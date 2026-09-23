"""GPU-free tests of the real worker boundary; no model downloads are allowed."""

import gc
import importlib.util
import inspect
import io
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("qwen_image21_worker", ROOT / "tools" / "qwen_image21_worker.py")
worker = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(worker)


class FakeGenerator:
    def __init__(self, device):
        self.device = device
        self.seed = None

    def manual_seed(self, seed):
        self.seed = seed
        return self


def fake_torch():
    return types.SimpleNamespace(
        bfloat16="bfloat16",
        Generator=FakeGenerator,
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            is_bf16_supported=lambda: True,
            is_initialized=lambda: False,
            empty_cache=mock.Mock(),
        ),
    )


class FakePipe:
    def __init__(self):
        self.calls = []
        self.before_callback = None
        self.fail = False

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("test inference failure")
        if self.before_callback:
            self.before_callback()
        callback_kwargs = {"sentinel": object()}
        returned = kwargs["callback_on_step_end"](self, 0, None, callback_kwargs)
        if returned is not callback_kwargs:
            raise AssertionError("callback must preserve pipeline state")
        return types.SimpleNamespace(images=[Image.new("RGBA", (kwargs["width"], kwargs["height"]), (1, 2, 3, 47))])


class WorkerJobTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "model_index.json").write_text(
            json.dumps({"_class_name": "QwenImage21Pipeline"}), encoding="utf-8"
        )
        self.job = self.root / "job"
        self.job.mkdir()
        self.payload = {
            "model_path": str(self.model),
            "job_dir": str(self.job),
            "precision": "int8",
            "memory_mode": "offload",
        }
        self.request = {
            "prompt": "a glass bird",
            "width": 256,
            "height": 320,
            "steps": 40,
            "seed": 123,
            "transparent": True,
            "input_images": [],
        }
        self.write_request()
        self.pipe = FakePipe()
        self.runtime = {
            "pipe": self.pipe,
            "int8_layers": {"transformer": 224, "text_encoder": 140},
            "versions": {"diffusers": "0.39.0.dev0"},
            "load_seconds": 1.2,
        }
        self.fake_torch = fake_torch()
        self.torch_patch = mock.patch.dict(sys.modules, {"torch": self.fake_torch})
        self.torch_patch.start()
        self.addCleanup(self.torch_patch.stop)
        self.output = redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)
        worker.clear_runtime()
        self.addCleanup(worker.clear_runtime)

    def write_request(self):
        (self.job / "request.json").write_text(json.dumps(self.request), encoding="utf-8")

    def run_job(self):
        with mock.patch.object(worker, "_load_runtime", return_value=self.runtime) as loader:
            result = worker.resident_run(self.payload)
        return result, loader

    def test_pipeline_arguments_and_saved_png_retain_native_alpha(self):
        reference1, reference2 = self.root / "one.png", self.root / "two.png"
        Image.new("RGBA", (20, 30), (10, 20, 30, 37)).save(reference1)
        Image.new("RGB", (40, 50), (40, 50, 60)).save(reference2)
        self.request["input_images"] = [str(reference1), str(reference2)]
        self.write_request()
        result, _ = self.run_job()
        actual = self.pipe.calls[0]
        self.assertEqual(actual["width"], 256)
        self.assertEqual(actual["height"], 320)
        self.assertEqual(actual["num_inference_steps"], 40)
        self.assertEqual(actual["true_cfg_scale"], 1.0)
        self.assertIs(actual["use_kv_cache"], True)
        self.assertEqual(actual["output_resolution"], 1024)
        self.assertEqual(actual["generator"].seed, 123)
        self.assertEqual(actual["generator"].device, "cpu")
        self.assertEqual(actual["callback_on_step_end_tensor_inputs"], [])
        self.assertEqual([image.size for image in actual["image"]], [(20, 30), (40, 50)])
        self.assertEqual(actual["image"][0].getpixel((0, 0)), (10, 20, 30, 37))
        self.assertEqual(
            actual["prompt"],
            "This is an RGBA image with transparency. a glass bird. The image has alpha channel and the background is transparent.",
        )
        with Image.open(result["output_path"]) as output:
            self.assertEqual(output.mode, "RGBA")
            self.assertEqual(output.getpixel((0, 0)), (1, 2, 3, 47))
        self.assertEqual(json.loads((self.job / "result.json").read_text(encoding="utf-8")), result)
        self.assertEqual(result["metadata"]["int8_layers"], self.runtime["int8_layers"])
        self.assertEqual(result["metadata"]["versions"], self.runtime["versions"])
        self.assertEqual(json.loads((self.job / "progress.json").read_text(encoding="utf-8"))["stage"], "complete")

    def test_opaque_request_keeps_original_prompt_and_t2i_uses_no_image(self):
        self.request["transparent"] = False
        self.write_request()
        result, _ = self.run_job()
        self.assertEqual(self.pipe.calls[0]["prompt"], "a glass bird")
        self.assertIsNone(self.pipe.calls[0]["image"])
        self.assertEqual(result["metadata"]["output_mode"], "RGBA")

    def test_rewrite_off_does_not_load_optional_model(self):
        with mock.patch("modules_forge.qwen_image21.prompt_rewriter.rewrite_prompt") as rewrite:
            result, _ = self.run_job()
        rewrite.assert_not_called()
        self.assertEqual(
            result["metadata"]["prompt_rewrite"], {"enabled": False, "applied": False, "reason": "disabled"}
        )

    def test_rewrite_on_feeds_expansion_to_pipeline_then_adds_transparency(self):
        self.request["rewrite_prompt"] = True
        self.write_request()
        rewritten = {"enabled": True, "applied": True, "rewritten_prompt": "Expanded glass bird", "wh_ratio": "1:1"}
        with mock.patch("modules_forge.qwen_image21.prompt_rewriter.rewrite_prompt", return_value=rewritten) as rewrite:
            result, _ = self.run_job()
        self.assertEqual(rewrite.call_args.args[:5], (self.model.parent.resolve(), "a glass bird", 256, 320, 123))
        self.assertIn("Expanded glass bird", self.pipe.calls[0]["prompt"])
        self.assertIn("background is transparent", self.pipe.calls[0]["prompt"])
        self.assertEqual(result["metadata"]["prompt"], "a glass bird")
        self.assertEqual(result["metadata"]["prompt_rewrite"], rewritten)
        self.assertEqual((result["metadata"]["width"], result["metadata"]["height"]), (256, 320))

    def test_reference_edit_skips_t2i_rewriter_even_when_enabled(self):
        path = self.root / "reference.png"
        Image.new("RGB", (8, 8)).save(path)
        self.request.update(rewrite_prompt=True, input_images=[str(path)])
        self.write_request()
        with mock.patch("modules_forge.qwen_image21.prompt_rewriter.rewrite_prompt") as rewrite:
            result, _ = self.run_job()
        rewrite.assert_not_called()
        self.assertEqual(result["metadata"]["prompt_rewrite"]["reason"], "image_edit")

    def test_turning_rewrite_off_keeps_cached_image_model_but_uses_the_new_original_prompt(self):
        self.request.update(rewrite_prompt=True, transparent=False)
        self.write_request()
        expanded = {"enabled": True, "applied": True, "rewritten_prompt": "Expanded first prompt"}
        with mock.patch("modules_forge.qwen_image21.prompt_rewriter.rewrite_prompt", return_value=expanded):
            self.run_job()
        self.request.update(rewrite_prompt=False, prompt="Unmodified second prompt")
        self.write_request()
        with mock.patch("modules_forge.qwen_image21.prompt_rewriter.rewrite_prompt") as rewrite:
            result, loader = self.run_job()
        rewrite.assert_not_called()
        loader.assert_not_called()
        self.assertEqual(self.pipe.calls[-1]["prompt"], "Unmodified second prompt")
        self.assertEqual(result["metadata"]["prompt_rewrite"]["reason"], "disabled")

    def test_edit_rewriter_receives_ordered_images_only_when_explicitly_enabled(self):
        path = self.job / "reference.png"
        Image.new("RGBA", (256, 320), (10, 20, 30, 47)).save(path)
        self.request.update(rewrite_edit_prompt=True, input_images=[str(path)], transparent=False)
        self.write_request()
        rewritten = {"enabled": True, "applied": True, "rewritten_prompt": "Change the cup to blue."}
        with mock.patch("modules_forge.qwen_image21.prompt_rewriter.rewrite_prompt", return_value=rewritten) as rewrite:
            result, _ = self.run_job()
        self.assertEqual(len(rewrite.call_args.kwargs["images"]), 1)
        self.assertEqual(rewrite.call_args.kwargs["images"][0].getpixel((0, 0)), (10, 20, 30, 47))
        self.assertEqual(self.pipe.calls[0]["prompt"], rewritten["rewritten_prompt"])
        self.assertTrue(result["metadata"]["prompt_rewrite"]["applied"])

    def test_preserved_and_raw_rgba_outputs_survive_worker_boundary(self):
        path = self.job / "reference.png"
        mask_path = self.job / "edit-mask.png"
        Image.new("RGBA", (256, 320), (10, 20, 30, 17)).save(path)
        mask = Image.new("L", (256, 320))
        mask.paste(255, (100, 100, 140, 140))
        mask.save(mask_path)
        self.request.update(
            input_images=[str(path)],
            preserve_unmasked=True,
            edit_mask={"original_path": str(path), "mask_path": str(mask_path), "feather": 2},
        )
        self.write_request()
        result, _ = self.run_job()
        with Image.open(result["preserved_output_path"]) as fixed, Image.open(result["output_path"]) as raw:
            self.assertEqual(fixed.getpixel((0, 0)), (10, 20, 30, 17))
            self.assertEqual(raw.getpixel((0, 0)), (1, 2, 3, 47))
            self.assertEqual(fixed.getpixel((120, 120)), raw.getpixel((120, 120)))
        self.assertTrue(result["metadata"]["preservation"]["applied"])

    def test_rewrite_failure_or_cancel_never_runs_diffusion(self):
        self.request["rewrite_prompt"] = True
        self.write_request()
        for error in (ValueError("invalid JSON"), worker.GenerationCancelled("cancelled while rewriting")):
            with (
                self.subTest(error=error),
                mock.patch("modules_forge.qwen_image21.prompt_rewriter.rewrite_prompt", side_effect=error),
                mock.patch.object(worker, "_load_runtime") as loader,
            ):
                with self.assertRaises(type(error)):
                    worker.resident_run(self.payload)
            loader.assert_not_called()
            self.assertFalse((self.job / "result.json").exists())
            self.assertIsNone(worker._RESIDENT_RUNTIME)

    def test_resident_image_model_is_offloaded_then_reused_after_rewrite(self):
        self.pipe.maybe_free_model_hooks = mock.Mock()
        self.run_job()
        self.request["rewrite_prompt"] = True
        self.write_request()
        rewritten = {"enabled": True, "applied": True, "rewritten_prompt": "Expanded", "wh_ratio": "1:1"}

        def rewrite(*args):
            self.pipe.maybe_free_model_hooks.assert_called_once()
            return rewritten

        with mock.patch("modules_forge.qwen_image21.prompt_rewriter.rewrite_prompt", side_effect=rewrite):
            result, loader = self.run_job()
        loader.assert_not_called()
        self.assertTrue(result["metadata"]["reused_model"])

    def test_gpu_resident_image_model_is_parked_before_rewrite_and_restored_afterward(self):
        self.payload["memory_mode"] = "gpu"
        self.pipe.to = mock.Mock()
        self.run_job()
        self.request["rewrite_prompt"] = True
        self.write_request()

        def rewrite(*args):
            self.pipe.to.assert_called_once_with("cpu")
            return {"enabled": True, "applied": True, "rewritten_prompt": "Expanded"}

        with mock.patch("modules_forge.qwen_image21.prompt_rewriter.rewrite_prompt", side_effect=rewrite):
            result, loader = self.run_job()
        self.assertEqual(self.pipe.to.call_args_list, [mock.call("cpu"), mock.call("cuda:0")])
        loader.assert_not_called()
        self.assertTrue(result["metadata"]["reused_model"])

    def test_offload_releases_unused_cache_without_touching_model_or_saved_pixels(self):
        self.fake_torch.cuda.is_initialized = lambda: True
        self.fake_torch.cuda.synchronize = mock.Mock()
        self.fake_torch.cuda.reset_peak_memory_stats = mock.Mock()
        self.fake_torch.cuda.memory_reserved = mock.Mock(side_effect=[8000 * 2**20, 300 * 2**20])
        self.fake_torch.cuda.memory_allocated = lambda: 200 * 2**20
        self.fake_torch.cuda.max_memory_allocated = lambda: 7000 * 2**20
        self.fake_torch.cuda.max_memory_reserved = lambda: 8000 * 2**20
        self.pipe.before_callback = self.fake_torch.cuda.empty_cache.reset_mock
        result, _ = self.run_job()
        self.fake_torch.cuda.empty_cache.assert_called_once()
        self.assertEqual(result["metadata"]["memory"]["released_cache_mib"], 7700.0)
        self.assertIs(worker._RESIDENT_RUNTIME["pipe"], self.pipe)
        with Image.open(result["output_path"]) as saved:
            self.assertEqual(saved.getpixel((0, 0)), (1, 2, 3, 47))

    def test_gpu_residency_does_not_trim_allocator_cache(self):
        cuda = types.SimpleNamespace(
            is_initialized=lambda: True,
            synchronize=mock.Mock(),
            memory_reserved=lambda: 8000 * 2**20,
            memory_allocated=lambda: 7000 * 2**20,
            max_memory_allocated=lambda: 7500 * 2**20,
            max_memory_reserved=lambda: 8000 * 2**20,
            empty_cache=mock.Mock(),
        )
        report = worker._idle_cuda_memory(types.SimpleNamespace(cuda=cuda), "gpu")
        cuda.empty_cache.assert_not_called()
        self.assertEqual(report["released_cache_mib"], 0)

    def test_model_reused_until_file_metadata_or_precision_or_memory_mode_changes(self):
        _, first = self.run_job()
        second_result, second = self.run_job()
        first.assert_called_once()
        second.assert_not_called()
        self.assertTrue(second_result["metadata"]["reused_model"])
        self.assertEqual(second_result["metadata"]["timings"]["load_seconds"], 0)
        (self.model / "model.safetensors").write_bytes(b"new shard")
        _, changed_file = self.run_job()
        changed_file.assert_called_once()
        self.payload["precision"] = "bf16"
        _, changed_precision = self.run_job()
        changed_precision.assert_called_once()
        self.payload["memory_mode"] = "gpu"
        _, changed_memory_mode = self.run_job()
        changed_memory_mode.assert_called_once()

    def test_overwritten_config_and_revision_marker_invalidate_cache(self):
        marker = self.model / ".revision"
        marker.write_text("first", encoding="utf-8")
        self.run_job()
        marker.write_text("second revision", encoding="utf-8")
        _, loader = self.run_job()
        loader.assert_called_once()
        path = self.model / "model_index.json"
        old = path.stat()
        os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns + 1_000_000))
        _, loader = self.run_job()
        loader.assert_called_once()

    def test_external_setup_manifest_changes_invalidate_cache_and_record_revision(self):
        manifest = self.model.parent / "model-files.json"
        manifest.write_text(json.dumps({"revision": "first"}), encoding="utf-8")
        result, _ = self.run_job()
        self.assertEqual(result["metadata"]["model_revision"], "first")
        manifest.write_text(json.dumps({"revision": "second_revision"}), encoding="utf-8")
        result, loader = self.run_job()
        loader.assert_called_once()
        self.assertEqual(result["metadata"]["model_revision"], "second_revision")

    def test_cancel_before_loading_discards_previous_cache_and_success_marker(self):
        self.run_job()
        (self.job / "cancel").touch()
        with mock.patch.object(worker, "_load_runtime") as loader:
            with self.assertRaises(worker.GenerationCancelled):
                worker.resident_run(self.payload)
        loader.assert_not_called()
        self.assertIsNone(worker._RESIDENT_RUNTIME)
        self.assertFalse((self.job / "result.json").exists())

    def test_cancel_after_loading_never_starts_inference(self):
        def load(*_args):
            (self.job / "cancel").touch()
            return self.runtime

        with mock.patch.object(worker, "_load_runtime", side_effect=load):
            with self.assertRaises(worker.GenerationCancelled):
                worker.resident_run(self.payload)
        self.assertEqual(self.pipe.calls, [])
        self.assertIsNone(worker._RESIDENT_RUNTIME)

    def test_cancel_callback_invalidates_cache_and_next_request_loads_fresh(self):
        self.pipe.before_callback = lambda: (self.job / "cancel").touch()
        with self.assertRaises(worker.GenerationCancelled):
            self.run_job()
        self.assertIsNone(worker._RESIDENT_RUNTIME)
        self.assertFalse((self.job / "result.json").exists())
        self.assertFalse((self.job / "output.png").exists())
        self.pipe.before_callback = None
        (self.job / "cancel").unlink()
        _, loader = self.run_job()
        loader.assert_called_once()

    def test_failure_during_inference_invalidates_cache(self):
        self.pipe.fail = True
        with self.assertRaisesRegex(RuntimeError, "test inference failure"):
            self.run_job()
        self.assertIsNone(worker._RESIDENT_RUNTIME)
        self.assertIsNone(worker._RESIDENT_KEY)
        self.assertFalse((self.job / "result.json").exists())

    def test_non_grid_size_is_rejected_instead_of_silently_rounded(self):
        self.request["width"] = 257
        self.write_request()
        with mock.patch.object(worker, "_load_runtime") as loader:
            with self.assertRaisesRegex(ValueError, "multiple of 32"):
                worker.resident_run(self.payload)
        loader.assert_not_called()

    def test_ten_references_are_accepted_but_urls_and_conflicting_modes_rejected(self):
        reference = self.root / "one.png"
        Image.new("RGBA", (2, 2)).save(reference)
        self.request["input_images"] = [str(reference)] * 10
        self.write_request()
        worker._read_request(self.payload)
        self.request["input_images"] = ["https://example.com/reference.png"]
        self.write_request()
        with self.assertRaisesRegex(ValueError, "absolute local"):
            worker._read_request(self.payload)
        self.request["input_images"] = []
        self.request["precision"] = "bf16"
        self.write_request()
        with self.assertRaisesRegex(ValueError, "disagree"):
            worker._read_request(self.payload)


class WorkerLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.job = Path(self.temp.name)
        self.events = []
        self.components = {}
        self.torch = fake_torch()
        events, components = self.events, self.components

        class Linear8bitLt:
            pass

        class Config:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class Component:
            is_loaded_in_8bit = True

            def __init__(self, name):
                self.name = name
                self.quantized_modules = [Linear8bitLt(), Linear8bitLt()]

            def modules(self):
                return iter([self, *self.quantized_modules])

            def to(self, device):
                events.append((self.name, "to", device))
                return self

            @classmethod
            def from_pretrained(cls, path, **kwargs):
                name = kwargs["subfolder"]
                events.append((name, "load", path, kwargs))
                component = cls(name)
                components[name] = component
                return component

        class Pipe:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                events.append(("pipeline", "load", path, kwargs))
                return cls()

            def enable_model_cpu_offload(self, **kwargs):
                events.append(("pipeline", "offload", kwargs))

            def to(self, device):
                events.append(("pipeline", "to", device))

            def set_progress_bar_config(self, **kwargs):
                pass

        class Scheduler:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                events.append(("scheduler", "load", path, kwargs))
                return types.SimpleNamespace(config=types.SimpleNamespace(shift_terminal=None))

        self.Component = Component
        self.validator = mock.Mock()
        self.modules = {
            "torch": self.torch,
            "diffusers": types.SimpleNamespace(
                BitsAndBytesConfig=Config,
                QwenImage21Pipeline=Pipe,
                QwenImage21Transformer2DModel=Component,
                FlowMatchEulerDiscreteScheduler=Scheduler,
            ),
            "transformers": types.SimpleNamespace(BitsAndBytesConfig=Config, Qwen3VLForConditionalGeneration=Component),
            "bitsandbytes.nn": types.SimpleNamespace(Linear8bitLt=Linear8bitLt),
            "modules_forge.qwen_image21_environment": types.SimpleNamespace(validate_running_versions=self.validator),
        }
        patcher = mock.patch.dict(sys.modules, self.modules)
        patcher.start()
        self.addCleanup(patcher.stop)
        environment = mock.patch.dict(os.environ, {})
        environment.start()
        self.addCleanup(environment.stop)
        versions = mock.patch.object(worker, "_versions", return_value={"diffusers": "pinned"})
        versions.start()
        self.addCleanup(versions.stop)
        cache = mock.patch(
            "modules_forge.qwen_image21.quantized_cache.load_or_create",
            side_effect=lambda _model, _identity, _load, create, _save, **_kwargs: (create(), {"status": "created"}),
        )
        cache.start()
        self.addCleanup(cache.stop)
        # These tests exercise mocked loader ordering, not installed GPU
        # distributions. Cache identity/serialization have their own tests.
        identity = mock.patch(
            "modules_forge.qwen_image21.quantized_cache.component_identity",
            side_effect=lambda _path, name, precision, **_kwargs: {"component": name, "precision": precision},
        )
        identity.start()
        self.addCleanup(identity.stop)
        output = redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def load(self, precision="int8", memory_mode="offload"):
        return worker._load_runtime(self.job, {"precision": precision, "memory_mode": memory_mode}, self.job)

    def test_both_components_quantize_on_gpu_then_park_on_cpu_before_pipeline_offload(self):
        runtime = self.load()
        self.validator.assert_called_once()
        self.assertEqual(
            [(event[0], event[1]) for event in self.events],
            [
                ("transformer", "load"),
                ("transformer", "to"),
                ("text_encoder", "load"),
                ("text_encoder", "to"),
                ("pipeline", "load"),
                ("pipeline", "offload"),
            ],
        )
        for event in (self.events[0], self.events[2]):
            kwargs = event[3]
            self.assertEqual(kwargs["device_map"], {"": "cuda:0"})
            self.assertEqual(kwargs["torch_dtype"], "bfloat16")
            self.assertTrue(kwargs["local_files_only"])
            self.assertTrue(kwargs["use_safetensors"])
            self.assertTrue(kwargs["quantization_config"].kwargs["load_in_8bit"])
        self.assertEqual(self.events[1][2], "cpu")
        self.assertEqual(self.events[3][2], "cpu")
        self.assertIs(self.events[4][3]["transformer"], self.components["transformer"])
        self.assertIs(self.events[4][3]["text_encoder"], self.components["text_encoder"])
        self.assertTrue(self.events[4][3]["local_files_only"])
        self.assertEqual(runtime["int8_layers"], {"transformer": 2, "text_encoder": 2})
        self.assertEqual(
            self.events[0][3]["quantization_config"].kwargs["llm_int8_skip_modules"], list(worker.INT8_SKIP_MODULES)
        )
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")

    def test_bf16_does_not_construct_quantized_components_and_respects_gpu_mode(self):
        runtime = self.load(precision="bf16", memory_mode="gpu")
        self.assertEqual([(event[0], event[1]) for event in self.events], [("pipeline", "load"), ("pipeline", "to")])
        self.assertEqual(self.events[1][2], "cuda:0")
        self.assertEqual(runtime["int8_layers"], {})
        self.assertEqual(self.events[0][3]["torch_dtype"], "bfloat16")

    def test_turbo_bf16_replaces_transformer_and_scheduler(self):
        runtime = self.load(precision="turbo_bf16")
        self.assertEqual(
            [event[:2] for event in self.events],
            [("transformer", "load"), ("pipeline", "load"), ("scheduler", "load"), ("pipeline", "offload")],
        )
        self.assertEqual(self.events[0][3]["subfolder"], "transformer")
        self.assertIs(self.events[1][3]["transformer"], self.components["transformer"])
        self.assertIsNone(runtime["pipe"].scheduler.config.shift_terminal)

    def test_turbo_q4_loads_gguf_transformer_and_int8_encoder(self):
        gguf = object()
        with (
            mock.patch("modules_forge.qwen_image21.turbo.load_gguf_transformer", return_value=gguf) as load_gguf,
            mock.patch.object(worker.importlib.metadata, "version", return_value="0.19.0"),
        ):
            runtime = self.load(precision="turbo_q4_k_m")
        load_gguf.assert_called_once_with(self.job)
        self.assertEqual(
            [event[:2] for event in self.events],
            [
                ("text_encoder", "load"),
                ("text_encoder", "to"),
                ("pipeline", "load"),
                ("scheduler", "load"),
                ("pipeline", "offload"),
            ],
        )
        self.assertIs(self.events[2][3]["transformer"], gguf)
        self.assertEqual(runtime["int8_layers"], {"text_encoder": 2})
        self.assertEqual(runtime["versions"]["gguf"], "0.19.0")

    def test_regular_q4_loads_gguf_transformer_with_base_scheduler(self):
        gguf = object()
        with (
            mock.patch("modules_forge.qwen_image21.regular_gguf.load_gguf_transformer", return_value=gguf) as load_gguf,
            mock.patch.object(worker.importlib.metadata, "version", return_value="0.19.0"),
        ):
            runtime = self.load(precision="base_q4_k_m")
        load_gguf.assert_called_once_with(self.job)
        self.assertEqual(
            [event[:2] for event in self.events],
            [
                ("text_encoder", "load"),
                ("text_encoder", "to"),
                ("pipeline", "load"),
                ("pipeline", "offload"),
            ],
        )
        self.assertIs(self.events[2][3]["transformer"], gguf)
        self.assertEqual(runtime["int8_layers"], {"text_encoder": 2})
        self.assertEqual(runtime["versions"]["gguf"], "0.19.0")

    def test_int8_gpu_mode_still_stages_loading_and_only_moves_pipeline_at_end(self):
        self.load(memory_mode="gpu")
        self.assertEqual(self.events[1][2], "cpu")
        self.assertEqual(self.events[3][2], "cpu")
        self.assertEqual(self.events[-1], ("pipeline", "to", "cuda:0"))

    def test_quantization_flag_without_actual_int8_layers_is_rejected(self):
        with mock.patch.object(self.Component, "modules", return_value=iter([])):
            with self.assertRaisesRegex(RuntimeError, "not loaded as bitsandbytes INT8"):
                self.load()
        self.assertEqual(len(self.events), 1)

    def test_cancellation_between_component_loads_skips_encoder_and_pipeline(self):
        original = self.Component.to

        def cancel_after_parking(component, device):
            result = original(component, device)
            (self.job / "cancel").touch()
            return result

        with mock.patch.object(self.Component, "to", cancel_after_parking):
            with self.assertRaises(worker.GenerationCancelled):
                self.load()
        self.assertEqual(
            [(event[0], event[1]) for event in self.events], [("transformer", "load"), ("transformer", "to")]
        )


class Int8OffloadMetadataTests(unittest.TestCase):
    def test_parent_apply_preserves_metadata_phase_and_aliases_without_copying_weight(self):
        import torch

        class TinyInt8Linear(torch.nn.Module):
            def __init__(self, initialized):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones((4, 4), dtype=torch.int8), requires_grad=False)
                scales = torch.arange(4, dtype=torch.float32)
                self.weight.CB = None if initialized else self.weight.data
                self.weight.SCB = None if initialized else scales
                self.state = types.SimpleNamespace(
                    CB=self.weight.data if initialized else None,
                    SCB=scales if initialized else None,
                    idx=torch.tensor([1, 2], dtype=torch.int64),
                )

        for initialized in (False, True):
            with self.subTest(initialized=initialized):
                layer = TinyInt8Linear(initialized)
                model = torch.nn.Sequential(layer)
                with mock.patch.dict(
                    sys.modules, {"bitsandbytes.nn": types.SimpleNamespace(Linear8bitLt=TinyInt8Linear)}
                ):
                    worker._install_int8_offload_fix(model)
                old_weight = layer.weight.data
                model._apply(lambda tensor: tensor.clone())
                self.assertNotEqual(layer.weight.data_ptr(), old_weight.data_ptr())
                active, inactive = (layer.state, layer.weight) if initialized else (layer.weight, layer.state)
                self.assertEqual(active.CB.data_ptr(), layer.weight.data_ptr())
                self.assertIsNone(inactive.CB)
                self.assertIsNone(inactive.SCB)
                self.assertEqual(active.SCB.dtype, torch.float32)
                torch.testing.assert_close(active.SCB, torch.arange(4, dtype=torch.float32))
                self.assertEqual(layer.state.idx.dtype, torch.int64)
                self.assertEqual(layer.state.idx.tolist(), [1, 2])


@unittest.skipUnless(
    os.environ.get("QWEN_IMAGE21_CPU_RUNTIME_TEST") == "1", "requires the pinned Qwen dedicated environment"
)
class QwenImage21RuntimeCpuTests(unittest.TestCase):
    """Opt-in checks of actual upstream modules with tiny random CPU weights.

    Run with QWEN_IMAGE21_CPU_RUNTIME_TEST=1 in the isolated worker environment.
    This exercises architecture/KV-cache/RGBA compatibility, not model quality
    or the CUDA-only Diffusers bitsandbytes loader.
    """

    @classmethod
    def setUpClass(cls):
        from modules_forge.qwen_image21_environment import validate_running_versions

        validate_running_versions()

    def test_real_transformer_cpu_cache_matches_uncached_target(self):
        import torch
        from diffusers import QwenImage21Transformer2DModel
        from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

        generator = torch.Generator(device="cpu").manual_seed(0)
        model = (
            QwenImage21Transformer2DModel(
                patch_size=1,
                in_channels=4,
                out_channels=4,
                num_layers=2,
                attention_head_dim=16,
                num_attention_heads=2,
                context_in_dim=8,
                mlp_ratio=2,
                axes_dims_rope=(4, 6, 6),
            )
            .to("cpu")
            .eval()
        )
        inputs = {
            "hidden_states": torch.randn((1, 4, 4), generator=generator),
            "encoder_hidden_states": torch.randn((1, 4, 8), generator=generator),
            "encoder_hidden_states_mask": torch.ones((1, 4), dtype=torch.long),
            "img_shapes": [[(1, 2, 2)]],
            "img_mask": torch.tensor([[False, False, False, False, True]]),
        }
        cache = QwenImage21KVCache(2)
        with torch.no_grad():
            model(**inputs, timestep=torch.tensor([0.9]), kv_cache=cache, kv_cache_mode="extract", return_dict=False)
            actual = model(
                **inputs, timestep=torch.tensor([0.4]), kv_cache=cache, kv_cache_mode="cached", return_dict=False
            )[0]
            reference = model(**inputs, timestep=torch.tensor([0.4]), return_dict=False)[0][:, -4:]
        self.assertEqual(tuple(actual.shape), (1, 4, 4))
        torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)

    def test_real_vae_cpu_decodes_four_channels_and_processor_returns_rgba(self):
        import torch
        from diffusers import AutoencoderKLQwenImage21
        from diffusers.image_processor import VaeImageProcessor

        model = (
            AutoencoderKLQwenImage21(
                base_dim=4,
                decoder_base_dim=4,
                z_dim=4,
                dim_mult=[1, 1, 1, 1, 1],
                num_res_blocks=1,
                attn_scales=[],
                temperal_downsample=[False, True, True, True],
                latents_mean=[0.0] * 4,
                latents_std=[1.0] * 4,
            )
            .to("cpu")
            .eval()
        )
        with torch.no_grad():
            latent = model.encode(torch.zeros((1, 4, 1, 96, 96))).latent_dist.mode()
            decoded = model.decode(latent, return_dict=False)[0][:, :, 0]
        self.assertEqual(tuple(latent.shape[-2:]), (6, 6))
        self.assertEqual(tuple(decoded.shape), (1, 4, 96, 96))
        image = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=4).postprocess(decoded, output_type="pil")[0]
        self.assertEqual(image.mode, "RGBA")

    def test_real_pipeline_accepts_every_worker_sampling_argument(self):
        from diffusers import QwenImage21Pipeline

        parameters = inspect.signature(QwenImage21Pipeline.__call__).parameters
        arguments = {
            "prompt",
            "image",
            "width",
            "height",
            "num_inference_steps",
            "true_cfg_scale",
            "use_kv_cache",
            "output_resolution",
            "generator",
            "num_images_per_prompt",
            "output_type",
            "return_dict",
            "callback_on_step_end",
            "callback_on_step_end_tensor_inputs",
        }
        self.assertTrue(arguments.issubset(parameters))
        self.assertEqual(QwenImage21Pipeline.model_cpu_offload_seq, "text_encoder->transformer->vae")


@unittest.skipUnless(
    os.environ.get("QWEN_IMAGE21_GPU_RUNTIME_TEST") == "1", "requires explicit tiny-model CUDA validation"
)
class QwenImage21RuntimeGpuTests(unittest.TestCase):
    """Actual production INT8 component loaders on tiny random local weights."""

    @classmethod
    def setUpClass(cls):
        from modules_forge.qwen_image21_environment import validate_running_versions

        validate_running_versions()

    def test_real_int8_components_survive_gpu_cpu_and_pipeline_offload(self):
        import torch
        from bitsandbytes.nn import Linear8bitLt
        from diffusers import (
            AutoencoderKLQwenImage21,
            FlowMatchEulerDiscreteScheduler,
            QwenImage21Pipeline,
            QwenImage21Transformer2DModel,
        )
        from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

        self.assertTrue(torch.cuda.is_available(), "CUDA is required by this explicitly enabled test")
        self.assertTrue(torch.cuda.is_bf16_supported())
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            transformer = QwenImage21Transformer2DModel(
                patch_size=1,
                in_channels=4,
                out_channels=4,
                num_layers=2,
                attention_head_dim=16,
                num_attention_heads=2,
                context_in_dim=32,
                mlp_ratio=2,
                axes_dims_rope=(4, 6, 6),
            )
            transformer.save_pretrained(directory / "transformer")
            text_encoder = Qwen3VLForConditionalGeneration(
                Qwen3VLConfig(
                    text_config={
                        "vocab_size": 64,
                        "hidden_size": 32,
                        "intermediate_size": 64,
                        "num_hidden_layers": 2,
                        "num_attention_heads": 2,
                        "num_key_value_heads": 2,
                        "head_dim": 16,
                        "max_position_embeddings": 64,
                        "rope_parameters": {"rope_type": "default", "mrope_section": [2, 3, 3]},
                    },
                    vision_config={
                        "depth": 1,
                        "hidden_size": 32,
                        "intermediate_size": 64,
                        "num_heads": 2,
                        "out_hidden_size": 32,
                        "patch_size": 2,
                        "temporal_patch_size": 1,
                        "num_position_embeddings": 16,
                        "deepstack_visual_indexes": [],
                    },
                    image_token_id=60,
                    video_token_id=61,
                    vision_start_token_id=62,
                    vision_end_token_id=63,
                )
            )
            text_encoder.save_pretrained(directory / "text_encoder")
            del transformer, text_encoder

            def assemble_pipeline(_path, **kwargs):
                # Only tokenization and non-quantized asset loading are replaced.
                # Both INT8 loaders, all actual model classes, and the real
                # Diffusers device/offload implementation remain in use.
                assert_int8(kwargs["transformer"], "cpu")
                assert_int8(kwargs["text_encoder"], "cpu")
                vae = AutoencoderKLQwenImage21(
                    base_dim=4,
                    decoder_base_dim=4,
                    z_dim=4,
                    dim_mult=[1, 1, 1, 1, 1],
                    num_res_blocks=1,
                    attn_scales=[],
                    temperal_downsample=[False, True, True, True],
                    latents_mean=[0.0] * 4,
                    latents_std=[1.0] * 4,
                ).to(dtype=torch.bfloat16)

                class TinyProcessor:
                    tokenizer = types.SimpleNamespace(encode=lambda _text: [60])

                    def apply_chat_template(self, *_args, **_kwargs):
                        return [[1]]

                processor = TinyProcessor()
                return QwenImage21Pipeline(
                    scheduler=FlowMatchEulerDiscreteScheduler(),
                    vae=vae,
                    processor=processor,
                    transformer=kwargs["transformer"],
                    text_encoder=kwargs["text_encoder"],
                )

            def assert_int8(model, device):
                layers = [module for module in model.modules() if isinstance(module, Linear8bitLt)]
                self.assertGreater(len(layers), 0)
                self.assertTrue(all(layer.weight.dtype == torch.int8 for layer in layers))
                self.assertTrue(all(layer.weight.device.type == device for layer in layers))
                for index, layer in enumerate(layers):
                    for owner_name, owner in (("weight", layer.weight), ("state", layer.state)):
                        for name, value in vars(owner).items():
                            if isinstance(value, torch.Tensor):
                                self.assertEqual(value.device.type, device, f"INT8 layer {index}: {owner_name}.{name}")
                                if name == "SCB":
                                    self.assertEqual(value.dtype, torch.float32)
                                if name == "CB":
                                    self.assertEqual(value.data_ptr(), layer.weight.data_ptr())
                return len(layers)

            def allocated():
                torch.cuda.synchronize()
                return torch.cuda.memory_allocated()

            def int8_bytes(model):
                return sum(module.weight.numel() for module in model.modules() if isinstance(module, Linear8bitLt))

            def forward_components(pipe):
                with torch.inference_mode():
                    encoded = pipe.text_encoder(
                        input_ids=torch.tensor([[1, 2, 3, 4]], device="cuda:0"),
                        attention_mask=torch.ones((1, 4), dtype=torch.long, device="cuda:0"),
                        output_hidden_states=True,
                    ).hidden_states[-1]
                    result = pipe.transformer(
                        hidden_states=torch.randn((1, 4, 4), device="cuda:0", dtype=torch.bfloat16),
                        encoder_hidden_states=encoded,
                        encoder_hidden_states_mask=torch.ones((1, 4), dtype=torch.long, device="cuda:0"),
                        timestep=torch.tensor([0.4], device="cuda:0", dtype=torch.bfloat16),
                        img_shapes=[[(1, 2, 2)]],
                        img_mask=torch.tensor([[False, False, False, False, True]], device="cuda:0"),
                        return_dict=False,
                    )[0]
                self.assertEqual(tuple(encoded.shape), (1, 4, 32))
                self.assertEqual(tuple(result.shape), (1, 8, 4))
                self.assertTrue(torch.isfinite(result).all().item())

            with mock.patch.object(QwenImage21Pipeline, "from_pretrained", side_effect=assemble_pipeline):
                runtime = worker._load_runtime(directory, {"precision": "int8", "memory_mode": "gpu"}, directory)
            pipe = runtime["pipe"]
            self.assertEqual(assert_int8(pipe.transformer, "cuda"), runtime["int8_layers"]["transformer"])
            self.assertEqual(assert_int8(pipe.text_encoder, "cuda"), runtime["int8_layers"]["text_encoder"])
            trace = {"int8_layers": runtime["int8_layers"], "cycles": []}
            trace["loaded_gpu_bytes"] = allocated()
            pipe.to("cpu")
            assert_int8(pipe.transformer, "cpu")
            assert_int8(pipe.text_encoder, "cpu")
            gc.collect()
            trace["parked_cpu_bytes"] = allocated()
            self.assertGreaterEqual(
                trace["loaded_gpu_bytes"] - trace["parked_cpu_bytes"],
                int8_bytes(pipe.transformer) + int8_bytes(pipe.text_encoder),
            )
            pipe.to("cuda:0")
            forward_components(pipe)
            pipe.to("cpu")
            assert_int8(pipe.transformer, "cpu")
            assert_int8(pipe.text_encoder, "cpu")
            pipe.enable_model_cpu_offload(gpu_id=0, device="cuda")
            self.assertEqual(len(pipe._all_hooks), 3)
            for _ in range(2):
                forward_components(pipe)
                assert_int8(pipe.text_encoder, "cpu")
                assert_int8(pipe.transformer, "cuda")
                before_offload = allocated()
                pipe.maybe_free_model_hooks()
                assert_int8(pipe.transformer, "cpu")
                assert_int8(pipe.text_encoder, "cpu")
                gc.collect()
                after_offload = allocated()
                self.assertGreaterEqual(before_offload - after_offload, int8_bytes(pipe.transformer))
                trace["cycles"].append({"before_bytes": before_offload, "after_bytes": after_offload})
            sys.stdout.write("INT8 offload regression: " + json.dumps(trace) + "\n")
            torch.cuda.synchronize()
            del pipe, runtime
            worker.clear_runtime()


if __name__ == "__main__":
    unittest.main()
