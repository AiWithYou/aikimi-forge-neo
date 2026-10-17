"""CPU contracts for the pinned native Comfy MOSS Image runtime."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from modules_forge.moss_image import core
from modules_forge.moss_image.inference import (
    Cancelled,
    ComfyRuntime,
    Runner,
    finite_vae_decode,
    import_arguments,
    read_hf_weights,
    verify_module_origin,
)


class FakeRuntime:
    def __init__(self):
        self.closed = False
        self.last_metadata = {"actual_steps": 49, "sampler": "euler", "scheduler": "sgm_uniform"}

    def generate(self, request, progress, check):
        check()
        progress("sampling", 1, request["steps"])
        return Image.new("RGB", (request["width"], request["height"]), "coral")

    def close(self):
        self.closed = True


class MossImageInferenceTests(unittest.TestCase):
    def test_png_retains_requested_precisions_loras_and_measured_conditions(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "job"
            request = core.validate_request({"prompt": "anime rainy street", "width": 512, "height": 768, "seed": 99})
            with (
                patch.object(core, "model_ready", return_value=True),
                patch("modules_forge.moss_image.inference.lora.bind"),
                patch("modules_forge.moss_image.inference.lora.record", return_value=[]),
            ):
                result = Runner(Path(temp), runtime=FakeRuntime()).run(directory, request)
            self.assertEqual(Path(result["image_path"]), directory / "output.png")
            self.assertEqual(json.loads((directory / "result.json").read_text(encoding="utf-8")), result)
            with Image.open(result["image_path"]) as image:
                metadata = json.loads(image.info["moss_image"])
                self.assertEqual(image.size, (512, 768))
            for key, value in request.items():
                self.assertEqual(metadata[key], value)
            self.assertEqual((metadata["model_id"], metadata["model_revision"]), (core.MODEL_ID, core.MODEL_REVISION))
            self.assertEqual((metadata["precision"], metadata["text_precision"]), ("int8", "int8"))
            self.assertEqual(
                (metadata["actual_steps"], metadata["sampler"], metadata["scheduler"]), (49, "euler", "sgm_uniform")
            )
            self.assertNotIn(str(Path(temp)), json.dumps(metadata))
            self.assertGreaterEqual(metadata["elapsed_seconds"], 0)
            self.assertEqual(json.loads((directory / "progress.json").read_text())["stage"], "complete")

    def test_cancelled_and_unverified_jobs_never_publish_results(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with (
                patch.object(core, "model_ready", return_value=False),
                patch("modules_forge.moss_image.inference.lora.bind"),
            ):
                with self.assertRaises(ValueError):
                    Runner(directory, runtime=FakeRuntime()).run(directory, {"prompt": "cat"})
            (directory / "cancel").touch()
            with self.assertRaises(Cancelled):
                Runner(directory, runtime=FakeRuntime()).run(directory, {"prompt": "cat"})
            self.assertFalse((directory / "output.png").exists())
            self.assertFalse((directory / "result.json").exists())

    def test_import_arguments_isolate_worker_options_and_restore_on_failure(self):
        previous = sys.argv
        with self.assertRaises(RuntimeError):
            with import_arguments(cpu=True):
                self.assertIn("--cpu", sys.argv)
                self.assertIn("--fp32-vae", sys.argv)
                self.assertNotIn("--script", sys.argv)
                raise RuntimeError("import failure")
        self.assertIs(sys.argv, previous)

    def test_module_origin_accepts_namespace_only_inside_pinned_checkout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            verify_module_origin(SimpleNamespace(__file__=None, __path__=[str(root / "comfy")]), root)
            with self.assertRaises(RuntimeError):
                verify_module_origin(SimpleNamespace(__file__=None, __path__=[str(root.parent / "other")]), root)

    def test_cpu_constructor_imports_no_server_and_never_initializes_cuda(self):
        with tempfile.TemporaryDirectory() as temp:
            repository = Path(temp)
            options = SimpleNamespace(args_parsing=False)
            modules = {
                "torch": SimpleNamespace(
                    cuda=SimpleNamespace(is_available=lambda: self.fail("CPU import must not query CUDA"))
                ),
                "comfy": SimpleNamespace(__file__=None, __path__=[str(repository / "comfy")]),
                "comfy.options": options,
                "comfy.cli_args": SimpleNamespace(args=SimpleNamespace(cpu=True)),
            }
            for name in (
                "comfy.model_management",
                "comfy.sd",
                "comfy.utils",
                "comfy.sample",
                "comfy.samplers",
                "comfy.lora",
                "comfy.ldm.moss_image.model",
                "comfy_extras.nodes_model_advanced",
                "comfy_extras.nodes_slg",
            ):
                modules[name] = SimpleNamespace()
            for name, module in modules.items():
                if name not in ("torch", "comfy"):
                    module.__file__ = str(repository / (name.replace(".", "/") + ".py"))
            with (
                patch.object(core, "runtime_ready", return_value=True),
                patch.object(core, "runtime_paths", return_value={"comfy": repository}),
                patch("modules_forge.moss_image.inference.sys.path", list(sys.path)),
                patch("modules_forge.moss_image.inference.importlib.metadata.version", return_value="0.2.37"),
                patch(
                    "modules_forge.moss_image.inference.importlib.import_module", side_effect=modules.__getitem__
                ) as imports,
            ):
                runtime = ComfyRuntime(repository / "models", cpu=True)
            self.assertTrue(runtime.cpu)
            self.assertFalse(options.args_parsing)
            self.assertNotIn("nodes", [call.args[0] for call in imports.call_args_list])

    def test_hf_shards_are_merged_once_with_strict_index_contents_and_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            index = directory / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {"a": "first.safetensors", "b": "second.safetensors"}}))
            utils = SimpleNamespace(
                load_torch_file=lambda path, **k: {"a": 1} if Path(path).name.startswith("first") else {"b": 2}
            )
            self.assertEqual(read_hf_weights(directory, "model", utils), {"a": 1, "b": 2})
            index.write_text(json.dumps({"weight_map": {"a": "../escape.safetensors"}}))
            with self.assertRaises(ValueError):
                read_hf_weights(directory, "model", utils)
            index.write_text(json.dumps({"weight_map": {"wrong": "first.safetensors"}}))
            with self.assertRaises(ValueError):
                read_hf_weights(directory, "model", utils)

    def test_selected_native_and_hf_loaders_never_change_precision(self):
        runtime = ComfyRuntime.__new__(ComfyRuntime)
        calls = []
        model = SimpleNamespace(model=SimpleNamespace(diffusion_model=object()))
        state = {"model.language_model.layers.0.input_layernorm.weight": object()}
        clip = SimpleNamespace(tokenizer=SimpleNamespace(qwen35_9b=object()))
        runtime.sd = SimpleNamespace(
            load_diffusion_model=lambda *a, **k: calls.append(("native_dit", a, k)) or model,
            load_diffusion_model_state_dict=lambda *a, **k: calls.append(("hf_dit", a, k)) or model,
            load_text_encoder_state_dicts=lambda *a, **k: calls.append(("text", a, k)) or clip,
            detect_te_model=lambda values: "qwen35",
            TEModel=SimpleNamespace(QWEN35_9B="qwen35"),
            CLIPType=SimpleNamespace(MOSS_IMAGE="moss"),
        )
        runtime.moss_model = SimpleNamespace(MOSSImage=object)
        runtime.torch = SimpleNamespace(bfloat16="bf16")
        runtime.utils = SimpleNamespace(
            load_torch_file=lambda *a, **k: (state, {}),
            convert_old_quants=lambda values, **kwargs: (values, {}),
        )
        with patch("modules_forge.moss_image.inference.read_hf_weights", return_value=state):
            runtime.load_dit(Path("int8.safetensors"), "int8")
            runtime.load_dit(Path("bundle"), "bf16")
            runtime.load_clip(Path("qwen_int8.safetensors"), "int8")
            runtime.load_clip(Path("qwen_w4a8.safetensors"), "w4a8")
            runtime.load_clip(Path("bundle"), "bf16")
        self.assertEqual(
            calls,
            [
                ("native_dit", ("int8.safetensors",), {}),
                ("hf_dit", (state,), {"model_options": {"dtype": "bf16"}}),
                ("text", ([state],), {"clip_type": "moss", "model_options": {}}),
                ("text", ([state],), {"clip_type": "moss", "model_options": {}}),
                ("text", ([state],), {"clip_type": "moss", "model_options": {"dtype": "bf16"}}),
            ],
        )

    def test_quantized_text_prefixes_reach_native_moss_without_fallback(self):
        from unittest.mock import Mock

        state = {
            "language_model.layers.0.input_layernorm.weight": object(),
            "language_model.layers.0.linear_attn.A_log": object(),
            "language_model.layers.0.mlp.up_proj.comfy_quant": object(),
            "language_model.layers.0.mlp.up_proj.weight_scale": object(),
            "visual.proj.weight": object(),
        }
        loaded = SimpleNamespace(tokenizer=SimpleNamespace(qwen35_9b=object()))
        runtime = ComfyRuntime.__new__(ComfyRuntime)
        runtime.utils = SimpleNamespace(
            load_torch_file=Mock(return_value=(state, {})),
            convert_old_quants=Mock(side_effect=lambda values, **kwargs: (values, {})),
        )
        runtime.sd = SimpleNamespace(
            load_clip=Mock(return_value=SimpleNamespace(tokenizer=object())),
            load_text_encoder_state_dicts=Mock(return_value=loaded),
            detect_te_model=lambda values: (
                "qwen35" if "model.language_model.layers.0.linear_attn.A_log" in values else None
            ),
            TEModel=SimpleNamespace(QWEN35_9B="qwen35"),
            CLIPType=SimpleNamespace(MOSS_IMAGE="moss"),
        )
        self.assertIs(runtime.load_clip(Path("int8.safetensors"), "int8"), loaded)
        converted = runtime.sd.load_text_encoder_state_dicts.call_args.args[0][0]
        self.assertEqual(set(converted), {"model." + key for key in state})
        for key, value in state.items():
            self.assertIs(converted["model." + key], value)
        runtime.sd.load_clip.assert_not_called()
        runtime.sd.detect_te_model = lambda values: None
        with self.assertRaisesRegex(ValueError, "Qwen3.5-9B"):
            runtime.load_clip(Path("int8.safetensors"), "int8")

    def test_encoding_counts_full_chat_before_plugin_can_truncate(self):
        import torch

        captured = []

        def tokenize(text, **kwargs):
            captured.append(text)
            return [[(index, 1.0) for index in range(4097)]]

        clip = SimpleNamespace(
            tokenizer=SimpleNamespace(qwen35_9b=SimpleNamespace(tokenize_with_weights=tokenize)),
            encode_from_tokens_scheduled=lambda *a: self.fail("oversize encoding must not run"),
        )
        runtime = ComfyRuntime.__new__(ComfyRuntime)
        runtime.torch = torch
        with self.assertRaisesRegex(ValueError, "4096"):
            runtime.encode_text(clip, "long prompt")
        self.assertEqual(captured, ["<|im_start|>user\nlong prompt<|im_end|>\n<|im_start|>assistant\n<think>\n"])

    def test_encode_finite_features_and_masks_are_returned_on_cpu(self):
        import torch

        clip = SimpleNamespace(
            tokenizer=SimpleNamespace(
                qwen35_9b=SimpleNamespace(tokenize_with_weights=lambda *a, **k: [[(1, 1.0), (2, 1.0), (3, 1.0)]])
            ),
            encode_from_tokens_scheduled=lambda tokens: [
                [torch.zeros(1, 3, 4096), {"attention_mask": torch.ones(1, 3)}]
            ],
        )
        runtime = ComfyRuntime.__new__(ComfyRuntime)
        runtime.torch = torch
        encoded, count = runtime.encode_text(clip, "cat")
        self.assertEqual(count, 3)
        self.assertEqual(encoded[0][0].device.type, "cpu")
        self.assertEqual(encoded[0][1]["attention_mask"].device.type, "cpu")
        clip.encode_from_tokens_scheduled = lambda tokens: [[torch.full((1, 3, 4096), float("nan")), {}]]
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            runtime.encode_text(clip, "cat")

    def test_sampling_uses_shift_then_simple_slg_and_cpu_fp32_noise(self):
        import torch

        events, captured = [], {}
        model_sampling = object()
        model = SimpleNamespace(get_model_object=lambda key: model_sampling)
        shifted = SimpleNamespace(get_model_object=lambda key: model_sampling)
        slg_model = SimpleNamespace(get_model_object=lambda key: model_sampling)
        runtime = ComfyRuntime.__new__(ComfyRuntime)
        runtime.torch = torch
        runtime.advanced = SimpleNamespace(
            ModelSamplingSD3=lambda: SimpleNamespace(
                patch=lambda model, **k: events.append(("shift", model, k)) or (shifted,)
            )
        )
        runtime.slg = SimpleNamespace(
            SkipLayerGuidanceDiTSimple=SimpleNamespace(
                execute=lambda model, **k: events.append(("slg", model, k)) or (slg_model,)
            )
        )
        runtime.samplers = SimpleNamespace(
            calculate_sigmas=lambda ms, name, steps: (
                events.append(("schedule", ms, name, steps)) or torch.linspace(1, 0, 4)
            ),
            sampler_object=lambda name: name,
        )

        def sample(*args, **kwargs):
            captured["args"], captured["kwargs"] = args, kwargs
            kwargs["callback"](0, torch.zeros(1), torch.zeros(1), 3)
            return args[7]

        runtime.sample_module = SimpleNamespace(sample_custom=sample)
        runtime.last_metadata = {}
        request = core.validate_request({"prompt": "cat", "width": 512, "height": 768, "seed": 7})
        progress = []
        result = runtime.sample(model, "positive", "negative", request, lambda *a: progress.append(a), lambda: None)
        args = captured["args"]
        self.assertEqual(events[0], ("shift", model, {"shift": 5.0, "multiplier": 1.0}))
        self.assertEqual(
            events[1],
            ("slg", shifted, {"start_percent": 0.0, "end_percent": 1.0, "double_layers": "", "single_layers": "22"}),
        )
        self.assertIs(args[0], slg_model)
        self.assertEqual((args[3], args[5], args[6]), ("euler", "positive", "negative"))
        self.assertEqual(tuple(result.shape), (1, 128, 48, 32))
        self.assertEqual(args[1].dtype, torch.float32)
        self.assertEqual(args[1].device.type, "cpu")
        expected = torch.randn(
            result.shape, generator=torch.Generator(device="cpu").manual_seed(7), dtype=torch.float32
        )
        self.assertTrue(torch.equal(args[1], expected))
        self.assertEqual(progress, [("sampling", 1, 3)])
        self.assertEqual(runtime.last_metadata["actual_steps"], 3)

    def test_vae_is_constructed_in_fp32_before_loading_statistics(self):
        import torch

        runtime = ComfyRuntime.__new__(ComfyRuntime)
        runtime.torch = torch
        calls = []
        vae = SimpleNamespace(throw_exception_if_invalid=lambda: None)
        runtime.sd = SimpleNamespace(VAE=lambda **k: calls.append(k) or vae)
        runtime.utils = SimpleNamespace(load_torch_file=lambda *a, **k: ({"statistics": "fp32"}, {"format": "pt"}))
        self.assertIs(runtime.load_vae(Path("flux2.safetensors")), vae)
        self.assertIs(calls[0]["dtype"], torch.float32)
        self.assertEqual(calls[0]["sd"], {"statistics": "fp32"})

    def test_decode_selects_budgeted_tiles_and_rejects_nonfinite_pixels(self):
        import torch

        runtime = ComfyRuntime.__new__(ComfyRuntime)
        runtime.torch = torch
        runtime.management = SimpleNamespace(get_free_memory=lambda device: 6 * 1024**3)
        events = []
        vae = SimpleNamespace(
            device="gpu",
            vae_dtype=torch.float32,
            memory_used_decode=lambda shape, dtype: 2178 * 64 * 4 * 4 * shape[-2] * shape[-1],
            decode=lambda latent: self.fail("full decode exceeds the budget"),
            decode_tiled=lambda *a, **k: events.append(k) or torch.zeros(1, 16, 16, 3),
        )
        self.assertEqual(runtime.decode(vae, torch.zeros(1, 128, 135, 240)).size, (16, 16))
        self.assertEqual(events, [{"tile_x": 32, "tile_y": 32, "overlap": 8}])
        vae.decode_tiled = lambda *a, **k: torch.full((1, 16, 16, 3), float("inf"))
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            runtime.decode(vae, torch.zeros(1, 128, 135, 240))

    def test_raw_vae_nonfinite_values_fail_before_comfy_clamps_them(self):
        import torch

        first_stage = SimpleNamespace(decode=lambda *a, **k: torch.full((1, 3, 8, 8), float("inf")))
        vae = SimpleNamespace(first_stage_model=first_stage)
        original = first_stage.decode
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            with finite_vae_decode(vae):
                first_stage.decode(None).clamp_(0, 1)
        self.assertIs(first_stage.decode, original)

    def test_generate_encodes_before_dit_and_applies_loras_to_fresh_model(self):
        import torch

        runtime = ComfyRuntime.__new__(ComfyRuntime)
        runtime.root = Path("models")
        runtime.torch = torch
        runtime.mapper = object()
        runtime.last_metadata = {}
        events = []
        base, patched = object(), object()
        from unittest.mock import Mock

        legacy_getter = Mock(side_effect=AttributeError("manual_cast_dtype"))
        clip = SimpleNamespace(
            cond_stage_model=SimpleNamespace(dtypes={torch.bfloat16}),
            patcher=SimpleNamespace(get_model_object=legacy_getter),
        )
        runtime.load_clip = lambda *a: events.append("clip") or clip
        runtime.encode_text = lambda clip, text: (events.append(("encode", text)) or [[torch.zeros(1), {}]], 8)
        runtime.offload = lambda: events.append("offload")
        runtime.load_dit = lambda *a: events.append("dit") or base
        runtime.sample = lambda model, *a: events.append(("sample", model)) or torch.zeros(1, 128, 32, 32)
        runtime.load_vae = lambda *a: events.append("vae") or "vae"
        runtime.decode = lambda *a: Image.new("RGB", (512, 512))
        request = core.validate_request({"prompt": "cat", "negative_prompt": "blur", "width": 512, "height": 512})
        with (
            patch.object(
                core, "asset_paths", return_value={"dit": Path("dit"), "text_encoder": Path("te"), "vae": Path("vae")}
            ),
            patch("modules_forge.moss_image.inference.lora.bind"),
            patch("modules_forge.moss_image.inference.lora.apply", return_value=(patched, 2)) as apply,
            patch("modules_forge.moss_image.inference.lora.record", return_value=[]),
        ):
            runtime.generate(request, lambda *a: None, lambda: None)
        self.assertLess(events.index(("encode", "blur")), events.index("dit"))
        self.assertLess(events.index("offload"), events.index("dit"))
        apply.assert_called_once_with(base, runtime.root, request, runtime.mapper, torch)
        self.assertIn(("sample", patched), events)
        self.assertEqual(runtime.last_metadata["lora_applied_targets"], 2)
        self.assertEqual(runtime.last_metadata["text_compute_precision"], "bfloat16")
        legacy_getter.assert_not_called()
        self.assertEqual(events[-1], "offload")

    def test_worker_reuses_runner_and_closes_after_malformed_job(self):
        from tools import moss_image_worker as worker

        runtime = SimpleNamespace(run=lambda *a: {"ok": True}, close=lambda: None)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            request_file = directory / "request.json"
            request_file.write_text('{"prompt": "cat"}', encoding="utf-8")
            with (
                patch.object(worker, "_runner", None),
                patch.object(worker, "_root", None),
                patch("modules_forge.moss_image.inference.Runner", return_value=runtime) as construct,
                patch.object(runtime, "close") as close,
            ):
                payload = {"root": str(directory), "job_dir": str(directory)}
                self.assertEqual(worker.resident_run(payload), {"ok": True})
                self.assertEqual(worker.resident_run(payload), {"ok": True})
                self.assertEqual(construct.call_count, 1)
                request_file.write_text("invalid json", encoding="utf-8")
                with self.assertRaises(ValueError):
                    worker.resident_run(payload)
                close.assert_called_once()
                self.assertIsNone(worker._runner)


if __name__ == "__main__":
    unittest.main()
