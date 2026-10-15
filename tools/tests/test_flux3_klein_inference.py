import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from modules_forge.flux3_klein.conditioning import pup_request
from modules_forge.flux3_klein.inference import (
    Cancelled,
    ComfyRuntime,
    Runner,
    greedy_logits,
    greedy_with_presence,
    official_sigmas,
)


class FakeRuntime:
    def __init__(self):
        self.events = []

    def ensure_precision(self, precision):
        self.events.append(("precision", precision))

    def verify_checkpoint(self, path, model):
        self.events.append(("verify", model))

    def load_clip(self, path, kind):
        self.events.append(("clip", kind))
        return kind

    def upsample(self, clip, spec, system_prompt, check):
        self.events.append(("pup", spec.presence_penalty))
        return "A detailed red balloon."

    def encode_text(self, clip, prompt):
        self.events.append(("text", prompt))
        return prompt

    def load_vae(self, path):
        self.events.append(("vae",))
        return object()

    def encode_reference(self, vae, image):
        return image.size

    def load_dit(self, path, model):
        self.events.append(("dit", model))
        return object()

    def sample(self, model, positive, negative, references, width, height, sigmas, cfg, seed, callback):
        self.events.append(("sample", negative, list(sigmas), cfg, references))
        callback(0, len(sigmas) - 1)
        return (width, height)

    def decode(self, vae, latent):
        return Image.new("RGB", latent, "red")

    def offload(self):
        self.events.append(("offload",))

    def close(self):
        self.events.append(("close",))


class InferenceTests(unittest.TestCase):
    def make_paths(self, root):
        paths = {key: root / key for key in ("dit", "text_encoder", "vae", "pup")}
        paths["pup_prompts"] = root / "system_prompts.json"
        paths["pup_prompts"].write_text(
            json.dumps({"t2i": "expand", "t2i_bbox": "layout", "i2i": "edit", "i2i_bbox": "edit boxes"}),
            encoding="utf-8",
        )
        return paths

    def test_arbitrary_steps_use_exact_official_shift(self):
        sigmas = official_sigmas(37)
        self.assertEqual(len(sigmas), 38)
        self.assertEqual(sigmas[-1], 0)
        t = 1 - 1 / 37
        self.assertAlmostEqual(sigmas[1], 6.93 * t / (1 + 5.93 * t), places=14)
        with self.assertRaises(ValueError):
            official_sigmas(0)

    def test_greedy_presence_changes_argmax_and_counts_once(self):
        import torch

        scores = torch.tensor([[3.0, 2.5, 1.0]])
        self.assertEqual(greedy_with_presence(scores, [0, 0], 1.5).item(), 1)
        self.assertEqual(scores[0, 0].item(), 3.0)

    def test_base_has_empty_unconditional_and_saves_png_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runtime = FakeRuntime()
            runner = Runner(root, runtime=runtime)
            with patch("modules_forge.flux3_klein.inference.model_paths", return_value=self.make_paths(root)):
                result = runner.run(
                    root / "job",
                    {
                        "model": "base",
                        "precision": "int8",
                        "prompt": "cat",
                        "seed": 7,
                        "steps": 37,
                        "cfg": 3.5,
                        "width": 64,
                        "height": 80,
                    },
                )
            sample = next(x for x in runtime.events if x[0] == "sample")
            self.assertEqual(sample[1], "")
            self.assertEqual(len(sample[2]), 38)
            self.assertFalse(any(x[:2] == ("clip", "pup") for x in runtime.events))
            self.assertEqual(Path(result["image_path"]).name, "output.png")
            saved = json.loads((root / "job" / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["seed"], 7)
            with Image.open(result["image_path"]) as image:
                self.assertEqual(image.size, (64, 80))
                self.assertEqual(json.loads(image.info["flux3_klein"])["model"], "base")

    def test_pup_runs_before_text_and_user_boxes_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runtime = FakeRuntime()
            boxes = [{"id": "obj_1", "desc": "balloon", "bbox": [0, 0, 1, 1]}]
            with patch("modules_forge.flux3_klein.inference.model_paths", return_value=self.make_paths(root)):
                result = Runner(root, runtime=runtime).run(
                    root / "job",
                    {
                        "model": "distilled",
                        "precision": "int8",
                        "prompt": "balloon",
                        "boxes": boxes,
                        "pup": True,
                        "seed": 4,
                        "width": 64,
                        "height": 64,
                    },
                )
            events = [event[0] for event in runtime.events]
            self.assertLess(events.index("pup"), events.index("text"))
            self.assertTrue(result["upsampled"])
            self.assertEqual(result["boxes"], boxes)
            self.assertIn('"bbox": [0, 0, 1000, 1000]', result["prompt"])

    def test_cancel_before_loading_never_produces_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            job = root / "job"
            job.mkdir()
            (job / "cancel").touch()
            runtime = FakeRuntime()
            with self.assertRaises(Cancelled):
                Runner(root, runtime=runtime).run(job, {"prompt": "cat"})
            self.assertFalse((job / "output.png").exists())
            self.assertFalse((job / "result.json").exists())
            self.assertFalse(runtime.events)

    def test_cancel_during_sampling_unloads_and_does_not_publish(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            job = root / "job"
            runtime = FakeRuntime()
            original = runtime.sample

            def cancel(*args):
                (job / "cancel").touch()
                return original(*args)

            runtime.sample = cancel
            with patch("modules_forge.flux3_klein.inference.model_paths", return_value=self.make_paths(root)):
                with self.assertRaises(Cancelled):
                    Runner(root, runtime=runtime).run(job, {"prompt": "cat", "pup": False})
            self.assertEqual(runtime.events[-1], ("offload",))
            self.assertFalse((job / "result.json").exists())
            self.assertFalse((job / "output.png").exists())

    def test_native_pup_greedy_penalty_images_and_restore(self):
        import torch

        class Transformer:
            def logits(self, hidden):
                return torch.tensor([[[3.0, 2.5, 1.0]]])

        model = Transformer()
        captured = {}

        class Clip:
            cond_stage_model = SimpleNamespace(qwen35_9b=SimpleNamespace(transformer=model))

            def tokenize(self, prompt, **kwargs):
                captured.update(kwargs)
                return "tokens"

            def generate(self, tokens, **kwargs):
                captured["generate"] = kwargs
                return [int(model.logits(None).argmax()) for _ in range(3)]

            def decode(self, tokens):
                captured["generated"] = tokens
                return "caption<|im_end|>"

        runtime = ComfyRuntime.__new__(ComfyRuntime)
        runtime.torch = torch
        self.assertEqual(runtime.upsample(Clip(), pup_request("cat", []), "system", lambda: None), "caption")
        self.assertEqual(captured["generated"], [0, 1, 0])
        self.assertFalse(captured["generate"]["do_sample"])
        self.assertFalse(captured["generate"]["mtp"])
        self.assertEqual(captured["generate"]["max_length"], 3584)
        self.assertNotIn("logits", model.__dict__)
        images = [Image.new("RGB", (64, 96)), Image.new("RGB", (96, 64))]
        runtime.upsample(Clip(), pup_request("edit", images), "system", lambda: None)
        self.assertNotEqual(captured["images"][0].shape, captured["images"][1].shape)

    def test_logits_restores_after_cancellation(self):
        class Transformer:
            def logits(self, hidden):
                return hidden

        model = Transformer()

        def cancel():
            raise Cancelled()

        with self.assertRaises(Cancelled):
            with greedy_logits(model, 1.5, cancel):
                model.logits(None)
        self.assertNotIn("logits", model.__dict__)

    def test_native_custom_sampler_receives_schedule_and_both_reference_conds(self):
        import torch

        runtime = ComfyRuntime.__new__(ComfyRuntime)
        runtime.torch = torch
        captured = {}

        def sample(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            kwargs["callback"](0, None, None, 37)
            return args[7]

        runtime.sample_module = SimpleNamespace(sample_custom=sample)
        runtime.samplers = SimpleNamespace(sampler_object=lambda name: name)
        cond = [[torch.zeros(1, 80, 20480), {}]]
        refs = [torch.zeros(1, 128, 4, 5)]
        progress = []
        result = runtime.sample(
            object(),
            cond,
            cond,
            refs,
            64,
            80,
            official_sigmas(37),
            3.5,
            5,
            lambda step, total: progress.append((step, total)),
        )
        args = captured["args"]
        self.assertEqual(args[3], "euler")
        self.assertEqual(args[4].shape[0], 38)
        self.assertIs(args[5][0][1]["reference_latents"], refs)
        self.assertIs(args[6][0][1]["reference_latents"], refs)
        self.assertEqual(tuple(result.shape), (1, 128, 5, 4))
        self.assertEqual(progress, [(0, 37)])

    def test_header_variant_mismatch_is_rejected(self):
        import torch
        from safetensors.torch import save_file

        runtime = ComfyRuntime.__new__(ComfyRuntime)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.safetensors"
            save_file({"txt_in.weight": torch.zeros(1), "ref_time_in.in_layer.weight": torch.zeros(1)}, str(path))
            runtime.verify_checkpoint(path, "distilled")
            with self.assertRaises(ValueError):
                runtime.verify_checkpoint(path, "base")

    def test_nvfp4_rejects_ampere_instead_of_silently_dequantizing(self):
        runtime = ComfyRuntime.__new__(ComfyRuntime)
        runtime.torch = SimpleNamespace(
            version=SimpleNamespace(hip=None),
            cuda=SimpleNamespace(is_available=lambda: True, get_device_capability=lambda: (8, 6)),
        )
        with self.assertRaisesRegex(ValueError, "Blackwell"):
            runtime.ensure_precision("nvfp4")
        runtime.ensure_precision("int8")

    def test_origin_guard_accepts_namespace_only_inside_pinned_checkout(self):
        from modules_forge.flux3_klein.inference import verify_module_origin

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            verify_module_origin(SimpleNamespace(__file__=None, __path__=[str(root / "comfy")]), root)
            with self.assertRaises(RuntimeError):
                verify_module_origin(SimpleNamespace(__file__=None, __path__=[str(root.parent / "other")]), root)

    def test_vae_is_constructed_in_fp32_before_loading_statistics(self):
        import torch

        runtime = ComfyRuntime.__new__(ComfyRuntime)
        runtime.torch = torch
        calls = []
        vae = SimpleNamespace(patcher=SimpleNamespace(), throw_exception_if_invalid=lambda: None)

        def construct(**kwargs):
            calls.append(kwargs)
            return vae

        runtime.sd = SimpleNamespace(VAE=construct, load_vae_patcher=lambda: None)
        runtime.utils = SimpleNamespace(
            load_torch_file=lambda path, **kwargs: ({"statistics": "fp32"}, {"format": "pt"})
        )
        runtime.nodes = SimpleNamespace(
            NODE_CLASS_MAPPINGS={"VAELoader": lambda: SimpleNamespace(load_vae=lambda name: ("bf16-default",))}
        )
        self.assertIs(runtime.load_vae(Path("flux2-vae.safetensors")), vae)
        self.assertIs(calls[0]["dtype"], torch.float32)
        self.assertEqual(calls[0]["sd"], {"statistics": "fp32"})

    def test_nonfinite_decode_pixels_are_rejected_before_uint8(self):
        import torch

        runtime = ComfyRuntime.__new__(ComfyRuntime)
        runtime.torch = torch
        runtime.management = SimpleNamespace(get_free_memory=lambda device: 1024)
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "non-finite"):
                vae = SimpleNamespace(
                    device="test-gpu",
                    vae_dtype=torch.float32,
                    memory_used_decode=lambda shape, dtype: 1,
                    decode=lambda latent, value=value: torch.full((1, 64, 64, 3), value),
                )
                runtime.decode(vae, torch.zeros(1, 128, 4, 4))

    def test_finite_black_decode_is_valid(self):
        import torch

        runtime = ComfyRuntime.__new__(ComfyRuntime)
        runtime.torch = torch
        runtime.management = SimpleNamespace(get_free_memory=lambda device: 1024)
        vae = SimpleNamespace(
            device="test-gpu",
            vae_dtype=torch.float32,
            memory_used_decode=lambda shape, dtype: 1,
            decode=lambda latent: torch.zeros(1, 64, 64, 3),
        )
        self.assertEqual(runtime.decode(vae, torch.zeros(1, 128, 4, 4)).getpixel((0, 0)), (0, 0, 0))

    def test_nonfinite_sampler_result_never_publishes_black_png(self):
        import torch

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runtime = FakeRuntime()
            runtime.sample = lambda *args: torch.full((1, 128, 4, 4), float("inf"))
            runtime.decode = lambda *args: Image.new("RGB", (64, 64), "black")
            with patch("modules_forge.flux3_klein.inference.model_paths", return_value=self.make_paths(root)):
                with self.assertRaisesRegex(RuntimeError, "non-finite"):
                    Runner(root, runtime=runtime).run(
                        root / "job", {"prompt": "black square", "width": 64, "height": 64}
                    )
            self.assertFalse((root / "job" / "output.png").exists())
            self.assertFalse((root / "job" / "result.json").exists())
            self.assertEqual(runtime.events[-1], ("offload",))

    def test_nonfinite_text_features_and_reference_latents_fail_early(self):
        import torch

        runtime = ComfyRuntime.__new__(ComfyRuntime)
        runtime.torch = torch
        runtime.management = SimpleNamespace(get_free_memory=lambda device: 1024)
        clip = SimpleNamespace(
            tokenize=lambda prompt: {"qwen3vl_4b": [[(1, 1.0)] * 80]},
            encode_from_tokens_scheduled=lambda tokens: [[torch.full((1, 80, 20480), float("nan")), {}]],
        )
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            runtime.encode_text(clip, "cat")
        vae = SimpleNamespace(
            device="test-gpu",
            vae_dtype=torch.float32,
            memory_used_encode=lambda shape, dtype: 1,
            encode=lambda pixels: torch.full((1, 128, 4, 4), float("inf")),
        )
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            runtime.encode_reference(vae, Image.new("RGB", (64, 64)))

    def test_decode_uses_budgeted_tiles_without_a_doomed_full_attempt(self):
        import torch

        gib = 1024**3
        coefficient = 2178 * 64 * 4 * 4
        for height, width, free, expected_tile in (
            (64, 64, 20 * gib, None),
            (135, 240, 20 * gib, 64),
            (135, 240, 6 * gib, 32),
            (135, 240, 2 * gib, 16),
        ):
            with self.subTest(height=height, width=width, free=free):
                runtime = ComfyRuntime.__new__(ComfyRuntime)
                runtime.torch = torch
                runtime.management = SimpleNamespace(get_free_memory=lambda device, free=free: free)
                calls = []

                def full(latent, calls=calls):
                    calls.append(("full", {}))
                    return torch.zeros(1, 16, 16, 3)

                def tiled(latent, calls=calls, **kwargs):
                    calls.append(("tiled", kwargs))
                    return torch.zeros(1, 16, 16, 3)

                vae = SimpleNamespace(
                    device="test-gpu",
                    vae_dtype=torch.float32,
                    memory_used_decode=lambda shape, dtype: coefficient * shape[-2] * shape[-1],
                    decode=full,
                    decode_tiled=tiled,
                )
                runtime.decode(vae, torch.zeros(1, 128, height, width))
                if expected_tile is None:
                    self.assertEqual(calls, [("full", {})])
                else:
                    self.assertEqual(
                        calls,
                        [("tiled", {"tile_x": expected_tile, "tile_y": expected_tile, "overlap": expected_tile // 4})],
                    )

    def test_decode_exact_budget_fit_and_insufficient_minimum(self):
        import torch

        runtime = ComfyRuntime.__new__(ComfyRuntime)
        runtime.torch = torch

        def full(latent):
            return torch.zeros(1, 16, 16, 3)

        vae = SimpleNamespace(
            device="test-gpu",
            vae_dtype=torch.float32,
            memory_used_decode=lambda shape, dtype: shape[-2] * shape[-1],
            decode=full,
            decode_tiled=lambda *args, **kwargs: self.fail("unexpected tiled decode"),
        )
        runtime.management = SimpleNamespace(get_free_memory=lambda device: 5120)
        runtime.decode(vae, torch.zeros(1, 128, 64, 64))
        runtime.management = SimpleNamespace(get_free_memory=lambda device: 100)
        with self.assertRaisesRegex(RuntimeError, "memory"):
            runtime.decode(vae, torch.zeros(1, 128, 64, 64))

    def test_constructor_needs_no_extra_nodes_or_server_initialization(self):
        with tempfile.TemporaryDirectory() as temp:
            repository = Path(temp)
            native = repository / "comfy/ldm/flux3/model.py"
            native.parent.mkdir(parents=True)
            native.touch()
            loads = []

            def forbidden(**kwargs):
                self.fail("extra-node initialization is not required by this adapter")

            modules = {
                "torch": SimpleNamespace(),
                "nodes": SimpleNamespace(__file__=str(repository / "nodes.py"), init_extra_nodes=forbidden),
                "comfy": SimpleNamespace(__file__=None, __path__=[str(repository / "comfy")]),
                "folder_paths": SimpleNamespace(add_model_folder_path=lambda *args, **kwargs: loads.append(args)),
            }
            for name in ("comfy.model_management", "comfy.sd", "comfy.utils", "comfy.sample", "comfy.samplers"):
                modules[name] = SimpleNamespace()
            with (
                patch("modules_forge.flux3_klein.core.comfy_path", return_value=repository),
                patch("modules_forge.flux3_klein.inference.sys.path", list(sys.path)),
                patch("modules_forge.flux3_klein.inference.importlib.import_module", side_effect=modules.__getitem__),
            ):
                ComfyRuntime(repository / "models")
            self.assertEqual([entry[0] for entry in loads], ["diffusion_models", "text_encoders", "vae"])

    def test_reference_encode_uses_pixel_tiles_with_the_same_memory_budget(self):
        import torch

        for height, width, free, expected_tile in (
            (1024, 1024, 20 * 1024**3, None),
            (2048, 2048, 20 * 1024**3, 1024),
            (2048, 2048, 6 * 1024**3, 512),
            (2048, 2048, 2 * 1024**3, 256),
            # Native three-pass tiling also uses a twice-wide tile. The short
            # image must not hide that larger peak behind a square estimate.
            (256, 4096, 700 * 1024**2, 256),
        ):
            with self.subTest(height=height, width=width, free=free):
                runtime = ComfyRuntime.__new__(ComfyRuntime)
                runtime.torch = torch
                runtime.management = SimpleNamespace(get_free_memory=lambda device, free=free: free)
                calls = []

                def full(pixels, calls=calls):
                    calls.append(("full", {}))
                    return torch.zeros(1, 128, 4, 4)

                def tiled(pixels, calls=calls, **kwargs):
                    calls.append(("tiled", kwargs))
                    return torch.zeros(1, 128, 4, 4)

                vae = SimpleNamespace(
                    device="test-gpu",
                    vae_dtype=torch.float32,
                    memory_used_encode=lambda shape, dtype: 1767 * 4 * shape[-2] * shape[-1],
                    encode=full,
                    encode_tiled=tiled,
                )
                runtime.encode_reference(vae, Image.new("RGB", (width, height)))
                if expected_tile is None:
                    self.assertEqual(calls, [("full", {})])
                else:
                    self.assertEqual(
                        calls,
                        [("tiled", {"tile_x": expected_tile, "tile_y": expected_tile, "overlap": expected_tile // 4})],
                    )

    def test_nonfinite_tiled_vae_results_are_rejected(self):
        import torch

        runtime = ComfyRuntime.__new__(ComfyRuntime)
        runtime.torch = torch
        runtime.management = SimpleNamespace(get_free_memory=lambda device: 20 * 1024**3)
        vae = SimpleNamespace(
            device="test-gpu",
            vae_dtype=torch.float32,
            memory_used_encode=lambda shape, dtype: 1767 * 4 * shape[-2] * shape[-1],
            memory_used_decode=lambda shape, dtype: 2178 * 64 * 4 * 4 * shape[-2] * shape[-1],
            encode_tiled=lambda *args, **kwargs: torch.full((1, 128, 4, 4), float("inf")),
            decode_tiled=lambda *args, **kwargs: torch.full((1, 16, 16, 3), float("nan")),
        )
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            runtime.encode_reference(vae, Image.new("RGB", (2048, 2048)))
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            runtime.decode(vae, torch.zeros(1, 128, 135, 240))


if __name__ == "__main__":
    unittest.main()
