"""Reference snapshots, explicit edit modes and distilled generation contracts."""

import base64
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from modules_forge.moss_image import core


class RequestTests(unittest.TestCase):
    def test_distilled_lora_uses_normal_settings_without_automatic_changes(self):
        for steps in (4, 8):
            request = core.validate_request(
                {
                    "prompt": "anime",
                    "steps": steps,
                    "cfg": 1,
                    "skip_layer": False,
                    "loras": [
                        {"name": "presets/lora_latest.safetensors", "strength": 0.7},
                        {"name": "style.safetensors", "strength": 0.5},
                    ],
                }
            )
            self.assertEqual((request["steps"], request["cfg"]), (steps, 1))
            self.assertEqual(len(request["loras"]), 2)
            self.assertEqual(core.validate_request(request, internal=True), request)
            self.assertNotIn("profile", request)

    def test_edit_requires_images_and_edit_lora_and_does_not_allow_stale_text_images(self):
        image = Image.new("RGB", (512, 768))
        request = core.validate_request({"prompt": "blue scarf", "mode": "img2img", "images": [image]})
        self.assertEqual((request["edit_method"], request["denoise"]), ("redraw", 0.65))
        for changes in (
            {"mode": "img2img"},
            {"images": [image]},
            {"mode": "img2img", "images": [image, image]},
            {"mode": "img2img", "images": [image], "denoise": 0},
            {"mode": "img2img", "images": [image], "kv_cache": "yes"},
            {"mode": "img2img", "images": [image], "edit_method": "reference"},
            {"mode": "img2img", "images": ["C:/private.png"]},
        ):
            with self.subTest(keys=list(changes)), self.assertRaises(ValueError):
                core.validate_request({"prompt": "cat", **changes})
        request = core.validate_request(
            {
                "prompt": "Picture 1 wears Picture 2",
                "mode": "img2img",
                "edit_method": "reference",
                "images": [image, image],
                "loras": [{"name": "edit.safetensors", "strength": 1}],
            }
        )
        self.assertEqual(len(request["images"]), 2)

    def test_setup_plan_includes_distillation_only_when_selected(self):
        from tools.setup_moss_image import download_plan

        standard = download_plan()
        fast = download_plan(with_dmd=True)
        self.assertEqual(len(fast), len(standard) + 1)
        spec = next(spec for spec in fast if spec["path"] == "loras/presets/lora_latest.safetensors")
        self.assertEqual(spec["revision"], "f95df83b5e90ae91ea412434b3f9b7f1c8c6314f")
        self.assertEqual(spec["sha256"], "be82ab2bef78194651815bb6de443cec67553e01c14559f2ad365449445f0452")


class ReferenceTests(unittest.TestCase):
    def test_upload_decode_exif_alpha_snapshots_and_rejects_paths_or_urls(self):
        from modules_forge.moss_image import references

        image = Image.new("RGBA", (40, 32), (255, 0, 0, 0))
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        images = references.decode_uploads([base64.b64encode(stream.getvalue()).decode()])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            request = {"images": images}
            references.snapshot(root, request)
            self.assertEqual(request["images"], ["reference-0.png"])
            self.assertEqual(
                request["reference_sha256"], [hashlib.sha256((root / "reference-0.png").read_bytes()).hexdigest()]
            )
            loaded = references.load(root, request)
            self.assertEqual(loaded[0].getpixel((0, 0)), (255, 255, 255))
            Image.new("RGB", (40, 32), "blue").save(root / "reference-0.png")
            with self.assertRaisesRegex(ValueError, "参照画像"):
                references.load(root, request)
        for value in (
            ["C:/private.png"],
            ["https://example.test/image.png"],
            [base64.b64encode(b"not an image").decode()],
            ["YQ=="] * 4,
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                references.decode_uploads(value)

    def test_internal_reference_filenames_and_hashes_cannot_escape_job(self):
        from modules_forge.moss_image import references

        with tempfile.TemporaryDirectory() as folder:
            for value in (["../secret.png"], ["reference-0.png", "reference-0.png"], ["C:/secret.png"]):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    references.load(Path(folder), {"images": value, "reference_sha256": ["0" * 64] * len(value)})

    def test_preparation_preserves_order_aspect_and_bounded_token_budget(self):
        from modules_forge.moss_image import references

        images = [Image.new("RGB", (1600, 800), "red"), Image.new("RGB", (800, 1600), "blue")]
        prepared = references.prepare(images)
        self.assertEqual(len(prepared), 2)
        self.assertGreater(prepared[0].width, prepared[0].height)
        self.assertLess(prepared[1].width, prepared[1].height)
        for image in prepared:
            self.assertEqual((image.width % 16, image.height % 16), (0, 0))
            self.assertLessEqual(image.width * image.height, 1024**2)
        self.assertEqual(images[0].size, (1600, 800))

    def test_extreme_aspect_reference_stays_within_pixel_budget_after_grid_snap(self):
        from modules_forge.moss_image import references

        for size in ((16, 100000), (100000, 16)):
            for budget, snap in ((1024**2, 16), (384**2, 32)):
                width, height = references.fit_size(size, budget, snap=snap)
                self.assertLessEqual(width * height, budget)
                self.assertEqual((width % snap, height % snap), (0, 0))

    def test_service_snapshots_before_worker_and_png_records_only_job_relative_names(self):
        from modules_forge.moss_image.service import Service

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            service = Service(root, outputs=root / "outputs")
            lease = Mock()
            lease.acquire.return_value = True
            with (
                patch("modules_forge.moss_image.service.model_ready", return_value=True),
                patch("modules_forge.gpu_ownership.GPUOwnership", return_value=lease),
                patch("modules_forge.moss_image.service.threading.Thread"),
            ):
                identifier = service.start(
                    {"prompt": "blue scarf", "mode": "img2img", "images": [Image.new("RGB", (512, 512))]}
                )
            directory = service.outputs / identifier
            request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
            self.assertEqual(request["images"], ["reference-0.png"])
            self.assertEqual(len(request["reference_sha256"]), 1)
            self.assertNotIn(str(root), json.dumps(request))


class SamplingTests(unittest.TestCase):
    def test_redraw_keeps_text_only_conditioning_and_samples_from_source_latent(self):
        import torch

        from modules_forge.moss_image.inference import ComfyRuntime

        runtime = ComfyRuntime.__new__(ComfyRuntime)
        runtime.root, runtime.torch, runtime.mapper = Path("models"), torch, object()
        latent = torch.ones(1, 128, 32, 32)
        clip = SimpleNamespace(cond_stage_model=SimpleNamespace(dtypes={torch.bfloat16}))
        runtime.load_clip = Mock(return_value=clip)
        runtime.encode_text = Mock(return_value=([[torch.zeros(1, 4, 16), {}]], 4))
        runtime.load_vae = Mock(return_value=SimpleNamespace(encode=lambda pixels: latent))
        runtime.offload = Mock()
        runtime.load_dit = Mock(return_value=object())
        runtime.sample = Mock(return_value=latent)
        runtime.decode = Mock(return_value=Image.new("RGB", (512, 512)))
        request = core.validate_request(
            {
                "prompt": "blue scarf",
                "mode": "img2img",
                "images": [Image.new("RGB", (512, 512))],
                "width": 512,
                "height": 512,
            }
        )
        with (
            patch.object(
                core, "asset_paths", return_value={"dit": Path("dit"), "text_encoder": Path("te"), "vae": Path("vae")}
            ),
            patch("modules_forge.moss_image.inference.lora.bind"),
            patch("modules_forge.moss_image.inference.lora.apply", side_effect=lambda model, *args: (model, 0)),
        ):
            runtime.generate(request, lambda *args: None, lambda: None, images=request["images"])
        runtime.load_clip.assert_called_once_with(Path("te"), "int8")
        self.assertTrue(all(not call.kwargs for call in runtime.encode_text.call_args_list))
        self.assertIs(runtime.sample.call_args.kwargs["initial_latent"], latent)
        self.assertEqual(runtime.last_metadata["vision_sizes"], [])

    def test_reference_vision_markers_are_inserted_in_order_before_encoding(self):
        import torch

        from modules_forge.moss_image.inference import ComfyRuntime

        captured = []
        images = [torch.zeros(1, 384, 384, 3), torch.zeros(1, 192, 384, 3)]

        def tokenize(text, **kwargs):
            captured.append(text)
            return [[(1, 1.0), (248056, 1.0), (2, 1.0), (248056, 1.0)]]

        tokens_seen = []
        clip = SimpleNamespace(
            tokenizer=SimpleNamespace(qwen35_9b=SimpleNamespace(tokenize_with_weights=tokenize)),
            encode_from_tokens_scheduled=lambda tokens: tokens_seen.append(tokens) or [[torch.zeros(1, 218, 4), {}]],
        )
        runtime = ComfyRuntime.__new__(ComfyRuntime)
        runtime.torch = torch
        encoded, count = runtime.encode_text(clip, "swap outfits", images=images)
        self.assertIn("Picture 1: <|vision_start|><|image_pad|><|vision_end|>Picture 2:", captured[0])
        self.assertIs(tokens_seen[0]["qwen35_9b"][0][1][0]["data"], images[0])
        self.assertIs(tokens_seen[0]["qwen35_9b"][0][3][0]["data"], images[1])
        self.assertEqual(count, 218)
        self.assertEqual(encoded[0][0].device.type, "cpu")

    def test_denoise_schedule_retains_requested_number_of_effective_steps(self):
        import torch

        from modules_forge.moss_image.inference import ComfyRuntime

        runtime = ComfyRuntime.__new__(ComfyRuntime)
        runtime.torch = torch
        model = SimpleNamespace(get_model_object=lambda key: None)
        runtime.samplers = SimpleNamespace(
            calculate_sigmas=lambda sampling, name, steps: torch.linspace(1, 0, steps + 1)
        )
        request = {"steps": 8, "denoise": 0.5}
        sigmas = runtime.sampling_sigmas(model, request)
        self.assertEqual(len(sigmas), 9)
        self.assertAlmostEqual(float(sigmas[0]), 0.5)
        self.assertEqual(float(sigmas[-1]), 0)


if __name__ == "__main__":
    unittest.main()
