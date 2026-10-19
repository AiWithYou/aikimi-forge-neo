"""Two-pass MOSS generation keeps output contracts and never publishes a draft."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from modules_forge.moss_image import core
from modules_forge.moss_image.api import public_result
from modules_forge.moss_image.inference import Cancelled, ComfyRuntime, Runner
from modules_forge.moss_image.service import Service
from tools.tests import test_moss_image_ui


def request(**changes):
    return core.validate_request(
        {
            "prompt": "an anime portrait",
            "width": 768,
            "height": 1024,
            "seed": 7,
            "steps": 8,
            "hires": {"base_width": 512, "base_height": 688, "steps": 4, "denoise": 0.35},
            **changes,
        }
    )


class HiresContractTests(unittest.TestCase):
    def test_display_sync_handles_invalid_raw_dimensions_without_blocking_off(self):
        screen = test_moss_image_ui.UITests().load_screen()
        blocks = screen.on_ui_tabs()[0][0]
        self.addCleanup(blocks.close)
        index = next(index for index, event in blocks.fns.items() if event.fn is screen.hires_controls)
        for width, height in ((511, 768), (1024, 4097), (None, None)):
            with self.subTest(off=(width, height)):
                result = asyncio.run(blocks.process_api(index, inputs=[False, width, height, 1.5]))["data"]
                self.assertEqual(result[0]["visible"], "hidden")
                self.assertEqual(result[1], "")
        for width in (511, 511.6, 4097):
            with self.subTest(invalid=width):
                result = asyncio.run(blocks.process_api(index, inputs=[True, width, 768, 1.5]))["data"]
                self.assertIs(result[0]["visible"], True)
                self.assertIn(str(width), result[1])
                self.assertNotIn("最終サイズ", result[1])
        result = asyncio.run(blocks.process_api(index, inputs=[True, 1024.4, 768, 1.5]))["data"]
        self.assertIn("1536×1152", result[1])

    def test_normal_requests_remain_single_pass_and_hires_is_explicit(self):
        self.assertNotIn("hires", core.validate_request({"prompt": "cat"}))
        value = request()
        self.assertEqual((value["width"], value["height"]), (768, 1024))
        self.assertEqual(value["hires"], {"base_width": 512, "base_height": 688, "steps": 4, "denoise": 0.35})
        self.assertEqual(core.validate_request(value, internal=True), value)
        defaults = request(hires={"base_width": 512, "base_height": 688})
        self.assertEqual(defaults["hires"]["steps"], 20)
        self.assertEqual(defaults["hires"]["denoise"], 0.35)

    def test_invalid_hires_settings_are_rejected_before_dispatch(self):
        for value in (
            None,
            False,
            [],
            {},
            {"base_width": 512, "base_height": 688, "private_path": "secret"},
            {"base_width": 511, "base_height": 688},
            {"base_width": 520, "base_height": 688},
            {"base_width": True, "base_height": 688},
            {"base_width": 784, "base_height": 688},
            {"base_width": 768, "base_height": 1024},
            {"base_width": 512, "base_height": 688, "steps": 0},
            {"base_width": 512, "base_height": 688, "steps": 201},
            {"base_width": 512, "base_height": 688, "steps": True},
            {"base_width": 512, "base_height": 688, "denoise": 0},
            {"base_width": 512, "base_height": 688, "denoise": 1.01},
            {"base_width": 512, "base_height": 688, "denoise": float("nan")},
            {"base_width": 512, "base_height": 688, "denoise": True},
        ):
            with self.subTest(hires=value), self.assertRaises(ValueError):
                request(hires=value)

    def test_img2img_cannot_inadvertently_use_a_retained_hires_request(self):
        with self.assertRaisesRegex(ValueError, "txt2img"):
            request(mode="img2img", images=[Image.new("RGB", (512, 512))])

    def test_denoise_rejects_unsafe_schedules_and_accepts_ui_boundaries(self):
        image = Image.new("RGB", (512, 512))
        for denoise in (5e-324, 1e-9, 0.009999):
            for value in (
                {"prompt": "cat", "mode": "img2img", "images": [image], "denoise": denoise},
                {"prompt": "cat", "width": 768, "height": 1024,
                 "hires": {"base_width": 512, "base_height": 688, "denoise": denoise}},
            ):
                with self.subTest(denoise=denoise, mode=value.get("mode", "hires")), self.assertRaises(ValueError):
                    core.validate_request(value)
        for denoise in (0.01, 0.35, 1):
            with self.subTest(denoise=denoise):
                self.assertEqual(request(hires={"base_width": 512, "base_height": 688,
                                              "denoise": denoise})["hires"]["denoise"], denoise)
                redraw = core.validate_request(
                    {"prompt": "cat", "mode": "img2img", "images": [image], "denoise": denoise}
                )
                self.assertEqual(redraw["denoise"], denoise)

    def test_ui_size_preview_and_submission_use_the_same_grid_and_limits(self):
        for dimensions, scale, expected in (
            ((1024, 1024), 1.5, (1536, 1536)),
            ((1360, 768), 1.5, (2048, 1152)),
            ((512, 512), 1.05, (544, 544)),
            ((1024, 1024), 4, (4096, 4096)),
        ):
            with self.subTest(dimensions=dimensions, scale=scale):
                self.assertEqual(core.hires_output_size(*dimensions, scale), expected)
        for width, height, scale in ((1024, 1024, 1), (1024, 1024, True), (4096, 512, 1.05), (512, 512, 4.01)):
            with self.subTest(width=width, height=height, scale=scale), self.assertRaises(ValueError):
                core.hires_output_size(width, height, scale)

    def test_ui_submits_final_dimensions_and_preserves_base_and_shared_settings(self):
        screen = test_moss_image_ui.UITests().load_screen()
        with patch.object(screen.SERVICE, "start", return_value="job") as start:
            screen.start(
                {},
                "cat",
                "blur",
                1024,
                768,
                23,
                4.5,
                2.5,
                7,
                False,
                hires_enabled=True,
                hires_scale=1.5,
                hires_steps=12,
                hires_denoise=0.25,
            )
        value = start.call_args.args[0]
        self.assertEqual((value["width"], value["height"]), (1536, 1152))
        self.assertEqual(value["hires"], {"base_width": 1024, "base_height": 768, "steps": 12, "denoise": 0.25})
        self.assertEqual((value["steps"], value["cfg"], value["shift"], value["seed"]), (23, 4.5, 2.5, 7))
        with patch.object(screen.SERVICE, "start", return_value="job") as start:
            screen.start(
                {},
                "cat",
                "",
                512,
                512,
                8,
                1,
                5,
                7,
                True,
                mode="img2img",
                images=[Image.new("RGB", (512, 512))],
                hires_enabled=True,
            )
        self.assertNotIn("hires", start.call_args.args[0], "hidden txt2img controls must not affect img2img")

    def test_ui_has_opt_in_controls_bound_to_generation_and_clear_stage_progress(self):
        screen = test_moss_image_ui.UITests().load_screen()
        blocks = screen.on_ui_tabs()[0][0]
        self.addCleanup(blocks.close)
        components = {item.elem_id: item for item in blocks.blocks.values() if item.elem_id}
        enabled = components["moss-image-hires-enabled"]
        self.assertFalse(enabled.value)
        self.assertEqual(components["moss-image-hires-settings"].visible, "hidden")
        run = next(event for event in blocks.fns.values() if event.fn is screen.start)
        for suffix in ("enabled", "scale", "steps", "denoise"):
            self.assertIn(components["moss-image-hires-" + suffix], run.inputs)
        self.assertIn("1536×1536", screen.hires_preview(True, 1024, 1024, 1.5))
        self.assertIn("4096", screen.hires_preview(True, 4096, 4096, 1.5))
        for stage, expected in (("hires_upscaling", "拡大"), ("hires_encoding", "拡大"), ("hires_sampling", "3/4")):
            with patch.object(
                screen.SERVICE, "poll", return_value={"status": "running", "stage": stage, "current": 3, "total": 4}
            ):
                self.assertIn(expected, screen.poll({"job": "job"})[1])


class HiresRuntimeTests(unittest.TestCase):
    def runtime(self):
        import torch

        runtime = ComfyRuntime.__new__(ComfyRuntime)
        runtime.root = Path("models")
        runtime.torch = torch
        runtime.mapper = object()
        runtime.offload = Mock()
        runtime.load_clip = Mock(
            return_value=SimpleNamespace(cond_stage_model=SimpleNamespace(dtypes={torch.bfloat16}))
        )
        runtime.encode_text = Mock(return_value=([[torch.zeros(1), {}]], 8))
        runtime.load_dit = Mock(return_value=object())
        runtime.load_vae = Mock(return_value=object())
        runtime.encode_image = Mock(return_value=torch.full((1, 128, 64, 48), 0.25))
        runtime.decode = Mock(
            side_effect=lambda vae, latent: Image.new("RGB", (latent.shape[-1] * 16, latent.shape[-2] * 16), "coral")
        )

        def sample(model, positive, negative, settings, progress, check, **kwargs):
            check()
            runtime.last_metadata.update(actual_steps=settings["steps"], sampler="euler", scheduler="sgm_uniform")
            progress("sampling", settings["steps"], settings["steps"])
            return torch.zeros(1, 128, settings["height"] // 16, settings["width"] // 16)

        runtime.sample = Mock(side_effect=sample)
        return runtime

    def generate(self, runtime, value, progress=lambda *args: None, check=lambda: None):
        with (
            patch.object(
                core, "asset_paths", return_value={"dit": Path("dit"), "text_encoder": Path("text"), "vae": Path("vae")}
            ),
            patch("modules_forge.moss_image.inference.lora.bind"),
            patch(
                "modules_forge.moss_image.inference.lora.apply", side_effect=lambda model, *args: (model, 2)
            ) as apply,
        ):
            result = runtime.generate(value, progress, check)
        return result, apply

    def test_second_pass_uses_upscaled_image_latent_and_reuses_model_conditioning_seed_and_lora(self):
        runtime = self.runtime()
        value = request()
        stages = []
        image, apply = self.generate(runtime, value, lambda stage, *args: stages.append(stage))
        self.assertEqual(image.size, (768, 1024))
        first, second = runtime.sample.call_args_list
        self.assertEqual((first.args[3]["width"], first.args[3]["height"], first.args[3]["steps"]), (512, 688, 8))
        self.assertEqual((second.args[3]["width"], second.args[3]["height"], second.args[3]["steps"]), (768, 1024, 4))
        self.assertEqual(second.args[3]["denoise"], 0.35)
        self.assertEqual((first.args[3]["seed"], second.args[3]["seed"]), (7, 7))
        self.assertIs(first.args[0], second.args[0])
        self.assertIs(first.args[1], second.args[1])
        self.assertIs(second.kwargs["initial_latent"], runtime.encode_image.return_value)
        self.assertEqual(runtime.encode_image.call_args.args[1].size, (768, 1024))
        runtime.load_clip.assert_called_once()
        runtime.load_dit.assert_called_once()
        apply.assert_called_once()
        self.assertEqual(runtime.last_metadata["lora_applied_targets"], 2)
        passes = runtime.last_metadata["sampling_passes"]
        self.assertEqual(
            [(item["width"], item["height"], item["actual_steps"]) for item in passes], [(512, 688, 8), (768, 1024, 4)]
        )
        self.assertEqual(runtime.last_metadata["actual_steps"], 12)
        self.assertEqual(runtime.last_metadata["upscaler"], "lanczos")
        self.assertIn("hires_sampling", stages)
        self.assertEqual(stages[-1], "decoding")

    def test_hires_encoding_is_cpu_fp32_finite_and_uses_the_native_vae_encoder(self):
        import torch

        runtime = ComfyRuntime.__new__(ComfyRuntime)
        runtime.torch = torch
        vae = SimpleNamespace(encode=Mock(return_value=torch.ones(1, 128, 32, 48, dtype=torch.bfloat16)))
        latent = runtime.encode_image(vae, Image.new("RGB", (768, 512), "coral"))
        pixels = vae.encode.call_args.args[0]
        self.assertEqual((pixels.shape, pixels.dtype), (torch.Size([1, 512, 768, 3]), torch.float32))
        self.assertEqual((latent.device.type, latent.dtype), ("cpu", torch.float32))
        vae.encode.return_value = torch.tensor([float("nan")])
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            runtime.encode_image(vae, Image.new("RGB", (512, 512)))

    def test_cancel_between_passes_does_not_start_second_sample(self):
        runtime = self.runtime()
        cancelled = False

        def progress(stage, *args):
            nonlocal cancelled
            if stage == "hires_encoding":
                cancelled = True

        def check():
            if cancelled:
                raise Cancelled("cancelled")

        with self.assertRaises(Cancelled):
            self.generate(runtime, request(), progress, check)
        self.assertEqual(runtime.sample.call_count, 1)
        runtime.offload.assert_called()

    def test_runner_and_service_agree_on_final_png_conditions_and_public_metadata(self):
        runtime = self.runtime()
        value = request()
        image, _ = self.generate(runtime, value)
        final_runtime = SimpleNamespace(
            generate=Mock(return_value=image), last_metadata={**runtime.last_metadata, "lora_applied_targets": 0}
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            core.atomic_json(directory / "request.json", value)
            with patch.object(core, "model_ready", return_value=True):
                result = Runner(directory, runtime=final_runtime).run(directory, value)
            Service.validate_result(directory)
            with Image.open(directory / "output.png") as saved:
                embedded = json.loads(saved.info["moss_image"])
                self.assertEqual(saved.size, (768, 1024))
            self.assertEqual(embedded["hires"], value["hires"])
            public = public_result({**result, "status": "complete"})
            self.assertEqual(public["sampling_passes"], embedded["sampling_passes"])
            self.assertEqual(public["hires"], embedded["hires"])
            self.assertEqual(public["upscaler"], "lanczos")
            self.assertNotIn(temporary, json.dumps(public))

    def test_cancelled_or_failed_second_pass_never_publishes_a_base_image(self):
        for error in (Cancelled("cancelled in pass 2"), RuntimeError("pass 2 failed")):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                runtime = SimpleNamespace(generate=Mock(side_effect=error))
                with patch.object(core, "model_ready", return_value=True), self.assertRaises(type(error)):
                    Runner(directory, runtime=runtime).run(directory, request())
                self.assertFalse((directory / "output.png").exists())
                self.assertFalse((directory / "result.json").exists())


if __name__ == "__main__":
    unittest.main()
