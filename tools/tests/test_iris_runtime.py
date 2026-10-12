"""Nonfinite model output must fail before clamping or conversion to pixels."""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from modules_forge.iris.img2img import ImageSchedule
from modules_forge.iris.runtime import Runner, check_model_output


class RuntimeTests(unittest.TestCase):
    def test_turbo_prepares_negative_text_only_for_manual_guidance(self):
        base = types.ModuleType("iris3b.text.base")
        base.TextEncoding = lambda embeddings, mask: SimpleNamespace(embeddings=embeddings, mask=mask)
        qwen = types.ModuleType("iris3b.text.qwen3_vl")
        qwen.Qwen3VLTextEncoder = object
        for cfg in (1, 2):
            with self.subTest(cfg=cfg):
                prompts, devices = [], []

                def encode(values, prompts=prompts):
                    prompts.extend(values)
                    return SimpleNamespace(embeddings=torch.ones(1, 2, 4), mask=torch.ones(1, 2))

                runner = Runner.__new__(Runner)
                runner.root, runner.precision = Path("runtime"), "int8"
                runner.cfg = SimpleNamespace(text_encoder=SimpleNamespace())
                runner.encoder = SimpleNamespace(encode=encode, to=devices.append)
                with (
                    patch.dict(sys.modules, {"iris3b.text.base": base, "iris3b.text.qwen3_vl": qwen}),
                    patch("torch.cuda.empty_cache"),
                ):
                    text = runner.prepare_text(
                        {"prompt": "fox", "negative_prompt": "dark", "cfg": cfg, "sampling": {"mode": "turbo"}}
                    )
                self.assertEqual(prompts, ["fox"] if cfg == 1 else ["fox", "dark"])
                self.assertEqual(devices, ["cuda", "cpu"])
                self.assertEqual(text.negative is None, cfg == 1)

    def test_turbo_uses_manual_settings_and_counts_guidance_calls(self):
        runner = Runner.__new__(Runner)
        runner.task, runner.precision = "generate", "int8"
        runner.root, runner.checkpoint, runner.source_identity = Path("runtime"), "base", {"weights": {}, "config": {}}
        runner.model = torch.nn.Identity()
        runner.model.cuda = lambda: runner.model
        runner.model.forward = lambda pixels: SimpleNamespace(x=pixels)
        runner.cfg = SimpleNamespace(flow=SimpleNamespace(num_train_timesteps=1000, prediction="v"))
        runner.prepare_text = lambda request: None
        runner.offload = lambda: None
        sampling = types.ModuleType("iris3b.sampling")
        sampling.generate = lambda *args, **kwargs: self.fail("Turbo must keep its sampler")
        received, progress = {}, []

        def turbo(model, text, prompts, height, width, steps, shift, generator, **kwargs):
            received.update(steps=steps, shift=shift, **kwargs)
            for _ in range(steps * 2):
                model(torch.zeros(1, 3, height, width))
            return torch.zeros(1, 3, height, width)

        from modules_forge.iris.core import atomic_json

        def record(path, value):
            if "current" in value:
                progress.append(value)
            atomic_json(path, value)

        request = {
            "prompt": "fox",
            "negative_prompt": "dark",
            "width": 16,
            "height": 16,
            "steps": 8,
            "cfg": 2,
            "seed": 42,
            "sampling": {"mode": "turbo", "default_steps": 4, "shift": 4},
        }
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.dict(sys.modules, {"iris3b.sampling": sampling}),
                patch("modules_forge.iris.turbo.generate_turbo", side_effect=turbo),
                patch("modules_forge.iris.runtime.atomic_json", side_effect=record),
                patch("torch.cuda.reset_peak_memory_stats"),
                patch("torch.cuda.synchronize"),
                patch("torch.cuda.max_memory_allocated", return_value=0),
                patch("torch.cuda.max_memory_reserved", return_value=0),
                patch("torch.Generator"),
                patch("torch.autocast"),
                patch("modules_forge.iris.runtime.checkpoint_identity", return_value=runner.source_identity),
            ):
                runner.run(directory, request)
            saved = json.loads((Path(directory) / "metadata.json").read_text())["settings"]
        self.assertEqual(received, {"steps": 8, "shift": 4, "cfg_scale": 2, "negative_prompt": "dark"})
        self.assertEqual((progress[-1]["current"], progress[-1]["total"]), (16, 16))
        self.assertEqual((saved["steps"], saved["cfg"], saved["negative_prompt"]), (8, 2, "dark"))

    def test_reported_time_and_completion_follow_model_release(self):
        clock = [0.0]
        runner = Runner.__new__(Runner)
        runner.task, runner.precision = "generate", "int8"
        runner.root, runner.checkpoint, runner.source_identity = Path("runtime"), "base", {"weights": {}, "config": {}}
        runner.model = torch.nn.Identity()
        runner.model.cuda = lambda: runner.model
        runner.cfg = SimpleNamespace(
            sample=SimpleNamespace(cfg_interval=(0, 1)),
            flow=SimpleNamespace(shift=4, num_train_timesteps=1000, prediction="v"),
        )
        runner.prepare_text = lambda request: None
        runner.offload = lambda: clock.__setitem__(0, 10.0)
        sampling = types.ModuleType("iris3b.sampling")
        sampling.generate = lambda *args, **kwargs: torch.zeros(1, 3, 16, 16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "request.json").write_text(
                json.dumps(
                    {
                        "prompt": "test",
                        "negative_prompt": "",
                        "width": 16,
                        "height": 16,
                        "steps": 1,
                        "cfg": 3,
                        "seed": 1,
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.dict(sys.modules, {"iris3b.sampling": sampling}),
                patch("modules_forge.iris.runtime.time.perf_counter", side_effect=lambda: clock[0]),
                patch("torch.cuda.reset_peak_memory_stats"),
                patch("torch.cuda.synchronize"),
                patch("torch.cuda.max_memory_allocated", return_value=0),
                patch("torch.cuda.max_memory_reserved", return_value=0),
                patch("torch.Generator"),
                patch("torch.autocast"),
                patch("modules_forge.iris.runtime.checkpoint_identity", return_value=runner.source_identity),
            ):
                result = runner.run(path)
            self.assertEqual(result["seconds"], 10.0)
            self.assertEqual(json.loads((path / "metadata.json").read_text())["seconds"], 10.0)
            self.assertEqual(
                {Path(file).name for file in result["files"]}, {"result.png", "request.json", "metadata.json"}
            )
            self.assertEqual(len(result["files"]), 3)
            self.assertTrue(all(Path(file).is_file() for file in result["files"]))

    def test_nonfinite_generation_model_output_fails_before_sampler_clamp(self):
        class PixelModel(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value

            def forward(self, pixels):
                return SimpleNamespace(x=torch.full_like(pixels, self.value))

            def cuda(self):
                return self

        for mode in ("txt2img", "img2img", "turbo"):
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(mode=mode, value=value), tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary)
                    runner = Runner.__new__(Runner)
                    runner.task, runner.precision = "generate", "int8"
                    runner.root, runner.checkpoint = Path("runtime"), "base"
                    runner.source_identity = {"weights": {}, "config": {}}
                    runner.model = PixelModel(value)
                    runner.cfg = SimpleNamespace(
                        sample=SimpleNamespace(cfg_interval=(0, 1)),
                        flow=SimpleNamespace(shift=4, num_train_timesteps=1000, prediction="v"),
                    )
                    runner.prepare_text = lambda request: None
                    released = []
                    runner.offload = lambda released=released: released.append(True)
                    request = {
                        "prompt": "test",
                        "negative_prompt": "",
                        "width": 16,
                        "height": 16,
                        "steps": 1,
                        "cfg": 3,
                        "seed": 1,
                    }
                    if mode == "img2img":
                        from PIL import Image

                        Image.new("RGB", (16, 16)).save(path / "input.png")
                        request.update(image_mode=mode, image=str(path / "input.png"), strength=1)
                    elif mode == "turbo":
                        request["sampling"] = {"mode": "turbo", "shift": 4}

                    def clamped(model, *args, **kwargs):
                        return model(torch.zeros(1, 3, 16, 16)).x.clamp(-1, 1)

                    sampling = types.ModuleType("iris3b.sampling")
                    sampling.generate = clamped
                    with (
                        patch.dict(sys.modules, {"iris3b.sampling": sampling}),
                        patch(
                            "modules_forge.iris.img2img.image_schedule", return_value=ImageSchedule(1, 1, 4, [0.9, 0])
                        ),
                        patch("modules_forge.iris.turbo.generate_turbo", side_effect=clamped),
                        patch("torch.cuda.reset_peak_memory_stats"),
                        patch("torch.cuda.synchronize"),
                        patch("torch.cuda.max_memory_allocated", return_value=0),
                        patch("torch.cuda.max_memory_reserved", return_value=0),
                        patch("torch.Generator"),
                        patch("torch.autocast"),
                        patch("modules_forge.iris.runtime.checkpoint_identity", return_value=runner.source_identity),
                        self.assertRaisesRegex(ValueError, "非有限値"),
                    ):
                        runner.run(path, request)
                    self.assertEqual(released, [True])
                    self.assertFalse((path / "result.png").exists())
                    self.assertEqual(len(runner.model._forward_hooks), 0)
                    self.assertEqual(len(runner.model._forward_pre_hooks), 0)

    def test_nonfinite_model_pixels_are_rejected(self):
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                check_model_output(None, (), SimpleNamespace(x=torch.full((1, 3, 8, 8), value)))
        pixels = torch.ones(1, 3, 8, 8)
        self.assertIsNone(check_model_output(None, (), SimpleNamespace(x=pixels)))
        torch.testing.assert_close(pixels, torch.ones_like(pixels))


if __name__ == "__main__":
    unittest.main()
