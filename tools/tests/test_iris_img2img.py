"""Img2img retains uploaded pixels and uses the matching partial flow schedule."""

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from PIL import Image

from modules_forge.iris.core import IrisError, validate_request
from modules_forge.iris.service import Service


class RequestTests(unittest.TestCase):
    def test_img2img_preserves_input_strength_and_arbitrary_full_ft(self):
        image = Image.new("RGB", (832, 1280))
        request = validate_request(
            {
                "prompt": "a red dress",
                "image_mode": "img2img",
                "image": image,
                "strength": 0.25,
                "checkpoint": "custom/illustration.safetensors",
                "precision": "normal",
            }
        )
        self.assertIs(request["image"], image)
        self.assertEqual(request["strength"], 0.25)
        self.assertEqual(request["checkpoint"], "custom/illustration.safetensors")
        for strength in (0, 1):
            self.assertEqual(validate_request({**request, "strength": strength})["strength"], strength)

    def test_img2img_requires_image_and_finite_valid_strength(self):
        valid = {"prompt": "a dress", "image_mode": "img2img", "image": Image.new("RGB", (2, 2))}
        for update in (
            {"image": None},
            {"strength": -0.1},
            {"strength": 1.1},
            {"strength": True},
            {"strength": float("nan")},
            {"strength": float("inf")},
            {"image_mode": "edit"},
        ):
            with self.subTest(update=update), self.assertRaises(IrisError):
                validate_request({**valid, **update})

    def test_txt2img_does_not_validate_or_keep_retained_image_strength(self):
        request = validate_request(
            {"prompt": "a dress", "image_mode": "txt2img", "image": object(), "strength": float("nan")}
        )
        self.assertEqual(request["image_mode"], "txt2img")
        self.assertNotIn("image", request)
        self.assertNotIn("strength", request)


class ServiceTests(unittest.TestCase):
    def test_input_is_snapshotted_before_worker_dispatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = Service(root)
            image = Image.new("RGBA", (3, 2), (255, 0, 0, 0))
            lease = Mock()
            lease.acquire.return_value = True
            with (
                patch("modules_forge.iris.service.ROOT", root),
                patch("modules_forge.iris.service.model_ready", return_value=True),
                patch("modules_forge.gpu_ownership.GPUOwnership", return_value=lease),
                patch("modules_forge.iris.service.threading.Thread") as thread,
            ):
                identifier = service.start(
                    {"prompt": "a dress", "image_mode": "img2img", "image": image, "strength": 0.25}
                )
            request = json.loads((root / "outputs/iris" / identifier / "request.json").read_text())
            self.assertEqual(request["image_mode"], "img2img")
            self.assertEqual(request["strength"], 0.25)
            with Image.open(request["image"]) as saved:
                self.assertEqual((saved.mode, saved.size, saved.getpixel((0, 0))), ("RGB", (3, 2), (255, 255, 255)))
            thread.return_value.start.assert_called_once()

    def test_turbo_is_rejected_before_gpu_or_dispatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            service = Service(Path(temporary))
            with (
                patch("modules_forge.iris.lora.snapshot", return_value=[]),
                patch("modules_forge.iris.lora.sampling_settings", return_value={"mode": "turbo", "default_steps": 4}),
                patch("modules_forge.gpu_ownership.GPUOwnership") as lease,
                self.assertRaisesRegex(IrisError, "Turbo"),
            ):
                service.start({"prompt": "dress", "image_mode": "img2img", "image": Image.new("RGB", (2, 2))})
            lease.assert_not_called()


class FlowSamplerTests(unittest.TestCase):
    def setUp(self):
        self.solver_module = types.ModuleType("iris3b.flow.solver")
        self.received = {}
        received = self.received

        class Solver:
            def __init__(self, model_fn, **kwargs):
                received["settings"] = kwargs

            @staticmethod
            def time_grid(steps, shift):
                return [0.9, 0.75, 0.4, 0.1, 0.0]

            def sample(self, z, cond, uncond, steps, order, shift):
                received.update(z=z, times=self.time_grid(steps, shift), steps=steps, order=order)
                return torch.zeros_like(z)

        self.solver_module.FlowDPMSolver = Solver
        self.model = torch.nn.Linear(1, 1)
        self.text = SimpleNamespace(
            encode=lambda prompts: SimpleNamespace(embeddings=torch.ones(1, 2, 4), mask=torch.ones(1, 2)),
            null=lambda prompt: SimpleNamespace(embeddings=torch.zeros(1, 2, 4), mask=torch.ones(1, 2)),
        )

    def test_partial_schedule_mixes_at_its_actual_start_sigma(self):
        from modules_forge.iris.img2img import generate_img2img, image_schedule

        pixels = torch.full((1, 3, 4, 4), 0.2)
        with patch.dict(sys.modules, {"iris3b.flow.solver": self.solver_module}):
            schedule = image_schedule(4, 0.5, 4)
            generate_img2img(
                self.model,
                self.text,
                ["dress"],
                pixels,
                schedule,
                cfg_scale=3,
                negative_prompt="blur",
                generator=torch.Generator().manual_seed(42),
                device="cpu",
            )
        noise = torch.randn(pixels.shape, generator=torch.Generator().manual_seed(42))
        torch.testing.assert_close(self.received["z"], 0.6 * pixels + 0.4 * noise)
        self.assertEqual(self.received["times"], [0.4, 0.1, 0.0])
        self.assertEqual(self.received["steps"], 2)
        self.assertEqual(self.received["settings"]["cfg_scale"], 3)

    def test_zero_strength_is_exact_input_without_text_or_model_calls(self):
        from modules_forge.iris.img2img import generate_img2img, image_schedule

        pixels = torch.rand(1, 3, 4, 4)
        with patch.dict(sys.modules, {"iris3b.flow.solver": self.solver_module}):
            schedule = image_schedule(4, 0, 4)
            result = generate_img2img(Mock(), Mock(), ["dress"], pixels, schedule, device="cpu")
        self.assertEqual(schedule.steps, 0)
        self.assertIs(result, pixels)
        self.assertFalse(self.received)

    def test_full_strength_delegates_to_identical_txt2img(self):
        from modules_forge.iris.img2img import generate_img2img, image_schedule

        sampling = types.ModuleType("iris3b.sampling")
        sampling.generate = Mock(return_value=torch.zeros(1, 3, 4, 4))
        with patch.dict(sys.modules, {"iris3b.flow.solver": self.solver_module, "iris3b.sampling": sampling}):
            schedule = image_schedule(4, 1, 4)
            generate_img2img(
                self.model,
                self.text,
                ["dress"],
                torch.ones(1, 3, 4, 4),
                schedule,
                cfg_scale=3,
                negative_prompt="blur",
                device="cpu",
            )
        self.assertEqual(sampling.generate.call_args.kwargs["steps"], 4)
        self.assertNotIn("noise", sampling.generate.call_args.kwargs)
        self.assertEqual(schedule.steps, 4)

    def test_small_nonzero_strength_has_one_step_and_terminal_zero(self):
        from modules_forge.iris.img2img import image_schedule

        with patch.dict(sys.modules, {"iris3b.flow.solver": self.solver_module}):
            schedule = image_schedule(4, 0.001, 4)
        self.assertEqual(schedule.steps, 1)
        self.assertEqual(schedule.times, [0.1, 0.0])

    def test_decimal_step_boundary_does_not_lose_a_step_to_float_rounding(self):
        from modules_forge.iris.img2img import image_schedule

        with (
            patch.dict(sys.modules, {"iris3b.flow.solver": self.solver_module}),
            patch.object(
                self.solver_module.FlowDPMSolver, "time_grid", return_value=torch.linspace(0.9, 0, 101).tolist()
            ),
        ):
            self.assertEqual(image_schedule(100, 0.58, 4).steps, 58)

    def test_invalid_ft_flow_grid_is_rejected(self):
        from modules_forge.iris.img2img import image_schedule

        for times in (
            [1, 0.75, 0.4, 0.1, 0],
            [0.9, 0.75, float("nan"), 0.1, 0],
            [0.9, 0.9, 0.4, 0.1, 0],
            [0.9, 0.75, 0.4, 0.1, 0.01],
        ):
            with (
                self.subTest(times=times),
                patch.dict(sys.modules, {"iris3b.flow.solver": self.solver_module}),
                patch.object(self.solver_module.FlowDPMSolver, "time_grid", return_value=times),
                self.assertRaisesRegex(ValueError, "時間刻み"),
            ):
                image_schedule(4, 0.5, 4)

    def test_nonfinite_solver_output_fails_before_clipping(self):
        from modules_forge.iris.img2img import generate_img2img, image_schedule

        for value in (float("inf"), float("-inf"), float("nan")):
            with (
                self.subTest(value=value),
                patch.dict(sys.modules, {"iris3b.flow.solver": self.solver_module}),
                patch.object(self.solver_module.FlowDPMSolver, "sample", return_value=torch.full((1, 3, 4, 4), value)),
                self.assertRaisesRegex(ValueError, "非有限"),
            ):
                generate_img2img(
                    self.model, self.text, ["dress"], torch.zeros(1, 3, 4, 4), image_schedule(4, 0.5, 4), device="cpu"
                )


class RuntimeTests(unittest.TestCase):
    def test_zero_strength_keeps_resized_rgb_and_skips_encoder_and_gpu_model(self):
        from modules_forge.iris.runtime import Runner

        runner = Runner.__new__(Runner)
        runner.task, runner.precision = "generate", "normal"
        runner.root, runner.checkpoint, runner.source_identity = (
            Path("runtime"),
            "custom.safetensors",
            {"weights": {}, "config": {}},
        )
        runner.model = torch.nn.Linear(1, 1)
        runner.model.cuda = Mock(side_effect=AssertionError("zero strength must not move the model to GPU"))
        runner.prepare_text = Mock(side_effect=AssertionError("zero strength must not encode text"))
        runner.offload = Mock()
        runner.cfg = SimpleNamespace(
            flow=SimpleNamespace(shift=4, prediction="v", num_train_timesteps=1000),
            sample=SimpleNamespace(cfg_interval=(0, 1)),
        )
        sampling = types.ModuleType("iris3b.sampling")
        sampling.generate = Mock(side_effect=AssertionError("zero strength must not sample"))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "input.png"
            Image.new("RGB", (3, 2), (21, 43, 65)).save(source)
            with (
                patch.dict(sys.modules, {"iris3b.sampling": sampling}),
                patch("torch.cuda.reset_peak_memory_stats"),
                patch("torch.cuda.synchronize"),
                patch("torch.cuda.max_memory_allocated", return_value=0),
                patch("torch.cuda.max_memory_reserved", return_value=0),
                patch("modules_forge.iris.runtime.checkpoint_identity", return_value=runner.source_identity),
            ):
                result = runner.run(
                    directory,
                    {
                        "image_mode": "img2img",
                        "strength": 0,
                        "image": str(source),
                        "width": 4,
                        "height": 4,
                        "steps": 100,
                    },
                )
            with Image.open(result["image"]) as image:
                self.assertEqual((image.size, image.getpixel((0, 0))), ((4, 4), (21, 43, 65)))
            metadata = json.loads((directory / "metadata.json").read_text())
            self.assertEqual((metadata["schedule"]["executed_steps"], metadata["schedule"]["forward_calls"]), (0, 0))
            self.assertEqual(metadata["input"]["resize"]["size"], [4, 4])
            self.assertIn(str(source), result["files"])
        runner.offload.assert_called_once()

    def test_mode_and_input_changes_reuse_the_same_full_ft_runner(self):
        from modules_forge.iris import core, runtime

        spec = importlib.util.spec_from_file_location("iris_img2img_worker", core.ROOT / "tools/iris_worker.py")
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            fake = Mock()
            with (
                patch.object(runtime, "Runner", return_value=fake) as build,
                patch.object(worker, "model_files", return_value=(Path("custom.safetensors"), Path("config.yaml"))),
                patch.object(worker, "checkpoint_identity", return_value={"sha256": "custom-full-ft"}),
                patch("modules_forge.iris.official_turbo.checkpoint_settings", return_value=None),
            ):
                for mode, strength in (("txt2img", None), ("img2img", 0.25), ("img2img", 0.65), ("txt2img", None)):
                    core.atomic_json(
                        directory / "request.json",
                        {
                            "task": "generate",
                            "precision": "normal",
                            "checkpoint": "custom.safetensors",
                            "image_mode": mode,
                            "strength": strength,
                            "image": str(directory / f"{strength}.png"),
                            "loras": [],
                        },
                    )
                    worker.resident_run({"root": temporary, "job_dir": temporary})
            build.assert_called_once()
            self.assertEqual(fake.run.call_count, 4)


class UITests(unittest.TestCase):
    def test_result_send_cannot_change_task_during_a_job(self):
        import gradio as gr

        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location(
            "test_iris_send_ui", root / "extensions-builtin/iris-studio/scripts/iris_studio.py"
        )
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"modules.script_callbacks": SimpleNamespace(on_ui_tabs=Mock())}):
            spec.loader.exec_module(module)
        with patch.object(module.local_assets, "selection", return_value={}):
            blocks = module.on_ui_tabs()[0][0]
        self.addCleanup(blocks.close)
        components = blocks.get_config_file()["components"]
        dependencies = blocks.get_config_file()["dependencies"]
        send_ids = {
            c["id"]
            for c in components
            if c["type"] == "button"
            and c["props"].get("value") in {"img2imgへ送る", "深度推定へ送る", "復元・4倍拡大へ送る"}
        }
        for operation in (module.start_with_loras, module.poll_with_loras):
            callback = next(fn for fn in blocks.fns.values() if fn.fn is operation)
            self.assertEqual(
                {c._id for c in callback.outputs[-3:]},
                send_ids,
                "Start and completion must update send controls directly",
            )
        for state, enabled in (
            ({}, False),
            ({"results": {"generate": {"image": "previous.png"}}}, True),
            ({"job": "running", "results": {"generate": {"image": "previous.png"}}}, False),
        ):
            self.assertEqual([value["interactive"] for value in module.send_controls(state)], [enabled] * 3)
        for label, destination in (
            ("img2imgへ送る", "generate"),
            ("深度推定へ送る", "depth"),
            ("復元・4倍拡大へ送る", "upscale"),
        ):
            with self.subTest(destination=destination):
                button = next(c for c in components if c["props"].get("value") == label)
                callback = next(fn for fn in blocks.fns.values() if (button["id"], "click") in fn.targets)
                previous = {"results": {"generate": {"image": "previous.png"}}}
                self.assertEqual(callback.fn(previous), (destination, "previous.png"))
                for kind in ("run", "setup"):
                    with self.subTest(kind=kind), self.assertRaises(gr.Error):
                        callback.fn({**previous, "job": "running", "kind": kind})
                successors = [dep for dep in dependencies if dep.get("trigger_after") == callback._id]
                self.assertEqual(len(successors), 1)
                self.assertTrue(successors[0]["trigger_only_on_success"])

    def test_common_img2img_upload_strength_and_run_inputs_are_wired(self):
        root = Path(__file__).resolve().parents[2]
        path = root / "extensions-builtin/iris-studio/scripts/iris_studio.py"
        spec = importlib.util.spec_from_file_location("test_iris_img2img_ui", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"modules.script_callbacks": SimpleNamespace(on_ui_tabs=Mock())}):
            spec.loader.exec_module(module)
        with patch.object(module.local_assets, "selection", return_value={}):
            blocks = module.on_ui_tabs()[0][0]
        self.addCleanup(blocks.close)
        components = blocks.get_config_file()["components"]
        edit = next(c for c in components if c["props"].get("elem_id") == "iris-common-img2img")
        self.assertTrue(edit["props"]["interactive"])
        upload = next(c for c in components if c["props"].get("elem_id") == "iris-img2img-input")
        strength = next(c for c in components if c["props"].get("elem_id") == "iris-img2img-strength")
        self.assertEqual(upload["props"]["type"], "pil")
        self.assertEqual((strength["props"]["minimum"], strength["props"]["maximum"]), (0, 1))
        run = next(fn for fn in blocks.fns.values() if fn.fn is module.start_with_loras)
        self.assertEqual([c._id for c in run.inputs[-2:]], [upload["id"], strength["id"]])
        with (
            patch.object(module, "ready", return_value=True),
            patch.object(module.SERVICE, "start", return_value="job") as start,
        ):
            image = Image.new("RGB", (2, 2))
            module.start(
                {},
                "generate",
                "int8",
                "dress",
                None,
                1,
                "1024×1024",
                100,
                3,
                "blur",
                "base",
                [],
                [],
                "img2img",
                image,
                0.25,
            )
        self.assertEqual(start.call_args.args[0]["image_mode"], "img2img")
        self.assertIs(start.call_args.args[0]["image"], image)
        self.assertEqual(start.call_args.args[0]["strength"], 0.25)
