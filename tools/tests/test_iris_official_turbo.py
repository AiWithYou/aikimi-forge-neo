"""Official distilled checkpoints keep their deterministic schedule and UI contract."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import yaml

from modules_forge import local_assets
from modules_forge.iris import core, local_source, official_turbo, service
from modules_forge.iris.runtime import Runner
from tools.tests.test_iris_local_source import checkpoint
from tools.tests.test_iris_lora_ui import screen_module

SIGMAS = [1.0, 0.938, 0.833, 0.625]
PROFILE = {"mode": "official_turbo", "sigmas": SIGMAS, "noise_device": "cpu"}


def distilled(directory, sigmas=None, **changes):
    path = checkpoint(directory)
    payload = {"dmd": {"sigmas": SIGMAS if sigmas is None else sigmas, "deterministic": True, **changes}}
    (directory / "distill.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


class OfficialTurboContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name, value in (
            ("ROOT", self.root),
            ("HASH_CACHE", self.root / "hashes.json"),
            ("LIBRARY", self.root / "library.json"),
        ):
            setting = patch.object(local_assets, name, value)
            setting.start()
            self.addCleanup(setting.stop)

    def test_schedule_is_read_and_included_in_checkpoint_identity(self):
        path = distilled(self.root / "checkpoints/turbo")
        self.assertEqual(local_source.resolve(self.root, str(path))["sampling"], PROFILE)
        before = core.checkpoint_identity(self.root, "normal", "generate", str(path))
        self.assertIn("distill", before)
        self.assertEqual(before["distill"]["path"], str(path.with_name("distill.yaml")))
        distilled(path.parent, [1.0, 0.9, 0.8, 0.6])
        after = core.checkpoint_identity(self.root, "normal", "generate", str(path))
        self.assertNotEqual(before["distill"]["sha256"], after["distill"]["sha256"])

    def test_malformed_schedule_is_rejected_during_discovery_and_selection(self):
        for sigmas in ([], [0.99, 0.9], [1, 1], [1, 0], [1, -0.1], [1, float("nan")], [1, True], "four"):
            with self.subTest(sigmas=sigmas):
                path = distilled(self.root / "checkpoints/turbo", sigmas)
                with self.assertRaises(ValueError):
                    local_source.resolve(self.root, str(path))
                self.assertEqual(local_source.discover(self.root, str(path.parent)), [])
        path = distilled(self.root / "checkpoints/turbo", deterministic=False)
        with self.assertRaisesRegex(ValueError, "deterministic"):
            local_source.resolve(self.root, str(path))

    def test_malformed_yaml_is_not_treated_as_an_ordinary_checkpoint(self):
        path = distilled(self.root / "checkpoints/turbo")
        for content in ("dmd: [", "[]", "dmd: {}", "dmd: null"):
            with self.subTest(content=content):
                path.with_name("distill.yaml").write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    local_source.resolve(self.root, str(path))

    def test_community_turbo_cannot_override_a_distilled_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "併用"):
            official_turbo.combine_settings(PROFILE, {"mode": "turbo", "default_steps": 4, "shift": 4})
        self.assertEqual(official_turbo.combine_settings(PROFILE, {"mode": "standard"}), PROFILE)
        self.assertEqual(official_turbo.combine_settings(None, {"mode": "standard"}), {"mode": "standard"})

    def test_service_records_effective_fixed_settings_and_official_sizes(self):
        path = distilled(self.root / "checkpoints/turbo")
        backend = service.Service(self.root)
        lease = Mock()
        lease.acquire.return_value = True
        with (
            patch.object(service, "ROOT", self.root),
            patch.object(service, "model_ready", return_value=True),
            patch("modules_forge.gpu_ownership.GPUOwnership", return_value=lease),
            patch.object(service.threading, "Thread"),
        ):
            identifier = backend.start(
                {
                    "checkpoint": str(path),
                    "precision": "normal",
                    "prompt": "fox",
                    "width": 1344,
                    "height": 752,
                    "steps": 80,
                    "cfg": 5,
                    "negative_prompt": "dark",
                    "seed": 1,
                }
            )
        request = json.loads((Path(backend.jobs[identifier]["directory"]) / "request.json").read_text(encoding="utf-8"))
        self.assertEqual((request["steps"], request["cfg"], request["negative_prompt"]), (4, 1.0, ""))
        self.assertEqual(request["sampling"], PROFILE)
        self.assertEqual((request["width"], request["height"]), (1344, 752))

    def test_distilled_img2img_is_rejected_before_gpu_acquisition(self):
        path = distilled(self.root / "checkpoints/turbo")
        with patch("modules_forge.gpu_ownership.GPUOwnership") as lease, self.assertRaisesRegex(ValueError, "img2img"):
            service.Service(self.root).start(
                {
                    "checkpoint": str(path),
                    "precision": "normal",
                    "prompt": "fox",
                    "image_mode": "img2img",
                    "image": object(),
                }
            )
        lease.assert_not_called()

    def test_worker_rejects_a_stale_schedule_before_loading(self):
        path = distilled(self.root / "checkpoints/turbo")
        spec = importlib.util.spec_from_file_location("iris_official_worker", core.ROOT / "tools/iris_worker.py")
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        directory = self.root / "job"
        core.atomic_json(
            directory / "request.json",
            {
                "task": "generate",
                "precision": "normal",
                "checkpoint": str(path),
                "sampling": PROFILE,
                "loras": [],
                "steps": 4,
                "cfg": 1,
                "negative_prompt": "",
            },
        )
        distilled(path.parent, [1, 0.9, 0.8, 0.6])
        with patch("modules_forge.iris.runtime.Runner") as constructor, self.assertRaisesRegex(ValueError, "生成方式"):
            worker.resident_run({"root": str(self.root), "job_dir": str(directory)})
        constructor.assert_not_called()

    def test_ui_uses_fixed_values_and_restores_standard_values_on_switch(self):
        path = distilled(self.root / "checkpoints/turbo")
        ui = screen_module()
        state = {"selection_checkpoint": str(path)}
        with patch.object(ui, "RUNTIME", self.root):
            updates = ui.lora_controls(state, "generate", [], [], {}, 80, 4, "dark", persist=False)
            for update, value in zip(updates[5:8], (4, 1, ""), strict=True):
                self.assertEqual(update["value"], value)
                self.assertFalse(update["interactive"])
            restored = ui.lora_controls(
                {"selection_checkpoint": "base"}, "generate", [], [], updates[0], 4, 1, "", persist=False
            )
            self.assertEqual([update["value"] for update in restored[5:8]], [80, 4, "dark"])
            self.assertTrue(all(update["interactive"] for update in restored[5:8]))
            sizes = ui.update_sizes(str(path), "832×1280")
            self.assertEqual(len(sizes["choices"]), 7)
            self.assertIn("1344×752", sizes["choices"])
            self.assertEqual(sizes["value"], "1024×1024")


class OfficialTurboSamplerTests(unittest.TestCase):
    def test_runtime_dispatches_fixed_schedule_and_counts_actual_forwards(self):
        runner = Runner.__new__(Runner)
        runner.task, runner.precision = "generate", "normal"
        runner.root, runner.checkpoint, runner.source_identity = (
            Path("runtime"),
            "turbo",
            {"weights": {}, "config": {}, "distill": {}},
        )
        runner.model = torch.nn.Identity()
        runner.model.cuda = lambda: runner.model
        runner.model.forward = lambda x: SimpleNamespace(x=x)
        runner.cfg = SimpleNamespace(flow=SimpleNamespace(num_train_timesteps=1000, prediction="v"))
        runner.prepare_text = lambda request: None
        runner.offload = lambda: None
        received = {}

        def sample(model, text, prompts, height, width, sigmas, generator, **kwargs):
            received.update(sigmas=sigmas, generator_device=str(generator.device), **kwargs)
            for _ in sigmas:
                model(torch.zeros(1, 3, height, width))
            return torch.zeros(1, 3, height, width)

        upstream = SimpleNamespace(generate=Mock(side_effect=AssertionError("wrong sampler")))
        request = {
            "prompt": "fox",
            "negative_prompt": "",
            "width": 16,
            "height": 16,
            "steps": 4,
            "cfg": 1,
            "seed": 42,
            "sampling": PROFILE,
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, {"iris3b.sampling": upstream}),
            patch.object(official_turbo, "generate", side_effect=sample),
            patch("torch.cuda.reset_peak_memory_stats"),
            patch("torch.cuda.synchronize"),
            patch("torch.cuda.max_memory_allocated", return_value=0),
            patch("torch.cuda.max_memory_reserved", return_value=0),
            patch("torch.autocast"),
            patch("modules_forge.iris.runtime.checkpoint_identity", return_value=runner.source_identity),
        ):
            runner.run(directory, request)
            metadata = json.loads((Path(directory) / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(
            received, {"sigmas": SIGMAS, "generator_device": "cpu", "num_train_timesteps": 1000, "prediction": "v"}
        )
        self.assertEqual(metadata["schedule"]["forward_calls"], 4)
        self.assertEqual(metadata["schedule"]["sigmas"], SIGMAS)

    def model(self, value=None):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.cfg = SimpleNamespace(in_channels=3)
                self.times = []

            def forward(self, x, t, cond, y_mask=None):
                self.times.append(t.tolist())
                output = 0.2 * x + t[:, None, None, None] / 10000 if value is None else torch.full_like(x, value)
                return SimpleNamespace(x=output)

        return Model()

    def encoder(self):
        encoding = SimpleNamespace(embeddings=torch.zeros(1, 1, 1), mask=torch.ones(1, 1))
        return SimpleNamespace(encode=Mock(return_value=encoding), null=Mock(side_effect=AssertionError("no CFG")))

    def test_four_calls_match_deterministic_euler_and_reuse_initial_noise(self):
        model = self.model()
        initial = torch.randn(1, 3, 2, 2, generator=torch.Generator().manual_seed(12))
        expected = initial.clone()
        for index, sigma in enumerate(SIGMAS):
            clean = expected - sigma * (0.2 * expected + sigma / 10)
            if index + 1 < len(SIGMAS):
                implied = (expected - (1 - sigma) * clean) / sigma
                expected = (1 - SIGMAS[index + 1]) * clean + SIGMAS[index + 1] * implied
        with patch("torch.randn", wraps=torch.randn) as random:
            result = official_turbo.generate(
                model, self.encoder(), ["fox"], 2, 2, SIGMAS, generator=torch.Generator().manual_seed(12), device="cpu"
            )
        torch.testing.assert_close(result, clean.clamp(-1, 1))
        self.assertEqual(model.times, [[sigma * 1000] for sigma in SIGMAS])
        random.assert_called_once()
        self.assertEqual(random.call_args.kwargs["device"], "cpu")

    def test_nonfinite_predictions_fail_before_clamping(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "非有限"):
                official_turbo.generate(self.model(value), self.encoder(), ["fox"], 2, 2, SIGMAS, device="cpu")


if __name__ == "__main__":
    unittest.main()
