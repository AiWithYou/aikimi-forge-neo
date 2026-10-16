"""LoRA recommendations never replace explicit generation settings."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from PIL import Image

from modules_forge.iris import core, lora, runtime, service
from tools.tests.test_iris_lora import adapter
from tools.tests.test_iris_lora_ui import screen_module


class LoRASamplingPolicyTests(unittest.TestCase):
    def test_recommendations_do_not_select_sampler_or_restrict_composition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extra = json.dumps({"turbo_steps": 4, "shift": 4})
            for name in ("first.safetensors", "second.safetensors"):
                adapter(root, name, meta={"alpha": "1", "extra": extra})
            sources = lora.snapshot(
                root,
                [
                    {"name": "first.safetensors", "strength": 0.5},
                    {"name": "second.safetensors", "strength": -0.25},
                ],
            )
            self.assertEqual(lora.sampling_settings(sources), {"mode": "standard"})
            self.assertEqual(lora.sampling_settings(sources, "turbo", 2.5), {"mode": "turbo", "shift": 2.5})
            model = torch.nn.Module()
            model.projection = torch.nn.Linear(2, 2, bias=False)
            before = model.projection.weight.detach().clone()
            lora.apply(model, sources)
            torch.testing.assert_close(
                model.projection.weight,
                before + 0.25 * torch.tensor([[0.5, 1], [-0.25, -0.5]]),
            )

    def test_broken_recommendations_do_not_reject_valid_factors(self):
        with tempfile.TemporaryDirectory() as root:
            for extra in ("{broken", "[]", json.dumps({"turbo_steps": 4}), json.dumps({"turbo_steps": 0, "shift": 4})):
                with self.subTest(extra=extra):
                    adapter(root, meta={"alpha": "2", "extra": extra})
                    self.assertEqual(lora.inspect(root, "test.safetensors"), {"rank": 1, "alpha": 2.0})

    def test_explicit_sampling_request_validation(self):
        self.assertEqual(core.validate_request({"prompt": "fox"})["lora_sampling"], "standard")
        for changes in (
            {"lora_sampling": "unknown"},
            {"lora_sampling": "turbo", "turbo_shift": 0},
            {"lora_sampling": "turbo", "turbo_shift": True},
            {"lora_sampling": "turbo", "turbo_shift": float("inf")},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                core.validate_request({"prompt": "fox", **changes})

    def test_service_keeps_manual_values_and_allows_turbo_lora_in_standard_img2img(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter(root, meta={"alpha": "1", "extra": json.dumps({"turbo_steps": 4, "shift": 4})})
            backend = service.Service(root)
            lease = Mock()
            lease.acquire.return_value = True
            with (
                patch.object(service, "ROOT", root),
                patch.object(service, "model_ready", return_value=True),
                patch("modules_forge.gpu_ownership.GPUOwnership", return_value=lease),
                patch.object(service.threading, "Thread"),
            ):
                for settings in (
                    {"image_mode": "img2img", "image": Image.new("RGB", (2, 2))},
                    {"lora_sampling": "turbo", "turbo_shift": 2.5},
                ):
                    identifier = backend.start(
                        {
                            "prompt": "fox",
                            "steps": 27,
                            "cfg": 3.5,
                            "negative_prompt": "dark",
                            "loras": [{"name": "test.safetensors", "strength": 0.5}],
                            **settings,
                        }
                    )
                    saved = json.loads((Path(backend.jobs[identifier]["directory"]) / "request.json").read_text())
                    self.assertEqual((saved["steps"], saved["cfg"], saved["negative_prompt"]), (27, 3.5, "dark"))
                    self.assertEqual(
                        saved["sampling"],
                        {"mode": "turbo", "shift": 2.5} if "lora_sampling" in settings else {"mode": "standard"},
                    )

    def test_ui_lo_ra_changes_preserve_manual_settings_and_sampler_has_explicit_control(self):
        ui = screen_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter(root, meta={"alpha": "1", "extra": json.dumps({"turbo_steps": 4, "shift": 4})})
            with patch.object(ui, "RUNTIME", root):
                updates = ui.lora_controls(
                    {},
                    "generate",
                    ["test.safetensors"],
                    [["test.safetensors", 0.5]],
                    {},
                    27,
                    3.5,
                    "dark",
                    persist=False,
                )
                self.assertTrue(all("value" not in item for item in updates[5:8]))
                self.assertIn("推奨", updates[4])
                blocks = ui.on_ui_tabs()[0][0]
                props = {item["props"].get("elem_id"): item["props"] for item in blocks.get_config_file()["components"]}
                self.assertEqual(props["iris-lora-sampling"]["value"], "standard")
                self.assertEqual(props["iris-turbo-shift"]["value"], 4)

    def test_worker_uses_explicit_sampling_and_rejects_mismatched_record(self):
        spec = importlib.util.spec_from_file_location(
            "iris_explicit_sampling_worker", core.ROOT / "tools/iris_worker.py"
        )
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = root / "job"
            request = {
                "task": "generate",
                "precision": "normal",
                "loras": [],
                "lora_sampling": "turbo",
                "turbo_shift": 2.5,
                "sampling": {"mode": "turbo", "shift": 2.5},
                "steps": 7,
                "cfg": 2,
                "negative_prompt": "dark",
            }
            core.atomic_json(job / "request.json", request)
            with (
                patch.object(worker, "checkpoint_identity", return_value={}),
                patch.object(runtime, "Runner") as constructor,
            ):
                constructor.return_value.run.return_value = "done"
                self.assertEqual(worker.resident_run({"root": root, "job_dir": job}), "done")
                core.atomic_json(job / "request.json", {**request, "turbo_shift": 4})
                with self.assertRaisesRegex(ValueError, "生成方式"):
                    worker.resident_run({"root": root, "job_dir": job})

    def test_official_profile_restores_latest_manual_community_settings(self):
        ui = screen_module()
        profile = {"mode": "official_turbo", "sigmas": [1, 0.5], "noise_device": "cpu"}
        previous = {"mode": "turbo", "steps": 100, "cfg": 3, "negative": "old"}
        with patch.object(ui, "checkpoint_settings", return_value=profile):
            fixed = ui.lora_controls({}, "generate", [], [], previous, 27, 3.5, "dark", persist=False)
        restored = ui.lora_controls({}, "generate", [], [], fixed[0], 2, 1, "", persist=False)
        self.assertEqual(tuple(item["value"] for item in restored[5:8]), (27, 3.5, "dark"))


if __name__ == "__main__":
    unittest.main()
