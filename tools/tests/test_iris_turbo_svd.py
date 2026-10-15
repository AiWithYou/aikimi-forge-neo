"""Extracted adapters preserve their base, dense tensors and official schedule."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from safetensors.torch import save_file
from torch import nn

from modules_forge.iris import lora, official_turbo, service
from tools.extract_iris_turbo import export, factorize
from tools.tests.test_iris_lora_ui import screen_module

PROFILE = {"mode": "official_turbo", "sigmas": [1.0, 0.938, 0.833, 0.625], "noise_device": "cpu"}
BASE_ID = {"weights": {"sha256": "a" * 64}, "config": {"sha256": "b" * 64}}


def hybrid(root, *, changes=None, tensors=None):
    path = Path(root) / "loras/turbo.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "iris-turbo-svd-v1",
        "rank": "1",
        "alpha": "1",
        "base_sha256": "a" * 64,
        "base_config_sha256": "b" * 64,
        "target_sha256": "c" * 64,
        "sampling": json.dumps(PROFILE),
        **(changes or {}),
    }
    save_file(
        tensors
        or {
            "projection.lora_down": torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
            "projection.lora_up": torch.tensor([[0.5], [-0.25]], dtype=torch.bfloat16),
            "dense.projection.bias": torch.tensor([0.125, -0.25], dtype=torch.bfloat16),
            "dense.position": torch.tensor([0.5], dtype=torch.bfloat16),
        },
        path,
        metadata=metadata,
    )
    return path


class TurboSVDTests(unittest.TestCase):
    def test_export_roundtrip_provenance_and_existing_output_protection(self):
        class Tiny(nn.Module):
            def __init__(self, cfg=None):
                super().__init__()
                self.projection = nn.Linear(2, 2)
                self.position = nn.Parameter(torch.ones(1))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, target, output = root / "official", root / "turbo", root / "output"
            for folder in (base, target):
                folder.mkdir()
                (folder / "config.yaml").write_text("model: {}\n", encoding="utf-8")
            (target / "distill.yaml").write_text(
                "dmd:\n  deterministic: true\n  sigmas: [1.0, 0.938, 0.833, 0.625]\n", encoding="utf-8"
            )
            original = Tiny()
            teacher = {name: (value + 0.125).bfloat16() for name, value in original.state_dict().items()}
            save_file(original.state_dict(), base / "model.safetensors")
            save_file(teacher, target / "model.safetensors")
            modules = {
                "iris3b.config": SimpleNamespace(inference_config=lambda raw, args: SimpleNamespace(model=None)),
                "iris3b.models.dit": SimpleNamespace(IrisDiT=Tiny),
            }
            with patch.dict(sys.modules, modules):
                manifest = export(root, target / "model.safetensors", output, (2, 4), "cpu")
                with self.assertRaisesRegex(ValueError, "既に"):
                    export(root, target / "model.safetensors", output, (2, 4), "cpu")
                save_file({name: value.float() for name, value in teacher.items()}, target / "model.safetensors")
                with self.assertRaisesRegex(ValueError, "FP32|BF16"):
                    export(root, target / "model.safetensors", root / "invalid-output", (2, 4), "cpu")
            self.assertEqual((manifest["linear_count"], manifest["dense_count"]), (1, 2))
            for record in manifest["files"]:
                model = Tiny()
                model.load_state_dict(original.state_dict())
                selected = lora.snapshot(root, [{"name": str(output / record["path"]), "strength": 1}])
                lora.apply(model, selected)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, teacher[name].float(), atol=0.002, rtol=0.002)

    def test_factorization_reconstructs_and_pads_small_layers(self):
        delta = torch.tensor([[3.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        down, up = factorize(delta, 4)
        self.assertEqual(tuple(down.shape), (4, 2))
        self.assertEqual(tuple(up.shape), (3, 4))
        torch.testing.assert_close(up @ down, delta)
        self.assertEqual(torch.count_nonzero(down[2:]).item(), 0)
        self.assertEqual(torch.count_nonzero(up[:, 2:]).item(), 0)
        down, up = factorize(delta, 1)
        torch.testing.assert_close(up @ down, torch.tensor([[3.0, 0.0], [0.0, 0.0], [0.0, 0.0]]))

    def test_profile_and_base_guard(self):
        with tempfile.TemporaryDirectory() as root:
            hybrid(root)
            sources = lora.snapshot(root, [{"name": "turbo.safetensors", "strength": 1}])
            self.assertEqual(lora.sampling_settings(sources), PROFILE)
            lora.verify_base(sources, BASE_ID, "normal")
            for identity, precision in ((BASE_ID, "int8"), ({**BASE_ID, "weights": {"sha256": "d" * 64}}, "normal")):
                with self.assertRaisesRegex(ValueError, "元モデル|通常版"):
                    lora.verify_base(sources, identity, precision)
            for invalid in (sources * 2, [{**sources[0], "strength": 0.5}]):
                with self.assertRaises(ValueError):
                    lora.sampling_settings(invalid)
            with self.assertRaisesRegex(ValueError, "併用"):
                official_turbo.combine_settings(PROFILE, PROFILE)

    def test_dense_replacement_is_exact_and_failure_does_not_merge(self):
        with tempfile.TemporaryDirectory() as root:
            hybrid(root)
            model = nn.Module()
            model.projection = nn.Linear(2, 2)
            model.position = nn.Parameter(torch.zeros(1))
            original = model.projection.weight.detach().clone()
            lora.apply(model, lora.snapshot(root, [{"name": "turbo.safetensors", "strength": 1}]))
            torch.testing.assert_close(model.projection.weight, original + torch.tensor([[0.5, 1.0], [-0.25, -0.5]]))
            torch.testing.assert_close(model.projection.bias, torch.tensor([0.125, -0.25]), rtol=0, atol=0)
            torch.testing.assert_close(model.position, torch.tensor([0.5]), rtol=0, atol=0)
            invalid = {
                "projection.lora_down": torch.ones(1, 2),
                "projection.lora_up": torch.ones(2, 1),
                "dense.projection.bias": torch.full((2,), float("nan"), dtype=torch.bfloat16),
            }
            hybrid(root, tensors=invalid)
            before = {k: v.clone() for k, v in model.state_dict().items()}
            with self.assertRaisesRegex(ValueError, "非有限"):
                lora.apply(model, lora.snapshot(root, [{"name": "turbo.safetensors", "strength": 1}]))
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, before[name])

    def test_dense_cannot_replace_linear_weights_or_unknown_parameters(self):
        with tempfile.TemporaryDirectory() as root:
            for key, value in (("projection.weight", torch.ones(2, 2)), ("missing.bias", torch.ones(2))):
                hybrid(
                    root,
                    tensors={
                        "projection.lora_down": torch.ones(1, 2),
                        "projection.lora_up": torch.ones(2, 1),
                        "dense." + key: value.bfloat16(),
                    },
                )
                model = nn.Module()
                model.projection = nn.Linear(2, 2)
                with self.assertRaises(ValueError):
                    lora.apply(model, lora.snapshot(root, [{"name": "turbo.safetensors", "strength": 1}]))

    def test_invalid_profile_and_unmarked_dense_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            for changes in (
                {"sampling": json.dumps({**PROFILE, "sigmas": [1, 1]})},
                {"base_sha256": "missing"},
                {"format": "ordinary"},
            ):
                hybrid(root, changes=changes)
                with self.assertRaises(ValueError):
                    lora.snapshot(root, [{"name": "turbo.safetensors", "strength": 1}])

    def test_service_accepts_official_adapter_sizes_and_records_fixed_settings(self):
        with tempfile.TemporaryDirectory() as root:
            hybrid(root)
            backend = service.Service(root)
            lease = Mock()
            lease.acquire.return_value = True
            with (
                patch.object(service, "ROOT", Path(root)),
                patch.object(service, "model_ready", return_value=True),
                patch.object(service, "checkpoint_identity", return_value=BASE_ID),
                patch("modules_forge.gpu_ownership.GPUOwnership", return_value=lease),
                patch.object(service.threading, "Thread"),
            ):
                identifier = backend.start(
                    {
                        "precision": "normal",
                        "prompt": "fox",
                        "width": 1344,
                        "height": 752,
                        "steps": 80,
                        "cfg": 5,
                        "negative_prompt": "dark",
                        "loras": [{"name": "turbo.safetensors", "strength": 1}],
                    }
                )
            request = json.loads((Path(backend.jobs[identifier]["directory"]) / "request.json").read_text())
            self.assertEqual(request["sampling"], PROFILE)
            self.assertEqual((request["steps"], request["cfg"], request["negative_prompt"]), (4, 1, ""))

    def test_ui_switches_sizes_and_restores_normal_settings(self):
        with tempfile.TemporaryDirectory() as root:
            hybrid(root)
            ui = screen_module()
            with patch.object(ui, "RUNTIME", Path(root)):
                updates = ui.lora_controls(
                    {"selection_checkpoint": "base"},
                    "generate",
                    ["turbo.safetensors"],
                    [],
                    {},
                    80,
                    4,
                    "dark",
                    persist=False,
                )
                self.assertEqual([value["value"] for value in updates[5:8]], [4, 1, ""])
                self.assertTrue(all(not value["interactive"] for value in updates[5:8]))
                sizes = ui.update_sizes("base", "1024×1024", ["turbo.safetensors"], [["turbo.safetensors", 1]])
                self.assertIn("1344×752", sizes["choices"])
                restored = ui.lora_controls(
                    {"selection_checkpoint": "base"}, "generate", [], [], updates[0], 4, 1, "", persist=False
                )
                self.assertEqual([value["value"] for value in restored[5:8]], [80, 4, "dark"])


if __name__ == "__main__":
    unittest.main()
