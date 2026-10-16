"""Strict Iris adapter selection, numerical application and source isolation."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

from modules_forge.iris import lora
from modules_forge.iris.quantization import Int8Linear
from modules_forge.iris.w4a8 import W4A8Linear


def adapter(root, name="test.safetensors", *, meta=None, tensors=None):
    path = Path(root) / "loras" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors
        if tensors is not None
        else {
            "projection.lora_down": torch.tensor([[1.0, 2.0]]),
            "projection.lora_up": torch.tensor([[0.5], [-0.25]]),
        },
        str(path),
        metadata=meta if meta is not None else {"rank": "1", "alpha": "2"},
    )
    return path


class IrisLoRATests(unittest.TestCase):
    def test_inventory_relative_paths_and_zero_skips_missing(self):
        with tempfile.TemporaryDirectory() as root:
            adapter(root, "日本語/a.safetensors")
            self.assertEqual(lora.inventory(root), ["日本語/a.safetensors"])
            self.assertEqual(lora.snapshot(root, [{"name": "absent.safetensors", "strength": 0}]), [])
            with self.assertRaises(ValueError):
                lora.resolve(root, "../test.safetensors")

    def test_duplicate_real_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = adapter(root)
            with self.assertRaises(ValueError):
                lora.snapshot(root, [{"name": path.name, "strength": 1}, {"name": str(path), "strength": 1}])

    def test_turbo_metadata_and_effective_settings(self):
        with tempfile.TemporaryDirectory() as root:
            adapter(root, meta={"rank": "1", "alpha": "1", "extra": json.dumps({"turbo_steps": 4, "shift": 4})})
            sources = lora.snapshot(root, [{"name": "test.safetensors", "strength": 1}])
            self.assertEqual(lora.sampling_settings(sources), {"mode": "standard"})
            for selected in (sources * 2, [{**sources[0], "strength": 0.5}]):
                self.assertEqual(lora.sampling_settings(selected, "turbo", 2), {"mode": "turbo", "shift": 2.0})

    def test_metadata_extra_cannot_override_scale(self):
        with tempfile.TemporaryDirectory() as root:
            adapter(root, meta={"rank": "1", "alpha": "2", "extra": json.dumps({"rank": 128, "alpha": 100})})
            meta = lora.inspect(root, "test.safetensors")
            self.assertEqual((meta["rank"], meta["alpha"]), (1, 2.0))

    def test_normal_training_shift_metadata_does_not_enable_turbo(self):
        with tempfile.TemporaryDirectory() as root:
            adapter(root, meta={"rank": "1", "alpha": "1", "extra": json.dumps({"shift": 4})})
            sources = lora.snapshot(root, [{"name": "test.safetensors", "strength": 1}])
            self.assertEqual(lora.sampling_settings(sources), {"mode": "standard"})

    def test_reject_unknown_unpaired_bad_rank_and_nonfinite(self):
        variants = [
            {"projection.weight": torch.ones(2, 2)},
            {"projection.lora_down": torch.ones(1, 2)},
            {"projection.lora_down": torch.ones(2, 2), "projection.lora_up": torch.ones(2, 1)},
            {"projection.lora_down": torch.ones(1, 2, dtype=torch.int8), "projection.lora_up": torch.ones(2, 1)},
        ]
        for tensors in variants:
            with self.subTest(keys=list(tensors)), tempfile.TemporaryDirectory() as root:
                adapter(root, tensors=tensors)
                with self.assertRaises(ValueError):
                    lora.snapshot(root, [{"name": "test.safetensors", "strength": 1}])
        with tempfile.TemporaryDirectory() as root:
            adapter(
                root,
                tensors={
                    "projection.lora_down": torch.full((1, 2), float("nan")),
                    "projection.lora_up": torch.ones(2, 1),
                },
            )
            model = nn.Module()
            model.projection = nn.Linear(2, 2)
            original = model.projection.weight.clone()
            sources = lora.snapshot(root, [{"name": "test.safetensors", "strength": 1}])
            with self.assertRaises(ValueError):
                lora.apply(model, sources)
            torch.testing.assert_close(model.projection.weight, original)

    def test_normal_merge_preserves_shared_reference_and_alpha(self):
        with tempfile.TemporaryDirectory() as root:
            adapter(root)
            model = nn.Module()
            model.projection = nn.Linear(2, 2, bias=False)
            shared = (model.projection,)
            original = model.projection.weight.clone()
            sources = lora.snapshot(root, [{"name": "test.safetensors", "strength": -0.5}])
            lora.apply(model, sources)
            delta = torch.tensor([[0.5, 1.0], [-0.25, -0.5]])
            torch.testing.assert_close(model.projection.weight, original - delta)
            self.assertIs(shared[0], model.projection)

    def test_normal_merge_preserves_parameter_without_writing_borrowed_base_storage(self):
        with tempfile.TemporaryDirectory() as root:
            adapter(root)
            model = nn.Module()
            model.projection = nn.Linear(2, 2, bias=False)
            parameter = model.projection.weight
            borrowed = parameter.detach()
            original = borrowed.clone()
            lora.apply(model, lora.snapshot(root, [{"name": "test.safetensors", "strength": 1}]))
            self.assertIs(model.projection.weight, parameter)
            torch.testing.assert_close(borrowed, original, rtol=0, atol=0)
            torch.testing.assert_close(parameter, original + torch.tensor([[1.0, 2.0], [-0.5, -1.0]]))

    def test_int8_keeps_base_buffers_and_tracks_device_moves(self):
        with tempfile.TemporaryDirectory() as root:
            adapter(root)
            model = nn.Module()
            model.projection = Int8Linear.from_linear(nn.Linear(2, 2, bias=False))
            original = {k: v.clone() for k, v in model.state_dict().items()}
            values = torch.tensor([[2.0, -1.0], [1.0, 1.0]])
            expected = model.projection(values) + values @ torch.tensor([[1.0, -0.5], [2.0, -1.0]])
            lora.apply(model, lora.snapshot(root, [{"name": "test.safetensors", "strength": 1}]))
            model.cpu()
            torch.testing.assert_close(model.projection(values), expected)
            self.assertEqual(set(model.state_dict()), set(original))
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, original[key])
            self.assertTrue(any("iris_lora" in name for name, _ in model.named_buffers()))

    def test_shape_or_unknown_target_failure_is_atomic(self):
        with tempfile.TemporaryDirectory() as root:
            adapter(
                root,
                tensors={
                    "projection.lora_down": torch.ones(1, 2),
                    "projection.lora_up": torch.ones(2, 1),
                    "wrong.lora_down": torch.ones(1, 2),
                    "wrong.lora_up": torch.ones(2, 1),
                },
            )
            model = nn.Module()
            model.projection = nn.Linear(2, 2)
            original = model.projection.weight.clone()
            with self.assertRaises(ValueError):
                lora.apply(model, lora.snapshot(root, [{"name": "test.safetensors", "strength": 1}]))
            torch.testing.assert_close(model.projection.weight, original)

    def test_w4a8_residual_uses_unpadded_input_and_keeps_packed_weights(self):
        with tempfile.TemporaryDirectory() as root:
            torch.manual_seed(13)
            down, up = torch.randn(2, 300) * 0.02, torch.randn(129, 2) * 0.02
            adapter(
                root, tensors={"projection.lora_down": down, "projection.lora_up": up}, meta={"rank": "2", "alpha": "2"}
            )
            model = nn.Module()
            model.projection = W4A8Linear.from_linear(nn.Linear(300, 129), device="cpu")
            original = {key: value.clone() for key, value in model.state_dict().items()}
            inputs = torch.randn(2, 300)
            expected = model.projection(inputs) + (inputs @ down.T @ up.T) * 0.5
            lora.apply(model, lora.snapshot(root, [{"name": "test.safetensors", "strength": 0.5}]))
            torch.testing.assert_close(model.projection(inputs), expected)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value.view(torch.uint8), original[key].view(torch.uint8)))

    def test_zero_up_is_exact_noop_and_shared_alias_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            adapter(root, tensors={"projection.lora_down": torch.ones(1, 2), "projection.lora_up": torch.zeros(2, 1)})
            model = nn.Module()
            model.projection = nn.Linear(2, 2)
            original = model.projection.weight.clone()
            lora.apply(model, lora.snapshot(root, [{"name": "test.safetensors", "strength": 1}]))
            self.assertTrue(torch.equal(original, model.projection.weight))
            model.alias = model.projection
            adapter(
                root,
                tensors={
                    "projection.lora_down": torch.ones(1, 2),
                    "projection.lora_up": torch.ones(2, 1),
                    "alias.lora_down": torch.ones(1, 2),
                    "alias.lora_up": torch.ones(2, 1),
                },
            )
            with self.assertRaisesRegex(ValueError, "重複"):
                lora.apply(model, lora.snapshot(root, [{"name": "test.safetensors", "strength": 1}]))

    def test_peft_direct_module_names(self):
        with tempfile.TemporaryDirectory() as root:
            adapter(
                root,
                tensors={
                    "base_model.model.projection.lora_A.weight": torch.ones(1, 2),
                    "base_model.model.projection.lora_B.weight": torch.ones(2, 1),
                },
                meta={},
            )
            self.assertEqual(lora.inspect(root, "test.safetensors")["alpha"], 1.0)

    def test_changed_file_after_acceptance_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            adapter(root)
            sources = lora.snapshot(root, [{"name": "test.safetensors", "strength": 1}])
            adapter(root, meta={"rank": "1", "alpha": "3"})
            with self.assertRaises(ValueError):
                lora.verify_sources(sources)

    def test_invalid_strengths_and_turbo_fields(self):
        with tempfile.TemporaryDirectory() as root:
            adapter(root)
            for strength in (True, float("nan"), float("inf"), -3, 3):
                with self.subTest(strength=strength), self.assertRaises(ValueError):
                    lora.snapshot(root, [{"name": "test.safetensors", "strength": strength}])
            for extra in (
                {"turbo_steps": 0, "shift": 4},
                {"turbo_steps": 4},
                {"turbo_steps": True, "shift": 4},
                {"turbo_steps": 4, "shift": float("nan")},
            ):
                adapter(root, meta={"rank": "1", "alpha": "1", "extra": json.dumps(extra)})
                self.assertEqual(lora.inspect(root, "test.safetensors"), {"rank": 1, "alpha": 1.0})


if __name__ == "__main__":
    unittest.main()
