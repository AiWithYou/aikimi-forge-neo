from __future__ import annotations

import unittest

import torch

from backend.patcher.lora import merge_lora_to_weight
from modules_forge.packages.comfy.weight_adapter.base import WeightAdapterBase
from modules_forge.packages.comfy.weight_adapter.lora import LoRAAdapter


class LoraWeightPreservationTests(unittest.TestCase):
    def test_offset_lora_with_reshape_metadata_copies_back_allocated_result(self):
        original = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        up = torch.tensor([[2.0], [1.0]])
        down = torch.tensor([[1.0, 2.0, 3.0]])
        adapter = LoRAAdapter.load("layer", {
            "layer.lora_up.weight": up,
            "layer.lora_down.weight": down,
            "layer.reshape_weight": torch.tensor([2, 3]),
        }, None, None)
        for strength in (0.0, 0.5, 1.0, -0.5):
            with self.subTest(strength=strength):
                expected = original.clone()
                expected[1:3] = original[1:3] * 0.8 + strength * (up @ down)
                expected += 0.25
                actual = merge_lora_to_weight([
                    (strength, adapter, 0.8, (0, 1, 2), None),
                    (0.25, (torch.ones_like(original),), 1.0, None, None),
                ], original.clone(), "weight")
                torch.testing.assert_close(actual, expected)

    def test_offset_lora_rejects_shape_expansion_instead_of_losing_update(self):
        adapter = LoRAAdapter(set(), (torch.ones(2, 1), torch.ones(1, 4), None, None, None, [2, 4]))
        with self.assertRaisesRegex(ValueError, "offset.*weight"):
            merge_lora_to_weight([(1.0, adapter, 1.0, (0, 1, 2), None)], torch.zeros(4, 3), "weight")

    def test_offset_diff_rejects_padding_outside_selected_shape(self):
        patch = ("diff", (torch.ones(2, 4), {"pad_weight": True}))
        with self.assertRaisesRegex(ValueError, "offset.*weight"):
            merge_lora_to_weight([(1.0, patch, 1.0, (0, 1, 2), None)], torch.zeros(4, 3), "weight")

    def test_failed_offset_adapter_preserves_full_weight_and_following_patch(self):
        class FailedAdapter(WeightAdapterBase):
            name = "failed-test-adapter"

            def calculate_weight(self, *args, **kwargs):
                return None

        original = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        for following in (False, True):
            with self.subTest(following=following):
                patches = [(1.0, FailedAdapter(), 1.0, (0, 1, 2), None)]
                if following:
                    patches.append((0.5, (torch.ones_like(original),), 1.0, None, None))
                with self.assertLogs("lora", level="ERROR"):
                    actual = merge_lora_to_weight(patches, original.clone(), "weight")
                torch.testing.assert_close(actual, original + (0.5 if following else 0.0))

    def test_dora_normalizes_the_updated_output_weight_at_full_and_partial_strength(self):
        original = torch.tensor([[3.0, 1.0, 2.0], [2.0, 4.0, 1.0]])
        up = torch.tensor([[2.0], [1.0]])
        down = torch.tensor([[1.0, 2.0, 3.0]])
        magnitude = torch.tensor([[5.0], [7.0]])
        adapter = LoRAAdapter(set(), (up, down, None, None, magnitude, None))
        updated = original + up @ down
        normalized = updated * magnitude / updated.norm(dim=1, keepdim=True)

        for strength in (1.0, 0.5):
            with self.subTest(strength=strength):
                actual = merge_lora_to_weight([(strength, adapter, 1.0, None, None)], original.clone(), "weight")
                expected = original + strength * (normalized - original)
                torch.testing.assert_close(actual, expected)

    def test_plain_lora_and_zero_strength_preserve_the_existing_weight_contract(self):
        original = torch.tensor([[3.0, 1.0], [2.0, 4.0]])
        up = torch.tensor([[2.0], [1.0]])
        down = torch.tensor([[1.0, 2.0]])
        for strength, magnitude in ((0.5, None), (0.0, torch.tensor([[5.0], [7.0]]))):
            with self.subTest(strength=strength):
                adapter = LoRAAdapter(set(), (up, down, None, None, magnitude, None))
                actual = merge_lora_to_weight([(strength, adapter, 1.0, None, None)], original.clone(), "weight")
                torch.testing.assert_close(actual, original + strength * (up @ down))


if __name__ == "__main__":
    unittest.main()
