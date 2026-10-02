from __future__ import annotations

import unittest

import torch

from backend.patcher.lora import merge_lora_to_weight
from modules_forge.packages.comfy.weight_adapter.lora import LoRAAdapter


class LoraWeightPreservationTests(unittest.TestCase):
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
