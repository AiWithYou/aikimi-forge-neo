from __future__ import annotations

import unittest

import torch

from backend.patcher.lora import merge_lora_to_weight
from modules_forge.packages.comfy.weight_adapter.glora import GLoRAAdapter


class GLoRAWeightValidationTests(unittest.TestCase):
    def test_invalid_or_empty_rank_keeps_weight_and_reports_error(self):
        for shapes in (
            ((2, 3), (3, 4), (2, 3), (4, 5)),
            ((0, 3), (3, 0), (0, 3), (4, 0)),
            ((3, 0), (0, 3), (4, 0), (0, 3)),
        ):
            for alpha in (None, 1.0):
                with self.subTest(shapes=shapes, alpha=alpha):
                    original = torch.arange(12, dtype=torch.float32).reshape(4, 3)
                    adapter = GLoRAAdapter(set(), (*[torch.ones(shape) for shape in shapes], alpha, None))
                    with self.assertLogs(level="ERROR") as logs:
                        result = merge_lora_to_weight([(1.0, adapter, 1.0, None, None)], original.clone(), "layer.weight")
                    torch.testing.assert_close(result, original)
                    self.assertIn("layer.weight", " ".join(logs.output))

    def test_old_and_new_layout_match_direct_matrix_equations(self):
        original = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10
        for old in (False, True):
            shapes = ((2, 3), (3, 2), (2, 3), (4, 2)) if old else ((3, 2), (2, 3), (4, 2), (2, 3))
            a1, a2, b1, b2 = [torch.arange(1, rows * cols + 1, dtype=torch.float32).reshape(rows, cols) / 10 for rows, cols in shapes]
            delta = b2 @ b1 + original @ a2 @ a1 if old else b1 @ b2 + original @ a1 @ a2
            for alpha in (None, 3.0):
                for strength in (0.0, 0.5, 1.0):
                    with self.subTest(old=old, alpha=alpha, strength=strength):
                        adapter = GLoRAAdapter(set(), (a1, a2, b1, b2, alpha, None))
                        actual = merge_lora_to_weight([(strength, adapter, 1.0, None, None)], original.clone(), "weight")
                        scale = 1.0 if alpha is None else alpha / 2
                        torch.testing.assert_close(actual, original + strength * scale * delta)


if __name__ == "__main__":
    unittest.main()
