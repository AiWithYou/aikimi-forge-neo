"""Checkpoint metadata and other components must not change weight inspection."""

import tempfile
import unittest
from pathlib import Path

import torch

from backend import utils


class WeightInspectionTests(unittest.TestCase):
    def test_direct_torch_checkpoint_metadata_is_not_a_weight(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            torch.save({"weight": torch.zeros(8, dtype=torch.float16), "epoch": 42, "name": "fixture"}, path)
            state = utils.load_torch_file(str(path))
            self.assertEqual(utils.weight_dtype(state), torch.float16)
            self.assertEqual(utils.calculate_parameters(state), 8)

    def test_largest_floating_weight_type_wins_over_integer_buffers(self):
        state = {
            "large": torch.nn.Parameter(torch.zeros(8, dtype=torch.bfloat16)),
            "small": torch.zeros(2, dtype=torch.float32),
            "buffer": torch.zeros(32, dtype=torch.int64),
        }
        self.assertEqual(utils.weight_dtype(state), torch.bfloat16)
        self.assertEqual(utils.calculate_parameters(state), 42)

    def test_no_floating_weights_returns_none(self):
        for state in ({}, {"metadata": "test"}, {"buffer": torch.zeros(3, dtype=torch.int64)}):
            with self.subTest(keys=list(state)):
                self.assertIsNone(utils.weight_dtype(state))

    def test_prefix_excludes_other_components_quantization_markers(self):
        state = {
            "unet.weight": torch.zeros(4, dtype=torch.float16),
            "encoder.weight": torch.zeros(4, dtype=torch.uint8),
            "encoder.weight.quant_state.bitsandbytes__nf4": torch.zeros(1, dtype=torch.uint8),
        }
        self.assertEqual(utils.weight_dtype(state, "unet."), torch.float16)
        self.assertEqual(utils.weight_dtype(state, "encoder."), "nf4")
        self.assertIsNone(utils.weight_dtype(state, "missing."))
        self.assertEqual(utils.calculate_parameters(state, "unet."), 4)


if __name__ == "__main__":
    unittest.main()
