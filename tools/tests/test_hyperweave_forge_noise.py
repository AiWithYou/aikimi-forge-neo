from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "extensions-builtin" / "hyperweave"
sys.path.insert(0, str(PACKAGE))

from hyperweave.forge_noise import coordinate_noise_tensor


class ForgeCoordinateNoiseTests(unittest.TestCase):
    def setUp(self):
        self.override = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)

    def test_adapts_standard_four_dimensional_image_latent(self):
        original = torch.zeros((1, 2, 3, 4), dtype=torch.float16)
        result = coordinate_noise_tensor(
            self.override, original, initial_noise_multiplier=0.5
        )
        self.assertEqual(tuple(original.shape), tuple(result.shape))
        np.testing.assert_allclose(
            result.float().numpy()[0],
            self.override * 0.5,
            rtol=0,
            atol=0,
        )

    def test_adapts_singleton_temporal_qwen_image_latent(self):
        original = torch.zeros((1, 2, 1, 3, 4), dtype=torch.bfloat16)
        result = coordinate_noise_tensor(self.override, original)
        self.assertEqual((1, 2, 1, 3, 4), tuple(result.shape))
        np.testing.assert_allclose(
            result.float().numpy()[0, :, 0],
            self.override,
            rtol=0,
            atol=0,
        )

    def test_rejects_non_singleton_temporal_latent(self):
        original = torch.zeros((1, 2, 2, 3, 4))
        with self.assertRaisesRegex(ValueError, "singleton temporal"):
            coordinate_noise_tensor(self.override, original)

    def test_preserves_prior_callback_delta(self):
        original = torch.full((1, 2, 3, 4), 7.0)
        delta = torch.full_like(original, 0.25)
        prior = original + delta
        result = coordinate_noise_tensor(
            self.override,
            original,
            prior_modified=prior,
        )
        expected = torch.from_numpy(self.override).unsqueeze(0) + delta
        torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    unittest.main()
