from __future__ import annotations

import unittest

import torch

from modules_forge.packages.comfy.weight_adapter.base import weight_decompose


class DoraPrecisionTests(unittest.TestCase):
    def test_flat_output_magnitude_matches_channel_norms_for_linear_and_conv(self):
        for shape in ((3, 3), (3, 5), (3, 2, 2, 2)):
            original = torch.arange(1, 1 + torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape) / 10
            delta = torch.ones_like(original) * 0.2
            for magnitude_shape in ((3,), (3, 1)):
                magnitude = torch.tensor([2.0, 3.0, 4.0]).reshape(magnitude_shape)
                updated = original + delta * 0.5
                norm = updated.flatten(1).norm(dim=1).reshape(3, *([1] * (len(shape) - 1)))
                normalized = updated * magnitude.reshape(norm.shape) / (norm + torch.finfo(torch.float32).eps)
                for strength in (0.0, 0.5, 1.0, -0.5):
                    with self.subTest(shape=shape, magnitude_shape=magnitude_shape, strength=strength):
                        result = weight_decompose(magnitude, original.clone(), delta.clone(), 0.5, strength, torch.float32, lambda x: x)
                        torch.testing.assert_close(result, original + strength * (normalized - original))

    def test_flat_input_magnitude_normalizes_input_channels(self):
        for shape in ((2, 3), (2, 3, 2, 2)):
            original = torch.arange(1, 1 + torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape) / 10
            magnitude = torch.tensor([2.0, 3.0, 4.0])
            axes = tuple(i for i in range(len(shape)) if i != 1)
            norm = torch.linalg.vector_norm(original, dim=axes, keepdim=True)
            expected = original * magnitude.reshape(norm.shape) / (norm + torch.finfo(torch.float32).eps)
            result = weight_decompose(magnitude, original.clone(), torch.zeros_like(original), 1.0, 1.0, torch.float32, lambda x: x)
            torch.testing.assert_close(result, expected)

    def test_low_precision_normalization_uses_requested_intermediate_dtype(self):
        for dtype, value in ((torch.float16, 60000.0), (torch.bfloat16, 0.001)):
            original = torch.full((2, 2), value, dtype=dtype)
            magnitude = torch.tensor([[2.0], [3.0]])
            full = original.float()
            normalized = full * magnitude / (full.norm(dim=1, keepdim=True) + torch.finfo(torch.float32).eps)
            for strength in (0.0, 0.5, 1.0):
                with self.subTest(dtype=dtype, strength=strength):
                    expected = (full + strength * (normalized - full)).to(dtype)
                    result = weight_decompose(magnitude, original.clone(), torch.zeros_like(full), 1.0, strength, torch.float32, lambda x: x)
                    self.assertEqual(result.dtype, dtype)
                    self.assertTrue(torch.isfinite(result).all())
                    torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    unittest.main()
