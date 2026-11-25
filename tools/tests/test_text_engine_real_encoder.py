"""Precision checks for shared prompt emphasis."""

import unittest

import torch

from backend.text_processing import emphasis


class EmphasisPrecisionReviewTests(unittest.TestCase):
    def test_uniform_half_weights_preserve_values_when_half_product_would_underflow(self):
        state = emphasis.EmphasisOriginal()
        state.z = torch.tensor([[[0.0001, 0.0002], [0.0003, 0.0004]]], dtype=torch.float16)
        state.multipliers = torch.full((1, 2), 1e-5, dtype=torch.float16)
        original = state.z.clone()
        self.assertTrue(torch.equal(state.z * state.multipliers.unsqueeze(-1), torch.zeros_like(state.z)))
        state.after_transformers()
        self.assertEqual(state.z.dtype, torch.float16)
        torch.testing.assert_close(state.z, original)

    def test_nonuniform_weights_preserve_nonzero_mean_and_existing_precision(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
            with self.subTest(dtype=dtype):
                state = emphasis.EmphasisOriginal()
                state.z = torch.tensor([[[1, 2], [3, 4]]], dtype=dtype)
                state.multipliers = torch.tensor([[2, 0.5]], dtype=dtype)
                state.after_transformers()
                expected = torch.tensor([[[40 / 19, 80 / 19], [30 / 19, 40 / 19]]], dtype=dtype)
                self.assertEqual(state.z.dtype, dtype)
                torch.testing.assert_close(state.z, expected)

    def test_double_precision_input_is_not_rounded_to_single_precision(self):
        state = emphasis.EmphasisOriginal()
        state.z = torch.tensor([[[1.0000000002, 2.0000000003]]], dtype=torch.float64)
        state.multipliers = torch.full((1, 1), 2.0, dtype=torch.float64)
        original = state.z.clone()
        state.after_transformers()
        torch.testing.assert_close(state.z, original, rtol=1e-12, atol=1e-12)

    def test_zero_mean_with_nonuniform_or_zero_weights_stays_finite(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for weights, expected in (([1, 2], [[1, -1], [4, -4]]), ([0, 0], [[0, 0], [0, 0]])):
                with self.subTest(dtype=dtype, weights=weights):
                    state = emphasis.EmphasisOriginal()
                    state.z = torch.tensor([[[1, -1], [2, -2]]], dtype=dtype)
                    state.multipliers = torch.tensor([weights], dtype=dtype)
                    state.after_transformers()
                    self.assertEqual(state.z.dtype, dtype)
                    torch.testing.assert_close(state.z, torch.tensor([expected], dtype=dtype))

    def test_small_finite_half_weights_do_not_overflow_normalization(self):
        state = emphasis.EmphasisOriginal()
        state.z = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float16)
        state.multipliers = torch.full((1, 2), 1e-5, dtype=torch.float16)
        original = state.z.clone()
        state.after_transformers()
        self.assertEqual(state.z.dtype, torch.float16)
        self.assertTrue(torch.isfinite(state.z).all())
        torch.testing.assert_close(state.z, original, atol=0.01, rtol=0.01)


if __name__ == "__main__":
    unittest.main()
