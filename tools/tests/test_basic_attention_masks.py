"""Basic attention preserves per-query masks and fully blocked rows on CPU."""

import unittest

import torch

from backend import attention


class BasicAttentionMaskTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator(device="cpu").manual_seed(27)
        self.q = torch.randn(3, 2, 5, 4, generator=generator)
        self.k = torch.randn(3, 2, 7, 4, generator=generator)
        self.v = torch.randn(3, 2, 7, 4, generator=generator)

    def assert_mask_matches_sdpa(self, mask, reference_mask=None):
        expected = torch.nn.functional.scaled_dot_product_attention(
            self.q, self.k, self.v, attn_mask=mask if reference_mask is None else reference_mask
        )
        for skip_reshape in (False, True):
            for skip_output_reshape in (False, True):
                with self.subTest(input_heads=skip_reshape, output_heads=skip_output_reshape):
                    inputs = (
                        (self.q, self.k, self.v)
                        if skip_reshape
                        else tuple(value.transpose(1, 2).flatten(2) for value in (self.q, self.k, self.v))
                    )
                    actual = attention.attention_basic(
                        *inputs,
                        heads=2,
                        mask=mask,
                        skip_reshape=skip_reshape,
                        skip_output_reshape=skip_output_reshape,
                    )
                    reference = expected if skip_output_reshape else expected.transpose(1, 2).flatten(2)
                    torch.testing.assert_close(actual, reference)

    def test_boolean_per_query_and_head_masks_preserve_their_axes(self):
        mask = torch.arange(3 * 2 * 5 * 7).reshape(3, 2, 5, 7) % 3 != 0
        self.assert_mask_matches_sdpa(mask)

    def test_broadcast_boolean_padding_mask_applies_to_every_batch(self):
        mask = torch.tensor([[[[True, False, True, False, True, True, False]]]])
        self.assert_mask_matches_sdpa(mask)

    def test_two_dimensional_query_mask_matches_additive_mask(self):
        mask = torch.arange(5 * 7).reshape(5, 7) % 4 != 0
        self.assert_mask_matches_sdpa(mask)

    def test_existing_per_batch_key_mask_keeps_its_padding_contract(self):
        mask = torch.arange(3 * 7).reshape(3, 7) % 3 != 0
        self.assert_mask_matches_sdpa(mask, reference_mask=mask[:, None, None, :])

    def test_fully_blocked_boolean_or_additive_rows_produce_zero(self):
        boolean = torch.ones(3, 1, 5, 7, dtype=torch.bool)
        boolean[:, :, 2, :] = False
        additive = torch.zeros_like(boolean, dtype=torch.float32).masked_fill(~boolean, float("-inf"))
        for mask in (boolean, additive):
            with self.subTest(dtype=mask.dtype):
                self.assert_mask_matches_sdpa(mask)


if __name__ == "__main__":
    unittest.main()
