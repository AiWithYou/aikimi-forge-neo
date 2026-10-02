"""CPU checks that attention fallbacks preserve tensor layout and masks."""

import types
import unittest
from unittest.mock import Mock, patch

import torch

from backend import attention


class AttentionFallbackLayoutTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator(device="cpu").manual_seed(17)
        self.q = torch.randn(3, 2, 5, 4, generator=generator)
        self.k = torch.randn(3, 2, 7, 4, generator=generator)
        self.v = torch.randn(3, 2, 7, 4, generator=generator)
        self.mask = torch.randn(3, 1, 5, 7, generator=generator)

    def flat_inputs(self):
        return tuple(t.transpose(1, 2).flatten(2) for t in (self.q, self.k, self.v))

    def reference(self, skip_output_reshape):
        out = torch.nn.functional.scaled_dot_product_attention(self.q, self.k, self.v, self.mask)
        return out if skip_output_reshape else out.transpose(1, 2).flatten(2)

    def test_xformers_failure_preserves_head_and_output_layouts(self):
        failing_ops = types.SimpleNamespace(memory_efficient_attention=Mock(side_effect=RuntimeError("fixture")))
        with patch.object(attention, "xformers", types.SimpleNamespace(ops=failing_ops), create=True):
            for skip_reshape in (False, True):
                for skip_output_reshape in (False, True):
                    with self.subTest(skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape):
                        inputs = (self.q, self.k, self.v) if skip_reshape else self.flat_inputs()
                        actual = attention.attention_xformers(
                            *inputs,
                            heads=2,
                            mask=self.mask,
                            skip_reshape=skip_reshape,
                            skip_output_reshape=skip_output_reshape,
                        )
                        torch.testing.assert_close(actual, self.reference(skip_output_reshape))

    def test_xformers_tracing_fallback_preserves_output_layout(self):
        with patch.object(torch.jit, "is_tracing", return_value=True):
            actual = attention.attention_xformers(
                *self.flat_inputs(), heads=2, mask=self.mask, skip_output_reshape=True
            )
        torch.testing.assert_close(actual, self.reference(True))

    def test_split_sdpa_preserves_output_layout_and_per_batch_mask(self):
        with patch.object(attention, "SDP_BATCH_LIMIT", 2):
            for skip_reshape in (False, True):
                for skip_output_reshape in (False, True):
                    with self.subTest(skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape):
                        inputs = (self.q, self.k, self.v) if skip_reshape else self.flat_inputs()
                        actual = attention.attention_pytorch(
                            *inputs,
                            heads=2,
                            mask=self.mask,
                            skip_reshape=skip_reshape,
                            skip_output_reshape=skip_output_reshape,
                        )
                        torch.testing.assert_close(actual, self.reference(skip_output_reshape))

    def test_slice_vae_reduces_prime_length_after_out_of_memory(self):
        q = self.q[:1, 0]
        k = self.k[:1, 0, :5].transpose(1, 2)
        v = self.v[:1, 0, :5].transpose(1, 2)
        expected = torch.bmm(v, (torch.bmm(q, k) * 0.5).softmax(dim=-1).transpose(1, 2))
        original_bmm = torch.bmm
        attempted_lengths = []

        def memory_limited_bmm(left, right):
            if left.shape[-1] == 4:
                attempted_lengths.append(left.shape[1])
                if left.shape[1] > 2:
                    raise torch.OutOfMemoryError("fixture")
            return original_bmm(left, right)

        with (
            patch.object(attention.memory_management, "get_free_memory", return_value=10**9),
            patch.object(
                attention.memory_management, "is_oom", side_effect=lambda e: isinstance(e, torch.OutOfMemoryError)
            ),
            patch.object(attention.memory_management, "soft_empty_cache"),
            patch.object(torch, "bmm", side_effect=memory_limited_bmm),
        ):
            actual = attention.slice_attention_vae(q, k, v)

        self.assertEqual(attempted_lengths[:3], [5, 3, 2])
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
