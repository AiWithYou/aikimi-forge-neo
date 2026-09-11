import unittest
from types import SimpleNamespace

import torch

from modules_forge.sensenova_u15_attention import headwise_eager_attention


class ReferenceAttentionTests(unittest.TestCase):
    def test_headwise_matches_full_attention_with_grouped_keys_and_masks(self):
        torch.manual_seed(44)
        module = SimpleNamespace(training=False, num_key_value_groups=2)
        for dtype in (torch.float32, torch.bfloat16):
            q = torch.randn(2, 8, 13, 16).to(dtype)
            k = torch.randn(2, 4, 19, 16).to(dtype)
            v = torch.randn(2, 4, 19, 12).to(dtype)
            for mask_heads in (0, 1, 8):
                mask = torch.zeros(2, mask_heads, 13, 19, dtype=dtype) if mask_heads else None
                if mask is not None:
                    mask[..., -3:] = float("-inf")
                weights = (q @ k.repeat_interleave(2, dim=1).transpose(2, 3)) * 0.25
                if mask is not None:
                    weights = weights + mask
                weights = torch.softmax(weights, dim=-1, dtype=torch.float32).to(dtype)
                expected = (weights @ v.repeat_interleave(2, dim=1)).transpose(1, 2).contiguous()
                with self.subTest(dtype=dtype, mask_heads=mask_heads):
                    actual, attention = headwise_eager_attention(module, q, k, v, mask, 0.25)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertIsNone(attention)

    def test_training_cannot_silently_use_inference_attention(self):
        with self.assertRaisesRegex(RuntimeError, "推論専用"):
            headwise_eager_attention(SimpleNamespace(training=True), None, None, None, None, 1.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDAのbatched matmul丸めを実機で確認")
    def test_cuda_bfloat16_preserves_batched_matmul_rounding(self):
        torch.manual_seed(67)
        q = torch.randn(1, 32, 503, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 8, 503, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, 8, 503, 128, device="cuda", dtype=torch.bfloat16)
        scale = 128**-0.5
        weights = torch.softmax(
            q @ k.repeat_interleave(4, dim=1).transpose(2, 3) * scale, dim=-1, dtype=torch.float32
        ).to(q.dtype)
        expected = (weights @ v.repeat_interleave(4, dim=1)).transpose(1, 2).contiguous()
        actual, _ = headwise_eager_attention(
            SimpleNamespace(training=False, num_key_value_groups=4), q, k, v, None, scale
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
