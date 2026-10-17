import unittest

import torch

from modules_forge.packages.comfy.weight_adapter.base import pad_tensor_to_shape


class AdapterTensorPaddingTests(unittest.TestCase):
    def test_rank_mismatch_is_rejected_before_dimension_indexing(self):
        original = torch.arange(6).reshape(2, 3)
        for shape in ([2, 3, 1], [2], [1, 3, 1]):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "same number of dimensions"):
                    pad_tensor_to_shape(original, shape)
        self.assertEqual(original.tolist(), [[0, 1, 2], [3, 4, 5]])

    def test_shrinking_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "larger"):
            pad_tensor_to_shape(torch.ones(2, 3), [2, 2])

    def test_padding_preserves_data_dtype_and_zero_fills_extension(self):
        original = torch.arange(6, dtype=torch.float16).reshape(2, 3)
        actual = pad_tensor_to_shape(original, [3, 5])
        self.assertEqual(actual.dtype, original.dtype)
        self.assertEqual(actual.device, original.device)
        self.assertTrue(torch.equal(actual[:2, :3], original))
        self.assertEqual(actual[2:].count_nonzero().item(), 0)
        self.assertEqual(actual[:, 3:].count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main()
