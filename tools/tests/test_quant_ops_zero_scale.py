import unittest

import comfy_kitchen as ck
import torch

from backend.quant_ops import (
    TensorCoreFP8E4M3Layout,
    TensorCoreFP8E5M2Layout,
    TensorCoreNVFP4Layout_,
)


class QuantOpsZeroScaleTests(unittest.TestCase):
    def test_fp8_all_zero_recalculated_scale_stays_finite(self):
        tensor = torch.zeros((16, 16), dtype=torch.float32)

        for layout in (TensorCoreFP8E4M3Layout, TensorCoreFP8E5M2Layout):
            for stochastic_rounding in (0, 1):
                with self.subTest(
                    layout=layout.__name__,
                    stochastic_rounding=stochastic_rounding,
                ):
                    quantized, params = layout.quantize(
                        tensor,
                        scale="recalculate",
                        stochastic_rounding=stochastic_rounding,
                    )
                    restored = ck.dequantize_per_tensor_fp8(
                        quantized,
                        params.scale,
                        torch.float32,
                    )

                    self.assertEqual(params.scale.item(), 1.0)
                    self.assertTrue(torch.isfinite(quantized.float()).all())
                    self.assertTrue(torch.equal(restored, tensor))

    def test_nvfp4_all_zero_recalculated_scale_stays_finite(self):
        tensor = torch.zeros((16, 16), dtype=torch.float32)

        for stochastic_rounding in (0, 1):
            with self.subTest(stochastic_rounding=stochastic_rounding):
                quantized, params = TensorCoreNVFP4Layout_.quantize(
                    tensor,
                    scale="recalculate",
                    stochastic_rounding=stochastic_rounding,
                )
                restored = ck.dequantize_nvfp4(
                    quantized,
                    params.scale,
                    params.block_scale,
                    output_type=torch.float32,
                )

                self.assertEqual(params.scale.item(), 1.0)
                self.assertTrue(torch.isfinite(params.block_scale.float()).all())
                self.assertTrue(torch.equal(restored, tensor))


if __name__ == "__main__":
    unittest.main()
