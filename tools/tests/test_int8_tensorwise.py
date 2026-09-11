import json
import unittest

import torch

from backend.operations import mixed_precision_ops
from backend.quant_ops import QUANT_ALGOS, QuantizedTensor, TensorWiseINT8Layout


def quant_config_tensor(**config):
    payload = json.dumps(config).encode("utf-8")
    return torch.tensor(list(payload), dtype=torch.uint8)


class Int8TensorWiseTests(unittest.TestCase):
    def test_quantization_format_is_registered_as_weight_only(self):
        self.assertEqual(
            QUANT_ALGOS["int8_tensorwise"],
            {
                "storage_t": torch.int8,
                "parameters": {"weight_scale"},
                "comfy_tensor_layout": "TensorWiseINT8Layout",
                "quantize_input": False,
            },
        )

    def test_convrot_metadata_loads_and_round_trips(self):
        ops = mixed_precision_ops(compute_dtype=torch.bfloat16)
        layer = ops.Linear(256, 4, bias=False, device=torch.device("cpu"))
        state_dict = {
            "weight": torch.zeros((4, 256), dtype=torch.int8),
            "weight_scale": torch.ones((4, 1), dtype=torch.float32),
            "comfy_quant": quant_config_tensor(
                format="int8_tensorwise",
                convrot=True,
                convrot_groupsize=256,
            ),
        }

        layer.load_state_dict(state_dict, strict=True)

        self.assertIsInstance(layer.weight, QuantizedTensor)
        self.assertEqual(layer.weight._params.scale.shape, (4, 1))
        self.assertTrue(layer.weight._params.convrot)
        self.assertEqual(layer.weight._params.convrot_groupsize, 256)

        serialized = layer.state_dict()
        serialized_config = json.loads(serialized["comfy_quant"].numpy().tobytes())
        self.assertEqual(
            serialized_config,
            {
                "format": "int8_tensorwise",
                "convrot": True,
                "convrot_groupsize": 256,
            },
        )
        self.assertEqual(serialized["weight"].dtype, torch.int8)
        self.assertEqual(serialized["weight_scale"].shape, (4, 1))

    def test_missing_int8_scale_fails_explicitly(self):
        ops = mixed_precision_ops(compute_dtype=torch.bfloat16)
        layer = ops.Linear(256, 4, bias=False, device=torch.device("cpu"))
        state_dict = {
            "weight": torch.zeros((4, 256), dtype=torch.int8),
            "comfy_quant": quant_config_tensor(
                format="int8_tensorwise",
                convrot=True,
                convrot_groupsize=256,
            ),
        }

        with self.assertRaisesRegex(ValueError, "Missing INT8 weight scale"):
            layer.load_state_dict(state_dict, strict=True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for native INT8 execution")
    def test_convrot_weight_only_cuda_forward(self):
        torch.manual_seed(123)
        device = torch.device("cuda")
        weight = torch.randn((64, 256), device=device, dtype=torch.bfloat16)
        qdata, params = TensorWiseINT8Layout.quantize(
            weight,
            per_channel=True,
            convrot=True,
            convrot_groupsize=256,
            stochastic_rounding=0,
        )
        ops = mixed_precision_ops(compute_dtype=torch.bfloat16)
        layer = ops.Linear(256, 64, bias=False, device=device)
        state_dict = {
            "weight": qdata.cpu(),
            "weight_scale": params.scale.cpu(),
            "comfy_quant": quant_config_tensor(
                format="int8_tensorwise",
                convrot=True,
                convrot_groupsize=256,
            ),
        }
        layer.load_state_dict(state_dict, strict=True)
        inputs = torch.randn((2, 3, 256), device=device, dtype=torch.bfloat16)

        actual = layer(inputs)
        expected = torch.nn.functional.linear(inputs, weight)

        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertLess(float((actual - expected).abs().mean()), 0.25)


if __name__ == "__main__":
    unittest.main()
