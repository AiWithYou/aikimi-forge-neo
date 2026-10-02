"""W4A8 checkpoint saves must preserve grouping and reject malformed packing."""

import json
import unittest

import torch

from backend.operations import mixed_precision_ops
from backend.quant_ops import QuantizedTensor


def config_tensor(**configuration):
    return torch.tensor(list(json.dumps(configuration).encode("utf-8")), dtype=torch.uint8)


class W4A8MetadataTests(unittest.TestCase):
    def layer(self):
        return mixed_precision_ops(compute_dtype=torch.bfloat16).Linear(256, 4, bias=False, device="cpu")

    def state(self, *, packed_width=128, group_size=32, convrot_groupsize=128):
        return {
            "weight": torch.zeros((4, packed_width), dtype=torch.int8),
            "weight_s_rel": torch.ones((4, 256 // group_size), dtype=torch.float8_e4m3fn),
            "weight_s_channel": torch.ones(4, dtype=torch.float32),
            "comfy_quant": config_tensor(
                format="asym_w4a8_int8", group_size=group_size, convrot_groupsize=convrot_groupsize
            ),
        }

    def test_save_reload_preserves_nondefault_group_parameters(self):
        layer = self.layer()
        layer.load_state_dict(self.state(), strict=True)
        self.assertIsInstance(layer.weight, QuantizedTensor)
        saved = layer.state_dict()
        metadata = json.loads(saved["comfy_quant"].numpy().tobytes())
        self.assertEqual(metadata["group_size"], 32)
        self.assertEqual(metadata["convrot_groupsize"], 128)
        restored = self.layer()
        restored.load_state_dict(saved, strict=True)
        self.assertEqual(restored.weight._params.group_size, layer.weight._params.group_size)
        self.assertEqual(restored.weight._params.convrot_groupsize, layer.weight._params.convrot_groupsize)
        self.assertTrue(torch.equal(restored.state_dict()["weight"], saved["weight"]))

    def test_bad_packed_width_is_rejected_before_model_use(self):
        for width in (127, 129, 256):
            with self.subTest(width=width), self.assertRaisesRegex(ValueError, "packed weight width"):
                self.layer().load_state_dict(self.state(packed_width=width), strict=True)


if __name__ == "__main__":
    unittest.main()
