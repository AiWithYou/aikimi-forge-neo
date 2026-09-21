"""Regression for upstream 3baffa8b: quant metadata must follow weight prefixes."""

import json
import unittest

import torch

from backend.state_dict import convert_quantization, detect_quantization, filter_state_dict_with_prefix


class QuantMetadataPrefixTests(unittest.TestCase):
    def convert(self, weights, layers):
        return convert_quantization(weights, {"_quantization_metadata": json.dumps({"layers": layers})})[0]

    def test_prefixed_weights_keep_w4a8_metadata_when_component_is_extracted(self):
        layer = "blocks.0.attn.to_q"
        prefix = "model.diffusion_model."
        config = {"format": "asym_w4a8_int8", "group_size": 16}
        weights = {prefix + layer + ".weight": torch.zeros((256, 128), dtype=torch.int8)}
        converted = self.convert(weights, {layer: config})
        component = filter_state_dict_with_prefix(converted, prefix)
        self.assertEqual(detect_quantization(component, is_unet=True), {"mixed_ops": True, "TE": False})
        self.assertEqual(json.loads(bytes(component[layer + ".comfy_quant"].tolist())), config)

    def test_already_prefixed_metadata_is_not_prefixed_twice(self):
        layer = "model.diffusion_model.blocks.0.attn.to_q"
        converted = self.convert({layer + ".weight": torch.zeros(1)}, {layer: {"format": "int8_tensorwise"}})
        self.assertIn(layer + ".comfy_quant", converted)
        self.assertEqual(len(converted), 2)

    def test_different_components_resolve_their_own_unique_prefix(self):
        weights = {"unet.blocks.0.weight": torch.zeros(1), "encoder.layers.0.weight": torch.zeros(1)}
        converted = self.convert(
            weights, {"blocks.0": {"format": "asym_w4a8_int8"}, "layers.0": {"format": "int8_tensorwise"}}
        )
        self.assertIn("unet.blocks.0.comfy_quant", converted)
        self.assertIn("encoder.layers.0.comfy_quant", converted)

    def test_ambiguous_suffix_is_rejected_without_partial_mutation(self):
        weights = {"a.blocks.0.weight": torch.zeros(1), "b.blocks.0.weight": torch.zeros(1)}
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            self.convert(weights, {"blocks.0": {"format": "asym_w4a8_int8"}})
        self.assertEqual(len(weights), 2)

    def test_suffix_matching_respects_module_boundary(self):
        weights = {"notblocks.0.weight": torch.zeros(1), "unet.blocks.0.weight": torch.zeros(1)}
        converted = self.convert(weights, {"blocks.0": {"format": "asym_w4a8_int8"}})
        self.assertIn("unet.blocks.0.comfy_quant", converted)
        self.assertNotIn("notblocks.0.comfy_quant", converted)


if __name__ == "__main__":
    unittest.main()
