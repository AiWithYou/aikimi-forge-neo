import unittest

from modules_forge.moss_image.checkpoints import QUANTIZED_KEYS
from tools.convert_moss_int8_convrot import MOSS_PROFILE, quantized_keys


class ConvertMossTests(unittest.TestCase):
    def test_only_main_moss_projections_are_quantized(self):
        self.assertEqual(quantized_keys(), QUANTIZED_KEYS)
        self.assertEqual(len(quantized_keys()), 252)
        self.assertNotIn("text_embedder.proj.weight", quantized_keys())
        self.assertNotIn("shared_block_modulation.weight", quantized_keys())
        self.assertEqual(MOSS_PROFILE, "moss_sft_main_252_v1")
