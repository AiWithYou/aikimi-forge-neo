from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modules_forge.packages.comfy import lora


class LoraKeyMapIsolationTests(unittest.TestCase):
    def model(self, key):
        return SimpleNamespace(
            state_dict=lambda: {key: None},
            diffusion_model=SimpleNamespace(config={}),
            config=SimpleNamespace(huggingface_repo="test/model"),
        )

    def test_implicit_maps_do_not_leak_between_models_or_change_old_results(self):
        with patch.object(lora, "unet_to_diffusers", return_value={}):
            for mapper, prefix in ((lora.model_lora_keys_clip, "text"),
                                   (lora.model_lora_keys_unet, "diffusion_model")):
                with self.subTest(mapper=mapper.__name__):
                    first_key = f"{prefix}.first.weight"
                    second_key = f"{prefix}.second.weight"
                    first = mapper(self.model(first_key))
                    snapshot = first.copy()
                    second = mapper(self.model(second_key))
                    self.assertNotIn(first_key, second.values())
                    self.assertIn(second_key, second.values())
                    self.assertEqual(first, snapshot)
                    self.assertIsNot(first, second)

    def test_explicit_map_is_extended_in_place(self):
        with patch.object(lora, "unet_to_diffusers", return_value={}):
            for mapper in (lora.model_lora_keys_clip, lora.model_lora_keys_unet):
                with self.subTest(mapper=mapper.__name__):
                    existing = {"caller_alias": "caller.weight"}
                    result = mapper(self.model("diffusion_model.new.weight"), existing)
                    self.assertIs(result, existing)
                    self.assertEqual(result["caller_alias"], "caller.weight")
                    self.assertIn("diffusion_model.new.weight", result.values())


if __name__ == "__main__":
    unittest.main()
