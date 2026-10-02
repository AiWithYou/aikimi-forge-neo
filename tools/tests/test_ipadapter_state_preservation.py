from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

import torch
import torchvision.transforms as transforms

from backend import utils
from tools.tests.test_runtime_efficiency import load_definitions


class IPAdapterStatePreservationTests(unittest.TestCase):
    source = "extensions-builtin/sd_forge_ipadapter/lib_ipadapter/IPAdapterPlus.py"

    def test_appending_an_adapter_to_a_clone_preserves_the_original_attention_patch(self):
        namespace = load_definitions(self.source, {"CrossAttentionPatch", "set_model_patch_replace"}, {"copy": copy})
        install = namespace["set_model_patch_replace"]
        original = SimpleNamespace(model_options={"transformer_options": {}})
        conditional = torch.ones((1, 1, 2))
        arguments = {
            "weight": 1.0,
            "ipadapter": object(),
            "number": 0,
            "cond": conditional,
            "uncond": torch.zeros_like(conditional),
            "weight_type": "original",
        }
        key = ("input", 0, 0)
        install(original, arguments, key)
        clone = SimpleNamespace(model_options=utils.deepcopy_(original.model_options))

        install(clone, dict(arguments, weight=0.5), key)

        original_patch = original.model_options["transformer_options"]["patches_replace"]["attn2"][key]
        clone_patch = clone.model_options["transformer_options"]["patches_replace"]["attn2"][key]
        self.assertEqual(original_patch.weights, [1.0])
        self.assertEqual(clone_patch.weights, [1.0, 0.5])
        self.assertIs(original_patch.conds[0], conditional)
        self.assertIs(clone_patch.conds[0], conditional)

    def test_negative_image_preprocessing_preserves_the_callers_random_state(self):
        noise = load_definitions(self.source, {"image_add_noise"}, {"torch": torch, "TT": transforms})[
            "image_add_noise"
        ]
        image = torch.linspace(0, 1, 8 * 8 * 3).reshape(1, 8, 8, 3)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(1234)
            before = torch.random.get_rng_state().clone()

            noise(image, 0.5)

            self.assertTrue(torch.equal(torch.random.get_rng_state(), before))

    def test_negative_image_preprocessing_is_reproducible_across_caller_seeds(self):
        noise = load_definitions(self.source, {"image_add_noise"}, {"torch": torch, "TT": transforms})[
            "image_add_noise"
        ]
        image = torch.ones((1, 8, 8, 3))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(1234)
            first = noise(image, 0.5)
            torch.random.default_generator.manual_seed(4321)
            second = noise(image, 0.5)

        torch.testing.assert_close(first, second)


if __name__ == "__main__":
    unittest.main()
