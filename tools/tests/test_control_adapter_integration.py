"""CPU integration checks for adapter cache geometry and cloned attention state."""

from __future__ import annotations

import copy
import math
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from backend import attention
from backend.nn.cnets.t2i_adapter import Adapter
from backend.patcher import controlnet
from backend.patcher.unet import UnetPatcher
from backend.sampling import sampling_function
from tools.tests.test_runtime_efficiency import load_definitions


class TinyAdapter(Adapter):
    def __init__(self):
        super().__init__(channels=[2, 4, 4, 4], nums_rb=1, cin=256, xl=True, sk=True)
        self.calls = 0

    def forward(self, image):
        self.calls += 1
        return super().forward(image)


class ControlAdapterIntegrationTests(unittest.TestCase):
    source = "extensions-builtin/sd_forge_ipadapter/lib_ipadapter/IPAdapterPlus.py"

    def setUp(self):
        self.enterContext(torch.no_grad())
        self.enterContext(torch.random.fork_rng(devices=[]))
        torch.random.default_generator.manual_seed(19)
        self.cpu = torch.device("cpu")

    def unet(self):
        return UnetPatcher(torch.nn.Linear(2, 2), load_device=self.cpu, offload_device=self.cpu)

    def test_xl_cache_uses_the_pixel_unshuffle_aligned_dimensions(self):
        # Resolution Step=8 is an existing UI choice; 72x88 becomes 80x96
        # for the XL adapter's PixelUnshuffle(16), independent of CFG grouping.
        model = TinyAdapter()
        prototype = controlnet.T2IAdapter(model, channels_in=1, device=self.cpu)
        image = torch.linspace(0, 1, 2 * 72 * 88).reshape(2, 1, 72, 88)
        unet = controlnet.apply_controlnet_advanced(self.unet(), prototype, image, 0.5, 0.0, 1.0)
        adapter = unet.controlnet_linked_list
        first = adapter.get_control(torch.zeros(4, 4, 9, 11), torch.ones(1), {}, 2)
        cached = [weakref.ref(value) for value in adapter.control_input if value is not None]
        second = adapter.get_control(torch.zeros(2, 4, 9, 11), torch.ones(1), {}, 1)

        self.assertEqual(model.calls, 1, "An unchanged aligned image must reuse its extracted features.")
        self.assertEqual(adapter.cond_hint.shape[-2:], (80, 96))
        for block in ("input", "middle"):
            for previous, current in zip(first[block], second[block], strict=True):
                if previous is not None:
                    torch.testing.assert_close(previous[:2], current)
                    torch.testing.assert_close(previous[2:], current)

        adapter.get_control(torch.zeros(2, 4, 11, 13), torch.ones(1), {}, 1)
        self.assertEqual(model.calls, 2, "A change in aligned geometry must invalidate the cache.")
        self.assertTrue(all(reference() is None for reference in cached))
        with patch.object(sampling_function.memory_management, "soft_empty_cache"):
            sampling_function.sampling_cleanup(unet)
        self.assertIsNone(adapter.control_input)
        self.assertIsNone(adapter.cond_hint)
        adapter.get_control(torch.zeros(2, 4, 11, 13), torch.ones(1), {}, 1)
        self.assertEqual(model.calls, 3, "A later sampling pass must reconstruct cleaned features.")

    def test_cloned_ipadapter_runs_independent_attention_and_releases_new_embeddings(self):
        namespace = load_definitions(
            self.source,
            {"CrossAttentionPatch", "set_model_patch_replace"},
            {
                "copy": copy,
                "torch": torch,
                "math": math,
                "F": F,
                "attention": SimpleNamespace(attention_function=attention.attention_pytorch),
            },
        )
        install = namespace["set_model_patch_replace"]
        projection = torch.nn.Linear(2, 2, bias=False)
        projection.weight.copy_(torch.tensor([[1.0, -0.5], [0.25, 2.0]]))
        ipadapter = SimpleNamespace(
            ip_layers=SimpleNamespace(
                to_kvs={
                    "1_to_k_ip": torch.nn.Identity(),
                    "1_to_v_ip": projection,
                }
            )
        )
        conditional = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
        unconditional = torch.tensor([[[0.0, 1.0], [2.0, 0.0]]])
        arguments = dict(
            weight=0.75,
            ipadapter=ipadapter,
            number=0,
            cond=conditional,
            uncond=unconditional,
            weight_type="original",
            sigma_start=1.0,
            sigma_end=0.0,
        )
        key = ("input", 0, 0)
        original = self.unet()
        install(original, arguments, key)
        clone = original.clone()

        def append_condition(target):
            embedding = conditional + 0.5
            install(target, dict(arguments, cond=embedding, uncond=unconditional + 0.5, weight=0.25), key)
            return weakref.ref(embedding)

        reference = append_condition(clone)
        query = torch.tensor([[[1.0, 0.5], [0.25, -1.0]], [[-0.5, 1.0], [1.0, 0.25]]])
        context = torch.ones(2, 2, 2)
        values = torch.tensor([[[0.25, 0.5], [0.75, 1.0]]]).repeat(2, 1, 1)
        options = dict(cond_or_uncond=[0, 1], sigmas=torch.tensor([0.5]), n_heads=1, original_shape=(2, 4, 1, 2))

        def expected_attention(keys, value):
            return torch.softmax(query @ keys.transpose(-2, -1) / math.sqrt(2), dim=-1) @ value

        embeddings = torch.cat((conditional, unconditional))
        expected = expected_attention(context, values) + 0.75 * expected_attention(embeddings, projection(embeddings))

        def invoke(model):
            return model.model_options["transformer_options"]["patches_replace"]["attn2"][key](
                query, context, values, options
            )

        torch.testing.assert_close(invoke(original), expected)
        torch.testing.assert_close(
            invoke(clone), expected + 0.25 * expected_attention(embeddings + 0.5, projection(embeddings + 0.5))
        )
        del clone
        self.assertIsNone(reference(), "The original UNet retained the clone's additional image embedding.")
        torch.testing.assert_close(invoke(original), expected)

    def test_negative_image_transform_failure_restores_random_state(self):
        add_noise = load_definitions(self.source, {"image_add_noise"}, {"torch": torch, "TT": transforms})[
            "image_add_noise"
        ]
        before = torch.random.get_rng_state().clone()

        def fail_after_random_draw(*_args, **_kwargs):
            torch.rand(3)
            raise RuntimeError("transform interrupted")

        with patch.object(transforms.ElasticTransform, "forward", new=fail_after_random_draw):
            with self.assertRaisesRegex(RuntimeError, "transform interrupted"):
                add_noise(torch.ones(1, 8, 8, 3), 0.5)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before))


if __name__ == "__main__":
    unittest.main()
