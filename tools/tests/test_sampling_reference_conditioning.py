"""Reference recording keeps conditional rows under weighted CFG."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from backend import attention
from backend.patcher.unet import UnetPatcher
from backend.sampling import sampling_function
from tools.tests.test_api_extras_boundaries import load_classes
from tools.tests.test_gpu_ownership import load_function

REFERENCE = "extensions-builtin/forge_preprocessor_reference/scripts/forge_reference.py"


class TinyReferenceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.predictor = SimpleNamespace(percent_to_sigma=lambda value: 1.0 - value)
        self.calls = []

    def get_dtype(self):
        return self.weight.dtype

    def memory_required(self, shape):
        return 0

    def apply_model(self, x, timestep, c_crossattn, transformer_options):
        self.calls.append(transformer_options["cond_or_uncond"][:])
        value = c_crossattn[:, 0, 0].reshape(-1, 1, 1, 1)
        grid = torch.arange(4, dtype=x.dtype).reshape(1, 1, 2, 2)
        h = (x + value + grid).repeat(1, 512, 1, 1)
        options = {**transformer_options, "block": ("input", 0), "block_index": 0, "n_heads": 1}
        for modifier in transformer_options["block_modifiers"]:
            h = modifier(h, "after", options)
        q = h[:, :256].flatten(2).transpose(1, 2)
        replace = transformer_options["patches_replace"]["attn1"][("input", 0, 0)]
        result = replace(q, q, q, options)
        return h[:, :1] + result[:, :, :1].transpose(1, 2).reshape(-1, 1, 2, 2)


class ReferenceConditioningTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(sampling_function.dynamic_args, "context_handler", None))
        namespace = {
            "torch": torch,
            "attention": attention,
            "Preprocessor": object,
            "PreprocessorParameter": lambda **kwargs: SimpleNamespace(**kwargs),
            "sampling_function_inner": sampling_function.sampling_function_inner,
        }
        for name in ("sdp", "adain", "zero_cat"):
            namespace[name] = load_function(REFERENCE, name, namespace)
        self.preprocessor_class = load_classes(REFERENCE, {"PreprocessorReference"}, namespace).PreprocessorReference

    def run_reference(self, weights, *, use_attn, use_adain):
        model = TinyReferenceModel()
        cpu = torch.device("cpu")
        original = UnetPatcher(model, load_device=cpu, offload_device=cpu)
        batch_size = len(weights)
        latent = torch.zeros(batch_size, 1, 2, 2)
        vae = SimpleNamespace(
            encode=lambda image: image.movedim(-1, 1), first_stage_model=SimpleNamespace(process_in=lambda x: x)
        )
        process = SimpleNamespace(
            seeds=[1],
            sd_model=SimpleNamespace(
                is_sdxl=False, is_inpaint=False, forge_objects=SimpleNamespace(unet=original, vae=vae)
            ),
        )
        reference = self.preprocessor_class("reference", use_attn=use_attn, use_adain=use_adain)
        unit = SimpleNamespace(weight=1.0, threshold_a=0.5, guidance_start=0.0, guidance_end=1.0)
        reference.process_before_every_sampling(process, latent, None, unit=unit)
        denoiser = SimpleNamespace(inner_model=SimpleNamespace(inner_model=process.sd_model), p=process)
        positive = torch.arange(1, batch_size + 1, dtype=torch.float32).reshape(batch_size, 1, 1)
        params = SimpleNamespace(
            x=latent,
            sigma=torch.ones(batch_size),
            text_cond=positive,
            text_uncond=torch.full_like(positive, 3.0),
            image_cond=None,
        )
        composition = [[(row, weight)] for row, weight in enumerate(weights)]
        first = sampling_function.sampling_function(denoiser, params, 2.0, composition)
        for output in first:
            self.assertEqual(output.shape, latent.shape)
            self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(model.calls[0], [0])
        self.assertEqual(model.calls[1], [1, 0])
        self.assertIsNot(process.sd_model.forge_objects.unet, original)
        self.assertEqual(original.model_options, {"transformer_options": {}})
        self.assertFalse(reference.is_recording_style)
        second = sampling_function.sampling_function(denoiser, params, 2.0, composition)
        for output in second:
            self.assertEqual(output.shape, latent.shape)
            self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(params.text_cond, positive)

    def test_weighted_reference_attention_keeps_recording_rows_for_single_and_mixed_batch(self):
        for weights in ([2.0], [0.0], [2.0, 1.0]):
            with self.subTest(weights=weights):
                self.run_reference(weights, use_attn=True, use_adain=False)

    def test_weighted_reference_adain_keeps_recording_rows_for_single_and_mixed_batch(self):
        for weights in ([2.0], [0.0], [2.0, 1.0]):
            with self.subTest(weights=weights):
                self.run_reference(weights, use_attn=False, use_adain=True)

    def test_unit_reference_keeps_existing_attention_and_adain_protocol(self):
        self.run_reference([1.0], use_attn=True, use_adain=True)


if __name__ == "__main__":
    unittest.main()
