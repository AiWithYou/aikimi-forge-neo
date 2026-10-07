"""GGUF norms must be usable when Diffusers skips the checkpoint name converter."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gguf as gguf_format
import torch
from diffusers.loaders import single_file_model
from diffusers.quantizers.gguf.utils import GGUFLinear, GGUFParameter

from modules_forge.qwen_image21 import gguf, local_source


def fixture():
    model = torch.nn.Module()
    model.text_norm = torch.nn.RMSNorm(4, eps=1e-6)
    model.norm_q = torch.nn.RMSNorm(2, eps=1e-6)
    model.norm_k = torch.nn.RMSNorm(2, eps=1e-6)
    model.plain_norm = torch.nn.RMSNorm(4)
    model.linear = GGUFLinear(256, 2, compute_dtype=torch.bfloat16)
    expected_state = model.state_dict()
    values = {
        "text_norm": torch.tensor([0.25, 0.5, 1, 2], dtype=torch.bfloat16),
        "norm_q": torch.tensor([0.5, 1.5], dtype=torch.bfloat16),
        "norm_k": torch.tensor([1.25, 0.75], dtype=torch.bfloat16),
    }
    for name, value in values.items():
        getattr(model, name).weight = GGUFParameter(
            value.view(torch.uint8), quant_type=gguf_format.GGMLQuantizationType.BF16
        )
    model.linear.weight = GGUFParameter(
        torch.zeros((2, 144), dtype=torch.uint8), quant_type=gguf_format.GGMLQuantizationType.Q4_K
    )
    model_class = SimpleNamespace(
        from_config=lambda _: SimpleNamespace(state_dict=lambda: expected_state),
        # Direct Diffusers tensor names bypass checkpoint_mapping_fn in the real loader.
        from_single_file=Mock(return_value=model),
    )
    return model, model_class, values


class GGUFNormalizationTests(unittest.TestCase):
    def assert_usable(self, result, model, values, linear_weight, plain_weight):
        self.assertIs(result, model)
        for name, expected in values.items():
            with self.subTest(norm=name):
                norm = getattr(result, name)
                inputs = torch.ones((1, 2, expected.numel()), dtype=torch.bfloat16)
                output = norm(inputs)
                torch.testing.assert_close(output, expected.expand_as(inputs), atol=1e-3, rtol=1e-3)
                self.assertEqual(norm.weight.dtype, torch.bfloat16)
                self.assertFalse(hasattr(norm.weight, "quant_type"))
        self.assertIs(result.linear.weight, linear_weight)
        self.assertEqual(result.linear.weight.quant_type, gguf_format.GGMLQuantizationType.Q4_K)
        self.assertIs(result.plain_norm.weight, plain_weight)

    def test_local_direct_names_restore_text_and_attention_norms_without_dequantizing_linear(self):
        model, model_class, values = fixture()
        linear, plain = model.linear.weight, model.plain_norm.weight
        previous = dict(single_file_model.SINGLE_FILE_LOADABLE_CLASSES)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = local_source.load_single(root / "direct.gguf", model_class, root)
        self.assert_usable(result, model, values, linear, plain)
        model_class.from_single_file.assert_called_once()
        self.assertEqual(single_file_model.SINGLE_FILE_LOADABLE_CLASSES, previous)

    def test_managed_direct_names_restore_norms_and_preserve_quantized_linear(self):
        model, model_class, values = fixture()
        linear, plain = model.linear.weight, model.plain_norm.weight
        previous = dict(single_file_model.SINGLE_FILE_LOADABLE_CLASSES)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("diffusers.QwenImage21Transformer2DModel", model_class, create=True),
        ):
            root = Path(directory)
            result = gguf.load_transformer(root / "model", root / "direct.gguf")
        self.assert_usable(result, model, values, linear, plain)
        self.assertEqual(single_file_model.SINGLE_FILE_LOADABLE_CLASSES, previous)


if __name__ == "__main__":
    unittest.main()
