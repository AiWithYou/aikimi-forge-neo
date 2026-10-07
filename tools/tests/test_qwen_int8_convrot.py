"""ConvRot checkpoint contracts independent of installed model weights or CUDA."""

from __future__ import annotations

import json
import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from modules_forge import local_assets
from modules_forge.qwen_image21 import convrot_int8, local_source
from modules_forge.qwen_image21.convrot_int8_runtime import ConvRotLinear, load_model
from tools.tests.test_additional_module_identity import stub_modules
from tools.tests.test_local_model_sources import qwen_fixture, tensor_file


def checkpoint_fixture(path):
    header, payload = {}, bytearray()
    for index in range(32):
        for name, shape in convrot_int8.LAYER_SHAPES.items():
            stem = f"transformer_blocks.{index}.{name}"
            header[stem + ".weight"] = {"dtype": "I8", "shape": list(shape)}
            header[stem + ".weight_scale"] = {"dtype": "F32", "shape": [shape[0], 1]}
            settings = json.dumps(convrot_int8.SETTINGS).encode()
            start = len(payload)
            payload.extend(settings)
            header[stem + ".comfy_quant"] = {
                "dtype": "U8",
                "shape": [len(settings)],
                "data_offsets": [start, len(payload)],
            }
    header.update({"img_in.weight": {"dtype": "BF16"}, "txt_in.in_layer.weight": {"dtype": "BF16"}})
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    return header


class ConvRotSourceTests(unittest.TestCase):
    def test_content_selects_int8_despite_neutral_filename(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "custom.safetensors"
            header = checkpoint_fixture(path)
            with patch.object(local_assets, "read_header", return_value=header):
                self.assertEqual(local_source.preferred_precision(path), "int8")
                info = convrot_int8.inspect_checkpoint(path)
            self.assertEqual(info["quantized_linears"], 224)
            self.assertEqual(info["convrot_groupsize"], 256)

    def test_unquantized_safetensors_with_int8_filename_selects_bf16(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "int8_convrot.safetensors"
            keys = {f"transformer_blocks.{i}.attn.to_q.weight": [1, 1] for i in range(32)}
            keys.update({"img_in.weight": [1, 1], "txt_in.in_layer.weight": [1, 1]})
            tensor_file(path, keys)
            self.assertEqual(local_source.preferred_precision(path), "bf16")

    def test_missing_layer_wrong_scale_and_wrong_weight_shape_are_rejected(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "weights.safetensors"
            for change in ("missing", "scale", "weight", "extra"):
                with self.subTest(change=change):
                    header = checkpoint_fixture(path)
                    stem = "transformer_blocks.0.attn.to_q"
                    if change == "missing":
                        del header[stem + ".comfy_quant"]
                    elif change == "scale":
                        header[stem + ".weight_scale"]["shape"] = [1]
                    elif change == "weight":
                        header[stem + ".weight"]["shape"] = [4096, 2048]
                    else:
                        header["unexpected.weight_scale"] = {"dtype": "F32", "shape": [1]}
                    with self.assertRaises(ValueError):
                        convrot_int8.validate_header(header)

    def test_quantization_recipe_must_match_including_types(self):
        for change in (
            {"convrot": False},
            {"convrot": 1},
            {"convrot_groupsize": 128},
            {"convrot_groupsize": 256.0},
            {"format": "fp8"},
            {"unknown": True},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                convrot_int8.validate_settings(convrot_int8.SETTINGS | change, "layer")

    def test_conflicting_prefixes_are_rejected(self):
        with self.assertRaises(ValueError):
            convrot_int8.normalize_keys({"img_in.weight": {}, "diffusion_model.img_in.weight": {}})

    def test_convrot_requires_int8_and_shared_components_without_standard_body(self):
        with TemporaryDirectory() as folder:
            runtime = Path(folder) / "runtime"
            qwen_fixture(runtime / "model", transformer=False)
            path = Path(folder) / "custom.safetensors"
            header = checkpoint_fixture(path)
            original = local_assets.read_header
            with patch.object(
                local_assets, "read_header", side_effect=lambda p: header if Path(p) == path else original(p)
            ):
                source = local_source.resolve(runtime, str(path), "", "int8")
                self.assertEqual(source["format"], "int8_convrot")
                self.assertFalse((runtime / "model/transformer").exists())
                for precision in ("bf16", "w4a8"):
                    with self.subTest(precision=precision), self.assertRaisesRegex(ValueError, "ConvRot"):
                        local_source.resolve(runtime, str(path), "", precision)


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(256, 256, bias=False)
        self.norm = nn.LayerNorm(256)

    @classmethod
    def from_config(cls, _config):
        return cls()


class ConvRotRuntimeTests(unittest.TestCase):
    def test_lora_residual_keeps_convrot_weights_and_fp32_scales(self):
        from modules_forge.qwen_image21.style_lora_runtime import StyleLinear

        base = ConvRotLinear(256, 256)
        weight = torch.zeros(256, 256, dtype=torch.int8)
        scales = torch.full((256, 1), 0.001234567, dtype=torch.float32)
        base.weight, base.weight_scale = weight, scales
        hidden = torch.ones(1, 2, 256, dtype=torch.bfloat16)
        down = torch.full((1, 256), 1 / 256, dtype=torch.bfloat16)
        up = torch.ones(256, 1, dtype=torch.bfloat16)
        adapter = StyleLinear(base, down, up, 0.5)
        kernel = Mock(return_value=torch.zeros_like(hidden))
        with stub_modules({"comfy_kitchen": SimpleNamespace(int8_linear=kernel)}):
            output = adapter(hidden)
        torch.testing.assert_close(output, torch.full_like(hidden, 0.5), rtol=0, atol=0)
        self.assertIs(adapter.weight, weight)
        self.assertIs(base.weight_scale, scales)
        self.assertEqual(base.weight.dtype, torch.int8)
        self.assertEqual(base.weight_scale.dtype, torch.float32)
        self.assertEqual((adapter.in_features, adapter.out_features), (256, 256))

    def test_loader_preserves_int8_and_float32_scales_with_bf16_norms(self):
        state = TinyTransformer().state_dict()
        state["projection.weight"] = torch.ones(256, 256, dtype=torch.int8)
        scales = torch.full((256, 1), 0.001234567, dtype=torch.float32)
        state["projection.weight_scale"] = scales
        state["projection.comfy_quant"] = torch.tensor(
            list(json.dumps(convrot_int8.SETTINGS).encode()), dtype=torch.uint8
        )
        with (
            patch("modules_forge.qwen_image21.convrot_int8_runtime.validate_dependency"),
            patch.object(
                convrot_int8, "inspect_checkpoint", return_value={"quantized_linears": 1, "convrot_groupsize": 256}
            ),
            patch("safetensors.torch.load_file", return_value=state),
        ):
            model, info = load_model(Path("weights.safetensors"), TinyTransformer)
        self.assertIsInstance(model.projection, ConvRotLinear)
        self.assertEqual((model.projection.in_features, model.projection.out_features), (256, 256))
        self.assertEqual(model.projection.weight.dtype, torch.int8)
        self.assertEqual(model.projection.weight_scale.dtype, torch.float32)
        torch.testing.assert_close(model.projection.weight_scale, scales, rtol=0, atol=0)
        self.assertEqual(model.norm.weight.dtype, torch.bfloat16)
        self.assertEqual(info["quantized_linears"], 1)
        model.to("cpu")
        self.assertEqual(model.projection.weight_scale.dtype, torch.float32)
        self.assertFalse(any(tensor.is_meta for tensor in model.state_dict().values()))

    def test_linear_dispatches_convrot_recipe_without_requantizing_weights(self):
        layer = ConvRotLinear(256, 256)
        layer.weight = torch.zeros(256, 256, dtype=torch.int8)
        layer.weight_scale = torch.ones(256, 1, dtype=torch.float32)
        hidden = torch.ones(1, 2, 256, dtype=torch.bfloat16)
        expected = torch.zeros_like(hidden)
        kernel = Mock(return_value=expected)
        with stub_modules({"comfy_kitchen": SimpleNamespace(int8_linear=kernel)}):
            self.assertIs(layer(hidden), expected)
        kernel.assert_called_once_with(
            hidden,
            layer.weight,
            layer.weight_scale,
            None,
            out_dtype=torch.bfloat16,
            convrot=True,
            convrot_groupsize=256,
        )


if __name__ == "__main__":
    unittest.main()
