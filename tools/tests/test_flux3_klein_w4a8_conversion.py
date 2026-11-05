"""CPU-only converter contract tests; real CUDA conversion is deliberately not run."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from modules_forge import local_assets
from modules_forge.flux3_klein import checkpoints
from tools.tests.test_flux3_klein_w4a8 import FUSED, shapes


def source(path, model="base", fused=True):
    weights = shapes(model)
    if not fused:
        for block in ("single_blocks.0", "single_blocks.1"):
            weights.pop(block + ".qkv.weight")
            for projection in ("q", "k", "v"):
                weights[block + "." + projection + "_proj.weight"] = (16, 256)
    save_file({key: torch.ones(shape, dtype=torch.bfloat16) for key, shape in weights.items()}, str(path))


def quantized(weight, device):
    del device
    rows, columns = weight.shape
    return (
        torch.zeros((rows, columns // 2), dtype=torch.int8),
        torch.ones((rows, columns // 16)).to(torch.float8_e4m3fn),
        torch.ones(rows),
        torch.linspace(-1, 1, 16),
    )


class Flux3W4ConversionTests(unittest.TestCase):
    def convert(self, src, dst, **kwargs):
        from tools import convert_flux3_klein_w4a8 as converter

        with (
            patch.object(checkpoints, "expected_shapes", side_effect=shapes),
            patch.object(checkpoints, "quantized_keys", return_value=FUSED),
            patch.object(converter, "quantize_weight", side_effect=quantized),
            patch.object(converter, "validate_conversion_resources"),
            patch.object(converter.gpu_fault.STATE, "ensure_available"),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "get_device_capability", return_value=(8, 6)),
        ):
            return converter.convert(src, dst, torch.device("cuda:0"), **kwargs)

    def test_fused_and_separate_convert_without_changing_source(self):
        for fused in (True, False):
            with self.subTest(fused=fused), tempfile.TemporaryDirectory() as directory:
                src, dst = Path(directory) / "original.safetensors", Path(directory) / "result.safetensors"
                source(src, fused=fused)
                before = src.read_bytes()
                self.convert(src, dst)
                self.assertEqual(before, src.read_bytes())
                self.assertFalse(Path(str(dst) + ".part").exists())
                with (
                    patch.object(checkpoints, "expected_shapes", side_effect=shapes),
                    patch.object(checkpoints, "quantized_keys", return_value=FUSED),
                ):
                    self.assertEqual(checkpoints.header_info(dst), {"model": "base", "precision": "w4a8"})
                with safe_open(dst, framework="pt") as out:
                    self.assertEqual(out.metadata()["forge.quantization.quantized_layers"], "8" if fused else "12")
                    self.assertEqual(len(out.metadata()["forge.quantization.source_sha256"]), 64)
                    self.assertEqual(tuple(out.get_tensor("txt_in.weight").shape), (16, 256))

    def test_changed_source_and_existing_destinations_are_never_published(self):
        with tempfile.TemporaryDirectory() as directory:
            src, dst = Path(directory) / "original.safetensors", Path(directory) / "result.safetensors"
            source(src)
            actual = local_assets.file_identity(src)
            changed = {**actual, "sha256": "f" * 64}
            with patch.object(local_assets, "file_identity", side_effect=[actual, changed]):
                with self.assertRaisesRegex(ValueError, "source.*changed"):
                    self.convert(src, dst)
            self.assertFalse(dst.exists())
            self.assertFalse(Path(str(dst) + ".part").exists())
            dst.write_bytes(b"keep existing output")
            with self.assertRaises(FileExistsError):
                self.convert(src, dst)
            self.assertEqual(dst.read_bytes(), b"keep existing output")
            with self.assertRaises(ValueError):
                self.convert(src, src)


if __name__ == "__main__":
    unittest.main()
