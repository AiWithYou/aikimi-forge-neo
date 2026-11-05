"""W4A8 conversion keeps the original model and publishes only verified output."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from modules_forge import gpu_fault
from modules_forge.moss_image import checkpoints

PREFIX = "layers.0.proj.q"
WEIGHT = PREFIX + ".weight"
SHAPES = {WEIGHT: (2, 256), "rope.bands": (1,)}


def source(path, value=1):
    save_file(
        {WEIGHT: torch.full((2, 256), value, dtype=torch.bfloat16), "rope.bands": torch.tensor([2.0])},
        str(path), metadata={"_quantization_metadata": '{"layers": {}}'},
    )


def quantized(weight, device):
    del weight, device
    return (
        torch.zeros(2, 128, dtype=torch.int8),
        torch.ones(2, 16).to(torch.float8_e4m3fn),
        torch.ones(2),
        torch.zeros(16),
    )


class ConvertMossW4A8Tests(unittest.TestCase):
    def run_conversion(self, src, dst, quantize=quantized):
        from tools import convert_moss_image_w4a8

        with (
            patch.object(checkpoints, "expected_shapes", return_value=SHAPES),
            patch.object(checkpoints, "QUANTIZED_KEYS", frozenset({WEIGHT})),
            patch.object(convert_moss_image_w4a8, "quantize_weight", side_effect=quantize),
            patch.object(convert_moss_image_w4a8, "validate_conversion_resources"),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "get_device_capability", return_value=(8, 6)),
        ):
            convert_moss_image_w4a8.convert(src, dst, torch.device("cuda:0"))

    def test_convert_writes_verified_separate_file_without_changing_source(self):
        with tempfile.TemporaryDirectory() as folder:
            src, dst = Path(folder) / "original.safetensors", Path(folder) / "result.safetensors"
            source(src)
            before = src.read_bytes()
            self.run_conversion(src, dst)
            self.assertEqual(src.read_bytes(), before)
            self.assertFalse(Path(str(dst) + ".part").exists())
            with (
                patch.object(checkpoints, "expected_shapes", return_value=SHAPES),
                patch.object(checkpoints, "QUANTIZED_KEYS", frozenset({WEIGHT})),
            ):
                self.assertEqual(checkpoints.precision(dst), "w4a8")
            with safe_open(dst, framework="pt") as output:
                self.assertTrue(torch.equal(output.get_tensor("rope.bands"), torch.tensor([2.0])))
                self.assertEqual(len(output.metadata()["forge.quantization.source_sha256"]), 64)
                self.assertNotIn("_quantization_metadata", output.metadata())

    def test_changed_source_and_existing_output_are_never_published_or_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            src, dst = Path(folder) / "original.safetensors", Path(folder) / "result.safetensors"
            source(src)

            def changed(weight, device):
                source(src, 2)
                return quantized(weight, device)

            with self.assertRaisesRegex(ValueError, "source.*changed"):
                self.run_conversion(src, dst, changed)
            self.assertFalse(dst.exists())
            self.assertFalse(Path(str(dst) + ".part").exists())
            dst.write_bytes(b"do not overwrite")
            with self.assertRaises(FileExistsError):
                self.run_conversion(src, dst)
            self.assertEqual(dst.read_bytes(), b"do not overwrite")
            with self.assertRaises(ValueError):
                self.run_conversion(src, src)

    def test_gpu_fault_stop_prevents_starting_cuda(self):
        from tools import convert_moss_image_w4a8

        with tempfile.TemporaryDirectory() as folder:
            src, dst = Path(folder) / "original.safetensors", Path(folder) / "result.safetensors"
            source(src)
            with (
                patch.object(gpu_fault.STATE, "ensure_available", side_effect=gpu_fault.GPUFaultError),
                patch.object(torch.cuda, "is_available") as available,
                self.assertRaises(gpu_fault.GPUFaultError),
            ):
                convert_moss_image_w4a8.convert(src, dst, torch.device("cuda:0"))
            available.assert_not_called()
            self.assertFalse(dst.exists())

    def test_fatal_cuda_failure_latches_fault_and_removes_partial_output(self):
        with tempfile.TemporaryDirectory() as folder:
            src, dst = Path(folder) / "original.safetensors", Path(folder) / "result.safetensors"
            source(src)

            def failure(weight, device):
                raise RuntimeError("CUDA error: illegal memory access")

            with patch.object(gpu_fault.STATE, "trip") as trip, self.assertRaisesRegex(RuntimeError, "CUDA"):
                self.run_conversion(src, dst, failure)
            trip.assert_called_once_with("gpu_execution", "moss-w4a8-converter")
            self.assertFalse(dst.exists())
            self.assertFalse(Path(str(dst) + ".part").exists())


if __name__ == "__main__":
    unittest.main()
