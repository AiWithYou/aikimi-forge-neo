import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from safetensors import safe_open
import torch

from tools.convert_krea2_int8_convrot import (
    GIB,
    KREA2_QUANTIZED_WEIGHT_KEYS,
    TensorSpec,
    build_safetensors_header,
    companion_keys,
    conversion_resource_requirements,
    quant_config,
    quant_config_tensor,
    validate_conversion_resources,
    write_tensor_bytes,
)


class ConvertKrea2Int8ConvRotTests(unittest.TestCase):
    def test_profile_matches_official_224_weight_layout(self):
        self.assertEqual(len(KREA2_QUANTIZED_WEIGHT_KEYS), 224)
        self.assertIn("blocks.0.attn.gate.weight", KREA2_QUANTIZED_WEIGHT_KEYS)
        self.assertIn("blocks.27.mlp.up.weight", KREA2_QUANTIZED_WEIGHT_KEYS)
        self.assertNotIn("blocks.28.attn.gate.weight", KREA2_QUANTIZED_WEIGHT_KEYS)
        self.assertNotIn("first.weight", KREA2_QUANTIZED_WEIGHT_KEYS)
        self.assertNotIn("last.linear.weight", KREA2_QUANTIZED_WEIGHT_KEYS)
        self.assertNotIn("txtfusion.projector.weight", KREA2_QUANTIZED_WEIGHT_KEYS)

    def test_quant_metadata_matches_native_comfy_format(self):
        tensor = quant_config_tensor(256)
        self.assertEqual(tensor.numel(), 72)
        self.assertEqual(json.loads(tensor.numpy().tobytes()), quant_config(256))
        self.assertEqual(
            quant_config(256),
            {
                "format": "int8_tensorwise",
                "convrot": True,
                "convrot_groupsize": 256,
            },
        )

    def test_companion_keys_use_safetensors_naming_contract(self):
        self.assertEqual(
            companion_keys("blocks.0.attn.gate.weight"),
            (
                "blocks.0.attn.gate.weight_scale",
                "blocks.0.attn.gate.comfy_quant",
            ),
        )

    def test_resource_requirements_use_bounded_streaming_ram_and_disk_headroom(self):
        requirements = conversion_resource_requirements(20 * GIB)

        self.assertEqual(requirements["estimated_output_bytes"], 14 * GIB)
        self.assertEqual(requirements["required_free_disk_bytes"], 16 * GIB)
        self.assertEqual(requirements["required_available_memory_bytes"], 6 * GIB)

    def test_streaming_header_produces_a_valid_safetensors_file(self):
        specs = [
            TensorSpec("a", "BF16", (2, 2)),
            TensorSpec("b", "U8", (3,)),
        ]
        header, data_bytes = build_safetensors_header(
            specs, {"modelspec.title": "streaming-test"}
        )

        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "streamed.safetensors"
            with checkpoint.open("wb") as destination:
                destination.write(header)
                destination.write(bytes(data_bytes))

            with safe_open(checkpoint, framework="pt", device="cpu") as handle:
                self.assertEqual(list(handle.keys()), ["a", "b"])
                self.assertEqual(tuple(handle.get_tensor("a").shape), (2, 2))
                self.assertEqual(tuple(handle.get_tensor("b").shape), (3,))
                self.assertEqual(handle.metadata()["modelspec.title"], "streaming-test")

    def test_tensor_writer_preserves_bfloat16_bytes(self):
        tensor = torch.tensor([[1.0, -2.5], [3.25, 0.0]], dtype=torch.bfloat16)
        with TemporaryDirectory() as directory:
            output = Path(directory) / "tensor.bin"
            with output.open("wb", buffering=0) as destination:
                write_tensor_bytes(destination, tensor, tensor.numel() * 2, "test")

            self.assertEqual(
                output.read_bytes(),
                bytes(memoryview(tensor.reshape(-1).view(torch.uint8).numpy())),
            )

    def test_resource_preflight_fails_clearly_when_ram_is_insufficient(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.safetensors"
            source.write_bytes(b"x" * 4096)

            with self.assertRaisesRegex(RuntimeError, "Insufficient available RAM"):
                validate_conversion_resources(
                    source,
                    root / "output.safetensors",
                    available_memory_bytes=1,
                    free_disk_bytes=10 * GIB,
                )

    def test_resource_preflight_fails_clearly_when_disk_is_insufficient(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.safetensors"
            source.write_bytes(b"x" * 4096)

            with self.assertRaisesRegex(RuntimeError, "Insufficient free output-disk"):
                validate_conversion_resources(
                    source,
                    root / "output.safetensors",
                    available_memory_bytes=10 * GIB,
                    free_disk_bytes=1,
                )


if __name__ == "__main__":
    unittest.main()
