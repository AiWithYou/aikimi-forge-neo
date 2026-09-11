import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import save_file

from tools.convert_anima38_int8_convrot import (
    ANIMA38_BLOCK_COUNT,
    ANIMA38_PROFILE,
    ANIMA38_QUANTIZED_WEIGHT_KEYS,
    ANIMA38_REQUIRED_METADATA,
    ANIMA38_V11_BUNDLE_ARCHITECTURE,
    ANIMA38_V11_CONNECTOR_PREFIX,
    ANIMA38_V11_PROFILE,
    conversion_profile,
    validate_anima38_source,
    write_sha256_sidecar,
)
from tools.convert_krea2_int8_convrot import output_metadata


class ConvertAnima38Int8ConvRotTests(unittest.TestCase):
    def test_profile_quantizes_only_main_attention_and_mlp_weights(self):
        self.assertEqual(ANIMA38_BLOCK_COUNT, 52)
        self.assertEqual(len(ANIMA38_QUANTIZED_WEIGHT_KEYS), 520)
        self.assertIn(
            "net.blocks.0.self_attn.q_proj.weight",
            ANIMA38_QUANTIZED_WEIGHT_KEYS,
        )
        self.assertIn(
            "net.blocks.51.mlp.layer2.weight",
            ANIMA38_QUANTIZED_WEIGHT_KEYS,
        )
        self.assertNotIn(
            "net.blocks.0.adaln_modulation_self_attn.1.weight",
            ANIMA38_QUANTIZED_WEIGHT_KEYS,
        )
        self.assertNotIn(
            "net.x_embedder.proj.1.weight",
            ANIMA38_QUANTIZED_WEIGHT_KEYS,
        )

    def test_source_validation_requires_the_pro52_metadata_and_all_weights(self):
        tensors = {
            key: torch.zeros((1, 256), dtype=torch.bfloat16)
            for key in ANIMA38_QUANTIZED_WEIGHT_KEYS
        }
        with TemporaryDirectory() as directory:
            source = Path(directory) / "Anima-3.8B.safetensors"
            save_file(tensors, source, metadata=ANIMA38_REQUIRED_METADATA)

            keys, metadata = validate_anima38_source(source, 256)

        self.assertEqual(set(keys), ANIMA38_QUANTIZED_WEIGHT_KEYS)
        self.assertEqual(metadata["new_block_count"], "52")

    def test_source_validation_rejects_a_different_expansion_manifest(self):
        tensors = {
            key: torch.zeros((1, 256), dtype=torch.bfloat16)
            for key in ANIMA38_QUANTIZED_WEIGHT_KEYS
        }
        metadata = dict(ANIMA38_REQUIRED_METADATA)
        metadata["new_block_count"] = "40"
        with TemporaryDirectory() as directory:
            source = Path(directory) / "wrong.safetensors"
            save_file(tensors, source, metadata=metadata)

            with self.assertRaisesRegex(ValueError, "pinned Anima 3.8B Pro52"):
                validate_anima38_source(source, 256)

    def test_v11_profile_preserves_the_bundled_semantic_connector(self):
        keys = [
            *ANIMA38_QUANTIZED_WEIGHT_KEYS,
            f"{ANIMA38_V11_CONNECTOR_PREFIX}quality_anchor.layer_mix_logits",
        ]
        metadata = {
            **ANIMA38_REQUIRED_METADATA,
            "architecture": ANIMA38_V11_BUNDLE_ARCHITECTURE,
            "anima_v2_bundle_format": "1",
        }

        profile, preserved = conversion_profile(keys, metadata)

        self.assertEqual(profile, ANIMA38_V11_PROFILE)
        self.assertIn("semantic_connector_v2", preserved)

    def test_v11_profile_rejects_missing_connector_tensors(self):
        metadata = {
            **ANIMA38_REQUIRED_METADATA,
            "architecture": ANIMA38_V11_BUNDLE_ARCHITECTURE,
            "anima_v2_bundle_format": "1",
        }

        with self.assertRaisesRegex(ValueError, "no Semantic Connector v2"):
            conversion_profile(list(ANIMA38_QUANTIZED_WEIGHT_KEYS), metadata)

    def test_output_metadata_records_the_anima_profile(self):
        metadata = output_metadata(
            Path("Anima-3.8B.safetensors"),
            Path("Anima-3.8B-int8-convrot.safetensors"),
            ANIMA38_REQUIRED_METADATA,
            256,
            quantized_weight_keys=ANIMA38_QUANTIZED_WEIGHT_KEYS,
            metadata_updates={"forge.quantization.profile": ANIMA38_PROFILE},
        )

        self.assertEqual(metadata["forge.quantization.quantized_layers"], "520")
        self.assertEqual(metadata["forge.quantization.profile"], ANIMA38_PROFILE)
        self.assertEqual(
            metadata["modelspec.quantization"], "int8_tensorwise+convrot"
        )

    def test_checksum_sidecar_is_written_atomically(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "Anima-3.8B-int8-convrot.safetensors"
            output.write_bytes(b"anima-38-test")

            sidecar = write_sha256_sidecar(output)

            self.assertEqual(
                sidecar.read_text(encoding="ascii"),
                "b04f97c0d138741d02f840c08d77578fb58917a59ec214ad441a681b6691d3c5  "
                "Anima-3.8B-int8-convrot.safetensors\n",
            )
            self.assertFalse(Path(f"{sidecar}.part").exists())
            with self.assertRaises(FileExistsError):
                write_sha256_sidecar(output)


if __name__ == "__main__":
    unittest.main()
