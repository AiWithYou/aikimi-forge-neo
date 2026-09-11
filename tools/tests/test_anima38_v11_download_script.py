import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "download_anima38_v11_int8_convrot_models.ps1"
BATCH_PATH = ROOT / "download_anima38_v11_int8_convrot_models.bat"


class Anima38V11DownloadScriptTests(unittest.TestCase):
    def test_batch_launcher_uses_the_v11_setup(self):
        launcher = BATCH_PATH.read_text(encoding="utf-8")

        self.assertIn("download_anima38_v11_int8_convrot_models.ps1", launcher)

    def test_model_urls_use_the_pinned_v11_revision(self):
        script = SCRIPT_PATH.read_text(encoding="utf-8")
        revision = re.search(r'^\$Revision = "([0-9a-f]{40})"$', script, re.MULTILINE)

        self.assertIsNotNone(revision)
        self.assertEqual(
            revision.group(1), "3ef641256377dc4e7efbf35d426ca31c1fe5180b"
        )
        self.assertNotIn("/resolve/main/", script)
        self.assertEqual(script.count("/resolve/$Revision/"), 2)

    def test_download_sizes_and_lfs_hashes_are_pinned(self):
        script = SCRIPT_PATH.read_text(encoding="utf-8")
        sizes = [
            int(value)
            for value in re.findall(
                r"^\s*Bytes = \[Int64\](\d+)$", script, flags=re.MULTILINE
            )
        ]
        hashes = re.findall(
            r'^\s*Sha256 = "([0-9a-f]{64})"$', script, flags=re.MULTILINE
        )

        self.assertEqual(sizes, [4_779_016_600, 8_809_227_318])
        self.assertEqual(
            hashes,
            [
                "ea289be7c916726d09953c7db9971c82b280e694b5d7c47f8ad9ffad6acb54ba",
                "4a458d26b21efa350073422f756d521b4397d9ca5964da4dc6bd9ae258a29629",
            ],
        )

    def test_setup_preserves_connector_v2_and_quantizes_520_dit_layers(self):
        script = SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("convert_anima38_int8_convrot.py", script)
        self.assertIn("-m $ConverterModule", script)
        self.assertIn("--group-size 256", script)
        self.assertIn("anima38_v11_main_attention_mlp_v1", script)
        self.assertIn("$ConvertedBytes = [Int64]5543364574", script)
        self.assertIn("net.anima_v2_connector.semantic_resampler.query_tokens", script)
        self.assertIn("$layerCount -ne 520", script)
        self.assertIn("checksum sidecar is missing", script)
        self.assertIn("Diffusion in Low Bits: Automatic", script)

    def test_v11_uses_bundled_connector_and_released_qwen(self):
        script = SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("bundled BF16 Semantic Connector v2", script)
        self.assertIn("mixed FP8 text encoder", script)
        self.assertIn(
            'RelativePath = "models\\text_encoder\\qwen35_4b.safetensors"',
            script,
        )
        self.assertNotIn("Anima-3.8B-expanded_adapter.safetensors", script)

    def test_temporary_bf16_source_is_removed_only_after_validation(self):
        script = SCRIPT_PATH.read_text(encoding="utf-8")
        validation = script.index("Assert-AnimaConvRotCheckpoint -Path $OutputPath")
        removal = script.index("Remove-Item -LiteralPath $resolvedSource")

        self.assertLess(validation, removal)
        self.assertIn("[switch]$KeepSource", script)
        self.assertIn("Refusing to remove a source outside", script)


if __name__ == "__main__":
    unittest.main()
