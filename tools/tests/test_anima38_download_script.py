from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "download_anima38_int8_convrot_models.ps1"
BATCH_PATH = ROOT / "download_anima38_int8_convrot_models.bat"


class Anima38DownloadScriptTests(unittest.TestCase):
    def test_batch_launcher_uses_the_anima38_setup(self):
        launcher = BATCH_PATH.read_text(encoding="utf-8")

        self.assertIn("download_anima38_int8_convrot_models.ps1", launcher)

    def test_all_model_urls_use_the_pinned_hugging_face_revision(self):
        script = SCRIPT_PATH.read_text(encoding="utf-8")
        revision = re.search(r'^\$Revision = "([0-9a-f]{40})"$', script, re.MULTILINE)

        self.assertIsNotNone(revision)
        self.assertEqual(
            revision.group(1), "dd05532037130bebe4d94f0d559b968c14ed1279"
        )
        self.assertNotIn("/resolve/main/", script)
        self.assertEqual(script.count("/resolve/$Revision/"), 3)

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

        self.assertEqual(sizes, [4_779_016_600, 88_131_712, 7_504_189_974])
        self.assertEqual(
            hashes,
            [
                "ea289be7c916726d09953c7db9971c82b280e694b5d7c47f8ad9ffad6acb54ba",
                "f9851ac4668ce069f7be7cf99755335c98879b463f3d486aaa731083978f0d71",
                "1432c925752447df86da7b277e3797f077d358bc24e3950685b13cc0e465c7d5",
            ],
        )

    def test_setup_uses_the_520_layer_convrot_profile(self):
        script = SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("convert_anima38_int8_convrot.py", script)
        self.assertIn("-m $ConverterModule", script)
        self.assertIn("--group-size 256", script)
        self.assertIn("anima38_main_attention_mlp_v1", script)
        self.assertIn("$ConvertedBytes = [Int64]4238326342", script)
        self.assertIn("$layerCount -ne 520", script)
        self.assertIn("checksum sidecar is missing", script)
        self.assertIn("Diffusion in Low Bits: Automatic", script)

    def test_released_qwen_is_kept_as_mixed_fp8(self):
        script = SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn("mixed FP8 text encoder", script)
        self.assertIn(
            'RelativePath = "models\\text_encoder\\qwen35_4b.safetensors"',
            script,
        )
        self.assertNotIn("convert_qwen", script.lower())

    def test_source_and_license_provenance_are_recorded(self):
        script = SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertEqual(len(re.findall(r"^\s*Source = \"https://", script, re.MULTILINE)), 3)
        self.assertEqual(len(re.findall(r"^\s*License = \"https://", script, re.MULTILINE)), 3)
        download_signature = script.split("function Download-ModelFile", 1)[1].split(")", 1)[0]
        assert_signature = script.split("function Assert-SafetensorsFile", 1)[1].split(")", 1)[0]
        self.assertIn("[string]$Source", download_signature)
        self.assertIn("[string]$License", download_signature)
        self.assertNotIn("[string]$Source", assert_signature)

    def test_temporary_bf16_source_is_removed_only_after_validation(self):
        script = SCRIPT_PATH.read_text(encoding="utf-8")
        validation = script.index("Assert-AnimaConvRotCheckpoint -Path $OutputPath")
        removal = script.index("Remove-Item -LiteralPath $resolvedSource")

        self.assertLess(validation, removal)
        self.assertIn("[switch]$KeepSource", script)
        self.assertIn("Refusing to remove a source outside", script)


if __name__ == "__main__":
    unittest.main()
