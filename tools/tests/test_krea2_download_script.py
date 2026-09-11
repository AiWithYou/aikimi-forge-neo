from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "download_krea2_int8_convrot_models.ps1"
BATCH_PATH = ROOT / "download_krea2_int8_convrot_models.bat"


class Krea2Int8DownloadScriptTests(unittest.TestCase):
    def test_batch_launcher_uses_int8_downloader(self):
        launcher = BATCH_PATH.read_text(encoding="utf-8")

        self.assertIn("download_krea2_int8_convrot_models.ps1", launcher)
        self.assertNotIn("download_krea2_bnb_nf4_models.ps1", launcher)

    def test_model_urls_use_immutable_hugging_face_revisions(self):
        script = SCRIPT_PATH.read_text(encoding="utf-8")
        urls = re.findall(r'^\s*Url = "([^"]+)"', script, flags=re.MULTILINE)

        self.assertEqual(len(urls), 3)
        for url in urls:
            with self.subTest(url=url):
                self.assertRegex(url, r"/resolve/[0-9a-f]{40}/")
                self.assertNotIn("/resolve/main/", url)

    def test_expected_sizes_remain_pinned_for_resume_validation(self):
        script = SCRIPT_PATH.read_text(encoding="utf-8")
        sizes = [
            int(value)
            for value in re.findall(
                r"^\s*Bytes = \[Int64\](\d+)$", script, flags=re.MULTILINE
            )
        ]

        self.assertEqual(sizes, [13_492_686_496, 5_242_467_968, 253_806_246])

    def test_every_download_has_a_pinned_lfs_sha256(self):
        script = SCRIPT_PATH.read_text(encoding="utf-8")
        hashes = re.findall(
            r'^\s*Sha256 = "([0-9a-f]{64})"$', script, flags=re.MULTILINE
        )

        self.assertEqual(
            hashes,
            [
                "8e4eeda70dd5037ab1ba2bef6b417f9f901e26093117cf397f741fc1fdaaf3f1",
                "54bd5144df0bbc25dd6ccadfcb826b521445a1b06ae5a42570bdd2974ca87094",
                "a70580f0213e67967ee9c95f05bb400e8fb08307e017a924bf3441223e023d1f",
            ],
        )
        self.assertIn("Assert-FileSha256", script)

    def test_default_model_is_native_int8_convrot(self):
        script = SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            'RelativePath = "models\\Stable-diffusion\\krea2_turbo_int8_convrot.safetensors"',
            script,
        )
        self.assertIn(
            'Markers = @("blocks.0.attn.gate.weight_scale", "blocks.0.attn.gate.comfy_quant")',
            script,
        )
        self.assertIn('Write-Host "Diffusion in Low Bits: Automatic"', script)

    def test_source_and_license_provenance_are_recorded(self):
        script = SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertEqual(
            len(re.findall(r'^\s*Source = "https://', script, flags=re.MULTILINE)),
            3,
        )
        self.assertEqual(
            len(re.findall(r'^\s*License = "https://', script, flags=re.MULTILINE)),
            3,
        )


if __name__ == "__main__":
    unittest.main()
