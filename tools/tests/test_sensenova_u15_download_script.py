import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "download_sensenova_u15_int8.ps1"
BATCH = ROOT / "download_sensenova_u15_int8.bat"


class SenseNovaDownloadScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = SCRIPT.read_text(encoding="utf-8")
        cls.batch = BATCH.read_text(encoding="utf-8")

    def test_final_convrot_download_is_immutable_and_integrity_checked(self):
        self.assertIn(
            'ModelRevision = "57de22ad4e2fc24c77f56dfe45dbb87a60dfebee"',
            self.script,
        )
        self.assertIn("ModelBytes = [Int64]17734813848", self.script)
        self.assertIn(
            'ModelSha256 = "cf6ed9ee3be516612b7fe083edfc7c9dd5d059cc759e300d2cf1f2726c0d250e"',
            self.script,
        )
        self.assertIn("Get-FileHash -LiteralPath $Path -Algorithm SHA256", self.script)
        self.assertIn("Assert-ConvRotSafetensorsHeader", self.script)
        self.assertIn('Contains(".comfy_quant")', self.script)
        self.assertIn("convRotLayerCount -ne 588", self.script)

    def test_runtime_source_is_pinned_and_blob_verified(self):
        self.assertIn(
            'SourceRevision = "e6dfd45762eb46f805067fe079c14bcb643ccccd"',
            self.script,
        )
        self.assertIn("Get-GitBlobSha1", self.script)
        self.assertIn('$_.path -like "SenseNova/*"', self.script)
        self.assertIn('$_.path -like "SenseNova-U1.5-8B-MoT/*"', self.script)
        self.assertIn("if ($tree.truncated)", self.script)
        self.assertIn("[System.IO.Path]::IsPathRooted($relativePath)", self.script)
        self.assertIn("$pathParts -contains \"..\"", self.script)
        self.assertIn("$targetPath.StartsWith($runtimePrefix", self.script)
        self.assertIn("comfy_kitchen", self.script)

    def test_official_8step_lora_is_pinned_and_integrity_checked(self):
        self.assertIn(
            'LoraRevision = "e909f4636d119d65fe4cba8770c19daff2ac102e"',
            self.script,
        )
        self.assertIn("LoraBytes = [Int64]814867236", self.script)
        self.assertIn(
            'LoraSha256 = "3ef32180cdf1e30a870a83f4f136e897ea50b7ee467f863d75633464ebb25708"',
            self.script,
        )
        self.assertIn("LoraTargetCount = 294", self.script)
        self.assertIn("Assert-Official8StepLoraHeader", self.script)
        self.assertIn('Contains(\'"tensor_kind":"neo_hf_lora"\')', self.script)
        self.assertIn("Install-Official8StepLora", self.script)

    def test_download_is_resumable_and_batch_calls_the_script(self):
        self.assertIn("--continue-at -", self.script)
        self.assertIn("--retry-all-errors", self.script)
        self.assertIn("ParallelDownloads = 16", self.script)
        self.assertIn("--parallel-max $ParallelDownloads", self.script)
        self.assertIn("$chunkBytes = [Int64](32MB)", self.script)
        self.assertIn("Merge-ChunkResume", self.script)
        self.assertIn('$resumePath = "$($chunk.Path).resume"', self.script)
        self.assertIn("$repair.SetLength($originalLength)", self.script)
        self.assertIn("Move-Item -Force -LiteralPath $ResumePath", self.script)
        self.assertIn('("chunk-{0:D3}.bin" -f $index)', self.script)
        self.assertIn("$input.CopyTo($output)", self.script)
        self.assertIn("download_sensenova_u15_int8.ps1", self.batch)

    def test_models_only_batch_selects_existing_download_mode(self):
        batch = (ROOT / "download_sensenova_u15_models.bat").read_text(encoding="utf-8")
        self.assertIn('download_sensenova_u15_int8.ps1" -ModelOnly', batch)
        self.assertIn('exit /b %EXITCODE%', batch)


if __name__ == "__main__":
    unittest.main()
