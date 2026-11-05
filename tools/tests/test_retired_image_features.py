"""Retired image studios must not be offered or alter pre-existing user files."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from modules.aikimi_capabilities import feature_check
from modules.aikimi_diagnostics import DiagnosticPaths
from modules_forge import model_catalog
from tools.aikimi_setup import PROFILES

ROOT = Path(__file__).resolve().parents[2]
RETIRED = ("sensenova", "nanosaur2", "ming_image")


class RetiredImageFeaturesTests(unittest.TestCase):
    def test_old_capabilities_and_profiles_are_not_exposed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = DiagnosticPaths(root=root, models_root=root / "models", output_root=root / "outputs")
            for feature in RETIRED:
                with self.subTest(feature=feature):
                    self.assertIsNone(feature_check(feature, paths))
                    self.assertNotIn(feature, PROFILES)
        self.assertIn("krea2", PROFILES)
        self.assertIn("anima38", PROFILES)

    def test_old_studios_are_not_registered_in_navigation(self):
        source = (ROOT / "extensions-builtin/aikimi-ui/javascript/aikimi_tabs.js").read_text(encoding="utf-8")
        for feature in RETIRED:
            with self.subTest(feature=feature):
                self.assertFalse(f'"{feature}"' in source, f"{feature} is still ordered in navigation")
                self.assertFalse(f"{feature}: Object.freeze(" in source, f"{feature} is still registered")
        self.assertIn('"minimax_h3"', source)
        self.assertIn('"qwen_image21"', source)

    def test_catalog_does_not_create_old_folders_or_change_user_models(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old_model = root / "models/SenseNova-U1/checkpoints/keep.safetensors"
            old_model.parent.mkdir(parents=True)
            old_model.write_bytes(b"existing model")
            with patch.object(model_catalog, "ROOT", root):
                model_catalog.prepare_directories()
            self.assertEqual(old_model.read_bytes(), b"existing model")
            self.assertFalse((root / "models/Nanosaur2").exists())
            self.assertFalse((root / "models/Ming-Image").exists())
            self.assertTrue((root / "models/MiniMax-H3/checkpoints").is_dir())

    @unittest.skipUnless(os.name == "nt" and shutil.which("pwsh"), "PowerShell 7 required")
    def test_launcher_rejects_retired_profiles_and_keeps_h3_dry_run(self):
        launcher = str(ROOT / "aikimi-setup.ps1")
        for model in ("sensenova", "nanosaur2", "ming-image"):
            with self.subTest(model=model):
                result = subprocess.run(  # noqa: S603 - fixed launcher and fixed profile names
                    [shutil.which("pwsh"), "-NoProfile", "-File", launcher, "-Model", model, "-DryRun", "-NoPause"],
                    capture_output=True, timeout=30, check=False,
                )
                self.assertNotEqual(result.returncode, 0)
        result = subprocess.run(  # noqa: S603 - fixed launcher and fixed profile name
            [shutil.which("pwsh"), "-NoProfile", "-File", launcher, "-Model", "h3", "-DryRun", "-NoPause"],
            capture_output=True, timeout=30, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
