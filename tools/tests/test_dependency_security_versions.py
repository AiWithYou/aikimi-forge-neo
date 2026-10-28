"""Security floors for the audited application and isolated runtimes."""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class DependencySecurityVersionTests(unittest.TestCase):
    def test_audited_requirements_use_fixed_setuptools_and_gitpython(self):
        files = [ROOT / "requirements.txt"] + [
            ROOT / "tools" / f"requirements-{name}.txt" for name in ("clef", "iris", "sensenova", "qwen-image21")
        ]
        for path in files:
            with self.subTest(path=path):
                self.assertIn("setuptools==84.1.0", path.read_text(encoding="utf-8"))
        self.assertIn("GitPython==3.2.1", files[0].read_text(encoding="utf-8"))

    def test_isolated_locks_use_fixed_setuptools_and_filelock(self):
        for name in ("flux3-klein", "moss-image", "minimax-h3"):
            with self.subTest(name=name):
                contents = (ROOT / "tools" / f"requirements-{name}.lock").read_text(encoding="utf-8")
                self.assertIn("setuptools==84.1.0", contents)
                self.assertIn("filelock==4.2.4", contents)
