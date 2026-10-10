"""The published README and changelog must agree with the runtime release version."""

import re
import unittest
from pathlib import Path

from modules.aikimi_version import VERSION

ROOT = Path(__file__).resolve().parents[2]


class ReleaseVersionTests(unittest.TestCase):
    def test_public_documents_match_the_runtime_version(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        readme_version = re.search(
            r"^\*\*\[v(\d+\.\d+\.\d+)\]\(https://github\.com/AiWithYou/aikimi-forge-neo/releases/tag/v(\d+\.\d+\.\d+)\)\*\*",
            readme,
            re.MULTILINE,
        )
        latest_release = re.search(r"^## v(\d+\.\d+\.\d+)\b", changelog, re.MULTILINE)
        self.assertIsNotNone(readme_version)
        self.assertIsNotNone(latest_release)
        self.assertEqual(readme_version.group(1), VERSION)
        self.assertEqual(readme_version.group(2), VERSION)
        self.assertEqual(latest_release.group(1), VERSION)

    def test_english_readme_matches_version_and_language_links(self):
        english = (ROOT / "README.en.md").read_text(encoding="utf-8")
        japanese = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn(
            f"**[v{VERSION}](https://github.com/AiWithYou/aikimi-forge-neo/releases/tag/v{VERSION})**", english
        )
        self.assertIn("[English](README.en.md)", japanese.split("\n\n", 2)[1])
        self.assertIn("[日本語](README.md)", english.split("\n\n", 2)[1])


if __name__ == "__main__":
    unittest.main()
