"""The published README and changelog must agree with the runtime release version."""

import re
import unittest
from datetime import date
from pathlib import Path

from modules.aikimi_version import VERSION

ROOT = Path(__file__).resolve().parents[2]


class ReleaseVersionTests(unittest.TestCase):
    def release_is_in_development(self):
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        latest = re.search(r"^## v(\d+\.\d+\.\d+)\s+—\s+([^\r\n]+)", changelog, re.MULTILINE)
        self.assertIsNotNone(latest)
        self.assertEqual(latest.group(1), VERSION)
        status = latest.group(2).strip()
        if status == "未リリース":
            return True
        try:
            released = date.fromisoformat(status)
        except ValueError:
            self.fail(f"Invalid release date: {status}")
        self.assertEqual(released.isoformat(), status)
        return False

    def test_public_documents_match_the_runtime_version(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        if self.release_is_in_development():
            self.assertIn(f"**v{VERSION}（開発中）**", readme)
            self.assertNotIn(f"/tree/v{VERSION}", readme)
        else:
            self.assertIn(f"**[v{VERSION}](https://github.com/AiWithYou/aikimi-forge-neo/tree/v{VERSION})**", readme)
            self.assertNotIn(f"v{VERSION}（開発中）", readme)

    def test_english_readme_matches_version_and_language_links(self):
        english = (ROOT / "README.en.md").read_text(encoding="utf-8")
        japanese = (ROOT / "README.md").read_text(encoding="utf-8")
        if self.release_is_in_development():
            self.assertIn(f"**v{VERSION} (in development)**", english)
            self.assertNotIn(f"/tree/v{VERSION}", english)
        else:
            self.assertIn(f"**[v{VERSION}](https://github.com/AiWithYou/aikimi-forge-neo/tree/v{VERSION})**", english)
            self.assertNotIn(f"v{VERSION} (in development)", english)
        self.assertIn("[English](README.en.md)", japanese.split("\n\n", 2)[1])
        self.assertIn("[日本語](README.md)", english.split("\n\n", 2)[1])


if __name__ == "__main__":
    unittest.main()
