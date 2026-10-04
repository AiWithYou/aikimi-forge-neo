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
        readme_version = re.search(r"^\*\*v(\d+\.\d+\.\d+)", readme, re.MULTILINE)
        latest_release = re.search(r"^## v(\d+\.\d+\.\d+)\b", changelog, re.MULTILINE)
        self.assertIsNotNone(readme_version)
        self.assertIsNotNone(latest_release)
        self.assertEqual(readme_version.group(1), VERSION)
        self.assertEqual(latest_release.group(1), VERSION)


if __name__ == "__main__":
    unittest.main()
