import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WINDOWS_USER_PATH = re.compile(r"(?i)\b[a-z]:\\users\\(?!<)[^\\\s`\"']+")


class RepositoryPrivacyTests(unittest.TestCase):
    def test_documentation_does_not_publish_local_windows_user_paths(self):
        documents = [
            *(ROOT / "docs").rglob("*.md"),
            *(path for path in ROOT.glob("*.md") if path.is_file()),
        ]
        findings = []
        for document in documents:
            for line_number, line in enumerate(document.read_text(encoding="utf-8").splitlines(), start=1):
                if WINDOWS_USER_PATH.search(line):
                    findings.append(f"{document.relative_to(ROOT)}:{line_number}")

        self.assertEqual(findings, [], f"local Windows user paths found in: {', '.join(findings)}")


if __name__ == "__main__":
    unittest.main()
