"""Iris implementation is covered by both change triggers and Ruff."""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class IrisCITests(unittest.TestCase):
    def test_lint_covers_iris_runtime_setup_tests_and_ui(self):
        workflow = (ROOT / ".github/workflows/lint.yml").read_text(encoding="utf-8")
        for target in (
            "modules_forge/iris",
            "extensions-builtin/iris-studio",
            "tools/*iris*.py",
            "tools/tests/test_iris*.py",
        ):
            with self.subTest(target=target):
                self.assertGreaterEqual(workflow.count(target), 3)

    def test_runtime_has_an_independent_strict_security_audit(self):
        workflow = (ROOT / ".github/workflows/security.yml").read_text(encoding="utf-8")
        self.assertTrue("  iris-pip-audit:" in workflow, "Iris needs an independent dependency audit")
        job = workflow.split("  iris-pip-audit:", 1)[1].split("  clef-pip-audit:", 1)[0]
        self.assertIn("inputs: tools/requirements-iris.txt", job)
        self.assertIn("vulnerability-service: OSV", job)
        self.assertIn("internal-be-careful-extra-flags: --strict", job)
        self.assertNotIn("ignore", job)


if __name__ == "__main__":
    unittest.main()
