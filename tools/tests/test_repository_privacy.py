import re
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WINDOWS_USER_PATH = re.compile(r"(?i)\b[a-z]:\\users\\(?!<)[^\\\s`\"']+")
PRIVATE_DIRECTORIES = ("docs/audits/", "docs/articles/", ".playwright-cli/")
PRIVATE_ASSET_DIRECTORIES = ("docs/assets/h3-orbit-editorial/", "docs/assets/h3-orbit-chibi-aikimi/")
PRIVATE_DOCUMENTS = (
    "docs/CODEX_HANDOFF.md",
    "docs/jev-sparse-validation.md",
    "docs/model-retention-validation.md",
    "docs/minimax-h3-union2-vae-benchmark.md",
    "docs/optimization-persistence-2026-09-22.md",
    "docs/quantization-audit-2026-09-21.md",
    "docs/yue2-windows-validation.md",
    "docs/qwen-image21-community-2026-09-26.md",
)
PRIVATE_WORKFLOW = re.compile(
    r"editor\.note\.com|相談の採用判断|"
    r"(?:Opus|Astra|Claude|Sonnet|Gemini)[^。\n]*(?:相談|助言|レビュー|整え)|"
    r"note[^。\n]*(?:下書き|ログイン)|"
    r"\]\([^\n)]*(?:docs/)?audits/"
)


def private_record(path):
    return (
        path.startswith(PRIVATE_DIRECTORIES)
        or path.startswith(PRIVATE_ASSET_DIRECTORIES)
        or path in PRIVATE_DOCUMENTS
        or (path.startswith("docs/note_") and path.endswith(".md"))
        or (
            path.startswith("docs/assets/")
            and (any("note" in part for part in path.split("/")) or path.endswith((".json", ".csv")))
        )
    )


class RepositoryPrivacyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.git_path = shutil.which("git")
        if cls.git_path is None:
            raise unittest.SkipTest("publication checks require Git")
        try:
            # Git is resolved locally and invoked with fixed arguments without a shell.
            result = subprocess.run(  # noqa: S603
                [cls.git_path, "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            raise unittest.SkipTest("publication checks require a Git checkout") from error
        cls.tracked = result.stdout.decode("utf-8").rstrip("\0").split("\0")
        cls.documents = [
            ROOT / path
            for path in cls.tracked
            if path.endswith(".md")
            and not private_record(path)
            and (path.startswith(("docs/", "extensions-builtin/")) or "/" not in path)
        ]

    def test_private_work_records_are_not_tracked(self):
        findings = [path for path in self.tracked if private_record(path)]
        self.assertEqual(findings, [], "private records must remain outside the published tree")

    def test_private_work_records_are_ignored(self):
        paths = [
            *PRIVATE_DOCUMENTS,
            "docs/audits/local-check.md",
            "docs/articles/local-draft.md",
            "docs/note_local_draft.md",
            ".playwright-cli/local-session.yml",
            "docs/assets/clef-note/draft.png",
            "docs/assets/h3-orbit-note/draft.png",
            "docs/assets/h3-orbit-editorial/thumbnail.png",
            "docs/assets/h3-orbit-chibi-aikimi/comparison.png",
            "docs/assets/example/measurements.json",
            "docs/assets/example/benchmark.csv",
        ]
        result = subprocess.run(  # noqa: S603 - Fixed Git arguments; document paths use stdin.
            [self.git_path, "check-ignore", "--no-index", "-z", "--stdin"],
            cwd=ROOT,
            input="\0".join(paths) + "\0",
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        self.assertEqual(set(result.stdout.rstrip("\0").split("\0")), set(paths))

    def test_public_documentation_has_no_private_workflow_notes(self):
        findings = []
        for document in self.documents:
            if document == ROOT / "docs/release-checklist.md":
                continue
            for line_number, line in enumerate(document.read_text(encoding="utf-8").splitlines(), start=1):
                if PRIVATE_WORKFLOW.search(line):
                    findings.append(f"{document.relative_to(ROOT)}:{line_number}")
        self.assertEqual(findings, [], f"private workflow notes found in: {', '.join(findings)}")

    def test_documentation_does_not_publish_local_windows_user_paths(self):
        findings = []
        for document in self.documents:
            for line_number, line in enumerate(document.read_text(encoding="utf-8").splitlines(), start=1):
                if WINDOWS_USER_PATH.search(line):
                    findings.append(f"{document.relative_to(ROOT)}:{line_number}")

        self.assertEqual(findings, [], f"local Windows user paths found in: {', '.join(findings)}")


if __name__ == "__main__":
    unittest.main()
