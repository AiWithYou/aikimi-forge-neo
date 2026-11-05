"""The P-H3 release version and date match the public documentation."""

from pathlib import Path

from modules.aikimi_version import VERSION

ROOT = Path(__file__).resolve().parents[2]


def test_release_version_matches_public_documentation():
    assert VERSION == "6.0.0"
    japanese = (ROOT / "README.md").read_text(encoding="utf-8")
    english = (ROOT / "README.en.md").read_text(encoding="utf-8")
    changes = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    link = "**[v6.0.0](https://github.com/AiWithYou/aikimi-forge-neo/tree/v6.0.0)**"
    assert link in japanese
    assert link in english
    assert "## v6.0.0 — 2026-11-05" in changes
    assert "v6.0.0（開発中）" not in japanese
    assert "v6.0.0 (in development)" not in english
