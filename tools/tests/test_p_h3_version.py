"""The development version is consistent without claiming a released tag."""

from pathlib import Path

from modules.aikimi_version import VERSION

ROOT = Path(__file__).resolve().parents[2]


def test_development_version_matches_public_documentation():
    assert VERSION == "6.0.0"
    japanese = (ROOT / "README.md").read_text(encoding="utf-8")
    english = (ROOT / "README.en.md").read_text(encoding="utf-8")
    changes = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "v6.0.0（開発中）" in japanese
    assert "v6.0.0 (in development)" in english
    assert "v6.0.0 — 未リリース" in changes
    assert "tree/v6.0.0" not in japanese + english
