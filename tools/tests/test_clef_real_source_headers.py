"""Use installed official assets to verify local Full FT structural baselines."""

import pytest

from modules_forge.clef.core import RUNTIME
from modules_forge.clef.custom_source import custom_manifest


@pytest.mark.parametrize(
    ("directory", "profile"),
    [
        ("source/clef-flash", "flash-bf16"),
        ("source/clef", "clef-bf16"),
        ("quantized/clef-flash-int8-bundle", "flash-int8"),
        ("quantized/clef-nf4-bundle", "clef-24gb"),
    ],
)
def test_installed_official_layout_matches_the_custom_full_ft_baseline(directory, profile):
    root = RUNTIME / directory
    if not (root / "model.safetensors.index.json").is_file():
        pytest.skip("official model files are not installed")
    path, metadata = custom_manifest(root, profile)
    assert path == root.resolve()
    assert metadata["source"] == "custom" and metadata["model"] and metadata["files"]
