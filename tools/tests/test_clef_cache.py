"""Clef storage configuration must stay explicit and share normal weights."""

import json

import pytest

from modules_forge.clef.cache import cache_directory


def test_normal_profiles_share_cache_and_bf16_does_not_cache(tmp_path):
    assert cache_directory(tmp_path, "clef-24gb") == cache_directory(tmp_path, "clef-16gb")
    assert cache_directory(tmp_path, "flash-bf16") is None


def test_relative_or_broken_storage_configuration_is_refused(tmp_path):
    for value in ('{"quantized": "relative"}', "[]", "{invalid"):
        (tmp_path / "storage.json").write_text(value)
        with pytest.raises(ValueError):
            cache_directory(tmp_path, "flash-int8")


def test_configured_ssd_cache_keeps_source_and_shares_normal_precision(tmp_path):
    root, ssd = tmp_path / "runtime", tmp_path / "ssd"
    root.mkdir()
    (root / "storage.json").write_text(json.dumps({"quantized": str(ssd.resolve())}))
    assert cache_directory(root, "flash-int8") == ssd / "clef-flash-int8"
    assert cache_directory(root, "clef-24gb") == cache_directory(root, "clef-16gb") == ssd / "clef-nf4"
    assert cache_directory(root, "flash-bf16") is None
