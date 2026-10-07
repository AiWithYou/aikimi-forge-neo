"""Only a complete, byte-matching Hub commit may become a download default."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from tools.package_clef import package_bundle
from tools.publish_clef import verify_remote
from tools.tests.test_clef_bundle import fixture_artifacts


def published_fixture(tmp_path):
    _, backbone, _ = fixture_artifacts(tmp_path, "flash-int8")
    directory = package_bundle(tmp_path, "flash-int8", backbone)
    manifest = json.loads((directory / "complete.json").read_text())
    paths = [x["path"] for x in manifest["files"]] + ["complete.json"]
    siblings = []
    for name in paths:
        data = (directory / name).read_bytes()
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data, usedforsecurity=False).hexdigest()
        siblings.append(SimpleNamespace(rfilename=name, size=len(data), lfs=None, blob_id=blob))
    return directory, SimpleNamespace(sha="a" * 40, siblings=siblings)


def test_only_complete_matching_commit_is_accepted(tmp_path):
    directory, info = published_fixture(tmp_path)
    assert verify_remote(directory, info) == "a" * 40
    info.siblings = [x for x in info.siblings if x.rfilename != "joint_head.safetensors"]
    with pytest.raises(RuntimeError, match="不足"):
        verify_remote(directory, info)


def test_wrong_remote_weight_digest_is_refused(tmp_path):
    directory, info = published_fixture(tmp_path)
    item = next(x for x in info.siblings if x.rfilename == "model.safetensors")
    item.lfs = SimpleNamespace(sha256="0" * 64)
    with pytest.raises(RuntimeError, match="SHA-256"):
        verify_remote(directory, info)
