"""Default setup downloads a complete quantized release, never raw BF16 weights."""

import json
import shutil
from contextlib import chdir
from types import SimpleNamespace

import pytest

from modules_forge.clef.bundle import bundle_directory
from modules_forge.clef.core import ClefError, sha256
from tools.package_clef import package_bundle
from tools.setup_clef import fetch_quantized
from tools.tests.test_clef_bundle import fixture_artifacts


def test_environment_updates_security_pins_and_resolves_vendor_from_repo_root(tmp_path, monkeypatch):
    import os

    import tools.setup_clef as setup

    root = tmp_path / "runtime"
    python = root / "worker-env" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python.parent.mkdir(parents=True)
    python.touch()
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(stdout="accelerate==1.15.0+aikimi.1\n")

    monkeypatch.setattr(setup.subprocess, "run", run)
    monkeypatch.setattr(setup.shutil, "which", lambda _: None)
    with chdir(tmp_path):
        assert setup.install_environment(root) == python
    assert all(kwargs["cwd"] == setup.ROOT for _, kwargs in calls)
    assert {"pip==26.2.1", "setuptools==83.0.0"} <= set(calls[0][0])
    requirements = (setup.ROOT / "tools/requirements-clef.txt").read_text(encoding="utf-8")
    assert "./vendor/accelerate" in requirements and "accelerate==1.15.0\n" not in requirements
    assert "pip==26.2.1" in requirements and "setuptools==83.0.0" in requirements
    assert "transformers==5.10.2" in requirements


def remote_fixture(tmp_path, monkeypatch, corrupt=False):
    root = tmp_path / "packaging"
    _, backbone, _ = fixture_artifacts(root, "flash-int8")
    remote = package_bundle(root, "flash-int8", backbone)
    import huggingface_hub

    import tools.setup_clef as setup

    commit = "f" * 40
    monkeypatch.setitem(setup.RELEASES, "clef-flash", {"repo": "Aikimi/clef-flash-int8", "revision": commit})
    files = [p for p in remote.rglob("*") if p.is_file()]
    info = SimpleNamespace(
        sha=commit,
        siblings=[
            SimpleNamespace(
                rfilename=p.relative_to(remote).as_posix(), size=p.stat().st_size, lfs=SimpleNamespace(sha256=sha256(p))
            )
            for p in files
        ],
    )
    calls = []

    def snapshot(repo_id, *, revision, local_dir, allow_patterns, **kwargs):
        calls.append((repo_id, revision, list(allow_patterns)))
        local_dir.mkdir(parents=True, exist_ok=True)
        for name in allow_patterns:
            destination = local_dir / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(remote / name, destination)
        if corrupt:
            path = local_dir / "joint_head.safetensors"
            data = bytearray(path.read_bytes())
            data[-1] ^= 1
            path.write_bytes(data)
        return str(local_dir)

    def download(repo_id, filename, *, revision, local_dir, **kwargs):
        local_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(remote / filename, local_dir / filename)
        return str(local_dir / filename)

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(model_info=lambda *args, **kwargs: info))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return calls, commit


def test_fresh_setup_uses_pinned_quantized_repo_and_reuses_complete_release(tmp_path, monkeypatch):
    calls, commit = remote_fixture(tmp_path, monkeypatch)
    fresh = tmp_path / "fresh"
    fetch_quantized(fresh, "clef-flash")
    target = bundle_directory(fresh, "flash-int8")
    assert target.is_dir() and not (fresh / "source").exists()
    assert calls[0][:2] == ("Aikimi/clef-flash-int8", commit)
    assert "lm-head.safetensors" in calls[0][2]
    fetch_quantized(fresh, "clef-flash")
    assert len(calls) == 1
    assert json.loads((target / "download.json").read_text())["revision"] == commit


def test_corrupt_download_is_not_published(tmp_path, monkeypatch):
    remote_fixture(tmp_path, monkeypatch, corrupt=True)
    fresh = tmp_path / "fresh"
    with pytest.raises((ClefError, RuntimeError), match="SHA-256"):
        fetch_quantized(fresh, "clef-flash")
    assert not bundle_directory(fresh, "flash-int8").exists()


def test_fresh_download_hashes_each_weight_file_only_once(tmp_path, monkeypatch):
    remote_fixture(tmp_path, monkeypatch)
    import modules_forge.clef.bundle as bundle
    import tools.setup_clef as setup

    calls = []

    def counted(path):
        calls.append(path.name)
        return sha256(path)

    monkeypatch.setattr(bundle, "sha256", counted)
    monkeypatch.setattr(setup, "sha256", counted)
    fetch_quantized(tmp_path / "fresh", "clef-flash")
    assert calls.count("model.safetensors") == 1
    assert calls.count("lm-head.safetensors") == 1
