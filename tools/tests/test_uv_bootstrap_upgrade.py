"""Pinned uv updates keep old and unknown local wheels untouched."""

import hashlib
import io
import sys
import zipfile
from unittest.mock import patch

import pytest

from tools import aikimi_setup, setup_flux3_klein, setup_minimax_h3, setup_moss_image

NAME = "uv-0.13.4-py3-none-win_amd64.whl"


def wheel():
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("uv-0.13.4.data/scripts/uv.exe", b"verified-new-uv")
    return output.getvalue()


class Response(io.BytesIO):
    status = 200

    def __init__(self, data):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}

    def getcode(self):
        return self.status


def source(data):
    return {
        "artifact_id": "test-uv", "relative_path": "bootstrap/uv.whl",
        "url": f"https://files.pythonhosted.org/packages/pinned/{NAME}",
        "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
        "license_url": "https://github.com/astral-sh/uv/blob/0.13.4/LICENSE-MIT",
    }


def setup_case(kind, repo, monkeypatch, data):
    spec = source(data)
    if kind == "h3":
        runtime = setup_minimax_h3.RuntimeInstaller(repo)
        runtime.data = {**runtime.data, "uv": spec}
        return runtime.base, runtime.bootstrap
    if kind == "flux":
        root = repo / "models/Flux-3-Klein"
        manifest = {**setup_flux3_klein.core.manifest(), "uv": spec}
        monkeypatch.setattr(setup_flux3_klein.core, "manifest", lambda: manifest)
        monkeypatch.setattr(setup_flux3_klein, "runtime_ready", lambda _: False)
        monkeypatch.setattr(setup_flux3_klein, "_worker_running", lambda _: False)
        monkeypatch.setattr(setup_flux3_klein, "prepare_comfy_checkout", lambda *_: None)
        monkeypatch.setattr(setup_flux3_klein, "runtime_command", lambda *_: "")
        monkeypatch.setattr(setup_flux3_klein.shutil, "which", lambda _: "git")
        return root, lambda: setup_flux3_klein.prepare_runtime(root, report=lambda _: None, verify=False)
    root = repo / "models/MOSS-Image"
    base = setup_moss_image.core.runtime_paths(root)["base"]
    manifest = {**setup_moss_image.manifest(), "uv": spec}
    monkeypatch.setattr(setup_moss_image, "manifest", lambda: manifest)
    monkeypatch.setattr(setup_moss_image.core, "runtime_ready", lambda _: False)
    monkeypatch.setattr(setup_moss_image, "_worker_running", lambda: False)
    monkeypatch.setattr(setup_moss_image, "prepare_comfy_checkout", lambda *_: None)
    monkeypatch.setattr(setup_moss_image, "runtime_command", lambda *_, **__: "")
    monkeypatch.setattr(setup_moss_image.shutil, "which", lambda _: "git")
    return base, lambda: setup_moss_image.prepare_runtime(root)


@pytest.mark.parametrize("kind", ["flux", "moss", "h3"])
@pytest.mark.parametrize("legacy", [b"previous-authentic-uv-0.12.12", b"unknown-local-wheel"])
def test_each_setup_installs_new_verified_wheel_without_touching_old(tmp_path, monkeypatch, kind, legacy):
    assert sys.version_info[:2] == (3, 13)
    data = wheel()
    base, run = setup_case(kind, tmp_path / "repo", monkeypatch, data)
    old = base / "bootstrap/uv.whl"
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_bytes(legacy)
    with patch.object(aikimi_setup.Installer, "_request", autospec=True, side_effect=lambda *_: Response(data)):
        run()
    assert old.read_bytes() == legacy
    assert (base / "bootstrap" / NAME).read_bytes() == data
    assert (base / "bootstrap/uv.exe").read_bytes() == b"verified-new-uv"


@pytest.mark.parametrize("kind", ["flux", "moss", "h3"])
def test_each_setup_preserves_invalid_new_version_and_never_runs_it(tmp_path, monkeypatch, kind):
    data = wheel()
    base, run = setup_case(kind, tmp_path / "repo", monkeypatch, data)
    bad = base / "bootstrap" / NAME
    bad.parent.mkdir(parents=True, exist_ok=True)
    tampered = bytearray(data)
    tampered[len(tampered) // 2] ^= 0x01
    bad.write_bytes(tampered)
    with patch.object(aikimi_setup.Installer, "_request", autospec=True) as request:
        with pytest.raises(aikimi_setup.IntegrityError):
            run()
    request.assert_not_called()
    assert bad.read_bytes() == tampered
    assert not (base / "bootstrap/uv.exe").exists()


@pytest.mark.parametrize("kind", ["flux", "moss", "h3"])
def test_each_setup_rejects_tampered_download_before_extraction(tmp_path, monkeypatch, kind):
    data = wheel()
    base, run = setup_case(kind, tmp_path / "repo", monkeypatch, data)
    old = base / "bootstrap/uv.whl"
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_bytes(b"keep-old-wheel")
    tampered = bytearray(data)
    tampered[len(tampered) // 2] ^= 0x01
    with patch.object(aikimi_setup.Installer, "_request", autospec=True, side_effect=lambda *_: Response(tampered)):
        with pytest.raises(aikimi_setup.IntegrityError, match="SHA-256"):
            run()
    assert old.read_bytes() == b"keep-old-wheel"
    assert not (base / "bootstrap" / NAME).exists()
    assert not (base / "bootstrap/uv.exe").exists()


@pytest.mark.parametrize("kind", ["flux", "moss", "h3"])
def test_versioned_wheel_symlink_outside_managed_root_is_rejected(tmp_path, monkeypatch, kind):
    data = wheel()
    base, run = setup_case(kind, tmp_path / "repo", monkeypatch, data)
    outside = tmp_path / "outside.whl"
    outside.write_bytes(b"preserve")
    target = base / "bootstrap" / NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"Symlinks unavailable: {error}")
    with patch.object(aikimi_setup.Installer, "_request", autospec=True) as request:
        with pytest.raises((aikimi_setup.SetupError, setup_flux3_klein.Flux3SetupError, setup_moss_image.MossImageSetupError)):
            run()
    request.assert_not_called()
    assert outside.read_bytes() == b"preserve"
    assert not (base / "bootstrap/uv.exe").exists()


@pytest.mark.parametrize("kind", ["flux", "moss", "h3"])
def test_unused_legacy_wheel_symlink_is_not_followed_or_modified(tmp_path, monkeypatch, kind):
    data = wheel()
    base, run = setup_case(kind, tmp_path / "repo", monkeypatch, data)
    outside = tmp_path / "outside.whl"
    outside.write_bytes(b"preserve")
    old = base / "bootstrap/uv.whl"
    old.parent.mkdir(parents=True, exist_ok=True)
    try:
        old.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"Symlinks unavailable: {error}")
    with patch.object(aikimi_setup.Installer, "_request", autospec=True, side_effect=lambda *_: Response(data)):
        run()
    assert old.is_symlink()
    assert outside.read_bytes() == b"preserve"
    assert (base / "bootstrap" / NAME).read_bytes() == data


def test_versioned_uv_rejects_unsafe_manifest_destinations_and_names():
    from tools.uv_bootstrap import versioned_uv_spec

    good = source(wheel())
    assert versioned_uv_spec(aikimi_setup.ArtifactSpec(**good)).relative_path == f"bootstrap/{NAME}"
    for change in ({"relative_path": "../uv.whl"}, {"relative_path": "other/uv.whl"},
                   {"url": "https://files.pythonhosted.org/packages/evil.whl"},
                   {"url": f"https://files.pythonhosted.org.evil.example/{NAME}"}):
        with pytest.raises(aikimi_setup.SetupError):
            versioned_uv_spec(aikimi_setup.ArtifactSpec(**(good | change)))
