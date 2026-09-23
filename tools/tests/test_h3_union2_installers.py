"""Downloader contracts and real local-Git transactions with stub Python install.

These tests deliberately never download real weights, invoke uv, or run CUDA.
"""

import hashlib
import importlib.util
import io
import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from modules_forge import minimax_h3_union2_vae as feature
from tools.tests.test_h3_union2_vae import header_fixture


def load_tool(name):
    runtime = ModuleType("modules_forge.minimax_h3_runtime")
    runtime.managed_runtime_root = lambda root: root / "repositories/minimax-h3/ComfyUI"
    runtime.configured_model_root = lambda root: root / "models/MiniMax-H3"
    runtime.setup_lock = lambda root: nullcontext()
    spec = importlib.util.spec_from_file_location("test_" + name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {runtime.__name__: runtime}):
        spec.loader.exec_module(module)
    return module


prepare = load_tool("prepare_minimax_h3_union2")
upgrade = load_tool("upgrade_minimax_h3_union2_vae")


def test_discovery_freezes_revision_and_uses_header_not_just_name():
    calls = []

    def get(url):
        calls.append(url)
        return {
            "sha": "a" * 40,
            "siblings": [
                {"rfilename": "model_patches/h3_union2_int8.safetensors", "lfs": {"sha256": "b" * 64, "size": 12345}},
                {"rfilename": "model_patches/h3_union_v2_bad.safetensors", "lfs": {"sha256": "c" * 64, "size": 12345}},
                {"rfilename": "model_patches/h3_union_v1.safetensors", "lfs": {"sha256": "d" * 64, "size": 12345}},
                {"rfilename": "diffusion_models/h3_union_v2.safetensors", "lfs": {"sha256": "d" * 64, "size": 12345}},
            ],
        }

    urls = []

    def probe(url):
        urls.append(url)
        if "bad" in url:
            raise ValueError("not native")
        return header_fixture("int8")

    found, messages = prepare.candidates(get=get, probe=probe)
    assert len(found) == 2 and len(messages) == 2
    assert all("/resolve/" + "a" * 40 + "/" in x["url"] for x in found)
    assert all(x["sha256"] == "b" * 64 for x in found)
    assert not any("v1" in url or "diffusion_models" in url for url in urls)
    assert len(calls) == 2


def test_discovery_filters_precision_and_absent_lfs_without_fabrication():
    info = {
        "sha": "a" * 40,
        "siblings": [{"rfilename": "model_patches/union_v2.safetensors", "lfs": {"sha256": "b" * 64, "size": 100}}],
    }
    assert prepare.candidates(get=lambda _: info, probe=lambda _: header_fixture("bf16"))[0] == []
    info["siblings"][0].pop("lfs")
    probe = Mock(side_effect=AssertionError("missing LFS must not probe"))
    found, diagnostics = prepare.candidates(get=lambda _: info, probe=probe)
    assert not found and len(diagnostics) == 2
    probe.assert_not_called()


def test_discovery_network_failure_is_not_claimed_no_models():
    def fail(_):
        raise OSError("offline")

    found, diagnostics = prepare.candidates(get=fail)
    assert not found and all("offline" in x for x in diagnostics) and len(diagnostics) == 2


def candidate(data):
    return {"url": "https://huggingface.co/test", "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


def test_streamed_download_size_hash_and_header_validation(tmp_path):
    target = tmp_path / "weight.safetensors"
    data = b"actual streamed data"
    with patch.object(prepare, "inspect_union2", return_value={"blocks": 10}) as inspect:
        prepare.download_verified(candidate(data), target, opener=lambda _: io.BytesIO(data))
        inspect.assert_called_once_with(target)
    assert target.read_bytes() == data


@pytest.mark.parametrize("actual", [b"tiny", b"larger than declared payload", b"bad!"])
def test_bad_download_removed(tmp_path, actual):
    target = tmp_path / "weight.safetensors"
    with pytest.raises(ValueError):
        prepare.download_verified(candidate(b"good"), target, opener=lambda _: io.BytesIO(actual))
    assert not target.exists()


def test_failed_download_never_deletes_existing_user_file(tmp_path):
    target = tmp_path / "weight.safetensors"
    target.write_bytes(b"user file")
    with pytest.raises(FileExistsError):
        prepare.download_verified(candidate(b"good"), target, opener=lambda _: io.BytesIO(b"good"))
    assert target.read_bytes() == b"user file"


def test_structurally_bad_download_removed_even_if_digest_matches(tmp_path):
    target = tmp_path / "weight.safetensors"
    with patch.object(prepare, "inspect_union2", side_effect=ValueError("v1 renamed")):
        with pytest.raises(ValueError):
            prepare.download_verified(candidate(b"good"), target, opener=lambda _: io.BytesIO(b"good"))
    assert not target.exists()


def test_install_atomic_copy_provenance_and_idempotency(tmp_path):
    source = tmp_path / "source.safetensors"
    source.write_bytes(b"fake compatible model data")
    models = tmp_path / "models"
    # Tiny synthetic bytes stand in only for the validated tensor payload.
    with patch.object(
        prepare, "inspect_union2", return_value={"precision": "int8", "blocks": 10, "size": source.stat().st_size}
    ):
        target = prepare.install_verified(source, models)
        assert target.read_bytes() == source.read_bytes()
        receipt = json.loads(target.with_suffix(".provenance.json").read_text())
        assert receipt["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert receipt["installed_name"] == feature.UNION2_MODEL
        assert prepare.install_verified(source, models) == target
        target.write_bytes(b"user changed model")
        with pytest.raises(ValueError):
            prepare.install_verified(source, models)
        assert target.read_bytes() == b"user changed model"
    assert not list((models / "model_patches").glob(".union2-*"))


def test_existing_receipt_never_overwritten_and_uncommitted_model_removed(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"abc")
    models = tmp_path / "models"
    dest = models / "model_patches"
    dest.mkdir(parents=True)
    receipt = (dest / feature.UNION2_MODEL).with_suffix(".provenance.json")
    receipt.write_text("user receipt")
    with patch.object(prepare, "inspect_union2", return_value={"precision": "int8", "size": 3}):
        with pytest.raises(FileExistsError):
            prepare.install_verified(source, models)
    assert receipt.read_text() == "user receipt" and not (dest / feature.UNION2_MODEL).exists()


def test_unknown_shape_rejected_before_writing(tmp_path):
    source = tmp_path / "raw"
    source.write_bytes(b"abc")
    with patch.object(prepare, "inspect_union2", side_effect=ValueError("raw checkpoint")):
        with pytest.raises(ValueError):
            prepare.install_verified(source, tmp_path / "models")
    assert not (tmp_path / "models").exists()


PATCH = """diff --git a/comfy/ldm/minimax/model.py b/comfy/ldm/minimax/model.py
--- a/comfy/ldm/minimax/model.py
+++ b/comfy/ldm/minimax/model.py
@@ -1,3 +1,3 @@
 header
-old body
+patched body
 footer
"""


def test_reviewed_patch_parser_and_unknown_context():
    assert (
        upgrade.apply_reviewed_file_patch("header\nold body\nfooter\n", PATCH, upgrade.MODEL)
        == "header\npatched body\nfooter\n"
    )
    assert upgrade.apply_reviewed_file_patch("unchanged\n", PATCH, "other.py") == "unchanged\n"
    with pytest.raises(ValueError):
        upgrade.apply_reviewed_file_patch("user code\n", PATCH, upgrade.MODEL)


def test_crlf_patch_blob_matches_git_blob():
    assert upgrade.blob("first\r\nsecond\r\n") == upgrade.blob("first\nsecond\n")


@pytest.fixture
def git_fixture(tmp_path, monkeypatch):
    """Real Git checkout/rollback, tiny fixture revisions, mock venv/compile."""
    runtime = tmp_path / "repositories/minimax-h3/ComfyUI"
    runtime.mkdir(parents=True)

    def git(*args):
        return subprocess.run(["git", *args], cwd=runtime, check=True, capture_output=True, text=True).stdout.strip()

    git("init")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "user.name", "Offline test fixture")
    model = runtime / upgrade.MODEL
    model.parent.mkdir(parents=True)
    model.write_text("header\nold body\nfooter\n")
    (runtime / "main.py").write_text("# fixture only\n")
    req = runtime / "requirements.txt"
    req.write_text("comfy-aimdo==0.5.2\n")
    git("add", ".")
    git("commit", "-m", "Old fixture")
    old = git("rev-parse", "HEAD")
    model.write_text("def _forward_with_memory_graph():\n    pass\n")
    req.write_text("comfy-aimdo==0.5.5\ncomfy-kitchen==0.2.35\n")
    git("add", ".")
    git("commit", "-m", "New fixture")
    new = git("rev-parse", "HEAD")
    new_req = req.read_text()
    git("checkout", "--detach", old)
    git("remote", "add", "origin", upgrade.CORE_URL)
    model.write_text("header\npatched body\nfooter\n")
    req.write_text("comfy-aimdo==0.5.3\n")
    python = runtime.parent / ".venv/Scripts/python.exe"
    python.parent.mkdir(parents=True)
    python.touch()
    patchpath = tmp_path / "patches/minimax-h3/comfyui-0.34.0-compiler.patch"
    patchpath.parent.mkdir(parents=True)
    patchpath.write_text(PATCH)
    new_patchpath = tmp_path / "patches/minimax-h3/comfyui-h3-compiler-912fca4.patch"
    new_patchpath.write_text(PATCH)
    security = ModuleType("modules.aikimi_security.redaction")
    security.sanitized_subprocess_environment = lambda env: dict(env)
    with patch.dict(sys.modules, {security.__name__: security}):
        u = upgrade.Upgrade(tmp_path)
    monkeypatch.setattr(upgrade, "OLD_CORE", old)
    monkeypatch.setattr(upgrade, "VAE_FIXED_COMMIT", new)
    monkeypatch.setattr(upgrade, "PATCH_BLOB", upgrade.blob(PATCH))
    monkeypatch.setattr(upgrade, "REQ_BLOB", upgrade.blob(new_req))
    monkeypatch.setattr(upgrade, "NEW_PATCH_BLOB", upgrade.blob(PATCH))
    monkeypatch.setattr(upgrade, "check_stopped", lambda _: None)
    actual = u.run

    def run(args, **kwargs):
        if args[0] == "git" and args[1] == "fetch":
            return SimpleNamespace(stdout="", returncode=0)
        if str(args[0]) == str(u.new_python):
            return SimpleNamespace(stdout="", returncode=0)
        return actual(args, **kwargs)

    monkeypatch.setattr(u, "run", run)

    def environment(_):
        u.new_python.parent.mkdir(parents=True)
        u.new_python.touch()
        return "fixture-freeze==1\n"

    monkeypatch.setattr(u, "make_environment", environment)
    return SimpleNamespace(u=u, runtime=runtime, old=old, new=new, git=git, model=model, req=req, run=run)


def test_real_git_apply_and_rollback_preserve_original_environment_and_edits(git_fixture):
    f = git_fixture
    model = f.model.read_bytes()
    req = f.req.read_bytes()
    assert f.u.preflight()[0] == f.old
    f.u.apply()
    assert f.git("rev-parse", "HEAD") == f.new
    assert f.u.record.exists() and not f.u.pending.exists()
    assert (f.u.base / ".venv/Scripts/python.exe").is_file()
    f.u.rollback()
    assert f.git("rev-parse", "HEAD") == f.old
    assert f.model.read_bytes() == model and f.req.read_bytes() == req
    assert not f.u.record.exists()
    assert f.u.new_python.is_file()  # Kept, not silently deleted.


def test_environment_failure_never_switches_core(git_fixture, monkeypatch):
    f = git_fixture

    def fail(_):
        raise OSError("simulated download failure")

    monkeypatch.setattr(f.u, "make_environment", fail)
    with pytest.raises(OSError):
        f.u.apply()
    assert f.git("rev-parse", "HEAD") == f.old and not f.u.pending.exists() and not f.u.record.exists()


def test_post_checkout_failure_restores_original_bytes(git_fixture, monkeypatch):
    f = git_fixture
    model = f.model.read_bytes()
    req = f.req.read_bytes()

    def fail_compile(args, **kwargs):
        if str(args[0]) == str(f.u.new_python):
            raise ValueError("simulated compile failure")
        return f.run(args, **kwargs)

    monkeypatch.setattr(f.u, "run", fail_compile)
    with pytest.raises(ValueError, match="compile failure"):
        f.u.apply()
    assert f.git("rev-parse", "HEAD") == f.old and f.model.read_bytes() == model and f.req.read_bytes() == req
    assert not f.u.pending.exists() and not f.u.record.exists()


def test_unknown_edits_abort_preflight(git_fixture):
    f = git_fixture
    f.model.write_text("user editing")
    with pytest.raises(ValueError):
        f.u.preflight()
    assert f.model.read_text() == "user editing"


def test_post_update_user_edit_blocks_rollback(git_fixture):
    f = git_fixture
    f.u.apply()
    f.model.write_text("later user editing")
    with pytest.raises(ValueError):
        f.u.rollback()
    assert f.git("rev-parse", "HEAD") == f.new and f.model.read_text() == "later user editing" and f.u.record.exists()


def test_staged_edit_blocks_rollback(git_fixture):
    f = git_fixture
    f.u.apply()
    f.req.write_text("user staged edit")
    f.git("add", "requirements.txt")
    with pytest.raises(ValueError):
        f.u.rollback()
    assert f.req.read_text() == "user staged edit"


def test_concurrent_edit_during_environment_build_not_overwritten(git_fixture, monkeypatch):
    f = git_fixture

    def environment(_):
        f.model.write_text("concurrent change")
        return "freeze"

    monkeypatch.setattr(f.u, "make_environment", environment)
    with pytest.raises(ValueError):
        f.u.apply()
    assert f.git("rev-parse", "HEAD") == f.old and f.model.read_text() == "concurrent change"
    assert not f.u.record.exists() and not f.u.pending.exists()
