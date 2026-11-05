"""P-H3 model selection keeps the selected absolute checkpoint identity."""

import hashlib
import json
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from modules_forge import local_assets, model_catalog, p_h3_image, p_h3_image_ui, studio_checkpoints


def make_model(path: Path):
    shapes = {
        "audio_patch_proj.weight": [5376, 32], "video_patch_proj.weight": [5376, 96],
        "adaln_t_table": [4097, 16], "blocks.0.adaln_proj.linear.weight": [96768, 16],
        "blocks.0.attn.q_norm.weight": [128], "blocks.0.attn.qkv_proj.weight": [21504, 5376],
        "blocks.0.mlp.fc1.weight": [28672, 5376],
    }
    header = {key: {"dtype": "BF16", "shape": shape, "data_offsets": [0, 0]} for key, shape in shapes.items()}
    encoded = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded)
    return path.resolve()


def test_gui_lists_nested_compatible_and_can_choose_external_same_named_file(tmp_path):
    managed = make_model(tmp_path / "models/MiniMax-H3/checkpoints/nested/portrait.safetensors")
    external = make_model(tmp_path / "elsewhere/portrait.safetensors")
    with patch.object(model_catalog, "ROOT", tmp_path), patch.object(local_assets, "LIBRARY", tmp_path / "library.json"):
        from modules_forge.studio_checkpoint_ui import inventory

        assert str(managed) in [value for _, value in inventory("MiniMax-H3", (), p_h3_image.compatible_checkpoint)]
        update, message = p_h3_image_ui._add_checkpoint(str(external))
        assert update["value"] == str(external)
        assert {str(managed), str(external)} <= {value for _, value in update["choices"]}
        assert "選択" in message
        assert local_assets.selection("p-h3-image")["checkpoint"] == str(external)


def test_custom_same_named_file_is_registered_at_its_absolute_parent_and_snapshotted(tmp_path):
    first = make_model(tmp_path / "models/diffusion_models/same.safetensors")
    selected = make_model(tmp_path / "other/nested/same.safetensors")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    for relative in ("text_encoders/" + p_h3_image.TEXT_ENCODER, "vae/" + p_h3_image.VAE):
        path = runtime / "models" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    ready = SimpleNamespace(connected=True, runtime_profile="fast", runtime_args=("--disable-comfy-compiler",))
    fake_client = Mock()
    bridge = SimpleNamespace(
        resolve_runtime_root=lambda path: path,
        normalize_loopback_url=lambda url: url,
        _RUNTIME_LIFECYCLE_LOCK=threading.RLock(), _RUNTIME_SETUP_ACTIVE=False,
        inspect_readiness=Mock(return_value=ready),
        ensure_checkpoint_registration=Mock(return_value=ready),
        _cli_option_values=Mock(return_value=("",)),
        require_checkpoint_registration=Mock(), ComfyH3Client=Mock(return_value=fake_client),
    )
    request = p_h3_image.PH3Request("photo", checkpoint=str(selected), use_upsampler=False)
    with patch.object(p_h3_image, "_bridge", return_value=bridge), patch.object(p_h3_image, "check_nodes"):
        p_h3_image.ensure_runtime(request, runtime, "http://127.0.0.1:8189", tmp_path)
    acceleration = bridge.ensure_checkpoint_registration.call_args.args[4]
    assert acceleration.model_path == str(selected)
    config = studio_checkpoints.write_comfy_config(runtime, acceleration.model_path)
    entry = json.loads(config.read_text(encoding="utf-8"))["neo_checkpoint"]
    assert entry["is_default"] is True
    resolved = (Path(entry["base_path"]) / entry["diffusion_models"] / selected.name).resolve()
    assert resolved == selected
    assert resolved != first
    assert p_h3_image.build_workflow(request, {"images": []}, 42)["model"]["inputs"]["unet_name"] == selected.name
    with patch.object(local_assets, "HASH_CACHE", tmp_path / "hashes.json"):
        snapshot = studio_checkpoints.snapshot_checkpoint(str(selected))
        selected.write_bytes(selected.read_bytes() + b"changed")
        with pytest.raises(ValueError, match="変更されました"):
            studio_checkpoints.verify_snapshot(snapshot)


def test_relative_custom_checkpoint_is_rejected():
    with pytest.raises(p_h3_image.PH3Error):
        p_h3_image.PH3Request("photo", checkpoint="relative.safetensors").validate()


def test_cancelled_hash_does_not_publish_an_incomplete_cache_entry(tmp_path):
    path = tmp_path / "large.safetensors"
    path.write_bytes(b"a" * (2 * 1024 * 1024))
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        if calls >= 3:
            raise p_h3_image.PH3Cancelled("cancelled during hash")

    cache = tmp_path / "hashes.json"
    with patch.object(local_assets, "HASH_CACHE", cache):
        with pytest.raises(p_h3_image.PH3Cancelled, match="cancelled during hash"):
            studio_checkpoints.snapshot_checkpoint(str(path), check_cancel=cancel)
    assert calls >= 3
    assert not cache.exists()


def test_cancellable_hash_matches_sha256_and_checks_cancel_on_cache_hits(tmp_path):
    path = tmp_path / "model.safetensors"
    data = b"x" * (2 * 1024 * 1024 + 23)
    path.write_bytes(data)
    cache = tmp_path / "hashes.json"
    versions = (len(data), 1, 123, 2)
    with patch.object(local_assets, "HASH_CACHE", cache), patch.object(local_assets, "file_version", return_value=versions):
        identity = local_assets.file_identity(path, check_cancel=lambda: None)
        assert identity["sha256"] == hashlib.sha256(data).hexdigest()

        def cancel_immediately():
            raise p_h3_image.PH3Cancelled("cancelled before lookup")

        with pytest.raises(p_h3_image.PH3Cancelled, match="cancelled before lookup"):
            local_assets.file_identity(path, check_cancel=cancel_immediately)
        calls = 0

        def cancel_after_lookup():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise p_h3_image.PH3Cancelled("cancelled on cache hit")

        with pytest.raises(p_h3_image.PH3Cancelled, match="cache hit"):
            local_assets.file_identity(path, check_cancel=cancel_after_lookup)
        with patch.object(local_assets.hashlib, "sha256", side_effect=AssertionError("cache was not reused")):
            assert local_assets.file_identity(path, check_cancel=lambda: None) == identity
        with patch.object(local_assets.hashlib, "file_digest", side_effect=AssertionError("cache was not reused")):
            assert local_assets.file_identity(path) == identity


def test_cancellable_hash_without_change_counter_never_reuses_cache(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"first")
    versions = (5, 1, None, 2)
    cache = tmp_path / "hashes.json"
    with patch.object(local_assets, "HASH_CACHE", cache), patch.object(local_assets, "file_version", return_value=versions):
        first = local_assets.file_identity(path, check_cancel=lambda: None)
        path.write_bytes(b"other")
        second = local_assets.file_identity(path, check_cancel=lambda: None)
    assert first["sha256"] != second["sha256"]
    assert not cache.exists()


def test_cancellable_hash_rejects_a_model_changed_during_read(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"a" * (2 * 1024 * 1024))
    calls = 0

    def change():
        nonlocal calls
        calls += 1
        if calls == 3:
            with path.open("ab") as file:
                file.write(b"changed")

    with patch.object(local_assets, "HASH_CACHE", tmp_path / "hashes.json"):
        with pytest.raises(ValueError, match="変更"):
            studio_checkpoints.snapshot_checkpoint(str(path), check_cancel=change)


def test_concurrent_cancellable_hashes_keep_both_cache_entries(tmp_path):
    files = [tmp_path / "first.safetensors", tmp_path / "second.safetensors"]
    for file in files:
        file.write_bytes(b"a" * (2 * 1024 * 1024))
    barrier = threading.Barrier(2)

    def calculate(file):
        calls = 0

        def wait_on_first_chunk():
            nonlocal calls
            calls += 1
            if calls == 2:
                barrier.wait(timeout=5)

        return local_assets.file_identity(file, check_cancel=wait_on_first_chunk)

    cache = tmp_path / "hashes.json"
    with (patch.object(local_assets, "HASH_CACHE", cache),
          patch.object(local_assets, "file_version", side_effect=lambda path: (path.stat().st_size, 1, 123, 2))):
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(calculate, files))
        records = json.loads(cache.read_text(encoding="utf-8"))
    assert {str(file.resolve()) for file in files} <= records.keys()
    assert all(result["sha256"] == hashlib.sha256(b"a" * (2 * 1024 * 1024)).hexdigest() for result in results)


def test_one_cancelled_hash_does_not_cancel_or_cache_another_partial_hash(tmp_path):
    files = [tmp_path / "cancelled.safetensors", tmp_path / "completed.safetensors"]
    for file in files:
        file.write_bytes(b"a" * (2 * 1024 * 1024))
    barrier = threading.Barrier(2)

    def calculate(file):
        calls = 0

        def check():
            nonlocal calls
            calls += 1
            if calls == 2:
                barrier.wait(timeout=5)
            if calls == 3 and file == files[0]:
                raise p_h3_image.PH3Cancelled("cancelled")

        try:
            return local_assets.file_identity(file, check_cancel=check)
        except p_h3_image.PH3Cancelled:
            return None

    cache = tmp_path / "hashes.json"
    with (patch.object(local_assets, "HASH_CACHE", cache),
          patch.object(local_assets, "file_version", side_effect=lambda path: (path.stat().st_size, 1, 123, 2))):
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(calculate, files))
        records = json.loads(cache.read_text(encoding="utf-8"))
    assert results[0] is None and results[1]["sha256"]
    assert str(files[0].resolve()) not in records
    assert str(files[1].resolve()) in records


def test_cancellation_while_waiting_for_cache_lock_does_not_publish_hash(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"test contents")
    cache = tmp_path / "hashes.json"
    ready_to_lock, continue_to_lock, cancelled = threading.Event(), threading.Event(), threading.Event()
    version_calls = 0

    def version(_path):
        nonlocal version_calls
        version_calls += 1
        if version_calls == 2:
            ready_to_lock.set()
            assert continue_to_lock.wait(timeout=5)
        return (len(b"test contents"), 1, 123, 2)

    def check():
        if cancelled.is_set():
            raise p_h3_image.PH3Cancelled("cancelled while waiting for cache")

    with patch.object(local_assets, "HASH_CACHE", cache), patch.object(local_assets, "file_version", side_effect=version):
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(local_assets.file_identity, path, check_cancel=check)
            assert ready_to_lock.wait(timeout=5)
            with local_assets._LOCK:
                continue_to_lock.set()
                cancelled.set()
            with pytest.raises(p_h3_image.PH3Cancelled):
                future.result(timeout=5)
    assert not cache.exists()
