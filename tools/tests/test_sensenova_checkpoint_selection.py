"""Local SenseNova choices preserve the final/pruned ConvRot architecture contract."""

import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from modules_forge import local_assets
from modules_forge import sensenova_u15_bridge as bridge


def checkpoint(path, *, value=b"a", extra=None):
    tensors = {
        "layer.comfy_quant": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
        "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [1, 5],
        },
        **(extra or {}),
    }
    encoded = json.dumps(tensors).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + value + b"0000")
    signature = hashlib.sha256(
        json.dumps(
            {name: {"dtype": item["dtype"], "shape": item["shape"]} for name, item in tensors.items()},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return signature


def test_custom_final_structure_is_accepted_without_official_digest(tmp_path):
    path = tmp_path / "custom.safetensors"
    signature = checkpoint(path)
    with (
        mock.patch.object(bridge, "CONVROT_STRUCTURE_SHA256", signature),
        mock.patch.object(bridge, "EXPECTED_CONVROT_LAYERS", 1),
    ):
        assert bridge.compatible_checkpoint(path)
        assert bridge.checkpoint_is_ready(path)
    assert not (tmp_path / "custom.safetensors.sha256").exists()


def test_other_architecture_and_unpruned_lm_head_are_refused(tmp_path):
    path = tmp_path / "custom.safetensors"
    signature = checkpoint(path)
    with (
        mock.patch.object(bridge, "CONVROT_STRUCTURE_SHA256", signature),
        mock.patch.object(bridge, "EXPECTED_CONVROT_LAYERS", 1),
    ):
        checkpoint(
            path, extra={"language_model.lm_head.weight": {"dtype": "F32", "shape": [0], "data_offsets": [5, 5]}}
        )
        assert not bridge.compatible_checkpoint(path)
        checkpoint(path)
        path.write_bytes(path.read_bytes()[:-1])
        assert not bridge.compatible_checkpoint(path)


def test_official_default_still_requires_pinned_sha_record(tmp_path):
    path = tmp_path / "official.safetensors"
    signature = checkpoint(path)
    with (
        mock.patch.object(bridge, "DEFAULT_CHECKPOINT_PATH", path),
        mock.patch.object(bridge, "CONVROT_STRUCTURE_SHA256", signature),
        mock.patch.object(bridge, "EXPECTED_CONVROT_LAYERS", 1),
        mock.patch.object(bridge, "CONVROT_EXPECTED_BYTES", path.stat().st_size),
    ):
        assert not bridge.checkpoint_is_ready(path)
        (tmp_path / "official.safetensors.sha256").write_text(bridge.CONVROT_SHA256)
        assert bridge.checkpoint_is_ready(path)


def test_catalog_scans_managed_subfolders_and_keeps_only_explicit_official_and_current(tmp_path, monkeypatch):
    from modules_forge import model_catalog

    storage = tmp_path / "models" / "SenseNova-U1"
    managed = storage / "checkpoints" / "nested" / "custom.safetensors"
    managed.parent.mkdir(parents=True)
    signature = checkpoint(managed)
    official = storage / "official.safetensors"
    unselected = storage / "another.safetensors"
    remembered = tmp_path / "remembered.safetensors"
    for path in (official, unselected, remembered):
        checkpoint(path)
    monkeypatch.setattr(model_catalog, "ROOT", tmp_path)
    monkeypatch.setattr(bridge, "DEFAULT_CHECKPOINT_PATH", official)
    monkeypatch.setattr(bridge, "CONVROT_STRUCTURE_SHA256", signature)
    monkeypatch.setattr(bridge, "EXPECTED_CONVROT_LAYERS", 1)
    monkeypatch.setattr(local_assets, "file_identity", mock.Mock(side_effect=AssertionError("listing must not hash")))

    values = {value for _, value in bridge.checkpoint_choices(current=str(remembered))}
    assert str(managed) in values and str(official) in values and str(remembered) in values
    assert str(unselected) not in values, "Official asset storage must not be scanned for additional checkpoints"


def test_custom_payload_uses_actual_file_identity_and_not_official_provenance(tmp_path):
    path = tmp_path / "custom.safetensors"
    checkpoint(path)
    request = bridge.SenseNovaRequest(
        mode=bridge.MODE_TEXT, prompt="test", generation_profile=bridge.PROFILE_QUALITY, checkpoint=str(path)
    )
    payload = bridge._request_payload(
        request, input_paths=[], output_path=tmp_path / "output.png", metadata_path=tmp_path / "output.json"
    )
    assert payload["checkpoint"] == str(path.resolve())
    assert payload["checkpoint_identity"] == local_assets.file_identity(path)
    assert payload["checkpoint_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert payload["checkpoint_revision"] == "local"


def test_custom_worker_rejects_selection_changed_since_request(tmp_path):
    from tools.tests.test_sensenova_u15_worker import worker

    path = tmp_path / "custom.safetensors"
    signature = checkpoint(path)
    payload = {
        "mode": "text",
        "prompt": "test",
        "generation_profile": bridge.PROFILE_QUALITY,
        "quantization": bridge.QUANT_INT8_CONVROT,
        "model_path": bridge.DEFAULT_MODEL_ID,
        "checkpoint_revision": "local",
        "checkpoint": str(path),
        "width": 1024,
        "height": 1024,
        "checkpoint_identity": local_assets.file_identity(path),
        "input_max_pixels": 512 * 512,
        "vram_mode": "low",
    }
    with (
        mock.patch.object(bridge, "CONVROT_STRUCTURE_SHA256", signature),
        mock.patch.object(bridge, "EXPECTED_CONVROT_LAYERS", 1),
    ):
        worker._validate_payload(payload)
        checkpoint(path, value=b"b")
        with pytest.raises(RuntimeError, match="changed|変更"):
            worker._validate_payload(payload)


@pytest.mark.parametrize("expected", [None, {}, [], ""])
def test_worker_rejects_missing_expected_identity_before_loading(tmp_path, monkeypatch, expected):
    from tools.tests.test_sensenova_u15_worker import worker

    path = tmp_path / "official.safetensors"
    checkpoint(path)
    payload = {
        "mode": "text",
        "prompt": "test",
        "generation_profile": worker.PROFILE_QUALITY,
        "quantization": "int8_convrot",
        "model_path": worker.FINAL_MODEL_ID,
        "checkpoint_revision": worker.CHECKPOINT_REVISION,
        "checkpoint": str(path),
        "width": 1024,
        "height": 1024,
        "vram_mode": "low",
    }
    if expected is not None:
        payload["checkpoint_identity"] = expected
    loader = mock.Mock(side_effect=AssertionError("missing identity must be rejected before loading"))
    monkeypatch.setattr(worker, "_load_runtime", loader)
    with pytest.raises(RuntimeError, match="identity|識別"):
        worker._validate_payload(payload)
    loader.assert_not_called()


def test_bridge_import_does_not_initialize_ui_framework_or_forge_gpu():
    source = (
        "import sys; from modules_forge import sensenova_u15_bridge; "
        "blocked = ('gradio', 'fastapi', 'modules.sd_models', 'modules.devices', 'backend.memory_management'); "
        "assert not any(name in sys.modules for name in blocked), [name for name in blocked if name in sys.modules]"
    )
    subprocess.run(  # noqa: S603 - fixed interpreter and literal probe, with no external input.
        [sys.executable, "-c", source], cwd=bridge._ROOT, capture_output=True, text=True, check=True
    )


@pytest.mark.parametrize("resident", [False, True])
@pytest.mark.parametrize("unreliable_counter", [False, True])
def test_worker_rejects_checkpoint_changed_during_load(tmp_path, monkeypatch, resident, unreliable_counter):
    from tools.tests.test_sensenova_u15_worker import worker

    path = tmp_path / "custom.safetensors"
    checkpoint(path)
    if unreliable_counter:
        monkeypatch.setattr(local_assets, "file_version", lambda path: (Path(path).stat().st_size, 0, None, 0))
    payload = {
        "checkpoint": str(path),
        "checkpoint_identity": local_assets.file_identity(path),
        "checkpoint_sha256": local_assets.file_identity(path)["sha256"],
        "generation_profile": worker.PROFILE_QUALITY,
    }
    engine = mock.Mock()

    def load(value):
        checkpoint(path, value=b"b")
        # A loader must not replace the captured expectation with the new file identity.
        value["checkpoint_identity"] = local_assets.file_identity(path)
        value["checkpoint_sha256"] = value["checkpoint_identity"]["sha256"]
        return (mock.Mock(), None, engine, 1, 588, None, 0.1, {})

    monkeypatch.setattr(worker, "_RESIDENT_MODE", resident)
    monkeypatch.setattr(worker, "_RESIDENT_RUNTIME", None)
    monkeypatch.setattr(worker, "_RESIDENT_KEY", None)
    monkeypatch.setattr(worker, "_load_runtime", load)
    monkeypatch.setattr(worker.torch.cuda, "empty_cache", mock.Mock())
    emitted = mock.Mock()
    monkeypatch.setattr(worker, "emit", emitted)
    with pytest.raises(RuntimeError, match="changed|変更"):
        worker._runtime_for_request(payload)
    engine.close.assert_called_once_with()
    assert worker._RESIDENT_RUNTIME is None and worker._RESIDENT_KEY is None
    assert not any(call.args[0] == "loaded" for call in emitted.call_args_list)


def test_worker_rejects_lora_changed_during_load(tmp_path, monkeypatch):
    from tools.tests.test_sensenova_u15_worker import worker

    path = tmp_path / "custom.safetensors"
    checkpoint(path)
    lora = tmp_path / "official-lora.safetensors"
    lora.write_bytes(b"first")
    monkeypatch.setattr(worker, "OFFICIAL_LORA_SHA256", hashlib.sha256(lora.read_bytes()).hexdigest())
    payload = {"checkpoint": str(path), "lora_path": str(lora), "generation_profile": worker.PROFILE_OFFICIAL_8STEP}
    engine = mock.Mock()

    def load(value):
        lora.write_bytes(b"other")
        return (mock.Mock(), None, engine, 1, 588, {"targets": 294}, 0.1, {})

    monkeypatch.setattr(worker, "_RESIDENT_MODE", True)
    monkeypatch.setattr(worker, "_RESIDENT_RUNTIME", None)
    monkeypatch.setattr(worker, "_RESIDENT_KEY", None)
    monkeypatch.setattr(worker, "_load_runtime", load)
    monkeypatch.setattr(worker.torch.cuda, "empty_cache", mock.Mock())
    monkeypatch.setattr(worker, "emit", mock.Mock())
    with pytest.raises(RuntimeError, match="changed|変更"):
        worker._runtime_for_request(payload)
    engine.close.assert_called_once_with()
    assert worker._RESIDENT_RUNTIME is None and worker._RESIDENT_KEY is None


@pytest.mark.parametrize("unreliable_counter", [False, True])
def test_worker_rejects_lora_changed_after_validation_before_loading(tmp_path, monkeypatch, unreliable_counter):
    from tools.tests.test_sensenova_u15_worker import worker

    path = tmp_path / "custom.safetensors"
    signature = checkpoint(path)
    lora = tmp_path / "official-lora.safetensors"
    lora.write_bytes(b"first")
    digest = hashlib.sha256(lora.read_bytes()).hexdigest()
    monkeypatch.setattr(bridge, "CONVROT_STRUCTURE_SHA256", signature)
    monkeypatch.setattr(bridge, "EXPECTED_CONVROT_LAYERS", 1)
    monkeypatch.setattr(worker, "OFFICIAL_LORA_EXPECTED_BYTES", lora.stat().st_size)
    monkeypatch.setattr(worker, "OFFICIAL_LORA_SHA256", digest)
    if unreliable_counter:
        monkeypatch.setattr(local_assets, "file_version", lambda path: (Path(path).stat().st_size, 0, None, 0))
    payload = {
        "mode": "text",
        "prompt": "test",
        "input_images": [],
        "quantization": bridge.QUANT_INT8_CONVROT,
        "model_path": bridge.DEFAULT_MODEL_ID,
        "checkpoint_revision": "local",
        "checkpoint": str(path),
        "checkpoint_identity": local_assets.file_identity(path),
        "width": 512,
        "height": 512,
        "generation_profile": worker.PROFILE_OFFICIAL_8STEP,
        "cfg_norm": "none",
        "steps": 8,
        "cfg_scale": 1.0,
        "timestep_shift": 3.0,
        "lora_path": str(lora),
        "lora_revision": worker.OFFICIAL_LORA_REVISION,
        "lora_sha256": digest,
        "device": "cpu",
        "source_path": str(tmp_path),
    }
    worker._validate_payload(payload)
    # File identity is already cached; replacing equal-size bytes must still be rejected.
    local_assets.file_identity(lora)
    lora.write_bytes(b"other")
    loaded = (mock.Mock(), None, mock.Mock(release_variant="final"), 1, 588, {"targets": 294}, 0.1, {})
    loader = mock.Mock(return_value=loaded)
    monkeypatch.setattr(worker, "_RESIDENT_MODE", False)
    monkeypatch.setattr(worker, "_load_runtime", loader)
    monkeypatch.setattr(worker, "emit", mock.Mock())
    with pytest.raises(RuntimeError, match="LoRA.*SHA-256"):
        worker._runtime_for_request(payload)
    loader.assert_not_called()


@pytest.mark.parametrize("resident", [False, True])
def test_unchanged_worker_model_becomes_ready_and_resident_model_is_reused(tmp_path, monkeypatch, resident):
    from tools.tests.test_sensenova_u15_worker import worker

    path = tmp_path / "custom.safetensors"
    checkpoint(path)
    payload = {"checkpoint": str(path), "generation_profile": worker.PROFILE_QUALITY}
    engine = mock.Mock(release_variant="final")
    loaded = (mock.Mock(), None, engine, 1, 588, None, 0.1, {})
    load = mock.Mock(return_value=loaded)
    monkeypatch.setattr(worker, "_RESIDENT_MODE", resident)
    monkeypatch.setattr(worker, "_RESIDENT_RUNTIME", None)
    monkeypatch.setattr(worker, "_RESIDENT_KEY", None)
    monkeypatch.setattr(worker, "_load_runtime", load)
    monkeypatch.setattr(worker.torch.cuda, "empty_cache", mock.Mock())
    emitted = mock.Mock()
    monkeypatch.setattr(worker, "emit", emitted)
    assert worker._runtime_for_request(payload) == loaded
    assert worker._runtime_for_request(payload)[2] is engine
    assert load.call_count == (1 if resident else 2)
    assert sum(call.args[0] == "loaded" for call in emitted.call_args_list) == load.call_count
    engine.close.assert_not_called()


def test_resident_worker_reloads_changed_content_without_change_counter(tmp_path, monkeypatch):
    from tools.tests.test_sensenova_u15_worker import worker

    path = tmp_path / "custom.safetensors"
    checkpoint(path)
    monkeypatch.setattr(local_assets, "file_version", lambda path: (Path(path).stat().st_size, 0, None, 0))
    payload = {"checkpoint": str(path), "generation_profile": worker.PROFILE_QUALITY}
    load = mock.Mock(return_value=(mock.Mock(), None, mock.Mock(release_variant="final"), 1, 588, None, 0.1, {}))
    monkeypatch.setattr(worker, "_RESIDENT_MODE", True)
    monkeypatch.setattr(worker, "_RESIDENT_RUNTIME", None)
    monkeypatch.setattr(worker, "_RESIDENT_KEY", None)
    monkeypatch.setattr(worker, "_load_runtime", load)
    monkeypatch.setattr(worker.torch.cuda, "empty_cache", mock.Mock())
    monkeypatch.setattr(worker, "emit", mock.Mock())
    worker._runtime_for_request(payload)
    checkpoint(path, value=b"b")
    worker._runtime_for_request(payload)
    assert load.call_count == 2


def test_catalog_and_ui_restore_selection_without_hashing_or_writing(tmp_path, monkeypatch):
    from tools.tests.test_sensenova_u15_studio import load_studio_module

    path = tmp_path / "custom.safetensors"
    signature = checkpoint(path)
    monkeypatch.setattr(bridge, "CONVROT_STRUCTURE_SHA256", signature)
    monkeypatch.setattr(bridge, "EXPECTED_CONVROT_LAYERS", 1)
    monkeypatch.setattr(
        local_assets, "selection", lambda _: {"checkpoint": str(path), "search_directory": str(tmp_path)}
    )
    save = mock.Mock()
    monkeypatch.setattr(local_assets, "save_selection", save)
    monkeypatch.setattr(local_assets, "file_identity", mock.Mock(side_effect=AssertionError("listing must not hash")))
    choices = bridge.checkpoint_choices(current=str(path))
    assert str(path.resolve()) in {value for _, value in choices}
    studio = load_studio_module()
    interface = studio._build_ui()[0][0]
    components = {getattr(component, "label", ""): component for component in interface.blocks.values()}
    assert components["チェックポイント"].value == str(path)
    assert components["チェックポイント"].do_not_save_to_config
    assert "探索フォルダー" not in components
    import gradio as gr

    assert not any(
        isinstance(component, gr.Accordion) and component.label == "チェックポイント"
        for component in interface.blocks.values()
    )
    assert studio._load_checkpoints()["value"] == str(path)
    loader = next(fn for fn in interface.fns.values() if fn.fn is studio._load_checkpoints)
    selected = components["チェックポイント"]
    assert loader.outputs == [selected] and loader.show_progress == "hidden"
    save.assert_not_called()
    saver = next(fn for fn in interface.fns.values() if fn.fn is bridge.save_checkpoint_selection)
    assert saver.inputs == [selected]
    saver.fn(str(path))
    save.assert_called_once_with("sensenova", {"checkpoint": str(path)})
