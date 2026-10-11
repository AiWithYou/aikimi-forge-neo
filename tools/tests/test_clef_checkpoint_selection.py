"""Checkpoint selection must reach the loader and distinguish actual model content."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from modules_forge.clef.bundle import REQUIRED_FILES
from modules_forge.clef.cache import identity
from modules_forge.clef.core import TEMPLATES, ClefError, atomic_json, sha256, validate_request


def bundle(directory, profile="flash-int8"):
    directory.mkdir(parents=True)
    for name in REQUIRED_FILES | {"model.safetensors"}:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    atomic_json(
        directory / "config.json",
        {"model_type": "qwen3_5", "architectures": ["Qwen3_5Model"]},
    )
    atomic_json(
        directory / "model.safetensors.index.json",
        {
            "weight_map": {
                "lm_head.weight": "lm-head.safetensors",
                "language_model.embed_tokens.weight": "model.safetensors",
            }
        },
    )
    files = [
        {"path": path.relative_to(directory).as_posix(), "size": path.stat().st_size, "sha256": sha256(path)}
        for path in directory.rglob("*")
        if path.is_file()
    ]
    atomic_json(directory / "complete.json", {**identity(profile), "bundle_format": 1, "files": files})
    return directory


def request(checkpoint):
    return {
        "profile": "flash-int8",
        "checkpoint": str(checkpoint),
        "questions": TEMPLATES["画像の評価"],
        "state": "test",
    }


def test_request_preserves_selected_checkpoint():
    assert validate_request(request("chosen-bundle"))["checkpoint"] == "chosen-bundle"


def test_external_complete_bundle_is_resolved_without_copying(tmp_path):
    from modules_forge.clef.local_source import checkpoint_manifest

    external = bundle(tmp_path / "external" / "chosen")
    directory, manifest = checkpoint_manifest(tmp_path / "runtime", "flash-int8", str(external))
    assert directory == external.resolve() and manifest["repo"] == identity("flash-int8")["repo"]
    with pytest.raises(ClefError, match="版・精度"):
        checkpoint_manifest(tmp_path / "runtime", "clef-24gb", str(external))


def test_broken_manifest_and_other_architecture_are_refused(tmp_path):
    from modules_forge.clef.local_source import checkpoint_manifest, compatible_profiles

    external = bundle(tmp_path / "external")
    original = (external / "complete.json").read_text()
    (external / "complete.json").write_text("[]")
    assert compatible_profiles(external) == []
    with pytest.raises(ClefError):
        checkpoint_manifest(tmp_path, "flash-int8", str(external))
    (external / "complete.json").write_text(original)
    atomic_json(external / "config.json", {"model_type": "llama", "architectures": ["LlamaModel"]})
    manifest = json.loads(original)
    for item in manifest["files"]:
        if item["path"] == "config.json":
            item.update(size=(external / "config.json").stat().st_size, sha256=sha256(external / "config.json"))
    atomic_json(external / "complete.json", manifest)
    assert compatible_profiles(external) == []


def test_discovery_does_not_hash_weights_or_import_models(tmp_path, monkeypatch):
    from modules_forge import model_catalog
    from modules_forge.clef import local_source

    managed = bundle(tmp_path / "models" / "Clef" / "checkpoints" / "nested" / "chosen")
    runtime = tmp_path / "runtime"
    unselected = bundle(runtime / "another-bundle")
    remembered = bundle(tmp_path / "remembered")
    monkeypatch.setattr(model_catalog, "ROOT", tmp_path)
    monkeypatch.setattr("modules_forge.clef.bundle.sha256", Mock(side_effect=AssertionError("listing must not hash")))
    monkeypatch.setattr(
        "modules_forge.local_assets.file_identity", Mock(side_effect=AssertionError("listing must not hash"))
    )
    choices = local_source.checkpoint_choices(runtime, current=str(remembered))
    values = {value for _, value in choices}
    assert str(managed.resolve()) in values
    assert str(unselected.resolve()) not in values, "Runtime storage must not be scanned for additional bundles"
    assert str(remembered.resolve()) in values, "A saved selection must not fall back to another model"
    assert {"official:" + profile for profile in local_source.OFFICIAL} <= values


def test_selection_content_identity_changes_when_weight_is_replaced(tmp_path):
    from modules_forge.clef.local_source import checkpoint_identity

    external = bundle(tmp_path / "external")
    first = checkpoint_identity(tmp_path / "runtime", "flash-int8", str(external))
    weight = external / "model.safetensors"
    weight.write_text("changed", encoding="utf-8")
    second = checkpoint_identity(tmp_path / "runtime", "flash-int8", str(external))
    assert first["path"] == second["path"] == str(external.resolve())
    assert first["sha256"] != second["sha256"]


def test_service_sends_selected_bundle_and_identity_to_worker(tmp_path):
    from modules_forge.clef.service import Studio

    external = bundle(tmp_path / "external")
    studio = Studio(tmp_path / "runtime", tmp_path / "outputs")
    studio._installed = lambda *args: "python"
    studio._dependencies = lambda: None
    studio._launch = lambda result, *args: result
    result = studio.start(request(external), [], "owner")
    assert result["request"]["checkpoint"] == str(external.resolve())
    assert result["model"]["path"] == str(external.resolve())
    assert result["request"]["model_identity"]["sha256"] == result["model"]["sha256"]


def test_default_profile_also_snapshots_actual_model_before_launch(tmp_path):
    from modules_forge.clef.bundle import bundle_directory
    from modules_forge.clef.service import Studio

    root = tmp_path / "runtime"
    selected = bundle(bundle_directory(root, "flash-int8"))
    studio = Studio(root, tmp_path / "outputs")
    studio._installed = lambda *args: "python"
    studio._dependencies = lambda: None
    studio._launch = lambda result, *args: result
    value = request("")
    value.pop("checkpoint")
    result = studio.start(value, [], "owner")
    assert result["request"]["model_identity"]["path"] == str(selected.resolve())
    assert result["model"]["path"] == str(selected.resolve())


@pytest.mark.parametrize("expected", [None, {}, [], ""])
def test_worker_rejects_missing_expected_identity_before_loading(tmp_path, monkeypatch, expected):
    import torch

    from modules_forge.clef import runtime
    from tools import clef_worker

    external = bundle(tmp_path / "external")
    directory = tmp_path / "job"
    value = request(external)
    if expected is not None:
        value["model_identity"] = expected
    atomic_json(directory / "request.json", value)
    loader = Mock(return_value=Mock(saved=external))
    monkeypatch.setattr(runtime.Runner, "from_directory", loader)
    monkeypatch.setattr(clef_worker, "_RUNNER", None)
    monkeypatch.setattr(clef_worker, "_KEY", None)
    monkeypatch.setattr(clef_worker, "process_run", Mock(return_value={}))
    for name in ("reset_peak_memory_stats", "max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    with pytest.raises(RuntimeError, match="identity|識別"):
        clef_worker.resident_run({"root": str(tmp_path / "runtime"), "job_dir": str(directory)})
    loader.assert_not_called()


def test_worker_rejects_expected_identity_mismatch_before_loading(tmp_path, monkeypatch):
    from modules_forge.clef import runtime
    from modules_forge.clef.local_source import checkpoint_identity
    from tools import clef_worker

    external = bundle(tmp_path / "external")
    expected = checkpoint_identity(tmp_path / "runtime", "flash-int8", str(external))
    (external / "model.safetensors").write_text("changed", encoding="utf-8")
    directory = tmp_path / "job"
    atomic_json(directory / "request.json", {**request(external), "model_identity": expected})
    loader = Mock(side_effect=AssertionError("mismatched identity must be rejected before loading"))
    monkeypatch.setattr(runtime.Runner, "from_directory", loader)
    with pytest.raises(RuntimeError, match="変更"):
        clef_worker.resident_run({"root": str(tmp_path / "runtime"), "job_dir": str(directory)})
    loader.assert_not_called()


def test_worker_reloads_same_path_when_model_content_changes(tmp_path, monkeypatch):
    import torch

    from modules_forge.clef import runtime
    from modules_forge.clef.local_source import checkpoint_identity
    from tools import clef_worker

    external = bundle(tmp_path / "external")
    directory = tmp_path / "job"
    atomic_json(
        directory / "request.json",
        {**request(external), "model_identity": checkpoint_identity(tmp_path / "runtime", "flash-int8", str(external))},
    )
    monkeypatch.setattr(clef_worker, "_RUNNER", None)
    monkeypatch.setattr(clef_worker, "_KEY", None)
    monkeypatch.setattr(clef_worker, "process_run", lambda *args: {})
    for name in ("reset_peak_memory_stats", "max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    created = []

    def load(path, profile):
        created.append((Path(path), profile))
        return Mock(saved=external)

    monkeypatch.setattr(runtime.Runner, "from_directory", load)
    payload = {"root": str(tmp_path / "runtime"), "job_dir": str(directory)}
    clef_worker.resident_run(payload)
    clef_worker.resident_run(payload)
    assert len(created) == 1
    (external / "model.safetensors").write_text("changed", encoding="utf-8")
    atomic_json(
        directory / "request.json",
        {**request(external), "model_identity": checkpoint_identity(tmp_path / "runtime", "flash-int8", str(external))},
    )
    clef_worker.resident_run(payload)
    assert created == [(external.resolve(), "flash-int8")] * 2
    assert json.loads((directory / "result.json").read_text())["model"]["path"] == str(external.resolve())


@pytest.mark.parametrize("unreliable_counter", [False, True])
def test_worker_rejects_model_changed_during_load(tmp_path, monkeypatch, unreliable_counter):
    import torch

    from modules_forge import local_assets
    from modules_forge.clef import runtime
    from modules_forge.clef.local_source import checkpoint_identity
    from tools import clef_worker

    external = bundle(tmp_path / "external")
    if unreliable_counter:
        monkeypatch.setattr(local_assets, "file_version", lambda path: (Path(path).stat().st_size, 0, None, 0))
    directory = tmp_path / "job"
    value = request(external)
    value["model_identity"] = checkpoint_identity(tmp_path / "runtime", "flash-int8", str(external))
    atomic_json(directory / "request.json", value)
    monkeypatch.setattr(clef_worker, "_RUNNER", None)
    monkeypatch.setattr(clef_worker, "_KEY", None)
    process = Mock(return_value={})
    monkeypatch.setattr(clef_worker, "process_run", process)
    for name in ("reset_peak_memory_stats", "max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    runner = Mock(saved=external)

    def load(*args):
        (external / "model.safetensors").write_text("changed", encoding="utf-8")
        return runner

    monkeypatch.setattr(runtime.Runner, "from_directory", load)
    with pytest.raises(RuntimeError, match="変更"):
        clef_worker.resident_run({"root": str(tmp_path / "runtime"), "job_dir": str(directory)})
    runner.close.assert_called_once_with()
    process.assert_not_called()
    assert clef_worker._RUNNER is None and clef_worker._KEY is None
    assert not (directory / "loading.json").exists() and not (directory / "result.json").exists()


def test_file_identity_rehashes_without_change_counter(tmp_path, monkeypatch):
    from modules_forge import local_assets

    path = tmp_path / "weight.safetensors"
    path.write_bytes(b"first")
    monkeypatch.setattr(local_assets, "file_version", lambda path: (5, 0, None, 0))
    first = local_assets.file_identity(path)
    path.write_bytes(b"other")
    second = local_assets.file_identity(path)
    assert first["path"] == second["path"] and first["size"] == second["size"]
    assert first["sha256"] != second["sha256"]


def test_ui_restores_without_writing_and_passes_checkpoint_to_submit(tmp_path, monkeypatch):
    import gradio as gr

    from modules_forge.clef import local_source, workspace_ui

    external = bundle(tmp_path / "external")
    monkeypatch.setattr(
        "modules_forge.local_assets.selection",
        lambda _: {"checkpoint": str(external), "search_directory": str(tmp_path), "profile": "flash-int8"},
    )
    save = Mock()
    monkeypatch.setattr("modules_forge.local_assets.save_selection", save)
    monkeypatch.setattr(workspace_ui, "environment_status", lambda: "ready")
    assert local_source.load_selection() == (str(external), "flash-int8")
    with gr.Blocks() as app:
        workspace_ui.build(tmp_path / "outputs")
    components = {getattr(component, "label", ""): component for component in app.blocks.values()}
    selected = components["チェックポイント"]
    assert selected.value == str(external) and selected.do_not_save_to_config
    assert "探索フォルダー" not in components
    assert not any(
        isinstance(component, gr.Accordion) and component.label == "チェックポイント"
        for component in app.blocks.values()
    )
    assert "実行設定" in components
    submits = [fn for fn in app.fns.values() if fn.fn and fn.fn.__name__ == "submit"]
    assert submits and all(selected in fn.inputs for fn in submits)
    loader = next(fn for fn in app.fns.values() if fn.fn and fn.fn.__name__ == "_load_checkpoints")
    assert loader.outputs == [selected, components["実行設定"]] and loader.show_progress == "hidden"
    assert loader.fn()[0]["value"] == str(external)
    save.assert_not_called()
    saver = next(fn for fn in app.fns.values() if fn.fn is local_source.save_selection)
    assert saver.inputs == [selected, components["実行設定"]]
    saver.fn(str(external), "flash-int8")
    save.assert_called_once_with("clef", {"checkpoint": str(external), "profile": "flash-int8"})
