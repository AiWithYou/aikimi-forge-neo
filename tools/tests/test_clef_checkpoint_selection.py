"""Checkpoint selection must reach the loader and distinguish actual model content."""

import json
import struct
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from modules_forge.clef.bundle import REQUIRED_FILES
from modules_forge.clef.cache import identity
from modules_forge.clef.core import TEMPLATES, ClefError, atomic_json, sha256, validate_request


def write_tensors(path, values):
    header, data = {}, bytearray()
    sizes = {"BF16": 2, "I8": 1, "F32": 4, "U8": 1}
    for key, dtype in values.items():
        start = len(data)
        data.extend(bytes(sizes[dtype]))
        header[key] = {"dtype": dtype, "shape": [1], "data_offsets": [start, len(data)]}
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


def change_tensor(path):
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)


@pytest.fixture(autouse=True)
def miniature_schema(monkeypatch):
    from modules_forge.clef import custom_source

    schema = {
        "language_model.embed_tokens.weight": ["BF16", [1]],
        "language_model.layers.0.linear_attn.in_proj_a.weight": ["I8", [1]],
        "visual.patch_embed.proj.weight": ["BF16", [1]],
        "lm_head.weight": ["BF16", [1]],
    }
    monkeypatch.setitem(custom_source.MODEL_SCHEMAS, ("clef-flash", "int8"), custom_source.schema_digest(schema))
    monkeypatch.setitem(
        custom_source.HEAD_SCHEMAS, "clef-flash", custom_source.schema_digest({"test.weight": ["BF16", [1]]})
    )


def bundle(directory, profile="flash-int8"):
    directory.mkdir(parents=True)
    for name in REQUIRED_FILES | {"model.safetensors"}:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    text = {
        "model_type": "qwen3_5_text",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "intermediate_size": 12288,
        "vocab_size": 248320,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_conv_kernel_dim": 4,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 8,
        "tie_word_embeddings": False,
        "attn_output_gate": True,
        "full_attention_interval": 4,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 262144,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
    }
    vision = {
        "model_type": "qwen3_5_vision",
        "depth": 27,
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "num_heads": 16,
        "out_hidden_size": 4096,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "in_channels": 3,
        "num_position_embeddings": 2304,
        "hidden_act": "gelu_pytorch_tanh",
    }
    config = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5Model"],
        "text_config": text,
        "vision_config": vision,
        "tie_word_embeddings": False,
        "image_token_id": 248056,
        "video_token_id": 248057,
        "vision_start_token_id": 248053,
        "vision_end_token_id": 248054,
        "quantization_config": {
            "quant_method": "bitsandbytes",
            "load_in_8bit": True,
            "load_in_4bit": False,
            "llm_int8_skip_modules": ["visual", "lm_head"],
        },
    }
    atomic_json(directory / "config.json", config)
    atomic_json(
        directory / "joint_head_config.json",
        {"hidden_size": 4096, "width": 1024, "routing_layers": 2, "layers": 4, "heads": 16, "feedforward": 4096},
    )
    atomic_json(
        directory / "tokenizer_config.json", {"tokenizer_class": "Qwen2Tokenizer", "image_token": "<|image_pad|>"}
    )
    special = {
        "<|endoftext|>": 248044,
        "<|im_start|>": 248045,
        "<|im_end|>": 248046,
        "<|vision_start|>": 248053,
        "<|vision_end|>": 248054,
        "<|image_pad|>": 248056,
        "<|video_pad|>": 248057,
    }
    atomic_json(
        directory / "tokenizer.json",
        {
            "model": {"type": "BPE", "vocab": {"example": 0}},
            "added_tokens": [{"content": k, "id": v} for k, v in special.items()],
        },
    )
    image = {
        "image_processor_type": "Qwen2VLImageProcessor",
        "patch_size": 16,
        "temporal_patch_size": 2,
        "merge_size": 2,
        "image_mean": [0.5] * 3,
        "image_std": [0.5] * 3,
        "do_convert_rgb": True,
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "rescale_factor": 1 / 255,
        "size": {"shortest_edge": 65536, "longest_edge": 16777216},
    }
    atomic_json(directory / "processor_config.json", {"processor_class": "Qwen3VLProcessor", "image_processor": image})
    write_tensors(directory / "joint_head.safetensors", {"test.weight": "BF16"})
    main = {
        "language_model.embed_tokens.weight": "BF16",
        "visual.patch_embed.proj.weight": "BF16",
        "language_model.layers.0.linear_attn.in_proj_a.weight": "I8",
    }
    write_tensors(directory / "model.safetensors", main)
    write_tensors(directory / "lm-head.safetensors", {"lm_head.weight": "BF16"})
    atomic_json(
        directory / "model.safetensors.index.json",
        {"weight_map": {"lm_head.weight": "lm-head.safetensors", **dict.fromkeys(main, "model.safetensors")}},
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
    assert directory == external.resolve() and manifest["source"] == "custom" and manifest["repo"] is None
    with pytest.raises(ClefError):
        checkpoint_manifest(tmp_path / "runtime", "clef-24gb", str(external))


def test_accepted_prefixed_checkpoint_provides_cpu_embedding(tmp_path):
    from modules_forge.clef.local_source import checkpoint_manifest
    from modules_forge.clef.runtime import tensor_from_source

    external = bundle(tmp_path / "prefixed")
    main = {
        "model.language_model.embed_tokens.weight": "BF16",
        "model.visual.patch_embed.proj.weight": "BF16",
        "model.language_model.layers.0.linear_attn.in_proj_a.weight": "I8",
    }
    write_tensors(external / "model.safetensors", main)
    atomic_json(external / "model.safetensors.index.json", {
        "weight_map": {"lm_head.weight": "lm-head.safetensors", **dict.fromkeys(main, "model.safetensors")}
    })
    directory, _ = checkpoint_manifest(tmp_path / "runtime", "flash-int8", str(external))
    embedding = tensor_from_source(directory, "language_model.embed_tokens.weight")
    assert embedding.device.type == "cpu" and embedding.shape == (1,)


def test_independently_finetuned_bundle_does_not_need_official_repo_revision(tmp_path):
    from modules_forge.clef.local_source import checkpoint_manifest

    external = bundle(tmp_path / "independent")
    manifest = json.loads((external / "complete.json").read_text(encoding="utf-8"))
    manifest.update(repo="independent/clef-finetune", revision="b" * 40)
    atomic_json(external / "complete.json", manifest)
    directory, metadata = checkpoint_manifest(tmp_path / "runtime", "flash-int8", str(external))
    assert directory == external.resolve()
    assert metadata["source"] == "custom" and metadata["repo"] is None and metadata["revision"] is None


def test_custom_auto_map_and_python_are_never_executed(tmp_path):
    from modules_forge.clef import runtime
    from modules_forge.clef.local_source import checkpoint_manifest

    external = bundle(tmp_path / "independent")
    marker = tmp_path / "evil-ran"
    (external / "evil.py").write_text(f"open({str(marker)!r}, 'w').write('bad')", encoding="utf-8")
    for name in ("config.json", "processor_config.json", "tokenizer_config.json"):
        path = external / name
        data = json.loads(path.read_text(encoding="utf-8"))
        data["auto_map"] = {
            "AutoModel": "evil.Payload",
            "AutoProcessor": "evil.Payload",
            "AutoTokenizer": "evil.Payload",
        }
        atomic_json(path, data)
    (external / "joint_schema_model.py").write_text(f"open({str(marker)!r}, 'w').write('bad')", encoding="utf-8")
    directory, _ = checkpoint_manifest(tmp_path / "runtime", "flash-int8", str(external))
    assert directory == external.resolve()
    assert hasattr(runtime.load_trusted_official(external), "JointSchemaHead")
    assert not marker.exists()


def test_real_auto_processor_ignores_selected_auto_map(tmp_path, monkeypatch):
    from transformers import AutoProcessor, Qwen3VLProcessor
    from transformers.models.auto import processing_auto

    external = bundle(tmp_path / "independent")
    path = external / "processor_config.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["auto_map"] = {"AutoProcessor": "evil.Payload"}
    atomic_json(path, data)
    marker = tmp_path / "evil-ran"
    (external / "evil.py").write_text(f"open({str(marker)!r}, 'w').write('bad')", encoding="utf-8")
    processor = object()
    load_local = Mock(return_value=processor)
    dynamic = Mock(side_effect=AssertionError("custom processor code must not load"))
    monkeypatch.setattr(Qwen3VLProcessor, "from_pretrained", load_local)
    monkeypatch.setattr(processing_auto, "get_class_from_dynamic_module", dynamic)
    result = AutoProcessor.from_pretrained(external, local_files_only=True, trust_remote_code=False)
    assert result is processor
    assert load_local.call_args.kwargs["trust_remote_code"] is False
    dynamic.assert_not_called()
    assert not marker.exists()


@pytest.mark.parametrize(
    "change",
    ["vision", "head", "vocab", "quant", "rope", "shard_escape", "weight_type", "tied_embeddings", "quant_flags"],
)
def test_custom_model_refuses_incompatible_component_and_unsafe_shard(tmp_path, change):
    from modules_forge.clef.local_source import checkpoint_manifest

    external = bundle(tmp_path / "independent")
    if change in {"vision", "quant", "rope", "tied_embeddings", "quant_flags"}:
        path = external / "config.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if change == "vision":
            value["vision_config"]["patch_size"] = 14
        elif change == "quant":
            value["quantization_config"]["quant_method"] = "w4a8"
        elif change == "tied_embeddings":
            value["text_config"]["tie_word_embeddings"] = True
        elif change == "quant_flags":
            value["quantization_config"]["_load_in_4bit"] = True
        else:
            value["text_config"]["rope_parameters"] = {"rope_type": "unsupported"}
        atomic_json(path, value)
    elif change == "head":
        write_tensors(external / "joint_head.safetensors", {"wrong.weight": "BF16"})
    elif change == "vocab":
        path = external / "tokenizer.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["added_tokens"][-1]["id"] = 7
        atomic_json(path, value)
    elif change == "shard_escape":
        path = external / "model.safetensors.index.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["weight_map"]["lm_head.weight"] = "../outside.safetensors"
        atomic_json(path, value)
    else:
        write_tensors(external / "lm-head.safetensors", {"lm_head.weight": "I8"})
    with pytest.raises(ClefError):
        checkpoint_manifest(tmp_path / "runtime", "flash-int8", str(external))


def test_trusted_joint_code_never_executes_python_from_selected_bundle(tmp_path):
    from modules_forge.clef import runtime

    marker = tmp_path / "selected-code-ran"
    selected = tmp_path / "selected"
    selected.mkdir()
    (selected / "joint_schema_model.py").write_text(f"open({str(marker)!r}, 'w').write('bad')", encoding="utf-8")
    module = runtime.load_trusted_official(selected)
    assert hasattr(module, "JointSchemaHead")
    assert not marker.exists()


def test_model_loads_disable_remote_code_and_require_safetensors():
    from modules_forge.clef import runtime

    model, processor = runtime.safe_pretrained_options()
    assert model["trust_remote_code"] is False and model["use_safetensors"] is True
    assert processor["trust_remote_code"] is False


def test_custom_runner_uses_only_trusted_code_and_passes_safe_loader_flags(tmp_path, monkeypatch):
    import safetensors.torch
    import torch
    import transformers

    from modules_forge.clef import runtime

    external = bundle(tmp_path / "independent")
    marker = tmp_path / "evil-ran"
    (external / "joint_schema_model.py").write_text(f"open({str(marker)!r}, 'w').write('bad')", encoding="utf-8")
    for name in ("config.json", "processor_config.json", "tokenizer_config.json"):
        path = external / name
        data = json.loads(path.read_text(encoding="utf-8"))
        data["auto_map"] = {"AutoModel": "evil.Payload", "AutoProcessor": "evil.Payload"}
        atomic_json(path, data)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *_: SimpleNamespace(total_memory=24 * 2**30))
    backbone = Mock()
    backbone.config = SimpleNamespace(use_cache=True)
    load_model = Mock(return_value=(backbone, {"missing_keys": [], "unexpected_keys": ["lm_head.weight"]}))
    load_processor = Mock(return_value=object())
    monkeypatch.setattr(transformers.Qwen3_5Model, "from_pretrained", load_model)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", load_processor)
    monkeypatch.setattr(transformers, "BitsAndBytesConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(safetensors.torch, "load_file", lambda *args, **kwargs: {})
    monkeypatch.setattr(runtime, "tensor_from_source", lambda *_: torch.zeros((2, 2), dtype=torch.bfloat16))
    head = Mock()
    head.to.return_value.eval.return_value = head
    monkeypatch.setattr(runtime, "load_trusted_official", lambda: SimpleNamespace(JointSchemaHead=lambda **_: head))
    runner = runtime.Runner.from_directory(external, "flash-int8", custom=True)
    assert runner.source == external.resolve()
    assert load_model.call_args.kwargs["trust_remote_code"] is False
    assert load_model.call_args.kwargs["use_safetensors"] is True
    assert load_processor.call_args.kwargs["trust_remote_code"] is False
    assert not marker.exists()


def test_broken_manifest_and_other_architecture_are_refused(tmp_path):
    from modules_forge.clef.local_source import checkpoint_manifest, compatible_profiles

    external = bundle(tmp_path / "external")
    original = (external / "model.safetensors.index.json").read_text()
    (external / "model.safetensors.index.json").write_text("[]")
    assert compatible_profiles(external) == []
    with pytest.raises(ClefError):
        checkpoint_manifest(tmp_path, "flash-int8", str(external))
    (external / "model.safetensors.index.json").write_text(original)
    atomic_json(external / "config.json", {"model_type": "llama", "architectures": ["LlamaModel"]})
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
    change_tensor(weight)
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
    change_tensor(external / "model.safetensors")
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

    def load(path, profile, *, custom=False):
        assert custom is True
        created.append((Path(path), profile))
        return Mock(saved=external)

    monkeypatch.setattr(runtime.Runner, "from_directory", load)
    payload = {"root": str(tmp_path / "runtime"), "job_dir": str(directory)}
    clef_worker.resident_run(payload)
    clef_worker.resident_run(payload)
    assert len(created) == 1
    change_tensor(external / "model.safetensors")
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

    def load(*args, **kwargs):
        change_tensor(external / "model.safetensors")
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
