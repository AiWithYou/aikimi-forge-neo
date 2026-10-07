"""A distributed model must run without the original BF16 checkpoint."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from modules_forge.clef.bundle import bundle_directory, bundle_manifest
from modules_forge.clef.cache import identity
from modules_forge.clef.core import MODELS, PROFILES, ClefError, atomic_json, sha256
from tools.package_clef import package_bundle


def fixture_artifacts(root, profile):
    model = PROFILES[profile]["model"]
    source, backbone = root / "source" / model, root / "conversion"
    source.mkdir(parents=True)
    backbone.mkdir()
    weight = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
    for name, text in {
        "joint_schema_model.py": "# Official encoding and head\n",
        "joint_head_config.json": "{}",
        "tokenizer.json": "{}",
        "tokenizer_config.json": "{}",
        "processor_config.json": "{}",
        "chat_template.jinja": "{{ text }}",
        "LICENSE": "Apache License, Version 2.0",
    }.items():
        (source / name).write_text(text, encoding="utf-8")
    save_file({"weight": weight}, source / "joint_head.safetensors")
    save_file({"lm_head.weight": weight}, source / "source.safetensors")
    atomic_json(source / "model.safetensors.index.json", {"weight_map": {"lm_head.weight": "source.safetensors"}})
    files = [{"path": p.name, "size": p.stat().st_size, "sha256": sha256(p)} for p in source.iterdir()]
    atomic_json(source / "release.json", {**MODELS[model], "files": files})
    save_file({"language_model.embed_tokens.weight": weight}, backbone / "model.safetensors")
    atomic_json(backbone / "config.json", {"model_type": "qwen3_5", "architectures": ["Qwen3_5Model"]})
    atomic_json(
        backbone / "model.safetensors.index.json",
        {
            "metadata": {"total_size": weight.numel() * 2},
            "weight_map": {"language_model.embed_tokens.weight": "model.safetensors"},
        },
    )
    files = [{"path": p.name, "size": p.stat().st_size, "sha256": sha256(p)} for p in backbone.iterdir()]
    atomic_json(backbone / "complete.json", {**identity(profile), "files": files})
    (backbone / "private-log.txt").write_text("Do not distribute")
    return source, backbone, weight


def test_bundle_contains_dense_vocab_and_official_assets_without_source(tmp_path):
    source, backbone, weight = fixture_artifacts(tmp_path, "clef-16gb")
    target = package_bundle(tmp_path, "clef-16gb", backbone)
    assert target == bundle_directory(tmp_path, "clef-24gb")
    assert target == bundle_directory(tmp_path, "clef-16gb")
    # A fresh installation has no source tree.
    source.rename(source.with_name("source-unavailable"))
    directory, manifest = bundle_manifest(tmp_path, "clef-24gb", verify_hashes=True)
    assert directory == target
    names = {x["path"] for x in manifest["files"]}
    assert {
        "lm-head.safetensors",
        "joint_schema_model.py",
        "joint_head.safetensors",
        "LICENSE",
        "inference.py",
    } <= names
    assert "private-log.txt" not in names and "complete.json" not in names
    assert "clef_runtime/runtime.py" in names
    assert {
        "vendor/accelerate/LICENSE",
        "vendor/accelerate/AIKIMI-PATCH.md",
        "vendor/accelerate/setup.py",
        "vendor/accelerate/src/accelerate/__init__.py",
        "vendor/accelerate/src/accelerate/utils/modeling.py",
    } <= names
    assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)
    assert "./vendor/accelerate" in (target / "requirements.txt").read_text(encoding="utf-8")
    from modules_forge.clef.runtime import tensor_from_source

    assert torch.equal(tensor_from_source(target, "lm_head.weight"), weight)
    assert torch.equal(tensor_from_source(target, "language_model.embed_tokens.weight"), weight)
    assert bundle_directory(tmp_path, "flash-bf16") is None
    from modules_forge.clef.service import Studio

    python = tmp_path / "worker-env" / ("Scripts/python.exe" if __import__("os").name == "nt" else "bin/python")
    python.parent.mkdir(parents=True)
    python.touch()
    studio = Studio(runtime=tmp_path, outputs=tmp_path / "outputs")
    assert studio._installed("clef-16gb") == str(python)


def test_refresh_code_keeps_weight_bytes_and_updates_verified_manifest(tmp_path):
    from tools.package_clef import refresh_bundle

    _, backbone, _ = fixture_artifacts(tmp_path, "flash-int8")
    target = package_bundle(tmp_path, "flash-int8", backbone)
    weight = target / "model.safetensors"
    weight_hash, modified = sha256(weight), weight.stat().st_mtime_ns
    script = target / "clef_runtime/runtime.py"
    script.write_text("# previous runtime\n", encoding="utf-8")
    manifest = json.loads((target / "complete.json").read_text())
    entry = next(x for x in manifest["files"] if x["path"] == "clef_runtime/runtime.py")
    entry.update(size=script.stat().st_size, sha256=sha256(script))
    atomic_json(target / "complete.json", manifest)

    assert refresh_bundle(tmp_path, "flash-int8") == target
    bundle_manifest(tmp_path, "flash-int8", verify_hashes=True)
    assert sha256(weight) == weight_hash and weight.stat().st_mtime_ns == modified
    assert script.read_bytes() == (Path(__file__).resolve().parents[2] / "modules_forge/clef/runtime.py").read_bytes()


def test_refresh_failure_restores_previous_complete_bundle(tmp_path, monkeypatch):
    import tools.package_clef as packaging

    _, backbone, _ = fixture_artifacts(tmp_path, "flash-int8")
    target = package_bundle(tmp_path, "flash-int8", backbone)
    originals = {
        name: (target / name).read_bytes() for name in ("clef_runtime/runtime.py", "README.md", "complete.json")
    }
    monkeypatch.setattr(packaging, "model_card", lambda profile: "new card")
    monkeypatch.setattr(packaging, "atomic_json", lambda *args: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        packaging.refresh_bundle(tmp_path, "flash-int8")
    for name, data in originals.items():
        assert (target / name).read_bytes() == data
    bundle_manifest(tmp_path, "flash-int8", verify_hashes=True)


def test_refresh_adds_missing_vendor_files_and_rolls_back_additions_on_failure(tmp_path, monkeypatch):
    import shutil

    import tools.package_clef as packaging

    _, backbone, _ = fixture_artifacts(tmp_path, "flash-int8")
    target = package_bundle(tmp_path, "flash-int8", backbone)
    vendor = target / "vendor/accelerate"
    if vendor.is_dir():
        shutil.rmtree(vendor)
    manifest = json.loads((target / "complete.json").read_text())
    manifest["files"] = [entry for entry in manifest["files"] if not entry["path"].startswith("vendor/")]
    atomic_json(target / "complete.json", manifest)
    before = (target / "complete.json").read_bytes()

    with monkeypatch.context() as failed:
        failed.setattr(packaging, "atomic_json", lambda *args: (_ for _ in ()).throw(OSError("disk full")))
        with pytest.raises(OSError, match="disk full"):
            packaging.refresh_bundle(tmp_path, "flash-int8")
    assert (target / "complete.json").read_bytes() == before
    assert not (vendor / "LICENSE").is_file()
    packaging.refresh_bundle(tmp_path, "flash-int8")
    _, refreshed = bundle_manifest(tmp_path, "flash-int8", verify_hashes=True)
    assert (vendor / "LICENSE").is_file()
    assert "vendor/accelerate/LICENSE" in {entry["path"] for entry in refreshed["files"]}


def test_incomplete_and_modified_bundles_are_refused(tmp_path):
    _, backbone, _ = fixture_artifacts(tmp_path, "flash-int8")
    target = package_bundle(tmp_path, "flash-int8", backbone)
    manifest = json.loads((target / "complete.json").read_text())
    manifest["files"] = [x for x in manifest["files"] if x["path"] != "joint_head.safetensors"]
    atomic_json(target / "complete.json", manifest)
    with pytest.raises(ClefError, match="構成"):
        bundle_manifest(tmp_path, "flash-int8")
    manifest["files"].append(
        {
            "path": "joint_head.safetensors",
            "size": (target / "joint_head.safetensors").stat().st_size,
            "sha256": sha256(target / "joint_head.safetensors"),
        }
    )
    atomic_json(target / "complete.json", manifest)
    script = target / "joint_schema_model.py"
    script.write_text(script.read_text().replace("Official", "Tampered"))
    with pytest.raises(ClefError, match="SHA-256"):
        bundle_manifest(tmp_path, "flash-int8")


def test_packaging_failure_never_publishes_model(tmp_path, monkeypatch):
    _, backbone, _ = fixture_artifacts(tmp_path, "flash-int8")
    import tools.package_clef as packaging

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(packaging, "save_lexical", fail)
    with pytest.raises(OSError, match="disk full"):
        package_bundle(tmp_path, "flash-int8", backbone)
    assert not bundle_directory(tmp_path, "flash-int8").exists()
    assert (backbone / "model.safetensors").is_file()


def test_bundle_backbone_and_lm_head_load_without_missing_weights(tmp_path):
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5Model

    config = Qwen3_5Config(
        text_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 128,
            "layer_types": ["full_attention"],
            "tie_word_embeddings": False,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 4,
            "out_hidden_size": 64,
            "num_position_embeddings": 16,
        },
    )
    original = Qwen3_5ForConditionalGeneration(config)
    original.model.save_pretrained(tmp_path, max_shard_size="2KB")
    save_file({"lm_head.weight": original.lm_head.weight.contiguous()}, tmp_path / "lm-head.safetensors")
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    index["weight_map"]["lm_head.weight"] = "lm-head.safetensors"
    atomic_json(tmp_path / "model.safetensors.index.json", index)
    from modules_forge.clef.runtime import stream_weights

    with stream_weights():
        base, info = Qwen3_5Model.from_pretrained(tmp_path, output_loading_info=True)
    assert not info["missing_keys"] and set(info["unexpected_keys"]) == {"lm_head.weight"}
    assert torch.equal(base.language_model.embed_tokens.weight, original.model.language_model.embed_tokens.weight)
    # The normal Transformers class can also resolve the separate dense head.
    with stream_weights():
        conditional, info = Qwen3_5ForConditionalGeneration.from_pretrained(tmp_path, output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"]
    assert torch.equal(conditional.lm_head.weight, original.lm_head.weight)
