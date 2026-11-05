"""Float single-file Qwen sources enter the existing quantizers safely."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load_file, save_file

from modules_forge import local_assets
from modules_forge.qwen_image21 import core, local_source, quantized_cache
from tools.tests.test_local_model_sources import qwen_fixture, tensor_file


@pytest.fixture(autouse=True)
def isolate_hash_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(local_assets, "HASH_CACHE", tmp_path / "hashes.json")


class TinyTransformer:
    @staticmethod
    def from_config(_):
        shapes = {"img_in.weight": (2, 2), "transformer_blocks.0.img_mlp.gate_layer.weight": (2, 3),
                  "transformer_blocks.0.img_mlp.proj.weight": (2, 3)}
        return SimpleNamespace(state_dict=lambda: {name: torch.empty(shape, device="meta") for name, shape in shapes.items()})


def source_file(path, *, extra=None):
    tensors = {
        "model.diffusion_model.img_in.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "model.diffusion_model.transformer_blocks.0.img_mlp.gate_up.weight": torch.arange(12, dtype=torch.float32).reshape(4, 3),
    }
    tensors.update(extra or {})
    save_file(tensors, str(path))
    return tensors


@pytest.mark.parametrize("precision", ["bf16", "int8", "w4a8"])
def test_float_single_source_accepts_all_normal_precisions(tmp_path, precision):
    components = qwen_fixture(tmp_path / "runtime/model", transformer=False)
    path = tmp_path / "custom.safetensors"
    keys = {f"transformer_blocks.{i}.attn.to_q.weight": [1, 1] for i in range(32)}
    keys.update({"img_in.weight": [1, 1], "txt_in.in_layer.weight": [1, 1]})
    tensor_file(path, keys)
    result = local_source.resolve(tmp_path / "runtime", str(path), str(components), precision)
    assert result["format"] == "float"
    assert result["identity"]["sha256"] == local_assets.file_identity(path)["sha256"]


@pytest.mark.parametrize("precision", ["int8", "w4a8"])
def test_local_single_precision_reaches_the_service_runtime_manifest(tmp_path, precision):
    runtime = tmp_path / "runtime"
    components = qwen_fixture(runtime / "model", transformer=False)
    python = runtime / "worker-env" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python.parent.mkdir(parents=True)
    python.write_bytes(b"interpreter")
    (runtime / "runtime.json").write_text(json.dumps({"schema": 1, "diffusers_revision": core.DIFFUSERS_REVISION}))
    path = tmp_path / "custom.safetensors"
    keys = {f"transformer_blocks.{i}.attn.to_q.weight": [1, 1] for i in range(32)}
    keys.update({"img_in.weight": [1, 1], "txt_in.in_layer.weight": [1, 1]})
    tensor_file(path, keys)
    result = core.runtime_manifest(runtime, precision, str(path), str(components))
    assert result["local_source"]["transformer"] == str(path.resolve())
    assert result["local_source"]["format"] == "float"


def test_worker_rejects_source_replaced_after_submission(tmp_path):
    from tools.tests.test_qwen_image21_worker import worker

    runtime = tmp_path / "runtime"
    components = qwen_fixture(runtime / "model", transformer=False)
    path = tmp_path / "custom.safetensors"
    keys = {f"transformer_blocks.{i}.attn.to_q.weight": [1, 1] for i in range(32)}
    keys.update({"img_in.weight": [1, 1], "txt_in.in_layer.weight": [1, 1]})
    tensor_file(path, keys)
    source = local_source.resolve(runtime, str(path), str(components), "int8")
    job = tmp_path / "job"
    job.mkdir()
    request = {"precision": "int8", "memory_mode": "offload", "prompt": "test", "seed": 1,
               "width": 256, "height": 256, "local_model": str(path), "local_components": str(components),
               "local_source": source, "runtime_root": str(runtime)}
    (job / "request.json").write_text(json.dumps(request), encoding="utf-8")
    payload = {"model_path": str(components), "job_dir": str(job), "runtime_root": str(runtime)}
    worker._read_request(payload)
    tensor_file(path, keys, value=2)
    with pytest.raises(ValueError, match="changed after submission"):
        worker._read_request(payload)


def test_streamed_conversion_normalizes_keys_splits_gate_up_and_preserves_data(tmp_path):
    from modules_forge.qwen_image21.single_file import staged_transformer

    path = tmp_path / "input.safetensors"
    source = source_file(path)
    with staged_transformer(path, TinyTransformer, tmp_path / "runtime", local_assets.file_identity(path)) as folder:
        assert json.loads((folder / "config.json").read_text())["_class_name"] == "QwenImage21Transformer2DModel"
        state = load_file(str(folder / "diffusion_pytorch_model.safetensors"))
        torch.testing.assert_close(state["img_in.weight"], source["model.diffusion_model.img_in.weight"])
        gate, projection = source["model.diffusion_model.transformer_blocks.0.img_mlp.gate_up.weight"].chunk(2, 0)
        torch.testing.assert_close(state["transformer_blocks.0.img_mlp.gate_layer.weight"], gate)
        torch.testing.assert_close(state["transformer_blocks.0.img_mlp.proj.weight"], projection)
        staged = folder
    assert not staged.exists()


@pytest.mark.parametrize("extra", [
    {"diffusion_model.img_in.weight": torch.ones(2, 2)},
    {"diffusion_model.transformer_blocks.0.img_mlp.proj.weight": torch.ones(2, 3)},
    {"unexpected.weight": torch.ones(1)},
])
def test_normalized_duplicate_or_extra_keys_are_rejected_before_copy(tmp_path, extra):
    from modules_forge.qwen_image21.single_file import staged_transformer

    path = tmp_path / "input.safetensors"
    source_file(path, extra=extra)
    with pytest.raises(ValueError), staged_transformer(path, TinyTransformer, tmp_path / "runtime", local_assets.file_identity(path)):
        pass


def test_wrong_shape_or_missing_key_cannot_enter_quantization(tmp_path):
    from modules_forge.qwen_image21.single_file import staged_transformer

    path = tmp_path / "bad.safetensors"
    save_file({"img_in.weight": torch.ones(3, 2)}, str(path))
    with pytest.raises(ValueError), staged_transformer(path, TinyTransformer, tmp_path / "runtime", local_assets.file_identity(path)):
        pass


def test_duplicate_safetensors_header_keys_are_rejected(tmp_path):
    from modules_forge.qwen_image21.single_file import staged_transformer

    path = tmp_path / "duplicate.safetensors"
    entry = '{"dtype":"F32","shape":[1],"data_offsets":[0,4]}'
    header = ('{"weight":' + entry + ',"weight":' + entry + '}').encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header + bytes(4))
    with pytest.raises(ValueError, match="重複"), staged_transformer(
        path, TinyTransformer, tmp_path / "runtime", local_assets.file_identity(path)
    ):
        pass


def test_source_change_during_conversion_rejects_and_removes_temporary_data(tmp_path):
    from modules_forge.qwen_image21.single_file import staged_transformer

    path = tmp_path / "input.safetensors"
    source_file(path)
    identity = local_assets.file_identity(path)
    calls = 0

    def change():
        nonlocal calls
        calls += 1
        if calls == 3:
            raw = bytearray(path.read_bytes())
            raw[-1] ^= 1
            stamp = path.stat().st_mtime_ns
            path.write_bytes(raw)
            os.utime(path, ns=(stamp, stamp))

    with pytest.raises(ValueError, match="変更"), staged_transformer(path, TinyTransformer, tmp_path / "runtime", identity, check_cancel=change):
        pass
    assert not list((tmp_path / "runtime").rglob("diffusion_pytorch_model.safetensors"))


def test_cancel_and_low_disk_never_leave_a_completed_temporary_model(tmp_path):
    from modules_forge.qwen_image21.single_file import staged_transformer

    path = tmp_path / "input.safetensors"
    source_file(path)
    identity = local_assets.file_identity(path)
    with patch("modules_forge.qwen_image21.single_file.shutil.disk_usage", return_value=SimpleNamespace(free=0)):
        with pytest.raises(OSError, match="空き容量"), staged_transformer(path, TinyTransformer, tmp_path / "runtime", identity):
            pass
    with pytest.raises(InterruptedError), staged_transformer(path, TinyTransformer, tmp_path / "runtime", identity,
                                                             check_cancel=lambda: (_ for _ in ()).throw(InterruptedError())):
        pass
    assert not list((tmp_path / "runtime").rglob("diffusion_pytorch_model.safetensors"))


def test_cancel_during_stream_removes_partial_temporary_file(tmp_path):
    from modules_forge.qwen_image21.single_file import staged_transformer

    path = tmp_path / "input.safetensors"
    source_file(path)
    identity = local_assets.file_identity(path)
    runtime = tmp_path / "runtime"

    def cancel_after_write_starts():
        if list(runtime.rglob("diffusion_pytorch_model.safetensors")):
            raise InterruptedError("cancel during stream")

    with pytest.raises(InterruptedError, match="during stream"), staged_transformer(
        path, TinyTransformer, runtime, identity, check_cancel=cancel_after_write_starts
    ):
        pass
    assert not list(runtime.rglob("diffusion_pytorch_model.safetensors"))


def test_source_guard_rehashes_when_filesystem_has_no_change_counter(tmp_path):
    from modules_forge.qwen_image21.single_file import source_guard

    path = tmp_path / "input.safetensors"
    source_file(path)
    actual_version = local_assets.file_version

    def no_counter(selected):
        size, modified, _, inode = actual_version(selected)
        return size, modified, None, inode

    with patch.object(local_assets, "file_version", side_effect=no_counter):
        identity = local_assets.file_identity(path)
        check, verify = source_guard(path, identity)
        before = path.stat()
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 1
        path.write_bytes(raw)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        check()
        with pytest.raises(ValueError, match="変更"):
            verify()


def test_cache_identity_is_bound_to_original_content_and_normalization_recipe(tmp_path):
    path = tmp_path / "input.safetensors"
    source_file(path)
    versions = dict.fromkeys(("torch", "diffusers", "transformers", "bitsandbytes", "accelerate", "safetensors", "comfy-kitchen"), "test")
    first = quantized_cache.component_identity(tmp_path / "model", "transformer", "w4a8", source_path=path,
                                                source_transform="single-v1", versions=versions)
    assert first["recipe"]["quantization"]["group_size"] == 16
    assert first["recipe"]["quantization"]["convrot_groupsize"] == 256
    with patch.dict(quantized_cache.W4A8_SETTINGS, {"group_size": 32}):
        different_quantization = quantized_cache.component_identity(
            tmp_path / "model", "transformer", "w4a8", source_path=path,
            source_transform="single-v1", versions=versions,
        )
    assert quantized_cache.cache_path(tmp_path / "model", first) != quantized_cache.cache_path(tmp_path / "model", different_quantization)
    changed_recipe = quantized_cache.component_identity(tmp_path / "model", "transformer", "w4a8", source_path=path,
                                                         source_transform="single-v2", versions=versions)
    assert quantized_cache.cache_path(tmp_path / "model", first) != quantized_cache.cache_path(tmp_path / "model", changed_recipe)
    source_file(path, extra={"other": torch.ones(1)})
    changed_file = quantized_cache.component_identity(tmp_path / "model", "transformer", "w4a8", source_path=path,
                                                       source_transform="single-v1", versions=versions)
    assert first["local_source"]["sha256"] != changed_file["local_source"]["sha256"]


def test_cache_reuses_converted_result_without_repeating_stream(tmp_path):
    from modules_forge.qwen_image21.single_file import source_guard, staged_transformer

    path = tmp_path / "input.safetensors"
    source_file(path)
    identity = local_assets.file_identity(path)
    check, verify = source_guard(path, identity)
    record = {"component": "transformer", "precision": "int8", "source_sha256": identity["sha256"]}
    calls = []

    def create():
        calls.append(True)
        with staged_transformer(path, TinyTransformer, tmp_path / "runtime", identity, check_cancel=check) as folder:
            return sorted(load_file(str(folder / "diffusion_pytorch_model.safetensors")))

    def save(value, folder):
        (folder / "result.json").write_text(json.dumps(value), encoding="utf-8")

    def load(folder):
        return json.loads((folder / "result.json").read_text(encoding="utf-8"))

    first, made = quantized_cache.load_or_create(tmp_path / "model", record, load, create, save,
                                                 check_cancel=check, verify_source=verify)
    second, hit = quantized_cache.load_or_create(tmp_path / "model", record, load, create, save,
                                                 check_cancel=check, verify_source=verify)
    assert first == second
    assert (made["status"], hit["status"]) == ("created", "hit")
    assert len(calls) == 1


def test_streamed_checkpoint_quantizes_and_reloads_w4a8_on_cpu(tmp_path):
    from torch import nn

    from modules_forge.qwen_image21.single_file import staged_transformer
    from modules_forge.qwen_image21.w4a8 import load_model, load_saved_model, save_model

    class Mlp(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_layer = nn.Linear(256, 256, bias=False)
            self.proj = nn.Linear(256, 256, bias=False)

        def forward(self, value):
            return self.gate_layer(value) + self.proj(value)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.img_mlp = Mlp()

        def forward(self, value):
            return self.img_mlp(value)

    class TinyQuant(nn.Module):
        def __init__(self):
            super().__init__()
            self.img_in = nn.Linear(256, 256, bias=False)
            self.transformer_blocks = nn.ModuleList([Block()])

        @classmethod
        def from_config(cls, _):
            return cls()

        def forward(self, value):
            return self.transformer_blocks[0](self.img_in(value))

    original = TinyQuant().to(dtype=torch.bfloat16).state_dict()
    stem = "transformer_blocks.0.img_mlp."
    gate_up = torch.cat((original[stem + "gate_layer.weight"], original[stem + "proj.weight"]))
    path = tmp_path / "single.safetensors"
    save_file({"diffusion_model.img_in.weight": original["img_in.weight"],
               "diffusion_model." + stem + "gate_up.weight": gate_up}, str(path))
    with staged_transformer(path, TinyQuant, tmp_path / "runtime", local_assets.file_identity(path)) as folder:
        packed, stats = load_model(folder, "transformer", TinyQuant, device="cpu")
    assert stats["layers"] == 2
    saved = tmp_path / "saved"
    save_model(packed, saved, stats, max_shard_bytes=100000)
    with patch("modules_forge.qwen_image21.w4a8._pack_weight", side_effect=AssertionError("repacked")):
        restored, restored_stats = load_saved_model(saved, "transformer", TinyQuant)
    sample = torch.randn(2, 256, dtype=torch.bfloat16)
    torch.testing.assert_close(packed(sample), restored(sample), rtol=0, atol=0)
    assert restored_stats == stats


def test_source_change_during_cache_write_never_publishes_result(tmp_path):
    from modules_forge.qwen_image21.single_file import source_guard

    path = tmp_path / "input.safetensors"
    source_file(path)
    identity = local_assets.file_identity(path)
    check, verify = source_guard(path, identity)
    record = {"component": "transformer", "precision": "int8", "source_sha256": identity["sha256"]}

    def save(_, folder):
        (folder / "weights").write_text("quantized", encoding="utf-8")
        source_file(path, extra={"changed": torch.ones(1)})

    with pytest.raises(ValueError, match="変更"):
        quantized_cache.load_or_create(tmp_path / "model", record, lambda _: None, lambda: None, save,
                                       check_cancel=check, verify_source=verify)
    assert not quantized_cache.cache_path(tmp_path / "model", record).exists()


def test_changed_source_does_not_quarantine_existing_valid_cache(tmp_path):
    from modules_forge.qwen_image21.single_file import source_guard

    path = tmp_path / "input.safetensors"
    source_file(path)
    identity = local_assets.file_identity(path)
    check, verify = source_guard(path, identity)
    record = {"component": "transformer", "precision": "int8", "source_sha256": identity["sha256"]}
    def save(_, folder):
        (folder / "weights").write_text("quantized", encoding="utf-8")
    _, created = quantized_cache.load_or_create(tmp_path / "model", record, lambda _: None, lambda: None, save,
                                                check_cancel=check, verify_source=verify)
    source_file(path, extra={"changed": torch.ones(1)})
    with pytest.raises(ValueError, match="変更"):
        quantized_cache.load_or_create(tmp_path / "model", record, lambda _: None, lambda: None, save,
                                       check_cancel=check, verify_source=verify)
    assert Path(created["path"]).is_dir()
    assert not list(Path(created["path"]).parent.glob("transformer.invalid-*"))
