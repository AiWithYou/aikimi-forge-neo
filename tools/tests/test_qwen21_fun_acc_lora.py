"""Contracts for the optional Qwen Image 2.1 Fun Acc 4-step adapter."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from modules_forge.qwen_image21 import core, fun_acc_lora, turbo
from modules_forge.qwen_image21.capabilities import PRECISIONS, sampling_recommendations
from tools import qwen_image21_worker as worker


def test_request_keeps_selected_model_and_accepts_sparse_fallback_with_four_steps():
    for precision in sorted(PRECISIONS):
        request = core.Request("a bird", precision=precision, steps=4, fun_acc=True, sparse_mode="fixed").resolved()
        assert request.precision == precision
        assert request.fun_acc is True
        assert request.steps == 4
        assert request.sparse_mode == "fixed"
        assert any("通常Attention" in notice for notice in sampling_recommendations(request.to_dict()))
    for fields in (
        {"steps": 40},
        {"fun_acc": "true"},
    ):
        with pytest.raises(core.QwenImage21Error, match="Fun Acc"):
            core.Request("a bird", **({"precision": "int8", "steps": 4, "fun_acc": True} | fields)).resolved()


def test_installed_requires_pinned_receipt_and_pdd_grid(tmp_path, monkeypatch):
    root = fun_acc_lora.adapter_dir(tmp_path)
    weights = fun_acc_lora.weights_path(tmp_path)
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"ok")
    (root / fun_acc_lora.CONFIG).write_text(
        json.dumps(
            {
                "pdd_num_steps": 4,
                "pdd_block_size": 1,
                "pdd_export_format": "qwenimage21_extracted_prefused_v1",
                "pdd_inference_only": True,
                "pdd_sampling_precision": "native_time_fp32_state",
                "pdd_sigmas": [1, 0.9, 0.7, 0.4, 0],
            }
        ),
        encoding="utf-8",
    )
    (root / "LICENSE").write_text("license", encoding="utf-8")
    monkeypatch.setattr(fun_acc_lora, "WEIGHTS_SIZE", 2)
    monkeypatch.setattr(
        fun_acc_lora, "CONFIG_SHA256", fun_acc_lora.sha256((root / fun_acc_lora.CONFIG).read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="未導入"):
        fun_acc_lora.installed(tmp_path)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "repository": fun_acc_lora.REPOSITORY,
                "revision": fun_acc_lora.REVISION,
                "weights_sha256": fun_acc_lora.WEIGHTS_SHA256,
                "config_sha256": fun_acc_lora.CONFIG_SHA256,
            }
        ),
        encoding="utf-8",
    )
    assert fun_acc_lora.installed(tmp_path)["steps"] == 4
    invalid_grid = json.loads((root / fun_acc_lora.CONFIG).read_text(encoding="utf-8"))
    invalid_grid["pdd_sigmas"] = [1, 1, 0.7, 0.4, 0]
    (root / fun_acc_lora.CONFIG).write_text(json.dumps(invalid_grid), encoding="utf-8")
    with pytest.raises(ValueError, match="4-step PDD"):
        fun_acc_lora._config(tmp_path)
    with pytest.raises(ValueError, match="固定revision"):
        fun_acc_lora.installed(tmp_path)


def test_worker_accepts_turbo_and_sparse_fallback_but_keeps_pdd_steps_and_cache(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "model_index.json").write_text(json.dumps({"_class_name": "QwenImage21Pipeline"}), encoding="utf-8")
    job = tmp_path / "job"
    job.mkdir()
    adapter = tmp_path / "adapter.safetensors"
    adapter.write_bytes(b"adapter")
    config = fun_acc_lora.adapter_dir(tmp_path) / fun_acc_lora.CONFIG
    config.parent.mkdir(parents=True)
    config.write_text("{}", encoding="utf-8")
    turbo_records = {}
    for precision in ("turbo_bf16", "turbo_q4_k_m"):
        model_id, revision, names = turbo.PROFILES[precision]
        files = {}
        for name in (*[f"{turbo.FOLDERS[precision]}/{name}" for name in names], turbo.SCHEDULER):
            path = tmp_path / "turbo" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
            files[name] = {"size": path.stat().st_size, "sha256": turbo.sha256(path)}
        turbo_records[precision] = {"model": model_id, "revision": revision, "files": files}
    (tmp_path / "turbo-files.json").write_text(json.dumps(turbo_records), encoding="utf-8")
    payload = {"model_path": str(model), "job_dir": str(job)}
    request = {
        "prompt": "a bird",
        "width": 512,
        "height": 512,
        "steps": 4,
        "seed": 1,
        "precision": "int8",
        "memory_mode": "offload",
        "fun_acc": True,
    }
    with patch.object(fun_acc_lora, "installed", return_value={"path": str(adapter)}):
        (job / "request.json").write_text(json.dumps(request), encoding="utf-8")
        assert worker._read_request(payload)[2]["fun_acc"] is True
        regular = worker._cache_key(model, "int8", "offload")
        accelerated = worker._cache_key(model, "int8", "offload", fun_acc=True)
        assert regular != accelerated
        assert any(isinstance(part, tuple) and part and part[0] == str(adapter) for part in accelerated)
        for precision in ("int8", "turbo_bf16", "turbo_q4_k_m"):
            selected = request | {"precision": precision, "sparse_mode": "fixed"}
            (job / "request.json").write_text(json.dumps(selected), encoding="utf-8")
            restored = worker._read_request(payload)[2]
            assert restored["precision"] == precision
            assert restored["steps"] == 4
            assert restored["fun_acc"] is True
            assert restored["sparse_mode"] == "fixed"
            assert any("通常Attention" in notice for notice in sampling_recommendations(restored))
        for invalid in ({"steps": 40}, {"fun_acc": "true"}):
            (job / "request.json").write_text(json.dumps(request | invalid), encoding="utf-8")
            with pytest.raises(ValueError, match="Fun Acc"):
                worker._read_request(payload)
