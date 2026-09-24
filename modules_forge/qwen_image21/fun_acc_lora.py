"""Pinned, optional Alibaba PAI Fun Acc PDD adapter for Qwen Image 2.1."""

from __future__ import annotations

import json
import math
from hashlib import sha256
from pathlib import Path

REPOSITORY = "alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs"
REVISION = "f7545234760e1847cd8e89e52bd951cb0b7e327f"
WEIGHTS = "models/Qwen-Image-2.1-Fun-Acc-4Step.safetensors"
CONFIG = "models/pdd_config.json"
WEIGHTS_SIZE = 345_632_504
WEIGHTS_SHA256 = "764c56ae94f330b6d06ccc322f95e1b8ce46424ddde5899a15432f95f720d558"
CONFIG_SHA256 = "f798c4a8e9225350e9c46be7a60f397df35e6bb90965d64be76d11e51fd41b97"
SOURCE = f"https://huggingface.co/{REPOSITORY}/tree/{REVISION}"


def adapter_dir(runtime: Path) -> Path:
    return Path(runtime) / "fun-acc"


def weights_path(runtime: Path) -> Path:
    return adapter_dir(runtime) / WEIGHTS


def _config(runtime: Path) -> dict:
    config = json.loads((adapter_dir(runtime) / CONFIG).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Fun Accの設定が不正です。")
    sigmas = config.get("pdd_sigmas")
    if (
        config.get("pdd_num_steps") != 4
        or config.get("pdd_block_size") != 1
        or config.get("pdd_export_format") != "qwenimage21_extracted_prefused_v1"
        or config.get("pdd_inference_only") is not True
        or config.get("pdd_sampling_precision") != "native_time_fp32_state"
        or not isinstance(sigmas, list)
        or len(sigmas) != 5
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in sigmas)
        or sigmas[0] != 1
        or sigmas[-1] != 0
        or any(left <= right for left, right in zip(sigmas, sigmas[1:], strict=False))
    ):
        raise ValueError("Fun Accの4-step PDD設定が配布仕様と一致しません。")
    return config


def installed(runtime: Path) -> dict:
    root = adapter_dir(runtime)
    path = weights_path(runtime)
    config_path = root / CONFIG
    license_path = root / "LICENSE"
    receipt_path = root / "manifest.json"
    if (
        any(item.is_symlink() for item in (root, path, config_path, license_path, receipt_path))
        or not path.is_file()
        or path.stat().st_size != WEIGHTS_SIZE
        or not config_path.is_file()
        or not license_path.is_file()
        or not receipt_path.is_file()
    ):
        raise ValueError("Fun Acc 4-step LoRAが未導入か不完全です。導入コマンドを実行してください。")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or any(
        receipt.get(key) != expected for key, expected in (
            ("repository", REPOSITORY), ("revision", REVISION),
            ("weights_sha256", WEIGHTS_SHA256), ("config_sha256", CONFIG_SHA256),
        )
    ):
        raise ValueError("Fun Accの導入記録が配布版と一致しません。導入コマンドを再実行してください。")
    if sha256(config_path.read_bytes()).hexdigest() != CONFIG_SHA256:
        raise ValueError("Fun AccのPDD設定が固定revisionと一致しません。導入コマンドを再実行してください。")
    config = _config(runtime)
    return {
        "path": str(path.resolve()),
        "repository": REPOSITORY,
        "revision": REVISION,
        "sha256": WEIGHTS_SHA256,
        "steps": 4,
        "sigmas": config["pdd_sigmas"],
    }


def status(runtime: Path) -> str:
    try:
        installed(runtime)
    except (OSError, ValueError, TypeError):
        return "Fun Acc · 4 steps: 未導入。専用の導入コマンドを実行してください。"
    return "Fun Acc · 4 steps: 導入済み（通常版INT8用）。"
