"""Check the isolated inference environment without importing model libraries."""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path

DIFFUSERS_REVISION = "6256aa7666cedd47443adc8f82da9a10e110b09c"
VERSIONS = {
    "torch": "2.13.0+cu130",
    "torchvision": "0.28.0+cu130",
    "transformers": "5.17.0",
    "bitsandbytes": "0.50.2",
    "comfy-kitchen": "0.2.31",
    "accelerate": "1.15.0+aikimi.1",
    "setuptools": "83.0.0",
    "pip": "26.2.1",
    "Pillow": "12.3.0",
    "safetensors": "0.8.0",
}


def _validate(distributions):
    installed = {
        item.metadata["Name"].lower().replace("_", "-"): item for item in distributions if item.metadata["Name"]
    }
    for name, expected in VERSIONS.items():
        distribution = installed.get(name.lower())
        if distribution is None or distribution.version != expected:
            actual = distribution.version if distribution else "未導入"
            raise RuntimeError(
                f"Qwen専用環境には{name}=={expected}が必要です（現在: {actual}）。セットアップを再実行してください。"
            )
    diffusers = installed.get("diffusers")
    direct = json.loads((diffusers.read_text("direct_url.json") if diffusers else None) or "{}")
    if direct.get("vcs_info", {}).get("commit_id") != DIFFUSERS_REVISION:
        raise RuntimeError("Qwen Image 2.1対応の固定Diffusersが必要です。セットアップを再実行してください。")


def validate_running_versions():
    _validate(importlib.metadata.distributions())


def environment_status(root: Path) -> tuple[bool, str]:
    environment = root / "worker-env"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        return False, "Qwen専用Pythonがありません。aikimi-qwen-image21-setup.batを実行してください。"
    paths = (
        [environment / "Lib" / "site-packages"]
        if os.name == "nt"
        else list((environment / "lib").glob("python*/site-packages"))
    )
    try:
        _validate(importlib.metadata.distributions(path=[str(path) for path in paths]))
    except (RuntimeError, ValueError, OSError) as exc:
        return False, str(exc)
    return True, "専用環境の固定バージョンを確認しました。"
