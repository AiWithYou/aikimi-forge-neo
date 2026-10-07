"""Storage location and fixed conversion provenance shared by release tools."""

from __future__ import annotations

import json
from pathlib import Path

from .core import MODELS, PROFILES


def cache_directory(root, profile):
    settings = PROFILES[profile]
    if settings["precision"] == "bf16":
        return None
    root = Path(root)
    try:
        storage = json.loads((root / "storage.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        base = root / "quantized"
    except (OSError, ValueError) as exc:
        raise ValueError("Clefのstorage.jsonを読み込めません。--cache-dirで保存先を設定し直してください。") from exc
    else:
        destination = storage.get("quantized") if isinstance(storage, dict) else None
        if not isinstance(destination, str) or not Path(destination).is_absolute():
            raise ValueError("Clefの量子化キャッシュ保存先は絶対パスで指定してください。")
        base = Path(destination)
    return base / f"{settings['model']}-{settings['precision']}"


def identity(profile):
    settings = PROFILES[profile]
    return {
        **MODELS[settings["model"]],
        "precision": settings["precision"],
        "format": 1,
        "transformers": "5.10.2",
        "bitsandbytes": "0.50.2",
        "vision": "bf16",
    }
