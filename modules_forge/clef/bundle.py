"""Validate complete, self-contained quantized releases without importing torch."""

from __future__ import annotations

import json
from pathlib import Path

from .cache import cache_directory, identity
from .core import PROFILES, ClefError, sha256, source_manifest

SUPPORT_FILES = (
    "joint_head.safetensors",
    "joint_head_config.json",
    "joint_schema_model.py",
    "tokenizer.json",
    "tokenizer_config.json",
    "processor_config.json",
    "chat_template.jinja",
    "LICENSE",
)
PORTABLE_MODULES = ("__init__.py", "core.py", "cache.py", "bundle.py", "runtime.py")
REQUIRED_FILES = {
    *SUPPORT_FILES,
    "config.json",
    "model.safetensors.index.json",
    "lm-head.safetensors",
    "README.md",
    "inference.py",
    "requirements.txt",
    *("clef_runtime/" + name for name in PORTABLE_MODULES),
}


def bundle_directory(root, profile):
    converted = cache_directory(root, profile)
    return converted.with_name(converted.name + "-bundle") if converted is not None else None


def read_bundle(directory, profile, *, verify_hashes=False):
    directory = Path(directory)
    try:
        manifest = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
        expected = {**identity(profile), "bundle_format": 1}
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ClefError("Clef量子化配布の版・精度が一致しません。セットアップを再実行してください。")
        files = manifest["files"]
        names = {item["path"] for item in files}
        if not REQUIRED_FILES <= names or len(names) != len(files):
            raise ClefError("Clef量子化配布の構成が不足しています。セットアップを再実行してください。")
        for item in files:
            path = directory / item["path"]
            if (
                not path.resolve().is_relative_to(directory.resolve())
                or not path.is_file()
                or path.stat().st_size != item["size"]
            ):
                raise ClefError(f"Clef量子化配布が不足・変更されています: {item['path']}")
            if (verify_hashes or path.suffix == ".py") and sha256(path) != item["sha256"]:
                raise ClefError(f"ClefのSHA-256が一致しません: {item['path']}")
        index = json.loads((directory / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
        if (
            index.get("lm_head.weight") != "lm-head.safetensors"
            or "language_model.embed_tokens.weight" not in index
            or not set(index.values()) <= names
        ):
            raise ClefError("Clef量子化配布の語彙・本体の構成が一致しません。")
        return directory, manifest
    except ClefError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ClefError("Clef量子化モデル未導入です。aikimi-clef-setup.batを実行してください。") from exc


def bundle_manifest(root, profile, *, verify_hashes=False):
    directory = bundle_directory(root, profile)
    if directory is None:
        return source_manifest(root, PROFILES[profile]["model"], verify_hashes=verify_hashes)
    return read_bundle(directory, profile, verify_hashes=verify_hashes)
