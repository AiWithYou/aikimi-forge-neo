"""Pinned Unsloth GGUF for the standard Qwen Image 2.1 denoiser."""

from __future__ import annotations

import json
from pathlib import Path

from .core import QwenImage21Error, atomic_json, inside
from .gguf import load_transformer, sha256

MODEL_ID = "unsloth/Qwen-Image-2.1-GGUF"
REVISION = "2c31ccd392b367a6637841a143813320a02dff55"
NAME = "qwen-image-2.1-Q4_K_M.gguf"
SIZE = 4_199_565_024
SHA256 = "631d532e7ca71e8d90a87c71d3699761a812039d22e3370e87498d87754660fe"
FOLDER = "regular-gguf"
INVENTORY = "regular-gguf-files.json"


def regular_manifest(root: Path, *, verify_hashes: bool = False) -> dict:
    """Check the completed download without importing GPU libraries."""
    root = Path(root)
    try:
        record = json.loads((root / INVENTORY).read_text(encoding="utf-8"))
        if not isinstance(record, dict) or any(
            record.get(key) != value
            for key, value in (
                ("model", MODEL_ID),
                ("revision", REVISION),
                ("file", NAME),
                ("size", SIZE),
                ("sha256", SHA256),
            )
        ):
            raise ValueError("固定モデルの導入記録が一致しません。")
        path = inside(root, root / FOLDER / NAME)
        if not path.is_file() or path.stat().st_size != SIZE:
            raise ValueError(f"不足またはサイズ不一致: {NAME}")
        if verify_hashes and sha256(path) != SHA256:
            raise ValueError(f"SHA-256不一致: {NAME}")
        return record
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise QwenImage21Error(
            f"通常版Q4_K_Mが未導入または不完全です。aikimi-qwen-image21-setup.bat を実行してください（{exc}）。"
        ) from exc


def regular_status(root: Path) -> str:
    try:
        regular_manifest(root)
    except QwenImage21Error as exc:
        return str(exc)
    return "通常版 Q4_K_M (Unsloth) · 導入済み。"


def download_regular(root: Path) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().model_info(MODEL_ID, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise RuntimeError("Unslothモデルの固定revisionを確認できません。")
    item = next((sibling for sibling in info.siblings if sibling.rfilename == NAME), None)
    if item is None or item.size != SIZE or getattr(item.lfs, "sha256", None) != SHA256:
        raise RuntimeError("Unslothモデルの固定ファイル情報が一致しません。")
    destination = Path(root) / FOLDER
    path = Path(hf_hub_download(MODEL_ID, filename=NAME, revision=REVISION, local_dir=destination))
    if not path.resolve().is_relative_to(destination.resolve()) or path.stat().st_size != SIZE:
        raise RuntimeError(f"通常版GGUFのファイルサイズが一致しません: {NAME}")
    if sha256(path) != SHA256:
        raise RuntimeError(f"通常版GGUFのSHA-256が一致しません: {NAME}")
    atomic_json(
        Path(root) / INVENTORY,
        {
            "model": MODEL_ID,
            "revision": REVISION,
            "file": NAME,
            "size": SIZE,
            "sha256": SHA256,
        },
    )
    regular_manifest(root)


def load_gguf_transformer(model_path: Path):
    return load_transformer(model_path, model_path.parent / FOLDER / NAME, unsloth=True)
