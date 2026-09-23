"""Pinned Viggle Turbo assets shared by the installer, UI, and worker."""

from __future__ import annotations

import json
from pathlib import Path

from .core import QwenImage21Error, atomic_json, inside, precision_label
from .gguf import load_transformer, sha256

VIGGLE_ID = "Viggle/Qwen-Image-2.1-viggle-turbo"
VIGGLE_REVISION = "bafc91e4cc934f5fb1406b22496a0bed9b99c548"
GGUF_ID = "Abiray/Qwen-Image-2.1-viggle-4-steps-turbo-GGUF"
GGUF_REVISION = "0016110b769f6625b92098ab9e3f4906dba367f6"
GGUF_NAME = "qwen_image_2.1_turbo_Q4_K_M.gguf"
PROFILES = {
    "turbo_bf16": (
        VIGGLE_ID,
        VIGGLE_REVISION,
        ("transformer/config.json", "transformer/diffusion_pytorch_model.safetensors"),
    ),
    "turbo_q4_k_m": (GGUF_ID, GGUF_REVISION, (GGUF_NAME,)),
}
FOLDERS = {"turbo_bf16": "bf16", "turbo_q4_k_m": "gguf"}
SCHEDULER = "scheduler/scheduler_config.json"


def turbo_manifest(root: Path, precision: str, *, verify_hashes: bool = False) -> dict:
    """Check a completed optional install without importing CUDA libraries."""
    if precision not in PROFILES:
        raise QwenImage21Error("Turboモデルの種類が不正です。")
    path = Path(root) / "turbo-files.json"
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, dict) or precision not in records:
            raise ValueError("導入記録がありません")
        record = records[precision]
        model_id, revision, names = PROFILES[precision]
        if record.get("model") != model_id or record.get("revision") != revision:
            raise ValueError("固定revisionが一致しません。")
        files = record["files"]
        expected = {f"{FOLDERS[precision]}/{name}" for name in names} | {SCHEDULER}
        if set(files) != expected:
            raise ValueError("ファイル一覧が一致しません。")
        for name, entry in files.items():
            file = inside(root, Path(root) / "turbo" / name)
            if not file.is_file() or file.stat().st_size != entry["size"]:
                raise ValueError(f"不足またはサイズ不一致: {name}")
            if verify_hashes and sha256(file) != entry["sha256"]:
                raise ValueError(f"SHA-256不一致: {name}")
        return record
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        flag = "--turbo-bf16-only" if precision == "turbo_bf16" else "--turbo-q4-only"
        raise QwenImage21Error(
            f"Turboモデルが未導入または不完全です。aikimi-qwen-image21-setup.bat {flag} を実行してください（{exc}）。"
        ) from exc


def turbo_status(root: Path, precision: str) -> str:
    try:
        turbo_manifest(root, precision)
    except QwenImage21Error as exc:
        return str(exc)
    return f"{precision_label(precision)} · 導入済み。"


def download_turbo(root: Path, precision: str) -> None:
    """Download exact optional files and publish their inventory last."""
    from huggingface_hub import HfApi, hf_hub_download

    model_id, revision, names = PROFILES[precision]
    api = HfApi()
    specs = ((model_id, revision, names, FOLDERS[precision]), (VIGGLE_ID, VIGGLE_REVISION, (SCHEDULER,), ""))
    files = {}
    for repo, pinned, wanted, folder in specs:
        info = api.model_info(repo, revision=pinned, files_metadata=True)
        if info.sha != pinned:
            raise RuntimeError(f"{repo}の固定revisionを確認できません。")
        siblings = {item.rfilename: item for item in info.siblings}
        for name in wanted:
            item = siblings[name]
            destination = root / "turbo" / folder
            path = Path(hf_hub_download(repo, filename=name, revision=pinned, local_dir=destination))
            if not path.resolve().is_relative_to(destination.resolve()) or (
                item.size and path.stat().st_size != item.size
            ):
                raise RuntimeError(f"Turboモデルのファイルサイズが一致しません: {name}")
            actual = sha256(path)
            expected = getattr(item.lfs, "sha256", None) if item.lfs else None
            if expected and actual != expected:
                raise RuntimeError(f"TurboモデルのSHA-256が一致しません: {name}")
            files[f"{folder}/{name}".lstrip("/")] = {"size": path.stat().st_size, "sha256": actual}
    manifest = root / "turbo-files.json"
    try:
        inventory = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        inventory = {}
    inventory[precision] = {"model": model_id, "revision": revision, "files": files}
    atomic_json(manifest, inventory)
    turbo_manifest(root, precision)


def load_gguf_transformer(model_path: Path):
    return load_transformer(model_path, model_path.parent / "turbo" / "gguf" / GGUF_NAME)
