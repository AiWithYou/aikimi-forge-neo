"""Pinned optional Orbit LoRA; importing this module never downloads weights."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from modules_forge.local_assets import file_identity
from modules_forge.minimax_h3_runtime import (
    REPOSITORY_ROOT as REPO_ROOT,
)
from modules_forge.minimax_h3_runtime import (
    configured_model_root,
    managed_runtime_root,
    model_root,
    setup_lock,
)

MODEL_NAME = "minimax_h3_flf2v_lora_v1.safetensors"
REVISION = "5ddbc2dbbe95edbbdaf5017c3e934b1d01791697"
MODEL_BYTES = 155_111_424
MODEL_SHA256 = "14f13e3effaf3e729fdc0c97680344aa63f473be0c55963f963d718b3db2a4d4"
MODEL_URL = f"https://huggingface.co/pablodawson/MiniMax-H3-360-Orbit-LoRA/resolve/{REVISION}/{MODEL_NAME}"
PACK = "Aikimi-MiniMax-H3-Orbit"
NODE = "MiniMaxH3OrbitLora"
BUNDLE = REPO_ROOT / "extensions-builtin/minimax-h3-studio/comfyui_nodes" / PACK


def snapshot(runtime_root: Path) -> dict:
    from modules_forge.studio_checkpoints import snapshot_checkpoint

    path = model_root(runtime_root) / "loras" / MODEL_NAME
    if path.is_symlink() or not path.is_file():
        raise ValueError("360° Orbit LoRAが未導入です。「Orbit LoRAを準備」を実行してください。")
    result = snapshot_checkpoint(str(path))
    if result["identity"]["size"] != MODEL_BYTES or result["identity"]["sha256"] != MODEL_SHA256:
        raise ValueError("360° Orbit LoRAのサイズまたはSHA-256が固定版と異なります。")
    from modules_forge.local_assets import read_header

    names = set(read_header(path)) - {"__metadata__"}
    down = {name.removesuffix(".lora_A.weight") for name in names if name.endswith(".lora_A.weight")}
    up = {name.removesuffix(".lora_B.weight") for name in names if name.endswith(".lora_B.weight")}
    if (
        not down
        or down != up
        or len(names) != 2 * len(down)
        or any(not name.startswith("diffusion_model.") for name in down)
    ):
        raise ValueError("Orbit LoRAの対象テンソル構成が不正です。")
    result["targets"] = sorted(name + ".weight" for name in down)
    verify(result)
    return result


def verify(value: dict | None) -> dict | None:
    from modules_forge.studio_checkpoints import verify_snapshot

    if value is None:
        return None
    try:
        return verify_snapshot(value)
    except (OSError, ValueError) as exc:
        raise ValueError("Orbit LoRAが受付後に変更されました。再実行してください。") from exc


def node_inputs(value: dict) -> dict:
    identity = verify(value)
    return {
        "canonical_path": identity["path"],
        "size": identity["size"],
        "sha256": identity["sha256"],
        "version": json.dumps(value["versions"][""]),
    }


def applied_identity(value: dict, strength_model: float = 1.0) -> dict:
    return dict(verify(value), applied_targets=value["targets"], strength_model=strength_model)


def receipt_identity(history: dict, prompt_id: str, value: dict, strength_model: float = 1.0) -> dict:
    expected = applied_identity(value, strength_model)
    receipts = history.get(prompt_id, {}).get("outputs", {}).get("40", {}).get("orbit_receipt", [])
    version = value["versions"][""]
    if receipts != [dict(expected, version=version)]:
        raise ValueError("Orbit LoRAの読み込み結果を受付時の実体と照合できません。")
    return expected


def install_node(runtime_root: Path) -> Path:
    """Install the local pack only before starting the selected stopped runtime."""
    root = Path(runtime_root).resolve(strict=True)
    if not (root / "main.py").is_file() or not (root / "models").is_dir():
        raise ValueError("Orbitノードの導入先はComfyUIルートを指定してください。")
    custom = root / "custom_nodes"
    target = custom / PACK
    if custom.is_symlink() or custom.resolve() != custom or target.is_symlink() or target.resolve() != target:
        raise ValueError("Orbitノードはリンクされたフォルダーへ導入しません。")
    files = {name: (BUNDLE / name).read_bytes() for name in ("__init__.py", "orbit_bytes.py")}
    if target.exists():
        actual = {p.name for p in target.iterdir()} - {"__pycache__"}
        if actual != files.keys() or any(
            (target / name).is_symlink() or (target / name).read_bytes() != data for name, data in files.items()
        ):
            raise ValueError("Orbitノードの既存ファイルが同梱版と異なります。退避してから再起動してください。")
        return target
    custom.mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".aikimi-orbit-", dir=custom))
    try:
        for name, data in files.items():
            (stage / name).write_bytes(data)
        os.rename(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return target


def _validate(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError("360° Orbit LoRAが未導入です。「Orbit LoRAを準備」を実行してください。")
    identity = file_identity(path)
    if identity["size"] != MODEL_BYTES or identity["sha256"] != MODEL_SHA256:
        raise ValueError(f"360° Orbit LoRAのサイズまたはSHA-256が一致しません。既存ファイルを確認してください: {path}")
    return path


def installed(repository_root: Path = REPO_ROOT, *, runtime_root: Path | None = None) -> bool:
    """Check complete local bytes, reusing file_identity's change-aware hash cache."""
    try:
        if runtime_root is not None:
            validate_model(runtime_root)
        else:
            _validate(configured_model_root(repository_root) / "loras" / MODEL_NAME)
    except (OSError, ValueError):
        return False
    return True


def validate_model(runtime_root: Path) -> Path:
    """Resolve the runtime's configured model directory and verify the LoRA."""
    return _validate(model_root(runtime_root) / "loras" / MODEL_NAME)


def install(repository_root: Path = REPO_ROOT, *, runtime_root: Path | None = None) -> Path:
    """Download fixed public weights and publish atomically without replacing files."""
    target = runtime_root if runtime_root is not None else managed_runtime_root(repository_root)
    with setup_lock(target):
        models = model_root(target) if runtime_root is not None else configured_model_root(repository_root)
        path = models / "loras" / MODEL_NAME
        if path.exists() or path.is_symlink():
            return _validate(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".orbit-", suffix=".part", delete=False) as output:
                temporary = Path(output.name)
                digest, received = hashlib.sha256(), 0
                with urllib.request.urlopen(MODEL_URL, timeout=90) as response:  # noqa: S310
                    while block := response.read(min(1024 * 1024, MODEL_BYTES + 1 - received)):
                        received += len(block)
                        if received > MODEL_BYTES:
                            raise ValueError("360° Orbit LoRAのデータが配布サイズを超えています。")
                        digest.update(block)
                        output.write(block)
                if received != MODEL_BYTES or digest.hexdigest() != MODEL_SHA256:
                    raise ValueError("360° Orbit LoRAのサイズまたはSHA-256が配布版と一致しません。")
                output.flush()
                os.fsync(output.fileno())
            try:
                # Linking a complete same-directory file is atomic and never
                # overwrites a file another process may have just published.
                os.link(temporary, path)
            except FileExistsError:
                return _validate(path)
            return path
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
