"""Lightweight compatibility probes and exact ComfyUI checkpoint roots."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from modules_forge import local_assets


def h3_role(path: Path) -> str:
    path = local_assets.local_path(str(path), suffixes=(".safetensors",))
    header = local_assets.read_header(path)
    if not {"audio_patch_proj.weight", "video_patch_proj.weight", "adaln_t_table"} <= header.keys():
        raise ValueError("MiniMax H3互換の本体を選択してください。")
    required_shapes = {
        "audio_patch_proj.weight": [5376, 32],
        "video_patch_proj.weight": [5376, 96],
        "adaln_t_table": [1025, 8],
    }
    if any(
        not isinstance(header[name], dict) or header[name].get("shape") != shape
        for name, shape in required_shapes.items()
    ):
        raise ValueError("MiniMax H3本体のテンソル形状が対応構成と一致しません。")
    metadata = header.get("__metadata__", {})
    role = metadata.get("h3_role", "") if isinstance(metadata, dict) else ""
    name = path.name.lower()
    if role in {"fl2va", "ref2va", "both"}:
        return role
    if "fused_refdelta" in name or "fused_turbo" in name:
        return "both"
    if "fl2va" in name and "ref2va" in name:
        raise ValueError("H3本体の役割が曖昧です。h3_roleメタデータでFL2VA / Ref2VAの役割を指定してください。")
    if "ref2va" in name:
        return "ref2va"
    if "fl2va" in name:
        return "fl2va"
    raise ValueError(
        "H3本体の役割を識別できません。ファイル名にFL2VA / Ref2VAを含めるか、h3_roleメタデータを指定してください。"
    )


def h3_variant(checkpoint: str) -> str:
    if not checkpoint or not Path(checkpoint).is_file():
        return "base"  # Keep a disappeared library selection visible; readiness rejects it.
    if checkpoint:
        try:
            metadata = local_assets.read_header(Path(checkpoint)).get("__metadata__", {})
            quantization = json.loads(metadata.get("_quantization_metadata", "{}"))
            if any("w4a8" in layer.get("format", "") for layer in quantization.get("layers", {}).values()):
                return "w4a8"
            if metadata.get("h3_variant") in {"base", "w4a8", "fused_turbo"}:
                return metadata["h3_variant"]
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    name = Path(checkpoint).name.lower()
    if "w4a8" in name:
        return "w4a8"
    if "fused_refdelta" in name or "fused_turbo" in name:
        return "fused_turbo"
    if "_pruned_int8_convrot" in name:
        return "base"
    raise ValueError(
        "H3本体の構成を識別できません。h3_variantメタデータへbase / w4a8 / fused_turboを指定してください。"
    )


def h3_compatible(path: Path) -> bool:
    try:
        return bool(h3_role(path) and h3_variant(str(path)))
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def nanosaur_compatible(path: Path) -> bool:
    try:
        header = local_assets.read_header(local_assets.local_path(str(path), suffixes=(".safetensors",)))
        required = {
            "s_embedder.proj.weight",
            "final_layer.linear.weight",
            "blocks.0.attn.q_norm.weight",
            "blocks.0.mlp.w12.weight",
        }
        if not required <= header.keys():
            return False
        input_shape = header["s_embedder.proj.weight"].get("shape", [])
        output_shape = header["final_layer.linear.weight"].get("shape", [])
        q_shape = header["blocks.0.attn.q_norm.weight"].get("shape", [])
        mlp_shape = header["blocks.0.mlp.w12.weight"].get("shape", [])
        return (
            len(input_shape) == 2
            and input_shape[1] == 64
            and output_shape == [64, input_shape[0]]
            and len(q_shape) == 1
            and q_shape[0] > 0
            and input_shape[0] % q_shape[0] == 0
            and len(mlp_shape) == 2
            and mlp_shape[1] == input_shape[0]
            and mlp_shape[0] % 2 == 0
        )
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def config_path(runtime: Path, checkpoint: str) -> Path:
    selected = Path(checkpoint).resolve()
    version = local_assets.file_version(selected)
    if version[2] is None:
        version = ("sha256", local_assets.file_identity(selected)["sha256"])
    key = hashlib.sha256(json.dumps((os.path.normcase(str(selected)), version)).encode("utf-8")).hexdigest()[:20]
    return Path(runtime) / f".neo-checkpoints-{key}.json"


def write_comfy_config(runtime: Path, checkpoint: str) -> Path:
    """is_default prepends the selected parent, resolving equal basename collisions."""
    selected = local_assets.local_path(checkpoint, suffixes=(".safetensors",))
    config = config_path(runtime, checkpoint)
    content = (
        json.dumps(
            {"neo_checkpoint": {"base_path": str(selected.parent), "diffusion_models": ".", "is_default": True}},
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    if not config.is_file() or config.read_text(encoding="utf-8") != content:
        config.write_text(content, encoding="utf-8")
    return config


def config_loaded(readiness, runtime: Path, checkpoint: str) -> bool:
    neo_configs = [
        Path(value).resolve()
        for value in _comfy_configs(readiness)
        if Path(value).name.startswith(".neo-checkpoints-") and Path(value).suffix == ".json"
    ]
    if not checkpoint:
        return not neo_configs
    expected = config_path(runtime, checkpoint)
    return neo_configs == [expected.resolve()]


def _comfy_configs(readiness) -> tuple[str, ...]:
    values = []
    collecting = False
    for value in getattr(readiness, "runtime_args", ()):
        if value.startswith("--extra-model-paths-config="):
            values.append(value.split("=", 1)[1])
            collecting = False
        elif value == "--extra-model-paths-config":
            collecting = True
        elif value.startswith("--"):
            collecting = False
        elif collecting:
            values.append(value)
    return tuple(values)


def user_comfy_configs(readiness) -> tuple[str, ...]:
    """Preserve user configs while replacing only this Studio's extra roots."""
    return tuple(value for value in _comfy_configs(readiness) if not Path(value).name.startswith(".neo-checkpoints-"))


def _snapshot_versions(path: Path, directory: bool) -> dict:
    paths = (
        sorted(item for item in path.rglob("*") if item.is_file() and ".cache" not in item.relative_to(path).parts)
        if directory
        else [path]
    )
    return {str(item.relative_to(path)) if directory else "": list(local_assets.file_version(item)) for item in paths}


def snapshot_checkpoint(checkpoint: str, *, directory: bool = False) -> dict | None:
    """Freeze the selected content before a job starts waiting for the GPU."""
    if not checkpoint:
        return None
    path = local_assets.local_path(checkpoint, directory=directory)
    before = _snapshot_versions(path, directory)
    identity = local_assets.identity(path) if directory else local_assets.file_identity(path)
    after = _snapshot_versions(path, directory)
    if before != after:
        raise ValueError("選択したチェックポイントが変更されました。実行し直してください。")
    return {"path": str(path), "directory": directory, "identity": dict(identity), "versions": after}


def verify_snapshot(snapshot: dict | None) -> dict | None:
    """Reject replacement and return the original identity for result metadata."""
    if snapshot is None:
        return None
    try:
        path = local_assets.local_path(snapshot["path"], directory=snapshot["directory"])
        versions = _snapshot_versions(path, snapshot["directory"])
        if versions != snapshot["versions"]:
            raise ValueError("version changed")
        if any(version[2] is None for version in versions.values()):
            identity = local_assets.identity(path) if snapshot["directory"] else local_assets.file_identity(path)
            if identity != snapshot["identity"]:
                raise ValueError("content changed")
    except (OSError, ValueError) as exc:
        raise ValueError("選択したチェックポイントが変更されました。実行し直してください。") from exc
    return snapshot["identity"]


def result_identity(checkpoint: str, snapshot: dict | None) -> dict | None:
    if checkpoint and (snapshot is None or Path(checkpoint).resolve() != Path(snapshot["path"]).resolve()):
        raise ValueError("選択したチェックポイントの開始時情報がありません。実行し直してください。")
    return verify_snapshot(snapshot)
