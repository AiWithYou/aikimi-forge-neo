"""Local safetensors inventory and strict Qwen 2.1 adapter validation (no torch)."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from pathlib import Path

SUSHI_SOURCE = "https://github.com/celll1/SushiUI/blob/69de838b18dbed8e8fc1e1294fb649a4f7dc5452/backend/core/pipeline_backends/qwen_image_21.py#L55-L78"
TRAINING_KEYS = {
    "qwen_partition_global_adapter." + key
    for key in ("alpha", "key.weight", "value.weight", "query.weight", "out.weight", "summary_queries")
}


def directory(runtime: Path) -> Path:
    return Path(runtime) / "loras"


def resolve(runtime: Path, name: str) -> Path:
    root = directory(runtime).resolve()
    if not isinstance(name, str) or not name:
        raise ValueError("LoRAは一覧から選択してください。")
    name = name.strip().strip('"')
    if Path(name).is_absolute():
        from modules_forge.local_assets import local_path

        return local_path(name, suffixes=(".safetensors",))
    path = (root / name).resolve()
    if not path.is_relative_to(root) or path.suffix.lower() != ".safetensors" or not path.is_file():
        raise ValueError("LoRAが見つかりません。一覧を更新して選び直してください。")
    return path


def inventory(runtime: Path) -> list[str]:
    from modules_forge.local_assets import library

    root = directory(runtime)
    local = [p.relative_to(root).as_posix() for p in root.rglob("*.safetensors") if p.is_file()]
    return sorted(set(local + library().get("qwen21_lora", [])))


def read_header(path: Path) -> dict:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("LoRAファイルが不完全です。")
        size = struct.unpack("<Q", prefix)[0]
        if not 2 <= size <= 16 * 1024 * 1024 or size + 8 > path.stat().st_size:
            raise ValueError("safetensorsのヘッダーが不正です。")
        header = json.loads(stream.read(size))
    if not isinstance(header, dict):
        raise ValueError("safetensorsの内容が不正です。")
    return header


def target_name(key: str) -> tuple[str, str] | None:
    if key.startswith("lora_unet_"):
        key = key.removeprefix("lora_unet_").replace("__", ".")
    else:
        for prefix in ("diffusion_model.", "transformer.", "base_model.model."):
            key = key.removeprefix(prefix)
    for suffix, side in (
        (".lora_down.weight", "down"),
        (".lora_up.weight", "up"),
        (".lora_A.weight", "down"),
        (".lora_B.weight", "up"),
        (".alpha", "alpha"),
    ):
        if key.endswith(suffix):
            name = key.removesuffix(suffix)
            if re.fullmatch(
                r"transformer_blocks\.([0-9]|[12][0-9]|3[01])\.(attn\.(to_q|to_k|to_v|to_out\.0)|img_mlp\.(proj|out|gate_layer|gate_up))",
                name,
            ) or name in {"txt_in.in_layer", "txt_in.out_layer"}:
                return name, side
    return None


def inspect(runtime: Path, name: str) -> dict:
    path = resolve(runtime, name)
    header = read_header(path)
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        raise ValueError("LoRAメタデータが不正です。")
    architecture = metadata.get("model_type", metadata.get("modelspec.architecture", ""))
    if architecture and architecture not in {"qwen_image_21", "qwen-image-2.1", "qwen_image21"}:
        raise ValueError(f"Qwen Image 2.1用ではありません: {architecture}")
    groups, ignored, unknown = {}, [], []
    for key, record in header.items():
        if key == "__metadata__":
            continue
        if key in TRAINING_KEYS and metadata.get("qwen_partition_global_adapter") == "latent_summary_v1":
            ignored.append(key)
            continue
        mapped = target_name(key)
        if mapped is None:
            unknown.append(key)
            continue
        name_, side = mapped
        group = groups.setdefault(name_, {})
        if side in group:
            raise ValueError(f"LoRAの対象が重複しています: {name_}")
        group[side] = key
        if not isinstance(record, dict) or not isinstance(record.get("shape"), list):
            raise ValueError(f"LoRAテンソルが不正です: {key}")
    if unknown:
        raise ValueError(f"未対応のLoRA形式です（黙って除外しません）: {unknown[0]}")
    if not groups:
        raise ValueError("Qwen 2.1用のLoRA差分がありません。")
    for name_, group in groups.items():
        if not {"down", "up"} <= group.keys():
            raise ValueError(f"LoRAのペアが不足しています: {name_}")
        down, up = (header[group[side]]["shape"] for side in ("down", "up"))
        if len(down) != 2 or len(up) != 2 or not 0 < down[0] == up[1] or min(*down, *up) <= 0:
            raise ValueError(f"LoRAの形状が不正です: {name_}")
    base_forward = metadata.get("qwen_base_forward", "")
    if base_forward and base_forward != "convrot_int8_bf16_backward_v1":
        raise ValueError(f"未対応の学習時モデルです: {base_forward}")
    from . import consistency_lora

    consistency_version = consistency_lora.identify(path, metadata)
    info = {
        "name": name,
        "path": str(path),
        "groups": groups,
        "metadata": metadata,
        "linear_layers": len(groups),
        "training_only_keys": sorted(ignored),
        "base_mismatch": bool(base_forward),
        "source": SUSHI_SOURCE if ignored or base_forward else None,
    }
    if consistency_version:
        info.update(consistency_version=consistency_version, source=consistency_lora.SOURCE)
    return info


def validate_options(values: dict) -> None:
    adapters = values.get("style_loras", ())
    if not isinstance(adapters, (list, tuple)):
        raise ValueError("LoRAは一覧から選択してください。")
    seen = set()
    for item in adapters:
        if not isinstance(item, dict) or set(item) != {"name", "strength"}:
            raise ValueError("LoRAの指定が不正です。")
        name, strength = item["name"], item["strength"]
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("LoRA名が空欄、または重複しています。")
        seen.add(name)
        if (
            isinstance(strength, bool)
            or not isinstance(strength, (int, float))
            or not math.isfinite(strength)
            or not -2 <= strength <= 2
        ):
            raise ValueError(f"{name}: 強度は−2〜2の数値で指定してください。")
    if not isinstance(values.get("allow_lora_base_mismatch", False), bool):
        raise ValueError("異なる量子化の指定が不正です。")
    if any(item["strength"] != 0 for item in adapters):
        if values.get("operation", "generate") != "generate":
            raise ValueError("追加LoRAは画像生成で使用してください。")


def validate_installed(runtime: Path, values: dict) -> list[dict]:
    validate_options(values)
    infos = []
    seen = set()
    for item in values.get("style_loras", ()):
        if item["strength"] == 0:
            continue
        try:
            info = inspect(runtime, item["name"])
            path = Path(info["path"]).resolve()
            if path in seen:
                raise ValueError("同じ実ファイルのLoRAを二重に指定できません。")
            seen.add(path)
            if info["base_mismatch"] and not values.get("allow_lora_base_mismatch"):
                raise ValueError("ConvRot INT8用です。「異なる量子化で試す」をONにした実験のみ可能です。")
            infos.append({**info, "strength": item["strength"]})
        except (ValueError, OSError) as exc:
            raise ValueError(f"{item['name']}: {exc}") from exc
    return infos


def cache_key(runtime: Path, values: dict) -> tuple:
    entries = []
    for item in values.get("style_loras", ()):
        if item["strength"] == 0:
            continue
        path = resolve(runtime, item["name"])
        from modules_forge.local_assets import file_identity

        entries.append((str(path), file_identity(path)["sha256"], item["strength"]))
    return tuple(entries), values.get("allow_lora_base_mismatch", False)


def fingerprint(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
