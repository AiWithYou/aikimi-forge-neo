"""Managed FLUX 3 native LoRAs: header selection and exact content identities."""

from __future__ import annotations

import math
import re
from pathlib import Path, PurePosixPath

from modules_forge import local_assets

from .core import Flux3Error
from .paths import canonical_path

_TARGET = re.compile(
    r"diffusion_model\.(?:content_mode_blocks\.(?:image|image_cond)\.[0-9]+|"
    r"txt_mode_blocks\.[0-9]+|single_blocks\.[0-9]+)\."
    r"(?:q_proj|k_proj|v_proj|qkv|attn_out|mlp_in|mlp_out)"
)
_SUFFIXES = (
    (".lora_down.weight", "down"),
    (".lora_up.weight", "up"),
    (".lora_A.weight", "down"),
    (".lora_B.weight", "up"),
    (".alpha", "alpha"),
)


def validate_name(name):
    if (
        not isinstance(name, str)
        or not name
        or len(name) > 512
        or "\\" in name
        or ":" in name
        or name.startswith("/")
        or any(not part or part.startswith(".") for part in name.split("/"))
        or PurePosixPath(name).suffix.lower() != ".safetensors"
    ):
        raise Flux3Error("LoRAは専用フォルダー内のsafetensors名で指定してください。")
    return name


def resolve(root, name):
    validate_name(name)
    try:
        directory = canonical_path(Path(root) / "loras")
        path = canonical_path(directory / name)
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError("missing")
        return path
    except (OSError, ValueError) as error:
        raise Flux3Error("LoRAが見つからないか、管理フォルダー外を参照しています。一覧を更新してください。") from error


def inventory(root):
    try:
        directory = canonical_path(Path(root) / "loras")
        result = []
        for path in directory.rglob("*.safetensors"):
            name = path.relative_to(directory).as_posix()
            try:
                resolve(root, name)
            except ValueError:
                continue
            result.append(name)
        return sorted(result, key=str.casefold)
    except (OSError, ValueError):
        return []


def pairs(values):
    result = {}
    for key in values:
        if key == "__metadata__":
            continue
        matched = False
        for suffix, side in _SUFFIXES:
            if key.endswith(suffix):
                prefix = key.removesuffix(suffix)
                if not _TARGET.fullmatch(prefix):
                    break
                entry = result.setdefault(prefix, {})
                if side in entry:
                    raise Flux3Error(f"LoRAの対象が重複しています: {prefix}")
                entry[side] = key
                matched = True
                break
        if not matched:
            raise Flux3Error(f"未対応のFLUX 3 LoRAキーです: {key}")
    if not result or any(not {"down", "up"} <= entry.keys() for entry in result.values()):
        raise Flux3Error("LoRAのdown/upの組が不完全です。")
    for prefix, entry in result.items():
        if entry["down"].endswith(".lora_down.weight") != entry["up"].endswith(".lora_up.weight"):
            raise Flux3Error(f"LoRAのdown/upまたはA/Bは同じ形式の組で指定してください: {prefix}")
    return result


def header_info(path):
    header = local_assets.read_header(path)
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()
    ):
        raise Flux3Error("LoRAのメタデータが不正です。")
    groups, ranks = pairs(header), set()
    for prefix, pair in groups.items():
        down, up = header[pair["down"]], header[pair["up"]]
        for value in (down, up):
            shape = value.get("shape")
            if (
                value.get("dtype") not in {"F32", "F16", "BF16"}
                or not isinstance(shape, list)
                or len(shape) != 2
                or any(type(x) is not int or x <= 0 for x in shape)
            ):
                raise Flux3Error(f"LoRAの形状・保存形式が不正です: {prefix}")
        if down["shape"][0] != up["shape"][1]:
            raise Flux3Error(f"LoRAのrankが一致しません: {prefix}")
        ranks.add(down["shape"][0])
        if "alpha" in pair:
            alpha = header[pair["alpha"]]
            if alpha.get("dtype") not in {"F32", "F16", "BF16"} or alpha.get("shape") not in ([], [1]):
                raise Flux3Error("LoRAのalphaが不正です。")
    source = metadata.get("base_model", "")
    training_source = (
        "base"
        if source.endswith("flux-3-image-klein-base")
        else ("distilled" if source.endswith("flux-3-image-klein") else None)
    )
    return {
        "rank": next(iter(ranks)) if len(ranks) == 1 else None,
        "targets": len(groups),
        "base_model": training_source,
    }


def inspect(root, name):
    path = resolve(root, name)
    before = local_assets.file_identity(path)
    info = header_info(path)
    if before != local_assets.file_identity(path):
        raise Flux3Error("確認中にLoRAが変更されました。再実行してください。")
    return {"name": name, "sha256": before["sha256"], "size": before["size"], **info}


def bind(root, request):
    name = request.get("lora")
    if not name:
        return None
    current = inspect(root, name)
    previous = request.get("lora_identity")
    if previous is not None and previous != current:
        raise Flux3Error("選択したLoRAが変更されました。選び直して再実行してください。")
    request["lora_identity"] = current
    return current


def record(request):
    if not request.get("lora"):
        return None
    identity = request.get("lora_identity")
    if not isinstance(identity, dict) or identity.get("name") != request["lora"]:
        raise Flux3Error("LoRAの保存条件がありません。")
    strength = request.get("lora_strength", 1.0)
    return {**identity, "strength": strength, "applied_targets": identity["targets"] if strength else 0}


def validate_tensors(state, key_map, model_state, torch):
    groups = pairs(state)
    for key, value in state.items():
        if (
            not torch.is_tensor(value)
            or value.dtype not in {torch.float32, torch.float16, torch.bfloat16}
            or not bool(torch.isfinite(value).all())
        ):
            raise Flux3Error(f"LoRAのテンソルは有限の浮動小数点数である必要があります: {key}")
    for prefix, pair in groups.items():
        down, up = state[pair["down"]], state[pair["up"]]
        target = key_map.get(prefix)
        key, narrow = target if isinstance(target, tuple) else (target, None)
        if key not in model_state:
            raise Flux3Error(f"LoRAの対象がモデルにありません: {prefix}")
        shape = list(model_state[key].shape)
        if narrow is not None:
            dimension, offset, length = narrow
            if dimension != 0 or offset < 0 or length <= 0 or offset + length > shape[0]:
                raise Flux3Error(f"LoRAの融合レイヤー範囲が不正です: {prefix}")
            shape[dimension] = length
        if (
            down.ndim != 2
            or up.ndim != 2
            or down.shape[0] <= 0
            or down.shape[0] != up.shape[1]
            or shape != [up.shape[0], down.shape[1]]
        ):
            raise Flux3Error(f"LoRAとモデルの形状・rankが一致しません: {prefix}")
        if "alpha" in pair:
            alpha = state[pair["alpha"]]
            if alpha.numel() != 1 or not math.isfinite(float(alpha.item())) or float(alpha.item()) <= 0:
                raise Flux3Error("LoRAのalphaは正の有限値で指定してください。")
    return len(groups)
