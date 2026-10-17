"""Managed, content-bound linear LoRAs applied through Comfy model patches."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path, PurePosixPath

from modules_forge import local_assets

from .core import MossImageError, safe_model_root

_SUFFIXES = (
    (".lora_A.weight", "down", "ab"),
    (".lora_B.weight", "up", "ab"),
    (".lora_down.weight", "down", "down_up"),
    (".lora_up.weight", "up", "down_up"),
    (".alpha", "alpha", None),
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
        raise MossImageError("LoRAは専用フォルダー内のsafetensors名で指定してください。")
    return name


def resolve(root, name):
    validate_name(name)
    try:
        directory = safe_model_root(Path(root) / "loras")
        path = safe_model_root(directory / name)
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError("missing")
        return path
    except (OSError, ValueError) as error:
        raise MossImageError(
            "LoRAが見つからないか、管理フォルダー外を参照しています。一覧を更新してください。"
        ) from error


def inventory(root):
    try:
        directory = safe_model_root(Path(root) / "loras")
        names = []
        for path in directory.rglob("*.safetensors"):
            name = path.relative_to(directory).as_posix()
            try:
                resolve(root, name)
            except ValueError:
                continue
            names.append(name)
        return sorted(names, key=str.casefold)
    except (OSError, ValueError):
        return []


def pairs(values):
    groups, formats = {}, set()
    for key in values:
        if key == "__metadata__":
            continue
        for suffix, side, family in _SUFFIXES:
            if key.endswith(suffix):
                prefix = key.removesuffix(suffix)
                if not prefix:
                    raise MossImageError("LoRAの対象名がありません。")
                entry = groups.setdefault(prefix, {})
                if side in entry:
                    raise MossImageError("LoRAの対象が重複しています。")
                entry[side] = key
                if family:
                    formats.add(family)
                break
        else:
            raise MossImageError(f"未対応のLoRA方式またはキーです: {key}")
    if len(formats) > 1:
        raise MossImageError("LoRAのA/Bとdown/upは同じ形式の組で指定してください。")
    if not groups or any(not {"down", "up"}.issubset(pair) for pair in groups.values()):
        raise MossImageError("LoRAのdown/upまたはA/Bの組が不完全です。")
    return groups


def header_info(path):
    try:
        header = local_assets.read_header(path)
        groups, ranks = pairs(header), set()
        for pair in groups.values():
            down, up = header[pair["down"]], header[pair["up"]]
            for tensor in (down, up):
                shape = tensor.get("shape")
                if (
                    tensor.get("dtype") not in {"F32", "F16", "BF16"}
                    or not isinstance(shape, list)
                    or len(shape) != 2
                    or any(type(x) is not int or x <= 0 for x in shape)
                ):
                    raise MossImageError("LoRAの形状・保存形式が不正です。")
            if down["shape"][0] != up["shape"][1]:
                raise MossImageError("LoRAのrankが一致しません。")
            ranks.add(down["shape"][0])
            if "alpha" in pair:
                alpha = header[pair["alpha"]]
                if alpha.get("dtype") not in {"F32", "F16", "BF16"} or alpha.get("shape") not in ([], [1]):
                    raise MossImageError("LoRAのalphaが不正です。")
        return {"targets": len(groups), "rank": next(iter(ranks)) if len(ranks) == 1 else None}
    except (OSError, ValueError, TypeError, KeyError) as error:
        if isinstance(error, MossImageError):
            raise
        raise MossImageError("LoRAのsafetensorsを読み取れません。") from error


def inspect(root, name):
    path = resolve(root, name)
    try:
        before = local_assets.file_version(path)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        info = header_info(path)
        if before != local_assets.file_version(path) or resolve(root, name) != path:
            raise MossImageError("確認中にLoRAが変更されました。再実行してください。")
    except OSError as error:
        raise MossImageError("LoRAを確認できません。一覧を更新して再実行してください。") from error
    return {"name": name, "sha256": digest, "size": before[0], "mtime_ns": before[1], **info}


def bind(root, request):
    selections = request.get("loras", [])
    identities, seen = [], set()
    for item in selections:
        name, strength = validate_name(item["name"]), item["strength"]
        if (
            name.casefold() in seen
            or type(strength) not in (int, float)
            or not -2 <= strength <= 2
            or not math.isfinite(strength)
        ):
            raise MossImageError("LoRAは重複しない名前と−2〜2の有限の強度で指定してください。")
        seen.add(name.casefold())
        identities.append(inspect(root, name))
    previous = request.get("lora_identities")
    if previous is not None and previous != identities:
        raise MossImageError("選択したLoRAが変更されました。選び直して再実行してください。")
    request["lora_identities"] = identities
    return identities


def record(request):
    selections, identities = request.get("loras", []), request.get("lora_identities", [])
    if len(selections) != len(identities) or any(
        item["name"] != identity.get("name") for item, identity in zip(selections, identities, strict=True)
    ):
        raise MossImageError("LoRAの保存条件がありません。")
    return [
        {**identity, "strength": item["strength"], "applied_targets": identity["targets"] if item["strength"] else 0}
        for item, identity in zip(selections, identities, strict=True)
    ]


def validate_tensors(state, key_map, model_state, torch):
    groups = pairs(state)
    targets, ranges = set(), []
    for key, value in state.items():
        if (
            not torch.is_tensor(value)
            or value.dtype not in {torch.float32, torch.float16, torch.bfloat16}
            or not bool(torch.isfinite(value).all())
        ):
            raise MossImageError(f"LoRAは有限の浮動小数点テンソルで指定してください: {key}")
    for prefix, pair in groups.items():
        target = key_map.get(prefix)
        key, narrow = target if isinstance(target, tuple) else (target, None)
        if key not in model_state:
            raise MossImageError(f"LoRAの対象がMOSSモデルにありません: {prefix}")
        shape = list(model_state[key].shape)
        if len(shape) != 2:
            raise MossImageError("LoRAはMOSSモデルの線形レイヤーを対象にしてください。")
        start, end = 0, shape[0]
        if narrow is not None:
            dimension, offset, length = narrow
            if dimension != 0 or offset < 0 or length <= 0 or offset + length > shape[0]:
                raise MossImageError("LoRAの融合レイヤー範囲が不正です。")
            shape[dimension] = length
            start, end = offset, offset + length
        down, up = state[pair["down"]], state[pair["up"]]
        if (
            down.ndim != 2
            or up.ndim != 2
            or down.shape[0] <= 0
            or down.shape[0] != up.shape[1]
            or shape != [up.shape[0], down.shape[1]]
        ):
            raise MossImageError(f"LoRAとMOSSモデルの形状・rankが一致しません: {prefix}")
        if "alpha" in pair:
            alpha = state[pair["alpha"]]
            if alpha.numel() != 1 or not math.isfinite(float(alpha.item())):
                raise MossImageError("LoRAのalphaは有限のスカラーで指定してください。")
        if any(key == other and max(start, left) < min(end, right) for other, left, right in ranges):
            raise MossImageError("LoRAの同じ対象が別の名前で重複しています。")
        ranges.append((key, start, end))
        targets.add(target)
    if len(targets) != len(groups):
        raise MossImageError("LoRAの同じ対象が別の名前で重複しています。")
    return targets


def apply(model, root, request, mapper, torch):
    bind(root, request)
    active = [item for item in request.get("loras", []) if item["strength"] != 0]
    if not active:
        return model, 0
    from safetensors import SafetensorError
    from safetensors.torch import load_file

    model_state = model.model.state_dict()
    key_map = mapper.model_lora_keys_unet(model.model, {})

    def add_alias(name, target):
        if name in key_map and key_map[name] != target:
            raise MossImageError("LoRAの対象名が複数のレイヤーと衝突しています。")
        key_map[name] = target

    for key in model_state:
        if key.endswith(".weight"):
            prefix = key.removesuffix(".weight")
            native = prefix.removeprefix("diffusion_model.")
            for name in (prefix, native, "transformer." + native, "base_model.model." + native):
                add_alias(name, key)
    for prefix, target in tuple(key_map.items()):
        if prefix.startswith("diffusion_model."):
            native = prefix.removeprefix("diffusion_model.")
            for name in (native, "transformer." + native, "base_model.model." + native):
                add_alias(name, target)
    plans = []
    for item in active:
        try:
            state = load_file(str(resolve(root, item["name"])), device="cpu")
        except (OSError, SafetensorError) as error:
            raise MossImageError("LoRAのsafetensorsを読み取れません。") from error
        targets = validate_tensors(state, key_map, model_state, torch)
        selected = {prefix: key_map[prefix] for prefix in pairs(state)}
        patches = mapper.load_lora(state, selected)
        if set(patches) != targets:
            raise MossImageError("LoRAの全対象を読み込めませんでした。")
        plans.append((patches, item["strength"], targets))
    bind(root, request)
    patched, count = model.clone(), 0
    for patches, strength, targets in plans:
        if set(patched.add_patches(patches, strength)) != targets:
            raise MossImageError("LoRAの全対象をモデルへ登録できませんでした。")
        count += len(targets)
    return patched, count
