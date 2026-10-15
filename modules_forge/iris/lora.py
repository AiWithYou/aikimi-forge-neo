"""Local Iris adapters: header-only selection and strict worker-side application."""

from __future__ import annotations

import json
import math
from pathlib import Path

from modules_forge import local_assets

SUFFIXES = (
    (".lora_down", "down"),
    (".lora_up", "up"),
    (".lora_down.weight", "down"),
    (".lora_up.weight", "up"),
    (".lora_A.weight", "down"),
    (".lora_B.weight", "up"),
)
HYBRID_FORMAT = "iris-turbo-svd-v1"
DENSE_PREFIX = "dense."


def inventory(root):
    directory = Path(root) / "loras"
    return sorted(
        p.relative_to(directory).as_posix()
        for p in directory.rglob("*.safetensors")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(directory).parts)
    )


def resolve(root, name):
    if not isinstance(name, str) or not name.strip() or "://" in name:
        raise ValueError("LoRAを選び直してください。")
    name = name.strip().strip('"')
    if Path(name).is_absolute():
        return local_assets.local_path(name, suffixes=(".safetensors",))
    directory = (Path(root) / "loras").resolve()
    path = (directory / name).resolve()
    if not path.is_relative_to(directory) or path.suffix.lower() != ".safetensors" or not path.is_file():
        raise ValueError("LoRAが見つかりません。一覧を更新して選び直してください。")
    return path


def target(key):
    for suffix, side in SUFFIXES:
        if key.endswith(suffix):
            name = key.removesuffix(suffix).removeprefix("base_model.model.")
            if name and all(part and (part.isidentifier() or part.isdecimal()) for part in name.split(".")):
                return name, side
    raise ValueError(f"未対応のLoRAキーです: {key}")


def pairs(header):
    result = {}
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        raise ValueError("LoRAのメタデータが不正です。")
    hybrid = metadata.get("format") == HYBRID_FORMAT
    for key in header:
        if key == "__metadata__":
            continue
        if hybrid and key.startswith(DENSE_PREFIX):
            continue
        name, side = target(key)
        entry = result.setdefault(name, {})
        if side in entry:
            raise ValueError(f"LoRAの対象が重複しています: {name}")
        entry[side] = key
    if not result or any(set(pair) != {"down", "up"} for pair in result.values()):
        raise ValueError("LoRAのdown/upの対が不完全です。")
    return result


def inspect(root, name):
    header = local_assets.read_header(resolve(root, name))
    targets = pairs(header)
    ranks = set()
    for name, pair in targets.items():
        down, up = (header[pair[side]] for side in ("down", "up"))
        for entry in (down, up):
            shape = entry.get("shape", [])
            if (
                entry.get("dtype") not in {"F32", "F16", "BF16"}
                or len(shape) != 2
                or any(type(x) is not int or x <= 0 for x in shape)
            ):
                raise ValueError(f"LoRAの形状・保存形式が不正です: {name}")
        if down["shape"][0] != up["shape"][1]:
            raise ValueError(f"LoRAのrankが一致しません: {name}")
        ranks.add(down["shape"][0])
    metadata = header.get("__metadata__", {})
    if len(ranks) != 1:
        raise ValueError("LoRAのrank・メタデータが不正です。")
    rank = ranks.pop()
    try:
        if "rank" in metadata and int(metadata["rank"]) != rank:
            raise ValueError
        alpha = float(metadata.get("alpha", rank))
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError
        extra = json.loads(metadata.get("extra", "{}"))
        if not isinstance(extra, dict):
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("LoRAのrank・alpha・メタデータが不正です。") from exc
    result = {"rank": rank, "alpha": alpha}
    if metadata.get("format") == HYBRID_FORMAT:
        from .official_turbo import validate_settings

        try:
            for field in ("base_sha256", "base_config_sha256", "target_sha256"):
                value = metadata[field]
                if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                    raise ValueError
            if alpha != rank or "turbo_steps" in extra:
                raise ValueError
            sampling = validate_settings(json.loads(metadata["sampling"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("抽出Turbo LoRAの元モデル・生成設定が不正です。") from exc
        dense = {key: value for key, value in header.items() if key.startswith(DENSE_PREFIX)}
        if not dense:
            raise ValueError("抽出Turbo LoRAの非Linearデータがありません。")
        for key, value in dense.items():
            name = key.removeprefix(DENSE_PREFIX)
            if (
                not name
                or not all(part and (part.isidentifier() or part.isdecimal()) for part in name.split("."))
                or value.get("dtype") != "BF16"
                or any(type(size) is not int or size <= 0 for size in value.get("shape", []))
            ):
                raise ValueError(f"非Linearデータの形状・名前が不正です: {key}")
        result.update(
            format=HYBRID_FORMAT,
            base_sha256=metadata["base_sha256"],
            base_config_sha256=metadata["base_config_sha256"],
            target_sha256=metadata["target_sha256"],
            sampling=sampling,
        )
    if "turbo_steps" in extra:
        steps, shift = extra.get("turbo_steps"), extra.get("shift")
        try:
            if type(steps) is not int or not 1 <= steps <= 200 or isinstance(shift, bool):
                raise ValueError
            shift = float(shift)
            if not math.isfinite(shift) or shift <= 0:
                raise ValueError
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Turbo LoRAのSteps・shiftが不正です。") from exc
        result.update(turbo_steps=steps, shift=shift)
    return result


def snapshot(root, loras):
    if not isinstance(loras, (list, tuple)):
        raise ValueError("LoRAの選択を確認してください。")
    result, seen = [], set()
    for item in loras:
        if not isinstance(item, dict):
            raise ValueError("LoRAの選択を確認してください。")
        try:
            strength = item.get("strength", 1)
            if isinstance(strength, bool):
                raise ValueError
            strength = float(strength)
            if not math.isfinite(strength) or not -2 <= strength <= 2:
                raise ValueError
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("LoRAの強度は−2〜2の数値で指定してください。") from exc
        if strength == 0:
            continue
        path = resolve(root, item.get("name"))
        if path in seen:
            raise ValueError("同じLoRAを重複して指定しています。")
        seen.add(path)
        source = local_assets.file_identity(path)
        metadata = inspect(root, str(path))
        result.append(
            {"name": item["name"], "path": str(path), "strength": strength, "source": source, "metadata": metadata}
        )
    verify_sources(result)
    sampling_settings(result)
    return result


def verify_sources(loras):
    for item in loras:
        try:
            if local_assets.file_identity(Path(item["path"])) != item["source"]:
                raise ValueError
        except (OSError, ValueError) as exc:
            raise ValueError("受付後にLoRAが更新されました。選び直して再実行してください。") from exc


def sampling_settings(loras):
    active = [
        item
        for item in loras
        if item["strength"] != 0 and ("turbo_steps" in item["metadata"] or "sampling" in item["metadata"])
    ]
    if not active:
        return {"mode": "standard"}
    if len(active) != 1 or active[0]["strength"] != 1:
        raise ValueError("Turbo LoRAは一つだけ選び、強度1で使用してください。")
    metadata = active[0]["metadata"]
    if "sampling" in metadata:
        return metadata["sampling"]
    return {"mode": "turbo", "default_steps": metadata["turbo_steps"], "shift": metadata["shift"]}


def verify_base(loras, identity, precision):
    for item in loras:
        metadata = item["metadata"]
        if metadata.get("format") != HYBRID_FORMAT:
            continue
        if precision != "normal":
            raise ValueError("抽出Turbo LoRAは通常版の元モデルで使用してください。")
        if (
            identity.get("weights", {}).get("sha256") != metadata["base_sha256"]
            or identity.get("config", {}).get("sha256") != metadata["base_config_sha256"]
        ):
            raise ValueError("抽出Turbo LoRAの元モデル・設定が一致しません。オリジナルの通常版を選んでください。")


def apply(model, loras):
    """Validate every tensor before altering a newly loaded CPU model."""
    import torch
    from safetensors import safe_open
    from torch import nn
    from torch.nn import functional as F

    from .quantization import Int8Linear
    from .w4a8 import W4A8Linear

    verify_sources(loras)
    sampling_settings(loras)
    prepared, replacements = [], []
    for item in loras:
        seen_modules = set()
        with safe_open(item["path"], framework="pt", device="cpu") as file:
            header = {**dict.fromkeys(file.keys()), "__metadata__": file.metadata() or {}}
            for name, pair in pairs(header).items():
                try:
                    module = model.get_submodule(name)
                except AttributeError as exc:
                    raise ValueError(f"Iris本体にLoRAの対象がありません: {name}") from exc
                if not isinstance(module, (nn.Linear, Int8Linear, W4A8Linear)):
                    raise ValueError(f"LoRAの対象がLinearではありません: {name}")
                if id(module) in seen_modules:
                    raise ValueError(f"同じ層へのLoRA指定が重複しています: {name}")
                seen_modules.add(id(module))
                down, up = (file.get_tensor(pair[side]).float() for side in ("down", "up"))
                rank = item["metadata"]["rank"]
                scale = item["strength"] * item["metadata"]["alpha"] / rank
                if down.shape != (rank, module.in_features) or up.shape != (module.out_features, rank):
                    raise ValueError(f"LoRAの形状が本体に一致しません: {name}")
                if not math.isfinite(scale) or not torch.isfinite(down).all() or not torch.isfinite(up).all():
                    raise ValueError(f"LoRAに非有限値が含まれています: {name}")
                prepared.append((module, down, up, scale))
            if item["metadata"].get("format") == HYBRID_FORMAT:
                for key in file.keys():
                    if not key.startswith(DENSE_PREFIX):
                        continue
                    name = key.removeprefix(DENSE_PREFIX)
                    try:
                        parameter = model.get_parameter(name)
                        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
                    except AttributeError as exc:
                        raise ValueError(f"非Linearデータの対象がありません: {name}") from exc
                    if name.endswith(".weight") and isinstance(parent, (nn.Linear, Int8Linear, W4A8Linear)):
                        raise ValueError(f"Linearの重みは非Linearデータで差し替えできません: {name}")
                    value = file.get_tensor(key).float()
                    if value.shape != parameter.shape:
                        raise ValueError(f"非Linearデータの形状が一致しません: {name}")
                    if not torch.isfinite(value).all():
                        raise ValueError(f"非Linearデータに非有限値があります: {name}")
                    replacements.append((parameter, value))
    verify_sources(loras)
    with torch.no_grad():
        for parameter, value in replacements:
            parameter.copy_(value.to(parameter.dtype))
        for index, (module, down, up, scale) in enumerate(prepared):
            if isinstance(module, nn.Linear):
                merged = module.weight.float() + (up @ down) * scale
                if not torch.isfinite(merged).all():
                    raise ValueError("LoRAの合成結果に非有限値が含まれています。")
                # Keep Parameter identity for shared modules without dirtying
                # every page of a large safetensors memory-mapped base.
                module.weight.set_(merged.to(module.weight.dtype))
            else:
                down_name, up_name = f"_iris_lora_down_{index}", f"_iris_lora_up_{index}"
                module.register_buffer(down_name, down, persistent=False)
                module.register_buffer(up_name, up, persistent=False)

                def residual(layer, args, output, down_name=down_name, up_name=up_name, scale=scale):
                    dtype = (
                        torch.get_autocast_dtype(args[0].device.type)
                        if torch.is_autocast_enabled(args[0].device.type)
                        else args[0].dtype
                    )
                    delta = F.linear(
                        F.linear(args[0].to(dtype), getattr(layer, down_name).to(dtype)),
                        getattr(layer, up_name).to(dtype),
                    )
                    return output + (delta * scale).to(output.dtype)

                module.register_forward_hook(residual)
