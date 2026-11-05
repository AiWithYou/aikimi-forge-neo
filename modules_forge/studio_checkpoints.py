"""Lightweight compatibility probes and exact ComfyUI checkpoint roots."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from pathlib import Path

from modules_forge import local_assets


def h3_variant(checkpoint: str, *, lora_targets=()) -> str:
    """Validate H3 block shapes and identify storage, never model purpose."""
    if not checkpoint:
        return "base"
    path = local_assets.local_path(checkpoint, suffixes=(".safetensors",))
    header = local_assets.read_header(path)
    targets = set()
    for target in lora_targets:
        if not isinstance(target, str) or not target.startswith("diffusion_model.") or not target.endswith(".weight"):
            raise ValueError("H3 LoRAの対象テンソル名が不正です。")
        targets.add(target.removeprefix("diffusion_model.").removesuffix(".weight"))

    def expect(name, shape, dtypes):
        entry = header.get(name)
        if not isinstance(entry, dict) or entry.get("shape") != shape or entry.get("dtype") not in dtypes:
            raise ValueError(f"MiniMax H3本体のテンソルが未対応です: {name}")
        return entry

    floating = {"F16", "BF16", "F32"}
    for name, shape in {
        "audio_patch_proj.weight": [5376, 32], "video_patch_proj.weight": [5376, 96],
        "adaln_t_table": [1025, 8],
    }.items():
        expect(name, shape, floating)
    for block in range(50):
        expect(f"blocks.{block}.adaln_proj.linear.weight", [96768, 8], floating)
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        raise ValueError("H3量子化メタデータが不正です。")
    try:
        quantization = json.loads(metadata["_quantization_metadata"]) if "_quantization_metadata" in metadata else {"layers": {}}
        layers = quantization["layers"]
        if not isinstance(layers, dict):
            raise ValueError("invalid layers")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("H3量子化メタデータが不正です。") from exc

    configs = dict(layers)
    comfy_entries = [(name, entry) for name, entry in header.items() if name.endswith(".comfy_quant")]
    if comfy_entries:
        with path.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            for name, entry in comfy_entries:
                stem = name.removesuffix(".comfy_quant")
                if stem in layers:
                    raise ValueError(f"H3量子化メタデータが重複しています: {stem}")
                if not isinstance(entry, dict) or entry.get("dtype") != "U8":
                    raise ValueError("H3 comfy_quantが不正です。")
                start, end = entry["data_offsets"]
                if end - start <= 0 or end - start > 4096 or entry.get("shape") != [end - start]:
                    raise ValueError("H3 comfy_quantが不正です。")
                stream.seek(8 + size + start)
                value = json.loads(stream.read(end - start))
                if not isinstance(value, dict):
                    raise ValueError("H3 comfy_quantが不正です。")
                configs[stem] = value

    # Match Comfy's descriptor-driven loading for every declared layer, not only
    # the representative blocks: otherwise a float tensor can be cast to INT8.
    formats = set()
    codebooks = []
    for stem, config in configs.items():
        if not isinstance(config, dict):
            raise ValueError("H3量子化メタデータが不正です。")
        params = config.get("params", {})
        if not isinstance(params, dict):
            raise ValueError(f"H3量子化設定が不正です: {stem}")
        weight = header.get(stem + ".weight", {})
        shape = weight.get("shape") if isinstance(weight, dict) else None
        if (not isinstance(shape, list) or len(shape) != 2 or
                any(type(number) is not int or number <= 0 for number in shape)):
            raise ValueError(f"H3量子化重みの形状が不正です: {stem}")
        rows, stored_columns = shape
        expect(stem + ".weight", [rows, stored_columns], {"I8"})
        group = config.get("convrot_groupsize", params.get("convrot_groupsize", 256))
        convrot = config.get("convrot", params.get("convrot", False))
        if type(convrot) is not bool:
            raise ValueError(f"H3 ConvRot設定が不正です: {stem}")
        if convrot or config.get("format") == "asym_w4a8_int8":
            columns = stored_columns * 2 if config.get("format") == "asym_w4a8_int8" else stored_columns
            if (type(group) is not int or group < 4 or group & (group - 1)
                    or (group.bit_length() - 1) % 2 or columns % group):
                raise ValueError(f"H3 ConvRot設定が不正です: {stem}")
        if config.get("format") == "int8_tensorwise":
            scale = header.get(stem + ".weight_scale", {})
            scale_shape = scale.get("shape") if isinstance(scale, dict) else None
            allowed_shapes = [[rows, 1], [], [1], [1, 1]]
            if scale_shape not in allowed_shapes or scale.get("dtype") not in floating:
                raise ValueError(f"H3 INT8重みとscaleが一致しません: {stem}")
            continue
        if config.get("format") != "asym_w4a8_int8":
            raise ValueError(f"H3本体の保存形式が未対応です: {stem}")
        group_size = config.get("group_size", params.get("group_size", 16))
        columns = stored_columns * 2
        if (type(group_size) is not int or group_size < 4
                or (16 % group_size and group_size % 16)
                or columns % 16 or columns % group_size):
            raise ValueError(f"H3 W4A8の形状が不正です: {stem}")
        if "convrot" in config and not convrot:
            raise ValueError(f"H3 W4A8のConvRot設定が不正です: {stem}")
        expect(stem + ".weight_s_rel", [rows, columns // group_size], {"F8_E4M3", "F32", "U8"})
        expect(stem + ".weight_s_channel", [rows], {"F32"})
        if stem + ".weight_codebook" in header:
            codebook = expect(stem + ".weight_codebook", [16], {"F32"})
            if stem in targets:
                codebooks.append((stem, codebook["data_offsets"]))
        formats.add("w4a8")

    if codebooks:
        with path.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            for stem, (start, end) in codebooks:
                if end - start != 64:
                    raise ValueError(f"H3 LoRA対象のcodebookが不正です: {stem}")
                stream.seek(8 + header_size + start)
                values = struct.unpack("<16f", stream.read(64))
                if any(not math.isfinite(value) for value in values) or any(a > b for a, b in zip(values, values[1:], strict=False)):
                    raise ValueError(f"H3 LoRA対象のcodebookが不正です: {stem}")

    for name, tensor in header.items():
        if (name.endswith(".weight") and isinstance(tensor, dict)
                and tensor.get("dtype") not in floating and name.removesuffix(".weight") not in configs):
            raise ValueError(f"H3量子化重みに設定がありません: {name}")

    for block in range(50):
        for layer, rows, columns in (
            ("attn.out_proj", 5376, 7168), ("attn.qkv_proj", 21504, 5376),
            ("mlp.fc1", 28672, 5376), ("mlp.fc2", 5376, 14336),
        ):
            stem = f"blocks.{block}.{layer}"
            weight = header.get(stem + ".weight", {})
            if not isinstance(weight, dict):
                raise ValueError("MiniMax H3本体のテンソルが不正です。")
            if weight.get("dtype") in floating:
                expect(stem + ".weight", [rows, columns], floating)
                continue
            config = configs.get(stem, {})
            fmt = config.get("format")
            if fmt == "int8_tensorwise":
                expect(stem + ".weight", [rows, columns], {"I8"})
            elif fmt == "asym_w4a8_int8":
                expect(stem + ".weight", [rows, columns // 2], {"I8"})
                formats.add("w4a8")
            else:
                raise ValueError(f"H3本体の保存形式が未対応です: {stem}")
    return "w4a8" if formats else "base"


def h3_compatible(path: Path) -> bool:
    try:
        return bool(h3_variant(str(path)))
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


def snapshot_checkpoint(checkpoint: str, *, directory: bool = False, check_cancel=None) -> dict | None:
    """Freeze the selected content before a job starts waiting for the GPU."""
    if not checkpoint:
        return None
    path = local_assets.local_path(checkpoint, directory=directory)
    before = _snapshot_versions(path, directory)
    if directory and check_cancel is not None:
        raise ValueError("フォルダーの中断可能な検証には対応していません。")
    if directory:
        identity = local_assets.identity(path)
    elif check_cancel is None:
        identity = local_assets.file_identity(path)
    else:
        identity = local_assets.file_identity(path, check_cancel=check_cancel)
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
