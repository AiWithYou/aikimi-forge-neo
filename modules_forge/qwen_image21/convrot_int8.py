"""Inspect Qwen Image 2.1 INT8 ConvRot checkpoints without importing CUDA libraries."""

from __future__ import annotations

import json
import struct
from pathlib import Path

from modules_forge import local_assets

FORMAT = "int8_convrot"
SETTINGS = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
LAYER_SHAPES = {
    "attn.to_q": (4096, 4096),
    "attn.to_k": (4096, 4096),
    "attn.to_v": (4096, 4096),
    "attn.to_out.0": (4096, 4096),
    "img_mlp.gate_layer": (12288, 4096),
    "img_mlp.proj": (12288, 4096),
    "img_mlp.out": (4096, 12288),
}


def normalize_keys(values: dict) -> dict:
    normalized = {}
    for key, value in values.items():
        if key == "__metadata__":
            continue
        name = key.removeprefix("model.diffusion_model.").removeprefix("diffusion_model.")
        if name in normalized:
            raise ValueError(f"本体のテンソル名が重複しています: {name}")
        normalized[name] = value
    return normalized


def validate_settings(settings: dict, name: str) -> None:
    if (
        settings != SETTINGS
        or type(settings.get("convrot")) is not bool
        or type(settings.get("convrot_groupsize")) is not int
    ):
        raise ValueError(f"未対応のConvRot量子化設定です: {name}")


def validate_header(header: dict) -> dict:
    tensors = normalize_keys(header)
    stems = {f"transformer_blocks.{index}.{name}" for index in range(32) for name in LAYER_SHAPES}
    if {key.removesuffix(".comfy_quant") for key in tensors if key.endswith(".comfy_quant")} != stems:
        raise ValueError("Qwen Image 2.1のINT8 ConvRotには32ブロック・224線形層が必要です。")
    consumed = set()
    for stem in stems:
        shape = LAYER_SHAPES[stem.split(".", 2)[2]]
        weight, scale, descriptor = (
            tensors.get(stem + suffix, {}) for suffix in (".weight", ".weight_scale", ".comfy_quant")
        )
        if weight.get("dtype") != "I8" or weight.get("shape") != list(shape):
            raise ValueError(f"INT8 ConvRot重みの寸法・精度が一致しません: {stem}")
        if scale.get("dtype") != "F32" or scale.get("shape") != [shape[0], 1]:
            raise ValueError(f"INT8 ConvRotのFP32スケールが一致しません: {stem}")
        descriptor_shape = descriptor.get("shape", [])
        if descriptor.get("dtype") != "U8" or len(descriptor_shape) != 1 or not 0 < descriptor_shape[0] <= 4096:
            raise ValueError(f"INT8 ConvRotの設定テンソルが不正です: {stem}")
        consumed.update(stem + suffix for suffix in (".weight", ".weight_scale", ".comfy_quant"))
    for key, item in tensors.items():
        if key not in consumed and (item.get("dtype") not in {"BF16", "F16", "F32"} or key.endswith(".weight_scale")):
            raise ValueError(f"未対応の事前量子化テンソルです: {key}")
    return {"format": FORMAT, "quantized_linears": len(stems), "convrot_groupsize": 256}


def inspect_checkpoint(path: Path, *, header: dict | None = None) -> dict:
    header = local_assets.read_header(path) if header is None else header
    info = validate_header(header)
    with Path(path).open("rb") as stream:
        offset = 8 + struct.unpack("<Q", stream.read(8))[0]
        for name, item in header.items():
            if not name.endswith(".comfy_quant"):
                continue
            start, end = item["data_offsets"]
            stream.seek(offset + start)
            raw = stream.read(end - start)
            if len(raw) != item["shape"][0]:
                raise ValueError(f"ConvRot設定のデータが不足しています: {name}")
            try:
                settings = json.loads(raw)
            except (ValueError, UnicodeDecodeError) as exc:
                raise ValueError(f"ConvRot設定のJSONが不正です: {name}") from exc
            if not isinstance(settings, dict):
                raise ValueError(f"ConvRot設定が不正です: {name}")
            validate_settings(settings, name)
    return info
