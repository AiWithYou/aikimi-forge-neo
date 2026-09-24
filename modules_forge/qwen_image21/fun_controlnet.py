"""Pinned Qwen Image 2.1 Fun ControlNet Union INT8 checkpoint contract."""

from __future__ import annotations

import json
import re
from pathlib import Path

from modules_forge.minimax_h3_union2_vae import read_header, validate_tensor_table

REPOSITORY = "Kijai/QwenImage_experimental"
REVISION = "04987755e10002ff33e4ea307a811487dddd79d9"
FILENAME = "model_patches/qwen_image_2.1_fun_controlnet_union_int8_convrot.safetensors"
SIZE = 3_779_298_944
SHA256 = "07aa961570ac0e03d4ca936aecd76854d077a33cde69b5092399afba01b3715d"
SOURCE = f"https://huggingface.co/{REPOSITORY}/blob/{REVISION}/{FILENAME}"
OFFICIAL = "https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union"


def checkpoint_path(runtime: Path) -> Path:
    return Path(runtime) / "controlnet" / FILENAME


def validate_header(header: dict) -> dict:
    validate_tensor_table(header)
    if header.get("control_img_in.weight", {}).get("shape") != [4096, 129]:
        raise ValueError("Qwen 2.1用の129ch制御入力がありません。")
    if header.get("control_img_in.weight", {}).get("dtype") != "BF16":
        raise ValueError("制御入力の精度が一致しません。")
    indices = {
        int(match.group(1))
        for key in header
        if (match := re.match(r"control_blocks\.(\d+)\.", key))
    }
    if indices != set(range(16)):
        raise ValueError("16個のControlNetブロックが必要です。")
    quantized = 0
    for index in range(16):
        prefix = f"control_blocks.{index}."
        names = ["after_proj", "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "img_mlp.gate_up", "img_mlp.out"]
        if index == 0:
            names.append("before_proj")
        for name in names:
            stem = prefix + name
            weight = header.get(stem + ".weight", {})
            scale = header.get(stem + ".weight_scale", {})
            settings = header.get(stem + ".comfy_quant", {})
            if weight.get("dtype") != "I8" or len(weight.get("shape", [])) != 2:
                raise ValueError(f"INT8重みが不足しています: {stem}")
            if scale.get("dtype") != "F32" or scale.get("shape") != [weight["shape"][0], 1]:
                raise ValueError(f"INT8スケールが不足しています: {stem}")
            if settings.get("dtype") != "U8" or settings.get("shape") != [72]:
                raise ValueError(f"ConvRot設定が不足しています: {stem}")
            quantized += 1
        if header.get(prefix + "attn.norm_q.weight", {}).get("shape") != [128]:
            raise ValueError(f"Attention normが不足しています: {index}")
        if header.get(prefix + "img_mlp.gate_up.weight", {}).get("shape") != [24576, 4096]:
            raise ValueError(f"MLPの形状が一致しません: {index}")
    return {"blocks": 16, "injection_layers": list(range(0, 32, 2)), "quantized_linears": quantized}


def inspect_checkpoint(path: Path) -> dict:
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size != SIZE:
        raise ValueError("Fun ControlNet INT8モデルがありません。導入コマンドを実行してください。")
    with path.open("rb") as stream:
        header, offset = read_header(stream)
    validate_tensor_table(header, path.stat().st_size - offset)
    return validate_header(header)


def installed(runtime: Path) -> dict:
    path = checkpoint_path(runtime)
    details = inspect_checkpoint(path)
    record = path.with_suffix(".json")
    if record.is_file():
        receipt = json.loads(record.read_text(encoding="utf-8"))
        if receipt.get("sha256") != SHA256 or receipt.get("revision") != REVISION:
            raise ValueError("Fun ControlNetの導入記録が重みと一致しません。")
    return {"path": str(path.resolve()), "repository": REPOSITORY, "revision": REVISION, "sha256": SHA256, **details}


def status(runtime: Path) -> str:
    try:
        installed(runtime)
    except (OSError, ValueError, json.JSONDecodeError):
        return "Fun ControlNet · INT8: 未導入。専用の導入コマンドを実行してください。"
    return "Fun ControlNet · INT8: 導入済み（16ブロック）。"
