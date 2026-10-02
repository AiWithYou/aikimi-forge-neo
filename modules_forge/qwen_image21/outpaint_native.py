"""Immutable job snapshots and validation for in-app Qwen outpainting."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from PIL import Image

from .outpaint import Plan, _integer, prepare, stitch


def validate_options(values: dict) -> None:
    version = values.get("outpaint_version", "")
    if not isinstance(version, str) or version not in {"", "none", "v1", "v2"}:
        raise ValueError("Outpaint LoRAはなし・v1・v2を選んでください。")
    if not version:
        return
    if values.get("operation", "generate") != "generate" or values.get("precision", "int8") not in {
        "int8",
        "base_q4_k_m",
    }:
        raise ValueError("Outpaintは通常版QwenのQ4_K_MまたはINT8を使用します。")
    if (
        values.get("fun_acc", False)
        or values.get("sparse_mode", "off") != "off"
        or values.get("control_kind", "off") != "off"
        or values.get("control_image")
        or values.get("annotation_layers")
        or values.get("preserve_unmasked", False)
        or values.get("rewrite_prompt", False)
        or values.get("rewrite_edit_prompt", False)
        or values.get("transparent", False)
    ):
        raise ValueError("Outpaintでは追加の制御・高速化・プロンプト書き換えをOFFにしてください。")
    if len(values.get("input_images", ())) != 1:
        raise ValueError("Outpaintの元画像を1枚指定してください。")
    pads = values.get("outpaint_margins")
    if not isinstance(pads, (tuple, list)) or len(pads) != 4:
        raise ValueError("Outpaintの余白を左・上・右・下の順で指定してください。")
    for value in pads:
        _integer(value, "余白", 0, 4096)
    _integer(values.get("outpaint_feather", 32), "境界幅", 0, 128)


def source_plan(source: str, values: dict):
    with Image.open(source) as image:
        original, canvas, plan = prepare(image, *values["outpaint_margins"])
    if plan.size != (values["width"], values["height"]):
        raise ValueError("Outpaintの完成サイズと元画像・余白が一致しません。")
    return original, canvas, plan


def snapshot(payload: dict, directory: Path) -> None:
    source = payload["clean_input_images"][0]
    original, canvas, plan = source_plan(source, payload)
    original.save(source, format="PNG")
    reference = directory / "outpaint-reference.png"
    canvas.save(reference, format="PNG")
    payload["input_images"] = [str(reference)]
    payload["outpaint"] = {"source": source, "plan": asdict(plan), "feather": payload["outpaint_feather"]}


def validate_snapshot(values: dict, job: Path) -> None:
    validate_options(values)
    data = values.get("outpaint")
    if not isinstance(data, dict) or set(data) != {"source", "plan", "feather"}:
        raise ValueError("Outpaintの元画像・余白の保存情報が不正です。")
    source = Path(data["source"])
    reference = Path(values["input_images"][0])
    if (
        source.resolve() != (job / "reference-01.png").resolve()
        or reference.resolve() != (job / "outpaint-reference.png").resolve()
    ):
        raise ValueError("Outpaintの画像はジョブ内の保存画像を使用してください。")
    if source.is_symlink() or reference.is_symlink():
        raise ValueError("Outpaintの保存画像にシンボリックリンクは使用できません。")
    _, expected, plan = source_plan(str(source), values)
    if data["plan"] != asdict(plan) or data["feather"] != values["outpaint_feather"]:
        raise ValueError("Outpaintの保存情報が要求と一致しません。")
    with Image.open(reference) as image:
        if image.mode != "RGB" or image.size != plan.size or image.tobytes() != expected.tobytes():
            raise ValueError("Outpaintの参照画像が元画像・余白と一致しません。")


def finish(image: Image.Image, values: dict) -> Image.Image:
    data = values["outpaint"]
    with Image.open(data["source"]) as source:
        return stitch(source, image, Plan(**data["plan"]), data["feather"]).convert("RGBA")
