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
    from .capabilities import validate_sampling

    validate_sampling(values)
    if values.get("operation", "generate") != "generate":
        raise ValueError("Outpaintは画像生成で使用してください。")
    if values.get("annotation_layers") or values.get("preserve_unmasked", False):
        raise ValueError("Outpaintの編集範囲はキャンバスで指定します。通常編集の注釈・範囲外固定は併用できません。")
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


def outside_mask(plan):
    mask = Image.new("L", plan.size, 255)
    mask.paste(0, (plan.left, plan.top, plan.left + plan.source_width, plan.top + plan.source_height))
    return mask


def snapshot(payload: dict, directory: Path) -> None:
    source = payload["clean_input_images"][0]
    original, canvas, plan = source_plan(source, payload)
    original.save(source, format="PNG")
    reference = directory / "outpaint-reference.png"
    canvas.save(reference, format="PNG")
    payload["input_images"] = [str(reference)]
    payload["outpaint"] = {"source": source, "plan": asdict(plan), "feather": payload["outpaint_feather"]}
    control = payload.get("control_image")
    if control:
        if Path(control).resolve() != (directory / "control-source.png").resolve() or Path(control).is_symlink():
            raise ValueError("Outpaintの制御画像はジョブ内へコピーしてから使用してください。")
        with Image.open(control) as image:
            image = image.convert("RGB")
            if image.size == original.size:
                expanded = Image.new("RGB", plan.size, "black")
                expanded.paste(image, (plan.left, plan.top))
                image = expanded
            elif image.size != plan.size:
                raise ValueError("Outpaintの制御画像は元画像または完成キャンバスと同じ寸法で指定してください。")
            image.save(control, format="PNG")
    if payload.get("control_inpaint"):
        mask = outside_mask(plan)
        mask_path = directory / "edit-mask.png"
        mask.save(mask_path)
        payload["edit_mask"] = {
            "original_path": str(reference),
            "mask_path": str(mask_path),
            "reference": 0,
            "feather": 0,
        }
        payload["edit_mask_path"] = str(mask_path)
        payload["edit_mask_reference"] = 0


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
    control = values.get("control_image")
    if control:
        path = Path(control)
        if path.resolve() != (job / "control-source.png").resolve() or path.is_symlink():
            raise ValueError("Outpaintの制御画像はジョブ内の保存画像を使用してください。")
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != plan.size:
                raise ValueError("Outpaintの制御画像が完成キャンバスの寸法と一致しません。")
    if values.get("control_inpaint"):
        info = values.get("edit_mask", {})
        mask_path = job / "edit-mask.png"
        if (
            info.get("original_path") != str(reference)
            or info.get("mask_path") != str(mask_path)
            or mask_path.is_symlink()
        ):
            raise ValueError("OutpaintのInpaintマスクはジョブ内の参照・外側マスクを使用してください。")
        with Image.open(mask_path) as mask:
            if mask.mode != "L" or mask.size != plan.size or mask.tobytes() != outside_mask(plan).tobytes():
                raise ValueError("OutpaintのInpaintマスクが元画像・余白と一致しません。")


def finish(image: Image.Image, values: dict) -> Image.Image:
    data = values["outpaint"]
    with Image.open(data["source"]) as source:
        return stitch(source, image, Plan(**data["plan"]), data["feather"]).convert("RGBA")
