"""Validate visual region guides and snapshot them without altering clean inputs."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter, ImageOps

from .core import QwenImage21Error, inside, validate_images

MAX_ANNOTATION_LAYERS = 8


def reference_index(paths: list[str] | tuple[str, ...], target_path: str) -> int:
    """Follow Gradio's byte-identical cache copies without matching other images."""
    target = Path(target_path).resolve()
    candidates = [Path(path).resolve() for path in paths]
    matches = [index for index, path in enumerate(candidates) if path == target]
    if not matches:
        try:
            size = target.stat().st_size
            if size > 64 * 1024 * 1024:
                raise OSError("reference exceeds upload limit")
            digest = hashlib.sha256(target.read_bytes()).digest()
            matches = [
                index
                for index, path in enumerate(candidates)
                if path.is_file()
                and path.stat().st_size == size
                and hashlib.sha256(path.read_bytes()).digest() == digest
            ]
        except OSError:
            matches = []
    if len(matches) != 1:
        raise QwenImage21Error("描画対象が削除されたか、同じ画像が複数あります。参照画像から選択し直してください。")
    return matches[0]


def annotation_preview(path: str) -> Image.Image:
    """Use an opaque drawing background; keep original RGBA for inference."""
    with _rgba(path) as original:
        preview = Image.new("RGBA", original.size, (255, 255, 255, 255))
        preview.alpha_composite(original)
        return preview


def _rgba(path: str, *, layer: bool = False) -> Image.Image:
    try:
        with Image.open(path) as source:
            source.load()
            oriented = ImageOps.exif_transpose(source)
            if layer and "A" not in oriented.getbands() and "transparency" not in oriented.info:
                raise QwenImage21Error(
                    "描画レイヤーにはアルファチャンネルが必要です。描画対象を読み込み直してください。"
                )
            return oriented.convert("RGBA")
    except (OSError, ValueError) as exc:
        if isinstance(exc, QwenImage21Error):
            raise
        raise QwenImage21Error(f"描画画像を読み込めません: {exc}") from exc


def _layers(paths) -> tuple[tuple[str, ...], list[tuple[int, int]], tuple[int, int, int, int] | None]:
    if not isinstance(paths, (tuple, list)) or len(paths) > MAX_ANNOTATION_LAYERS:
        raise QwenImage21Error(f"描画レイヤーは最大{MAX_ANNOTATION_LAYERS}枚です。")
    checked = validate_images(paths)
    sizes = []
    bounds = None
    for path in checked:
        with _rgba(path, layer=True) as image:
            sizes.append(image.size)
            box = image.getchannel("A").getbbox()
            if box is not None:
                bounds = (
                    box
                    if bounds is None
                    else (
                        min(bounds[0], box[0]),
                        min(bounds[1], box[1]),
                        max(bounds[2], box[2]),
                        max(bounds[3], box[3]),
                    )
                )
    return checked, sizes, bounds


def _same_visible_pixels(left: Image.Image, right: Image.Image) -> bool:
    if left.size != right.size:
        return False
    if ImageChops.difference(left.getchannel("A"), right.getchannel("A")).getbbox() is not None:
        return False
    # Browser canvases may zero the otherwise invisible RGB values of pixels
    # with alpha=0. Those differences do not make the selected source stale.
    difference = ImageChops.difference(left.convert("RGB"), right.convert("RGB"))
    invisible = left.getchannel("A").point(lambda alpha: 255 if alpha == 0 else 0)
    difference.paste((0, 0, 0), mask=invisible)
    return difference.getbbox() is None


def validate_annotation_layers(reference_path: str, layer_paths) -> tuple[tuple[str, ...], tuple[int, int, int, int]]:
    """Check a resolved request before it acquires the GPU lease."""
    reference = validate_images([reference_path])[0]
    checked, sizes, bounds = _layers(layer_paths)
    if not checked or bounds is None:
        raise QwenImage21Error("描画がありません。対象を囲むか、描画指定を解除してください。")
    with _rgba(reference) as image:
        if any(size != image.size for size in sizes):
            raise QwenImage21Error("描画レイヤーと参照画像のサイズが一致しません。対象を読み込み直してください。")
    return checked, bounds


def resolve_annotation(
    gallery_paths: list[str], target_path: str | None, editor_value: dict | None
) -> tuple[int, tuple[str, ...]]:
    """Bind editor strokes to the current reference list, ignoring its composite."""
    if not target_path or editor_value is None:
        return -1, ()
    if not isinstance(editor_value, dict):
        raise QwenImage21Error("描画データの形式が不正です。対象を読み込み直してください。")
    raw_layers = editor_value.get("layers")
    if raw_layers is None:
        raw_layers = []
    checked, sizes, bounds = _layers(raw_layers)
    if bounds is None:
        return -1, ()
    paths = validate_images(gallery_paths)
    if not isinstance(target_path, (str, Path)):
        raise QwenImage21Error("描画対象が不正です。参照画像から選択し直してください。")
    index = reference_index(paths, str(target_path))
    target = paths[index]
    background_path = editor_value.get("background")
    if not isinstance(background_path, (str, Path)):
        raise QwenImage21Error("描画の背景画像がありません。対象を読み込み直してください。")
    background = validate_images([background_path])[0]
    with _rgba(target) as original, _rgba(background) as editor_background:
        if not _same_visible_pixels(original, editor_background):
            # Gradio's browser canvas can flatten alpha while exporting its
            # background. Compare to the deliberately opaque preview instead
            # of accepting arbitrary changes to the clean reference pixels.
            with annotation_preview(target) as preview:
                if not _same_visible_pixels(preview, editor_background):
                    raise QwenImage21Error("描画対象の画像が変わりました。現在の参照画像を読み込み直してください。")
        if any(size != original.size for size in sizes):
            raise QwenImage21Error("描画レイヤーと参照画像のサイズが一致しません。対象を読み込み直してください。")
    return index, checked


def snapshot_annotation(
    paths: list[str], index: int, layers: tuple[str, ...], jobdir: Path
) -> tuple[list[str], str, dict]:
    """Save clean inputs and visual guides separately; guides are not edit masks."""
    if index == -1 and not layers:
        return list(paths), "", {}
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(paths):
        raise QwenImage21Error("描画対象の参照番号が不正です。")
    directory = Path(jobdir)
    original = inside(directory, Path(paths[index]))
    checked, bounds = validate_annotation_layers(str(original), layers)
    number = index + 1
    annotated = inside(directory, directory / f"reference-{number:02d}-annotated.png")
    if annotated in [Path(path).resolve() for path in paths]:
        raise QwenImage21Error("描画付き画像と元の参照画像の保存先が重複しています。")
    saved_layers = []
    with _rgba(str(original)) as composite:
        for layer_number, path in enumerate(checked, 1):
            destination = inside(directory, directory / f"annotation-{number:02d}-layer-{layer_number:02d}.png")
            if destination in [Path(value).resolve() for value in paths]:
                raise QwenImage21Error("描画レイヤーと元の参照画像の保存先が重複しています。")
            with _rgba(path, layer=True) as layer:
                layer.save(destination, format="PNG")
                composite.alpha_composite(layer)
            saved_layers.append(str(destination))
        composite.save(annotated, format="PNG")
    model_paths = list(paths)
    model_paths[index] = str(annotated)
    instruction = (
        f"\n\nImage {number} contains colored annotation marks indicating the region to edit. "
        "Apply the requested edit to the indicated region and remove the annotation marks from the final image. "
        "Preserve the other content as much as possible."
    )
    info = {
        "reference_index": number,
        "original_path": str(original),
        "annotated_path": str(annotated),
        "layer_paths": saved_layers,
        "bounds": list(bounds),
    }
    return model_paths, instruction, info


def _mask(path: str) -> Image.Image:
    """White edits, black preserves; transparent upload pixels never edit."""
    checked = validate_images([path])[0]
    with Image.open(checked) as source:
        source.load()
        oriented = ImageOps.exif_transpose(source)
        mask = oriented.convert("L")
        if "A" in oriented.getbands() or "transparency" in oriented.info:
            mask = ImageChops.multiply(mask, oriented.convert("RGBA").getchannel("A"))
        return mask


def validate_edit_mask(reference_path: str, mask_path: str, *, output_size=None) -> tuple[str, tuple[int, ...]]:
    """An explicit mask has the clean reference's exact oriented coordinates."""
    reference = validate_images([reference_path])[0]
    checked = validate_images([mask_path])[0]
    with _rgba(reference) as original, _mask(checked) as mask:
        if mask.size != original.size:
            raise QwenImage21Error("マスクと編集元のサイズが一致しません。同じ大きさのマスクを指定してください。")
        if output_size is not None and original.size != tuple(output_size):
            raise QwenImage21Error(
                f"範囲外を固定するには編集元と同じ出力サイズ（{original.width}×{original.height}）を指定してください。"
            )
        bounds = mask.getbbox()
        if bounds is None:
            raise QwenImage21Error(
                "編集マスクが空です。変更する範囲を塗るか、白い編集領域のあるマスクを指定してください。"
            )
    return checked, bounds


def validate_mask_feather(value) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 64
    ):
        raise QwenImage21Error("マスク境界のぼかしは0〜64 pxで指定してください。")
    return float(value)


def snapshot_edit_mask(paths: list[str], index: int, mask_path: str, feather: float, jobdir: Path) -> dict:
    """Archive a separate, explicit edit mask; annotation guides remain guides."""
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(paths):
        raise QwenImage21Error("マスクの編集元がありません。参照画像から選択し直してください。")
    directory = Path(jobdir)
    original = inside(directory, Path(paths[index]))
    checked, bounds = validate_edit_mask(str(original), mask_path)
    destination = inside(directory, directory / "edit-mask.png")
    if destination in [Path(value).resolve() for value in paths]:
        raise QwenImage21Error("マスクと編集元の保存先が重複しています。")
    feather = validate_mask_feather(feather)
    with _mask(checked) as mask:
        mask.save(destination, format="PNG")
        size = list(mask.size)
    return {
        "reference_index": index + 1,
        "original_path": str(original),
        "mask_path": str(destination),
        "feather": feather,
        "bounds": list(bounds),
        "size": size,
    }


def composite_preserving_outside(
    original: Image.Image, generated: Image.Image, mask: Image.Image, feather=0
) -> Image.Image:
    """Blend premultiplied RGBA inside the mask without resizing any image."""
    feather = validate_mask_feather(feather)
    if original.size != generated.size or original.size != mask.size:
        raise QwenImage21Error("編集元・生成画像・マスクのサイズが一致しないため、範囲外を固定できません。")
    coverage = mask.convert("L")
    if coverage.getbbox() is None:
        raise QwenImage21Error("編集マスクが空です。")
    if feather:
        # Clamp to the original mask, including holes, so Gaussian tails can
        # never modify an outside pixel or its otherwise invisible RGB bytes.
        coverage = ImageChops.darker(coverage, coverage.filter(ImageFilter.GaussianBlur(feather)))
    original_rgba, generated_rgba = original.convert("RGBA"), generated.convert("RGBA")
    result = Image.composite(generated_rgba, original_rgba, coverage)
    transition = coverage.point(lambda value: 255 if 0 < value < 255 else 0).getbbox()
    if transition is None:
        return result

    import numpy as np

    left, top, right, bottom = transition
    # Work in small strips so a soft mask need not allocate several full-size
    # float images. Integer numerators retain low-alpha colors without the
    # rounding losses of Pillow's 8-bit premultiplied RGBa conversion.
    for row in range(top, bottom, 128):
        box = (left, row, right, min(row + 128, bottom))
        weights = np.asarray(coverage.crop(box), dtype=np.uint32)
        source = np.asarray(original_rgba.crop(box), dtype=np.uint32)
        target = np.asarray(generated_rgba.crop(box), dtype=np.uint32)
        source_alpha = source[..., 3] * (255 - weights)
        target_alpha = target[..., 3] * weights
        total_alpha = source_alpha + target_alpha
        divisor = np.maximum(total_alpha, 1)[..., None]
        numerator = source[..., :3] * source_alpha[..., None] + target[..., :3] * target_alpha[..., None]
        colors = ((numerator + divisor // 2) // divisor).astype(np.uint8)
        mixed = (weights > 0) & (weights < 255) & (total_alpha > 0)
        pixels = np.array(result.crop(box))
        np.copyto(pixels[..., :3], colors, where=mixed[..., None])
        # Alpha is already linearly interpolated correctly by Image.composite.
        # At coverage 0/255, keep every original/target byte, including hidden RGB.
        result.paste(Image.fromarray(pixels), (left, row))
    return result


def save_preserved_output(request: dict, generated: Image.Image, jobdir: Path) -> dict:
    """Worker hook after output.png is saved; keep both raw and fixed results."""
    if not request.get("preserve_unmasked", False):
        return {}
    directory = Path(jobdir)
    info = request.get("edit_mask")
    if not isinstance(info, dict) or not info.get("original_path") or not info.get("mask_path"):
        raise QwenImage21Error("範囲外固定用の編集元とマスクが保存されていません。")
    original_path = inside(directory, Path(info["original_path"]))
    mask_path = inside(directory, Path(info["mask_path"]))
    validate_edit_mask(str(original_path), str(mask_path), output_size=generated.size)
    feather = validate_mask_feather(info.get("feather", 0))
    output = inside(directory, directory / "output-preserved.png")
    partial = inside(directory, directory / "output-preserved.png.part")
    with _rgba(str(original_path)) as original, _mask(str(mask_path)) as mask:
        with composite_preserving_outside(original, generated, mask, feather) as preserved:
            preserved.save(partial, format="PNG")
    os.replace(partial, output)
    raw = inside(directory, directory / "output.png")
    return {
        "original_output_path": str(raw),
        "preserved_output_path": str(output),
        "output_paths": [str(output), str(raw)],
        "preservation": {**info, "applied": True, "outside_pixels": "original_rgba", "feather_direction": "inside"},
    }
