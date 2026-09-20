"""Validate visual region guides and snapshot them without altering clean inputs."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageChops, ImageOps

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
