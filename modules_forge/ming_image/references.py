"""Bounded image uploads and immutable, job-local reference snapshots."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import re

from .core import MingImageError, safe_model_root


def validate_inputs(values, *, internal=False):
    if not isinstance(values, list) or len(values) > 8:
        raise MingImageError("参照画像は入力順に最大8枚です。")
    for index, image in enumerate(values):
        if internal:
            if image != f"reference-{index}.png":
                raise MingImageError("参照画像の保存名が不正です。")
        else:
            from PIL import Image

            if not isinstance(image, Image.Image) or min(image.size) < 16 or image.width * image.height > 40_000_000:
                raise MingImageError("参照画像は各辺16ピクセル以上、1枚40メガピクセル以内で入力してください。")
    return list(values)


def validate_hashes(images, hashes):
    if (
        not isinstance(hashes, list)
        or len(hashes) != len(images)
        or any(not isinstance(item, str) or re.fullmatch(r"[0-9a-f]{64}", item) is None for item in hashes)
    ):
        raise MingImageError("参照画像のSHA-256が一致しません。")
    return list(hashes)


def decode_uploads(values):
    from PIL import Image, ImageOps, UnidentifiedImageError

    if not isinstance(values, list) or len(values) > 8:
        raise MingImageError("参照画像はbase64配列で最大8枚です。")
    images = []
    for value in values:
        if not isinstance(value, str) or len(value) > 64_000_000:
            raise MingImageError("参照画像をbase64形式で送信してください。")
        if value.startswith("data:image/") and ";base64," in value:
            value = value.split(";base64,", 1)[1]
        try:
            raw = base64.b64decode(value, validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                validate_inputs([image])
                image.load()
                images.append(ImageOps.exif_transpose(image).copy())
        except MingImageError:
            raise
        except (ValueError, UnidentifiedImageError, OSError, Image.DecompressionBombError, binascii.Error) as error:
            raise MingImageError("参照画像は40メガピクセル以内のbase64画像で送信してください。") from error
    return images


def snapshot(directory, request):
    from PIL import ImageOps

    images = validate_inputs(request.get("images", []))
    directory = safe_model_root(directory)
    names, hashes = [], []
    for index, source in enumerate(images):
        image = ImageOps.exif_transpose(source).convert("RGBA")
        image.info.clear()
        name = f"reference-{index}.png"
        path = safe_model_root(directory / name)
        image.save(path, format="PNG")
        names.append(name)
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    request.update(images=names, reference_sha256=hashes)


def load(directory, request):
    from PIL import Image

    names = validate_inputs(request.get("images", []), internal=True)
    hashes = validate_hashes(names, request.get("reference_sha256", []))
    directory = safe_model_root(directory)
    images = []
    for name, digest in zip(names, hashes, strict=True):
        path = safe_model_root(directory / name)
        if (
            not path.is_relative_to(directory)
            or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise MingImageError("参照画像が変更されたか、保存条件と一致しません。")
        with Image.open(path) as image:
            validate_inputs([image])
            image.load()
            images.append(image.convert("RGBA"))
    return images
