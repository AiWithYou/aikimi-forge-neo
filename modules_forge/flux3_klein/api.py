"""Authenticated job API using decoded uploads, never remote URLs or paths."""

from __future__ import annotations

import base64
import binascii
import io
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.requests import Request

from .core import Flux3Error
from .service import SERVICE


def decode_references(values):
    if not isinstance(values, list) or len(values) > 10:
        raise Flux3Error("参照画像は配列で最大10枚です。")
    images = []
    for value in values:
        if not isinstance(value, str) or len(value) > 64_000_000:
            raise Flux3Error("参照画像をbase64形式で送信してください。")
        if value.startswith("data:image/") and ";base64," in value:
            value = value.split(";base64,", 1)[1]
        try:
            raw = base64.b64decode(value, validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                if image.width * image.height > 40_000_000:
                    raise Flux3Error("参照画像は1枚40メガピクセルまでです。")
                image.load()
                images.append(ImageOps.exif_transpose(image).copy())
        except (binascii.Error, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise Flux3Error("参照画像をbase64形式の画像で送信してください。") from exc
    return images


def public_result(value):
    allowed = {
        "status",
        "id",
        "stage",
        "current",
        "total",
        "error",
        "model",
        "precision",
        "width",
        "height",
        "prompt",
        "input_prompt",
        "steps",
        "cfg",
        "seed",
        "upsampled",
        "boxes",
        "reference_sha256",
        "reference_hashes",
        "references",
        "model_revision",
        "comfy_revision",
        "sampling_width",
        "sampling_height",
    }
    result = {key: item for key, item in value.items() if key in allowed}
    if value.get("status") == "complete":
        path = Path(value["image_path"])
        result["image"] = base64.b64encode(path.read_bytes()).decode("ascii")
    return result


def register_api(app):
    from fastapi import Depends, HTTPException

    from modules.aikimi_security.auth import require_api_auth

    def authenticate(request: Request):
        require_api_auth(request)

    def start(value: dict):
        try:
            value = dict(value)
            value["images"] = decode_references(value.get("images", []))
            return {"id": SERVICE.start(value), "status": "running"}
        except ValueError as exc:
            raise HTTPException(422, detail=str(exc)) from exc

    def poll(identifier: str):
        result = SERVICE.poll(identifier)
        if result["status"] == "idle":
            raise HTTPException(404, detail="ジョブが見つかりません。")
        return public_result(result)

    def cancel(identifier: str):
        return {"message": SERVICE.cancel(identifier)}

    for path, endpoint, method in (
        ("/sdapi/v1/flux3/jobs", start, "POST"),
        ("/sdapi/v1/flux3/jobs/{identifier}", poll, "GET"),
        ("/sdapi/v1/flux3/jobs/{identifier}/cancel", cancel, "POST"),
    ):
        app.add_api_route(path, endpoint, methods=[method], dependencies=[Depends(authenticate)])
