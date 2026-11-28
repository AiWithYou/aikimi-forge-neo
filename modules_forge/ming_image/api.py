"""Authenticated Ming generation jobs return public metadata and PNG bytes."""

from __future__ import annotations

import base64
from pathlib import Path

from starlette.requests import Request

from modules_forge import gpu_fault

from .core import png_metadata
from .references import decode_uploads
from .service import SERVICE


def public_result(value):
    allowed = {"status", "id", "stage", "current", "total", "error"}
    result = {key: item for key, item in value.items() if key in allowed}
    result.update(png_metadata(value))
    if result.get("error"):
        result["error"] = "生成に失敗しました。Forge Neoのログを確認してください。"
        if value.get("error_code") == "gpu_stopped":
            result.update(error_code="gpu_stopped", error=gpu_fault.MESSAGE)
    if value.get("status") == "complete":
        result["image"] = base64.b64encode(Path(value["image_path"]).read_bytes()).decode("ascii")
    return result


def register_api(app):
    from fastapi import Depends, HTTPException

    from modules.aikimi_security.auth import require_api_auth

    def authenticate(request: Request):
        require_api_auth(request)

    def start(value: dict):
        try:
            if "images" in value:
                value = {**value, "images": decode_uploads(value["images"])}
            return {"id": SERVICE.start(value), "status": "running"}
        except gpu_fault.GPUFaultError as exc:
            raise HTTPException(503, detail=gpu_fault.MESSAGE) from exc
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
        ("/sdapi/v1/ming-image/jobs", start, "POST"),
        ("/sdapi/v1/ming-image/jobs/{identifier}", poll, "GET"),
        ("/sdapi/v1/ming-image/jobs/{identifier}/cancel", cancel, "POST"),
    ):
        app.add_api_route(path, endpoint, methods=[method], dependencies=[Depends(authenticate)])
