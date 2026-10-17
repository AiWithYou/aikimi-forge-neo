"""Authenticated MOSS generation jobs return public metadata and PNG bytes."""

from __future__ import annotations

import base64
from pathlib import Path

from starlette.requests import Request

from .service import SERVICE


def public_result(value):
    allowed = {
        "status",
        "id",
        "stage",
        "current",
        "total",
        "error",
        "prompt",
        "negative_prompt",
        "width",
        "height",
        "steps",
        "cfg",
        "shift",
        "seed",
        "skip_layer",
        "model_id",
        "model_revision",
        "elapsed_seconds",
        "actual_steps",
        "scheduler",
        "precision",
        "vae_precision",
        "text_encoder_precision",
        "transformer_precision",
        "initial_noise_precision",
        "cfg_precision",
        "latent_update_precision",
        "token_limit",
        "prompt_tokens",
        "negative_prompt_tokens",
        "prompt_truncated",
        "generator_device",
        "text_precision",
        "loras",
        "lora_identities",
        "lora",
        "lora_applied_targets",
        "comfy_revision",
        "runtime_backend",
        "weight_sources",
        "sampler",
        "comfy_kitchen_version",
        "dit_storage_precision",
        "text_storage_precision",
        "dit_compute_precision",
        "dit_patcher_class",
        "dit_patcher_mode",
        "text_compute_precision",
        "text_conditioning_precision",
    }
    result = {key: item for key, item in value.items() if key in allowed}
    if "weight_sources" in result:
        source_fields = {"repo_id", "revision", "filename", "sha256", "precision"}
        result["weight_sources"] = {
            name: {key: item[key] for key in source_fields if key in item}
            for name, item in result["weight_sources"].items()
            if name in {"dit", "text"}
        }
    identity_fields = {"name", "sha256", "size", "mtime_ns", "targets", "rank"}
    for key, fields in (
        ("loras", {"name", "strength"}),
        ("lora_identities", identity_fields),
        ("lora", identity_fields | {"strength", "applied_targets"}),
    ):
        if key in result:
            result[key] = [{name: item[name] for name in fields if name in item} for item in result[key]]
    if result.get("error"):
        result["error"] = "生成に失敗しました。Forge Neoのログを確認してください。"
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
        ("/sdapi/v1/moss-image/jobs", start, "POST"),
        ("/sdapi/v1/moss-image/jobs/{identifier}", poll, "GET"),
        ("/sdapi/v1/moss-image/jobs/{identifier}/cancel", cancel, "POST"),
    ):
        app.add_api_route(path, endpoint, methods=[method], dependencies=[Depends(authenticate)])
