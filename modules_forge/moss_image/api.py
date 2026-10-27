"""Authenticated MOSS generation jobs return public metadata and PNG bytes."""

from __future__ import annotations

import base64
from pathlib import Path

from starlette.requests import Request

from modules_forge import gpu_fault

from .references import decode_uploads
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
        "mode",
        "edit_method",
        "denoise",
        "kv_cache",
        "image_strength",
        "reference_sha256",
        "reference_sizes",
        "vision_sizes",
        "reference_latent_tokens",
        "hires",
        "sampling_passes",
        "upscaler",
        "rope",
        "inference_policy",
    }
    result = {key: item for key, item in value.items() if key in allowed}
    if "inference_policy" in result:
        policy = result["inference_policy"]
        expected = {
            "sdpa_backends": ["efficient"], "situ_activation": "eager_fp32",
            "kernel_identity_observed": False,
        }
        cleaned = {key: item for key, item in expected.items() if key in policy and policy[key] == item}
        for key in ("situ_rows_per_chunk", "sdpa_calls_checked", "situ_eager_calls"):
            if type(policy.get(key)) is int and policy[key] >= 0:
                cleaned[key] = policy[key]
        if isinstance(policy.get("attention_dtypes"), list):
            cleaned["attention_dtypes"] = [name for name in policy["attention_dtypes"]
                                           if name in {"float16", "bfloat16", "float32"}]
        result["inference_policy"] = cleaned
    if "rope" in result:
        result["rope"] = {key: item for key, item in result["rope"].items() if key in {"scale_x", "scale_y"}}
    if "hires" in result:
        result["hires"] = {
            key: item for key, item in result["hires"].items() if key in {"base_width", "base_height", "steps", "denoise"}
        }
    if "sampling_passes" in result:
        fields = {"width", "height", "steps", "denoise", "seed", "cfg", "shift", "skip_layer", "actual_steps", "sampler", "scheduler", "rope"}
        result["sampling_passes"] = [{key: item for key, item in stage.items() if key in fields} for stage in result["sampling_passes"]]
        for stage in result["sampling_passes"]:
            if "rope" in stage:
                stage["rope"] = {key: item for key, item in stage["rope"].items() if key in {"scale_x", "scale_y"}}
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
        messages = {"gpu_stopped": gpu_fault.MESSAGE,
                    "unsupported_attention": "この設定では指定のattention計算方式を使用できません。"}
        if value.get("error_code") in messages:
            result.update(error_code=value["error_code"], error=messages[value["error_code"]])
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
        ("/sdapi/v1/moss-image/jobs", start, "POST"),
        ("/sdapi/v1/moss-image/jobs/{identifier}", poll, "GET"),
        ("/sdapi/v1/moss-image/jobs/{identifier}/cancel", cancel, "POST"),
    ):
        app.add_api_route(path, endpoint, methods=[method], dependencies=[Depends(authenticate)])
