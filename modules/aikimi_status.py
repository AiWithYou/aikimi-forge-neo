"""Read-only runtime snapshot for the Aikimi assistant UI."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from modules.aikimi_security.redaction import safe_error_message


def _display_name(path: object | None) -> str | None:
    return Path(str(path).replace("\\", "/")).name if path else None


def request_is_authorized(app, request) -> bool:
    """Match Gradio's login boundary for an add-on FastAPI route."""
    from modules.aikimi_security.auth import request_has_gradio_auth

    return request_has_gradio_auth(app, request)


def _model_snapshot(model_data=None) -> dict[str, Any]:
    if model_data is None:
        from modules import sd_models

        model_data = sd_models.model_data
    loading_parameters = model_data.forge_loading_parameters or {}
    checkpoint_info = loading_parameters.get("checkpoint_info")
    selected_path = getattr(checkpoint_info, "filename", None)
    selected_name = getattr(checkpoint_info, "name", None) or _display_name(selected_path)
    loaded = getattr(model_data.sd_model, "forge_objects", None) is not None
    loaded_checkpoint_info = getattr(model_data.sd_model, "sd_checkpoint_info", None)
    loaded_path = getattr(model_data.sd_model, "filename", None) if loaded else None
    loaded_name = getattr(loaded_checkpoint_info, "name", None) or _display_name(loaded_path)

    return {
        "loaded": loaded,
        "loading": bool(model_data.forge_loading),
        "loaded_name": loaded_name,
        "selected_name": selected_name,
        "reload_pending": model_data.forge_hash != str(loading_parameters),
        "last_load_seconds": model_data.last_load_seconds,
    }


def _generation_progress(
    job_count: int,
    job_no: int,
    sampling_steps: int,
    sampling_step: int,
) -> float:
    value = 0.0

    if job_count > 0:
        value += job_no / job_count
        if sampling_steps > 0:
            value += sampling_step / sampling_steps / job_count

    return min(max(value, 0.0), 1.0)


def _generation_snapshot(state=None, pending_tasks=None) -> dict[str, Any]:
    if state is None:
        from modules import shared

        state = shared.state
    if pending_tasks is None:
        from modules import progress

        pending_tasks = progress.pending_tasks

    job = state.job
    job_count = state.job_count
    job_no = state.job_no
    sampling_steps = state.sampling_steps
    sampling_step = state.sampling_step
    time_start = state.time_start
    textinfo = state.textinfo
    active = bool(job) or job_count != 0
    value = (
        _generation_progress(job_count, job_no, sampling_steps, sampling_step)
        if active
        else 0.0
    )
    eta = None

    if active and value > 0 and time_start:
        elapsed = max(time.time() - time_start, 0.0)
        eta = max(elapsed / value - elapsed, 0.0)

    return {
        "active": active,
        "progress": value,
        "eta": eta,
        "text": safe_error_message(textinfo, limit=240) if textinfo else None,
        "queue_size": len(pending_tasks),
    }


def _memory_snapshot() -> dict[str, Any]:
    try:
        import torch

        from backend import memory_management

        device = memory_management.get_torch_device()
        if getattr(device, "type", None) in {"cpu", "mps"}:
            return {"available": False, "device": str(device), "error": "VRAM unavailable"}

        backend = torch.xpu if memory_management.is_intel_xpu() else torch.cuda
        free, total = backend.mem_get_info(device)
        stats = dict(backend.memory_stats(device))

        return {
            "available": True,
            "device": str(device),
            "used": total - free,
            "free": free,
            "total": total,
            "allocated": stats.get("allocated_bytes.all.current", 0),
            "reserved": stats.get("reserved_bytes.all.current", 0),
            "oom_count": stats.get("num_ooms", 0),
        }
    except Exception as exc:
        return {
            "available": False,
            "device": None,
            "error": f"{type(exc).__name__}: {safe_error_message(exc, limit=160)}",
        }


def snapshot() -> dict[str, Any]:
    """Return a JSON-safe view of existing Forge runtime state."""

    from modules import shared

    server_start = shared.state.server_start or time.time()
    return {
        "model": _model_snapshot(),
        "generation": _generation_snapshot(),
        "memory": _memory_snapshot(),
        "backend": {
            "ready": True,
            "uptime_seconds": max(time.time() - server_start, 0.0),
        },
    }
