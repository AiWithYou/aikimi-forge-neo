"""Safe, lightweight diagnostics shared by the Aikimi UI and versioned API."""

from __future__ import annotations

import html
import importlib.metadata
import math
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from modules.aikimi_security.redaction import safe_error_message

API_VERSION = "1"
_GIB = 1024**3
_WINDOWS_ABSOLUTE_PATH = re.compile(r"(?i)(?<![a-z0-9_])(?:[a-z]:[\\/]|\\\\)[^\s<>\"|]+")
_POSIX_SENSITIVE_PATH = re.compile(
    r"(?<![a-z0-9_])/(?:home|users|tmp|var|opt|mnt|workspace)/[^\s<>\"|]+",
    re.IGNORECASE,
)


class CheckState(StrEnum):
    """Stable state vocabulary used by both HTML and JSON consumers."""

    READY = "ready"
    WARNING = "warning"
    BLOCKED = "blocked"

    @property
    def label(self) -> str:
        return self.value.title()


@dataclass(frozen=True)
class DiagnosticCheck:
    """One client-safe check result with a concrete remediation."""

    id: str
    label: str
    state: CheckState
    summary: str
    action: str = ""
    available: bool | None = None

    def public_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "state": self.state.value,
            "summary": _public_line(self.summary),
            "action": _public_line(self.action),
        }
        if self.available is not None:
            result["available"] = bool(self.available)
        return result


@dataclass(frozen=True)
class DiagnosticPaths:
    """Internal paths used for checks. These values are never serialized."""

    root: Path
    models_root: Path
    output_root: Path


class _CudaLike(Protocol):
    def is_available(self) -> bool: ...

    def current_device(self) -> int: ...

    def get_device_properties(self, device: int) -> Any: ...


def _public_line(value: object, *, limit: int = 320) -> str:
    """Return one bounded line with generic local absolute paths removed."""

    if value is None or not str(value).strip():
        return ""
    text = safe_error_message(value, limit=limit).replace("\t", " ")
    text = _WINDOWS_ABSOLUTE_PATH.sub("<local-path>", text)
    return _POSIX_SENSITIVE_PATH.sub("<local-path>", text)


def app_version() -> str:
    try:
        from modules_forge import forge_version

        return f"Aikimi Studio Neo {forge_version.version} {forge_version.release}"
    except (AttributeError, ImportError):
        return "Aikimi Studio Neo"


def _short_commit(root: Path) -> str | None:
    git_executable = shutil.which("git")
    if git_executable is None:
        return None
    try:
        # The executable is resolved locally and every argument is a fixed literal.
        result = subprocess.run(  # noqa: S603
            [git_executable, "rev-parse", "--short=12", "HEAD"],
            cwd=root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    candidate = result.stdout.strip().lower()
    if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{7,40}", candidate):
        return candidate
    return None


def default_paths() -> DiagnosticPaths:
    """Resolve internal roots without exposing them to a response."""

    from modules.paths_internal import default_output_dir, models_path, script_path

    return DiagnosticPaths(
        root=Path(script_path).resolve(),
        models_root=Path(models_path).resolve(),
        output_root=Path(default_output_dir).resolve(),
    )


def feature_checks(paths: DiagnosticPaths | None = None) -> tuple[DiagnosticCheck, ...]:
    """Delegate model and workflow checks to the focused capability module."""

    from modules.aikimi_capabilities import feature_checks as collect_feature_checks

    return collect_feature_checks(paths or default_paths())


def _python_check() -> DiagnosticCheck:
    version = ".".join(str(item) for item in sys.version_info[:3])
    supported = sys.version_info[:2] == (3, 13)
    return DiagnosticCheck(
        "python",
        "Python",
        CheckState.READY if supported else CheckState.WARNING,
        f"Python {version} is running.",
        "Use 64-bit Python 3.13 on Windows." if not supported else "No action is required.",
    )


def _package_check(distribution: str, label: str) -> DiagnosticCheck:
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return DiagnosticCheck(
            distribution,
            label,
            CheckState.BLOCKED,
            f"{label} is not installed.",
            "Install the repository requirements and retry the check.",
        )
    except Exception:
        return DiagnosticCheck(
            distribution,
            label,
            CheckState.WARNING,
            f"{label} version could not be read.",
            "Reinstall the package in the Aikimi virtual environment.",
        )
    return DiagnosticCheck(
        distribution,
        label,
        CheckState.READY,
        f"{label} {version} is installed.",
        "No action is required.",
    )


def _cuda_check(torch_module: Any | None = None) -> DiagnosticCheck:
    try:
        if torch_module is None:
            import torch as torch_module
        cuda: _CudaLike = torch_module.cuda
        if not cuda.is_available():
            return DiagnosticCheck(
                "cuda",
                "CUDA / GPU",
                CheckState.BLOCKED,
                "CUDA is unavailable; GPU generation is not ready.",
                "Check the NVIDIA driver, CUDA-enabled PyTorch build, and launch profile.",
            )
        device = int(cuda.current_device())
        properties = cuda.get_device_properties(device)
        name = _public_line(getattr(properties, "name", "GPU"), limit=120)
        total = float(getattr(properties, "total_memory", 0)) / _GIB
        cuda_version = _public_line(getattr(getattr(torch_module, "version", None), "cuda", "unknown"), limit=40)
    except Exception:
        return DiagnosticCheck(
            "cuda",
            "CUDA / GPU",
            CheckState.BLOCKED,
            "CUDA device information could not be read safely.",
            "Check the NVIDIA driver and the installed PyTorch build.",
        )
    return DiagnosticCheck(
        "cuda",
        "CUDA / GPU",
        CheckState.READY,
        f"{name} · {total:.1f} GiB VRAM · CUDA {cuda_version}.",
        "No action is required.",
    )


def _ram_check() -> DiagnosticCheck:
    try:
        import psutil

        memory = psutil.virtual_memory()
        total = float(memory.total) / _GIB
        available = float(memory.available) / _GIB
    except Exception:
        return DiagnosticCheck(
            "ram",
            "System RAM",
            CheckState.WARNING,
            "System RAM information is unavailable.",
            "Install psutil or review memory in Windows Task Manager.",
        )
    state = CheckState.READY if total >= 32 and available >= 8 else CheckState.WARNING
    return DiagnosticCheck(
        "ram",
        "System RAM",
        state,
        f"{available:.1f} GiB available of {total:.1f} GiB.",
        "Close memory-heavy applications or select a lower-memory workflow."
        if state is CheckState.WARNING
        else "No action is required.",
    )


def _storage_check(output_root: Path) -> DiagnosticCheck:
    if not output_root.is_dir():
        return DiagnosticCheck(
            "storage",
            "Output storage",
            CheckState.BLOCKED,
            "The managed output directory does not exist.",
            "Create the configured output directory and retry the check.",
        )
    probe: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".aikimi-diagnostic-",
            suffix=".tmp",
            dir=output_root,
            delete=False,
        ) as stream:
            stream.write(b"ok")
            probe = Path(stream.name)
        probe.unlink()
        probe = None
        free = shutil.disk_usage(output_root).free / _GIB
    except OSError:
        if probe is not None:
            try:
                probe.unlink(missing_ok=True)
            except OSError:
                pass
        return DiagnosticCheck(
            "storage",
            "Output storage",
            CheckState.BLOCKED,
            "The managed output directory is not writable.",
            "Check directory permissions and free space.",
        )
    if free < 5:
        state = CheckState.BLOCKED
    elif free < 20:
        state = CheckState.WARNING
    else:
        state = CheckState.READY
    return DiagnosticCheck(
        "storage",
        "Output storage",
        state,
        f"The output directory is writable with {free:.1f} GiB free.",
        "Free disk space before model installation or high-resolution generation."
        if state is not CheckState.READY
        else "No action is required.",
    )


def _security_check(options: Any | None = None) -> DiagnosticCheck:
    if options is None:
        from modules.shared_cmd_options import cmd_opts as options
    try:
        from modules.aikimi_security.remote_access import exposure_reasons

        reasons = exposure_reasons(options)
    except (AttributeError, ImportError, TypeError):
        reasons = ()
    api_enabled = bool(getattr(options, "api", False) or getattr(options, "nowebui", False))
    web_enabled = not bool(getattr(options, "nowebui", False))
    web_auth = bool(getattr(options, "gradio_auth", None) or getattr(options, "gradio_auth_path", None))
    api_auth = bool(getattr(options, "api_auth", None) or getattr(options, "api_auth_path", None))
    if not reasons:
        api_note = "API authentication configured" if api_auth else "API is loopback-only"
        return DiagnosticCheck(
            "security",
            "Exposure / authentication",
            CheckState.READY,
            f"Local Safe loopback mode is active; {api_note}.",
            "No action is required.",
        )
    authenticated = bool(getattr(options, "aikimi_remote", False))
    authenticated = authenticated and (not web_enabled or web_auth)
    authenticated = authenticated and (not api_enabled or api_auth)
    if authenticated:
        return DiagnosticCheck(
            "security",
            "Exposure / authentication",
            CheckState.WARNING,
            "Authenticated remote mode is active.",
            "Disable remote exposure when it is no longer needed and keep extensions trusted.",
        )
    return DiagnosticCheck(
        "security",
        "Exposure / authentication",
        CheckState.BLOCKED,
        "A remote exposure setting is active without the required Aikimi authentication policy.",
        "Stop the server and restart with Local Safe, or explicitly configure authenticated remote mode.",
    )


def _application_check(paths: DiagnosticPaths) -> DiagnosticCheck:
    commit = _short_commit(paths.root)
    suffix = f" · commit {commit}" if commit else ""
    return DiagnosticCheck(
        "application",
        "Aikimi Studio Neo",
        CheckState.READY if commit else CheckState.WARNING,
        f"{app_version()}{suffix}.",
        "No action is required." if commit else "Use a Git checkout to make updates and support reports reproducible.",
    )


def system_checks(
    paths: DiagnosticPaths | None = None,
    *,
    options: Any | None = None,
    torch_module: Any | None = None,
    include_features: bool = True,
) -> tuple[DiagnosticCheck, ...]:
    """Collect bounded checks without downloads, model hashing, or generation."""

    paths = paths or default_paths()
    checks: list[DiagnosticCheck] = [
        _application_check(paths),
        _python_check(),
        _package_check("torch", "PyTorch"),
        _package_check("gradio", "Gradio"),
        _cuda_check(torch_module),
        _ram_check(),
        _storage_check(paths.output_root),
        _security_check(options),
    ]
    if include_features:
        checks.extend(feature_checks(paths))
    return tuple(checks)


def overall_state(checks: Iterable[DiagnosticCheck]) -> CheckState:
    states = {check.state for check in checks}
    if CheckState.BLOCKED in states:
        return CheckState.BLOCKED
    if CheckState.WARNING in states:
        return CheckState.WARNING
    return CheckState.READY


def diagnostics_payload(
    paths: DiagnosticPaths | None = None,
    *,
    options: Any | None = None,
    torch_module: Any | None = None,
) -> dict[str, Any]:
    checks = system_checks(paths, options=options, torch_module=torch_module)
    return {
        "api_version": API_VERSION,
        "app_version": app_version(),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "state": overall_state(checks).value,
        "checks": [
            {
                "id": check.id,
                "label": check.label,
                **check.public_dict(),
            }
            for check in checks
        ],
    }


def health_payload() -> dict[str, str]:
    """Return a deliberately minimal liveness response."""

    return {
        "api_version": API_VERSION,
        "app_version": app_version(),
        "status": "ok",
    }


def _finite_number(value: object, *, minimum: float = 0.0) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        return None
    return int(value) if isinstance(value, int) else number


def _safe_model_name(value: object) -> str | None:
    if not value:
        return None
    name = str(value).replace("\\", "/").rsplit("/", 1)[-1]
    return _public_line(name, limit=160)


def status_payload(snapshot: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Whitelist the assistant snapshot for an API client."""

    if snapshot is None:
        try:
            from modules.aikimi_status import snapshot as collect_snapshot

            snapshot = collect_snapshot()
        except Exception:
            snapshot = {}
    model_raw = snapshot.get("model") if isinstance(snapshot, Mapping) else {}
    generation_raw = snapshot.get("generation") if isinstance(snapshot, Mapping) else {}
    memory_raw = snapshot.get("memory") if isinstance(snapshot, Mapping) else {}
    backend_raw = snapshot.get("backend") if isinstance(snapshot, Mapping) else {}
    model = model_raw if isinstance(model_raw, Mapping) else {}
    generation = generation_raw if isinstance(generation_raw, Mapping) else {}
    memory = memory_raw if isinstance(memory_raw, Mapping) else {}
    backend = backend_raw if isinstance(backend_raw, Mapping) else {}

    loading = bool(model.get("loading"))
    active = bool(generation.get("active"))
    queue_size = _finite_number(generation.get("queue_size")) or 0
    phase = "loading_model" if loading else "generating" if active else "queued" if queue_size else "idle"
    result: dict[str, Any] = {
        "api_version": API_VERSION,
        "app_version": app_version(),
        "status": "ok" if bool(backend.get("ready", False)) else "degraded",
        "phase": phase,
        "model": {
            "loaded": bool(model.get("loaded")),
            "loading": loading,
            "loaded_name": _safe_model_name(model.get("loaded_name")),
            "selected_name": _safe_model_name(model.get("selected_name")),
            "reload_pending": bool(model.get("reload_pending")),
            "last_load_seconds": _finite_number(model.get("last_load_seconds")),
        },
        "generation": {
            "active": active,
            "progress": _finite_number(generation.get("progress")),
            "eta": _finite_number(generation.get("eta")),
            "queue_size": int(queue_size),
        },
        "memory": {
            "available": bool(memory.get("available")),
        },
        "backend": {
            "ready": bool(backend.get("ready", False)),
            "uptime_seconds": _finite_number(backend.get("uptime_seconds")),
        },
    }
    if result["memory"]["available"]:
        for key in ("used", "free", "total", "allocated", "reserved", "oom_count"):
            result["memory"][key] = _finite_number(memory.get(key))
    return result


def capabilities_payload(paths: DiagnosticPaths | None = None) -> dict[str, Any]:
    checks = feature_checks(paths)
    commit = _short_commit((paths or default_paths()).root)
    return {
        "api_version": API_VERSION,
        "app_version": app_version(),
        "app_commit": commit,
        "state": overall_state(checks).value,
        "features": {check.id: check.public_dict() for check in checks},
    }


def render_diagnostics_html(
    checks: Sequence[DiagnosticCheck] | None = None,
) -> str:
    """Render accessible, escaped Settings UI markup."""

    collected = tuple(checks) if checks is not None else system_checks()
    state = overall_state(collected)
    counts = {item: sum(check.state is item for check in collected) for item in CheckState}
    rows = []
    for check in collected:
        action = (
            f'<p class="aikimi-diagnostic-action"><strong>Action:</strong> {html.escape(_public_line(check.action))}</p>'
            if check.action
            else ""
        )
        rows.append(
            f'<article class="aikimi-diagnostic-card is-{check.state.value}">'
            '<div class="aikimi-diagnostic-card-heading">'
            f'<span class="aikimi-diagnostic-state">{check.state.label}</span>'
            f"<h4>{html.escape(check.label)}</h4>"
            "</div>"
            f"<p>{html.escape(_public_line(check.summary))}</p>"
            f"{action}</article>"
        )
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    return (
        f'<section class="aikimi-diagnostics is-{state.value}" aria-labelledby="aikimi-diagnostics-title">'
        '<header class="aikimi-diagnostics-header">'
        '<div><p class="aikimi-diagnostics-eyebrow">AIKIMI NEO · SYSTEM CHECK</p>'
        '<h3 id="aikimi-diagnostics-title">Diagnostics</h3>'
        "<p>Checks are local and do not download models, generate media, or expose credentials and paths.</p></div>"
        f'<span class="aikimi-diagnostics-overall">{state.label}</span>'
        "</header>"
        '<div class="aikimi-diagnostics-counts" aria-label="Diagnostic result counts">'
        f"<span>Ready {counts[CheckState.READY]}</span>"
        f"<span>Warning {counts[CheckState.WARNING]}</span>"
        f"<span>Blocked {counts[CheckState.BLOCKED]}</span>"
        "</div>"
        f'<div class="aikimi-diagnostics-grid">{"".join(rows)}</div>'
        f'<p class="aikimi-diagnostics-generated">Checked {generated}. Full model hashes and live GPU generation were not run.</p>'
        "</section>"
    )


def register_api_routes(api: Any) -> None:
    """Register read-only endpoints through Api.add_api_route authentication."""

    def api_status() -> dict[str, Any]:
        return status_payload()

    def api_capabilities() -> dict[str, Any]:
        return capabilities_payload()

    api.add_api_route(
        "/aikimi/api/v1/health",
        health_payload,
        methods=["GET"],
    )
    api.add_api_route(
        "/aikimi/api/v1/status",
        api_status,
        methods=["GET"],
    )
    api.add_api_route(
        "/aikimi/api/v1/capabilities",
        api_capabilities,
        methods=["GET"],
    )


__all__ = [
    "API_VERSION",
    "CheckState",
    "DiagnosticCheck",
    "DiagnosticPaths",
    "app_version",
    "capabilities_payload",
    "default_paths",
    "diagnostics_payload",
    "feature_checks",
    "health_payload",
    "overall_state",
    "register_api_routes",
    "render_diagnostics_html",
    "status_payload",
    "system_checks",
]
