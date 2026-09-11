"""Fail-closed launch policy for local and explicitly remote Aikimi sessions."""

from __future__ import annotations

import ipaddress
from typing import Any


class RemoteAccessError(ValueError):
    """Raised when launch arguments would weaken the network boundary."""


def is_loopback_host(host: object | None) -> bool:
    """Return True for explicit loopback literals and ``localhost`` only."""

    if host is None:
        return True
    candidate = str(host).strip().strip("[]").rstrip(".").casefold()
    if candidate == "localhost":
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def exposure_reasons(options: Any) -> tuple[str, ...]:
    reasons: list[str] = []
    if bool(getattr(options, "listen", False)):
        reasons.append("--listen")
    server_name = getattr(options, "server_name", None)
    if server_name and not is_loopback_host(server_name):
        reasons.append("non-loopback --server-name")
    if bool(getattr(options, "share", False)):
        reasons.append("--share")
    if getattr(options, "ngrok", None) is not None:
        reasons.append("--ngrok")
    return tuple(reasons)


def _has_web_auth(options: Any) -> bool:
    return bool(getattr(options, "gradio_auth", None) or getattr(options, "gradio_auth_path", None))


def _has_api_auth(options: Any) -> bool:
    return bool(getattr(options, "api_auth", None) or getattr(options, "api_auth_path", None))


def validate_remote_access(options: Any) -> tuple[str, ...]:
    """Validate remote opt-in and authentication before either server starts."""

    reasons = exposure_reasons(options)
    remote_opt_in = bool(getattr(options, "aikimi_remote", False))
    api_enabled = bool(getattr(options, "api", False) or getattr(options, "nowebui", False))
    web_enabled = not bool(getattr(options, "nowebui", False))

    if reasons and not remote_opt_in:
        joined = ", ".join(reasons)
        raise RemoteAccessError(
            f"Remote exposure requested by {joined}. Add --aikimi-remote and authentication explicitly."
        )
    if remote_opt_in and not reasons:
        raise RemoteAccessError("--aikimi-remote requires --listen, a non-loopback --server-name, --share, or --ngrok.")
    if getattr(options, "ngrok", None) == "":
        raise RemoteAccessError("--ngrok requires a non-empty token source.")
    if getattr(options, "nowebui", False) and (getattr(options, "share", False) or getattr(options, "ngrok", None)):
        raise RemoteAccessError("--share and --ngrok cannot be used with --nowebui.")
    if reasons and web_enabled and not _has_web_auth(options):
        raise RemoteAccessError("Remote WebUI requires --gradio-auth-path (recommended) or --gradio-auth.")
    if reasons and api_enabled and not _has_api_auth(options):
        raise RemoteAccessError("Remote API requires --api-auth-path (recommended) or --api-auth.")
    return reasons


def server_bind_name(options: Any) -> str:
    """Return an explicit bind address; local mode never relies on framework defaults."""

    server_name = getattr(options, "server_name", None)
    if server_name:
        return str(server_name)
    if bool(getattr(options, "listen", False)):
        return "0.0.0.0"
    return "127.0.0.1"
