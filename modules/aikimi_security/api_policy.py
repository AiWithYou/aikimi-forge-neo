"""Explicit disclosure policy for legacy WebUI API compatibility routes."""

from __future__ import annotations

import re
from argparse import Namespace
from typing import Any

from modules.aikimi_security.redaction import is_sensitive_key, redact_mapping

PUBLIC_CMD_FLAG_ALLOWLIST = frozenset(
    {
        "aikimi_remote",
        "api",
        "disable_all_extensions",
        "disable_extra_extensions",
        "freeze_settings",
        "gradio_debug",
        "hide_ui_dir_config",
        "listen",
        "lowvram",
        "medvram",
        "no_gradio_queue",
        "nowebui",
        "port",
        "share",
        "theme",
        "ui_debug_mode",
    }
)

_NON_PUBLIC_OPTION_NAME = re.compile(
    r"(?:^|_)(?:auth|cache|credential|directory|dir|endpoint|file|filename|folder|"
    r"host|listen|ngrok|path|proxy|remote|secret|share|tls|token|url)(?:_|$)",
    re.IGNORECASE,
)


def option_name_is_public(key: str) -> bool:
    """Fail closed for settings likely to carry paths, network policy, or secrets."""

    return (
        not key.startswith("forge_additional_modules")
        and not key.startswith("outdir_")
        and not is_sensitive_key(key)
        and not _NON_PUBLIC_OPTION_NAME.search(key)
    )


def public_cmd_flags(options: Namespace | Any) -> dict[str, Any]:
    """Return only reviewed, non-path command-line state."""

    from modules.aikimi_security.remote_access import exposure_reasons

    values = vars(options)
    public = {key: values[key] for key in PUBLIC_CMD_FLAG_ALLOWLIST if key in values}
    public.update(
        {
            "bind_scope": "remote" if exposure_reasons(options) else "loopback",
            "api_auth_enabled": bool(getattr(options, "api_auth", None) or getattr(options, "api_auth_path", None)),
            "gradio_auth_enabled": bool(
                getattr(options, "gradio_auth", None) or getattr(options, "gradio_auth_path", None)
            ),
        }
    )
    return redact_mapping(public)


def public_options(options: Any) -> dict[str, Any]:
    """Return options explicitly admitted by ``Options.api_accessible``."""

    return redact_mapping(
        {
            key: options.data.get(key, info.default)
            for key, info in options.data_labels.items()
            if options.api_accessible(key, write=False)
        }
    )
