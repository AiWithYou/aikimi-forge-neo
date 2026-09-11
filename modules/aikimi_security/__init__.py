"""Security primitives shared by Aikimi Studio Neo launch, API, and diagnostics code."""

# Web-framework middleware is imported from its concrete module so lightweight
# asset checks can use redaction without installing FastAPI or Starlette.
from modules.aikimi_security.paths import (
    UnsafeAllowedPathError,
    build_gradio_allowed_paths,
    build_gradio_blocked_paths,
)
from modules.aikimi_security.redaction import (
    redact_argv,
    redact_mapping,
    redact_text,
    redact_url,
    safe_error_message,
)
from modules.aikimi_security.remote_access import (
    RemoteAccessError,
    exposure_reasons,
    is_loopback_host,
    server_bind_name,
    validate_remote_access,
)

__all__ = [
    "RemoteAccessError",
    "UnsafeAllowedPathError",
    "build_gradio_allowed_paths",
    "build_gradio_blocked_paths",
    "exposure_reasons",
    "is_loopback_host",
    "redact_argv",
    "redact_mapping",
    "redact_text",
    "redact_url",
    "safe_error_message",
    "server_bind_name",
    "validate_remote_access",
]
