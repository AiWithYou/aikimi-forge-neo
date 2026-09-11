"""Conservative redaction for logs, API responses, and support bundles."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

REDACTED = "<redacted>"

_SENSITIVE_KEY = re.compile(
    r"(?:^|[_\-.])(?:access[_\-.]?key|api[_\-.]?key|auth(?:entication|orization)?|"
    r"authtoken|cookie|credential|passwd|password|private[_\-.]?key|secret|session|"
    r"signature|token)(?:$|[_\-.])",
    re.IGNORECASE,
)
_SENSITIVE_CLI_OPTION = re.compile(
    r"^--(?:(?:ngrok(?:-options)?)|.*(?:access[-_]?key|api[-_]?key|auth|authtoken|"
    r"cookie|credential|passwd|password|private[-_]?key|secret|signature|token)|"
    r"tls-keyfile)(?:$|=)",
    re.IGNORECASE,
)
_URL = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_HEADER = re.compile(
    r"(?im)\b(authorization|proxy-authorization|cookie|set-cookie|x-api-key|"
    r"api-key|x-auth-token)\s*:\s*[^\r\n]+"
)
_ASSIGNMENT = re.compile(
    r"(?i)\b([A-Z0-9_.-]*(?:access[_-]?key|api[_-]?key|auth|authtoken|cookie|"
    r"credential|passwd|password|private[_-]?key|secret|signature|token)"
    r"[A-Z0-9_.-]*)\s*=\s*([^\s,;]+)"
)
_NGROK_ASSIGNMENT = re.compile(r"(?i)\b(ngrok(?:_options)?)\s*=\s*([^\s,;]+)")
_JSON_SECRET = re.compile(
    r"(?i)([\"']?(?:access[_-]?key|api[_-]?key|auth|authtoken|basic[_-]?auth|"
    r"cookie|credential|passwd|password|private[_-]?key|secret|signature|token|"
    r"ngrok(?:[_-]?options)?|tls[_-]?keyfile)[\"']?\s*:\s*)([\"'])(.*?)(\2)"
)
_INLINE_FLAG = re.compile(
    r"(?i)(--(?:(?:ngrok(?:-options)?)|[^\s=]*(?:access[-_]?key|api[-_]?key|auth|"
    r"authtoken|cookie|credential|passwd|password|private[-_]?key|secret|signature|"
    r"token|tls-keyfile))\s*=\s*)([^\s]+)"
)
_SEPARATE_FLAG = re.compile(
    r"(?i)(--(?:(?:ngrok(?:-options)?)|[^\s=]*(?:access[-_]?key|api[-_]?key|auth|"
    r"authtoken|cookie|credential|passwd|password|private[-_]?key|secret|signature|"
    r"token|tls-keyfile))\s+)([^\s]+)"
)
_WINDOWS_ABSOLUTE_PATH = re.compile(r"(?i)(?<![A-Z0-9])(?:[A-Z]:[\\/]|\\\\)[^\s\"'<>|]+")
_POSIX_PRIVATE_PATH = re.compile(
    r"(?<![:A-Z0-9])/(?:home|Users|workspace|workspaces|tmp|var/tmp|mnt|media)/"
    r"[^\s\"'<>|]+",
    re.IGNORECASE,
)
_PATH_KEY = re.compile(
    r"(?:^|[_\s.-])(?:directory|dir|file|filename|folder|path)(?:$|[_\s.-])",
    re.IGNORECASE,
)


def is_sensitive_key(key: object) -> bool:
    """Return whether a mapping key or option name is secret-bearing."""

    normalized = re.sub(r"[^a-z0-9]+", "_", str(key).casefold()).strip("_")
    if normalized in {"ngrok", "ngrok_options", "tls_keyfile"}:
        return True
    return bool(_SENSITIVE_KEY.search(normalized))


def _is_sensitive_query_key(key: object) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(key).casefold()).strip("_")
    return is_sensitive_key(normalized) or normalized in {
        "key",
        "sig",
        "signature",
        "x_amz_signature",
        "x_goog_signature",
    }


def _redact_home(value: str) -> str:
    home = str(Path.home())
    if home:
        value = re.sub(re.escape(home), "<user-home>", value, flags=re.IGNORECASE)
    user_profile = os.environ.get("USERPROFILE")
    if user_profile and user_profile.casefold() != home.casefold():
        value = re.sub(re.escape(user_profile), "<user-home>", value, flags=re.IGNORECASE)
    return value


def redact_url(value: str) -> str:
    """Remove URL userinfo and redact credential-like query parameters."""

    try:
        parsed = urlsplit(value)
    except ValueError:
        return REDACTED

    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return _redact_home(value)

    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        port = f":{parsed.port}" if parsed.port is not None else ""
    except ValueError:
        port = ""
    netloc = f"{host}{port}"

    query_items = []
    try:
        for key, item in parse_qsl(parsed.query, keep_blank_values=True):
            query_items.append((key, REDACTED if _is_sensitive_query_key(key) else item))
        query = urlencode(query_items, doseq=True)
    except ValueError:
        query = REDACTED if parsed.query else ""

    return urlunsplit((parsed.scheme, netloc, parsed.path, query, ""))


def redact_text(value: object) -> str:
    """Redact common secret forms without logging the original on parse errors."""

    text = str(value)
    text = _HEADER.sub(lambda match: f"{match.group(1)}: {REDACTED}", text)
    text = _INLINE_FLAG.sub(lambda match: f"{match.group(1)}{REDACTED}", text)
    text = _SEPARATE_FLAG.sub(lambda match: f"{match.group(1)}{REDACTED}", text)
    text = _ASSIGNMENT.sub(lambda match: f"{match.group(1)}={REDACTED}", text)
    text = _NGROK_ASSIGNMENT.sub(lambda match: f"{match.group(1)}={REDACTED}", text)
    text = _JSON_SECRET.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{REDACTED}{match.group(4)}",
        text,
    )
    text = _URL.sub(lambda match: redact_url(match.group(0)), text)
    text = _WINDOWS_ABSOLUTE_PATH.sub("<local-path>", text)
    text = _POSIX_PRIVATE_PATH.sub("<local-path>", text)
    return _redact_home(text)


def redact_argv(argv: Sequence[object]) -> list[str]:
    """Redact both ``--secret value`` and ``--secret=value`` argument forms."""

    result: list[str] = []
    hide_next = False
    for raw in argv:
        item = str(raw)
        if hide_next:
            result.append(REDACTED)
            hide_next = False
            continue
        if _SENSITIVE_CLI_OPTION.match(item):
            if "=" in item:
                result.append(f"{item.split('=', 1)[0]}={REDACTED}")
            else:
                result.append(item)
                hide_next = True
            continue
        result.append(redact_text(item))
    return result


def redact_mapping(value: Any, *, _key: object | None = None) -> Any:
    """Recursively redact JSON-like data and exception strings."""

    if _key is not None and is_sensitive_key(_key):
        # Diagnostics may intentionally expose only whether authentication is
        # configured. Preserve a boolean state without ever preserving a
        # credential-bearing string, number, sequence, or mapping.
        if isinstance(value, bool) or value is None:
            return value
        return REDACTED
    if (
        _key is not None
        and _PATH_KEY.search(str(_key))
        and isinstance(value, (str, Path))
        and (str(value).startswith("/") or _WINDOWS_ABSOLUTE_PATH.search(str(value)))
    ):
        return "<local-path>"
    if isinstance(value, Mapping):
        return {str(key): redact_mapping(item, _key=key) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(redact_mapping(item) for item in value)
    if isinstance(value, list):
        return [redact_mapping(item) for item in value]
    if isinstance(value, set):
        return sorted((redact_mapping(item) for item in value), key=str)
    if isinstance(value, Path):
        return redact_text(value)
    if isinstance(value, bytes):
        return redact_text(value.decode("utf-8", errors="replace"))
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, BaseException):
        return safe_error_message(value)
    return value


def safe_error_message(error: BaseException | object, *, limit: int = 400) -> str:
    """Return a bounded, redacted message suitable for an untrusted client."""

    message = redact_text(error).replace("\r", " ").replace("\n", " ").strip()
    if not message:
        message = type(error).__name__ if isinstance(error, BaseException) else "Error"
    if len(message) > limit:
        message = f"{message[: limit - 1]}…"
    return message


def sanitized_subprocess_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Remove parent-process credentials that local model workers do not need."""

    blocked_names = {
        "ALL_PROXY",
        "COMMANDLINE_ARGS",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "INDEX_URL",
        "NO_PROXY",
        "PIP_CONFIG_FILE",
        "TORCH_COMMAND",
        "TORCH_INDEX_URL",
    }
    return {
        str(key): str(value)
        for key, value in environment.items()
        if str(key).upper() not in blocked_names and not is_sensitive_key(key)
    }
