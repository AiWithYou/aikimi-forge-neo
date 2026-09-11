"""Shared authentication helpers for core and extension FastAPI routes."""

from __future__ import annotations

import base64
import inspect
from collections.abc import Iterable
from pathlib import Path
from secrets import compare_digest
from typing import Any

from starlette.middleware import Middleware
from starlette.requests import HTTPConnection
from starlette.responses import JSONResponse


class AuthenticationConfigError(ValueError):
    """Raised for malformed or unreadable authentication configuration."""


_PUBLIC_WEB_EXACT_ROUTES = {
    ("GET", "/"),
    ("HEAD", "/"),
    ("POST", "/login"),
    ("POST", "/login/"),
    ("GET", "/gradio_api/user"),
    ("GET", "/gradio_api/user/"),
    ("GET", "/gradio_api/login_check"),
    ("GET", "/gradio_api/login_check/"),
    ("GET", "/gradio_api/token"),
    ("GET", "/gradio_api/token/"),
    ("GET", "/gradio_api/app_id"),
    ("GET", "/gradio_api/app_id/"),
    ("GET", "/favicon.ico"),
    ("HEAD", "/favicon.ico"),
    ("GET", "/theme.css"),
    ("HEAD", "/theme.css"),
    ("GET", "/manifest.json"),
    ("HEAD", "/manifest.json"),
    ("GET", "/pwa_icon"),
    ("HEAD", "/pwa_icon"),
}
_PUBLIC_WEB_STATIC_PREFIXES = ("/assets/", "/static/", "/svelte/", "/pwa_icon/")
_AUTH_INSTALLED_ATTRIBUTE = "_aikimi_remote_auth_installed"


def _credential_lines(value: str) -> Iterable[str]:
    for line in value.replace("\r", "\n").split("\n"):
        for item in line.split(","):
            candidate = item.strip()
            if candidate:
                yield candidate


def parse_credentials(value: str) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    for line in _credential_lines(value):
        if ":" not in line:
            raise AuthenticationConfigError("Each authentication entry must use the username:password format.")
        username, password = line.split(":", 1)
        if not username or not password:
            raise AuthenticationConfigError("Authentication usernames and passwords must not be empty.")
        result.append((username, password))
    if not result:
        raise AuthenticationConfigError("The authentication source is empty.")
    return tuple(result)


def read_credentials_file(path: str | Path) -> tuple[tuple[str, str], ...]:
    source = Path(path)
    try:
        if not source.is_file() or source.stat().st_size > 64 * 1024:
            raise AuthenticationConfigError("The authentication file is missing, not a file, or too large.")
        return parse_credentials(source.read_text(encoding="utf-8-sig"))
    except AuthenticationConfigError:
        raise
    except (OSError, UnicodeError) as exc:
        raise AuthenticationConfigError("The authentication file could not be read safely.") from exc


def credentials_from_options(options: Any, kind: str) -> tuple[tuple[str, str], ...]:
    inline = getattr(options, f"{kind}_auth", None)
    path = getattr(options, f"{kind}_auth_path", None)
    result: list[tuple[str, str]] = []
    if inline:
        result.extend(parse_credentials(inline))
    if path:
        result.extend(read_credentials_file(path))
    return tuple(result)


def validate_auth_configuration(options: Any) -> None:
    """Validate configured sources before model initialization or server launch."""

    for kind in ("api", "gradio"):
        if getattr(options, f"{kind}_auth", None) or getattr(options, f"{kind}_auth_path", None):
            credentials_from_options(options, kind)


def _basic_credentials(request) -> tuple[str, str] | None:
    header = request.headers.get("Authorization", "")
    scheme, _, encoded = header.partition(" ")
    if scheme.casefold() != "basic" or not encoded:
        return None
    try:
        decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
    except (ValueError, UnicodeError):
        return None
    if ":" not in decoded:
        return None
    return tuple(decoded.split(":", 1))  # type: ignore[return-value]


def request_has_basic_auth(request, credentials: Iterable[tuple[str, str]]) -> bool:
    supplied = _basic_credentials(request)
    if supplied is None:
        return False
    supplied_user, supplied_password = supplied
    matched = False
    for username, password in credentials:
        username_ok = compare_digest(supplied_user, username)
        password_ok = compare_digest(supplied_password, password)
        matched = matched or (username_ok and password_ok)
    return matched


def request_has_gradio_auth(app, request) -> bool:
    """Match Gradio's login cookie/dependency boundary for custom routes."""

    auth_dependency = getattr(app, "auth_dependency", None)
    auth = getattr(app, "auth", None)
    if auth_dependency is None and auth is None:
        return True
    if auth_dependency is not None:
        result = auth_dependency(request)
        if inspect.isawaitable(result):
            close = getattr(result, "close", None)
            if close is not None:
                close()
            return False
        return result is not None
    cookie_id = getattr(app, "cookie_id", "")
    token = request.cookies.get(f"access-token-{cookie_id}") or request.cookies.get(
        f"access-token-unsecure-{cookie_id}"
    )
    return getattr(app, "tokens", {}).get(token) is not None


async def request_has_gradio_auth_async(app, request) -> bool:
    """Async-aware Gradio authentication check used by the ASGI boundary."""

    auth_dependency = getattr(app, "auth_dependency", None)
    auth = getattr(app, "auth", None)
    if auth_dependency is None and auth is None:
        return True
    if auth_dependency is not None:
        result = auth_dependency(request)
        if inspect.isawaitable(result):
            result = await result
        return result is not None
    return request_has_gradio_auth(app, request)


def require_gradio_auth(app, request) -> None:
    if request_has_gradio_auth(app, request):
        return
    from fastapi import HTTPException, status

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
    )


def require_api_auth(request) -> None:
    from modules.shared_cmd_options import cmd_opts

    credentials = credentials_from_options(cmd_opts, "api")
    if not credentials or request_has_basic_auth(request, credentials):
        return
    from fastapi import HTTPException, status

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Basic"},
    )


class RemoteAuthPolicy:
    """Authentication policy shared by every route on a remotely exposed app."""

    __slots__ = ("_api_credentials", "_gradio_app", "api_only")

    def __init__(
        self,
        gradio_app: Any,
        api_credentials: Iterable[tuple[str, str]],
        *,
        api_only: bool,
    ) -> None:
        self._gradio_app = gradio_app
        self._api_credentials = tuple(api_credentials)
        self.api_only = api_only

        if api_only and not self._api_credentials:
            raise AuthenticationConfigError("Remote API-only mode requires API Basic authentication.")

    def __repr__(self) -> str:
        return (
            "RemoteAuthPolicy("
            f"api_only={self.api_only!r}, "
            f"api_auth_configured={bool(self._api_credentials)!r}, "
            f"gradio_auth_configured={self._gradio_auth_configured()!r})"
        )

    def _gradio_auth_configured(self) -> bool:
        return bool(
            getattr(self._gradio_app, "auth_dependency", None) is not None
            or getattr(self._gradio_app, "auth", None) is not None
        )

    def is_public_bootstrap_request(self, scope: dict[str, Any]) -> bool:
        """Allow only the resources needed to reach Gradio's own login boundary."""

        if self.api_only or scope.get("type") != "http":
            return False

        method = str(scope.get("method", "GET")).upper()
        path = str(scope.get("path", ""))
        client = scope.get("client")
        client_host = client[0] if isinstance(client, (tuple, list)) and client else None
        from modules.aikimi_security.remote_access import is_loopback_host

        if (
            method == "GET"
            and path == "/gradio_api/startup-events"
            and client_host is not None
            and is_loopback_host(client_host)
            and not getattr(self._gradio_app, "startup_events_triggered", False)
        ):
            return True
        if (method, path) in _PUBLIC_WEB_EXACT_ROUTES:
            return True
        return method in {"GET", "HEAD"} and path.startswith(_PUBLIC_WEB_STATIC_PREFIXES)

    async def is_authenticated(self, connection: HTTPConnection) -> bool:
        if self._api_credentials and request_has_basic_auth(connection, self._api_credentials):
            return True

        if self.api_only or not self._gradio_auth_configured():
            return False
        return await request_has_gradio_auth_async(self._gradio_app, connection)


class RemoteAuthenticationMiddleware:
    """Fail closed around core, internal, and extension routes in remote mode."""

    def __init__(self, app, *, policy: RemoteAuthPolicy) -> None:
        self.app = app
        self.policy = policy

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return

        if self.policy.is_public_bootstrap_request(scope):
            await self.app(scope, receive, send)
            return

        connection = HTTPConnection(scope)
        if await self.policy.is_authenticated(connection):
            await self.app(scope, receive, send)
            return

        if scope["type"] == "websocket":
            await send(
                {
                    "type": "websocket.close",
                    "code": 4401,
                    "reason": "Authentication required",
                }
            )
            return

        headers = {"WWW-Authenticate": "Basic"} if self.policy.api_only else None
        response = JSONResponse(
            {"detail": "Authentication required"},
            status_code=401,
            headers=headers,
        )
        await response(scope, receive, send)


def install_remote_auth_middleware(app, options: Any) -> bool:
    """Install one global auth boundary when the validated launch is remote."""

    from modules.aikimi_security.remote_access import exposure_reasons

    if not exposure_reasons(options):
        return False
    if getattr(app, _AUTH_INSTALLED_ATTRIBUTE, False):
        return False

    api_only = bool(getattr(options, "nowebui", False))
    policy = RemoteAuthPolicy(
        app,
        credentials_from_options(options, "api"),
        api_only=api_only,
    )
    insert_at = 0
    for index, middleware in enumerate(app.user_middleware):
        if "cors" in middleware.cls.__name__.casefold():
            # Keep CORS outside authentication so it can answer credential-free
            # preflight requests without ever dispatching an application route.
            insert_at = index + 1
            break
    app.user_middleware.insert(insert_at, Middleware(RemoteAuthenticationMiddleware, policy=policy))
    # A prepared Gradio app has no stack yet; keep it unbuilt so Gradio can add
    # its own middleware before listening. Running apps are rebuilt immediately.
    if app.middleware_stack is not None:
        app.middleware_stack = app.build_middleware_stack()
    setattr(app, _AUTH_INSTALLED_ATTRIBUTE, True)
    return True
