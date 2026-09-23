"""Fail-closed protection for Gradio file routes that receive URL targets."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from urllib.parse import quote, unquote, urlsplit

from starlette.middleware import Middleware
from starlette.requests import HTTPConnection, Request
from starlette.responses import JSONResponse, RedirectResponse
from starlette.types import ASGIApp, Receive, Scope, Send

_FILE_ROUTE_MARKERS = (
    # Gradio 6.17.3 redirects external targets from the current and deprecated
    # variants of this route. Keep the guard narrower than general file paths.
    "/gradio_api/file=",
    "/gradio_api/file/",
    # Tag Autocomplete still requests the pre-Gradio-6 file route.
    "/file=",
)
_GUARD_INSTALLED_ATTRIBUTE = "_aikimi_gradio_file_url_guard_installed"
_LEGACY_ROUTE_INSTALLED_ATTRIBUTE = "_aikimi_gradio_legacy_file_route_installed"
_MAX_DECODE_PASSES = 4


def _decoded_variants(value: str) -> Iterator[str]:
    current = value
    yield current
    for _ in range(_MAX_DECODE_PASSES):
        decoded = unquote(current, errors="replace")
        if decoded == current:
            return
        yield decoded
        current = decoded


def _file_route_target(path: str) -> str | None:
    for decoded_path in _decoded_variants(path):
        for marker in _FILE_ROUTE_MARKERS:
            marker_index = decoded_path.find(marker)
            if marker_index >= 0:
                return decoded_path[marker_index + len(marker) :]
    return None


def _is_external_url_target(target: str) -> bool:
    for decoded_target in _decoded_variants(target):
        candidate = decoded_target.strip()
        slash_normalized = candidate.replace("\\", "/")
        lowered = slash_normalized.casefold()
        if candidate.startswith("//") or lowered.startswith(("http://", "https://")):
            return True
        try:
            parsed = urlsplit(slash_normalized)
        except ValueError:
            continue
        if parsed.username is not None or parsed.password is not None:
            return True
    return False


def _request_targets_external_url(scope: Scope) -> bool:
    path_values = [str(scope.get("path", ""))]
    raw_path = scope.get("raw_path")
    if isinstance(raw_path, bytes):
        path_values.append(raw_path.decode("latin-1", errors="replace"))

    for path in path_values:
        target = _file_route_target(path)
        if target is not None and _is_external_url_target(target):
            return True
    return False


class GradioExternalFileURLGuardMiddleware:
    """Reject external URL targets before Gradio can redirect to them."""

    def __init__(self, app: ASGIApp, *, gradio_app: Any) -> None:
        self.app = app
        self.gradio_app = gradio_app

    async def _defer_to_gradio_auth(self, scope: Scope) -> bool:
        if getattr(self.gradio_app, "auth", None) is None and getattr(self.gradio_app, "auth_dependency", None) is None:
            return False
        from modules.aikimi_security.auth import request_has_gradio_auth_async

        return not await request_has_gradio_auth_async(self.gradio_app, HTTPConnection(scope))

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        method = str(scope.get("method", "")).upper()
        if scope.get("type") == "http" and method in {"GET", "HEAD"} and _request_targets_external_url(scope):
            if await self._defer_to_gradio_auth(scope):
                await self.app(scope, receive, send)
                return
            response = JSONResponse(
                {"detail": "External URL file targets are not allowed."},
                status_code=403,
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


def install_gradio_file_url_guard(app: Any) -> bool:
    """Install the guard inside any existing authentication middleware."""

    if getattr(app, _GUARD_INSTALLED_ATTRIBUTE, False):
        return False
    app.user_middleware.append(Middleware(GradioExternalFileURLGuardMiddleware, gradio_app=app))
    if app.middleware_stack is not None:
        app.middleware_stack = app.build_middleware_stack()
    setattr(app, _GUARD_INSTALLED_ATTRIBUTE, True)
    return True


def install_gradio_legacy_file_route(app: Any) -> bool:
    """Pass old ``/file=`` requests through Gradio's current file policy."""

    if getattr(app, _LEGACY_ROUTE_INSTALLED_ATTRIBUTE, False):
        return False

    async def legacy_file(request: Request) -> RedirectResponse:
        target = quote(request.path_params["path"], safe="/:")
        query = f"?{request.url.query}" if request.url.query else ""
        root_path = str(request.scope.get("root_path", "")).rstrip("/")
        return RedirectResponse(f"{root_path}/gradio_api/file={target}{query}", status_code=307)

    app.add_api_route("/file={path:path}", legacy_file, methods=["GET", "HEAD"], include_in_schema=False)
    setattr(app, _LEGACY_ROUTE_INSTALLED_ATTRIBUTE, True)
    return True
