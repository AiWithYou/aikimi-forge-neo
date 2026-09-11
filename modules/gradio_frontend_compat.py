"""Exact, fail-closed frontend compatibility patches for the pinned Gradio build."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio
from fastapi import FastAPI, Request, Response
from fastapi.responses import ORJSONResponse
from fastapi.routing import APIRoute
from gradio.routes import App

SUPPORTED_VERSION = "6.17.3"
TABS_ASSET_NAME = "Walkthrough.svelte_svelte_type_style_lang-DBgsQkoF.js"
ORIGINAL_ASSET_SHA256 = "e9be5d9fd700f1521287465e80c768ba326c8d274f094a7cea4ecb8bf4cdb8f5"
PATCHED_ASSET_SHA256 = "62076871b459f46dddfc920c2a756c574231b607e84eb714304decabd371b344"
PATCH_ROUTE_NAME = "aikimi_gradio_tabs_compat"

# Gradio 6.17.3 predates the upstream Tabs mount-storm fix in PR #13509.
# These are exact compiled snippets from that one audited wheel. The bounded
# workaround disables overflow measurement, batches initial_tabs updates, and
# removes per-Tab registration invalidations. Any upstream byte change fails
# before the server listens and requires a fresh compatibility review.
_ORIGINAL_OVERFLOW_FUNCTION = (
    b"async function ne(){if(!e(m)||(await Ce(),await new Promise(n=>requestAnimationFrame(n)),!e(m)))return;"
    b"const t=e(m).clientWidth;let s=0,_=e(l).length;for(let n=0;n<e(l).length;n++){const g=e(l)[n];"
    b'if(!g||g.visible===!1||g.visible==="hidden")continue;const z=e(T)[g.id];if(!z)continue;'
    b's+=z.getBoundingClientRect().width;const Y=e(l).slice(n+1).some(A=>A&&A.visible!==!1&&A.visible!=="hidden")?t-$e:t;'
    b"if(s>Y){_=n;break}}d(he,e(l).slice(0,_)),d(I,e(l).slice(_)),d(le,pe(v())),"
    b"d(we,e(I).filter(n=>n&&n.visible!==!1).length>0)}"
)

_NO_OVERFLOW_FUNCTION = b"async function ne(){}"

_ORIGINAL_INITIAL_TAB_SYNC = (
    b"function Se(t){Ce().then(()=>{for(let s=0;s<t.length;s++)t[s]&&!ae.has(s)&&J(l,e(l)[s]=t[s])})}"
)
_STATIC_INITIAL_TAB_SYNC = (
    b"function Se(t){Ce().then(()=>{const s=t.slice();J(l,s),d(he,s),d(I,[]),d(le,!1),d(we,!1)})}"
)

_ORIGINAL_TAB_REGISTRATION = (
    b"register_tab:(t,s)=>(ae.add(s),J(l,e(l)[s]=t),"
    b"v()===!1&&t.visible!==!1&&t.interactive&&(N(P,t.id),N(X,s)),s),"
    b"unregister_tab:(t,s)=>{ae.delete(s),v()===t.id&&N(P,e(l)[0]?.id||!1),"
    b"J(l,e(l)[s]=null)},selected_tab:P,selected_tab_index:X"
)
_STATIC_TAB_REGISTRATION = (
    b"register_tab:(t,s)=>(ae.add(s),"
    b"v()===!1&&t.visible!==!1&&t.interactive&&(N(P,t.id),N(X,s)),s),"
    b"unregister_tab:(t,s)=>{ae.delete(s),v()===t.id&&N(P,e(l)[0]?.id||!1)},"
    b"selected_tab:P,selected_tab_index:X"
)


class GradioFrontendCompatibilityError(RuntimeError):
    """Raised when the pinned third-party asset no longer matches the audited build."""


@dataclass(frozen=True)
class PatchedFrontendAsset:
    filename: str
    content: bytes
    original_sha256: str
    patched_sha256: str


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _replace_exact(source: bytes, old: bytes, new: bytes, description: str) -> bytes:
    if source.count(old) != 1:
        raise GradioFrontendCompatibilityError(
            f"The audited Gradio tabs {description} snippet was not found exactly once; "
            "the compatibility patch was not applied."
        )
    return source.replace(old, new, 1)


def _asset_path() -> Path:
    package_root = Path(gradio.__file__).resolve().parent
    asset = package_root / "templates" / "frontend" / "assets" / TABS_ASSET_NAME
    resolved = asset.resolve()
    if not resolved.is_relative_to(package_root) or not resolved.is_file():
        raise GradioFrontendCompatibilityError(
            "The audited Gradio tabs asset is missing; reinstall the pinned dependencies."
        )
    return resolved


def build_patched_tabs_asset() -> PatchedFrontendAsset:
    """Read and patch the exact audited Gradio asset without changing site-packages."""

    if gradio.__version__ != SUPPORTED_VERSION:
        raise GradioFrontendCompatibilityError(
            f"Gradio {gradio.__version__} is not supported by the tabs compatibility patch; "
            f"expected {SUPPORTED_VERSION}."
        )

    source = _asset_path().read_bytes()
    source_hash = _sha256(source)
    if source_hash != ORIGINAL_ASSET_SHA256:
        raise GradioFrontendCompatibilityError(
            "The Gradio tabs asset does not match the audited SHA-256; the compatibility patch was not applied."
        )
    patched = _replace_exact(
        source,
        _ORIGINAL_OVERFLOW_FUNCTION,
        _NO_OVERFLOW_FUNCTION,
        "overflow function",
    )
    patched = _replace_exact(
        patched,
        _ORIGINAL_INITIAL_TAB_SYNC,
        _STATIC_INITIAL_TAB_SYNC,
        "initial-tab sync",
    )
    patched = _replace_exact(
        patched,
        _ORIGINAL_TAB_REGISTRATION,
        _STATIC_TAB_REGISTRATION,
        "registration",
    )
    if patched == source:
        raise GradioFrontendCompatibilityError("The Gradio tabs compatibility patch did not apply cleanly.")
    patched_hash = _sha256(patched)
    if patched_hash != PATCHED_ASSET_SHA256:
        raise GradioFrontendCompatibilityError("The patched Gradio tabs asset does not match the audited SHA-256.")

    return PatchedFrontendAsset(
        filename=TABS_ASSET_NAME,
        content=patched,
        original_sha256=source_hash,
        patched_sha256=patched_hash,
    )


def _validate_patched_asset(asset: PatchedFrontendAsset) -> None:
    if (
        asset.filename != TABS_ASSET_NAME
        or asset.original_sha256 != ORIGINAL_ASSET_SHA256
        or asset.patched_sha256 != PATCHED_ASSET_SHA256
        or _sha256(asset.content) != PATCHED_ASSET_SHA256
    ):
        raise GradioFrontendCompatibilityError("The prepared Gradio tabs asset failed integrity validation.")


def install_gradio_tabs_compatibility_route(
    app: FastAPI,
    asset: PatchedFrontendAsset | None = None,
) -> PatchedFrontendAsset:
    """Serve the audited tabs patch before Gradio's generic assets route."""

    existing = next(
        (route for route in app.router.routes if getattr(route, "name", None) == PATCH_ROUTE_NAME),
        None,
    )
    if existing is not None:
        return existing.endpoint.__aikimi_patched_asset__

    asset = asset or build_patched_tabs_asset()
    _validate_patched_asset(asset)

    async def serve_patched_tabs(request: Request) -> Response:
        etag = f'"{asset.patched_sha256}"'
        headers = {
            "Cache-Control": "max-age=0, must-revalidate",
            "ETag": etag,
            "X-Aikimi-Gradio-Compat": SUPPORTED_VERSION,
            "X-Content-Type-Options": "nosniff",
        }
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers=headers)
        return Response(asset.content, media_type="text/javascript", headers=headers)

    serve_patched_tabs.__aikimi_patched_asset__ = asset
    route = APIRoute(
        path=f"/assets/{asset.filename}",
        endpoint=serve_patched_tabs,
        methods=["GET", "HEAD"],
        name=PATCH_ROUTE_NAME,
        include_in_schema=False,
    )
    app.router.routes.insert(0, route)
    return asset


def create_gradio_compatibility_app(
    asset: PatchedFrontendAsset,
    *,
    app_kwargs: Mapping[str, Any] | None = None,
    debug: bool = False,
) -> App:
    """Prepare Gradio's FastAPI app with the exact route before a listener starts."""

    prepared_kwargs = dict(app_kwargs or {})
    prepared_kwargs.setdefault("default_response_class", ORJSONResponse)
    prepared_app = App(debug=debug, **prepared_kwargs)
    install_gradio_tabs_compatibility_route(prepared_app, asset)
    return prepared_app
