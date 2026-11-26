"""Workspace routes shared by the regular WebUI and API-only server."""

import mimetypes
import os
from contextlib import contextmanager
from pathlib import Path

from fastapi import Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from modules.aikimi_security.auth import request_has_gradio_auth_async
from modules_forge.aikimi_gallery import Gallery
from modules_forge.prompt_library import PromptLibrary, expand_wildcards
from modules_forge.workspace_store import open_regular


class MediaAction(BaseModel):
    id: str = Field(max_length=2048)
    enabled: bool = False


class Expansion(BaseModel):
    text: str = Field(max_length=16384)
    seed: int | None = None


@contextmanager
def _checked():
    try:
        yield
    except (OSError, ValueError, RecursionError):
        raise HTTPException(400, "入力またはローカルファイルを確認してください。") from None


def register_workspace_routes(api):
    from modules import aikimi_diagnostics
    from modules.paths_internal import data_path
    from modules_forge.gpu_residency import release_idle
    from modules_forge.model_summary import model_summary

    gallery = Gallery(Path(data_path).resolve())
    library = PromptLibrary(gallery.root)

    async def authorize(request: Request):
        if not api.credentials and not await request_has_gradio_auth_async(request.app, request):
            raise HTTPException(401, "Authentication required")
        if (
            request.method in {"POST", "DELETE"}
            and request.headers.get("content-type", "").split(";")[0] != "application/json"
        ):
            raise HTTPException(415, "Use application/json")

    def browse(
        studio: str = "",
        media: str = "",
        query: str = "",
        favorites: bool = False,
        order: str = "newest",
        offset: int = 0,
        limit: int = 40,
    ):
        with _checked():
            return gallery.list_items(studio, media, query, favorites, order, offset, limit)

    def preview(id: str):
        with _checked():
            path = gallery.media_path(id)
            stream = open_regular(path)
            size = os.fstat(stream.fileno()).st_size
            return StreamingResponse(
                iter(lambda: stream.read(65536), b""),
                media_type=mimetypes.guess_type(path.name)[0],
                headers={
                    "Content-Length": str(size),
                    "X-Content-Type-Options": "nosniff",
                    "Content-Security-Policy": "sandbox",
                },
                background=BackgroundTask(stream.close),
            )

    def favorite(action: MediaAction):
        with _checked():
            gallery.favorite(action.id, action.enabled)
            return {"favorite": action.enabled}

    def quarantine(action: MediaAction):
        with _checked():
            gallery.quarantine(action.id)
            return {"quarantined": True}

    def entries():
        with _checked():
            return library.entries()

    def save(entry: dict):
        with _checked():
            return library.save(entry)

    def delete(entry_id: str):
        with _checked():
            library.delete(entry_id)
            return {"deleted": True}

    def expand(body: Expansion):
        with _checked():
            prompt, seed = expand_wildcards(body.text, body.seed)
            return {"prompt": prompt, "seed": seed}

    def release():
        with _checked():
            return {"message": release_idle()}

    def model_summary_api():
        return model_summary()

    dependencies = [Depends(authorize)]
    aikimi_diagnostics.register_api_routes(api, dependencies=dependencies)
    for suffix, method, handler in (
        ("gallery", "GET", browse),
        ("gallery/media", "GET", preview),
        ("gallery/favorite", "POST", favorite),
        ("gallery/quarantine", "POST", quarantine),
        ("prompt-library", "GET", entries),
        ("prompt-library", "POST", save),
        ("prompt-library/{entry_id}", "DELETE", delete),
        ("prompt-library/expand", "POST", expand),
        ("models/summary", "GET", model_summary_api),
        ("gpu/release", "POST", release),
    ):
        api.add_api_route("/aikimi/api/v1/" + suffix, handler, methods=[method], dependencies=dependencies)
