"""Authenticated browser image imports through the shared safe downloader."""

from fastapi import Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from modules.aikimi_security.auth import request_has_gradio_auth_async
from modules.aikimi_security.url_fetch import SafeFetchError, fetch_remote_image


class RemoteImageRequest(BaseModel):
    url: str = Field(min_length=1, max_length=4096)


def register_remote_image_route(app) -> None:
    async def require_auth(request: Request):
        if not await request_has_gradio_auth_async(app, request):
            raise HTTPException(status_code=401, detail="Not authenticated")

    def remote_image(value: RemoteImageRequest):
        try:
            payload = fetch_remote_image(value.url)
        except SafeFetchError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return Response(
            content=payload,
            media_type="application/octet-stream",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    app.add_api_route(
        "/internal/remote-image",
        remote_image,
        methods=["POST"],
        dependencies=[Depends(require_auth)],
    )
