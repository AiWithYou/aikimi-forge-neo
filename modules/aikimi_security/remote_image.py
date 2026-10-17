"""Authenticated browser image imports through the shared safe downloader."""

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field

from modules.aikimi_security.auth import request_has_gradio_auth_async
from modules.aikimi_security.url_fetch import SafeFetchError, fetch_remote_image


class RemoteImageRequest(BaseModel):
    url: str = Field(min_length=1, max_length=4096)


class RemoteImageRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def validate_request(request: Request):
            try:
                return await handler(request)
            except RequestValidationError as error:
                raise HTTPException(status_code=422, detail="Invalid image import request.") from error

        return validate_request


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

    router = APIRouter(route_class=RemoteImageRoute)
    router.add_api_route(
        "/internal/remote-image",
        remote_image,
        methods=["POST"],
        dependencies=[Depends(require_auth)],
    )
    app.include_router(router)
