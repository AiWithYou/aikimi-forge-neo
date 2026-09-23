"""Read-only localhost workflow handoff. No model nodes, no queue endpoint."""

import asyncio
import ipaddress
from urllib.parse import urlsplit

import folder_paths
from aiohttp import web
from server import PromptServer

from .store import HandoffError, read_snapshot, token_value

WEB_DIRECTORY = "./web"
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _local_request(request):
    try:
        remote = ipaddress.ip_address(request.remote or "")
        host = urlsplit("http://" + request.host)
        host_ok = host.hostname in {"localhost", "127.0.0.1", "::1"}
        origin = request.headers.get("Origin")
        origin_ok = not origin or origin == f"{request.scheme}://{request.host}"
        return remote.is_loopback and host_ok and origin_ok and request.headers.get("Sec-Fetch-Site") != "cross-site"
    except ValueError:
        return False


@PromptServer.instance.routes.get("/aikimi/h3/workflows/{token}")
async def get_h3_workflow(request):
    headers = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}
    if not _local_request(request):
        return web.json_response(
            {"error": "この機能は同じPCのComfyUIからのみ使用できます。"}, status=403, headers=headers
        )
    try:
        token = token_value(request.match_info["token"])
        # Hash large media without blocking ComfyUI's event loop.
        record = await asyncio.to_thread(read_snapshot, folder_paths.get_input_directory(), token)
    except (HandoffError, OSError, TypeError) as exc:
        message = str(exc) if isinstance(exc, HandoffError) else "保存ワークフローの素材を読み込めません。"
        return web.json_response({"error": message}, status=404, headers=headers)
    return web.json_response(record, headers=headers)
