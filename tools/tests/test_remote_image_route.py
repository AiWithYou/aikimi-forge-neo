"""The browser image importer shares the API's real SSRF and authentication policy."""

from __future__ import annotations

import importlib
import io
import socket
from unittest import TestCase, main, mock

import gradio as gr
from fastapi import FastAPI
from fastapi.testclient import TestClient
from gradio.routes import App
from PIL import Image

from modules.aikimi_security import url_fetch


class RemoteImageRouteTests(TestCase):
    def setUp(self):
        self.route = importlib.import_module("modules.aikimi_security.remote_image")
        image = io.BytesIO()
        Image.new("RGB", (2, 2), "blue").save(image, format="PNG")
        self.png = image.getvalue()

    def app(self, authenticated=False):
        with gr.Blocks() as blocks:
            gr.Markdown("image import")
        app = App.create_app(blocks)
        if authenticated:
            app.auth = {"test-user": "test-password"}
            app.tokens = {"test-token": "test-user"}
        self.route.register_remote_image_route(app)
        return app

    def test_image_import_works_at_root_and_subpath(self):
        for mount in ("/", "/aikimi"):
            with self.subTest(mount=mount):
                app = self.app()
                outer = app if mount == "/" else FastAPI()
                if mount != "/":
                    outer.mount(mount, app)
                with (
                    TestClient(outer) as client,
                    mock.patch.object(self.route, "fetch_remote_image", return_value=self.png) as fetch,
                ):
                    response = client.post(
                        mount.rstrip("/") + "/internal/remote-image", json={"url": "https://images.example/a.png"}
                    )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.content, self.png)
                self.assertEqual(response.headers["cache-control"], "no-store")
                self.assertEqual(response.headers["x-content-type-options"], "nosniff")
                fetch.assert_called_once_with("https://images.example/a.png")

    def test_anonymous_request_is_rejected_before_any_network_access(self):
        app = self.app(authenticated=True)
        with TestClient(app) as client, mock.patch.object(self.route, "fetch_remote_image") as fetch:
            response = client.post("/internal/remote-image", json={"url": "https://images.example/a.png"})
        self.assertEqual(response.status_code, 401)
        fetch.assert_not_called()

    def test_logged_in_browser_can_import_an_image(self):
        app = self.app(authenticated=True)
        with (
            TestClient(app) as client,
            mock.patch.object(self.route, "fetch_remote_image", return_value=self.png),
        ):
            client.cookies.set(f"access-token-{app.cookie_id}", "test-token")
            response = client.post("/internal/remote-image", json={"url": "https://images.example/a.png"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.content, self.png)

    def test_unicode_and_escaped_image_urls_reach_the_real_http_transport(self):
        cases = (
            (
                "http://images.example/写真 1.png?説明=猫&literal=%2F+plus",
                "/%E5%86%99%E7%9C%9F%201.png?%E8%AA%AC%E6%98%8E=%E7%8C%AB&literal=%2F+plus",
            ),
            ("http://images.example/already%20quoted.png?raw=%2f%2B", "/already%20quoted.png?raw=%2f%2B"),
        )
        for url, expected_target in cases:
            with self.subTest(url=url):
                transport = mock.Mock()
                transport.makefile.return_value = io.BytesIO(
                    b"HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: "
                    + str(len(self.png)).encode("ascii")
                    + b"\r\nConnection: close\r\n\r\n"
                    + self.png
                )
                with (
                    TestClient(self.app()) as client,
                    mock.patch.object(
                        socket,
                        "getaddrinfo",
                        return_value=[
                            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", 80))
                        ],
                    ),
                    mock.patch.object(socket, "create_connection", return_value=transport) as connect,
                ):
                    response = client.post("/internal/remote-image", json={"url": url})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.content, self.png)
                sent = b"".join(call.args[0] for call in transport.sendall.call_args_list)
                self.assertEqual(sent.split(b"\r\n", 1)[0], f"GET {expected_target} HTTP/1.1".encode("ascii"))
                self.assertEqual(connect.call_args.args[0], ("93.184.216.34", 80))

    def test_internal_network_urls_are_rejected_without_opening_a_connection(self):
        app = self.app()
        with TestClient(app) as client:
            for url in (
                "http://localhost/private",
                "file:///private.png",
                "https://user:password@example/a.png",
                "http://169.254.169.254/latest",
            ):
                with (
                    self.subTest(url=url),
                    mock.patch.object(
                        socket,
                        "getaddrinfo",
                        return_value=[
                            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("169.254.169.254", 80))
                        ],
                    ),
                    mock.patch.object(url_fetch, "_request_once") as request,
                ):
                    response = client.post("/internal/remote-image", json={"url": url})
                    self.assertEqual(response.status_code, 400, response.text)
                    self.assertNotIn("password", response.text)
                    self.assertNotIn("169.254.169.254", response.text)
                    request.assert_not_called()

    def test_get_and_cross_origin_simple_post_cannot_start_a_download(self):
        with TestClient(self.app()) as client, mock.patch.object(self.route, "fetch_remote_image") as fetch:
            self.assertEqual(client.get("/internal/remote-image").status_code, 405)
            for content_type in (None, "text/plain", "application/x-www-form-urlencoded", "multipart/form-data"):
                headers = {"Origin": "https://untrusted.example"}
                if content_type:
                    headers["Content-Type"] = content_type
                with self.subTest(content_type=content_type):
                    response = client.post(
                        "/internal/remote-image",
                        content=b'{"url":"https://images.example/a.png"}',
                        headers=headers,
                    )
                    self.assertEqual(response.status_code, 422)
        fetch.assert_not_called()

    def test_invalid_unicode_url_is_rejected_without_opening_a_connection(self):
        with (
            TestClient(self.app()) as client,
            mock.patch.object(
                socket,
                "getaddrinfo",
                return_value=[(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", 80))],
            ),
            mock.patch.object(socket, "create_connection") as connect,
        ):
            response = client.post(
                "/internal/remote-image",
                content=b'{"url":"http://images.example/\\ud800.png"}',
                headers={"Content-Type": "application/json"},
            )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json(), {"detail": "Invalid image import request."})
        connect.assert_not_called()


if __name__ == "__main__":
    main()
