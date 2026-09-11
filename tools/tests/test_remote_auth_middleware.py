import ast
import base64
import socket
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gradio as gr
import httpx
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect

from modules.aikimi_security.auth import (
    AuthenticationConfigError,
    RemoteAuthenticationMiddleware,
    RemoteAuthPolicy,
    install_remote_auth_middleware,
)

DIRECT_ROUTES = (
    ("GET", "/internal/sysinfo"),
    ("GET", "/internal/profile-startup"),
    ("GET", "/internal/pending-tasks"),
    ("POST", "/internal/progress"),
    ("GET", "/sdapi/v1/loras"),
    ("POST", "/sdapi/v1/refresh-loras"),
    ("POST", "/controlnet/detect"),
)

PUBLIC_WEB_ROUTES = (
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
    ("GET", "/pwa_icon/192"),
    ("GET", "/assets/app.js"),
    ("HEAD", "/assets/app.js"),
    ("GET", "/static/app.js"),
    ("GET", "/svelte/app.js"),
)


def options(
    *,
    remote: bool,
    api_only: bool,
    api_auth: str | None,
    api_auth_path: str | None = None,
):
    return SimpleNamespace(
        listen=remote,
        server_name=None,
        share=False,
        ngrok=None,
        nowebui=api_only,
        api=True,
        api_auth=api_auth,
        api_auth_path=api_auth_path,
    )


def basic_auth(values: tuple[str, str] | None = None) -> str:
    username, credential = values or ("api-user", "api-pass")
    encoded = base64.b64encode(f"{username}:{credential}".encode()).decode("ascii")
    return f"Basic {encoded}"


def route_handler(path: str):
    def handler():
        return {"path": path}

    return handler


def add_direct_routes(app: FastAPI) -> None:
    for method, path in DIRECT_ROUTES:
        app.add_api_route(path, route_handler(path), methods=[method])


def add_public_web_routes(app: FastAPI) -> None:
    for method, path in PUBLIC_WEB_ROUTES:
        app.add_api_route(path, route_handler(path), methods=[method])


class RemoteAuthMiddlewareTests(unittest.TestCase):
    def test_webui_installs_boundary_for_api_only_and_webui_workers(self):
        source = Path(__file__).resolve().parents[2] / "webui.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

        for function_name in ("api_only_worker", "webui_worker"):
            with self.subTest(function=function_name):
                function = functions[function_name]
                self.assertTrue(
                    any(
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "install_remote_auth_middleware"
                        for node in ast.walk(function)
                    )
                )

    def test_local_mode_does_not_install_or_change_direct_routes(self):
        app = FastAPI()
        add_direct_routes(app)

        installed = install_remote_auth_middleware(app, options(remote=False, api_only=False, api_auth=None))

        self.assertFalse(installed)
        self.assertFalse(any(item.cls is RemoteAuthenticationMiddleware for item in app.user_middleware))
        with TestClient(app) as client:
            for method, path in DIRECT_ROUTES:
                with self.subTest(method=method, path=path):
                    self.assertEqual(client.request(method, path).status_code, 200)

    def test_prelaunch_startup_event_exception_is_loopback_only(self):
        app = SimpleNamespace(startup_events_triggered=False, auth=None, auth_dependency=None)
        policy = RemoteAuthPolicy(app, (), api_only=False)
        base_scope = {
            "type": "http",
            "method": "GET",
            "path": "/gradio_api/startup-events",
        }

        self.assertTrue(policy.is_public_bootstrap_request({**base_scope, "client": ("127.0.0.1", 53000)}))
        self.assertFalse(policy.is_public_bootstrap_request({**base_scope, "client": ("203.0.113.10", 53000)}))
        app.startup_events_triggered = True
        self.assertFalse(policy.is_public_bootstrap_request({**base_scope, "client": ("127.0.0.1", 53000)}))

    def test_remote_boundary_is_active_before_gradio_auto_launch(self):
        from gradio.routes import App

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]

        prepared_app = App()
        remote_options = options(
            remote=True,
            api_only=False,
            api_auth="api-user:api-pass",
        )
        self.assertTrue(install_remote_auth_middleware(prepared_app, remote_options))
        observed: list[tuple[int, str | None]] = []
        startup_after_launch: list[int] = []

        def probe_before_launch_returns(local_url: str) -> bool:
            response = httpx.get(local_url.rstrip("/") + "/config", follow_redirects=False, timeout=5)
            observed.append((response.status_code, response.headers.get("www-authenticate")))
            return True

        with gr.Blocks(analytics_enabled=False) as demo:
            gr.Markdown("remote pre-listener auth boundary")

        try:
            with patch("webbrowser.open", side_effect=probe_before_launch_returns):
                app, local_url, _ = demo.launch(
                    server_name="127.0.0.1",
                    server_port=port,
                    prevent_thread_lock=True,
                    quiet=True,
                    inbrowser=True,
                    auth=[("web-user", "web-pass")],
                    _app=prepared_app,
                )
            startup_after_launch.append(
                httpx.get(
                    local_url.rstrip("/") + "/gradio_api/startup-events",
                    follow_redirects=False,
                    timeout=5,
                ).status_code
            )
        finally:
            demo.close()

        self.assertIs(app, prepared_app)
        self.assertEqual(observed, [(401, None)])
        self.assertEqual(startup_after_launch, [401])

    def test_remote_api_only_requires_basic_auth_for_every_route(self):
        app = FastAPI()
        add_direct_routes(app)
        self.assertTrue(
            install_remote_auth_middleware(
                app,
                options(
                    remote=True,
                    api_only=True,
                    api_auth="api-user:api-pass",
                ),
            )
        )
        # Prove that routes registered later by extensions remain inside the boundary.
        app.add_api_route(
            "/future-extension/action",
            route_handler("/future-extension/action"),
            methods=["POST"],
        )

        with TestClient(app) as client:
            malformed = client.get(
                "/internal/sysinfo",
                headers={"Authorization": "Basic not-base64"},
            )
            self.assertEqual(malformed.status_code, 401)
            for method, path in (*DIRECT_ROUTES, ("POST", "/future-extension/action")):
                with self.subTest(method=method, path=path):
                    response = client.request(method, path)
                    self.assertEqual(response.status_code, 401)
                    self.assertEqual(response.headers["www-authenticate"], "Basic")
                    self.assertNotIn("api-pass", response.text)

                    bad = client.request(
                        method,
                        path,
                        headers={"Authorization": basic_auth(("api-user", "wrong"))},
                    )
                    self.assertEqual(bad.status_code, 401)

                    allowed = client.request(
                        method,
                        path,
                        headers={"Authorization": basic_auth()},
                    )
                    self.assertEqual(allowed.status_code, 200)

            # API-only mode has no unauthenticated Gradio bootstrap exception.
            self.assertEqual(client.get("/").status_code, 401)

    def test_remote_webui_accepts_cookie_or_api_basic_for_direct_routes(self):
        app = FastAPI()
        app.auth = {"web-user": "hashed-password"}
        app.auth_dependency = None
        app.cookie_id = "cookie-id"
        app.tokens = {"valid-cookie": "web-user"}
        add_direct_routes(app)
        add_public_web_routes(app)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["https://client.example"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
        app.add_api_route(
            "/assets-private",
            route_handler("/assets-private"),
            methods=["GET"],
        )
        app.add_api_route(
            "/aikimi-assets/manifest.json",
            route_handler("/aikimi-assets/manifest.json"),
            methods=["GET"],
        )

        self.assertTrue(
            install_remote_auth_middleware(
                app,
                options(
                    remote=True,
                    api_only=False,
                    api_auth="api-user:api-pass",
                ),
            )
        )

        with TestClient(app) as anonymous:
            for method, path in PUBLIC_WEB_ROUTES:
                with self.subTest(public_method=method, public_path=path):
                    self.assertEqual(anonymous.request(method, path).status_code, 200)

            for method, path in DIRECT_ROUTES:
                with self.subTest(anonymous_method=method, anonymous_path=path):
                    response = anonymous.request(method, path)
                    self.assertEqual(response.status_code, 401)
                    self.assertNotIn("www-authenticate", response.headers)

            self.assertEqual(anonymous.get("/assets-private").status_code, 401)
            self.assertEqual(anonymous.get("/aikimi-assets/manifest.json").status_code, 401)
            preflight = anonymous.options(
                "/internal/sysinfo",
                headers={
                    "Origin": "https://client.example",
                    "Access-Control-Request-Method": "GET",
                },
            )
            self.assertEqual(preflight.status_code, 200)

        with TestClient(app) as cookie_client:
            cookie_client.cookies.set("access-token-unsecure-cookie-id", "valid-cookie")
            for method, path in DIRECT_ROUTES:
                with self.subTest(cookie_method=method, cookie_path=path):
                    self.assertEqual(cookie_client.request(method, path).status_code, 200)

        with TestClient(app) as basic_client:
            for method, path in DIRECT_ROUTES:
                with self.subTest(basic_method=method, basic_path=path):
                    self.assertEqual(
                        basic_client.request(
                            method,
                            path,
                            headers={"Authorization": basic_auth()},
                        ).status_code,
                        200,
                    )

    def test_remote_webui_awaits_async_gradio_auth_dependency(self):
        app = FastAPI()

        async def auth_dependency(connection):
            return "web-user" if connection.headers.get("X-Web-Session") == "valid" else None

        app.auth = None
        app.auth_dependency = auth_dependency
        app.add_api_route("/internal/sysinfo", route_handler("/internal/sysinfo"), methods=["GET"])
        install_remote_auth_middleware(
            app,
            options(remote=True, api_only=False, api_auth="api-user:api-pass"),
        )

        with TestClient(app) as client:
            self.assertEqual(client.get("/internal/sysinfo").status_code, 401)
            self.assertEqual(
                client.get(
                    "/internal/sysinfo",
                    headers={"X-Web-Session": "valid"},
                ).status_code,
                200,
            )

    def test_gradio_6_login_bootstrap_reaches_cookie_protected_direct_route(self):
        import gradio as gr
        from gradio.routes import App

        # Gradio 6.17.3 creates short-lived component event loops while building
        # a Blocks app without launching it. They are dependency-owned and do not
        # belong to the middleware under test.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            web_values = ("web-user", "web-pass")
            with gr.Blocks() as blocks:
                gr.Markdown("Remote authentication test")
            blocks.auth = [web_values]
            blocks.auth_message = None
            app = App.create_app(blocks)
            app.add_api_route(
                "/internal/sysinfo",
                route_handler("/internal/sysinfo"),
                methods=["GET"],
            )
            install_remote_auth_middleware(
                app,
                options(
                    remote=True,
                    api_only=False,
                    api_auth="api-user:api-pass",
                ),
            )

            with TestClient(app) as client:
                self.assertEqual(client.get("/").status_code, 200)
                self.assertEqual(client.get("/manifest.json").status_code, 200)
                login_check = client.get("/gradio_api/login_check")
                self.assertEqual(login_check.status_code, 401)
                self.assertIsInstance(login_check.json()["detail"], dict)
                self.assertEqual(client.get("/internal/sysinfo").status_code, 401)

                login = client.post(
                    "/login",
                    data=dict(zip(("username", "password"), web_values, strict=True)),
                )
                self.assertEqual(login.status_code, 200)
                self.assertTrue(login.json()["success"])
                self.assertEqual(client.get("/internal/sysinfo").status_code, 200)

    def test_remote_middleware_is_idempotent_and_repr_hides_password(self):
        app = FastAPI()
        remote_options = options(
            remote=True,
            api_only=True,
            api_auth="api-user:do-not-print-this",
        )

        self.assertTrue(install_remote_auth_middleware(app, remote_options))
        self.assertFalse(install_remote_auth_middleware(app, remote_options))
        matching = [item for item in app.user_middleware if item.cls is RemoteAuthenticationMiddleware]
        self.assertEqual(len(matching), 1)
        self.assertNotIn("do-not-print-this", repr(matching[0].kwargs["policy"]))

    def test_remote_api_only_fails_closed_without_api_credentials(self):
        app = FastAPI()

        with self.assertRaisesRegex(AuthenticationConfigError, "requires API Basic authentication"):
            install_remote_auth_middleware(app, options(remote=True, api_only=True, api_auth=None))

    def test_remote_api_only_rejects_unauthenticated_websocket(self):
        app = FastAPI()

        @app.websocket("/socket")
        async def socket(websocket: WebSocket):
            await websocket.accept()
            await websocket.send_text("ready")
            await websocket.close()

        install_remote_auth_middleware(
            app,
            options(
                remote=True,
                api_only=True,
                api_auth="api-user:api-pass",
            ),
        )

        with TestClient(app) as client:
            with self.assertRaises(WebSocketDisconnect) as error:
                with client.websocket_connect("/socket"):
                    pass
            self.assertEqual(error.exception.code, 4401)

            with client.websocket_connect("/socket", headers={"Authorization": basic_auth()}) as websocket:
                self.assertEqual(websocket.receive_text(), "ready")

    def test_remote_api_only_accepts_credentials_from_auth_file(self):
        with tempfile.TemporaryDirectory() as directory:
            auth_file = Path(directory) / "api-auth.txt"
            auth_file.write_text("api-user:api-pass\n", encoding="utf-8")
            app = FastAPI()
            app.add_api_route(
                "/extension",
                route_handler("/extension"),
                methods=["GET"],
            )
            install_remote_auth_middleware(
                app,
                options(
                    remote=True,
                    api_only=True,
                    api_auth=None,
                    api_auth_path=str(auth_file),
                ),
            )

            with TestClient(app) as client:
                self.assertEqual(client.get("/extension").status_code, 401)
                self.assertEqual(
                    client.get(
                        "/extension",
                        headers={"Authorization": basic_auth()},
                    ).status_code,
                    200,
                )


if __name__ == "__main__":
    unittest.main()
