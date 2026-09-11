import base64
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gradio as gr
from fastapi import FastAPI
from fastapi.testclient import TestClient

from modules.aikimi_security.auth import install_remote_auth_middleware
from modules.aikimi_security.gradio_file_guard import (
    GradioExternalFileURLGuardMiddleware,
    install_gradio_file_url_guard,
)


def _remote_api_options():
    return SimpleNamespace(
        listen=True,
        server_name=None,
        share=False,
        ngrok=None,
        nowebui=True,
        api=True,
        api_auth="api-user:api-pass",
        api_auth_path=None,
    )


def _basic_auth() -> str:
    encoded = base64.b64encode(b"api-user:api-pass").decode("ascii")
    return f"Basic {encoded}"


class GradioFileURLGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.asset = Path(self.temp.name) / "asset.js"
        self.asset.write_text("window.guardAsset = true;", encoding="utf-8")

    def build_app(self, *, subpath: str = "/aikimi"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            with gr.Blocks() as blocks:
                gr.Markdown("file guard")
        app = gr.mount_gradio_app(
            FastAPI(),
            blocks,
            path=subpath,
            allowed_paths=[str(self.asset)],
        )
        install_gradio_file_url_guard(app)
        return app

    def test_external_current_and_legacy_routes_are_rejected_without_redirect(self):
        cases = (
            "/aikimi/gradio_api/file=http://redirect.invalid/path",
            "/aikimi/gradio_api/file=https://redirect.invalid/path",
            "/aikimi/gradio_api/file=//redirect.invalid/path",
            "/aikimi/gradio_api/file/https://redirect.invalid/legacy",
            "/aikimi/gradio_api/file=HTTPS://redirect.invalid/mixed-case",
            "/aikimi/gradio_api/file=https://user:private@redirect.invalid/path",
        )
        with TestClient(self.build_app()) as client:
            for method in ("GET", "HEAD"):
                for path in cases:
                    with self.subTest(method=method, path=path):
                        response = client.request(method, path, follow_redirects=False)
                        self.assertEqual(response.status_code, 403)
                        self.assertNotIn("location", response.headers)
                        self.assertNotIn("redirect.invalid", response.text)
                        self.assertNotIn("private", response.text)

    def test_percent_encoded_external_targets_are_rejected(self):
        cases = (
            "/aikimi/gradio_api/file=https%3A%2F%2Fredirect.invalid/path",
            "/aikimi/gradio_api/file=http%253A%252F%252Fredirect.invalid/double",
            "/aikimi/gradio_api/file=%2F%2Fredirect.invalid/protocol-relative",
            "/aikimi/gradio_api/file=https%3A%2F%2Fuser%3Aprivate%40redirect.invalid/path",
            "/aikimi/%67radio_api%2Ffile%3Dhttps%3A%2F%2Fredirect.invalid/encoded-route",
        )
        with TestClient(self.build_app()) as client:
            for path in cases:
                with self.subTest(path=path):
                    response = client.get(path, follow_redirects=False)
                    self.assertEqual(response.status_code, 403)
                    self.assertNotIn("location", response.headers)
                    self.assertNotIn("redirect.invalid", response.text)
                    self.assertNotIn("private", response.text)

    def test_exact_local_asset_is_unchanged_at_root_and_subpath(self):
        for subpath in ("/", "/aikimi"):
            with self.subTest(subpath=subpath), TestClient(self.build_app(subpath=subpath)) as client:
                route = "gradio_api/file=" + self.asset.as_posix()
                request_path = f"/{route}" if subpath == "/" else f"{subpath}/{route}"
                response = client.get(request_path)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertIn("window.guardAsset", response.text)

    def test_remote_authentication_still_runs_before_the_guard(self):
        app = self.build_app()
        install_remote_auth_middleware(app, _remote_api_options())
        external = "/aikimi/gradio_api/file=https://redirect.invalid/path"
        local = "/aikimi/gradio_api/file=" + self.asset.as_posix()

        with TestClient(app) as client:
            anonymous = client.get(external, follow_redirects=False)
            self.assertEqual(anonymous.status_code, 401)
            self.assertEqual(anonymous.headers.get("www-authenticate"), "Basic")

            authenticated_external = client.get(
                external,
                headers={"Authorization": _basic_auth()},
                follow_redirects=False,
            )
            self.assertEqual(authenticated_external.status_code, 403)
            self.assertNotIn("location", authenticated_external.headers)

            authenticated_local = client.get(
                local,
                headers={"Authorization": _basic_auth()},
            )
            self.assertEqual(authenticated_local.status_code, 200)

    def test_gradio_login_dependency_remains_the_unauthenticated_boundary(self):
        from gradio.routes import App

        credentials = ("web-user", "web-pass")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            with gr.Blocks() as blocks:
                gr.Markdown("authenticated file guard")
        blocks.auth = [credentials]
        blocks.auth_message = None
        blocks.allowed_paths = [str(self.asset)]
        app = App.create_app(blocks)
        install_gradio_file_url_guard(app)
        external = "/gradio_api/file=https://redirect.invalid/path"

        with TestClient(app) as client:
            anonymous = client.get(external, follow_redirects=False)
            self.assertEqual(anonymous.status_code, 401)
            self.assertNotIn("location", anonymous.headers)

            login = client.post(
                "/login",
                data=dict(zip(("username", "password"), credentials, strict=True)),
            )
            self.assertEqual(login.status_code, 200)
            self.assertTrue(login.json()["success"])

            authenticated = client.get(external, follow_redirects=False)
            self.assertEqual(authenticated.status_code, 403)
            self.assertNotIn("location", authenticated.headers)

    def test_installation_is_idempotent_and_setup_middleware_integrates_guard(self):
        from modules import initialize_util

        app = FastAPI()
        with patch.object(initialize_util, "configure_cors_middleware"):
            initialize_util.setup_middleware(app)
        self.assertTrue(any(item.cls is GradioExternalFileURLGuardMiddleware for item in app.user_middleware))
        self.assertFalse(install_gradio_file_url_guard(app))
        self.assertEqual(
            sum(item.cls is GradioExternalFileURLGuardMiddleware for item in app.user_middleware),
            1,
        )


if __name__ == "__main__":
    unittest.main()
