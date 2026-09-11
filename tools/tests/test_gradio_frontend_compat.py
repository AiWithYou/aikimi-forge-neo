from __future__ import annotations

import hashlib
import socket
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urljoin
from urllib.request import urlopen

import gradio
import httpx
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from fastapi.testclient import TestClient

from modules import gradio_frontend_compat
from modules.aikimi_security.gradio_file_guard import install_gradio_file_url_guard


class GradioFrontendCompatibilityTests(unittest.TestCase):
    def test_webui_validates_patch_before_starting_gradio_listener(self):
        root = Path(__file__).resolve().parents[2]
        source = (root / "webui.py").read_text(encoding="utf-8")

        self.assertLess(
            source.index("patched_tabs = build_patched_tabs_asset()"),
            source.index("app, local_url, share_url = shared.demo.launch("),
        )

    def test_exact_pinned_asset_is_patched_without_modifying_site_packages(self):
        asset_path = (
            Path(gradio.__file__).resolve().parent
            / "templates"
            / "frontend"
            / "assets"
            / gradio_frontend_compat.TABS_ASSET_NAME
        )
        before = asset_path.read_bytes()

        patched = gradio_frontend_compat.build_patched_tabs_asset()

        self.assertEqual(hashlib.sha256(before).hexdigest(), patched.original_sha256)
        self.assertEqual(before, asset_path.read_bytes())
        self.assertNotEqual(before, patched.content)
        self.assertNotIn(gradio_frontend_compat._ORIGINAL_OVERFLOW_FUNCTION, patched.content)
        self.assertIn(gradio_frontend_compat._NO_OVERFLOW_FUNCTION, patched.content)
        self.assertNotIn(gradio_frontend_compat._ORIGINAL_INITIAL_TAB_SYNC, patched.content)
        self.assertIn(gradio_frontend_compat._STATIC_INITIAL_TAB_SYNC, patched.content)
        self.assertNotIn(gradio_frontend_compat._ORIGINAL_TAB_REGISTRATION, patched.content)
        self.assertIn(gradio_frontend_compat._STATIC_TAB_REGISTRATION, patched.content)
        self.assertEqual(hashlib.sha256(patched.content).hexdigest(), patched.patched_sha256)
        self.assertEqual(patched.patched_sha256, gradio_frontend_compat.PATCHED_ASSET_SHA256)

    def test_preconfigured_route_is_available_during_gradio_auto_launch(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]

        asset = gradio_frontend_compat.build_patched_tabs_asset()
        prepared_app = gradio_frontend_compat.create_gradio_compatibility_app(asset)
        self.assertIs(prepared_app.router.default_response_class, ORJSONResponse)
        install_gradio_file_url_guard(prepared_app)
        fetched_during_open: list[bytes] = []
        external_status_during_open: list[tuple[int, str | None]] = []

        def fetch_from_browser_open(local_url: str) -> bool:
            with urlopen(urljoin(local_url, f"assets/{asset.filename}"), timeout=5) as response:  # noqa: S310
                fetched_during_open.append(response.read())
            external = httpx.get(
                local_url.rstrip("/") + "/gradio_api/file=https://redirect.invalid/pre-listener",
                follow_redirects=False,
                timeout=5,
            )
            external_status_during_open.append((external.status_code, external.headers.get("location")))
            return True

        with gradio.Blocks(analytics_enabled=False) as demo:
            gradio.Markdown("pre-listener compatibility route")

        try:
            with patch("webbrowser.open", side_effect=fetch_from_browser_open):
                app, _, _ = demo.launch(
                    server_name="127.0.0.1",
                    server_port=port,
                    prevent_thread_lock=True,
                    quiet=True,
                    inbrowser=True,
                    _app=prepared_app,
                )
        finally:
            demo.close()

        self.assertIs(app, prepared_app)
        self.assertEqual(fetched_during_open, [asset.content])
        self.assertEqual(external_status_during_open, [(403, None)])

    def test_webui_opens_browser_only_after_routes_and_middleware_are_ready(self):
        root = Path(__file__).resolve().parents[2]
        source = (root / "webui.py").read_text(encoding="utf-8")

        self.assertIn("inbrowser=False", source)
        self.assertIn("debug=False", source)
        self.assertIn("show_error=cmd_opts.gradio_debug", source)
        self.assertIn("ssr_mode=False", source)
        self.assertIn('os.environ["GRADIO_DEBUG"] = "0"', source)
        self.assertIn('os.environ["GRADIO_DEBUG"] = previous_gradio_debug', source)
        self.assertIn("tunnel_baseline = gradio_runtime.tunnel_snapshot()", source)
        self.assertNotIn("shared.demo.close()", source)
        self.assertGreaterEqual(source.count("gradio_runtime.close_gradio_runtime(shared.demo, tunnel_baseline)"), 5)
        launch_block = source.split("app, local_url, share_url = shared.demo.launch(", 1)[1].split(
            'startup_timer.record("gradio launch")',
            1,
        )[0]
        self.assertIn("except Exception:", launch_block)
        self.assertIn("gradio_runtime.close_gradio_runtime(shared.demo, tunnel_baseline)", launch_block)
        self.assertLess(
            source.index("install_gradio_file_url_guard(prepared_app)"), source.index("shared.demo.launch(")
        )
        self.assertLess(source.index("ui.setup_ui_api(app)"), source.index("webbrowser.open(browser_url)"))
        self.assertLess(
            source.index('with startup_timer.subcategory("app_started_callback")'),
            source.index("webbrowser.open(browser_url)"),
        )
        post_launch_block = source.split('startup_timer.record("gradio launch")', 1)[1].split(
            "server_command = shared.state.wait_for_server_command",
            1,
        )[0]
        self.assertIn("initialize_util.setup_middleware(app)", post_launch_block)
        self.assertIn("ui.setup_ui_api(app)", post_launch_block)
        self.assertIn("webbrowser.open(browser_url)", post_launch_block)
        self.assertIn("except Exception:", post_launch_block)
        self.assertIn("gradio_runtime.close_gradio_runtime(shared.demo, tunnel_baseline)", post_launch_block)

    def test_version_mismatch_fails_closed(self):
        with patch.object(gradio, "__version__", "6.18.0"):
            with self.assertRaisesRegex(
                gradio_frontend_compat.GradioFrontendCompatibilityError,
                "not supported",
            ):
                gradio_frontend_compat.build_patched_tabs_asset()

    def test_hash_mismatch_fails_closed(self):
        with patch.object(gradio_frontend_compat, "ORIGINAL_ASSET_SHA256", "0" * 64):
            with self.assertRaisesRegex(
                gradio_frontend_compat.GradioFrontendCompatibilityError,
                "does not match",
            ):
                gradio_frontend_compat.build_patched_tabs_asset()

    def test_tampered_prebuilt_asset_fails_closed(self):
        asset = gradio_frontend_compat.build_patched_tabs_asset()
        tampered = gradio_frontend_compat.PatchedFrontendAsset(
            filename=asset.filename,
            content=asset.content + b" ",
            original_sha256=asset.original_sha256,
            patched_sha256=asset.patched_sha256,
        )

        with self.assertRaisesRegex(
            gradio_frontend_compat.GradioFrontendCompatibilityError,
            "integrity validation",
        ):
            gradio_frontend_compat.install_gradio_tabs_compatibility_route(
                FastAPI(),
                tampered,
            )

    def test_exact_route_precedes_generic_assets_and_supports_etag(self):
        app = FastAPI()

        @app.get("/assets/{path:path}")
        async def generic_asset(path: str):
            return {"generic": path}

        asset = gradio_frontend_compat.build_patched_tabs_asset()
        installed = gradio_frontend_compat.install_gradio_tabs_compatibility_route(app, asset)
        self.assertIs(installed, asset)
        client = TestClient(app)

        response = client.get(f"/assets/{asset.filename}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, asset.content)
        self.assertEqual(response.headers["content-length"], str(len(asset.content)))
        self.assertEqual(response.headers["etag"], f'"{asset.patched_sha256}"')
        self.assertEqual(response.headers["x-aikimi-gradio-compat"], "6.17.3")
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")

        head = client.head(f"/assets/{asset.filename}")
        self.assertEqual(head.status_code, 200)
        self.assertEqual(head.content, b"")
        self.assertEqual(head.headers["content-length"], str(len(asset.content)))
        self.assertEqual(head.headers["etag"], f'"{asset.patched_sha256}"')

        cached = client.get(
            f"/assets/{asset.filename}",
            headers={"If-None-Match": f'"{asset.patched_sha256}"'},
        )
        self.assertEqual(cached.status_code, 304)
        self.assertEqual(client.get("/assets/other.js").json(), {"generic": "other.js"})

    def test_route_is_idempotent_and_works_with_asgi_root_path(self):
        app = FastAPI(root_path="/aikimi")
        app.router.routes.insert(0, object())
        first = gradio_frontend_compat.install_gradio_tabs_compatibility_route(app)
        second = gradio_frontend_compat.install_gradio_tabs_compatibility_route(app)

        self.assertIs(first, second)
        self.assertEqual(
            sum(getattr(route, "name", None) == gradio_frontend_compat.PATCH_ROUTE_NAME for route in app.router.routes),
            1,
        )
        response = TestClient(app, root_path="/aikimi").get(f"/assets/{first.filename}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, first.content)


if __name__ == "__main__":
    unittest.main()
