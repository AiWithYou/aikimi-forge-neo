"""Every Ming route authenticates and exposes image bytes without local paths."""

import base64
import io
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from modules_forge.ming_image.api import register_api


class ApiTests(unittest.TestCase):
    def test_reference_uploads_are_decoded_and_paths_are_rejected_before_dispatch(self):
        app = FastAPI()
        register_api(app)
        options = SimpleNamespace(cmd_opts=SimpleNamespace(api_auth=None, api_auth_path=None))
        stream = io.BytesIO()
        Image.new("RGB", (512, 768), "red").save(stream, format="PNG")
        encoded = base64.b64encode(stream.getvalue()).decode()
        with (
            patch.dict(sys.modules, {"modules.shared_cmd_options": options}),
            patch("modules_forge.ming_image.api.SERVICE") as service,
            TestClient(app) as client,
        ):
            service.start.return_value = "abc"
            response = client.post(
                "/sdapi/v1/ming-image/jobs", json={"prompt": "blue coat", "mode": "img2img", "images": [encoded]}
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(service.start.call_args.args[0]["images"][0].size, (512, 768))
            service.start.reset_mock()
            for value in ("/private/image.png", "https://example.test/image.png"):
                response = client.post(
                    "/sdapi/v1/ming-image/jobs", json={"prompt": "edit", "mode": "img2img", "images": [value]}
                )
                self.assertEqual(response.status_code, 422)
                self.assertNotIn(value, response.text)
            service.start.assert_not_called()

    def test_cfg_below_one_returns_422(self):
        app = FastAPI()
        register_api(app)
        options = SimpleNamespace(cmd_opts=SimpleNamespace(api_auth=None, api_auth_path=None))
        with patch.dict(sys.modules, {"modules.shared_cmd_options": options}), TestClient(app) as client:
            for cfg in (0, 0.5):
                response = client.post("/sdapi/v1/ming-image/jobs", json={"prompt": "design", "cfg": cfg})
                self.assertEqual(response.status_code, 422)
                self.assertIn("CFG", response.text)

    def test_all_routes_require_configured_api_credentials(self):
        app = FastAPI()
        register_api(app)
        options = SimpleNamespace(cmd_opts=SimpleNamespace(api_auth="ming:test", api_auth_path=None))
        with patch.dict(sys.modules, {"modules.shared_cmd_options": options}), TestClient(app) as client:
            for method, path, body in (
                ("POST", "/sdapi/v1/ming-image/jobs", {"prompt": "a fox"}),
                ("GET", "/sdapi/v1/ming-image/jobs/missing", None),
                ("POST", "/sdapi/v1/ming-image/jobs/missing/cancel", None),
            ):
                self.assertEqual(client.request(method, path, json=body).status_code, 401)
            self.assertEqual(client.get("/sdapi/v1/ming-image/jobs/missing", auth=("ming", "test")).status_code, 404)

    def test_job_routes_submit_poll_and_cancel(self):
        app = FastAPI()
        register_api(app)
        options = SimpleNamespace(cmd_opts=SimpleNamespace(api_auth=None, api_auth_path=None))
        with (
            patch.dict(sys.modules, {"modules.shared_cmd_options": options}),
            patch("modules_forge.ming_image.api.SERVICE") as service,
            TestClient(app) as client,
        ):
            service.start.return_value = "abc"
            service.poll.return_value = {"status": "running", "id": "abc", "stage": "sampling", "directory": "private"}
            service.cancel.return_value = "中断処理中…"
            response = client.post("/sdapi/v1/ming-image/jobs", json={"prompt": "a fox", "profile": "comfy_org_w4a8"})
            self.assertEqual(response.json(), {"id": "abc", "status": "running"})
            service.start.assert_called_once_with({"prompt": "a fox", "profile": "comfy_org_w4a8"})
            self.assertEqual(
                client.get("/sdapi/v1/ming-image/jobs/abc").json(),
                {"status": "running", "id": "abc", "stage": "sampling"},
            )
            self.assertIn("中断", client.post("/sdapi/v1/ming-image/jobs/abc/cancel").json()["message"])
