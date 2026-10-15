"""The FLUX 3 API accepts image bytes and never caller filesystem paths."""

import base64
import io
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from modules_forge.flux3_klein.api import decode_references, public_result, register_api


class ApiTests(unittest.TestCase):
    def test_all_routes_require_configured_api_credentials(self):
        app = FastAPI()
        register_api(app)
        options = SimpleNamespace(cmd_opts=SimpleNamespace(api_auth="flux:test", api_auth_path=None))
        with patch.dict(sys.modules, {"modules.shared_cmd_options": options}), TestClient(app) as client:
            for method, path, body in (
                ("POST", "/sdapi/v1/flux3/jobs", {"prompt": "a fox"}),
                ("GET", "/sdapi/v1/flux3/jobs/missing", None),
                ("POST", "/sdapi/v1/flux3/jobs/missing/cancel", None),
            ):
                response = client.request(method, path, json=body)
                self.assertEqual(response.status_code, 401)
            response = client.get("/sdapi/v1/flux3/jobs/missing", auth=("flux", "test"))
            self.assertEqual(response.status_code, 404)

    def test_decode_pixels_and_reject_urls_paths_and_excess_images(self):
        stream = io.BytesIO()
        Image.new("RGB", (16, 32), "red").save(stream, format="PNG")
        encoded = base64.b64encode(stream.getvalue()).decode("ascii")
        images = decode_references([encoded, "data:image/png;base64," + encoded])
        self.assertEqual([image.size for image in images], [(16, 32), (16, 32)])
        for value in (["C:/private.png"], ["https://example.org/image.png"], [encoded] * 11):
            with self.subTest(value=value[:1]), self.assertRaises(ValueError):
                decode_references(value)

    def test_status_does_not_disclose_private_paths(self):
        result = public_result(
            {"status": "running", "id": "abc", "directory": "C:/private", "image_path": "private.png", "stage": "text"}
        )
        self.assertEqual(result, {"status": "running", "id": "abc", "stage": "text"})


if __name__ == "__main__":
    unittest.main()
