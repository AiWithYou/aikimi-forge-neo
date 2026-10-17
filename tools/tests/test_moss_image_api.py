"""Every MOSS route authenticates and exposes image bytes without local paths."""

import base64
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from modules_forge.moss_image.api import public_result, register_api


class ApiTests(unittest.TestCase):
    def test_all_routes_require_configured_api_credentials(self):
        app = FastAPI()
        register_api(app)
        options = SimpleNamespace(cmd_opts=SimpleNamespace(api_auth="moss:test", api_auth_path=None))
        with patch.dict(sys.modules, {"modules.shared_cmd_options": options}), TestClient(app) as client:
            for method, path, body in (
                ("POST", "/sdapi/v1/moss-image/jobs", {"prompt": "a fox"}),
                ("GET", "/sdapi/v1/moss-image/jobs/missing", None),
                ("POST", "/sdapi/v1/moss-image/jobs/missing/cancel", None),
            ):
                self.assertEqual(client.request(method, path, json=body).status_code, 401)
            self.assertEqual(client.get("/sdapi/v1/moss-image/jobs/missing", auth=("moss", "test")).status_code, 404)

    def test_job_routes_submit_poll_and_cancel(self):
        app = FastAPI()
        register_api(app)
        options = SimpleNamespace(cmd_opts=SimpleNamespace(api_auth=None, api_auth_path=None))
        with (
            patch.dict(sys.modules, {"modules.shared_cmd_options": options}),
            patch("modules_forge.moss_image.api.SERVICE") as service,
            TestClient(app) as client,
        ):
            service.start.return_value = "abc"
            service.poll.return_value = {"status": "running", "id": "abc", "stage": "sampling", "directory": "private"}
            service.cancel.return_value = "中断処理中…"
            response = client.post("/sdapi/v1/moss-image/jobs", json={"prompt": "a fox", "skip_layer": False})
            self.assertEqual(response.json(), {"id": "abc", "status": "running"})
            service.start.assert_called_once_with({"prompt": "a fox", "skip_layer": False})
            self.assertEqual(
                client.get("/sdapi/v1/moss-image/jobs/abc").json(),
                {"status": "running", "id": "abc", "stage": "sampling"},
            )
            self.assertIn("中断", client.post("/sdapi/v1/moss-image/jobs/abc/cancel").json()["message"])

    def test_invalid_request_is_422(self):
        app = FastAPI()
        register_api(app)
        options = SimpleNamespace(cmd_opts=SimpleNamespace(api_auth=None, api_auth_path=None))
        with (
            patch.dict(sys.modules, {"modules.shared_cmd_options": options}),
            patch("modules_forge.moss_image.api.SERVICE.start", side_effect=ValueError("invalid")),
            TestClient(app) as client,
        ):
            self.assertEqual(client.post("/sdapi/v1/moss-image/jobs", json={"prompt": ""}).status_code, 422)

    def test_completed_image_is_base64_with_public_metadata_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "output.png"
            Image.new("RGB", (16, 16)).save(path)
            result = public_result(
                {
                    "status": "complete",
                    "id": "abc",
                    "image_path": str(path),
                    "directory": temporary,
                    "files": [str(path)],
                    "root": temporary,
                    "prompt": "a fox",
                    "negative_prompt": "blur",
                    "shift": 5,
                    "skip_layer": True,
                }
            )
            self.assertEqual(base64.b64decode(result.pop("image")), path.read_bytes())
            self.assertEqual(
                result,
                {
                    "status": "complete",
                    "id": "abc",
                    "prompt": "a fox",
                    "negative_prompt": "blur",
                    "shift": 5,
                    "skip_layer": True,
                },
            )

    def test_running_job_never_reads_image_path(self):
        self.assertEqual(
            public_result({"status": "running", "directory": "private", "image_path": "private.png"}),
            {"status": "running"},
        )

    def test_worker_error_does_not_disclose_filesystem_details(self):
        result = public_result({"status": "error", "error": "Cannot load H:/private/model.safetensors"})
        self.assertTrue(result["error"])
        self.assertNotIn("H:/private", result["error"])

    def test_reproduction_metadata_survives_public_filter(self):
        metadata = {
            "actual_steps": 48,
            "scheduler": "MOSSImageScheduler",
            "precision": "bf16",
            "vae_precision": "fp32",
            "token_limit": 4096,
            "prompt_truncated": False,
            "generator_device": "cpu",
            "prompt_tokens": 100,
            "negative_prompt_tokens": 12,
            "text_compute_precision": "torch.bfloat16",
            "text_conditioning_precision": "torch.float32",
        }
        self.assertEqual(public_result({"status": "running", **metadata}), {"status": "running", **metadata})

    def test_lora_public_records_keep_identity_strength_and_precision_without_paths(self):
        identity = {
            "name": "nested/style.safetensors",
            "sha256": "a" * 64,
            "size": 100,
            "mtime_ns": 7,
            "targets": 2,
            "rank": 1,
        }
        record = {**identity, "strength": 0.5, "applied_targets": 2}
        result = public_result(
            {
                "status": "running",
                "precision": "int8",
                "text_precision": "w4a8",
                "loras": [{"name": identity["name"], "strength": 0.5}],
                "lora_identities": [{**identity, "path": "H:/private/style.safetensors"}],
                "lora": [{**record, "path": "H:/private/style.safetensors"}],
                "lora_applied_targets": 2,
            }
        )
        self.assertEqual(result["text_precision"], "w4a8")
        self.assertEqual(result["lora_identities"], [identity])
        self.assertEqual(result["lora"], [record])
        self.assertEqual(result["lora_applied_targets"], 2)
        self.assertNotIn("H:/private", str(result))

    def test_runtime_and_asset_revisions_survive_without_internal_source_paths(self):
        source = {
            "repo_id": "community/model",
            "revision": "a" * 40,
            "filename": "model.safetensors",
            "sha256": "b" * 64,
        }
        result = public_result(
            {
                "status": "running",
                "comfy_revision": "c" * 40,
                "runtime_backend": "ComfyUI",
                "sampler": "euler",
                "weight_sources": {"dit": {**source, "path": "H:/private/model"}},
            }
        )
        self.assertEqual(result["weight_sources"], {"dit": source})
        self.assertEqual(result["comfy_revision"], "c" * 40)
        self.assertEqual(result["sampler"], "euler")
        self.assertNotIn("H:/private", str(result))


if __name__ == "__main__":
    unittest.main()
