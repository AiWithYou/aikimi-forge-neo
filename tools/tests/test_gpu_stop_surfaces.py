"""Public stop responses and memory polling, without importing Forge GPU code."""

import ast
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from modules_forge import gpu_fault
from modules_forge.moss_image.api import public_result, register_api


class StopSurfaceTests(unittest.TestCase):
    def test_restart_gate_precedes_forge_gpu_warmup(self):
        source = Path("modules_forge/initialization.py").read_text(encoding="utf-8")
        node = next(
            n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == "initialize_forge"
        )
        namespace = {"INITIALIZED": False}
        exec(  # noqa: S102 -- Fixed local source under a CPU-only guard.
            compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "init-contract", "exec"),
            namespace,
        )
        with tempfile.TemporaryDirectory() as directory:
            state = gpu_fault.FaultState(Path(directory) / "fault.json")
            state.trip("gpu_execution", "test")
            with patch.object(gpu_fault, "STATE", state):
                with self.assertRaises(gpu_fault.GPUFaultError):
                    namespace["initialize_forge"]()
        self.assertFalse(namespace["INITIALIZED"])

    def test_public_error_codes_are_allowlisted(self):
        result = public_result({"status": "error", "error": "private/path", "error_code": "gpu_stopped"})
        self.assertEqual(result["error"], gpu_fault.MESSAGE)
        self.assertEqual(result["error_code"], "gpu_stopped")
        unknown = public_result({"status": "error", "error": "private/path", "error_code": "private/path"})
        self.assertNotIn("error_code", unknown)
        self.assertNotIn("private", unknown["error"])

    def test_policy_metadata_cannot_expose_paths(self):
        result = public_result(
            {
                "inference_policy": {
                    "sdpa_backends": ["efficient"],
                    "situ_activation": "eager_fp32",
                    "sdpa_calls_checked": 5,
                    "kernel_identity_observed": False,
                    "path": "private",
                }
            }
        )
        self.assertEqual(
            result["inference_policy"],
            {
                "sdpa_backends": ["efficient"],
                "situ_activation": "eager_fp32",
                "sdpa_calls_checked": 5,
                "kernel_identity_observed": False,
            },
        )

    def test_stopped_start_returns_503(self):
        app = FastAPI()
        register_api(app)
        options = SimpleNamespace(cmd_opts=SimpleNamespace(api_auth=None, api_auth_path=None))
        with (
            patch.dict(sys.modules, {"modules.shared_cmd_options": options}),
            patch("modules_forge.moss_image.api.SERVICE.start", side_effect=gpu_fault.GPUFaultError()),
            TestClient(app) as client,
        ):
            response = client.post("/sdapi/v1/moss-image/jobs", json={"prompt": "cat"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], gpu_fault.MESSAGE)

    def test_memory_poll_does_not_even_import_torch_when_stopped(self):
        source = Path("modules/api/api.py").read_text(encoding="utf-8")
        node = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == "get_memory")
        namespace = {"models": SimpleNamespace(MemoryResponse=lambda **values: values)}
        exec(  # noqa: S102 -- Fixed local source under a CPU-only guard.
            compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "memory-contract", "exec"),
            namespace,
        )
        with tempfile.TemporaryDirectory() as directory:
            state = gpu_fault.FaultState(Path(directory) / "fault.json")
            state.trip("gpu_execution", "test")
            original = __import__

            def guarded(name, *args, **kwargs):
                if name == "torch":
                    self.fail("memory polling imported Torch after a fault")
                return original(name, *args, **kwargs)

            with patch.object(gpu_fault, "STATE", state), patch("builtins.__import__", side_effect=guarded):
                result = namespace["get_memory"](None)
        self.assertEqual(result["cuda"], {"error": gpu_fault.MESSAGE})
