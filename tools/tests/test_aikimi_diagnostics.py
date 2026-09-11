import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from modules import aikimi_capabilities as capabilities
from modules import aikimi_diagnostics as diagnostics

ROOT = Path(__file__).resolve().parents[2]


class RecordingApi:
    def __init__(self):
        self.routes = []

    def add_api_route(self, path, endpoint, **kwargs):
        self.routes.append((path, endpoint, kwargs))


class FakeCuda:
    def is_available(self):
        return True

    def current_device(self):
        return 0

    def get_device_properties(self, _device):
        return SimpleNamespace(name="Test GPU", total_memory=24 * 1024**3)


class AikimiDiagnosticsTests(unittest.TestCase):
    def paths(self, root):
        output = root / "output"
        output.mkdir(parents=True, exist_ok=True)
        return diagnostics.DiagnosticPaths(
            root=root,
            models_root=root / "models",
            output_root=output,
        )

    def test_health_is_minimal_and_versioned(self):
        payload = diagnostics.health_payload()

        self.assertEqual(payload["api_version"], "1")
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(set(payload), {"api_version", "app_version", "status"})

    def test_status_whitelists_runtime_fields_and_drops_secrets_and_paths(self):
        sentinel = "diagnostic-secret-123"
        payload = diagnostics.status_payload(
            {
                "model": {
                    "loaded": True,
                    "loading": False,
                    "loaded_name": r"H:\private\models\safe-model.safetensors",
                    "selected_name": "/home/user/models/next-model.safetensors",
                    "reload_pending": True,
                    "last_load_seconds": 2.5,
                    "token": sentinel,
                },
                "generation": {
                    "active": True,
                    "progress": 0.5,
                    "eta": 3.0,
                    "queue_size": 2,
                    "text": rf"token={sentinel} H:\private\job.json",
                },
                "memory": {
                    "available": False,
                    "error": rf"password={sentinel} H:\private\driver.log",
                },
                "backend": {"ready": True, "uptime_seconds": 10.0},
                "future_secret": sentinel,
            }
        )

        rendered = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn(sentinel, rendered)
        self.assertNotIn("H:\\", rendered)
        self.assertNotIn("/home/", rendered)
        self.assertNotIn("text", payload["generation"])
        self.assertNotIn("error", payload["memory"])
        self.assertEqual(payload["model"]["loaded_name"], "safe-model.safetensors")
        self.assertEqual(payload["model"]["selected_name"], "next-model.safetensors")
        self.assertEqual(payload["phase"], "generating")

    def test_status_rejects_non_finite_or_negative_numbers(self):
        payload = diagnostics.status_payload(
            {
                "model": {"last_load_seconds": float("inf")},
                "generation": {
                    "progress": float("nan"),
                    "eta": -1,
                    "queue_size": -2,
                },
                "memory": {"available": True, "used": -1, "total": float("inf")},
                "backend": {"ready": False, "uptime_seconds": -1},
            }
        )

        self.assertEqual(payload["status"], "degraded")
        self.assertIsNone(payload["generation"]["progress"])
        self.assertIsNone(payload["generation"]["eta"])
        self.assertEqual(payload["generation"]["queue_size"], 0)
        self.assertIsNone(payload["memory"]["used"])
        self.assertIsNone(payload["memory"]["total"])

    def test_status_fails_closed_before_runtime_state_is_initialized(self):
        with mock.patch(
            "modules.aikimi_status.snapshot",
            side_effect=RuntimeError(r"token=secret H:\private\state.json"),
        ):
            payload = diagnostics.status_payload()

        rendered = json.dumps(payload)
        self.assertEqual(payload["status"], "degraded")
        self.assertEqual(payload["phase"], "idle")
        self.assertNotIn("secret", rendered)
        self.assertNotIn("H:\\", rendered)

    def test_public_check_redacts_secret_forms_and_absolute_paths(self):
        sentinel = "abc123456789"
        check = diagnostics.DiagnosticCheck(
            "unsafe",
            "Unsafe",
            diagnostics.CheckState.WARNING,
            rf"token={sentinel} H:\private\config.json /home/user/config.json",
            rf"Open \\server\share\secret.txt password={sentinel}",
        )

        rendered = json.dumps(check.public_dict(), ensure_ascii=False)
        self.assertNotIn(sentinel, rendered)
        self.assertNotIn("H:\\", rendered)
        self.assertNotIn("\\\\server", rendered)
        self.assertNotIn("/home/", rendered)
        self.assertIn("<local-path>", rendered)

    def test_public_check_preserves_an_empty_optional_action(self):
        check = diagnostics.DiagnosticCheck("ready", "Ready", diagnostics.CheckState.READY, "Ready.")

        self.assertEqual(check.public_dict()["action"], "")

    def test_rendered_diagnostics_escape_labels_summaries_and_actions(self):
        check = diagnostics.DiagnosticCheck(
            "escape",
            '<script id="label">',
            diagnostics.CheckState.BLOCKED,
            '<img src=x onerror="alert(1)">',
            "<b>repair</b>",
        )

        rendered = diagnostics.render_diagnostics_html([check])

        self.assertNotIn("<script", rendered)
        self.assertNotIn("<img", rendered)
        self.assertNotIn("<b>repair", rendered)
        self.assertIn("&lt;script", rendered)
        self.assertIn("&lt;img", rendered)
        self.assertIn("&lt;b&gt;repair", rendered)
        self.assertIn("Blocked 1", rendered)

    def test_overall_state_uses_blocked_then_warning_priority(self):
        ready = diagnostics.DiagnosticCheck("ready", "Ready", diagnostics.CheckState.READY, "ok")
        warning = diagnostics.DiagnosticCheck("warning", "Warning", diagnostics.CheckState.WARNING, "warn")
        blocked = diagnostics.DiagnosticCheck("blocked", "Blocked", diagnostics.CheckState.BLOCKED, "blocked")

        self.assertIs(diagnostics.overall_state([ready]), diagnostics.CheckState.READY)
        self.assertIs(diagnostics.overall_state([ready, warning]), diagnostics.CheckState.WARNING)
        self.assertIs(
            diagnostics.overall_state([ready, warning, blocked]),
            diagnostics.CheckState.BLOCKED,
        )

    def test_register_api_routes_uses_read_only_versioned_names(self):
        api = RecordingApi()

        diagnostics.register_api_routes(api)

        self.assertEqual(
            [path for path, _, _ in api.routes],
            [
                "/aikimi/api/v1/health",
                "/aikimi/api/v1/status",
                "/aikimi/api/v1/capabilities",
            ],
        )
        self.assertTrue(all(route[2]["methods"] == ["GET"] for route in api.routes))
        self.assertTrue(all(not inspect.signature(route[1]).parameters for route in api.routes))

    def test_health_route_builds_in_fastapi_without_ui_or_gpu(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        api = SimpleNamespace(add_api_route=app.add_api_route)
        diagnostics.register_api_routes(api)

        response = TestClient(app).get("/aikimi/api/v1/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["api_version"], "1")
        self.assertEqual(response.json()["status"], "ok")
        schema = app.openapi()
        for path in (
            "/aikimi/api/v1/health",
            "/aikimi/api/v1/status",
            "/aikimi/api/v1/capabilities",
        ):
            operation = schema["paths"][path]["get"]
            self.assertEqual(operation.get("parameters", []), [])
            self.assertNotIn("requestBody", operation)

    def test_api_constructor_registers_diagnostics_through_authenticated_helper(self):
        source = (ROOT / "modules" / "api" / "api.py").read_text(encoding="utf-8")

        self.assertIn("aikimi_diagnostics.register_api_routes(self)", source)
        self.assertNotIn('self.app.add_api_route("/aikimi/api/', source)

    def test_settings_contains_refreshable_diagnostics_tab(self):
        source = (ROOT / "modules" / "ui_settings.py").read_text(encoding="utf-8")
        style = (ROOT / "extensions-builtin" / "aikimi-ui" / "style.css").read_text(encoding="utf-8")

        self.assertIn('gr.TabItem("Diagnostics"', source)
        self.assertIn("aikimi_diagnostics.render_diagnostics_html", source)
        self.assertIn('elem_id="aikimi_system_check"', source)
        self.assertIn("diagnostics_refresh.click", source)
        self.assertIn("queue=False", source)
        self.assertIn(".aikimi-diagnostic-card.is-ready", style)
        self.assertIn(".aikimi-diagnostic-card.is-warning", style)
        self.assertIn(".aikimi-diagnostic-card.is-blocked", style)

    def test_diagnostics_component_and_refresh_event_build_in_current_gradio(self):
        import gradio as gr

        with gr.Blocks() as demo:
            output = gr.HTML(
                value=lambda: diagnostics.render_diagnostics_html(
                    [
                        diagnostics.DiagnosticCheck(
                            "ready",
                            "Ready",
                            diagnostics.CheckState.READY,
                            "Ready.",
                        )
                    ]
                ),
                elem_id="aikimi_system_check_fixture",
            )
            refresh = gr.Button("Run System Check")
            refresh.click(
                fn=lambda: "<section>Ready</section>",
                inputs=[],
                outputs=[output],
                queue=False,
                show_progress=False,
            )

        config = demo.get_config_file()
        self.assertTrue(
            any(
                component.get("props", {}).get("elem_id") == "aikimi_system_check_fixture"
                for component in config["components"]
            )
        )
        self.assertTrue(config["dependencies"])

    def test_security_check_reports_local_remote_and_invalid_remote_modes(self):
        local = SimpleNamespace(
            listen=False,
            server_name="127.0.0.1",
            share=False,
            ngrok=None,
            aikimi_remote=False,
            api=True,
            nowebui=False,
            api_auth=None,
            api_auth_path=None,
            gradio_auth=None,
            gradio_auth_path=None,
        )
        remote = SimpleNamespace(
            listen=True,
            server_name=None,
            share=False,
            ngrok=None,
            aikimi_remote=True,
            api=True,
            nowebui=False,
            api_auth_path="api-auth.txt",
            api_auth=None,
            gradio_auth_path="gradio-auth.txt",
            gradio_auth=None,
        )
        invalid = SimpleNamespace(**vars(remote))
        invalid.gradio_auth_path = None

        self.assertIs(diagnostics._security_check(local).state, diagnostics.CheckState.READY)
        self.assertIs(diagnostics._security_check(remote).state, diagnostics.CheckState.WARNING)
        self.assertIs(diagnostics._security_check(invalid).state, diagnostics.CheckState.BLOCKED)
        rendered = json.dumps(diagnostics._security_check(remote).public_dict())
        self.assertNotIn("api-auth.txt", rendered)
        self.assertNotIn("gradio-auth.txt", rendered)

    def test_storage_check_writes_and_removes_only_its_probe(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            existing = output / "keep.png"
            existing.write_bytes(b"keep")

            result = diagnostics._storage_check(output)

            self.assertIsNot(result.state, diagnostics.CheckState.BLOCKED)
            self.assertEqual(existing.read_bytes(), b"keep")
            self.assertEqual(list(output.glob(".aikimi-diagnostic-*")), [])

    def test_safetensors_contract_reads_only_bounded_header(self):
        with tempfile.TemporaryDirectory() as temporary:
            models = Path(temporary)
            target = models / "fixture.safetensors"
            header = json.dumps(
                {
                    "tensor": {
                        "dtype": "F32",
                        "shape": [],
                        "data_offsets": [0, 0],
                    },
                    "__metadata__": {"profile": "expected-marker"},
                },
                separators=(",", ":"),
            ).encode("utf-8")
            target.write_bytes(len(header).to_bytes(8, "little") + header)
            contract = capabilities._FileContract("fixture.safetensors", target.stat().st_size, ("expected-marker",))

            self.assertTrue(capabilities._matches_file_contract(models, contract))
            wrong = capabilities._FileContract("fixture.safetensors", target.stat().st_size, ("missing-marker",))
            self.assertFalse(capabilities._matches_file_contract(models, wrong))

    def test_cuda_check_uses_metadata_without_generation(self):
        torch_module = SimpleNamespace(
            cuda=FakeCuda(),
            version=SimpleNamespace(cuda="13.0"),
        )

        result = diagnostics._cuda_check(torch_module)

        self.assertIs(result.state, diagnostics.CheckState.READY)
        self.assertIn("24.0 GiB", result.summary)
        self.assertIn("CUDA 13.0", result.summary)

    def test_capabilities_payload_never_serializes_internal_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.paths(Path(temporary))
            checks = (
                diagnostics.DiagnosticCheck(
                    "feature",
                    "Feature",
                    diagnostics.CheckState.READY,
                    "Feature is ready.",
                    "No action is required.",
                    available=True,
                ),
            )
            with (
                mock.patch.object(diagnostics, "feature_checks", return_value=checks),
                mock.patch.object(diagnostics, "_short_commit", return_value="abcdef123456"),
            ):
                payload = diagnostics.capabilities_payload(paths)

        rendered = json.dumps(payload, ensure_ascii=False)
        self.assertEqual(payload["api_version"], "1")
        self.assertTrue(payload["features"]["feature"]["available"])
        self.assertNotIn(temporary, rendered)


if __name__ == "__main__":
    unittest.main()
