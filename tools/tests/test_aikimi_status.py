import ast
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from modules import aikimi_status

ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = ROOT / "assets" / "aikimi"
AIKIMI_UI_ROOT = ROOT / "extensions-builtin" / "aikimi-ui"
ASSISTANT_SOURCE = AIKIMI_UI_ROOT / "javascript" / "aikimiStatus.js"
ASSISTANT_CSS = AIKIMI_UI_ROOT / "style.css"
EXPECTED_STATES = {
    "idle",
    "loading_model",
    "generating",
    "completed",
    "queued",
    "warning",
    "error",
    "out_of_memory",
    "updating",
}


class AikimiStatusTests(unittest.TestCase):
    def test_manifest_covers_states_and_references_local_assets(self):
        manifest = json.loads((ASSET_ROOT / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["version"], 4)
        self.assertEqual(set(manifest["states"]), EXPECTED_STATES)
        self.assertIn(manifest["default_state"], manifest["states"])
        self.assertEqual(Path(manifest["portrait"]).name, manifest["portrait"])
        self.assertTrue(all(state["message"] for state in manifest["states"].values()))
        with Image.open(ASSET_ROOT / manifest["portrait"]) as pet:
            self.assertEqual(pet.mode, "RGBA")
            self.assertEqual(pet.n_frames, 1)
            self.assertEqual(pet.getchannel("A").getextrema(), (0, 255))
            self.assertIsNone(pet.getchannel("A").crop((0, 0, pet.width, 1)).getbbox())

    def test_legacy_character_sources_remain_available(self):
        for name in ("idle", "working", "happy", "troubled"):
            with self.subTest(name=name):
                with Image.open(ASSET_ROOT / f"{name}.webp") as image:
                    self.assertEqual(image.size, (512, 640))
                    self.assertEqual(image.mode, "RGBA")
                    self.assertEqual(image.getchannel("A").getextrema(), (0, 255))

        with Image.open(ASSET_ROOT / "favicon.png") as favicon:
            self.assertEqual(favicon.size, (128, 128))
            self.assertEqual(favicon.mode, "RGBA")

    def test_model_snapshot_uses_shallow_loader_state(self):
        model_data = SimpleNamespace(
            sd_model=SimpleNamespace(
                forge_objects=object(),
                filename=r"C:\models\loaded-aikimi.safetensors",
                sd_checkpoint_info=SimpleNamespace(name="folder-a/loaded-aikimi.safetensors"),
            ),
            forge_loading_parameters={
                "checkpoint_info": SimpleNamespace(
                    filename=r"C:\models\folder-b\aikimi.safetensors",
                    name="folder-b/aikimi.safetensors",
                )
            },
            forge_hash="old-hash",
            forge_loading=True,
            last_load_seconds=8.21,
            last_load_error=None,
        )

        status = aikimi_status._model_snapshot(model_data)

        self.assertTrue(status["loaded"])
        self.assertTrue(status["loading"])
        self.assertEqual(status["selected_name"], "folder-b/aikimi.safetensors")
        self.assertEqual(status["loaded_name"], "folder-a/loaded-aikimi.safetensors")
        self.assertTrue(status["reload_pending"])
        self.assertNotIn("path", status)
        self.assertEqual(status["last_load_seconds"], 8.21)

    def test_status_auth_matches_gradio_cookie_boundary(self):
        open_app = SimpleNamespace(auth=None, auth_dependency=None)
        request = SimpleNamespace(cookies={})
        self.assertTrue(aikimi_status.request_is_authorized(open_app, request))

        protected_app = SimpleNamespace(
            auth={"user": "hash"},
            auth_dependency=None,
            cookie_id="cookie",
            tokens={"valid-token": "user"},
        )
        self.assertFalse(aikimi_status.request_is_authorized(protected_app, request))
        request.cookies["access-token-unsecure-cookie"] = "valid-token"
        self.assertTrue(aikimi_status.request_is_authorized(protected_app, request))

    def test_generation_snapshot_reports_pending_queue_separately(self):
        state = SimpleNamespace(
            job_count=2,
            job_no=1,
            sampling_steps=10,
            sampling_step=5,
            time_start=None,
            job="task(test)",
            textinfo="Sampling",
        )

        status = aikimi_status._generation_snapshot(
            state,
            pending_tasks={"task(waiting-1)": 1.0, "task(waiting-2)": 2.0},
        )

        self.assertTrue(status["active"])
        self.assertEqual(status["progress"], 0.75)
        self.assertEqual(status["queue_size"], 2)
        self.assertNotIn("task", status)
        self.assertNotIn("job", status)
        self.assertNotIn("queue_tasks", status)

    def test_loader_status_defaults_do_not_change_model_contract(self):
        tree = ast.parse((ROOT / "modules" / "sd_models.py").read_text(encoding="utf-8"))
        model_data_class = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SdModelData"
        )
        constructor = next(
            node for node in model_data_class.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        assigned_attributes = {
            target.attr
            for node in ast.walk(constructor)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self"
        }

        self.assertIn("forge_loading", assigned_attributes)
        self.assertIn("last_load_seconds", assigned_attributes)
        self.assertIn("last_load_error", assigned_attributes)

    def test_model_loading_flag_is_cleared_by_outer_finally(self):
        tree = ast.parse((ROOT / "modules" / "sd_models.py").read_text(encoding="utf-8"))
        reload_function = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "forge_model_reload"
        )
        tracking_try = next(node for node in reload_function.body if isinstance(node, ast.Try))
        final_assignments = [
            node
            for node in ast.walk(ast.Module(body=tracking_try.finalbody, type_ignores=[]))
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Attribute) and target.attr == "forge_loading" for target in node.targets)
        ]

        self.assertEqual(len(final_assignments), 1)
        self.assertIs(final_assignments[0].value.value, False)

    def test_frontend_subscribes_without_replacing_progress_api(self):
        progress_source = (ROOT / "javascript" / "progressbar.js").read_text(encoding="utf-8")
        assistant_source = ASSISTANT_SOURCE.read_text(encoding="utf-8")

        for event_name in (
            "webui:task-start",
            "webui:task-progress",
            "webui:task-error",
            "webui:task-end",
        ):
            self.assertIn(event_name, progress_source)
            self.assertIn(event_name, assistant_source)
        self.assertIn("window.AikimiStatus", assistant_source)
        self.assertIn('matchMedia("(prefers-reduced-motion: reduce)")', assistant_source)
        self.assertIn("failedAssetUrls", assistant_source)
        self.assertIn("preloadConfiguredAssets", assistant_source)
        self.assertIn("let enabled = false", assistant_source)
        self.assertIn("const ACTIVE_POLL_MS = 1500", assistant_source)
        self.assertIn("const MAX_BACKOFF_MS = 60000", assistant_source)
        self.assertIn("const FETCH_TIMEOUT_MS = 8000", assistant_source)
        self.assertIn("pollingController === controller", assistant_source)
        self.assertIn("pollingGeneration", assistant_source)
        self.assertIn("window.__aikimiStatusInitialized", assistant_source)
        self.assertIn("publicTechnicalDetail", assistant_source)
        self.assertIn("window.AikimiTabs", assistant_source)
        self.assertIn('"aikimi:feature-tab-change"', assistant_source)
        self.assertIn("navigationIssue", assistant_source)
        self.assertIn("message: navigationIssue", assistant_source)
        self.assertIn("errorDetails: navigationIssue", assistant_source)
        self.assertIn("statusIsActive()", assistant_source)
        self.assertIn("if (statusIsActive()) scanOutputErrors();", assistant_source)
        self.assertIn("activeContainer", assistant_source)
        self.assertNotIn("createBrandHeader", assistant_source)
        self.assertIn("aikimi_assistant_position", assistant_source)
        self.assertNotIn("aikimi-working", (ROOT / "style.css").read_text(encoding="utf-8"))

    def test_assistant_preferences_are_registered_and_manifest_driven(self):
        options_source = (ROOT / "modules" / "shared_options.py").read_text(encoding="utf-8")
        assistant_source = ASSISTANT_SOURCE.read_text(encoding="utf-8")
        css_source = ASSISTANT_CSS.read_text(encoding="utf-8")
        root_css_source = (ROOT / "style.css").read_text(encoding="utf-8")

        for option in (
            "aikimi_assistant_size",
            "aikimi_assistant_dialogue_enabled",
            "aikimi_assistant_animation_enabled",
        ):
            self.assertIn(option, options_source)
            self.assertIn(option, assistant_source)
        for value in ("small", "medium", "large"):
            self.assertIn(f'[data-size="{value}"]', css_source)
        self.assertIn('"aikimi_assistant_position"', options_source)
        self.assertIn('"最初の置き場所"', options_source)
        self.assertNotIn("data-position", css_source)
        self.assertIn("position: fixed", css_source)
        self.assertNotIn("#aikimi-status", root_css_source)
        self.assertNotIn(".aikimi-diagnostics", root_css_source)
        self.assertNotIn(".aikimi-about", root_css_source)
        self.assertIn("--aikimi-character-size: 80px", css_source)
        self.assertIn("--aikimi-character-size: 104px", css_source)
        self.assertIn("--aikimi-character-size: 128px", css_source)
        self.assertIn("object-fit: contain", css_source)
        self.assertIn("#aikimi-status .aikimi-status__summary", css_source)
        self.assertIn("aikimi-pet-float", css_source)
        self.assertNotIn('content: "  ▾"', css_source)
        self.assertNotIn('content: "  ▴"', css_source)
        self.assertIn("@media (forced-colors: active)", css_source)
        self.assertIn("border-color: CanvasText", css_source)
        self.assertIn("outline-color: Highlight", css_source)
        self.assertIn('warning: "確認してください"', assistant_source)
        self.assertIn('error: "エラー"', assistant_source)
        self.assertIn('out_of_memory: "メモリ不足"', assistant_source)
        self.assertIn("stillMode", assistant_source)
        self.assertIn("message.hidden = !dialogueEnabled", assistant_source)


if __name__ == "__main__":
    unittest.main()
