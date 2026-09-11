import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

from modules_forge.minimax_h3_bridge import RuntimeReadiness


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STUDIO_PATH = (
    REPOSITORY_ROOT
    / "extensions-builtin"
    / "minimax-h3-studio"
    / "scripts"
    / "minimax_h3_studio.py"
)
JAVASCRIPT_PATH = STUDIO_PATH.parents[1] / "javascript" / "minimax_h3_studio.js"
STYLE_PATH = STUDIO_PATH.parents[1] / "style.css"


def load_studio_module():
    modules_package = ModuleType("modules")
    callbacks_module = ModuleType("modules.script_callbacks")
    callbacks_module.on_ui_tabs = mock.Mock()
    paths_module = ModuleType("modules.paths")
    paths_module.data_path = str(REPOSITORY_ROOT)
    paths_module.script_path = str(REPOSITORY_ROOT)
    modules_package.script_callbacks = callbacks_module

    spec = importlib.util.spec_from_file_location("_test_minimax_h3_studio", STUDIO_PATH)
    module = importlib.util.module_from_spec(spec)
    stubs = {
        "modules": modules_package,
        "modules.script_callbacks": callbacks_module,
        "modules.paths": paths_module,
    }
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


class MiniMaxH3StudioCallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.studio = load_studio_module()

    def test_manual_setting_update_refreshes_summary_and_marks_custom(self):
        summary, state, validation = self.studio._custom_settings_updates(
            "16:9",
            "balanced",
            5.0,
            20,
            -1,
            "simple",
            "match",
        )
        self.assertIn("960 × 544", summary)
        self.assertIn('data-h3-preset="custom"', state)
        self.assertEqual(validation["__type__"], "update")

    def test_mode_update_resets_hidden_reference_max_and_summary(self):
        updates = self.studio._mode_updates(
            self.studio.MODE_TEXT,
            "16:9",
            "preview",
            5.0,
            20,
            "simple",
            "max",
        )
        self.assertEqual(len(updates), 6)
        self.assertEqual(updates[1]["visible"], "hidden")
        self.assertEqual(updates[3]["value"], "match")
        self.assertNotIn("Reference Max", updates[4])
        self.assertEqual(updates[5]["__type__"], "update")

    def test_generate_callback_disables_and_restores_actions(self):
        generation_updates = [
            {"stage": "prepare", "message": "prepare", "progress": 0.06, "prompt_id": ""},
            {"stage": "queued", "message": "queued", "progress": 0.12, "prompt_id": "job-1"},
            {
                "stage": "complete",
                "message": "complete",
                "progress": 1.0,
                "prompt_id": "job-1",
                "path": "result.mp4",
            },
        ]
        with mock.patch.object(self.studio, "resolve_runtime_root", return_value=Path("runtime")), mock.patch.object(
            self.studio,
            "run_generation",
            return_value=iter(generation_updates),
        ), mock.patch.object(
            self.studio,
            "_history_state",
            return_value=([], "history", [("latest", "result.mp4")]),
        ):
            results = list(
                self.studio._generate(
                    "runtime",
                    "http://127.0.0.1:8188",
                    "fast",
                    "text",
                    "prompt",
                    None,
                    None,
                    None,
                    None,
                    None,
                    "16:9",
                    "preview",
                    5.0,
                    20,
                    -1,
                    "simple",
                    "match",
                )
            )

        self.assertTrue(all(len(result) == 8 for result in results))
        self.assertFalse(results[0][5]["interactive"])
        self.assertFalse(results[0][6]["interactive"])
        self.assertEqual(results[0][6]["value"], "生成を準備中…")
        self.assertTrue(results[2][5]["interactive"])
        self.assertFalse(results[2][6]["interactive"])
        self.assertFalse(results[-1][5]["interactive"])
        self.assertTrue(results[-1][6]["interactive"])
        self.assertEqual(results[-1][6]["value"], "映像＋音声を生成")
        self.assertEqual(results[0][7], "")

    def test_generate_uses_specific_runtime_and_reconnect_button_labels(self):
        generation_updates = [
            {"stage": "runtime", "message": "runtime", "progress": 0.03, "prompt_id": ""},
            {
                "stage": "reconnecting",
                "message": "retry",
                "progress": 0.13,
                "prompt_id": "job-1",
            },
        ]
        with mock.patch.object(self.studio, "resolve_runtime_root", return_value=Path("runtime")), mock.patch.object(
            self.studio,
            "run_generation",
            return_value=iter(generation_updates),
        ):
            results = list(
                self.studio._generate(
                    "runtime",
                    "http://127.0.0.1:8188",
                    "fast",
                    "text",
                    "prompt",
                    None,
                    None,
                    None,
                    None,
                    None,
                    "16:9",
                    "preview",
                    5.0,
                    20,
                    -1,
                    "simple",
                    "match",
                )
            )
        self.assertEqual(results[1][6]["value"], "H3 backendを確認中…")
        self.assertEqual(results[2][6]["value"], "H3 backendへ再接続中…")
        self.assertFalse(results[-1][6]["interactive"])

    def test_generate_callback_restores_button_after_validation_error(self):
        with mock.patch.object(
            self.studio,
            "resolve_runtime_root",
            side_effect=self.studio.H3BridgeError("invalid runtime"),
        ):
            results = list(
                self.studio._generate(
                    "runtime",
                    "http://127.0.0.1:8188",
                    "fast",
                    "text",
                    "prompt",
                    None,
                    None,
                    None,
                    None,
                    None,
                    "16:9",
                    "preview",
                    5.0,
                    20,
                    -1,
                    "simple",
                    "match",
                )
            )
        self.assertEqual(len(results), 2)
        self.assertIn("invalid runtime", results[-1][0])
        self.assertTrue(results[-1][6]["interactive"])

    def test_cancelled_generation_is_a_non_error_terminal_state(self):
        def cancelled_updates(*_args, **_kwargs):
            yield {
                "stage": "queued",
                "message": "queued",
                "progress": 0.12,
                "prompt_id": "job-1",
            }
            raise self.studio.H3GenerationCancelled("生成を停止しました。")

        with mock.patch.object(self.studio, "resolve_runtime_root", return_value=Path("runtime")), mock.patch.object(
            self.studio,
            "run_generation",
            side_effect=cancelled_updates,
        ):
            results = list(
                self.studio._generate(
                    "runtime",
                    "http://127.0.0.1:8188",
                    "fast",
                    "text",
                    "prompt",
                    None,
                    None,
                    None,
                    None,
                    None,
                    "16:9",
                    "preview",
                    5.0,
                    20,
                    -1,
                    "simple",
                    "match",
                )
            )
        self.assertIn('data-stage="cancelled"', results[-1][0])
        self.assertNotIn('data-stage="error"', results[-1][0])
        self.assertFalse(results[-1][5]["interactive"])
        self.assertTrue(results[-1][6]["interactive"])

    def test_empty_prompt_returns_inline_alert_and_focus_target(self):
        results = list(
            self.studio._generate(
                "runtime",
                "http://127.0.0.1:8188",
                "fast",
                "text",
                "",
                None,
                None,
                None,
                None,
                None,
                "16:9",
                "preview",
                5.0,
                20,
                -1,
                "simple",
                "match",
            )
        )
        self.assertEqual(len(results), 2)
        self.assertIn('data-h3-invalid="prompt"', results[-1][7])
        self.assertIn('data-h3-control="prompt"', results[-1][7])
        self.assertIn('role="alert"', results[-1][7])
        self.assertTrue(results[-1][6]["interactive"])

    def test_validation_only_clears_after_the_value_is_valid(self):
        prompt_error = '<div data-h3-invalid="prompt">prompt error</div>'
        self.assertEqual(
            self.studio._clear_prompt_validation("   ", prompt_error)["__type__"],
            "update",
        )
        self.assertEqual(self.studio._clear_prompt_validation("valid", prompt_error), "")

        keyframe_error = '<div data-h3-invalid="keyframes">keyframe error</div>'
        self.assertEqual(
            self.studio._clear_keyframe_validation(None, None, keyframe_error)["__type__"],
            "update",
        )
        self.assertEqual(
            self.studio._clear_keyframe_validation("first.png", None, keyframe_error),
            "",
        )

    def test_prompt_action_clears_input_validation(self):
        prompt, validation = self.studio._prompt_camera_updates(
            "",
            '<div data-h3-invalid="prompt">prompt error</div>',
        )
        self.assertIn("Camera:", prompt)
        self.assertEqual(validation, "")

    def test_settings_error_targets_settings_and_clears_on_seed_input(self):
        results = list(
            self.studio._generate(
                "runtime",
                "http://127.0.0.1:8188",
                "fast",
                "keyframes",
                "prompt",
                None,
                None,
                None,
                None,
                None,
                "16:9",
                "preview",
                5.0,
                20,
                -2,
                "simple",
                "match",
            )
        )
        validation = results[-1][7]
        self.assertIn('data-h3-invalid="settings"', validation)
        self.assertIn('data-h3-control="seed"', validation)
        invalid = self.studio._clear_settings_validation(
            "16:9", "preview", 5.0, 20, -2, "simple", "match", validation
        )
        self.assertEqual(invalid["__type__"], "update")
        self.assertEqual(
            self.studio._clear_settings_validation(
                "16:9", "preview", 5.0, 20, -1, "simple", "match", validation
            ),
            "",
        )
        unchanged = self.studio._clear_settings_validation(
            "16:9",
            "preview",
            5.0,
            20,
            -1,
            "simple",
            "match",
            '<div data-h3-invalid="keyframes">keyframe error</div>',
        )
        self.assertEqual(unchanged["__type__"], "update")

    def test_non_finite_seed_restores_generate_button_with_inline_error(self):
        results = list(
            self.studio._generate(
                "runtime",
                "http://127.0.0.1:8188",
                "fast",
                "text",
                "prompt",
                None,
                None,
                None,
                None,
                None,
                "16:9",
                "preview",
                5.0,
                20,
                float("inf"),
                "simple",
                "match",
            )
        )
        self.assertEqual(len(results), 2)
        self.assertIn('data-h3-control="seed"', results[-1][7])
        self.assertTrue(results[-1][6]["interactive"])

    def test_history_refresh_preserves_a_selection_that_still_exists(self):
        choices = [("latest", "one.mp4"), ("selected", "two.mp4")]
        with mock.patch.object(
            self.studio,
            "_history_state",
            return_value=([], "history", choices),
        ):
            rendered, selector = self.studio._refresh_history("runtime", "two.mp4")
        self.assertEqual(rendered, "history")
        self.assertEqual(selector["value"], "two.mp4")

    def test_history_restore_updates_prompt_settings_and_warns_about_media(self):
        request = self.studio.H3Request(
            mode=self.studio.MODE_REFERENCES,
            prompt="Restored prompt",
            aspect="9:16",
            quality="balanced",
            duration_seconds=7.5,
            steps=24,
            seed=42,
            scheduler="beta",
            ref_image_size="max",
        )
        with mock.patch.object(
            self.studio,
            "_history_state",
            return_value=([mock.Mock()], "history", [("item", "video.mp4")]),
        ), mock.patch.object(
            self.studio,
            "load_history_request",
            return_value=request,
        ):
            updates = self.studio._restore_history_settings("video.mp4", "runtime")
        self.assertEqual(len(updates), 22)
        self.assertEqual(updates[0]["value"], self.studio.MODE_REFERENCES)
        self.assertEqual(updates[1]["value"], "Restored prompt")
        self.assertTrue(all(updates[index]["value"] is None for index in range(2, 7)))
        self.assertTrue(updates[8]["visible"])
        self.assertEqual(updates[11]["value"], "9:16")
        self.assertEqual(updates[16]["value"], "beta")
        self.assertEqual(updates[17]["value"], "max")
        self.assertIn("参照素材はもう一度追加", updates[-1])

    def test_history_failure_does_not_turn_a_completed_generation_into_an_error(self):
        generation_updates = [
            {
                "stage": "complete",
                "message": "complete",
                "progress": 1.0,
                "prompt_id": "job-1",
                "path": "result.mp4",
            }
        ]
        with mock.patch.object(self.studio, "resolve_runtime_root", return_value=Path("runtime")), mock.patch.object(
            self.studio,
            "run_generation",
            return_value=iter(generation_updates),
        ), mock.patch.object(
            self.studio,
            "_history_state",
            side_effect=self.studio.H3BridgeError("history unavailable"),
        ):
            results = list(
                self.studio._generate(
                    "runtime",
                    "http://127.0.0.1:8188",
                    "fast",
                    "text",
                    "prompt",
                    None,
                    None,
                    None,
                    None,
                    None,
                    "16:9",
                    "preview",
                    5.0,
                    20,
                    -1,
                    "simple",
                    "match",
                )
            )
        self.assertEqual(results[-1][1], "result.mp4")
        self.assertIn('data-stage="complete"', results[-1][0])
        self.assertIn("history unavailable", results[-1][3])
        self.assertTrue(results[-1][6]["interactive"])

    def test_ui_dependencies_match_fast_manual_update_contract(self):
        readiness = RuntimeReadiness(
            runtime_root=None,
            server_url=self.studio.H3_SERVER_URL,
            connected=False,
        )
        with mock.patch.object(self.studio, "_initial_runtime", return_value=None), mock.patch.object(
            self.studio,
            "inspect_readiness",
            return_value=readiness,
        ) as inspect, mock.patch.object(
            self.studio,
            "_initial_history_state",
            return_value=([], self.studio.history_html([]), []),
        ):
            interface = self.studio._build_ui()[0][0]
        inspect.assert_not_called()
        config = interface.get_config_file()
        element_ids = [
            component["props"].get("elem_id")
            for component in config["components"]
            if component.get("props", {}).get("elem_id")
        ]
        self.assertEqual(len(element_ids), len(set(element_ids)))
        component_ids = {
            component["props"].get("elem_id"): component["id"]
            for component in config["components"]
            if component.get("props", {}).get("elem_id")
        }
        self.assertIn("h3-history-load", component_ids)
        self.assertIn("h3-history-restore", component_ids)
        self.assertIn("h3-history-refresh", component_ids)
        self.assertIn("h3-mobile-action-bar", component_ids)
        self.assertIn("h3-initialize-trigger", component_ids)
        self.assertIn("h3-prompt-assists", component_ids)
        self.assertLess(
            element_ids.index("h3-generate"),
            element_ids.index("h3-settings-summary"),
        )
        self.assertLess(
            element_ids.index("h3-generate"),
            element_ids.index("h3-prompt-assists"),
        )
        self.assertLess(
            element_ids.index("h3-runtime-setup"),
            element_ids.index("h3-history-accordion"),
        )
        self.assertGreater(
            element_ids.index("h3-mobile-action-bar"),
            element_ids.index("h3-generate"),
        )
        component_props = {
            component["props"].get("elem_id"): component["props"]
            for component in config["components"]
            if component.get("props", {}).get("elem_id")
        }
        for elem_id in (
            "h3-generate",
            "h3-history-restore",
            "h3-runtime-profile",
            "h3-runtime-path",
            "h3-server-url",
            "h3-connect",
            "h3-restart",
            "h3-rescan",
        ):
            self.assertFalse(component_props[elem_id]["interactive"], elem_id)

        dependencies = config["dependencies"]
        # 旧UI設定の外部ComfyUIや8188が、管理対象の固定値を上書きしない。
        importlib.import_module("modules.shared")
        from modules.ui_loadsave import UiLoadsave

        for elem_id in ("h3-runtime-path", "h3-server-url", "h3-shared-models"):
            component = interface.blocks[component_ids[elem_id]]
            expected = component.value
            path = f"minimax_h3_studio/{component.label}"
            loader = SimpleNamespace(
                finalized_ui=False, ui_settings={f"{path}/value": "old-external-runtime"}, component_mapping={},
            )
            UiLoadsave.add_component(loader, path, component)
            self.assertEqual(component.value, expected)
        generate_id = component_ids["h3-generate"]
        generate = next(
            dependency
            for dependency in dependencies
            if any(target[0] == generate_id and target[1] == "click" for target in dependency["targets"])
        )
        from modules_forge.minimax_h3_acceleration import H3Acceleration
        control_count = len(H3Acceleration().values())
        self.assertEqual(len(generate["inputs"]), 20 + control_count)
        self.assertEqual(generate["inputs"][-14], component_ids["h3-negpip-enabled"])
        self.assertEqual(generate["inputs"][-6], component_ids["h3-clip-cache"])
        self.assertEqual(generate["inputs"][-control_count:-control_count + 8], [component_ids[elem_id] for elem_id in (
            "h3-model-variant", "h3-video-vae", "h3-decode-mode", "h3-tile-batch",
            "h3-attention-mode", "h3-sparse-tau", "h3-sparse-keep", "h3-sparse-start",
        )])
        self.assertEqual(len(generate["outputs"]), 8)
        self.assertEqual(generate["trigger_mode"], "once")

        quick_id = component_ids["h3-preset-quick"]
        quick_preset = next(
            dependency
            for dependency in dependencies
            if any(target[0] == quick_id and target[1] == "click" for target in dependency["targets"])
        )
        self.assertEqual(quick_preset["api_name"], "h3_apply_quick_preset")
        self.assertEqual(len(quick_preset["inputs"]), 1)
        self.assertEqual(len(quick_preset["outputs"]), 7)

        quality_id = component_ids["h3-quality"]
        quality_input = next(
            dependency
            for dependency in dependencies
            if any(target[0] == quality_id and target[1] == "input" for target in dependency["targets"])
        )
        self.assertEqual(len(quality_input["inputs"]), 8)
        self.assertEqual(len(quality_input["outputs"]), 3)
        self.assertFalse(quality_input["queue"])

        mode_id = component_ids["h3-mode"]
        mode_change = next(
            dependency
            for dependency in dependencies
            if any(target[0] == mode_id and target[1] == "change" for target in dependency["targets"])
        )
        self.assertEqual(len(mode_change["inputs"]), 8)
        self.assertEqual(len(mode_change["outputs"]), 6)

        seed_id = component_ids["h3-seed"]
        seed_input = next(
            dependency
            for dependency in dependencies
            if any(target[0] == seed_id and target[1] == "change" for target in dependency["targets"])
        )
        self.assertEqual(len(seed_input["inputs"]), 8)
        self.assertEqual(len(seed_input["outputs"]), 1)
        self.assertFalse(seed_input["queue"])

        prompt_id = component_ids["h3-prompt"]
        prompt_input = next(
            dependency
            for dependency in dependencies
            if any(target[0] == prompt_id and target[1] == "blur" for target in dependency["targets"])
        )
        self.assertEqual(len(prompt_input["inputs"]), 2)

        duration_id = component_ids["h3-duration"]
        duration_input = next(
            dependency
            for dependency in dependencies
            if any(target[0] == duration_id and target[1] == "input" for target in dependency["targets"])
        )
        self.assertEqual(len(duration_input["outputs"]), 3)

        initialize_id = component_ids["h3-initialize-trigger"]
        initialize = next(
            dependency
            for dependency in dependencies
            if any(target[0] == initialize_id and target[1] == "click" for target in dependency["targets"])
        )
        self.assertEqual(len(initialize["inputs"]), 4)
        self.assertEqual(len(initialize["outputs"]), 27 + control_count)
        self.assertEqual(initialize["outputs"][21:21 + control_count], generate["inputs"][-control_count:])
        self.assertEqual(initialize["trigger_mode"], "once")

        refresh_id = component_ids["h3-history-refresh"]
        history_refresh = next(
            dependency
            for dependency in dependencies
            if any(target[0] == refresh_id and target[1] == "click" for target in dependency["targets"])
        )
        self.assertEqual(len(history_refresh["inputs"]), 2)

        restore_id = component_ids["h3-history-restore"]
        restore = next(
            dependency
            for dependency in dependencies
            if any(target[0] == restore_id and target[1] == "click" for target in dependency["targets"])
        )
        self.assertEqual(len(restore["inputs"]), 2)
        self.assertEqual(len(restore["outputs"]), 25 + control_count)
        self.assertEqual(restore["outputs"][-control_count - 3:-3], generate["inputs"][-control_count:])
        self.assertFalse(restore["queue"])

        runtime_functions = [
            function
            for function in interface.fns.values()
            if function.name
            in {
                "_connect_runtime_updates",
                "_restart_runtime_updates",
                "_rescan_runtime_updates",
            }
        ]
        self.assertEqual(len(runtime_functions), 3)
        for function in runtime_functions:
            self.assertEqual(function.concurrency_id, "h3-runtime-control")
            self.assertEqual(function.concurrency_limit, 1)
            self.assertEqual(len(function.outputs), 5)

    def test_runtime_operation_disables_and_restores_all_runtime_controls(self):
        callback = mock.Mock(return_value="ready")
        updates = list(
            self.studio._runtime_operation(
                callback,
                "checking",
                "runtime",
                "http://127.0.0.1:8188",
                "fast",
            )
        )
        self.assertEqual(len(updates), 2)
        self.assertTrue(all(not update["interactive"] for update in updates[0][1:]))
        self.assertTrue(all(update["interactive"] for update in updates[1][1:]))
        self.assertIn("checking", updates[0][0])
        self.assertEqual(updates[1][0], "ready")
        callback.assert_called_once_with(
            "runtime",
            "http://127.0.0.1:8188",
            "fast",
        )

    def test_runtime_operation_restores_controls_after_runtime_failure(self):
        callback = mock.Mock(side_effect=RuntimeError("runtime unavailable"))
        updates = list(
            self.studio._runtime_operation(
                callback,
                "checking",
                "runtime",
                "http://127.0.0.1:8188",
                "fast",
            )
        )
        self.assertEqual(len(updates), 2)
        self.assertTrue(all(update["interactive"] for update in updates[1][1:]))
        self.assertIn("runtime unavailable", updates[1][0])

    def test_async_initialization_preserves_connected_low_ram_profile(self):
        readiness = RuntimeReadiness(
            runtime_root=Path("runtime"),
            server_url=self.studio.H3_SERVER_URL,
            connected=True,
            runtime_profile=self.studio.RUNTIME_PROFILE_LOW_RAM,
            ram_free_gib=4.0,
            commit_free_gib=4.0,
        )
        with mock.patch.object(
            self.studio,
            "resolve_runtime_root",
            return_value=Path("runtime"),
        ), mock.patch.object(
            self.studio,
            "inspect_readiness",
            return_value=readiness,
        ):
            updates = self.studio._initial_ui_updates(
                "runtime",
                self.studio.H3_SERVER_URL,
                self.studio.RUNTIME_PROFILE_FAST,
                "16:9",
            )
        self.assertEqual(len(updates), 21)
        self.assertEqual(updates[1]["value"], self.studio.RUNTIME_PROFILE_LOW_RAM)
        for update in (*updates[1:9], *updates[11:21]):
            self.assertTrue(update["interactive"])
        self.assertIn(
            'data-h3-preset="recommended"',
            updates[10],
        )

    def test_initial_preset_uses_quick_when_fast_profile_memory_is_too_low(self):
        readiness = RuntimeReadiness(
            runtime_root=Path("runtime"),
            server_url=self.studio.H3_SERVER_URL,
            connected=True,
            runtime_profile=self.studio.RUNTIME_PROFILE_FAST,
            ram_free_gib=4.0,
            commit_free_gib=8.0,
        )
        self.assertEqual(
            self.studio._initial_generation_preset(
                readiness,
                self.studio.RUNTIME_PROFILE_FAST,
            ),
            "quick",
        )
        self.assertEqual(
            self.studio._initial_generation_preset(
                readiness,
                self.studio.RUNTIME_PROFILE_LOW_RAM,
            ),
            "recommended",
        )

    def test_javascript_exposes_keyboard_and_busy_accessibility_contract(self):
        source = JAVASCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn('input.name = "h3-generation-mode"', source)
        self.assertIn('event.key === "ArrowRight"', source)
        self.assertIn('event.key === "Home"', source)
        self.assertIn('setH3Attribute(generate, "aria-keyshortcuts"', source)
        self.assertIn('setH3Data(studio, "h3Busy"', source)
        self.assertIn('validation.dataset.h3Control', source)
        self.assertIn('labelH3RadioGroup("#h3-mode", "生成モード"', source)
        self.assertIn("h3-mobile-generate-proxy", source)
        self.assertIn("if (node && node.textContent !== value)", source)
        self.assertIn("function scheduleH3StudioChrome()", source)
        self.assertIn('window.addEventListener("scroll", scheduleH3StudioChrome', source)
        self.assertIn("candidate.offsetParent !== null", source)
        self.assertIn("window.setTimeout(function ()", source)
        self.assertIn('"aria-label", "H3 プロンプト"', source)
        self.assertIn('"aria-label",\n            "履歴を選択"', source)
        self.assertIn('window.matchMedia("(max-width: 620px)")', source)
        self.assertIn("function addH3DescribedBy", source)
        self.assertIn("function removeH3DescribedBy", source)
        self.assertEqual(source.count("[aria-describedby~='h3-input-validation-message']"), 2)
        self.assertIn("generate.tabIndex = visible ? -1 : 0", source)
        self.assertIn("H3_PROMPT_DRAFT_KEY", source)
        self.assertIn("window.localStorage.setItem", source)
        self.assertIn("window.localStorage.getItem", source)
        self.assertIn('prompt.dispatchEvent(new window.Event("input"', source)
        self.assertIn("function syncH3ProgressAnnouncement()", source)
        self.assertIn("signature === h3LastProgressAnnouncement", source)
        self.assertIn("function requestH3Initialization()", source)
        self.assertIn("trigger === h3InitializationTrigger", source)
        self.assertIn("h3InitializationTrigger = trigger", source)

    def test_apple_inspired_workspace_keeps_web_accessibility_fallbacks(self):
        script = STUDIO_PATH.read_text(encoding="utf-8")
        style = STYLE_PATH.read_text(encoding="utf-8")
        self.assertIn("<h2>MiniMax H3 Studio</h2>", script)
        self.assertNotIn('class="h3-eyebrow"', script)
        self.assertIn("Apple-inspired workspace v2", style)
        self.assertIn(".h3-runtime-summary", style)
        self.assertIn("min-height: 44px", style)
        self.assertIn("prefers-reduced-motion", style)
        self.assertIn("prefers-reduced-transparency", style)
        self.assertIn("prefers-contrast: more", style)
        self.assertIn("forced-colors: active", style)


if __name__ == "__main__":
    unittest.main()
