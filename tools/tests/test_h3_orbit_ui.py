import unittest
from pathlib import Path
from unittest import mock

import gradio  # Preload before the loader's temporary modules stubs.

from tools.tests.test_minimax_h3_studio import load_studio_module


class OrbitUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ui = load_studio_module()

    def test_round_trip_preserves_separate_custom_settings(self):
        ui = self.ui
        slots = ui._initial_mode_slots()
        normal = ("9:16", "custom:736x1120", 7.5, 24, "beta", 1.0)
        entering = ui._switch_mode_settings("orbit", slots, *normal)
        self.assertEqual(entering[0]["normal"], normal)
        self.assertEqual([value["value"] for value in entering[1:7]], ["1:1", "native", 3.0, 28, "simple", 1.0])
        custom_orbit = ("1:1", "custom:864x768", 4, 32, "simple", 0.8)
        leaving = ui._switch_mode_settings("keyframes", entering[0], *custom_orbit)
        self.assertEqual([value["value"] for value in leaving[1:7]], list(normal))
        returned = ui._switch_mode_settings("orbit", leaving[0], *normal)
        self.assertEqual([value["value"] for value in returned[1:7]], list(custom_orbit))
        self.assertEqual(entering[3]["minimum"], 3)
        self.assertEqual(leaving[3]["minimum"], 5)

    def test_fixed_duration_ranges_and_switch_select_active_value(self):
        ui = self.ui
        interface = ui._build_ui()[0][0]
        self.assertIsInstance(interface, gradio.Blocks)
        components = {item["props"].get("elem_id"): item["props"] for item in interface.get_config_file()["components"]}
        self.assertEqual((components["h3-duration"]["minimum"], components["h3-duration"]["maximum"]), (5, 15))
        self.assertEqual(
            (components["h3-orbit-duration"]["minimum"], components["h3-orbit-duration"]["maximum"]), (3, 15)
        )
        entering = ui._switch_mode_settings_ui(
            "orbit", ui._initial_mode_slots(), "16:9", "preview", 7.5, 24, "beta", 1.0, 3.0
        )
        self.assertNotIn("value", entering[3])
        self.assertEqual(entering[-1]["value"], 3.0)
        leaving = ui._switch_mode_settings_ui("keyframes", entering[0], "1:1", "native", 7.5, 28, "simple", 1.0, 4.0)
        self.assertEqual(leaving[3]["value"], 7.5)
        self.assertNotIn("value", leaving[-1])
        self.assertEqual(leaving[0]["orbit"][2], 4.0)

    def test_ui_generation_uses_active_duration_without_changing_normal_api(self):
        ui = self.ui
        update = ("", None, "", None, None, {}, {"interactive": True}, None)
        for mode, expected in (("orbit", 3.0), ("text", 7.5)):
            values = (mode, "subject", None, None, None, None, None, "1:1", "native", 7.5, 28, -1, "simple", "match")
            with mock.patch.object(ui, "_generate", return_value=iter([update])) as generate:
                list(ui._generate_ui("runtime", "url", "fast", "photo.png", 1.0, 3.0, *values))
            self.assertEqual(generate.call_args.args[12], expected)

    def test_reset_only_changes_orbit_slot(self):
        ui = self.ui
        slots = ui._initial_mode_slots()
        slots["normal"] = ("9:16", "draft", 8, 30, "beta", 1.0)
        slots["orbit"] = ("16:9", "draft", 4, 12, "beta", 0.5)
        reset = ui._reset_orbit_settings(slots)
        self.assertEqual(reset[0]["normal"], slots["normal"])
        self.assertEqual(reset[0]["orbit"], ("1:1", "native", 3.0, 28, "simple", 1.0))

    def test_canvas_updates_do_not_react_to_programmatic_slot_restore(self):
        interface = self.ui._build_ui()[0][0]
        config = interface.get_config_file()
        ids = {component["props"].get("elem_id"): component["id"] for component in config["components"]}
        dimensions = [ids["h3-width"], ids["h3-height"]]
        dependencies = config["dependencies"]
        for control in ("h3-aspect", "h3-quality"):
            callbacks = [
                item
                for item in dependencies
                if item["outputs"] == dimensions and any(target[0] == ids[control] for target in item["targets"])
            ]
            self.assertEqual(len(callbacks), 1)
            self.assertEqual(callbacks[0]["targets"][0][1], "input")
        for button in ("h3-preset-quick", "h3-preset-recommended", "h3-preset-final", "h3-initialize-trigger"):
            clicked = next(
                item
                for item in dependencies
                if any(target[0] == ids[button] and target[1] == "click" for target in item["targets"])
            )
            after = [
                item
                for item in dependencies
                if item["trigger_after"] == clicked["id"] and item["outputs"] == dimensions
            ]
            self.assertEqual(len(after), 1, button)
            result = interface.fns[after[0]["id"]].fn("16:9", "custom:736x512")
            self.assertEqual([update["value"] for update in result], [736, 512])

    def test_incomplete_or_invalid_dimensions_survive_mode_round_trip(self):
        ui = self.ui
        for quality, expected_width in (("custom:257x768", 257), ("custom:Nonex768", None)):
            slots = ui._initial_mode_slots()
            slots["active"] = "orbit"
            slots["orbit"] = ("1:1", quality, 3, 28, "simple", 1.0)
            leaving = ui._switch_mode_settings("text", slots, *slots["orbit"])
            returning = ui._switch_mode_settings("orbit", leaving[0], *leaving[0]["normal"])
            self.assertEqual(returning[2]["value"], quality)
            self.assertEqual(returning[7]["value"], expected_width)
            self.assertEqual(returning[8]["value"], 768)

    def test_normal_settings_keep_existing_seed_validation(self):
        ui = self.ui
        validation = ui._input_validation_html("Seed is invalid", "settings", "seed")
        result = ui._custom_settings_for_mode(
            "text", "16:9", "preview", 5, 20, "oops", "simple", "match", 1.0, "off", validation
        )
        self.assertNotIn("value", result[2])
        fixed = ui._custom_settings_for_mode(
            "orbit", "1:1", "native", 3, 28, -1, "simple", "match", 1.0, "off", validation
        )
        self.assertEqual(fixed[2], "")

    def test_orbit_validation_targets_its_photo_and_strength_controls(self):
        for control in ("orbit_photo", "orbit_strength"):
            rendered = self.ui._input_validation_html("Orbit input invalid", "orbit", control)
            self.assertIn('data-h3-invalid="orbit"', rendered)
            self.assertIn(f'data-h3-control="{control}"', rendered)

    def test_explicit_orbit_prompt_action_appends(self):
        prompt, _ = self.ui._prompt_template_for_mode("orbit", "My subject description", "")
        self.assertTrue(prompt.startswith("My subject description\n\n"))
        self.assertIn("360", prompt)
        self.assertIn("camera", prompt.lower())
        empty, _ = self.ui._prompt_template_for_mode("orbit", "", "")
        self.assertEqual(prompt.removeprefix("My subject description\n\n"), empty)
        repeated, _ = self.ui._prompt_template_for_mode("orbit", prompt, "")
        self.assertEqual(repeated, prompt)

    def test_orbit_request_uses_only_dedicated_photo(self):
        request = self.ui._request_from_ui(
            "orbit",
            "subject",
            "normal-start.png",
            "normal-end.png",
            ["ref.png"],
            None,
            None,
            "1:1",
            "native",
            3,
            28,
            -1,
            "simple",
            "match",
            orbit_photo="orbit.png",
            orbit_strength=0.8,
        )
        self.assertEqual(request.first_frame, "orbit.png")
        self.assertIsNone(request.last_frame)
        self.assertEqual(request.reference_images, ())
        self.assertEqual(request.orbit_strength, 0.8)
        self.assertFalse(request.has_audio)

    def test_orbit_summary_reports_real_timing_and_deviations(self):
        ui = self.ui
        summary = ui._mode_settings_summary("orbit", "1:1", "native", 3, 28, "simple", "match", 1.0)
        self.assertIn("73 frames", summary)
        self.assertIn("3.04 sec", summary)
        self.assertIn("無音", summary)
        self.assertNotIn("作者推奨から変更:", summary)
        changed = ui._mode_settings_summary("orbit", "16:9", "preview", 5, 20, "beta", "match", 0.8)
        self.assertIn("作者推奨から変更:", changed)
        self.assertIn("未検証", changed)

    def test_gate_and_explicit_install_follow_selected_runtime(self):
        ui = self.ui
        selected = Path("external-runtime")
        with mock.patch.object(ui.orbit_assets, "installed", return_value=True) as installed:
            gate, _ = ui._generation_gate("orbit", photo="photo.png", runtime_value=str(selected))
        self.assertTrue(gate["interactive"])
        installed.assert_called_once_with(Path(ui.script_path), runtime_root=selected)
        with (
            mock.patch.object(ui, "resolve_runtime_root", return_value=selected),
            mock.patch.object(ui.orbit_assets, "install") as install,
        ):
            list(ui._install_orbit(str(selected)))
        install.assert_called_once_with(Path(ui.script_path), runtime_root=selected)
        interface = ui._build_ui()[0][0]
        callbacks = [fn for fn in interface.fns.values() if fn.fn and fn.fn.__name__ == "_mode_chrome_ui"]
        self.assertTrue(all(any(component.elem_id == "h3-runtime-path" for component in fn.inputs) for fn in callbacks))

    def test_install_gate_is_local_and_keeps_normal_modes_available(self):
        ui = self.ui
        with (
            mock.patch.object(ui.orbit_assets, "installed", return_value=False),
            mock.patch.object(ui.orbit_assets, "install") as install,
        ):
            update, reason = ui._generation_gate("orbit")
            self.assertFalse(update["interactive"])
            self.assertIn("準備", reason)
            normal, _ = ui._generation_gate("keyframes")
            self.assertTrue(normal["interactive"])
            install.assert_not_called()
        with mock.patch.object(ui.orbit_assets, "installed", return_value=True):
            busy, reason = ui._generation_gate("orbit", installing=True)
            self.assertFalse(busy["interactive"])
            self.assertIn("導入中", reason)
            ready, _ = ui._generation_gate("orbit", photo="photo.png")
            self.assertTrue(ready["interactive"])
            self.assertEqual(ready["value"], "周回動画を生成（無音）")

    def test_empty_orbit_photo_disables_generation_even_when_lora_ready(self):
        ui = self.ui
        with mock.patch.object(ui.orbit_assets, "installed", return_value=True):
            update, reason = ui._generation_gate("orbit", photo=None)
            ready, ready_reason = ui._generation_gate("orbit", photo="photo.png")
        self.assertFalse(update["interactive"])
        self.assertIn("写真を追加", reason)
        self.assertTrue(ready["interactive"])
        self.assertEqual(ready_reason, "")

    def test_install_generator_reports_failure_and_reenables_retry(self):
        ui = self.ui
        with mock.patch.object(ui.orbit_assets, "install", side_effect=ValueError("SHA-256 mismatch")):
            updates = list(ui._install_orbit())
        self.assertEqual([item[2] for item in updates], [True, False])
        self.assertFalse(updates[0][1]["interactive"])
        self.assertTrue(updates[-1][1]["interactive"])
        self.assertIn("SHA-256 mismatch", updates[-1][0])
        self.assertNotIn("%", updates[0][0])

    def test_install_success_reports_ready_without_progress_percentage(self):
        ui = self.ui
        with mock.patch.object(ui.orbit_assets, "install", return_value=Path("orbit.safetensors")):
            updates = list(ui._install_orbit())
        self.assertEqual([item[2] for item in updates], [True, False])
        self.assertIn("準備済み", updates[-1][0])
        self.assertFalse(updates[-1][1]["interactive"])

    def test_hybrid_stops_orbit_and_controlnet_only_adds_untested_note(self):
        from modules_forge.minimax_h3_hybrid import H3Hybrid

        ui = self.ui
        option = ui.H3Acceleration(hybrid=H3Hybrid(enabled=True))
        with mock.patch.object(ui.orbit_assets, "installed", return_value=True):
            update, reason = ui._generation_gate("orbit", acceleration=option)
        self.assertFalse(update["interactive"])
        self.assertIn("長尺生成をオフ", reason)
        self.assertTrue(option.hybrid.enabled)
        summary = ui._mode_settings_summary("orbit", "1:1", "native", 3, 28, "simple", "match", 1, control_mode="canny")
        self.assertIn("未検証の併用: ControlNet", summary)

    def test_generation_finishes_with_orbit_label_and_missing_lora_never_runs(self):
        ui = self.ui
        values = (
            "runtime",
            "url",
            "fast",
            "orbit",
            "subject",
            None,
            None,
            None,
            None,
            None,
            "1:1",
            "native",
            3,
            28,
            -1,
            "simple",
            "match",
        )
        complete = [{"stage": "complete", "message": "done", "path": "result.mp4"}]
        with (
            mock.patch.object(ui.orbit_assets, "installed", return_value=True),
            mock.patch.object(ui, "resolve_runtime_root", return_value=Path("runtime")),
            mock.patch.object(ui, "run_generation", return_value=iter(complete)),
            mock.patch.object(ui, "_history_state", return_value=([], "history", [])),
        ):
            results = list(ui._generate(*values, orbit_photo="orbit.png"))
        self.assertEqual(results[-1][6]["value"], "周回動画を生成（無音）")
        self.assertTrue(results[-1][6]["interactive"])
        with (
            mock.patch.object(ui.orbit_assets, "installed", return_value=False),
            mock.patch.object(ui, "run_generation") as generate,
        ):
            missing = list(ui._generate(*values, orbit_photo="orbit.png"))
        generate.assert_not_called()
        self.assertFalse(missing[-1][6]["interactive"])

    def test_history_routes_to_target_slot_without_losing_opposite_slot(self):
        ui = self.ui
        slots = ui._initial_mode_slots()
        normal = ("9:16", "balanced", 8, 22, "beta", 1.0)
        request = ui.H3Request(
            mode="orbit",
            prompt="saved",
            aspect="1:1",
            quality="native",
            duration_seconds=4,
            steps=32,
            orbit_strength=0.6,
        )
        with (
            mock.patch.object(ui, "_history_state", return_value=([], "", [])),
            mock.patch.object(ui, "load_history_request", return_value=request),
        ):
            updates = ui._restore_history_ui("saved.mp4", "runtime", slots, *normal)
        saved_slots, photo, strength = updates[-3:]
        self.assertEqual(saved_slots["normal"], normal)
        self.assertEqual(saved_slots["orbit"], ("1:1", "native", 4, 32, "simple", 0.6))
        self.assertTrue(all("value" not in item for item in updates[2:7]))
        self.assertIsNone(photo["value"])
        self.assertEqual(strength["value"], 0.6)
        self.assertIn("写真を再指定", updates[21])

    def test_normal_history_restored_from_orbit_leaves_orbit_slot_and_photo(self):
        ui = self.ui
        slots = ui._initial_mode_slots()
        slots["active"] = "orbit"
        orbit = ("1:1", "custom:864x768", 4, 32, "simple", 0.8)
        request = ui.H3Request(
            mode="keyframes", prompt="saved", aspect="9:16", quality="draft", duration_seconds=7, steps=20
        )
        with (
            mock.patch.object(ui, "_history_state", return_value=([], "", [])),
            mock.patch.object(ui, "load_history_request", return_value=request),
        ):
            updates = ui._restore_history_ui("saved.mp4", "runtime", slots, *orbit)
        self.assertEqual(updates[-3]["orbit"], orbit)
        self.assertEqual(updates[-3]["normal"][:5], ("9:16", "draft", 7, 20, "simple"))
        self.assertNotIn("value", updates[-2])


if __name__ == "__main__":
    unittest.main()
