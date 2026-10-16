"""Iris LoRA controls preserve selection contracts and show effective sampling."""

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import gradio as gr

from modules_forge import local_assets

ROOT = Path(__file__).resolve().parents[2]


def screen_module():
    spec = importlib.util.spec_from_file_location(
        "iris_lora_screen", ROOT / "extensions-builtin/iris-studio/scripts/iris_studio.py"
    )
    module = importlib.util.module_from_spec(spec)
    callbacks = types.ModuleType("modules.script_callbacks")
    callbacks.on_ui_tabs = lambda callback: None
    with patch.dict(sys.modules, {"modules.script_callbacks": callbacks}):
        spec.loader.exec_module(module)
    return module


class IrisLoRAUITests(unittest.TestCase):
    def setUp(self):
        self.module = screen_module()
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.library = patch.object(local_assets, "LIBRARY", Path(self.folder.name) / "library.json")
        self.library.start()
        self.addCleanup(self.library.stop)
        self.backend = types.SimpleNamespace(
            inventory=Mock(return_value=["style.safetensors", "turbo.safetensors"]),
            inspect=Mock(
                side_effect=lambda root, name: (
                    {"turbo_steps": 6 if "other" in name else 4, "shift": 4.0} if "turbo" in name else {}
                )
            ),
            sampling_settings=Mock(side_effect=self.sampling),
        )
        self.backend_patch = patch.object(self.module, "iris_lora", self.backend, create=True)
        self.backend_patch.start()
        self.addCleanup(self.backend_patch.stop)

    @staticmethod
    def sampling(loras, mode="standard", shift=4.0):
        return {"mode": "turbo", "shift": float(shift)} if mode == "turbo" else {"mode": "standard"}

    def controls(self, names, rows, standard=None, values=(80, 4, "blurry"), state=None, task="generate"):
        return self.module.lora_controls(state or {}, task, names, rows, standard or {}, *values)

    def test_native_controls_and_existing_load_contract(self):
        self.assertIs(self.module.gr, gr)
        blocks, _, _ = self.module.on_ui_tabs()[0]
        config = blocks.get_config_file()
        props = {item["props"].get("elem_id"): item["props"] for item in config["components"]}
        self.assertTrue(props["iris-loras"]["multiselect"])
        self.assertTrue(props["iris-loras"]["allow_custom_value"])
        self.assertGreaterEqual(props["iris-loras"]["min_width"], 240)
        self.assertEqual(props["iris-lora-strengths"]["headers"], ["LoRA", "強度"])
        self.assertEqual(props["iris-lora-strengths"]["datatype"], ["str", "number"])
        self.assertIn("iris-lora-refresh", props)
        self.assertTrue(
            any(item["props"].get("label") == "ネガティブプロンプト（CFG > 1で反映）" for item in config["components"])
        )
        loads = [item for item in config["dependencies"] if any(event == "load" for _, event in item["targets"])]
        self.assertEqual(len(loads), 1)
        self.assertIs(blocks.fns[loads[0]["id"]].fn, self.module.load_selection)
        self.assertEqual(len(loads[0]["outputs"]), 16)

    def test_turbo_selection_and_removal_only_show_recommendations(self):
        turbo = self.controls(["turbo.safetensors"], [["turbo.safetensors", 1]])
        self.assertTrue(all("value" not in turbo[index] for index in (5, 6, 7)))
        self.assertTrue(all(turbo[index]["interactive"] for index in (5, 6, 7)))
        self.assertIn("4ステップ", turbo[4])
        self.assertIn("推奨", turbo[4])
        restored = self.controls([], [], turbo[0], values=(4, 1, ""))
        self.assertTrue(all("value" not in restored[index] for index in (5, 6, 7)))
        self.assertTrue(all(restored[index]["interactive"] for index in (5, 6, 7)))

    def test_manual_turbo_values_survive_refresh_strength_edits_and_downstream_tasks(self):
        selected = self.controls(["turbo.safetensors"], [["turbo.safetensors", 1]])
        names = ["turbo.safetensors", "style.safetensors"]
        rows = [["turbo.safetensors", 1], ["style.safetensors", 0.5]]
        manual = (8, 2, "grain")
        added = self.controls(names, rows, selected[0], values=manual)
        refreshed = self.module.sync_loras({}, "generate", names, rows, added[0], *manual)
        rows[1][1] = 0.25
        changed = self.module.edit_loras({}, "generate", names, rows, refreshed[0], *manual)
        for updates in (added, refreshed, changed):
            self.assertTrue(all("value" not in updates[index] for index in (5, 6, 7)))
        self.backend.inspect.reset_mock()
        depth = self.controls(names, rows, changed[0], values=manual, task="depth")
        upscale = self.controls(names, rows, depth[0], values=manual, task="upscale")
        self.backend.inspect.assert_not_called()
        for updates in (depth, upscale):
            self.assertTrue(all("value" not in updates[index] for index in (5, 6, 7)))
        returned = self.controls(names, rows, upscale[0], values=manual)
        self.assertTrue(all("value" not in returned[index] for index in (5, 6, 7)))
        restored = self.controls(["style.safetensors"], rows[1:], returned[0], values=manual)
        self.assertTrue(all("value" not in restored[index] for index in (5, 6, 7)))

    def test_different_turbo_adapter_displays_its_own_recommendations_without_writing_values(self):
        selected = self.controls(["turbo.safetensors"], [["turbo.safetensors", 1]])
        switched = self.controls(
            ["turbo-other.safetensors"], [["turbo-other.safetensors", 1]], selected[0], (8, 2, "grain")
        )
        self.assertTrue(all("value" not in switched[index] for index in (5, 6, 7)))
        self.assertIn("6ステップ", switched[4])

    def test_zero_strength_skips_inspection_and_does_not_enable_turbo(self):
        result = self.controls(["turbo.safetensors"], [["turbo.safetensors", 0]])
        self.backend.inspect.assert_not_called()
        self.assertEqual(result[0]["mode"], "standard")
        self.assertNotIn("value", result[5])

    def test_dropdown_changes_cover_removal_without_echoing_selected_values(self):
        blocks, _, _ = self.module.on_ui_tabs()[0]
        config = blocks.get_config_file()
        dropdown = next(item for item in config["components"] if item["props"].get("elem_id") == "iris-loras")
        changes = [
            item
            for item in config["dependencies"]
            if any(identifier == dropdown["id"] and event == "change" for identifier, event in item["targets"])
        ]
        selection_changes = [item for item in changes if blocks.fns[item["id"]].fn is self.module.edit_loras]
        self.assertEqual(len(selection_changes), 1, "Gradio token removal and clear emit one selection edit")
        by_id = {item["id"]: item for item in config["components"]}
        for item in changes:
            if item not in selection_changes:
                self.assertTrue(
                    all(by_id[identifier]["type"] == "button" for identifier in item["outputs"]),
                    "Other selection listeners may update disclosure badges only",
                )
        previous = {"mode": "turbo", "steps": 80, "cfg": 4, "negative": "blurry"}
        result = self.module.edit_loras({}, "generate", [], [], previous, 4, 1, "")
        self.assertNotIn("value", result[1], "A user selection callback must not write an older selection back")
        self.assertTrue(all("value" not in result[index] for index in (5, 6, 7)))

    def test_strength_table_row_count_follows_selection(self):
        result = self.controls(
            ["turbo.safetensors", "style.safetensors"],
            [["turbo.safetensors", 1], ["style.safetensors", 0.5]],
        )
        self.assertEqual(result[2]["row_count"], (2, "fixed"))
        cleared = self.controls([], [], result[0], values=(4, 1, ""))
        self.assertEqual(cleared[2]["row_count"], (0, "fixed"))

    def test_strength_edits_do_not_retrigger_dropdown_selection_changes(self):
        blocks, _, _ = self.module.on_ui_tabs()[0]
        config = blocks.get_config_file()
        strengths = next(item for item in config["components"] if item["props"].get("elem_id") == "iris-lora-strengths")
        inputs = [
            item
            for item in config["dependencies"]
            if any(identifier == strengths["id"] and event == "input" for identifier, event in item["targets"])
        ]
        self.assertEqual(len(inputs), 1)
        self.assertIs(blocks.fns[inputs[0]["id"]].fn, self.module.edit_loras)
        result = self.module.edit_loras(
            {}, "generate", ["style.safetensors"], [["style.safetensors", 0.5]], {}, 80, 4, ""
        )
        self.assertNotIn("value", result[1])
        self.assertEqual(result[2]["value"], [["style.safetensors", 0.5]])

    def test_initial_change_cannot_clear_persisted_selection_before_load(self):
        saved = {"loras": [{"name": "turbo.safetensors", "strength": 1}]}
        local_assets.save_selection("iris_lora", saved)
        result = self.module.edit_loras({}, "generate", [], [], {"initialized": False}, 100, 3, "")
        self.assertEqual(local_assets.selection("iris_lora"), saved)
        self.assertTrue(all("value" not in item for item in result))
        loaded = self.module.load_loras()
        self.assertTrue(loaded[0]["initialized"])
        self.assertEqual(loaded[1]["value"], ["turbo.safetensors"])
        blocks, _, _ = self.module.on_ui_tabs()[0]
        states = [item.value for item in blocks.blocks.values() if isinstance(item, gr.State)]
        self.assertTrue(any(item.get("initialized") is False for item in states if isinstance(item, dict)))

    def test_task_and_refresh_sync_cannot_overwrite_initialization_during_page_load(self):
        for task in ("generate", "depth", "upscale"):
            with self.subTest(task=task):
                updates = self.module.sync_loras({}, task, [], [], {"initialized": False}, 100, 3, "")
                self.assertTrue(all(item == gr.skip() for item in updates))
        self.backend.inspect.assert_not_called()
        loaded = self.module.load_loras(task="depth")
        self.assertTrue(loaded[0]["initialized"])
        returned = self.module.edit_loras(
            {}, "generate", ["turbo.safetensors"], [["turbo.safetensors", 1]], loaded[0], 100, 3, ""
        )
        self.assertTrue(all("value" not in returned[index] for index in (5, 6, 7)))

    def test_standard_lora_edits_and_refresh_do_not_echo_sampling_inputs(self):
        for callback in (self.module.edit_loras, self.module.sync_loras):
            with self.subTest(callback=callback.__name__):
                updates = callback(
                    {},
                    "generate",
                    ["style.safetensors"],
                    [["style.safetensors", 0.5]],
                    {"initialized": True, "mode": "standard"},
                    80,
                    4,
                    "blurry",
                )
                self.assertTrue(all("value" not in updates[index] for index in (5, 6, 7)))

    def test_task_and_refresh_sync_do_not_echo_dropdown_values(self):
        result = self.module.sync_loras(
            {}, "generate", ["style.safetensors"], [["style.safetensors", 0.5]], {}, 80, 4, ""
        )
        self.assertNotIn("value", result[1])
        self.assertIn("style.safetensors", result[1]["choices"])

    def test_lora_persistence_does_not_overwrite_checkpoint_or_precision(self):
        checkpoint = {
            "checkpoint": "custom.safetensors",
            "precision": "normal",
            "base_precisions": {"generate": "w4a8"},
        }
        local_assets.save_selection("iris", checkpoint)
        self.controls(["style.safetensors"], [["style.safetensors", 0.75]])
        self.assertEqual(local_assets.selection("iris"), checkpoint)
        self.assertEqual(
            local_assets.selection("iris_lora"), {"loras": [{"name": "style.safetensors", "strength": 0.75}]}
        )
        with patch.object(local_assets, "save_selection", side_effect=AssertionError("reload is read-only")):
            loaded = self.module.load_loras()
        self.assertEqual(loaded[1]["value"], ["style.safetensors"])
        self.assertEqual(loaded[2]["value"], [["style.safetensors", 0.75]])

    def test_turbo_fractional_strength_is_allowed_and_all_lora_controls_lock_during_job(self):
        result = self.controls(["turbo.safetensors"], [["turbo.safetensors", 0.5]])
        self.assertIn("推奨", result[4])
        locked = self.controls(["style.safetensors"], [["style.safetensors", 1]], state={"job": "job"})
        self.assertTrue(all(not locked[index]["interactive"] for index in (1, 2, 3)))
        hidden = self.controls(["style.safetensors"], [["style.safetensors", 1]], task="depth")
        self.assertFalse(hidden[8]["visible"])

    def test_validation_errors_keep_last_valid_turbo_and_manual_values_until_corrected(self):
        selected = self.controls(["turbo.safetensors"], [["turbo.safetensors", 1]])
        manual = (8, 2, "grain")
        original_inspect = self.backend.inspect.side_effect

        def inspect(root, name):
            if name == "missing.safetensors":
                raise ValueError("LoRAが見つかりません。")
            return original_inspect(root, name)

        self.backend.inspect.side_effect = inspect
        for names, rows, error in (
            (
                ["turbo.safetensors", "missing.safetensors"],
                [["turbo.safetensors", 1], ["missing.safetensors", 1]],
                "見つかりません",
            ),
        ):
            with self.subTest(error=error):
                invalid = self.controls(names, rows, selected[0], values=manual)
                self.assertIn(error, invalid[4])
                self.assertTrue(all("value" not in invalid[index] for index in (5, 6, 7)))
                corrected = self.controls(["turbo.safetensors"], [["turbo.safetensors", 1]], invalid[0], values=manual)
                self.assertTrue(all("value" not in corrected[index] for index in (5, 6, 7)))
                self.assertIn("推奨", corrected[4])

    def test_invalid_saved_selection_still_loads_initial_controls(self):
        local_assets.save_selection("iris_lora", {"loras": [{"name": "turbo.safetensors", "strength": 3}]})
        loaded = self.module.load_loras()
        self.assertIn("強度", loaded[4])
        self.assertEqual(tuple(loaded[index]["value"] for index in (5, 6, 7)), (100, 3, ""))
        blocks, _, _ = self.module.on_ui_tabs()[0]
        self.assertIsInstance(blocks, gr.Blocks)

    def test_start_includes_only_generation_loras_and_keeps_manual_turbo_settings(self):
        with (
            patch.object(self.module, "ready", return_value=True),
            patch.object(self.module.SERVICE, "start", return_value="job") as service,
        ):
            response = self.module.start(
                {},
                "generate",
                "int8",
                "fox",
                None,
                1,
                "1024×1024",
                8,
                2,
                "grain",
                "base",
                ["turbo.safetensors"],
                [["turbo.safetensors", 1]],
            )
        request = service.call_args.args[0]
        self.assertEqual(request["loras"], [{"name": "turbo.safetensors", "strength": 1.0}])
        self.assertEqual((request["steps"], request["cfg"], request["negative_prompt"]), (8, 2, "grain"))
        self.assertEqual(len(response), 8)
        self.backend.inspect.reset_mock()
        with (
            patch.object(self.module, "ready", return_value=True),
            patch.object(self.module.SERVICE, "start", return_value="job") as service,
        ):
            self.module.start(
                {},
                "depth",
                "int8",
                "",
                "image",
                1,
                "1024×1024",
                80,
                4,
                "",
                "base",
                ["missing.safetensors"],
                [["missing.safetensors", 1]],
            )
        self.assertNotIn("loras", service.call_args.args[0])
        self.backend.inspect.assert_not_called()

    def test_start_and_terminal_poll_wrappers_lock_and_restore_controls(self):
        with (
            patch.object(self.module, "ready", return_value=True),
            patch.object(self.module.SERVICE, "start", return_value="job"),
        ):
            started = self.module.start_with_loras(
                {}, "generate", "int8", "fox", None, 1, "1024×1024", 80, 4, "", "base", [], []
            )
        self.assertEqual(len(started), 14)
        self.assertTrue(all(not item["interactive"] for item in started[8:]))
        with patch.object(self.module.SERVICE, "poll", return_value={"status": "cancelled"}):
            stopped = self.module.poll_with_loras(started[0])
        self.assertEqual(len(stopped), 19)
        self.assertTrue(all(item["interactive"] for item in stopped[13:16]))
        self.assertTrue(all(not item["interactive"] for item in stopped[16:]))

    def test_multiple_turbo_loras_and_fractional_strength_start_without_replacing_sampler(self):
        names = ["turbo.safetensors", "turbo-other.safetensors"]
        rows = [[names[0], 0.5], [names[1], -0.25]]
        with (
            patch.object(self.module, "ready", return_value=True),
            patch.object(self.module.SERVICE, "start", return_value="job") as service,
        ):
            self.module.start({}, "generate", "int8", "fox", None, 1, "1024×1024", 80, 4, "grain", "base", names, rows)
        request = service.call_args.args[0]
        self.assertEqual(request["lora_sampling"], "standard")
        self.assertEqual((request["steps"], request["cfg"], request["negative_prompt"]), (80, 4, "grain"))
        self.assertEqual([item["strength"] for item in request["loras"]], [0.5, -0.25])


if __name__ == "__main__":
    unittest.main()
