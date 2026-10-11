"""Selection must reach the exact local artifact, including same-path updates."""

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import gradio as gr
import torch
from safetensors.torch import save_file

from modules_forge import local_assets
from modules_forge.iris import core, runtime

ROOT = Path(__file__).resolve().parents[2]


def artifact(directory, name="model.safetensors", value=1):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.yaml").write_text("model:\n  hidden_size: 2\n", encoding="utf-8")
    path = directory / name
    save_file(
        {
            key: torch.full((2, 2), float(value))
            for key in (
                "s_embedder.proj.weight",
                "t_embedder.mlp.0.weight",
                "final_layer.linear.weight",
                "blocks.0.weight",
            )
        },
        path,
    )
    return path.resolve()


def screen_module():
    spec = importlib.util.spec_from_file_location(
        "iris_selection_screen", ROOT / "extensions-builtin/iris-studio/scripts/iris_studio.py"
    )
    module = importlib.util.module_from_spec(spec)
    callbacks = types.ModuleType("modules.script_callbacks")
    callbacks.on_ui_tabs = lambda callback: None
    with patch.dict(sys.modules, {"modules.script_callbacks": callbacks}):
        spec.loader.exec_module(module)
    return module


class SelectionTests(unittest.TestCase):
    def test_reload_uses_latest_library_after_ui_construction_without_writing(self):
        module = screen_module()
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(local_assets, "LIBRARY", Path(folder) / "library.json"),
        ):
            root = Path(folder)
            local_assets.save_selection("iris", {"checkpoint": "base", "base_precisions": {"generate": "int8"}})
            module.on_ui_tabs()
            selected = str(artifact(root / "outside/新しい FT"))
            directory = root / "search"
            directory.mkdir()
            local_assets.save_selection(
                "iris",
                {
                    "checkpoint": selected,
                    "precision": "int8",
                    "directory": str(directory),
                    "base_precisions": {"generate": "w4a8"},
                },
            )
            before = local_assets.LIBRARY.read_bytes()
            with (
                patch.object(local_assets, "selection", wraps=local_assets.selection) as read,
                patch.object(local_assets, "save_selection", side_effect=AssertionError("reload must be read-only")),
            ):
                view = module.load_selection()
            read.assert_called_once_with("iris")
            self.assertEqual(len(view), 16)
            self.assertEqual(view[13]["value"], selected)
            self.assertIn(selected, [value for _, value in view[13]["choices"]])
            self.assertEqual(view[12]["value"], "normal")
            self.assertFalse(view[12]["interactive"])
            self.assertEqual(view[14]["selection_checkpoint"], selected)
            self.assertEqual(view[14]["base_precisions"]["generate"], "w4a8")
            self.assertIn(Path(selected).name, view[8])
            self.assertEqual(view[15]["value"], "generate")
            self.assertEqual(local_assets.LIBRARY.read_bytes(), before)

    def test_reload_keeps_missing_selection_and_saved_precision_without_fallback(self):
        module = screen_module()
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(local_assets, "LIBRARY", Path(folder) / "library.json"),
        ):
            root = Path(folder)
            selected = artifact(root / "outside")
            selected.unlink()
            local_assets.save_selection(
                "iris", {"checkpoint": str(selected), "precision": "normal", "directory": str(root / "missing-search")}
            )
            before = local_assets.LIBRARY.read_bytes()
            view = module.load_selection()
            self.assertEqual(view[13]["value"], str(selected))
            self.assertIn(str(selected), [value for _, value in view[13]["choices"]])
            self.assertEqual(view[12]["value"], "normal")
            self.assertFalse(view[12]["interactive"])
            self.assertFalse(view[7]["interactive"])
            self.assertEqual(view[14]["selection_checkpoint"], str(selected))
            self.assertIn(selected.name, view[8])
            self.assertEqual(local_assets.LIBRARY.read_bytes(), before)

    def test_initial_ui_selection_is_read_only_and_switch_can_use_one_saved_snapshot(self):
        module = screen_module()
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(local_assets, "LIBRARY", Path(folder) / "library.json"),
        ):
            selected = str(artifact(Path(folder) / "ft"))
            saved = {"checkpoint": selected, "precision": "normal", "directory": folder}
            local_assets.save_selection("iris", saved)
            before = local_assets.LIBRARY.read_bytes()
            with patch.object(
                local_assets, "save_selection", side_effect=AssertionError("construction must be read-only")
            ):
                blocks, _, _ = module.on_ui_tabs()[0]
                with patch.object(local_assets, "selection", side_effect=AssertionError("use the passed snapshot")):
                    view = module.switch({}, "generate", "int8", selected, persist=False, saved=saved)
            self.assertEqual(view[12]["value"], "normal")
            self.assertEqual(local_assets.LIBRARY.read_bytes(), before)
            self.assertTrue(any(item["type"] == "state" for item in blocks.get_config_file()["components"]))

    def test_rendered_root_retains_read_only_load_and_user_input_persistence_bindings(self):
        module = screen_module()
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(local_assets, "LIBRARY", Path(folder) / "library.json"),
        ):
            blocks, _, _ = module.on_ui_tabs()[0]
            with gr.Blocks() as root:
                blocks.render()
        config = root.get_config_file()
        dependencies = config["dependencies"]
        loads = [item for item in dependencies if any(event == "load" for _, event in item["targets"])]
        self.assertEqual(len(loads), 1)
        load = loads[0]
        self.assertIs(root.fns[load["id"]].fn, module.load_selection)
        self.assertEqual(load["inputs"], [])
        self.assertEqual(len(load["outputs"]), 16)
        self.assertFalse(load["queue"])
        self.assertEqual(load["show_progress"], "hidden")
        components = {item["id"]: item for item in config["components"]}
        self.assertEqual(components[load["outputs"][14]]["type"], "state")
        self.assertEqual(components[load["outputs"][15]]["props"]["elem_id"], "iris-task")
        self.assertFalse(any(item["props"].get("label") == "探索フォルダー" for item in components.values()))
        targets = {(identifier, event) for item in dependencies for identifier, event in item["targets"]}
        for identifier in (load["outputs"][12], load["outputs"][13]):
            self.assertIn((identifier, "input"), targets)
            self.assertNotIn((identifier, "change"), targets)
        self.assertIn((load["outputs"][15], "change"), targets)

    def test_forge_ui_defaults_cannot_replace_library_selection(self):
        module = screen_module()
        spec = importlib.util.spec_from_file_location("iris_ui_loadsave", ROOT / "modules/ui_loadsave.py")
        loadsave_module = importlib.util.module_from_spec(spec)
        components = types.ModuleType("modules.ui_components")
        components.InputAccordionImpl = type("InputAccordionImpl", (), {})
        components.ToolButton = type("ToolButton", (), {})
        with patch.dict(
            sys.modules, {"modules.errors": types.ModuleType("modules.errors"), "modules.ui_components": components}
        ):
            spec.loader.exec_module(loadsave_module)
        self.assertIs(module.gr, gr)
        self.assertIs(loadsave_module.gr, gr)
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(local_assets, "LIBRARY", Path(folder) / "library.json"),
        ):
            selected = str(artifact(Path(folder) / "ft"))
            local_assets.save_selection("iris", {"checkpoint": selected, "precision": "normal", "directory": folder})
            blocks, _, _ = module.on_ui_tabs()[0]
            loadsave = loadsave_module.UiLoadsave(str(Path(folder) / "ui-config.json"))
            loadsave.ui_settings = {
                "iris_studio/チェックポイント/value": "base",
                "iris_studio/保存形式/value": "w4a8",
                "iris_studio/探索フォルダー/value": "",
            }
            by_id = {
                component.elem_id: component for component in blocks.blocks.values() if hasattr(component, "elem_id")
            }
            formats = [
                component for component in blocks.blocks.values() if getattr(component, "label", None) == "保存形式"
            ]
            loadsave.add_block(blocks, "iris_studio")
            self.assertEqual(by_id["iris-checkpoint"].value, selected)
            self.assertNotIn("iris-search-directory", by_id)
            self.assertEqual(formats[0].value, "normal")
            self.assertFalse(formats[0].interactive)
            for component in (by_id["iris-checkpoint"], formats[0]):
                self.assertNotIn(component, loadsave.component_mapping.values())

    def test_refresh_during_job_keeps_run_button_disabled(self):
        module = screen_module()
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(local_assets, "LIBRARY", Path(folder) / "library.json"),
        ):
            result = module.switch({"job": "running"}, "generate", "int8", "base")
        self.assertFalse(result[7]["interactive"])

    def test_worker_rejects_model_changed_during_load(self):
        spec = importlib.util.spec_from_file_location("iris_mutating_worker", ROOT / "tools/iris_worker.py")
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        events = []
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            weights = artifact(root / "chosen")
            job = root / "job"
            core.atomic_json(
                job / "request.json", {"task": "generate", "precision": "normal", "checkpoint": str(weights.parent)}
            )

            class MutatingRunner:
                def __init__(self, root, precision, task, checkpoint="base"):
                    self.checkpoint = checkpoint
                    events.append(checkpoint)
                    artifact(weights.parent, value=2)

                def close(self):
                    events.append("close")

                def set_loras(self, loras):
                    self.loras = loras

                def run(self, directory, request=None):
                    raise AssertionError("changed model must not run")

            with patch.object(runtime, "Runner", MutatingRunner):
                with self.assertRaisesRegex(ValueError, "読み込み中"):
                    worker.resident_run({"root": str(root), "job_dir": str(job)})
        self.assertEqual(events, [str(weights), "close"])

    def test_short_relative_folder_means_the_actual_folder(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(local_assets, "ROOT", Path(folder)):
            selected = artifact(Path(folder) / "myft")
            self.assertEqual(core.model_files(Path(folder) / "runtime", "normal", "generate", "myft")[0], selected)

    def test_custom_format_does_not_erase_base_or_downstream_precision(self):
        module = screen_module()
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(local_assets, "LIBRARY", Path(folder) / "library.json"),
        ):
            selected = str(artifact(Path(folder) / "ft"))
            state = module.switch({}, "generate", "int8", "base")[-1]
            state = module.switch(state, "generate", "int8", selected)[-1]
            result = module.switch(state, "generate", "normal", "base")
            self.assertEqual(result[12]["value"], "int8")
            state = module.switch(result[-1], "depth", "int8", selected)[-1]
            state = module.switch(state, "depth", "w4a8", selected)[-1]
            state = module.switch(state, "generate", "w4a8", selected)[-1]
            result = module.switch(state, "depth", "normal", selected)
            self.assertEqual(result[12]["value"], "w4a8")
            self.assertTrue(result[12]["interactive"])

    def test_valid_ft_prepares_shared_components_without_base_weights(self):
        module = screen_module()
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(local_assets, "LIBRARY", Path(folder) / "library.json"),
        ):
            selected = str(artifact(Path(folder) / "ft"))
            with (
                patch.object(module, "ready", return_value=False),
                patch.object(module.PREPARATION, "start", return_value="prepare") as prepare,
            ):
                response = module.start({}, "generate", "int8", "a fox", None, 42, "1024×1024", 2, 3, "", selected)
            prepare.assert_called_once_with("generate", "normal", components_only=True)
            self.assertEqual(response[0]["checkpoint"], selected)

    def test_direct_file_and_single_file_folder_resolve_exact_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            selected = artifact(root / "日本語 FT")
            self.assertEqual(
                core.model_files(root, "normal", "generate", str(selected)),
                (selected, selected.with_name("config.yaml")),
            )
            self.assertEqual(core.model_files(root, "normal", "generate", str(selected.parent))[0], selected)
            with self.assertRaisesRegex(ValueError, "形式"):
                core.model_files(root, "int8", "generate", str(selected))

    def test_discovery_is_independent_of_environment_and_keeps_deleted_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = artifact(root / "checkpoints/deep")
            ignored = artifact(root / "outside")
            values = [value for _, value in core.checkpoint_choices(root)]
            self.assertIn(str(path), values)
            self.assertNotIn(str(ignored), values)
            path.unlink()
            self.assertIn(str(path), [value for _, value in core.checkpoint_choices(root, current=str(path))])
            self.assertFalse(core.model_ready(root, "normal", "generate", str(path)))

    def test_worker_reloads_overwritten_weights_and_config(self):
        spec = importlib.util.spec_from_file_location("iris_selection_worker", ROOT / "tools/iris_worker.py")
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        events = []

        class FakeRunner:
            def __init__(self, root, precision, task, checkpoint="base"):
                events.append("load")

            def close(self):
                events.append("close")

            def set_loras(self, loras):
                self.loras = loras

            def run(self, directory, request=None):
                return "result"

        with tempfile.TemporaryDirectory() as folder, patch.object(runtime, "Runner", FakeRunner):
            root = Path(folder)
            weights = artifact(root / "chosen")
            job = root / "job"
            core.atomic_json(
                job / "request.json", {"task": "generate", "precision": "normal", "checkpoint": str(weights)}
            )
            payload = {"root": str(root), "job_dir": str(job)}
            worker.resident_run(payload)
            worker.resident_run(payload)
            stamp = weights.stat().st_mtime_ns
            artifact(weights.parent, value=2)
            os.utime(weights, ns=(stamp, stamp))
            worker.resident_run(payload)
            config = weights.with_name("config.yaml")
            config.write_text(config.read_text(encoding="utf-8") + "flow:\n  shift: 3\n", encoding="utf-8")
            worker.resident_run(payload)
        self.assertEqual(events, ["load", "close", "load", "close", "load"])

    def test_ui_selection_persists_without_external_search_or_fallback(self):
        module = screen_module()
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(local_assets, "LIBRARY", Path(folder) / "library.json"),
        ):
            selected = artifact(Path(folder) / "探索/子")
            local_assets.save_selection("iris", {"directory": "old external root"})
            update = module.refresh_checkpoints(str(selected))
            self.assertEqual(update["value"], str(selected))
            module.switch({}, "generate", "int8", str(selected))
            self.assertEqual(local_assets.selection("iris")["checkpoint"], str(selected))
            self.assertNotIn("directory", local_assets.selection("iris"))
            selected.unlink()
            update = module.refresh_checkpoints(str(selected))
            self.assertEqual(update["value"], str(selected))
            with patch.object(module.PREPARATION, "start") as download:
                response = module.start(
                    {}, "generate", "normal", "a fox", None, 42, "1024×1024", 2, 3, "", str(selected)
                )
            download.assert_not_called()
            self.assertIn("チェックポイント", response[4])

    def test_real_ui_has_checkpoint_and_format_without_search_control(self):
        module = screen_module()
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(local_assets, "LIBRARY", Path(folder) / "library.json"),
        ):
            blocks, _, _ = module.on_ui_tabs()[0]
        components = blocks.get_config_file()["components"]
        props = {item.get("props", {}).get("elem_id"): item.get("props", {}) for item in components}
        self.assertNotIn("iris-search-directory", props)
        self.assertTrue(props["iris-checkpoint"]["allow_custom_value"])
        self.assertTrue(any(item.get("props", {}).get("label") == "保存形式" for item in components))


if __name__ == "__main__":
    unittest.main()
