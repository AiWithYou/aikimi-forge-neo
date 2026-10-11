"""Checkpoint selectors share layout and state while keeping family loaders."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gradio as gr

from modules_forge import local_assets, model_catalog
from tools.tests.test_additional_module_identity import load_main_entry
from tools.tests.test_iris_selection import artifact, screen_module
from tools.tests.test_local_model_sources import qwen_fixture, qwen_transformer, tensor_file
from tools.tests.test_ming_image_studio import load_ui as ming_ui
from tools.tests.test_qwen_image21_service import load_ui as qwen_ui


def ancestors(config, identifier):
    def visit(node, parents):
        if node["id"] == identifier:
            return parents
        for child in node.get("children", []):
            found = visit(child, [*parents, node["id"]])
            if found is not None:
                return found
        return None

    return visit(config["layout"], [])


class CheckpointSelectorUiTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.enterContext(patch.object(local_assets, "LIBRARY", self.root / "library.json"))
        self.enterContext(patch.object(local_assets, "HASH_CACHE", self.root / "hashes.json"))
        self.enterContext(patch.object(local_assets, "ROOT", self.root))
        self.enterContext(patch.object(model_catalog, "ROOT", self.root))

    def assert_selector_visible(self, tab, elem_id):
        config = tab.get_config_file()
        components = {item["id"]: item for item in config["components"]}
        selector = next(item for item in components.values() if item["props"].get("elem_id") == elem_id)
        self.assertEqual(selector["props"]["label"], "チェックポイント")
        parents = ancestors(config, selector["id"])
        self.assertFalse(any(components[key]["type"] == "accordion" for key in parents if key in components))
        row = next(
            components[key] for key in reversed(parents) if key in components and components[key]["type"] == "row"
        )
        refresh = next(
            item
            for item in components.values()
            if item["type"] == "button" and item["props"].get("value") == "一覧を更新"
        )
        self.assertIn(row["id"], ancestors(config, refresh["id"]))
        self.assertTrue(tab.blocks[selector["id"]].do_not_save_to_config)
        self.assertIn(
            (selector["id"], "input"), [target for item in config["dependencies"] for target in item["targets"]]
        )

    def test_qwen_checkpoint_is_visible_with_refresh_and_details(self):
        ui = qwen_ui()
        with patch.object(ui, "RUNTIME", self.root / "runtime"):
            tab = ui.on_ui_tabs()[0][0]
        self.addCleanup(tab.close)
        self.assert_selector_visible(tab, "qwen21-local-model")
        config = tab.get_config_file()
        self.assertFalse(any(item["props"].get("label") == "探索フォルダー" for item in config["components"]))

    def test_ming_checkpoint_is_visible_with_refresh_and_details(self):
        ui = ming_ui()
        with patch.object(ui.studio, "runtime_root", return_value=self.root / "runtime"):
            tab = ui.on_ui_tabs()[0][0]
        self.addCleanup(tab.close)
        self.assert_selector_visible(tab, "ming-local-model")
        config = tab.get_config_file()
        self.assertFalse(any(item["props"].get("label") == "探索フォルダー" for item in config["components"]))

    def test_qwen_catalog_accepts_bundles_and_excludes_their_parts_and_lora(self):
        ui = qwen_ui()
        managed = self.root / "models/Qwen-Image-2.1/checkpoints"
        bundle = qwen_fixture(managed / "日本語 bundle")
        weights = qwen_transformer(managed / "standalone") / "diffusion_pytorch_model.safetensors"
        lora = tensor_file(managed / "lora.safetensors")
        missing = self.root / "outside/missing.safetensors"
        with patch.object(local_assets, "file_identity", side_effect=AssertionError("discovery must not hash")):
            choices = ui.model_choices(current=str(missing))
        values = [value for _, value in choices]
        self.assertIn(str(bundle.resolve()), values)
        self.assertNotIn(str(bundle / "transformer"), values)
        self.assertNotIn(str(bundle / "text_encoder/model.safetensors"), values)
        self.assertNotIn(str(lora), values)
        self.assertIn(str(weights.parent.resolve()), values)
        self.assertIn(str(missing.resolve()), values)
        self.assertNotIn(str(self.root), " ".join(label for label, _ in choices))

    def test_ming_catalog_filters_family_without_shared_parts(self):
        ui = ming_ui()
        selected = tensor_file(
            self.root / "models/Ming-Image/checkpoints/新しい FT/model.safetensors",
            {"layers.0.weight": [1]},
            metadata={"config": '{"transformer":{"image_model":"ming_image"}}'},
        )
        foreign = tensor_file(
            self.root / "models/Ming-Image/checkpoints/zimage.safetensors",
            {"layers.0.weight": [1], "cap_pad_token": [1]},
        )
        lora = tensor_file(self.root / "models/Ming-Image/checkpoints/lora.safetensors")
        with patch.object(local_assets, "file_identity", side_effect=AssertionError("discovery must not hash")):
            choices = ui.model_choices(current=str(selected))
        self.assertIn(str(selected.resolve()), [value for _, value in choices])
        self.assertNotIn(str(foreign), [value for _, value in choices])
        self.assertNotIn(str(lora), [value for _, value in choices])
        self.assertNotIn(str(self.root), " ".join(label for label, _ in choices))

    def test_qwen_input_selection_restores_latest_library_without_writing(self):
        ui = qwen_ui()
        selected = qwen_transformer(self.root / "FT") / "diffusion_pytorch_model.safetensors"
        local_assets.save_selection("qwen21", {"directory": "old external root"})
        ui.save_model_selection(str(selected), "", "bf16", [], [])
        self.assertEqual(local_assets.selection("qwen21")["local_model"], str(selected.resolve()))
        self.assertNotIn("directory", local_assets.selection("qwen21"))
        before = local_assets.LIBRARY.read_bytes()
        with patch.object(local_assets, "save_selection", side_effect=AssertionError("load must be read-only")):
            view = ui.load_model_selection()
        self.assertEqual(view[0]["value"], str(selected.resolve()))
        self.assertEqual(view[2]["value"], "bf16")
        self.assertEqual(local_assets.LIBRARY.read_bytes(), before)

    def test_managed_discovery_ignores_remembered_models_but_keeps_current_path(self):
        for engine, ui, make in (
            ("qwen21", qwen_ui(), qwen_transformer),
            ("ming", ming_ui(), lambda path: tensor_file(path / "model.safetensors", {"layers.0.weight": [1]})),
        ):
            with self.subTest(engine=engine):
                current = make(self.root / engine / "selected")
                ignored = make(self.root / engine / "remembered")
                local_assets.remember(engine + "_model", [str(ignored)])
                choices = ui.model_choices(current=str(current))
                self.assertIn(str(current.resolve()), [value for _, value in choices])
                self.assertNotIn(str(ignored.resolve()), [value for _, value in choices])

    def test_ming_loras_use_family_folder_and_shared_runtime_without_legacy_root(self):
        ui = ming_ui()
        managed = tensor_file(self.root / "models/Ming-Image/loras/日本語/style.safetensors")
        runtime = self.root / "runtime"
        shared = tensor_file(runtime / "models/loras/style.safetensors")
        old = tensor_file(self.root / "models/Lora/Ming/old.safetensors")
        with patch.object(ui.studio, "runtime_root", return_value=runtime):
            tab = ui.on_ui_tabs()[0][0]
        self.addCleanup(tab.close)
        selector = next(item for item in tab.blocks.values() if getattr(item, "elem_id", None) == "ming-loras")
        values = [value for _, value in selector.choices]
        self.assertIn(str(managed), values)
        self.assertIn(str(shared), values)
        self.assertNotIn(str(old), values)

    def test_lora_discovery_ignores_remembered_external_but_keeps_current(self):
        from modules_forge.qwen_image21 import style_lora

        current = tensor_file(self.root / "selected/style.safetensors")
        ignored = tensor_file(self.root / "remembered/style.safetensors")
        local_assets.remember("qwen21_lora", [str(ignored)])
        local_assets.remember("ming_lora", [str(ignored)])
        self.assertNotIn(str(ignored), style_lora.inventory(self.root))
        qwen = qwen_ui()
        with patch.object(qwen, "RUNTIME", self.root):
            choices = qwen.refresh_loras([str(current)])["choices"]
        self.assertIn(str(current), [value for _, value in choices])
        ming = ming_ui()
        with patch.object(ming.studio, "runtime_root", return_value=self.root / "runtime"):
            choices = ming.lora_choices([str(current)])
        self.assertNotIn(str(ignored), [value for _, value in choices])
        self.assertIn(str(current), [value for _, value in choices])

    def test_qwen_lora_restore_keeps_empty_strength_table_mounted(self):
        ui = qwen_ui()
        for loras in ([], [{"name": "custom.safetensors", "strength": 0.5}]):
            with self.subTest(loras=loras):
                local_assets.save_selection("qwen21", {"style_loras": loras})
                table = ui.load_model_selection()[-1]
                self.assertEqual(table["visible"], True if loras else "hidden")
                self.assertEqual(table["value"], [[item["name"], item["strength"]] for item in loras])

    def test_ming_input_selection_restores_latest_library_without_writing(self):
        ui = ming_ui()
        selected = tensor_file(self.root / "FT/model.safetensors", {"layers.0.weight": [1]})
        local_assets.save_selection("ming", {"directory": "old external root"})
        ui.save_model_selection("int8", str(selected), "", "", [], [])
        self.assertEqual(local_assets.selection("ming")["model_path"], str(selected.resolve()))
        self.assertNotIn("directory", local_assets.selection("ming"))
        before = local_assets.LIBRARY.read_bytes()
        with patch.object(local_assets, "save_selection", side_effect=AssertionError("load must be read-only")):
            view = ui.load_model_selection()
        self.assertEqual(view[0]["value"], str(selected.resolve()))
        self.assertEqual(view[3]["value"], "int8")
        self.assertEqual(local_assets.LIBRARY.read_bytes(), before)

    def test_ming_lora_restore_and_selection_keep_empty_strength_table_mounted(self):
        ui = ming_ui()
        with patch.object(ui.studio, "runtime_root", return_value=self.root / "runtime"):
            tab = ui.on_ui_tabs()[0][0]
        self.addCleanup(tab.close)
        table = next(item for item in tab.blocks.values() if getattr(item, "elem_id", None) == "ming-lora-strengths")
        self.assertEqual(table.visible, "hidden")
        updates = [function.fn for function in tab.fns.values() if function.outputs == [table]]
        self.assertEqual(len(updates), 2)
        for loras in ([], [{"name": "custom.safetensors", "strength": 0.5}]):
            with self.subTest(loras=loras):
                local_assets.save_selection("ming", {"loras": loras})
                expected_rows = [[item["name"], item["strength"]] for item in loras]
                expected_visibility = True if loras else "hidden"
                restored = ui.load_model_selection()[-1]
                self.assertEqual((restored["value"], restored["visible"]), (expected_rows, expected_visibility))
                for update in updates:
                    selected = update([item["name"] for item in loras], expected_rows)
                    self.assertEqual((selected["value"], selected["visible"]), (expected_rows, expected_visibility))

    def test_ming_load_keeps_w4a8_when_only_that_official_model_is_installed(self):
        ui = ming_ui()
        with patch.object(
            ui.studio, "model_ready", side_effect=lambda root, verify_hash=False, precision="int8": precision == "w4a8"
        ):
            view = ui.load_model_selection()
        self.assertEqual(view[3]["value"], "w4a8")

    def test_iris_task_specific_official_selection_stays_visible(self):
        ui = screen_module()
        selected = str(artifact(self.root / "FT"))
        local_assets.save_selection(
            "iris", {"checkpoint": selected, "precision": "normal", "directory": str(self.root)}
        )
        view = ui.switch({}, "depth", "int8", selected, persist=False)
        self.assertTrue(view[11]["visible"])
        self.assertEqual(view[13]["value"], "base")
        self.assertFalse(view[13]["interactive"])
        self.assertFalse(view[13]["allow_custom_value"])
        self.assertEqual(view[13]["choices"], [("公式 · 深度推定", "base")])
        self.assertEqual(local_assets.selection("iris")["checkpoint"], selected)
        restored = ui.switch(view[14], "generate", "int8", "base", persist=False)
        self.assertEqual(restored[13]["value"], selected)

    def test_iris_terminal_job_keeps_dedicated_task_checkpoint_read_only(self):
        ui = screen_module()
        state = {"job": "test-job", "kind": "run", "task": "depth", "precision": "int8", "checkpoint": "base"}
        with (
            patch.object(ui.SERVICE, "poll", return_value={"status": "error", "error": "test error"}),
            patch.object(ui, "button", return_value={}),
        ):
            view = ui.poll(state)
        self.assertFalse(view[12]["interactive"])

    def test_forge_library_selection_uses_exact_path_and_restores_without_writing(self):
        ui, opts, _ = load_main_entry()
        left = self.root / "left/model.safetensors"
        right = self.root / "right/model.safetensors"
        tensor_file(left)
        tensor_file(right)
        records = {
            str(path): SimpleNamespace(
                filename=str(path), name=f"{path.parent.name}/model.safetensors", register=lambda: None
            )
            for path in (left, right)
        }
        opts.sd_model_checkpoint = str(left)
        ui.sd_models.get_closet_checkpoint_match = lambda value: next(
            (info for path, info in records.items() if value in (path, info.name)), None
        )
        ui.sd_models.CheckpointInfo = lambda value: records[value]
        ui.sd_models.checkpoint_choices = lambda current="": [
            (f"{path.parent.name}/model.safetensors", str(path)) for path in (left, right)
        ]
        with patch.object(ui, "refresh_model_loading_parameters"):
            ui.checkpoint_change(str(right), "krea", refresh=False)
        self.assertEqual(local_assets.selection("forge")["checkpoints"]["krea"], str(right))
        before = local_assets.LIBRARY.read_bytes()
        with patch.object(local_assets, "save_selection", side_effect=AssertionError("restore must be read-only")):
            view = ui.preset_checkpoint_selection("krea")
        self.assertEqual(view["value"], str(right))
        self.assertEqual(local_assets.LIBRARY.read_bytes(), before)

    def test_forge_can_select_again_after_empty_or_missing_checkpoint(self):
        selected = tensor_file(self.root / "selected/model.safetensors")
        for previous in ("", str(self.root / "missing.safetensors")):
            ui, opts, _ = load_main_entry()
            opts.sd_model_checkpoint = previous
            info = SimpleNamespace(filename=str(selected), name="selected/model.safetensors")
            ui.sd_models.get_closet_checkpoint_match = lambda value, info=info: info if value == str(selected) else None
            with patch.object(ui, "refresh_model_loading_parameters") as refresh:
                self.assertTrue(ui.checkpoint_change(str(selected), "sd"))
            self.assertEqual(opts.data["sd_model_checkpoint"], str(selected))
            self.assertEqual(local_assets.selection("forge")["checkpoints"]["sd"], str(selected))
            refresh.assert_called_once()

    def test_forge_checkpoint_ui_is_visible_and_excludes_config_defaults(self):
        ui, _, _ = load_main_entry()
        with patch.object(ui, "refresh_models", return_value=([], [])), gr.Blocks() as tab:
            ui.make_checkpoint_manager_ui()
        self.addCleanup(tab.close)
        self.assert_selector_visible(tab, "setting_sd_model_checkpoint")

    def test_forge_header_groups_have_distinct_layout_hooks(self):
        ui, _, _ = load_main_entry()
        with patch.object(ui, "refresh_models", return_value=([], [])), gr.Blocks() as tab:
            ui.make_checkpoint_manager_ui()
        self.addCleanup(tab.close)
        config = tab.get_config_file()
        components = {item["id"]: item for item in config["components"]}
        by_name = {item["props"].get("elem_id"): item for item in components.values()}
        selector = by_name["forge_checkpoint_selector"]
        details = by_name["forge_model_details"]
        self.assertEqual(selector["type"], "row")
        self.assertEqual(details["type"], "accordion")
        self.assertIn(selector["id"], ancestors(config, by_name["setting_sd_model_checkpoint"]["id"]))
        self.assertNotIn("forge-search-directory", by_name)
        self.assertIn(details["id"], ancestors(config, by_name["setting_sd_modules"]["id"]))
        refresh = by_name["forge_refresh_checkpoint"]
        self.assertIn(selector["id"], ancestors(config, refresh["id"]))
        self.assertEqual(refresh["props"]["scale"], 0)
        self.assertEqual(refresh["props"]["min_width"], 105)


if __name__ == "__main__":
    unittest.main()
