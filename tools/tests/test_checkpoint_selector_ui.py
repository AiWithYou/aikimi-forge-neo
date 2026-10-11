"""Checkpoint selectors share layout and state while keeping family loaders."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gradio as gr

from modules_forge import local_assets
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
        directory = next(
            item for item in config["components"] if item["props"].get("elem_id") == "qwen21-search-directory"
        )
        self.assertTrue(
            any(
                item["type"] == "accordion" and item["id"] in ancestors(config, directory["id"])
                for item in config["components"]
            )
        )

    def test_ming_checkpoint_is_visible_with_refresh_and_details(self):
        ui = ming_ui()
        with patch.object(ui.studio, "runtime_root", return_value=self.root / "runtime"):
            tab = ui.on_ui_tabs()[0][0]
        self.addCleanup(tab.close)
        self.assert_selector_visible(tab, "ming-local-model")
        config = tab.get_config_file()
        directory = next(
            item for item in config["components"] if item["props"].get("elem_id") == "ming-search-directory"
        )
        self.assertTrue(
            any(
                item["type"] == "accordion" and item["id"] in ancestors(config, directory["id"])
                for item in config["components"]
            )
        )

    def test_qwen_catalog_accepts_bundles_and_excludes_their_parts_and_lora(self):
        ui = qwen_ui()
        bundle = qwen_fixture(self.root / "explore/日本語 bundle")
        weights = qwen_transformer(self.root / "explore/standalone") / "diffusion_pytorch_model.safetensors"
        lora = tensor_file(self.root / "explore/lora.safetensors")
        missing = self.root / "outside/missing.safetensors"
        with patch.object(local_assets, "file_identity", side_effect=AssertionError("discovery must not hash")):
            choices = ui.model_choices(str(self.root / "explore"), str(missing))
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
            self.root / "explore/新しい FT/model.safetensors",
            {"layers.0.weight": [1]},
            metadata={"config": '{"transformer":{"image_model":"ming_image"}}'},
        )
        foreign = tensor_file(self.root / "explore/zimage.safetensors", {"layers.0.weight": [1], "cap_pad_token": [1]})
        lora = tensor_file(self.root / "explore/lora.safetensors")
        with patch.object(local_assets, "file_identity", side_effect=AssertionError("discovery must not hash")):
            choices = ui.model_choices(str(self.root / "explore"), str(selected))
        self.assertIn(str(selected.resolve()), [value for _, value in choices])
        self.assertNotIn(str(foreign), [value for _, value in choices])
        self.assertNotIn(str(lora), [value for _, value in choices])
        self.assertNotIn(str(self.root), " ".join(label for label, _ in choices))

    def test_qwen_input_selection_restores_latest_library_without_writing(self):
        ui = qwen_ui()
        selected = qwen_transformer(self.root / "FT") / "diffusion_pytorch_model.safetensors"
        ui.save_model_selection(str(selected), "", "bf16", str(self.root), [], [])
        self.assertEqual(local_assets.selection("qwen21")["local_model"], str(selected.resolve()))
        self.assertEqual(local_assets.selection("qwen21")["directory"], str(self.root))
        before = local_assets.LIBRARY.read_bytes()
        with patch.object(local_assets, "save_selection", side_effect=AssertionError("load must be read-only")):
            view = ui.load_model_selection()
        self.assertEqual(view[0]["value"], str(selected.resolve()))
        self.assertEqual(view[2]["value"], "bf16")
        self.assertEqual(local_assets.LIBRARY.read_bytes(), before)

    def test_ming_input_selection_restores_latest_library_without_writing(self):
        ui = ming_ui()
        selected = tensor_file(self.root / "FT/model.safetensors", {"layers.0.weight": [1]})
        ui.save_model_selection("int8", str(selected), "", "", str(self.root), [], [])
        self.assertEqual(local_assets.selection("ming")["model_path"], str(selected.resolve()))
        before = local_assets.LIBRARY.read_bytes()
        with patch.object(local_assets, "save_selection", side_effect=AssertionError("load must be read-only")):
            view = ui.load_model_selection()
        self.assertEqual(view[0]["value"], str(selected.resolve()))
        self.assertEqual(view[3]["value"], "int8")
        self.assertEqual(local_assets.LIBRARY.read_bytes(), before)

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
        self.assertIn(details["id"], ancestors(config, by_name["forge-search-directory"]["id"]))
        self.assertIn(details["id"], ancestors(config, by_name["setting_sd_modules"]["id"]))
        refresh = by_name["forge_refresh_checkpoint"]
        self.assertIn(selector["id"], ancestors(config, refresh["id"]))
        self.assertEqual(refresh["props"]["scale"], 0)
        self.assertEqual(refresh["props"]["min_width"], 105)


if __name__ == "__main__":
    unittest.main()
