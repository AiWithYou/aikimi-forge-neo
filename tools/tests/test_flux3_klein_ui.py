"""Build and operate the real FLUX 3 Gradio components without inference."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import gradio as gr
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


class UITests(unittest.TestCase):
    def test_feature_navigation_and_common_mode_switch_include_flux3(self):
        folder = ROOT / "extensions-builtin/aikimi-ui"
        self.assertTrue(
            'buttonId: "aikimi-tab-flux3"' in (folder / "javascript/aikimi_tabs.js").read_text(encoding="utf-8")
        )
        self.assertTrue('"flux3"' in (folder / "javascript/aikimi_image_workspace.js").read_text(encoding="utf-8"))
        self.assertTrue('[data-active-feature="flux3"]' in (folder / "style.css").read_text(encoding="utf-8"))

    def load_screen(self):
        path = ROOT / "extensions-builtin/flux3-klein-studio/scripts/flux3_klein_studio.py"
        spec = importlib.util.spec_from_file_location("test_flux3_screen", path)
        module = importlib.util.module_from_spec(spec)
        callbacks = types.ModuleType("modules.script_callbacks")
        callbacks.on_ui_tabs = lambda callback: None
        callbacks.on_app_started = lambda callback: None
        with patch.dict(sys.modules, {"modules.script_callbacks": callbacks}):
            spec.loader.exec_module(module)
        return module

    def test_real_surface_has_ordered_references_and_both_models(self):
        module = self.load_screen()
        blocks, label, identifier = module.on_ui_tabs()[0]
        self.assertEqual((label, identifier), ("FLUX 3", "flux3_klein_studio"))
        components = blocks.get_config_file()["components"]
        props = [item.get("props", {}) for item in components]
        ids = [item["elem_id"] for item in props if item.get("elem_id")]
        self.assertEqual(len(ids), len(set(ids)), "Native controls and workspace wrappers need distinct DOM ids")
        model = next(item for item in props if item.get("elem_id") == "flux3-model-choice")
        self.assertEqual(model["value"], "distilled")
        self.assertEqual({value for _, value in model["choices"]}, {"base", "distilled"})
        gallery = next(item for item in props if item.get("elem_id") == "flux3-reference-images")
        input_gallery = next(
            item
            for item in blocks.blocks.values()
            if isinstance(item, gr.Gallery) and item.elem_id == "flux3-reference-images"
        )
        self.assertEqual(input_gallery.type, "pil")
        self.assertTrue(gallery["interactive"])
        self.assertTrue(any(item.get("label") == "プロンプト補完（PUP）" for item in props))
        self.assertFalse(next(item for item in props if item.get("label") == "プロンプト補完（PUP）")["value"])
        self.assertTrue(any(item.get("label") == "配置ボックス（JSON）" for item in props))

    def test_start_uses_only_current_mode_references(self):
        module = self.load_screen()
        with patch.object(module.SERVICE, "start", return_value="job") as start:
            module.start(
                {},
                "txt2img",
                "distilled",
                "int8",
                "a fox",
                [(Image.new("RGB", (16, 16)), "old")],
                "[]",
                "1024×1024",
                16,
                1,
                7,
                False,
            )
        self.assertEqual(start.call_args.args[0]["images"], [])

    def test_lora_selection_strength_and_refresh_are_connected_to_generation(self):
        module = self.load_screen()
        blocks = module.on_ui_tabs()[0][0]
        self.addCleanup(blocks.close)
        props = [item.get("props", {}) for item in blocks.get_config_file()["components"]]
        self.assertTrue(any(item.get("elem_id") == "flux3-lora-choice" for item in props))
        self.assertTrue(any(item.get("label") == "LoRA強度" for item in props))
        lora = next(item for item in blocks.blocks.values() if item.elem_id == "flux3-lora-choice")
        strength = next(item for item in blocks.blocks.values() if getattr(item, "label", None) == "LoRA強度")
        generation = next(item for item in blocks.fns.values() if item.fn is module.start)
        self.assertEqual(generation.inputs[-2:], [lora, strength])
        selected = next(item for item in blocks.fns.values() if item.fn is module.select_lora)
        self.assertEqual(selected.inputs, [lora])
        self.assertEqual(selected.outputs[1], strength)
        defaults = next(item for item in blocks.fns.values() if item.fn is module.model_defaults)
        self.assertEqual(defaults.inputs, [selected.outputs[0]])
        with patch.object(module.SERVICE, "start", return_value="job") as start:
            generation.fn(
                {},
                "txt2img",
                "base",
                "int8",
                "cat",
                [],
                "[]",
                "1024×1024",
                50,
                3.5,
                7,
                False,
                "AikimiLoRA.safetensors",
                0.75,
            )
        self.assertEqual(start.call_args.args[0]["lora"], "AikimiLoRA.safetensors")
        self.assertEqual(start.call_args.args[0]["lora_strength"], 0.75)
        with patch.object(module.local_lora, "required_model", return_value="base"):
            model, strength = module.select_lora("AikimiLoRA.safetensors")
        self.assertEqual(model["value"], "base")
        self.assertTrue(strength["interactive"])
        with patch.object(
            module, "lora_choices", return_value=[("なし", ""), ("AikimiLoRA", "AikimiLoRA.safetensors")]
        ):
            self.assertEqual(module.refresh_loras("AikimiLoRA.safetensors")["value"], "AikimiLoRA.safetensors")
            self.assertEqual(module.refresh_loras("removed.safetensors")["value"], "")

    def test_lora_starts_unselected_even_with_saved_ui_defaults(self):
        from tools.tests.test_ui_loadsave_selection import load_add_component

        module = self.load_screen()
        with patch.object(module, "lora_choices", return_value=[("なし", ""), ("Base", "base.safetensors")]):
            blocks = module.on_ui_tabs()[0][0]
        self.addCleanup(blocks.close)
        choice = next(item for item in blocks.blocks.values() if item.elem_id == "flux3-lora-choice")
        loader = types.SimpleNamespace(
            finalized_ui=False,
            ui_settings={"flux3/LoRA/value": "base.safetensors", "flux3/LoRA/visible": False},
            component_mapping={},
        )
        load_add_component()(loader, "flux3/LoRA", choice)
        self.assertEqual(choice.value, "", "Startup must not restore a LoRA without running its model callback")
        self.assertTrue(choice.visible)
        self.assertNotIn("flux3/LoRA/value", loader.component_mapping)

    def test_history_panel_has_persistent_selection_files_and_guide(self):
        module = self.load_screen()
        blocks = module.on_ui_tabs()[0][0]
        props = [item.get("props", {}) for item in blocks.get_config_file()["components"]]
        self.assertTrue(any(item.get("elem_id") == "flux3-history-choice" for item in props))
        self.assertTrue(any(item.get("value") == "履歴" for item in props))
        self.assertTrue(any(item.get("value") == "更新" for item in props))
        self.assertTrue(any(item.get("label") == "履歴のPNG・生成条件" for item in props))
        self.assertTrue(any("docs/flux3-klein.md" in str(item.get("value", "")) for item in props))
        choice = next(item for item in blocks.blocks.values() if item.elem_id == "flux3-history-choice")
        self.assertTrue(getattr(choice, "do_not_save_to_config", False))

    def test_history_refresh_and_selection_use_validated_persistent_results(self):
        module = self.load_screen()
        with patch.object(module.HISTORY, "list", return_value=[("saved image", "a" * 32)]):
            self.assertEqual(module.refresh_history("a" * 32)["value"], "a" * 32)
            self.assertIsNone(module.refresh_history("missing")["value"])
        record = {"image_path": "output.png", "files": ["output.png", "result.json"], "prompt": "a vase"}
        with patch.object(module.HISTORY, "load", return_value=record) as load:
            self.assertEqual(module.select_history("a" * 32), ("output.png", record["files"], "a vase"))
        load.assert_called_once_with("a" * 32)
        self.assertEqual(module.select_history(None), (None, [], ""))


if __name__ == "__main__":
    unittest.main()
