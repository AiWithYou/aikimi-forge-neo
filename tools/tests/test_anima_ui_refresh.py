"""Anima UI callback regressions with real Gradio and no model imports."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import gradio as gr

from modules_forge import local_assets

ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def accordion_for_ui_tests(value, **kwargs):
    """Keep the wrapper open: only the panel's real controls/callbacks are tested."""
    checkbox = gr.Checkbox(value=value, visible=False)
    with gr.Accordion(label=kwargs["label"], elem_id=kwargs["elem_id"], open=True):
        yield checkbox


def load_ui_module():
    catalog = {"loras": ["Style A", "Style B"], "adapters": ["adapter-a.safetensors"]}
    files = SimpleNamespace(
        adapters=lambda: dict.fromkeys(catalog["adapters"], "fixture"),
        standard_anima_loras=lambda: dict.fromkeys(catalog["loras"], "fixture"),
    )
    modules = ModuleType("modules")
    modules.__path__ = []
    modules.scripts = SimpleNamespace(Script=object, AlwaysVisible=True)
    stubs = {
        "modules": modules,
        "modules.scripts": modules.scripts,
        "modules.processing": SimpleNamespace(StableDiffusionProcessing=object),
        "modules.ui_components": SimpleNamespace(InputAccordion=accordion_for_ui_tests),
        "anima3b": ModuleType("anima3b"),
        "anima3b.files": files,
        "anima3b.runtime": SimpleNamespace(Anima3BRuntime=Mock),
        "anima3b.standard_lora": SimpleNamespace(NO_STANDARD_LORA="None", apply_standard_lora_selections=Mock()),
    }
    path = ROOT / "extensions-builtin/anima-3-8b/scripts/anima_3_8b.py"
    spec = importlib.util.spec_from_file_location("_anima_ui_refresh", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module, catalog


class AnimaUIRefreshTests(unittest.TestCase):
    def setUp(self):
        self.module, self.catalog = load_ui_module()
        self.script = self.module.Anima3BScript()
        with gr.Blocks(analytics_enabled=False) as self.demo:
            self.controls = self.script.ui(False)
        self.addCleanup(self.demo.close)

    def callback(self, name):
        return next(fn for fn in self.demo.fns.values() if getattr(fn.fn, "__name__", "") == name)

    def refresh(self, names):
        callback = self.callback("refresh_standard_lora_slots")
        return callback.fn(*names) if callback.inputs else callback.fn()

    def test_refresh_preserves_selected_slots_without_writing_weights(self):
        updates = self.refresh(["Style A", "Style B", "None", "Style A"])
        self.assertEqual([update["value"] for update in updates[::2]], ["Style A", "Style B", "None", "Style A"])
        self.assertEqual([update["visible"] for update in updates[1::2]], [True, True, False, True])
        for update in updates[1::2]:
            self.assertNotIn("value", update)

    def test_refresh_only_clears_missing_selections_and_reports_them(self):
        self.catalog["loras"] = ["Style B", "New Style"]
        with patch.object(gr, "Warning") as warning:
            updates = self.refresh(["Style A", "Style B", None, ""])
        self.assertEqual([update["value"] for update in updates[::2]], ["None", "Style B", "None", "None"])
        self.assertIn("New Style", updates[0]["choices"])
        warning.assert_called_once()
        self.assertIn("Style A", warning.call_args.args[0])

    def test_programmatic_name_changes_only_update_visibility(self):
        for name_control, weight in zip(self.controls[6::2], self.controls[7::2], strict=True):
            callback = next(
                fn for fn in self.demo.fns.values() if fn.inputs == [name_control] and fn.outputs == [weight]
            )
            self.assertNotIn("value", callback.fn("Style B"))
            self.assertTrue(callback.fn("Style B")["visible"])
            self.assertFalse(callback.fn("None")["visible"])

    def test_refresh_reads_all_four_current_names_and_preserves_argument_order(self):
        callback = self.callback("refresh_standard_lora_slots")
        self.assertEqual(callback.inputs, self.controls[6::2])
        self.assertEqual(callback.outputs, self.controls[6:])
        self.assertEqual(len(self.controls), 14)

    def test_explicit_clear_resets_all_slots_and_weights(self):
        callback = self.callback("clear_standard_lora_slots")
        updates = callback.fn()
        self.assertEqual([update["value"] for update in updates[::2]], ["None"] * 4)
        self.assertEqual([update["value"] for update in updates[1::2]], [1.0] * 4)
        self.assertTrue(all(not update["visible"] for update in updates[1::2]))
        self.assertEqual(callback.outputs, self.controls[6:])

    def test_png_infotext_resets_all_names_and_strengths_without_saving(self):
        fields = getattr(self.script, "infotext_fields", [])
        self.assertEqual([component for component, _ in fields], self.controls[6:])
        params = {"Prompt": "portrait <lora:Style A:0.7>", "Anima 3.8B standard LoRA": "Style A:0.7"}
        with patch.object(local_assets, "save_selection", side_effect=AssertionError("PNG restore must not save")):
            updates = [callback(params) for _, callback in fields]
        self.assertEqual([update["value"] for update in updates[::2]], ["None"] * 4)
        self.assertEqual([update["value"] for update in updates[1::2]], [1.0] * 4)
        self.assertTrue(all(update["visible"] is False for update in updates[1::2]))
        self.assertEqual(params["Prompt"], "portrait <lora:Style A:0.7>")

    def test_png_prompt_tags_reproduce_without_previous_form_loras(self):
        spec = importlib.util.spec_from_file_location(
            "_anima_png_standard_lora", ROOT / "extensions-builtin/anima-3-8b/anima3b/standard_lora.py"
        )
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        prompt = "portrait <lora:Style A:0.7>"
        params = {"Prompt": prompt, "Anima 3.8B standard LoRA": "Style A:0.7"}
        values = ["Style A", 0.7, "None", 1.0, "None", 1.0, "None", 1.0]
        replacements = {
            component: callback(params) for component, callback in getattr(self.script, "infotext_fields", [])
        }
        values = [
            replacements[component]["value"] if component in replacements else value
            for component, value in zip(self.controls[6:], values, strict=True)
        ]
        processing = SimpleNamespace(prompt=prompt, all_prompts=[prompt], extra_generation_params={})
        with TemporaryDirectory() as directory:
            path = Path(directory) / "style.safetensors"
            path.touch()
            with (
                patch.object(self.module, "standard_anima_loras", return_value={"Style A": str(path)}),
                patch.object(self.module, "apply_standard_lora_selections", helper.apply_standard_lora_selections),
            ):
                self.script.process(processing, True, "adapter", 1.0, False, 1.0, False, *values)
        self.assertEqual(processing.prompt, prompt)
        self.assertEqual(processing.all_prompts, [prompt])
        self.assertEqual(processing.extra_generation_params, {})

    def test_gallery_send_to_filters_all_target_slots_and_keeps_ui_transfer(self):
        target = self.module.Anima3BScript()
        with gr.Blocks(analytics_enabled=False) as demo:
            target_controls = target.ui(True)
        self.addCleanup(demo.close)
        names = getattr(self.script, "paste_field_names", [])
        target_fields = [(component, key) for component, key in target.infotext_fields if key in names]
        self.assertEqual([component for component, _ in target_fields], target_controls[6:])
        source_fields = [(component, key) for component, key in self.script.infotext_fields if key in names]
        self.assertEqual([component for component, _ in source_fields], self.controls[6:])
        self.assertEqual(len(names), 2)
        self.assertEqual([key for _, key in source_fields], [key for _, key in target_fields])
        params = {"Prompt": "portrait <lora:Style A:0.7>"}
        updates = [key(params) for _, key in target_fields]
        self.assertEqual([update["value"] for update in updates[::2]], ["None"] * 4)
        self.assertEqual([update["value"] for update in updates[1::2]], [1.0] * 4)
        self.assertTrue(all(update["visible"] is False for update in updates[1::2]))

    def test_adapter_refresh_retains_selection_and_discovers_downloads(self):
        self.catalog["adapters"].append("adapter-b.safetensors")
        callback = self.callback("refresh_adapter_choices")
        update = callback.fn("adapter-a.safetensors")
        self.assertEqual(update["value"], "adapter-a.safetensors")
        self.assertIn("adapter-b.safetensors", update["choices"])
        self.assertEqual(callback.inputs, [self.controls[1]])
        self.assertEqual(callback.outputs, [self.controls[1]])

    def test_adapter_refresh_handles_removed_and_empty_catalogs(self):
        callback = self.callback("refresh_adapter_choices")
        self.assertEqual(callback.fn("removed.safetensors")["value"], "adapter-a.safetensors")
        self.catalog["adapters"] = []
        update = callback.fn("removed.safetensors")
        self.assertEqual(update["value"], "Anima-3.8B-expanded_adapter.safetensors")
        self.assertEqual(update["choices"], [update["value"]])

    def test_txt2img_and_img2img_refresh_buttons_have_distinct_ids(self):
        with gr.Blocks(analytics_enabled=False) as demo:
            script = self.module.Anima3BScript()
            script.ui(False)
            script.ui(True)
        self.addCleanup(demo.close)
        ids = [block.elem_id for block in demo.blocks.values() if isinstance(block, gr.Button) and block.elem_id]
        self.assertEqual(len(ids), 6)
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()
