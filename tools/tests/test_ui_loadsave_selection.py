"""Saved textbox values must not corrupt hidden generation dropdown inputs."""

import ast
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import gradio as gr

from modules import gradio_compat


def load_add_component():
    path = Path(__file__).resolve().parents[2] / "modules/ui_loadsave.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    choices = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "radio_choices")
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "UiLoadsave")
    method = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "add_component")
    namespace = {
        "gr": gr,
        "math": math,
        "InputAccordionImpl": type("InputAccordionImpl", (), {}),
        "ToolButton": type("ToolButton", (), {}),
    }
    exec(compile(ast.Module(body=[choices, method], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102 - isolated repository code, no user input
    return namespace["add_component"]


class SavedSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.add_component = staticmethod(load_add_component())

    def loader(self, saved):
        return SimpleNamespace(
            finalized_ui=False,
            ui_settings={"xyz/X values/value": saved},
            component_mapping={},
        )

    def test_xyz_textbox_and_hidden_dropdown_can_restore_the_same_legacy_key(self):
        loader = self.loader("")
        text = gr.Textbox(label="X values")
        dropdown = gr.Dropdown(label="X values", choices=[], multiselect=True, visible=False)
        self.add_component(loader, "xyz/X values", text)
        self.add_component(loader, "xyz/X values", dropdown)
        self.assertEqual(text.value, "")
        self.assertEqual(dropdown.preprocess(dropdown.value), [])
        self.assertEqual(loader.ui_settings["xyz/X values/value"], "")

    def test_invalid_saved_shapes_cannot_enter_generation_inputs(self):
        for saved in ("", "a", 42, {"a": True}, ["missing"]):
            with self.subTest(saved=saved):
                dropdown = gr.Dropdown(choices=["a"], multiselect=True, value=[])
                self.add_component(self.loader(saved), "xyz/X values", dropdown)
                self.assertEqual(dropdown.preprocess(dropdown.value), [])

    def test_valid_saved_selections_still_restore(self):
        for saved in ([], ["a"], ["a", "b"]):
            with self.subTest(saved=saved):
                dropdown = gr.Dropdown(choices=["a", "b"], multiselect=True, value=[])
                self.add_component(self.loader(saved), "xyz/X values", dropdown)
                self.assertEqual(dropdown.preprocess(dropdown.value), saved)

    def test_hidden_controlnet_defaults_survive_legacy_slider_bounds(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "extensions-builtin/sd_forge_controlnet/lib_controlnet/controlnet_ui/controlnet_ui_group.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {"processor_res", "threshold_a", "threshold_b"}
        columns = [node for node in ast.walk(tree) if isinstance(node, ast.With)]
        block = next(
            node
            for node in columns
            if {
                target.attr
                for statement in node.body
                if isinstance(statement, ast.Assign)
                for target in statement.targets
                if isinstance(target, ast.Attribute)
            }
            == names
        )
        group = SimpleNamespace(default_unit=SimpleNamespace(**dict.fromkeys(names, -1)))
        namespace = {
            "self": group,
            "gr": gr,
            "gradio_compat": gradio_compat,
            "elem_id_tabname": "txt2img",
            "tabname": "ControlNet-0",
        }
        exec(compile(ast.Module(body=block.body, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102 - isolated repository code, no user input
        for name in names:
            slider = getattr(group, name)
            loader = SimpleNamespace(
                finalized_ui=False,
                component_mapping={},
                ui_settings={f"txt2img/{slider.label}/value": -1, f"txt2img/{slider.label}/minimum": 64},
            )
            self.add_component(loader, f"txt2img/{slider.label}", slider)
            self.assertEqual(slider.preprocess(slider.value), -1)


if __name__ == "__main__":
    unittest.main()
