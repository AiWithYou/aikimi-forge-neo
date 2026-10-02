"""Exercise actual settings save/restore boundaries without starting Forge."""

import ast
import html
import json
import tempfile
import unittest
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[2]


def load_editor():
    path = ROOT / "modules/ui_loadsave.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    choices = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "radio_choices")
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "UiLoadsave")
    names = {"iter_changes", "read_from_file", "write_to_file", "ui_view", "ui_apply"}
    owner.body = [node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"html": html, "json": json}
    exec(compile(ast.Module(body=[choices, owner], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    return namespace["UiLoadsave"]


class UiDefaultsChangesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.editor_class = load_editor()

    def editor(self, filename, component, saved, *, key="settings/Choice/value"):
        editor = self.editor_class()
        editor.filename = filename
        editor.component_mapping = {key: component}
        Path(filename).write_text(json.dumps({key: saved}), encoding="utf-8")
        return editor

    def test_clear_multiselect_is_saved_and_stays_cleared_after_reload(self):
        with tempfile.TemporaryDirectory() as folder:
            component = gr.Dropdown(choices=["a", "b"], multiselect=True)
            editor = self.editor(Path(folder) / "ui.json", component, ["a"])
            self.assertEqual(editor.ui_apply(component.preprocess([])), "Wrote 1 changes.")
            self.assertEqual(editor.read_from_file()["settings/Choice/value"], [])
            self.assertEqual(editor.ui_apply([]), "No changes.")

    def test_numeric_values_are_saved_as_values(self):
        for component in (gr.Dropdown(choices=[10, 20]), gr.Radio(choices=[10, 20])):
            with self.subTest(component=type(component)), tempfile.TemporaryDirectory() as folder:
                editor = self.editor(Path(folder) / "ui.json", component, 10)
                self.assertEqual(editor.ui_apply(component.preprocess(20)), "Wrote 1 changes.")
                self.assertEqual(editor.read_from_file()["settings/Choice/value"], 20)

    def test_index_components_save_underlying_values_including_multiselect(self):
        components = (
            (gr.Dropdown(choices=[("Alpha", "a"), ("Beta", "b")], type="index"), "b"),
            (gr.Radio(choices=[("Alpha", "a"), ("Beta", "b")], type="index"), "b"),
            (gr.Dropdown(choices=[("Alpha", "a"), ("Beta", "b")], type="index", multiselect=True), ["b", "a"]),
        )
        for component, value in components:
            with self.subTest(component=type(component), value=value), tempfile.TemporaryDirectory() as folder:
                editor = self.editor(Path(folder) / "ui.json", component, None)
                self.assertEqual(editor.ui_apply(component.preprocess(value)), "Wrote 1 changes.")
                self.assertEqual(editor.read_from_file()["settings/Choice/value"], value)

    def test_invalid_indices_cannot_select_from_end_or_crash_review(self):
        component = gr.Dropdown(choices=["a", "b"], type="index")
        with tempfile.TemporaryDirectory() as folder:
            editor = self.editor(Path(folder) / "ui.json", component, "a")
            for value in (-1, 2, True):
                with self.subTest(value=value):
                    self.assertEqual(editor.ui_apply(value), "No changes.")
            self.assertEqual(editor.read_from_file()["settings/Choice/value"], "a")

    def test_review_displays_prompt_markup_as_text(self):
        with tempfile.TemporaryDirectory() as folder:
            old, new = "<b>old & text</b>", '<img src=x onerror="alert(1)">'
            key = "settings/<label>/value"
            editor = self.editor(Path(folder) / "ui.json", gr.Textbox(), old, key=key)
            review = editor.ui_view(new)
            for value in (key, old, new):
                self.assertIn(html.escape(value), review)
                self.assertNotIn(value, review)
            self.assertTrue(review.endswith("</tbody></table>"))


if __name__ == "__main__":
    unittest.main()
