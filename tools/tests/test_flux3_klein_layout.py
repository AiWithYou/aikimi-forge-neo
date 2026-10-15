"""The visual editor shares the real JSON component consumed by generation."""

import unittest

import gradio as gr

from tools.tests import test_flux3_klein_ui


class LayoutSurfaceTests(unittest.TestCase):
    def test_canvas_and_advanced_json_share_the_generation_input(self):
        screen = test_flux3_klein_ui.UITests().load_screen()
        blocks = screen.on_ui_tabs()[0][0]
        editor = next(
            item
            for item in blocks.blocks.values()
            if isinstance(item, gr.HTML) and item.elem_id == "flux3-layout-editor"
        )
        self.assertIn('viewBox="0 0 1000 1000"', editor.value)
        self.assertIn("flux3-layout-rows", editor.value)
        self.assertIn("描いた枠をクリア", editor.value)
        boxes = next(
            item
            for item in blocks.blocks.values()
            if isinstance(item, gr.Textbox) and item.elem_id == "flux3-layout-json"
        )
        self.assertEqual(boxes.label, "配置ボックス（JSON）")
        disclosure = next(
            item
            for item in blocks.blocks.values()
            if isinstance(item, gr.HTML) and item.elem_id == "flux3-layout-disclosure"
        )
        self.assertIn("<details>", disclosure.value)
        self.assertNotIn("<details open", disclosure.value)
        parents = []
        parent = boxes.parent
        while parent is not None:
            parents.append(parent)
            parent = getattr(parent, "parent", None)
        self.assertFalse(any(isinstance(item, gr.Accordion) for item in parents))
        self.assertTrue(any("flux3-layout-advanced" in (item.elem_classes or []) for item in parents))
        start = next(item for item in blocks.fns.values() if item.fn is screen.start)
        self.assertIn(boxes, start.inputs)


if __name__ == "__main__":
    unittest.main()
