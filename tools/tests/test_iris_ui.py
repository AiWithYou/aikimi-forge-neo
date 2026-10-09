"""Build the real Gradio surface without loading a model or downloading."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


class IrisUITests(unittest.TestCase):
    def test_real_screen_is_a_native_tab_and_has_three_tasks(self):
        path = ROOT / "extensions-builtin/iris-studio/scripts/iris_studio.py"
        spec = importlib.util.spec_from_file_location("test_iris_screen", path)
        module = importlib.util.module_from_spec(spec)
        callbacks = types.ModuleType("modules.script_callbacks")
        callbacks.on_ui_tabs = lambda callback: None
        with patch.dict(sys.modules, {"modules.script_callbacks": callbacks}):
            spec.loader.exec_module(module)
        blocks, label, identifier = module.on_ui_tabs()[0]
        self.assertEqual((label, identifier), ("Iris", "iris_studio"))
        components = blocks.get_config_file()["components"]
        task = next(x for x in components if x.get("props", {}).get("elem_id") == "iris-task")
        self.assertEqual(len(task["props"]["choices"]), 3)
        precision = next(x for x in components if x.get("props", {}).get("label") == "モデル")
        self.assertEqual(precision["props"]["choices"], [("INT8", "int8"), ("W4A8", "w4a8"), ("通常版", "normal")])
        self.assertEqual(precision["props"]["value"], "int8")
        preparation = "\n".join(x.get("props", {}).get("value", "") for x in components if x["type"] == "markdown")
        self.assertIn("https://huggingface.co/Aikimi/iris-3b-w4a8", preparation)
        labels = [x.get("props", {}).get("label", "") for x in components]
        image_input = next(x for x in components if x.get("props", {}).get("label") == "入力画像")
        self.assertEqual(image_input["props"]["type"], "pil", "Callbacks must receive decoded pixels, not client paths")
        sizes = [x for x in components if x.get("props", {}).get("label") == "生成サイズ"]
        self.assertEqual(len(sizes), 1, "One trained-size selector is required")
        self.assertEqual(sizes[0]["type"], "dropdown")
        self.assertEqual(len(sizes[0]["props"]["choices"]), 7)
        self.assertEqual(sizes[0]["props"]["value"], "1024×1024")
        self.assertNotIn("幅", labels)
        self.assertNotIn("高さ", labels)
        self.assertIn("ステップ数", labels)
        self.assertIn("CFG", labels)
        self.assertTrue(any(x["type"] == "imageslider" for x in components))
        self.assertTrue(any(x["type"] == "file" for x in components))
        with (
            patch.object(module, "ready", return_value=True),
            patch.object(module.SERVICE, "start", return_value="job") as start,
        ):
            module.start({}, "generate", "int8", "a fox", None, 1, "1344×768", 100, 3, "")
        request = start.call_args.args[0]
        self.assertEqual((request["width"], request["height"]), (1344, 768))
        with (
            patch.object(module, "ready", return_value=True),
            patch.object(module.SERVICE, "start", return_value="w4-job") as start,
        ):
            module.start({}, "generate", "w4a8", "a fox", None, 1, "1024×1024", 100, 3, "")
        self.assertEqual(start.call_args.args[0]["precision"], "w4a8")

    def test_navigation_entry_is_visible_for_the_native_panel(self):
        source = (ROOT / "extensions-builtin/aikimi-ui/javascript/aikimi_tabs.js").read_text(encoding="utf-8")
        self.assertTrue('buttonId: "aikimi-tab-iris"' in source, "Iris navigation is missing")
        self.assertTrue('containerId: "tab_iris_studio"' in source, "Iris native target is missing")


if __name__ == "__main__":
    unittest.main()
