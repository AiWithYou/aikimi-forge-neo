"""Consistency preparation keeps user choices and exposes edit limitations."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.tests.test_qwen_image21_service import isolate_local_assets, load_ui


class ConsistencyUiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        isolate_local_assets(self, self.root)
        self.ui = load_ui()
        self.enterContext(patch.object(self.ui, "RUNTIME", self.root))

    def test_preparation_adds_default_once_and_keeps_existing_strengths(self):
        from modules_forge.qwen_image21 import consistency_lora

        name = "ausboss--Qwen-Image-2.1-Consistency-LoRA/qwen-image-2.1-consistency.safetensors"
        with (
            patch.object(
                consistency_lora, "install", return_value={"name": name, "path": str(self.root / "loras" / name)}
            ),
            patch.object(self.ui, "lora_choices", return_value=[("other", "other"), ("Consistency", name)]),
        ):
            selection, strengths, message = self.ui.prepare_consistency(["other"], [["other", 0.65]])
            self.assertEqual(selection["value"], ["other", name])
            self.assertEqual(strengths["value"], [["other", 0.65], [name, 1.0]])
            self.assertIn("1500", message)
            selection, strengths, _ = self.ui.prepare_consistency(["other", name], [["other", 0.65], [name, 0.0]])
            self.assertEqual(selection["value"], ["other", name])
            self.assertEqual(strengths["value"], [["other", 0.65], [name, 0.0]])

    def test_preparation_keeps_absolute_selection_of_the_same_file(self):
        from modules_forge.qwen_image21 import consistency_lora

        path = consistency_lora.adapter_path(self.root)
        path.parent.mkdir(parents=True)
        path.touch()
        name = path.relative_to(self.root / "loras").as_posix()
        absolute = str(path.resolve())
        with patch.object(consistency_lora, "install", return_value={"name": name, "path": absolute}):
            selection, strengths, _ = self.ui.prepare_consistency([absolute], [[absolute, 0.45]])
        self.assertEqual(selection["value"], [absolute])
        self.assertEqual(
            self.ui.lora_settings(selection["value"], strengths["value"]), ({"name": absolute, "strength": 0.45},)
        )

    def test_prepare_button_uses_existing_library_without_changing_sampling(self):
        demo = self.ui.on_ui_tabs()[0][0]
        self.addCleanup(demo.close)
        config = demo.get_config_file()
        components = {item["id"]: item["props"] for item in config["components"]}
        button = next(key for key, props in components.items() if props.get("elem_id") == "qwen21-consistency-prepare")
        event = next(
            item for item in config["dependencies"] if any(target == (button, "click") for target in item["targets"])
        )
        self.assertEqual(
            [components[key].get("elem_id") for key in event["inputs"]],
            ["qwen21-style-lora", "qwen21-lora-strengths"],
        )
        self.assertEqual(
            [components[key].get("elem_id") for key in event["outputs"][:2]],
            ["qwen21-style-lora", "qwen21-lora-strengths"],
        )
        self.assertEqual(len(event["outputs"]), 3)
        selection = event["outputs"][0]
        changed = next(item for item in config["dependencies"] if (selection, "change") in item["targets"])
        mismatch = next(key for key, props in components.items() if props.get("label") == "異なる量子化で試す（実験）")
        self.assertEqual(changed["inputs"], [selection, event["outputs"][1], mismatch])

    def test_known_adapter_selection_explains_geometry_and_pose_limit(self):
        info = {"base_mismatch": False, "consistency_version": "1500"}
        with patch.object(self.ui.lora_library, "inspect", return_value=info):
            _, message, mismatch = self.ui.lora_selection(["renamed.safetensors"], [["renamed.safetensors", 1]])
        for phrase in ("Consistency", "ポーズ", "同じサイズ", "ConvRot", "25", "Q4_K_M", "作者"):
            self.assertIn(phrase, message)
        self.assertFalse(mismatch["value"])


if __name__ == "__main__":
    unittest.main()
