import ast
from pathlib import Path
import unittest

from modules.api.script_args import overlay_script_args


ROOT = Path(__file__).resolve().parents[2]


class ScriptArgumentOverlayTests(unittest.TestCase):
    def test_api_selectable_script_path_uses_layout_preserving_overlay(self):
        tree = ast.parse((ROOT / "modules" / "api" / "api.py").read_text("utf-8"))
        init_method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "init_script_args"
        )

        self.assertTrue(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "overlay_script_args"
                for node in ast.walk(init_method)
            )
        )

    def test_short_legacy_args_keep_new_defaults_and_following_slots(self):
        original = [f"default-{index}" for index in range(30)]
        script_args = original.copy()
        legacy = [f"legacy-{index}" for index in range(13)]

        overlay_script_args(script_args, 1, 19, legacy)

        self.assertEqual(len(script_args), len(original))
        self.assertEqual(script_args[1:14], legacy)
        self.assertEqual(script_args[14:19], original[14:19])
        self.assertEqual(script_args[19:], original[19:])

    def test_extra_values_are_bounded_to_selectable_script_range(self):
        original = [f"default-{index}" for index in range(12)]
        script_args = original.copy()

        overlay_script_args(script_args, 2, 5, list(range(20)))

        self.assertEqual(script_args[2:5], [0, 1, 2])
        self.assertEqual(script_args[5:], original[5:])
        self.assertEqual(len(script_args), len(original))

    def test_none_keeps_initialized_defaults(self):
        script_args = [None, "a", "b"]

        overlay_script_args(script_args, 1, 3, None)

        self.assertEqual(script_args, [None, "a", "b"])

    def test_invalid_layout_range_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            overlay_script_args([None], 1, 2, ["value"])


if __name__ == "__main__":
    unittest.main()
