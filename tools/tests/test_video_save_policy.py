"""Respect the explicit no-save policy set by generation API requests."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class VideoSavePolicyTests(unittest.TestCase):
    def test_video_output_honors_explicit_sample_save_policy(self):
        path = Path(__file__).resolve().parents[2] / "modules/processing.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        branch = next(
            n for n in ast.walk(tree) if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "_is_video"
            and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "save_video" for call in ast.walk(n))
        )
        for no_save in (True, False):
            with self.subTest(no_save=no_save):
                saver = Mock(return_value="output.mp4")
                scope = dict(_is_video=True, p=SimpleNamespace(do_not_save_samples=no_save), frames=["frame"],
                             images=SimpleNamespace(save_video=saver), infotext=lambda index: "current-iteration", video_path=None)
                exec(compile(ast.Module(body=[branch], type_ignores=[]), str(path), "exec"), scope)  # noqa: S102
                if no_save:
                    saver.assert_not_called()
                    self.assertIsNone(scope["video_path"])
                else:
                    saver.assert_called_once_with(scope["p"], ["frame"], info="current-iteration")
                    self.assertEqual(scope["video_path"], "output.mp4")
                self.assertNotIn("frames", scope)


if __name__ == "__main__":
    unittest.main()
