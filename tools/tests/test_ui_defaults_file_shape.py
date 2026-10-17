from __future__ import annotations

import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class UiDefaultsFileShapeTests(unittest.TestCase):
    def loader_class(self):
        source = Path(__file__).resolve().parents[2] / "modules/ui_loadsave.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "UiLoadsave")
        owner.body = [node for node in owner.body if isinstance(node, ast.FunctionDef)
                      and node.name in {"__init__", "read_from_file", "write_to_file", "dump_defaults"}]
        self.report = Mock()
        scope = {"json": json, "os": os, "errors": SimpleNamespace(display=self.report)}
        exec(compile(ast.Module(body=[owner], type_ignores=[]), str(source), "exec"), scope)  # noqa: S102 -- isolated production persistence methods
        return scope["UiLoadsave"]

    def test_non_object_defaults_report_error_and_preserve_original_file(self):
        loader_class = self.loader_class()
        for value in ([], [1], None, 42, True, "text"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "ui-config.json"
                payload = json.dumps(value)
                path.write_text(payload, encoding="utf-8")
                self.report.reset_mock()
                loader = loader_class(path)
                self.assertEqual(loader.ui_settings, {})
                self.assertTrue(loader.error_loading)
                self.report.assert_called_once()
                self.assertIsInstance(self.report.call_args.args[0], ValueError)
                loader.dump_defaults()
                self.assertEqual(path.read_text(encoding="utf-8"), payload)

    def test_valid_object_and_new_file_remain_writable(self):
        loader_class = self.loader_class()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ui-config.json"
            loader = loader_class(path)
            self.assertFalse(loader.error_loading)
            loader.ui_settings = {"tab/Seed/value": 42}
            loader.dump_defaults()
            restored = loader_class(path)
            self.assertEqual(restored.ui_settings, loader.ui_settings)
            self.assertFalse(restored.error_loading)
            self.report.assert_not_called()


if __name__ == "__main__":
    unittest.main()
