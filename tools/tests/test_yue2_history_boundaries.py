from __future__ import annotations

import importlib.util
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from modules_forge.yue2_studio import core, service


class YuE2HistoryBoundaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.studio = service.Studio(self.root / "runtime", self.root / "outputs/yue2")
        self.addCleanup(self.studio.shutdown)
        self.key = f"{'a' * 32}/take-1"
        self.take = self.studio.outputs / self.key
        self.take.mkdir(parents=True)
        core.atomic_json(self.take / "studio-result.json", {"state": "complete", "title": "Generated", "seed": 42})

    def symlink(self, link, target):
        try:
            link.symlink_to(target)
        except OSError as exc:
            self.skipTest(f"File symlinks are unavailable: {exc}")

    def ui(self):
        importlib.import_module("gradio")
        repository = Path(__file__).resolve().parents[2]
        modules = ModuleType("modules")
        modules.script_callbacks = SimpleNamespace(on_ui_tabs=lambda callback: None)
        paths = ModuleType("modules.paths")
        paths.data_path, paths.script_path = str(self.root), str(repository)
        spec = importlib.util.spec_from_file_location(
            "test_yue2_history_ui", repository / "extensions-builtin/yue2-studio/scripts/yue2_studio.py"
        )
        ui = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {"modules": modules, "modules.paths": paths}):
            spec.loader.exec_module(ui)
        self.addCleanup(ui.STUDIO.shutdown)
        return ui

    def test_history_skips_a_broken_link_and_keeps_valid_takes(self):
        self.symlink(self.studio.outputs / ("b" * 32), self.root / "missing")

        self.assertEqual(self.studio.history(), [("Generated / take-1 / seed 42", self.key)])

    def test_history_does_not_read_metadata_outside_outputs(self):
        outside = self.root / "private.json"
        core.atomic_json(outside, {"state": "complete", "title": "Private", "seed": 9})
        metadata = self.take / "studio-result.json"
        metadata.unlink()
        self.symlink(metadata, outside)

        self.assertEqual(self.studio.history(), [])

    def test_loading_a_take_rejects_external_metadata(self):
        ui = self.ui()
        outside = self.root / "private.json"
        core.atomic_json(outside, {"state": "complete", "plan_only": True})
        metadata = self.take / "studio-result.json"
        metadata.unlink()
        self.symlink(metadata, outside)
        (self.take / "audio.wav").write_bytes(b"generated")

        with patch.object(ui, "read_json", wraps=core.read_json) as read:
            audio, _score, files, warning = ui.load_result(self.key)
        read.assert_not_called()

        self.assertIsNone(audio)
        self.assertEqual(files, [])
        self.assertIn("外側", warning)

    def test_restoring_a_take_rejects_an_external_project(self):
        ui = self.ui()
        outside = self.root / "private-project.json"
        request = core.Request(style="piano", lyrics="[instrumental]", seed=42)
        core.atomic_json(outside, {"schema": 1, "request": asdict(request)})
        self.symlink(self.take / "project.json", outside)

        with self.assertRaisesRegex(core.YuE2Error, "外側"):
            ui.restore(self.key)


if __name__ == "__main__":
    unittest.main()
