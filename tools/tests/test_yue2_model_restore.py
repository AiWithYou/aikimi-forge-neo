"""YuE2 must not submit the wrong model while restoring or changing selection."""

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import gradio as gr

from modules_forge import local_assets, model_catalog
from modules_forge.yue2_studio.core import Request

ROOT = Path(__file__).resolve().parents[2]


class YuE2ModelRestoreTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.enterContext(patch.object(model_catalog, "ROOT", self.root))
        self.enterContext(patch.object(local_assets, "LIBRARY", self.root / "library.json"))
        package = types.ModuleType("modules")
        package.__path__ = [str(ROOT / "modules")]
        callbacks = types.ModuleType("modules.script_callbacks")
        callbacks.on_ui_tabs = Mock()
        package.script_callbacks = callbacks
        paths = types.ModuleType("modules.paths")
        paths.script_path = paths.data_path = str(self.root)
        spec = importlib.util.spec_from_file_location(
            "test_yue2_model_restore", ROOT / "extensions-builtin/yue2-studio/scripts/yue2_studio.py"
        )
        self.ui = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"modules": package, "modules.script_callbacks": callbacks, "modules.paths": paths}):
            spec.loader.exec_module(self.ui)
        self.addCleanup(self.ui.STUDIO.shutdown)
        self.request = types.SimpleNamespace(session_hash="session", username=None)
        self.values = self.ui.values_for(Request(style="piano", lyrics="[instrumental]"))

    def test_initial_ui_is_unselected_and_cannot_submit_before_restore(self):
        app = self.ui.on_ui_tabs()[0][0]
        self.addCleanup(app.close)
        checkpoint = next(item for item in app.blocks.values() if getattr(item, "label", None) == "チェックポイント")
        engine = next(item for item in app.blocks.values() if getattr(item, "label", None) == "エンジン")
        buttons = [item for item in app.blocks.values() if isinstance(item, gr.Button) and item.value in {"曲を生成", "楽譜だけ作る"}]
        self.assertIsNone(checkpoint.value)
        self.assertFalse(checkpoint.interactive)
        self.assertFalse(engine.interactive)
        self.assertEqual(len(buttons), 2)
        self.assertTrue(all(not item.interactive for item in buttons))
        restore = next(event for event in app.fns.values() if event.fn is self.ui.restore_checkpoint)
        submit = next(event for event in app.fns.values() if event.fn is self.ui.run)
        poll = next(event for event in app.fns.values() if event.fn is self.ui.poll)
        controls = next(event for event in app.fns.values() if event.fn is self.ui.generation_controls)
        self.assertIs(restore.outputs[0], checkpoint)
        self.assertIs(submit.inputs[-1], restore.outputs[1])
        self.assertIs(submit.outputs[-1], poll.outputs[-1])
        self.assertIs(controls.inputs[-1], submit.outputs[-1])
        engine_refresh = [event for event in app.fns.values()
                          if event.fn is not None and event.fn.__name__ == "refresh_checkpoint_engine"]
        self.assertTrue(any((engine._id, "input") in event.targets for event in engine_refresh))
        self.assertTrue(all((engine._id, "change") not in event.targets for event in engine_refresh))
        for name in ("restore", "restore_file"):
            first = next(index for index, event in app.fns.items() if event.fn and event.fn.__name__ == name)
            refresh_index = next(index for index, event in app.fns.items()
                                 if event.trigger_after == first and event in engine_refresh)
            verify = next(event for event in app.fns.values() if event.trigger_after == refresh_index)
            self.assertIs(verify.fn, self.ui.confirm_checkpoint)
            self.assertEqual(verify.inputs, [checkpoint, engine])
            self.assertIs(verify.outputs[0], restore.outputs[1])
        with patch.object(self.ui.STUDIO, "start", return_value="job") as start:
            for plan_only in (False, True):
                result = self.ui.start(self.values, plan_only, self.request)
                self.assertIn("モデル", result[1])
            start.assert_not_called()
            confirmed, _ = self.ui.confirm_checkpoint("", "official")
            self.ui.start(self.values, False, self.request, confirmed)
            self.assertEqual(start.call_args.args[0].checkpoint, "")

    def test_direct_submit_rejects_unconfirmed_standard_model(self):
        with patch.object(self.ui.STUDIO, "start", return_value="job") as start:
            for plan_only in (False, True):
                result = self.ui.start(self.values, plan_only, self.request)
                self.assertIn("モデル", result[1])
            start.assert_not_called()

    def test_confirmation_binds_both_model_and_engine_before_submission(self):
        custom = self.root / "models/YuE2/checkpoints/custom"
        custom.mkdir(parents=True)
        (custom / "config.json").write_text('{"model_type":"yue2"}', encoding="utf-8")
        (custom / "qwen.tiktoken").write_text("fixture", encoding="utf-8")
        (custom / "model.safetensors").write_text("fixture", encoding="utf-8")
        path = str(custom.resolve())
        confirmed, message = self.ui.confirm_checkpoint(path, "official")
        self.assertEqual(confirmed, {"checkpoint": path, "engine": "official"})
        self.assertFalse(message)
        values = list(self.values)
        values[self.ui.FIELDS.index("checkpoint")] = path
        with patch.object(self.ui.STUDIO, "start", return_value="job") as start:
            self.ui.start(values, False, self.request, confirmed)
            self.assertEqual(start.call_args.args[0].checkpoint, path)
            start.reset_mock()
            values[self.ui.FIELDS.index("checkpoint")] = ""
            self.ui.start(values, False, self.request, confirmed)
            values[self.ui.FIELDS.index("checkpoint")] = path
            values[self.ui.FIELDS.index("engine")] = "cpp"
            self.ui.start(values, False, self.request, confirmed)
            start.assert_not_called()

        missing = custom / "model.safetensors"
        missing.unlink()
        confirmed, message = self.ui.confirm_checkpoint(path, "official")
        self.assertIsNone(confirmed)
        self.assertTrue(message)

    def test_restore_keeps_custom_selection_and_does_not_fallback_when_missing(self):
        custom = self.root / "models/YuE2/checkpoints/custom"
        custom.mkdir(parents=True)
        (custom / "config.json").write_text('{"model_type":"yue2"}', encoding="utf-8")
        (custom / "qwen.tiktoken").write_text("fixture", encoding="utf-8")
        (custom / "model.safetensors").write_text("fixture", encoding="utf-8")
        path = str(custom.resolve())
        local_assets.save_selection("yue2", {"checkpoint": path})
        before = local_assets.LIBRARY.read_bytes()
        restored = self.ui.restore_checkpoint("official")
        self.assertEqual(restored[0]["value"], path)
        self.assertEqual(restored[1], {"checkpoint": path, "engine": "official"})
        self.assertEqual(local_assets.LIBRARY.read_bytes(), before)
        (custom / "model.safetensors").unlink()
        restored = self.ui.restore_checkpoint("official")
        self.assertEqual(restored[0]["value"], path)
        self.assertIsNone(restored[1])
        self.assertTrue(restored[2])
        self.assertEqual(local_assets.LIBRARY.read_bytes(), before)

    def test_buttons_remain_disabled_for_stale_confirmation_or_active_job(self):
        confirmed = {"checkpoint": "saved", "engine": "official"}
        self.assertTrue(self.ui.generation_controls(confirmed, "saved", "official", False)[0]["interactive"])
        for current, engine, busy in (("other", "official", False), ("saved", "cpp", False),
                                      ("saved", "official", True)):
            with self.subTest(current=current, engine=engine, busy=busy):
                self.assertTrue(all(not item["interactive"] for item in
                                    self.ui.generation_controls(confirmed, current, engine, busy)))

    def test_job_completion_does_not_enable_a_different_unconfirmed_model(self):
        confirmed = {"checkpoint": "saved", "engine": "official"}
        status = {"message": "完了", "elapsed": 1, "done": True}
        with patch.object(self.ui.STUDIO, "status", return_value=status), patch.object(self.ui.STUDIO, "history", return_value=[]):
            stale = self.ui.poll("job", confirmed, "other", "official", self.request)
            valid = self.ui.poll("job", confirmed, "saved", "official", self.request)
        self.assertFalse(stale[1]["interactive"])
        self.assertFalse(stale[2]["interactive"])
        self.assertFalse(stale[-1])
        self.assertTrue(valid[1]["interactive"])
        self.assertTrue(valid[2]["interactive"])
