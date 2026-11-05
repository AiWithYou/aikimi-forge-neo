"""Verify native UI registration and revision-pinned isolated setup."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


class SetupTests(unittest.TestCase):
    def test_isolated_python_ignores_parent_pythonhome_but_preserves_model_cache(self):
        from modules_forge.irodori.core import child_environment

        with patch.dict(
            "os.environ",
            {
                "PYTHONHOME": "other-python",
                "PYTHONPATH": "other-site",
                "HF_HOME": "model-cache",
                "HF_TOKEN": "test-only",
            },
        ):
            for offline in (True, False):
                environment = child_environment(offline=offline)
                self.assertNotIn("PYTHONHOME", environment)
                self.assertNotIn("PYTHONPATH", environment)
                self.assertEqual(environment["HF_HOME"], "model-cache")
                if offline:
                    self.assertNotIn("HF_TOKEN", environment)
                    self.assertEqual(environment["HF_HUB_OFFLINE"], "1")

    def test_download_plan_pins_every_model_component(self):
        from tools.setup_irodori import download_plan

        plan = download_plan()
        self.assertEqual(len(plan), 3)
        for record in plan:
            self.assertEqual(len(record["revision"]), 40)
            self.assertTrue(record["files"])
            self.assertNotIn("*", record["files"])

    def test_setup_runs_frozen_in_its_own_project(self):
        from tools.setup_irodori import sync_command

        command = [str(x) for x in sync_command(Path("uv"), Path("runtime/source"))]
        self.assertIn("--frozen", command)
        self.assertIn("cu128", command)
        self.assertIn("--project", command)
        self.assertIn(str(Path("runtime/source")), command)

    def test_ready_rejects_missing_or_changed_model_files(self):
        import tempfile

        from modules_forge.irodori.core import (
            CODE_REVISION,
            CODEC_REVISION,
            MODEL_REVISION,
            REQUIRED_FILES,
            WATERMARK_REVISION,
            atomic_json,
            python_path,
            ready,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / "models"
            python_path(root).parent.mkdir(parents=True)
            python_path(root).touch()
            records = []
            for name in REQUIRED_FILES:
                record = models / name
                record.parent.mkdir(parents=True, exist_ok=True)
                record.write_bytes(b"model")
                records.append({"file": name, "size": 5, "mtime_ns": record.stat().st_mtime_ns})
            atomic_json(
                root / "runtime.json",
                {
                    "code_revision": CODE_REVISION,
                    "model_revision": MODEL_REVISION,
                    "codec_revision": CODEC_REVISION,
                    "watermark_revision": WATERMARK_REVISION,
                },
            )
            atomic_json(root / "model-files.json", records)
            self.assertTrue(ready(root, models))
            (models / "v4-Large/model.safetensors").write_bytes(b"")
            self.assertFalse(ready(root, models))
            custom = models / "v4-Large/checkpoints/MyVoice.safetensors"
            custom.parent.mkdir()
            custom.write_bytes(b"custom")
            self.assertTrue(ready(root, models, checkpoint=custom.name))
            custom.unlink()
            self.assertFalse(ready(root, models, checkpoint=custom.name))


class UITests(unittest.TestCase):
    def test_empty_restore_event_cannot_overwrite_restored_ui_state(self):
        from modules_forge.irodori import ui

        with patch.object(ui.checkpoints, "inventory", return_value=[("公式 v4-Large", "official")]):
            model, precision, notice = ui.restore_model(None, None)
        self.assertEqual((model, precision, notice), (ui.gr.skip(), ui.gr.skip(), ui.gr.skip()))

    def test_load_before_saved_state_keeps_model_unselected_and_allows_later_restore(self):
        from modules_forge.irodori import ui

        identity = {"name": "Voice.safetensors", "sha256": "a" * 64, "format": "float", "size": 10}
        options = [("公式 v4-Large", "official"), ("Voice", "Voice.safetensors")]
        with (
            patch.object(ui.checkpoints, "inventory", return_value=options),
            patch.object(ui.checkpoints, "bind", return_value=identity),
            patch.object(ui, "ready", return_value=True),
        ):
            before, _, _ = ui.restore_model(None, None)
            self.assertEqual(before, ui.gr.skip())
            restored, _, notice = ui.restore_model(identity, None)
            self.assertEqual(restored["value"], identity["name"])
            self.assertFalse(notice["visible"])

            blocks = ui.build_ui()
            self.addCleanup(blocks.close)
            component = next(item for item in blocks.blocks.values() if item.elem_id == "irodori-checkpoint")
            self.assertIsNone(component.value)

    def test_official_submit_requires_explicit_confirmed_selection(self):
        from modules_forge.irodori import ui

        request = types.SimpleNamespace(session_hash="session")
        args = ("文", "", "official", None, "bf16", 40, 3, 3, 5, 7, 0, 1, [], request)
        with patch.object(ui.SERVICE, "start", return_value="job") as start:
            with self.assertRaises(ui.gr.Error):
                ui.submit(*args)
            start.assert_not_called()

            identity = {"name": "official", "sha256": "a" * 64, "format": "float", "size": 10}
            ui.submit(*args[:3], identity, *args[4:])
            self.assertEqual(start.call_args.args[0]["checkpoint_identity"], identity)

    def test_missing_official_requires_explicit_custom_selection(self):
        from modules_forge.irodori import ui

        options = [("公式 v4-Large", "official"), ("Voice", "Voice.safetensors")]
        with (
            patch.object(ui.checkpoints, "inventory", return_value=options),
            patch.object(ui, "ready", side_effect=lambda checkpoint="official": checkpoint != "official"),
        ):
            blocks = ui.build_ui()
        self.addCleanup(blocks.close)
        model = next(component for component in blocks.blocks.values() if component.elem_id == "irodori-checkpoint")
        self.assertIsNone(model.value)

    def test_model_choices_preserve_selection_and_never_fall_back(self):
        from modules_forge.irodori import ui

        with patch.object(ui.checkpoints, "inventory", return_value=[("公式 v4-Large", "official"), ("Voice", "Voice.safetensors")]):
            self.assertEqual(ui.refresh_models("Voice.safetensors")[0]["value"], "Voice.safetensors")
            model, notice = ui.refresh_models("missing.safetensors")
            self.assertIsNone(model["value"])
            self.assertTrue(notice["visible"])
        with (
            patch.object(ui.checkpoints, "bind", return_value={
                "name": "Voice.safetensors", "sha256": "a" * 64, "format": "int4_weight_only", "size": 10,
            }),
        ):
            update, identity, notice = ui.select_model("Voice.safetensors")
            self.assertEqual(update["value"], "bf16")
            self.assertFalse(update["interactive"])
            self.assertEqual(identity["name"], "Voice.safetensors")
            self.assertEqual(identity["sha256"], "a" * 64)
            self.assertFalse(notice["visible"])

    def test_reload_restores_only_the_same_model_contents(self):
        from modules_forge.irodori import ui

        identity = {"name": "Voice.safetensors", "sha256": "a" * 64, "format": "float", "size": 10}
        options = [("公式 v4-Large", "official"), ("Voice", "Voice.safetensors")]
        with (
            patch.object(ui.checkpoints, "inventory", return_value=options),
            patch.object(ui.checkpoints, "bind", return_value=identity) as bind,
        ):
            model, _, notice = ui.restore_model(identity, None)
            self.assertEqual(model["value"], "Voice.safetensors")
            self.assertFalse(notice["visible"])
            self.assertEqual(bind.call_args.args[1]["checkpoint_identity"], identity)
        with (
            patch.object(ui.checkpoints, "inventory", return_value=options),
            patch.object(ui.checkpoints, "bind", side_effect=ValueError("changed")),
        ):
            model, _, notice = ui.restore_model(identity, None)
            self.assertIsNone(model["value"])
            self.assertTrue(notice["visible"])
            self.assertIn("選び直", notice["value"])
        with patch.object(ui.checkpoints, "inventory", return_value=options[:1]):
            model, _, notice = ui.restore_model(identity, None)
            self.assertIsNone(model["value"])
            self.assertTrue(notice["visible"])
        with (
            patch.object(ui.checkpoints, "inventory", return_value=options),
            patch.object(ui.checkpoints, "bind", side_effect=ValueError("changed")),
        ):
            model, notice = ui.refresh_models("Voice.safetensors", identity)
            self.assertIsNone(model["value"])
            self.assertTrue(notice["visible"])

    def test_custom_selection_requires_its_recorded_identity_before_submit(self):
        from modules_forge.irodori import ui

        request = types.SimpleNamespace(session_hash="session")
        with patch.object(ui.SERVICE, "start") as start, self.assertRaises(ui.gr.Error):
            ui.submit("文", "", "Voice.safetensors", None, "bf16", 40, 3, 3, 5, 7, 0, 1, [], request)
        start.assert_not_called()

    def test_custom_tokenizer_resolution_is_forced_to_official_and_restored(self):
        from tools.irodori_worker import local_tokenizer

        original = unittest.mock.Mock()
        upstream = types.SimpleNamespace(_resolve_tokenizer_source=original)
        with self.assertRaisesRegex(RuntimeError, "stop"), local_tokenizer(upstream, Path("official-tokenizer")):
            self.assertEqual(
                upstream._resolve_tokenizer_source(Path("local/model.safetensors"), "google/t5gemma-2-1b-1b"),
                (str(Path("official-tokenizer")), True),
            )
            with self.assertRaises(ValueError):
                upstream._resolve_tokenizer_source(Path("local/model.safetensors"), "other/tokenizer")
            raise RuntimeError("stop")
        self.assertIs(upstream._resolve_tokenizer_source, original)

    def test_watermark_loader_uses_local_weights_and_restores_official_loader(self):
        from tools.irodori_worker import local_watermark

        original = unittest.mock.Mock()
        backend = types.ModuleType("silentcipher")
        backend.get_model = original
        with patch.dict(sys.modules, {"silentcipher": backend}):
            with self.assertRaisesRegex(RuntimeError, "stop"), local_watermark(Path("local-models")):
                backend.get_model(model_type="44.1k", device="cuda")
                raise RuntimeError("stop")
        self.assertIs(backend.get_model, original)
        self.assertEqual(original.call_args.kwargs["ckpt_path"], str(Path("local-models/44_1_khz/73999_iteration")))
        self.assertEqual(original.call_args.kwargs["device"], "cuda")

    def test_status_payload_reaches_the_shared_generation_indicator(self):
        from modules_forge.irodori.ui import assistant_state

        value = assistant_state("job-a", {"status": "running", "error": ""})
        self.assertIn('data-aikimi-source="irodori"', value)
        self.assertIn('data-stage="running"', value)
        self.assertIn('data-job="job-a"', value)

    def test_native_tab_and_controls_build_without_gpu_or_network(self):
        import gradio.routes

        self.assertIsNotNone(gradio.routes.Request)

        path = ROOT / "extensions-builtin/irodori-studio/scripts/irodori_studio.py"
        spec = importlib.util.spec_from_file_location("test_irodori_ui", path)
        module = importlib.util.module_from_spec(spec)
        callbacks = types.ModuleType("modules.script_callbacks")
        callbacks.on_ui_tabs = lambda callback: None
        with patch.dict(sys.modules, {"modules.script_callbacks": callbacks}):
            spec.loader.exec_module(module)
        blocks, label, identifier = module.on_ui_tabs()[0]
        self.assertEqual((label, identifier), ("Irodori TTS", "irodori_studio"))
        components = blocks.get_config_file()["components"]
        ids = {x.get("props", {}).get("elem_id") for x in components}
        self.assertTrue(
            {
                "irodori-text",
                "irodori-caption",
                "irodori-references",
                "irodori-audio",
                "irodori-generate",
                "irodori-cancel",
            }.issubset(ids)
        )
        precision = next(x for x in components if x.get("props", {}).get("elem_id") == "irodori-precision")
        self.assertEqual(precision["props"]["value"], "bf16")
        refs = next(x for x in components if x.get("props", {}).get("elem_id") == "irodori-references")
        self.assertEqual(refs["props"]["file_count"], "multiple")

    def test_token_overflow_fails_before_sampling(self):
        from tools.irodori_worker import check_token_lengths

        runtime = unittest.mock.Mock()
        runtime.tokenizer.encode.return_value.numel.return_value = 257
        runtime.default_text_max_len = 256
        with self.assertRaisesRegex(ValueError, "257 / 256"):
            check_token_lengths(runtime, {"text": "長い文", "caption": None})


if __name__ == "__main__":
    unittest.main()
