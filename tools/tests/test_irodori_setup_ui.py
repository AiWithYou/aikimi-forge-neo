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


class UITests(unittest.TestCase):
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
