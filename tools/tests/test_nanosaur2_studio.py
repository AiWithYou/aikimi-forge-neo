"""Offline contracts for the pinned Nanosaur2 setup and ComfyUI graph."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from modules_forge import nanosaur2_studio as studio
from tools import setup_nanosaur2


class Nanosaur2Contracts(unittest.TestCase):
    def test_manifest_tracks_upstream_files_and_destinations(self):
        manifest = setup_nanosaur2.manifest()
        self.assertEqual(manifest["revision"], "dcd61cd6c3f9e2cc62619da620bd86219191c3ed")
        profiles = setup_nanosaur2.profiles()
        self.assertEqual(len(profiles["source"].artifacts), 5)
        self.assertEqual(len(profiles["models"].artifacts), 3)
        self.assertTrue(
            all(
                item.relative_path.startswith("repositories/nanosaur2/ComfyUI/")
                for profile in profiles.values()
                for item in profile.artifacts
            )
        )

    def test_official_inference_graph_and_reproducible_seed(self):
        request = studio.Nanosaur2Request(
            prompt="newest, masterpiece, little fox under a lantern",
            width=832,
            height=1216,
            steps=50,
            cfg=4,
            seed=42,
        )
        graph = studio.build_workflow(request, request.resolved_seed())
        self.assertEqual(graph["loader"]["inputs"]["guidance"], "alternate")
        self.assertEqual(graph["sample"]["inputs"]["sampler_name"], "euler")
        self.assertEqual(graph["sample"]["inputs"]["scheduler"], "simple")
        self.assertEqual(graph["sample"]["inputs"]["latent_image"], ["latent", 0])
        self.assertEqual(graph["sample"]["inputs"]["seed"], 42)
        self.assertEqual(graph["decode"]["inputs"]["vae"], ["loader", 2])
        self.assertEqual(graph["save"]["inputs"]["images"], ["decode", 0])
        self.assertEqual(len(graph), 7)

    def test_invalid_requests_fail_before_runtime(self):
        for request in (
            studio.Nanosaur2Request(prompt=""),
            studio.Nanosaur2Request(prompt="x", width=513),
            studio.Nanosaur2Request(prompt="x", width=2048, height=2048),
            studio.Nanosaur2Request(prompt="x", cfg=float("nan")),
            studio.Nanosaur2Request(prompt="x", guidance="unknown"),
        ):
            with self.subTest(request=request), self.assertRaises(studio.Nanosaur2Error):
                studio.build_workflow(request, 42)

    def test_whitelist_is_opt_in_to_verified_node(self):
        from modules_forge.minimax_h3_bridge import _runtime_command

        plain = _runtime_command(Path("python.exe"), 8189)
        enabled = _runtime_command(Path("python.exe"), 8189, trusted_custom_nodes=(studio.NODE_NAME,))
        self.assertNotIn(studio.NODE_NAME, plain)
        self.assertIn(studio.NODE_NAME, enabled)

    def test_node_schema_rejects_missing_model(self):
        names = {
            "unet_name": [["nanosaur2_diffusion_model.safetensors"]],
            "text_encoder_name": [["nanosaur2_text_encoder.safetensors"]],
            "vae_name": [["nanosaur2_vae.safetensors"]],
            "guidance": [["alternate", "cfg", "path_drop"]],
        }
        client = Mock()
        client.object_info.return_value = {
            node: {"input": {"required": names if node == "Nanosaur2Loader" else {}}} for node in studio.REQUIRED_NODES
        }
        studio.check_nodes(client)
        names["vae_name"] = [[]]
        with self.assertRaisesRegex(studio.Nanosaur2Error, "vae"):
            studio.check_nodes(client)

    def load_ui(self):
        if importlib.util.find_spec("gradio") is None:
            self.skipTest("Gradio unavailable")
        callbacks = SimpleNamespace(on_ui_tabs=Mock())
        modules = types.ModuleType("modules")
        modules.script_callbacks = callbacks
        status = types.ModuleType("modules.aikimi_status")
        status.studio_status_html = Mock(return_value="")
        location = (
            Path(__file__).resolve().parents[2] / "extensions-builtin/nanosaur2-studio/scripts/nanosaur2_studio.py"
        )
        spec = importlib.util.spec_from_file_location("nanosaur2_ui_test", location)
        ui = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules, {"modules": modules, "modules.aikimi_status": status, "modules.script_callbacks": callbacks}
        ):
            spec.loader.exec_module(ui)
        return ui

    def test_ui_tab_can_render_without_starting_backend(self):
        ui = self.load_ui()
        with patch.object(studio, "ensure_runtime") as backend:
            tabs = ui.on_ui_tabs()
            backend.assert_not_called()
        self.assertEqual(tabs[0][1:], ("Nanosaur2", "nanosaur2_studio"))
        config = tabs[0][0].get_config_file()
        self.assertTrue(config["components"])
        self.assertTrue(
            all(item["api_visibility"] == "private" for item in config["dependencies"] if item.get("backend_fn"))
        )

    def test_setup_state_checks_missing_and_present_files_without_starting_or_hashing_models(self):
        ui = self.load_ui()
        with (
            TemporaryDirectory() as temporary,
            patch.object(studio, "runtime_root", return_value=Path(temporary)),
            patch.object(setup_nanosaur2, "runtime_ready", return_value=False) as runtime,
            patch.object(studio, "source_ready", return_value=True),
            patch.object(studio, "_entries", return_value=[{"path": "model.safetensors", "size": 4}]),
            patch.object(studio, "ensure_runtime") as backend,
            patch.object(studio, "model_ready") as full_hash,
        ):
            self.assertIn("未導入", ui._setup_state())
            runtime.return_value = True
            self.assertIn("不足分", ui._setup_state())
            models = Path(temporary) / "models"
            models.mkdir()
            (models / "model.safetensors").write_bytes(b"test")
            self.assertIn("生成時にモデルを検証", ui._setup_state())
            backend.assert_not_called()
            full_hash.assert_not_called()

    def test_connect_reports_progress_before_starting_and_keeps_failure_visible(self):
        ui = self.load_ui()
        with patch.object(studio, "ensure_runtime", side_effect=studio.Nanosaur2Error("test failure")) as backend:
            updates = ui._connect()
            self.assertIn("接続・起動しています", next(updates))
            backend.assert_not_called()
            self.assertIn("起動に失敗しました", next(updates))
            backend.assert_called_once_with(restart=False)


if __name__ == "__main__":
    unittest.main()
