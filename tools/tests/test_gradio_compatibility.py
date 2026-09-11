import importlib.metadata
import inspect
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gradio as gr
import gradio.processing_utils
from packaging.requirements import Requirement
from PIL import Image

from modules import gradio_compat, gradio_runtime

ROOT = Path(__file__).resolve().parents[2]


def import_ui_tempdir():
    with patch.object(sys, "argv", ["gradio-compatibility-test"]):
        from modules import (
            shared,  # noqa: F401
            ui_tempdir,
        )

    return ui_tempdir


class GradioDependencyContractTests(unittest.TestCase):
    def test_runtime_matches_the_audited_direct_dependency_set(self):
        expected = {
            "diffusers": "0.38.0",
            "GitPython": "3.1.61",
            "gradio": gradio_compat.SUPPORTED_GRADIO_VERSION,
            "gradio-client": "2.5.0",
            "fastapi": "0.141.1",
            "starlette": "1.6.0",
            "huggingface-hub": "1.5.0",
            "peft": "0.20.0",
            "transformers": "5.10.4",
        }
        self.assertEqual(
            {name: importlib.metadata.version(name) for name in expected},
            expected,
        )

    def test_requirements_are_the_single_source_for_gradio_installation(self):
        requirements = {}
        for raw_line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if " #" in line:
                line = line.split(" #", 1)[0].rstrip()
            requirement = Requirement(line)
            requirements[requirement.name] = requirement

        self.assertEqual(
            str(requirements["gradio"].specifier),
            f"=={gradio_compat.SUPPORTED_GRADIO_VERSION}",
        )
        self.assertEqual(str(requirements["gradio-client"].specifier), "==2.5.0")
        self.assertNotIn("gradio-rangeslider", requirements)
        self.assertNotIn("torch", requirements)

        launch_source = (ROOT / "modules" / "launch_utils.py").read_text(encoding="utf-8")
        self.assertNotIn("GRADIO_PACKAGE", launch_source)
        self.assertNotIn('is_installed("gradio")', launch_source)


class GradioFileHandlingTests(unittest.TestCase):
    def test_temp_override_keeps_gradio_official_async_path_validator(self):
        ui_tempdir = import_ui_tempdir()
        official_validator = gradio.processing_utils.async_move_files_to_cache

        ui_tempdir.install_ui_tempdir_override()

        self.assertIs(
            gradio.processing_utils.async_move_files_to_cache,
            official_validator,
        )
        self.assertEqual(official_validator.__module__, "gradio.processing_utils")

    def test_png_cache_override_accepts_gradio6_signature_and_keeps_metadata(self):
        ui_tempdir = import_ui_tempdir()
        image = Image.new("RGB", (4, 4), "white")
        image.info["parameters"] = "seed=123"

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(
                ui_tempdir,
                "shared",
                SimpleNamespace(opts=SimpleNamespace(temp_dir=""), demo=None),
            ),
        ):
            output = Path(
                ui_tempdir.save_pil_to_file(
                    image,
                    cache_dir=temp_dir,
                    name="preview",
                    format="png",
                )
            )
            with Image.open(output) as restored:
                self.assertEqual(restored.info["parameters"], "seed=123")

            self.assertTrue(output.name.startswith(ui_tempdir.MANAGED_TEMP_PREFIX))
            self.assertEqual(output.suffix, ".png")


class FakeTunnelProcess:
    def __init__(self) -> None:
        self.returncode = None
        self.killed = False
        self.wait_count = 0

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        self.wait_count += 1
        if self.wait_count == 1:
            raise subprocess.TimeoutExpired("frpc", timeout)
        self.returncode = -1
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -1


class FakeTunnel:
    def __init__(self, local_port: int) -> None:
        self.local_port = local_port
        self.proc = FakeTunnelProcess()
        self.terminated = False

    def kill(self):
        self.terminated = True


class GradioRuntimeCleanupTests(unittest.TestCase):
    def test_close_removes_only_new_share_tunnels_and_waits_for_process(self):
        existing = FakeTunnel(7000)
        owned = FakeTunnel(7860)
        unrelated = FakeTunnel(9000)
        demo = SimpleNamespace(close=Mock(), server_port=7860)

        with patch.object(gradio_runtime, "CURRENT_TUNNELS", [existing, owned, unrelated]):
            gradio_runtime.close_gradio_runtime(demo, {id(existing)})

            demo.close.assert_called_once_with()
            self.assertFalse(existing.terminated)
            self.assertTrue(owned.terminated)
            self.assertFalse(unrelated.terminated)
            self.assertTrue(owned.proc.killed)
            self.assertEqual(gradio_runtime.CURRENT_TUNNELS, [existing, unrelated])

    def test_close_releases_a_real_local_gradio_listener(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]

        with gr.Blocks(analytics_enabled=False) as demo:
            gr.Markdown("runtime cleanup")
        baseline = gradio_runtime.tunnel_snapshot()
        demo.launch(
            server_name="127.0.0.1",
            server_port=port,
            prevent_thread_lock=True,
            quiet=True,
            ssr_mode=False,
        )
        gradio_runtime.close_gradio_runtime(demo, baseline)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as rebound:
            rebound.bind(("127.0.0.1", port))


class GradioUiContractTests(unittest.TestCase):
    def test_gradio6_launch_keeps_security_and_app_level_parameters(self):
        parameters = inspect.signature(gr.Blocks.launch).parameters
        for name in ("auth", "allowed_paths", "blocked_paths", "theme", "head"):
            self.assertIn(name, parameters)

    def test_representative_upload_gallery_and_file_components_build(self):
        import_ui_tempdir()
        with gr.Blocks() as demo:
            image = gr.Image(sources="upload", type="pil", buttons=[])
            gallery = gr.Gallery(type="pil", buttons=[])
            upload = gr.File(file_count="multiple", file_types=["image"])

        config = demo.get_config_file()
        component_types = {component["type"] for component in config["components"]}
        self.assertTrue({"image", "gallery", "file"}.issubset(component_types))
        self.assertEqual(image.sources, ["upload"])
        self.assertEqual(gallery.buttons, [])
        self.assertEqual(upload.file_count, "multiple")

    def test_hidden_components_are_only_mounted_with_explicit_opt_in(self):
        import_ui_tempdir()
        with gr.Blocks():
            slider = gr.Slider(visible=False)
            with gr.Column(visible=False) as column:
                gr.Textbox()
            mounted_slider = gr.Slider(visible=gradio_compat.keep_hidden_component_mounted(False))

        self.assertIs(slider.visible, False)
        self.assertIs(column.visible, False)
        self.assertEqual(mounted_slider.visible, "hidden")
        self.assertIs(gr.update(visible=False)["visible"], False)

    def test_input_accordion_internal_checkbox_stays_mounted(self):
        import_ui_tempdir()
        from modules.ui_components import InputAccordionImpl

        with gr.Blocks():
            accordion_input = InputAccordionImpl(value=False, setup=True, elem_id="test-input-accordion")

        self.assertEqual(accordion_input.visible, "hidden")
        self.assertEqual(accordion_input.elem_id, "test-input-accordion-checkbox")

    def test_unnamed_ui_events_are_private_without_hiding_named_apis(self):
        import_ui_tempdir()
        with gr.Blocks() as demo:
            first = gr.Button("Internal")
            second = gr.Button("Named")
            third = gr.Button("Explicit public")
            callable_html = gr.HTML(value=lambda: "callable initial value")
            chained = first.click(lambda: None)
            second.click(lambda: None, api_name="named_contract")
            third.click(lambda: None, api_visibility="public")
            chained_then = chained.then(lambda: None)
            chained_success = chained.success(lambda: None)
            chained_failure = chained.failure(lambda: None)
            load_event = demo.load(lambda: None)
            combined_load_event = gr.on(fn=lambda: None)
            named_chain = chained.then(lambda: None, api_name="named_chain_contract")
            explicit_public_load = demo.load(lambda: None, api_visibility="public")

        dependencies = demo.get_config_file()["dependencies"]
        by_id = {dependency["id"]: dependency for dependency in dependencies}
        by_name = {dependency["api_name"]: dependency for dependency in dependencies}
        unnamed = next(
            dependency for dependency in dependencies if dependency["api_name"] not in {"named_contract", None}
        )

        self.assertEqual(unnamed["api_visibility"], "private")
        self.assertEqual(by_name["named_contract"]["api_visibility"], "public")
        explicit_public = next(
            dependency
            for dependency in dependencies
            if dependency["api_visibility"] == "public" and dependency["api_name"] != "named_contract"
        )
        self.assertEqual(explicit_public["api_visibility"], "public")
        for dependency in (chained_then, chained_success, chained_failure, load_event, combined_load_event):
            self.assertEqual(by_id[dependency["id"]]["api_visibility"], "private")
        self.assertEqual(by_id[named_chain["id"]]["api_visibility"], "public")
        self.assertEqual(by_id[explicit_public_load["id"]]["api_visibility"], "public")
        callable_dependency = next(
            dependency for dependency in dependencies if callable_html._id in dependency["outputs"]
        )
        self.assertEqual(callable_dependency["api_visibility"], "private")

    def test_multi_control_visibility_workarounds_are_explicit_and_bounded(self):
        h3_source = (ROOT / "extensions-builtin" / "minimax-h3-studio" / "scripts" / "minimax_h3_studio.py").read_text(
            encoding="utf-8"
        )
        hyperweave_source = (ROOT / "extensions-builtin" / "hyperweave" / "scripts" / "hyperweave.py").read_text(
            encoding="utf-8"
        )
        controlnet_source = (
            ROOT
            / "extensions-builtin"
            / "sd_forge_controlnet"
            / "lib_controlnet"
            / "controlnet_ui"
            / "controlnet_ui_group.py"
        ).read_text(encoding="utf-8")

        self.assertIn('visible="hidden", elem_id="h3-keyframes"', h3_source)
        self.assertIn('visible="hidden", elem_id="h3-references"', h3_source)
        self.assertGreaterEqual(
            hyperweave_source.count("gradio_compat.keep_hidden_component_mounted("),
            6,
        )
        self.assertGreaterEqual(
            controlnet_source.count("gradio_compat.keep_hidden_component_mounted("),
            5,
        )

    def test_settings_are_initialized_without_a_full_page_load_replay(self):
        settings_source = (ROOT / "modules" / "ui_settings.py").read_text(encoding="utf-8")

        self.assertIn("value=fun()", settings_source)
        self.assertIn("value=opts.dumpjson()", settings_source)
        self.assertIn('elem_id="settings_json"', settings_source)
        self.assertIn('visible="hidden"', settings_source)
        self.assertNotIn("value=lambda: opts.dumpjson()", settings_source)
        self.assertNotIn("demo.load(", settings_source)
        self.assertNotIn("get_settings_values", settings_source)

    def test_late_ui_loaded_callbacks_run_immediately(self):
        source = (ROOT / "script.js").read_text(encoding="utf-8")
        callback_block = source.split("function onUiLoaded(callback) {", 1)[1].split("\n}", 1)[0]

        self.assertLess(source.index("let executedOnLoaded = false;"), source.index("function onUiLoaded"))
        self.assertIn("if (executedOnLoaded)", callback_block)
        self.assertIn("callback();", callback_block)

    def test_options_bootstrap_waits_for_the_hydrated_settings_value(self):
        source = (ROOT / "javascript" / "ui.js").read_text(encoding="utf-8")
        bootstrap = source.split("function load_webui_settings", 1)[1].split("onOptionsChanged(function", 1)[0]

        self.assertIn("attempt >= 200", bootstrap)
        self.assertIn("textarea == null || !textarea.value", bootstrap)
        self.assertIn("opts = JSON.parse(jsdata)", bootstrap)
        self.assertIn("catch (_error)", bootstrap)
        self.assertNotIn("console.error(_error", bootstrap)
        self.assertIn("onUiLoaded(load_webui_settings);", source)

    def test_runtime_import_does_not_generate_type_stubs_in_checkout(self):
        with patch.object(sys, "argv", ["gradio-compatibility-test"]):
            import modules.ui_components  # noqa: F401

        self.assertFalse((ROOT / "modules" / "ui_components.pyi").exists())

    def test_single_style_editor_rejects_gradio6_list_values_safely(self):
        self.assertEqual(gradio_compat.normalize_single_selection("style"), "style")
        self.assertEqual(gradio_compat.normalize_single_selection([]), "")

    def test_controlnet_range_replacement_preserves_order_and_bounds(self):
        self.assertEqual(gradio_compat.normalize_unit_interval(0.2, 0.8), (0.2, 0.8))
        self.assertEqual(gradio_compat.normalize_unit_interval(0.9, 0.1), (0.1, 0.9))
        self.assertEqual(gradio_compat.normalize_unit_interval(-1, 2), (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
