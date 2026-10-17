"""Construct the actual MOSS Image controls and exercise their service bindings."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import gradio as gr

ROOT = Path(__file__).resolve().parents[2]


class UITests(unittest.TestCase):
    def load_screen(self):
        path = ROOT / "extensions-builtin/moss-image-studio/scripts/moss_image_studio.py"
        spec = importlib.util.spec_from_file_location("test_moss_screen", path)
        module = importlib.util.module_from_spec(spec)
        callbacks = types.ModuleType("modules.script_callbacks")
        callbacks.on_ui_tabs = lambda callback: None
        callbacks.on_app_started = lambda callback: None
        with patch.dict(sys.modules, {"modules.script_callbacks": callbacks}):
            spec.loader.exec_module(module)
        return module

    def test_navigation_common_mode_switch_and_model_directory_include_moss(self):
        from modules_forge.model_catalog import STUDIO_FAMILIES

        folder = ROOT / "extensions-builtin/aikimi-ui"
        script = (folder / "javascript/aikimi_tabs.js").read_text(encoding="utf-8")
        self.assertIn('buttonId: "aikimi-tab-moss-image"', script)
        self.assertIn('containerId: "tab_moss_image_studio"', script)
        self.assertIn('"moss_image"', (folder / "javascript/aikimi_image_workspace.js").read_text(encoding="utf-8"))
        self.assertIn('[data-active-feature="moss_image"]', (folder / "style.css").read_text(encoding="utf-8"))
        self.assertIn("MOSS-Image", STUDIO_FAMILIES)

    def test_real_screen_has_fixed_sft_model_expected_controls_and_private_events(self):
        module = self.load_screen()
        blocks, label, identifier = module.on_ui_tabs()[0]
        self.addCleanup(blocks.close)
        self.assertEqual((label, identifier), ("MOSS Image", "moss_image_studio"))
        components = list(blocks.blocks.values())
        props = [item.get("props", {}) for item in blocks.get_config_file()["components"]]
        ids = [item["elem_id"] for item in props if item.get("elem_id")]
        self.assertEqual(len(ids), len(set(ids)))
        fixed = next(item for item in props if item.get("elem_id") == "moss-image-model-name")
        self.assertEqual(fixed["value"], "MOSS-Image-SFT")
        self.assertFalse(fixed["interactive"])
        self.assertFalse(
            any(isinstance(item, gr.Dropdown) and getattr(item, "label", None) == "モデル" for item in components)
        )
        for label, value in (("幅", 1024), ("高さ", 1024), ("Steps", 50), ("CFG", 3), ("Shift", 5), ("Seed", -1)):
            with self.subTest(label=label):
                self.assertEqual(next(item for item in props if item.get("label") == label)["value"], value)
        self.assertTrue(any(item.get("label") == "ネガティブプロンプト" for item in props))
        skip = next(item for item in props if item.get("elem_id") == "moss-image-skip-layer")
        self.assertTrue(skip["value"])
        self.assertTrue(any(item.get("label") == "PNG・生成条件" for item in props))
        mode = next(item for item in components if item.elem_id == "moss-image-img2img")
        self.assertFalse(mode.interactive)
        for event in blocks.fns.values():
            self.assertEqual(event.api_visibility, "private")

    def test_start_preserves_explicit_generation_settings(self):
        module = self.load_screen()
        with patch.object(module.SERVICE, "start", return_value="job") as start:
            result = module.start({}, "a fox", "blur", 768, 1024, 23, 4.5, 2.5, 7, False)
        self.assertEqual(
            start.call_args.args[0],
            dict(
                prompt="a fox",
                negative_prompt="blur",
                width=768,
                height=1024,
                steps=23,
                cfg=4.5,
                shift=2.5,
                seed=7,
                skip_layer=False,
                precision="int8",
                text_precision="int8",
                loras=[],
            ),
        )
        self.assertEqual(result[0], {"job": "job"})
        self.assertFalse(result[2]["interactive"])
        self.assertTrue(result[-1]["active"])

    def test_precision_and_multi_lora_controls_do_not_change_model_or_sampling(self):
        module = self.load_screen()
        with patch.object(
            module.local_lora, "inventory", return_value=["first.safetensors", "nested/second.safetensors"]
        ):
            blocks = module.on_ui_tabs()[0][0]
        self.addCleanup(blocks.close)
        components = list(blocks.blocks.values())
        precision = next(item for item in components if item.elem_id == "moss-image-precision")
        text = next(item for item in components if item.elem_id == "moss-image-text-precision")
        vae = next(item for item in components if item.elem_id == "moss-image-vae-precision")
        names = next(item for item in components if item.elem_id == "moss-image-loras")
        rows = next(item for item in components if item.elem_id == "moss-image-lora-strengths")
        self.assertEqual(precision.value, "int8")
        self.assertEqual(text.value, "int8")
        self.assertEqual({value for _, value in precision.choices}, {"int8", "bf16"})
        self.assertEqual({value for _, value in text.choices}, {"int8", "w4a8", "bf16"})
        self.assertEqual(vae.value, "FP32")
        self.assertFalse(vae.interactive)
        self.assertEqual(names.value, [])
        self.assertTrue(names.multiselect)
        self.assertTrue(names.do_not_save_to_config)
        selection = next(event for event in blocks.fns.values() if event.fn is module.select_loras)
        self.assertEqual(selection.outputs, [rows])
        with patch.object(module.SERVICE, "start", return_value="job") as start:
            module.start(
                {},
                "cat",
                "blur",
                768,
                1024,
                23,
                4.5,
                2.5,
                7,
                False,
                "bf16",
                "w4a8",
                ["first.safetensors", "nested/second.safetensors"],
                [["first.safetensors", 0.75], ["nested/second.safetensors", -0.5]],
            )
        request = start.call_args.args[0]
        self.assertEqual(
            (request["precision"], request["text_precision"], request["steps"], request["cfg"]),
            ("bf16", "w4a8", 23, 4.5),
        )
        self.assertEqual(
            request["loras"],
            [{"name": "first.safetensors", "strength": 0.75}, {"name": "nested/second.safetensors", "strength": -0.5}],
        )

    def test_lora_selection_and_refresh_keep_valid_strengths_without_startup_restore(self):
        module = self.load_screen()
        with patch.object(module.local_lora, "header_info"), patch.object(module.local_lora, "resolve"):
            result = module.select_loras(["first.safetensors", "second.safetensors"], [["first.safetensors", 0.75]])
        self.assertEqual(result["value"], [["first.safetensors", 0.75], ["second.safetensors", 1.0]])
        with patch.object(module.local_lora, "inventory", return_value=["second.safetensors"]):
            update = module.refresh_loras(["first.safetensors", "second.safetensors"])
        self.assertEqual(update["value"], ["second.safetensors"])

    def test_poll_returns_completed_files_and_unlocks_generation(self):
        module = self.load_screen()
        record = {
            "status": "complete",
            "width": 768,
            "height": 1024,
            "seed": 7,
            "image_path": "output.png",
            "files": ["output.png", "result.json"],
        }
        with patch.object(module.SERVICE, "poll", return_value=record):
            state, message, image, files, run, cancel, timer = module.poll({"job": "job"})
        self.assertEqual(state, {})
        self.assertIn("768×1024", message)
        self.assertEqual(image, "output.png")
        self.assertEqual(files, record["files"])
        self.assertTrue(run["interactive"])
        self.assertFalse(cancel["interactive"])
        self.assertFalse(timer["active"])

    def test_poll_translates_progress_without_displaying_backend_details(self):
        module = self.load_screen()
        for stage, message in (
            ("verifying", "生成準備中…"),
            ("initializing", "生成準備中…"),
            ("loading_text_encoder", "生成準備中…"),
            ("encoding_prompt", "プロンプトを処理中…"),
            ("loading_diffusion", "生成準備中…"),
            ("sampling", "画像を生成中… 3/50"),
            ("decoding", "画像を仕上げ中…"),
            ("saving", "保存中…"),
            ("H:/private", "生成中…"),
        ):
            with (
                self.subTest(stage=stage),
                patch.object(
                    module.SERVICE,
                    "poll",
                    return_value={
                        "status": "running",
                        "stage": stage,
                        "current": 3,
                        "total": 50,
                        "detail": "H:/private/model.safetensors",
                    },
                ),
            ):
                result = module.poll({"job": "job"})
                self.assertEqual(result[1], message)
                self.assertEqual(result[0], {"job": "job"})

    def test_app_started_registers_authenticated_api(self):
        module = self.load_screen()
        app = object()
        with patch("modules_forge.moss_image.api.register_api") as register:
            module.app_started(None, app)
        register.assert_called_once_with(app)


if __name__ == "__main__":
    unittest.main()
