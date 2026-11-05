"""Construct the actual MOSS Image controls and exercise their service bindings."""

import ast
import importlib.util
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import gradio as gr

ROOT = Path(__file__).resolve().parents[2]


def restore_defaults(blocks, values):
    """Restore actual Gradio components through the production defaults methods."""
    path = ROOT / "modules/ui_loadsave.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    choices = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "radio_choices")
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "UiLoadsave")
    names = {"__init__", "add_component", "add_block", "read_from_file"}
    owner.body = [node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {
        "gr": gr, "json": json, "math": math,
        "InputAccordionImpl": type("InputAccordionImpl", (), {}), "ToolButton": type("ToolButton", (), {}),
    }
    exec(compile(ast.Module(body=[choices, owner], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    with tempfile.TemporaryDirectory() as folder:
        filename = Path(folder) / "ui.json"
        filename.write_text(json.dumps(values), encoding="utf-8")
        loader = namespace["UiLoadsave"](filename)
        loader.add_block(blocks, "moss_image_studio")


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

    def test_generation_is_disabled_until_the_saved_model_has_been_verified(self):
        module = self.load_screen()
        with patch.object(module.local_assets, "selection", return_value={"checkpoint": "official"}):
            blocks = module.on_ui_tabs()[0][0]
        self.addCleanup(blocks.close)
        run = next(item for item in blocks.blocks.values() if isinstance(item, gr.Button) and item.value == "生成")
        self.assertFalse(run.interactive)
        generation = next(event for event in blocks.fns.values() if event.fn is module.start)
        self.assertIsInstance(generation.inputs[-1], gr.State)
        self.assertIs(generation.inputs[-1].value, False)
        with patch.object(module.SERVICE, "start") as service, self.assertRaises(gr.Error):
            module.start({}, "cat", "", 512, 512, 20, 3, 5, 7, True, checkpoint_ready=False)
        service.assert_not_called()

    def test_navigation_common_mode_switch_and_model_directory_include_moss(self):
        from modules_forge.model_catalog import STUDIO_FAMILIES

        folder = ROOT / "extensions-builtin/aikimi-ui"
        script = (folder / "javascript/aikimi_tabs.js").read_text(encoding="utf-8")
        self.assertIn('buttonId: "aikimi-tab-moss-image"', script)
        self.assertIn('containerId: "tab_moss_image_studio"', script)
        self.assertIn('"moss_image"', (folder / "javascript/aikimi_image_workspace.js").read_text(encoding="utf-8"))
        self.assertIn('[data-active-feature="moss_image"]', (folder / "style.css").read_text(encoding="utf-8"))
        self.assertIn("MOSS-Image", STUDIO_FAMILIES)

    def test_real_screen_has_model_selection_expected_controls_and_private_events(self):
        module = self.load_screen()
        blocks, label, identifier = module.on_ui_tabs()[0]
        self.addCleanup(blocks.close)
        self.assertEqual((label, identifier), ("MOSS Image", "moss_image_studio"))
        components = list(blocks.blocks.values())
        props = [item.get("props", {}) for item in blocks.get_config_file()["components"]]
        ids = [item["elem_id"] for item in props if item.get("elem_id")]
        self.assertEqual(len(ids), len(set(ids)))
        fixed = next(item for item in props if item.get("elem_id") == "moss-image-model-name")
        self.assertEqual(fixed["value"], "official")
        self.assertFalse(fixed["interactive"])
        self.assertNotIn("moss-image-profile", ids)
        self.assertTrue(
            any(isinstance(item, gr.Dropdown) and getattr(item, "label", None) == "モデル" for item in components)
        )
        for label, value in (("幅", 1024), ("高さ", 1024), ("Steps", 50), ("CFG", 3), ("Shift", 5), ("Seed", -1)):
            with self.subTest(label=label):
                self.assertEqual(next(item for item in props if item.get("label") == label)["value"], value)
        for label in ("幅", "高さ"):
            with self.subTest(maximum=label):
                self.assertEqual(next(item for item in props if item.get("label") == label)["maximum"], 4096)
        self.assertTrue(any(item.get("label") == "ネガティブプロンプト" for item in props))
        skip = next(item for item in props if item.get("elem_id") == "moss-image-skip-layer")
        self.assertTrue(skip["value"])
        attention = next(item for item in props if item.get("elem_id") == "moss-image-reference-attention")
        self.assertEqual((attention["value"], attention["minimum"], attention["maximum"]), (1, 0, 4))
        self.assertIn("0でも", attention["info"])
        self.assertTrue(any(item.get("label") == "PNG・生成条件" for item in props))
        mode = next(item for item in components if item.elem_id == "moss-image-img2img")
        self.assertTrue(mode.interactive)
        for event in blocks.fns.values():
            self.assertEqual(event.api_visibility, "private")

    def test_start_preserves_explicit_generation_settings(self):
        module = self.load_screen()
        with patch.object(module.SERVICE, "start", return_value="job") as start:
            result = module.start({}, "a fox", "blur", 768, 1024, 23, 4.5, 2.5, 7, False, checkpoint_ready=True)
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
                rope={"scale_x": 1.0, "scale_y": 1.0},
            ),
        )
        self.assertEqual(result[0], {"job": "job"})
        self.assertFalse(result[2]["interactive"])
        self.assertTrue(result[-1]["active"])

    def test_reference_attention_is_sent_with_reference_edit(self):
        module = self.load_screen()
        with patch.object(module.SERVICE, "start", return_value="job") as start:
            module.start({}, "edit", "", 512, 512, 24, 3, 5, 42, True,
                         mode="img2img", images=["image"], edit_method="reference",
                         edit_lora="edit.safetensors", image_strength=0.35, checkpoint_ready=True)
        self.assertEqual(start.call_args.args[0]["image_strength"], 0.35)
        self.assertEqual(start.call_args.args[0]["loras"], [{"name": "edit.safetensors", "strength": 1.0}])

    def test_relevant_controls_follow_mode_and_cfg_without_overwriting_values(self):
        module = self.load_screen()
        blocks = module.on_ui_tabs()[0][0]
        self.addCleanup(blocks.close)
        components = list(blocks.blocks.values())
        hires = next(item for item in components if item.elem_id == "moss-image-hires-controls")
        for mode, visible in (("img2img", "hidden"), ("txt2img", True)):
            tab = next(item for item in components if item.elem_id == f"moss-image-{mode}")
            mode_event = next(event for event in blocks.fns.values() if event.outputs == [hires] and (tab._id, "select") in event.targets)
            self.assertEqual(mode_event.fn()["visible"], visible)
        negative = next(item for item in components if getattr(item, "label", None) == "ネガティブプロンプト")
        skip = next(item for item in components if item.elem_id == "moss-image-skip-layer")
        cfg_event = next(event for event in blocks.fns.values() if event.outputs == [negative, skip])
        for cfg in (0.9, 1, 1.1, 3):
            with self.subTest(cfg=cfg):
                updates = cfg_event.fn(cfg)
                self.assertEqual("不使用" in updates[0]["label"], cfg == 1)
                self.assertEqual(updates[1]["interactive"], cfg != 1)
                self.assertTrue(all("value" not in update for update in updates))

    def test_browser_restores_only_its_saved_job_and_resumes_polling(self):
        module = self.load_screen()
        blocks = module.on_ui_tabs()[0][0]
        self.addCleanup(blocks.close)
        saved = next(item for item in blocks.blocks.values() if isinstance(item, gr.BrowserState))
        event = next(event for event in blocks.fns.values() if (saved._id, "change") in event.targets)
        self.assertIs(event.fn, module.restore_job)
        self.assertIsNone(saved.default_value)
        server_state = event.inputs[0]
        self.assertIsInstance(server_state, gr.State)
        persist = next(event for event in blocks.fns.values() if event.outputs == [saved])
        self.assertEqual(persist.fn({"job": "my-job", "prompt": "private text"}), "my-job")
        self.assertIsNone(persist.fn({}))
        with patch.object(module.SERVICE, "poll", return_value={"status": "running", "stage": "sampling"}) as poll:
            result = event.fn({}, "my-job")
            poll.assert_called_once_with("my-job")
        self.assertEqual(result[0], {"job": "my-job"})
        self.assertFalse(result[4]["interactive"])
        self.assertTrue(result[5]["interactive"])
        self.assertTrue(result[6]["active"])
        with patch.object(module.SERVICE, "poll") as poll:
            event.fn({}, None)
            event.fn({"job": "new-job"}, "old-job")
            poll.assert_not_called()

    def test_text_mode_excludes_retained_images_and_edit_lora(self):
        from PIL import Image

        module = self.load_screen()
        with patch.object(module.SERVICE, "start", return_value="job") as dispatch:
            module.start(
                {},
                "cat",
                "",
                512,
                512,
                50,
                3,
                5,
                7,
                True,
                mode="txt2img",
                images=[(Image.new("RGB", (512, 512)), None)],
                edit_method="reference",
                edit_lora="edit.safetensors",
                checkpoint_ready=True,
            )
        request = dispatch.call_args.args[0]
        self.assertNotIn("images", request)
        self.assertEqual(request["loras"], [])

    def test_redraw_hides_edit_adapter_controls_until_reference_method_is_selected(self):
        module = self.load_screen()
        blocks = module.on_ui_tabs()[0][0]
        self.addCleanup(blocks.close)
        edit = next(item for item in blocks.blocks.values() if item.elem_id == "moss-image-edit-lora")
        parent = edit.parent
        while not isinstance(parent, gr.Column):
            parent = parent.parent
        self.assertFalse(parent.visible)
        self.assertFalse(module.edit_controls("redraw")[1]["visible"])

    def test_reference_size_uses_4096_as_its_maximum_side(self):
        from types import SimpleNamespace

        module = self.load_screen()
        with patch("PIL.ImageOps.exif_transpose", return_value=SimpleNamespace(size=(7680, 4320))):
            self.assertEqual(module.reference_size([object()]), (4096, 2304))
        self.assertTrue(module.edit_controls("reference")[1]["visible"])

    def test_reference_mode_dispatches_ordered_images_and_separate_edit_adapter(self):
        from PIL import Image

        module = self.load_screen()
        images = [Image.new("RGB", (512, 512), "red"), Image.new("RGB", (512, 768), "blue")]
        with patch.object(module.SERVICE, "start", return_value="job") as dispatch:
            module.start(
                {},
                "Picture 1 wears Picture 2",
                "",
                512,
                512,
                30,
                3,
                5,
                7,
                True,
                mode="img2img",
                images=[(image, None) for image in images],
                edit_method="reference",
                edit_lora="edit.safetensors",
                edit_strength=0.8,
                checkpoint_ready=True,
            )
        request = dispatch.call_args.args[0]
        self.assertEqual(request["images"], images)
        self.assertEqual(request["loras"], [{"name": "edit.safetensors", "strength": 0.8}])
        self.assertEqual((request["mode"], request["edit_method"]), ("img2img", "reference"))
        with self.assertRaises(gr.Error):
            module.start(
                {}, "edit", "", 512, 512, 30, 3, 5, 7, True, mode="img2img", images=images, edit_method="reference", checkpoint_ready=True
            )

    def test_custom_model_is_dispatched_and_precision_selection_does_not_change_sampling(self):
        module = self.load_screen()
        identity = {"name": "Pastel.safetensors", "precision": "bf16", "size": 10, "mtime_ns": 1, "sha256": "a" * 64}
        with patch.object(module.SERVICE, "start", return_value="job") as start:
            module.start(
                {},
                "pastelstyle, a fox",
                "blur",
                768,
                1024,
                50,
                3,
                5,
                7,
                True,
                "bf16",
                "int8",
                [],
                [],
                "Pastel.safetensors",
                checkpoint_identity=identity,
                checkpoint_ready=True,
            )
        request = start.call_args.args[0]
        self.assertEqual(request["checkpoint"], "Pastel.safetensors")
        self.assertEqual(start.call_args.kwargs["checkpoint_identity"], identity)
        self.assertEqual((request["steps"], request["cfg"], request["precision"]), (50, 3, "bf16"))
        with (
            patch.object(module.checkpoints, "resolve", return_value=Path("Pastel.safetensors")),
            patch.object(module.checkpoints.local_assets, "read_header", return_value={}),
            patch.object(module.checkpoints, "header_precision", return_value="bf16"),
        ):
            self.assertEqual(module.select_model("Pastel.safetensors", "int8"), gr.update(
                choices=[("INT8 ConvRot", "int8"), ("W4A8", "w4a8"), ("BF16", "bf16")],
                value="bf16", interactive=False,
            ))
        self.assertEqual(module.select_model("official", "w4a8"), gr.update(
            choices=[("INT8 ConvRot", "int8"), ("BF16", "bf16")], value="int8", interactive=True,
        ))
        self.assertEqual(module.select_model("official", "bf16"), gr.update(
            choices=[("INT8 ConvRot", "int8"), ("BF16", "bf16")], value="bf16", interactive=True,
        ))

    def test_custom_model_selection_restores_only_after_identity_verification(self):
        module = self.load_screen()
        identity = {"name": "nested/Pastel.safetensors", "precision": "w4a8", "size": 10, "mtime_ns": 1, "sha256": "a" * 64}
        stored = {"checkpoint": identity["name"], "identity": identity}
        with (
            patch.object(module.local_assets, "selection", return_value=stored),
            patch.object(module.checkpoints, "inventory", return_value=[("MOSS-Image-SFT", "official"), ("Pastel", identity["name"])]),
            patch.object(module.checkpoints, "inspect", return_value=identity) as inspect,
            patch.object(module.local_assets, "save_selection") as save,
        ):
            blocks = module.on_ui_tabs()[0][0]
            self.addCleanup(blocks.close)
            model = next(item for item in blocks.blocks.values() if item.elem_id == "moss-image-model-name")
            self.assertEqual(model.value, identity["name"])
            self.assertFalse(model.interactive)
            selection = next(event for event in blocks.fns.values() if event.fn is module.bind_model_selection)
            self.assertEqual([event for _, event in selection.targets], ["input"])
            restored = next(event for event in blocks.fns.values() if event.fn is module.restore_model)
            self.assertEqual([event for _, event in restored.targets], ["load"])
            selected, verified, precision, status, ready = module.restore_model(identity["name"], "int8")
            self.assertTrue(ready)
            self.assertEqual((selected["value"], verified, precision["value"]), (identity["name"], identity, "w4a8"))
            self.assertTrue(selected["interactive"])
            self.assertIn("確認", status)
            inspect.assert_called_with(module.MODEL_ROOT, identity["name"])
            save.assert_not_called()

    def test_reload_applies_saved_selection_even_if_server_started_with_official(self):
        module = self.load_screen()
        identity = {"name": "Pastel.safetensors", "precision": "bf16", "size": 10, "mtime_ns": 1, "sha256": "a" * 64}
        with (
            patch.object(module.local_assets, "selection", return_value={"checkpoint": identity["name"], "identity": identity}),
            patch.object(module.checkpoints, "inventory", return_value=[("MOSS-Image-SFT", "official"), ("Pastel", identity["name"])]),
            patch.object(module.checkpoints, "inspect", return_value=identity),
        ):
            selected, verified, precision, status, ready = module.restore_model("official", "int8")
        self.assertTrue(ready)
        self.assertEqual((selected["value"], verified, precision["value"]), (identity["name"], identity, "bf16"))
        self.assertIn("確認", status)

    def test_selecting_custom_model_saves_the_verified_identity(self):
        module = self.load_screen()
        identity = {"name": "Pastel.safetensors", "precision": "w4a8", "size": 10, "mtime_ns": 1, "sha256": "a" * 64}
        with (
            patch.object(module.checkpoints, "inspect", return_value=identity) as inspect,
            patch.object(module.local_assets, "save_selection") as save,
        ):
            selected, precision, status, ready = module.bind_model_selection(identity["name"], "int8")
        self.assertTrue(ready)
        self.assertEqual((selected, precision["value"]), (identity, "w4a8"))
        self.assertFalse(precision["interactive"])
        self.assertIn("確認", status)
        inspect.assert_called_once_with(module.MODEL_ROOT, identity["name"])
        save.assert_called_once_with("moss_image", {"checkpoint": identity["name"], "identity": identity})

    def test_changed_or_missing_model_stays_selected_and_cannot_run_as_official(self):
        module = self.load_screen()
        identity = {"name": "Pastel.safetensors", "precision": "bf16", "size": 10, "mtime_ns": 1, "sha256": "a" * 64}
        stored = {"checkpoint": identity["name"], "identity": identity}
        for current in (None, {**identity, "sha256": "b" * 64}):
            with (
                self.subTest(current=current),
                patch.object(module.local_assets, "selection", return_value=stored),
                patch.object(module.checkpoints, "inventory", return_value=[("MOSS-Image-SFT", "official")]),
                patch.object(module.checkpoints, "inspect", side_effect=ValueError("missing") if current is None else None, return_value=current),
                patch.object(module.local_assets, "save_selection") as save,
            ):
                selected, verified, _, status, ready = module.restore_model(identity["name"], "bf16")
                self.assertFalse(ready)
                self.assertEqual(selected["value"], identity["name"])
                self.assertIsNone(verified)
                self.assertIn("選び直", status)
                save.assert_not_called()

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
                checkpoint_ready=True,
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

    def test_edit_lora_refresh_preserves_only_available_selection(self):
        module = self.load_screen()
        blocks = module.on_ui_tabs()[0][0]
        self.addCleanup(blocks.close)
        refresh = next(item for item in blocks.blocks.values() if getattr(item, "value", None) == "編集用LoRA一覧を更新")
        event = next(event for event in blocks.fns.values() if (refresh._id, "click") in event.targets)
        with patch.object(module.local_lora, "inventory", return_value=["keep.safetensors"]):
            for selection, expected in (("keep.safetensors", "keep.safetensors"), ("deleted.safetensors", ""), (None, "")):
                with self.subTest(selection=selection):
                    update = event.fn(selection)
                    self.assertEqual(update["value"], expected)
                    dropdown = gr.Dropdown(choices=update["choices"], value=update["value"])
                    self.assertEqual(dropdown.preprocess(update["value"]), expected)

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

    def test_restored_finished_job_cannot_unlock_generation_before_model_validation(self):
        module = self.load_screen()
        record = {"status": "complete", "width": 64, "height": 64, "seed": 7,
                  "image_path": "output.png", "files": ["output.png", "result.json"]}
        with patch.object(module.SERVICE, "poll", return_value=record):
            restored = module.restore_job({}, "job", False)
        self.assertFalse(restored[4]["interactive"])
        self.assertFalse(module.generation_ready({"job": "job"}, True)["interactive"])
        self.assertTrue(module.generation_ready({}, True)["interactive"])

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
