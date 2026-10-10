"""The folder button reads paths without changing the gallery's PIL contract."""

import asyncio
import json
import os
import tempfile
import unittest
from contextlib import closing
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gradio as gr
from gradio.context import LocalContext
from gradio.state_holder import SessionState
from PIL import Image

from modules import ui_tempdir
from tools.tests.test_gpu_ownership import load_function


class OutputFolderTests(unittest.TestCase):
    def callback(self, choice="Subdirectory", hidden=False):
        util = SimpleNamespace(open_folder=Mock())
        namespace = {
            "os": os,
            "json": json,
            "shared": SimpleNamespace(
                cmd_opts=SimpleNamespace(hide_ui_dir_config=hidden),
                opts=SimpleNamespace(open_dir_button_choice=choice, temp_dir=""),
                demo=SimpleNamespace(temp_file_sets=[set()]),
            ),
            "ui_tempdir": ui_tempdir,
            "util": util,
        }
        return load_function("modules/ui_common.py", "open_folder", namespace), namespace, util

    def test_gradio_gallery_saved_path_opens_selected_subfolder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "saved-subfolder"
            folder.mkdir()
            filename = folder / "result.png"
            Image.new("RGB", (2, 2)).save(filename)
            gallery = gr.Gallery(type="pil")
            payload = gallery.postprocess([str(filename)]).model_dump()
            callback, namespace, util = self.callback()
            ui_tempdir.register_tmp_file(namespace["shared"].demo, filename)
            with patch.object(ui_tempdir, "shared", namespace["shared"]):
                callback(str(root), payload, 0)
            util.open_folder.assert_called_once_with(str(folder))
            self.assertEqual(gallery.type, "pil")

    def test_missing_unselected_unregistered_and_temp_paths_fall_back(self):
        callback, namespace, util = self.callback()
        for payload, index in (
            (None, 0),
            ([], 0),
            ([{"image": {"path": "other.png"}}], -1),
            ([{"image": {"path": "other.png"}}], 0),
        ):
            with self.subTest(payload=payload, index=index):
                callback("output-root", payload, index)
                util.open_folder.assert_called_with("output-root")
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "temp.png"
            filename.touch()
            ui_tempdir.register_tmp_file(namespace["shared"].demo, filename)
            with patch.object(ui_tempdir, "is_gradio_temp_path", return_value=True):
                callback("output-root", [{"image": {"path": str(filename)}}], 0)
            util.open_folder.assert_called_with("output-root")

    def test_root_choice_and_hidden_configuration(self):
        callback, _, util = self.callback(choice="Output Root")
        callback("output-root", [{"image": {"path": "result.png"}}], 0)
        util.open_folder.assert_called_once_with("output-root")
        callback, _, util = self.callback(hidden=True)
        callback("output-root", None, None)
        util.open_folder.assert_not_called()

    def output_panel(self, directory, tabname="review"):
        callback, namespace, util = self.callback()
        shared = namespace["shared"]
        shared.opts.gallery_height = 0
        shared.opts.video_player_auto = False
        shared.opts.video_player_loop = False
        shared.opts.outdir_samples = str(directory)
        shared.opts.outdir_save = str(directory)
        bindings = []
        prepare_event = load_function("modules/gradio_extensions.py", "_prepare_event_kwargs", {})

        def tool_button(value="", *, tooltip=None, **kwargs):
            del tooltip
            button = gr.Button(value=value, **kwargs)
            listener = button.click
            button.click = lambda *args, **event_kwargs: listener(*args, **prepare_event(event_kwargs))
            return button

        def param_binding(**kwargs):
            return SimpleNamespace(source_text_component=None, override_settings_component=None, **kwargs)

        panel_namespace = {
            "gr": gr,
            "shared": shared,
            "OutputPanel": SimpleNamespace,
            "ToolButton": tool_button,
            "folder_symbol": "folder",
            "open_folder": callback,
            "update_generation_info": lambda *args: args[:2],
            "save_files": lambda *args: args,
            "call_queue": SimpleNamespace(wrap_gradio_call_no_job=lambda fn: fn),
            "parameters_copypaste": SimpleNamespace(
                ParamBinding=param_binding,
                register_paste_params_button=bindings.append,
            ),
        }
        create_panel = load_function("modules/ui_common.py", "create_output_panel", panel_namespace)
        namespace["Image"] = Image
        image_from_gallery = load_function("modules/infotext_utils.py", "image_from_url_text", namespace)
        send_namespace = {
            "gr": gr,
            "ParamBinding": SimpleNamespace,
            "paste_fields": {},
            "image_from_url_text": image_from_gallery,
        }
        connect_send = load_function("modules/infotext_utils.py", "_connect_paste_params_buttons", send_namespace)
        with gr.Blocks(analytics_enabled=False) as app:
            panel = create_panel(tabname, str(directory))
            for binding in bindings:
                destination = gr.Image(type="pil")
                send_namespace["paste_fields"][binding.tabname] = {
                    "fields": [],
                    "init_img": destination,
                    "override_settings_component": None,
                }
                connect_send(binding)
            emitter = gr.Button()
            outputs = [panel.gallery, panel.saved_paths] if tabname == "extras" else [panel.gallery]
            output_event = emitter.click(lambda: [], outputs=outputs)
        shared.demo = app
        app.has_launched = True
        app.allowed_paths = [str(directory)]
        return app, panel, namespace, util, output_event["id"]

    def gradio_stage(self, app, method, *args):
        async def execute():
            token = LocalContext.blocks.set(app)
            try:
                return await method(*args)
            finally:
                LocalContext.blocks.reset(token)

        return asyncio.run(execute())

    def test_actual_folder_event_receives_raw_gallery_data_without_pil_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "saved-subfolder" / "result.png"
            filename.parent.mkdir()
            Image.new("RGB", (2, 2), (10, 20, 30)).save(filename)
            app, panel, namespace, util, output_event = self.output_panel(directory)
            ui_tempdir.register_tmp_file(app, filename)
            output = self.gradio_stage(app, app.postprocess_data, app.fns[output_event], [str(filename)], None)[0]
            folder_event = next(
                fn for fn in app.fns.values() if len(fn.inputs) in (2, 3) and fn.inputs[0] is panel.gallery
            )
            self.assertFalse(folder_event.preprocess)
            with patch.object(panel.gallery, "preprocess", side_effect=AssertionError("folder must not decode images")):
                metadata = json.dumps({"saved_paths": [str(filename)]})
                inputs = self.gradio_stage(app, app.preprocess_data, folder_event, [output, 0, metadata], None)

            self.assertIsInstance(inputs[0][0], dict)
            self.assertEqual(inputs[0][0]["image"]["path"], output[0]["image"]["path"])
            self.assertEqual(inputs[1], 0)
            self.assertEqual(panel.gallery.type, "pil")
            with patch.object(ui_tempdir, "shared", namespace["shared"]):
                folder_event.fn(*inputs)
            util.open_folder.assert_called_once_with(str(filename.parent))
            filename.unlink()
            with patch.object(ui_tempdir, "shared", namespace["shared"]):
                folder_event.fn(*inputs)
            util.open_folder.assert_called_with(str(directory))
            namespace["shared"].opts.open_dir_button_choice = "Subdirectory (or temp dir)"
            with patch.object(ui_tempdir, "shared", namespace["shared"]):
                folder_event.fn(*inputs)
            util.open_folder.assert_called_with(os.path.dirname(output[0]["image"]["path"]))

    def test_actual_save_and_send_events_keep_gallery_pil_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "result.png"
            Image.new("RGB", (2, 2), (10, 20, 30)).save(filename)
            app, panel, _, _, output_event = self.output_panel(directory)
            output = self.gradio_stage(app, app.postprocess_data, app.fns[output_event], [str(filename)], None)[0]
            save_events = [fn for fn in app.fns.values() if len(fn.inputs) == 4 and fn.inputs[1] is panel.gallery]
            send_events = [fn for fn in app.fns.values() if fn.inputs == [panel.gallery] and fn.outputs]
            self.assertEqual(len(save_events), 2)
            self.assertEqual(len(send_events), 3)

            for event in save_events:
                self.assertTrue(event.preprocess)
                inputs = self.gradio_stage(app, app.preprocess_data, event, ["{}", output, False, 0], None)
                image = inputs[1][0][0]
                try:
                    self.assertIsInstance(image, Image.Image)
                    self.assertEqual(image.getpixel((0, 0)), (10, 20, 30))
                finally:
                    image.close()

            for event in send_events:
                self.assertTrue(event.preprocess)
                inputs = self.gradio_stage(app, app.preprocess_data, event, [output], None)
                image = inputs[0][0][0]
                try:
                    self.assertIsInstance(image, Image.Image)
                    self.assertIs(event.fn(*inputs), image)
                    self.assertEqual(image.getpixel((0, 0)), (10, 20, 30))
                finally:
                    image.close()

    def test_saved_metadata_requires_registered_existing_file_and_valid_row(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "sub" / "result.png"
            filename.parent.mkdir()
            Image.new("RGB", (2, 2)).save(filename)
            callback, namespace, util = self.callback()
            payload = [{"image": {"path": "missing-cache.png"}}]
            metadata = json.dumps({"saved_paths": [str(filename)]})
            callback("output-root", payload, 0, metadata)
            util.open_folder.assert_called_with("output-root")
            ui_tempdir.register_tmp_file(namespace["shared"].demo, filename)
            with patch.object(ui_tempdir, "shared", namespace["shared"]):
                for index in (-1, 1, None):
                    callback("output-root", payload, index, metadata)
                    util.open_folder.assert_called_with("output-root")
                callback("output-root", payload, 0, metadata)
                util.open_folder.assert_called_with(str(filename.parent))
                namespace["shared"].opts.open_dir_button_choice = "Output Root"
                callback("output-root", payload, 0, metadata)
                util.open_folder.assert_called_with("output-root")

    def test_extras_state_keeps_saved_path_through_official_cache_and_sessions(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "extras-subfolder" / "result.png"
            filename.parent.mkdir()
            Image.new("RGB", (2, 2)).save(filename)
            app, panel, namespace, util, output_event = self.output_panel(directory, "extras")
            ui_tempdir.register_tmp_file(app, filename)
            state = SessionState(app)
            output = self.gradio_stage(
                app, app.postprocess_data, app.fns[output_event], [[str(filename)], [str(filename)]], state
            )
            self.assertNotEqual(output[0][0]["image"]["path"], str(filename))
            folder_event = next(
                fn
                for fn in app.fns.values()
                if len(fn.inputs) == 3 and fn.inputs[0] is panel.gallery and fn.inputs[2] is panel.saved_paths
            )
            inputs = self.gradio_stage(app, app.preprocess_data, folder_event, [output[0], 0, None], state)
            self.assertEqual(inputs[2], [str(filename)])
            with patch.object(ui_tempdir, "shared", namespace["shared"]):
                folder_event.fn(*inputs)
            util.open_folder.assert_called_once_with(str(filename.parent))

            other_state = SessionState(app)
            inputs = self.gradio_stage(app, app.preprocess_data, folder_event, [output[0], 0, None], other_state)
            self.assertEqual(inputs[2], [])
            with patch.object(ui_tempdir, "shared", namespace["shared"]):
                folder_event.fn(*inputs)
            util.open_folder.assert_called_with(str(directory))

    def test_extras_wrapper_preserves_existing_outputs_and_adds_row_paths(self):
        wrapper = load_function(
            "modules/ui_common.py", "wrap_gradio_gallery_paths", {"wraps": wraps, "ui_tempdir": ui_tempdir}
        )
        image = Image.new("RGB", (2, 2))
        image.already_saved_as = "saved-result.png"
        original = ([image, (Image.new("RGB", (1, 1)), "caption")], "info", "log")
        result = wrapper(lambda: original)()
        self.assertEqual(result[:3], original)
        self.assertEqual(result[3], ["saved-result.png", None])

    def test_processed_metadata_adds_gallery_row_paths_only_when_ui_requests_them(self):
        class Metadata(SimpleNamespace):
            def __getattr__(self, _name):
                return None

        image = Image.new("RGB", (2, 2))
        image.already_saved_as = "first.png"
        extra = Image.new("RGB", (2, 2))
        extra.already_saved_as = "extra.png"
        record = Metadata(
            all_prompts=["portrait"],
            all_negative_prompts=[""],
            images=[image, Image.new("RGB", (2, 2))],
            extra_images=[extra],
        )
        serialize = load_function("modules/processing.py", "js", {"json": json, "ui_tempdir": ui_tempdir})

        self.assertNotIn("saved_paths", json.loads(serialize(record)))
        self.assertEqual(
            json.loads(serialize(record, include_saved_paths=True))["saved_paths"],
            ["first.png", None, "extra.png"],
        )

    def test_hires_metadata_keeps_old_row_paths_for_insert_and_replace(self):
        for insert in (False, True):
            with self.subTest(insert=insert):
                old_images = [Image.new("RGB", (2, 2)) for _ in range(3)]
                for image in old_images:
                    image.filename = "cache.png"
                hires_images = [Image.new("RGB", (4, 4)), Image.new("RGB", (4, 4))]
                for index, image in enumerate(hires_images):
                    image.already_saved_as = f"hires-{index}.png"
                processed = SimpleNamespace(images=hires_images, infotexts=["hr0", "hr1"], comments="")
                processing = SimpleNamespace(
                    script_args=(),
                    override_settings={},
                    extra_generation_params={},
                    close=Mock(),
                )
                namespace = {
                    "gr": gr,
                    "json": json,
                    "closing": closing,
                    "ui_tempdir": ui_tempdir,
                    "txt2img_create_processing": Mock(return_value=processing),
                    "infotext_utils": SimpleNamespace(image_from_url_text=lambda image: image[0]),
                    "opts": SimpleNamespace(txt2img_upscale_single_batch=False, txt2img_upscale_same_seed=False),
                    "modules": SimpleNamespace(
                        scripts=SimpleNamespace(scripts_txt2img=SimpleNamespace(run=Mock(return_value=processed)))
                    ),
                    "shared": SimpleNamespace(
                        opts=SimpleNamespace(hires_button_gallery_insert=insert),
                        total_tqdm=SimpleNamespace(clear=Mock()),
                    ),
                    "plaintext_to_html": lambda value, **_kwargs: value,
                }
                upscale = load_function("modules/txt2img.py", "txt2img_upscale_function", namespace)
                metadata = json.dumps({"infotexts": ["old0", "old1"], "saved_paths": ["old0.png", "old1.png", None]})
                result = upscale("task", None, [(image, None) for image in old_images], 1, metadata, *range(9))
                expected = ["old0.png", *(["old1.png"] if insert else []), "hires-0.png", "hires-1.png", None]
                self.assertEqual(json.loads(result[1])["saved_paths"], expected)
                self.assertEqual(len(result[0]["value"]), len(expected))

    def test_txt2img_hidden_paths_match_displayed_rows_and_stay_out_of_stdout(self):
        class Metadata(SimpleNamespace):
            def __getattr__(self, _name):
                return None

        serialize = load_function("modules/processing.py", "js", {"json": json, "ui_tempdir": ui_tempdir})
        for hide_images in (False, True):
            with self.subTest(hide_images=hide_images):
                image = Image.new("RGB", (2, 2))
                image.already_saved_as = "personal-output.png"
                extra = Image.new("RGB", (2, 2))
                extra.already_saved_as = "personal-extra.png"
                processed = Metadata(
                    images=[image],
                    extra_images=[extra],
                    all_prompts=["portrait"],
                    all_negative_prompts=[""],
                    comments="",
                )
                processed.js = lambda record=processed, **kwargs: serialize(record, **kwargs)
                processing = SimpleNamespace(script_args=(), close=Mock())
                printer = Mock()
                namespace = {
                    "gr": gr,
                    "closing": closing,
                    "txt2img_create_processing": Mock(return_value=processing),
                    "modules": SimpleNamespace(
                        scripts=SimpleNamespace(scripts_txt2img=SimpleNamespace(run=Mock(return_value=processed)))
                    ),
                    "shared": SimpleNamespace(total_tqdm=SimpleNamespace(clear=Mock())),
                    "opts": SimpleNamespace(samples_log_stdout=True, do_not_show_images=hide_images),
                    "print": printer,
                    "plaintext_to_html": lambda value, **_kwargs: value,
                }
                generate = load_function("modules/txt2img.py", "txt2img_function", namespace)
                result = generate("task", None)
                paths = json.loads(result[2])["saved_paths"]
                expected = ["personal-extra.png"] if hide_images else ["personal-output.png", "personal-extra.png"]
                self.assertEqual(paths, expected)
                self.assertEqual(len(paths), len(result[0]["value"]))
                self.assertNotIn("saved_paths", json.loads(printer.call_args.args[0]))


if __name__ == "__main__":
    unittest.main()
