"""The common image surface keeps the original model events and their state."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gradio as gr


def layout_ids(node):
    yield node["id"]
    for child in node.get("children", []):
        yield from layout_ids(child)


class ImageWorkspaceTests(unittest.TestCase):
    def build(self, editable=True):
        from modules_forge.image_workspace import ImageWorkspace

        with gr.Blocks() as tab:
            with gr.Row():
                with gr.Column():
                    checkpoint = gr.Dropdown(["base"], value="base", label="チェックポイント")
                    prompt = gr.Textbox(value="draft", label="Prompt")
                    negative = gr.Textbox(value="blur", label="Negative Prompt")
                    seed = gr.Number(value=123, label="Seed")
                    steps = gr.Number(value=20, label="Steps")
                    with gr.Group(visible="hidden") as refs:
                        image = gr.Image(label="入力画像")
                    with gr.Accordion("LoRA", open=False) as lora_panel:
                        loras = gr.Dropdown(["style"], value=["style"], multiselect=True)
                    run = gr.Button("Generate")
                    stop = gr.Button("Interrupt", interactive=False)
                with gr.Column():
                    output = gr.Image(label="生成結果")
            state = gr.State({"job": "owned"})
            run.click(lambda text, job: (text, job), [prompt, state], [prompt, state], api_visibility="private")
            original_fn = next(iter(tab.fns.values()))
            workspace = ImageWorkspace(
                tab,
                "test-image",
                editable=editable,
                models=[checkpoint],
                prompts=[prompt, negative],
                actions=[run, stop],
                settings=[steps, seed],
                references=[refs],
                outputs=[output],
                panels={"LoRA": [lora_panel]},
                badges={"LoRA": loras},
            )
        self.addCleanup(tab.close)
        return tab, workspace, original_fn, (prompt, negative, seed, image, output, state)

    def test_every_model_uses_one_pair_of_modes_and_the_same_slots(self):
        tab, workspace, _, _ = self.build()
        config = tab.get_config_file()
        ids = {x["props"].get("elem_id") for x in config["components"]}
        for suffix in (
            "model",
            "modes",
            "tools",
            "toprow",
            "prompts",
            "actions",
            "workspace",
            "settings",
            "references",
            "output",
        ):
            self.assertIn(f"test-image-{suffix}", ids)
        tabs = [x for x in config["components"] if x["type"] == "tabitem"]
        self.assertEqual([x["props"]["label"] for x in tabs], ["txt2img", "img2img"])
        self.assertTrue(tabs[1]["props"]["interactive"])
        self.assertEqual(workspace.mode.value, "txt2img")
        self.assertIs(workspace.settings.children[0], workspace.references, "img2img input precedes sampling controls")

    def test_rearrangement_preserves_all_event_inputs_outputs_and_values(self):
        tab, _, original_fn, components = self.build()
        config = tab.get_config_file()
        mounted = list(layout_ids(config["layout"]))
        self.assertEqual(len(mounted), len(set(mounted)))
        self.assertIs(tab.fns[original_fn._id], original_fn)
        self.assertEqual([c.value for c in components[:3]], ["draft", "blur", 123])
        for event in config["dependencies"]:
            for identifier in [*event["inputs"], *event["outputs"]]:
                self.assertIn(identifier, mounted, "Event components must stay mounted, including hidden state")

    def test_mode_changes_visibility_without_erasing_the_inputs(self):
        _, workspace, _, components = self.build()
        update = workspace.select_mode("img2img")
        self.assertEqual(update[0], "img2img")
        self.assertEqual(update[1]["selected"], "img2img")
        self.assertIs(update[2]["visible"], True)
        update = workspace.select_mode("txt2img")
        self.assertEqual(update[2]["visible"], "hidden")
        self.assertEqual([c.value for c in components[:3]], ["draft", "blur", 123])

    def test_unsupported_img2img_is_disabled_and_cannot_be_selected(self):
        tab, workspace, _, _ = self.build(editable=False)
        item = next(
            c
            for c in tab.get_config_file()["components"]
            if c["type"] == "tabitem" and c["props"]["label"] == "img2img"
        )
        self.assertFalse(item["props"]["interactive"])
        with self.assertRaisesRegex(ValueError, "img2img"):
            workspace.select_mode("img2img")

    def test_panel_visibility_and_enabled_badge_are_independent(self):
        _, workspace, _, _ = self.build()
        panel = workspace.panels["LoRA"]
        self.assertEqual(panel.group.visible, "hidden")
        self.assertEqual(panel.button.value, "LoRA · 1")
        opened = panel.toggle(False)
        closed = panel.toggle(True)
        self.assertEqual(opened, (True, gr.update(visible=True)))
        self.assertEqual(closed, (False, gr.update(visible="hidden")))
        self.assertEqual(panel.button.value, "LoRA · 1")

    def test_mode_echo_is_ignored_and_tab_selection_does_not_update_itself(self):
        _, workspace, _, _ = self.build()
        self.assertEqual(workspace.sync_native_mode("edit", "img2img"), (gr.skip(),) * 3)
        result = workspace.tab_selected("img2img", "txt2img")
        self.assertEqual(result[0], "img2img")
        self.assertEqual(result[1], gr.skip())
        self.assertEqual(workspace.tab_selected("img2img", "img2img"), (gr.skip(),) * 3)


class QwenImageModeTests(unittest.TestCase):
    def test_txt2img_excludes_retained_references_annotations_and_edit_masks(self):
        from tools.tests.test_qwen_image21_service import load_ui

        ui = load_ui()
        result = ui.canvas_inputs_for_mode("txt2img", ["retained.png"], "retained.png", True, True, "reference")
        self.assertEqual(result, ([], "", False, False, "custom"))

    def test_img2img_requires_a_reference_and_keeps_edit_controls(self):
        from tools.tests.test_qwen_image21_service import load_ui

        ui = load_ui()
        with self.assertRaisesRegex(ui.QwenImage21Error, "参照画像"):
            ui.canvas_inputs_for_mode("img2img", [], "", False, False, "custom")
        result = ui.canvas_inputs_for_mode("img2img", ["image.png"], "image.png", True, True, "reference")
        self.assertEqual(result, (["image.png"], "image.png", True, True, "reference"))

    def test_the_generation_callback_sends_the_selected_mode_to_the_service(self):
        from unittest.mock import patch

        from tools.tests.test_qwen_image21_service import load_ui

        ui = load_ui()
        with patch.object(ui.STUDIO, "start", return_value="owned-job") as start:
            result = ui.start_canvas(
                "a fox",
                ["retained.png"],
                "reference",
                False,
                "base_q4_k_m",
                "offload",
                "123",
                40,
                gr.Request(session_hash="workspace-test"),
                annotation_target="retained.png",
                preserve_unmasked=True,
                control_inpaint=True,
                control_kind="canny",
                control_image="guide.png",
                custom_width=832,
                custom_height=1280,
                image_mode="txt2img",
            )
        self.assertEqual(result[0], "owned-job")
        request = start.call_args.args[0]
        self.assertEqual(request.input_images, ())
        self.assertEqual((request.width, request.height), (832, 1280))
        self.assertEqual(request.annotation_reference, -1)
        self.assertFalse(request.preserve_unmasked)
        self.assertFalse(request.control_inpaint)
        self.assertEqual(request.edit_mask_path, "")
        self.assertEqual((request.control_kind, request.control_image), ("canny", "guide.png"))

    def test_missing_img2img_input_does_not_start_a_job(self):
        from unittest.mock import patch

        from tools.tests.test_qwen_image21_service import load_ui

        ui = load_ui()
        with patch.object(ui.STUDIO, "start") as start:
            result = ui.start_canvas(
                "a fox",
                [],
                "1024x1024",
                False,
                "base_q4_k_m",
                "offload",
                "123",
                40,
                gr.Request(session_hash="workspace-test"),
                image_mode="img2img",
            )
        start.assert_not_called()
        self.assertIn("参照画像", result[1])


class ForgePanelTests(unittest.TestCase):
    def test_toolbar_accepts_forges_wrapped_change_events(self):
        from types import SimpleNamespace

        from modules_forge.image_workspace import forge_toolbar

        with gr.Blocks() as tab:
            with gr.Row() as top:
                gr.Textbox()
            with gr.Group() as group:
                enabled = gr.Checkbox(value=False, label="Enable")
            original = enabled.change

            class EventWrapper:
                def __call__(self, *args, **kwargs):
                    return original(*args, **kwargs)

            enabled.change = EventWrapper()
            forge_toolbar(tab, "txt2img", [SimpleNamespace(group=group, title=lambda: "ControlNet")], before=top)
        self.addCleanup(tab.close)

    def test_toolbar_is_above_the_forge_prompt_and_keeps_native_enabled_values(self):
        from types import SimpleNamespace

        from modules_forge.image_workspace import forge_toolbar

        with gr.Blocks() as tab:
            with gr.Row() as top:
                prompt = gr.Textbox(value="kept")
            with gr.Group() as group:
                with gr.Accordion("ControlNet", open=False):
                    enabled = gr.Checkbox(value=True, label="Enable")
            script = SimpleNamespace(group=group, title=lambda: "ControlNet")
            panels = forge_toolbar(tab, "txt2img", [script], before=top)
        self.addCleanup(tab.close)
        self.assertEqual(tab.children[0].elem_id, "txt2img-image-toolbar")
        self.assertEqual(enabled.value, True)
        self.assertEqual(prompt.value, "kept")
        self.assertEqual(panels["ControlNet"].button.value, "ControlNet · ON")
        self.assertFalse(any(x["type"] == "tabs" for x in tab.get_config_file()["components"]))


def studio_ui(path):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("workspace_" + Path(path).stem, root / path)
    module = importlib.util.module_from_spec(spec)
    paths = SimpleNamespace(data_path=str(root), script_path=str(root))
    with patch.dict(
        sys.modules,
        {
            "modules.script_callbacks": SimpleNamespace(on_ui_tabs=Mock()),
            "modules.paths": paths,
        },
    ):
        spec.loader.exec_module(module)
    return module


class ReferenceAdapterTests(unittest.TestCase):
    def test_sensenova_excludes_hidden_references_in_txt2img(self):
        from tools.tests.test_sensenova_u15_studio import load_studio_module

        ui = load_studio_module()
        with patch.object(ui, "normalize_gallery_images", side_effect=AssertionError("must not read retained images")):
            request = ui._request_from_ui(
                ui.MODE_TEXT,
                "a fox",
                ["retained.png"],
                ui.DEFAULT_MODEL_ID,
                ui.QUANT_INT8_CONVROT,
                "",
                "",
                "2048x2048",
                str(512 * 512),
                ui.PROFILE_QUALITY,
                50,
                4,
                1,
                3,
                123,
                "low",
                "sdpa",
                "bfloat16",
                should_validate=False,
            )
        self.assertEqual(request.input_images, ())

    def test_h3_excludes_hidden_references_in_txt2img(self):
        ui = studio_ui("extensions-builtin/minimax-h3-studio/scripts/minimax_h3_image_studio.py")
        with patch.object(ui.images, "_bridge", side_effect=AssertionError("must not read retained images")):
            request = ui._request(
                "text",
                "a fox",
                ["retained.png"],
                768,
                768,
                20,
                123,
                "simple",
                0,
                False,
                False,
                1,
                True,
                0,
                999,
                1,
                False,
                False,
                "",
            )
        self.assertEqual(request.reference_images, ())


class StudioSurfaceTests(unittest.TestCase):
    def test_all_image_studios_use_one_mode_pair_and_keep_job_status_visible(self):
        models = [
            ("iris-studio/scripts/iris_studio.py", "on_ui_tabs", True, "iris-status"),
            ("qwen-image21-studio/scripts/qwen_image21_studio.py", "on_ui_tabs", True, "qwen21-status"),
            ("sensenova-u15-studio/scripts/sensenova_u15_studio.py", "_build_ui", True, "sn-progress"),
            ("ming-image-studio/scripts/ming_image_studio.py", "on_ui_tabs", False, "ming-status"),
            ("nanosaur2-studio/scripts/nanosaur2_studio.py", "on_ui_tabs", False, "nanosaur2-status"),
            ("minimax-h3-studio/scripts/minimax_h3_image_studio.py", "on_ui_tabs", True, "h3-image-status"),
        ]
        from modules_forge import local_assets

        with patch.object(local_assets, "selection", return_value={}):
            for path, factory, editable, status_id in models:
                with self.subTest(model=path):
                    ui = studio_ui("extensions-builtin/" + path)
                    if hasattr(ui, "_initial_runtime"):
                        ui._initial_runtime = lambda: ""
                    interfaces = getattr(ui, factory)()
                    for blocks, _, _ in interfaces:
                        self.addCleanup(blocks.close)
                    blocks = interfaces[0][0]
                    config = blocks.get_config_file()
                    mounted = set().union(
                        *(set(layout_ids(item[0].get_config_file()["layout"])) for item in interfaces)
                    )
                    tabs = [item for item in config["components"] if item["type"] == "tabitem"]
                    self.assertEqual([item["props"]["label"] for item in tabs], ["txt2img", "img2img"])
                    self.assertEqual(tabs[1]["props"]["interactive"], editable)
                    status = next(item for item in blocks.blocks.values() if item.elem_id == status_id)
                    ancestors = list()
                    while status is not None:
                        ancestors.extend(getattr(status, "elem_classes", None) or [])
                        status = getattr(status, "parent", None)
                    self.assertIn("aikimi-image-actions", ancestors)
                    self.assertNotIn("aikimi-image-panel", ancestors)
                    for dependency in config["dependencies"]:
                        for identifier in [*dependency["inputs"], *dependency["outputs"]]:
                            self.assertIn(identifier, mounted)


if __name__ == "__main__":
    unittest.main()
