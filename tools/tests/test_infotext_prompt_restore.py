"""Infotext and Seed restoration use real parsers without model generation."""

import doctest
import inspect
import json
import os
import re
import unittest
from types import SimpleNamespace
from typing import Any, Optional, Union, get_args, get_origin
from unittest.mock import Mock, patch

import gradio as gr
from inflection import underscore
from pydantic import BaseModel, ConfigDict, Field, create_model

from backend.text_processing import parsing
from backend.text_processing.emphasis import uses_emphasis
from modules import prompt_parser
from tools.tests.test_api_extras_boundaries import load_classes
from tools.tests.test_gpu_ownership import load_function
from tools.tests.test_image_roundtrip import load_boundary_functions


class RestoreFixture:
    def __init__(self):
        self.opts = SimpleNamespace(
            infotext_skip_pasting=[],
            infotext_styles="Ignore",
            use_old_hires_fix_width_height=False,
            sd_model_checkpoint="fixture.safetensors",
            forge_additional_modules=[],
            forge_unet_storage_dtype="Automatic",
            CLIP_stop_at_last_layers=1,
            eta_noise_seed_delta=0,
            add_model_name_to_info=False,
            add_model_hash_to_info=False,
            inpainting_mask_weight=1,
            scaling_factor=1,
            save_init_img=False,
            randn_source="CPU",
            tiling=False,
            add_version_to_infotext=False,
            add_user_name_to_info=False,
            data_labels={},
        )
        self.shared = SimpleNamespace(opts=self.opts)
        namespace = {
            "json": json,
            "os": os,
            "re": re,
            "Any": Any,
            "shared": self.shared,
            "uses_emphasis": uses_emphasis,
            "main_entry": Mock(),
        }
        load_boundary_functions(
            "modules/infotext_utils.py",
            {
                "re_param_code",
                "re_param",
                "re_imagesize",
                "INFOTEXT_TO_SETTING",
                "quote",
                "unquote",
                "restore_old_hires_fix_params",
                "_extract_styles",
                "_populate_defaults",
                "_resolve_additional_module_parameters",
                "parse_generation_parameters",
                "get_override_settings",
            },
            namespace,
        )
        self.info = SimpleNamespace(**{key: value for key, value in namespace.items() if callable(value)})
        self.create = load_function(
            "modules/processing.py",
            "create_infotext",
            {
                "opts": self.opts,
                "shared": self.shared,
                "_overridden_modules": None,
                "infotext_utils": self.info,
                "errors": Mock(),
            },
        )

    def infotext(self, prompt="landscape", negative="blur", seed=123, variation_strength=0, extras=None):
        p = SimpleNamespace(
            batch_size=1,
            all_negative_prompts=[negative],
            all_seeds=[seed],
            all_subseeds=[456],
            get_token_merging_ratio=lambda **_kwargs: 0,
            steps=20,
            sampler_name="Euler",
            scheduler="Simple",
            cfg_scale=7,
            sd_model=SimpleNamespace(use_distilled_cfg_scale=False, use_shift=False, is_sd1=False),
            restore_faces=False,
            width=512,
            height=384,
            subseed_strength=variation_strength,
            seed_resize_from_w=768,
            seed_resize_from_h=512,
            extra_generation_params=extras or {},
            is_using_inpainting_conditioning=False,
        )
        return self.create(p, [prompt], [seed], [456])


class PromptAttentionTests(unittest.TestCase):
    def test_malformed_weights_remain_prompt_text_in_both_parsers(self):
        for weight in (".", "..", "1..2", "+.", "-..3"):
            text = f"(subject:{weight})"
            expected = [[f"subject:{weight}", 1.1]]
            for name, parser in (
                ("frontend", prompt_parser.parse_prompt_attention),
                ("backend", lambda text: parsing.parse_prompt_attention(text, "Original")),
            ):
                with self.subTest(text=text, parser=name):
                    self.assertEqual(parser(text), expected)
            self.assertEqual(parsing.parse_prompt_attention(text, "None"), [[text, 1.0]])

    def test_existing_valid_weight_syntax_stays_compatible(self):
        for weight in ("1", "1.", ".5", "-0.25", "+1.3", "0"):
            with self.subTest(weight=weight):
                text = f"before (subject:{weight}) after"
                expected = [["before ", 1.0], ["subject", float(weight)], [" after", 1.0]]
                if float(weight) == 1:
                    expected = [["before subject after", 1.0]]
                self.assertEqual(prompt_parser.parse_prompt_attention(text), expected)
                self.assertEqual(parsing.parse_prompt_attention(text, "Original"), expected)

    def test_attention_and_schedule_documented_examples(self):
        finder = doctest.DocTestFinder()
        runner = doctest.DocTestRunner()
        for function in (
            prompt_parser.parse_prompt_attention,
            prompt_parser.get_learned_conditioning_prompt_schedules,
        ):
            for test in finder.find(function, globs=vars(prompt_parser)):
                runner.run(test)
        results = runner.summarize()
        self.assertGreater(results.attempted, 20)
        self.assertEqual(results.failed, 0)


class InfotextRestoreTests(unittest.TestCase):
    def setUp(self):
        self.fixture = RestoreFixture()

    def test_negative_prompt_keeps_literal_marker_inside_its_text(self):
        negative = (
            'text reading "Negative prompt: omit this", blur\nNegative prompt: a literal second marker\nlast line'
        )
        text = self.fixture.infotext(negative=negative)
        params = self.fixture.info.parse_generation_parameters(text)
        self.assertEqual(params["Negative prompt"], negative)
        self.assertEqual(params["Seed"], "123")
        self.assertEqual((params["Size-1"], params["Size-2"]), ("512", "384"))

    def test_malformed_attention_does_not_block_saved_infotext_restore(self):
        for prompt, negative in (("(subject:1..2)", "blur"), ("landscape", "(blur:..)")):
            with self.subTest(prompt=prompt, negative=negative):
                text = self.fixture.infotext(prompt=prompt, negative=negative)
                params = self.fixture.info.parse_generation_parameters(text)
                self.assertEqual(params["Prompt"], prompt)
                self.assertEqual(params["Negative prompt"], negative)
                self.assertEqual(params["Seed"], "123")
                self.assertEqual(params["Emphasis"], "Original")

    def test_quoted_metadata_and_skip_fields_keep_existing_restore_behavior(self):
        hires_prompt = '日本語, path C:\\fixture: "quoted"\nsecond line'
        text = self.fixture.infotext(extras={"Hires prompt": hires_prompt})
        params = self.fixture.info.parse_generation_parameters(text, skip_fields=["Seed"])
        self.assertEqual(params["Hires prompt"], hires_prompt)
        self.assertEqual(params["Prompt"], "landscape")
        self.assertNotIn("Seed", params)
        self.assertEqual(params["Hires VAE/TE"], ["Use same choices"])


def seed_request_model():
    class ProcessingFixture:
        def __init__(
            self,
            seed: int = -1,
            subseed: int = -1,
            subseed_strength: float = 0,
            seed_resize_from_w: int = 0,
            seed_resize_from_h: int = 0,
            override_settings: dict | None = None,
        ):
            pass

    namespace = {
        "__name__": __name__,
        "inspect": inspect,
        "underscore": underscore,
        "Optional": Optional,
        "Any": Any,
        "BaseModel": BaseModel,
        "ConfigDict": ConfigDict,
        "Field": Field,
        "create_model": create_model,
    }
    load_boundary_functions("modules/api/models.py", {"API_NOT_ALLOWED"}, namespace)
    classes = load_classes("modules/api/models.py", {"ModelDef", "PydanticModelGenerator"}, namespace)
    return classes.PydanticModelGenerator(
        "SeedRestoreRequest", ProcessingFixture, [{"key": "infotext", "type": str, "default": ""}]
    ).generate_model()


class SeedRestoreTests(unittest.TestCase):
    def setUp(self):
        self.fixture = RestoreFixture()
        self.errors = Mock()
        namespace = {
            "gr": gr,
            "json": json,
            "infotext_utils": self.fixture.info,
            "errors": self.errors,
        }
        self.connect = load_function("modules/processing_scripts/seed.py", "connect_reuse_seed", namespace)

    def reuse_callback(self, is_subseed):
        button = Mock()
        self.connect(None, button, None, is_subseed)
        return button.click.call_args.kwargs["fn"]

    def test_variation_seed_reuse_falls_back_to_normal_seed_when_unused(self):
        callback = self.reuse_callback(is_subseed=True)
        info = json.dumps({"infotexts": [self.fixture.infotext(seed=123), self.fixture.infotext(seed=789)]})
        self.assertEqual(callback(info, 0)[0], 123)
        self.assertEqual(callback(info, 1)[0], 789)
        self.errors.report.assert_not_called()

    def test_used_variation_seed_and_normal_seed_reuse_stay_distinct(self):
        info = json.dumps({"infotexts": [self.fixture.infotext(variation_strength=0.5)]})
        self.assertEqual(self.reuse_callback(is_subseed=True)(info, 0)[0], 456)
        self.assertEqual(self.reuse_callback(is_subseed=False)(info, 0)[0], 123)
        self.errors.report.assert_not_called()

    def test_seed_ui_registration_restores_resize_axes_to_matching_api_fields(self):
        paste_field = load_classes("modules/infotext_utils.py", {"PasteField"}, {"gr": gr}).PasteField
        ui_namespace = {
            "gr": gr,
            "cmd_opts": SimpleNamespace(use_textbox_seed=False),
            "ToolButton": lambda value, tooltip=None, **kwargs: gr.Button(value, **kwargs),
            "PasteField": paste_field,
        }
        ui = load_function(
            "modules/processing_scripts/seed.py",
            "ui",
            ui_namespace,
        )
        ui_namespace["ui"] = SimpleNamespace(random_symbol="random", reuse_symbol="reuse")
        script = SimpleNamespace(elem_id=lambda name: f"test_{name}", tabname="txt2img", on_after_component=Mock())

        def no_event(_self, *_args, **_kwargs):
            return None

        with (
            patch.object(gr.Button, "click", no_event),
            patch.object(gr.Checkbox, "change", no_event),
            gr.Blocks(analytics_enabled=False),
        ):
            ui(script, False)
        self.fixture.info.paste_fields = {"txt2img": {"fields": script.infotext_fields}}
        apply = load_function(
            "modules/api/api.py",
            "apply_infotext",
            {
                "infotext_utils": self.fixture.info,
                "Union": Union,
                "get_args": get_args,
                "get_origin": get_origin,
            },
        )
        request = seed_request_model()(infotext=self.fixture.infotext())
        apply(None, request, "txt2img")
        self.assertEqual((request.seed_resize_from_w, request.seed_resize_from_h), (768, 512))
        self.assertEqual(request.seed, 123)


if __name__ == "__main__":
    unittest.main()
