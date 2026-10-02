"""Batch PNG metadata parsing must not leave temporary global preferences behind."""

from __future__ import annotations

import ast
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[2]


class BatchMetadataSettingsTests(unittest.TestCase):
    def setUp(self):
        source = ROOT / "modules/img2img.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        process = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "process_batch")
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.options = SimpleNamespace(
            infotext_styles="Apply if any",
            img2img_batch_use_original_name=False,
            img2img_batch_show_results_limit=10,
        )
        self.generation_settings = []
        self.p = SimpleNamespace(
            prompt="base",
            negative_prompt="negative",
            seed=10,
            cfg_scale=7.0,
            sampler_name="Euler",
            steps=20,
            override_settings={},
            n_iter=1,
            batch_size=1,
        )

        def generate(_p):
            self.generation_settings.append((self.p.prompt, self.options.infotext_styles))
            return SimpleNamespace(images=[Image.new("RGB", (64, 64))], infotexts=["fixture"])

        self.namespace = {
            "print": lambda *_args, **_kwargs: None,
            "os": os,
            "Path": Path,
            "processing": SimpleNamespace(fix_seed=lambda _p: None),
            "shared": SimpleNamespace(
                opts=self.options,
                walk_files=lambda *_args, **_kwargs: [str(self.root / "input.png")],
            ),
            "state": SimpleNamespace(skipped=False, interrupted=False, stopping_generation=False),
            "images": SimpleNamespace(
                read=lambda _path: Image.new("RGB", (64, 64)),
                read_info_from_image=lambda _image: ("metadata", {}),
            ),
            "ImageOps": ImageOps,
            "UnidentifiedImageError": UnidentifiedImageError,
            "_STEP": 8,
            "get_closet_checkpoint_match": lambda _name: None,
            "parse_generation_parameters": self.parse_success,
            "modules": SimpleNamespace(
                scripts=SimpleNamespace(scripts_img2img=SimpleNamespace(run=lambda *_args: None))
            ),
            "opts": self.options,
            "process_images": generate,
        }
        exec(  # noqa: S102 - execute only the extracted repository batch function in isolation
            compile(ast.Module(body=[process], type_ignores=[]), str(source), "exec"), self.namespace
        )
        self.process_batch = self.namespace["process_batch"]

    def parse_success(self, _info):
        self.assertEqual(self.options.infotext_styles, "Ignore")
        return {"Prompt": "source"}

    def run_batch(self):
        return self.process_batch(self.p, str(self.root), "", "", [], use_png_info=True, png_info_props=["Prompt"])

    def test_success_restores_style_preference_before_generation(self):
        self.run_batch()
        self.assertEqual(self.options.infotext_styles, "Apply if any")
        self.assertEqual(self.generation_settings, [("base source", "Apply if any")])

    def test_failed_metadata_restores_preference_and_uses_default_prompt(self):
        def parse_failure(_info):
            self.assertEqual(self.options.infotext_styles, "Ignore")
            raise ValueError("invalid PNG metadata")

        self.namespace["parse_generation_parameters"] = parse_failure
        self.run_batch()
        self.assertEqual(self.options.infotext_styles, "Apply if any")
        self.assertEqual(self.generation_settings, [("base", "Apply if any")])

    def test_interruption_restores_preference_and_propagates(self):
        error = KeyboardInterrupt("cancelled")

        def interrupt(_info):
            self.assertEqual(self.options.infotext_styles, "Ignore")
            raise error

        self.namespace["parse_generation_parameters"] = interrupt
        with self.assertRaises(KeyboardInterrupt) as caught:
            self.run_batch()
        self.assertIs(caught.exception, error)
        self.assertEqual(self.options.infotext_styles, "Apply if any")
        self.assertEqual(self.generation_settings, [])


if __name__ == "__main__":
    unittest.main()
