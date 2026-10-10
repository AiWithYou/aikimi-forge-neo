"""Exercise real ImageStitch processing and PIL resize without loading Forge models."""
# ruff: noqa: S102 - execute only selected AST nodes from trusted repository source.

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


class _Tensor:
    def __init__(self, array):
        self.array = array

    def to(self, **_kwargs):
        return self

    def unsqueeze(self, axis):
        return _Tensor(np.expand_dims(self.array, axis))


def _load_script():
    source = ROOT / "extensions-builtin/sd_forge_image_stitch/scripts/image_stitch.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    script = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ImageStitch")
    image_source = ROOT / "modules/images.py"
    image_tree = ast.parse(image_source.read_text(encoding="utf-8"), filename=str(image_source))
    resize = next(node for node in image_tree.body if isinstance(node, ast.FunctionDef) and node.name == "resize_image")
    image_namespace = dict(
        Image=Image, LANCZOS=Image.Resampling.LANCZOS, opts=SimpleNamespace(upscaler_for_img2img="None")
    )
    exec(compile(ast.Module([resize], type_ignores=[]), str(image_source), "exec"), image_namespace)
    flags = SimpleNamespace(
        kontext=False, edit=False, klein=False, wan=False, anima=True, krea2=False, is_referencing=False
    )
    samples = Mock()
    namespace = dict(
        scripts=SimpleNamespace(Script=object),
        Image=Image,
        np=np,
        torch=SimpleNamespace(from_numpy=_Tensor),
        dynamic_args=flags,
        device="cpu",
        images=SimpleNamespace(
            resize_image=image_namespace["resize_image"], flatten=lambda image, _background: image.convert("RGB")
        ),
        sd_models=SimpleNamespace(model_data=SimpleNamespace(forge_loading_parameters={"model": "anima"})),
        opts=SimpleNamespace(img2img_background_color="white"),
        images_tensor_to_samples=samples,
        StableDiffusionProcessing=object,
        StableDiffusionProcessingTxt2Img=object,
        logger=Mock(),
    )
    exec(compile(ast.Module([script], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["ImageStitch"], flags, samples


class ImageStitchAnimaCanvasTests(unittest.TestCase):
    def setUp(self):
        script, self.flags, self.samples = _load_script()
        self.script = script()
        self.processing = SimpleNamespace(
            width=512, height=512, batch_size=1, clear_prompt_cache=Mock(), sd_model=Mock()
        )
        self.reference = Image.new("RGB", (384, 512), "green")
        self.reference.paste("red", (0, 0, 384, 32))
        self.reference.paste("blue", (0, 480, 384, 512))

    def process(self, max_dim=512):
        self.script.process(self.processing, True, [(self.reference, None)], max_dim)

    def test_portrait_is_center_cropped_to_square_before_vae_encoding(self):
        self.process()
        encoded = self.samples.call_args.args[0].array
        self.assertEqual(encoded.shape, (1, 3, 512, 512))
        np.testing.assert_allclose(encoded[0, :, 0, 256], [0, 128 / 255, 0], atol=1 / 255)
        np.testing.assert_allclose(encoded[0, :, -1, 256], [0, 128 / 255, 0], atol=1 / 255)
        self.assertFalse(self.flags.is_referencing)

    def test_anima_output_canvas_takes_priority_over_maximum_side_length(self):
        self.processing.width, self.processing.height = 768, 512
        self.process(max_dim=256)
        self.assertEqual(self.samples.call_args.args[0].array.shape, (1, 3, 512, 768))

    def test_same_reference_reencodes_when_output_width_or_height_changes(self):
        self.process()
        self.processing.width = 768
        self.process()
        self.processing.height = 640
        self.process()
        self.assertEqual(self.samples.call_count, 3)
        self.assertEqual(self.samples.call_args.args[0].array.shape, (1, 3, 640, 768))

    def test_maximum_side_length_change_reuses_anima_encoded_canvas(self):
        self.process(max_dim=512)
        self.process(max_dim=256)
        self.assertEqual(self.samples.call_count, 1)
        self.assertEqual(self.samples.call_args.args[0].array.shape, (1, 3, 512, 512))

    def test_unchanged_reference_and_canvas_reuse_encoded_latents(self):
        self.process()
        self.process()
        self.samples.assert_called_once()
        self.processing.clear_prompt_cache.assert_called_once()

    def test_other_models_keep_reference_aspect_and_maximum_side_length(self):
        self.flags.anima, self.flags.krea2 = False, True
        self.process(max_dim=256)
        self.assertEqual(self.samples.call_args.args[0].array.shape, (1, 3, 256, 192))


if __name__ == "__main__":
    unittest.main()
