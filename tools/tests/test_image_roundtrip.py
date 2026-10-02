"""Real Pillow round trips through Forge image and API boundary functions."""

import ast
import base64
import io
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import piexif
import piexif.helper
from fastapi import HTTPException
from PIL import Image, ImageOps, PngImagePlugin

from modules.aikimi_security.url_fetch import SafeFetchError, validate_decoded_image

ROOT = Path(__file__).resolve().parents[2]


def load_boundary_functions(path, names, namespace):
    source = ROOT / path
    tree = ast.parse(source.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)
        )
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102
    return namespace


class ImageRoundTripTests(unittest.TestCase):
    def setUp(self):
        self.opts = SimpleNamespace(
            enable_pnginfo=True,
            jpeg_quality=90,
            webp_lossless=True,
            samples_format="webp",
            api_enable_requests=False,
        )
        self.images = load_boundary_functions(
            "modules/images.py",
            {
                "save_image_with_geninfo",
                "read_info_from_image",
                "IGNORED_INFO_KEYS",
                "read",
                "fix_image",
                "fix_png_transparency",
            },
            {
                "Image": Image,
                "ImageOps": ImageOps,
                "PngImagePlugin": PngImagePlugin,
                "piexif": piexif,
                "json": json,
                "os": os,
                "struct": struct,
                "opts": self.opts,
                "errors": Mock(),
                "sd_samplers": SimpleNamespace(samplers_map={}),
            },
        )
        self.api = load_boundary_functions(
            "modules/api/api.py",
            {"encode_pil_to_base64", "decode_base64_to_image"},
            {
                "base64": base64,
                "io": io,
                "BytesIO": io.BytesIO,
                "PngImagePlugin": PngImagePlugin,
                "piexif": piexif,
                "opts": self.opts,
                "images": SimpleNamespace(read=self.images["read"]),
                "HTTPException": HTTPException,
                "SafeFetchError": SafeFetchError,
                "validate_decoded_image": validate_decoded_image,
                "fetch_remote_image": Mock(side_effect=AssertionError("network must not be used")),
            },
        )
        self.image = Image.new("RGBA", (3, 2))
        self.image.putdata([(10, 20, 30, alpha) for alpha in (0, 64, 128, 192, 255, 255)])
        self.image.info["parameters"] = "日本語のprompt\nSteps: 20"
        self.addCleanup(self.image.close)

    def assert_alpha_and_parameters(self, image):
        rgba = image.convert("RGBA")
        self.addCleanup(rgba.close)
        self.assertEqual(list(rgba.getchannel("A").getdata()), [0, 64, 128, 192, 255, 255])
        geninfo, _ = self.images["read_info_from_image"](image)
        self.assertEqual(geninfo, self.image.info["parameters"])

    def test_saved_webp_keeps_alpha_and_generation_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.webp"
            for filename in (str(path), path):
                with self.subTest(path_type=type(filename).__name__):
                    self.images["save_image_with_geninfo"](self.image, self.image.info["parameters"], filename)
                    with Image.open(path) as reopened:
                        self.assert_alpha_and_parameters(reopened)

    def test_api_webp_keeps_alpha_and_generation_metadata(self):
        encoded = self.api["encode_pil_to_base64"](self.image)
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as reopened:
            self.assert_alpha_and_parameters(reopened)

    def test_api_png_keeps_alpha_and_generation_metadata(self):
        self.opts.samples_format = "png"
        encoded = self.api["encode_pil_to_base64"](self.image)
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as reopened:
            self.assert_alpha_and_parameters(reopened)

    def test_api_palette_webp_keeps_palette_transparency(self):
        image = Image.new("P", (3, 1))
        self.addCleanup(image.close)
        image.putpalette([255, 0, 0, 0, 255, 0, 0, 0, 255])
        image.putdata([0, 1, 2])
        image.info["transparency"] = bytes([0, 128, 255])

        encoded = self.api["encode_pil_to_base64"](image)

        with Image.open(io.BytesIO(base64.b64decode(encoded))) as reopened:
            self.assertEqual(list(reopened.getchannel("A").getdata()), [0, 128, 255])

    def test_jpeg_encoders_keep_rgb_conversion_and_generation_metadata(self):
        self.opts.samples_format = "jpeg"
        encoded = self.api["encode_pil_to_base64"](self.image)
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as reopened:
            self.assertEqual(reopened.mode, "RGB")
            self.assertEqual(self.images["read_info_from_image"](reopened)[0], self.image.info["parameters"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.JPG"
            self.images["save_image_with_geninfo"](self.image, self.image.info["parameters"], path)
            with Image.open(path) as reopened:
                self.assertEqual(reopened.mode, "RGB")
                self.assertEqual(self.images["read_info_from_image"](reopened)[0], self.image.info["parameters"])

    def test_png_filename_extension_is_case_insensitive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.PNG"
            self.images["save_image_with_geninfo"](self.image, self.image.info["parameters"], path)
            with Image.open(path) as reopened:
                self.assertEqual(reopened.format, "PNG")
                self.assert_alpha_and_parameters(reopened)

    def test_truncated_exif_does_not_discard_valid_png_parameters(self):
        output = io.BytesIO()
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("parameters", self.image.info["parameters"])
        self.image.save(output, format="PNG", pnginfo=pnginfo, exif=b"Exif\0\0II*\0")

        with Image.open(io.BytesIO(output.getvalue())) as reopened:
            geninfo, items = self.images["read_info_from_image"](reopened)

        self.assertEqual(geninfo, self.image.info["parameters"])
        self.assertNotIn("exif", items)

    def test_valid_exif_comment_keeps_precedence_over_png_parameters(self):
        self.image.info["exif"] = piexif.dump(
            {"Exif": {piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump("EXIF prompt", encoding="unicode")}}
        )

        geninfo, _ = self.images["read_info_from_image"](self.image)

        self.assertEqual(geninfo, "EXIF prompt")

    def test_api_decodes_data_uri_with_additional_image_parameters(self):
        output = io.BytesIO()
        self.image.save(output, format="PNG")
        encoded = base64.b64encode(output.getvalue()).decode("ascii")

        for payload in (
            encoded,
            f"data:image/png;base64,{encoded}",
            f"data:image/png;name=result.png;base64,{encoded}",
        ):
            with self.subTest(payload=payload[:45]):
                image = self.api["decode_base64_to_image"](payload)
                self.addCleanup(image.close)
                self.assertEqual(image.tobytes(), self.image.tobytes())

    def test_api_malformed_data_uri_uses_the_existing_invalid_image_error(self):
        with self.assertRaises(HTTPException) as caught:
            self.api["decode_base64_to_image"]("data:image/png;base64")

        self.assertEqual(caught.exception.detail, "Invalid encoded image")


if __name__ == "__main__":
    unittest.main()
