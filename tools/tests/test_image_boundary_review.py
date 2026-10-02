"""Additional image modes and corrupt optional EXIF must preserve usable output."""

from __future__ import annotations

import base64
import io
import struct
import tempfile
import unittest
from pathlib import Path

import piexif
from PIL import Image, PngImagePlugin

from tools.tests import test_image_roundtrip as roundtrip


class ImageBoundaryReviewTests(unittest.TestCase):
    def setUp(self):
        self.fixture = roundtrip.ImageRoundTripTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()

    def transparent_image(self, mode):
        image = Image.new(mode, (3, 2), (80, 128) if mode == "LA" else 0)
        self.addCleanup(image.close)
        if mode == "P":
            image.putpalette([255, 0, 0] + [0] * 765)
            image.info["transparency"] = 0
        image.info["parameters"] = self.fixture.image.info["parameters"]
        return image

    def test_api_la_jpeg_keeps_conversion_and_unicode_parameters(self):
        self.fixture.opts.samples_format = "jpeg"
        encoded = self.fixture.api["encode_pil_to_base64"](self.transparent_image("LA"))
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as reopened:
            self.assertEqual(reopened.mode, "RGB")
            self.assertEqual(
                self.fixture.images["read_info_from_image"](reopened)[0], self.fixture.image.info["parameters"]
            )

    def test_saved_la_and_palette_jpeg_with_unicode_path_keep_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("LA", "P"):
                with self.subTest(mode=mode):
                    path = Path(directory) / f"画像-{mode}.JPEG"
                    self.fixture.images["save_image_with_geninfo"](
                        self.transparent_image(mode), self.fixture.image.info["parameters"], path
                    )
                    with Image.open(path) as reopened:
                        self.assertEqual(reopened.mode, "RGB")
                        self.assertEqual(
                            self.fixture.images["read_info_from_image"](reopened)[0],
                            self.fixture.image.info["parameters"],
                        )

    def test_wrong_exif_comment_type_keeps_valid_png_parameters(self):
        exif = bytearray(piexif.dump({"Exif": {piexif.ExifIFD.UserComment: b"ASCII\0\0\0fixture"}}))
        tag = exif.index(struct.pack(">HH", piexif.ExifIFD.UserComment, 7))
        # A damaged TIFF entry reports UserComment as LONG instead of bytes.
        exif[tag + 2 : tag + 12] = struct.pack(">HII", 4, 1, 123)
        output = io.BytesIO()
        pnginfo = PngImagePlugin.PngInfo()
        parameters = self.fixture.image.info["parameters"]
        pnginfo.add_text("parameters", parameters)
        self.fixture.image.save(output, format="PNG", pnginfo=pnginfo, exif=bytes(exif))
        with Image.open(io.BytesIO(output.getvalue())) as reopened:
            original_info = reopened.info.copy()
            geninfo, items = self.fixture.images["read_info_from_image"](reopened)
            self.assertEqual(reopened.info, original_info)
        self.assertEqual(geninfo, parameters)
        self.assertNotIn("exif", items)

    def test_la_and_palette_webp_keep_alpha_and_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode, alpha in (("LA", 128), ("P", 0)):
                with self.subTest(mode=mode):
                    image = self.transparent_image(mode)
                    encoded = self.fixture.api["encode_pil_to_base64"](image)
                    path = Path(directory) / f"透過-{mode}.webp"
                    self.fixture.images["save_image_with_geninfo"](image, image.info["parameters"], path)
                    for source in (io.BytesIO(base64.b64decode(encoded)), path):
                        with self.subTest(boundary=type(source).__name__), Image.open(source) as reopened:
                            self.assertEqual(reopened.convert("RGBA").getchannel("A").tobytes(), bytes([alpha] * 6))
                            self.assertEqual(
                                self.fixture.images["read_info_from_image"](reopened)[0],
                                image.info["parameters"],
                            )


if __name__ == "__main__":
    unittest.main()
