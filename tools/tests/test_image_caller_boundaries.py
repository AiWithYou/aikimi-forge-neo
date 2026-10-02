"""Caller checks for 16-bit images and optional EXIF pointers in valid PNG files."""

from __future__ import annotations

import base64
import io
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import piexif
from PIL import Image, ImageOps, PngImagePlugin

from tools.tests import test_image_roundtrip as fixtures
from tools.tests.test_gpu_ownership import load_function


class ImageCallerBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ImageRoundTripTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        namespace = {
            "models": SimpleNamespace(
                PNGInfoRequest=object, PNGInfoResponse=lambda **values: SimpleNamespace(**values)
            ),
            "decode_base64_to_image": self.decode,
            "images": SimpleNamespace(read_info_from_image=self.fixture.images["read_info_from_image"]),
            "infotext_utils": SimpleNamespace(parse_generation_parameters=lambda text: {"Prompt": text}),
            "script_callbacks": SimpleNamespace(infotext_pasted_callback=Mock()),
        }
        self.pnginfo = load_function("modules/api/api.py", "pnginfoapi", namespace)

    def decode(self, payload):
        image = self.fixture.api["decode_base64_to_image"](payload)
        self.addCleanup(image.close)
        return image

    def test_pnginfo_api_preserves_parameters_with_damaged_exif_pointer_type(self):
        exif = bytearray(piexif.dump({"Exif": {piexif.ExifIFD.UserComment: b"ASCII\0\0\0comment"}}))
        tag = exif.index(struct.pack(">HH", piexif.ImageIFD.ExifTag, 4))
        # A corrupt IFD pointer is stored as four bytes instead of a LONG offset.
        exif[tag + 2 : tag + 8] = struct.pack(">HI", 7, 4)
        metadata = PngImagePlugin.PngInfo()
        parameters = self.fixture.image.info["parameters"]
        metadata.add_text("parameters", parameters)
        output = io.BytesIO()
        self.fixture.image.save(output, format="PNG", pnginfo=metadata, exif=bytes(exif))
        payload = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")
        response = self.pnginfo(None, SimpleNamespace(image=payload))
        self.assertEqual(response.info, parameters)
        self.assertEqual(response.parameters, {"Prompt": parameters})
        self.assertNotIn("exif", response.items)

    def test_decoded_16_bit_png_saves_existing_jpeg_conversion_with_unicode_path(self):
        original = Image.frombytes("I;16", (32, 4), struct.pack("<H", 32768) * 128)
        self.addCleanup(original.close)
        metadata = PngImagePlugin.PngInfo()
        parameters = self.fixture.image.info["parameters"]
        metadata.add_text("parameters", parameters)
        buffer = io.BytesIO()
        original.save(buffer, format="PNG", pnginfo=metadata)
        decoded = self.fixture.api["decode_base64_to_image"](base64.b64encode(buffer.getvalue()).decode("ascii"))
        self.addCleanup(decoded.close)
        self.assertEqual(decoded.mode, "I;16")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "階調画像.JPEG"
            self.fixture.images["save_image_with_geninfo"](decoded, parameters, output)
            with Image.open(output) as reopened:
                self.assertEqual(reopened.mode, "L")
                self.assertLessEqual(abs(reopened.getpixel((0, 0)) - 128), 1)
                self.assertEqual(self.fixture.images["read_info_from_image"](reopened)[0], parameters)

    def test_pnginfo_api_keeps_unicode_comment_after_jpeg_orientation_correction(self):
        image = self.fixture.image.convert("RGB")
        self.addCleanup(image.close)
        parameters = self.fixture.image.info["parameters"]
        exif = piexif.dump(
            {
                "0th": {piexif.ImageIFD.Orientation: 6},
                "Exif": {piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(parameters, encoding="unicode")},
            }
        )
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=100, subsampling=0, exif=exif)
        with Image.open(io.BytesIO(buffer.getvalue())) as reopened:
            expected = ImageOps.exif_transpose(reopened)
            self.addCleanup(expected.close)
        decoded = self.fixture.api["decode_base64_to_image"](base64.b64encode(buffer.getvalue()).decode("ascii"))
        self.addCleanup(decoded.close)
        self.assertEqual(decoded.size, (2, 3))
        self.assertEqual(decoded.tobytes(), expected.tobytes())
        self.assertNotIn(piexif.ImageIFD.Orientation, decoded.getexif())
        comment = decoded.getexif().get_ifd(piexif.ImageIFD.ExifTag)[piexif.ExifIFD.UserComment]
        self.assertEqual(piexif.helper.UserComment.load(comment), parameters)
        self.assertEqual(self.fixture.images["read_info_from_image"](decoded)[0], parameters)
        payload = base64.b64encode(buffer.getvalue()).decode("ascii")
        self.assertEqual(self.pnginfo(None, SimpleNamespace(image=payload)).info, parameters)


if __name__ == "__main__":
    unittest.main()
