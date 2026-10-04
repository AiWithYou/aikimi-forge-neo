"""Existing flattening and API output must preserve transparent and 16-bit input."""

from __future__ import annotations

import base64
import io
import struct
import unittest

from PIL import Image, PngImagePlugin

from tools.tests import test_image_roundtrip as roundtrip


class ImageOutputBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = roundtrip.ImageRoundTripTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.flatten = roundtrip.load_boundary_functions("modules/images.py", {"flatten"}, {"Image": Image})["flatten"]

    def assert_flatten(self, image, expected):
        self.addCleanup(image.close)
        original_pixels = image.tobytes()
        original_info = image.info.copy()
        result = self.flatten(image, "#204060")
        self.addCleanup(result.close)
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(list(result.getdata()), expected)
        self.assertEqual(image.tobytes(), original_pixels)
        self.assertEqual(image.info, original_info)
        return result

    def test_flatten_uses_alpha_channels_and_palette_byte_transparency(self):
        rgba = Image.new("RGBA", (3, 1))
        rgba.putdata([(100, 140, 200, alpha) for alpha in (0, 128, 255)])
        luminance = Image.new("LA", (3, 1))
        luminance.putdata([(80, alpha) for alpha in (0, 128, 255)])
        palette = Image.new("P", (3, 1))
        palette.putpalette([80, 0, 0, 0, 80, 0, 0, 0, 80] + [0] * 759)
        palette.putdata([0, 1, 2])
        palette.info["transparency"] = bytes([0, 128, 255])
        for image, expected in (
            (rgba, [(32, 64, 96), (66, 102, 148), (100, 140, 200)]),
            (luminance, [(32, 64, 96), (56, 72, 88), (80, 80, 80)]),
            (palette, [(32, 64, 96), (16, 72, 48), (0, 0, 80)]),
        ):
            with self.subTest(mode=image.mode):
                self.assert_flatten(image, expected)

    def test_flatten_decoded_png_uses_palette_and_grayscale_rgb_color_keys(self):
        for mode, transparent in (("P", 0), ("L", 80), ("RGB", (80, 0, 0))):
            with self.subTest(mode=mode):
                original = Image.new(mode, (2, 1))
                self.addCleanup(original.close)
                if mode == "P":
                    original.putpalette([80, 0, 0, 0, 0, 80] + [0] * 762)
                    original.putdata([0, 1])
                    opaque = (0, 0, 80)
                elif mode == "L":
                    original.putdata([80, 120])
                    opaque = (120, 120, 120)
                else:
                    original.putdata([(80, 0, 0), (0, 0, 80)])
                    opaque = (0, 0, 80)
                original.info["transparency"] = transparent
                output = io.BytesIO()
                original.save(output, format="PNG")
                decoded = self.fixture.api["decode_base64_to_image"](
                    base64.b64encode(output.getvalue()).decode("ascii")
                )
                self.assert_flatten(decoded, [(32, 64, 96), opaque])

    def test_flatten_opaque_inputs_keeps_pixels(self):
        for mode, color, expected in (
            ("RGB", (80, 120, 160), (80, 120, 160)),
            ("L", 80, (80, 80, 80)),
            ("RGBA", (80, 120, 160, 255), (80, 120, 160)),
        ):
            with self.subTest(mode=mode):
                self.assert_flatten(Image.new(mode, (1, 1), color), [expected])

    def test_flatten_png_transparency_text_keeps_text_and_uses_only_real_alpha(self):
        for text in ("user note", "透過メモ"):
            for mode, color, expected in (
                ("RGB", (80, 120, 160), (80, 120, 160)),
                ("L", 80, (80, 80, 80)),
                ("P", 0, (80, 120, 160)),
                ("1", 1, (255, 255, 255)),
                ("I;16", 32768, (255, 255, 255)),
                ("LA", (80, 128), (56, 72, 88)),
                ("RGBA", (100, 140, 200, 128), (66, 102, 148)),
            ):
                with self.subTest(mode=mode, text=text):
                    image = Image.new(mode, (1, 1), color)
                    self.addCleanup(image.close)
                    if mode == "P":
                        image.putpalette([80, 120, 160] + [0] * 765)
                    metadata = PngImagePlugin.PngInfo()
                    metadata.add_text("transparency", text)
                    output = io.BytesIO()
                    image.save(output, format="PNG", pnginfo=metadata)
                    with Image.open(io.BytesIO(output.getvalue())) as reopened:
                        self.assertEqual(reopened.info["transparency"], text)
                        result = self.assert_flatten(reopened, [expected])
                        self.assertEqual(result.info["transparency"], text)

    def test_api_16_bit_jpeg_webp_scale_luminance_and_png_preserves_samples(self):
        samples = (0, 257, 32768, 65535)
        row = b"".join(struct.pack("<H", value) * 16 for value in samples)
        image = Image.frombytes("I;16", (64, 8), row * 8)
        self.addCleanup(image.close)
        image.info["parameters"] = self.fixture.image.info["parameters"]
        original_pixels = image.tobytes()
        original_info = image.info.copy()
        self.fixture.opts.jpeg_quality = 100

        for format_name in ("jpeg", "webp", "png"):
            with self.subTest(format=format_name):
                self.fixture.opts.samples_format = format_name
                encoded = self.fixture.api["encode_pil_to_base64"](image)
                with Image.open(io.BytesIO(base64.b64decode(encoded))) as reopened:
                    self.assertEqual(self.fixture.images["read_info_from_image"](reopened)[0], image.info["parameters"])
                    expected = samples if format_name == "png" else (0, 1, 127, 255)
                    for index, value in enumerate(expected):
                        actual = reopened.getpixel((index * 16 + 8, 4))
                        if isinstance(actual, tuple):
                            self.assertEqual(actual[0], actual[1])
                            self.assertEqual(actual[1], actual[2])
                            actual = actual[0]
                        self.assertLessEqual(abs(actual - value), 1 if format_name == "jpeg" else 0)
                self.assertEqual(image.mode, "I;16")
                self.assertEqual(image.tobytes(), original_pixels)
                self.assertEqual(image.info, original_info)


if __name__ == "__main__":
    unittest.main()
