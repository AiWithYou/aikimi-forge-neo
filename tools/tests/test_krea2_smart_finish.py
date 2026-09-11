from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from tools.krea2_smart_finish import prepare_source_image, save_png, validate_paths


class SmartFinishPathTests(unittest.TestCase):
    def test_rejects_output_report_collision(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "distinct"):
                validate_paths(
                    root / "input.png", root / "result.png", root / "result.png"
                )

    def test_rejects_input_report_collision(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "distinct"):
                validate_paths(
                    root / "input.png", root / "result.png", root / "input.png"
                )

    def test_requires_png_output_suffix(self):
        with self.assertRaisesRegex(ValueError, r"\.png"):
            validate_paths(Path("input.png"), Path("result.jpg"), Path("result.json"))

    def test_requires_json_report_suffix(self):
        with self.assertRaisesRegex(ValueError, r"\.json"):
            validate_paths(Path("input.png"), Path("result.png"), Path("report.txt"))


class SmartFinishTransparencyTests(unittest.TestCase):
    def test_palette_transparency_is_preserved(self):
        image = Image.new("P", (2, 1))
        image.putpalette([255, 0, 0, 0, 0, 255] + [0] * (768 - 6))
        image.putdata([0, 1])
        image.info["transparency"] = 0

        prepared = prepare_source_image(image)

        self.assertEqual(prepared.mode, "RGBA")
        self.assertEqual(prepared.getpixel((0, 0))[3], 0)
        self.assertEqual(prepared.getpixel((1, 0)), (0, 0, 255, 255))

    def test_luminance_alpha_is_preserved(self):
        image = Image.new("LA", (1, 1), (90, 37))

        prepared = prepare_source_image(image)

        self.assertEqual(prepared.mode, "RGBA")
        self.assertEqual(prepared.getpixel((0, 0)), (90, 90, 90, 37))

    def test_png_metadata_is_preserved_with_quality_report(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "result.png"
            save_png(
                path,
                Image.new("RGBA", (2, 2), (10, 20, 30, 40)),
                parameters="prompt\nSteps: 8",
                report={"version": 1},
                overwrite=False,
                preserved_info={
                    "Description": "kept text",
                    "icc_profile": b"test profile bytes",
                    "dpi": (72.0, 72.0),
                },
            )

            with Image.open(path) as result:
                self.assertEqual(result.info["Description"], "kept text")
                self.assertEqual(result.info["parameters"], "prompt\nSteps: 8")
                self.assertEqual(result.info["icc_profile"], b"test profile bytes")
                self.assertIn('"version":1', result.info["krea2_smart_finish"])


if __name__ == "__main__":
    unittest.main()
