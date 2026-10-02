from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from modules_forge import ming_image_studio as studio


class MingTransparencyContractTests(unittest.TestCase):
    def test_published_png_transparency_is_measured_in_all_accepted_modes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = studio.MingImageRequest("A cutout", transparent=True, width=256, height=256)
            readiness = SimpleNamespace(core_revision="fixture", comfy_version="fixture")
            for mode in ("LA", "P", "RGB"):
                with self.subTest(mode=mode):
                    source = root / f"{mode}.png"
                    options = {}
                    if mode == "LA":
                        image = Image.new(mode, (256, 256), (128, 0))
                        image.putpixel((0, 0), (128, 255))
                    elif mode == "P":
                        image = Image.new(mode, (256, 256), 0)
                        image.putpalette([0, 0, 0, 255, 0, 0] + [0] * 762)
                        image.putpixel((0, 0), 1)
                        options["transparency"] = 0
                    else:
                        image = Image.new(mode, (256, 256), (1, 2, 3))
                        image.putpixel((0, 0), (4, 5, 6))
                        options["transparency"] = (1, 2, 3)
                    try:
                        image.save(source, **options)
                    finally:
                        image.close()

                    result = studio.save_result(source, request, 42, "fixture", readiness, root / "output")

                    self.assertEqual(Path(result["path"]).read_bytes(), source.read_bytes())
                    self.assertEqual(result["metadata"]["image_mode"], mode)
                    self.assertTrue(result["metadata"]["has_transparency"])
                    self.assertEqual(result["metadata"]["alpha_range"], [0, 255])
                    self.assertAlmostEqual(result["metadata"]["near_transparent_fraction"], 1 - 1 / 256**2)


if __name__ == "__main__":
    unittest.main()
