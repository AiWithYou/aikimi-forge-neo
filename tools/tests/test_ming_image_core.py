import unittest

from PIL import Image

from modules_forge.ming_image.core import validate_request


class CoreTests(unittest.TestCase):
    def test_defaults(self):
        r = validate_request({"prompt": "layout"})
        self.assertEqual((r["steps"], r["shift"], r["cfg"]), (12, 1.15, 1))

    def test_width(self):
        for w in (511, 513, 3088, True):
            with self.subTest(w=w), self.assertRaises(ValueError):
                validate_request({"prompt": "layout", "width": w})

    def test_edit_reference_limit(self):
        refs = [Image.new("RGB", (16, 16))] * 8
        value = dict(prompt="edit", mode="img2img", images=refs)
        self.assertEqual(validate_request(value)["images"], refs)
        with self.assertRaises(ValueError):
            validate_request({**value, "images": refs + refs})

    def test_cfg_bounds(self):
        for invalid in (0, 0.5, 20.1):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "CFG"):
                validate_request({"prompt": "layout", "cfg": invalid})
        for valid in (1.0, 20.0):
            self.assertEqual(validate_request({"prompt": "layout", "cfg": valid})["cfg"], valid)
