"""Visual framing input must preserve original geometry and reject stale data."""

import base64
import io
import json
import re
import unittest
from html import unescape

from PIL import Image

from modules_forge.qwen_image21.outpaint import prepare
from modules_forge.qwen_image21.outpaint_gui import canvas_markup, parse_canvas_commit


class CanvasGuiTests(unittest.TestCase):
    def test_zero_or_excess_padding_keeps_visual_editor_available(self):
        source = Image.new("RGB", (736, 512))
        for pads in ((0, 0, 0, 0), (4096, 4096, 4096, 4096)):
            html = canvas_markup(source, *pads)
            self.assertIn('class="qoc-viewport"', html)
            data = json.loads(unescape(re.search(r'data-layout="([^"]+)"', html)[1]))
            self.assertEqual(data, {"w": 736, "h": 512, "pads": list(pads)})
        self.assertEqual(canvas_markup(None, 0, 0, 0, 0), "")

    def test_thumbnail_does_not_resize_original_or_drop_alpha(self):
        source = Image.new("RGBA", (1536, 1024), (255, 0, 0, 0))
        html = canvas_markup(source, 32, 0, 32, 0)
        data = re.search(r"data:image/png;base64,([^\"]+)", html)[1]
        with Image.open(io.BytesIO(base64.b64decode(data))) as thumb:
            self.assertEqual(thumb.size, (1024, 683))
            self.assertEqual(thumb.getpixel((0, 0))[3], 0)
        self.assertEqual(source.size, (1536, 1024))
        self.assertEqual(source.getpixel((0, 0)), (255, 0, 0, 0))

    def test_atomic_commit_feeds_native_geometry_without_resizing(self):
        source = Image.new("RGB", (736, 512))
        pads = parse_canvas_commit(json.dumps({"w": 736, "h": 512, "pads": [192, 0, 64, 0]}), source)
        original, _, plan = prepare(source, *pads)
        self.assertEqual(plan.size, (992, 512))
        self.assertEqual(plan.box, (192, 0, 928, 512))
        self.assertEqual(original.tobytes(), source.tobytes())

    def test_rejects_stale_source_and_untrusted_layout(self):
        source = Image.new("RGB", (736, 512))
        for value in (
            "not json",
            "[]",
            "null",
            "{}",
            "x" * 1025,
            json.dumps({"w": 512, "h": 736, "pads": [1, 0, 0, 0]}),
            *[
                json.dumps({"w": 736, "h": 512, "pads": pads})
                for pads in ([0, 0, 0], [True, 0, 0, 0], [1.5, 0, 0, 0], [-1, 0, 0, 0], [4097, 0, 0, 0])
            ],
        ):
            with self.subTest(value=value[:100]), self.assertRaises(ValueError):
                parse_canvas_commit(value, source)


if __name__ == "__main__":
    unittest.main()
