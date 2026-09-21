"""Image fidelity and cache boundaries for the lightweight canvas transport."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from modules_forge.forge_canvas.canvas import LogicalImage


class CanvasTransportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="canvas-transport-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.component = LogicalImage(numpy=False, file_transport=True)
        self.component.GRADIO_CACHE = str(self.root / "cache")
        self.image = Image.new("RGBA", (31, 23), (120, 40, 210, 75))

    def test_file_roundtrip_preserves_alpha_and_uses_a_short_reference(self):
        payload = self.component.postprocess(self.image)
        self.assertTrue(payload.startswith("forge-file:"))
        self.assertLess(len(payload), 1024)
        reference = json.loads(payload.removeprefix("forge-file:"))
        self.assertIn("gradio_api/file=", reference["url"])
        self.assertEqual(self.component.preprocess(payload).tobytes(), self.image.tobytes())

    def test_rejects_existing_image_outside_its_cache(self):
        outside = self.root / "private.png"
        self.image.save(outside)
        payload = "forge-file:" + json.dumps({"path": str(outside), "url": "/ignored"})
        with self.assertRaisesRegex(ValueError, "cache file"):
            self.component.preprocess(payload)

    def test_file_component_rejects_inline_payload(self):
        with self.assertRaisesRegex(ValueError, "Invalid canvas"):
            self.component.preprocess("data:image/png;base64,AAAA")
        self.assertIsNone(self.component.preprocess(""))

    def test_forge_redirected_temp_cache_is_accepted_only_after_emitting_the_file(self):
        redirected = self.root / "forge-temp.png"
        self.image.save(redirected)
        with patch("gradio.processing_utils.save_pil_to_cache", return_value=str(redirected)):
            payload = self.component.postprocess(self.image)
        self.assertEqual(self.component.preprocess(payload).tobytes(), self.image.tobytes())
        other = LogicalImage(numpy=False, file_transport=True)
        with self.assertRaisesRegex(ValueError, "cache file"):
            other.preprocess(payload)

    def test_initial_images_work_with_both_transports(self):
        for file_transport in (False, True):
            with self.subTest(file_transport=file_transport):
                component = LogicalImage(numpy=False, file_transport=file_transport, value=self.image)
                self.assertEqual(component.preprocess(component.value).tobytes(), self.image.tobytes())

    def test_legacy_numpy_transport_is_unchanged(self):
        pixels = np.asarray(self.image)
        component = LogicalImage(numpy=True)
        payload = component.postprocess(pixels)
        self.assertTrue(payload.startswith("data:image/png;base64,"))
        np.testing.assert_array_equal(component.preprocess(payload), pixels)


if __name__ == "__main__":
    unittest.main()
