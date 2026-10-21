"""Keep image decoding and certificate dependencies on the audited patch level."""

import io
import re
import unittest
from pathlib import Path

from PIL import Image, PngImagePlugin

ROOT = Path(__file__).resolve().parents[2]
DIRECT = (
    "requirements.txt",
    "tools/requirements-iris.txt",
    "tools/requirements-clef.txt",
    "tools/requirements-qwen-image21.txt",
    "tools/requirements-sensenova.txt",
)
LOCKED = tuple(f"tools/requirements-{name}.lock" for name in ("moss-image", "flux3-klein", "minimax-h3", "ming-image"))


class DependencyPatchVersionsTests(unittest.TestCase):
    def assert_pin(self, filename, package, version):
        text = (ROOT / filename).read_text(encoding="utf-8")
        pattern = rf"(?im)^{package}=={re.escape(version)}(?:\s*\\)?$"
        self.assertIsNotNone(re.search(pattern, text), f"{filename}: {package} must use the security patch")

    def test_direct_runtime_requirements_use_fixed_image_and_certificate_libraries(self):
        for filename in DIRECT:
            with self.subTest(filename=filename):
                self.assert_pin(filename, "pillow", "13.0.0")
                self.assert_pin(filename, "certifi", "2026.10.19")
        self.assert_pin("tools/requirements-aikimi-assets.txt", "pillow", "13.0.0")

    def test_all_locked_runtimes_use_fixed_versions_and_keep_hashes(self):
        for filename in LOCKED:
            with self.subTest(filename=filename):
                self.assert_pin(filename, "pillow", "13.0.0")
                self.assert_pin(filename, "certifi", "2026.10.19")
                text = (ROOT / filename).read_text(encoding="utf-8")
                for package in ("pillow", "certifi"):
                    self.assertRegex(text, rf"(?m)^{package}==[^\n]+\n\s+--hash=sha256:[0-9a-f]{{64}}")

    def test_png_pixel_modes_and_generation_metadata_round_trip(self):
        for mode, value in (("RGB", (13, 71, 149)), ("RGBA", (13, 71, 149, 117)), ("L", 137)):
            with self.subTest(mode=mode):
                original = Image.new(mode, (7, 5), value)
                metadata = PngImagePlugin.PngInfo()
                metadata.add_text("moss_image", '{"steps":36}')
                stream = io.BytesIO()
                original.save(stream, format="PNG", pnginfo=metadata)
                stream.seek(0)
                with Image.open(stream) as restored:
                    restored.load()
                    self.assertEqual(restored.tobytes(), original.tobytes())
                    self.assertEqual(restored.mode, mode)
                    self.assertEqual(restored.info["moss_image"], '{"steps":36}')

    def test_jpeg_exif_survives_image_io(self):
        original = Image.new("RGB", (8, 8), (50, 100, 150))
        exif = original.getexif()
        exif[305] = "Aikimi Forge Neo"
        stream = io.BytesIO()
        original.save(stream, format="JPEG", exif=exif)
        stream.seek(0)
        with Image.open(stream) as restored:
            restored.load()
            self.assertEqual(restored.size, original.size)
            self.assertEqual(restored.getexif()[305], "Aikimi Forge Neo")


if __name__ == "__main__":
    unittest.main()
