from __future__ import annotations

import ast
import os
import stat
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


class ImageSaveFilenameTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[2] / "modules/images.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "save_image")
        self.options = SimpleNamespace(save_images_replace_action="Override", target_side_length=1024, export_for_4chan=False, save_txt=True)
        host_os = SimpleNamespace(**{key: getattr(os, key) for key in dir(os) if not key.startswith("__")})
        host_os.statvfs = lambda path: SimpleNamespace(f_namemax=48)
        self.saved = []
        callbacks = SimpleNamespace(
            ImageSaveParams=lambda image, p, filename, pnginfo: SimpleNamespace(image=image, filename=filename, pnginfo=pnginfo),
            before_image_saved_callback=lambda params: None,
            image_saved_callback=lambda params: self.saved.append(params.filename),
        )
        self.host_os, self.callbacks = host_os, callbacks
        scope = {
            "os": host_os, "tempfile": tempfile, "uuid": uuid, "opts": self.options, "shared": SimpleNamespace(opts=self.options),
            "FilenameGenerator": lambda *args, **kwargs: None, "script_callbacks": callbacks,
            "save_image_with_geninfo": lambda image, info, filename, extension, **kwargs: image.save(filename, format="JPEG" if extension == ".jpg" else "PNG"),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), scope)  # noqa: S102 -- isolated production save path with real file writes
        self.save = scope["save_image"]
        self.scope = scope

    def test_temp_creation_is_exclusive_with_normal_creation_permissions(self):
        calls = []
        def create(path, flags, mode=0o777):
            calls.append((flags, mode))
            return os.open(path, flags, mode)
        self.host_os.open = create
        with tempfile.TemporaryDirectory() as temporary:
            self.save(Image.new("RGB", (8, 8)), temporary, "", forced_filename="mode", save_to_dirs=False)
        self.assertEqual(len(calls), 1)
        flags, mode = calls[0]
        self.assertTrue(flags & os.O_EXCL)
        self.assertTrue(flags & os.O_CREAT)
        self.assertEqual(mode, 0o666)

    def test_existing_temp_name_is_never_overwritten_or_deleted(self):
        self.scope["uuid"] = SimpleNamespace(uuid4=lambda: SimpleNamespace(hex="collision"))
        with tempfile.TemporaryDirectory() as temporary:
            existing = Path(temporary) / ".forge-image-collision.tmp"
            existing.write_bytes(b"other writer")
            with self.assertRaises(FileExistsError):
                self.save(Image.new("RGB", (8, 8)), temporary, "", forced_filename="image", save_to_dirs=False)
            self.assertEqual(existing.read_bytes(), b"other writer")
            self.assertEqual(list(Path(temporary).iterdir()), [existing])

    @unittest.skipIf(os.name == "nt", "POSIX file permissions")
    def test_saved_image_permissions_follow_umask(self):
        for mask in (0o022, 0o077):
            with self.subTest(mask=mask), tempfile.TemporaryDirectory() as temporary:
                previous = os.umask(mask)
                try:
                    filename, _ = self.save(Image.new("RGB", (8, 8)), temporary, "", forced_filename="mode", save_to_dirs=False)
                finally:
                    os.umask(previous)
                self.assertEqual(stat.S_IMODE(os.stat(filename).st_mode), 0o666 & ~mask)

    @unittest.skipUnless(os.name == "nt", "Native Windows filename limits")
    def test_windows_long_names_and_collision_suffixes_fit_utf16_limit(self):
        del self.host_os.statvfs
        self.options.save_images_replace_action = "Add number"
        for name in ("x" * 300, "絵" * 300, "😀" * 180):
            with self.subTest(name=name[:1]), tempfile.TemporaryDirectory() as temporary:
                paths = []
                for index in range(2):
                    filename, _ = self.save(Image.new("RGB", (8, 8), (index, 0, 0)), temporary, "",
                                            forced_filename=name, save_to_dirs=False)
                    paths.append(filename)
                    self.assertLessEqual(len(Path(filename).name.encode("utf-16-le")) // 2, 255)
                self.assertNotEqual(*paths)
                for index, filename in enumerate(paths):
                    with Image.open(filename) as image:
                        self.assertEqual(image.getpixel((0, 0)), (index, 0, 0))

    def test_failed_save_cleans_its_temp_and_preserves_existing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "image.tmp").write_bytes(b"unrelated temp")
            (directory / "image.png").write_bytes(b"existing image")

            def fail(image, info, filename, extension, **kwargs):
                Path(filename).write_bytes(b"partial image")
                raise OSError("encoder failed")

            self.scope["save_image_with_geninfo"] = fail
            with self.assertRaisesRegex(OSError, "encoder failed"):
                self.save(Image.new("RGB", (8, 8)), temporary, "", forced_filename="image", save_to_dirs=False)
            self.assertEqual((directory / "image.tmp").read_bytes(), b"unrelated temp")
            self.assertEqual((directory / "image.png").read_bytes(), b"existing image")
            self.assertEqual({path.name for path in directory.iterdir()}, {"image.tmp", "image.png"})
            self.assertEqual(self.saved, [])

    def test_thin_image_export_keeps_both_dimensions_positive(self):
        self.options.export_for_4chan = True
        self.scope["LANCZOS"] = Image.Resampling.LANCZOS
        for size, expected in (((1, 4096), (1, 1024)), ((4096, 1), (1024, 1))):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as temporary:
                filename, _ = self.save(Image.new("RGB", size), temporary, "", forced_filename="thin", save_to_dirs=False)
                with Image.open(Path(filename).with_name("thin-downscaled.jpg")) as exported:
                    self.assertEqual(exported.format, "JPEG")
                    self.assertEqual(exported.size, expected)

    def test_jpeg_export_preserves_full_size_original_even_with_long_name(self):
        self.options.export_for_4chan = True
        self.scope["LANCZOS"] = Image.Resampling.LANCZOS
        with tempfile.TemporaryDirectory() as temporary:
            filename, _ = self.save(Image.new("RGB", (2048, 16)), temporary, "", extension="jpg",
                                    forced_filename="original" * 20, save_to_dirs=False)
            with Image.open(filename) as original:
                self.assertEqual(original.size, (2048, 16))
            copies = list(Path(temporary).glob("*-downscaled.jpg"))
            self.assertEqual(len(copies), 1)
            self.assertLessEqual(len(os.fsencode(copies[0].name)), 48)
            with Image.open(copies[0]) as copy:
                self.assertEqual(copy.size, (1024, 8))
            self.assertEqual(self.saved, [filename])

    def test_export_suffix_cannot_collapse_to_the_original_filename(self):
        self.options.export_for_4chan = True
        self.scope["LANCZOS"] = Image.Resampling.LANCZOS
        with tempfile.TemporaryDirectory() as temporary:
            name = "x" * 33 + "-downscaled"
            filename, _ = self.save(Image.new("RGB", (2048, 16)), temporary, "", extension="jpg",
                                    forced_filename=name, save_to_dirs=False)
            with Image.open(filename) as original:
                self.assertEqual(original.size, (2048, 16))
            copies = list(Path(temporary).glob("*-downscaled-1.jpg"))
            self.assertEqual(len(copies), 1)
            with Image.open(copies[0]) as copy:
                self.assertEqual(copy.size, (1024, 8))

    def test_collision_numbers_fit_byte_limit_and_callbacks_report_saved_path(self):
        self.options.save_images_replace_action = "Add number"
        with tempfile.TemporaryDirectory() as temporary:
            outputs = []
            for index in range(12):
                filename, metadata = self.save(Image.new("RGB", (8, 8), (index, 0, 0)), temporary, "",
                                               forced_filename="画像" * 30, save_to_dirs=False, info=str(index))
                outputs.append(filename)
                self.assertLessEqual(len(os.fsencode(Path(filename).name)), 48)
                self.assertEqual(self.saved[-1], filename)
                self.assertEqual(Path(metadata).read_text(encoding="utf-8"), f"{index}\n")
            self.assertEqual(len(set(outputs)), 12)
            for index, filename in enumerate(outputs):
                with Image.open(filename) as saved:
                    self.assertEqual(saved.getpixel((0, 0)), (index, 0, 0))

    def test_callback_redirect_uses_destination_filesystem_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "redirect"
            destination.mkdir()
            checked = []

            def limits(path):
                checked.append(Path(path))
                return SimpleNamespace(f_namemax=32 if Path(path) == destination else 48)

            def redirect(params):
                params.filename = str(destination / ("x" * 70 + ".png"))

            self.host_os.statvfs = limits
            self.callbacks.before_image_saved_callback = redirect
            filename, _ = self.save(Image.new("RGB", (8, 8)), temporary, "", forced_filename="original", save_to_dirs=False)
            self.assertEqual(Path(filename).parent, destination)
            self.assertLessEqual(len(os.fsencode(Path(filename).name)), 32)
            self.assertIn(destination, checked)

    def test_override_policy_replaces_existing_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for color in ("red", "blue"):
                filename, _ = self.save(Image.new("RGB", (8, 8), color), temporary, "", forced_filename="same", save_to_dirs=False)
                paths.append(filename)
            self.assertEqual(paths[0], paths[1])
            with Image.open(paths[1]) as saved:
                self.assertEqual(saved.getpixel((0, 0)), (0, 0, 255))

    def test_without_statvfs_collision_still_reports_actual_filename(self):
        del self.host_os.statvfs
        self.options.save_images_replace_action = "Add number"
        with tempfile.TemporaryDirectory() as temporary:
            for index in range(2):
                filename, _ = self.save(Image.new("RGB", (8, 8)), temporary, "", forced_filename="image", save_to_dirs=False)
                self.assertEqual(Path(filename).name, "image.png" if index == 0 else "image-1.png")
                self.assertEqual(self.saved[-1], filename)

    def test_long_directory_is_preserved_and_filename_respects_byte_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "deep-output-directory-longer-than-the-filename-limit" / "images"
            for name in ("short", "画像" * 30, "long-name-" * 20):
                with self.subTest(name=name):
                    image = Image.new("RGB", (8, 8), "red")
                    filename, metadata = self.save(image, str(directory), "", forced_filename=name, save_to_dirs=False, info="test")
                    output = Path(filename)
                    self.addCleanup(output.unlink, missing_ok=True)
                    self.addCleanup(Path(metadata).unlink, missing_ok=True)
                    self.assertEqual(output.parent, directory)
                    self.assertLessEqual(len(os.fsencode(output.name)), 48)
                    self.assertEqual(output.suffix, ".png")
                    with Image.open(output) as saved:
                        self.assertEqual(saved.getpixel((0, 0)), (255, 0, 0))
                    self.assertEqual(Path(metadata).read_text(encoding="utf-8"), "test\n")
                    self.assertEqual(image.already_saved_as, filename)


if __name__ == "__main__":
    unittest.main()
