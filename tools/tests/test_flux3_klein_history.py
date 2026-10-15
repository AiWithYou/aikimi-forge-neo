"""Persistent FLUX 3 results stay within their output root and are revalidated."""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image, PngImagePlugin

from modules_forge.flux3_klein.history import History


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "outputs"
        self.root.mkdir()
        self.history = History(self.root)

    def completed(self, number=1):
        identifier = f"{number:032x}"
        directory = self.root / identifier
        directory.mkdir()
        request = {
            "prompt": "a white vase",
            "model": "distilled",
            "precision": "int8",
            "width": 64,
            "height": 64,
            "steps": 16,
            "cfg": 1,
            "seed": number,
            "boxes": [],
            "images": [],
            "reference_sha256": [],
            "pup": False,
        }
        record = {
            **{key: request[key] for key in ("model", "precision", "width", "height", "steps", "cfg", "seed", "boxes")},
            "prompt": request["prompt"],
            "input_prompt": request["prompt"],
            "reference_hashes": [],
            "upsampled": False,
        }
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("flux3_klein", json.dumps(record))
        Image.new("RGB", (64, 64), "white").save(directory / "output.png", pnginfo=metadata)
        record.update(image_path=str(directory / "output.png"), result_path=str(directory / "result.json"))
        for filename, value in (
            ("request.json", request),
            ("result.json", record),
            ("progress.json", {"stage": "complete"}),
        ):
            (directory / filename).write_text(json.dumps(value), encoding="utf-8")
        os.utime(directory, (number, number))
        return identifier, directory

    def test_existing_results_survive_a_new_history_instance_and_select_files(self):
        identifier, directory = self.completed()
        choices = History(self.root).list()
        self.assertEqual([value for _, value in choices], [identifier])
        self.assertIn("a white vase", choices[0][0])
        result = self.history.load(identifier)
        self.assertEqual(result["id"], identifier)
        self.assertEqual(result["files"], [str(directory / "output.png"), str(directory / "result.json")])
        self.assertEqual(result["prompt"], "a white vase")

    def test_newest_fifty_candidates_are_bounded_and_ordered(self):
        for number in range(1, 54):
            self.completed(number)
        choices = self.history.list()
        self.assertEqual(len(choices), 50)
        self.assertEqual([value for _, value in choices], [f"{number:032x}" for number in range(53, 3, -1)])

    def test_failed_cancelled_incomplete_and_corrupt_results_are_excluded(self):
        mutations = (
            lambda d: (d / "cancel").touch(),
            lambda d: (d / "progress.json").write_text('{"stage":"sampling"}'),
            lambda d: (d / "result.json").write_text('{"status":"error"}'),
            lambda d: (d / "output.png").write_bytes(b"broken"),
            lambda d: (d / "request.json").unlink(),
            lambda d: (d / "result.json").write_text("{"),
        )
        for number, mutate in enumerate(mutations, 1):
            _, directory = self.completed(number)
            mutate(directory)
        self.assertEqual(self.history.list(), [])

    def test_selection_revalidates_after_listing(self):
        identifier, directory = self.completed()
        self.assertTrue(self.history.list())
        record = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        record["seed"] += 1
        (directory / "result.json").write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.history.load(identifier)

    def test_modified_reference_pixels_are_rejected_even_with_the_same_dimensions(self):
        identifier, directory = self.completed()
        reference = directory / "reference-00.png"
        Image.new("RGB", (64, 64), "red").save(reference)
        expected = hashlib.sha256(reference.read_bytes()).hexdigest()
        request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
        request.update(images=[str(reference)], reference_sha256=[expected])
        record = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        record["reference_hashes"] = [expected]
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("flux3_klein", json.dumps(record))
        Image.new("RGB", (64, 64), "white").save(directory / "output.png", pnginfo=metadata)
        (directory / "request.json").write_text(json.dumps(request), encoding="utf-8")
        (directory / "result.json").write_text(json.dumps(record), encoding="utf-8")
        self.assertTrue(self.history.load(identifier))
        Image.new("RGB", (64, 64), "blue").save(reference)
        with self.assertRaises(ValueError):
            self.history.load(identifier)
        self.assertEqual(self.history.list(), [])

    def test_external_image_and_reference_paths_are_never_opened(self):
        identifier, directory = self.completed()
        record = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        record["image_path"] = str(self.root.parent / "outside.png")
        (directory / "result.json").write_text(json.dumps(record), encoding="utf-8")
        self.assertEqual(self.history.list(), [])
        with self.assertRaises(ValueError):
            self.history.load(identifier)
        for value in ("../outside", str(directory), "a" * 31, "g" * 32, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.history.load(value)
        record["image_path"] = str(directory / "output.png")
        (directory / "result.json").write_text(json.dumps(record), encoding="utf-8")
        request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
        request["images"] = [str(self.root.parent / "outside.png")]
        (directory / "request.json").write_text(json.dumps(request), encoding="utf-8")
        self.assertEqual(self.history.list(), [])

    def test_symlinked_artifacts_and_directories_are_excluded(self):
        identifier, directory = self.completed()
        original = directory / "original.png"
        (directory / "output.png").rename(original)
        try:
            (directory / "output.png").symlink_to(original)
        except OSError as error:
            self.skipTest(f"Symlink creation unavailable: {error}")
        self.assertEqual(self.history.list(), [])
        with self.assertRaises(ValueError):
            self.history.load(identifier)
        (directory / "output.png").unlink()
        original.rename(directory / "output.png")
        link = self.root / ("f" * 32)
        link.symlink_to(directory, target_is_directory=True)
        self.assertEqual([value for _, value in self.history.list()], [identifier])
        with self.assertRaises(ValueError):
            self.history.load(link.name)
        alias = self.root.parent / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        self.assertEqual(History(alias).list(), [])

    def test_missing_root_is_empty_and_history_does_not_create_it(self):
        root = self.root / "missing"
        self.assertEqual(History(root).list(), [])
        self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
