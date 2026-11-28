import unittest
from pathlib import Path

from modules_forge.ming_image.checkpoints import resolve


class CheckpointTests(unittest.TestCase):
    def test_escape(self):
        with self.assertRaises(ValueError):
            resolve(Path.cwd(), "../other.safetensors")

    def test_ming_shapes_in_native_and_split_layouts(self):
        from unittest.mock import patch

        from modules_forge.ming_image import checkpoints

        for key, shape in (
            ("layers.0.attention.qkv.weight", [11520, 3840]),
            ("layers.0.attention.to_q.weight", [3840, 3840]),
        ):
            header = {
                "__metadata__": {"config": '{"transformer":{"image_model":"ming_image"}}'},
                "cap_embedder.0.weight": {"shape": [2560]},
                key: {"shape": shape},
            }
            with patch.object(checkpoints.local_assets, "read_header", return_value=header):
                self.assertEqual(checkpoints.role("sample"), "dit")
                header[key]["shape"] = [1, 1]
                with self.assertRaises(ValueError):
                    checkpoints.role("sample")

    def test_invalid_metadata_excluded_and_selection_explained(self):
        import tempfile
        from unittest.mock import patch

        from modules_forge.ming_image import checkpoints

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "checkpoints/bad.safetensors"
            path.parent.mkdir()
            path.touch()
            for config in (None, {}, [], "{", "null", '{"transformer":2}'):
                header = {"__metadata__": {"config": config}}
                with (
                    self.subTest(config=config),
                    patch.object(checkpoints.local_assets, "read_header", return_value=header),
                ):
                    self.assertEqual(checkpoints.inventory(root, "dit"), [("標準モデル", "official")])
                    with self.assertRaisesRegex(checkpoints.MingImageError, "config"):
                        checkpoints.bind(root, {"checkpoint": path.name})
