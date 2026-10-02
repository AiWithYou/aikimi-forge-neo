"""Local model preflight rejects incomplete tensor payloads without loading a model."""

from __future__ import annotations

import json
import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from safetensors import SafetensorError, safe_open

from modules_forge import local_assets
from modules_forge.qwen_image21 import local_source
from tools.tests import test_local_model_sources as fixtures


class LocalAssetHeaderBoundsTests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        for name, filename in (("LIBRARY", "library.json"), ("HASH_CACHE", "hashes.json")):
            context = patch.object(local_assets, name, self.root / filename)
            context.start()
            self.addCleanup(context.stop)

    def test_truncated_payload_is_rejected_despite_complete_header(self):
        path = fixtures.tensor_file(self.root / "途中取得.safetensors", {"weight": [2]})
        with safe_open(str(path), framework="numpy") as tensors:
            self.assertEqual(tensors.keys(), ["weight"])
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaises(SafetensorError), safe_open(str(path), framework="numpy"):
            pass
        with self.assertRaises(ValueError):
            local_assets.read_header(path)

    def test_qwen_external_selection_rejects_partial_tensor_before_identity(self):
        fixtures.qwen_fixture(self.root / "runtime/model", transformer=False)
        transformer = fixtures.qwen_transformer(self.root / "外部本体")
        weights = transformer / "diffusion_pytorch_model.safetensors"
        weights.write_bytes(weights.read_bytes()[:-1])
        with (
            patch.object(local_assets, "identity", side_effect=AssertionError("incomplete model was identified")),
            self.assertRaises(ValueError),
        ):
            local_source.resolve(self.root / "runtime", str(transformer), "", "int8")

    def test_invalid_tensor_offsets_are_rejected(self):
        path = self.root / "invalid.safetensors"
        for offsets in ([-1, 4], [4, 0], [0, 5], [False, 4], [0, 4.0], [0], "0,4", None):
            with self.subTest(offsets=offsets):
                raw = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": offsets}}).encode()
                path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\x00" * 4)
                with self.assertRaises(ValueError):
                    local_assets.read_header(path)
        raw = json.dumps({"weight": "not a tensor record"}).encode()
        path.write_bytes(struct.pack("<Q", len(raw)) + raw)
        with self.assertRaises(ValueError):
            local_assets.read_header(path)

    def test_scalar_empty_tensor_and_unicode_metadata_keep_their_header(self):
        path = fixtures.tensor_file(
            self.root / "日本語モデル.safetensors",
            {"scalar": [], "empty": [0], "matrix": [2, 2]},
            metadata={"title": "外部モデル"},
        )
        with safe_open(str(path), framework="numpy") as tensors:
            self.assertEqual(tensors.keys(), ["empty", "matrix", "scalar"])
        header = local_assets.read_header(path)
        self.assertEqual(header["__metadata__"], {"title": "外部モデル"})
        self.assertEqual(header["scalar"]["shape"], [])
        self.assertEqual(header["empty"]["data_offsets"], [4, 4])
        self.assertEqual(header["matrix"]["data_offsets"], [4, 20])


if __name__ == "__main__":
    unittest.main()
