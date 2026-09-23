"""Unsloth Qwen 2.1 GGUF name conversion and pinned local inventory."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from modules_forge.qwen_image21 import gguf, regular_gguf


class FakeFusedWeight:
    def chunk(self, pieces, dim):
        if (pieces, dim) != (2, 0):
            raise AssertionError("SwiGLU output dimension must be split in half")
        return "gate", "projection"


class RegularGGUFTests(unittest.TestCase):
    def test_unsloth_mapping_splits_gate_up_and_removes_prefix(self):
        prefix = "model.diffusion_model."
        checkpoint = {f"{prefix}other.{index}.weight": index for index in range(233)}
        checkpoint.update(
            {f"{prefix}transformer_blocks.{index}.img_mlp.gate_up.weight": FakeFusedWeight() for index in range(32)}
        )
        result = gguf.map_unsloth_checkpoint(checkpoint)
        self.assertEqual(len(result), 297)
        self.assertEqual(result["transformer_blocks.0.img_mlp.gate_layer.weight"], "gate")
        self.assertEqual(result["transformer_blocks.0.img_mlp.proj.weight"], "projection")
        self.assertEqual(result["other.0.weight"], 0)
        with self.assertRaisesRegex(ValueError, "予期しない"):
            gguf.map_unsloth_checkpoint({"other.weight": FakeFusedWeight()})

    def test_corrupt_download_never_publishes_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / regular_gguf.FOLDER
            destination.mkdir()
            path = destination / regular_gguf.NAME
            path.write_bytes(b"bad")
            sha = hashlib.sha256(b"expected").hexdigest()
            item = SimpleNamespace(rfilename=regular_gguf.NAME, size=3, lfs=SimpleNamespace(sha256=sha))
            hub = SimpleNamespace(
                HfApi=lambda: SimpleNamespace(
                    model_info=lambda *args, **kwargs: SimpleNamespace(
                        sha=regular_gguf.REVISION,
                        siblings=[item],
                    )
                ),
                hf_hub_download=lambda *args, **kwargs: str(path),
            )
            with (
                mock.patch.object(regular_gguf, "SIZE", 3),
                mock.patch.object(regular_gguf, "SHA256", sha),
                mock.patch.dict("sys.modules", {"huggingface_hub": hub}),
            ):
                with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                    regular_gguf.download_regular(root)
            self.assertFalse((root / regular_gguf.INVENTORY).exists())


if __name__ == "__main__":
    unittest.main()
