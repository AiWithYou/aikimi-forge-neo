"""Checkpoint read failures retain their actual cause instead of claiming corruption."""

import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

import safetensors
import torch
from safetensors.torch import save_file

from backend import utils


class CheckpointLoadErrorTests(unittest.TestCase):
    def test_open_errors_keep_the_original_exception(self):
        for error in (
            FileNotFoundError("checkpoint moved"),
            PermissionError("checkpoint access denied"),
            MemoryError("cannot allocate mapping"),
        ):
            with self.subTest(error=type(error).__name__):
                with (
                    patch.object(utils.safetensors, "safe_open", side_effect=error),
                    self.assertRaises(type(error)) as raised,
                ):
                    utils.load_torch_file("fixture.safetensors")
                self.assertIs(raised.exception, error)

    def test_tensor_read_errors_keep_the_original_exception(self):
        for error in (
            MemoryError("cannot allocate tensor"),
            torch.OutOfMemoryError("device allocation failed"),
            RuntimeError("device is unavailable"),
        ):
            reader = Mock()
            reader.keys.return_value = ["weight"]
            reader.get_tensor.side_effect = error
            with self.subTest(error=type(error).__name__):
                with (
                    patch.object(utils.safetensors, "safe_open", return_value=nullcontext(reader)),
                    self.assertRaises(type(error)) as raised,
                ):
                    utils.load_torch_file("fixture.safetensors")
                self.assertIs(raised.exception, error)

    def test_invalid_file_keeps_the_parser_detail_and_cause(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.safetensors"
            path.write_bytes(b"broken")
            with self.assertRaises(ValueError) as raised:
                utils.load_torch_file(str(path))
        cause = raised.exception.__cause__
        self.assertIsInstance(cause, safetensors.SafetensorError)
        self.assertIn(str(cause), str(raised.exception))
        self.assertIn(str(path), str(raised.exception))
        self.assertNotIn("download the model again", str(raised.exception))

    def test_valid_weights_and_metadata_still_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "valid.safetensors"
            weight = torch.arange(4, dtype=torch.float32)
            save_file({"weight": weight}, str(path), metadata={"format": "fixture"})
            state, metadata = utils.load_torch_file(str(path), return_metadata=True)
        torch.testing.assert_close(state["weight"], weight)
        self.assertEqual(metadata, {"format": "fixture"})


if __name__ == "__main__":
    unittest.main()
