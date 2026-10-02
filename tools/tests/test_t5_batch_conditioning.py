"""Existing T5 batch and BREAK behavior with a deterministic CPU encoder."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from backend.text_processing import emphasis


def load_engine():
    source = Path(__file__).resolve().parents[2] / "backend/text_processing/t5_engine.py"
    spec = importlib.util.spec_from_file_location("_test_t5_batch_engine", source)
    module = importlib.util.module_from_spec(spec)
    shared = types.ModuleType("modules.shared")
    shared.opts = types.SimpleNamespace(emphasis="Ignore")
    previous = sys.modules.get("modules.shared")
    sys.modules["modules.shared"] = shared
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop("modules.shared", None)
        else:
            sys.modules["modules.shared"] = previous
    return module


class CharacterTokenizer:
    def __call__(self, texts, **kwargs):
        return {"input_ids": [[ord(char) for char in text] for text in texts]}


class RecordingEncoder:
    def __init__(self):
        self.calls = []

    def __call__(self, *, input_ids):
        self.calls.append(input_ids.clone())
        return input_ids.float().unsqueeze(-1).repeat(1, 1, 2)


class T5BatchConditioningTests(unittest.TestCase):
    def setUp(self):
        self.module = load_engine()
        self.encoder = RecordingEncoder()
        self.engine = self.module.T5TextProcessingEngine(
            types.SimpleNamespace(transformer=self.encoder), CharacterTokenizer(), min_length=1, min_padding=0
        )
        self.enterContext(patch.object(self.module.memory_management, "text_encoder_device", return_value="cpu"))

    def test_different_prompt_lengths_share_one_batch(self):
        actual = self.engine(["a", "abcd", "a"])
        expected = torch.tensor([[97, 1, 0, 0, 0], [97, 98, 99, 100, 1], [97, 1, 0, 0, 0]]).float()
        torch.testing.assert_close(actual, expected.unsqueeze(-1).repeat(1, 1, 2))
        self.assertEqual(len(self.encoder.calls), 2)

    def test_break_sections_stay_in_their_prompt(self):
        actual = self.engine(["a BREAK bc", "de"])
        expected = torch.tensor([[97, 1, 0, 98, 99, 1], [100, 101, 1, 1, 0, 0]]).float()
        torch.testing.assert_close(actual, expected.unsqueeze(-1).repeat(1, 1, 2))

    def test_homogeneous_batch_keeps_previous_padding_and_values(self):
        self.engine.min_length = 4
        self.engine.min_padding = 2
        actual = self.engine(["abc", "xyz"])
        expected = torch.tensor([[97, 98, 99, 1, 0, 0], [120, 121, 122, 1, 0, 0]]).float()
        torch.testing.assert_close(actual, expected.unsqueeze(-1).repeat(1, 1, 2))


class ZeroMeanEmphasisTests(unittest.TestCase):
    def test_zero_mean_embeddings_stay_finite(self):
        for values in ([[0.0, 0.0]], [[1.0, -1.0]]):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                with self.subTest(values=values, dtype=dtype):
                    state = emphasis.EmphasisOriginal()
                    state.z = torch.tensor([values], dtype=dtype)
                    state.multipliers = torch.ones(1, 1, dtype=dtype)
                    expected = state.z.clone()
                    state.after_transformers()
                    torch.testing.assert_close(state.z, expected)

    def test_nonzero_mean_preserves_existing_normalization(self):
        state = emphasis.EmphasisOriginal()
        state.z = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        state.multipliers = torch.tensor([[2.0, 0.5]])
        weighted = state.z * state.multipliers.unsqueeze(-1)
        expected = weighted * (state.z.mean() / weighted.mean())
        state.after_transformers()
        torch.testing.assert_close(state.z, expected)


if __name__ == "__main__":
    unittest.main()
