"""UMT5 tokenization preserves prompt characters and pads chunks with its pad token."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from backend.text_processing import emphasis, parsing

ROOT = Path(__file__).resolve().parents[2]


class CharacterTokenizer:
    eos = 1

    def __init__(self):
        self.calls = []

    def __call__(self, texts):
        self.calls.append(texts)

        def tokens(text):
            return [ord(character) for character in text] + [self.eos]

        return {"input_ids": tokens(texts) if isinstance(texts, str) else [tokens(text) for text in texts]}


class RecordingEncoder:
    def __init__(self):
        self.calls = []

    def __call__(self, *, input_ids, attention_mask):
        self.calls.append((input_ids.clone(), attention_mask.clone()))
        return input_ids.to(dtype=torch.float32).unsqueeze(-1)


class Umt5TokenizationTests(unittest.TestCase):
    def setUp(self):
        source = ROOT / "backend/text_processing/umt5_engine.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        namespace = {
            "torch": torch,
            "emphasis": emphasis,
            "parsing": parsing,
            "memory_management": SimpleNamespace(text_encoder_device=lambda: torch.device("cpu")),
            "dynamic_args": SimpleNamespace(last_extra_generation_params={}),
            "opts": SimpleNamespace(emphasis="Ignore"),
        }
        exec(  # noqa: S102 - execute the actual repository classes without GPU device discovery
            compile(ast.Module(body=classes, type_ignores=[]), str(source), "exec"), namespace
        )
        self.tokenizer = CharacterTokenizer()
        self.encoder = RecordingEncoder()
        self.engine = namespace["UMT5TextProcessingEngine"](
            SimpleNamespace(transformer=self.encoder), self.tokenizer, min_length=4
        )

    def test_all_prompt_characters_reach_tokenizer_and_encoder(self):
        result = self.engine(["cat"])
        self.assertEqual(self.tokenizer.calls[-1], ["cat"])
        self.assertEqual(self.encoder.calls[0][0].tolist(), [[ord("c"), ord("a"), ord("t"), 1]])
        self.assertEqual(result.shape, (1, 4, 1))

    def test_single_weighted_unicode_character_is_not_removed(self):
        chunks, count = self.engine.tokenize_line("(猫)")
        self.assertEqual(self.tokenizer.calls[-1], ["猫"])
        self.assertEqual(chunks[0].tokens, [ord("猫"), 1, 0, 0])
        self.assertEqual(chunks[0].multipliers, [1.1, 1.0, 1.0, 1.0])
        self.assertEqual(count, 2)

    def test_empty_prompt_has_one_eos_then_padding(self):
        chunks, count = self.engine.tokenize_line("")
        self.assertEqual(chunks[0].tokens, [1, 0, 0, 0])
        self.assertEqual(count, 1)
        self.assertEqual(self.engine.process_attn_mask([chunks[0].tokens]).tolist(), [[1, 0, 0, 0]])

    def test_shorter_break_chunk_uses_pad_token_and_attention_mask(self):
        self.engine.min_length = 2
        result = self.engine(["a BREAK abc"])
        self.assertEqual(result.shape, (1, 8, 1))
        first_tokens, first_mask = self.encoder.calls[0]
        second_tokens, second_mask = self.encoder.calls[1]
        self.assertEqual(first_tokens.tolist(), [[ord("a"), 1, self.engine.pad_token, self.engine.pad_token]])
        self.assertEqual(first_mask.tolist(), [[1, 1, 0, 0]])
        self.assertEqual(second_tokens.tolist(), [[ord("a"), ord("b"), ord("c"), 1]])
        self.assertEqual(second_mask.tolist(), [[1, 1, 1, 1]])
        self.assertEqual(result[0, 2:4].tolist(), [[0.0], [0.0]])

    def test_different_prompt_lengths_share_sequence_length(self):
        self.engine.min_length = 2
        result = self.engine(["a", "abcd"])
        self.assertEqual(result.shape, (2, 5, 1))
        self.assertEqual(result[0, :, 0].tolist(), [float(ord("a")), 1.0, 0.0, 0.0, 0.0])
        self.assertEqual(result[1, :, 0].tolist(), [float(ord(c)) for c in "abcd"] + [1.0])
        self.assertEqual(self.encoder.calls[0][1].tolist(), [[1, 1, 0, 0, 0]])

    def test_missing_break_chunks_use_empty_eos_chunk(self):
        self.engine.min_length = 2
        result = self.engine(["a BREAK bc", "d"])
        self.assertEqual(result.shape, (2, 6, 1))
        self.assertEqual(result[0, :, 0].tolist(), [float(ord("a")), 1.0, 0.0, float(ord("b")), float(ord("c")), 1.0])
        self.assertEqual(result[1, :, 0].tolist(), [float(ord("d")), 1.0, 0.0, 1.0, 0.0, 0.0])
        self.assertEqual(self.encoder.calls[-1][0].tolist(), [[1, 0, 0]])
        self.assertEqual(self.encoder.calls[-1][1].tolist(), [[1, 0, 0]])

    def test_repeated_prompt_reuses_encoding_with_global_padding(self):
        self.engine.min_length = 2
        result = self.engine(["a", "abcd", "a"])
        self.assertEqual(result.shape, (3, 5, 1))
        self.assertEqual(len(self.encoder.calls), 2)
        self.assertTrue(torch.equal(result[0], result[2]))


if __name__ == "__main__":
    unittest.main()
