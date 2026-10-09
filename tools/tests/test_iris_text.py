"""Packed Qwen must preserve Iris's prompt template and selected states."""

import unittest
from types import SimpleNamespace

import torch

from modules_forge.iris.text import initialize_encoder


class PackedTextTests(unittest.TestCase):
    def test_assistant_suffix_budget_pad_fallback_and_hidden_states(self):
        tokenizer = SimpleNamespace(
            pad_token_id=None,
            eos_token_id=9,
            encode=lambda value, **kwargs: [1, 2] if value == "prefix" else [3, 4],
        )
        cfg = SimpleNamespace(dim=256, max_length=6, hidden_layers=[1, 2], on_caption_overflow="error")
        decoder = SimpleNamespace(config=SimpleNamespace(hidden_size=256, num_hidden_layers=2))
        encoder = SimpleNamespace()
        initialize_encoder(encoder, cfg, tokenizer, decoder, "prefix", "suffix")
        self.assertEqual(encoder.caption_budget, 4)
        self.assertEqual(encoder._pad_id, 9)
        self.assertEqual(encoder.hidden_layers, (1, 2))
        torch.testing.assert_close(encoder._suffix_ids, torch.tensor([3, 4]))
        self.assertIs(encoder.decoder, decoder)
        self.assertEqual(encoder.device, torch.device("cpu"))

    def test_rejects_incompatible_decoder_and_exhausted_caption_budget(self):
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=9, encode=lambda *args, **kwargs: [1, 2])
        decoder = SimpleNamespace(config=SimpleNamespace(hidden_size=256, num_hidden_layers=2))
        for dim, length, layers in ((512, 6, [1]), (256, 2, [1]), (256, 6, [3])):
            with self.subTest(dim=dim, length=length, layers=layers), self.assertRaises(ValueError):
                cfg = SimpleNamespace(dim=dim, max_length=length, hidden_layers=layers, on_caption_overflow="warn")
                initialize_encoder(SimpleNamespace(), cfg, tokenizer, decoder, "prefix", "suffix")


if __name__ == "__main__":
    unittest.main()
