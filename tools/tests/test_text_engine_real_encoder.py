"""Tiny real T5 encoders check interfaces and finite values, not image quality."""

import types
import unittest

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from backend.nn.t5 import T5, T5LayerNorm
from backend.text_processing import emphasis, parsing
from tools.tests.test_api_extras_boundaries import load_classes


def tiny_tokenizer():
    pad, eos, unknown = "<pad>", "</s>", "<unk>"
    vocabulary = {pad: 0, eos: 1, unknown: 2, "a": 3, "b": 4, "c": 5, "猫": 6}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token=unknown))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(single=f"$A {eos}", special_tokens=[(eos, 1)])
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token=eos, pad_token=pad, unk_token=unknown)


def tiny_encoder(kind, seed=61):
    config = {
        "num_layers": 2,
        "d_model": 8,
        "d_ff": 16,
        "dense_act_fn": "relu",
        "is_gated_act": False,
        "num_heads": 2,
        "model_type": kind,
        "vocab_size": 8,
    }
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = T5(config).eval()
        # Production loads these torch.empty parameters from checkpoints.
        for module in model.modules():
            if isinstance(module, T5LayerNorm):
                torch.nn.init.ones_(module.weight)
    return model


class RecordingRealEncoder(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.calls = []

    def forward(self, *, input_ids, attention_mask=None):
        self.calls.append((input_ids.clone(), attention_mask.clone() if attention_mask is not None else None))
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask)


def engine_for(kind, dtype=torch.float32):
    name = "T5TextProcessingEngine" if kind == "t5" else "UMT5TextProcessingEngine"
    opts = types.SimpleNamespace(emphasis="Original")
    namespace = {
        "torch": torch,
        "emphasis": emphasis,
        "parsing": parsing,
        "memory_management": types.SimpleNamespace(text_encoder_device=lambda: torch.device("cpu")),
        "dynamic_args": types.SimpleNamespace(last_extra_generation_params={}),
        "opts": opts,
    }
    module = load_classes(f"backend/text_processing/{kind}_engine.py", {"PromptChunk", name}, namespace)
    encoder = RecordingRealEncoder(tiny_encoder(kind).to(dtype=dtype))
    engine = getattr(module, name)(types.SimpleNamespace(transformer=encoder), tiny_tokenizer(), min_length=2)
    return engine, encoder, opts


class RealTextEncoderTests(unittest.TestCase):
    @torch.inference_mode()
    def test_real_zero_encoder_and_weighted_breaks_keep_original_dtype(self):
        for kind in ("t5", "umt5"):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                with self.subTest(kind=kind, dtype=dtype):
                    engine, encoder, _ = engine_for(kind, dtype)
                    encoder.encoder.shared.weight.zero_()
                    actual = engine(["(a:2) BREAK (b:0)", "", "a"])
                    self.assertEqual(actual.shape, (3, 4, 8))
                    self.assertEqual(actual.dtype, dtype)
                    torch.testing.assert_close(actual, torch.zeros_like(actual))

    @torch.inference_mode()
    def test_mixed_break_empty_and_zero_weight_prompts_preserve_shape_and_eos_masks(self):
        prompts = ["a BREAK b c", "", "a BREAK b c", "BREAK", "(猫:0)", "a"]
        for kind in ("t5", "umt5"):
            with self.subTest(kind=kind):
                engine, encoder, _ = engine_for(kind)
                actual = engine(prompts)
                self.assertEqual(actual.shape, (6, 6, 8))
                self.assertEqual(actual.dtype, torch.float32)
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual[0], actual[2])
                self.assertEqual(len(encoder.calls), 10)
                for tokens, mask in encoder.calls:
                    row = tokens[0].tolist()
                    eos = row.index(1)
                    self.assertEqual(row[eos + 1 :], [0] * (len(row) - eos - 1))
                    if kind == "umt5":
                        self.assertEqual(mask[0].tolist(), [1] * (eos + 1) + [0] * (len(row) - eos - 1))

    @torch.inference_mode()
    def test_repeated_calls_recompute_after_encoder_and_emphasis_change(self):
        for kind in ("t5", "umt5"):
            with self.subTest(kind=kind):
                engine, encoder, opts = engine_for(kind)
                prompts = ["(a:2)", "", "(a:2)"]
                first = engine(prompts)
                again = engine(prompts)
                torch.testing.assert_close(again, first)
                self.assertEqual(len(encoder.calls), 4)
                replacement = RecordingRealEncoder(tiny_encoder(kind, seed=73))
                engine.text_encoder = replacement
                changed = engine(prompts)
                self.assertEqual(changed.shape, first.shape)
                self.assertTrue(torch.isfinite(changed).all())
                self.assertEqual(len(replacement.calls), 2)
                self.assertFalse(torch.equal(changed, first))
                opts.emphasis = "Ignore"
                without_weights = engine(prompts)
                self.assertEqual(without_weights.shape, changed.shape)
                self.assertTrue(torch.isfinite(without_weights).all())
                self.assertFalse(torch.equal(without_weights, changed))


class EmphasisPrecisionReviewTests(unittest.TestCase):
    def test_uniform_half_weights_preserve_values_when_half_product_would_underflow(self):
        state = emphasis.EmphasisOriginal()
        state.z = torch.tensor([[[0.0001, 0.0002], [0.0003, 0.0004]]], dtype=torch.float16)
        state.multipliers = torch.full((1, 2), 1e-5, dtype=torch.float16)
        original = state.z.clone()
        self.assertTrue(torch.equal(state.z * state.multipliers.unsqueeze(-1), torch.zeros_like(state.z)))
        state.after_transformers()
        self.assertEqual(state.z.dtype, torch.float16)
        torch.testing.assert_close(state.z, original)

    def test_nonuniform_weights_preserve_nonzero_mean_and_existing_precision(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
            with self.subTest(dtype=dtype):
                state = emphasis.EmphasisOriginal()
                state.z = torch.tensor([[[1, 2], [3, 4]]], dtype=dtype)
                state.multipliers = torch.tensor([[2, 0.5]], dtype=dtype)
                state.after_transformers()
                expected = torch.tensor([[[40 / 19, 80 / 19], [30 / 19, 40 / 19]]], dtype=dtype)
                self.assertEqual(state.z.dtype, dtype)
                torch.testing.assert_close(state.z, expected)

    def test_double_precision_input_is_not_rounded_to_single_precision(self):
        state = emphasis.EmphasisOriginal()
        state.z = torch.tensor([[[1.0000000002, 2.0000000003]]], dtype=torch.float64)
        state.multipliers = torch.full((1, 1), 2.0, dtype=torch.float64)
        original = state.z.clone()
        state.after_transformers()
        torch.testing.assert_close(state.z, original, rtol=1e-12, atol=1e-12)

    def test_zero_mean_with_nonuniform_or_zero_weights_stays_finite(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for weights, expected in (([1, 2], [[1, -1], [4, -4]]), ([0, 0], [[0, 0], [0, 0]])):
                with self.subTest(dtype=dtype, weights=weights):
                    state = emphasis.EmphasisOriginal()
                    state.z = torch.tensor([[[1, -1], [2, -2]]], dtype=dtype)
                    state.multipliers = torch.tensor([weights], dtype=dtype)
                    state.after_transformers()
                    self.assertEqual(state.z.dtype, dtype)
                    torch.testing.assert_close(state.z, torch.tensor([expected], dtype=dtype))

    def test_small_finite_half_weights_do_not_overflow_normalization(self):
        state = emphasis.EmphasisOriginal()
        state.z = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float16)
        state.multipliers = torch.full((1, 2), 1e-5, dtype=torch.float16)
        original = state.z.clone()
        state.after_transformers()
        self.assertEqual(state.z.dtype, torch.float16)
        self.assertTrue(torch.isfinite(state.z).all())
        torch.testing.assert_close(state.z, original, atol=0.01, rtol=0.01)


if __name__ == "__main__":
    unittest.main()
