import sys
import unittest

import torch

sys.argv = sys.argv[:1]

from backend.text_processing.qwen3vl_engine import Qwen3VLTextProcessingEngine


VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"


class RecordingTokenizer:
    def __init__(self, forced_image_tokens=None):
        self.calls = []
        self.forced_image_tokens = forced_image_tokens

    def __call__(self, texts):
        self.calls.append(list(texts))
        input_ids = []
        for text in texts:
            count = (
                text.count("<|image_pad|>")
                if self.forced_image_tokens is None
                else self.forced_image_tokens
            )
            input_ids.append([1, *([151655] * count)])
        return {"input_ids": input_ids}


class Qwen3VLTokenizeTests(unittest.TestCase):
    def make_engine(self, tokenizer=None):
        tokenizer = tokenizer or RecordingTokenizer()
        engine = object.__new__(Qwen3VLTextProcessingEngine)
        engine.tokenizer = tokenizer
        engine.llama_template = "text:{}"
        engine.vision_block = VISION_BLOCK
        engine.id_template = 151644
        engine.id_image = 151655
        return engine, tokenizer

    def test_empty_text_uses_nonempty_text_template(self):
        engine, tokenizer = self.make_engine()

        tokenized = engine.tokenize([""])

        self.assertGreater(len(tokenized[0]), 0)
        self.assertEqual(["text: "], tokenizer.calls[0])

    def test_empty_text_with_images_uses_llama_template_and_vision_block(self):
        engine, tokenizer = self.make_engine()

        tokenized = engine.tokenize([""], images=[object()])

        self.assertGreater(len(tokenized[0]), 0)
        self.assertEqual([f"text:{VISION_BLOCK}"], tokenizer.calls[0])

    def test_image_prompt_keeps_full_llama_template(self):
        tokenizer = RecordingTokenizer()
        engine = Qwen3VLTextProcessingEngine(text_encoder=object(), tokenizer=tokenizer)

        engine.tokenize(["a red cat"], images=[object()])

        rendered = tokenizer.calls[0][0]
        self.assertTrue(rendered.startswith("<|im_start|>system\nDescribe the image"))
        self.assertIn(
            f"<|im_start|>user\n{VISION_BLOCK}a red cat<|im_end|>",
            rendered,
        )
        self.assertTrue(rendered.endswith("<|im_start|>assistant\n"))

    def test_image_arguments_do_not_leak_between_calls(self):
        engine, tokenizer = self.make_engine()

        engine.tokenize(["first"], images=[object(), object()])
        engine.tokenize(["second"])

        self.assertEqual(
            f"text:{VISION_BLOCK}{VISION_BLOCK}first",
            tokenizer.calls[0][0],
        )
        self.assertEqual("text:second", tokenizer.calls[1][0])

    def test_image_parameters_use_none_defaults(self):
        self.assertEqual((None,), Qwen3VLTextProcessingEngine.tokenize.__defaults__)
        self.assertEqual((None,), Qwen3VLTextProcessingEngine.tokenize_line.__defaults__)
        self.assertEqual((None,), Qwen3VLTextProcessingEngine.__call__.__defaults__)

    def test_plain_prompt_is_templated_once(self):
        engine, tokenizer = self.make_engine()

        chunks = engine.tokenize_line("a red cat")

        self.assertEqual([["text:a red cat"]], tokenizer.calls)
        self.assertEqual(1, len(chunks))
        self.assertEqual([1.0] * len(chunks[0].tokens), chunks[0].multipliers)

    def test_emphasis_syntax_is_literal_and_templated_once(self):
        engine, tokenizer = self.make_engine()

        chunks = engine.tokenize_line("a (red:1.2) cat")

        self.assertEqual([["text:a (red:1.2) cat"]], tokenizer.calls)
        self.assertEqual(1, len(chunks))
        self.assertEqual([1.0] * len(chunks[0].tokens), chunks[0].multipliers)

    def test_image_prompt_is_templated_once_and_replaces_each_image_token(self):
        engine, tokenizer = self.make_engine()
        images = [object(), object()]

        chunks = engine.tokenize_line("a (red:1.2) cat", images=images)

        self.assertEqual(
            [[f"text:{VISION_BLOCK}{VISION_BLOCK}a (red:1.2) cat"]],
            tokenizer.calls,
        )
        image_tokens = [
            token for token in chunks[0].tokens if isinstance(token, dict)
        ]
        self.assertEqual(2, len(image_tokens))
        self.assertIs(images[0], image_tokens[0]["data"])
        self.assertIs(images[1], image_tokens[1]["data"])
        self.assertEqual([1.0] * len(chunks[0].tokens), chunks[0].multipliers)

    def test_more_image_tokens_than_images_fails(self):
        engine, _ = self.make_engine(RecordingTokenizer(forced_image_tokens=2))

        with self.assertRaisesRegex(ValueError, "more image tokens"):
            engine.tokenize_line("cat", images=[object()])

    def test_fewer_image_tokens_than_images_fails(self):
        engine, _ = self.make_engine(RecordingTokenizer(forced_image_tokens=1))

        with self.assertRaisesRegex(ValueError, "fewer image tokens"):
            engine.tokenize_line("cat", images=[object(), object()])

    def test_empty_line_keeps_nonempty_template(self):
        engine, tokenizer = self.make_engine()

        chunks = engine.tokenize_line("")

        self.assertEqual([["text: "]], tokenizer.calls)
        self.assertEqual(1, len(chunks))
        self.assertEqual([1.0] * len(chunks[0].tokens), chunks[0].multipliers)


class Qwen3VLConditioningShapeTests(unittest.TestCase):
    def make_engine(self):
        tokenizer = RecordingTokenizer()
        engine = object.__new__(Qwen3VLTextProcessingEngine)
        engine.tokenizer = tokenizer
        engine.llama_template = "text:{}"
        engine.vision_block = VISION_BLOCK
        engine.id_template = 151644
        engine.id_image = 151655
        return engine

    def test_call_returns_sequence_layer_hidden_shape(self):
        engine = self.make_engine()
        calls = []

        def process_tokens(batch_tokens, batch_multipliers):
            calls.append((batch_tokens, batch_multipliers))
            taps = torch.arange(12, dtype=torch.float32).reshape(1, 12, 1, 1)
            return taps.expand(1, 12, 3, 2560).clone()

        engine.process_tokens = process_tokens

        conditioning = engine(["cat", "cat"])

        self.assertEqual(1, len(calls), "duplicate prompt should use the per-call cache")
        self.assertEqual(2, len(conditioning))
        self.assertEqual((3, 12, 2560), tuple(conditioning[0].shape))
        self.assertTrue(
            torch.equal(
                conditioning[0][:, :, 0],
                torch.arange(12).repeat(3, 1),
            )
        )
        self.assertIs(conditioning[0], conditioning[1])

    def test_call_rejects_invalid_qwen_tap_shape(self):
        engine = self.make_engine()
        engine.process_tokens = lambda *_: torch.zeros(1, 11, 2, 2560)

        with self.assertRaisesRegex(ValueError, "12x2560"):
            engine(["cat"])


if __name__ == "__main__":
    unittest.main()
