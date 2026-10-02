"""Small CPU fixtures for native, Forge, and quantized Llama token embeddings."""

import json
import unittest
from unittest.mock import patch

import torch

from backend import operations
from backend.nn.llm import llama
from backend.quant_ops import QuantizedTensor


class LlamaEmbeddingDtypeTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(operations, "current_device", torch.device("cpu")))
        self.enterContext(patch.object(operations, "current_manual_cast_enabled", False))
        self.tokens = torch.tensor([[1, 2]], dtype=torch.long, device="cpu")
        self.weights = torch.arange(-16, 16, dtype=torch.float32, device="cpu").reshape(8, 4) / 4
        config = llama.Qwen3_06BConfig(vocab_size=8, hidden_size=4, num_hidden_layers=0, final_norm=False)
        config.head_dim = 4
        self.model = llama.Llama2_(config)

    def assert_embedding_output(self, embedding):
        self.model.embed_tokens = embedding
        for dtype in (None, torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                actual, intermediate = self.model(self.tokens, dtype=dtype)
                expected = self.weights[self.tokens].to(dtype=dtype)

                self.assertEqual(actual.dtype, expected.dtype)
                self.assertEqual(actual.device.type, "cpu")
                self.assertIsNone(intermediate)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_native_embedding_accepts_requested_output_dtype(self):
        embedding = torch.nn.Embedding(8, 4, device="cpu", dtype=torch.float32)
        embedding.load_state_dict({"weight": self.weights})

        self.assert_embedding_output(embedding)
        self.assertEqual(embedding.weight.dtype, torch.float32)

    def test_forge_embedding_accepts_requested_output_dtype(self):
        embedding = operations.ForgeOperations.Embedding(8, 4, dtype=torch.float32)
        embedding.load_state_dict({"weight": self.weights})

        self.assert_embedding_output(embedding)
        self.assertEqual(embedding.weight.dtype, torch.float32)

    def test_int8_embedding_preserves_quantization_and_requested_output_dtype(self):
        mixed_ops = operations.mixed_precision_ops(compute_dtype=torch.float32)
        embedding = mixed_ops.Embedding(8, 4, dtype=torch.float32)
        metadata = json.dumps({"format": "int8_tensorwise"}).encode("utf-8")
        embedding.load_state_dict(
            {
                "weight": (self.weights * 4).to(dtype=torch.int8),
                "weight_scale": torch.tensor(0.25, dtype=torch.float32),
                "comfy_quant": torch.tensor(list(metadata), dtype=torch.uint8),
            }
        )
        self.assertIsInstance(embedding.weight, QuantizedTensor)

        self.assert_embedding_output(embedding)
        self.assertIsInstance(embedding.weight, QuantizedTensor)
        self.assertEqual(embedding.weight._qdata.dtype, torch.int8)


if __name__ == "__main__":
    unittest.main()
