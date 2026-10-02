"""Exercise post-context CPU weight loading with real operation classes."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from backend import operations
from backend.patcher.base import OnlineLoRAPatch
from backend.patcher.controlnet import ControlLoraOps
from backend.quant_ops import QuantizedTensor


class OperationContextIntegrationTests(unittest.TestCase):
    def roundtrip_state(self, state):
        with tempfile.TemporaryDirectory() as temporary:
            filename = Path(temporary) / "小さい 重み.pt"
            torch.save(state, filename)
            return torch.load(filename, map_location="cpu", weights_only=True)

    def test_linear_and_embedding_load_after_nested_contexts_exit(self):
        linear_state = self.roundtrip_state(
            {"weight": torch.arange(12, dtype=torch.float32).reshape(3, 4) / 7, "bias": torch.tensor([1.0, 2.0, 3.0])}
        )
        embedding_state = self.roundtrip_state({"weight": torch.arange(20, dtype=torch.float32).reshape(5, 4) / 5})
        native_linear = torch.nn.Linear
        native_embedding = torch.nn.Embedding
        for quantization in (None, "gguf"):
            with self.subTest(quantization=quantization):
                with operations.using_forge_operations(device="cpu", dtype=torch.float64, manual_cast_enabled=True):
                    with operations.using_forge_operations(device="cpu", dtype=torch.float32, bnb_dtype=quantization):
                        linear = torch.nn.Linear(4, 3)
                        embedding = torch.nn.Embedding(5, 4)
                    outer = torch.nn.Linear(4, 3)
                self.assertIs(torch.nn.Linear, native_linear)
                self.assertIs(torch.nn.Embedding, native_embedding)
                self.assertEqual(outer.weight.dtype, torch.float64)
                for layer, state in ((linear, linear_state), (embedding, embedding_state), (outer, linear_state)):
                    loaded = layer.load_state_dict(state)
                    self.assertEqual(loaded.missing_keys, [])
                    self.assertEqual(loaded.unexpected_keys, [])
                values = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 3
                expected = torch.nn.functional.linear(values, **linear_state)
                torch.testing.assert_close(linear(values), expected)
                torch.testing.assert_close(outer(values.double()), expected.double(), rtol=1e-6, atol=1e-6)
                tokens = torch.tensor([[0, 4], [3, 1]])
                torch.testing.assert_close(
                    embedding(tokens), torch.nn.functional.embedding(tokens, embedding_state["weight"])
                )

    def test_reused_te_config_loads_real_int8_linear_and_embedding(self):
        configuration = {"TE": True, "linear": {"format": "int8_tensorwise"}}
        quant_conf = torch.tensor(list(json.dumps({"format": "int8_tensorwise"}).encode()), dtype=torch.uint8)
        weight = torch.arange(-10, 10, dtype=torch.int8).reshape(5, 4)
        scale = torch.tensor(0.25)
        linear_state = self.roundtrip_state(
            {
                "weight": weight,
                "weight_scale": scale,
                "comfy_quant": quant_conf,
                "bias": torch.arange(5, dtype=torch.float32),
            }
        )
        embedding_state = self.roundtrip_state({"weight": weight, "weight_scale": scale, "comfy_quant": quant_conf})
        with (
            patch.object(operations.memory_management, "get_torch_device", return_value=torch.device("cpu")),
            patch.object(operations.memory_management, "should_use_bf16", return_value=False),
            patch.object(operations.memory_management, "supports_fp8_compute", return_value=False),
            patch.object(operations.memory_management, "supports_nvfp4_compute", return_value=False),
            patch.object(operations.memory_management, "supports_mxfp8_compute", return_value=False),
        ):
            for _ in range(2):
                with operations.using_forge_operations(device="cpu", dtype=torch.float32, bnb_dtype=configuration):
                    linear = torch.nn.Linear(4, 5)
                    embedding = torch.nn.Embedding(5, 4)
                self.assertTrue(linear._full_precision_mm)
                for layer, state in ((linear, linear_state), (embedding, embedding_state)):
                    loaded = layer.load_state_dict(state)
                    self.assertEqual(loaded.missing_keys, [])
                    self.assertEqual(loaded.unexpected_keys, [])
                values = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 3
                with torch.no_grad():
                    torch.testing.assert_close(
                        linear(values), torch.nn.functional.linear(values, weight.float() * scale, linear_state["bias"])
                    )
                    tokens = torch.tensor([[4, 0], [1, 3]])
                    torch.testing.assert_close(
                        embedding(tokens), torch.nn.functional.embedding(tokens, weight.float() * scale)
                    )
        self.assertEqual(configuration, {"TE": True, "linear": {"format": "int8_tensorwise"}})

    def test_control_lora_custom_operations_forward_after_context_exit(self):
        with operations.using_forge_operations(device="cpu", dtype=torch.float64):
            with operations.using_forge_operations(operations=ControlLoraOps, dtype=torch.float32):
                linear = torch.nn.Linear(4, 3)
                convolution = torch.nn.Conv2d(4, 3, 1)
        state = self.roundtrip_state(
            {
                "weight": torch.arange(12, dtype=torch.float32).reshape(3, 4) / 7,
                "bias": torch.arange(3, dtype=torch.float32),
                "up": torch.arange(6, dtype=torch.float32).reshape(3, 2) / 5,
                "down": torch.arange(8, dtype=torch.float32).reshape(2, 4) / 11,
            }
        )
        for layer in (linear, convolution):
            for name, value in state.items():
                if layer is convolution and name != "bias":
                    value = value[:, :, None, None]
                setattr(layer, name, torch.nn.Parameter(value, requires_grad=False))
        values = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 3
        weight = state["weight"] + state["up"] @ state["down"]
        torch.testing.assert_close(linear(values), torch.nn.functional.linear(values, weight, state["bias"]))
        image = values[:, :, None, None]
        torch.testing.assert_close(
            convolution(image), torch.nn.functional.conv2d(image, weight[:, :, None, None], state["bias"])
        )

    def test_full_precision_keeps_dense_math_and_other_int8_paths_keep_dynamic_math(self):
        weight = torch.arange(-10, 10, dtype=torch.int8).reshape(5, 4)
        scale = torch.tensor(0.25)
        bias = torch.arange(5, dtype=torch.float32)
        values = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 3
        delta = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 13
        row_scale = values.abs().amax(dim=-1, keepdim=True) / 127
        quantized_values = (values / row_scale).round().clamp(-127, 127)
        dynamic_expected = (quantized_values @ weight.float().T) * (row_scale * scale) + bias
        dense_expected = torch.nn.functional.linear(values, weight.float() * scale, bias)
        self.assertGreater((dense_expected - dynamic_expected).abs().max().item(), 0.03)
        for full_precision, layer_override in ((False, False), (True, False), (False, True)):
            metadata = {"format": "int8_tensorwise", "full_precision_matrix_mult": layer_override}
            state = self.roundtrip_state(
                {
                    "weight": weight,
                    "weight_scale": scale,
                    "bias": bias,
                    "comfy_quant": torch.tensor(list(json.dumps(metadata).encode()), dtype=torch.uint8),
                }
            )
            factory = operations.mixed_precision_ops(compute_dtype=torch.float32, full_precision_mm=full_precision)
            for online_lora in (False, True):
                with self.subTest(
                    full_precision=full_precision, layer_override=layer_override, online_lora=online_lora
                ):
                    linear = factory.Linear(4, 5, device="cpu")
                    linear.load_state_dict(state)
                    if online_lora:
                        linear.weight_function = [OnlineLoRAPatch("weight", (0.5, (delta,), 1.0, None, None))]
                        expected = torch.nn.functional.linear(values, weight.float() * scale + delta * 0.5, bias)
                    else:
                        expected = dense_expected if full_precision or layer_override else dynamic_expected
                    with torch.no_grad():
                        torch.testing.assert_close(linear(values), expected)
                    self.assertIsInstance(linear.weight, QuantizedTensor)
                    torch.testing.assert_close(linear.weight._qdata, weight)

    def test_full_precision_dequantization_failure_waits_for_original_transfer_signal(self):
        from backend import operations_mixed_precision

        factory = operations.mixed_precision_ops(compute_dtype=torch.float32, full_precision_mm=True)
        linear = factory.Linear(4, 5, bias=False, device="cpu")
        linear.load_state_dict(
            self.roundtrip_state(
                {
                    "weight": torch.arange(-10, 10, dtype=torch.int8).reshape(5, 4),
                    "weight_scale": torch.tensor(0.25),
                    "comfy_quant": torch.tensor(list(b'{"format":"int8_tensorwise"}'), dtype=torch.uint8),
                }
            )
        )
        manual_cast = operations_mixed_precision.weights_manual_cast
        offload = Mock()
        current = object()

        def transfer_with_signal(*args, **kwargs):
            weight, bias, _ = manual_cast(*args, **kwargs)
            return weight, bias, (offload, weight, bias)

        error = RuntimeError("dequantization failed")
        with (
            patch.object(operations_mixed_precision, "weights_manual_cast", side_effect=transfer_with_signal),
            patch.object(operations.memory_management, "current_stream", return_value=current),
            patch.object(QuantizedTensor, "dequantize", side_effect=error),
            torch.no_grad(),
        ):
            with self.assertRaises(RuntimeError) as raised:
                linear(torch.ones(2, 4))
        self.assertIs(raised.exception, error)
        offload.wait_stream.assert_called_once_with(current)

    def test_tiny_clipvision_load_and_forward_match_native_layers(self):
        from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

        from backend.patcher import clipvision

        configuration = {
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 4,
            "patch_size": 2,
            "projection_dim": 4,
            "_attn_implementation": "eager",
        }
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(123)
            reference = CLIPVisionModelWithProjection(CLIPVisionConfig(**configuration)).eval()
        state = self.roundtrip_state(reference.state_dict())
        with (
            patch.object(clipvision.memory_management, "text_encoder_device", return_value=torch.device("cpu")),
            patch.object(clipvision.memory_management, "text_encoder_offload_device", return_value=torch.device("cpu")),
            patch.object(clipvision.memory_management, "should_use_fp16", return_value=False),
            operations.using_forge_operations(device="cpu", dtype=torch.float64),
        ):
            loaded_clip = clipvision.ClipVisionModel(configuration)
        loaded = loaded_clip.load_sd(state)
        self.assertEqual(loaded.missing_keys, [])
        self.assertEqual(loaded.unexpected_keys, [])
        pixels = torch.arange(96, dtype=torch.float32).reshape(2, 3, 4, 4) / 100
        with torch.no_grad():
            expected = reference(pixel_values=pixels, output_hidden_states=True)
            actual = loaded_clip.model.eval()(pixel_values=pixels, output_hidden_states=True)
        torch.testing.assert_close(actual.image_embeds, expected.image_embeds)
        torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state)


if __name__ == "__main__":
    unittest.main()
