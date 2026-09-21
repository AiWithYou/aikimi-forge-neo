"""Numerical packing, placement and cancellation checks with small real weights."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from modules_forge.qwen_image21.w4a8 import W4A8Linear, convert_model, load_model, target_linears, validate_dependency


def tiny_model():
    torch.manual_seed(921)
    model = nn.Module()
    model.transformer_blocks = nn.ModuleList([nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 256))])
    model.img_in = nn.Linear(256, 256)
    model.transformer_blocks[0].add_module("modulation", nn.Linear(256, 256))
    return model.eval().requires_grad_(False)


class W4A8Tests(unittest.TestCase):
    def test_streamed_load_matches_conversion_and_keeps_nonpersistent_buffers(self):
        from safetensors.torch import save_file

        def factory():
            model = tiny_model()
            model.register_buffer("inv_freq", torch.arange(8, dtype=torch.float32), persistent=False)
            return model

        source = factory().to(dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as directory:
            save_file(source.state_dict(), str(Path(directory) / "model.safetensors"))
            loaded, stats = load_model(directory, "transformer", factory, device="cpu")
        convert_model(source, "transformer", device="cpu")
        inputs = torch.randn(1, 4, 256, dtype=torch.bfloat16)
        torch.testing.assert_close(
            loaded.transformer_blocks[0](inputs), source.transformer_blocks[0](inputs), rtol=0, atol=0
        )
        torch.testing.assert_close(loaded.inv_freq, torch.arange(8, dtype=torch.float32))
        self.assertEqual(stats["layers"], 2)
        self.assertFalse(any(value.is_meta for value in (*loaded.parameters(), *loaded.buffers())))
        # Diffusers uses no_grad, and CUDA Linear reads weight version counters.
        # Loading parameters inside inference_mode breaks that execution path.
        self.assertFalse(any(value.is_inference() for value in (*loaded.parameters(), *loaded.buffers())))

    def test_streaming_rejects_missing_weights_before_quantizing(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as directory:
            state = tiny_model().state_dict()
            del state["img_in.weight"]
            save_file(state, str(Path(directory) / "model.safetensors"))
            with mock.patch("modules_forge.qwen_image21.w4a8._pack_weight") as pack:
                with self.assertRaisesRegex(ValueError, "Missing Qwen weights"):
                    load_model(directory, "transformer", tiny_model, device="cpu")
            pack.assert_not_called()

    def test_streaming_rejects_shape_mismatch_before_quantizing(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as directory:
            state = tiny_model().state_dict()
            state["img_in.weight"] = torch.zeros(1)
            save_file(state, str(Path(directory) / "model.safetensors"))
            with mock.patch("modules_forge.qwen_image21.w4a8._pack_weight") as pack:
                with self.assertRaisesRegex(ValueError, "shape mismatch"):
                    load_model(directory, "transformer", tiny_model, device="cpu")
            pack.assert_not_called()

    def test_packed_weights_reduce_storage_and_produce_finite_close_outputs(self):
        model = tiny_model()
        x = torch.randn(1, 4, 256)
        before = model.transformer_blocks[0](x)
        original_input = model.img_in.weight
        original_modulation = model.transformer_blocks[0].modulation.weight
        stats = convert_model(model, "transformer", device="cpu")
        after = model.transformer_blocks[0](x)
        self.assertEqual(stats["layers"], 2)
        self.assertLess(stats["packed_weight_bytes"], stats["source_weight_bytes"] * 0.3)
        self.assertTrue(torch.isfinite(after).all())
        self.assertLess(((after - before).norm() / before.norm()).item(), 0.20)
        self.assertIs(model.img_in.weight, original_input)
        self.assertIs(model.transformer_blocks[0].modulation.weight, original_modulation)
        self.assertEqual(target_linears(model, "transformer"), [])

    def test_dtype_conversion_preserves_packed_scales_and_zero_layers_stay_finite(self):
        model = tiny_model()
        with torch.no_grad():
            model.transformer_blocks[0][0].weight.zero_()
        convert_model(model, "transformer", device="cpu")
        model.to(dtype=torch.bfloat16)
        layer = model.transformer_blocks[0][0]
        self.assertIsInstance(layer, W4A8Linear)
        self.assertEqual(layer.weight._params.scale.dtype, torch.float8_e4m3fn)
        self.assertTrue(torch.isfinite(layer(torch.ones(1, 4, 256, dtype=torch.bfloat16))).all())

    def test_encoder_vision_and_language_head_remain_unmodified(self):
        model = nn.Module()
        model.model = nn.Module()
        model.model.language_model = nn.Module()
        model.model.language_model.layers = nn.ModuleList([nn.Linear(256, 256)])
        model.model.visual = nn.Linear(256, 256)
        model.lm_head = nn.Linear(256, 256)
        visual, head = model.model.visual, model.lm_head
        stats = convert_model(model, "text_encoder", device="cpu")
        self.assertEqual(stats["layers"], 1)
        self.assertIs(model.model.visual, visual)
        self.assertIs(model.lm_head, head)

    def test_cancel_stops_packing_between_layers(self):
        model = tiny_model()

        def progress(done, _total):
            if done == 1:
                raise InterruptedError("cancelled")

        with self.assertRaisesRegex(InterruptedError, "cancelled"):
            convert_model(model, "transformer", device="cpu", progress=progress)
        self.assertIsInstance(model.transformer_blocks[0][0], W4A8Linear)
        self.assertNotIsInstance(model.transformer_blocks[0][2], W4A8Linear)

    def test_empty_or_incompatible_model_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "no compatible"):
            convert_model(nn.Linear(8, 8), "transformer", device="cpu")

    def test_missing_optional_dependency_gives_targeted_setup_instruction(self):
        with mock.patch("importlib.metadata.version", return_value="0.1.0"):
            with self.assertRaisesRegex(RuntimeError, "runtime-only"):
                validate_dependency()

    @unittest.skipUnless(os.environ.get("QWEN_W4A8_GPU_TEST") == "1", "opt-in CUDA/offload test")
    def test_native_cuda_and_real_accelerate_offload_repeat_without_stale_scales(self):
        from accelerate import cpu_offload_with_hook

        model = tiny_model().to(dtype=torch.bfloat16)
        convert_model(model, "transformer")
        block, hook = cpu_offload_with_hook(model.transformer_blocks[0], execution_device="cuda:0")
        x = torch.randn(2, 32, 256, device="cuda", dtype=torch.bfloat16)
        outputs = []
        with torch.no_grad():
            for _ in range(2):
                outputs.append(block(x))
                self.assertTrue(torch.isfinite(outputs[-1]).all())
                hook.offload()
                for layer in block.modules():
                    if isinstance(layer, W4A8Linear):
                        self.assertEqual(layer.weight._qdata.device.type, "cpu")
                        self.assertTrue(all(t.device.type == "cpu" for t in layer.weight.state_dict().values()))
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)

    @unittest.skipUnless(os.environ.get("QWEN_W4A8_GPU_TEST") == "1", "opt-in dedicated Qwen runtime test")
    def test_real_qwen_components_stream_load_and_forward_under_no_grad_twice(self):
        from accelerate import cpu_offload_with_hook
        from diffusers import QwenImage21Transformer2DModel
        from safetensors.torch import save_file
        from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

        text_config = Qwen3VLConfig(
            text_config={
                "vocab_size": 64,
                "hidden_size": 256,
                "intermediate_size": 512,
                "num_hidden_layers": 1,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "head_dim": 16,
                "max_position_embeddings": 64,
                "rope_parameters": {"rope_type": "default", "mrope_section": [2, 3, 3]},
            },
            vision_config={
                "depth": 1,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_heads": 2,
                "out_hidden_size": 256,
                "patch_size": 2,
                "temporal_patch_size": 1,
                "num_position_embeddings": 16,
                "deepstack_visual_indexes": [],
            },
            image_token_id=60,
            video_token_id=61,
            vision_start_token_id=62,
            vision_end_token_id=63,
        )

        def text_factory():
            return Qwen3VLForConditionalGeneration(text_config)

        def transformer_factory():
            return QwenImage21Transformer2DModel(
                patch_size=1,
                in_channels=4,
                out_channels=4,
                num_layers=1,
                attention_head_dim=16,
                num_attention_heads=16,
                context_in_dim=256,
                mlp_ratio=2,
                axes_dims_rope=(4, 6, 6),
            )

        with tempfile.TemporaryDirectory() as temporary:
            models = {}
            for name, factory in (("text_encoder", text_factory), ("transformer", transformer_factory)):
                directory = Path(temporary) / name
                directory.mkdir()
                source = factory().to(dtype=torch.bfloat16)
                save_file(source.state_dict(), str(directory / "model.safetensors"))
                del source
                models[name], stats = load_model(directory, name, factory)
                self.assertEqual(stats["layers"], 7)
            encoder, encoder_hook = cpu_offload_with_hook(models["text_encoder"], execution_device="cuda:0")
            transformer, transformer_hook = cpu_offload_with_hook(
                models["transformer"],
                execution_device="cuda:0",
                prev_module_hook=encoder_hook,
            )
            inputs = torch.randn(1, 4, 4, device="cuda", dtype=torch.bfloat16)
            outputs = []
            with torch.no_grad():
                for _ in range(2):
                    encoded = encoder(
                        input_ids=torch.tensor([[1, 2, 3, 4]], device="cuda"),
                        attention_mask=torch.ones(1, 4, device="cuda", dtype=torch.long),
                        output_hidden_states=True,
                    ).hidden_states[-1]
                    output = transformer(
                        hidden_states=inputs,
                        encoder_hidden_states=encoded,
                        encoder_hidden_states_mask=torch.ones(1, 4, device="cuda", dtype=torch.long),
                        timestep=torch.tensor([0.4], device="cuda", dtype=torch.bfloat16),
                        img_shapes=[[(1, 2, 2)]],
                        img_mask=torch.tensor([[False, False, False, False, True]], device="cuda"),
                        return_dict=False,
                    )[0]
                    self.assertTrue(torch.isfinite(output).all())
                    outputs.append(output)
                    transformer_hook.offload()
                    self.assertTrue(all(p.device.type == "cpu" for p in encoder.parameters()))
                    self.assertTrue(all(p.device.type == "cpu" for p in transformer.parameters()))
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
