"""Additional CPU review boundaries for attention masks and clone lifetimes."""

import gc
import types
import unittest
import weakref
from unittest.mock import Mock, patch

import torch

from backend import attention, memory_management
from backend.patcher.base import ModelPatcher
from backend.patcher.vae import VAE
from tools.tests.test_vae_clone import TinyAutoencoder


class AttentionReviewTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator(device="cpu").manual_seed(24)
        self.q = torch.randn(3, 4, 5, 4, generator=generator)
        self.k = torch.randn(3, 2, 7, 4, generator=generator)
        self.v = torch.randn(3, 2, 7, 4, generator=generator)

    def test_split_gqa_keeps_broadcast_masks_and_nondefault_scale(self):
        masks = (torch.randn(5, 7), torch.rand(1, 4, 1, 7) > 0.4)
        with patch.object(attention, "SDP_BATCH_LIMIT", 2):
            for mask in masks:
                expected = torch.nn.functional.scaled_dot_product_attention(
                    self.q, self.k, self.v, attn_mask=mask, enable_gqa=True, scale=0.25
                )
                for skip_reshape in (False, True):
                    inputs = (
                        (self.q, self.k, self.v)
                        if skip_reshape
                        else tuple(t.transpose(1, 2).flatten(2) for t in (self.q, self.k, self.v))
                    )
                    for skip_output_reshape in (False, True):
                        with self.subTest(mask=mask.shape, skip_reshape=skip_reshape, output=skip_output_reshape):
                            actual = attention.attention_pytorch(
                                *inputs,
                                heads=4,
                                mask=mask,
                                enable_gqa=True,
                                scale=0.25,
                                skip_reshape=skip_reshape,
                                skip_output_reshape=skip_output_reshape,
                            )
                            reference = expected if skip_output_reshape else expected.transpose(1, 2).flatten(2)
                            torch.testing.assert_close(actual, reference)

    def test_xformers_success_and_failure_keep_boolean_mask_semantics(self):
        # Anima's LLM adapter supplies this bool [batch, 1, 1, key] padding mask.
        q, k, v = self.q[:, :2], self.k, self.v
        mask = torch.tensor([[[[True, False, True, False, False, True, False]]]])
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        def cpu_xformers(q, k, v, attn_bias):
            # Preserve xformers' NHD layout and additive-mask contract on CPU.
            return torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=attn_bias
            ).transpose(1, 2)

        for failure in (False, True):
            operation = Mock(side_effect=RuntimeError("fixture") if failure else cpu_xformers)
            stand_in = types.SimpleNamespace(ops=types.SimpleNamespace(memory_efficient_attention=operation))
            with self.subTest(failure=failure), patch.object(attention, "xformers", stand_in, create=True):
                actual = attention.attention_xformers(q, k, v, heads=2, mask=mask, skip_reshape=True)
                torch.testing.assert_close(actual, expected.transpose(1, 2).flatten(2))

    def test_xformers_failure_keeps_grouped_heads_after_layout_conversion(self):
        failing = types.SimpleNamespace(memory_efficient_attention=Mock(side_effect=RuntimeError("fixture")))
        mask = torch.randn(1, 1, 5, 7)
        expected = torch.nn.functional.scaled_dot_product_attention(
            self.q, self.k, self.v, attn_mask=mask, enable_gqa=True
        )
        with patch.object(attention, "xformers", types.SimpleNamespace(ops=failing), create=True):
            actual = attention.attention_xformers(
                self.q,
                self.k,
                self.v,
                heads=4,
                mask=mask,
                enable_gqa=True,
                skip_reshape=True,
                skip_output_reshape=True,
            )
        torch.testing.assert_close(actual, expected)


class ModelLifetimeReviewTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(memory_management, "current_loaded_models", []))
        self.enterContext(patch("modules_forge.gpu_residency.release_resource"))
        self.cpu = torch.device("cpu")

    def test_three_clone_generations_fall_back_then_release_their_wrapper(self):
        root = ModelPatcher(torch.nn.Linear(2, 2), self.cpu, self.cpu)
        first = root.clone()
        second = first.clone()
        third = second.clone()
        loaded = memory_management.LoadedModel(third)
        del third
        gc.collect()
        self.assertIs(loaded.model, second)
        del second
        gc.collect()
        self.assertIs(loaded.model, first)
        del first
        gc.collect()
        self.assertIs(loaded.model, root)
        self.assertIsNone(loaded._patcher_finalizer)
        reference = weakref.ref(loaded)
        del loaded
        gc.collect()
        self.assertIsNone(reference())

    def test_real_cpu_model_switch_reclaims_dead_wrapper_before_reloading_same_module(self):
        module_a, module_b = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
        model_a = ModelPatcher(module_a, self.cpu, self.cpu)
        model_b = ModelPatcher(module_b, self.cpu, self.cpu)
        memory_management.load_models_gpu([model_a], force_full_load=True)
        loaded_a = weakref.ref(memory_management.current_loaded_models[0])
        memory_management.load_models_gpu([model_b], force_full_load=True)
        del model_a
        gc.collect()
        replacement_a = ModelPatcher(module_a, self.cpu, self.cpu)
        memory_management.load_models_gpu([replacement_a], force_full_load=True)
        gc.collect()
        self.assertIsNone(loaded_a())
        loaded = memory_management.current_loaded_models
        self.assertEqual(len(loaded), 2)
        self.assertEqual(sum(item.model is replacement_a for item in loaded), 1)
        self.assertEqual(sum(item.model is model_b for item in loaded), 1)


class VaeLifetimeReviewTests(unittest.TestCase):
    @torch.inference_mode()
    def test_video_clone_survives_parent_gc_and_matches_direct_multi_tile_results(self):
        cpu = torch.device("cpu")
        with (
            patch("backend.memory_management.load_models_gpu"),
            patch("backend.memory_management.vae_offload_device", return_value=cpu),
            patch("backend.memory_management.intermediate_device", return_value=cpu),
        ):
            model = TinyAutoencoder(video=True)
            parent = VAE(model=model, device=cpu, dtype=torch.float32, is_wan=True)
            intermediate = parent.clone()
            clone = intermediate.clone()
            parent_ref, intermediate_ref = weakref.ref(parent), weakref.ref(intermediate)
            del parent, intermediate
            gc.collect()
            self.assertIsNone(parent_ref())
            self.assertIsNone(intermediate_ref())
            pixels = torch.linspace(0, 1, 9 * 40 * 48 * 3).reshape(9, 40, 48, 3)
            expected_latents = model.encode(VAE.process_input(pixels.movedim(-1, 1).movedim(1, 0).unsqueeze(0)))
            actual_latents = clone.encode_tiled(pixels, tile_x=24, tile_y=24, overlap=8)
            torch.testing.assert_close(actual_latents, expected_latents)
            expected_pixels = VAE.process_output(model.decode(expected_latents)).movedim(1, -1)
            actual_pixels = clone.decode_tiled(actual_latents, tile_x=3, tile_y=3, overlap=1)
            torch.testing.assert_close(actual_pixels, expected_pixels)


if __name__ == "__main__":
    unittest.main()
