# ruff: noqa: E402

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "modules_forge" / "packages", ROOT / "extensions-builtin" / "anima-3-8b"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from anima3b.runtime import Anima3BRuntime

from backend.args import dynamic_args
from backend.diffusion_engine.anima import Anima as AnimaEngine
from backend.nn.anima import Anima as AnimaModel


class _Prompt(list):
    def __init__(self, *, negative=False):
        super().__init__(["portrait"])
        self.is_negative_prompt = negative


class AnimaReferenceConditioningTests(unittest.TestCase):
    def setUp(self):
        self.initial = torch.full((1, 2, 1, 2, 2), 3.0)
        self.reference = torch.full((1, 2, 1, 2, 2), 7.0)
        self.stale = torch.full((1, 2, 1, 2, 2), -1.0)
        self.previous_references = dynamic_args.ref_latents
        dynamic_args.ref_latents = [self.stale]
        self.addCleanup(setattr, dynamic_args, "ref_latents", self.previous_references)

        self.model = AnimaEngine.__new__(AnimaEngine)
        self.model.ini_latent = self.initial
        self.model.ref_latents = [self.reference]
        self.model.text_processing_engine_anima = Mock(return_value=[torch.zeros(1, 512, 4)])
        self.native_adapter = Mock()
        self.native_adapter.embed.weight = torch.ones(1, 4)
        self.clip = SimpleNamespace(
            patcher=SimpleNamespace(offload_device="cpu"),
            cond_stage_model=SimpleNamespace(qwen3_06b=SimpleNamespace(llm_adapter=self.native_adapter)),
        )
        self.model.forge_objects = SimpleNamespace(
            clip=self.clip,
            unet=SimpleNamespace(model=SimpleNamespace(diffusion_model=SimpleNamespace(blocks=[None] * 52))),
        )
        self.original = Mock(name="original_conditioning", return_value="native-result")
        self.model._anima3b_original_get_learned_conditioning = self.original

    def assert_prepared_references(self):
        self.assertEqual(len(dynamic_args.ref_latents), 2)
        self.assertIs(dynamic_args.ref_latents[0], self.initial)
        self.assertIs(dynamic_args.ref_latents[1], self.reference)
        self.assertIsNone(self.model.ini_latent)
        self.assertIsNot(dynamic_args.ref_latents, self.model.ref_latents)
        self.assertEqual(len(self.model.ref_latents), 1)

    def test_shared_preparation_consumes_initial_latent_once(self):
        with patch("backend.diffusion_engine.anima.opts", SimpleNamespace(anima_do_reference=True)):
            self.model.prepare_reference_latents(_Prompt())
            self.assert_prepared_references()
            self.model.prepare_reference_latents(_Prompt())

        self.assertEqual(len(dynamic_args.ref_latents), 1)
        self.assertIs(dynamic_args.ref_latents[0], self.reference)

    def test_native_conditioning_prepares_references_and_encodes_once(self):
        prompt = _Prompt()
        with (
            patch("backend.diffusion_engine.anima.opts", SimpleNamespace(anima_do_reference=True)),
            patch("backend.diffusion_engine.anima.memory_management.load_model_gpu"),
        ):
            result = self.model.get_learned_conditioning(prompt)

        self.assert_prepared_references()
        self.assertIs(result, self.model.text_processing_engine_anima.return_value)
        self.model.text_processing_engine_anima.assert_called_once_with(prompt)

    def _runtime(self, bundled):
        runtime = Anima3BRuntime()
        if bundled:
            runtime._active_bundle_metadata = {"architecture": "semantic-v2"}
            runtime._encode_v2 = Mock(return_value="v2-result")
        else:
            runtime._load_qwen = Mock(return_value=(object(), object(), SimpleNamespace(patcher=object())))
            runtime._load_adapter = Mock(return_value=Mock(return_value=torch.ones(1, 2, 4)))
            runtime._native_inputs = Mock(
                return_value=(torch.ones(1, 2, 4), torch.ones(1, 2, dtype=torch.long), torch.ones(1, 2, 1))
            )
            runtime._semantic_layers = Mock(return_value=([torch.ones(1, 2, 4)], torch.ones(1, 2)))
        return runtime

    def test_custom_encoders_prepare_references_without_running_native_encoder(self):
        for bundled in (False, True):
            with self.subTest(bundled=bundled):
                self.model.ini_latent = self.initial
                dynamic_args.ref_latents = [self.stale]
                runtime = self._runtime(bundled)
                with (
                    patch("backend.diffusion_engine.anima.opts", SimpleNamespace(anima_do_reference=True)),
                    patch("anima3b.runtime.memory_management.load_model_gpu"),
                ):
                    runtime.encode(self.model, _Prompt(), "adapter", 1.0, 1.0)

                self.assert_prepared_references()
                self.original.assert_not_called()
                self.model.text_processing_engine_anima.assert_not_called()
                if not bundled:
                    runtime._native_inputs.assert_called_once()
                    runtime._semantic_layers.assert_called_once()

    def test_custom_positive_encoders_clear_stale_references_when_disabled(self):
        for bundled in (False, True):
            with self.subTest(bundled=bundled):
                dynamic_args.ref_latents = [self.stale]
                runtime = self._runtime(bundled)
                with (
                    patch("backend.diffusion_engine.anima.opts", SimpleNamespace(anima_do_reference=False)),
                    patch("anima3b.runtime.memory_management.load_model_gpu"),
                ):
                    runtime.encode(self.model, _Prompt(), "adapter", 1.0, 1.0)

                self.assertEqual(dynamic_args.ref_latents, [])
                self.assertIs(self.model.ini_latent, self.initial)
                self.original.assert_not_called()

    def test_custom_negative_encoders_preserve_positive_reference_state(self):
        for bundled in (False, True):
            with self.subTest(bundled=bundled):
                prepared = [self.initial, self.reference]
                dynamic_args.ref_latents = prepared
                runtime = self._runtime(bundled)
                with (
                    patch("backend.diffusion_engine.anima.opts", SimpleNamespace(anima_do_reference=False)),
                    patch("anima3b.runtime.memory_management.load_model_gpu"),
                ):
                    runtime.encode(self.model, _Prompt(negative=True), "adapter", 1.0, 1.0)

                self.assertIs(dynamic_args.ref_latents, prepared)
                self.assertIs(self.model.ini_latent, self.initial)
                self.original.assert_not_called()

    def test_zero_strength_fallback_prepares_references_only_once(self):
        prompt = _Prompt()
        self.original.side_effect = self.model.get_learned_conditioning
        with (
            patch("backend.diffusion_engine.anima.opts", SimpleNamespace(anima_do_reference=True)),
            patch("backend.diffusion_engine.anima.memory_management.load_model_gpu"),
            patch.object(
                self.model, "prepare_reference_latents", wraps=self.model.prepare_reference_latents
            ) as prepare,
        ):
            Anima3BRuntime().encode(self.model, prompt, "adapter", 0.0, None)

        prepare.assert_called_once_with(prompt)
        self.assert_prepared_references()
        self.original.assert_called_once_with(prompt)
        self.model.text_processing_engine_anima.assert_called_once_with(prompt)


class AnimaReferenceBatchTests(unittest.TestCase):
    def setUp(self):
        self.previous_references = dynamic_args.ref_latents
        dynamic_args.ref_latents = []
        self.addCleanup(setattr, dynamic_args, "ref_latents", self.previous_references)
        self.model = AnimaModel(
            in_channels=2,
            out_channels=2,
            patch_spatial=1,
            patch_temporal=1,
            model_channels=12,
            crossattn_emb_channels=4,
            adaln_lora_dim=4,
            num_blocks=0,
            num_heads=1,
        )

    def _forward(self, sampling_batch):
        samples = torch.zeros(sampling_batch, 2, 1, 2, 2)
        with patch.object(
            self.model, "prepare_embedded_sequence", wraps=self.model.prepare_embedded_sequence
        ) as prepare:
            output = self.model(samples, torch.ones(sampling_batch), torch.zeros(sampling_batch, 2, 4))
        self.assertEqual(output.shape, samples.shape)
        return prepare.call_args.args[0]

    def test_single_reference_matches_sampling_batches_one_through_four(self):
        reference = torch.full((1, 2, 1, 2, 2), 5.0, dtype=torch.float64)
        dynamic_args.ref_latents = [reference]
        for batch in (1, 2, 3, 4):
            with self.subTest(batch=batch):
                embedded_input = self._forward(batch)
                self.assertEqual(embedded_input.shape, (batch, 2, 2, 2, 2))
                torch.testing.assert_close(embedded_input[:, :, 1:], reference.float().repeat(batch, 1, 1, 1, 1))
        self.assertEqual(reference.shape[0], 1)
        self.assertEqual(reference.dtype, torch.float64)

    def test_matching_reference_batch_keeps_each_sample_row(self):
        reference = torch.tensor([10.0, 20.0]).reshape(2, 1, 1, 1, 1).expand(-1, 2, 1, 2, 2)
        dynamic_args.ref_latents = [reference]

        embedded_input = self._forward(2)

        torch.testing.assert_close(embedded_input[:, :, 1:], reference)

    def test_cfg_repeats_complete_reference_batch_in_condition_group_order(self):
        reference = torch.tensor([10.0, 20.0]).reshape(2, 1, 1, 1, 1).expand(-1, 2, 1, 2, 2)
        dynamic_args.ref_latents = [reference]

        embedded_input = self._forward(4)

        self.assertEqual(embedded_input[:, 0, 1, 0, 0].tolist(), [10.0, 20.0, 10.0, 20.0])

    def test_multiple_references_keep_temporal_order_and_crop_output(self):
        dynamic_args.ref_latents = [
            torch.full((1, 2, 1, 2, 2), 5.0),
            torch.full((1, 2, 1, 2, 2), 8.0),
        ]

        embedded_input = self._forward(3)

        self.assertEqual(embedded_input[0, 0, :, 0, 0].tolist(), [0.0, 5.0, 8.0])

    def test_invalid_reference_batches_fail_before_embedding(self):
        for reference_batch, sampling_batch in ((2, 3), (3, 2), (0, 1), (1, 0)):
            with self.subTest(reference=reference_batch, sampling=sampling_batch):
                dynamic_args.ref_latents = [torch.zeros(reference_batch, 2, 1, 2, 2)]
                with (
                    patch.object(self.model, "prepare_embedded_sequence") as prepare,
                    self.assertRaisesRegex(
                        ValueError, f"reference batch {reference_batch}.*sampling batch {sampling_batch}"
                    ),
                ):
                    self._forward(sampling_batch)
                prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
