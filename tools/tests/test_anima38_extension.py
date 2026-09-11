# ruff: noqa: E402

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
EXTENSION = ROOT / "extensions-builtin" / "anima-3-8b"
if str(EXTENSION) not in sys.path:
    sys.path.insert(0, str(EXTENSION))

from anima3b.adapter import ProgressiveCrossAdapter
from anima3b.files import (
    ARCHITECTURE,
    BUNDLE_ARCHITECTURE,
    BUNDLE_FORMAT,
    V2_ARCHITECTURE,
    adapters,
    bundle_metadata,
    qwen35_models,
    standard_anima_loras,
    tokenizer_dir,
)
from anima3b.qwen35 import CPUEmbedding, Qwen35HybridModel
from anima3b.runtime import ANIMA38_BLOCK_COUNT, Anima3BRuntime
from anima3b.semantic_v2 import QualityAnchoredSemanticConnectorV2
from anima3b.standard_lora import (
    apply_standard_lora_selection,
    apply_standard_lora_selections,
)


def fake_anima(block_count: int):
    clip = object()
    diffusion_model = SimpleNamespace(blocks=[object()] * block_count)
    unet = SimpleNamespace(model=SimpleNamespace(diffusion_model=diffusion_model))
    return SimpleNamespace(
        text_processing_engine_anima=object(),
        forge_objects=SimpleNamespace(clip=clip, unet=unet),
        filename="legacy-anima.safetensors",
    )


class _RecordingLayer(torch.nn.Module):
    def __init__(self, increment: float):
        super().__init__()
        self.increment = increment
        self.last_output = None

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        freqs_cis=None,
        linear_attention_mask=None,
    ):
        del attention_mask, freqs_cis, linear_attention_mask
        self.last_output = hidden_states + self.increment
        return self.last_output


class _RecordingNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states):
        self.calls += 1
        return hidden_states


class _FlagSelfAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads = 1


class _FlagBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _FlagSelfAttention()
        self.seen_flag = False
        self.fail = False

    def forward(self, x, context, **kwargs):
        del context, kwargs
        self.seen_flag = getattr(self, "_anima38_inplace", False)
        if self.fail:
            raise RuntimeError("intentional adapter failure")
        return x


class _NullRotary(torch.nn.Module):
    def forward(self, x, positions):
        del x, positions
        return None


class _NativeAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(32, 4)
        self.in_proj = torch.nn.Identity()
        self.rotary_emb = _NullRotary()
        self.blocks = torch.nn.ModuleList([_FlagBlock()])
        self.out_proj = torch.nn.Identity()
        self.norm = torch.nn.Identity()


class _ZeroAttention(torch.nn.Module):
    def forward(self, x, **kwargs):
        del kwargs
        return torch.zeros_like(x)


class Anima38ExtensionTests(unittest.TestCase):
    def test_progressive_adapter_restores_temporary_inplace_flags(self):
        native = _NativeAdapter()
        adapter = ProgressiveCrossAdapter(
            native,
            semantic_source_dim=4,
            layer_indices=(0,),
        )
        adapter.semantic_attentions[0] = _ZeroAttention()
        source = torch.ones(1, 2, 4)
        target_ids = torch.tensor([[1, 2]])
        semantic = [torch.ones(1, 2, 4)]

        adapter(source, target_ids, semantic)
        self.assertTrue(native.blocks[0].seen_flag)
        self.assertFalse(hasattr(native.blocks[0], "_anima38_inplace"))

        native.blocks[0]._anima38_inplace = False
        native.blocks[0].fail = True
        with self.assertRaisesRegex(RuntimeError, "intentional adapter failure"):
            adapter(source, target_ids, semantic)
        self.assertFalse(native.blocks[0]._anima38_inplace)

    def test_v11_connector_restores_temporary_inplace_flags(self):
        native = _NativeAdapter()
        connector = QualityAnchoredSemanticConnectorV2(
            native,
            semantic_source_dim=4,
            layer_indices=(0,),
            num_queries=2,
            resampler_blocks=1,
            resampler_dim=4,
            resampler_heads=1,
            mlp_hidden_dim=4,
        )
        source = torch.ones(1, 2, 4)
        target_ids = torch.tensor([[1, 2]])
        semantic = [torch.ones(1, 2, 4)]

        connector(source, target_ids, semantic, timesteps=torch.tensor([0.5]))

        self.assertTrue(native.blocks[0].seen_flag)
        self.assertFalse(hasattr(native.blocks[0], "_anima38_inplace"))

    def test_cpu_embedding_keeps_table_unregistered_and_gathers_exact_rows(self):
        weight = torch.arange(40, dtype=torch.bfloat16).reshape(10, 4)
        embedding = CPUEmbedding(weight)
        ids = torch.tensor([[7, 1, 7]], dtype=torch.long)

        actual = embedding(ids)
        expected = torch.nn.functional.embedding(ids, weight)

        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(list(embedding.named_parameters()), [])
        self.assertEqual(list(embedding.named_buffers()), [])
        embedding.to("meta")
        self.assertEqual(embedding.weight.device.type, "cpu")

    def test_bundled_qwen35_tokenizer_is_available_offline(self):
        directory = tokenizer_dir()

        self.assertEqual(directory, EXTENSION / "qwen35_tokenizer")
        config = json.loads(
            (directory / "tokenizer_config.json").read_text(encoding="utf-8")
        )
        self.assertIn("tokenizer_class", config)

    def test_adapter_discovery_uses_architecture_metadata(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "adapter.safetensors"
            unrelated = root / "other.safetensors"
            save_file(
                {"query_norms.0.weight": torch.ones(2)},
                valid,
                metadata={"architecture": ARCHITECTURE},
            )
            save_file(
                {"query_norms.0.weight": torch.ones(2)},
                unrelated,
                metadata={"architecture": "different"},
            )

            with patch("anima3b.files.text_encoder_roots", return_value=[root]):
                found = adapters()

        self.assertEqual(found, {"adapter.safetensors": str(valid)})

    def test_v11_bundle_discovery_requires_official_metadata(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "v11.safetensors"
            legacy = root / "legacy.safetensors"
            save_file(
                {"net.anima_v2_connector.test": torch.ones(1)},
                valid,
                metadata={
                    "architecture": BUNDLE_ARCHITECTURE,
                    "anima_v2_bundle_format": BUNDLE_FORMAT,
                    "anima_v2_adapter_architecture": V2_ARCHITECTURE,
                },
            )
            save_file(
                {"net.blocks.0.test": torch.ones(1)},
                legacy,
                metadata={"architecture": "legacy"},
            )

            valid_metadata = bundle_metadata(valid)
            legacy_metadata = bundle_metadata(legacy)

        self.assertIsNotNone(valid_metadata)
        self.assertEqual(valid_metadata["architecture"], BUNDLE_ARCHITECTURE)
        self.assertIsNone(legacy_metadata)

    def test_qwen_discovery_accepts_the_paired_filename(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paired = root / "qwen35_4b.safetensors"
            paired.touch()
            (root / "qwen3_06b_base.safetensors").touch()

            with patch("anima3b.files.text_encoder_roots", return_value=[root]):
                found = qwen35_models()

        self.assertEqual(found, {"qwen35_4b.safetensors": str(paired)})

    def test_standard_lora_discovery_accepts_complete_anima_layouts_only(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for block_count in (28, 40, 52):
                save_file(
                    {
                        f"diffusion_model.blocks.{block}.{module}.lora_A.weight": torch.ones(1)
                        for block in range(block_count)
                        for module in (
                            "adaln_modulation_mlp.1",
                            "cross_attn.q_proj",
                            "self_attn.q_proj",
                            "mlp.layer1",
                        )
                    },
                    root / f"anima-{block_count}.safetensors",
                )
            save_file(
                {
                    "diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight": torch.ones(1),
                    "diffusion_model.blocks.27.self_attn.q_proj.lora_A.weight": torch.ones(1),
                },
                root / "sparse.safetensors",
            )
            save_file(
                {"unrelated.weight": torch.ones(1)},
                root / "unrelated.safetensors",
            )
            save_file(
                {
                    f"diffusion_model.blocks.{block}.cross_attn.q.lora_down.weight": torch.ones(1)
                    for block in range(40)
                },
                root / "wan-40.safetensors",
            )

            duplicate_root = root / "later-root"
            duplicate_root.mkdir()
            save_file(
                {
                    f"diffusion_model.blocks.{block}.cross_attn.q.lora_down.weight": torch.ones(1)
                    for block in range(28)
                },
                duplicate_root / "anima-28.safetensors",
            )

            with patch(
                "anima3b.files.lora_roots",
                return_value=[root, duplicate_root],
            ):
                found = standard_anima_loras()

        self.assertEqual(
            found,
            {
                "anima-40": str(root / "anima-40.safetensors"),
                "anima-52": str(root / "anima-52.safetensors"),
            },
        )

    def test_standard_lora_selection_uses_forge_prompt_pipeline_once(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "regular-anima.safetensors"
            path.touch()
            processing = SimpleNamespace(
                prompt="portrait",
                all_prompts=["styled portrait", "already <lora:regular-anima:0.5>"],
                extra_generation_params={},
            )

            applied = apply_standard_lora_selection(
                processing,
                "regular-anima",
                1.25,
                {"regular-anima": str(path)},
            )
            applied_again = apply_standard_lora_selection(
                processing,
                "regular-anima",
                1.25,
                {"regular-anima": str(path)},
            )

        self.assertTrue(applied)
        self.assertTrue(applied_again)
        self.assertEqual(
            processing.prompt,
            "portrait <lora:regular-anima:1.25>",
        )
        self.assertEqual(
            processing.all_prompts,
            [
                "styled portrait <lora:regular-anima:1.25>",
                "already <lora:regular-anima:0.5>",
            ],
        )
        self.assertEqual(
            processing.extra_generation_params["Anima 3.8B standard LoRA"],
            "regular-anima:1.25",
        )

    def test_standard_lora_selection_rejects_stale_or_nonfinite_values(self):
        processing = SimpleNamespace(
            prompt="portrait",
            all_prompts=["portrait"],
            extra_generation_params={},
        )
        with self.assertRaises(FileNotFoundError):
            apply_standard_lora_selection(
                processing,
                "missing",
                1.0,
                {},
            )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "regular-anima.safetensors"
            path.touch()
            with self.assertRaisesRegex(ValueError, "finite"):
                apply_standard_lora_selection(
                    processing,
                    "regular-anima",
                    float("nan"),
                    {"regular-anima": str(path)},
                )

    def test_multiple_standard_lora_selections_use_individual_weights(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            regular = root / "regular-anima.safetensors"
            cinematic = root / "cinematic-anima.safetensors"
            regular.touch()
            cinematic.touch()
            processing = SimpleNamespace(
                prompt="portrait",
                all_prompts=["first portrait", "second portrait"],
                extra_generation_params={},
            )

            applied = apply_standard_lora_selections(
                processing,
                [
                    ("regular-anima", 0.75),
                    ("cinematic-anima", 0.35),
                    ("regular-anima", 1.5),
                    (None, 1.0),
                ],
                {
                    "regular-anima": str(regular),
                    "cinematic-anima": str(cinematic),
                },
            )

        expected_tags = (
            "<lora:regular-anima:0.75> <lora:cinematic-anima:0.35>"
        )
        self.assertTrue(applied)
        self.assertEqual(processing.prompt, f"portrait {expected_tags}")
        self.assertEqual(
            processing.all_prompts,
            [
                f"first portrait {expected_tags}",
                f"second portrait {expected_tags}",
            ],
        )
        self.assertEqual(
            processing.extra_generation_params["Anima 3.8B standard LoRA"],
            "regular-anima:0.75, cinematic-anima:0.35",
        )

    def test_multiple_standard_lora_validation_is_atomic(self):
        with TemporaryDirectory() as directory:
            regular = Path(directory) / "regular-anima.safetensors"
            regular.touch()
            processing = SimpleNamespace(
                prompt="portrait",
                all_prompts=["portrait"],
                extra_generation_params={},
            )

            with self.assertRaisesRegex(FileNotFoundError, "missing-anima"):
                apply_standard_lora_selections(
                    processing,
                    [("regular-anima", 0.75), ("missing-anima", 0.35)],
                    {"regular-anima": str(regular)},
                )

        self.assertEqual(processing.prompt, "portrait")
        self.assertEqual(processing.all_prompts, ["portrait"])
        self.assertEqual(processing.extra_generation_params, {})

    def test_runtime_requires_the_paired_52_block_checkpoint(self):
        engine, clip = Anima3BRuntime._require_anima(
            fake_anima(ANIMA38_BLOCK_COUNT)
        )

        self.assertIsNotNone(engine)
        self.assertIsNotNone(clip)
        with self.assertRaisesRegex(RuntimeError, "52-block checkpoint"):
            Anima3BRuntime._require_anima(fake_anima(40))

    def test_install_salts_existing_conditioning_cache_and_restore_is_lossless(self):
        runtime = Anima3BRuntime()
        model = fake_anima(ANIMA38_BLOCK_COUNT)
        original_conditioning = Mock(name="original_conditioning")
        model.get_learned_conditioning = original_conditioning
        original_cached_params = Mock(
            name="original_cached_params",
            return_value=("base-cache-key",),
        )
        cached_c = ["positive-key", "positive-value", "positive-metadata"]
        cached_uc = ["negative-key", "negative-value", "negative-metadata"]
        processing = SimpleNamespace(
            sd_model=model,
            cached_params=original_cached_params,
            cached_c=cached_c,
            cached_uc=cached_uc,
            extra_generation_params={},
        )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            qwen = root / "qwen35_4b.safetensors"
            adapter = root / "adapter.safetensors"
            qwen.write_bytes(b"qwen-checkpoint")
            adapter.write_bytes(b"adapter-checkpoint")
            qwen_fingerprint = runtime._file_fingerprint(str(qwen))
            adapter_fingerprint = runtime._file_fingerprint(str(adapter))

            with (
                patch(
                    "anima3b.runtime.qwen35_models",
                    return_value={qwen.name: str(qwen)},
                ),
                patch(
                    "anima3b.runtime.adapters",
                    return_value={adapter.name: str(adapter)},
                ),
            ):
                runtime.install(
                    processing,
                    adapter.name,
                    strength=1.25,
                    negative_strength=0.75,
                )

            cache_key = processing.cached_params(
                ["prompt"],
                32,
                {"lora": "unchanged"},
                None,
            )

        original_cached_params.assert_called_once_with(
            ["prompt"],
            32,
            {"lora": "unchanged"},
            None,
        )
        self.assertEqual(cache_key[:-1], ("base-cache-key",))
        cache_namespace, signature = cache_key[-1]
        self.assertEqual(cache_namespace, "anima3b")
        self.assertEqual(signature[0], ARCHITECTURE)
        self.assertEqual(signature[1], qwen_fingerprint)
        self.assertEqual(signature[2], adapter_fingerprint)
        self.assertEqual(signature[3:], (1.25, 0.75))
        self.assertIs(processing.cached_c, cached_c)
        self.assertIs(processing.cached_uc, cached_uc)
        self.assertEqual(
            processing.cached_c,
            ["positive-key", "positive-value", "positive-metadata"],
        )
        self.assertEqual(
            processing.cached_uc,
            ["negative-key", "negative-value", "negative-metadata"],
        )

        replacement_model = fake_anima(ANIMA38_BLOCK_COUNT)
        replacement_conditioning = Mock(name="replacement_conditioning")
        replacement_model.get_learned_conditioning = replacement_conditioning
        processing.sd_model = replacement_model
        runtime.restore(processing)

        self.assertIs(processing.cached_params, original_cached_params)
        self.assertIs(model.get_learned_conditioning, original_conditioning)
        self.assertIs(
            replacement_model.get_learned_conditioning, replacement_conditioning
        )
        self.assertFalse(hasattr(processing, "_anima3b_original_cached_params"))
        self.assertFalse(hasattr(processing, "_anima3b_patched_model"))
        self.assertFalse(
            hasattr(model, "_anima3b_original_get_learned_conditioning")
        )

    def test_v11_install_is_automatic_and_uses_fixed_strength(self):
        runtime = Anima3BRuntime()
        model = fake_anima(ANIMA38_BLOCK_COUNT)
        original_conditioning = Mock(name="original_conditioning")
        model.get_learned_conditioning = original_conditioning
        original_cached_params = Mock(return_value=("base",))
        cached_c = ["positive-key", "positive-value", "positive-metadata"]
        cached_uc = ["negative-key", "negative-value", "negative-metadata"]
        processing = SimpleNamespace(
            sd_model=model,
            cached_params=original_cached_params,
            cached_c=cached_c,
            cached_uc=cached_uc,
            extra_generation_params={},
        )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "Anima-3.8B-v1.1.safetensors"
            qwen = root / "qwen35_4b.safetensors"
            qwen.write_bytes(b"qwen-checkpoint")
            save_file(
                {"net.anima_v2_connector.test": torch.ones(1)},
                bundle,
                metadata={
                    "architecture": BUNDLE_ARCHITECTURE,
                    "anima_v2_bundle_format": BUNDLE_FORMAT,
                    "anima_v2_adapter_architecture": V2_ARCHITECTURE,
                    "anima_v2_adapter_filename": "Semantic Connector v2",
                },
            )
            model.filename = str(bundle)

            with (
                patch(
                    "anima3b.runtime.qwen35_models",
                    return_value={qwen.name: str(qwen)},
                ),
                patch.object(runtime, "_install_v2") as install_v2,
            ):
                self.assertTrue(runtime.is_v2_bundle(model))
                runtime.install(
                    processing,
                    "ignored-legacy-adapter.safetensors",
                    strength=0.25,
                    negative_strength=None,
                )

            signature = processing.cached_params(["prompt"], 32, {}, None)[-1][1]

        install_v2.assert_called_once()
        self.assertEqual(signature[0], V2_ARCHITECTURE)
        self.assertEqual(signature[3:], (1.0, None))
        self.assertEqual(
            processing.extra_generation_params["Anima 3.8B strength"],
            1.0,
        )
        runtime.restore(processing)
        self.assertIs(model.get_learned_conditioning, original_conditioning)
        self.assertEqual(cached_c, [None, None, None])
        self.assertEqual(cached_uc, [None, None, None])

    def test_v11_cleanup_invalidates_shared_hires_caches_for_next_request(self):
        # Forgeは次のリクエストにもクラス共有のHiresキャッシュを渡す。
        class Processing:
            cached_c = ["base", "conditioning", None]
            cached_uc = ["base-negative", "conditioning", None]
            cached_hr_c = ["hires", {"vector": torch.tensor([[2]])}, None]
            cached_hr_uc = ["hires-negative", {"vector": torch.tensor([[1]])}, None]

        for bundled in (False, True):
            with self.subTest(bundled=bundled):
                runtime = Anima3BRuntime()
                processing = Processing()
                processing.sd_model = fake_anima(ANIMA38_BLOCK_COUNT)
                if bundled:
                    runtime._active_bundle_metadata = {"architecture": V2_ARCHITECTURE}
                    runtime._v2_runs[2] = object()
                runtime.restore(processing)
                next_request = Processing()
                for name in ("cached_c", "cached_uc", "cached_hr_c", "cached_hr_uc"):
                    cache = getattr(next_request, name)
                    if bundled:
                        self.assertEqual(cache, [None, None, None], name)
                    else:
                        self.assertIsNotNone(cache[0], name)
                self.assertFalse(runtime._v2_runs)

    def test_missing_assets_fail_during_conditioning_and_restore_cleanly(self):
        runtime = Anima3BRuntime()
        model = fake_anima(ANIMA38_BLOCK_COUNT)
        original_conditioning = Mock(name="original_conditioning")
        model.get_learned_conditioning = original_conditioning
        original_cached_params = Mock(name="original_cached_params")
        processing = SimpleNamespace(
            sd_model=model,
            cached_params=original_cached_params,
            extra_generation_params={},
        )

        with (
            patch(
                "anima3b.runtime.qwen35_models",
                return_value={"qwen35_4b.safetensors": "missing-qwen.safetensors"},
            ),
            patch(
                "anima3b.runtime.adapters",
                return_value={"adapter.safetensors": "missing-adapter.safetensors"},
            ),
        ):
            runtime.install(processing, "adapter.safetensors", 1.0, None)
            with self.assertRaises(FileNotFoundError):
                model.get_learned_conditioning(["prompt"])

        self.assertIsNot(model.get_learned_conditioning, original_conditioning)
        runtime.restore(processing)
        self.assertIs(model.get_learned_conditioning, original_conditioning)
        self.assertIs(processing.cached_params, original_cached_params)

    def test_offload_text_encoders_detaches_patchers_and_moves_adapter_to_cpu(self):
        runtime = Anima3BRuntime()
        model = fake_anima(ANIMA38_BLOCK_COUNT)
        native_patcher = Mock(name="native_patcher")
        native_patcher.loaded_size.return_value = 100
        qwen_patcher = Mock(name="qwen_patcher")
        qwen_patcher.loaded_size.return_value = 200
        model.forge_objects.clip = SimpleNamespace(patcher=native_patcher)
        runtime._qwen_clip = SimpleNamespace(patcher=qwen_patcher)
        runtime._adapter = Mock(name="adapter")

        with (
            patch(
                "anima3b.runtime.memory_management.unload_model",
                return_value=True,
            ) as unload,
            patch("anima3b.runtime.memory_management.soft_empty_cache") as empty,
        ):
            released = runtime.offload_text_encoders(model)

        self.assertEqual(released, 300)
        native_patcher.detach.assert_not_called()
        qwen_patcher.detach.assert_not_called()
        self.assertEqual(
            unload.call_args_list,
            [call(native_patcher), call(qwen_patcher)],
        )
        runtime._adapter.to.assert_called_once_with("cpu")
        empty.assert_called_once_with()

    def test_reinstall_after_model_switch_restores_the_exact_previous_model(self):
        runtime = Anima3BRuntime()
        first_model = fake_anima(ANIMA38_BLOCK_COUNT)
        second_model = fake_anima(ANIMA38_BLOCK_COUNT)
        first_original = Mock(name="first_original")
        second_original = Mock(name="second_original")
        first_model.get_learned_conditioning = first_original
        second_model.get_learned_conditioning = second_original
        original_cached_params = Mock(
            name="original_cached_params", return_value=("base",)
        )
        processing = SimpleNamespace(
            sd_model=first_model,
            cached_params=original_cached_params,
            extra_generation_params={},
        )

        with patch.object(
            runtime,
            "_conditioning_cache_signature",
            return_value=("signature",),
        ):
            runtime.install(processing, "adapter.safetensors", 1.0, None)
            processing.sd_model = second_model
            runtime.install(processing, "adapter.safetensors", 1.0, None)

        self.assertIs(first_model.get_learned_conditioning, first_original)
        self.assertIsNot(second_model.get_learned_conditioning, second_original)
        cache_key = processing.cached_params(["prompt"], 32, {}, None)
        self.assertEqual(cache_key.count(("anima3b", ("signature",))), 1)

        runtime.restore(processing)
        self.assertIs(second_model.get_learned_conditioning, second_original)

    def test_qwen_can_skip_unused_final_output_without_cloning_intermediates(self):
        model = Qwen35HybridModel.__new__(Qwen35HybridModel)
        torch.nn.Module.__init__(model)
        first = _RecordingLayer(1.0)
        second = _RecordingLayer(2.0)
        norm = _RecordingNorm()
        model.layers = torch.nn.ModuleList([first, second])
        model.norm = norm
        embeds = torch.zeros(1, 3, 2)

        output, intermediate = model(
            input_ids=None,
            embeds=embeds,
            attention_mask=None,
            intermediate_output=[0, 1],
            return_final_output=False,
        )

        self.assertIsNone(output)
        self.assertEqual(norm.calls, 0)
        self.assertIs(intermediate[0], first.last_output)
        self.assertIs(intermediate[1], second.last_output)
        torch.testing.assert_close(intermediate[0], torch.ones_like(embeds))
        torch.testing.assert_close(intermediate[1], torch.full_like(embeds, 3.0))


if __name__ == "__main__":
    unittest.main()
