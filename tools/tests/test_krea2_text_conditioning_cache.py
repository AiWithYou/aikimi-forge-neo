"""Sampling-scoped Krea text reuse preserves real forwards and releases tensors."""

import ast
import collections
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from backend.args import dynamic_args
from backend.nn import krea
from backend.nn.krea import SingleStreamDiT
from backend.sampling.condition import compile_conditions

ROOT = Path(__file__).resolve().parents[2]


class Interrupted(BaseException):
    pass


def load_launch_sampling():
    path = ROOT / "modules/sd_samplers_common.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    sampler = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Sampler")
    method = next(n for n in sampler.body if isinstance(n, ast.FunctionDef) and n.name == "launch_sampling")
    namespace = {
        "nullcontext": nullcontext,
        "state": SimpleNamespace(current_latent="interrupted latent"),
        "InterruptedException": Interrupted,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    return namespace["launch_sampling"]


def load_calc_batch(batch_cfg):
    path = ROOT / "backend/sampling/sampling_function.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        "get_area_and_mult",
        "cond_equal_size",
        "can_concat_cond",
        "cond_cat",
        "compute_cond_mark",
        "compute_cond_indices",
        "calc_cond_uncond_batch",
    }
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {
        "torch": torch,
        "collections": collections,
        "args": SimpleNamespace(disable_gpu_warning=True),
        "shared": SimpleNamespace(batch_cond_uncond=batch_cfg),
        "memory_management": SimpleNamespace(signal_empty_cache=False, get_free_memory=lambda _device: float("inf")),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    return namespace["calc_cond_uncond_batch"]


class KreaTextConditioningCacheTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.model = SingleStreamDiT(
            features=64,
            txtdim=64,
            heads=1,
            kvheads=1,
            layers=1,
            txtlayers=2,
            txtheads=1,
            txtkvheads=1,
            multiplier=1,
        ).eval()
        for parameter in self.model.parameters():
            torch.nn.init.normal_(parameter, std=0.03)
        self.context = torch.randn(1, 4, 2, 64)
        self.input = torch.randn(1, 16, 1, 8, 8)
        self.enterContext(patch.object(dynamic_args, "ref_latents", []))

    def forward(self, context=None, *, step=1, options=None):
        context = self.context if context is None else context
        x = self.input.repeat(context.shape[0], 1, 1, 1, 1)
        return self.model(
            x,
            torch.full((context.shape[0],), float(step)),
            context.clone(),
            transformer_options={} if options is None else options,
        )

    def test_equal_new_context_tensors_skip_text_work_and_keep_exact_step_outputs(self):
        with torch.inference_mode():
            expected = [self.forward(step=step) for step in (1, 2, 3)]
            with patch.object(self.model.txtfusion, "forward", wraps=self.model.txtfusion.forward) as fusion:
                with self.model.text_conditioning_cache() as cache:
                    actual = [self.forward(step=step) for step in (1, 2, 3)]
                    self.assertEqual(fusion.call_count, 1)
                    self.assertGreater(cache.retained_bytes, 0)
                    self.assertLessEqual(cache.retained_bytes, 64 * 1024 * 1024)
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True)))
                self.assertEqual(cache.retained_bytes, 0)
                self.assertIsNone(self.model._text_conditioning_cache)

    def test_prompt_change_and_same_storage_mutation_recompute(self):
        changed = self.context.clone()
        with (
            torch.inference_mode(),
            patch.object(self.model.txtfusion, "forward", wraps=self.model.txtfusion.forward) as fusion,
        ):
            with self.model.text_conditioning_cache():
                first = self.forward(changed)
                changed.add_(0.5)
                second = self.forward(changed)
                third = self.forward(changed)
            self.assertEqual(fusion.call_count, 2)
            self.assertFalse(torch.equal(first, second))
            self.assertTrue(torch.equal(second, third))

    def test_batch_one_to_four_and_cfg_row_order_match_uncached(self):
        positive, negative = self.context, self.context + 0.25
        contexts = [self.context.repeat(batch, 1, 1, 1) for batch in (1, 2, 3, 4)]
        contexts += [torch.cat((positive, negative)), torch.cat((negative, positive))]
        with torch.inference_mode():
            expected = [self.forward(c) for c in contexts]
            with self.model.text_conditioning_cache():
                actual = [self.forward(c) for c in contexts]
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True)))

    def test_split_cfg_keeps_two_entries_but_evicts_a_third_prompt(self):
        negative, later = self.context + 0.2, self.context + 0.4
        with (
            torch.inference_mode(),
            patch.object(self.model.txtfusion, "forward", wraps=self.model.txtfusion.forward) as fusion,
        ):
            with self.model.text_conditioning_cache() as cache:
                for context in (self.context, negative, self.context, later, negative):
                    self.forward(context)
                    self.assertLessEqual(len(cache.entries), 2)
                self.assertEqual(fusion.call_count, 4)

    def test_reference_and_resolution_changes_do_not_stale_the_text_result(self):
        with torch.inference_mode():
            with self.model.text_conditioning_cache():
                self.forward()
                dynamic_args.ref_latents = [torch.randn(1, 16, 8, 8)]
                self.input = torch.randn(1, 16, 1, 6, 10)
                cached = self.forward()
            expected = self.forward()
        self.assertTrue(torch.equal(expected, cached))

    def test_nested_sampling_releases_outer_entries_before_allocating_inner_cache(self):
        with torch.inference_mode(), self.model.text_conditioning_cache() as outer:
            self.forward()
            self.assertGreater(outer.retained_bytes, 0)
            with self.model.text_conditioning_cache() as inner:
                self.assertEqual(outer.retained_bytes, 0)
                self.forward()
                self.assertGreater(inner.retained_bytes, 0)
            self.assertEqual(inner.retained_bytes, 0)
            self.assertEqual(outer.retained_bytes, 0)
            self.forward()
            self.assertEqual(outer.misses, 2)

    def test_byte_limit_bypasses_oversized_context_and_bounds_retained_entries(self):
        with (
            torch.inference_mode(),
            patch.object(self.model.txtfusion, "forward", wraps=self.model.txtfusion.forward) as fusion,
        ):
            with self.model.text_conditioning_cache(max_bytes=1) as cache:
                self.forward()
                self.forward()
                self.assertEqual(cache.retained_bytes, 0)
                self.assertEqual(fusion.call_count, 2)
            with self.model.text_conditioning_cache(max_bytes=4500) as cache:
                self.forward()
                self.forward(self.context + 0.3)
                self.assertLessEqual(cache.retained_bytes, 4500)
                self.assertEqual(len(cache.entries), 1)

    def test_weight_update_and_module_replacement_invalidate_cached_text(self):
        with (
            torch.inference_mode(),
            patch.object(self.model.txtfusion, "forward", wraps=self.model.txtfusion.forward) as fusion,
        ):
            with self.model.text_conditioning_cache():
                before = self.forward()
                self.model.txtmlp[-1].weight.add_(0.1)
                updated = self.forward()
                self.assertEqual(fusion.call_count, 2)
                self.assertFalse(torch.equal(before, updated))
                original = self.model.txtmlp[-1]
                replacement = torch.nn.Linear(64, 64)
                replacement.load_state_dict(original.state_dict())
                self.model.txtmlp[-1] = replacement
                replaced = self.forward()
                self.assertEqual(fusion.call_count, 3)
                self.assertTrue(torch.equal(updated, replaced))

    def test_real_sampler_cat_and_split_cfg_preserve_outputs_and_source_conditioning(self):
        positive, negative = self.context.clone(), self.context.clone() + 0.2
        cond, uncond = compile_conditions(positive), compile_conditions(negative)

        def apply(x, timestep, *, c_crossattn, transformer_options):
            return self.model(x.unsqueeze(2), timestep, c_crossattn, transformer_options=transformer_options).squeeze(2)

        model = SimpleNamespace(apply_model=apply, memory_required=lambda _shape: 0)
        for batched in (True, False):
            calc = load_calc_batch(batched)
            with self.subTest(batched=batched), torch.inference_mode():
                expected = [
                    calc(model, cond, uncond, self.input.squeeze(2), torch.tensor([float(step)]), {}) for step in (1, 2)
                ]
                with self.model.text_conditioning_cache():
                    actual = [
                        calc(model, cond, uncond, self.input.squeeze(2), torch.tensor([float(step)]), {})
                        for step in (1, 2)
                    ]
                for baseline, cached in zip(expected, actual, strict=True):
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(baseline, cached, strict=True)))
                self.assertTrue(torch.equal(positive, self.context))
                self.assertTrue(torch.equal(negative, self.context + 0.2))

    def test_inference_loaded_parameters_reuse_only_with_model_revision_provider(self):
        with torch.inference_mode():
            inference_model = SingleStreamDiT(
                features=64,
                txtdim=64,
                heads=1,
                kvheads=1,
                layers=1,
                txtlayers=2,
                txtheads=1,
                txtkvheads=1,
                multiplier=1,
            ).eval()
            inference_model.load_state_dict(self.model.state_dict())
            self.model = inference_model
            revision = ["original patch"]
            with patch.object(self.model.txtfusion, "forward", wraps=self.model.txtfusion.forward) as fusion:
                with self.model.text_conditioning_cache(weight_revision=lambda: revision[0]):
                    first = self.forward()
                    self.assertTrue(torch.equal(first, self.forward()))
                    self.assertEqual(fusion.call_count, 1)
                    self.model.txtmlp[-1].weight.data.add_(0.1)
                    revision[0] = "updated patch"
                    updated = self.forward()
                    self.assertEqual(fusion.call_count, 2)
                    self.assertFalse(torch.equal(first, updated))
                with self.model.text_conditioning_cache() as cache:
                    self.forward()
                    self.forward()
                    self.assertEqual(cache.retained_bytes, 0)

    def test_attention_function_replacement_invalidates_cached_fusion(self):
        original = krea.attention_function
        with (
            torch.inference_mode(),
            patch.object(self.model.txtfusion, "forward", wraps=self.model.txtfusion.forward) as fusion,
        ):
            with self.model.text_conditioning_cache():
                self.forward()
                with patch.object(krea, "attention_function", wraps=original):
                    self.forward()
                    self.assertEqual(fusion.call_count, 2)

    def test_training_and_grad_enabled_calls_bypass_reuse(self):
        with patch.object(self.model.txtfusion, "forward", wraps=self.model.txtfusion.forward) as fusion:
            self.model.train()
            with torch.no_grad(), self.model.text_conditioning_cache() as cache:
                self.forward()
                self.forward()
                self.assertEqual(cache.retained_bytes, 0)
            self.model.eval().requires_grad_(False)
            with torch.enable_grad(), self.model.text_conditioning_cache() as cache:
                self.forward()
                self.forward()
                self.assertEqual(cache.retained_bytes, 0)
            self.assertEqual(fusion.call_count, 4)

    def test_text_forward_hooks_run_each_time_and_do_not_leave_a_cached_result(self):
        calls = []

        def alter_text(_module, _args, output):
            calls.append(1)
            return output + len(calls) * 0.05

        handle = self.model.txtmlp.register_forward_hook(alter_text)
        self.addCleanup(handle.remove)
        with torch.inference_mode(), self.model.text_conditioning_cache() as cache:
            first, second = self.forward(), self.forward()
            self.assertEqual(len(calls), 2)
            self.assertFalse(torch.equal(first, second))
            self.assertEqual(cache.retained_bytes, 0)

    def test_online_weight_hooks_and_custom_text_attention_bypass_reuse(self):
        with (
            torch.inference_mode(),
            patch.object(self.model.txtfusion, "forward", wraps=self.model.txtfusion.forward) as fusion,
        ):
            self.model.txtmlp[-1].weight_function = [object()]
            with self.model.text_conditioning_cache() as cache:
                self.forward()
                self.forward()
                self.assertEqual(cache.retained_bytes, 0)
            self.model.txtmlp[-1].weight_function = []

            def attention(q, k, v, heads, mask, options, dense):
                return dense(q, k, v, heads, mask=mask, skip_reshape=True, transformer_options=options)

            with self.model.text_conditioning_cache() as cache:
                options = {"krea2_attention_override": attention, "krea2_block_index": 0}
                self.forward(options=options)
                self.forward(options=options)
                self.assertEqual(cache.retained_bytes, 0)
            self.assertEqual(fusion.call_count, 4)

    def test_sampler_requires_known_weight_revision_tracking(self):
        launch = load_launch_sampling()
        for model_attributes, patcher_attributes, expected in (
            ({}, {}, None),
            ({"current_weight_patches_uuid": None}, {}, None),
            ({}, {"patches_uuid": "loaded"}, None),
            ({"current_weight_patches_uuid": None}, {"patches_uuid": None}, None),
            ({"current_weight_patches_uuid": None}, {"patches_uuid": "loaded"}, (None, "loaded")),
        ):
            with self.subTest(model=model_attributes, patcher=patcher_attributes):
                unet = SimpleNamespace(
                    model=SimpleNamespace(diffusion_model=self.model, **model_attributes),
                    model_options={},
                    **patcher_attributes,
                )
                sampler = SimpleNamespace(
                    p=SimpleNamespace(sd_model=SimpleNamespace(forge_objects=SimpleNamespace(unet=unet))),
                    model_wrap_cfg=SimpleNamespace(),
                    config=SimpleNamespace(total_steps=lambda count: count),
                )
                with patch.object(self.model, "text_conditioning_cache") as scope:
                    observed = launch(sampler, 2, lambda: scope.call_args.kwargs["weight_revision"]())
                self.assertEqual(observed, expected)

    def test_sampler_scope_releases_cache_on_success_cancel_and_exception(self):
        launch = load_launch_sampling()
        engine = SimpleNamespace(
            forge_objects=SimpleNamespace(
                unet=SimpleNamespace(
                    model=SimpleNamespace(diffusion_model=self.model, current_weight_patches_uuid=None),
                    patches_uuid="base",
                )
            )
        )
        sampler = SimpleNamespace(
            p=SimpleNamespace(sd_model=engine),
            model_wrap_cfg=SimpleNamespace(),
            config=SimpleNamespace(total_steps=lambda count: count),
        )
        for error in (None, Interrupted(), RuntimeError("sampling failed")):
            with self.subTest(error=type(error).__name__), torch.inference_mode():
                caches = []

                def run(caches=caches, error=error):
                    caches.append(self.model._text_conditioning_cache)
                    self.assertIsNotNone(caches[-1])
                    self.forward()
                    self.assertGreater(caches[-1].retained_bytes, 0)
                    if error is not None:
                        raise error
                    return "completed"

                if isinstance(error, RuntimeError):
                    with self.assertRaisesRegex(RuntimeError, "sampling failed"):
                        launch(sampler, 3, run)
                else:
                    result = launch(sampler, 3, run)
                    self.assertEqual(result, "completed" if error is None else "interrupted latent")
                self.assertIsNone(self.model._text_conditioning_cache)
                self.assertEqual(caches[0].retained_bytes, 0)

    def test_next_sampling_and_hires_stage_start_with_fresh_cache(self):
        launch = load_launch_sampling()
        engine = SimpleNamespace(
            forge_objects=SimpleNamespace(
                unet=SimpleNamespace(
                    model=SimpleNamespace(diffusion_model=self.model, current_weight_patches_uuid=None),
                    patches_uuid="base",
                )
            )
        )
        sampler = SimpleNamespace(
            p=SimpleNamespace(sd_model=engine),
            model_wrap_cfg=SimpleNamespace(),
            config=SimpleNamespace(total_steps=lambda count: count),
        )
        with (
            torch.inference_mode(),
            patch.object(self.model.txtfusion, "forward", wraps=self.model.txtfusion.forward) as fusion,
        ):
            for _stage in range(3):
                launch(sampler, 2, lambda: (self.forward(), self.forward()))
            self.assertEqual(fusion.call_count, 3)

    def test_dynamic_sampler_wrappers_and_lora_bypass_even_if_added_after_first_step(self):
        launch = load_launch_sampling()
        options = {}
        unet = SimpleNamespace(
            model=SimpleNamespace(diffusion_model=self.model, current_weight_patches_uuid="loaded"),
            model_options=options,
            patches_uuid="loaded",
            has_online_lora=lambda: False,
        )
        sampler = SimpleNamespace(
            p=SimpleNamespace(sd_model=SimpleNamespace(forge_objects=SimpleNamespace(unet=unet))),
            model_wrap_cfg=SimpleNamespace(),
            config=SimpleNamespace(total_steps=lambda count: count),
        )
        for key in ("model_function_wrapper", "conditioning_modifiers", "sampler_post_cfg_function"):
            with (
                self.subTest(key=key),
                torch.inference_mode(),
                patch.object(self.model.txtfusion, "forward", wraps=self.model.txtfusion.forward) as fusion,
            ):
                options.clear()

                def run(key=key):
                    self.forward()
                    options[key] = [object()]
                    self.forward()
                    self.forward()
                    self.assertEqual(self.model._text_conditioning_cache.retained_bytes, 0)

                launch(sampler, 3, run)
                self.assertEqual(fusion.call_count, 3)
        options.clear()
        unet.has_online_lora = lambda: True
        with (
            torch.inference_mode(),
            patch.object(self.model.txtfusion, "forward", wraps=self.model.txtfusion.forward) as fusion,
        ):
            launch(sampler, 2, lambda: (self.forward(), self.forward()))
            self.assertEqual(fusion.call_count, 2)

    def test_non_krea_sampler_still_launches(self):
        launch = load_launch_sampling()
        sampler = SimpleNamespace(
            p=SimpleNamespace(
                sd_model=SimpleNamespace(
                    forge_objects=SimpleNamespace(unet=SimpleNamespace(model=SimpleNamespace(diffusion_model=object())))
                )
            ),
            model_wrap_cfg=SimpleNamespace(),
            config=SimpleNamespace(total_steps=lambda count: count),
        )
        callback = Mock(return_value="ordinary generation")
        self.assertEqual(launch(sampler, 4, callback), "ordinary generation")
        callback.assert_called_once_with()

    def test_mid_sampling_model_patcher_replacement_disables_old_scope(self):
        launch = load_launch_sampling()
        unet = SimpleNamespace(
            model=SimpleNamespace(diffusion_model=self.model, current_weight_patches_uuid="loaded"),
            model_options={},
            patches_uuid="loaded",
        )
        forge_objects = SimpleNamespace(unet=unet)
        sampler = SimpleNamespace(
            p=SimpleNamespace(sd_model=SimpleNamespace(forge_objects=forge_objects)),
            model_wrap_cfg=SimpleNamespace(),
            config=SimpleNamespace(total_steps=lambda count: count),
        )

        def run():
            self.forward()
            self.model.txtmlp[-1].weight.data.add_(0.1)
            forge_objects.unet = SimpleNamespace(model=unet.model, model_options={}, patches_uuid="loaded")
            self.forward()
            self.assertEqual(self.model._text_conditioning_cache.retained_bytes, 0)

        with (
            torch.inference_mode(),
            patch.object(self.model.txtfusion, "forward", wraps=self.model.txtfusion.forward) as fusion,
        ):
            launch(sampler, 2, run)
            self.assertEqual(fusion.call_count, 2)


if __name__ == "__main__":
    unittest.main()
