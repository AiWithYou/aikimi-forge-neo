"""Zero-sum prompt weights keep finite conditional contrasts and hook shapes."""

import unittest

import torch

from backend.misc.context_windows import IndexListContextHandler
from backend.sampling import sampling_function
from backend.sampling.condition import compile_conditions, compile_weighted_conditions
from tools.tests import test_sampling_composition_batches as batch_fixture


class ZeroSumWeightsTests(unittest.TestCase):
    def setUp(self):
        batch_fixture.SamplingCompositionBatchTests.setUp(self)

    def run_euler(self, *args, **kwargs):
        return batch_fixture.SamplingCompositionBatchTests.run_euler(self, *args, **kwargs)

    def test_opposite_weights_match_existing_combine_for_nonzero_negative_prompt(self):
        for prompts in (["a:1 AND b:-1"], ["a:1 AND a:-1"], ["a:1 AND b:-1", "c:1 AND a:-1"]):
            with self.subTest(prompts=prompts):
                self.run_euler(prompts, negative_prompt="b")

    def test_mixed_zero_and_nonzero_totals_keep_image_video_and_dictionary_shapes(self):
        for ndim in (4, 5):
            for dictionary in (False, True):
                for scale in (1.0, 2.0):
                    with self.subTest(ndim=ndim, dictionary=dictionary, scale=scale):
                        self.run_euler(
                            ["a:1 AND b:-1", "c:1 AND a:0.5"],
                            negative_prompt="b",
                            ndim=ndim,
                            dictionary=dictionary,
                            scale=scale,
                        )

    def test_all_zero_weights_return_negative_prediction(self):
        for scale in (1.0, 2.0):
            with self.subTest(scale=scale):
                self.run_euler(["a:0 AND b:0", "c:0"], negative_prompt="b", scale=scale)

    def test_decimal_cancellation_matches_combine_in_both_orders_and_mixed_batch(self):
        for prompts in (
            ["a:0.1 AND b:0.2 AND c:-0.3"],
            ["a:0.3 AND b:-0.1 AND c:-0.2"],
            ["a:0.1 AND b:0.2 AND c:-0.3", "c"],
            ["c", "a:0.3 AND b:-0.1 AND c:-0.2"],
        ):
            with self.subTest(prompts=prompts):
                self.run_euler(prompts, negative_prompt="b")

    def test_small_uncancelled_weight_keeps_existing_average_hook(self):
        calls = []

        def post_cfg(args):
            calls.append(args)
            return args["denoised"]

        self.run_euler(["a:0.000001"], negative_prompt="b", model_options={"sampler_post_cfg_function": [post_cfg]})
        torch.testing.assert_close(calls[0]["cond_denoised"], torch.ones(1, 1, 2, 2))

    def test_post_cfg_gets_effective_zero_sum_condition_and_existing_nonzero_average(self):
        calls = []

        def post_cfg(args):
            calls.append(args)
            return args["denoised"]

        self.run_euler(
            ["a:1 AND b:-1", "c:1 AND a:0.5"],
            negative_prompt="b",
            model_options={"sampler_post_cfg_function": [post_cfg]},
        )
        self.assertEqual(len(calls), 1)
        expected = torch.tensor([1.0, 3.0]).reshape(2, 1, 1, 1).expand(2, 1, 2, 2)
        torch.testing.assert_close(calls[0]["cond_denoised"], expected)
        torch.testing.assert_close(calls[0]["uncond_denoised"], torch.full_like(expected, 2.0))

    def test_custom_linear_cfg_keeps_existing_protocol_for_zero_sum_and_zero_weights(self):
        def linear_cfg(args):
            self.assertTrue(torch.isfinite(args["cond_denoised"]).all())
            self.assertEqual(args["cond"].shape, args["uncond"].shape)
            return args["uncond"] + args["cond_scale"] * (args["cond"] - args["uncond"])

        for prompts in (["a:1 AND b:-1", "c"], ["a:0", "c:0"]):
            with self.subTest(prompts=prompts):
                self.run_euler(prompts, negative_prompt="b", model_options={"sampler_cfg_function": linear_cfg})

    def test_model_wrapper_does_not_receive_private_calc_keyword(self):
        calls = []

        def wrapper(apply_model, data):
            self.assertEqual(set(data), {"input", "timestep", "c", "cond_or_uncond"})
            calls.append(data)
            return apply_model(data["input"], data["timestep"], **data["c"])

        self.run_euler(["a:1 AND b:-1", "c"], negative_prompt="b", model_options={"model_function_wrapper": wrapper})
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["input"].shape, (6, 1, 2, 2))

    def test_zero_sum_calc_keeps_original_dtype_and_finite_values(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                x = torch.zeros(1, 1, 2, 2, dtype=dtype)
                cond = compile_weighted_conditions(torch.tensor([[[1.0]], [[2.0]]]), [[(0, 1.0), (1, -1.0)]])
                uncond = compile_conditions(torch.tensor([[[2.0]]]))
                actual, effective, negative = sampling_function.sampling_function_inner(
                    batch_fixture.TinyConditionModel(False),
                    x,
                    torch.ones(1, dtype=dtype),
                    uncond,
                    cond,
                    2.0,
                    return_full=True,
                )
                self.assertEqual(actual.dtype, dtype)
                torch.testing.assert_close(actual, torch.zeros_like(x))
                torch.testing.assert_close(effective, torch.ones_like(x))
                torch.testing.assert_close(negative, torch.full_like(x, 2.0))

    def test_real_context_windows_keep_zero_sum_contrast_and_two_tensor_protocol(self):
        context = IndexListContextHandler(context_length=3, context_overlap=1, dim=2)
        self.run_euler(["a:1 AND b:-1"], negative_prompt="b", ndim=5, frames=5, context_handler=context)

    def test_split_context_windows_apply_each_selected_conditions_signed_contrast(self):
        context = IndexListContextHandler(context_length=3, context_overlap=1, dim=2, split_conds_to_windows=True)
        expected = torch.tensor([0.0, 0.0, 1.0, 2.0, 2.0]).reshape(1, 1, 5, 1, 1).expand(1, 1, 5, 2, 2)
        self.run_euler(
            ["a:1 AND b:-1"], negative_prompt="b", ndim=5, frames=5, context_handler=context, expected=expected
        )

    def test_disjoint_regional_masks_keep_local_signed_cfg_and_uncovered_pixels(self):
        x = torch.zeros(1, 1, 2, 2)
        cond = compile_weighted_conditions(torch.tensor([[[1.0]], [[2.0]]]), [[(0, 1.0), (1, -1.0)]])
        cond[0]["mask"] = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
        cond[1]["mask"] = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]])
        uncond = compile_conditions(torch.tensor([[[3.0]]]))
        actual, effective, negative = sampling_function.sampling_function_inner(
            batch_fixture.TinyConditionModel(False), x, torch.ones(1), uncond, cond, 2.0, return_full=True
        )
        torch.testing.assert_close(actual, torch.tensor([[[[-1.0, 5.0], [3.0, 3.0]]]]))
        torch.testing.assert_close(effective, torch.tensor([[[[1.0, 4.0], [3.0, 3.0]]]]))
        torch.testing.assert_close(negative, torch.full_like(x, 3.0))

    def test_partial_areas_keep_signed_feathering(self):
        x = torch.zeros(1, 1, 2, 2)
        cond = compile_weighted_conditions(torch.tensor([[[1.0]], [[2.0]]]), [[(0, 1.0), (1, -1.0)]])
        cond[0]["area"] = (1, 1, 0, 0)
        cond[1]["area"] = (1, 1, 1, 1)
        uncond = compile_conditions(torch.tensor([[[3.0]]]))
        expected = torch.full_like(x, 3.0)
        for value, conditioning in zip((1.0, 2.0), cond, strict=True):
            region = sampling_function.get_area_and_mult(conditioning, x, torch.ones(1))
            height, width, y, x_start = region.area
            expected[:, :, y : y + height, x_start : x_start + width] += 2.0 * (value - 3.0) * region.mult
        actual = sampling_function.sampling_function_inner(
            batch_fixture.TinyConditionModel(False), x, torch.ones(1), uncond, cond, 2.0
        )
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
