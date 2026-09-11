import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from modules_forge.krea2_highres import (
    internal_exact_img2img_steps,
    uses_prompt_only_conditioning_cache,
)
from modules_forge.vram_canvas import (
    adaptive_step_count,
    axis_positions,
    axis_blend_weights,
    balanced_virtual_axis_origin,
    consensus_gated_residual,
    coordinate_seed,
    detail_score,
    extract_tile_context,
    frequency_detail_delta,
    novel_detail_delta,
    phase_normalized_tile_weight,
    phase_weave_residual,
    phase_weight_normalizers,
    plan_tiles,
    progressive_stage_sizes,
    replace_infotext_seed,
    resolve_core_overlap,
    resolve_halo,
    resolve_tile_size,
    spatial_activation_reduction,
    tile_weight_mask,
    vram_canvas_work_bytes_per_pixel,
)
from tools.vram_canvas_highres import (
    _find_numeric_total_bytes,
    build_tile_payload,
    krea2_profile_cli_defaults,
    pad_tile_for_diffusion,
    replace_infotext_prompts,
)

ROOT = Path(__file__).resolve().parents[2]


class Krea2ProfileTests(unittest.TestCase):
    def test_prompt_only_conditioning_cache_is_explicit_opt_in(self):
        class PromptOnly:
            conditioning_cache_is_prompt_only = True

        self.assertTrue(uses_prompt_only_conditioning_cache(PromptOnly()))
        self.assertFalse(uses_prompt_only_conditioning_cache(object()))

    def test_texture_rich_profile_maps_shared_values_to_cli_destinations(self):
        defaults = krea2_profile_cli_defaults("texture_rich_4k")

        self.assertEqual(defaults["phase_count"], 2)
        self.assertEqual((defaults["minimum_steps"], defaults["steps"]), (6, 6))
        self.assertEqual((defaults["coarse_denoise"], defaults["denoise"]), (0.22, 0.18))
        self.assertEqual(defaults["detail_gain"], 1.55)
        self.assertEqual(defaults["novel_detail_gain"], 1.6)
        self.assertEqual(defaults["novel_detail_consensus_sigma"], 2.0)
        self.assertEqual(defaults["novel_detail_consensus_strength"], 4.0)
        self.assertEqual(defaults["finish_detail_strength"], 0.85)
        self.assertEqual(defaults["finish_detail_radius"], 1.4)
        self.assertEqual(defaults["finish_detail_threshold"], 0.4)
        self.assertEqual(defaults["finish_max_detail_delta"], 10.0)

    def test_phaseweave_profile_maps_the_shared_merge_mode_and_exact_budget(self):
        defaults = krea2_profile_cli_defaults("phaseweave_4k")

        self.assertEqual(defaults["merge_mode"], "phase_weave")
        self.assertEqual(defaults["phase_count"], 2)
        self.assertEqual((defaults["minimum_steps"], defaults["steps"]), (6, 6))
        self.assertEqual((defaults["coarse_denoise"], defaults["denoise"]), (0.20, 0.16))

    def test_exact_steps_scope_is_nested_and_exception_safe(self):
        original = {"existing": 7, "img2img_fix_steps": False}
        processing = SimpleNamespace(
            override_settings=original,
            override_settings_restore_afterwards=False,
        )

        with self.assertRaisesRegex(RuntimeError, "stop"):
            with internal_exact_img2img_steps(processing):
                outer = processing.override_settings
                self.assertIsNot(outer, original)
                self.assertTrue(outer["img2img_fix_steps"])
                self.assertTrue(processing.override_settings_restore_afterwards)
                with internal_exact_img2img_steps(processing):
                    self.assertIsNot(processing.override_settings, outer)
                    self.assertTrue(processing.override_settings["img2img_fix_steps"])
                self.assertIs(processing.override_settings, outer)
                raise RuntimeError("stop")

        self.assertIs(processing.override_settings, original)
        self.assertEqual(original["img2img_fix_steps"], False)
        self.assertFalse(processing.override_settings_restore_afterwards)


class InfotextPromptTests(unittest.TestCase):
    def test_effective_prompts_replace_source_prompts_and_preserve_settings(self):
        source = (
            "old prompt\n"
            "Negative prompt: old negative\n"
            "Steps: 4, Sampler: test, Seed: 123, Size: 64x48"
        )
        updated = replace_infotext_prompts(source, "effective prompt", "effective negative")
        self.assertEqual(
            updated,
            "effective prompt\n"
            "Negative prompt: effective negative\n"
            "Steps: 4, Sampler: test, Seed: 123, Size: 64x48",
        )

    def test_empty_effective_negative_removes_stale_source_negative(self):
        source = "old prompt\nNegative prompt: stale\nSteps: 4, Size: 64x48"
        updated = replace_infotext_prompts(source, "effective prompt", "")
        self.assertEqual(updated, "effective prompt\nSteps: 4, Size: 64x48")


class BudgetPlannerTests(unittest.TestCase):
    def test_tile_grows_with_vram_and_stays_in_bounds(self):
        sizes = [resolve_tile_size(value) for value in (8, 12, 24)]
        self.assertEqual(sizes, sorted(sizes))
        self.assertEqual(sizes[-1], 1280)

    def test_explicit_tile_is_aligned_down(self):
        self.assertEqual(resolve_tile_size(24, requested_tile_size=1000), 960)

    def test_impossible_budget_and_oversized_explicit_tile_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "cannot fit"):
            resolve_tile_size(4)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            resolve_tile_size(8, requested_tile_size=1024)

    def test_rejects_impossible_vram_fraction(self):
        with self.assertRaisesRegex(ValueError, "fraction"):
            resolve_tile_size(8, use_fraction=1.1)

    def test_8k_spatial_ratio_is_independent_of_model_weights(self):
        self.assertEqual(spatial_activation_reduction(8192, 8192, 1024), 64.0)

    def test_memory_endpoint_parser_uses_cuda_system_total(self):
        value = {"ram": {"total": 999}, "cuda": {"system": {"total": 24 * 1024**3}}}
        self.assertEqual(_find_numeric_total_bytes(value), 24 * 1024**3)

    def test_phaseweave_disk_preflight_counts_both_completed_phases(self):
        self.assertEqual(
            vram_canvas_work_bytes_per_pixel(
                phase_count=2, merge_mode="consensus", novel_detail=False
            ),
            32,
        )
        self.assertEqual(
            vram_canvas_work_bytes_per_pixel(
                phase_count=2, merge_mode="phase_weave", novel_detail=False
            ),
            52,
        )
        self.assertEqual(
            vram_canvas_work_bytes_per_pixel(
                phase_count=2, merge_mode="phase_weave", novel_detail=True
            ),
            84,
        )

    def test_cli_tile_payload_forces_and_restores_exact_img2img_steps(self):
        args = SimpleNamespace(
            sampler="test sampler",
            scheduler="test scheduler",
            cfg=1.0,
            distilled_cfg=1.15,
        )
        payload, _ = build_tile_payload(
            args,
            Image.new("RGB", (16, 16)),
            "prompt",
            "negative",
            seed=1,
            steps=6,
            denoise=0.16,
        )

        self.assertEqual(payload["steps"], 6)
        self.assertEqual(payload["override_settings"], {"img2img_fix_steps": True})
        self.assertTrue(payload["override_settings_restore_afterwards"])


class ProgressivePlanTests(unittest.TestCase):
    def test_8x_scale_uses_three_2x_stages(self):
        self.assertEqual(
            progressive_stage_sizes(1024, 1024, 8192, 8192),
            [(2048, 2048), (4096, 4096), (8192, 8192)],
        )

    def test_final_delivery_size_is_exact(self):
        sizes = progressive_stage_sizes(1024, 768, 3840, 2160)
        self.assertEqual(sizes[-1], (3840, 2160))
        previous = (1024, 768)
        for width, height in sizes:
            self.assertLessEqual(width / previous[0], 2.0)
            self.assertLessEqual(height / previous[1], 2.0)
            previous = (width, height)


class TileGeometryTests(unittest.TestCase):
    def test_auto_halo_and_overlap_leave_a_positive_core(self):
        halo = resolve_halo(768)
        core = 768 - halo * 2
        overlap = resolve_core_overlap(core, halo)
        self.assertEqual((halo, core, overlap), (96, 576, 48))

    def test_context_never_exceeds_payload_and_canvas_is_covered(self):
        plans = plan_tiles(1500, 900, tile_size=768, halo=96, core_overlap=48, phase_count=1)
        coverage = np.zeros((900, 1500), dtype=np.float32)
        for tile in plans:
            self.assertLessEqual(tile.context_width, 768)
            self.assertLessEqual(tile.context_height, 768)
            coverage[tile.core_y0 : tile.core_y1, tile.core_x0 : tile.core_x1] += tile_weight_mask(tile)
        self.assertTrue(np.all(coverage > 0))

    def test_second_phase_changes_grid_but_keeps_coverage(self):
        one_phase = plan_tiles(1500, 900, tile_size=768, halo=96, core_overlap=48, phase_count=1)
        two_phase = plan_tiles(1500, 900, tile_size=768, halo=96, core_overlap=48, phase_count=2)
        self.assertGreater(len(two_phase), len(one_phase))
        self.assertEqual({tile.phase for tile in two_phase}, {0, 1})

    def test_smoothstep_overlap_is_complementary(self):
        left = axis_blend_weights(7, 0, 4)[-4:]
        right = axis_blend_weights(7, 4, 0)[:4]
        np.testing.assert_allclose(left + right, np.ones(4), atol=1e-6)

    def test_each_shifted_phase_is_normalized_to_unit_coverage(self):
        plans = plan_tiles(1500, 900, tile_size=768, halo=96, core_overlap=48, phase_count=2)
        normalizers = phase_weight_normalizers(plans, 1500, 900)
        for phase in (0, 1):
            coverage = np.zeros((900, 1500), dtype=np.float32)
            for tile in (item for item in plans if item.phase == phase):
                coverage[tile.core_y0 : tile.core_y1, tile.core_x0 : tile.core_x1] += phase_normalized_tile_weight(
                    tile,
                    normalizers,
                )
            np.testing.assert_allclose(coverage, 1.0, atol=2e-6)

    def test_phaseweave_virtual_grid_keeps_uniform_stride_and_nominal_overlap(self):
        plans = plan_tiles(
            2897,
            4096,
            tile_size=1280,
            halo=160,
            core_overlap=80,
            phase_count=2,
            virtual_padding=True,
        )
        phase0 = [tile for tile in plans if tile.phase == 0]
        phase1 = [tile for tile in plans if tile.phase == 1]
        self.assertEqual((len(phase0), len(phase1)), (20, 24))
        self.assertEqual(
            sorted({tile.grid_core_x0 for tile in phase0}),
            [-572, 308, 1188, 2068],
        )
        self.assertEqual(
            sorted({tile.grid_core_y0 for tile in phase0}),
            [-192, 688, 1568, 2448, 3328],
        )
        self.assertEqual(
            sorted({tile.grid_core_x0 for tile in phase1}),
            [-132, 748, 1628, 2508],
        )
        self.assertEqual(
            sorted({tile.grid_core_y0 for tile in phase1}),
            [-632, 248, 1128, 2008, 2888, 3768],
        )
        self.assertEqual(
            (
                balanced_virtual_axis_origin(2897, 960, 80, phase_count=2),
                balanced_virtual_axis_origin(4096, 960, 80, phase_count=2),
            ),
            (308, 688),
        )
        for phase in (phase0, phase1):
            for coordinates in (
                sorted({tile.grid_core_x0 for tile in phase}),
                sorted({tile.grid_core_y0 for tile in phase}),
            ):
                self.assertTrue(
                    all(b - a == 880 for a, b in zip(coordinates, coordinates[1:]))
                )
        self.assertLessEqual(
            max(
                max(tile.previous_x_overlap, tile.next_x_overlap)
                for tile in plans
            ),
            80,
        )
        self.assertGreaterEqual(min(tile.core_width for tile in plans), 388)
        self.assertGreaterEqual(min(tile.core_height for tile in plans), 328)
        self.assertLessEqual(
            max(
                max(tile.previous_y_overlap, tile.next_y_overlap)
                for tile in plans
            ),
            80,
        )

    def test_phaseweave_virtual_grid_normalizes_each_phase_to_unit_coverage(self):
        plans = plan_tiles(
            2897,
            4096,
            tile_size=1280,
            halo=160,
            core_overlap=80,
            phase_count=2,
            virtual_padding=True,
        )
        normalizers = phase_weight_normalizers(plans, 2897, 4096)
        for phase in (0, 1):
            coverage = np.zeros((4096, 2897), dtype=np.float32)
            for tile in (item for item in plans if item.phase == phase):
                coverage[
                    tile.core_y0 : tile.core_y1,
                    tile.core_x0 : tile.core_x1,
                ] += phase_normalized_tile_weight(tile, normalizers)
            np.testing.assert_allclose(coverage, 1.0, atol=2e-6)

    def test_virtual_context_uses_edge_padding_without_changing_canvas_pixels(self):
        values = np.arange(64 * 128 * 3, dtype=np.uint8).reshape(64, 128, 3)
        image = Image.fromarray(values, "RGB")
        plans = plan_tiles(
            128,
            64,
            tile_size=384,
            halo=48,
            core_overlap=48,
            phase_count=2,
            virtual_padding=True,
        )
        shifted = next(tile for tile in plans if tile.phase == 1)
        context = np.asarray(extract_tile_context(image, shifted))
        self.assertEqual(context.shape, (384, 384, 3))
        left, top, _, _ = shifted.context_padding
        np.testing.assert_array_equal(context[0, 0], values[0, 0])
        np.testing.assert_array_equal(
            context[top : top + 64, left : left + 128],
            values,
        )

    def test_padding_repeats_edges_to_model_alignment(self):
        values = np.arange(17 * 19 * 3, dtype=np.uint8).reshape(17, 19, 3)
        padded = np.asarray(pad_tile_for_diffusion(Image.fromarray(values, "RGB")))
        self.assertEqual(padded.shape, (32, 32, 3))
        np.testing.assert_array_equal(padded[:17, :19], values)
        np.testing.assert_array_equal(padded[-1, -1], values[-1, -1])


class ResidualMergeTests(unittest.TestCase):
    def test_infotext_seed_rewrite_does_not_touch_prompt_literal(self):
        infotext = "prompt contains Seed: 999 as text\n" "Steps: 4, Sampler: test, Size: 512x512, Seed: 456"
        updated = replace_infotext_seed(infotext, 123)
        self.assertIn("prompt contains Seed: 999 as text", updated)
        self.assertIn("Size: 512x512, Seed: 123", updated)

    def test_constant_color_shift_does_not_replace_global_tone(self):
        base = np.full((32, 32, 3), [100, 120, 140], dtype=np.uint8)
        refined = np.full((32, 32, 3), [140, 90, 160], dtype=np.uint8)
        delta, stats = frequency_detail_delta(refined, base, radius=3)
        np.testing.assert_allclose(delta, 0, atol=1e-5)
        self.assertLess(stats["mean_gate"], 1.0)

    def test_new_checkerboard_detail_is_suppressed_on_flat_base(self):
        base = np.full((32, 32, 3), 128, dtype=np.uint8)
        checker = ((np.indices((32, 32)).sum(axis=0) % 2) * 255).astype(np.uint8)
        refined = np.repeat(checker[..., None], 3, axis=2)
        delta, stats = frequency_detail_delta(refined, base, radius=2, max_delta=12.0)
        np.testing.assert_allclose(delta, 0, atol=1e-6)
        self.assertEqual(stats["mean_base_detail_gate"], 0.0)
        self.assertEqual(stats["clipped_fraction"], 0.0)

    def test_novel_branch_can_propose_bounded_luminance_detail_on_flat_base(self):
        base = np.full((48, 48, 3), 128, dtype=np.uint8)
        refined = base.copy()
        refined[22:26, 8:40] = 154
        delta, stats = novel_detail_delta(
            refined,
            base,
            inner_radius=1,
            outer_radius=4,
            max_delta=8.0,
        )
        self.assertGreater(float(np.mean(np.abs(delta))), 0.05)
        self.assertLessEqual(float(np.max(np.abs(delta))), 8.0)
        np.testing.assert_allclose(delta[..., 0], delta[..., 1], atol=1e-6)
        np.testing.assert_allclose(delta[..., 1], delta[..., 2], atol=1e-6)
        self.assertGreater(stats["mean_novelty_gate"], 0.9)

    def test_novel_branch_requires_cross_phase_agreement_at_merge(self):
        base = np.full((48, 48, 3), 128, dtype=np.uint8)
        refined = base.copy()
        refined[22:26, 8:40] = 154
        proposal, _ = novel_detail_delta(refined, base)
        weights = np.full((48, 48), 2.0, dtype=np.float32)
        agreed_energy = (
            np.mean(np.square(proposal), axis=2, dtype=np.float32) * 2.0
        )
        agreed, _, _ = consensus_gated_residual(
            proposal * 2.0,
            weights,
            agreed_energy,
            sigma=0.75,
            strength=8.0,
        )
        one_phase_energy = np.mean(
            np.square(proposal), axis=2, dtype=np.float32
        )
        disagreed, _, _ = consensus_gated_residual(
            proposal,
            weights,
            one_phase_energy,
            sigma=0.75,
            strength=8.0,
        )
        self.assertGreater(float(np.mean(np.abs(agreed))), 0.05)
        self.assertLess(
            float(np.mean(np.abs(disagreed))),
            float(np.mean(np.abs(agreed))) * 0.2,
        )

    def test_base_detail_gate_preserves_refinement_of_existing_texture(self):
        signs = np.where(np.indices((32, 32)).sum(axis=0) % 2, 1, -1).astype(np.int16)
        base = np.repeat(np.clip(128 + signs[..., None] * 24, 0, 255), 3, axis=2).astype(np.uint8)
        refined = np.repeat(np.clip(128 + signs[..., None] * 48, 0, 255), 3, axis=2).astype(np.uint8)
        delta, stats = frequency_detail_delta(refined, base, radius=2)
        self.assertGreater(float(np.mean(np.abs(delta))), 5.0)
        self.assertGreater(stats["mean_base_detail_gate"], 0.5)

    def test_zero_base_detail_sigma_disables_flat_region_gate(self):
        base = np.full((32, 32, 3), 128, dtype=np.uint8)
        checker = ((np.indices((32, 32)).sum(axis=0) % 2) * 255).astype(np.uint8)
        refined = np.repeat(checker[..., None], 3, axis=2)
        delta, stats = frequency_detail_delta(
            refined,
            base,
            radius=2,
            max_delta=12.0,
            base_detail_sigma=0,
        )
        self.assertGreater(float(np.mean(np.abs(delta))), 1.0)
        self.assertLessEqual(float(np.max(np.abs(delta))), 12.0)
        self.assertEqual(stats["mean_base_detail_gate"], 1.0)
        self.assertGreater(stats["clipped_fraction"], 0.0)

    def test_base_detail_sigma_rejects_negative_and_nonfinite_values(self):
        base = np.full((8, 8, 3), 128, dtype=np.uint8)
        for value in (-1, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "base detail sigma"):
                    frequency_detail_delta(base, base, base_detail_sigma=value)

    def test_structure_disagreement_reduces_residual(self):
        base = np.full((32, 32, 3), 128, dtype=np.uint8)
        texture = ((np.indices((32, 32)).sum(axis=0) % 2) * 30 - 15).astype(np.int16)
        close = np.clip(base.astype(np.int16) + texture[..., None], 0, 255).astype(np.uint8)
        far = np.clip(close.astype(np.int16) + 80, 0, 255).astype(np.uint8)
        close_delta, _ = frequency_detail_delta(close, base, radius=2, base_detail_sigma=0)
        far_delta, _ = frequency_detail_delta(far, base, radius=2, base_detail_sigma=0)
        self.assertLess(
            float(np.mean(np.abs(far_delta))),
            float(np.mean(np.abs(close_delta))),
        )

    def test_detail_score_and_step_budget_distinguish_flat_texture(self):
        flat = np.full((64, 64, 3), 128, dtype=np.uint8)
        checker = np.repeat(
            (((np.indices((64, 64)).sum(axis=0) % 2) * 255)[..., None]),
            3,
            axis=2,
        ).astype(np.uint8)
        flat_score = detail_score(flat)
        checker_score = detail_score(checker)
        self.assertEqual(adaptive_step_count(flat_score, 2, 6), 2)
        self.assertGreater(
            adaptive_step_count(checker_score, 2, 6),
            adaptive_step_count(flat_score, 2, 6),
        )

    def test_coordinate_seed_is_repeatable_and_spatially_distinct(self):
        first = coordinate_seed(1234, 0, 0, 0)
        self.assertEqual(first, coordinate_seed(1234, 0, 0, 0))
        self.assertNotEqual(first, coordinate_seed(1234, 0, 576, 0))
        self.assertNotEqual(first, coordinate_seed(1234, 0, -288, 0))
        self.assertLess(first, 2**32)

    def test_consensus_preserves_identical_and_single_residuals(self):
        residual = np.full((2, 3, 3), [30.0, -12.0, 6.0], dtype=np.float32)
        weights = np.full((2, 3), 2.0, dtype=np.float32)
        first = residual * weights[..., None]
        energy = np.mean(np.square(residual), axis=2, dtype=np.float32) * weights
        merged, gate, disagreement = consensus_gated_residual(first, weights, energy)
        np.testing.assert_allclose(merged, residual, atol=1e-6)
        np.testing.assert_allclose(gate, 1.0, atol=1e-6)
        np.testing.assert_allclose(disagreement, 0.0, atol=1e-6)

    def test_consensus_relative_gate_rejects_directional_conflict(self):
        same_sign_first = np.full((1, 1, 3), 52.0, dtype=np.float32)
        same_sign_energy = np.full((1, 1), 30.0**2 + 22.0**2, dtype=np.float32)
        opposite_first = np.zeros((1, 1, 3), dtype=np.float32)
        opposite_energy = np.full((1, 1), 2 * 8.0**2, dtype=np.float32)
        weights = np.full((1, 1), 2.0, dtype=np.float32)
        same_sign, same_gate, _ = consensus_gated_residual(same_sign_first, weights, same_sign_energy)
        opposite, opposite_gate, _ = consensus_gated_residual(opposite_first, weights, opposite_energy)
        self.assertGreater(float(same_gate[0, 0]), 0.9)
        self.assertAlmostEqual(float(opposite_gate[0, 0]), float(np.exp(-2.0)), places=6)
        np.testing.assert_allclose(
            same_sign,
            np.full_like(same_sign, 26.0 * float(same_gate[0, 0])),
            atol=1e-5,
        )
        np.testing.assert_allclose(opposite, 0.0, atol=1e-6)

    def test_consensus_weighted_moments_match_hand_calculation(self):
        weights = np.array([[1.0]], dtype=np.float32)
        first = np.full((1, 1, 3), 5.0, dtype=np.float32)
        energy = np.array([[100.0]], dtype=np.float32)
        merged, gate, disagreement = consensus_gated_residual(first, weights, energy)
        expected_gate = np.exp(-4.0 * 75.0 / (100.0 + 8.0**2))
        self.assertAlmostEqual(float(gate[0, 0]), float(expected_gate), places=6)
        self.assertAlmostEqual(float(disagreement[0, 0]), float(np.sqrt(75.0)), places=6)
        np.testing.assert_allclose(merged, 5.0 * expected_gate, atol=1e-6)

    def test_consensus_zero_coverage_and_disabled_gate_are_safe(self):
        zero = np.zeros((1, 2), dtype=np.float32)
        merged, gate, disagreement = consensus_gated_residual(
            np.zeros((1, 2, 3), dtype=np.float32),
            zero,
            zero,
        )
        np.testing.assert_array_equal(merged, 0.0)
        np.testing.assert_array_equal(gate, 0.0)
        np.testing.assert_array_equal(disagreement, 0.0)

        first = np.full((1, 1, 3), 5.0, dtype=np.float32)
        weights = np.ones((1, 1), dtype=np.float32)
        energy = np.full((1, 1), 100.0, dtype=np.float32)
        disabled, disabled_gate, _ = consensus_gated_residual(first, weights, energy, sigma=0)
        np.testing.assert_allclose(disabled, 5.0)
        np.testing.assert_allclose(disabled_gate, 1.0)

    def test_phaseweave_does_not_average_opposite_fine_lines_to_zero(self):
        base = np.full((64, 64, 3), 128.0, dtype=np.float32)
        base[:, 31:33] = 150.0
        phase0 = np.zeros_like(base)
        phase1 = np.zeros_like(base)
        phase0[:, 31:33] = 12.0
        phase1[:, 31:33] = -8.0
        weights = np.ones((64, 64), dtype=np.float32)
        energy0 = np.mean(np.square(phase0), axis=2, dtype=np.float32)
        energy1 = np.mean(np.square(phase1), axis=2, dtype=np.float32)

        consensus, _, _ = consensus_gated_residual(
            phase0 + phase1,
            weights * 2.0,
            energy0 + energy1,
        )
        woven, diagnostics = phase_weave_residual(
            phase0,
            weights,
            energy0,
            phase1,
            weights,
            energy1,
            base_rgb=base,
            quality_radius=2,
            propagation_radius=2,
            island_min_area=0,
            low_frequency_sigma=2,
        )

        self.assertLess(float(np.mean(np.abs(consensus[:, 31:33]))), 1.0)
        self.assertGreater(float(np.mean(woven[:, 31:33])), 5.0)
        self.assertLess(float(np.mean(diagnostics["support_weight"])), 0.01)
        self.assertGreaterEqual(float(np.min(diagnostics["confidence_gain"])), 0.8999)

    def test_phaseweave_selects_local_detail_and_feathers_only_the_boundary(self):
        yy, xx = np.indices((64, 96))
        checker = ((xx + yy) % 2).astype(np.float32) * 2.0 - 1.0
        base = np.repeat((128.0 + checker * 5.0)[..., None], 3, axis=2)
        amplitude0 = np.where(xx < 48, 12.0, 2.0)
        amplitude1 = np.where(xx < 48, 2.0, 12.0)
        phase0 = np.repeat((checker * amplitude0)[..., None], 3, axis=2)
        phase1 = np.repeat((checker * amplitude1)[..., None], 3, axis=2)
        weights = np.ones((64, 96), dtype=np.float32)
        energy0 = np.mean(np.square(phase0), axis=2, dtype=np.float32)
        energy1 = np.mean(np.square(phase1), axis=2, dtype=np.float32)

        woven, diagnostics = phase_weave_residual(
            phase0,
            weights,
            energy0,
            phase1,
            weights,
            energy1,
            base_rgb=base,
            quality_radius=3,
            propagation_radius=3,
            island_min_area=0,
            low_frequency_sigma=2,
        )

        np.testing.assert_array_equal(diagnostics["selected_phase"][:, :32], 0)
        np.testing.assert_array_equal(diagnostics["selected_phase"][:, 64:], 1)
        np.testing.assert_allclose(diagnostics["phase1_mix"][:, :28], 0.0)
        np.testing.assert_allclose(diagnostics["phase1_mix"][:, 68:], 1.0)
        self.assertFalse(np.any(diagnostics["boundary"][:, :28]))
        self.assertFalse(np.any(diagnostics["boundary"][:, 68:]))
        self.assertTrue(np.any(diagnostics["boundary"][:, 42:54]))
        self.assertGreater(float(np.mean(np.abs(woven[:, :28]))), 7.0)
        self.assertGreater(float(np.mean(np.abs(woven[:, 68:]))), 7.0)

    def test_phaseweave_single_phase_coverage_passes_without_attenuation(self):
        yy, xx = np.indices((16, 16))
        checker = ((xx + yy) % 2).astype(np.float32) * 2.0 - 1.0
        phase0 = np.repeat((checker * 5.0)[..., None], 3, axis=2)
        base = np.repeat((128.0 + checker * 4.0)[..., None], 3, axis=2)
        weights0 = np.ones((16, 16), dtype=np.float32)
        energies0 = np.mean(np.square(phase0), axis=2, dtype=np.float32)
        zeros = np.zeros((16, 16), dtype=np.float32)

        woven, diagnostics = phase_weave_residual(
            phase0,
            weights0,
            energies0,
            np.zeros_like(phase0),
            zeros,
            zeros,
            base_rgb=base,
            low_frequency_luma_gain=1.0,
            low_frequency_chroma_gain=1.0,
            highlight_low_frequency_scale=1.0,
            island_min_area=0,
        )

        np.testing.assert_allclose(woven, phase0, atol=1e-6)
        np.testing.assert_allclose(diagnostics["confidence_gain"], 1.0)

    def test_phaseweave_near_tie_can_reject_both_candidates_for_the_input(self):
        base = np.full((96, 96, 3), 240.0, dtype=np.float32)
        phase0 = np.full_like(base, 24.0)
        phase1 = np.full_like(base, -24.0)
        weights = np.ones((96, 96), dtype=np.float32)
        energy0 = np.mean(np.square(phase0), axis=2, dtype=np.float32)
        energy1 = np.mean(np.square(phase1), axis=2, dtype=np.float32)

        woven, diagnostics = phase_weave_residual(
            phase0,
            weights,
            energy0,
            phase1,
            weights,
            energy1,
            base_rgb=base,
            island_min_area=0,
        )

        self.assertFalse(np.any(diagnostics["boundary"]))
        np.testing.assert_array_equal(diagnostics["selected_phase"], 2)
        np.testing.assert_allclose(woven, 0.0, atol=1e-6)
        self.assertLess(float(np.max(np.abs(diagnostics["selection_score"]))), 0.03)

    def test_phaseweave_rejects_a_decisive_but_unfaithful_winner(self):
        yy, xx = np.indices((64, 64))
        checker = ((xx + yy) % 2).astype(np.float32) * 2.0 - 1.0
        base = np.full((64, 64, 3), 128.0, dtype=np.float32)
        phase0 = np.repeat((24.0 + checker * 30.0)[..., None], 3, axis=2)
        phase1 = np.repeat((24.0 + checker * 10.0)[..., None], 3, axis=2)
        weights = np.ones((64, 64), dtype=np.float32)

        woven, diagnostics = phase_weave_residual(
            phase0,
            weights,
            np.mean(np.square(phase0), axis=2, dtype=np.float32),
            phase1,
            weights,
            np.mean(np.square(phase1), axis=2, dtype=np.float32),
            base_rgb=base,
            quality_radius=1,
            propagation_radius=1,
            island_min_area=0,
            low_frequency_sigma=2,
            fidelity_reject_threshold=0.8,
        )

        self.assertGreater(
            float(np.mean(np.abs(diagnostics["selection_score"]))),
            0.03,
        )
        np.testing.assert_array_equal(diagnostics["selected_phase"], 2)
        np.testing.assert_allclose(woven, 0.0, atol=1e-6)

    def test_phaseweave_preserves_high_frequency_more_than_low_frequency(self):
        yy, xx = np.indices((64, 64))
        checker = ((xx + yy) % 2).astype(np.float32) * 2.0 - 1.0
        base = np.repeat((128.0 + checker * 4.0)[..., None], 3, axis=2)
        weights = np.ones((64, 64), dtype=np.float32)
        zeros = np.zeros((64, 64), dtype=np.float32)
        low = np.full_like(base, 10.0)
        high = np.repeat((checker * 10.0)[..., None], 3, axis=2)

        low_woven, _ = phase_weave_residual(
            low,
            weights,
            np.mean(np.square(low), axis=2, dtype=np.float32),
            np.zeros_like(low),
            zeros,
            zeros,
            base_rgb=base,
            island_min_area=0,
        )
        high_woven, _ = phase_weave_residual(
            high,
            weights,
            np.mean(np.square(high), axis=2, dtype=np.float32),
            np.zeros_like(high),
            zeros,
            zeros,
            base_rgb=base,
            island_min_area=0,
        )

        self.assertLess(float(np.mean(np.abs(low_woven))), 4.0)
        self.assertGreater(float(np.mean(np.abs(high_woven))), 8.0)

    def test_phaseweave_removes_small_selection_islands_before_propagation(self):
        yy, xx = np.indices((80, 80))
        checker = ((xx + yy) % 2).astype(np.float32) * 2.0 - 1.0
        base = np.repeat((128.0 + checker * 4.0)[..., None], 3, axis=2)
        phase0 = np.repeat((checker * 9.0)[..., None], 3, axis=2)
        amplitude1 = np.full((80, 80), 5.0, dtype=np.float32)
        amplitude1[34:46, 34:46] = 14.0
        phase1 = np.repeat((checker * amplitude1)[..., None], 3, axis=2)
        weights = np.ones((80, 80), dtype=np.float32)

        _, diagnostics = phase_weave_residual(
            phase0,
            weights,
            np.mean(np.square(phase0), axis=2, dtype=np.float32),
            phase1,
            weights,
            np.mean(np.square(phase1), axis=2, dtype=np.float32),
            base_rgb=base,
            quality_radius=1,
            propagation_radius=4,
            island_min_area=200,
            low_frequency_sigma=2,
        )

        self.assertFalse(np.any(diagnostics["selected_phase"][36:44, 36:44] == 1))

    def test_phaseweave_support_is_squared_and_never_exceeds_ten_percent(self):
        yy, xx = np.indices((48, 48))
        checker = ((xx + yy) % 2).astype(np.float32) * 2.0 - 1.0
        base = np.repeat((128.0 + checker * 4.0)[..., None], 3, axis=2)
        phase = np.repeat((checker * 8.0)[..., None], 3, axis=2)
        weights = np.ones((48, 48), dtype=np.float32)
        energy = np.mean(np.square(phase), axis=2, dtype=np.float32)

        _, diagnostics = phase_weave_residual(
            phase,
            weights,
            energy,
            phase.copy(),
            weights,
            energy,
            base_rgb=base,
            quality_radius=2,
            propagation_radius=2,
            island_min_area=0,
            low_frequency_sigma=2,
        )

        self.assertLessEqual(float(np.max(diagnostics["support_weight"])), 0.100001)
        self.assertTrue(np.all(diagnostics["selected_phase"] == 3))


class DryRunIntegrationTests(unittest.TestCase):
    def test_cli_writes_a_reproducible_plan_without_forge(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.png"
            Image.new("RGB", (64, 48), (100, 120, 140)).save(source_path)
            output_root = root / "out"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "vram_canvas_highres.py"),
                    "--input",
                    str(source_path),
                    "--prompt",
                    "test image",
                    "--long-edge",
                    "128",
                    "--vram-budget-gib",
                    "8",
                    "--output-root",
                    str(output_root),
                    "--dry-run",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("DRY_RUN=1", completed.stdout)
            run_dirs = list(output_root.glob("vram_canvas_*"))
            self.assertEqual(len(run_dirs), 1)
            manifest = json.loads((run_dirs[0] / "run_manifest.json").read_text())
            self.assertEqual(manifest["target_size"], [128, 96])
            self.assertEqual(manifest["tile_size"], 576)
            self.assertEqual(manifest["stage_reports"][0]["tile_count"], 1)
            self.assertEqual(manifest["prompt"], "test image")
            self.assertEqual(manifest["negative_prompt"], "")


class _EchoForgeHandler(BaseHTTPRequestHandler):
    def log_message(self, _format, *_args):
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        if self.path == "/sdapi/v1/img2img":
            result = {"images": [payload["init_images"][0]], "info": "{}"}
        elif self.path == "/sdapi/v1/interrupt":
            result = {}
        else:
            self.send_error(404)
            return
        body = json.dumps(result).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class FullPipelineIntegrationTests(unittest.TestCase):
    def test_phaseweave_exports_both_independent_phase_candidates(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _EchoForgeHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with TemporaryDirectory() as directory:
                root = Path(directory)
                source_path = root / "source.png"
                yy, xx = np.indices((48, 64))
                values = np.stack(
                    (
                        (xx * 3) % 256,
                        (yy * 5) % 256,
                        ((xx + yy) * 2) % 256,
                    ),
                    axis=2,
                ).astype(np.uint8)
                Image.fromarray(values, "RGB").save(source_path)
                output_root = root / "out"
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "tools" / "vram_canvas_highres.py"),
                        "--input",
                        str(source_path),
                        "--prompt",
                        "test image",
                        "--long-edge",
                        "128",
                        "--vram-budget-gib",
                        "8",
                        "--api",
                        f"http://127.0.0.1:{server.server_port}",
                        "--output-root",
                        str(output_root),
                        "--phase-count",
                        "2",
                        "--merge-mode",
                        "phase_weave",
                        "--save-phase-candidates",
                        "--progress-interval",
                        "0",
                        "--no-progress-timeout",
                        "0",
                    ],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=True,
                )
                self.assertIn("IMAGE=", completed.stdout)
                run_dir = next(output_root.glob("vram_canvas_*"))
                phase_a = run_dir / "stage_01_phase_a_128x96.png"
                phase_b = run_dir / "stage_01_phase_b_128x96.png"
                selection_map = run_dir / "stage_01_phase_selection_128x96.png"
                self.assertTrue(phase_a.is_file())
                self.assertTrue(phase_b.is_file())
                self.assertTrue(selection_map.is_file())
                manifest = json.loads(
                    (run_dir / "run_manifest.json").read_text(encoding="utf-8")
                )
                self.assertTrue(manifest["save_phase_candidates"])
                stats = manifest["stage_reports"][0]["consensus_stats"]
                self.assertEqual(stats["phaseweave_phase0_candidate"], str(phase_a))
                self.assertEqual(stats["phaseweave_phase1_candidate"], str(phase_b))
                self.assertEqual(stats["phaseweave_selection_map"], str(selection_map))
                self.assertEqual(manifest["phaseweave"]["selection_mode"], "ternary_input_fallback")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_one_tile_echo_api_completes_frequency_merge_and_metadata(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), _EchoForgeHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with TemporaryDirectory() as directory:
                root = Path(directory)
                source_path = root / "source.png"
                values = np.zeros((48, 64, 3), dtype=np.uint8)
                values[..., 0] = np.arange(64, dtype=np.uint8)[None, :] * 3
                values[..., 1] = 100
                values[..., 2] = 160
                Image.fromarray(values, "RGB").save(source_path)
                output_root = root / "out"
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "tools" / "vram_canvas_highres.py"),
                        "--input",
                        str(source_path),
                        "--prompt",
                        "test image",
                        "--negative-prompt",
                        "avoid this",
                        "--long-edge",
                        "256",
                        "--vram-budget-gib",
                        "8",
                        "--api",
                        f"http://127.0.0.1:{server.server_port}",
                        "--output-root",
                        str(output_root),
                        "--progress-interval",
                        "0",
                        "--no-progress-timeout",
                        "0",
                    ],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=True,
                )
                self.assertIn("IMAGE=", completed.stdout)
                run_dir = next(output_root.glob("vram_canvas_*"))
                final_path = run_dir / "vram_canvas_highres.png"
                with Image.open(final_path) as final:
                    self.assertEqual(final.size, (256, 192))
                    report = json.loads(final.info["vram_canvas"])
                    parameters = final.info["parameters"]
                self.assertEqual(report["prompt"], "test image")
                self.assertEqual(report["negative_prompt"], "avoid this")
                self.assertTrue(parameters.startswith("test image\n"))
                self.assertIn("\nNegative prompt: avoid this\n", parameters)
                self.assertIn("Size: 256x192", parameters)
                self.assertEqual(report["stage_reports"][0]["processed_tile_count"], 1)
                self.assertEqual(len(report["stage_reports"]), 2)
                for stage in report["stage_reports"]:
                    self.assertEqual(stage["processed_tile_count"], 1)
                    self.assertAlmostEqual(stage["delta_stats"]["mean_abs_delta"], 0.0, places=5)
                work_files = list((run_dir / "work").glob("*"))
                self.assertEqual(work_files, [])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
