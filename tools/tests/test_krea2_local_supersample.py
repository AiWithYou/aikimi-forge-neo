import unittest

import numpy as np

from modules_forge.krea2_local_supersample import (
    Box,
    CandidateEvaluation,
    FOCUSED_FACE_PROMPT_SUFFIX,
    LOCAL_DETAIL_PROMPT_SUFFIX,
    MODE_FOCUSED_ROI_REWRITE,
    MODE_FULL_IMAGE_GRID,
    MODE_ROI_BOXES,
    PROFILE_FOCUSED_FACE_1536,
    PROFILE_ROI_ULTRA_2048,
    PROFILE_SAFE_1536,
    PROFILE_ULTRA_1536,
    agreement_mask,
    append_focused_face_guidance,
    append_local_detail_guidance,
    apply_canvas_residual,
    build_axis_normalizers,
    candidate_seed,
    cap_luma_chroma,
    enforce_maximum_tile_count,
    extract_padded_payload,
    filter_local_residual,
    focused_roi_difference_metrics,
    get_profile,
    lanczos_upscale,
    low_frequency_reject,
    normalized_tile_weight,
    parse_roi_boxes,
    plan_focused_rois,
    plan_local_tiles,
    roi_core_mask,
    round_trip_residual,
    select_candidate,
    select_tiles_for_rois,
    split_luma_chroma,
    strong_edge_guard,
    tile_composition_weights,
    validate_krea2_module_names,
    validate_request,
)


def valid_request(**overrides):
    values = {
        "mode": MODE_FULL_IMAGE_GRID,
        "profile": PROFILE_SAFE_1536,
        "roi_boxes": [],
        "payload": 512,
        "core": 384,
        "overlap": 64,
        "process_edge": 1536,
        "steps": 4,
        "denoise": 0.10,
        "candidate_count": 1,
        "luma_cap": 8.0,
        "chroma_cap": 2.0,
        "low_frequency_reject_radius": 12.0,
        "focused_context_scale": 2.0,
        "focused_rewrite_feather": 20.0,
        "allow_expensive_2048_full_grid": False,
        "maximum_tile_count": 256,
    }
    values.update(overrides)
    return values


def evaluation(
    residual,
    *,
    score=0.0,
    accepted=True,
    reason=None,
    c0=None,
    c1=None,
):
    zeros = np.zeros_like(residual, dtype=np.float32)
    return CandidateEvaluation(
        residual=np.asarray(residual, dtype=np.float32),
        c0_linear=zeros if c0 is None else np.asarray(c0, dtype=np.float32),
        c1_linear=zeros if c1 is None else np.asarray(c1, dtype=np.float32),
        stats={"quality_score": float(score)},
        accepted=accepted,
        rejection_reason=reason,
    )


class ProfileAndPromptTests(unittest.TestCase):
    def test_profiles_match_the_documented_defaults_and_are_isolated(self):
        self.assertEqual(
            get_profile(PROFILE_SAFE_1536),
            {
                "payload": 512,
                "core": 384,
                "overlap": 64,
                "process_edge": 1536,
                "steps": 4,
                "denoise": 0.10,
                "candidates": 1,
                "luma_cap": 8.0,
                "chroma_cap": 2.0,
                "low_frequency_reject_radius": 12.0,
                "context_scale": 2.0,
                "rewrite_feather": 20.0,
            },
        )
        self.assertEqual(get_profile(PROFILE_ULTRA_1536)["candidates"], 2)
        self.assertEqual(get_profile(PROFILE_ROI_ULTRA_2048)["process_edge"], 2048)
        focused = get_profile(PROFILE_FOCUSED_FACE_1536)
        self.assertEqual(focused["denoise"], 0.38)
        self.assertEqual(focused["steps"], 6)
        self.assertEqual(focused["context_scale"], 2.0)
        self.assertEqual(focused["rewrite_feather"], 20.0)
        copy = get_profile(PROFILE_SAFE_1536)
        copy["payload"] = 1
        self.assertEqual(get_profile(PROFILE_SAFE_1536)["payload"], 512)

    def test_guidance_preserves_the_exact_prefix_and_is_idempotent(self):
        prompt = "  original prompt with trailing spaces  "
        result = append_local_detail_guidance(prompt)
        self.assertEqual(result[: len(prompt)], prompt)
        self.assertEqual(result.count(LOCAL_DETAIL_PROMPT_SUFFIX), 1)
        self.assertEqual(append_local_detail_guidance(result), result)
        self.assertNotIn("slime", LOCAL_DETAIL_PROMPT_SUFFIX.lower())

        focused = append_focused_face_guidance(prompt)
        self.assertEqual(focused[: len(prompt)], prompt)
        self.assertEqual(focused.count(FOCUSED_FACE_PROMPT_SUFFIX), 1)
        self.assertEqual(append_focused_face_guidance(focused), focused)

    def test_roi_parser_accepts_semicolon_boxes_and_rejects_outside_values(self):
        self.assertEqual(
            parse_roi_boxes("10,20,110,120; 200,30,250,90", 300, 200),
            [Box(10, 20, 110, 120), Box(200, 30, 250, 90)],
        )
        with self.assertRaisesRegex(ValueError, "fit inside"):
            parse_roi_boxes("10,20,400,120", 300, 200)


class RequestValidationTests(unittest.TestCase):
    def test_accepts_standard_and_explicit_full_grid_2048(self):
        validate_request(**valid_request())
        validate_request(
            **valid_request(
                process_edge=2048,
                allow_expensive_2048_full_grid=True,
            )
        )

    def test_roi_ultra_requires_roi_mode_and_boxes_before_processing(self):
        with self.assertRaisesRegex(ValueError, "requires ROI Boxes"):
            validate_request(
                **valid_request(
                    profile=PROFILE_ROI_ULTRA_2048,
                    mode=MODE_ROI_BOXES,
                    process_edge=2048,
                )
            )
        with self.assertRaisesRegex(ValueError, "requires ROI Boxes mode"):
            validate_request(
                **valid_request(
                    profile=PROFILE_ROI_ULTRA_2048,
                    process_edge=2048,
                    roi_boxes=[Box(0, 0, 64, 64)],
                    allow_expensive_2048_full_grid=True,
                )
            )

    def test_full_grid_2048_requires_explicit_permission(self):
        with self.assertRaisesRegex(ValueError, "Allow expensive 2048 full-grid"):
            validate_request(**valid_request(process_edge=2048))

    def test_focused_rewrite_requires_its_profile_and_tight_rois(self):
        roi = [Box(100, 80, 220, 230)]
        validate_request(
            **valid_request(
                mode=MODE_FOCUSED_ROI_REWRITE,
                profile=PROFILE_FOCUSED_FACE_1536,
                roi_boxes=roi,
                steps=6,
                denoise=0.38,
                candidate_count=2,
            )
        )
        with self.assertRaisesRegex(ValueError, "requires at least one tight"):
            validate_request(
                **valid_request(
                    mode=MODE_FOCUSED_ROI_REWRITE,
                    profile=PROFILE_FOCUSED_FACE_1536,
                    roi_boxes=[],
                )
            )
        with self.assertRaisesRegex(ValueError, "requires the Focused Face"):
            validate_request(
                **valid_request(
                    mode=MODE_FOCUSED_ROI_REWRITE,
                    roi_boxes=roi,
                )
            )

    def test_rejects_invalid_numeric_settings(self):
        cases = (
            ("payload", 0),
            ("core", 513),
            ("overlap", 384),
            ("process_edge", 1024),
            ("steps", 0),
            ("denoise", float("nan")),
            ("candidate_count", 3),
            ("luma_cap", float("inf")),
            ("chroma_cap", -1),
            ("low_frequency_reject_radius", 0),
            ("focused_context_scale", 0.5),
            ("focused_rewrite_feather", float("nan")),
            ("maximum_tile_count", 0),
        )
        for name, value in cases:
            with self.subTest(name=name, value=value):
                with self.assertRaises(ValueError):
                    validate_request(**valid_request(**{name: value}))

    def test_maximum_tile_count_is_checked_before_generation(self):
        plans = plan_local_tiles(1003, 777)
        enforce_maximum_tile_count(plans, len(plans))
        with self.assertRaisesRegex(ValueError, "exceeding Maximum Tile Count"):
            enforce_maximum_tile_count(plans, len(plans) - 1)


class TilePlanningTests(unittest.TestCase):
    def test_focused_roi_plan_keeps_one_face_in_one_square_context(self):
        roi = Box(2608, 635, 2728, 785)
        plan = plan_focused_rois(4096, 1756, [roi], context_scale=2.0)[0]
        self.assertEqual(plan.core_box, roi)
        self.assertEqual(
            [plan.payload_x0, plan.payload_y0, plan.payload_x1, plan.payload_y1],
            [2518, 560, 2818, 860],
        )
        self.assertEqual(plan.payload_x1 - plan.payload_x0, 300)
        self.assertEqual(plan.local_core_box, (90, 75, 210, 225))

        source = np.arange(4096 * 1756 * 3, dtype=np.uint8).reshape(1756, 4096, 3)
        payload = extract_padded_payload(source, plan)
        self.assertEqual(payload.shape, (300, 300, 3))
        x0, y0, x1, y1 = plan.local_core_box
        np.testing.assert_array_equal(
            payload[y0:y1, x0:x1],
            source[roi.top : roi.bottom, roi.left : roi.right],
        )

    def test_focused_roi_plan_pads_edges_and_rejects_overlapping_targets(self):
        edge = plan_focused_rois(
            320,
            240,
            [Box(0, 0, 40, 50)],
            context_scale=2.0,
        )[0]
        self.assertGreater(edge.pad_left, 0)
        self.assertGreater(edge.pad_top, 0)
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            plan_focused_rois(
                320,
                240,
                [Box(10, 10, 80, 80), Box(60, 60, 120, 120)],
            )

    def test_plan_covers_non_divisible_canvas(self):
        width, height = 1003, 777
        plans = plan_local_tiles(width, height)
        coverage = np.zeros((height, width), dtype=bool)
        for tile in plans:
            coverage[tile.core_y0 : tile.core_y1, tile.core_x0 : tile.core_x1] = True
        self.assertTrue(np.all(coverage))

    def test_normalized_smoothstep_weights_sum_to_one(self):
        width, height = 1003, 777
        plans = plan_local_tiles(width, height)
        normalizers = build_axis_normalizers(plans, width, height)
        weight_sum = np.zeros((height, width), dtype=np.float32)
        for tile in plans:
            weight_sum[tile.core_y0 : tile.core_y1, tile.core_x0 : tile.core_x1] += normalized_tile_weight(tile, normalizers)
        np.testing.assert_allclose(weight_sum, 1.0, atol=2e-6)

    def test_edge_padding_keeps_core_mapping_exact(self):
        height, width = 91, 137
        yy, xx = np.indices((height, width))
        source = np.stack((xx % 251, yy % 251, (xx + yy) % 251), axis=2).astype(np.uint8)
        plans = plan_local_tiles(width, height, payload=64, core=48, overlap=8)
        for tile in (plans[0], plans[-1]):
            with self.subTest(tile=tile.index):
                payload = extract_padded_payload(source, tile)
                self.assertEqual(payload.shape, (64, 64, 3))
                lx0, ly0, lx1, ly1 = tile.local_core_box
                np.testing.assert_array_equal(
                    payload[ly0:ly1, lx0:lx1],
                    source[tile.core_y0 : tile.core_y1, tile.core_x0 : tile.core_x1],
                )
        self.assertGreater(plans[0].pad_left, 0)
        self.assertGreater(plans[0].pad_top, 0)
        self.assertGreater(plans[-1].pad_right, 0)
        self.assertGreater(plans[-1].pad_bottom, 0)

    def test_roi_tile_selection_and_union_mask_do_not_double_count(self):
        plans = plan_local_tiles(640, 480, payload=128, core=96, overlap=16)
        rois = [Box(100, 80, 250, 220), Box(180, 150, 300, 260)]
        selected = select_tiles_for_rois(plans, rois)
        self.assertLess(len(selected), len(plans))
        for tile in selected:
            mask = roi_core_mask(tile, rois)
            self.assertTrue(np.all((mask == 0) | (mask == 1)))

        normalizers = build_axis_normalizers(plans, 640, 480)
        roi_weight_sum = np.zeros((480, 640), dtype=np.float32)
        expected_union = np.zeros((480, 640), dtype=bool)
        for roi in rois:
            expected_union[roi.top : roi.bottom, roi.left : roi.right] = True
        for tile in selected:
            canvas_slice = np.s_[
                tile.core_y0 : tile.core_y1,
                tile.core_x0 : tile.core_x1,
            ]
            roi_weight_sum[canvas_slice] += normalized_tile_weight(
                tile,
                normalizers,
            ) * roi_core_mask(tile, rois)
        np.testing.assert_allclose(roi_weight_sum[expected_union], 1.0, atol=2e-6)
        np.testing.assert_array_equal(roi_weight_sum[~expected_union], 0.0)

    def test_focused_rewrite_roi_mask_feathers_inward_and_preserves_exterior(self):
        plans = plan_local_tiles(128, 128, payload=128, core=96, overlap=16)
        tile = plans[0]
        roi = Box(12, 20, 76, 92)
        mask = roi_core_mask(tile, [roi], feather=16)
        self.assertEqual(float(mask[0, 0]), 0.0)
        self.assertEqual(float(mask[50 - tile.core_y0, 44 - tile.core_x0]), 1.0)
        self.assertGreater(float(mask[20 - tile.core_y0, 44 - tile.core_x0]), 0.0)
        self.assertLess(float(mask[20 - tile.core_y0, 44 - tile.core_x0]), 0.01)
        residual_weight, normalization_weight = tile_composition_weights(
            tile,
            build_axis_normalizers(plans, 128, 128),
            [roi],
            feather=16,
        )
        effective_strength = np.divide(
            residual_weight,
            normalization_weight,
            out=np.zeros_like(residual_weight),
            where=normalization_weight > 0,
        )
        np.testing.assert_allclose(effective_strength, mask, atol=1e-7)


class RoundTripAndResidualTests(unittest.TestCase):
    def test_identical_high_resolution_candidate_has_zero_roundtrip_residual(self):
        yy, xx = np.indices((32, 32))
        payload = np.stack(
            ((xx * 7) % 256, (yy * 9) % 256, ((xx + yy) * 5) % 256),
            axis=2,
        ).astype(np.uint8)
        process_input = lanczos_upscale(payload, 1536)
        c0, c1, delta = round_trip_residual(process_input, process_input.copy(), (32, 32))
        np.testing.assert_array_equal(c0, c1)
        np.testing.assert_array_equal(delta, 0.0)

    def test_uniform_low_frequency_shift_is_rejected(self):
        base = np.full((48, 48, 3), 128, dtype=np.uint8)
        c0 = np.full((48, 48, 3), 0.25, dtype=np.float32)
        c1 = c0 + 0.03
        evaluation_result = filter_local_residual(
            base,
            c0,
            c1,
            low_frequency_reject_radius=8,
            luma_cap=8,
            chroma_cap=2,
        )
        np.testing.assert_allclose(evaluation_result.residual, 0.0, atol=1e-7)
        self.assertFalse(evaluation_result.accepted)

    def test_luma_and_chroma_caps_are_independent(self):
        luma_input = np.full((5, 7, 3), 0.25, dtype=np.float32)
        bounded, luma, chroma = cap_luma_chroma(luma_input, luma_cap=8, chroma_cap=2)
        self.assertLessEqual(float(np.max(np.abs(luma))), 8 / 255 + 1e-7)
        np.testing.assert_allclose(chroma, 0.0, atol=1e-7)
        bounded_luma, bounded_chroma = split_luma_chroma(bounded)
        self.assertLessEqual(float(np.max(np.abs(bounded_luma))), 8 / 255 + 1e-7)
        self.assertLessEqual(float(np.max(np.abs(bounded_chroma))), 2 / 255 + 1e-7)

        chroma_input = np.zeros((5, 7, 3), dtype=np.float32)
        chroma_input[..., 0] = 0.2
        chroma_input[..., 1] = -0.2 * 0.2126 / 0.7152
        bounded, luma, chroma = cap_luma_chroma(chroma_input, luma_cap=8, chroma_cap=2)
        np.testing.assert_allclose(luma, 0.0, atol=1e-6)
        self.assertLessEqual(float(np.max(np.abs(chroma))), 2 / 255 + 1e-7)
        self.assertGreater(float(np.max(np.abs(bounded))), 0.0)

    def test_nonfinite_and_invalid_shapes_fail_closed(self):
        good = np.zeros((8, 8, 3), dtype=np.float32)
        bad = good.copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            low_frequency_reject(bad)
        with self.assertRaisesRegex(ValueError, "HxWx3"):
            cap_luma_chroma(np.zeros((8, 8), dtype=np.float32), luma_cap=8, chroma_cap=2)
        with self.assertRaisesRegex(ValueError, "share one shape"):
            round_trip_residual(
                np.zeros((16, 16, 3), dtype=np.uint8),
                np.zeros((8, 8, 3), dtype=np.uint8),
                (4, 4),
            )

    def test_strong_edge_guard_attenuates_without_zeroing(self):
        base = np.zeros((32, 32, 3), dtype=np.float32)
        base[:, 16:] = 1.0
        guard = strong_edge_guard(base)
        self.assertGreaterEqual(float(np.min(guard)), 0.25 - 1e-6)
        self.assertLess(float(np.min(guard)), 1.0)
        self.assertEqual(float(np.max(guard)), 1.0)


class CandidateSelectionTests(unittest.TestCase):
    def test_full_rewrite_uses_downsampled_delta_even_when_gate_rejects(self):
        shape = (8, 8, 3)
        c0 = np.full(shape, 0.25, dtype=np.float32)
        c1 = np.full(shape, 0.40, dtype=np.float32)
        decision = select_candidate(
            [
                evaluation(
                    np.zeros(shape, dtype=np.float32),
                    accepted=False,
                    reason="detail_energy_did_not_increase",
                    c0=c0,
                    c1=c1,
                )
            ],
            apply_full_rewrite=True,
        )
        self.assertEqual(decision.selected_index, 0)
        self.assertIsNone(decision.rejection_reason)
        self.assertEqual(
            decision.quality_gate_override_reason,
            "detail_energy_did_not_increase",
        )
        np.testing.assert_allclose(decision.residual, 0.15, atol=1e-7)

    def test_full_rewrite_prefers_an_accepted_candidate_over_a_lower_score_rejection(self):
        shape = (8, 8, 3)
        rejected = evaluation(
            np.zeros(shape, dtype=np.float32),
            accepted=False,
            score=1.0,
            reason="detail_energy_did_not_increase",
            c0=np.zeros(shape, dtype=np.float32),
            c1=np.full(shape, 0.10, dtype=np.float32),
        )
        accepted = evaluation(
            np.zeros(shape, dtype=np.float32),
            accepted=True,
            score=2.0,
            c0=np.zeros(shape, dtype=np.float32),
            c1=np.full(shape, 0.20, dtype=np.float32),
        )
        decision = select_candidate(
            [rejected, accepted],
            apply_full_rewrite=True,
        )
        self.assertEqual(decision.selected_index, 1)
        self.assertIsNone(decision.quality_gate_override_reason)
        np.testing.assert_allclose(decision.residual, 0.20, atol=1e-7)

    def test_two_candidates_use_one_representative_and_never_average(self):
        representative = np.full((16, 16, 3), 2.0 / 255.0, dtype=np.float32)
        support = np.full((16, 16, 3), 4.0 / 255.0, dtype=np.float32)
        decision = select_candidate(
            [
                evaluation(representative, score=1.0),
                evaluation(support, score=2.0),
            ]
        )
        self.assertEqual(decision.selected_index, 0)
        np.testing.assert_allclose(
            decision.residual,
            representative * decision.agreement[..., None],
            atol=1e-8,
        )
        averaged = (representative + support) * 0.5
        self.assertFalse(np.allclose(decision.residual, averaged))

    def test_agreement_passes_matching_detail_and_rejects_opposite_or_noise(self):
        yy, xx = np.indices((64, 64))
        pattern = (np.sin(xx * 0.7) + np.cos(yy * 0.5)).astype(np.float32) / 255.0
        first = np.repeat(pattern[..., None], 3, axis=2)
        matching = first * 0.8
        opposite = -first
        rng = np.random.default_rng(1234)
        noise = rng.normal(0, 1 / 255.0, first.shape).astype(np.float32)
        same_mask = agreement_mask(first, matching)
        opposite_mask = agreement_mask(first, opposite)
        noise_mask = agreement_mask(first, noise)
        active = np.abs(pattern) > 0.2 / 255.0
        self.assertGreater(float(np.mean(same_mask[active])), 0.6)
        self.assertLess(float(np.mean(opposite_mask[active])), 0.01)
        self.assertLess(float(np.mean(noise_mask[active])), 0.25)

    def test_fixed_seed_is_deterministic_and_candidate_specific(self):
        first = candidate_seed(12345, 320, 640, 0)
        self.assertEqual(first, candidate_seed(12345, 320, 640, 0))
        self.assertNotEqual(first, candidate_seed(12345, 320, 640, 1))
        self.assertNotEqual(first, candidate_seed(12345, 321, 640, 0))
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 2**32)

    def test_all_failed_candidates_make_a_reasoned_noop(self):
        residual = np.ones((8, 8, 3), dtype=np.float32)
        decision = select_candidate(
            [
                evaluation(residual, accepted=False, reason="drift"),
                evaluation(residual, accepted=False, reason="clipping"),
            ]
        )
        self.assertIsNone(decision.selected_index)
        np.testing.assert_array_equal(decision.residual, 0.0)
        self.assertIn("drift", decision.rejection_reason)
        self.assertIn("clipping", decision.rejection_reason)

    def test_one_accepted_candidate_is_representative_and_failed_candidate_is_support_only(self):
        representative = np.full((16, 16, 3), 3.0 / 255.0, dtype=np.float32)
        support = np.full((16, 16, 3), 2.0 / 255.0, dtype=np.float32)
        decision = select_candidate(
            [
                evaluation(representative, score=1.0, accepted=True),
                evaluation(support, score=0.5, accepted=False, reason="drift"),
            ]
        )
        self.assertEqual(decision.selected_index, 0)
        np.testing.assert_allclose(
            decision.residual,
            representative * decision.agreement[..., None],
            atol=1e-8,
        )
        self.assertFalse(np.allclose(decision.residual, (representative + support) * 0.5))


class CanvasCompositionTests(unittest.TestCase):
    def test_focused_difference_metrics_separate_target_and_exact_exterior(self):
        source = np.zeros((12, 16, 3), dtype=np.uint8)
        output = source.copy()
        output[3:9, 4:12] = 10
        metrics = focused_roi_difference_metrics(
            source,
            output,
            [Box(4, 3, 12, 9)],
        )
        self.assertEqual(metrics["target_pixel_count"], 48)
        self.assertEqual(metrics["changed_pixels_inside_target"], 48)
        self.assertEqual(metrics["changed_percent_inside_target"], 100.0)
        self.assertEqual(metrics["changed_pixels_outside_target"], 0)
        self.assertEqual(metrics["mean_abs_rgb_delta_inside_target"], 10.0)
        output[0, 0] = 1
        metrics = focused_roi_difference_metrics(
            source,
            output,
            [Box(4, 3, 12, 9)],
        )
        self.assertEqual(metrics["changed_pixels_outside_target"], 1)

    def test_noop_is_bit_identical_and_size_is_unchanged(self):
        rng = np.random.default_rng(20260713)
        source = rng.integers(0, 256, (79, 113, 3), dtype=np.uint8)
        result, stats = apply_canvas_residual(
            source,
            np.zeros_like(source, dtype=np.float32),
            np.ones(source.shape[:2], dtype=np.float32),
        )
        np.testing.assert_array_equal(result, source)
        self.assertEqual(result.shape, source.shape)
        self.assertEqual(stats["clipping_fraction"], 0.0)

    def test_roi_outside_pixels_are_bit_identical(self):
        source = np.full((40, 60, 3), 128, dtype=np.uint8)
        residual = np.zeros_like(source, dtype=np.float32)
        weights = np.zeros(source.shape[:2], dtype=np.float32)
        roi = np.s_[10:30, 20:45]
        residual[roi] = 2.0 / 255.0
        weights[roi] = 1.0
        result, _ = apply_canvas_residual(source, residual, weights)
        outside = np.ones(source.shape[:2], dtype=bool)
        outside[roi] = False
        np.testing.assert_array_equal(result[outside], source[outside])
        self.assertTrue(np.any(result[roi] != source[roi]))
        self.assertEqual(result.shape, source.shape)

    def test_negative_and_nonfinite_weights_are_rejected(self):
        source = np.zeros((4, 5, 3), dtype=np.uint8)
        residual = np.zeros_like(source, dtype=np.float32)
        for value in (-1.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                weights = np.ones(source.shape[:2], dtype=np.float32)
                weights[0, 0] = value
                with self.assertRaises(ValueError):
                    apply_canvas_residual(source, residual, weights)


class BackendPreflightTests(unittest.TestCase):
    def test_requires_krea2_qwen_vae_and_qwen3vl_names(self):
        report = validate_krea2_module_names(
            "Krea2_Center NF4.safetensors",
            ["qwen_image_vae.safetensors", "qwen3vl_4b.safetensors"],
        )
        self.assertIn("Krea2", report["checkpoint"])
        with self.assertRaisesRegex(ValueError, "Qwen Image VAE"):
            validate_krea2_module_names("Krea2.safetensors", ["qwen3vl_4b.safetensors"])
        with self.assertRaisesRegex(ValueError, "Qwen3-VL"):
            validate_krea2_module_names("Krea2.safetensors", ["qwen_image_vae.safetensors"])


if __name__ == "__main__":
    unittest.main()
