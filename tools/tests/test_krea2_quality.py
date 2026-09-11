import unittest
import time

import cv2
import numpy as np
from PIL import Image

from modules.krea2_quality import (
    ChromaMuraParams,
    adaptive_chroma_correct,
    adaptive_detail_guard,
    adaptive_despeckle,
    analyze_chroma_mura,
    build_adaptive_speckle_mask,
    rgb_to_lab_float,
    smart_finish_image,
)


def lab_blotch(width: int, height: int, amplitude: float = 16.0) -> Image.Image:
    rng = np.random.default_rng(20260710)
    field = rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32)
    sigma = max(width, height) * 0.012
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=sigma, sigmaY=sigma)
    field /= max(float(np.std(field)), 1e-6)
    field = np.clip(field, -2.0, 2.0)
    lab = np.zeros((height, width, 3), dtype=np.uint8)
    lab[..., 0] = 150
    lab[..., 1] = np.clip(128 + field * amplitude, 0, 255).astype(np.uint8)
    lab[..., 2] = 128
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb, mode="RGB")


class ChromaMuraAnalysisTests(unittest.TestCase):
    def test_luminance_noise_is_not_color_mura(self):
        rng = np.random.default_rng(1234)
        luma = np.clip(128 + rng.normal(0.0, 8.0, size=(256, 256)), 0, 255).astype(
            np.uint8
        )
        rgb = np.repeat(luma[..., None], 3, axis=2)

        _, _, _, metrics, _ = analyze_chroma_mura(Image.fromarray(rgb, "RGB"))

        self.assertLess(metrics.p95_chroma_delta, 0.5)
        self.assertTrue(metrics.rough_judgement.startswith("OK"))

    def test_detects_low_frequency_chroma_blotch(self):
        _, _, _, metrics, _ = analyze_chroma_mura(lab_blotch(512, 512))

        self.assertGreater(metrics.p95_chroma_delta, 2.5)
        self.assertGreater(metrics.area_chroma_delta_gt_2_pct, 1.0)

    def test_normalized_score_is_similar_across_resolutions(self):
        params = ChromaMuraParams(analysis_long_edge=768)
        _, _, _, small, _ = analyze_chroma_mura(lab_blotch(384, 256), params)
        _, _, _, large, _ = analyze_chroma_mura(lab_blotch(1536, 1024), params)

        self.assertAlmostEqual(
            small.p95_chroma_delta, large.p95_chroma_delta, delta=0.8
        )

    def test_analysis_long_edge_is_capped(self):
        _, _, _, _, analysis_size = analyze_chroma_mura(
            lab_blotch(2048, 1024), ChromaMuraParams(analysis_long_edge=512)
        )

        self.assertEqual(max(analysis_size), 512)


class AdaptiveChromaCorrectionTests(unittest.TestCase):
    def test_clean_image_is_bit_identical_noop(self):
        image = Image.new("RGB", (256, 192), (120, 100, 90))

        result, report = adaptive_chroma_correct(image)

        self.assertFalse(report["applied"])
        np.testing.assert_array_equal(np.asarray(result), np.asarray(image))

    def test_correction_reduces_chroma_mura_and_preserves_luminance(self):
        image = lab_blotch(512, 512, amplitude=24.0)
        before_lab = rgb_to_lab_float(np.asarray(image, dtype=np.uint8))

        result, report = adaptive_chroma_correct(image, strength=1.0)
        after_lab = rgb_to_lab_float(np.asarray(result.convert("RGB"), dtype=np.uint8))

        self.assertTrue(report["applied"])
        self.assertLess(
            report["after"]["p95_chroma_delta"],
            report["before"]["p95_chroma_delta"],
        )
        self.assertLess(
            float(np.mean(np.abs(after_lab[..., 0] - before_lab[..., 0]))), 0.6
        )

    def test_alpha_is_preserved(self):
        image = lab_blotch(256, 256, amplitude=24.0).convert("RGBA")
        alpha = np.linspace(0, 255, 256, dtype=np.uint8)[None, :]
        alpha = np.repeat(alpha, 256, axis=0)
        image.putalpha(Image.fromarray(alpha, "L"))

        result, _ = adaptive_chroma_correct(image, strength=1.0)

        np.testing.assert_array_equal(np.asarray(result)[..., 3], alpha)

    def test_pixels_outside_correction_mask_are_bit_exact(self):
        image_array = np.asarray(lab_blotch(512, 512, amplitude=24.0)).copy()
        checker = (np.indices((512, 220)).sum(axis=0) % 2).astype(bool)
        image_array[:, :220] = np.where(
            checker[..., None],
            np.array([15, 220, 45], dtype=np.uint8),
            np.array([235, 30, 210], dtype=np.uint8),
        )
        image = Image.fromarray(image_array, "RGB")

        result, report = adaptive_chroma_correct(image, strength=1.0)

        self.assertTrue(report["applied"])
        np.testing.assert_array_equal(np.asarray(result)[:, :210], image_array[:, :210])
        self.assertGreater(
            int(np.count_nonzero(np.asarray(result)[:, 260:] != image_array[:, 260:])),
            0,
        )

    def test_transparent_hidden_rgb_does_not_change_visible_metrics_or_pixels(self):
        visible = np.asarray(lab_blotch(320, 256, amplitude=24.0)).copy()
        alpha = np.zeros((256, 320), dtype=np.uint8)
        alpha[:, 120:] = 255
        first = np.dstack((visible.copy(), alpha))
        second = first.copy()
        first[:, :120, :3] = [255, 0, 255]
        second[:, :120, :3] = [0, 255, 0]
        image_a = Image.fromarray(first, "RGBA")
        image_b = Image.fromarray(second, "RGBA")

        *_, metrics_a, _ = analyze_chroma_mura(image_a)
        *_, metrics_b, _ = analyze_chroma_mura(image_b)
        self.assertAlmostEqual(
            metrics_a.p95_chroma_delta, metrics_b.p95_chroma_delta, delta=0.02
        )

        result, report = adaptive_chroma_correct(image_a, strength=1.0)
        self.assertTrue(report["applied"])
        result_array = np.asarray(result)
        np.testing.assert_array_equal(result_array[:, :120], first[:, :120])
        # Alpha-boundary colors are protected against compositing halos.
        np.testing.assert_array_equal(result_array[:, 120:122], first[:, 120:122])

    def test_hard_color_boundary_does_not_create_chroma_halo(self):
        source = np.empty((512, 256, 3), dtype=np.uint8)
        source[:, :128] = [180, 80, 80]
        source[:, 128:] = [80, 180, 80]
        image = Image.fromarray(source, "RGB")

        result, report = adaptive_chroma_correct(image, strength=1.0)
        difference = np.max(
            np.abs(np.asarray(result).astype(np.int16) - source.astype(np.int16)),
            axis=2,
        )

        self.assertFalse(report["applied"])
        self.assertEqual(int(np.max(difference)), 0)

    def test_4k_clean_image_smoke_is_noop(self):
        image = Image.new("RGB", (3840, 2160), (110, 120, 130))

        result, report = adaptive_chroma_correct(
            image,
            params=ChromaMuraParams(analysis_long_edge=512),
        )

        self.assertEqual(result.size, (3840, 2160))
        self.assertFalse(report["applied"])
        self.assertEqual(report["analysis_size"], [512, 288])


class AdaptiveSpeckleTests(unittest.TestCase):
    def test_isolated_speckles_are_repaired_but_line_is_preserved(self):
        rgb = np.full((192, 256, 3), 120, dtype=np.uint8)
        rgb[40, 40] = 255
        rgb[120, 180] = 0
        rgb[80:83, 30:220] = 235
        image = Image.fromarray(rgb, "RGB")

        mask, report = build_adaptive_speckle_mask(image)
        mask_array = np.asarray(mask)

        self.assertTrue(report["accepted"])
        self.assertGreater(mask_array[40, 40], 0)
        self.assertGreater(mask_array[120, 180], 0)
        self.assertEqual(int(np.count_nonzero(mask_array[80:83, 30:220])), 0)

        result, result_report = adaptive_despeckle(image)
        result_array = np.asarray(result)
        self.assertTrue(result_report["applied"])
        self.assertLess(abs(int(result_array[40, 40, 0]) - 120), 20)
        np.testing.assert_array_equal(result_array[81, 50], rgb[81, 50])

    def test_large_mask_is_rejected_by_fail_safe(self):
        rgb = np.full((128, 128, 3), 120, dtype=np.uint8)
        rgb[::4, ::4] = 255

        mask, report = build_adaptive_speckle_mask(
            Image.fromarray(rgb, "RGB"), max_masked_percent=0.01
        )

        self.assertFalse(report["accepted"])
        self.assertEqual(int(np.count_nonzero(np.asarray(mask))), 0)

    def test_transparent_hidden_speckles_do_not_hide_visible_repair(self):
        rgba = np.full((256, 320, 4), [120, 120, 120, 255], dtype=np.uint8)
        rgba[:, :128, 3] = 0
        rgba[::4, :128:4, :3] = 255
        rgba[100, 220, :3] = 255

        mask, report = build_adaptive_speckle_mask(Image.fromarray(rgba, "RGBA"))
        mask_array = np.asarray(mask)

        self.assertTrue(report["accepted"])
        self.assertEqual(int(np.count_nonzero(mask_array[:, :128])), 0)
        self.assertGreater(mask_array[100, 220], 0)

    def test_many_components_are_filtered_in_linear_time(self):
        rgb = np.full((768, 768, 3), 120, dtype=np.uint8)
        rgb[::8, ::8] = 255
        started = time.perf_counter()

        _, report = build_adaptive_speckle_mask(
            Image.fromarray(rgb, "RGB"), max_masked_percent=0.01
        )

        self.assertGreater(report["candidate_components"], 8000)
        self.assertLess(time.perf_counter() - started, 1.5)


class AdaptiveDetailGuardTests(unittest.TestCase):
    @staticmethod
    def coherent_lines(width: int = 256, height: int = 192) -> Image.Image:
        rgb = np.full((height, width, 3), 120, dtype=np.uint8)
        for x in range(12, width - 12, 16):
            rgb[16 : height - 16, x : x + 2] = 134
        return Image.fromarray(rgb, "RGB")

    def test_flat_image_is_bit_identical_noop(self):
        image = Image.new("RGB", (256, 192), (120, 120, 120))

        result, report = adaptive_detail_guard(image)

        self.assertFalse(report["applied"])
        self.assertTrue(report["accepted"])
        np.testing.assert_array_equal(np.asarray(result), np.asarray(image))

    def test_coherent_source_lines_gain_detail_without_flat_region_edits(self):
        image = self.coherent_lines()

        result, report = adaptive_detail_guard(image, strength=0.7)
        delta = np.max(
            np.abs(np.asarray(result).astype(np.int16) - np.asarray(image).astype(np.int16)),
            axis=2,
        )

        self.assertTrue(report["applied"])
        self.assertTrue(report["accepted"])
        self.assertGreater(report["detail_energy_ratio"], 1.002)
        self.assertEqual(report["flat_region_changed_pixels"], 0)
        self.assertLessEqual(int(np.max(delta)), 4)
        np.testing.assert_array_equal(np.asarray(result)[:, :8], np.asarray(image)[:, :8])

    def test_low_amplitude_isotropic_noise_is_not_amplified(self):
        rng = np.random.default_rng(20260713)
        noise = rng.integers(-1, 2, size=(192, 256, 1), dtype=np.int16)
        rgb = np.clip(120 + noise, 0, 255).astype(np.uint8)
        rgb = np.repeat(rgb, 3, axis=2)
        image = Image.fromarray(rgb, "RGB")

        result, report = adaptive_detail_guard(image, detail_threshold=1.25)

        self.assertFalse(report["applied"])
        np.testing.assert_array_equal(np.asarray(result), rgb)

    def test_hard_edge_does_not_create_a_halo(self):
        rgb = np.empty((192, 256, 3), dtype=np.uint8)
        rgb[:, :128] = 60
        rgb[:, 128:] = 200
        image = Image.fromarray(rgb, "RGB")

        result, _ = adaptive_detail_guard(image, strength=1.0)
        result_array = np.asarray(result)

        np.testing.assert_array_equal(result_array[:, :120], rgb[:, :120])
        np.testing.assert_array_equal(result_array[:, 136:], rgb[:, 136:])

    def test_alpha_and_stripe_boundaries_are_preserved(self):
        image = self.coherent_lines().convert("RGBA")
        alpha = np.linspace(0, 255, image.width, dtype=np.uint8)[None, :]
        alpha = np.repeat(alpha, image.height, axis=0)
        image.putalpha(Image.fromarray(alpha, "L"))

        narrow, narrow_report = adaptive_detail_guard(image, stripe_height=48)
        wide, wide_report = adaptive_detail_guard(image, stripe_height=512)

        self.assertTrue(narrow_report["applied"])
        self.assertTrue(wide_report["applied"])
        np.testing.assert_array_equal(np.asarray(narrow)[..., 3], alpha)
        np.testing.assert_array_equal(np.asarray(narrow), np.asarray(wide))

    def test_smart_finish_records_detail_guard_report(self):
        result, report = smart_finish_image(
            self.coherent_lines(),
            color_strength=0.0,
            detail_guard=True,
            detail_strength=0.7,
        )

        self.assertEqual(result.size, (256, 192))
        self.assertEqual(report["version"], 2)
        self.assertTrue(report["detail_guard"]["applied"])


if __name__ == "__main__":
    unittest.main()
