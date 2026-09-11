from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from pathlib import Path

from modules_forge.krea2_upscale import (
    KREA2_MAX_TILE_PIXELS,
    KREA2_TURBO_NATIVE_LONG_EDGE,
    SAFE_DIFFUSION_LONG_EDGE,
    auto_first_pass_long_edge,
    capped_diffusion_size,
    require_native_diffusion_size,
    require_safe_diffusion_size,
    replace_infotext_size,
    target_size,
    two_stage_sizes,
    validate_tile_geometry,
)
from PIL import Image

from tools.krea2_8k_img2img import (
    build_img2img_payload,
    flatten_source_image,
    resolve_seed,
    resolve_upscale_mode,
    save_png,
    validate_args,
    validate_delivery_size,
    validate_generated_image,
)


def valid_args(**overrides):
    values = {
        "long_edge": 4096,
        "width": None,
        "height": None,
        "first_pass_long_edge": 0,
        "diffusion_long_edge_cap": KREA2_TURBO_NATIVE_LONG_EDGE,
        "model_profile": "custom",
        "allow_non_native_diffusion": False,
        "allow_unsafe_large_diffusion": False,
        "steps": 8,
        "tile_width": 768,
        "tile_height": 768,
        "tile_overlap": 96,
        "tile_batch_size": 1,
        "timeout": 43200,
        "denoise": 0.12,
        "first_pass_denoise": 0.10,
        "cfg": 1.0,
        "distilled_cfg": 1.15,
        "progress_interval": 30.0,
        "no_progress_timeout": 600.0,
        "smart_color_strength": 0.8,
        "smart_analysis_long_edge": 1536,
        "smart_max_speckle_percent": 0.35,
        "always_tiled_vae": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TargetSizeTests(unittest.TestCase):
    def test_preserves_landscape_ratio_at_long_edge(self):
        self.assertEqual(target_size(1000, 500, 8192, None, None), (8192, 4096))

    def test_preserves_portrait_ratio_at_long_edge(self):
        self.assertEqual(target_size(500, 1000, 8192, None, None), (4096, 8192))

    def test_long_edge_delivery_does_not_round_short_edge_to_model_alignment(self):
        self.assertEqual(target_size(1024, 1536, 4096, None, None), (2731, 4096))

    def test_preserves_exact_explicit_delivery_size(self):
        self.assertEqual(target_size(3000, 2000, 8192, 3840, 2160), (3840, 2160))

    def test_rejects_single_explicit_dimension(self):
        with self.assertRaisesRegex(ValueError, "--width and --height"):
            target_size(3000, 2000, 8192, 8192, None)


class ArgumentValidationTests(unittest.TestCase):
    def test_accepts_default_arguments(self):
        validate_args(valid_args())

    def test_rejects_non_positive_steps(self):
        with self.assertRaisesRegex(ValueError, "--steps"):
            validate_args(valid_args(steps=0))

    def test_rejects_negative_tile_overlap(self):
        with self.assertRaisesRegex(ValueError, "--tile-overlap"):
            validate_args(valid_args(tile_overlap=-1))

    def test_rejects_denoise_outside_unit_interval(self):
        with self.assertRaisesRegex(ValueError, "--denoise"):
            validate_args(valid_args(denoise=1.01))

    def test_rejects_first_pass_denoise_outside_unit_interval(self):
        with self.assertRaisesRegex(ValueError, "--first-pass-denoise"):
            validate_args(valid_args(first_pass_denoise=-0.01))

    def test_rejects_single_explicit_dimension(self):
        with self.assertRaisesRegex(ValueError, "--width and --height"):
            validate_args(valid_args(width=8192))

    def test_rejects_negative_first_pass_long_edge(self):
        with self.assertRaisesRegex(ValueError, "--first-pass-long-edge"):
            validate_args(valid_args(first_pass_long_edge=-64))

    def test_rejects_negative_diffusion_long_edge_cap(self):
        with self.assertRaisesRegex(ValueError, "--diffusion-long-edge-cap"):
            validate_args(valid_args(diffusion_long_edge_cap=-1))

    def test_rejects_negative_progress_interval(self):
        with self.assertRaisesRegex(ValueError, "--progress-interval"):
            validate_args(valid_args(progress_interval=-0.1))

    def test_rejects_negative_no_progress_timeout(self):
        with self.assertRaisesRegex(ValueError, "--no-progress-timeout"):
            validate_args(valid_args(no_progress_timeout=-0.1))

    def test_rejects_no_progress_timeout_without_progress_polling(self):
        with self.assertRaisesRegex(ValueError, "--no-progress-timeout requires"):
            validate_args(valid_args(no_progress_timeout=120, progress_interval=0))

    def test_rejects_overlap_at_tile_dimension(self):
        with self.assertRaisesRegex(ValueError, "smaller than"):
            validate_args(valid_args(tile_width=768, tile_height=512, tile_overlap=512))

    def test_rejects_smart_strength_outside_unit_interval(self):
        with self.assertRaisesRegex(ValueError, "--smart-color-strength"):
            validate_args(valid_args(smart_color_strength=1.1))


class RunPolicyTests(unittest.TestCase):
    def test_multidiffusion_payload_uses_internal_exact_steps_only(self):
        args = valid_args()
        args.seed = 7
        args.sampler = "DPM++ 2M SDE"
        args.scheduler = "Simple"
        args.method = "MultiDiffusion"
        payload = build_img2img_payload(
            args,
            Image.new("RGB", (32, 48), "white"),
            "detail",
            "",
            64,
            96,
            0.16,
            True,
            False,
        )

        self.assertEqual(payload["override_settings"], {"img2img_fix_steps": True})
        self.assertTrue(payload["override_settings_restore_afterwards"])

    def test_multidiffusion_payload_can_force_tiled_vae(self):
        args = valid_args(always_tiled_vae=True)
        args.seed = 7
        args.sampler = "DPM++ 2M SDE"
        args.scheduler = "Simple"
        args.method = "MultiDiffusion"

        payload = build_img2img_payload(
            args,
            Image.new("RGB", (32, 48), "white"),
            "detail",
            "",
            64,
            96,
            0.16,
            True,
            False,
        )

        self.assertEqual(
            payload["alwayson_scripts"]["never oom integrated"],
            {"args": [False, True]},
        )

    def test_resolves_explicit_seed(self):
        self.assertEqual(resolve_seed(1234), 1234)

    def test_resolves_random_seed_once_to_uint32(self):
        seed = resolve_seed(-1)
        self.assertGreaterEqual(seed, 0)
        self.assertLess(seed, 2**32)

    def test_rejects_seed_below_random_sentinel(self):
        with self.assertRaisesRegex(ValueError, "--seed"):
            resolve_seed(-2)

    def test_auto_uses_two_stages_for_meaningful_native_refine(self):
        self.assertEqual(
            resolve_upscale_mode("auto", 1024, 1536, 1360, 2048), "two-stage"
        )

    def test_auto_uses_single_stage_when_source_is_already_near_proxy(self):
        self.assertEqual(
            resolve_upscale_mode("auto", 1365, 2048, 1360, 2048), "single-stage"
        )

    def test_rejects_near_black_api_output(self):
        with self.assertRaisesRegex(RuntimeError, "near-empty"):
            validate_generated_image(
                Image.new("RGB", (64, 64), (0, 0, 0)), (64, 64), "TEST"
            )

    def test_png_metadata_round_trip_keeps_parameters_and_quality_report(self):
        report = {"version": 1, "chroma_mura": {"applied": False}}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "result.png"
            save_png(
                path,
                Image.new("RGB", (16, 16), (120, 100, 90)),
                parameters="prompt\nSteps: 8",
                quality_report=report,
            )

            with Image.open(path) as image:
                self.assertEqual(image.info["parameters"], "prompt\nSteps: 8")
                self.assertIn('"version":1', image.info["krea2_smart_finish"])

    def test_delivery_infotext_updates_only_settings_size(self):
        infotext = (
            "poster text says Size: 2048x1360\n"
            "Negative prompt: none\n"
            "Steps: 8, Sampler: DPM++ SDE, Size: 2048x1360, Seed: 7"
        )

        updated = replace_infotext_size(infotext, 2048, 1360, 4096, 2731)

        self.assertIn("poster text says Size: 2048x1360", updated)
        self.assertIn("Size: 4096x2731", updated.splitlines()[-1])

    def test_palette_transparency_is_flattened_on_white(self):
        image = Image.new("P", (2, 1))
        palette = [255, 0, 0, 0, 0, 255] + [0] * (768 - 6)
        image.putpalette(palette)
        image.putdata([0, 1])
        image.info["transparency"] = 0

        flattened = flatten_source_image(image)

        self.assertEqual(flattened.mode, "RGB")
        self.assertEqual(flattened.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(flattened.getpixel((1, 0)), (0, 0, 255))

    def test_rejects_host_memory_risk_without_explicit_opt_in(self):
        with self.assertRaisesRegex(ValueError, "safe limit"):
            validate_delivery_size(8192, 8192)

        validate_delivery_size(8192, 8192, allow_unsafe_large_delivery=True)


class DiffusionCapTests(unittest.TestCase):
    def test_returns_target_when_cap_is_disabled(self):
        self.assertEqual(capped_diffusion_size(1254, 1254, 6144, 6144, 0), (6144, 6144))

    def test_returns_target_when_target_is_below_cap(self):
        self.assertEqual(
            capped_diffusion_size(1254, 1254, 4096, 4096, 6144), (4096, 4096)
        )

    def test_caps_diffusion_size_to_requested_long_edge(self):
        self.assertEqual(
            capped_diffusion_size(1254, 1254, 6144, 6144, 4096), (4096, 4096)
        )

    def test_rejects_cap_below_source_long_edge(self):
        with self.assertRaisesRegex(ValueError, "diffusion long edge cap"):
            capped_diffusion_size(1254, 1254, 6144, 6144, 1024)

    def test_accepts_safe_diffusion_size(self):
        require_safe_diffusion_size(SAFE_DIFFUSION_LONG_EDGE, SAFE_DIFFUSION_LONG_EDGE)

    def test_rejects_unsafe_diffusion_size(self):
        with self.assertRaisesRegex(ValueError, "exceeds safe limit"):
            require_safe_diffusion_size(
                SAFE_DIFFUSION_LONG_EDGE + 64, SAFE_DIFFUSION_LONG_EDGE + 64
            )

    def test_accepts_unsafe_diffusion_size_when_explicitly_allowed(self):
        require_safe_diffusion_size(
            SAFE_DIFFUSION_LONG_EDGE + 64,
            SAFE_DIFFUSION_LONG_EDGE + 64,
            allow_unsafe=True,
        )

    def test_aligns_only_the_diffusion_proxy(self):
        self.assertEqual(
            capped_diffusion_size(1024, 768, 1921, 1081, 4096), (1920, 1088)
        )

    def test_native_proxy_uses_16_pixel_alignment_and_preserves_ratio(self):
        self.assertEqual(
            capped_diffusion_size(1024, 1536, 2731, 4096, 2048), (1360, 2048)
        )

    def test_alignment_never_exceeds_requested_cap(self):
        width, height = capped_diffusion_size(1024, 576, 3840, 2160, 2041)

        self.assertEqual((width, height), (2032, 1136))
        self.assertLessEqual(max(width, height), 2041)

    def test_alignment_does_not_exceed_cap_when_target_equals_cap(self):
        width, height = capped_diffusion_size(1024, 512, 2041, 1000, 2041)

        self.assertEqual(max(width, height), 2032)

    def test_rejects_cap_that_floors_below_source(self):
        with self.assertRaisesRegex(ValueError, "aligned diffusion"):
            capped_diffusion_size(2040, 1024, 4096, 2048, 2041)

    def test_rejects_cap_smaller_than_model_alignment(self):
        with self.assertRaisesRegex(ValueError, "alignment"):
            capped_diffusion_size(8, 8, 64, 64, 15)

    def test_rejects_non_native_turbo_diffusion_without_opt_in(self):
        with self.assertRaisesRegex(ValueError, "resolution guard"):
            require_native_diffusion_size(
                KREA2_TURBO_NATIVE_LONG_EDGE + 16,
                1024,
                "turbo",
            )

    def test_accepts_non_native_diffusion_with_opt_in(self):
        require_native_diffusion_size(3072, 1728, "turbo", allow_non_native=True)


class TileSafetyTests(unittest.TestCase):
    def test_accepts_tile_at_pixel_limit(self):
        validate_tile_geometry(1280, 1280, 128, 1)
        self.assertEqual(1280 * 1280, KREA2_MAX_TILE_PIXELS)

    def test_rejects_tile_above_pixel_limit(self):
        with self.assertRaisesRegex(ValueError, "tile area"):
            validate_tile_geometry(1280, 1344, 128, 1)

    def test_rejects_overlap_at_short_dimension(self):
        with self.assertRaisesRegex(ValueError, "smaller than"):
            validate_tile_geometry(768, 512, 512, 1)

    def test_rejects_tile_batch_above_one(self):
        with self.assertRaisesRegex(ValueError, "batch size 1"):
            validate_tile_geometry(768, 768, 96, 2)

    def test_rejects_tile_below_multidiffusion_minimum(self):
        with self.assertRaisesRegex(ValueError, ">= 256"):
            validate_tile_geometry(16, 256, 0, 1)

    def test_rejects_unaligned_tile_dimensions(self):
        with self.assertRaisesRegex(ValueError, "divisible by 16"):
            validate_tile_geometry(257, 768, 96, 1)

    def test_rejects_unaligned_nonzero_overlap(self):
        with self.assertRaisesRegex(ValueError, "overlap.*divisible by 16"):
            validate_tile_geometry(768, 768, 1, 1)


class TwoStageSizeTests(unittest.TestCase):
    def test_uses_final_aspect_ratio_for_intermediate_size(self):
        self.assertEqual(
            two_stage_sizes(1000, 500, 8192, 8192, 4096), ((4096, 4096), (8192, 8192))
        )

    def test_uses_auto_intermediate_size(self):
        self.assertEqual(auto_first_pass_long_edge(1000, 500, 8192, 4096), 2880)
        self.assertEqual(
            two_stage_sizes(1000, 500, 8192, 4096, 0), ((2880, 1408), (8192, 4096))
        )

    def test_rejects_auto_when_no_intermediate_multiple_exists(self):
        with self.assertRaisesRegex(ValueError, "too close"):
            two_stage_sizes(8150, 4000, 8192, 4032, 0)

    def test_rejects_first_pass_smaller_than_source(self):
        with self.assertRaisesRegex(ValueError, "first pass long edge"):
            two_stage_sizes(5000, 3000, 8192, 4928, 4096)

    def test_rejects_first_pass_at_final_size(self):
        with self.assertRaisesRegex(ValueError, "first pass long edge"):
            two_stage_sizes(1000, 500, 8192, 4096, 8192)

    def test_ceil_aligns_first_pass_instead_of_downscaling_source(self):
        stage1, _ = two_stage_sizes(1490, 800, 2048, 1024, 1500)

        self.assertEqual(stage1, (1536, 768))
        self.assertGreaterEqual(max(stage1), 1490)


if __name__ == "__main__":
    unittest.main()
