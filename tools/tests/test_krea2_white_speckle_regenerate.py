import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from tools.krea2_8k_img2img import decode_b64_image, image_to_b64_png
from tools.krea2_white_speckle_regenerate import (
    Box,
    build_inpaint_payload,
    build_protection_mask,
    build_region_plans,
    composite_refined_region,
    detect_white_speckles,
    parse_box_argument,
    process_size_for_roi,
    region_mask,
    run,
    validate_args,
)


def detection_kwargs(**overrides):
    values = {
        "white_luma_min": 235,
        "local_contrast_min": 30,
        "max_chroma": 48,
        "median_size": 7,
        "min_area": 1,
        "max_area": 32,
        "max_span": 10,
        "min_fill_ratio": 0.2,
        "max_background_gradient": 28.0,
        "max_components": 16,
        "max_masked_percent": 1.0,
    }
    values.update(overrides)
    return values


def valid_args(input_path: Path, output_root: Path, **overrides):
    values = {
        "input": str(input_path),
        "output_root": str(output_root),
        "white_luma_min": 235,
        "local_contrast_min": 30,
        "max_chroma": 48,
        "median_size": 7,
        "min_area": 1,
        "max_area": 32,
        "max_span": 10,
        "min_fill_ratio": 0.2,
        "max_background_gradient": 28.0,
        "max_components": 16,
        "max_regions": 16,
        "max_masked_percent": 1.0,
        "limit_regions": 0,
        "protect_mask": None,
        "exclude_box": [],
        "mask_radius": 1,
        "roi_padding": 16,
        "minimum_roi_edge": 32,
        "merge_distance": 0,
        "max_group_edge": 128,
        "process_long_edge": 256,
        "max_process_pixels": 256 * 256,
        "composite_feather": 0.0,
        "api": "http://127.0.0.1:7861",
        "prompt": "repair the masked dot",
        "negative_prompt": "",
        "steps": 4,
        "sampler": "DPM++ 2M SDE",
        "scheduler": "Simple",
        "cfg": 1.0,
        "distilled_cfg": 1.15,
        "denoise": 0.28,
        "seed": 1234,
        "mask_blur": 1,
        "inpainting_fill": 1,
        "timeout": 60,
        "progress_interval": 0.0,
        "no_progress_timeout": 0.0,
        "dry_run": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def flat_image_with_dot(
    *,
    size: tuple[int, int] = (64, 64),
    background: tuple[int, int, int] = (80, 80, 80),
    dot: tuple[int, int, int] = (255, 255, 255),
    box: tuple[int, int, int, int] = (30, 30, 33, 33),
) -> Image.Image:
    image = Image.new("RGB", size, background)
    array = np.array(image)
    left, top, right, bottom = box
    array[top:bottom, left:right] = dot
    return Image.fromarray(array, mode="RGB")


class DetectionTests(unittest.TestCase):
    def test_detects_small_neutral_white_component(self):
        image = flat_image_with_dot()

        mask, components, report = detect_white_speckles(image, **detection_kwargs())

        self.assertEqual(len(components), 1)
        self.assertEqual(components[0].box, Box(30, 30, 33, 33))
        self.assertEqual(components[0].area, 9)
        self.assertEqual(np.count_nonzero(np.asarray(mask)), 9)
        self.assertEqual(report["kept_components"], 1)

    def test_rejects_saturated_bright_component(self):
        image = flat_image_with_dot(dot=(0, 255, 255))

        mask, components, report = detect_white_speckles(image, **detection_kwargs())

        self.assertEqual(components, [])
        self.assertEqual(np.count_nonzero(np.asarray(mask)), 0)
        self.assertEqual(report["kept_components"], 0)

    def test_rejects_component_above_area_limit(self):
        image = flat_image_with_dot(box=(20, 20, 30, 30))

        _, components, report = detect_white_speckles(
            image,
            **detection_kwargs(
                median_size=15,
                max_area=32,
                max_span=20,
            ),
        )

        self.assertEqual(components, [])
        self.assertEqual(report["rejected_by_area"], 1)

    def test_component_limit_guard_fails_before_generation(self):
        image = flat_image_with_dot(box=(10, 10, 12, 12))
        array = np.array(image)
        array[45:47, 45:47] = (255, 255, 255)
        image = Image.fromarray(array, mode="RGB")

        with self.assertRaisesRegex(ValueError, "exceeding --max-components"):
            detect_white_speckles(image, **detection_kwargs(max_components=1))

    def test_protection_mask_excludes_legitimate_white_highlight(self):
        image = flat_image_with_dot()
        protected = np.zeros((image.height, image.width), dtype=np.uint8)
        protected[30:33, 30:33] = 255
        protection = Image.fromarray(protected, mode="L")

        mask, components, report = detect_white_speckles(
            image,
            **detection_kwargs(protection_mask=protection),
        )

        self.assertEqual(components, [])
        self.assertEqual(np.count_nonzero(np.asarray(mask)), 0)
        self.assertEqual(report["protected_candidate_pixels"], 9)


class RegionPlanningTests(unittest.TestCase):
    def setUp(self):
        image = flat_image_with_dot()
        _, self.components, _ = detect_white_speckles(image, **detection_kwargs())

    def test_micro_mask_is_tiny_and_roi_contains_context(self):
        plans = build_region_plans(
            self.components,
            image_width=64,
            image_height=64,
            mask_radius=1,
            roi_padding=8,
            minimum_roi_edge=32,
        )

        self.assertEqual(len(plans), 1)
        plan = plans[0]
        self.assertEqual(plan.roi_box.width, 32)
        self.assertEqual(plan.roi_box.height, 32)
        self.assertLess(plan.masked_pixels, 32 * 32 * 0.1)
        self.assertLessEqual(plan.roi_box.left, plan.mask_box.left)
        self.assertGreaterEqual(plan.roi_box.right, plan.mask_box.right)

    def test_region_mask_uses_roi_coordinates(self):
        plan = build_region_plans(
            self.components,
            image_width=64,
            image_height=64,
            mask_radius=1,
            roi_padding=8,
            minimum_roi_edge=32,
        )[0]

        mask = region_mask(plan)

        self.assertEqual(mask.size, (32, 32))
        self.assertEqual(np.count_nonzero(np.asarray(mask)), plan.masked_pixels)

    def test_nearby_components_share_one_context_roi(self):
        image = flat_image_with_dot(box=(20, 20, 22, 22))
        array = np.array(image)
        array[20:22, 32:34] = (255, 255, 255)
        _, components, _ = detect_white_speckles(Image.fromarray(array, mode="RGB"), **detection_kwargs())

        plans = build_region_plans(
            components,
            image_width=64,
            image_height=64,
            mask_radius=1,
            roi_padding=8,
            minimum_roi_edge=32,
            merge_distance=16,
            max_group_edge=64,
        )

        self.assertEqual(len(plans), 1)
        self.assertEqual(len(plans[0].components), 2)

    def test_process_size_is_aligned_and_bounded(self):
        self.assertEqual(process_size_for_roi(96, 96, 512, 512 * 512), (512, 512))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            process_size_for_roi(96, 96, 512, 200_000)


class PayloadAndCompositeTests(unittest.TestCase):
    def test_payload_processes_only_pre_cropped_mask(self):
        args = argparse.Namespace(
            sampler="DPM++ 2M SDE",
            scheduler="Simple",
            steps=4,
            cfg=1.0,
            distilled_cfg=1.15,
            denoise=0.28,
            inpainting_fill=1,
        )
        image = Image.new("RGB", (256, 256), (80, 80, 80))
        mask = Image.new("L", (256, 256), 0)
        mask.putpixel((128, 128), 255)

        payload = build_inpaint_payload(args, image, mask, "repair", "white dot", 123, 4)

        self.assertFalse(payload["inpaint_full_res"])
        self.assertEqual(payload["width"], 256)
        self.assertEqual(payload["height"], 256)
        self.assertEqual(payload["seed"], 123)
        self.assertEqual(payload["mask_blur"], 4)
        self.assertEqual(
            np.count_nonzero(np.asarray(decode_b64_image(payload["mask"]).convert("L"))),
            1,
        )

    def test_composite_cannot_change_pixels_outside_micro_mask(self):
        source = Image.new("RGB", (32, 32), (0, 0, 0))
        refined = Image.new("RGB", (32, 32), (255, 255, 255))
        mask = Image.new("L", (32, 32), 0)
        mask.putpixel((16, 16), 255)

        result = composite_refined_region(source, refined, mask, 0.0)

        self.assertEqual(result.getpixel((16, 16)), (255, 255, 255))
        self.assertEqual(result.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(result.getpixel((16, 15)), (0, 0, 0))


class ValidationTests(unittest.TestCase):
    def test_build_protection_mask_combines_source_and_boxes(self):
        source = Image.new("L", (16, 16), 0)
        source.putpixel((2, 2), 255)

        protection = build_protection_mask(
            (16, 16),
            source,
            [Box(8, 8, 12, 12)],
        )

        self.assertIsNotNone(protection)
        self.assertEqual(protection.getpixel((2, 2)), 255)
        self.assertEqual(protection.getpixel((8, 8)), 255)
        self.assertEqual(protection.getpixel((11, 11)), 255)
        self.assertEqual(protection.getpixel((12, 12)), 0)

    def test_build_protection_mask_rejects_wrong_size(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            build_protection_mask(
                (16, 16),
                Image.new("L", (8, 8), 255),
                [],
            )

    def test_parse_box_argument(self):
        self.assertEqual(parse_box_argument("1, 2, 10, 20"), Box(1, 2, 10, 20))
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "LEFT < RIGHT"):
            parse_box_argument("10,2,1,20")

    def test_rejects_even_median_kernel(self):
        args = valid_args(Path("input.png"), Path("output"), median_size=6)
        with self.assertRaisesRegex(ValueError, "must be odd"):
            validate_args(args)

    def test_rejects_watchdog_without_progress_polling(self):
        args = valid_args(
            Path("input.png"),
            Path("output"),
            progress_interval=0.0,
            no_progress_timeout=10.0,
        )
        with self.assertRaisesRegex(ValueError, "requires"):
            validate_args(args)


class RunIntegrationTests(unittest.TestCase):
    def test_dry_run_writes_masks_regions_and_manifest_without_api(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.png"
            flat_image_with_dot().save(input_path)
            args = valid_args(input_path, root / "output", dry_run=True)

            with patch("tools.krea2_white_speckle_regenerate.post_img2img") as post:
                artifacts = run(args)

            post.assert_not_called()
            self.assertEqual(artifacts.region_count, 1)
            self.assertIsNone(artifacts.output_path)
            self.assertTrue((artifacts.output_dir / "detected_white_speckles.png").exists())
            self.assertTrue((artifacts.output_dir / "micro_mask.png").exists())
            self.assertTrue((artifacts.output_dir / "region_001_process_mask.png").exists())
            manifest = json.loads(artifacts.manifest_path.read_text("utf-8"))
            self.assertTrue(manifest["dry_run"])
            self.assertEqual(manifest["selected_region_count"], 1)
            self.assertEqual(manifest["regions"][0]["status"], "dry-run")

    def test_live_path_calls_one_api_request_and_replaces_only_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.png"
            flat_image_with_dot().save(input_path)
            args = valid_args(input_path, root / "output", dry_run=False)

            def fake_post(_args, payload, _stage_name):
                process_image = decode_b64_image(payload["init_images"][0])
                process_mask = decode_b64_image(payload["mask"]).convert("L")
                array = np.array(process_image)
                selected = np.asarray(process_mask) > 0
                array[selected] = (80, 80, 80)
                repaired = Image.fromarray(array, mode="RGB")
                return {"images": [image_to_b64_png(repaired)]}

            with patch(
                "tools.krea2_white_speckle_regenerate.post_img2img",
                side_effect=fake_post,
            ) as post:
                artifacts = run(args)

            self.assertEqual(post.call_count, 1)
            self.assertIsNotNone(artifacts.output_path)
            with Image.open(artifacts.output_path) as output:
                self.assertEqual(output.getpixel((31, 31)), (80, 80, 80))
                self.assertEqual(output.getpixel((0, 0)), (80, 80, 80))
                self.assertIn("krea2_white_speckle_regenerate", output.info)


if __name__ == "__main__":
    unittest.main()
