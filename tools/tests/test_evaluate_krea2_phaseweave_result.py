from dataclasses import asdict
import unittest

from modules_forge.vram_canvas import balanced_virtual_axis_origin, plan_tiles
from tools.evaluate_krea2_phaseweave_result import regions_for_size, validate_run


def phaseweave_run(size: tuple[int, int]) -> tuple[dict, dict]:
    width, height = size
    tile_size = 1280
    halo = 160
    core_size = tile_size - 2 * halo
    core_overlap = 80
    phase_count = 2
    stride = core_size - core_overlap
    origin = [
        balanced_virtual_axis_origin(
            width, core_size, core_overlap, phase_count=phase_count
        ),
        balanced_virtual_axis_origin(
            height, core_size, core_overlap, phase_count=phase_count
        ),
    ]
    tiles = [
        asdict(tile)
        for tile in plan_tiles(
            width,
            height,
            tile_size=tile_size,
            halo=halo,
            core_overlap=core_overlap,
            phase_count=phase_count,
            virtual_padding=True,
        )
    ]
    stage = {
        "stage": 1,
        "size": [width, height],
        "tile_count": len(tiles),
        "processed_tile_count": len(tiles),
        "skipped_tile_count": 0,
        "grid_origin": origin,
        "tiles": tiles,
    }
    grid = {
        "grid_layout": "uniform_virtual_edge_balanced",
        "grid_stride": stride,
        "grid_phase_offset": round(stride / phase_count),
        "grid_padding_mode": "edge",
        "grid_origin": origin,
    }
    manifest = {
        "target_size": [width, height],
        "merge_mode": "phase_weave",
        "krea2_profile": "phaseweave_4k",
        "exact_img2img_steps": True,
        "exact_img2img_steps_scope": "internal_tiles_only",
        "tile_size": tile_size,
        "core_size": core_size,
        "core_overlap": core_overlap,
        "phase_count": phase_count,
        "stage_reports": [stage],
        **grid,
    }
    metadata = {
        "krea2_phaseweave": {
            "product_name": "Krea2 PhaseWeave 4K",
            "profile_key": "phaseweave_4k",
            "merge_mode": "phase_weave",
            "exact_img2img_steps": True,
            "exact_img2img_steps_scope": "internal_tiles_only",
            **grid,
        }
    }
    return manifest, metadata


class PhaseWeaveResultEvaluationTests(unittest.TestCase):
    def test_validate_run_recomputes_balanced_grid_for_each_delivery_size(self):
        for size, expected_origin in (
            ((2897, 4096), [308, 688]),
            ((5793, 8192), [436, 316]),
        ):
            with self.subTest(size=size):
                manifest, metadata = phaseweave_run(size)
                stage = validate_run(manifest, size, metadata)
                self.assertEqual(stage["grid_origin"], expected_origin)

    def test_validate_run_rejects_origin_from_another_delivery_size(self):
        size = (5793, 8192)
        manifest, metadata = phaseweave_run(size)
        manifest["grid_origin"] = [308, 688]
        with self.assertRaisesRegex(ValueError, "grid_origin"):
            validate_run(manifest, size, metadata)

    def test_review_regions_scale_to_the_delivery_size(self):
        regions = regions_for_size((5793, 8192))
        self.assertEqual(regions["A"][0], (840, 1640, 2740, 3540))
        self.assertEqual(regions["D"][0], (1700, 5380, 4479, 8100))


if __name__ == "__main__":
    unittest.main()
