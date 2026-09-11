from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
EXTENSION = ROOT / "extensions-builtin" / "hyperweave"
sys.path.insert(0, str(EXTENSION))

from hyperweave.config import (
    AccumulatorMode,
    ContentProfile,
    HyperWeaveConfig,
    HyperWeavePreset,
    TargetMode,
)
from hyperweave.engine import HyperWeaveEngine, HyperWeaveInterrupted
from hyperweave.generator import StubGenerator
from hyperweave.color import image_to_linear_rgb
from hyperweave.quality import roundtrip_metrics


class ProtectedPassGenerator(StubGenerator):
    """Leave Anchor/Global unchanged and add coherent Material/Micro detail."""

    def __init__(self):
        super().__init__(mode_cycle=("coherent",))

    def generate(self, image, request):
        if request.pass_name in ("material", "micro"):
            return super().generate(image, request)
        self.calls.append(request)
        return image.copy()


def synthetic_rgba() -> Image.Image:
    yy, xx = np.mgrid[:64, :64]
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(xx * 4, 0, 255)
    rgb[..., 1] = np.clip(yy * 4, 0, 255)
    rgb[..., 2] = 96
    cv2.circle(rgb, (24, 24), 9, (220, 180, 150), -1)
    cv2.line(rgb, (10, 45), (54, 45), (20, 20, 20), 2)
    alpha = np.clip((xx + yy) * 2, 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA")


def test_config(tempdir: str, seed: int = 123) -> HyperWeaveConfig:
    return HyperWeaveConfig.from_preset(
        HyperWeavePreset.STRUCTURE_SAFE,
        target_mode=TargetMode.X2,
        content_profile=ContentProfile.ILLUSTRATION,
        seed=seed,
        exact_steps=2,
        tile_input_size=128,
        core_size=96,
        context_size=16,
        stride=72,
        global_candidates=1,
        enable_face_redraw=False,
        enable_hair_redraw=False,
        enable_material_redraw=False,
        enable_micro_pass=False,
        back_projection_iterations=1,
        accumulator_mode=AccumulatorMode.MEMORY,
        temp_directory=tempdir,
        save_debug_images=False,
    )


class EngineIntegrationTests(unittest.TestCase):
    def test_stub_integration_size_alpha_metadata_and_cleanup(self):
        source = synthetic_rgba()
        with tempfile.TemporaryDirectory() as temporary:
            config = test_config(temporary)
            generator = StubGenerator(mode_cycle=("coherent",))
            result = HyperWeaveEngine(config, generator).run(source)
            self.assertEqual((128, 128), result.image.size)
            self.assertEqual("RGBA", result.image.mode)
            expected_alpha = np.asarray(
                source.getchannel("A").resize((128, 128), Image.Resampling.LANCZOS)
            )
            actual_alpha = np.asarray(result.image.getchannel("A"))
            self.assertLess(float(np.mean(np.abs(actual_alpha.astype(float) - expected_alpha))), 1.5)
            self.assertEqual(123, result.metadata["seed"])
            self.assertEqual("hyper_weave", result.metadata["mode_id"])
            self.assertEqual("1.2.0", result.metadata["version"])
            self.assertTrue(result.metadata["local_roundtrip_gate_enabled"])
            self.assertTrue(result.metadata["symmetric_edge_metrics_enabled"])
            self.assertIn("runtime", result.metadata)
            self.assertIn("memory", result.metadata)
            self.assertIn("stage_reports", result.metadata["quality"])
            self.assertIn("hyperweave", result.image.info)
            json.dumps(result.metadata, allow_nan=False)
            json.dumps(result.metrics, allow_nan=False)
            leftovers = list(Path(temporary).glob("hyperweave_*"))
            self.assertEqual([], leftovers)
            self.assertGreaterEqual(len(generator.calls), 2)

    def test_same_seed_is_deterministic_and_different_seed_changes_output(self):
        source = synthetic_rgba()
        with tempfile.TemporaryDirectory() as temporary:
            first = HyperWeaveEngine(
                test_config(temporary, 44),
                StubGenerator(mode_cycle=("coherent",)),
            ).run(source)
            second = HyperWeaveEngine(
                test_config(temporary, 44),
                StubGenerator(mode_cycle=("coherent",)),
            ).run(source)
            different = HyperWeaveEngine(
                test_config(temporary, 45),
                StubGenerator(mode_cycle=("coherent",)),
            ).run(source)
            np.testing.assert_array_equal(np.asarray(first.image), np.asarray(second.image))
            self.assertFalse(
                np.array_equal(np.asarray(first.image), np.asarray(different.image))
            )

    def test_memmap_and_ram_outputs_match(self):
        source = synthetic_rgba()
        with tempfile.TemporaryDirectory() as temporary:
            memory_config = test_config(temporary, 81)
            disk_config = replace(
                memory_config, accumulator_mode=AccumulatorMode.MEMMAP
            )
            memory = HyperWeaveEngine(
                memory_config, StubGenerator(mode_cycle=("coherent",))
            ).run(source)
            disk = HyperWeaveEngine(
                disk_config, StubGenerator(mode_cycle=("coherent",))
            ).run(source)
            np.testing.assert_array_equal(np.asarray(memory.image), np.asarray(disk.image))

    def test_manual_face_roi_runs_face_pass(self):
        source = synthetic_rgba()
        mask = Image.new("L", source.size, 0)
        mask_array = np.asarray(mask).copy()
        cv2.circle(mask_array, (24, 24), 10, 255, -1)
        mask = Image.fromarray(mask_array, mode="L")
        with tempfile.TemporaryDirectory() as temporary:
            config = replace(
                test_config(temporary, 91),
                manual_face_mask=mask,
                enable_face_redraw=True,
                face_candidates=1,
                roi_stages="Final stage only",
            )
            generator = StubGenerator(mode_cycle=("coherent",))
            result = HyperWeaveEngine(config, generator).run(source)
            self.assertEqual(1, result.metadata["detected_faces"])
            self.assertTrue(
                any(call.pass_name == "face" for call in generator.calls)
            )

    def test_overlapping_face_contexts_keep_separate_face_ids(self):
        source = synthetic_rgba()
        mask_array = np.zeros((64, 64), np.uint8)
        cv2.circle(mask_array, (21, 25), 6, 255, -1)
        cv2.circle(mask_array, (43, 25), 6, 255, -1)
        mask = Image.fromarray(mask_array, mode="L")
        with tempfile.TemporaryDirectory() as temporary:
            config = replace(
                test_config(temporary, 93),
                manual_face_mask=mask,
                enable_face_redraw=True,
                face_candidates=2,
                roi_stages="Final stage only",
            )
            generator = StubGenerator(mode_cycle=("coherent",))
            result = HyperWeaveEngine(config, generator).run(source)
            face_reports = [
                item
                for item in result.metrics["stage_reports"][0]["rois"]
                if item["kind"] == "face"
            ]
            self.assertEqual([0, 1], [item["roi_id"] for item in face_reports])
            self.assertEqual([2, 2], [item["candidate_count"] for item in face_reports])
            self.assertTrue(
                all(item["processing_size"] >= 768 for item in face_reports)
            )
            first, second = (item["box"] for item in face_reports)
            overlap_width = min(first[2], second[2]) - max(first[0], second[0])
            overlap_height = min(first[3], second[3]) - max(first[1], second[1])
            self.assertGreater(overlap_width, 0)
            self.assertGreater(overlap_height, 0)
            face_calls = [
                call for call in generator.calls if call.pass_name == "face"
            ]
            self.assertEqual({0, 1}, {call.roi_id for call in face_calls})

    def test_semantic_pass_order_is_material_micro_hair_face(self):
        source = synthetic_rgba()
        mask_array = np.zeros((64, 64), np.uint8)
        cv2.circle(mask_array, (24, 24), 9, 255, -1)
        with tempfile.TemporaryDirectory() as temporary:
            config = replace(
                test_config(temporary, 94),
                manual_face_mask=Image.fromarray(mask_array, mode="L"),
                enable_face_redraw=True,
                enable_hair_redraw=True,
                enable_material_redraw=True,
                enable_micro_pass=True,
                face_candidates=1,
                hair_candidates=1,
                material_candidates=1,
                roi_stages="Final stage only",
            )
            generator = StubGenerator(mode_cycle=("coherent",))
            HyperWeaveEngine(config, generator).run(source)
            semantic = [
                call.pass_name
                for call in generator.calls
                if call.pass_name in ("material", "micro", "hair", "face")
            ]
            collapsed = [
                name
                for index, name in enumerate(semantic)
                if index == 0 or name != semantic[index - 1]
            ]
            self.assertEqual(
                ["material", "micro", "hair", "face"], collapsed
            )
            self.assertEqual("face", semantic[-1])

    def test_material_and_micro_cannot_write_face_core(self):
        source = synthetic_rgba().convert("RGB")
        mask_array = np.zeros((64, 64), np.uint8)
        cv2.circle(mask_array, (24, 24), 9, 255, -1)
        manual = Image.fromarray(mask_array, mode="L")
        with tempfile.TemporaryDirectory() as temporary:
            baseline_config = replace(
                test_config(temporary, 95),
                manual_face_mask=manual,
                enable_face_redraw=False,
                enable_hair_redraw=False,
                enable_material_redraw=False,
                enable_micro_pass=False,
                back_projection_iterations=0,
                tile_input_size=128,
                core_size=128,
                context_size=0,
                stride=128,
            )
            protected_config = replace(
                baseline_config,
                enable_material_redraw=True,
                enable_micro_pass=True,
                material_candidates=1,
            )
            baseline = HyperWeaveEngine(
                baseline_config, ProtectedPassGenerator()
            ).run(source)
            protected = HyperWeaveEngine(
                protected_config, ProtectedPassGenerator()
            ).run(source)
            baseline_array = np.asarray(baseline.image.convert("RGB"))
            protected_array = np.asarray(protected.image.convert("RGB"))
            core = cv2.resize(
                mask_array.astype(np.float32) / 255.0,
                protected.image.size,
                interpolation=cv2.INTER_LINEAR,
            )
            np.testing.assert_array_equal(
                protected_array[core > 0.95],
                baseline_array[core > 0.95],
            )
            self.assertTrue(
                np.any(protected_array[core < 0.05] != baseline_array[core < 0.05])
            )

    def test_final_face_metrics_exist_and_measure_final_output(self):
        source = synthetic_rgba().convert("RGB")
        mask_array = np.zeros((64, 64), np.uint8)
        cv2.circle(mask_array, (24, 24), 9, 255, -1)
        with tempfile.TemporaryDirectory() as temporary:
            config = replace(
                test_config(temporary, 96),
                manual_face_mask=Image.fromarray(mask_array, mode="L"),
                enable_face_redraw=True,
                face_candidates=1,
                roi_stages="Final stage only",
            )
            result = HyperWeaveEngine(
                config, StubGenerator(mode_cycle=("coherent",))
            ).run(source)
            report = result.metrics["stage_reports"][0]
            self.assertEqual(1, len(report["final_face_metrics"]))
            final_metric = report["final_face_metrics"][0]
            self.assertEqual(
                report["final_face_metrics"],
                result.metadata["final_face_metrics"],
            )
            source_linear, _ = image_to_linear_rgb(source)
            output_linear, _ = image_to_linear_rgb(result.image.convert("RGB"))
            sx0, sy0, sx1, sy1 = final_metric["source_bbox"]
            tx0, ty0, tx1, ty1 = final_metric["target_bbox"]
            expected = roundtrip_metrics(
                source_linear[sy0:sy1, sx0:sx1],
                output_linear[ty0:ty1, tx0:tx1],
                evaluation_mask=(
                    mask_array[sy0:sy1, sx0:sx1].astype(np.float32) / 255.0
                ),
            )
            self.assertAlmostEqual(
                expected.ssim, final_metric["roundtrip_ssim"], places=6
            )
            face_candidate = next(
                item
                for item in report["rois"]
                if item["kind"] == "face"
            )
            if face_candidate["selected_score"] is not None:
                selected_ssim = face_candidate["selected_score"]["roundtrip"]["ssim"]
                self.assertNotEqual(
                    round(selected_ssim, 8),
                    round(final_metric["roundtrip_ssim"], 8),
                )

    def test_interruption_discards_partial_memmap_and_temp_directory(self):
        source = synthetic_rgba()
        with tempfile.TemporaryDirectory() as temporary:
            config = replace(
                test_config(temporary, 101),
                accumulator_mode=AccumulatorMode.MEMMAP,
            )
            generator = StubGenerator(mode_cycle=("coherent",))
            engine = HyperWeaveEngine(
                config,
                generator,
                interrupted=lambda: len(generator.calls) >= 1,
            )
            with self.assertRaisesRegex(HyperWeaveInterrupted, "interrupted"):
                engine.run(source)
            self.assertEqual([], list(Path(temporary).glob("hyperweave_*")))

    def test_full_candidate_cycle_selects_coherent_candidate(self):
        source = synthetic_rgba()
        with tempfile.TemporaryDirectory() as temporary:
            config = replace(
                test_config(temporary, 111),
                global_candidates=4,
            )
            result = HyperWeaveEngine(
                config,
                StubGenerator(
                    mode_cycle=("coherent", "shift", "noise", "cross")
                ),
            ).run(source)
            report = result.metrics["stage_reports"][0]
            self.assertEqual(0, report["selected_global_candidate"])
            global_rows = [
                row
                for row in result.metrics["candidate_scores"]
                if row["pass"] == "global"
            ]
            self.assertEqual(4, len(global_rows))
            self.assertTrue(global_rows[0]["accepted"])
            self.assertTrue(
                any(not row["accepted"] for row in global_rows[1:])
            )

    def test_rejected_global_fallback_keeps_boundaries_for_seam_analysis(self):
        source = synthetic_rgba()
        with tempfile.TemporaryDirectory() as temporary:
            result = HyperWeaveEngine(
                test_config(temporary, 116),
                StubGenerator(mode_cycle=("shift",)),
            ).run(source)
            report = result.metrics["stage_reports"][0]
            self.assertIsNone(report["selected_global_candidate"])
            self.assertGreater(report["seam"]["boundary_count"], 0)

    def test_spatial_rescue_salvages_connected_cells_after_global_reject(self):
        source = synthetic_rgba()
        with tempfile.TemporaryDirectory() as temporary:
            config = replace(
                test_config(temporary, 118),
                candidate_rejection_strictness=1.0,
                global_overdraw_strength=0.90,
                spatial_decision_size=32,
                spatial_transition_width=16,
                spatial_score_margin=0.0,
                spatial_fragmentation_limit=0.80,
                spatial_minimum_component_cells=2,
            )
            result = HyperWeaveEngine(
                config,
                StubGenerator(mode_cycle=("localized_failure",)),
            ).run(source)
            report = result.metrics["stage_reports"][0]
            selection = report["selection_reports"][0]
            self.assertEqual("spatial_residual_rescue", selection["mode"])
            self.assertIsNone(report["selected_global_candidate"])
            self.assertGreater(selection["selected_cells"], 0)
            self.assertLess(
                selection["selected_cells"], selection["total_cells"]
            )
            self.assertTrue(selection["final_validation"]["accepted"])

    def test_spatial_rescue_can_be_disabled(self):
        source = synthetic_rgba()
        with tempfile.TemporaryDirectory() as temporary:
            config = replace(
                test_config(temporary, 119),
                candidate_rejection_strictness=1.0,
                global_overdraw_strength=0.90,
                enable_spatial_rescue=False,
            )
            result = HyperWeaveEngine(
                config,
                StubGenerator(mode_cycle=("localized_failure",)),
            ).run(source)
            report = result.metrics["stage_reports"][0]
            selection = report["selection_reports"][0]
            self.assertEqual("anchor", selection["mode"])
            self.assertFalse(selection["spatial_rescue_enabled"])
            self.assertIsNone(selection["spatial_rescue"])

    def test_debug_maps_include_roundtrip_frequency_and_seam_artifacts(self):
        source = synthetic_rgba()
        with tempfile.TemporaryDirectory() as temporary:
            debug_destination = Path(temporary) / "published"
            config = replace(
                test_config(temporary, 121),
                save_debug_images=True,
                save_all_candidates=True,
                save_maps=True,
                save_metrics_json=True,
            )
            result = HyperWeaveEngine(
                config,
                StubGenerator(mode_cycle=("coherent",)),
            ).run(
                source,
                debug_stem="debug",
                debug_destination=debug_destination,
            )
            names = {path.name for path in result.debug_files}
            for expected in (
                "debug_hw_stage01_roundtrip_confidence.png",
                "debug_hw_stage01_frequency_high.png",
                "debug_hw_stage01_frequency_mid.png",
                "debug_hw_stage01_frequency_midlow.png",
                "debug_hw_stage01_seam_map.png",
            ):
                self.assertIn(expected, names)


if __name__ == "__main__":
    unittest.main()
