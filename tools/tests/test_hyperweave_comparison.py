from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
EXTENSION = ROOT / "extensions-builtin" / "hyperweave"
sys.path.insert(0, str(EXTENSION))

from hyperweave.comparison import build_comparison_artifacts, comparison_metrics


class ComparisonTests(unittest.TestCase):
    def test_comparison_metrics_prefers_final_face_metric(self):
        source = Image.fromarray(
            np.tile(np.arange(64, dtype=np.uint8)[None, :, None], (64, 1, 3))
            * 4,
            mode="RGB",
        )
        candidate = source.resize((128, 128), Image.Resampling.LANCZOS)
        candidate.info["hyperweave"] = json.dumps(
            {
                "quality": {
                    "stage_reports": [
                        {
                            "rois": [
                                {
                                    "kind": "face",
                                    "selected_score": {
                                        "roundtrip": {"ssim": 0.99}
                                    },
                                }
                            ],
                            "final_face_metrics": [
                                {
                                    "roundtrip_ssim": 0.61,
                                    "edge_f1": 0.72,
                                }
                            ],
                        }
                    ]
                }
            }
        )
        metrics = comparison_metrics(source, candidate)
        self.assertEqual(0.99, metrics["selected_face_candidate_roundtrip_ssim"])
        self.assertEqual(0.61, metrics["final_face_roundtrip_ssim"])
        self.assertEqual(0.72, metrics["final_face_edge_f1"])
        self.assertEqual(0.61, metrics["face_structure_score"])

    def test_builds_metrics_contact_sheet_crops_and_maps(self):
        source = Image.fromarray(
            np.tile(np.arange(64, dtype=np.uint8)[None, :, None], (64, 1, 3))
            * 4,
            mode="RGB",
        )
        candidate = source.resize((128, 128), Image.Resampling.LANCZOS)
        with tempfile.TemporaryDirectory() as temporary:
            report = build_comparison_artifacts(
                source,
                {"HyperWeave": candidate},
                temporary,
                crop_boxes={"face_center": (32, 32, 96, 96)},
            )
            root = Path(temporary)
            self.assertTrue((root / "metrics.json").is_file())
            self.assertTrue((root / "contact_sheet.png").is_file())
            self.assertTrue((root / "crop_face_center.png").is_file())
            self.assertTrue((root / "HyperWeave_frequency_mid.png").is_file())
            self.assertTrue((root / "HyperWeave_confidence_map.png").is_file())
            self.assertTrue((root / "HyperWeave_seam_map.png").is_file())
            self.assertIn("roundtrip_ssim", report["metrics"]["HyperWeave"])
            self.assertIn("peak_vram_bytes", report["metrics"]["HyperWeave"])
            self.assertIsNotNone(
                report["metrics"]["HyperWeave"]["face_structure_score"]
            )


if __name__ == "__main__":
    unittest.main()
