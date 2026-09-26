"""Custom sizes must reach generation, summaries and H3 replay without rounding."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from modules_forge import minimax_h3_bridge as h3
from modules_forge.studio_dimensions import custom_dimensions


class CanvasSizeTests(unittest.TestCase):
    def test_exact_and_integral_float_sizes(self):
        self.assertEqual(custom_dimensions(736.0, "1120"), (736, 1120))

    def test_invalid_values_are_never_rounded(self):
        for value in (None, "", True, float("nan"), float("inf"), 768.5, 767, 224, 4128):
            with self.subTest(value=value), self.assertRaises(ValueError):
                custom_dimensions(value, 768)

    def test_pixel_budget_and_studio_minimum_are_enforced(self):
        with self.assertRaisesRegex(ValueError, "MP"):
            custom_dimensions(2048, 2048, max_pixels=2_097_152)
        with self.assertRaises(ValueError):
            custom_dimensions(480, 768, minimum=512)

    def test_h3_custom_request_reaches_workflow_and_summary(self):
        request = h3.H3Request(mode=h3.MODE_TEXT, prompt="Quiet coast", quality="custom:736x1120")
        h3.validate_request(request)
        workflow = h3.build_workflow(request, {}, seed=42)
        self.assertEqual((workflow["5"]["inputs"]["width"], workflow["5"]["inputs"]["height"]), (736, 1120))
        self.assertIn("736 × 1120", h3.settings_summary_html("16:9", request.quality, 5, 20))
        self.assertNotIn("公式Fast Preview相当の標準設定です", h3.settings_summary_html("16:9", request.quality, 5, 20))

    def test_h3_invalid_custom_request_fails_before_backend(self):
        for quality in ("custom:", "custom:768", "custom:768x513", "custom:nanx512", "custom:4096x4096"):
            with self.subTest(quality=quality), self.assertRaises(h3.H3BridgeError):
                h3.validate_request(h3.H3Request(mode=h3.MODE_TEXT, prompt="Coast", quality=quality))

    def test_h3_custom_size_survives_history_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            video = output / "custom.mp4"
            video.write_bytes(b"history fixture")
            video.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "model": "MiniMax H3",
                        "mode": h3.MODE_TEXT,
                        "prompt": "Coast",
                        "aspect": "16:9",
                        "quality": "custom:736x1120",
                        "requested_seconds": 5,
                        "steps": 20,
                        "seed": 42,
                        "scheduler": "simple",
                        "ref_image_size": "match",
                    }
                ),
                encoding="utf-8",
            )
            item = h3.HistoryItem(video.resolve(), video.stat().st_mtime, "Forge Neo")
            restored = h3.load_history_request(str(video), [item], output)
            self.assertEqual(restored.dimensions, (736, 1120))


class QwenCanvasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tools.tests.test_qwen_image21_service import load_ui

        cls.ui = load_ui()

    def test_custom_canvas_passes_both_dimensions_to_submission(self):
        request = SimpleNamespace(session_hash="dimensions", username=None)
        with patch.object(self.ui.STUDIO, "start", return_value="job") as start:
            result = self.ui.start_canvas(
                "Coast", [], "custom", False, "int8", "offload", 42, 40, request, custom_width=736, custom_height=1120
            )
        self.assertEqual(result[0], "job")
        submitted = start.call_args.args[0].resolved()
        self.assertEqual((submitted.width, submitted.height), (736, 1120))

    def test_invalid_custom_size_does_not_start_job(self):
        request = SimpleNamespace(session_hash="dimensions", username=None)
        with patch.object(self.ui.STUDIO, "start") as start:
            result = self.ui.start(
                "Coast", [], "custom", False, "int8", "offload", 42, 40, request, custom_width=767, custom_height=1120
            )
        start.assert_not_called()
        self.assertIn("32", result[1])

    def test_reference_size_disables_numbers_without_destroying_draft(self):
        updates = self.ui.resolution_settings("reference")
        self.assertTrue(all(not update["interactive"] for update in updates))
        self.assertTrue(all("value" not in update for update in updates))
        self.assertEqual([u["value"] for u in self.ui.resolution_settings("1280x1024")[:2]], [1280, 1024])

    def test_summary_uses_custom_size_and_reports_invalid_input(self):
        self.assertIn("736×1120", self.ui.generation_summary("w4a8", 40, "custom", 736, 1120))
        self.assertIn("32", self.ui.generation_summary("w4a8", 40, "custom", 737, 1120))


class SenseNovaCanvasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tools.tests.test_sensenova_u15_studio import load_studio_module

        cls.ui = load_studio_module()

    def request(self, resolution="custom", **changes):
        values = dict(
            mode=self.ui.MODE_TEXT,
            prompt="Coast",
            gallery=[],
            model_path="model",
            quantization=self.ui.QUANT_INT8_CONVROT,
            checkpoint_path="checkpoint",
            source_path="source",
            resolution=resolution,
            input_max_pixels="auto",
            generation_profile=self.ui.PROFILE_QUALITY,
            steps=50,
            cfg_scale=4,
            img_cfg_scale=1,
            timestep_shift=3,
            seed=42,
            vram_mode="low",
            attn_backend="auto",
            dtype="bfloat16",
            custom_width=736,
            custom_height=1120,
            should_validate=False,
        )
        values.update(changes)
        return self.ui._request_from_ui(**values)

    def test_custom_request_uses_dimensions_and_presets_override_them(self):
        custom = self.request()
        self.assertEqual((custom.width, custom.height), (736, 1120))
        preset = self.request("2048x2048")
        self.assertEqual((preset.width, preset.height), (2048, 2048))

    def test_auto_edit_preserves_auto_size_contract(self):
        automatic = self.request("auto_1mp", mode=self.ui.MODE_EDIT)
        self.assertEqual((automatic.width, automatic.height, automatic.target_pixels), (None, None, 1024 * 1024))
        updates = self.ui._resolution_updates("auto_1mp", self.ui.MODE_EDIT)
        self.assertTrue(all(not update["interactive"] and "value" not in update for update in updates))

    def test_custom_mode_switch_preserves_size_choice(self):
        for mode in (self.ui.MODE_EDIT, self.ui.MODE_TEXT):
            updates = self.ui._mode_updates(mode, "custom", True)
            self.assertEqual(updates[1]["value"], "custom")

    def test_invalid_custom_request_fails_even_for_summary(self):
        with self.assertRaises(ValueError):
            self.request(custom_height=511)


if __name__ == "__main__":
    unittest.main()
