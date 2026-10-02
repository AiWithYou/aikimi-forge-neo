from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules_forge import minimax_h3_bridge as bridge
from tools.tests.test_minimax_h3_studio import load_studio_module


class H3InputPreservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ui = load_studio_module()

    def request_from_ui(self, steps=20, seed=-1):
        return self.ui._request_from_ui(
            "text",
            "A scene",
            None,
            None,
            None,
            None,
            None,
            "16:9",
            "preview",
            5.0,
            steps,
            seed,
            "simple",
            "match",
        )

    def test_seed_control_preserves_the_full_supported_integer_range(self):
        with (
            patch.object(self.ui, "_initial_runtime", return_value=None),
            patch.object(self.ui, "_initial_history_state", return_value=([], "", [])),
        ):
            tabs = self.ui._build_ui()
        interface = tabs[0][0]
        control = next(component for component in interface.blocks.values() if component.elem_id == "h3-seed")

        self.assertEqual(control.get_block_name(), "textbox")
        seed = 2**63 - 1
        browser_value = control.postprocess(seed)
        self.assertEqual(browser_value, str(seed))
        self.assertEqual(self.request_from_ui(seed=browser_value).seed, seed)

    def test_history_restores_seed_as_text_without_losing_digits(self):
        seed = 2**63 - 1
        request = bridge.H3Request(mode="text", prompt="A scene", seed=seed)
        with (
            patch.object(self.ui, "_history_state", return_value=([], "", [])),
            patch.object(self.ui, "load_history_request", return_value=request),
        ):
            values = self.ui._restore_history_settings("saved", "runtime")

        self.assertEqual(values[15]["value"], str(seed))
        self.assertEqual(self.request_from_ui(seed=values[15]["value"]).seed, seed)

    def test_ui_and_bridge_reject_fractional_and_boolean_steps_and_seed(self):
        for field, value in (("steps", 20.9), ("seed", 42.9), ("steps", True), ("seed", True)):
            with self.subTest(field=field, value=value):
                with self.assertRaises(bridge.H3BridgeError):
                    self.request_from_ui(**{field: value})
                request = bridge.H3Request(mode="text", prompt="A scene", **{field: value})
                with self.assertRaises(bridge.H3BridgeError):
                    bridge.validate_request(request)

        bridge.validate_request(self.request_from_ui(steps=20.0, seed="42"))

    def test_history_rejects_fractional_steps_and_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            video = output / "MiniMax_H3_result.mp4"
            video.write_bytes(b"generated")
            item = bridge.HistoryItem(video, video.stat().st_mtime, "Forge Neo")
            metadata = {
                "model": "MiniMax H3",
                "mode": "text",
                "prompt": "A scene",
                "aspect": "16:9",
                "quality": "preview",
                "requested_seconds": 5.0,
                "steps": 20,
                "seed": 42,
                "scheduler": "simple",
                "ref_image_size": "match",
            }
            for field, value in (("steps", 20.9), ("seed", 42.9)):
                with self.subTest(field=field):
                    video.with_suffix(".json").write_text(json.dumps(metadata | {field: value}), encoding="utf-8")
                    with self.assertRaises(bridge.H3BridgeError):
                        bridge.load_history_request(item.public_id, [item], output)


if __name__ == "__main__":
    unittest.main()
