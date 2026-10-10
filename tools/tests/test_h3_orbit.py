"""Orbit uses one image at both ends; ordinary H3 keeps its AV contract."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from PIL import Image

from modules_forge import minimax_h3_bridge as bridge
from modules_forge.minimax_h3_acceleration import H3Acceleration
from modules_forge.minimax_h3_hybrid import H3Hybrid
from modules_forge.minimax_h3_negpip import H3NegPiP
from tools.tests.media_fixtures import write_video


class H3OrbitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.photo = self.root / "photo.png"
        Image.new("RGB", (640, 384), "blue").save(self.photo)

    def request(self, **changes):
        return bridge.H3Request(
            mode="orbit",
            prompt="Camera orbits a still subject.",
            first_frame=str(self.photo),
            aspect="1:1",
            quality="native",
            duration_seconds=3,
            steps=28,
            seed=42,
            **changes,
        )

    def test_author_recipe_uses_73_frames_and_preserves_normal_duration_bounds(self):
        request = self.request()
        bridge.validate_request(request)
        self.assertEqual(request.frame_count, 73)
        self.assertEqual(request.dimensions, (768, 768))
        self.assertFalse(request.has_audio)
        for mode in ("text", "keyframes"):
            with self.subTest(mode=mode), self.assertRaises(bridge.H3BridgeError):
                bridge.validate_request(replace(request, mode=mode))
        self.assertEqual(bridge.H3Request(mode="text", prompt="A scene").frame_count, 124)

    def test_orbit_rejects_missing_image_invalid_strength_or_multiple_turn_windows(self):
        for changes in (
            {"first_frame": None},
            {"orbit_strength": float("nan")},
            {"orbit_strength": True},
            {"orbit_strength": 0},
            {"orbit_strength": 2.1},
            {"last_frame": str(self.photo)},
            {"acceleration": H3Acceleration(hybrid=H3Hybrid(enabled=True))},
        ):
            with self.subTest(changes=changes), self.assertRaises(bridge.H3BridgeError):
                bridge.validate_request(replace(self.request(), **changes))

    def test_graph_scales_once_and_patches_sampling_model_without_audio_output(self):
        request = self.request(orbit_strength=0.75)
        graph = bridge.build_workflow(request, {"first_frame": "photo.png"}, seed=42)
        nodes = {node["class_type"]: (key, node["inputs"]) for key, node in graph.items()}
        lora_id, lora = nodes["LoraLoaderModelOnly"]
        self.assertEqual(lora["strength_model"], 0.75)
        self.assertEqual(lora["model"], ["1", 0])
        self.assertEqual(graph["15"]["inputs"]["model"], [lora_id, 0])
        self.assertEqual(graph["8"]["inputs"]["model"], [lora_id, 0])
        scale_id, scale = nodes["ImageScale"]
        self.assertEqual((scale["width"], scale["height"], scale["crop"]), (768, 768, "center"))
        cond = graph["5"]["inputs"]
        self.assertEqual(cond["first_frame"], [scale_id, 0])
        self.assertEqual(cond["last_frame"], [scale_id, 0])
        self.assertEqual(cond["length"], 73)
        self.assertNotIn("VAEDecodeAudio", nodes)
        self.assertNotIn("4", graph)
        self.assertNotIn("audio", graph["13"]["inputs"])
        with self.assertRaises(bridge.H3BridgeError):
            bridge.build_workflow(request, {})

    def test_media_preparation_copies_only_one_input(self):
        (self.root / "input").mkdir()
        with mock.patch.object(bridge, "_copy_to_comfy_input", return_value="prepared.png") as copy:
            prepared = bridge.prepare_media(self.request(), self.root)
        copy.assert_called_once()
        self.assertEqual(prepared["first_frame"], "prepared.png")
        self.assertIsNone(prepared["last_frame"])

    def test_orbit_preserves_negpip_sparse_and_clip_cache_model_routing(self):
        acceleration = H3Acceleration(negpip=H3NegPiP(enabled=True), clip_cache="auto")
        graph = bridge.build_workflow(self.request(acceleration=acceleration), {"first_frame": "photo.png"})
        self.assertEqual(graph["8"]["inputs"]["model"], ["17", 0])
        self.assertEqual(graph["9"]["inputs"]["model"], ["17", 0])
        self.assertEqual(graph["17"]["inputs"]["model"], ["15", 0])
        self.assertEqual(graph["15"]["inputs"]["model"], ["40", 0])
        self.assertEqual(graph["5"]["inputs"]["first_frame"], graph["5"]["inputs"]["last_frame"])
        graph = bridge.build_workflow(
            self.request(acceleration=H3Acceleration(attention="sol")), {"first_frame": "photo.png"}
        )
        self.assertEqual(graph["9"]["inputs"]["model"], ["16", 0])
        self.assertEqual(graph["16"]["inputs"]["model"], ["15", 0])

    def test_missing_orbit_weights_stop_before_submission(self):
        ready = bridge.RuntimeReadiness(self.root, bridge.H3_SERVER_URL, connected=True)
        with (
            mock.patch.object(
                bridge.RuntimeReadiness, "ready_for_fl2va", new_callable=mock.PropertyMock, return_value=True
            ),
            self.assertRaisesRegex(bridge.H3BridgeError, "Orbit"),
        ):
            bridge._validate_request_runtime_constraints(self.request(), ready, "fast")

    def test_runtime_requires_the_verified_lora_to_be_visible_to_comfyui(self):
        ready = bridge.RuntimeReadiness(self.root, bridge.H3_SERVER_URL, connected=True)
        specs = {
            "LoraLoaderModelOnly": {
                "input": {
                    "required": {
                        "model": ["MODEL"],
                        "lora_name": [["different.safetensors"]],
                        "strength_model": ["FLOAT"],
                    }
                }
            },
            "ImageScale": {
                "input": {"required": {key: ["INT"] for key in ("image", "width", "height", "crop", "upscale_method")}}
            },
        }
        with (
            mock.patch.object(bridge.orbit_assets, "validate_model"),
            mock.patch.object(bridge, "ComfyH3Client") as api,
        ):
            api.return_value.object_info.return_value = specs
            with self.assertRaisesRegex(bridge.H3BridgeError, "接続先"):
                bridge.validate_orbit_runtime(ready)
            specs["LoraLoaderModelOnly"]["input"]["required"]["lora_name"][0] = [bridge.orbit_assets.MODEL_NAME]
            bridge.validate_orbit_runtime(ready)
            self.assertEqual(api.return_value.close.call_count, 2)

    def test_silent_video_is_published_with_orbit_identity_and_restorable_settings(self):
        request = replace(self.request(), quality="draft")
        source = self.root / "source.mp4"
        write_video(source, width=448, height=448, frames=73, audio=False)
        ready = bridge.RuntimeReadiness(self.root, bridge.H3_SERVER_URL, connected=True)
        output = self.root / "output"
        target = bridge.mirror_result(source, output, request, "orbit-1", 42, ready)
        meta = json.loads(target.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual(meta["orbit"]["strength"], 1.0)
        self.assertEqual(len(meta["orbit"]["sha256"]), 64)
        self.assertEqual(meta["output_validation"]["frames"], 73)
        self.assertEqual(meta["output_validation"]["audio_channels"], 0)
        items = bridge.list_history(self.root, output)
        selected = next(item for item in items if item.path == target)
        restored = bridge.load_history_request(selected.public_id, items, output)
        self.assertEqual(restored.mode, "orbit")
        self.assertEqual(restored.orbit_strength, 1.0)
        self.assertEqual(restored.frame_count, 73)
        self.assertIsNone(restored.first_frame)

    def test_orbit_output_rejects_audio_and_incomplete_video(self):
        request = replace(self.request(), quality="draft")
        source = self.root / "invalid.mp4"
        for kwargs in ({"frames": 73, "audio": True}, {"frames": 56, "audio": False}):
            write_video(source, width=448, height=448, **kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(bridge.H3BridgeError):
                bridge._validate_video_result(source, request)

    def test_orbit_history_refuses_missing_or_changed_model_identity(self):
        target = self.root / "history.mp4"
        target.write_bytes(b"history-item")
        item = bridge.HistoryItem(target, 0, "Forge Neo")
        record = {
            "model": "MiniMax H3",
            "mode": "orbit",
            "prompt": "Camera orbits a still subject.",
            "aspect": "1:1",
            "quality": "native",
            "requested_seconds": 3,
            "steps": 28,
            "seed": 42,
            "scheduler": "simple",
            "ref_image_size": "match",
        }
        valid = bridge.orbit_metadata(self.request())
        broken = [
            None,
            {},
            *[{key: value for key, value in valid.items() if key != missing} for missing in valid],
            *[valid | {key: "changed"} for key in ("model", "revision", "sha256")],
        ]
        for orbit in broken:
            target.with_suffix(".json").write_text(json.dumps(record | {"orbit": orbit}), encoding="utf-8")
            with self.subTest(orbit=orbit), self.assertRaises(bridge.H3BridgeError):
                bridge.load_history_request(item.public_id, [item], self.root)

    def test_orbit_summary_reports_real_duration_and_silent_output(self):
        rendered = bridge.settings_summary_html("1:1", "native", 3, 28, orbit=True)
        self.assertIn("73 frames", rendered)
        self.assertIn("3.04 sec", rendered)
        self.assertIn("無音", rendered)
        self.assertNotIn("stereo", rendered)
        self.assertNotIn("音声付き", bridge.history_html([]))


if __name__ == "__main__":
    unittest.main()
