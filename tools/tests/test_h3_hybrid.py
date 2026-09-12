"""長尺の全尺計算・条件の割当て・実行設定・履歴を検証する。"""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from modules_forge import minimax_h3_bridge as bridge
from modules_forge.minimax_h3_acceleration import H3Acceleration
from modules_forge.minimax_h3_hybrid import PACK, REVISION, H3Hybrid, validate_nodes
from tools.tests.media_fixtures import write_video


class HybridTests(unittest.TestCase):
    def request(self, **changes):
        return replace(
            bridge.H3Request(
                mode="text",
                prompt="A continuous shot",
                duration_seconds=10,
                acceleration=H3Acceleration(hybrid=H3Hybrid(enabled=True)),
            ),
            **changes,
        )

    def test_full_timeline_and_shared_sampling_schedule(self):
        request = self.request(seed=42)
        graph = bridge.build_workflow(request, {})
        self.assertEqual(request.frame_count, 651)
        self.assertEqual(request.effective_seconds, 27.125)
        self.assertEqual(graph["h3_latent"]["inputs"]["length"], 651)
        first, last = graph["h3_warmup"]["inputs"], graph["10"]["inputs"]
        for name in ("noise_seed", "steps", "cfg", "sampler_name", "scheduler", "positive", "negative"):
            self.assertEqual(first[name], last[name])
        self.assertEqual(first["end_at_step"], last["start_at_step"])
        self.assertEqual((first["add_noise"], last["add_noise"]), ("enable", "disable"))
        self.assertEqual(first["sampler_name"], "euler")
        self.assertEqual(graph["11"]["inputs"]["samples"], ["10", 0])
        self.assertEqual(graph["12"]["inputs"]["samples"], ["10", 0])
        for node in graph.values():
            for value in node["inputs"].values():
                if isinstance(value, list):
                    self.assertIn(value[0], graph)

    def test_first_frame_is_local_and_end_guide_recurs(self):
        option = H3Hybrid(enabled=True, prompts="Looks left\nLooks forward\nLooks right")
        request = self.request(
            mode="keyframes", first_frame="first.png", last_frame="last.png", acceleration=H3Acceleration(hybrid=option)
        )
        graph = bridge.build_workflow(request, {"first_frame": "first.png", "last_frame": "last.png"})
        for index, node_id in enumerate(("5", "h3_cond_1", "h3_cond_2")):
            inputs = graph[node_id]["inputs"]
            self.assertEqual(inputs["length"], 243)
            self.assertEqual("first_frame" in inputs, index == 0)
            self.assertEqual(inputs["last_frame"], ["21", 0])
            self.assertEqual(inputs["prompt"], request.prompt + "\n" + option.prompts.splitlines()[index])

    def test_references_are_shared_without_copying_input_loaders(self):
        request = self.request(mode="references", prompt="<Picture 1> turns slowly", reference_images=("portrait.png",))
        graph = bridge.build_workflow(request, {"images": ["portrait.png"]})
        for node_id in ("5", "h3_cond_1", "h3_cond_2"):
            self.assertEqual(graph[node_id]["inputs"]["ref_images.ref_image_0"], ["20", 0])
        self.assertEqual(sum(node["class_type"] == "LoadImage" for node in graph.values()), 1)

    def test_invalid_intervals_and_prompts_fail_before_submission(self):
        for option in (
            H3Hybrid(True, 3, 39, 21),
            H3Hybrid(True, 3, 260),
            H3Hybrid(True, 3, 39, 16, "one\ntwo"),
            H3Hybrid(True, 3, 39, 16, "one\n\nthree"),
            H3Hybrid(True, 2.5),
            H3Hybrid(True, overlap=40),
        ):
            with self.subTest(option=option), self.assertRaises(bridge.H3BridgeError):
                bridge.validate_request(self.request(acceleration=H3Acceleration(hybrid=option)))
        with self.assertRaisesRegex(bridge.H3BridgeError, "画像"):
            bridge.validate_request(self.request(reference_audios=("audio.wav",)))

    def test_sequential_baseline_and_disabled_mode(self):
        request = self.request(acceleration=H3Acceleration(hybrid=H3Hybrid(True, switch_step=20)))
        self.assertEqual(bridge.build_workflow(request, {})["10"]["inputs"]["start_at_step"], 20)
        disabled = replace(request, acceleration=H3Acceleration())
        graph = bridge.build_workflow(disabled, {})
        self.assertEqual(disabled.frame_count, 243)
        self.assertEqual(graph["10"]["class_type"], "SamplerCustomAdvanced")
        self.assertNotIn("h3_windows", graph)

    def test_runtime_permits_only_explicitly_selected_pack(self):
        request = self.request()
        command = bridge._runtime_command(Path("python"), 8188, acceleration=request.acceleration)
        self.assertIn(PACK, command)
        self.assertTrue(bridge._runtime_arguments_are_allowed(command[1:]))
        self.assertNotIn(PACK, bridge._runtime_command(Path("python"), 8188))
        self.assertEqual(H3Acceleration.from_dict(request.acceleration.to_dict()), request.acceleration)
        self.assertEqual(H3Acceleration.from_values(request.acceleration.values()), request.acceleration)

    def test_missing_schema_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "H3HybridWindows"):
            validate_nodes({})

    def test_long_generation_uses_a_bounded_longer_status_timeout(self):
        for enabled, timeout in ((False, 15.0), (True, 120.0)):
            request = self.request(acceleration=H3Acceleration(hybrid=H3Hybrid(enabled)))
            ready = bridge.RuntimeReadiness(
                Path("runtime"), bridge.H3_SERVER_URL, connected=True, acceleration=request.acceleration
            )
            with (
                self.subTest(enabled=enabled),
                patch.object(bridge, "ensure_ready", return_value=ready),
                patch.object(bridge, "_validate_request_runtime_constraints"),
                patch.object(bridge, "cleanup_stale_prepared_media"),
                patch.object(bridge, "prepare_media", return_value={}),
                patch.object(bridge, "build_workflow", return_value={}),
                patch.object(bridge, "cleanup_prepared_media"),
                patch.object(bridge, "_schedule_deferred_cleanup"),
                patch.object(bridge, "_loopback_server_process"),
                patch.object(bridge, "pending_jobs"),
                patch.object(bridge, "GPUOwnership"),
                patch.object(bridge, "_GPU_OWNERSHIPS", {}),
                patch.object(bridge, "_ACTIVE_GENERATION_IDS", set()),
                patch.object(bridge, "ComfyH3Client") as client,
            ):
                client.return_value.submit.return_value = "timeout-test"
                events = bridge.run_generation(
                    request, Path("runtime"), bridge.H3_SERVER_URL, Path("logs"), Path("outputs")
                )
                try:
                    self.assertEqual([next(events)["stage"] for _ in range(3)], ["runtime", "prepare", "queued"])
                    client.assert_called_once_with(bridge.H3_SERVER_URL, timeout=timeout)
                finally:
                    events.close()

    def test_saved_history_keeps_window_settings_and_total_frames(self):
        request = self.request()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            write_video(source, width=864, height=480, frames=651)
            output = root / "output"
            ready = bridge.RuntimeReadiness(runtime_root=root, server_url=bridge.H3_SERVER_URL, connected=True)
            result = bridge.mirror_result(source, output, request, "hybrid-test", 42, ready)
            metadata = json.loads(result.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["frames"], 651)
            self.assertEqual(metadata["hybrid_revision"], REVISION)
            item = bridge.HistoryItem(result, 1.0, "forge")
            restored = bridge.load_history_request(item.public_id, [item], output)
            self.assertEqual(restored.acceleration.hybrid, request.acceleration.hybrid)
            self.assertEqual(restored.frame_count, 651)
