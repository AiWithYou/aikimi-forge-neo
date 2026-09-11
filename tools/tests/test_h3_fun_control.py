"""Fun ControlNetの入力整形、モデル接続、片付けを検証する。"""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import av
import numpy as np

from modules_forge import minimax_h3_bridge as bridge
from modules_forge import minimax_h3_fun_control as control
from modules_forge.minimax_h3_acceleration import H3Acceleration
from modules_forge.minimax_h3_negpip import H3NegPiP


def write_video(path, frames, fps=12):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264rgb", rate=fps)
        stream.width = stream.height = 32
        stream.pix_fmt = "rgb24"
        stream.options = {"crf": "0", "preset": "ultrafast"}
        for value in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(value, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class FunControlTests(unittest.TestCase):
    def test_v3_attention_options_are_detected(self):
        nodes = {"ModelAttentionBackend": {"input": {"required": {"attention": ["COMBO", {"options": ["pytorch attention", "comfy kitchen attention"]}]}}}}
        self.assertTrue(bridge.ck_attention_available(nodes))
        nodes["ModelAttentionBackend"]["input"]["required"]["attention"][1]["options"] = ["pytorch attention"]
        self.assertFalse(bridge.ck_attention_available(nodes))

    def test_h3_launch_keeps_memory_compiler_enabled(self):
        command = bridge._runtime_command(Path("python"), 8199)
        self.assertNotIn("--disable-comfy-compiler", command)
        self.assertTrue(bridge._runtime_arguments_are_allowed(command[1:]))

    def test_controls_validate_and_missing_video_fails_before_generation(self):
        for value in ({"mode": []}, {"mode": "pose"}, {"strength": float("nan")}, {"strength": True}, {"unexpected": 1}):
            with self.assertRaises(ValueError):
                control.H3FunControl.from_dict(value)
        with self.assertRaisesRegex(bridge.H3BridgeError, "制御動画"):
            bridge.validate_request(bridge.H3Request(mode="text", prompt="test", control=control.H3FunControl("canny")))

    def test_off_does_not_change_graph(self):
        graph = bridge.build_workflow(bridge.H3Request(mode="text", prompt="test"), {}, seed=1)
        before = copy.deepcopy(graph)
        control.apply_workflow(graph, control.H3FunControl(), None)
        self.assertEqual(graph, before)

    def test_native_patch_composes_after_negpip_and_cached_clip(self):
        request = bridge.H3Request(
            mode="text", prompt="(shake:-0.7)", control=control.H3FunControl("canny", 0.8), control_video="motion.mp4",
            acceleration=H3Acceleration(clip_cache="auto", negpip=H3NegPiP(enabled=True)),
        )
        graph = bridge.build_workflow(request, {"control_video": "forge_h3/prepared.mp4"}, seed=5)
        self.assertEqual(graph["103"]["inputs"]["model"], ["17", 0])
        self.assertEqual(graph["9"]["inputs"]["model"], ["103", 0])
        self.assertEqual(graph["8"]["inputs"]["model"], ["103", 0])
        self.assertEqual(graph["100"]["inputs"]["name"], control.MODEL)
        self.assertEqual(graph["103"]["inputs"]["strength"], 0.8)
        self.assertNotIn("2", graph)
        self.assertEqual(graph["13"]["inputs"]["audio"], ["12", 0])

    def test_video_resamples_to_24fps_and_pads_with_last_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = (Path(directory) / name for name in ("source.mp4", "target.mp4"))
            frames = [np.full((32, 32, 3), value, dtype=np.uint8) for value in (20, 180)]
            write_video(source, frames)
            control.prepare_video(source, target, 64, 32, 5, "preprocessed")
            with av.open(str(target)) as container:
                self.assertEqual(float(container.streams.video[0].average_rate), 24)
                decoded = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
            self.assertEqual([int(f[0, 0, 0]) for f in decoded], [20, 20, 180, 180, 180])
            self.assertTrue(all(f.shape == (32, 64, 3) for f in decoded))
            self.assertTrue(source.exists())

    def test_canny_produces_binary_edges_in_three_channels(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = (Path(directory) / name for name in ("source.mp4", "target.mp4"))
            frame = np.zeros((32, 32, 3), dtype=np.uint8)
            frame[8:24, 8:24] = 255
            write_video(source, [frame])
            control.prepare_video(source, target, 32, 32, 5, "canny")
            with av.open(str(target)) as container:
                frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
            self.assertEqual(len(frames), 5)
            self.assertEqual(set(np.unique(frames[0])), {0, 255})
            self.assertTrue(np.array_equal(frames[0][..., 0], frames[0][..., 2]))
            self.assertLess(np.count_nonzero(frames[0][..., 0]), 100)

    def test_failed_preparation_removes_only_the_generated_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input").mkdir()
            source = root / "source.mp4"
            source.write_bytes(b"source")
            request = bridge.H3Request(mode="text", prompt="test", control=control.H3FunControl("canny"), control_video=str(source))

            def fail(source, target, *args):
                target.write_bytes(b"partial")
                raise ValueError("decode failed")

            with mock.patch.object(bridge, "_validate_media_path", return_value=source), mock.patch.object(control, "prepare_video", side_effect=fail):
                with self.assertRaisesRegex(ValueError, "decode failed"):
                    bridge.prepare_media(request, root)
            self.assertEqual(source.read_bytes(), b"source")
            self.assertEqual(list((root / "input/forge_h3").glob("*.mp4")), [])

    def test_missing_runtime_model_is_actionable(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "INT8モデル"):
                control.validate_model(Path(directory))


if __name__ == "__main__":
    unittest.main()
