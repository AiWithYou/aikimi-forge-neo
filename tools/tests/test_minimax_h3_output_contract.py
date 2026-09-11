"""動画の成功判定を、実際のMP4・音声と独立した期待値で検証する。"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from modules_forge import minimax_h3_bridge as bridge
from tools.tests.media_fixtures import write_video


class H3OutputContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source.mp4"
        self.output = self.root / "output"
        self.request = bridge.H3Request(mode="text", prompt="A blue field", quality="draft", aspect="1:1", seed=42)
        self.ready = bridge.RuntimeReadiness(self.root, bridge.H3_SERVER_URL, connected=True)

    def publish(self):
        return bridge.mirror_result(self.source, self.output, self.request, "output-test", 42, self.ready)

    def write_video(self, **changes):
        # 5秒の指定はH3の17n+5制約により124フレーム。実装から期待値をコピーしない。
        values = {"width": 448, "height": 448, "frames": 124, "fps": 24}
        write_video(self.source, **(values | changes))

    def test_valid_video_publishes_measured_evidence_and_identical_bytes(self):
        self.write_video()
        target = self.publish()
        self.assertEqual(target.read_bytes(), self.source.read_bytes())
        metadata = json.loads(target.with_suffix(".json").read_text(encoding="utf-8"))
        measured = metadata["output_validation"]
        self.assertEqual((measured["width"], measured["height"], measured["frames"]), (448, 448, 124))
        self.assertEqual((measured["fps"], measured["audio_sample_rate"], measured["audio_channels"]), (24, 32000, 2))
        self.assertEqual(metadata["output_sha256"], hashlib.sha256(target.read_bytes()).hexdigest())

    def test_malformed_or_incomplete_result_is_not_published(self):
        cases = {
            "wrong_size": {"width": 416},
            "short_video": {"frames": 100},
            "extra_frames": {"frames": 125},
            "wrong_fps": {"fps": 25},
            "missing_audio": {"audio": False},
            "short_audio": {"audio_seconds": 1.0},
            "long_audio": {"audio_seconds": 7.0},
        }
        for name, changes in cases.items():
            with self.subTest(name=name):
                self.write_video(**changes)
                with self.assertRaises(bridge.H3BridgeError):
                    self.publish()
                self.assertEqual(list(self.output.iterdir()), [])
        self.source.write_bytes(b"not an MP4")
        with self.assertRaises(bridge.H3BridgeError):
            self.publish()
        self.assertEqual(list(self.output.iterdir()), [])

    def test_retry_save_failure_keeps_previous_video_and_metadata(self):
        self.write_video()
        with patch.object(bridge, "datetime") as clock:
            clock.now.return_value = datetime(2026, 9, 10, 12)
            target = self.publish()
            previous_video = target.read_bytes()
            previous_metadata = target.with_suffix(".json").read_bytes()
            replace_file = bridge.os.replace

            def fail_final_video(source, destination):
                if Path(destination).suffix == ".mp4":
                    raise OSError("disk full while publishing video")
                return replace_file(source, destination)

            with patch.object(bridge.os, "replace", side_effect=fail_final_video):
                with self.assertRaises(bridge.H3BridgeError):
                    self.publish()
        self.assertEqual(target.read_bytes(), previous_video)
        self.assertEqual(target.with_suffix(".json").read_bytes(), previous_metadata)
        self.assertEqual(len(list(self.output.iterdir())), 2)


if __name__ == "__main__":
    unittest.main()
