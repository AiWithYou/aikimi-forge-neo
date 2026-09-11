"""CPU integration checks using a real encoder, with no models or downloads."""

from __future__ import annotations

import json
import shutil
import stat
import subprocess
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

from modules.video_writer import VideoEncodingCancelled, VideoEncodingError, write_video

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


@unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe are required for encoder integration tests")
class VideoWriterIntegrationTests(unittest.TestCase):
    def test_stream_and_list_have_identical_decoded_frames_and_rational_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            frames = [np.full((24, 32, 3), value * 30, dtype=np.uint8) for value in range(8)]
            permission_reference = Path(directory) / "permissions-reference"
            permission_reference.write_bytes(b"reference")
            default_mode = stat.S_IMODE(permission_reference.stat().st_mode)
            permission_reference.unlink()
            decoded = []
            for name, source in (("list", frames), ("stream", iter(frames))):
                path = Path(directory) / f"{name}.mp4"
                expected_mode = default_mode
                if name == "stream":
                    path.write_bytes(b"existing video")
                    path.chmod(0o640)
                    expected_mode = stat.S_IMODE(path.stat().st_mode)
                write_video(path, source, Fraction(30000, 1001), preset="ultrafast", info="seed: 123")
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), expected_mode)
                probe = subprocess.run(  # noqa: S603 - resolved executable and temporary test paths
                    [FFPROBE, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                details = json.loads(probe.stdout)
                stream = details["streams"][0]
                self.assertEqual((stream["width"], stream["height"]), (32, 24))
                self.assertEqual(stream["avg_frame_rate"], "30000/1001")
                self.assertEqual(int(stream["nb_frames"]), len(frames))
                self.assertEqual(details["format"]["tags"]["description"], "seed: 123")
                result = subprocess.run(  # noqa: S603 - resolved executable and temporary test paths
                    [FFMPEG, "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                    check=True,
                    capture_output=True,
                    timeout=20,
                )
                decoded.append(result.stdout)
            self.assertEqual(len(decoded[0]), 8 * 24 * 32 * 3)
            self.assertEqual(decoded[0], decoded[1])
            self.assertFalse(list(Path(directory).glob(".aikimi-video-*")))

    def test_encoder_failure_preserves_old_output_and_removes_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.mp4"
            path.write_bytes(b"existing video")
            with self.assertRaises(VideoEncodingError):
                write_video(path, [np.zeros((16, 16, 3), dtype=np.uint8)], preset="invalid-test-preset")
            self.assertEqual(path.read_bytes(), b"existing video")
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_cancelled_producer_preserves_old_output_and_reaps_encoder(self):
        def frames():
            yield np.zeros((16, 16, 3), dtype=np.uint8)
            raise VideoEncodingCancelled()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.mp4"
            path.write_bytes(b"existing video")
            with self.assertRaises(VideoEncodingCancelled):
                write_video(path, frames(), preset="ultrafast")
            self.assertEqual(path.read_bytes(), b"existing video")
            self.assertEqual(list(Path(directory).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
