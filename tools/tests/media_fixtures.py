"""GPUや外部ファイルを使わず、復号できる検証用メディアを作る。"""

from fractions import Fraction
from pathlib import Path

import av
import numpy as np


def write_video(
    path: Path,
    *,
    width: int,
    height: int,
    frames: int,
    fps: int = 24,
    audio: bool = True,
    audio_seconds: float | None = None,
) -> None:
    with av.open(str(path), "w", format="mp4") as container:
        video = container.add_stream("libx264", rate=fps)
        video.width, video.height, video.pix_fmt = width, height, "yuv420p"
        video.options = {"preset": "ultrafast", "crf": "30"}
        video.codec_context.thread_count = 1
        sound = container.add_stream("aac", rate=32000) if audio else None
        if sound is not None:
            sound.layout = "stereo"
        pixels = np.zeros((height, width, 3), dtype=np.uint8)
        pixels[:, :, 2] = 160
        for index in range(frames):
            pixels[:, :, 0] = index % 256
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, fps)
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        if sound is not None:
            samples = round((frames / fps if audio_seconds is None else audio_seconds) * 32000)
            for start in range(0, samples, 1024):
                wave = (np.sin(np.arange(start, min(start + 1024, samples)) * (440 * 2 * np.pi / 32000)) * 0.1).astype(
                    np.float32
                )
                frame = av.AudioFrame.from_ndarray(np.stack((wave, wave)), format="fltp", layout="stereo")
                frame.sample_rate, frame.pts, frame.time_base = 32000, start, Fraction(1, 32000)
                for packet in sound.encode(frame):
                    container.mux(packet)
            for packet in sound.encode():
                container.mux(packet)
