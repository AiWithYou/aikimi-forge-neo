"""Stream RGB frames to FFmpeg without publishing failed or cancelled output."""

from __future__ import annotations

import math
import os
import stat
import subprocess
import tempfile
from collections.abc import Iterable
from contextlib import suppress
from fractions import Fraction
from itertools import chain
from pathlib import Path

import numpy as np


class VideoEncodingError(RuntimeError):
    """FFmpeg did not produce a complete output file."""


class VideoEncodingCancelled(Exception):
    """The frame producer was cancelled; discard the incomplete output."""


def _validate_frame(frame: np.ndarray, shape: tuple[int, ...] | None = None) -> None:
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise ValueError("Video frames must be uint8 NumPy arrays")
    if frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) <= 0:
        raise ValueError("Video frames must have (height, width, 3) RGB shape")
    if shape is not None and frame.shape != shape:
        raise ValueError("All video frames must have the same dimensions")


def write_video(
    filename: str | os.PathLike[str],
    frames: Iterable[np.ndarray],
    fps: int | float | Fraction = 16,
    *,
    crf: int = 23,
    preset: str = "medium",
    profile: str = "main",
    info: str | None = None,
    audio_copy: str | os.PathLike[str] | None = None,
) -> None:
    """Consume frames once, keeping a constant number of frames in Python memory.

    Errors from the producer propagate after FFmpeg has been reaped. Encoding
    always targets a unique sibling file, so failure preserves any older output.
    Stderr is inherited rather than piped to avoid filling an unread error pipe.
    """
    if not math.isfinite(float(fps)) or fps <= 0:
        raise ValueError("Video frame rate must be finite and positive")
    iterator = iter(frames)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("Cannot save a video without frames") from exc
    _validate_frame(first)
    height, width, _ = first.shape
    shape = first.shape
    target = Path(filename)
    # Let FFmpeg create the file with the process umask, inside a private sibling
    # directory. Precreating it with mkstemp would publish owner-only videos.
    temporary_directory = tempfile.TemporaryDirectory(prefix=".aikimi-video-", dir=target.parent)
    temporary = Path(temporary_directory.name) / f"output{target.suffix}"
    process = None
    try:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-hwaccel",
            "auto",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
        ]
        if audio_copy is not None:
            command += ["-i", os.fspath(audio_copy), "-map", "0:v", "-map", "1:a?", "-acodec", "copy"]
        command += [
            "-vcodec",
            "h264",
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-pix_fmt",
            "yuv420p",
            "-profile:v",
            profile,
            "-metadata",
            f"description={str(info)}",
            os.fspath(temporary),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)  # noqa: S603 - fixed executable, no shell
        for frame in chain((first,), iterator):
            _validate_frame(frame, shape)
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        returncode = process.wait()
        if returncode != 0:
            raise VideoEncodingError(f"FFmpeg failed with exit code {returncode}; no video was published")
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise VideoEncodingError("FFmpeg produced an empty output; no video was published")
        try:
            mode = stat.S_IMODE(target.stat().st_mode)
        except FileNotFoundError:
            # New outputs retain FFmpeg's normal umask-derived permissions.
            pass
        else:
            os.chmod(temporary, mode)
        os.replace(temporary, target)
    except BrokenPipeError as exc:
        raise VideoEncodingError("FFmpeg stopped accepting frames; no video was published") from exc
    finally:
        try:
            if process is not None:
                if process.poll() is None:
                    with suppress(OSError):
                        process.kill()
                with suppress(OSError):
                    process.stdin.close()
                process.wait()
        finally:
            temporary_directory.cleanup()
