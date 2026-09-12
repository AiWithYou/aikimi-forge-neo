"""Regression tests for user settings and video postprocessing without a GPU."""

from __future__ import annotations

import ast
import contextlib
import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from modules.atomic_file import atomic_write_json
from modules.video_writer import VideoEncodingCancelled, VideoEncodingError, write_video

ROOT = Path(__file__).resolve().parents[2]


def source_functions(path, names, namespace, *, owner=None):
    """Execute the actual boundary functions without booting Forge or loading models."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = tree.body
    if owner:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == owner).body
    selected = [node for node in nodes if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=selected, type_ignores=[]), path, "exec"), namespace)  # noqa: S102
    return namespace


class AtomicFileTests(unittest.TestCase):
    def test_write_preserves_unicode_and_existing_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text("{}", encoding="utf-8")
            path.chmod(0o600)
            mode = path.stat().st_mode
            atomic_write_json(path, {"prompt": "日本語"})
            self.assertEqual(path.read_text(encoding="utf-8"), '{\n    "prompt": "日本語"\n}')
            self.assertEqual(path.stat().st_mode, mode)

    def test_disk_failures_leave_existing_settings_untouched(self):
        for stage in ("os.fsync", "os.replace", "os.chmod"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "settings.json"
                path.write_text('{"original": true}', encoding="utf-8")
                with patch(f"modules.atomic_file.{stage}", side_effect=OSError("disk error")):
                    with self.assertRaisesRegex(OSError, "disk error"):
                        atomic_write_json(path, {"new": True})
                self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"original": True})
                self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_symlink_still_targets_original_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "real.json"
            link = Path(directory) / "settings.json"
            target.write_text("{}", encoding="utf-8")
            try:
                link.symlink_to(target)
            except OSError:
                self.skipTest("Symlink creation is unavailable on this host")
            atomic_write_json(link, {"new": True})
            self.assertTrue(link.is_symlink())
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), {"new": True})

    def test_concurrent_writes_leave_one_complete_document(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            documents = [{"writer": i, "payload": str(i) * 4000} for i in range(8)]
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda data: atomic_write_json(path, data), documents))
            self.assertIn(json.loads(path.read_text(encoding="utf-8")), documents)
            self.assertEqual(list(Path(directory).iterdir()), [path])


class SettingsPersistenceTests(unittest.TestCase):
    def test_failed_serialization_preserves_settings_and_cleans_temporary_files(self):
        save = source_functions(
            "modules/options.py",
            {"save"},
            {"json": json, "cmd_opts": SimpleNamespace(freeze_settings=False)},
            owner="Options",
        )["save"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            original = '{"prompt": "keep my settings"}'
            path.write_text(original, encoding="utf-8")
            with self.assertRaises(TypeError):
                save(SimpleNamespace(data={"valid": 1, "invalid": object()}), path)
            self.assertEqual(path.read_text(encoding="utf-8"), original)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_failed_serialization_preserves_ui_defaults(self):
        save = source_functions(
            "modules/ui_loadsave.py",
            {"write_to_file"},
            {"json": json},
            owner="UiLoadsave",
        )["write_to_file"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ui-config.json"
            path.write_text('{"old": true}', encoding="utf-8")
            with self.assertRaises(TypeError):
                save(SimpleNamespace(filename=path), {"invalid": object()})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"old": True})


class VideoSaveRegressionTests(unittest.TestCase):
    def save_fixture(self, directory, *, exit_code=0):
        class Encoder:
            def __init__(self, command, **kwargs):
                self.stdin = Mock()
                Path(command[-1]).write_bytes(b"encoded-video")

            def wait(self):
                return exit_code

            def poll(self):
                return exit_code

            def kill(self):
                pass

        opts = SimpleNamespace(
            outdir_samples="",
            outdir_videos=directory,
            video_container="mp4",
            video_crf=23,
            video_preset="ultrafast",
            video_profile="main",
            save_txt=True,
        )
        namespace = source_functions(
            "modules/images.py",
            {"save_video"},
            {
                "os": os,
                "np": np,
                "opts": opts,
                "get_next_sequence_number": lambda *args: 0,
                "subprocess": SimpleNamespace(Popen=Encoder),
                "chain": chain,
            },
        )
        return namespace["save_video"], Encoder

    def test_encoder_failure_does_not_report_success(self):
        with tempfile.TemporaryDirectory() as directory:
            save, encoder = self.save_fixture(directory, exit_code=1)
            with patch("subprocess.Popen", encoder), self.assertRaises(RuntimeError):
                save("clip", [np.zeros((16, 16, 3), np.uint8)])
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_sidecar_uses_the_actual_numbered_video_name(self):
        with tempfile.TemporaryDirectory() as directory:
            save, encoder = self.save_fixture(directory)
            with patch("subprocess.Popen", encoder):
                filename = save("clip", [np.zeros((16, 16, 3), np.uint8)], info="seed: 123")
            self.assertEqual(Path(filename).with_suffix(".txt").read_text(encoding="utf-8"), "seed: 123\n")
            self.assertFalse((Path(directory) / ".txt").exists())

    def test_save_video_accepts_single_pass_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            save, encoder = self.save_fixture(directory)
            frames = (np.full((16, 16, 3), value, np.uint8) for value in range(3))
            with patch("subprocess.Popen", encoder):
                result = save("clip", frames)
            self.assertTrue(Path(result).is_file())


class VideoWriterTests(unittest.TestCase):
    def test_rejects_empty_invalid_frames_and_frame_rates_before_launch(self):
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        cases = [
            ([], 16),
            ([np.zeros((16, 16, 3))], 16),
            ([frame[:, :, 0]], 16),
            ([np.zeros((0, 16, 3), dtype=np.uint8)], 16),
            ([frame], 0),
            ([frame], -1),
            ([frame], float("nan")),
            ([frame], float("inf")),
        ]
        for frames, rate in cases:
            with self.subTest(rate=rate), tempfile.TemporaryDirectory() as directory:
                with patch("modules.video_writer.subprocess.Popen") as launch, self.assertRaises(ValueError):
                    write_video(Path(directory) / "output.mp4", frames, rate)
                launch.assert_not_called()
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_missing_encoder_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.mp4"
            path.write_bytes(b"old")
            with patch("modules.video_writer.subprocess.Popen", side_effect=FileNotFoundError("ffmpeg")):
                with self.assertRaises(FileNotFoundError):
                    write_video(path, [np.zeros((16, 16, 3), dtype=np.uint8)])
            self.assertEqual(path.read_bytes(), b"old")
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_producer_failures_reap_encoder_and_preserve_output(self):
        for failure in (VideoEncodingCancelled(), ValueError("processor failure")):
            with self.subTest(failure=type(failure)), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "output.mp4"
                path.write_bytes(b"old")
                process = Mock()
                process.poll.return_value = None

                def frames(failure=failure):
                    yield np.zeros((16, 16, 3), dtype=np.uint8)
                    raise failure

                with patch("modules.video_writer.subprocess.Popen", return_value=process):
                    with self.assertRaises(type(failure)):
                        write_video(path, frames())
                process.kill.assert_called_once_with()
                process.wait.assert_called_once_with()
                process.stdin.close.assert_called_once_with()
                self.assertEqual(path.read_bytes(), b"old")
                self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_dimension_change_is_rejected_and_reaped(self):
        process = Mock()
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as directory:
            frames = [np.zeros((16, size, 3), dtype=np.uint8) for size in (16, 32)]
            with patch("modules.video_writer.subprocess.Popen", return_value=process):
                with self.assertRaisesRegex(ValueError, "same dimensions"):
                    write_video(Path(directory) / "output.mp4", frames)
            process.kill.assert_called_once_with()
            process.wait.assert_called_once_with()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_broken_pipe_is_an_encoding_error_and_reaps_process(self):
        process = Mock()
        process.stdin.write.side_effect = BrokenPipeError()
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as directory:
            with patch("modules.video_writer.subprocess.Popen", return_value=process):
                with self.assertRaises(VideoEncodingError):
                    write_video(Path(directory) / "output.mp4", [np.zeros((16, 16, 3), dtype=np.uint8)])
            process.kill.assert_called_once_with()
            process.wait.assert_called_once_with()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_empty_successful_encoder_output_is_not_published(self):
        process = Mock()
        process.poll.return_value = process.wait.return_value = 0
        with tempfile.TemporaryDirectory() as directory:
            with patch("modules.video_writer.subprocess.Popen", return_value=process):
                with self.assertRaisesRegex(VideoEncodingError, "empty output"):
                    write_video(Path(directory) / "output.mp4", [np.zeros((16, 16, 3), dtype=np.uint8)])
            self.assertEqual(list(Path(directory).iterdir()), [])


class PostprocessingReliabilityTests(unittest.TestCase):
    def fixture(self, *, count=3, processor=None, interrupted=False):
        state = SimpleNamespace(
            begin=Mock(),
            end=Mock(),
            nextjob=Mock(),
            assign_current_image=Mock(),
            interrupted=interrupted,
            stopping_generation=False,
            skipped=False,
        )
        image = Image.new("RGB", (16, 16))
        frames = [SimpleNamespace(to_image=Mock(return_value=image)) for _ in range(count)]
        stream = SimpleNamespace(frames=count, average_rate=Fraction(30000, 1001))
        container = SimpleNamespace(
            streams=SimpleNamespace(best=Mock(return_value=stream), video=[stream]),
            decode=Mock(return_value=iter(frames)),
            close=Mock(),
        )
        postprocess = Mock(side_effect=processor)
        image_helpers = SimpleNamespace(
            save_video=Mock(),
            fix_image=Mock(side_effect=lambda value: value),
            read=Mock(return_value=image),
            read_info_from_image=Mock(return_value=(None, {})),
            save_image=Mock(),
        )
        opts = SimpleNamespace(outdir_samples="", outdir_extras_samples="outputs")
        namespace = {
            "os": os,
            "np": np,
            "Image": Image,
            "chain": chain,
            "contextmanager": contextlib.contextmanager,
            "devices": SimpleNamespace(torch_gc=Mock()),
            "images": image_helpers,
            "shared": SimpleNamespace(state=state, cmd_opts=SimpleNamespace(hide_ui_dir_config=False)),
            "opts": opts,
            "scripts": SimpleNamespace(
                scripts_postproc=SimpleNamespace(run=postprocess, scripts_in_preferred_order=lambda: [])
            ),
            "scripts_postprocessing": SimpleNamespace(
                PostprocessedImage=lambda image: SimpleNamespace(image=image, info={})
            ),
            "infotext_utils": SimpleNamespace(quote=str),
            "result_html": lambda _: "result",
            "av": SimpleNamespace(open=Mock(return_value=container)),
            "tqdm": lambda values, **kwargs: values,
            "ui_common": SimpleNamespace(plaintext_to_html=lambda value: value),
        }
        source_functions(
            "modules/postprocessing.py",
            {"_extras_job", "run_postprocessing", "run_postprocessing_video"},
            namespace,
        )
        return namespace, state, image_helpers, postprocess, container

    def test_image_processing_exception_always_ends_the_job(self):
        ns, state, _, _, _ = self.fixture(processor=ValueError("processor failed"))
        with self.assertRaisesRegex(ValueError, "processor failed"):
            ns["run_postprocessing"](0, Image.new("RGB", (16, 16)), None, "", "", True, "", save_output=False)
        state.end.assert_called_once_with()
        self.assertEqual(ns["devices"].torch_gc.call_count, 2)

    def test_uploaded_batch_is_not_eagerly_decoded(self):
        ns, _, image_helpers, _, _ = self.fixture(processor=ValueError("stop on first image"))
        batch = [SimpleNamespace(name=f"image-{i}.png") for i in range(5)]
        with self.assertRaises(ValueError):
            ns["run_postprocessing"](1, None, batch, "", "", True, "", save_output=False)
        self.assertEqual(image_helpers.read.call_count, 1)

    def test_video_processing_exception_closes_the_input_and_job(self):
        ns, state, _, _, container = self.fixture(processor=ValueError("processor failed"))
        with self.assertRaisesRegex(ValueError, "processor failed"):
            ns["run_postprocessing_video"](3, None, None, "", "", True, "input.mp4")
        container.close.assert_called_once_with()
        state.end.assert_called_once_with()

    def test_background_removal_video_rejected_before_opening_or_inference(self):
        ns, state, _, process, _ = self.fixture()
        script = SimpleNamespace(name="Background Removal", controls={"enable": None}, args_from=0, args_to=1)
        ns["scripts"].scripts_postproc.scripts_in_preferred_order = lambda: [script]
        with self.assertRaisesRegex(ValueError, "画像タブ"):
            ns["run_postprocessing_video"](3, None, None, "", "", True, "input.mp4", True)
        ns["av"].open.assert_not_called()
        process.assert_not_called()
        state.end.assert_called_once_with()

    def test_background_removal_saves_png_in_single_and_batch_modes(self):
        class Output:
            def __init__(self, image):
                self.image = image.convert("RGBA")
                self.info = {"Background Removal": "BiRefNet HR"}
                self.extra_images = []

            def get_suffix(self, _):
                return ""

        for mode in (0, 1, 2):
            with self.subTest(mode=mode):
                ns, _, helpers, _, _ = self.fixture()
                ns["opts"].samples_format = "jpg"
                ns["opts"].use_original_name_batch = True
                ns["opts"].enable_pnginfo = False
                ns["shared"].listfiles = lambda _: ["input.png"]
                ns["scripts_postprocessing"].PostprocessedImage = Output
                helpers.save_image.return_value = ("output.png", None)
                ns["run_postprocessing"](
                    mode, Image.new("RGB", (16, 16)), [Image.new("RGB", (16, 16))], "in", "", True, ""
                )
                self.assertEqual(helpers.save_image.call_args.kwargs["extension"], "png")
                self.assertEqual(helpers.save_image.call_args.args[0].mode, "RGBA")

    def test_extras_preserves_palette_and_grayscale_transparency_before_processing(self):
        palette = Image.new("P", (16, 16), 0)
        palette.info["transparency"] = 0
        for image in (palette, Image.new("LA", (16, 16), (120, 64))):
            with self.subTest(mode=image.mode):
                ns, _, _, process, _ = self.fixture(processor=ValueError("checked"))
                with self.assertRaisesRegex(ValueError, "checked"):
                    ns["run_postprocessing"](0, image, None, "", "", True, "", save_output=False)
                received = process.call_args.args[0].image
                self.assertEqual(received.mode, "RGBA")
                self.assertEqual(
                    received.getchannel("A").getextrema(), image.convert("RGBA").getchannel("A").getextrema()
                )

    def test_save_output_false_does_not_encode_or_write_a_video(self):
        ns, state, image_helpers, process, _ = self.fixture()
        result = ns["run_postprocessing_video"](3, None, None, "", "", True, "input.mp4", save_output=False)
        image_helpers.save_video.assert_not_called()
        self.assertEqual(process.call_count, 3)
        self.assertEqual(len(result[0]), 1)
        state.end.assert_called_once_with()

    def test_frames_are_streamed_and_fractional_frame_rate_is_preserved(self):
        ns, _, image_helpers, process, _ = self.fixture(count=10)

        def save(_name, frames, **kwargs):
            self.assertEqual(process.call_count, 1)
            self.assertEqual(kwargs["fps"], Fraction(30000, 1001))
            self.assertFalse(isinstance(frames, list))
            self.assertEqual(sum(1 for _ in frames), 10)

        image_helpers.save_video.side_effect = save
        result = ns["run_postprocessing_video"](3, None, None, "", "", True, "input.mp4")
        self.assertEqual(len(result[0]), 1)

    def test_encoder_failure_still_closes_decoder_and_job(self):
        ns, state, image_helpers, _, container = self.fixture()
        image_helpers.save_video.side_effect = VideoEncodingError("failed")
        with self.assertRaises(VideoEncodingError):
            ns["run_postprocessing_video"](3, None, None, "", "", True, "input.mp4")
        container.close.assert_called_once_with()
        state.end.assert_called_once_with()

    def test_cancellation_after_final_frame_aborts_publication(self):
        ns, state, image_helpers, _, container = self.fixture(count=1)

        def save(_name, frames, **kwargs):
            next(frames)
            state.interrupted = True
            with self.assertRaises(VideoEncodingCancelled):
                next(frames)
            raise VideoEncodingCancelled()

        image_helpers.save_video.side_effect = save
        ns["run_postprocessing_video"](3, None, None, "", "", True, "input.mp4")
        state.end.assert_called_once_with()
        container.close.assert_called_once_with()

    def test_missing_stream_closes_decoder_and_job(self):
        ns, state, image_helpers, _, container = self.fixture()
        container.streams.best.return_value = None
        with self.assertRaisesRegex(ValueError, "video stream"):
            ns["run_postprocessing_video"](3, None, None, "", "", True, "input.mp4")
        image_helpers.save_video.assert_not_called()
        state.end.assert_called_once_with()
        container.close.assert_called_once_with()

    def test_empty_video_does_not_reach_encoder(self):
        ns, state, image_helpers, _, container = self.fixture(count=0)
        result = ns["run_postprocessing_video"](3, None, None, "", "", True, "input.mp4")
        image_helpers.save_video.assert_not_called()
        self.assertEqual(result[0], [])
        state.end.assert_called_once_with()
        container.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
