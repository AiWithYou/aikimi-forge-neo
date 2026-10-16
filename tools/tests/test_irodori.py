"""Irodori request boundaries and completed audio contracts, without GPU imports."""

import json
import math
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from modules_forge.irodori.core import (
    CODE_REVISION,
    MODEL_REVISION,
    IrodoriError,
    atomic_json,
    stage_references,
    validate_request,
    verify_result,
)


def wav(path, frames=4800, rate=48000):
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(b"\x10\x00" * frames)


class RequestTests(unittest.TestCase):
    def test_text_and_emoji_survive_and_random_seed_is_recorded(self):
        request = validate_request({"text": "こんにちは🤭"})
        self.assertEqual(request["text"], "こんにちは🤭")
        self.assertIsInstance(request["seed"], int)
        self.assertGreaterEqual(request["seed"], 0)
        self.assertEqual(request["steps"], 40)
        self.assertIsNone(request["seconds"])
        self.assertEqual(request["precision"], "bf16")

    def test_bad_inputs_are_rejected_before_gpu_acquisition(self):
        for values in (
            {"text": " "},
            {"text": "文", "steps": 0},
            {"text": "文", "steps": 2.5},
            {"text": "文", "seconds": 31},
            {"text": "文", "seconds": -1},
            {"text": "文", "cfg_text": math.nan},
            {"text": "文", "cfg_speaker": math.inf},
            {"text": "文", "precision": "fp8"},
            {"text": "文", "seed": -2},
            {"text": "文", "duration_scale": 0},
            {"text": "a" * 2001},
        ):
            with self.subTest(values=values), self.assertRaises(IrodoriError):
                validate_request(values)

    def test_manual_and_auto_duration_are_distinct(self):
        self.assertIsNone(validate_request({"text": "文", "seconds": 0})["seconds"])
        self.assertEqual(validate_request({"text": "文", "seconds": 4})["seconds"], 4)

    def test_references_are_copied_hashed_and_capped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "a.wav"
            wav(source)
            target = root / "job"
            target.mkdir()
            result = stage_references([source], target)
            self.assertEqual(result[0]["seconds"], 0.1)
            self.assertEqual(len(result[0]["sha256"]), 64)
            source.unlink()
            self.assertTrue((target / result[0]["file"]).is_file())
            long = root / "long.wav"
            wav(long, frames=121, rate=1)
            with self.assertRaisesRegex(IrodoriError, "120"):
                stage_references([long], target)

    def test_wrong_audio_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = root / "text.wav"
            bad.write_text("not audio", encoding="utf-8")
            with self.assertRaises(IrodoriError):
                stage_references([bad], root)


class ResultTests(unittest.TestCase):
    def test_success_requires_wav_and_matching_result_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = validate_request({"text": "テスト", "seed": 7})
            atomic_json(root / "request.json", request)
            with self.assertRaises(IrodoriError):
                verify_result(root)
            wav(root / "audio.wav")
            atomic_json(
                root / "result.json",
                {
                    "job_id": root.name,
                    "sample_rate": 48000,
                    "samples": 4800,
                    "seconds": 0.1,
                    "seed": 7,
                    "model_revision": MODEL_REVISION,
                    "code_revision": CODE_REVISION,
                    "watermarked": False,
                },
            )
            self.assertEqual(verify_result(root)["seed"], 7)
            value = json.loads((root / "result.json").read_text(encoding="utf-8"))
            value["seed"] = 8
            atomic_json(root / "result.json", value)
            with self.assertRaises(IrodoriError):
                verify_result(root)

    def test_missing_or_invalid_wave_is_never_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic_json(root / "request.json", validate_request({"text": "文"}))
            atomic_json(root / "result.json", {"sample_rate": 48000})
            wav(root / "audio.wav", frames=0)
            with self.assertRaises(IrodoriError):
                verify_result(root)


class ServiceTests(unittest.TestCase):
    def test_progress_file_cannot_override_server_state(self):
        from modules_forge.irodori.service import Service

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = Service(root=root, outputs=root)
            service.jobs["a"] = service.new_job("owner", root)
            atomic_json(root / "progress.json", {"status": "complete", "directory": "elsewhere"})
            self.assertEqual(service.poll("a", "owner")["status"], "running")

    def test_worker_success_without_audio_fails_and_releases_after_close(self):
        from modules_forge.irodori.service import Service

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = validate_request({"text": "文"})
            atomic_json(root / "request.json", request)
            events = []
            worker = unittest.mock.Mock()
            worker.result.return_value = {"ok": True}
            worker.close.side_effect = lambda: events.append("close")
            lease = unittest.mock.Mock()
            lease.release.side_effect = lambda: events.append("release")
            service = Service(root=root, outputs=root / "outputs", worker=worker)
            job = service.new_job("owner", root)
            with (
                patch("modules_forge.gpu_ownership.release_forge_vram"),
                patch("modules_forge.gpu_residency.register"),
                patch("modules_forge.gpu_residency.release_resource", side_effect=lambda *args: worker.close()),
            ):
                service.execute(job, lease)
            self.assertEqual(job["status"], "error")
            self.assertEqual(events, ["close", "release"])

    def test_failed_cleanup_retains_lease_until_retry_and_cancel_wins(self):
        from modules_forge.irodori.service import Service

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            atomic_json(root / "request.json", validate_request({"text": "文"}))
            events = []
            worker = unittest.mock.Mock()
            service = Service(root=root, outputs=root / "outputs", worker=worker)
            job = service.new_job("owner", root)
            job["cancel"].set()
            lease = unittest.mock.Mock()
            lease.release.side_effect = lambda: events.append("release")

            def cleanup(*args):
                events.append("close")
                if len(events) == 1:
                    raise PermissionError("sharing conflict")
                self.assertFalse(lease.release.called)

            with (
                patch("modules_forge.gpu_ownership.release_forge_vram"),
                patch("modules_forge.gpu_residency.register"),
                patch("modules_forge.gpu_residency.release_resource", side_effect=cleanup),
                patch("modules_forge.irodori.service.time.sleep"),
            ):
                service.execute(job, lease)
            self.assertEqual(job["status"], "cancelled")
            self.assertEqual(events, ["close", "close", "release"])

    def test_other_session_cannot_poll_or_cancel(self):
        from modules_forge.irodori.service import Service

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = Service(root=root, outputs=root)
            service.jobs["a"] = service.new_job("owner", root)
            with self.assertRaises(IrodoriError):
                service.poll("a", "different")
            with self.assertRaises(IrodoriError):
                service.cancel("a", "different")


if __name__ == "__main__":
    unittest.main()
