"""Cancellation and successful completion have one commit boundary."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from modules_forge import gpu_residency
from modules_forge import sensenova_u15_bridge as bridge


class CompletedResident:
    def __init__(self):
        self.process = Mock()
        self.process.poll.return_value = None
        self.process.stdout = io.StringIO()
        self.closes = 0

    def start(self, python, script, environment, payload, log_path):
        log_path.write_text("", encoding="utf-8")
        path = Path(payload["output_path"])
        Image.new("RGB", (payload["width"], payload["height"]), (10, 20, 30)).save(path)
        metadata = {
            "schema_version": 3,
            "mode": payload["mode"],
            "prompt": payload["prompt"],
            "seed": payload["seed"],
            "steps": payload["steps"],
            "generation_profile": payload["generation_profile"],
            "input_image_count": len(payload["input_images"]),
            "width": payload["width"],
            "height": payload["height"],
            "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        Path(payload["metadata_path"]).write_text(json.dumps(metadata), encoding="utf-8")
        return False

    def result(self):
        return {"ok": True}

    def close(self):
        self.closes += 1
        self.process.poll.return_value = -15


class SenseNovaCompletionCancelTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.script = self.root / "worker.py"
        self.script.touch()
        self.resident = CompletedResident()
        self.addCleanup(self.resident.process.stdout.close)
        self.ownership = Mock()
        self.ownership.acquire.return_value = True
        self.register = self.enterContext(patch.object(gpu_residency, "register"))
        for name, value in (
            ("_ACTIVE_JOB_ID", None),
            ("_ACTIVE_JOB_COMMITTED", False),
            ("_ACTIVE_PROCESS", None),
            ("_CANCELLED_JOB_IDS", set()),
            ("_PENDING_CLEANUP", None),
            ("_GPU_OWNERSHIP", None),
            ("_RESIDENT_WORKER", self.resident),
        ):
            self.enterContext(patch.object(bridge, name, value, create=True))
        self.enterContext(patch.object(bridge, "GPUOwnership", return_value=self.ownership))
        self.enterContext(patch.object(bridge, "inspect_runtime", return_value=SimpleNamespace(ready=True)))
        self.enterContext(patch.object(bridge, "_release_forge_vram"))
        self.request = bridge.SenseNovaRequest(
            mode=bridge.MODE_TEXT,
            prompt="検証中の取消",
            generation_profile=bridge.PROFILE_QUALITY,
            width=512,
            height=512,
            steps=1,
            seed=42,
        )

    def generation(self):
        updates = bridge.run_generation(
            self.request,
            output_directory=self.root / "outputs",
            cache_directory=self.root / "cache",
            log_directory=self.root / "logs",
            worker_path=self.script,
        )
        self.addCleanup(updates.close)
        return updates

    def assert_idle(self):
        self.assertIsNone(bridge._ACTIVE_JOB_ID)
        self.assertIsNone(bridge._ACTIVE_PROCESS)
        self.assertFalse(bridge._ACTIVE_JOB_COMMITTED)
        self.assertEqual(bridge._CANCELLED_JOB_IDS, set())
        self.assertEqual(list((self.root / "cache/jobs").iterdir()), [])

    def test_cancel_accepted_during_result_validation_prevents_completion(self):
        validate = bridge._validate_worker_result

        def cancel_after_validation(*args):
            metadata = validate(*args)
            self.assertIn("受け付け", bridge.cancel_generation(bridge._ACTIVE_JOB_ID))
            return metadata

        stages = []
        with (
            patch.object(bridge, "_validate_worker_result", side_effect=cancel_after_validation),
            self.assertRaises(bridge.SenseNovaGenerationCancelled),
        ):
            for event in self.generation():
                stages.append(event["stage"])
        self.assertNotIn("complete", stages)
        self.register.assert_not_called()
        self.assertGreaterEqual(self.resident.closes, 1)
        self.ownership.release.assert_called_once()
        self.assert_idle()

    def test_cancel_after_completion_preserves_idle_worker_and_next_job_can_cancel(self):
        updates = self.generation()
        for event in updates:
            if event["stage"] == "complete":
                break
        else:
            self.fail("valid result never completed")
        self.assertEqual(bridge._ACTIVE_JOB_ID, event["job_id"])
        self.assertIn("終了", bridge.cancel_generation(event["job_id"]))
        self.assertEqual(self.resident.closes, 0)
        self.assertNotIn(event["job_id"], bridge._CANCELLED_JOB_IDS)
        self.ownership.release.assert_not_called()
        updates.close()
        self.assert_idle()

        retry = self.generation()
        retry_id = next(retry)["job_id"]
        self.assertIn("受け付け", bridge.cancel_generation(retry_id))
        with self.assertRaises(bridge.SenseNovaGenerationCancelled):
            next(retry)
        self.assert_idle()


if __name__ == "__main__":
    unittest.main()
