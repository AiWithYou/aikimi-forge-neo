"""準備から結果受領までの所有権と、実ファイルによる成功条件を検証する。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from modules_forge import sensenova_u15_bridge as bridge

WORKER = """
import hashlib
import json
import sys
from pathlib import Path
from PIL import Image

payload = json.loads(Path(sys.argv[-1]).read_text(encoding="utf-8"))
path = Path(payload["output_path"])
image = Image.new("RGB", (payload["width"], payload["height"]), (80, 120, 160))
image.save(path, format="PNG")
metadata = {
    "schema_version": 3,
    "width": image.width, "height": image.height,
    "mode": payload["mode"], "prompt": payload["prompt"],
    "seed": payload["seed"], "steps": payload["steps"],
    "generation_profile": payload["generation_profile"],
    "input_image_count": len(payload["input_images"]),
    "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
}
{change}
Path(payload["metadata_path"]).write_text(json.dumps(metadata), encoding="utf-8")
"""


class SenseNovaJobLifecycleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.worker = self.root / "worker.py"
        self.write_worker()
        self.request = bridge.SenseNovaRequest(
            mode=bridge.MODE_TEXT,
            prompt="A blue circle",
            generation_profile=bridge.PROFILE_QUALITY,
            width=512,
            height=512,
            steps=1,
        )
        for name, value in (
            ("inspect_runtime", lambda *args, **kwargs: SimpleNamespace(ready=True)),
            ("_release_forge_vram", lambda: None),
            ("WORKER_PYTHON", Path(sys.executable)),
        ):
            self.enterContext(patch.object(bridge, name, value))

    def write_worker(self, change=""):
        self.worker.write_text(WORKER.replace("{change}", change), encoding="utf-8")

    def generation(self):
        updates = bridge.run_generation(
            self.request,
            output_directory=self.root / "outputs",
            cache_directory=self.root / "cache",
            log_directory=self.root / "logs",
            worker_path=self.worker,
        )
        self.addCleanup(updates.close)
        return updates

    def assert_idle(self):
        self.assertIsNone(bridge._ACTIVE_JOB_ID)
        self.assertIsNone(bridge._ACTIVE_PROCESS)
        self.assertEqual(list((self.root / "cache/jobs").iterdir()), [])

    def test_preparing_job_cannot_be_replaced_and_remains_cancellable(self):
        first = self.generation()
        own_id = next(first)["job_id"]
        with self.assertRaises(bridge.SenseNovaBridgeError):
            next(self.generation())
        self.assertEqual(bridge._ACTIVE_JOB_ID, own_id)
        bridge.cancel_generation(own_id)
        with self.assertRaises(bridge.SenseNovaGenerationCancelled):
            next(first)
        self.assert_idle()
        retry = self.generation()
        self.assertEqual(next(retry)["stage"], "prepare")
        retry.close()
        self.assert_idle()

    def test_completed_worker_keeps_ownership_until_result_is_received(self):
        first = self.generation()
        for event in first:
            if event["stage"] == "complete":
                break
        else:
            self.fail("成功結果が返りませんでした")
        with self.assertRaises(bridge.SenseNovaBridgeError):
            next(self.generation())
        self.assertEqual(bridge._ACTIVE_JOB_ID, event["job_id"])
        first.close()
        self.assert_idle()

    def test_cancel_at_spawn_boundary_does_not_start_worker(self):
        updates = self.generation()
        own_id = next(updates)["job_id"]
        self.assertEqual(next(updates)["stage"], "vram")

        def cancel_before_spawn(environment):
            bridge.cancel_generation(own_id)
            return dict(environment)

        with (
            patch.object(bridge, "sanitized_subprocess_environment", side_effect=cancel_before_spawn),
            patch.object(bridge.subprocess, "Popen", wraps=bridge.subprocess.Popen) as spawn,
        ):
            with self.assertRaises(bridge.SenseNovaGenerationCancelled):
                list(updates)
            spawn.assert_not_called()
        self.assert_idle()

    def test_invalid_worker_results_never_emit_complete_and_allow_retry(self):
        cases = {
            "undecodable": 'path.write_bytes(b"not a PNG")',
            "truncated": "path.write_bytes(path.read_bytes()[:80])",
            "wrong_size": 'Image.new("RGB", (256, 512)).save(path)',
            "wrong_hash": 'metadata["output_sha256"] = "0" * 64',
            "wrong_seed": 'metadata["seed"] += 1',
            "wrong_references": 'metadata["input_image_count"] = 1',
            "wrong_prompt": 'metadata["prompt"] = "a different request"',
            "missing_metadata": "metadata = {}",
        }
        for name, change in cases.items():
            with self.subTest(name=name):
                self.write_worker(change)
                stages = []
                with self.assertRaises(bridge.SenseNovaBridgeError):
                    for event in self.generation():
                        stages.append(event["stage"])
                self.assertNotIn("complete", stages)
                self.assert_idle()
        self.write_worker()
        self.assertEqual(list(self.generation())[-1]["stage"], "complete")
        self.assert_idle()


if __name__ == "__main__":
    unittest.main()
