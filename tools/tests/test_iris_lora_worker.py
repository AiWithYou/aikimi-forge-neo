"""Resident LoRA identity changes rebuild from the base, including errors."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules_forge.iris import core, lora, runtime
from tools.tests.test_iris_lora import adapter


class LoRAWorkerTests(unittest.TestCase):
    def test_switch_strength_remove_and_failure_do_not_reuse_stale_runner(self):
        spec = importlib.util.spec_from_file_location("iris_lora_worker", core.ROOT / "tools/iris_worker.py")
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        events = []
        requests = []

        class FakeRunner:
            def __init__(self, *args):
                events.append("load")

            def close(self):
                events.append("close")

            def set_loras(self, loras):
                self.loras = loras

            def run(self, directory, request=None):
                requests.append(request)
                if request.get("fail"):
                    raise ValueError("run failed")
                return self.loras

        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            adapter(path)
            sources = lora.snapshot(path, [{"name": "test.safetensors", "strength": 1}])
            job = path / "job"
            payload = {"root": root, "job_dir": str(job)}
            with (
                patch.object(runtime, "Runner", FakeRunner),
                patch.object(worker, "checkpoint_identity", return_value={"sha256": "base"}),
            ):
                for selection in ([], sources, sources, [{**sources[0], "strength": 0.5}], []):
                    request = {"task": "generate", "precision": "int8", "loras": selection}
                    core.atomic_json(job / "request.json", request)
                    self.assertEqual(worker.resident_run(payload), selection)
                    self.assertEqual(requests[-1], request)
                self.assertEqual(events, ["load", "close", "load", "close", "load", "close", "load"])
                core.atomic_json(job / "request.json", {**request, "fail": True})
                with self.assertRaisesRegex(ValueError, "run failed"):
                    worker.resident_run(payload)
                core.atomic_json(job / "request.json", request)
                self.assertEqual(worker.resident_run(payload), [])
                self.assertEqual(events[-2:], ["close", "load"])

    def test_loading_failures_retry_with_a_fresh_runner(self):
        for failure in ("constructor", "apply", "changed_checkpoint"):
            with self.subTest(failure=failure):
                self.assert_loading_retry(failure)

    def assert_loading_retry(self, failure):
        with tempfile.TemporaryDirectory() as root:
            spec = importlib.util.spec_from_file_location("iris_retry_worker", core.ROOT / "tools/iris_worker.py")
            worker = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(worker)
            events = []
            attempts = []

            class FakeRunner:
                def __init__(self, *args):
                    attempts.append(self)
                    events.append("load")
                    if failure == "constructor" and len(attempts) == 1:
                        raise ValueError("constructor failed")

                def set_loras(self, loras):
                    events.append("apply")
                    if failure == "apply" and len(attempts) == 1:
                        raise ValueError("apply failed")

                def close(self):
                    events.append("close")

                def run(self, directory, request=None):
                    events.append("run")
                    return {"status": "complete"}

            job = Path(root) / "job"
            core.atomic_json(job / "request.json", {"task": "generate", "precision": "int8", "loras": []})
            identities = (
                [{"sha256": "base"}, {"sha256": "changed"}, {"sha256": "base"}, {"sha256": "base"}]
                if failure == "changed_checkpoint"
                else None
            )
            with (
                patch.object(runtime, "Runner", FakeRunner),
                patch.object(worker, "checkpoint_identity", return_value={"sha256": "base"}, side_effect=identities),
            ):
                payload = {"root": root, "job_dir": str(job)}
                with self.assertRaises(ValueError):
                    worker.resident_run(payload)
                self.assertEqual(worker.resident_run(payload), {"status": "complete"})
            self.assertEqual(len(attempts), 2)
            self.assertEqual(events.count("run"), 1)
            self.assertEqual(events.count("close"), 0 if failure == "constructor" else 1)

    def test_changed_lora_is_rejected_before_runner_use(self):
        spec = importlib.util.spec_from_file_location("iris_changed_lora_worker", core.ROOT / "tools/iris_worker.py")
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            adapter(path)
            sources = lora.snapshot(path, [{"name": "test.safetensors", "strength": 1}])
            adapter(path, meta={"rank": "1", "alpha": "4"})
            job = path / "job"
            core.atomic_json(job / "request.json", {"task": "generate", "precision": "int8", "loras": sources})
            with (
                patch.object(worker, "checkpoint_identity", return_value={}),
                patch.object(runtime, "Runner") as constructor,
            ):
                with self.assertRaisesRegex(ValueError, "LoRA"):
                    worker.resident_run({"root": root, "job_dir": str(job)})
                constructor.assert_not_called()

    def test_turbo_manual_settings_are_forwarded_and_invalid_values_rejected_before_runner_creation(self):
        spec = importlib.util.spec_from_file_location(
            "iris_turbo_conditions_worker", core.ROOT / "tools/iris_worker.py"
        )
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            adapter(
                path,
                meta={"rank": "1", "alpha": "1", "extra": json.dumps({"turbo_steps": 4, "shift": 4})},
            )
            with patch("modules_forge.local_assets.HASH_CACHE", path / "hashes.json"):
                sources = lora.snapshot(path, [{"name": "test.safetensors", "strength": 1}])
            job = path / "job"
            payload = {"root": root, "job_dir": str(job)}
            request = {
                "task": "generate",
                "precision": "int8",
                "loras": sources,
                "lora_sampling": "turbo",
                "turbo_shift": 2.5,
                "sampling": {"mode": "turbo", "shift": 2.5},
                "steps": 8,
                "cfg": 2.0,
                "negative_prompt": "blurry",
            }
            with (
                patch.object(worker, "checkpoint_identity", return_value={"sha256": "base"}),
                patch.object(runtime, "Runner") as constructor,
            ):
                invalid_values = (
                    ("steps", 0),
                    ("steps", 201),
                    ("steps", 4.5),
                    ("steps", True),
                    ("cfg", 0),
                    ("cfg", 16),
                    ("cfg", True),
                    ("cfg", float("nan")),
                    ("cfg", float("inf")),
                    ("cfg", "invalid"),
                    ("negative_prompt", None),
                    ("negative_prompt", ["blurry"]),
                )
                for field, value in invalid_values:
                    with self.subTest(field=field):
                        job.mkdir(exist_ok=True)
                        (job / "request.json").write_text(json.dumps({**request, field: value}), encoding="utf-8")
                        with self.assertRaisesRegex(ValueError, "Turbo.*Steps|Turbo.*CFG|Turbo.*Negative"):
                            worker.resident_run(payload)
                        constructor.assert_not_called()
                core.atomic_json(job / "request.json", {**request, "sampling": {**request["sampling"], "shift": 8}})
                with self.assertRaisesRegex(ValueError, "LoRAと生成方式"):
                    worker.resident_run(payload)
                constructor.assert_not_called()
                core.atomic_json(job / "request.json", request)
                constructor.return_value.run.return_value = {"status": "complete"}
                self.assertEqual(worker.resident_run(payload), {"status": "complete"})
                constructor.assert_called_once_with(root, "int8", "generate", "base")
                constructor.return_value.run.assert_called_once_with(job, request=request)


if __name__ == "__main__":
    unittest.main()
