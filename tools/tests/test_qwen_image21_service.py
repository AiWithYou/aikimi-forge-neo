"""Qwen Studio validation, process/GPU lifetime, and real Gradio event contracts."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import gradio as gr
from PIL import Image

from modules import gradio_compat
from modules_forge.qwen_image21 import core, service

ROOT = Path(__file__).resolve().parents[2]


def installed_runtime(root):
    python = root / "worker-env" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python.parent.mkdir(parents=True)
    python.write_bytes(b"test interpreter")
    model = root / "model"
    model.mkdir()
    core.atomic_json(model / "model_index.json", {"_class_name": "QwenImage21Pipeline"})
    files = []
    for component in ("transformer", "text_encoder", "vae"):
        relative = f"{component}/weights.safetensors"
        path = model / relative
        path.parent.mkdir()
        path.write_bytes(b"test weights")
        files.append({"path": relative, "size": path.stat().st_size, "sha256": "test"})
    core.atomic_json(root / "model-files.json", {"revision": core.MODEL_REVISION, "files": files})
    core.atomic_json(
        root / "runtime.json",
        {
            "schema": 1,
            "python": str(python),
            "model": str(model),
            "model_revision": core.MODEL_REVISION,
            "diffusers_revision": core.DIFFUSERS_REVISION,
        },
    )


class Lease:
    def __init__(self):
        self.engine = None
        self.owned = False
        self.releases = 0

    def acquire(self, blocking=False):
        self.owned = True
        return True

    def release(self):
        self.owned = False
        self.releases += 1


class Residency:
    def __init__(self):
        self.resources = {}

    def register(self, name, cleanup, engine):
        self.resources[name] = cleanup

    def release_resource(self, name):
        if name in self.resources:
            self.resources[name]()
            self.resources.pop(name)


class FakeResident:
    def __init__(self, mode="complete"):
        self.mode = mode
        self.process = SimpleNamespace(pid=42, poll=lambda: self.returncode)
        self.returncode = None
        self.starts = 0
        self.polls = 0
        self.closed = threading.Event()
        self.started = threading.Event()
        self.close_attempt = threading.Event()
        self.allow_close = threading.Event()
        self.allow_close.set()

    def start(self, python, script, environment, payload, log_path):
        self.starts += 1
        self.returncode = None
        self.payload = payload
        self.started.set()
        if self.mode == "start-failure":
            raise OSError("injected startup failure after process creation")
        return self.starts > 1

    def result(self):
        self.polls += 1
        if self.mode == "wait":
            return None
        if self.mode == "error":
            return {"ok": False, "error": "injected inference error"}
        directory = Path(self.payload["job_dir"])
        request = core.read_json(directory / "request.json")
        output = directory / "output.png"
        Image.new("RGBA", (request["width"], request["height"]), (30, 60, 90, 80)).save(output)
        core.atomic_json(directory / "result.json", {"output_path": str(output), "metadata": {"seed": request["seed"]}})
        return {"ok": True}

    def close(self):
        self.close_attempt.set()
        if not self.allow_close.is_set():
            raise OSError("injected stop failure")
        self.returncode = 0
        self.closed.set()


class QwenCoreTests(unittest.TestCase):
    def test_rgba_reference_snapshots_preserve_pixels_and_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "first.png", root / "second.png"
            Image.new("RGBA", (8, 6), (12, 34, 56, 78)).save(first)
            Image.new("RGB", (6, 8), (56, 34, 12)).save(second)
            output = root / "job"
            output.mkdir()
            paths = core.copy_inputs([second, first], output)
            self.assertEqual([Path(path).name for path in paths], ["reference-01.png", "reference-02.png"])
            with Image.open(paths[1]) as image:
                self.assertEqual(image.mode, "RGBA")
                self.assertEqual(image.getpixel((0, 0)), (12, 34, 56, 78))
            first.unlink()
            self.assertTrue(Path(paths[1]).is_file())

    def test_requests_reject_invalid_numbers_and_allow_all_official_sizes(self):
        for fields in (
            {"seed": float("nan")},
            {"steps": True},
            {"width": 513},
            {"transparent": "yes"},
            {"precision": "fp8"},
        ):
            with self.subTest(fields=fields), self.assertRaises(core.QwenImage21Error):
                core.Request("test", **fields).resolved()
        for width, height in (
            (2048, 2048),
            (2400, 1792),
            (1792, 2400),
            (2528, 1696),
            (1696, 2528),
            (2752, 1536),
            (1536, 2752),
        ):
            self.assertEqual(core.Request("test", width=width, height=height, seed=7).resolved().seed, 7)
        self.assertGreaterEqual(core.Request("test").resolved().seed, 0)

    def test_runtime_checks_pinned_versions_and_weight_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            installed_runtime(root)
            self.assertEqual(core.runtime_manifest(root)["model_revision"], core.MODEL_REVISION)
            weight = root / "model/transformer/weights.safetensors"
            weight.unlink()
            with self.assertRaisesRegex(core.QwenImage21Error, "不足"):
                core.runtime_manifest(root)
            weight.write_bytes(b"test weights")
            data = core.read_json(root / "runtime.json")
            data["diffusers_revision"] = "wrong"
            core.atomic_json(root / "runtime.json", data)
            with self.assertRaisesRegex(core.QwenImage21Error, "バージョン"):
                core.runtime_manifest(root)

    def test_image_limit_and_inventory_escape_rejected(self):
        with self.assertRaisesRegex(core.QwenImage21Error, "10"):
            core.validate_images(["unused"] * 11)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            installed_runtime(root)
            data = core.read_json(root / "model-files.json")
            data["files"][0]["path"] = "../../outside.safetensors"
            core.atomic_json(root / "model-files.json", data)
            with self.assertRaisesRegex(core.QwenImage21Error, "外側"):
                core.runtime_manifest(root)


class QwenServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runtime = self.root / "runtime"
        installed_runtime(self.runtime)
        self.lease = Lease()
        self.residency = Residency()
        self.worker = FakeResident()
        self.studio = service.Studio(
            self.runtime,
            self.root / "outputs",
            ownership_factory=lambda: self.lease,
            release_vram=lambda: None,
            worker_factory=lambda *_: self.worker,
            residency=self.residency,
            poll_interval=0.01,
            cancel_grace=0.02,
        )
        self.addCleanup(self.studio.shutdown)

    def request(self, **kwargs):
        return core.Request("test", width=256, height=256, seed=123, **kwargs)

    def wait_done(self, identifier, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.studio.status(identifier, "owner")
            if state["done"]:
                return state
            time.sleep(0.01)
        self.fail(f"job did not finish: {self.studio.status(identifier, 'owner')}")

    def test_success_reuses_worker_and_locks_runtime_until_eviction(self):
        first = self.studio.start(self.request(), "owner")
        self.assertEqual(self.wait_done(first)["state"], "complete")
        self.assertFalse(self.lease.owned)
        self.assertFalse(self.worker.closed.is_set())
        self.assertEqual(self.studio.artifact(first, "owner").name, "output.png")
        with self.assertRaises(core.QwenImage21Error):
            core.runtime_lock(self.runtime)
        second = self.studio.start(self.request(precision="bf16"), "owner")
        self.assertEqual(self.wait_done(second)["state"], "complete")
        self.assertEqual(self.worker.starts, 2)
        self.assertEqual(self.worker.payload["precision"], "bf16")
        self.assertTrue(self.studio.status(first, "owner")["done"])
        self.residency.release_resource(service.ENGINE)
        self.assertTrue(self.worker.closed.is_set())
        core.runtime_lock(self.runtime).close()

    def test_validation_happens_before_gpu_acquisition(self):
        with self.assertRaises(core.QwenImage21Error):
            self.studio.start(self.request(input_images=(str(self.root / "missing.png"),)), "owner")
        self.assertFalse(self.lease.owned)
        self.assertEqual(self.lease.releases, 0)
        self.assertFalse(self.worker.started.is_set())

    def test_failed_input_snapshot_and_lock_close_still_return_unused_gpu(self):
        lock = Mock()
        lock.close.side_effect = OSError("injected lock close failure")
        with (
            patch.object(service, "runtime_lock", return_value=lock),
            patch.object(service, "copy_inputs", side_effect=OSError("injected disk failure")),
        ):
            with self.assertRaises(OSError):
                self.studio.start(self.request(), "owner")
        self.assertFalse(self.lease.owned)
        self.assertEqual(self.lease.releases, 1)
        self.assertFalse(self.worker.started.is_set())
        lock.close.side_effect = None

    def test_cancel_owner_and_confirmed_exit_before_gpu_release(self):
        self.worker.mode = "wait"
        self.worker.allow_close.clear()
        identifier = self.studio.start(self.request(), "owner")
        self.assertTrue(self.worker.started.wait(2))
        self.assertFalse(self.studio.cancel(identifier, "other"))
        with self.assertRaises(service.JobNotFound):
            self.studio.status(identifier, "other")
        self.assertTrue(self.studio.cancel(identifier, "owner"))
        try:
            self.assertTrue(self.worker.close_attempt.wait(2))
            self.assertTrue(self.lease.owned)
            self.assertFalse(self.studio.status(identifier, "owner")["done"])
            with self.assertRaises(core.QwenImage21Error):
                self.studio.start(self.request(), "owner")
        finally:
            self.worker.allow_close.set()
        self.assertEqual(self.wait_done(identifier)["state"], "cancelled")
        self.assertTrue(self.worker.closed.is_set())
        self.assertFalse(self.lease.owned)
        self.assertFalse(self.studio.cancel(identifier, "owner"))
        self.assertLess(self.worker.polls, 30)
        core.runtime_lock(self.runtime).close()

    def test_start_failure_after_process_creation_keeps_gpu_until_exit(self):
        self.worker.mode = "start-failure"
        self.worker.allow_close.clear()
        identifier = self.studio.start(self.request(), "owner")
        try:
            self.assertTrue(self.worker.close_attempt.wait(2))
            self.assertTrue(self.lease.owned)
            self.assertFalse(self.studio.status(identifier, "owner")["done"])
        finally:
            self.worker.allow_close.set()
        state = self.wait_done(identifier)
        self.assertEqual(state["state"], "failed")
        self.assertIn("startup failure", state["message"])
        self.assertTrue(self.worker.closed.is_set())
        self.assertFalse(self.lease.owned)

    def test_failed_inference_closes_worker_and_releases_runtime(self):
        self.worker.mode = "error"
        identifier = self.studio.start(self.request(), "owner")
        self.assertEqual(self.wait_done(identifier)["state"], "failed")
        self.assertTrue(self.worker.closed.is_set())
        self.assertFalse(self.lease.owned)
        core.runtime_lock(self.runtime).close()

    def test_cancel_accepted_during_output_validation_is_honored(self):
        original_open = Image.open

        def cancel_on_validation(path, *args, **kwargs):
            if Path(path).name == "output.png":
                self.assertTrue(self.studio.cancel(Path(path).parent.name, "owner"))
            return original_open(path, *args, **kwargs)

        with patch.object(service.Image, "open", side_effect=cancel_on_validation):
            identifier = self.studio.start(self.request(), "owner")
            self.assertEqual(self.wait_done(identifier)["state"], "cancelled")
        self.assertTrue(self.worker.closed.is_set())
        self.assertNotIn(service.ENGINE, self.residency.resources)

    def test_poll_cannot_report_running_and_done_from_different_snapshots(self):
        job = service.Job("job", "owner", self.root)
        job.final = {"state": "complete", "message": "complete"}
        job.done = Mock()
        job.done.is_set.side_effect = [False, True, True]
        self.studio._jobs["job"] = job
        state = self.studio.status("job", "owner")
        self.assertEqual((state["state"], state["done"]), ("running", False))
        job.done.is_set.assert_called_once()
        self.studio._jobs.clear()

    def test_real_resident_process_completes_without_browser_poll_and_reuses(self):
        from modules_forge.resident_worker import ResidentWorker

        engine = self.root / "dummy_engine.py"
        engine.write_text(
            "import json\nfrom pathlib import Path\nfrom PIL import Image\ncount=0\n"
            "def resident_run(payload):\n"
            " global count\n count+=1\n d=Path(payload['job_dir'])\n"
            " r=json.loads((d/'request.json').read_text())\n"
            " Image.new('RGBA',(r['width'],r['height']),(10,20,30,50)).save(d/'output.png')\n"
            " (d/'result.json').write_text(json.dumps({'output_path':str(d/'output.png'),'count':count}))\n",
            encoding="utf-8",
        )

        class LocalResident(ResidentWorker):
            def start(self, python, script, environment, payload, log_path):
                return super().start(sys.executable, engine, environment, payload, log_path)

        self.studio._worker_factory = LocalResident
        first = self.studio.start(self.request(), "owner")
        self.assertTrue(self.studio._jobs[first].done.wait(15))
        self.assertEqual(self.studio.status(first, "owner")["state"], "complete")
        process = self.studio._resident.process
        second = self.studio.start(self.request(), "owner")
        self.assertEqual(self.wait_done(second)["state"], "complete")
        self.assertEqual(self.studio._resident.process.pid, process.pid)
        second_path = self.studio.artifact(second, "owner")
        self.assertEqual(core.read_json(second_path.with_name("result.json"))["count"], 2)
        self.residency.release_resource(service.ENGINE)
        self.assertIsNotNone(process.poll())


def load_ui():
    package = ModuleType("modules")
    callbacks = ModuleType("modules.script_callbacks")
    callbacks.on_ui_tabs = Mock()
    paths = ModuleType("modules.paths")
    paths.data_path, paths.script_path = str(ROOT), str(ROOT)
    package.script_callbacks = callbacks
    package.gradio_compat = gradio_compat
    path = ROOT / "extensions-builtin/qwen-image21-studio/scripts/qwen_image21_studio.py"
    spec = importlib.util.spec_from_file_location("_test_qwen_image21_ui", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"modules": package, "modules.script_callbacks": callbacks, "modules.paths": paths}):
        spec.loader.exec_module(module)
    return module


class QwenUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ui = load_ui()

    def test_real_ui_builds_and_all_callbacks_are_private(self):
        tab, label, identifier = self.ui.on_ui_tabs()[0]
        self.assertEqual((label, identifier), ("Qwen Image 2.1", "qwen_image21_studio"))
        config = tab.get_config_file()
        for function in tab.fns.values():
            self.assertEqual(function.api_visibility, "private")
        props = {item["props"].get("elem_id"): item["props"] for item in config["components"]}
        self.assertEqual(props["qwen21-references"]["type"], "filepath")
        self.assertEqual(props["qwen21-references"]["sources"], ["upload", "clipboard"])
        self.assertFalse(props["qwen21-reference-controls"]["visible"])
        output = next(component for component in tab.blocks.values() if isinstance(component, gr.Image))
        self.assertIsNone(output.image_mode)
        self.assertFalse(props["qwen21-stop"]["interactive"])
        self.assertEqual(gr.utils.get_type_hints(self.ui.start)["request"], gr.Request)
        files = next(component for component in tab.blocks.values() if isinstance(component, gr.File))
        self.assertEqual(files.visible, "hidden")

    def test_reorder_remove_and_owner_bound_result_reuse(self):
        gallery = [("first.png", None), ("second.png", None)]
        self.assertFalse(self.ui.reference_controls_visibility([])["visible"])
        self.assertTrue(self.ui.reference_controls_visibility(gallery)["visible"])
        update, selected = self.ui.move_reference(gallery, 1, -1)
        self.assertEqual(self.ui.reference_paths(update["value"]), ["second.png", "first.png"])
        self.assertEqual(selected, 0)
        update, selected, controls = self.ui.remove_reference(update["value"], selected)
        self.assertTrue(controls["visible"])
        self.assertEqual(self.ui.reference_paths(update["value"]), ["first.png"])
        request = SimpleNamespace(session_hash="session", username="user")
        with patch.object(self.ui.STUDIO, "artifact", return_value=Path("output.png")) as artifact:
            update, selected, controls = self.ui.use_result("job-id", update["value"], request)
        self.assertTrue(controls["visible"])
        artifact.assert_called_once_with("job-id", "user:session")
        self.assertEqual(self.ui.reference_paths(update["value"]), ["first.png", "output.png"])
        self.assertEqual(selected, 1)
        _, _, controls = self.ui.remove_reference([("only.png", None)], 0)
        self.assertFalse(controls["visible"])

    def test_invalid_start_does_not_reset_another_running_job(self):
        request = SimpleNamespace(session_hash="session", username=None)
        with patch.object(self.ui.STUDIO, "start", side_effect=core.QwenImage21Error("実行中")):
            result = self.ui.start("test", [], "1024x1024", False, "int8", "offload", "-1", 40, request)
        self.assertEqual(result[0], {"__type__": "update"})
        self.assertIn("実行中", result[1])
        self.assertNotIn("interactive", result[2])

    def test_completed_poll_reveals_both_downloads_and_stops_timer(self):
        request = SimpleNamespace(session_hash="session", username=None)
        output = Path("completed-job/output.png").resolve()
        state = {"done": True, "state": "complete", "message": "完了", "elapsed": 2.5}
        with (
            patch.object(self.ui.STUDIO, "status", return_value=state),
            patch.object(self.ui.STUDIO, "artifact", return_value=output),
        ):
            result = self.ui.poll("completed-job", request)
        self.assertEqual(result[4], str(output))
        self.assertEqual(
            result[5],
            {"__type__": "update", "visible": True, "value": [str(output), str(output.with_name("result.json"))]},
        )
        self.assertFalse(result[3]["active"])
        self.assertTrue(result[6]["interactive"])

    def test_new_generation_hides_downloads_without_unmounting(self):
        request = SimpleNamespace(session_hash="session", username=None)
        with patch.object(self.ui.STUDIO, "start", return_value="new-job"):
            result = self.ui.start("test", [], "1024x1024", False, "int8", "offload", "-1", 40, request)
        self.assertEqual(result[6], {"__type__": "update", "value": None, "visible": "hidden"})
        self.assertTrue(result[4]["active"])


if __name__ == "__main__":
    unittest.main()
