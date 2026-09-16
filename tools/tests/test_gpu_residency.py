"""モデル保持・GPU切替・待機回収と常駐workerの回帰テスト。"""

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from modules_forge import gpu_ownership, gpu_residency
from modules_forge.resident_worker import ResidentWorker
from modules_forge.yue2_studio.core import safe_environment


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.lock = gpu_ownership.GPUQueueLock()
        self.lock._restored = True
        self.enterContext(patch.object(gpu_ownership, "queue_lock", self.lock))
        self.enterContext(patch.object(gpu_ownership, "release_forge_vram"))
        self.enterContext(patch.object(gpu_residency, "_resources", {}))
        self.enterContext(patch.object(gpu_residency, "_active", False))
        self.enterContext(patch.object(gpu_residency, "_idle_since", None))
        self.mode = self.enterContext(patch.object(gpu_residency, "policy", return_value="keep"))
        self.addCleanup(self.clear_timer)

    def clear_timer(self):
        if gpu_residency._timer:
            gpu_residency._timer.cancel()
            gpu_residency._timer = None

    def enter(self, engine):
        owner = gpu_ownership.GPUOwnership()
        owner.engine = engine
        self.assertTrue(owner.acquire())
        return owner

    def test_same_engine_reuses_other_engine_evicts_before_acquire(self):
        release = Mock()
        first = self.enter("yue2")
        gpu_residency.register("yue2", release, "yue2")
        first.release()
        second = self.enter("yue2")
        release.assert_not_called()
        second.release()
        third = self.enter("sensenova")
        release.assert_called_once()
        third.release()

    def test_release_mode_cleans_before_queue_becomes_available(self):
        self.mode.return_value = "release"
        owner = self.enter("yue2")
        observations = []

        def cleanup():
            observations.append(self.lock.acquire(False))

        gpu_residency.register("yue2", cleanup, "yue2")
        owner.release()
        self.assertEqual(observations, [False])
        self.assertEqual(gpu_residency._resources, {})

    def test_manual_release_does_not_interrupt_active_generation(self):
        release = Mock()
        owner = self.enter("yue2")
        gpu_residency.register("yue2", release, "yue2")
        self.assertIn("生成中", gpu_residency.release_idle())
        release.assert_not_called()
        owner.release()
        self.assertIn("解放しました", gpu_residency.release_idle())
        release.assert_called_once()

    def test_idle_timeout_and_keep_mode(self):
        release = Mock()
        owner = self.enter("yue2")
        gpu_residency.register("yue2", release, "yue2")
        owner.release()
        gpu_residency._idle_since = time.monotonic() - 301
        gpu_residency.expire_idle()
        release.assert_not_called()
        self.mode.return_value = "auto"
        gpu_residency.expire_idle()
        release.assert_called_once()

    def test_failed_eviction_blocks_next_engine_and_can_retry(self):
        release = Mock(side_effect=[OSError("stop failed"), None])
        owner = self.enter("yue2")
        gpu_residency.register("yue2", release, "yue2")
        owner.release()
        following = gpu_ownership.GPUOwnership()
        with self.assertRaisesRegex(OSError, "stop failed"):
            following.acquire()
        self.assertIn("yue2", gpu_residency._resources)
        self.assertTrue(following.acquire())
        following.release()


class ResidentProcessTests(unittest.TestCase):
    def test_same_process_two_jobs_and_confirmed_close(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            script = root / "engine.py"
            script.write_text(
                "import json,os\nfrom pathlib import Path\ncount=0\n"
                "def resident_run(payload):\n global count\n count+=1\n"
                " Path(payload['output']).write_text(json.dumps({'pid':os.getpid(),'count':count}))\n",
                encoding="utf-8",
            )
            worker = ResidentWorker("test", root / "sessions")
            try:
                records = []
                for index in range(2):
                    output = root / f"output-{index}.json"
                    reused = worker.start(
                        sys.executable, script, safe_environment(), {"output": str(output)}, root / f"job-{index}.log"
                    )
                    self.assertEqual(reused, index > 0)
                    deadline = time.monotonic() + 15
                    while worker.result() is None and time.monotonic() < deadline:
                        time.sleep(0.05)
                    self.assertTrue(worker.result()["ok"])
                    records.append(json.loads(output.read_text()))
                self.assertEqual(records[0]["pid"], records[1]["pid"])
                self.assertEqual([record["count"] for record in records], [1, 2])
                process = worker.process
                worker.close()

                self.assertIsNotNone(process.poll())
            finally:
                worker.close()


class ModelCacheTests(unittest.TestCase):
    def test_settings_rows_survive_ui_class_reload(self):
        from types import SimpleNamespace

        import gradio as gr

        from modules import shared  # noqa: F401 -- 実アプリと同じ順でOptionsを初期化する。
        from modules.options import OptionRow
        from modules.ui_components import FormRow
        from tools.tests.test_gpu_ownership import load_function

        class ReloadedFormRow(FormRow):
            pass

        namespace = {
            "opts": SimpleNamespace(data={}, data_labels={"begin": OptionRow(), "end": OptionRow()}),
            "gr": gr,
            "OptionRow": OptionRow,
            "FormRow": ReloadedFormRow,
            "CURRENT_ROW": None,
        }
        create = load_function("modules/ui_settings.py", "create_setting_component", namespace)
        with gr.Blocks():
            self.assertIsInstance(create("begin"), gr.State)
            self.assertIsInstance(create("end"), gr.State)
        self.assertIsNone(namespace["CURRENT_ROW"])

    def test_ui_reloads_current_policy_without_saving_on_load(self):
        import gradio as gr

        from modules import ui_gpu_residency as ui

        current = ["auto"]

        def read_policy():
            return current[0]

        with patch.object(ui.gpu_residency, "policy", read_policy):
            with gr.Blocks() as demo:
                ui.create_ui(demo)
            current[0] = "keep"
            loader = next(function for function in demo.fns.values() if function.fn is read_policy)
            self.assertEqual(loader.fn(), "keep")
            events = [event for dependency in demo.config["dependencies"] for _, event in dependency["targets"]]
            self.assertIn("load", events)
            self.assertIn("input", events)
            self.assertNotIn("change", events)

    def test_sensenova_seed_reuses_profile_and_file_changes_reload(self):
        from tools import sensenova_u15_worker as worker

        with tempfile.TemporaryDirectory() as root:
            checkpoint = Path(root) / "weights.safetensors"
            checkpoint.write_bytes(b"fixture")
            payload = {"checkpoint": str(checkpoint), "generation_profile": "quality", "seed": 1}
            runtime = (None, None, object(), 1, 588, {}, 12.0, {})
            with (
                patch.object(worker, "_RESIDENT_MODE", True),
                patch.object(worker, "_RESIDENT_RUNTIME", None),
                patch.object(worker, "_RESIDENT_KEY", None),
                patch.object(worker, "_load_runtime", return_value=runtime) as load,
            ):
                self.assertEqual(worker._runtime_for_request(payload)[6], 12.0)
                self.assertEqual(worker._runtime_for_request({**payload, "seed": 2, "steps": 3})[6], 0.0)
                self.assertEqual(load.call_count, 1)
                worker._runtime_for_request({**payload, "generation_profile": "official_8step"})
                self.assertEqual(load.call_count, 2)
                checkpoint.write_bytes(b"changed fixture")
                worker._runtime_for_request(payload)
                self.assertEqual(load.call_count, 3)

    def test_h3_release_confirms_runtime_exit(self):
        from modules_forge import minimax_h3_bridge as bridge

        process = Mock()
        process.poll.return_value = 0
        with (
            patch.object(bridge, "_MANAGED_PROCESS", process),
            patch.object(bridge, "_loopback_server_process", side_effect=[object(), None]),
            patch.object(bridge, "_stop_managed_runtime") as stop,
        ):
            bridge._release_retained_runtime("http://127.0.0.1:8189")
        stop.assert_called_once()

    def test_h3_external_runtime_is_not_killed(self):
        from modules_forge import minimax_h3_bridge as bridge

        with (
            patch.object(bridge, "_MANAGED_PROCESS", None),
            patch.object(bridge, "_loopback_server_process", return_value=object()),
            patch.object(bridge, "_stop_managed_runtime") as stop,
        ):
            with self.assertRaisesRegex(bridge.H3BridgeError, "外部起動"):
                bridge._release_retained_runtime("http://127.0.0.1:8189")
        stop.assert_not_called()
