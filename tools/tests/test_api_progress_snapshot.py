"""API progress uses live task IDs and stable CPU state counters."""

import datetime
import threading
import unittest
from collections import Counter, OrderedDict
from types import ModuleType, SimpleNamespace
from typing import Optional
from unittest.mock import Mock

import torch
from fastapi import Depends
from PIL import Image
from pydantic import BaseModel, Field

from tools.tests.test_api_extras_boundaries import load_classes
from tools.tests.test_gpu_ownership import load_function


class ChangingCounters:
    """Advance selected counters between reads without scheduling a thread."""

    def __init__(self, **sequences):
        self.values = {
            "job_count": 1,
            "job_no": 0,
            "sampling_steps": 4,
            "sampling_step": 2,
            "time_start": 100.0,
            "current_image": None,
            "textinfo": "Sampling",
        }
        self.sequences = sequences
        self.reads = Counter()
        self.set_current_image = Mock()

    def __getattr__(self, name):
        if name not in self.values:
            raise AttributeError(name)
        index = self.reads[name]
        self.reads[name] += 1
        sequence = self.sequences.get(name, [self.values[name]])
        return sequence[min(index, len(sequence) - 1)]

    def dict(self):
        return self.values.copy()


class ApiProgressTests(unittest.TestCase):
    def setUp(self):
        self.models = load_classes(
            "modules/api/models.py", {"ProgressRequest", "ProgressResponse"}, {"BaseModel": BaseModel, "Field": Field}
        )
        self.shared = SimpleNamespace(opts=SimpleNamespace(live_previews_enable=False))
        state_models = load_classes(
            "modules/shared_state.py",
            {"State"},
            {
                "datetime": datetime,
                "threading": threading,
                "time": SimpleNamespace(time=lambda: 100.0),
                "Optional": Optional,
                "torch": torch,
                "Image": Image,
                "stream": SimpleNamespace(should_use_stream=lambda: False),
                "devices": Mock(),
                "log": Mock(),
                "shared": self.shared,
            },
        )
        self.state = state_models.State()
        self.state.begin("CPU fixture")
        self.state.job_count = 2
        self.state.job_no = 1
        self.state.sampling_steps = 4
        self.state.sampling_step = 2
        self.state.textinfo = "Sampling"
        self.shared.state = self.state

        # Execute the production task lifecycle with isolated globals.
        self.progress = ModuleType("_api_progress_fixture")
        self.progress.current_task = None
        self.progress.pending_tasks = OrderedDict()
        self.progress.finished_tasks = []
        for name in ("start_task", "finish_task"):
            load_function("modules/progress.py", name, vars(self.progress))

        namespace = {
            "models": self.models,
            "Depends": Depends,
            "shared": self.shared,
            "time": SimpleNamespace(time=lambda: 120.0),
            "progress_module": self.progress,
            # Match the old import-time value so this fixture catches rebinding.
            "current_task": self.progress.current_task,
            "encode_pil_to_base64": Mock(side_effect=AssertionError("Preview encoding was not requested")),
        }
        self.callback = load_function("modules/api/api.py", "progressapi", namespace)

    def request(self):
        return self.callback(SimpleNamespace(), self.models.ProgressRequest(skip_current_image=True))

    def test_real_task_start_and_switch_are_visible_in_response_model(self):
        for task in ("task(first)", "task(second)"):
            self.progress.start_task(task)
            response = self.request()
            self.assertEqual(response.model_dump().get("current_task"), task)
            self.assertIsInstance(response, self.models.ProgressResponse)

    def test_finished_idle_task_is_explicitly_none(self):
        self.progress.start_task("task(finished)")
        self.progress.finish_task("task(finished)")
        self.state.end()
        response = self.request()
        self.assertIn("current_task", response.model_dump())
        self.assertIsNone(response.current_task)
        self.assertEqual(response.progress, 0)
        self.assertEqual(response.eta_relative, 0)

    def test_job_finishing_between_counter_reads_does_not_divide_by_zero(self):
        self.shared.state = ChangingCounters(job_count=[1, 0])
        response = self.request()
        self.assertAlmostEqual(response.progress, 0.51)
        self.assertGreaterEqual(response.eta_relative, 0)
        self.assertEqual(self.shared.state.reads["job_count"], 1)

    def test_sampling_restart_between_counter_reads_does_not_divide_by_zero(self):
        self.shared.state = ChangingCounters(sampling_steps=[4, 0])
        response = self.request()
        self.assertAlmostEqual(response.progress, 0.51)
        for field in ("job_count", "job_no", "sampling_steps", "sampling_step", "time_start"):
            self.assertEqual(self.shared.state.reads[field], 1, field)

    def test_unknown_job_count_does_not_subtract_sampling_progress(self):
        # begin() leaves the count unknown; a later sampling step can be read
        # during the same polling request after the next job starts.
        self.state.begin("Preparing")
        self.assertEqual(self.state.job_count, -1)
        self.shared.state = ChangingCounters(job_count=[self.state.job_count], sampling_step=[2])
        response = self.request()
        self.assertEqual(response.progress, 0.01)
        self.assertGreaterEqual(response.eta_relative, 0)

    def test_complete_progress_has_zero_eta(self):
        self.state.job_no = self.state.job_count
        self.state.sampling_step = 0
        response = self.request()
        self.assertEqual(response.progress, 1)
        self.assertEqual(response.eta_relative, 0)

    def test_stable_running_state_keeps_existing_progress_and_eta(self):
        response = self.request()
        self.assertAlmostEqual(response.progress, 0.76)
        self.assertAlmostEqual(response.eta_relative, 20 / 0.76 - 20)
        self.assertEqual(response.textinfo, "Sampling")
        self.assertEqual(response.state["job_count"], 2)


if __name__ == "__main__":
    unittest.main()
