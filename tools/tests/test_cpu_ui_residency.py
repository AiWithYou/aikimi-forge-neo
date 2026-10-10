"""CPU UI callbacks serialize safely without evicting idle GPU resources."""

import html
import time
import traceback
import unittest
from functools import wraps
from types import SimpleNamespace
from unittest.mock import Mock, patch

from modules_forge import gpu_ownership, gpu_residency
from tools.tests.test_gpu_ownership import load_function


class CpuUiResidencyTests(unittest.TestCase):
    def test_token_callback_preserves_resident_worker_and_idle_deadline(self):
        lock = gpu_ownership.GPUQueueLock()
        lock._restored = True
        cleanup = Mock()
        with (
            patch.object(gpu_residency, "_resources", {"qwen": ("qwen", cleanup)}),
            patch.object(gpu_residency, "policy", return_value="keep"),
            patch.object(gpu_residency, "_idle_since", 123.0),
            patch.object(gpu_residency, "_active", False),
        ):
            wrapper = load_function(
                "modules/call_queue.py",
                "wrap_queued_call",
                {"queue_lock": lock, "engine_scope": gpu_residency.engine_scope},
            )

            def tokenize():
                self.assertFalse(lock.acquire(False))
                return 7

            self.assertEqual(wrapper(tokenize)(), 7)
            cleanup.assert_not_called()
            self.assertIn("qwen", gpu_residency._resources)
            self.assertEqual(gpu_residency._idle_since, 123.0)
            self.assertFalse(gpu_residency._active)
            with gpu_residency.engine_scope(None):
                self.assertTrue(lock.acquire(False))
                lock.release()

    def test_save_and_pnginfo_wrappers_never_collect_gpu_cache(self):
        devices = SimpleNamespace(torch_gc=Mock())
        wrapper = load_function(
            "modules/call_queue.py",
            "wrap_gradio_call_no_job",
            {
                "wraps": wraps,
                "time": time,
                "html": html,
                "traceback": traceback,
                "devices": devices,
                "main_thread": SimpleNamespace(last_exception=None),
                "shared": SimpleNamespace(opts=SimpleNamespace(memmon_poll_rate=0)),
            },
        )
        self.assertEqual(wrapper(lambda: ["saved", "info"])(), ("saved", "info"))
        with patch.object(traceback, "print_exc"):
            result = wrapper(Mock(side_effect=ValueError("invalid PNG")))()
        self.assertIn("invalid PNG", result[-1])
        devices.torch_gc.assert_not_called()


if __name__ == "__main__":
    unittest.main()
