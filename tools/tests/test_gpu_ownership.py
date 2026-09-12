"""通常UI生成とAPIモデル退避を、外部jobの所有権で待機させる。"""

import ast
import threading
import unittest
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from modules.fifo_lock import FIFOLock
from modules_forge import gpu_ownership

ROOT = Path(__file__).resolve().parents[2]


def load_function(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    code = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(code, path, "exec"), namespace)  # noqa: S102 -- リポジトリ内の実処理だけをCPUで実行する。
    return namespace[name]


class GPUEntryTests(unittest.TestCase):
    def setUp(self):
        self.lock = FIFOLock()
        self.enterContext(patch.object(gpu_ownership, "queue_lock", self.lock))

    def assert_waits_for_owner(self, callback, entered):
        owner = gpu_ownership.GPUOwnership()
        self.assertTrue(owner.acquire())
        started = threading.Event()
        errors = []

        def run():
            started.set()
            try:
                callback()
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            self.assertTrue(started.wait(2))
            self.assertFalse(entered.wait(0.1))
        finally:
            owner.release()
            thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(entered.is_set())

    def test_unload_api_waits_until_gpu_owner_has_stopped(self):
        entered = threading.Event()
        unload = load_function(
            "modules/api/api.py",
            "unloadapi",
            {
                "sd_models": SimpleNamespace(unload_model_weights=entered.set),
            },
        )
        self.assert_waits_for_owner(lambda: unload(SimpleNamespace(queue_lock=self.lock)), entered)

    def test_ui_generation_uses_the_same_queue_before_model_processing(self):
        entered = threading.Event()
        wrapper = load_function(
            "modules/call_queue.py",
            "wrap_gradio_gpu_call",
            {
                "wraps": wraps,
                "queue_lock": self.lock,
                "shared": SimpleNamespace(state=Mock()),
                "progress": Mock(),
                "wrap_gradio_call": lambda callback, *args, **kwargs: callback,
            },
        )
        self.assert_waits_for_owner(wrapper(entered.set), entered)
