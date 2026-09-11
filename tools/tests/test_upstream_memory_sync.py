"""本家のOOM判定と部分ロード更新をCPUだけで検証する。"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from backend.patcher.base import ModelPatcher

ROOT = Path(__file__).resolve().parents[2]


def load_oom_classifier(torch):
    path = ROOT / "backend/memory_management.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == "is_oom")
        or (isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in {"OOM_EXCEPTION", "ACCELERATOR_ERROR"}
            for target in node.targets
        ))
    ]
    namespace = {"torch": torch, "discard_cuda_async_error": Mock()}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    return namespace


class OomClassificationTests(unittest.TestCase):
    def test_missing_torch_exception_types_do_not_classify_unrelated_errors_as_oom(self):
        namespace = load_oom_classifier(SimpleNamespace())
        for error in (ValueError("invalid shape"), RuntimeError("invalid device"), OSError("read failed")):
            with self.subTest(error=type(error).__name__):
                self.assertFalse(namespace["is_oom"](error))
        namespace["discard_cuda_async_error"].assert_not_called()

    def test_explicit_oom_and_accelerator_types_are_recognized(self):
        class OutOfMemoryError(Exception):
            pass

        class AcceleratorError(Exception):
            pass

        namespace = load_oom_classifier(SimpleNamespace(
            OutOfMemoryError=OutOfMemoryError, AcceleratorError=AcceleratorError,
        ))
        self.assertTrue(namespace["is_oom"](OutOfMemoryError()))
        namespace["discard_cuda_async_error"].assert_not_called()
        self.assertTrue(namespace["is_oom"](AcceleratorError()))
        namespace["discard_cuda_async_error"].assert_called_once_with()
        self.assertFalse(namespace["is_oom"](RuntimeError("invalid shape")))

    def test_oom_message_is_recognized_without_torch_exception_types(self):
        namespace = load_oom_classifier(SimpleNamespace())
        self.assertTrue(namespace["is_oom"](RuntimeError("CUDA OUT OF MEMORY")))
        namespace["discard_cuda_async_error"].assert_called_once_with()


class PartialLoadTests(unittest.TestCase):
    def make_patcher(self, current_uuid="old"):
        patcher = ModelPatcher.__new__(ModelPatcher)
        patcher.model = SimpleNamespace(
            current_weight_patches_uuid=current_uuid,
            model_loaded_weight_memory=100,
            model_offload_buffer_memory=0,
            model_lowvram=False,
            modules=lambda: [],
            to=Mock(),
        )
        patcher.patches_uuid = "new"
        patcher.backup = {}
        patcher.object_patches_backup = {}
        patcher.offload_device = "cpu"
        patcher.unpin_all_weights = Mock()
        patcher.patch_model = Mock()
        patcher.model_size = Mock(return_value=1000)
        patcher.partially_unload = Mock()
        patcher.detach = Mock()
        patcher.load = Mock()
        return patcher

    def test_changed_patches_reuse_residency_and_keep_memory_budget(self):
        patcher = self.make_patcher()

        def load(*args, **kwargs):
            patcher.model.model_loaded_weight_memory = 150

        patcher.load.side_effect = load
        self.assertEqual(patcher.partially_load("cuda", extra_memory=50), 150)
        patcher.model.to.assert_not_called()
        patcher.unpin_all_weights.assert_called_once_with()
        patcher.load.assert_called_once_with(
            "cuda", lowvram_model_memory=150, force_patch_weights=False, full_load=False,
        )

    def test_negative_budget_still_offloads_unchanged_patches(self):
        patcher = self.make_patcher(current_uuid="new")
        self.assertEqual(patcher.partially_load("cuda", extra_memory=-25), 0)
        patcher.partially_unload.assert_called_once_with("cpu", 25, force_patch_weights=False)
        patcher.load.assert_not_called()

    def test_load_failure_still_detaches(self):
        patcher = self.make_patcher()
        patcher.load.side_effect = RuntimeError("load failed")
        with self.assertRaisesRegex(RuntimeError, "load failed"):
            patcher.partially_load("cuda", extra_memory=50)
        patcher.detach.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
