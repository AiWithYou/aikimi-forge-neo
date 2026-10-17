from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from backend import memory_management


class OomCleanupTests(unittest.TestCase):
    def test_cpu_classification_does_not_touch_cuda(self):
        with (patch.object(memory_management, "get_torch_device", return_value=torch.device("cpu")),
              patch.object(torch.cuda, "synchronize", side_effect=AssertionError("CUDA accessed")) as sync):
            self.assertTrue(memory_management.is_oom(RuntimeError("CPU out of memory")))
            sync.assert_not_called()

    def test_cleanup_runtime_failure_does_not_replace_original_error(self):
        with (patch.object(memory_management, "get_torch_device", return_value=torch.device("cuda:0")),
              patch.object(torch, "tensor", side_effect=RuntimeError("failed to clear device error"))):
            self.assertTrue(memory_management.is_oom(RuntimeError("CUDA out of memory")))

    def test_successful_cleanup_synchronizes_the_selected_device(self):
        device = torch.device("cuda:1")
        tensor = torch.ones(1, dtype=torch.uint8)
        with (patch.object(memory_management, "get_torch_device", return_value=device),
              patch.object(torch, "tensor", return_value=tensor),
              patch.object(torch.cuda, "synchronize") as sync):
            memory_management.discard_cuda_async_error()
            sync.assert_called_once_with(device)


if __name__ == "__main__":
    unittest.main()
