"""CPU model loading must not leak temporary wrappers for a live clone."""

import gc
import unittest
import weakref
from unittest.mock import patch

import torch

from backend import memory_management
from backend.patcher.base import ModelPatcher


class LoadedModelLifetimeTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(memory_management, "current_loaded_models", []))
        self.enterContext(patch("modules_forge.gpu_residency.release_resource"))
        self.parent = ModelPatcher(torch.nn.Linear(2, 2), torch.device("cpu"), torch.device("cpu"))
        self.child = self.parent.clone()

    def test_existing_patcher_does_not_allocate_another_wrapper(self):
        created = []
        original = memory_management.LoadedModel

        class TrackedLoadedModel(original):
            def __init__(self, model):
                super().__init__(model)
                created.append(weakref.ref(self))

        with (
            patch.object(memory_management, "LoadedModel", TrackedLoadedModel),
            patch.object(self.child, "partially_load", return_value=0),
        ):
            for _ in range(4):
                memory_management.load_models_gpu([self.child], force_full_load=True)

        gc.collect()
        self.assertEqual(len(memory_management.current_loaded_models), 1)
        self.assertEqual(sum(item() is not None for item in created), 1)

    def test_clone_wrapper_retains_and_can_detach_parent_finalizer(self):
        loaded = memory_management.LoadedModel(self.child)
        self.assertIsNotNone(loaded._patcher_finalizer)
        self.assertTrue(loaded._patcher_finalizer.alive)
        loaded._patcher_finalizer.detach()
        del self.child
        gc.collect()
        self.assertIsNone(loaded.model)

    def test_releasing_wrapper_does_not_wait_for_live_child(self):
        loaded = memory_management.LoadedModel(self.child)
        reference = weakref.ref(loaded)
        del loaded
        gc.collect()
        self.assertIsNone(reference())

    def test_child_collection_falls_back_to_parent(self):
        loaded = memory_management.LoadedModel(self.child)
        del self.child
        gc.collect()
        self.assertIs(loaded.model, self.parent)


if __name__ == "__main__":
    unittest.main()
