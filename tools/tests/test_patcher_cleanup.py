import unittest
from types import SimpleNamespace
from unittest import mock

from backend.patcher.base import ModelPatcher


class PatcherCleanupTests(unittest.TestCase):
    def test_full_load_unpatch_removes_online_weight_wrappers(self):
        module = SimpleNamespace(
            parameters_manual_cast=True,
            prev_parameters_manual_cast=False,
            weight_function=[object()],
            bias_function=[object()],
            forge_patched_weights=True,
        )
        model = SimpleNamespace(
            model_lowvram=False,
            lowvram_patch_counter=0,
            current_weight_patches_uuid="patched",
            model_loaded_weight_memory=123,
            model_offload_buffer_memory=456,
            modules=lambda: [module],
        )
        patcher = ModelPatcher.__new__(ModelPatcher)
        patcher.model = model
        patcher.backup = {}
        patcher.object_patches_backup = {}
        patcher.unpin_all_weights = mock.Mock()
        patcher.detach = mock.Mock()

        patcher.unpatch_model()

        patcher.unpin_all_weights.assert_called_once_with()
        self.assertFalse(module.parameters_manual_cast)
        self.assertFalse(hasattr(module, "prev_parameters_manual_cast"))
        self.assertEqual(module.weight_function, [])
        self.assertEqual(module.bias_function, [])
        self.assertFalse(hasattr(module, "forge_patched_weights"))
        self.assertIsNone(model.current_weight_patches_uuid)
        self.assertEqual(model.model_loaded_weight_memory, 0)
        self.assertEqual(model.model_offload_buffer_memory, 0)


if __name__ == "__main__":
    unittest.main()
