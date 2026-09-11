from contextlib import nullcontext
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

_ORIGINAL_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]
ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ROOT / "modules_forge" / "packages"
if str(PACKAGES) not in sys.path:
    sys.path.insert(0, str(PACKAGES))

import modules.shared_init as shared_init

shared_init.initialize()
sys.argv = _ORIGINAL_ARGV

from backend import memory_management
from modules import processing


class ProcessCleanupTests(unittest.TestCase):
    @staticmethod
    def _processing(runner):
        return SimpleNamespace(
            scripts=runner,
            override_settings={},
            override_settings_restore_afterwards=False,
        )

    def test_cleanup_hook_runs_after_success(self):
        runner = Mock()
        request = self._processing(runner)
        expected = object()
        with (
            patch.object(processing, "set_config"),
            patch.object(processing, "manage_model_and_prompt_cache"),
            patch.object(processing.sd_samplers, "fix_p_invalid_sampler_and_scheduler"),
            patch.object(processing.profiling, "Profiler", return_value=nullcontext()),
            patch.object(processing, "process_images_inner", return_value=expected),
        ):
            actual = processing.process_images(request)

        self.assertIs(actual, expected)
        runner.before_process.assert_called_once_with(request)
        runner.on_process_cleanup.assert_called_once_with(request)

    def test_cleanup_hook_runs_after_processing_error(self):
        runner = Mock()
        request = self._processing(runner)
        with (
            patch.object(processing, "set_config"),
            patch.object(
                processing,
                "manage_model_and_prompt_cache",
                side_effect=RuntimeError("intentional failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "intentional failure"),
        ):
            processing.process_images(request)

        runner.on_process_cleanup.assert_called_once_with(request)

    def test_generation_overrides_use_narrow_module_boundary_without_mutating_request(self):
        runner = Mock()
        request = self._processing(runner)
        original = {
            "forge_additional_modules": ["safe-selector.safetensors"],
            "sd_model_checkpoint": "missing-checkpoint.safetensors",
        }
        request.override_settings = original
        with (
            patch.object(processing, "set_config") as set_config,
            patch.object(processing, "manage_model_and_prompt_cache"),
            patch.object(processing.sd_samplers, "fix_p_invalid_sampler_and_scheduler"),
            patch.object(processing.profiling, "Profiler", return_value=nullcontext()),
            patch.object(processing, "process_images_inner", return_value=object()),
        ):
            processing.process_images(request)

        self.assertIs(request.override_settings, original)
        self.assertEqual(
            original,
            {
                "forge_additional_modules": ["safe-selector.safetensors"],
                "sd_model_checkpoint": "missing-checkpoint.safetensors",
            },
        )
        applied = set_config.call_args_list[0]
        self.assertIsNot(applied.args[0], original)
        self.assertEqual(
            applied.args[0],
            {"forge_additional_modules": ["safe-selector.safetensors"]},
        )
        self.assertTrue(applied.kwargs["allow_generation_module_override"])

    def test_unload_model_dispatches_tracked_model_cleanup_once(self):
        patcher = object()
        loaded = Mock()
        loaded.model = patcher
        original = list(memory_management.current_loaded_models)
        memory_management.current_loaded_models[:] = [loaded]
        try:
            self.assertTrue(memory_management.unload_model(patcher))
            self.assertEqual(memory_management.current_loaded_models, [])
            loaded.model_unload.assert_called_once_with()
        finally:
            memory_management.current_loaded_models[:] = original


if __name__ == "__main__":
    unittest.main()
