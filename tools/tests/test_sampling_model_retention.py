"""ControlNet sampling preparation must not retain per-pass dependencies."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from backend.patcher.base import ModelPatcher
from backend.patcher.controlnet import ControlNet
from backend.patcher.unet import UnetPatcher
from backend.sampling import sampling_function


class TinySamplingModel(torch.nn.Linear):
    def __init__(self):
        super().__init__(1, 1)
        self.predictor = SimpleNamespace(percent_to_sigma=lambda percent: 1.0 - percent)

    def memory_required(self, input_shape):
        return 64

    def get_dtype(self):
        return self.weight.dtype


class SamplingModelRetentionTests(unittest.TestCase):
    def setUp(self):
        cpu = torch.device("cpu")
        self.unet = UnetPatcher(TinySamplingModel(), load_device=cpu, offload_device=cpu)
        self.additional = ModelPatcher(torch.nn.Linear(1, 1), load_device=cpu, offload_device=cpu)
        self.unet.add_extra_model_patcher_during_sampling(self.additional)
        self.controls = [ControlNet(torch.nn.Linear(1, 1), device=cpu, load_device=cpu) for _ in range(2)]
        for control in self.controls:
            self.unet.add_patched_controlnet(control)
        self.latent = torch.zeros(1, 4, 2, 2)
        self.load = self.enterContext(patch.object(sampling_function.memory_management, "load_models_gpu"))

    def test_repeated_prepare_keeps_control_models_out_of_persistent_extra_list(self):
        for _ in range(3):
            sampling_function.sampling_prepare(self.unet, self.latent)
        self.assertEqual(self.unet.extra_model_patchers_during_sampling, [self.additional])
        expected = [self.unet, self.additional] + [control.control_model_wrapped for control in self.controls]
        self.assertEqual(self.load.call_count, 3)
        for call in self.load.call_args_list:
            self.assertEqual(call.kwargs["models"], expected)
            self.assertEqual(call.kwargs["memory_required"], 64)
        for control in self.controls:
            self.assertIs(control.model_sampling_current, self.unet.model.predictor)
            self.assertEqual(control.timestep_range, (1.0, 0.0))

    def test_removed_control_is_not_loaded_by_next_sampling_pass(self):
        sampling_function.sampling_prepare(self.unet, self.latent)
        self.unet.controlnet_linked_list = None
        sampling_function.sampling_prepare(self.unet, self.latent)
        self.assertEqual(self.load.call_args.kwargs["models"], [self.unet, self.additional])

    def test_prepare_failure_does_not_change_retained_dependencies(self):
        self.load.side_effect = RuntimeError("load failed")
        with self.assertRaisesRegex(RuntimeError, "load failed"):
            sampling_function.sampling_prepare(self.unet, self.latent)
        self.assertEqual(self.unet.extra_model_patchers_during_sampling, [self.additional])


if __name__ == "__main__":
    unittest.main()
