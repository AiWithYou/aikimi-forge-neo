from __future__ import annotations

import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch

from backend.patcher import controlnet


class HintModel(torch.nn.Module):
    dtype = torch.float32
    unshuffle_amount = 1
    xl = False

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hint=None, **_kwargs):
        self.calls += 1
        return [hint.mean(dim=(2, 3), keepdim=True)]


class ControlNetBatchLifecycleTests(unittest.TestCase):
    @staticmethod
    def hints():
        return torch.tensor([10.0, 20.0])[:, None, None, None].expand(2, 1, 8, 8).clone()

    @staticmethod
    def noise(batch):
        return torch.zeros((batch, 1, 1, 1))

    def test_controlnet_cfg_group_changes_keep_the_original_reference_order(self):
        model = HintModel()
        cnet = controlnet.ControlNet(model, device=torch.device("cpu"), load_device=torch.device("cpu"))
        cnet.set_cond_hint(self.hints())
        cnet.model_sampling_current = SimpleNamespace(timestep=lambda t: t, calculate_input=lambda _t, x: x)
        cond = {"c_crossattn": torch.zeros((1, 1, 1))}
        for batch, groups, expected in ((2, 2, [10.0, 10.0]), (1, 1, [10.0]), (4, 2, [10.0, 20.0, 10.0, 20.0])):
            with self.subTest(batch=batch, groups=groups):
                output = cnet.get_control(self.noise(batch), torch.ones(1), cond, groups)
                torch.testing.assert_close(output["middle"][0].flatten(), torch.tensor(expected))

    def test_t2i_cached_features_follow_cfg_group_changes_without_recomputing(self):
        model = HintModel()
        adapter = controlnet.T2IAdapter(model, channels_in=1, device=torch.device("cpu"))
        adapter.set_cond_hint(self.hints())
        for batch, groups, expected in ((4, 2, [10.0, 20.0, 10.0, 20.0]), (2, 1, [10.0, 20.0])):
            with self.subTest(batch=batch, groups=groups):
                output = adapter.get_control(self.noise(batch), torch.ones(1), {}, groups)
                torch.testing.assert_close(output["input"][0].flatten(), torch.tensor(expected))
        self.assertEqual(model.calls, 1)

    def test_t2i_cleanup_releases_cached_feature_tensors(self):
        adapter = controlnet.T2IAdapter(HintModel(), channels_in=1, device=torch.device("cpu"))
        adapter.set_cond_hint(self.hints())
        adapter.get_control(self.noise(2), torch.ones(1), {}, 1)
        cached = weakref.ref(adapter.control_input[0])

        adapter.cleanup()

        self.assertIsNone(cached(), "Cleanup retained the adapter's computed feature tensor.")
        self.assertIsNone(adapter.cond_hint)

    def test_copies_preserve_the_explicit_execution_device(self):
        device = torch.device("cpu")
        controls = (
            controlnet.ControlNet(HintModel(), device=device, load_device=device),
            controlnet.ControlLora({}, device=device),
            controlnet.T2IAdapter(HintModel(), channels_in=1, device=device),
        )
        for original in controls:
            with (
                self.subTest(control=type(original).__name__),
                patch.object(controlnet.memory_management, "get_torch_device", return_value=torch.device("meta")),
            ):
                self.assertEqual(original.copy().device, device)

    def test_control_lora_cleanup_is_safe_before_a_completed_pre_run(self):
        cnet = controlnet.ControlLora({}, device=torch.device("cpu"))
        cnet.cleanup()
        cnet.cleanup()
        self.assertIsNone(cnet.model_sampling_current)


if __name__ == "__main__":
    unittest.main()
