from __future__ import annotations

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from backend.nn.cnets import cldm
from backend.patcher import controlnet


class ControlLoraLoadErrorTests(unittest.TestCase):
    def run_load(self, error):
        hint = torch.ones(2, 3, 1, 1)
        adapter = controlnet.ControlLora({"input_hint_block.0.weight": hint}, device=torch.device("cpu"))
        model = SimpleNamespace(
            storage_dtype=torch.float32, computation_dtype=torch.float32,
            diffusion_model=SimpleNamespace(config={"out_channels": 4}, state_dict=lambda: {"base.weight": hint}),
        )
        with (patch.object(controlnet.ControlNet, "pre_run"),
              patch.object(controlnet, "using_forge_operations", return_value=nullcontext()),
              patch.object(cldm, "ControlNet"),
              patch.object(controlnet.memory_management, "get_torch_device", return_value=torch.device("cpu")),
              patch.object(controlnet.utils, "set_attr", side_effect=[error, None]) as assign):
            adapter.pre_run(model, lambda value: value)
            return assign.call_args_list

    def test_missing_base_layer_still_allows_control_weights(self):
        calls = self.run_load(AttributeError("layer absent from control model"))
        self.assertEqual([call.args[1] for call in calls], ["base.weight", "input_hint_block.0.weight"])

    def test_execution_errors_cannot_silently_skip_base_weights(self):
        for error in (torch.OutOfMemoryError("allocation failed"), RuntimeError("invalid parameter"),
                      KeyboardInterrupt(), SystemExit(1)):
            with self.subTest(error=type(error).__name__):
                with self.assertRaises(type(error)) as raised:
                    self.run_load(error)
                self.assertIs(raised.exception, error)


if __name__ == "__main__":
    unittest.main()
