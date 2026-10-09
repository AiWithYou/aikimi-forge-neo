"""Nonfinite restoration output must fail before conversion to pixels."""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from modules_forge.iris.runtime import Runner, check_restoration_output


class RuntimeTests(unittest.TestCase):
    def test_reported_time_and_completion_follow_model_release(self):
        clock = [0.0]
        runner = Runner.__new__(Runner)
        runner.task, runner.precision = "generate", "int8"
        runner.model = torch.nn.Identity()
        runner.model.cuda = lambda: runner.model
        runner.cfg = SimpleNamespace(
            sample=SimpleNamespace(cfg_interval=(0, 1)),
            flow=SimpleNamespace(shift=4, num_train_timesteps=1000, prediction="v"),
        )
        runner.prepare_text = lambda request: None
        runner.offload = lambda: clock.__setitem__(0, 10.0)
        sampling = types.ModuleType("iris3b.sampling")
        sampling.generate = lambda *args, **kwargs: torch.zeros(1, 3, 16, 16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "request.json").write_text(
                json.dumps(
                    {
                        "prompt": "test",
                        "negative_prompt": "",
                        "width": 16,
                        "height": 16,
                        "steps": 1,
                        "cfg": 3,
                        "seed": 1,
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.dict(sys.modules, {"iris3b.sampling": sampling}),
                patch("modules_forge.iris.runtime.time.perf_counter", side_effect=lambda: clock[0]),
                patch("torch.cuda.reset_peak_memory_stats"),
                patch("torch.cuda.synchronize"),
                patch("torch.cuda.max_memory_allocated", return_value=0),
                patch("torch.cuda.max_memory_reserved", return_value=0),
                patch("torch.Generator"),
                patch("torch.autocast"),
            ):
                result = runner.run(path)
            self.assertEqual(result["seconds"], 10.0)
            self.assertEqual(json.loads((path / "metadata.json").read_text())["seconds"], 10.0)

    def test_nonfinite_model_pixels_are_rejected(self):
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                check_restoration_output(None, (), SimpleNamespace(x=torch.full((1, 3, 8, 8), value)))
        pixels = torch.ones(1, 3, 8, 8)
        self.assertIsNone(check_restoration_output(None, (), SimpleNamespace(x=pixels)))
        torch.testing.assert_close(pixels, torch.ones_like(pixels))


if __name__ == "__main__":
    unittest.main()
