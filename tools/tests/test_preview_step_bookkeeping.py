"""Live previews track completed sampler steps instead of zero-based indices."""

import ast
import datetime
import logging
import sys
import threading
import time
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Optional
from unittest.mock import Mock, patch

import torch
from PIL import Image

import modules
from tools.tests.test_gpu_ownership import load_function

ROOT = Path(__file__).resolve().parents[2]


class PreviewStepBookkeepingTests(unittest.TestCase):
    def setUp(self):
        self.opts = SimpleNamespace(
            live_previews_enable=True,
            show_progress_every_n_steps=1,
            show_progress_grid=False,
            live_previews_image_format="png",
        )
        self.errors = Mock()
        namespace = {
            "datetime": datetime,
            "logging": logging,
            "threading": threading,
            "time": time,
            "nullcontext": nullcontext,
            "Optional": Optional,
            "torch": torch,
            "Image": Image,
            "log": logging.getLogger(__name__),
            "stream": Mock(should_use_stream=Mock(return_value=False)),
            "shared": SimpleNamespace(opts=self.opts),
            "devices": Mock(),
            "errors": self.errors,
        }
        source = ROOT / "modules/shared_state.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        state_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "State")
        exec(compile(ast.Module(body=[state_class], type_ignores=[]), str(source), "exec"), namespace)  # noqa: S102
        self.state = namespace["State"]()
        self.state.sampling_steps = 20
        self.state.current_latent = torch.zeros(1, 4, 2, 2)
        self.image = Image.new("RGB", (2, 2))
        self.addCleanup(self.image.close)
        self.decoded_steps = []

        def decode(_latent):
            self.decoded_steps.append(self.state.preview_step)
            return self.image

        self.samplers = ModuleType("modules.sd_samplers")
        self.samplers.sample_to_image = Mock(side_effect=decode)
        self.samplers.samples_to_image_grid = Mock(side_effect=decode)
        self.samplers.sample_to_video = Mock(side_effect=decode)
        key = "modules.sd_samplers"
        missing = object()
        previous = sys.modules.get(key, missing)

        def restore_sampler_module():
            if previous is missing:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = previous

        sys.modules[key] = self.samplers
        self.addCleanup(restore_sampler_module)
        self.enterContext(patch.object(modules, "sd_samplers", self.samplers, create=True))
        self.callback = load_function(
            "modules/sd_samplers_common.py",
            "callback_state",
            {"state": self.state, "shared": SimpleNamespace(total_tqdm=Mock()), "InterruptedException": RuntimeError},
        )
        self.sampler = SimpleNamespace(stop_at=None)

    def advance(self, completed_step):
        self.callback(self.sampler, {"i": completed_step - 1})

    def test_repeated_poll_decodes_each_completed_step_once(self):
        self.advance(1)
        for _ in range(4):
            self.state.set_current_image()
        self.assertEqual(self.decoded_steps, [1])
        self.assertEqual(self.state.id_live_preview, 1)

    def test_preview_interval_has_no_early_decode(self):
        self.opts.show_progress_every_n_steps = 5
        for step in range(1, 16):
            self.advance(step)
            for _ in range(2):
                self.state.set_current_image()
        self.assertEqual(self.decoded_steps, [5, 10, 15])

    def test_decode_records_starting_step_when_sampler_advances(self):
        self.advance(5)

        def advance_during_decode(_latent):
            self.advance(8)
            return self.image

        self.samplers.sample_to_image.side_effect = advance_during_decode
        self.state.set_current_image()
        self.assertEqual(self.state.current_image_sampling_step, 5)

    def test_failed_decode_keeps_step_available_for_retry(self):
        self.advance(1)
        self.samplers.sample_to_image.side_effect = RuntimeError("preview failed")
        self.state.set_current_image()
        self.errors.record_exception.assert_called_once_with()
        self.assertEqual(self.state.current_image_sampling_step, 0)
        self.samplers.sample_to_image.side_effect = lambda _latent: self.image
        self.state.set_current_image()
        self.assertEqual(self.state.current_image_sampling_step, 1)
        self.assertEqual(self.state.id_live_preview, 1)

    def test_grid_and_video_use_the_same_step_bookkeeping(self):
        for video in (False, True):
            with self.subTest(video=video):
                self.opts.show_progress_grid = not video
                self.state.current_latent = torch.zeros(1, 4, 3, 2, 2) if video else torch.zeros(1, 4, 2, 2)
                self.state.current_image_sampling_step = 0
                self.decoded_steps.clear()
                self.advance(1)
                self.state.set_current_image()
                self.state.set_current_image()
                self.assertEqual(self.decoded_steps, [1])


if __name__ == "__main__":
    unittest.main()
