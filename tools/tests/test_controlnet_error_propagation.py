from __future__ import annotations

import contextlib
import io
import unittest

import torch

from backend.nn.unet import apply_control


class ControlNetErrorPropagationTests(unittest.TestCase):
    def test_fatal_errors_are_propagated(self):
        for error in (KeyboardInterrupt(), SystemExit(1), MemoryError(), torch.OutOfMemoryError("allocation failed"),
                      RuntimeError("CUDA out of memory")):
            with self.subTest(error=type(error).__name__):
                class FailingTensor:
                    shape = (1, 2)

                    def __init__(self, exception):
                        self.exception = exception

                    def __iadd__(self, other):
                        raise self.exception

                with self.assertRaises(type(error)) as raised:
                    apply_control(FailingTensor(error), {"input": [torch.ones(1, 2)]}, "input")
                self.assertIs(raised.exception, error)

    def test_shape_mismatch_keeps_original_and_warns(self):
        original = torch.ones(1, 2)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            actual = apply_control(original, {"input": [torch.ones(1, 3)]}, "input")
        self.assertIs(actual, original)
        torch.testing.assert_close(actual, torch.ones(1, 2))
        self.assertIn("control could not be applied", output.getvalue())

    def test_valid_control_is_consumed_and_added_once(self):
        original = torch.ones(1, 2)
        control = {"input": [torch.full((1, 2), 2.0)]}
        actual = apply_control(original, control, "input")
        self.assertIs(actual, original)
        torch.testing.assert_close(actual, torch.full((1, 2), 3.0))
        self.assertEqual(control["input"], [])
        self.assertIs(apply_control(actual, control, "input"), actual)


if __name__ == "__main__":
    unittest.main()
