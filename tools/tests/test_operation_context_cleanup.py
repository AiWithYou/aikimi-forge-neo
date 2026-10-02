"""Check operation contexts restore resources and construction settings on CPU."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from backend import operations


class OperationContextCleanupTests(unittest.TestCase):
    def test_stream_waits_after_operation_failure_or_interrupt(self):
        for error in (RuntimeError("kernel failed"), KeyboardInterrupt()):
            for weight, bias in ((torch.ones(2), None), (None, torch.ones(2))):
                with self.subTest(error=type(error).__name__, bias_only=weight is None):
                    offload = Mock()
                    current = object()
                    with patch.object(operations.memory_management, "current_stream", return_value=current):
                        with self.assertRaises(type(error)) as raised:
                            with operations.main_stream_worker(weight, bias, (offload, weight, bias)):
                                raise error
                    self.assertIs(raised.exception, error)
                    offload.wait_stream.assert_called_once_with(current)

    def test_missing_stream_or_weights_preserves_body_exception(self):
        for signal in (None, (None, torch.ones(1), None), (Mock(), None, None)):
            with self.subTest(signal=signal):
                with self.assertRaisesRegex(RuntimeError, "body failed"):
                    with operations.main_stream_worker(None, None, signal):
                        raise RuntimeError("body failed")

    def test_nested_context_restores_outer_layer_construction(self):
        native_linear = torch.nn.Linear
        with operations.using_forge_operations(
            device=torch.device("cpu"), dtype=torch.float64, manual_cast_enabled=True
        ):
            with operations.using_forge_operations(device=torch.device("meta"), dtype=torch.float32):
                inner = torch.nn.Linear(2, 3)
                self.assertEqual(inner.weight.device.type, "meta")
                self.assertFalse(inner.parameters_manual_cast)
            outer = torch.nn.Linear(2, 3)
            self.assertEqual(outer.weight.device.type, "cpu")
            self.assertEqual(outer.weight.dtype, torch.float64)
            self.assertTrue(outer.parameters_manual_cast)
        self.assertIs(torch.nn.Linear, native_linear)

    def test_quantization_configuration_is_reusable(self):
        configuration = {"TE": True, "layer": {"format": "int8"}}
        factory = Mock(return_value=operations.ForgeOperations)
        with (
            patch.object(operations, "mixed_precision_ops", factory),
            patch.object(operations.memory_management, "get_torch_device", return_value=torch.device("cpu")),
            patch.object(operations.memory_management, "should_use_bf16", return_value=False),
            patch.object(operations.memory_management, "supports_fp8_compute", return_value=False),
            patch.object(operations.memory_management, "supports_nvfp4_compute", return_value=False),
            patch.object(operations.memory_management, "supports_mxfp8_compute", return_value=False),
        ):
            for _ in range(2):
                with operations.using_forge_operations(bnb_dtype=configuration):
                    pass
        self.assertEqual(configuration, {"TE": True, "layer": {"format": "int8"}})
        self.assertEqual([call.kwargs["full_precision_mm"] for call in factory.call_args_list], [True, True])

    def test_failed_context_setup_restores_all_construction_settings(self):
        previous = (
            operations.current_device,
            operations.current_dtype,
            operations.current_manual_cast_enabled,
            operations.current_bnb_dtype,
        )
        native_linear = torch.nn.Linear
        invalid = SimpleNamespace(Linear=operations.ForgeOperations.Linear, __name__="incomplete")
        with self.assertRaises(AttributeError):
            with operations.using_forge_operations(
                operations=invalid, device="meta", dtype=torch.float16, bnb_dtype="fixture"
            ):
                pass
        self.assertIs(torch.nn.Linear, native_linear)
        self.assertEqual(
            (
                operations.current_device,
                operations.current_dtype,
                operations.current_manual_cast_enabled,
                operations.current_bnb_dtype,
            ),
            previous,
        )


if __name__ == "__main__":
    unittest.main()
