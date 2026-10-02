from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from backend.patcher.lora import merge_lora_to_weight
from modules_forge.packages.comfy.weight_adapter.loha import LoHaAdapter
from modules_forge.packages.comfy.weight_adapter.lokr import LoKrAdapter
from modules_forge.packages.comfy.weight_adapter.lora import LoRAAdapter


def tucker_weight(up, middle, down):
    return torch.einsum("or, rsxy, si -> oixy", up, middle, down)


class LoraAdapterBoundaryTests(unittest.TestCase):
    def test_conv_dora_normalizes_the_selected_axis_including_one_output_channel(self):
        for outputs, axis in ((2, 0), (2, 1), (1, 0), (1, 1)):
            original = torch.arange(1, outputs * 3 * 4 + 1, dtype=torch.float32).reshape(outputs, 3, 2, 2) / 10
            up = torch.arange(1, outputs * 2 + 1, dtype=torch.float32).reshape(outputs, 2, 1, 1) / 10
            down = torch.arange(1, 25, dtype=torch.float32).reshape(2, 3, 2, 2) / 15
            dimensions = [1, 1, 1, 1]
            dimensions[axis] = original.shape[axis]
            magnitude = torch.arange(1, original.shape[axis] + 1, dtype=torch.float32).reshape(dimensions)
            adapter = LoRAAdapter.load(
                "conv", {"conv.lora_up.weight": up, "conv.lora_down.weight": down}, 3.0, magnitude
            )
            delta = (up.flatten(1) @ down.flatten(1)).reshape(original.shape) * 1.5
            updated = original + delta
            norm_dimensions = tuple(dimension for dimension in range(4) if dimension != axis)
            normalized = (
                updated
                * magnitude
                / (
                    torch.linalg.vector_norm(updated, dim=norm_dimensions, keepdim=True)
                    + torch.finfo(updated.dtype).eps
                )
            )

            for strength in (1.0, 0.5):
                with self.subTest(outputs=outputs, axis=axis, strength=strength):
                    actual = merge_lora_to_weight([(strength, adapter, 1.0, None, None)], original.clone(), "weight")
                    torch.testing.assert_close(actual, original + strength * (normalized - original))

    def test_locon_mid_kernel_rank_scale_and_offset_match_tensor_contraction(self):
        up = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        down = torch.tensor([[2.0, 1.0], [4.0, 3.0]])
        middle = torch.arange(1, 17, dtype=torch.float32).reshape(2, 2, 2, 2) / 10
        adapter = LoRAAdapter.load(
            "conv",
            {
                "conv.lora_up.weight": up[:, :, None, None],
                "conv.lora_down.weight": down[:, :, None, None],
                "conv.lora_mid.weight": middle,
            },
            3.0,
            None,
        )
        self.assert_scaled_offset(adapter, tucker_weight(up, middle, down) * 1.5)

    def test_loha_matrix_and_cp_forms_match_the_hadamard_product_and_rank_scale(self):
        up1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        down1 = torch.tensor([[2.0, 1.0], [4.0, 3.0]])
        up2 = torch.tensor([[0.5, 1.5], [2.5, 3.5]])
        down2 = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
        middle1 = torch.arange(1, 17, dtype=torch.float32).reshape(2, 2, 2, 2) / 10
        middle2 = middle1.flip(-1) / 2
        for cp in (False, True):
            state = {
                "conv.hada_w1_a": up1.t() if cp else up1,
                "conv.hada_w1_b": down1,
                "conv.hada_w2_a": up2.t() if cp else up2,
                "conv.hada_w2_b": down2,
            }
            if cp:
                state.update({"conv.hada_t1": middle1, "conv.hada_t2": middle2})
                delta = tucker_weight(up1, middle1, down1) * tucker_weight(up2, middle2, down2)
            else:
                delta = (up1 @ down1) * (up2 @ down2)
            with self.subTest(cp=cp):
                self.assert_scaled_offset(LoHaAdapter.load("conv", state, 3.0, None), delta * 1.5)

    def test_lokr_full_factorized_and_cp_forms_match_the_kronecker_product(self):
        first = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        up = torch.tensor([[2.0, 1.0], [4.0, 3.0]])
        down = torch.tensor([[0.5, 1.5], [2.5, 3.5]])
        middle = torch.arange(1, 17, dtype=torch.float32).reshape(2, 2, 2, 2) / 10
        for kind in ("full", "factorized", "cp"):
            state = {"conv.lokr_w1": first}
            if kind == "full":
                state["conv.lokr_w2"] = up @ down
                second = up @ down
                scale = 1.0
            elif kind == "factorized":
                state.update({"conv.lokr_w2_a": up, "conv.lokr_w2_b": down})
                second = up @ down
                scale = 1.5
            else:
                state.update({"conv.lokr_w2_a": up.t(), "conv.lokr_w2_b": down, "conv.lokr_t2": middle})
                second = tucker_weight(up, middle, down)
                scale = 1.5
            expanded = first[:, :, None, None] if kind == "cp" else first
            with self.subTest(kind=kind):
                self.assert_scaled_offset(
                    LoKrAdapter.load("conv", state, 3.0, None),
                    torch.kron(expanded.contiguous(), second.contiguous()) * scale,
                )

    def test_saved_noncontiguous_lokr_factors_are_not_silently_dropped(self):
        first = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        second = torch.tensor([[2.0, 1.0], [4.0, 3.0]])
        with tempfile.TemporaryDirectory(prefix="LoKr 既存重み ") as temporary:
            filename = Path(temporary) / "adapter.pt"
            for transposed in ("first", "second"):
                state = {
                    "conv.lokr_w1": first.t() if transposed == "first" else first,
                    "conv.lokr_w2": second.t() if transposed == "second" else second,
                }
                torch.save(state, filename)
                loaded = torch.load(filename, map_location="cpu", weights_only=True)
                field = "conv.lokr_w1" if transposed == "first" else "conv.lokr_w2"
                self.assertFalse(loaded[field].is_contiguous())
                delta = torch.kron(loaded["conv.lokr_w1"].contiguous(), loaded["conv.lokr_w2"].contiguous())
                with self.subTest(transposed=transposed):
                    self.assert_scaled_offset(LoKrAdapter.load("conv", loaded, None, None), delta)

    def assert_scaled_offset(self, adapter, delta):
        shape = [delta.shape[0] + 2, *delta.shape[1:]]
        original = torch.ones(shape)
        offset = (0, 1, delta.shape[0])
        for strength in (1.0, 0.5, -0.5, 0.0):
            expected = original.clone()
            expected[1:-1] = original[1:-1] * 0.8 + delta * strength
            with self.subTest(strength=strength):
                actual = merge_lora_to_weight([(strength, adapter, 0.8, offset, None)], original.clone(), "weight")
                torch.testing.assert_close(actual, expected)

        for axis in (0, 1):
            dimensions = [1] * delta.ndim
            dimensions[axis] = delta.shape[axis]
            magnitude = torch.arange(1, delta.shape[axis] + 1, dtype=torch.float32).reshape(dimensions)
            fields = list(adapter.weights)
            fields[{LoRAAdapter: 4, LoHaAdapter: 7, LoKrAdapter: 8}[type(adapter)]] = magnitude
            decomposed = type(adapter)(adapter.loaded_keys.copy(), tuple(fields))
            base = original[1:-1] * 0.8
            updated = base + delta
            norm_dimensions = tuple(dimension for dimension in range(delta.ndim) if dimension != axis)
            normalized = (
                updated
                * magnitude
                / (
                    torch.linalg.vector_norm(updated, dim=norm_dimensions, keepdim=True)
                    + torch.finfo(updated.dtype).eps
                )
            )
            for strength in (1.0, 0.5):
                expected = original.clone()
                expected[1:-1] = base + strength * (normalized - base)
                with self.subTest(dora_axis=axis, strength=strength):
                    actual = merge_lora_to_weight(
                        [(strength, decomposed, 0.8, offset, None)], original.clone(), "weight"
                    )
                    torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
