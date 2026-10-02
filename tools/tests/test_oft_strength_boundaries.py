"""Compare existing OFT/BOFT strengths with dense rotation operators on CPU."""

import tempfile
import unittest
from pathlib import Path

import torch

from backend.patcher.lora import merge_lora_to_weight
from modules_forge.packages.comfy.weight_adapter.boft import BOFTAdapter
from modules_forge.packages.comfy.weight_adapter.oft import OFTAdapter


def constrained_rotation(blocks, alpha, outputs):
    identity = torch.eye(blocks.shape[-1])
    skew = blocks - blocks.transpose(-1, -2)
    constraint = alpha * outputs
    if constraint > 0:
        skew = skew * min(1.0, constraint / (skew.norm().item() + 1e-8))
    return (identity + skew) @ torch.linalg.inv(identity - skew)


class OftStrengthBoundaryTests(unittest.TestCase):
    def saved_weights(self, blocks, rescale=None):
        with tempfile.TemporaryDirectory() as temporary:
            filename = Path(temporary) / "回転 重み.pt"
            state = {"layer.oft_blocks": blocks}
            if rescale is not None:
                state["layer.rescale"] = rescale
            torch.save(state, filename)
            return torch.load(filename, map_location="cpu", weights_only=True)

    def test_oft_strength_applies_once_to_dense_block_rotation(self):
        blocks = torch.tensor([[[0.0, 0.2], [-0.1, 0.0]], [[0.0, -0.15], [0.05, 0.0]]])
        for alpha in (0.0, 0.03):
            rotations = constrained_rotation(blocks, alpha, 4)
            rotation = torch.block_diag(*rotations.transpose(-1, -2))
            adapter = OFTAdapter.load("layer", self.saved_weights(blocks), alpha, None)
            for dimensions in ((4, 3), (4, 3, 1, 1)):
                original = torch.arange(1, 13, dtype=torch.float32).reshape(dimensions) / 10
                for strength in (0.0, 0.5, 1.0, -0.5):
                    with self.subTest(alpha=alpha, dimensions=dimensions, strength=strength):
                        expected = (
                            original.flatten(1) + strength * (rotation - torch.eye(4)) @ original.flatten(1)
                        ).reshape(dimensions)
                        actual = merge_lora_to_weight(
                            [(strength, adapter, 1.0, None, None)], original.clone(), "weight"
                        )
                        torch.testing.assert_close(actual, expected)

    def test_boft_strength_interpolates_each_butterfly_stage_once(self):
        blocks = torch.tensor(
            [
                [[[0.0, 0.2], [-0.1, 0.0]], [[0.0, -0.15], [0.05, 0.0]]],
                [[[0.0, -0.3], [0.1, 0.0]], [[0.0, 0.1], [-0.15, 0.0]]],
            ]
        )
        for stages in (1, 2):
            for alpha in (0.0, 0.03):
                selected = blocks[:stages]
                rotations = constrained_rotation(selected, alpha, 4)
                adapter = BOFTAdapter.load("layer", self.saved_weights(selected), alpha, None)
                for dimensions in ((4, 3), (4, 3, 1, 1)):
                    original = torch.arange(1, 13, dtype=torch.float32).reshape(dimensions) / 10
                    for strength in (0.0, 0.5, 1.0, -0.5):
                        with self.subTest(stages=stages, alpha=alpha, dimensions=dimensions, strength=strength):
                            expected = original.flatten(1)
                            for stage_number in range(stages):
                                stage = torch.eye(4)
                                groups = ((0, 1), (2, 3)) if stage_number == 0 else ((0, 2), (1, 3))
                                for rotation, group in zip(rotations[stage_number], groups, strict=True):
                                    interpolated = rotation * strength + (1 - strength) * torch.eye(2)
                                    for row, output in enumerate(group):
                                        for column, input_channel in enumerate(group):
                                            stage[output, input_channel] = interpolated[row, column]
                                expected = stage @ expected
                            actual = merge_lora_to_weight(
                                [(strength, adapter, 1.0, None, None)], original.clone(), "weight"
                            )
                            torch.testing.assert_close(actual, expected.reshape(dimensions))

    def test_offset_model_strength_and_weight_function_preserve_surrounding_rows(self):
        blocks = torch.tensor([[[0.0, 0.2], [-0.1, 0.0]], [[0.0, -0.15], [0.05, 0.0]]])
        rotations = constrained_rotation(blocks, 0.0, 4)
        original = torch.arange(1, 19, dtype=torch.float32).reshape(6, 3) / 10
        for adapter_class in (OFTAdapter, BOFTAdapter):
            saved_blocks = blocks if adapter_class is OFTAdapter else blocks[None]
            adapter = adapter_class.load("layer", self.saved_weights(saved_blocks), 0.0, None)
            rotation = torch.block_diag(*(rotations.transpose(-1, -2) if adapter_class is OFTAdapter else rotations))
            for strength in (0.5, -0.5):
                with self.subTest(adapter=adapter_class.name, strength=strength):
                    expected = original.clone()
                    selected = original[1:5] * 0.8
                    expected[1:5] = selected + 1.5 * strength * ((rotation - torch.eye(4)) @ selected)
                    actual = merge_lora_to_weight(
                        [(strength, adapter, 0.8, (0, 1, 4), lambda delta: delta * 1.5)], original.clone(), "weight"
                    )
                    torch.testing.assert_close(actual, expected)

    def test_zero_strength_with_saved_rescale_keeps_the_disabled_lora_contract(self):
        blocks = torch.tensor([[[0.0, 0.2], [-0.1, 0.0]], [[0.0, -0.15], [0.05, 0.0]]])
        for dimensions in ((4, 3), (4, 3, 1, 1)):
            original = torch.arange(1, 13, dtype=torch.float32).reshape(dimensions) / 10
            rescale = torch.tensor([0.7, 1.2, 1.4, 0.8]).reshape(4, *([1] * (len(dimensions) - 1)))
            for adapter_class in (OFTAdapter, BOFTAdapter):
                with self.subTest(adapter=adapter_class.name, dimensions=dimensions):
                    saved_blocks = blocks if adapter_class is OFTAdapter else blocks[None]
                    adapter = adapter_class.load("layer", self.saved_weights(saved_blocks, rescale), 0.0, None)
                    actual = merge_lora_to_weight([(0.0, adapter, 1.0, None, None)], original.clone(), "weight")
                    torch.testing.assert_close(actual, original)


if __name__ == "__main__":
    unittest.main()
