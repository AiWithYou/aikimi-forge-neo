import unittest

import torch
from torch import nn

from modules_forge.sensenova_u15_streaming import (
    ALL_TENSOR_GROUPS,
    BranchAwareSynchronousStreamingWrapper,
    GROUP_GENERATION,
    GROUP_SHARED,
    GROUP_UNDERSTANDING,
    classify_layer_tensors,
    required_tensor_groups,
)


def _storage_pointer(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def _cloning_cpu_mover(
    tensor: torch.Tensor, target_device: torch.device
) -> torch.Tensor:
    if target_device.type != "cpu":
        raise AssertionError("CUDA-free tests require a CPU target")
    return tensor.clone(memory_format=torch.preserve_format)


class ToyBranchLayer(nn.Module):
    def __init__(self, width: int = 2) -> None:
        super().__init__()
        self.proj = nn.Linear(width, width, bias=False)
        self.proj_mot_gen = nn.Linear(width, width, bias=False)
        self.shared_scale = nn.Parameter(torch.ones(width))
        self.register_buffer("shared_bias", torch.full((width,), 0.25))
        self.seen_storage: dict[str, int] = {}
        self.raise_after_observation = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        exist_non_image_gen_tokens=None,
        exist_image_gen_tokens=None,
    ) -> torch.Tensor:
        self.seen_storage = {
            name: _storage_pointer(tensor)
            for name, tensor in list(self.named_parameters())
            + list(self.named_buffers())
        }
        if self.raise_after_observation:
            raise RuntimeError("intentional layer failure")
        if exist_non_image_gen_tokens is True and exist_image_gen_tokens is False:
            output = self.proj(hidden_states)
        elif exist_image_gen_tokens is True and exist_non_image_gen_tokens is False:
            output = self.proj_mot_gen(hidden_states)
        else:
            output = self.proj(hidden_states) + self.proj_mot_gen(hidden_states)
        return output * self.shared_scale + self.shared_bias


class ToySenseNova(nn.Module):
    def __init__(self, layer_count: int = 2) -> None:
        super().__init__()
        self.prefix_weight = nn.Parameter(torch.eye(2))
        self.register_buffer("prefix_bias", torch.tensor([0.5, -0.5]))
        self.layers = nn.ModuleList(
            ToyBranchLayer() for _ in range(layer_count)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        exist_non_image_gen_tokens=None,
        exist_image_gen_tokens=None,
    ) -> torch.Tensor:
        hidden_states = hidden_states @ self.prefix_weight + self.prefix_bias
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                exist_non_image_gen_tokens=exist_non_image_gen_tokens,
                exist_image_gen_tokens=exist_image_gen_tokens,
            )
        return hidden_states


class AmbiguousClassificationLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.outer = nn.Module()
        self.outer.inner = nn.Linear(2, 2, bias=False)
        self.outer.inner_mot_gen = nn.Linear(2, 2, bias=False)
        self.outer_mot_gen = nn.Module()
        self.outer_mot_gen.inner = nn.Linear(2, 2, bias=False)
        self.almost_mot_generation = nn.Linear(2, 2, bias=False)


def _layer_group_bytes(layer: nn.Module) -> dict[str, int]:
    groups = classify_layer_tensors(layer)
    values = {group: 0 for group in ALL_TENSOR_GROUPS}
    for name, tensor in list(layer.named_parameters()) + list(layer.named_buffers()):
        values[groups[name]] += tensor.numel() * tensor.element_size()
    return values


class SenseNovaTensorClassificationTests(unittest.TestCase):
    def test_strict_mot_gen_counterparts_and_shared_tensors(self):
        layer = ToyBranchLayer()

        groups = classify_layer_tensors(layer)

        self.assertEqual(groups["proj.weight"], GROUP_UNDERSTANDING)
        self.assertEqual(groups["proj_mot_gen.weight"], GROUP_GENERATION)
        self.assertEqual(groups["shared_scale"], GROUP_SHARED)
        self.assertEqual(groups["shared_bias"], GROUP_SHARED)

    def test_ambiguous_counterpart_is_shared_and_suffix_lookalike_is_not_generation(self):
        groups = classify_layer_tensors(AmbiguousClassificationLayer())

        self.assertEqual(groups["outer.inner.weight"], GROUP_SHARED)
        self.assertEqual(groups["outer.inner_mot_gen.weight"], GROUP_GENERATION)
        self.assertEqual(groups["outer_mot_gen.inner.weight"], GROUP_GENERATION)
        self.assertEqual(groups["almost_mot_generation.weight"], GROUP_SHARED)

    def test_unknown_and_mixed_flags_fail_closed_to_all_groups(self):
        self.assertEqual(required_tensor_groups({}), ALL_TENSOR_GROUPS)
        self.assertEqual(
            required_tensor_groups(
                {
                    "exist_non_image_gen_tokens": True,
                    "exist_image_gen_tokens": True,
                }
            ),
            ALL_TENSOR_GROUPS,
        )
        self.assertEqual(
            required_tensor_groups(
                {
                    "exist_non_image_gen_tokens": False,
                    "exist_image_gen_tokens": False,
                }
            ),
            ALL_TENSOR_GROUPS,
        )
        self.assertEqual(
            required_tensor_groups(
                {
                    "exist_non_image_gen_tokens": torch.tensor([True, False]),
                    "exist_image_gen_tokens": False,
                }
            ),
            ALL_TENSOR_GROUPS,
        )

    def test_scalar_torch_bool_flags_match_the_pinned_runtime_contract(self):
        self.assertEqual(
            required_tensor_groups(
                {
                    "exist_non_image_gen_tokens": torch.tensor(True),
                    "exist_image_gen_tokens": torch.tensor(False),
                }
            ),
            frozenset({GROUP_SHARED, GROUP_UNDERSTANDING}),
        )
        self.assertEqual(
            required_tensor_groups(
                {
                    "exist_non_image_gen_tokens": torch.tensor(False),
                    "exist_image_gen_tokens": torch.tensor(True),
                }
            ),
            frozenset({GROUP_SHARED, GROUP_GENERATION}),
        )


class SenseNovaBranchStreamingTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = ToySenseNova().eval()
        self.original_layer_storage = [
            {
                name: _storage_pointer(tensor)
                for name, tensor in list(layer.named_parameters())
                + list(layer.named_buffers())
            }
            for layer in self.model.layers
        ]
        self.original_prefix_storage = {
            "weight": _storage_pointer(self.model.prefix_weight),
            "bias": _storage_pointer(self.model.prefix_bias),
        }
        self.wrapper = BranchAwareSynchronousStreamingWrapper(
            self.model,
            layers_attr="layers",
            target_device="cpu",
            tensor_mover=_cloning_cpu_mover,
        )

    def tearDown(self) -> None:
        self.wrapper.teardown()

    def _assert_layers_restored(self) -> None:
        for layer, originals in zip(
            self.model.layers, self.original_layer_storage, strict=True
        ):
            current = {
                name: _storage_pointer(tensor)
                for name, tensor in list(layer.named_parameters())
                + list(layer.named_buffers())
            }
            self.assertEqual(current, originals)

    def test_understanding_generation_and_mixed_forwards_move_only_safe_groups(self):
        sample = torch.ones(1, 2)

        self.wrapper(
            sample,
            exist_non_image_gen_tokens=True,
            exist_image_gen_tokens=False,
        )
        for layer, originals in zip(
            self.model.layers, self.original_layer_storage, strict=True
        ):
            self.assertNotEqual(layer.seen_storage["proj.weight"], originals["proj.weight"])
            self.assertEqual(
                layer.seen_storage["proj_mot_gen.weight"],
                originals["proj_mot_gen.weight"],
            )
            self.assertNotEqual(
                layer.seen_storage["shared_scale"], originals["shared_scale"]
            )
        self._assert_layers_restored()

        self.wrapper(
            sample,
            exist_non_image_gen_tokens=False,
            exist_image_gen_tokens=True,
        )
        for layer, originals in zip(
            self.model.layers, self.original_layer_storage, strict=True
        ):
            self.assertEqual(layer.seen_storage["proj.weight"], originals["proj.weight"])
            self.assertNotEqual(
                layer.seen_storage["proj_mot_gen.weight"],
                originals["proj_mot_gen.weight"],
            )
            self.assertNotEqual(
                layer.seen_storage["shared_bias"], originals["shared_bias"]
            )
        self._assert_layers_restored()

        self.wrapper(
            sample,
            exist_non_image_gen_tokens=True,
            exist_image_gen_tokens=True,
        )
        for layer, originals in zip(
            self.model.layers, self.original_layer_storage, strict=True
        ):
            for name, original in originals.items():
                self.assertNotEqual(layer.seen_storage[name], original)
        self._assert_layers_restored()

    def test_group_transfer_bytes_and_forward_counts_are_cumulative(self):
        sample = torch.ones(1, 2)
        branches = ((True, False), (False, True), (True, True))
        for understanding, generation in branches:
            self.wrapper(
                sample,
                exist_non_image_gen_tokens=understanding,
                exist_image_gen_tokens=generation,
            )

        per_layer = [_layer_group_bytes(layer) for layer in self.model.layers]
        expected_layer_bytes = {
            GROUP_UNDERSTANDING: sum(
                values[GROUP_UNDERSTANDING] * 2 for values in per_layer
            ),
            GROUP_GENERATION: sum(
                values[GROUP_GENERATION] * 2 for values in per_layer
            ),
            GROUP_SHARED: sum(values[GROUP_SHARED] * 3 for values in per_layer),
        }
        expected_non_layer = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.model.prefix_weight, self.model.prefix_bias)
        )
        telemetry = self.wrapper.telemetry

        self.assertEqual(telemetry.total_layer_forwards, 6)
        self.assertEqual(
            telemetry.layer_forward_counts_by_group,
            {
                GROUP_UNDERSTANDING: 4,
                GROUP_GENERATION: 4,
                GROUP_SHARED: 6,
            },
        )
        self.assertEqual(
            telemetry.layer_transfer_bytes_by_group, expected_layer_bytes
        )
        self.assertEqual(telemetry.non_layer_transfer_bytes, expected_non_layer)
        self.assertEqual(
            telemetry.total_transfer_bytes,
            expected_non_layer + sum(expected_layer_bytes.values()),
        )

    def test_unknown_flags_load_all_groups(self):
        self.wrapper(torch.ones(1, 2))

        for layer, originals in zip(
            self.model.layers, self.original_layer_storage, strict=True
        ):
            for name, original in originals.items():
                self.assertNotEqual(layer.seen_storage[name], original)
        telemetry = self.wrapper.telemetry
        self.assertEqual(
            telemetry.layer_forward_counts_by_group,
            {GROUP_UNDERSTANDING: 2, GROUP_GENERATION: 2, GROUP_SHARED: 2},
        )
        self._assert_layers_restored()

    def test_reused_scalar_flags_are_rechecked_at_each_model_forward(self):
        understanding = torch.tensor(True)
        generation = torch.tensor(False)

        self.wrapper(
            torch.ones(1, 2),
            exist_non_image_gen_tokens=understanding,
            exist_image_gen_tokens=generation,
        )
        understanding.fill_(False)
        generation.fill_(True)
        self.wrapper(
            torch.ones(1, 2),
            exist_non_image_gen_tokens=understanding,
            exist_image_gen_tokens=generation,
        )

        telemetry = self.wrapper.telemetry
        self.assertEqual(
            telemetry.layer_forward_counts_by_group,
            {GROUP_UNDERSTANDING: 2, GROUP_GENERATION: 2, GROUP_SHARED: 4},
        )
        self._assert_layers_restored()

    def test_exception_path_and_teardown_restore_exact_cpu_storage(self):
        self.model.layers[0].raise_after_observation = True
        with self.assertRaisesRegex(RuntimeError, "intentional layer failure"):
            self.wrapper(
                torch.ones(1, 2),
                exist_non_image_gen_tokens=False,
                exist_image_gen_tokens=True,
            )

        self._assert_layers_restored()
        self.assertNotEqual(
            _storage_pointer(self.model.prefix_weight),
            self.original_prefix_storage["weight"],
        )
        self.assertNotEqual(
            _storage_pointer(self.model.prefix_bias),
            self.original_prefix_storage["bias"],
        )

        self.wrapper.teardown()
        self.wrapper.teardown()

        self.assertTrue(self.wrapper.closed)
        self.assertEqual(
            _storage_pointer(self.model.prefix_weight),
            self.original_prefix_storage["weight"],
        )
        self.assertEqual(
            _storage_pointer(self.model.prefix_bias),
            self.original_prefix_storage["bias"],
        )


if __name__ == "__main__":
    unittest.main()
