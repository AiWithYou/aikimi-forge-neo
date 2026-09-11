"""Branch-aware synchronous layer streaming for SenseNova U1.5.

SenseNova decoder layers contain parallel understanding and image-generation
branches.  Generation modules use a ``*_mot_gen`` path component while their
understanding counterparts use the same path without that suffix.  Moving an
entire layer for every forward therefore transfers roughly twice the weight
data required by a branch-specific invocation.

This module keeps the conservative synchronous offload contract: the model
starts on CPU, only tensors needed by the current layer invocation are moved
to the target device, and their original CPU storage references are restored
as soon as that invocation returns.  Unknown or mixed branch flags request all
groups, which is deliberately slower but cannot omit a required tensor.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
import itertools
import threading
from typing import Any

import torch
from torch import nn


GROUP_SHARED = "shared"
GROUP_UNDERSTANDING = "understanding"
GROUP_GENERATION = "generation"
ALL_TENSOR_GROUPS = frozenset(
    {GROUP_SHARED, GROUP_UNDERSTANDING, GROUP_GENERATION}
)

_TensorMover = Callable[[torch.Tensor, torch.device], torch.Tensor]


def _named_tensors(module: nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield every parameter and persistent/non-persistent buffer by path."""

    yield from module.named_parameters()
    yield from module.named_buffers()


def classify_layer_tensors(layer: nn.Module) -> dict[str, str]:
    """Classify layer tensors as shared, understanding, or generation.

    A module-path component ending in ``_mot_gen`` is generation-specific.  A
    non-generation tensor is understanding-specific only when exactly one
    otherwise-identical generation path exists.  No match means shared.  More
    than one possible match is ambiguous and is also classified as shared, so
    it is transferred for every branch rather than being unsafely omitted.
    """

    names = {name for name, _tensor in _named_tensors(layer)}
    groups: dict[str, str] = {}

    for name in names:
        parts = name.split(".")
        module_parts = parts[:-1]
        if any(part.endswith("_mot_gen") for part in module_parts):
            groups[name] = GROUP_GENERATION
            continue

        candidates = 0
        for index in range(len(module_parts)):
            candidate = parts.copy()
            candidate[index] = f"{candidate[index]}_mot_gen"
            if ".".join(candidate) in names:
                candidates += 1

        groups[name] = GROUP_UNDERSTANDING if candidates == 1 else GROUP_SHARED

    return groups


def required_tensor_groups(kwargs: Mapping[str, Any]) -> frozenset[str]:
    """Return the safe tensor groups for one decoder-layer invocation.

    SenseNova supplies two boolean keyword arguments to every decoder layer.
    The pinned runtime currently passes scalar ``torch.bool`` tensors while
    newer runtimes normalize them to Python ``bool``.  Both forms are accepted.
    Missing, non-scalar/non-boolean, both-false, and mixed both-true inputs load
    all groups.
    """

    def branch_bool(value: Any) -> bool | None:
        if type(value) is bool:
            return value
        if (
            isinstance(value, torch.Tensor)
            and value.dtype == torch.bool
            and value.numel() == 1
        ):
            try:
                return bool(value.detach().item())
            except (RuntimeError, ValueError):
                return None
        return None

    use_understanding = branch_bool(kwargs.get("exist_non_image_gen_tokens"))
    use_generation = branch_bool(kwargs.get("exist_image_gen_tokens"))

    if use_understanding is None or use_generation is None:
        return ALL_TENSOR_GROUPS
    if use_understanding and not use_generation:
        return frozenset({GROUP_SHARED, GROUP_UNDERSTANDING})
    if use_generation and not use_understanding:
        return frozenset({GROUP_SHARED, GROUP_GENERATION})
    return ALL_TENSOR_GROUPS


@dataclass(frozen=True)
class StreamingTelemetry:
    """Cumulative logical H2D work performed by a streaming wrapper.

    ``layer_forward_counts_by_group`` counts decoder-layer invocations that
    requested each group.  ``layer_transfer_bytes_by_group`` counts source
    tensor bytes passed to the device mover.  Tests may use a CPU cloning mover;
    in production with a CUDA target these values represent logical H2D bytes.
    """

    total_transfer_bytes: int
    non_layer_transfer_bytes: int
    layer_transfer_bytes_by_group: dict[str, int]
    total_layer_forwards: int
    layer_forward_counts_by_group: dict[str, int]


@dataclass
class _LayerRecord:
    module: nn.Module
    tensors: dict[str, torch.Tensor]
    cpu_data: dict[str, torch.Tensor]
    groups: dict[str, str]


def _default_tensor_mover(
    tensor: torch.Tensor, target_device: torch.device
) -> torch.Tensor:
    return tensor.to(device=target_device)


class BranchAwareSynchronousStreamingWrapper(nn.Module):
    """Synchronously stream only the SenseNova branch required by each layer.

    The wrapped model must be in evaluation mode with all parameters and
    buffers on CPU.  Non-layer tensors retain the established SenseNova
    streaming contract and remain on ``target_device`` until :meth:`teardown`.
    Layer tensors are moved immediately before their forward and rebound to
    their exact original CPU storage immediately afterwards.

    ``tensor_mover`` exists to make the storage transition testable without a
    CUDA device.  Production callers should leave it unset.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        layers_attr: str,
        target_device: str | torch.device,
        tensor_mover: _TensorMover | None = None,
    ) -> None:
        super().__init__()
        if model.training:
            raise RuntimeError(
                "Branch-aware SenseNova streaming is inference-only; call model.eval() first."
            )

        self._model = model
        self._layers = self._resolve_layers(model, layers_attr)
        self._target_device = torch.device(target_device)
        self._tensor_mover = tensor_mover or _default_tensor_mover
        self._records = tuple(self._make_layer_record(layer) for layer in self._layers)
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._active_groups: dict[int, list[frozenset[str]]] = {
            id(record.module): [] for record in self._records
        }
        self._last_branch_flags: tuple[Any, Any] | None = None
        self._last_required_groups = ALL_TENSOR_GROUPS
        self._lock = threading.RLock()
        self._closed = False

        self._layer_transfer_bytes = {
            group: 0 for group in ALL_TENSOR_GROUPS
        }
        self._layer_forward_counts = {
            group: 0 for group in ALL_TENSOR_GROUPS
        }
        self._total_layer_forwards = 0
        self._non_layer_transfer_bytes = 0
        self._non_layer_cpu_data: list[tuple[torch.Tensor, torch.Tensor]] = []

        self._validate_cpu_sources()
        try:
            self._move_non_layer_tensors_to_target()
            self._register_hooks()
        except Exception:
            self._remove_hooks()
            self._restore_all_cpu_references()
            raise

    @staticmethod
    def _resolve_layers(model: nn.Module, dotted_path: str) -> nn.ModuleList:
        value: Any = model
        for part in dotted_path.split("."):
            value = getattr(value, part)
        if not isinstance(value, nn.ModuleList):
            raise TypeError(
                f"Expected nn.ModuleList at {dotted_path!r}, got {type(value).__name__}."
            )
        return value

    @staticmethod
    def _make_layer_record(layer: nn.Module) -> _LayerRecord:
        tensors: dict[str, torch.Tensor] = {}
        for name, tensor in _named_tensors(layer):
            if name in tensors:
                raise RuntimeError(f"Duplicate layer tensor path: {name!r}")
            tensors[name] = tensor
        groups = classify_layer_tensors(layer)
        if groups.keys() != tensors.keys():
            raise RuntimeError("SenseNova layer tensor classification is incomplete.")
        return _LayerRecord(
            module=layer,
            tensors=tensors,
            cpu_data={name: tensor.data for name, tensor in tensors.items()},
            groups=groups,
        )

    def _validate_cpu_sources(self) -> None:
        layer_tensor_ids = {
            id(tensor)
            for record in self._records
            for tensor in record.tensors.values()
        }
        seen: set[int] = set()
        tensors = itertools.chain(self._model.parameters(), self._model.buffers())
        for tensor in tensors:
            tensor_id = id(tensor)
            if tensor_id in seen:
                continue
            seen.add(tensor_id)
            if tensor.device.type != "cpu":
                scope = "layer" if tensor_id in layer_tensor_ids else "non-layer"
                raise RuntimeError(
                    f"SenseNova {scope} tensors must start on CPU, got {tensor.device}."
                )

    def _move_tensor(self, source: torch.Tensor) -> torch.Tensor:
        moved = self._tensor_mover(source, self._target_device)
        if not isinstance(moved, torch.Tensor):
            raise TypeError("SenseNova tensor mover must return a torch.Tensor.")
        device_matches = moved.device.type == self._target_device.type and (
            self._target_device.index is None
            or moved.device.index == self._target_device.index
        )
        if not device_matches:
            raise RuntimeError(
                f"SenseNova tensor mover returned {moved.device}, expected {self._target_device}."
            )
        if moved.shape != source.shape or moved.dtype != source.dtype:
            raise RuntimeError(
                "SenseNova tensor mover changed tensor shape or dtype."
            )
        return moved

    def _move_non_layer_tensors_to_target(self) -> None:
        layer_tensor_ids = {
            id(tensor)
            for record in self._records
            for tensor in record.tensors.values()
        }
        seen: set[int] = set()
        tensors = itertools.chain(self._model.parameters(), self._model.buffers())
        for tensor in tensors:
            tensor_id = id(tensor)
            if tensor_id in seen or tensor_id in layer_tensor_ids:
                continue
            seen.add(tensor_id)
            source = tensor.data
            tensor.data = self._move_tensor(source)
            self._non_layer_cpu_data.append((tensor, source))
            self._non_layer_transfer_bytes += source.numel() * source.element_size()

    def _move_layer_groups(
        self, record: _LayerRecord, groups: frozenset[str]
    ) -> dict[str, int]:
        moved_bytes = {group: 0 for group in ALL_TENSOR_GROUPS}
        moved_names: list[str] = []
        try:
            for name, tensor in record.tensors.items():
                group = record.groups[name]
                if group not in groups:
                    continue
                source = record.cpu_data[name]
                tensor.data = self._move_tensor(source)
                moved_names.append(name)
                moved_bytes[group] += source.numel() * source.element_size()
        except Exception:
            for name in moved_names:
                record.tensors[name].data = record.cpu_data[name]
            raise
        return moved_bytes

    @staticmethod
    def _restore_layer_groups(
        record: _LayerRecord, groups: frozenset[str]
    ) -> None:
        for name, tensor in record.tensors.items():
            if record.groups[name] in groups:
                tensor.data = record.cpu_data[name]

    def _register_hooks(self) -> None:
        record_by_id = {id(record.module): record for record in self._records}
        first_layer_id = id(self._records[0].module) if self._records else None

        def pre_hook(
            module: nn.Module,
            _args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> None:
            record = record_by_id[id(module)]
            flags = (
                kwargs.get("exist_non_image_gen_tokens"),
                kwargs.get("exist_image_gen_tokens"),
            )
            with self._lock:
                cached_flags = self._last_branch_flags
                if (
                    id(module) != first_layer_id
                    and cached_flags is not None
                    and flags[0] is cached_flags[0]
                    and flags[1] is cached_flags[1]
                ):
                    groups = self._last_required_groups
                else:
                    groups = required_tensor_groups(kwargs)
                    self._last_branch_flags = flags
                    self._last_required_groups = groups
            moved_bytes = self._move_layer_groups(record, groups)
            with self._lock:
                self._active_groups[id(module)].append(groups)
                self._total_layer_forwards += 1
                for group in groups:
                    self._layer_forward_counts[group] += 1
                for group, value in moved_bytes.items():
                    self._layer_transfer_bytes[group] += value

        def post_hook(
            module: nn.Module,
            _args: tuple[Any, ...],
            _kwargs: dict[str, Any],
            _output: Any,
        ) -> None:
            record = record_by_id[id(module)]
            with self._lock:
                stack = self._active_groups[id(module)]
                groups = stack.pop() if stack else ALL_TENSOR_GROUPS
            self._restore_layer_groups(record, groups)

        for record in self._records:
            self._hooks.append(
                record.module.register_forward_pre_hook(
                    pre_hook,
                    with_kwargs=True,
                )
            )
            self._hooks.append(
                record.module.register_forward_hook(
                    post_hook,
                    with_kwargs=True,
                    always_call=True,
                )
            )

    def _remove_hooks(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def _restore_all_cpu_references(self) -> None:
        for record in self._records:
            for name, tensor in record.tensors.items():
                tensor.data = record.cpu_data[name]
        for tensor, source in self._non_layer_cpu_data:
            tensor.data = source

    @property
    def telemetry(self) -> StreamingTelemetry:
        with self._lock:
            layer_bytes = dict(self._layer_transfer_bytes)
            forward_counts = dict(self._layer_forward_counts)
            non_layer_bytes = self._non_layer_transfer_bytes
            return StreamingTelemetry(
                total_transfer_bytes=non_layer_bytes + sum(layer_bytes.values()),
                non_layer_transfer_bytes=non_layer_bytes,
                layer_transfer_bytes_by_group=layer_bytes,
                total_layer_forwards=self._total_layer_forwards,
                layer_forward_counts_by_group=forward_counts,
            )

    @property
    def closed(self) -> bool:
        return self._closed

    def teardown(self) -> None:
        """Remove hooks and restore every exact CPU storage reference."""

        with self._lock:
            if self._closed:
                return
            self._remove_hooks()
            self._restore_all_cpu_references()
            for stack in self._active_groups.values():
                stack.clear()
            self._last_branch_flags = None
            self._closed = True

    def __enter__(self) -> BranchAwareSynchronousStreamingWrapper:
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> bool:
        self.teardown()
        return False

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._model(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)


__all__ = [
    "ALL_TENSOR_GROUPS",
    "BranchAwareSynchronousStreamingWrapper",
    "GROUP_GENERATION",
    "GROUP_SHARED",
    "GROUP_UNDERSTANDING",
    "StreamingTelemetry",
    "classify_layer_tensors",
    "required_tensor_groups",
]
