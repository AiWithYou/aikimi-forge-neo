from abc import abstractmethod
from typing import Optional

import torch

from backend import memory_management


class WeightAdapterBase:
    name: str
    loaded_keys: set[str]
    weights: list[torch.Tensor]

    @classmethod
    def load(cls, x: str, lora: dict[str, torch.Tensor], alpha: float, dora_scale: torch.Tensor) -> Optional["WeightAdapterBase"]:
        raise NotImplementedError

    def to_train(self):
        raise NotImplementedError

    @classmethod
    def create_train(cls, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def calculate_weight(
        self,
        weight,
        key,
        strength,
        strength_model,
        offset,
        function,
        intermediate_dtype=torch.float32,
        original_weight=None,
    ):
        raise NotImplementedError


def weight_decompose(dora_scale, weight, lora_diff, alpha, strength, intermediate_dtype, function):
    dora_scale = memory_management.cast_to_device(dora_scale, weight.device, intermediate_dtype)
    lora_diff = lora_diff * alpha
    weight_orig = memory_management.cast_to_device(weight, weight.device, intermediate_dtype)
    weight_calc = weight_orig + function(lora_diff).to(device=weight.device, dtype=intermediate_dtype)

    wd_on_output_axis = dora_scale.shape[0] == weight_calc.shape[0] and dora_scale.numel() == dora_scale.shape[0]
    if wd_on_output_axis:
        weight_norm = weight_calc.reshape(weight_calc.shape[0], -1).norm(dim=1, keepdim=True).reshape(weight_calc.shape[0], *[1] * (weight_calc.dim() - 1))
    else:
        weight_norm = weight_calc.transpose(0, 1).reshape(weight_calc.shape[1], -1).norm(dim=1, keepdim=True).reshape(weight_calc.shape[1], *[1] * (weight_calc.dim() - 1)).transpose(0, 1)
    weight_norm = weight_norm + torch.finfo(intermediate_dtype).eps

    weight_calc *= dora_scale.reshape_as(weight_norm) / weight_norm
    if strength != 1.0:
        weight_calc = weight_orig + strength * (weight_calc - weight_orig)
    weight.copy_(weight_calc)
    return weight


def pad_tensor_to_shape(tensor: torch.Tensor, new_shape: list[int]) -> torch.Tensor:
    """
    Pad a tensor to a new shape with zeros.

    Args:
        tensor (torch.Tensor): The original tensor to be padded.
        new_shape (List[int]): The desired shape of the padded tensor.

    Returns:
        torch.Tensor: A new tensor padded with zeros to the specified shape.

    Raises:
        ValueError: If the rank differs or any dimension would shrink.
    """
    if len(new_shape) != len(tensor.shape):
        raise ValueError("The new shape must have the same number of dimensions as the original tensor")

    if any(new < old for new, old in zip(new_shape, tensor.shape)):
        raise ValueError("The new shape must be larger than the original tensor in all dimensions")

    # Create a new tensor filled with zeros
    padded_tensor = torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.device)

    # Copy the original data into the matching region of the padded tensor.
    orig_slices = tuple(slice(0, dim) for dim in tensor.shape)
    padded_tensor[orig_slices] = tensor

    return padded_tensor
