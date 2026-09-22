"""ConvRot W4A8 for the isolated Qwen pipeline, using Comfy Kitchen kernels.

Only block attention/MLP projections are converted. The vision tower, embeddings,
modulation, normalization, output heads and VAE retain their original precision.
Packed weights and all scales travel together as a QuantizedTensor parameter.
"""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import torch
from torch import nn

KITCHEN_VERSION = "0.2.31"
LAYOUT = "AsymW4A8Int8Layout"


def validate_dependency():
    try:
        installed = importlib.metadata.version("comfy-kitchen")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed != KITCHEN_VERSION:
        raise RuntimeError(
            f"W4A8には専用環境のcomfy-kitchen=={KITCHEN_VERSION}が必要です。"
            "aikimi-qwen-image21-setup.bat --runtime-onlyを実行してください。"
        )


class W4A8Linear(nn.Linear):
    def __init__(self, weight, bias):
        nn.Module.__init__(self)
        self.out_features, self.in_features = weight.shape
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None

    def forward(self, inputs):
        import comfy_kitchen as ck

        # A missing CUDA kernel must not silently become a full BF16 GEMM.
        with ck.use_backend("cuda" if inputs.is_cuda else "eager"):
            return nn.functional.linear(inputs, self.weight, self.bias)


def target_linears(model, component):
    prefixes = {"transformer": "transformer_blocks.", "text_encoder": "model.language_model.layers."}
    if component not in prefixes:
        raise ValueError(f"Unsupported W4A8 component: {component}")
    return [
        (name, layer)
        for name, layer in model.named_modules()
        if name.startswith(prefixes[component])
        and isinstance(layer, nn.Linear)
        and not isinstance(layer, W4A8Linear)
        and "modulation" not in name.split(".")
        and layer.in_features % 256 == 0
    ]


def _pack_weight(weight, device):
    import comfy_kitchen as ck
    from comfy_kitchen.tensor import QuantizedTensor

    with ck.use_backend("cuda" if device.type == "cuda" else "eager"):
        return QuantizedTensor.from_float(
            weight.detach().to(device=device),
            LAYOUT,
            group_size=16,
            convrot_groupsize=256,
            symmetric=True,
            codebook=True,
            scale_dtype=torch.float8_e4m3fn,
        ).to("cpu")


@torch.no_grad()
def load_model(folder, component, factory, *, device="cuda:0", dtype=torch.bfloat16, progress=None):
    """Materialize a meta skeleton from safetensors, packing each weight at once.

    Nonpersistent buffers (including RoPE frequencies) are built on CPU normally.
    Validate all keys/shapes before the first GPU copy; never zero-fill omissions.
    """
    from accelerate import init_empty_weights
    from safetensors import safe_open

    folder = Path(folder).resolve()
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.get_device_capability(device) < (8, 0):
        raise RuntimeError("W4A8 CUDAにはAmpere以降（compute capability 8.0以上）が必要です。")
    with init_empty_weights(include_buffers=False):
        model = factory()
    model.eval().requires_grad_(False)
    expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    parameters = set(dict(model.named_parameters()))
    targets = {name for name, _layer in target_linears(model, component)}
    if not targets:
        raise RuntimeError(f"{component} has no compatible W4A8 block projections.")
    indexes = sorted(folder.glob("*.safetensors.index.json"))
    if len(indexes) > 1:
        raise ValueError("Multiple safetensors indexes in Qwen component")
    if indexes:
        index = json.loads(indexes[0].read_text(encoding="utf-8"))
        filenames = sorted(set(index["weight_map"].values()))
    else:
        filenames = [path.name for path in sorted(folder.glob("*.safetensors"))]
    inventory = {}
    for filename in filenames:
        path = (folder / filename).resolve()
        if not path.is_relative_to(folder) or not path.is_file():
            raise ValueError("Missing or escaping safetensors shard")
        with safe_open(path, framework="pt", device="cpu") as source:
            for name in source.keys():
                if name in inventory or name not in expected:
                    raise ValueError(f"Unexpected or duplicate Qwen weight: {name}")
                if tuple(source.get_slice(name).get_shape()) != expected[name]:
                    raise ValueError(f"Qwen weight shape mismatch: {name}")
                inventory[name] = path
    missing = expected.keys() - inventory.keys()
    if missing:
        raise ValueError(f"Missing Qwen weights: {', '.join(sorted(missing)[:5])}")
    source_bytes = packed_bytes = packed_count = 0
    for path in dict.fromkeys(inventory.values()):
        with safe_open(path, framework="pt", device="cpu") as source:
            for name in source.keys():
                if progress is not None:
                    progress(packed_count, len(targets))
                parent_name, _, field = name.rpartition(".")
                parent = model.get_submodule(parent_name)
                tensor = source.get_tensor(name)
                if name in parameters and tensor.is_floating_point():
                    tensor = tensor.to(dtype=dtype)
                if field == "weight" and parent_name in targets:
                    source_bytes += tensor.nbytes
                    packed = _pack_weight(tensor, device)
                    packed_bytes += sum(value.nbytes for value in packed.state_dict().values())
                    grandparent_name, _, child_name = parent_name.rpartition(".")
                    replacement = W4A8Linear(packed, parent.bias).eval()
                    setattr(model.get_submodule(grandparent_name), child_name, replacement)
                    packed_count += 1
                elif name in parameters:
                    # A small unquantized tensor must not retain the mapping of
                    # an entire multi-GB BF16 shard after the reader closes.
                    parent._parameters[field] = nn.Parameter(tensor.clone(), requires_grad=False)
                else:
                    parent._buffers[field] = tensor.clone()
                del tensor
    if any(value.is_meta for value in (*model.parameters(), *model.buffers())):
        raise RuntimeError("Unmaterialized Qwen tensor after W4A8 loading")
    if progress is not None:
        progress(packed_count, len(targets))
    return model, {"layers": packed_count, "source_weight_bytes": source_bytes, "packed_weight_bytes": packed_bytes}


@torch.no_grad()
def convert_model(model, component, *, device="cuda:0", progress=None):
    """Pack one Linear at a time on the GPU, retaining only packed CPU weights."""
    selected = target_linears(model, component)
    if not selected:
        raise RuntimeError(f"{component} has no compatible W4A8 block projections.")
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.get_device_capability(device) < (8, 0):
        raise RuntimeError("W4A8 CUDAにはAmpere以降（compute capability 8.0以上）が必要です。")
    source_bytes = packed_bytes = 0
    for index, (name, layer) in enumerate(selected):
        if progress is not None:
            progress(index, len(selected))
        source_bytes += layer.weight.nbytes
        packed = _pack_weight(layer.weight, device)
        packed_bytes += sum(tensor.nbytes for tensor in packed.state_dict().values())
        parent_name, _, child_name = name.rpartition(".")
        replacement = W4A8Linear(packed, layer.bias.detach().cpu() if layer.bias is not None else None)
        replacement.train(layer.training)
        setattr(model.get_submodule(parent_name), child_name, replacement)
        # The selection list must not keep each replaced BF16 layer alive.
        selected[index] = (name, None)
    if progress is not None:
        progress(len(selected), len(selected))
    return {"layers": len(selected), "source_weight_bytes": source_bytes, "packed_weight_bytes": packed_bytes}


@torch.no_grad()
def save_model(model, folder, stats, *, check_cancel=lambda: None, max_shard_bytes=512 * 1024**2):
    """Store packed bytes/scales and BF16 exceptions without dequantization."""
    from comfy_kitchen.tensor import QuantizedTensor
    from safetensors.torch import save_file

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    weights, weight_map, shard = {}, {}, {}
    shard_size = 0
    shard_number = 0

    def flush():
        nonlocal shard_size, shard_number
        if not shard:
            return
        check_cancel()
        filename = f"weights-{shard_number:05d}.safetensors"
        save_file(shard, str(folder / filename))
        weight_map.update(dict.fromkeys(shard, filename))
        shard.clear()
        shard_number += 1
        shard_size = 0

    for name, tensor in model.state_dict().items():
        check_cancel()
        if isinstance(tensor, QuantizedTensor):
            params = tensor._params
            if tensor._layout_cls != LAYOUT or params.transposed:
                raise ValueError("Unsupported packed Qwen weight layout")
            group = tensor.state_dict(name)
            weights[name] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "group_size": params.group_size,
                "convrot_groupsize": params.convrot_groupsize,
                "keys": list(group),
            }
        else:
            group = {name: tensor}
        size = sum(value.nbytes for value in group.values())
        if shard and shard_size + size > max_shard_bytes:
            flush()
        # Clone each group so tied/aliased source storage is serialized safely.
        shard.update({key: value.detach().cpu().contiguous().clone() for key, value in group.items()})
        shard_size += size
    flush()
    (folder / "w4a8.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "layout": LAYOUT,
                "weights": weights,
                "weight_map": weight_map,
                "stats": stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


@torch.no_grad()
def load_saved_model(folder, component, factory, *, check_cancel=lambda: None):
    """Restore the exact packed representation into a freshly built skeleton."""
    from accelerate import init_empty_weights
    from comfy_kitchen.tensor import AsymW4A8Int8Layout, QuantizedTensor
    from safetensors import safe_open

    folder = Path(folder).resolve()
    record = json.loads((folder / "w4a8.json").read_text(encoding="utf-8"))
    if record.get("schema") != 1 or record.get("layout") != LAYOUT:
        raise ValueError("Unsupported W4A8 checkpoint format")
    with init_empty_weights(include_buffers=False):
        model = factory()
    model.eval().requires_grad_(False)
    expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    parameters = set(dict(model.named_parameters()))
    targets = {f"{name}.weight" for name, _layer in target_linears(model, component)}
    packed_weights = record["weights"]
    if targets != set(packed_weights):
        raise ValueError("W4A8 checkpoint target layers mismatch")
    physical_keys = set(expected) - targets
    for name, spec in packed_weights.items():
        if tuple(spec["shape"]) != expected[name] or spec["dtype"] not in {"float32", "float16", "bfloat16"}:
            raise ValueError(f"W4A8 checkpoint shape/dtype mismatch: {name}")
        required = {name, name + "_s_rel", name + "_s_channel"}
        allowed = required | {name + "_correction", name + "_codebook"}
        if not required.issubset(spec["keys"]) or not set(spec["keys"]).issubset(allowed):
            raise ValueError(f"Incomplete W4A8 scales: {name}")
        physical_keys.update(spec["keys"])
    if physical_keys != set(record["weight_map"]):
        raise ValueError("W4A8 checkpoint inventory mismatch")
    loaded = set()
    for filename in dict.fromkeys(record["weight_map"].values()):
        check_cancel()
        path = (folder / filename).resolve()
        if not path.is_relative_to(folder):
            raise ValueError("Escaping W4A8 shard path")
        with safe_open(path, framework="pt", device="cpu") as source:
            keys = set(source.keys())
            if keys != {key for key, file in record["weight_map"].items() if file == filename} or loaded & keys:
                raise ValueError("Unexpected W4A8 shard keys")
            loaded.update(keys)
            for name in expected.keys() & keys:
                check_cancel()
                parent_name, _, field = name.rpartition(".")
                parent = model.get_submodule(parent_name)
                tensor = source.get_tensor(name).clone()
                if name in targets:
                    spec = packed_weights[name]
                    if not set(spec["keys"]).issubset(keys):
                        raise ValueError("Packed weight and scales must share a shard")
                    n, k = spec["shape"]
                    if tuple(tensor.shape) != (n, k // 2) or tensor.dtype != torch.int8:
                        raise ValueError(f"Invalid packed W4A8 storage: {name}")

                    def scale(suffix, name=name, spec=spec):
                        key = name + suffix
                        return source.get_tensor(key).clone() if key in spec["keys"] else None

                    params = AsymW4A8Int8Layout.Params(
                        scale=scale("_s_rel"),
                        s_channel=scale("_s_channel"),
                        correction=scale("_correction"),
                        codebook=scale("_codebook"),
                        orig_dtype=getattr(torch, spec["dtype"]),
                        orig_shape=tuple(spec["shape"]),
                        group_size=spec["group_size"],
                        convrot_groupsize=spec["convrot_groupsize"],
                    )
                    packed = QuantizedTensor(tensor, LAYOUT, params)
                    grandparent, _, child = parent_name.rpartition(".")
                    setattr(model.get_submodule(grandparent), child, W4A8Linear(packed, parent.bias).eval())
                else:
                    if tuple(tensor.shape) != expected[name]:
                        raise ValueError(f"W4A8 exception tensor shape mismatch: {name}")
                    if name in parameters:
                        parent._parameters[field] = nn.Parameter(tensor, requires_grad=False)
                    else:
                        parent._buffers[field] = tensor
    if loaded != physical_keys or any(value.is_meta for value in (*model.parameters(), *model.buffers())):
        raise ValueError("Incomplete saved W4A8 model")
    return model, record["stats"]
