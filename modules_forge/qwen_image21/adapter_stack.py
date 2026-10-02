"""Compose sampling, outpaint and style residuals without changing base files."""

from pathlib import Path


def materialize_pdd_projection(transformer):
    """PDD copies its decoder heads; only this small GGUF Linear must be dense."""
    import torch
    from torch import nn

    source = transformer.proj_out
    weight = source.weight
    if not hasattr(weight, "quant_type"):
        if not weight.is_floating_point() or tuple(weight.shape) != (source.out_features, source.in_features):
            raise ValueError("Fun Accの出力層は論理寸法に一致する浮動小数点重みが必要です。")
        return
    from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor

    dense = dequantize_gguf_tensor(weight).to(dtype=torch.bfloat16)
    if tuple(dense.shape) != (source.out_features, source.in_features) or not torch.isfinite(dense).all():
        raise ValueError("GGUF出力層をFun Acc用に復元できません。")
    projection = nn.Linear(
        source.in_features, source.out_features, bias=source.bias is not None, device=dense.device, dtype=dense.dtype
    )
    with torch.no_grad():
        projection.weight.copy_(dense)
        if source.bias is not None:
            bias = source.bias
            if hasattr(bias, "quant_type"):
                bias = dequantize_gguf_tensor(bias)
            projection.bias.copy_(bias.to(device=dense.device, dtype=dense.dtype))
    projection.requires_grad_(False)
    transformer.proj_out = projection


def load_stack(pipe, runtime: Path, request: dict, progress):
    """PDD sees native Linear targets; later residuals can wrap those targets."""
    fun_acc, config, outpaint, styles = None, None, None, []
    if request.get("fun_acc"):
        from .fun_acc_lora import installed
        from .pdd_vendor.qwenimage21_pdd import QwenImage21PDDScheduler, load_pdd_lora

        fun_acc = installed(runtime)
        progress("Fun Acc 4-step LoRAを読み込み中")
        materialize_pdd_projection(pipe.transformer)
        config = load_pdd_lora(pipe.transformer, fun_acc["path"])
        pipe.scheduler = QwenImage21PDDScheduler.from_config(pipe.scheduler.config)
        pipe.scheduler.register_to_config(**config)
    if request.get("outpaint_version") in {"v1", "v2"}:
        from .outpaint_lora import installed
        from .outpaint_runtime import load_adapter

        outpaint = installed(runtime, request["outpaint_version"], verify=True)
        progress("Outpaint LoRAを読み込み中")
        outpaint.update(load_adapter(pipe.transformer, outpaint["path"]))
    if request.get("style_loras"):
        from .style_lora_runtime import load_adapters

        progress("追加LoRAを読み込み中")
        styles = load_adapters(pipe.transformer, runtime, request)
    if fun_acc or outpaint or styles:
        pipe.transformer.eval()
    return styles, fun_acc, config, outpaint
