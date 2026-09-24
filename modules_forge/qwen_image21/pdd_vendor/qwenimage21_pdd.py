# ruff: noqa
"""Qwen-Image 2.1 glue for the shared PDD heads, plans and LoRA modules.

The native diffusers model/pipeline own loading, text conditioning, RoPE and VAE
decoding. Unlike Qwen-Image 20B, 2.1 uses CFG=1 and no guided norm rescaling.
"""

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21AttnProcessor, _qwenimage21_prepare_qkv,
)
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift
from safetensors.torch import load_file

from .lora_utils_pdd import (
    PDDParallelHead, add_pdd_lora, load_pdd_config, pdd_sampling_plan,
    pdd_state_dict, pdd_training_plan, resolve_pdd_lora_path,
)


PDD_DEFAULT_CONFIG = {
    "pdd_num_steps": 32,
    "pdd_block_size": 8,
    "lora_rank": 64,
    "lora_alpha": 64.0,
    "lora_targets": "to_q,to_k,to_v,to_out.0,img_mlp.proj,img_mlp.out,img_mlp.gate_layer",
    # Optional inference-only exports from the older full-parameter student.
    "pdd_export_format": None,
    "pdd_inference_only": False,
    "pdd_sigmas": None,
    "pdd_full_parameters": [],
    "pdd_sampling_precision": None,
    "pdd_sample_size": None,
}


class QwenImage21FlashAttnProcessor(QwenImage21AttnProcessor):
    """Local FA4 for unpadded T2I; reuse the native processor for other layouts.

Text attends causally to text; target image queries attend to all keys. Keeping
these two calls separate preserves the native block-causal attention exactly.
No torch.compile, flex mask compilation, or downloaded kernel is needed.
"""

    def __call__(self, attn, hidden_states, attention_mask=None, rotary_emb=None,
                 layer_cache=None, kv_cache_mode=None, cache_write_slice=None,
                 segments=None, key_valid=None):
        if (hidden_states.device.type != "cuda" or attention_mask is not None or key_valid is not None
                or (segments is not None and (len(segments) != 1 or segments[0][0] != 0 or not segments[0][2]))):
            return super().__call__(
                attn, hidden_states, attention_mask, rotary_emb, layer_cache,
                kv_cache_mode, cache_write_slice, segments, key_valid,
            )
        from flash_attn.cute import flash_attn_func

        query, key, value, seq_len = _qwenimage21_prepare_qkv(
            attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice,
        )
        def attend(q, k, v, causal=False):
            result = flash_attn_func(q, k, v, causal=causal)
            return result[0] if isinstance(result, tuple) else result

        if segments is None:
            output = attend(query, key, value)
        else:
            prefix = segments[0][1]
            text = attend(query[:, :prefix], key[:, :prefix], value[:, :prefix], causal=True)
            target = attend(query[:, prefix:], key, value)
            output = torch.cat([text, target], dim=1)
        output = output[:, :seq_len].flatten(2, 3).to(query.dtype)
        return attn.to_out[1](attn.to_out[0](output))


def set_attention_backend(transformer, backend):
    processor = QwenImage21FlashAttnProcessor if backend == "flash4" else QwenImage21AttnProcessor
    for block in transformer.transformer_blocks:
        block.attn.set_processor(processor())


def attach_parallel_decoder(transformer, num_steps):
    transformer.proj_out = PDDParallelHead(transformer.proj_out, num_steps).float()


def pdd_block_loss(student, teacher, latents, conditioning, sigmas, start, block_size, target, solver):
    """One on-policy block; detach teacher states and the next block (paper Eq. 11).

The caller backpropagates loss / number_of_blocks, and updates only after whole
independent trajectories have accumulated. No optimizer state lives here.
"""
    head = getattr(student, "module", student).proj_out
    step_sizes = sigmas.diff()
    head.set_plan(pdd_training_plan(step_sizes, start, [target], block_size))
    dtype = conditioning["encoder_hidden_states"].dtype

    def predict(model, state, sigma):
        timestep = (sigma * 1000).expand(state.shape[0]).to(dtype) / 1000
        return model(hidden_states=state.to(dtype), timestep=timestep, **conditioning, return_dict=False)[0][
            :, -state.shape[1]:
        ]

    output = predict(student, latents, sigmas[start]).unflatten(-1, (3, latents.shape[-1])).float()
    with torch.no_grad():
        target_state = latents.float() + output[:, :, 0].detach()
        target_velocity = predict(teacher, target_state, sigmas[target]).float()
        if solver == "midpoint":
            half_step = 0.5 * step_sizes[target]
            target_velocity = predict(
                teacher, target_state + half_step * target_velocity, sigmas[target] + half_step,
            ).float()
        next_latents = latents.float() + output[:, :, 2].detach()
    loss = F.mse_loss(output[:, :, 1], target_velocity)
    return loss, next_latents.detach()


def pdd_time_grid(scheduler, num_steps, image_seq_len, device="cpu"):
    """Exact native N-step sigmas, including dynamic shift, terminal stretch and zero."""
    config = scheduler.config
    if config.get("pdd_sigmas") is not None:
        grid = torch.as_tensor(config.pdd_sigmas, dtype=torch.float32, device=device)
        if (grid.shape != (num_steps + 1,) or not torch.isfinite(grid).all()
                or grid[0] != 1 or grid[-1] != 0 or not (grid.diff() < 0).all()):
            raise ValueError("Invalid stored PDD sigma grid.")
        return grid
    mu = calculate_shift(
        image_seq_len, config.get("base_image_seq_len", 256), config.get("max_image_seq_len", 4096),
        config.get("base_shift", 0.5), config.get("max_shift", 1.15),
    )
    scheduler.set_timesteps(sigmas=np.linspace(1.0, 1.0 / num_steps, num_steps), mu=mu, device=device)
    return scheduler.sigmas.clone()


class QwenImage21PDDScheduler(FlowMatchEulerDiscreteScheduler):
    """Euler on the exact block boundaries of the native N-step grid.

Register pdd_num_steps and pdd_block_size in the config before use. The native
pipeline supplies its resolution-dependent mu. Building an ordinary 4-step
schedule, or shifting already-shifted sigmas again, would use a different grid.
"""

    def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None, mu=None, timesteps=None):
        num_steps, block = self.config.pdd_num_steps, self.config.pdd_block_size
        requested = num_inference_steps if num_inference_steps is not None else len(sigmas)
        if timesteps is not None or requested != num_steps // block:
            raise ValueError(f"This checkpoint is trained for {num_steps // block} NFE.")
        expected = np.linspace(1.0, 1.0 / requested, requested)
        if sigmas is not None and not np.allclose(sigmas, expected):
            raise ValueError("Custom sigmas would change the trained PDD grid.")
        if self.config.get("pdd_sigmas") is not None:
            # Already shifted/stretched at conversion time; never shift twice.
            self.pdd_sigmas = pdd_time_grid(self, num_steps, 0)
            self.sigmas = self.pdd_sigmas[::block].clone()
            self.timesteps = (self.sigmas[:-1] * self.config.num_train_timesteps).to(device)
            self.num_inference_steps = requested
            self._step_index = self._begin_index = None
            return
        super().set_timesteps(sigmas=np.linspace(1.0, 1.0 / num_steps, num_steps), mu=mu, device=device)
        self.pdd_sigmas = self.sigmas.clone()
        self.sigmas = self.sigmas[::block].clone()
        self.timesteps = self.timesteps[::block].clone()
        self.num_inference_steps = requested


def pdd_step_callback(transformer, sigmas, block_size):
    """Arm block-mean heads while the native pipeline performs Euler and decoding."""
    step_sizes = sigmas.diff()

    def arm(index):
        transformer._pdd_block_start = index * block_size
        transformer.proj_out.set_plan(pdd_sampling_plan(step_sizes, index * block_size, block_size))

    arm(0)

    def callback(pipe, step_index, timestep, callback_kwargs):
        if (step_index + 1) * block_size < len(step_sizes):
            arm(step_index + 1)
        return {}

    return callback


def load_pdd_lora(transformer, path):
    path = resolve_pdd_lora_path(path)
    config = load_pdd_config(path, defaults=PDD_DEFAULT_CONFIG)
    converted = config["pdd_export_format"] == "qwenimage21_extracted_prefused_v1"
    if config["pdd_export_format"] is not None and not converted:
        raise ValueError("Unknown PDD export format.")
    if converted and (not config["pdd_inference_only"] or config["pdd_block_size"] != 1
                      or config["pdd_sampling_precision"] != "native_time_fp32_state"):
        raise ValueError("A converted bundle requires inference-only, single-interval fused heads.")
    if isinstance(transformer.proj_out, PDDParallelHead):
        raise ValueError("Load PDD on an unmodified base transformer.")
    dtype = transformer.proj_out.weight.dtype
    transformer.requires_grad_(False)
    add_pdd_lora(transformer, config["lora_targets"].split(","), config["lora_rank"], config["lora_alpha"])
    attach_parallel_decoder(transformer, config["pdd_num_steps"])
    state = load_file(path)
    full_names = set(config["pdd_full_parameters"])
    for name in full_names:
        parameter = transformer.get_parameter(name)
        if name not in state or state[name].shape != parameter.shape or parameter.ndim == 2:
            raise ValueError(f"Invalid full parameter: {name}")
    expected = set(pdd_state_dict(transformer)) | full_names
    if set(state) != expected:
        raise ValueError(f"PDD checkpoint keys differ: missing={expected - set(state)}, extra={set(state) - expected}")
    transformer.load_state_dict(state, strict=False)
    if converted:
        # Offline fusion already rounded each effective Linear to the source dtype.
        # The original shared head now selects one of these with a one-hot plan.
        transformer.proj_out.to(dtype=dtype)
        transformer.requires_grad_(False)
        transformer._pdd_block_start = 0
        grid = torch.as_tensor(config["pdd_sigmas"], dtype=torch.float32)

        def native_time(module, positional, kwargs):
            if module.training:
                raise ValueError("Converted PDD exports are inference-only; use the original training checkpoint.")
            embeds = kwargs["encoder_hidden_states"]
            sigma = grid[module._pdd_block_start].to(embeds.device)
            kwargs["timestep"] = ((sigma * 1000).to(embeds.dtype) / 1000).expand(embeds.shape[0])
            kwargs["hidden_states"] = kwargs["hidden_states"].to(embeds.dtype)
            return positional, kwargs

        def fp32_prediction(module, positional, output):
            return (output[0].float(), *output[1:])

        transformer.register_forward_pre_hook(native_time, with_kwargs=True)
        transformer.register_forward_hook(fp32_prediction)
    return config
