"""Request-scoped Krea2 attention routing, including repeated upscale passes.

Q/K normalization, RoPE, GQA expansion, sigmoid gating and output projection stay
in Krea2. Text/reference KV and query rows use the kernel's exact sink ranges.
"""

from __future__ import annotations

import contextvars
import importlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .anima import AnimaRun
from .common import AnimaOptions, ExperimentCancelled, JevClient, RunLog, sdk_python
from .credentials import cloud_source

_ACTIVE = contextvars.ContextVar("aikimi_krea2_active", default=None)
KEEP_CHOICES = (1.0, 3.0, 5.0, 10.0, 25.0, 50.0, 100.0)


@dataclass(frozen=True)
class Options(AnimaOptions):
    keep_percent: float = 10.0
    min_tokens: int = 4096
    timeout: float = 10.0
    tile_mode: str = "off"
    decision_cadence: str = "once"
    tile_cadence: str = "once"

    def validate(self):
        super().validate()
        if self.tile_mode not in {"off", "rules", "jev"}:
            raise ValueError("Unknown Krea2 tile allocation mode")
        if self.max_calls != 1:
            raise ValueError("Krea2 uses decision_cadence instead of max_calls")
        if self.decision_cadence not in {"once", "interval", "step"}:
            raise ValueError("Unknown Krea2 decision cadence")
        if self.tile_cadence not in {"once", "stage"}:
            raise ValueError("Unknown Krea2 tile decision cadence")


class KreaRun(AnimaRun):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layout = None
        self.layouts = set()
        self.sampling_pass = -1
        self.sampling_step = -1
        self.clock_ready = False
        self.last_decision_step = None

    @property
    def repeating(self):
        return self.options.mode == "jev" and self.options.decision_cadence != "once"

    def begin_sampling(self):
        self.sampling_pass += 1
        self.sampling_step = -1
        self.clock_ready = False
        self.last_decision_step = None
        if self.repeating:
            # A new tile/pass must not be assessed using another tile's statistics.
            self.observations.clear()
            self.previous.clear()
            self.pending.clear()

    def begin_step(self):
        # Called before CFG batching, once per sampler denoiser evaluation.
        self.clock_ready = True
        self.sampling_step += 1

    def begin_evaluation(self):
        if self.cancelled():
            raise ExperimentCancelled("Generation cancelled")
        self.evaluation += 1
        if not self.clock_ready:
            self.sampling_step = self.evaluation
        self.counts.clear()
        self.pending.clear()
        interval = self.options.update_interval if self.options.decision_cadence == "interval" else 1
        if (
            self.evaluation < self.options.warmup_evaluations
            or self.sampling_step < self.options.warmup_evaluations
            or self.options.mode not in {"jev", "rules"}
            or self.circuit_open
            or len(self.observations) != self.layers
            or (not self.repeating and self.last_decision is not None)
            or (self.last_decision_step is not None and self.sampling_step - self.last_decision_step < interval)
        ):
            return
        self.last_decision = self.evaluation
        self.last_decision_step = self.sampling_step
        if self.options.mode == "rules":
            order = sorted(self.observations, key=lambda i: self.observations[i]["relative_output_norm"])
            self.keeps = {
                layer: 3.0 if rank < self.layers / 3 else 10.0 if rank < 2 * self.layers / 3 else 25.0
                for rank, layer in enumerate(order)
            }
            self.log.write("decision", evaluation=self.evaluation, source="rules", keep_percent=self.keeps)
            return
        try:
            self.keeps = self.client.decide(
                {
                    "target": "Krea2 image-to-image and text-to-image joint attention",
                    "constraints": "SPEED-FIRST: choose KEEP percentages 1,3,5,10,25,50,100 separately for all layers. Text and reference-image key/value blocks AND query rows remain exact. All layers, projections, gates and MLPs run. Rank attention output norm and drift against peers: use 1 or 3 for weak contributions, 5 or 10 for typical contributions, 25 or 50 for strong contributions, reserve 100 for extreme contributions. Changed detail/composition is acceptable; preserve plausible, useful images. Proxies are not measured visual quality. Missing first drift is normal. No forced quotas. Apply this choice until the next scheduled decision; once cadence reuses it across remaining steps and upscale tiles.",
                    "next_model_evaluation": self.evaluation,
                    "sampling_pass": self.sampling_pass,
                    "sampling_step": self.sampling_step,
                    "decision_cadence": self.options.decision_cadence,
                    "blocks": self.observations,
                    "current_keep": self.keeps,
                },
                {str(i): KEEP_CHOICES for i in range(self.layers)},
                100.0,
            )
            self.log.write(
                "decision",
                evaluation=self.evaluation,
                sampling_pass=self.sampling_pass,
                sampling_step=self.sampling_step,
                source="jev",
                keep_percent=self.keeps,
                api_calls=self.client.calls,
                api_wait_seconds=self.client.wait_seconds,
                diagnostics=self.client.last_diagnostics,
                observations=self.observations,
            )
        except ExperimentCancelled:
            raise
        except Exception as exc:
            self.circuit_open = True
            self.keeps = {str(i): 100.0 for i in range(self.layers)}
            self.log.write("decision", source="dense_fallback", error_type=type(exc).__name__, keep_percent=self.keeps)

    def wants_observations(self):
        return (
            self.options.mode in {"jev", "rules"}
            and not self.circuit_open
            and (self.repeating or self.last_decision is None)
        )

    def keep(self, layer):
        if self.repeating and self.sampling_step < self.options.warmup_evaluations:
            return 100.0
        return super().keep(layer)

    def end_evaluation(self):
        if self.repeating:
            # Unsupported/masked evaluations must not leave stale observations.
            self.observations.clear()
        super().end_evaluation()


def sparse_attention(q, k, v, keep, protected_tokens, *, kernel=None):
    import torch

    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or q.shape[-1] != 128:
        raise ValueError("Krea2 sparse attention requires matching B,H,N,128 tensors")
    if not 0 <= protected_tokens <= q.shape[2]:
        raise ValueError("Invalid Krea2 conditioning range")
    ck = kernel or importlib.import_module("comfy_kitchen")
    if q.device.type != "cuda" or not ck.sol_attn_is_available(q.device):
        raise RuntimeError("Krea2 Sparseには対応するCUDA版Comfy-Kitchenが必要です。")
    original_dtype = q.dtype
    if (
        original_dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or k.dtype != original_dtype
        or v.dtype != original_dtype
    ):
        raise ValueError("Unsupported Krea2 attention dtype")
    return blocked_attention(q, k, v, keep, (protected_tokens + 63) // 64, ck)


def blocked_attention(q, k, v, keep, protected_blocks, kernel):
    """Bound each native launch and explicitly mask a partial 64-token block.

    Attention heads are independent. Krea2's 48 heads are processed as three
    groups; this avoids the large mixed-shape launch that stalled in a 3090
    tiled run. Padded keys are excluded by block_len, never treated as live KV.
    """
    import torch
    from torch.nn import functional as F

    batch, heads, tokens, _ = q.shape
    dtype = torch.bfloat16 if q.dtype == torch.float32 else q.dtype
    padding = (-tokens) % 64
    block_len = None
    if padding:
        block_len = torch.full(((tokens + 63) // 64,), 64, dtype=torch.int32, device=q.device)
        block_len[-1] = tokens % 64
    outputs = []
    for first in range(0, heads, 16):
        tensors = [t[:, first : first + 16].transpose(1, 2).to(dtype) for t in (q, k, v)]
        tensors = [(F.pad(t, (0, 0, 0, 0, 0, padding)) if padding else t).contiguous() for t in tensors]
        out = kernel.sol_attn(
            *tensors,
            tau=1.3,
            sink_blocks=[0, protected_blocks],
            sink_q=[0, protected_blocks],
            topk_ratio=keep / 100.0,
            tail=True,
            token_aug=0,
            block_len=block_len,
        )
        outputs.append(out[:, :tokens])
    return torch.cat(outputs, dim=2).to(q.dtype).reshape(batch, tokens, -1)


def attention_override(run, kernel_fn=sparse_attention):
    def compute(q, k, v, heads, mask, options, dense):
        def normal():
            return dense(q, k, v, heads, mask=mask, skip_reshape=True, transformer_options=options)

        if _ACTIVE.get() is not run or run.closed:
            return normal()
        layer = options["krea2_block_index"]
        protected = options["krea2_text_tokens"] + options["krea2_reference_tokens"]
        image_tokens = options["krea2_image_tokens"]
        if protected < 0 or image_tokens <= 0 or protected + image_tokens != q.shape[2]:
            raise ValueError("Krea2 token layout mismatch")
        layout = (protected, image_tokens)
        if layout not in run.layouts:
            run.layouts.add(layout)
            run.log.write(
                "layout", protected_tokens=protected, image_tokens=image_tokens, protected_blocks=(protected + 63) // 64
            )
        keep = run.keep(layer)
        if mask is not None:
            reason, result = "dense_mask", normal()
        elif image_tokens < run.options.min_tokens:
            reason, result = "dense_short_sequence", normal()
        elif keep >= 100:
            reason, result = "dense", normal()
        else:
            reason, result = "sparse", kernel_fn(q, k, v, keep, protected)
        run.counts[reason] += 1
        if mask is None:
            run.observe(layer, result[:, protected:], v[:, :, protected:].transpose(1, 2))
        return result

    return compute


def new_run(options, layers, log_root, *, prompt="", cancelled=None):
    client = (
        JevClient(sdk_python(), options.timeout, cancelled, environment=cloud_source())
        if options.mode == "jev"
        else None
    )
    settings = {**asdict(options), "kernel_head_group": 16, "kernel_token_block": 64}
    settings.pop("max_calls")
    settings["api_call_limit"] = 1 if options.decision_cadence == "once" else "sampling_cadence"
    return KreaRun(options, layers, RunLog(Path(log_root), "krea2", settings, prompt), client, cancelled)


def attach(unet, options, log_root, *, run=None, prompt="", cancelled=None):
    options.validate()
    if options.mode == "off":
        return unet, run
    model = unet.get_model_object("diffusion_model")
    if type(model).__name__ != "SingleStreamDiT" or type(model).__module__ != "backend.nn.krea":
        raise ValueError("Krea2 SparseはKrea2専用です。")
    if not model.blocks or any(b.attn.headdim != 128 for b in model.blocks):
        raise ValueError("Unsupported Krea2 attention structure")
    if options.mode != "dense":
        ck = importlib.import_module("comfy_kitchen")
        if not ck.sol_attn_is_available(unet.load_device):
            raise RuntimeError("このGPUではKrea2 Sparseを利用できません。")
    if run is not None and (run.closed or run.layers != len(model.blocks) or run.options != options):
        raise ValueError("Krea2 generation session changed during sampling")
    patched = unet.clone()
    transformer_options = patched.model_options.setdefault("transformer_options", {})
    if transformer_options.get("krea2_attention_override") is not None:
        raise ValueError("Another Krea2 attention override is already installed")
    run = run or new_run(options, len(model.blocks), log_root, prompt=prompt, cancelled=cancelled)
    run.begin_sampling()
    transformer_options["krea2_attention_override"] = attention_override(run)
    previous = patched.model_options.get("model_function_wrapper")

    def step_clock(model, x, timestep, uncond, cond, cond_scale, model_options, seed):
        run.begin_step()
        return model, x, timestep, uncond, cond, cond_scale, model_options, seed

    patched.model_options["conditioning_modifiers"] = [
        *patched.model_options.get("conditioning_modifiers", []),
        step_clock,
    ]

    def wrapper(apply_model, args):
        def delegate():
            return (
                previous(apply_model, args) if previous else apply_model(args["input"], args["timestep"], **args["c"])
            )

        if run.closed:
            return delegate()
        import torch

        device = args["input"].device

        def sync():
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        sync()
        started = time.perf_counter()
        token = _ACTIVE.set(run)
        try:
            run.begin_evaluation()
            result = delegate()
            run.end_evaluation()
            sync()
            run.model_seconds += time.perf_counter() - started
            return result
        except BaseException as exc:
            run.close("cancelled" if isinstance(exc, ExperimentCancelled) or run.cancelled() else "failed")
            raise
        finally:
            _ACTIVE.reset(token)

    patched.set_model_unet_function_wrapper(wrapper)
    return patched, run
