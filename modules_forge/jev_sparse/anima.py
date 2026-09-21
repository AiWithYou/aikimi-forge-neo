"""Anima self-attention only. Object patches live on a cloned Forge patcher.

Kernel contract: ComfyUI 7a0b5eed nodes_sparse_attention.py, make_attention_override.
The existing Q/K/V projections, norms, RoPE, output projection and cross-attention
are preserved. No global attention function is replaced.
"""
from __future__ import annotations

import contextvars
import importlib
import math
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .common import AnimaOptions, ExperimentCancelled, JevClient, RunLog, sdk_python

_ACTIVE = contextvars.ContextVar("aikimi_anima_sparse_run", default=None)


class AnimaRun:
    def __init__(self, options: AnimaOptions, layers: int, log: RunLog, client=None, cancelled=None):
        options.validate()
        self.options, self.layers, self.log = options, layers, log
        self.client = client
        self.cancelled = cancelled or (lambda: False)
        self.keeps = {str(i): float(options.keep_percent) for i in range(layers)}
        if options.mode in {"dense", "rules", "jev"}:
            self.keeps = {str(i): 100.0 for i in range(layers)}
        self.evaluation = -1
        self.last_decision = None
        self.previous = {}
        self.pending = {}
        self.observations = {}
        self.counts = Counter()
        self.total_counts = Counter()
        self.circuit_open = False
        self.closed = False
        self.model_seconds = 0.0

    def begin_evaluation(self):
        if self.cancelled():
            raise ExperimentCancelled("Generation cancelled")
        self.evaluation += 1
        self.counts.clear()
        self.pending.clear()
        opt = self.options
        if self.evaluation < opt.warmup_evaluations or opt.mode not in {"rules", "jev"} or self.circuit_open:
            return
        if self.last_decision is not None and self.evaluation - self.last_decision < opt.update_interval:
            return
        if len(self.observations) != self.layers:
            return
        if opt.mode == "jev" and (self.client is None or self.client.calls >= opt.max_calls):
            return
        self.last_decision = self.evaluation
        if opt.mode == "rules":
            order = sorted(self.observations, key=lambda i: self.observations[i]["relative_output_norm"])
            self.keeps = {layer: (50.0 if rank < self.layers / 3 else 75.0 if rank < 2 * self.layers / 3 else 100.0) for rank, layer in enumerate(order)}
            self.log.write("decision", evaluation=self.evaluation, source="rules", keep_percent=self.keeps)
            return
        try:
            self.keeps = self.client.decide({
                "target": "Anima image self-attention", "next_model_evaluation": self.evaluation,
                "constraints": "Choose 50, 75 or 100 percent. Text cross-attention is unchanged. No reference-image tokens are present. All transformer layers execute. Statistics are uncalibrated output-norm/drift proxies, not measured quality or sparse approximation error. Prefer 100 under uncertainty.",
                "blocks": self.observations, "current_keep": self.keeps,
            }, {str(i): (50.0, 75.0, 100.0) for i in range(self.layers)}, 100.0)
            self.log.write("decision", evaluation=self.evaluation, source="jev", keep_percent=self.keeps, api_calls=self.client.calls, api_wait_seconds=self.client.wait_seconds)
        except ExperimentCancelled:
            raise
        except Exception as exc:
            self.circuit_open = True
            self.keeps = {str(i): 100.0 for i in range(self.layers)}
            self.log.write("decision", evaluation=self.evaluation, source="dense_fallback", error_type=type(exc).__name__, keep_percent=self.keeps)

    def keep(self, layer: int) -> float:
        if self.closed or self.evaluation < self.options.warmup_evaluations:
            return 100.0
        return self.keeps[str(layer)]

    def observe(self, layer, result, v):
        if self.options.mode not in {"rules", "jev"} or self.circuit_open:
            return
        import torch
        with torch.no_grad():
            stride = max(1, result.shape[1] // 32)
            sample = result[0, ::stride, :16][:32].detach().float().clone()
            vs = v[0, ::stride, 0, :16][:32].detach().float()
            relative = sample.norm() / vs.norm().clamp_min(1e-8)
            prior = self.previous.get(layer)
            drift = sample.new_tensor(-1.0) if prior is None or prior.shape != sample.shape else (sample - prior).norm() / prior.norm().clamp_min(1e-8)
            self.previous[layer] = sample
            self.pending[layer] = torch.stack((relative, drift))

    def end_evaluation(self):
        if self.pending:
            import torch
            ids = sorted(self.pending)
            rows = torch.stack([self.pending[i] for i in ids]).detach().cpu().tolist()
            if not all(math.isfinite(v) for row in rows for v in row):
                self.circuit_open = True
                self.keeps = {str(i): 100.0 for i in range(self.layers)}
                self.observations = {}
                self.log.write("nonfinite_statistics", action="dense_for_remaining_evaluations")
            else:
                self.observations = {str(i): {"relative_output_norm": row[0], "cross_evaluation_drift": None if row[1] < 0 else row[1]} for i, row in zip(ids, rows)}
        self.total_counts.update(self.counts)
        self.log.write("evaluation", evaluation=self.evaluation, counts=dict(self.counts), keep_percent={str(i): self.keep(i) for i in range(self.layers)}, circuit_open=self.circuit_open)
        self.pending.clear()

    def close(self, status="completed"):
        if not self.closed:
            self.closed = True
            self.previous.clear()
            self.pending.clear()
            self.observations.clear()
            self.log.finish(status, timing_scope="sum_instrumented_model_evaluations", model_seconds=self.model_seconds, model_evaluations=self.evaluation + 1, attention_calls=dict(self.total_counts), api_calls=self.client.calls if self.client else 0, api_wait_seconds=self.client.wait_seconds if self.client else 0)


def sparse_attention(q, k, v, keep, *, kernel=None):
    """B,N,H,D -> B,N,H,D. Use a real block-sparse kernel, never a dense mask."""
    import torch
    ck = kernel if kernel is not None else importlib.import_module("comfy_kitchen")
    if q.device.type != "cuda":
        raise RuntimeError("Anima SparseはCUDA専用です。CPUで高速化済みとは扱いません。")
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or q.shape[-1] != 128:
        raise RuntimeError("Anima Sparse requires equal B,N,H,128 Q/K/V tensors")
    if q.dtype != k.dtype or q.dtype != v.dtype or q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError("Unsupported Q/K/V dtype")
    if not hasattr(ck, "sol_attn_is_available") or not ck.sol_attn_is_available(q.device):
        raise RuntimeError("このGPU用のComfy-Kitchen sol_attnカーネルがありません。通常モードを使うか対応版を導入してください。")
    dtype = q.dtype
    if dtype == torch.float32:
        q, k, v = (t.to(torch.bfloat16) for t in (q, k, v))
    return ck.sol_attn(q.contiguous(), k.contiguous(), v.contiguous(), tau=1.3, scale=None, sink_blocks=[0, 0], sink_q=[0, 0], topk_ratio=keep / 100.0, token_aug=0).to(dtype)


def make_attention_patch(module, original, layer, run, kernel_fn=sparse_attention):
    def compute(q, k, v, transformer_options=None):
        if _ACTIVE.get() is not run or run.closed:
            return original(q, k, v, transformer_options=transformer_options or {})
        keep = run.keep(layer)
        if q.shape[1] < run.options.min_tokens:
            reason = "dense_short_sequence"
            result = original(q, k, v, transformer_options=transformer_options or {})
        elif keep >= 100:
            reason = "dense"
            result = original(q, k, v, transformer_options=transformer_options or {})
        else:
            reason = "sparse"
            out = kernel_fn(q, k, v, keep)
            result = module.output_dropout(module.output_proj(out.reshape(out.shape[0], out.shape[1], -1)))
        run.counts[reason] += 1
        run.observe(layer, result, v)
        return result
    return compute


def attach(unet, options: AnimaOptions, log_root: Path, *, prompt="", cancelled=None):
    """Return (cloned_patcher, run). Original weights and cross-attention untouched."""
    options.validate()
    if options.mode == "off":
        return unet, None
    model = unet.get_model_object("diffusion_model")
    if type(model).__name__ != "Anima" or type(model).__module__ != "backend.nn.anima":
        raise ValueError("この実験はAnima専用です。別モデルではOFFにしてください。")
    from backend.args import dynamic_args
    if dynamic_args.ref_latents:
        raise ValueError("Anima Sparseの初版は参照latentなしに限定しています。参照画像を間引く設定ではありません。")
    if unet.model_options.get("aikimi_anima_sparse"):
        raise ValueError("Anima Sparse patch is already installed on this patcher")
    if not model.blocks or any(not block.self_attn.is_SelfAttn for block in model.blocks):
        raise ValueError("Anima self-attention structure mismatch")
    if options.mode not in {"dense", "off"}:
        ck = importlib.import_module("comfy_kitchen")
        import torch
        if any(block.self_attn.head_dim != 128 for block in model.blocks) or not hasattr(ck, "sol_attn_is_available") or not torch.cuda.is_available() or not ck.sol_attn_is_available(unet.load_device):
            raise RuntimeError("Anima Sparseにはhead_dim=128と、このGPUで使えるComfy-Kitchen sol_attnが必要です。")
    client = JevClient(sdk_python(), options.timeout, cancelled) if options.mode == "jev" else None
    log = RunLog(log_root, "anima", asdict(options), prompt)
    run = AnimaRun(options, len(model.blocks), log, client, cancelled)
    try:
        patched = unet.clone()
        for i, block in enumerate(model.blocks):
            name = f"diffusion_model.blocks.{i}.self_attn.compute_attention"
            if name in unet.object_patches:
                raise ValueError("Another extension already patches this Anima attention method")
            original = unet.get_model_object(name)
            patched.add_object_patch(name, make_attention_patch(block.self_attn, original, i, run))
        previous = patched.model_options.get("model_function_wrapper")
        def wrapper(apply_model, args):
            def delegate():
                return previous(apply_model, args) if previous else apply_model(args["input"], args["timestep"], **args["c"])
            if run.closed:
                return delegate()
            # Recheck at execution: another extension may add references after setup.
            if dynamic_args.ref_latents:
                run.close("rejected_references")
                raise ValueError("Anima Sparse does not support reference latents")
            import torch
            device = getattr(args["input"], "device", None)
            def sync():
                if device is not None and device.type == "cuda":
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
        patched.model_options["aikimi_anima_sparse"] = True
        return patched, run
    except BaseException:
        run.close("failed")
        raise
