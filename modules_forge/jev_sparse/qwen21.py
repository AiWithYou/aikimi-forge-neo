"""Qwen-Image 2.1 ONLY: preserve prefill/cache, sparsify cached target attention.

The adapter contract is Diffusers 6256aa7666cedd47443adc8f82da9a10e110b09c,
transformer_qwenimage21.py. This is NOT QwenDoubleStreamAttnProcessor.
The portable block-gather SDPA backend is not Comfy-Kitchen's native SLA.
"""
from __future__ import annotations

import importlib
import json
import math
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .common import ExperimentCancelled, JevClient, RunLog, sdk_python

OPTIONS_ENV = "AIKIMI_QWEN21_SPARSE_OPTIONS"
REVISION = "6256aa7666cedd47443adc8f82da9a10e110b09c"
UPSTREAM = "diffusers.models.transformers.transformer_qwenimage21"


@dataclass(frozen=True)
class Options:
    mode: str = "off"
    keep_percent: float = 75.0
    min_tokens: int = 1024
    warmup_evaluations: int = 1
    update_interval: int = 4
    max_calls: int = 4
    timeout: float = 3.0
    block_size: int = 256

    def validate(self):
        if not isinstance(self.mode, str) or self.mode not in {"off", "dense", "fixed", "rules", "jev"}:
            raise ValueError("Unknown Qwen 2.1 experiment mode")
        for name, low, high in (("keep_percent", 1, 100), ("timeout", .5, 20)):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not low <= v <= high:
                raise ValueError(f"{name} must be finite in {low}..{high}")
        for name, low, high in (("min_tokens", 64, 1048576), ("warmup_evaluations", 1, 100), ("update_interval", 1, 100), ("max_calls", 0, 8)):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or not low <= v <= high:
                raise ValueError(f"{name} must be an integer in {low}..{high}")
        if type(self.block_size) is not int or self.block_size not in {64, 128, 256, 512}:
            raise ValueError("block_size must be 64, 128, 256 or 512")

    @classmethod
    def parse(cls, value):
        if isinstance(value, str):
            if len(value) > 4096:
                raise ValueError("Qwen experiment options are too large")
            value = json.loads(value)
        if value is None:
            value = {}
        if not isinstance(value, dict) or set(value) - {f.name for f in fields(cls)}:
            raise ValueError("Invalid Qwen 2.1 experiment options")
        result = cls(**value)
        result.validate()
        return result


def _key_mask(mask, batch, length):
    """Only the pinned cached-decode padding mask, never an arbitrary causal mask."""
    import torch
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.ndim != 4:
        raise ValueError("Unsupported cached-decode attention mask")
    if mask.shape[0] not in (1, batch) or mask.shape[1:3] != (1, 1) or mask.shape[-1] != length:
        raise ValueError("Unsupported cached-decode mask shape")
    return mask.expand(batch, 1, 1, length)[:, 0, 0]


def block_gather_attention(q, k, v, keep_percent, prefix_len, mask=None, block_size=256):
    """B,N,H,D -> B,N,H,D, with K/V length prefix+N.

    Pool target Q/K into blocks; choose top-k target blocks per head/query block.
    Gather fewer K/V rows BEFORE SDPA. All prefix rows retain their original mask.
    Prefix and target participate in ONE softmax. No dense N-by-N mask is built.
    CPU works for correctness tests; it is not a GPU speed measurement.
    """
    import torch
    import torch.nn.functional as F
    if q.ndim != 4 or k.ndim != 4 or k.shape != v.shape or q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise ValueError("Expected matching B,N,H,D tensors")
    if any(t.device != q.device or t.dtype != q.dtype for t in (k, v)) or not q.is_floating_point():
        raise ValueError("Q/K/V device and dtype must match")
    if not isinstance(prefix_len, int) or prefix_len < 0 or k.shape[1] != prefix_len + q.shape[1] or not q.shape[1]:
        raise ValueError("Invalid Qwen cached prefix length")
    if isinstance(keep_percent, bool) or not math.isfinite(keep_percent) or not 0 < keep_percent <= 100:
        raise ValueError("Invalid target keep percentage")
    if type(block_size) is not int or block_size not in {64, 128, 256, 512}:
        raise ValueError("Invalid block size")
    batch, length, heads, dim = q.shape
    valid = _key_mask(mask, batch, k.shape[1])
    qh, kh, vh = (t.transpose(1, 2) for t in (q, k, v))
    if keep_percent >= 100:
        return F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask, dropout_p=0.0).transpose(1, 2)
    nblocks = math.ceil(length / block_size)
    kept = max(1, math.ceil(nblocks * keep_percent / 100))
    padding = nblocks * block_size - length
    target_k = kh[:, :, prefix_len:]
    target_v = vh[:, :, prefix_len:]
    qp = F.pad(qh, (0, 0, 0, padding))
    kp = F.pad(target_k, (0, 0, 0, padding))
    vp = F.pad(target_v, (0, 0, 0, padding))
    # Padding is excluded from pooled means and from the gathered softmax.
    counts = torch.full((nblocks,), block_size, device=q.device, dtype=torch.float32)
    counts[-1] = block_size - padding
    qm = qp.reshape(batch, heads, nblocks, block_size, dim).float().sum(-2) / counts[None, None, :, None]
    km = kp.reshape(batch, heads, nblocks, block_size, dim).float().sum(-2) / counts[None, None, :, None]
    scores = (qm @ km.transpose(-1, -2)) * dim**-.5
    chosen = scores.topk(kept, dim=-1, sorted=True).indices
    offsets = torch.arange(block_size, device=q.device)
    indices = (chosen[..., None] * block_size + offsets).flatten(-2)
    target_valid = torch.arange(nblocks * block_size, device=q.device) < length
    if valid is not None:
        target_valid = target_valid[None, :] & F.pad(valid[:, prefix_len:], (0, padding), value=False)
    else:
        target_valid = target_valid[None, :].expand(batch, -1)
    # Prefix masking is preserved, including noncontiguous text padding.
    prefix_valid = valid[:, :prefix_len] if valid is not None else torch.ones(batch, prefix_len, device=q.device, dtype=torch.bool)
    outputs = []
    for i in range(nblocks):
        ids = indices[:, :, i]
        gather = ids[..., None].expand(batch, heads, ids.shape[-1], dim)
        selected_k = kp.gather(2, gather)
        selected_v = vp.gather(2, gather)
        selected_valid = target_valid[:, None, :].expand(batch, heads, -1).gather(2, ids)
        keys = torch.cat((kh[:, :, :prefix_len], selected_k), dim=2)
        values = torch.cat((vh[:, :, :prefix_len], selected_v), dim=2)
        allowed = torch.cat((prefix_valid[:, None, :].expand(batch, heads, -1), selected_valid), dim=-1).unsqueeze(-2)
        outputs.append(F.scaled_dot_product_attention(qh[:, :, i * block_size:(i + 1) * block_size], keys, values, attn_mask=allowed, dropout_p=0.0))
    return torch.cat(outputs, dim=2).transpose(1, 2)


class Run:
    def __init__(self, options, layers, log, client=None, cancelled=None):
        self.options, self.layers, self.log, self.client = options, layers, log, client
        self.cancelled = cancelled or (lambda: False)
        self.evaluation = -1
        self.keeps = {str(i): (options.keep_percent if options.mode == "fixed" else 100.) for i in range(layers)}
        self.previous, self.pending, self.observations = {}, {}, {}
        self.counts, self.total = Counter(), Counter()
        self.last_decision = None
        self.circuit_open = False
        self.model_seconds = 0.
        self.started = None
        self.closed = False
        self.summary = {}

    def begin(self):
        if self.cancelled():
            raise ExperimentCancelled("Qwen generation cancelled")
        self.evaluation += 1
        self.started = time.perf_counter()
        self.counts.clear()
        self.pending.clear()
        o = self.options
        if self.circuit_open or self.evaluation < o.warmup_evaluations or len(self.observations) != self.layers:
            return
        if self.last_decision is not None and self.evaluation - self.last_decision < o.update_interval:
            return
        if o.mode == "rules":
            order = sorted(self.observations, key=lambda x: self.observations[x]["relative_output_norm"])
            self.keeps = {i: (50. if n < self.layers / 3 else 75. if n < 2 * self.layers / 3 else 100.) for n, i in enumerate(order)}
        elif o.mode == "jev" and self.client is not None and self.client.calls < o.max_calls:
            try:
                self.keeps = self.client.decide({
                    "target": "Qwen-Image-2.1 cached TARGET image attention only",
                    "evaluation": self.evaluation,
                    "blocks": self.observations,
                    "current_keep": self.keeps,
                    "constraints": "Choose 50,75,100 percent of TARGET key blocks. All prefix KV tokens and padding masks remain. First prefill is unchanged. These output norm/drift statistics are uncalibrated proxies, not quality or approximation-error measurements. Prefer 100 under uncertainty. No prompt, image, raw tensor or model weights are supplied.",
                }, {str(i): (50., 75., 100.) for i in range(self.layers)}, 100.)
            except ExperimentCancelled:
                raise
            except Exception as exc:
                self.circuit_open = True
                self.keeps = {str(i): 100. for i in range(self.layers)}
                self.log.write("controller_failure", error_type=type(exc).__name__, action="dense_remaining_run")
        else:
            return
        self.last_decision = self.evaluation
        self.log.write("decision", evaluation=self.evaluation, keep_percent=self.keeps, circuit_open=self.circuit_open)

    def keep(self, layer):
        return 100. if self.evaluation < self.options.warmup_evaluations or self.circuit_open else self.keeps[str(layer)]

    def observe(self, layer, result, source):
        if self.options.mode not in {"rules", "jev"} or self.circuit_open or result.shape[1] == 0:
            return
        import torch
        with torch.no_grad():
            stride = max(1, result.shape[1] // 32)
            sample = result[0, ::stride, :16][:32].detach().float().clone()
            before = source[0, ::stride, :16][:32].detach().float()
            prior = self.previous.get(layer)
            relative = sample.norm() / before.norm().clamp_min(1e-8)
            drift = sample.new_tensor(-1.) if prior is None or prior.shape != sample.shape else (sample - prior).norm() / prior.norm().clamp_min(1e-8)
            self.pending[layer] = torch.stack((relative, drift))
            self.previous[layer] = sample

    def end(self):
        if self.pending:
            import torch
            ids = sorted(self.pending)
            rows = torch.stack([self.pending[i] for i in ids]).cpu().tolist()
            if all(math.isfinite(v) for row in rows for v in row):
                self.observations = {str(i): {"relative_output_norm": r[0], "cross_evaluation_drift": None if r[1] < 0 else r[1]} for i, r in zip(ids, rows)}
            else:
                self.circuit_open = True
                self.observations.clear()
                self.log.write("nonfinite_statistics", action="dense_remaining_run")
        self.total.update(self.counts)
        self.log.write("evaluation", evaluation=self.evaluation, counts=dict(self.counts), keep_percent={str(i): self.keep(i) for i in range(self.layers)})
        self.pending.clear()
        if self.started is not None:
            self.model_seconds += time.perf_counter() - self.started
            self.started = None

    def finish(self, status):
        if self.closed:
            return
        self.closed = True
        self.summary = dict(status=status, backend="pytorch-block-gather-sdpa", settings=asdict(self.options), timing_scope="sum_instrumented_transformer_evaluations", model_seconds=self.model_seconds, model_evaluations=self.evaluation + 1, attention_calls=dict(self.total), api_calls=self.client.calls if self.client else 0, api_wait_seconds=self.client.wait_seconds if self.client else 0., log_path=str(self.log.path))
        self.log.finish(status, **{k: v for k, v in self.summary.items() if k != "status"})
        self.previous.clear()
        self.pending.clear()
        self.observations.clear()


class Processor:
    def __init__(self, original, run, layer, prepare_qkv, kernel=block_gather_attention):
        self.original, self.run, self.layer = original, run, layer
        self.prepare_qkv, self.kernel = prepare_qkv, kernel

    def __call__(self, attn, hidden_states, attention_mask=None, rotary_emb=None, layer_cache=None, kv_cache_mode=None, cache_write_slice=None, segments=None, key_valid=None):
        r = self.run
        if r.cancelled():
            raise ExperimentCancelled("Qwen generation cancelled")
        kwargs = dict(attention_mask=attention_mask, rotary_emb=rotary_emb, layer_cache=layer_cache, kv_cache_mode=kv_cache_mode, cache_write_slice=cache_write_slice, segments=segments, key_valid=key_valid)
        keep = r.keep(self.layer)
        target_start = segments[-1][1] if segments else 0
        reason = None
        if r.closed or r.options.mode in {"off", "dense"} or keep >= 100:
            reason = "dense"
        elif kv_cache_mode != "cached" or segments is not None or layer_cache is None:
            reason = "dense_prefill_or_uncached"
        elif hidden_states.shape[1] < r.options.min_tokens:
            reason = "dense_short_target"
        elif math.ceil(math.ceil(hidden_states.shape[1] / r.options.block_size) * keep / 100) >= math.ceil(hidden_states.shape[1] / r.options.block_size):
            reason = "dense_rounded_full_coverage"
        else:
            ck, _ = layer_cache.get()
            try:
                _key_mask(attention_mask, hidden_states.shape[0], ck.shape[1] + hidden_states.shape[1])
            except ValueError:
                reason = "dense_unsupported_mask"
        if reason:
            result = self.original(attn, hidden_states, **kwargs)
        else:
            q, k, v, n = self.prepare_qkv(attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice)
            prefix = k.shape[1] - n
            out = self.kernel(q, k, v, keep, prefix, mask=attention_mask, block_size=r.options.block_size)
            result = attn.to_out[1](attn.to_out[0](out.flatten(2, 3).type_as(q)))
            reason = "sparse_target"
        r.counts[reason] += 1
        # Dense prefill observations let the first adaptive decision actually run.
        r.observe(self.layer, result[:, target_start:], hidden_states[:, target_start:])
        return result


@contextmanager
def experiment(transformer, options, log_root, *, prompt="", cancelled=None, upstream=None, client=None):
    options.validate()
    if options.mode == "off":
        yield None
        return
    upstream = upstream or importlib.import_module(UPSTREAM)
    if type(transformer).__name__ != "QwenImage21Transformer2DModel":
        raise ValueError("This adapter is Qwen-Image 2.1 ONLY, not legacy Qwen-Image")
    blocks = transformer.transformer_blocks
    originals = [b.attn.processor for b in blocks]
    if not originals or any(type(p) is not upstream.QwenImage21AttnProcessor or getattr(p, "_parallel_config", None) is not None for p in originals):
        raise ValueError("Qwen 2.1 requires its unmodified default processor; compiled/custom/parallel processors are not supported")
    if getattr(transformer, "_aikimi_qwen21_experiment", False):
        raise ValueError("Qwen experiment already active")
    if options.mode == "jev" and client is None and options.max_calls:
        client = JevClient(sdk_python(), options.timeout, cancelled)
    log = RunLog(Path(log_root), "qwen21", asdict(options), prompt)
    run = Run(options, len(blocks), log, client, cancelled)
    handles = []
    changed = []
    transformer._aikimi_qwen21_experiment = True
    def sync():
        import torch
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
    def before(module, args):
        sync()
        run.begin()
    def after(module, args, output):
        sync()
        run.end()
    status = "failed"
    try:
        for i, block in enumerate(blocks):
            block.attn.set_processor(Processor(originals[i], run, i, upstream._qwenimage21_prepare_qkv))
            changed.append(i)
        handles.append(transformer.register_forward_pre_hook(before))
        handles.append(transformer.register_forward_hook(after))
        yield run
        status = "completed"
    except BaseException:
        status = "cancelled" if run.cancelled() else "failed"
        raise
    finally:
        for handle in handles:
            handle.remove()
        for i in changed:
            blocks[i].attn.set_processor(originals[i])
        del transformer._aikimi_qwen21_experiment
        run.finish(status)
