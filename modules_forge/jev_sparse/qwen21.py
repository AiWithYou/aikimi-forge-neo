"""Qwen-Image 2.1 ONLY: preserve prefill/cache, sparsify cached target attention.

The adapter contract is Diffusers 6256aa7666cedd47443adc8f82da9a10e110b09c,
transformer_qwenimage21.py. This is NOT QwenDoubleStreamAttnProcessor.
The portable block-gather SDPA backend is not Comfy-Kitchen's native SLA.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .common import (
    ExperimentCancelled,
    JevBudget,
    ReplayError,
    RunLog,
    cadence_has_budget,
    cadence_interval,
    client_available,
    close_client,
    create_client,
    decision_count,
)

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
    max_calls: int = 1
    timeout: float = 3.0
    block_size: int = 256
    decision_cadence: str = "legacy"
    query_batch_blocks: int = 4
    max_batch_workspace_mb: int = 256
    job_max_calls: int = 0
    job_max_wait_seconds: float = 0.0

    def validate(self):
        if not isinstance(self.mode, str) or self.mode not in {"off", "dense", "fixed", "rules", "jev"}:
            raise ValueError("Unknown Qwen 2.1 experiment mode")
        if not isinstance(self.decision_cadence, str) or self.decision_cadence not in {
            "legacy",
            "once",
            "interval",
            "step",
        }:
            raise ValueError("Unknown Jev decision cadence")
        for name, low, high in (
            ("keep_percent", 1, 100),
            ("timeout", 0.5, 20),
            ("job_max_wait_seconds", 0, 3600),
        ):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not low <= v <= high:
                raise ValueError(f"{name} must be finite in {low}..{high}")
        for name, low, high in (
            ("min_tokens", 64, 1048576),
            ("warmup_evaluations", 1, 100),
            ("update_interval", 1, 100),
            ("max_calls", 0, 8),
            ("query_batch_blocks", 1, 16),
            ("max_batch_workspace_mb", 1, 1024),
            ("job_max_calls", 0, 1000),
        ):
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


def _gather_batch_plan(batch, heads, dim, block_size, selected_tokens, prefix_len, element_size, blocks, workspace_mb):
    """Bound explicit gathered/catenated Q/K/V, indices and masks (not SDPA's workspace).

    Split heads as well as query blocks if even one full-head block exceeds the
    budget. Base Q/K/V, pooled block scores and the final output are not temporary
    batch workspace. Backends may allocate additional attention workspace.
    """
    all_tokens = prefix_len + selected_tokens
    per_head_block = batch * (
        (2 * selected_tokens + 2 * all_tokens + 2 * block_size) * dim * element_size
        + 16 * selected_tokens
        + 2 * all_tokens
    )
    capacity = workspace_mb * 1024 * 1024 // per_head_block
    if capacity < 1:
        raise ValueError("Sparse workspace budget cannot hold one query block and head")
    head_batch = min(heads, capacity)
    return head_batch, min(blocks, capacity // head_batch)


def block_gather_attention(
    q, k, v, keep_percent, prefix_len, mask=None, block_size=256, *, query_batch_blocks=4, max_batch_workspace_mb=256
):
    """B,N,H,D -> B,N,H,D, with K/V length prefix+N.

    Pool target Q/K into blocks; choose top-k target blocks per head/query block.
    Gather fewer K/V rows BEFORE SDPA. All prefix rows retain their original mask.
    Prefix and target participate in ONE softmax. No dense N-by-N mask is built.
    Batch a bounded number of query blocks per SDPA call; split heads if needed.
    The workspace budget covers explicit batch tensors, not total VRAM or SDPA.
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
    if type(query_batch_blocks) is not int or not 1 <= query_batch_blocks <= 16:
        raise ValueError("query_batch_blocks must be an integer in 1..16")
    if type(max_batch_workspace_mb) is not int or not 1 <= max_batch_workspace_mb <= 1024:
        raise ValueError("max_batch_workspace_mb must be an integer in 1..1024")
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
    qp = F.pad(qh, (0, 0, 0, padding)) if padding else qh
    kp = F.pad(target_k, (0, 0, 0, padding)) if padding else target_k
    vp = F.pad(target_v, (0, 0, 0, padding)) if padding else target_v
    # Padding is excluded from pooled means and from the gathered softmax.
    counts = torch.full((nblocks,), block_size, device=q.device, dtype=torch.float32)
    # fill_ keeps the scalar in the CUDA kernel arguments. Indexed assignment
    # creates an unpinned CPU scalar tensor and prevents CUDA Graph capture.
    counts[-1:].fill_(block_size - padding)
    qm = qp.reshape(batch, heads, nblocks, block_size, dim).float().sum(-2) / counts[None, None, :, None]
    km = kp.reshape(batch, heads, nblocks, block_size, dim).float().sum(-2) / counts[None, None, :, None]
    scores = (qm @ km.transpose(-1, -2)) * dim**-0.5
    chosen = scores.topk(kept, dim=-1, sorted=True).indices
    offsets = torch.arange(block_size, device=q.device)
    target_valid = torch.arange(nblocks * block_size, device=q.device) < length
    if valid is not None:
        target_valid = target_valid[None, :] & F.pad(valid[:, prefix_len:], (0, padding), value=False)
    else:
        target_valid = target_valid[None, :].expand(batch, -1)
    # Prefix masking is preserved, including noncontiguous text padding.
    prefix_valid = (
        valid[:, :prefix_len] if valid is not None else torch.ones(batch, prefix_len, device=q.device, dtype=torch.bool)
    )
    head_batch, block_batch = _gather_batch_plan(
        batch,
        heads,
        dim,
        block_size,
        kept * block_size,
        prefix_len,
        q.element_size(),
        min(nblocks, query_batch_blocks),
        max_batch_workspace_mb,
    )
    output = torch.empty_like(qp)
    for h in range(0, heads, head_batch):
        h_end = min(h + head_batch, heads)
        nh = h_end - h
        for start in range(0, nblocks, block_batch):
            end = min(start + block_batch, nblocks)
            nb = end - start
            # Build token indices only for this batch, not all N query blocks.
            ids = (chosen[:, h:h_end, start:end, :, None] * block_size + offsets).flatten(-2)
            gather = ids[..., None].expand(batch, nh, nb, ids.shape[-1], dim)
            selected_k = kp[:, h:h_end, None].expand(-1, -1, nb, -1, -1).gather(3, gather)
            selected_v = vp[:, h:h_end, None].expand(-1, -1, nb, -1, -1).gather(3, gather)
            selected_valid = target_valid[:, None, None, :].expand(batch, nh, nb, -1).gather(3, ids)
            keys = torch.cat((kh[:, h:h_end, None, :prefix_len].expand(-1, -1, nb, -1, -1), selected_k), dim=3)
            values = torch.cat((vh[:, h:h_end, None, :prefix_len].expand(-1, -1, nb, -1, -1), selected_v), dim=3)
            allowed = torch.cat((prefix_valid[:, None, None, :].expand(batch, nh, nb, -1), selected_valid), dim=-1)
            # Fold blocks into the head axis, keeping 4-D SDPA for fused backends.
            query = qp[:, h:h_end, start * block_size : end * block_size].reshape(batch, nh * nb, block_size, dim)
            attended = F.scaled_dot_product_attention(
                query,
                keys.flatten(1, 2),
                values.flatten(1, 2),
                attn_mask=allowed.flatten(1, 2).unsqueeze(-2),
                dropout_p=0.0,
            )
            output[:, h:h_end, start * block_size : end * block_size] = attended.reshape(
                batch, nh, nb * block_size, dim
            )
            del selected_k, selected_v, keys, values, query, attended, ids, gather, selected_valid, allowed
    return output[:, :, :length].transpose(1, 2)


class Run:
    def __init__(self, options, layers, log, client=None, cancelled=None):
        self.options, self.layers, self.log, self.client = options, layers, log, client
        self.cancelled = cancelled or (lambda: False)
        self.evaluation = -1
        self.keeps = {str(i): (options.keep_percent if options.mode == "fixed" else 100.0) for i in range(layers)}
        self.previous, self.pending, self.observations = {}, {}, {}
        self.observation_shapes = {}
        self.counts, self.total = Counter(), Counter()
        self.last_decision = None
        self.circuit_open = False
        self.model_seconds = 0.0
        self.controller_wall_seconds = 0.0
        self.transformer_cpu_wall_seconds = 0.0
        self.run_started = time.perf_counter()
        self.started = None
        self.forward_started = None
        self.cuda_events = []
        self.active_cuda_events = None
        self.closed = False
        self.summary = {}

    def begin(self):
        if self.cancelled():
            raise ExperimentCancelled("Qwen generation cancelled")
        self.evaluation += 1
        self.started = time.perf_counter()
        self.counts.clear()
        self.pending.clear()
        started = time.perf_counter()
        try:
            self._update_decision()
        finally:
            self.controller_wall_seconds += time.perf_counter() - started

    def _update_decision(self):
        o = self.options
        if self.circuit_open or self.evaluation < o.warmup_evaluations or len(self.observations) != self.layers:
            return
        if self.last_decision is not None and self.evaluation - self.last_decision < cadence_interval(
            o.decision_cadence, o.update_interval
        ):
            return
        if o.mode == "rules":
            order = sorted(self.observations, key=lambda x: self.observations[x]["relative_output_norm"])
            self.keeps = {
                i: (50.0 if n < self.layers / 3 else 75.0 if n < 2 * self.layers / 3 else 100.0)
                for n, i in enumerate(order)
            }
        elif (
            o.mode == "jev"
            and client_available(self.client)
            and cadence_has_budget(o.decision_cadence, decision_count(self.client), o.max_calls)
        ):
            try:
                self.keeps = self.client.decide(
                    {
                        "target": "Qwen-Image-2.1 cached TARGET image attention only",
                        "evaluation": self.evaluation,
                        "blocks": self.observations,
                        "tensor_shapes": self.observation_shapes,
                        "decision_cadence": o.decision_cadence,
                        "current_keep": self.keeps,
                        "constraints": "SPEED-FIRST profile: choose 25,50,75,100 percent of TARGET key blocks. All prefix KV tokens and padding masks remain. First prefill is unchanged. Compare output-norm/drift proxies with peers: favor 25 for weak/stable contributions, 50 for typical/ambiguous contributions, 75 for unusually strong contributions. Reserve 100 for extreme contributions with supporting measurements. Visually good results matter; different fine details and compositions are acceptable. Missing first drift alone does not require 100. These are proxies, not measured visual quality. No forced quota. No prompt, image, raw tensor or weights are supplied.",
                    },
                    {str(i): (25.0, 50.0, 75.0, 100.0) for i in range(self.layers)},
                    100.0,
                )
            except (ExperimentCancelled, ReplayError):
                raise
            except Exception as exc:
                self.circuit_open = True
                self.keeps = {str(i): 100.0 for i in range(self.layers)}
                self.log.write("controller_failure", error_type=type(exc).__name__, action="dense_remaining_run")
        else:
            return
        self.last_decision = self.evaluation
        self.log.write(
            "decision",
            evaluation=self.evaluation,
            keep_percent=self.keeps,
            circuit_open=self.circuit_open,
            source="dense_fallback" if self.circuit_open else o.mode,
            controller_policy="speed_v3",
            diagnostics=getattr(self.client, "last_diagnostics", {}),
            observations=self.observations,
            replay=getattr(self.client, "last_replay", None),
            replay_calls=getattr(self.client, "replay_calls", 0),
        )

    def start_forward(self, args, kwargs):
        """Measure host dispatch separately from the device stream interval.

        CUDA events exclude the controller request, but the elapsed stream time
        still includes host launch gaps and device copies. It is not a sum of
        kernel durations. Defer synchronization until the run ends.
        """
        import torch

        def cuda_tensor(values):
            if isinstance(values, torch.Tensor):
                return values if values.is_cuda else None
            if isinstance(values, dict):
                values = tuple(values.values())
            if isinstance(values, (tuple, list)):
                for value in values:
                    found = cuda_tensor(value)
                    if found is not None:
                        return found
            return None

        tensor = cuda_tensor((args, kwargs))
        if tensor is not None:
            stream = torch.cuda.current_stream(tensor.device)
            start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record(stream)
            self.active_cuda_events = (start, stop, stream)
        self.forward_started = time.perf_counter()

    def end_forward(self):
        if self.forward_started is not None:
            self.transformer_cpu_wall_seconds += time.perf_counter() - self.forward_started
            self.forward_started = None
        if self.active_cuda_events is not None:
            start, stop, stream = self.active_cuda_events
            stop.record(stream)
            self.cuda_events.append((start, stop))
            self.active_cuda_events = None

    def keep(self, layer):
        return (
            100.0 if self.evaluation < self.options.warmup_evaluations or self.circuit_open else self.keeps[str(layer)]
        )

    def observe(self, layer, result, source, prefix_len=0):
        if result.shape[1] < self.options.min_tokens:
            return
        if (
            self.options.mode == "jev"
            and self.client is not None
            and (
                not client_available(self.client)
                or not cadence_has_budget(
                    self.options.decision_cadence, decision_count(self.client), self.options.max_calls
                )
            )
        ):
            return
        if self.options.mode not in {"rules", "jev"} or self.circuit_open or result.shape[1] == 0:
            return
        import torch

        self.observation_shapes[str(layer)] = {"target": list(result.shape), "prefix_tokens": prefix_len}
        with torch.no_grad():
            stride = max(1, result.shape[1] // 32)
            sample = result[0, ::stride, :16][:32].detach().float().clone()
            before = source[0, ::stride, :16][:32].detach().float()
            prior = self.previous.get(layer)
            relative = sample.norm() / before.norm().clamp_min(1e-8)
            drift = (
                sample.new_tensor(-1.0)
                if prior is None or prior.shape != sample.shape
                else (sample - prior).norm() / prior.norm().clamp_min(1e-8)
            )
            self.pending[layer] = torch.stack((relative, drift))
            self.previous[layer] = sample

    def end(self):
        if self.pending:
            import torch

            ids = sorted(self.pending)
            rows = torch.stack([self.pending[i] for i in ids]).cpu().tolist()
            if all(math.isfinite(v) for row in rows for v in row):
                self.observations = {
                    str(i): {"relative_output_norm": r[0], "cross_evaluation_drift": None if r[1] < 0 else r[1]}
                    for i, r in zip(ids, rows, strict=True)
                }
            else:
                self.circuit_open = True
                self.observations.clear()
                self.log.write("nonfinite_statistics", action="dense_remaining_run")
        self.total.update(self.counts)
        self.log.write(
            "evaluation",
            evaluation=self.evaluation,
            counts=dict(self.counts),
            keep_percent={str(i): self.keep(i) for i in range(self.layers)},
        )
        self.pending.clear()
        if self.started is not None:
            self.model_seconds += time.perf_counter() - self.started
            self.started = None

    def finish(self, status):
        if self.closed:
            return
        self.closed = True
        close_error = None
        try:
            close_client(self.client, status)
        except ReplayError as exc:
            status, close_error = "failed", exc
        self.end_forward()
        if self.started is not None:
            self.model_seconds += time.perf_counter() - self.started
            self.started = None
        cuda_seconds = None
        if self.cuda_events:
            cuda_seconds = 0.0
            for start, stop in self.cuda_events:
                stop.synchronize()
                cuda_seconds += start.elapsed_time(stop) / 1000
        self.summary = dict(
            status=status,
            backend="pytorch-block-gather-sdpa",
            settings=asdict(self.options),
            timing_scope="experiment_context_wall_and_separate_transformer_intervals",
            total_wall_seconds=time.perf_counter() - self.run_started,
            evaluation_cpu_wall_seconds=self.model_seconds,
            transformer_cpu_wall_seconds=self.transformer_cpu_wall_seconds,
            controller_wall_seconds=self.controller_wall_seconds,
            cuda_event_seconds=cuda_seconds,
            cuda_event_evaluations=len(self.cuda_events),
            cuda_event_scope="transformer_stream_intervals_include_host_launch_gaps_and_device_copies",
            model_seconds=self.model_seconds,
            model_seconds_kind="legacy_alias_evaluation_cpu_wall_including_controller_not_gpu_time",
            model_evaluations=self.evaluation + 1,
            attention_calls=dict(self.total),
            api_calls=self.client.calls if self.client else 0,
            api_wait_seconds=self.client.wait_seconds if self.client else 0.0,
            replay_calls=getattr(self.client, "replay_calls", 0),
            log_path=str(self.log.path),
        )
        self.log.finish(status, **{k: v for k, v in self.summary.items() if k != "status"})
        self.previous.clear()
        self.pending.clear()
        self.observations.clear()
        self.observation_shapes.clear()
        self.cuda_events.clear()
        if close_error is not None:
            raise close_error


class Processor:
    def __init__(self, original, run, layer, prepare_qkv, kernel=block_gather_attention):
        self.original, self.run, self.layer = original, run, layer
        self.prepare_qkv, self.kernel = prepare_qkv, kernel

    def __call__(
        self,
        attn,
        hidden_states,
        attention_mask=None,
        rotary_emb=None,
        layer_cache=None,
        kv_cache_mode=None,
        cache_write_slice=None,
        segments=None,
        key_valid=None,
    ):
        r = self.run
        if r.cancelled():
            raise ExperimentCancelled("Qwen generation cancelled")
        kwargs = dict(
            attention_mask=attention_mask,
            rotary_emb=rotary_emb,
            layer_cache=layer_cache,
            kv_cache_mode=kv_cache_mode,
            cache_write_slice=cache_write_slice,
            segments=segments,
            key_valid=key_valid,
        )
        keep = r.keep(self.layer)
        target_start = segments[-1][1] if segments else 0
        reason = None
        if r.closed or r.options.mode in {"off", "dense"} or keep >= 100:
            reason = "dense"
        elif kv_cache_mode != "cached" or segments is not None or layer_cache is None:
            reason = "dense_prefill_or_uncached"
        elif hidden_states.shape[1] < r.options.min_tokens:
            reason = "dense_short_target"
        elif math.ceil(math.ceil(hidden_states.shape[1] / r.options.block_size) * keep / 100) >= math.ceil(
            hidden_states.shape[1] / r.options.block_size
        ):
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
            q, k, v, n = self.prepare_qkv(
                attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice
            )
            prefix = k.shape[1] - n
            out = self.kernel(
                q,
                k,
                v,
                keep,
                prefix,
                mask=attention_mask,
                block_size=r.options.block_size,
                query_batch_blocks=r.options.query_batch_blocks,
                max_batch_workspace_mb=r.options.max_batch_workspace_mb,
            )
            result = attn.to_out[1](attn.to_out[0](out.flatten(2, 3).type_as(q)))
            reason = "sparse_target"
        r.counts[reason] += 1
        # Dense prefill observations let the first adaptive decision actually run.
        prefix_tokens = target_start
        if kv_cache_mode == "cached" and layer_cache is not None:
            prefix_tokens = layer_cache.get()[0].shape[1]
        r.observe(self.layer, result[:, target_start:], hidden_states[:, target_start:], prefix_tokens)
        return result


@contextmanager
def experiment(
    transformer, options, log_root, *, prompt="", cancelled=None, upstream=None, client=None, replay_identity=None
):
    options.validate()
    if options.mode == "off":
        yield None
        return
    upstream = upstream or importlib.import_module(UPSTREAM)
    if type(transformer).__name__ != "QwenImage21Transformer2DModel":
        raise ValueError("This adapter is Qwen-Image 2.1 ONLY, not legacy Qwen-Image")
    blocks = transformer.transformer_blocks
    originals = [b.attn.processor for b in blocks]
    if not originals or any(
        type(p) is not upstream.QwenImage21AttnProcessor or getattr(p, "_parallel_config", None) is not None
        for p in originals
    ):
        raise ValueError(
            "Qwen 2.1 requires its unmodified default processor; compiled/custom/parallel processors are not supported"
        )
    if getattr(transformer, "_aikimi_qwen21_experiment", False):
        raise ValueError("Qwen experiment already active")
    if options.mode == "jev" and client is None and options.max_calls:
        client = create_client(
            timeout=options.timeout,
            cancelled=cancelled,
            budget=JevBudget(options.job_max_calls, options.job_max_wait_seconds),
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            replay_identity=replay_identity,
        )
    try:
        log = RunLog(Path(log_root), "qwen21", {**asdict(options), "replay_identity": replay_identity}, prompt)
        run = Run(options, len(blocks), log, client, cancelled)
    except BaseException:
        close_client(client, "failed")
        raise
    handles = []
    changed = []
    transformer._aikimi_qwen21_experiment = True

    def before(module, args, kwargs):
        run.begin()
        run.start_forward(args, kwargs)

    def after(module, args, kwargs, output):
        run.end_forward()
        run.end()

    status = "failed"
    try:
        for i, block in enumerate(blocks):
            block.attn.set_processor(Processor(originals[i], run, i, upstream._qwenimage21_prepare_qkv))
            changed.append(i)
        handles.append(transformer.register_forward_pre_hook(before, with_kwargs=True))
        handles.append(transformer.register_forward_hook(after, with_kwargs=True))
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
