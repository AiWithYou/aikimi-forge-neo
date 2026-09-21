# SPDX-License-Identifier: GPL-3.0-only
"""H3 four-step experiment node, derived from sepiablue-ai's native_sla.py.

Reference: ComfyUI-MiniMax-H3-W4A4-VSA@fa29225664909ea3dca69d0cdc35752bc38cd7fa.
Uses ComfyUI's native H3 sparse producer; no weight conversion or W4A4 code.
Fixed modes never construct the SDK client and never call a cloud service.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

from .common import ExperimentCancelled, JevClient, RunLog


def _samples(x, layout):
    import torch

    result = {}
    for start, end, kind in layout.segments:
        if kind in ("audio", "video") and end > start:
            ids = torch.linspace(start, end - 1, min(64, end - start), device=x.device).long()
            result[kind] = x.index_select(0, ids)[:, :: max(1, x.shape[-1] // 16)][:, :16].detach().float()
    if set(result) != {"audio", "video"}:
        raise RuntimeError("H3 experiment requires separate target audio/video segments")
    return result


class H3Controller:
    def __init__(self, mode, log, client=None, prompt="", *, max_calls=4, initial_decision=True):
        self.mode, self.log, self.client, self.prompt = mode, log, client, prompt
        self.max_calls, self.initial_decision = max_calls, initial_decision
        self.step = 0
        self.keeps = [100.0 if mode == "dense" else 10.0 if mode == "fixed10" else 5.0] * 50
        self.previous, self.pending, self.actual = {}, {}, {}
        self.disabled = False

    def initialize(self):
        if self.mode != "jev" or not self.initial_decision:
            return
        allowed = {str(i): ((5.0, 10.0) if i == 0 else (1.0, 3.0, 5.0, 10.0)) for i in range(50)}
        self._decide(
            {
                "initialization": True,
                "next_step": 1,
                "prompt": self.prompt,
                "constraints": "First denoising step, no current-run observations and no historical statistics. Layer 0 allows only 5 or 10 percent. All layers run. Exact conditioning KV/audio query protection comes from native SLA. No quota or forced variation; uncertain layers should use 10.",
            },
            allowed,
            10.0,
            initial=True,
        )
        self.prompt = ""

    def _decide(self, state, allowed, fallback, initial=False):
        try:
            result = self.client.decide(state, allowed, fallback)
            self.keeps = (
                [result[str(i)] for i in range(50)] if initial else [5.0] + [result[str(i)] for i in range(1, 50)]
            )
            reason = "jev"
        except ExperimentCancelled:
            raise
        except Exception as exc:
            self.disabled = True
            self.keeps = [fallback] * 50
            reason = "fallback:" + type(exc).__name__
        self.log.write(
            "decision",
            next_step=self.step + (1 if initial else 2),
            source=reason,
            keep_percent=self.keeps[:],
            api_calls=self.client.calls,
            api_wait_seconds=self.client.wait_seconds,
            controller_policy="speed_v3",
            diagnostics=getattr(self.client, "last_diagnostics", {}),
        )

    def observe(self, layer, before, after):
        import torch

        vals = []
        for kind in ("audio", "video"):
            base = before[kind]
            delta = after[kind] - base
            prev = self.previous.get((layer, kind))
            drift = (
                delta.new_tensor(-1.0)
                if prev is None or prev.shape != delta.shape
                else (delta - prev).norm() / prev.norm().clamp_min(1e-8)
            )
            vals.extend((delta.norm() / base.norm().clamp_min(1e-8), drift))
            self.previous[layer, kind] = delta
        self.pending[layer] = torch.stack(vals)

    def done(self, index):
        if index != self.step:
            raise RuntimeError("H3 callback step order does not match four-step res_multistep")
        self.log.write("step", step=index + 1, keep_percent=self.keeps[:], actual_attention=dict(self.actual))
        if self.mode == "jev" and index < 3 and not self.disabled and self.client.calls < self.max_calls:
            import torch

            if set(self.pending) != set(range(50)):
                raise RuntimeError("H3 did not report all 50 transformer blocks")
            rows = torch.stack([self.pending[i] for i in range(50)]).cpu().tolist()
            if not all(math.isfinite(v) for row in rows for v in row):
                self.disabled = True
                self.keeps = [5.0] * 50
                self.log.write("nonfinite_statistics", action="fixed5_for_remaining_steps")
            else:
                ranks = {kind: sorted(range(50), key=lambda i: rows[i][j]) for kind, j in (("audio", 0), ("video", 2))}
                state = {
                    "step_measured": index + 1,
                    "next_step": index + 2,
                    "constraints": "SPEED-FIRST: layer 0 is fixed at 5 percent and is not a question. Choose 1,3,5,10 for layers 1..49. Favor 1 for weak/stable residual contributions, 3 for typical/ambiguous contributions, 5 for strong contributions, and reserve 10 for extreme contributions. All blocks run. Exact conditioning KV/audio query rows are protected by native SLA. Changes in detail and motion are acceptable; preserve visually plausible video. Residual magnitude, peer rank and drift are proxies, not measured quality. Missing first drift alone does not require maximum keep. No forced quota.",
                    "blocks": {
                        str(i): {
                            kind: {
                                "residual_relative_l2": rows[i][j],
                                "rank": (ranks[kind].index(i) + 1) / 50,
                                "cross_step_change": None if rows[i][j + 1] < 0 else rows[i][j + 1],
                            }
                            for kind, j in (("audio", 0), ("video", 2))
                        }
                        for i in range(50)
                    },
                    "current_keep": self.keeps[:],
                }
                self._decide(state, {str(i): (1.0, 3.0, 5.0, 10.0) for i in range(1, 50)}, 5.0)
        self.pending.clear()
        self.actual.clear()
        self.step += 1


class AikimiH3SparseExperiment:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # A repeated benchmark must execute the sampler and controller again.
        return float("nan")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (["dense", "fixed5", "fixed10", "jev"],),
                "sdk_python": ("STRING", {"default": ""}),
                "prompt_context": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Aikimi/Experiments"

    def patch(self, model, mode, sdk_python="", prompt_context=""):
        import comfy.model_management as mm
        import comfy.patcher_extension as pe
        from comfy_extras.nodes_sparse_attention import (
            SparseAttnPatch,
            h3_eligible,
            h3_sparse_attention,
            install_override,
        )

        if mode not in {"dense", "fixed5", "fixed10", "jev"}:
            raise ValueError("Unknown H3 experiment mode")
        if mode == "jev":
            from .common import cloud_environment

            cloud_environment()
            if not prompt_context.strip() or not Path(sdk_python).is_file():
                raise ValueError("Jev requires a prompt and the dedicated SDK Python")
        blocks = model.get_model_object("diffusion_model").blocks
        if len(blocks) != 50:
            raise ValueError("H3 experiment requires the native 50-block H3 model")
        sampling = model.get_model_object("model_sampling")
        p = SparseAttnPatch(
            tau=1.3,
            topk_ratio=0.05,
            vsa=False,
            sigma_start=float(sampling.percent_to_sigma(0)),
            sigma_end=float(sampling.percent_to_sigma(1)),
            min_tokens=12288,
            dense_blocks=set(),
            sink_conditioning="exact_kv_and_rows",
            extra_tokens=0,
            verbose=False,
        )
        m = model.clone()
        p.controller = None
        if mode != "dense":
            install_override(p, m.model_options["transformer_options"])
            m.add_callback_with_key(
                pe.CallbacksMP.ON_PREPARE_STATE,
                "aikimi_jev",
                lambda mp, t, opts: install_override(p, opts["transformer_options"]),
            )
        m.add_callback_with_key(pe.CallbacksMP.ON_CLEANUP, "aikimi_jev", lambda mp: p.reset())

        def make(block, index):
            def attention(h, rope_freqs=None, transformer_options=None):
                return h3_sparse_attention(block.attn, h, rope_freqs, transformer_options, p, index)

            def call(args, extra):
                c = p.controller
                if c is None:
                    raise RuntimeError("H3 sampling controller is not active")
                p.topk_ratio = c.keeps[index] / 100
                eligible = mode != "dense" and h3_eligible(
                    block.attn, args["img"], args["rope_freqs"], args["transformer_options"], p, index
                )
                c.actual[str(index)] = (
                    "native_sla_producer" if eligible else "dense" if mode == "dense" else "native_producer_not_used"
                )
                observe = mode == "jev" and c.step < 3 and not c.disabled and c.client.calls < c.max_calls
                before = _samples(args["img"], args["layout"]) if observe else None
                out = extra["original_block"]({**args, "attention": attention} if eligible else args)
                if observe:
                    c.observe(index, before, _samples(out["img"], args["layout"]))
                return out

            return call

        for i, block in enumerate(blocks):
            m.set_model_patch_replace(make(block, i), "dit", "double_block", i)

        def cancelled():
            try:
                mm.throw_exception_if_processing_interrupted()
                return False
            except mm.InterruptProcessingException:
                return True

        def wrapper(
            executor,
            model_wrap,
            sigmas,
            extra_args,
            callback,
            noise,
            latent_image=None,
            denoise_mask=None,
            disable_pbar=False,
        ):
            if (
                len(sigmas) != 5
                or getattr(executor.class_obj.sampler_function, "__name__", "") != "sample_res_multistep"
            ):
                raise ValueError("H3 Jev/fixed comparison is restricted to 4 steps / res_multistep")
            p.reset()
            root = Path(os.environ.get("AIKIMI_SPARSE_LOG_DIR", "output/aikimi-sparse"))
            log = RunLog(
                root,
                "h3",
                {
                    "mode": mode,
                    "steps": 4,
                    "sampler": "res_multistep",
                    "timing_scope": "sampling_only",
                    "min_tokens": 12288,
                    "extra_tokens": 0,
                    "sigmas": sigmas.detach().cpu().tolist(),
                },
                prompt_context,
            )
            client = None
            c = None
            status = "failed"
            try:
                client = JevClient(Path(sdk_python), 20, cancelled) if mode == "jev" else None
                c = H3Controller(mode, log, client, prompt_context, max_calls=1, initial_decision=False)
                p.controller = c
                c.initialize()

                def done(i, denoised, x, total):
                    c.done(i)
                    if callback is not None:
                        callback(i, denoised, x, total)

                result = executor(model_wrap, sigmas, extra_args, done, noise, latent_image, denoise_mask, disable_pbar)
                status = "completed"
                return result
            except BaseException:
                if cancelled():
                    status = "cancelled"
                raise
            finally:
                log.finish(
                    status,
                    api_calls=client.calls if client else 0,
                    api_wait_seconds=client.wait_seconds if client else 0,
                )
                p.controller = None
                p.reset()
                if c is not None:
                    c.previous.clear()
                    c.pending.clear()

        m.add_wrapper_with_key(pe.WrappersMP.SAMPLER_SAMPLE, "aikimi_jev", wrapper)
        return (m,)


NODE_CLASS_MAPPINGS = {"AikimiH3SparseExperiment": AikimiH3SparseExperiment}
NODE_DISPLAY_NAME_MAPPINGS = {"AikimiH3SparseExperiment": "Aikimi H3 · 4-step Jev / fixed SLA experiment"}
