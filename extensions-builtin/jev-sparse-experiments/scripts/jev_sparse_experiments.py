"""Forge UI entrypoint. No model, SDK or network activity while disabled."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import gradio as gr

from modules import script_callbacks, scripts, shared
from modules.paths import data_path
from modules_forge.jev_sparse import h3_integration
from modules_forge.jev_sparse.common import (
    AnimaOptions,
    JevBudget,
    ReplayError,
    close_client,
    create_client,
    replay_requested,
)


def _install_h3():
    try:
        h3_integration.install()
    except Exception as exc:
        h3_integration.uninstall()
        logging.error(
            "H3 Sparse experiment integration disabled (%s). Existing H3 modes are unchanged.", type(exc).__name__
        )


def _after_component(component, **kwargs):
    if h3_integration._RESTORES:
        h3_integration.after_component(component, **kwargs)


script_callbacks.on_before_ui(_install_h3)
script_callbacks.on_after_component(_after_component)
script_callbacks.on_script_unloaded(h3_integration.uninstall)


def _integer(value):
    if isinstance(value, bool) or int(value) != value:
        raise ValueError("整数の設定値が必要です。")
    return int(value)


class Script(scripts.Script):
    sorting_priority = 95

    def title(self):
        return "Anima Sparse Attention (experimental)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Anima Self-Attention · 実験", open=False):
            from modules_forge.jev_sparse.ui import budget_controls, credential_controls, decision_controls

            credential_controls("anima-i2i" if is_img2img else "anima-t2i")
            mode = gr.Dropdown(
                choices=[
                    ("OFF · 通常生成", "off"),
                    ("Dense · 比較ログのみ", "dense"),
                    ("固定SLA · 通信なし", "fixed"),
                    ("数値ルールSLA · 通信なし", "rules"),
                    ("Jev速度優先 · 集約統計を外部送信", "jev"),
                ],
                value="off",
                label="実験方式",
            )
            keep = gr.Slider(1, 100, value=75, step=1, label="固定モードの保持率 %")
            cadence, interval = decision_controls("anima-i2i" if is_img2img else "anima-t2i", mode)
            job_calls, job_wait = budget_controls(
                "anima-i2i" if is_img2img else "anima-t2i", mode, cadence=cadence, interval=interval
            )
            with gr.Row():
                minimum = gr.Number(value=4096, precision=0, label="Sparseを使う最小token数")
                warmup = gr.Slider(0, 10, value=1, step=1, label="最初のDenseモデル評価回数")
            with gr.Row():
                maximum = gr.Number(value=1, visible=False)
                timeout = gr.Slider(0.5, 20, value=3, step=0.5, label="Jev timeout（秒）")
            gr.Markdown(
                "画像の自己Attentionだけが対象です。参照latentは対象外です。Jevは25・50・75・100%、数値ルールは50・75・100%から選びます。Jevのstepはモデル評価単位です。保存したキーを使い、選んだ頻度で再判定します。API失敗後はDenseへ戻ります。ログ：`outputs/jev-sparse`。"
            )
        self.infotext_fields = [
            (mode, "Anima Sparse mode"),
            (keep, "Anima Sparse keep"),
            (minimum, "Anima Sparse min tokens"),
            (warmup, "Anima Sparse warmup"),
            (interval, "Anima Sparse interval"),
            (maximum, "Anima Sparse max calls"),
            (timeout, "Anima Sparse timeout"),
            (cadence, "Anima Sparse cadence"),
            (job_calls, "Anima Jev job max calls"),
            (job_wait, "Anima Jev job wait seconds"),
        ]
        return [mode, keep, minimum, warmup, interval, maximum, timeout, cadence, job_calls, job_wait]

    def process(
        self,
        p,
        mode,
        keep,
        minimum,
        warmup,
        interval,
        maximum,
        timeout,
        cadence="legacy",
        job_max_calls=0,
        job_max_wait_seconds=0,
    ):
        rule_interval = 4 if mode == "rules" and cadence != "legacy" else _integer(interval)
        options = AnimaOptions(
            mode,
            float(keep),
            _integer(minimum),
            _integer(warmup),
            rule_interval,
            _integer(maximum),
            float(timeout),
            decision_cadence=cadence,
            job_max_calls=_integer(job_max_calls),
            job_max_wait_seconds=job_max_wait_seconds,
        )
        options.validate()
        p._aikimi_sparse_options = options
        p._aikimi_sparse_runs = []
        p._aikimi_sparse_client = None
        p._aikimi_sparse_completed = False
        if mode != "off":
            p.extra_generation_params.update(
                {
                    "Anima Sparse mode": mode,
                    "Anima Sparse keep": keep,
                    "Anima Sparse min tokens": minimum,
                    "Anima Sparse warmup": warmup,
                    "Anima Sparse interval": options.update_interval,
                    "Anima Sparse max calls": maximum,
                    "Anima Sparse timeout": timeout,
                    "Anima Sparse cadence": cadence,
                    "Anima Jev job max calls": options.job_max_calls,
                    "Anima Jev job wait seconds": options.job_max_wait_seconds,
                }
            )

    def process_before_every_sampling(self, p, *args, **kwargs):
        options = getattr(p, "_aikimi_sparse_options", AnimaOptions())
        if options.mode == "off":
            return
        from modules_forge.jev_sparse.anima import attach

        current = p.sd_model.forge_objects.unet
        if current is getattr(p, "_aikimi_sparse_patch", None):
            current = p._aikimi_sparse_base
        for run in p._aikimi_sparse_runs:
            run.close("sampling_pass_completed")

        def cancelled():
            return bool(shared.state.interrupted or shared.state.skipped)

        p.extra_generation_params["Anima Sparse status"] = "requested_not_active"
        try:
            if options.mode == "jev" and options.max_calls and p._aikimi_sparse_client is None:
                from modules_forge.jev_sparse.credentials import cloud_source

                p._aikimi_sparse_client = create_client(
                    timeout=options.timeout,
                    cancelled=cancelled,
                    environment=None if replay_requested() else cloud_source(),
                    budget=JevBudget(options.job_max_calls, options.job_max_wait_seconds),
                    prompt_sha256=hashlib.sha256(str(p.prompt).encode()).hexdigest(),
                )
            patched, run = attach(
                current,
                options,
                Path(data_path) / "outputs" / "jev-sparse",
                prompt=str(p.prompt),
                cancelled=cancelled,
                client=p._aikimi_sparse_client,
                owns_client=False,
            )
        except Exception as exc:
            close_client(p._aikimi_sparse_client, "failed")
            reason = type(exc).__name__
            p.extra_generation_params["Anima Sparse status"] = "not_active:" + reason
            # Forge catches setup callback exceptions. Carry a rejected replay
            # into sampling so it cannot silently complete using an unpatched model.
            patched = current.clone()

            def fail(_apply_model, _args):
                raise RuntimeError("Anima Jevを開始できません。設定と判定ログを確認してください: " + reason)

            patched.set_model_unet_function_wrapper(fail)
            p._aikimi_sparse_base, p._aikimi_sparse_patch = current, patched
            p.sd_model.forge_objects.unet = patched
            return
        p.extra_generation_params["Anima Sparse status"] = "active_experiment"
        run.log.write(
            "generation_context",
            seed=getattr(p, "seed", None),
            width=getattr(p, "width", None),
            height=getattr(p, "height", None),
            steps=getattr(p, "steps", None),
            batch_size=getattr(p, "batch_size", None),
        )
        p._aikimi_sparse_base = current
        p._aikimi_sparse_patch = patched
        p._aikimi_sparse_runs.append(run)
        p.sd_model.forge_objects.unet = patched
        p.extra_generation_params["Anima Sparse log"] = str(run.log.path)

    def postprocess(self, p, processed, *args):
        p._aikimi_sparse_completed = True
        self.on_process_cleanup(p)

    def on_process_cleanup(self, p, *args):
        status = "completed" if getattr(p, "_aikimi_sparse_completed", False) else "failed"
        if shared.state.interrupted or shared.state.skipped:
            status = "cancelled"
        try:
            try:
                close_client(getattr(p, "_aikimi_sparse_client", None), status)
            except ReplayError:
                status = "failed"
                raise
        finally:
            try:
                for run in getattr(p, "_aikimi_sparse_runs", []):
                    run.close(status)
            finally:
                patched = getattr(p, "_aikimi_sparse_patch", None)
                if patched is not None and p.sd_model.forge_objects.unet is patched:
                    p.sd_model.forge_objects.unet = p._aikimi_sparse_base
