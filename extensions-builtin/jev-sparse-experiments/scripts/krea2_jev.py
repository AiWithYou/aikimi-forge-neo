"""Krea2 generation controls; secrets use the shared external credential store."""

from __future__ import annotations

import gradio as gr

from modules import scripts
from modules_forge.jev_sparse import krea2, krea2_jobs
from modules_forge.jev_sparse.ui import credential_controls


def api_usage(mode, tile_mode, cadence="once", interval=2, tile_cadence="once"):
    calls = int(mode == "jev") + int(tile_mode == "jev")
    if not calls:
        return "**Jev API：0回** · キーなしで使えます。"
    parts = []
    if mode == "jev":
        parts.append(
            "層：生成全体で1回"
            if cadence == "once"
            else "層：毎step（最初の評価を除く）"
            if cadence == "step"
            else f"層：最初の評価後、{int(interval)} stepごと"
        )
    if tile_mode == "jev":
        parts.append("タイル：生成全体で1回" if tile_cadence == "once" else "タイル：拡大段階ごとに1回")
    repeating = (mode == "jev" and cadence != "once") or (tile_mode == "jev" and tile_cadence != "once")
    title = "Jev API" if repeating else f"Jev API：最大{calls}回／1生成"
    return f"**{title}** · " + " ／ ".join(parts)


def update_controls(mode, tile_mode, cadence, interval, tile_cadence):
    return (
        gr.update(interactive=mode == "fixed"),
        gr.update(visible=mode == "jev"),
        gr.update(visible=mode == "jev" and cadence == "interval"),
        gr.update(visible=tile_mode == "jev"),
        api_usage(mode, tile_mode, cadence, interval, tile_cadence),
    )


class Script(scripts.Script):
    sorting_priority = 96

    def title(self):
        return krea2_jobs.SCRIPT_TITLE

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        prefix = "krea2-jev-i2i" if is_img2img else "krea2-jev-t2i"
        with gr.Accordion("Krea2 · Jev高速化", open=False, elem_id=prefix):
            mode = gr.Radio(
                choices=[
                    ("OFF", "off"),
                    ("固定率", "fixed"),
                    ("Jev自動", "jev"),
                    ("数値ルール", "rules"),
                    ("Dense記録", "dense"),
                ],
                value="off",
                label="層ごとの高速化",
                elem_id=prefix + "-mode",
            )
            keep = gr.Slider(
                1,
                100,
                value=10,
                step=1,
                label="固定率：画像Attentionの保持率 %",
                info="固定率を選ぶと調整できます。小さいほど計算範囲を絞り、100%で通常の計算になります。",
                interactive=False,
                elem_id=prefix + "-keep",
            )
            with gr.Group(visible=False) as cadence_group:
                cadence = gr.Radio(
                    choices=[("初回のみ", "once"), ("指定間隔", "interval"), ("毎step", "step")],
                    value="once",
                    label="層のJev再判定頻度",
                    info="stepはサンプラーのモデル評価単位。最初の評価で統計を集め、次の評価から判定します。",
                    elem_id=prefix + "-cadence",
                )
                with gr.Group(visible=False) as interval_group:
                    interval = gr.Slider(
                        1,
                        100,
                        value=2,
                        step=1,
                        label="再判定する間隔（step）",
                        elem_id=prefix + "-interval",
                    )
            tile_mode = gr.Radio(
                choices=[("OFF", "off"), ("数値ルール", "rules"), ("Jev自動", "jev")],
                value="off",
                label="4K/8Kタイルの高速化（VRAM-Canvas）",
                info="層の設定と別に切り替えられます。OFFでは既存のstep配分を使います。",
                visible=is_img2img,
                elem_id=prefix + "-tiles",
            )
            with gr.Group(visible=False) as tile_cadence_group:
                tile_cadence = gr.Radio(
                    choices=[("初回のみ", "once"), ("拡大段階ごと", "stage")],
                    value="once",
                    label="タイル配分のJev再判定頻度",
                    elem_id=prefix + "-tile-cadence",
                )
            usage = gr.Markdown(api_usage("off", "off"), elem_id=prefix + "-api-usage")
            frequency_inputs = [mode, tile_mode, cadence, interval, tile_cadence]
            for control in frequency_inputs:
                control.change(
                    update_controls,
                    inputs=frequency_inputs,
                    outputs=[keep, cadence_group, interval_group, tile_cadence_group, usage],
                    queue=False,
                    show_progress="hidden",
                    api_visibility="private",
                )
            with gr.Accordion("詳細", open=False):
                minimum = gr.Number(
                    value=4096,
                    minimum=64,
                    maximum=1048576,
                    precision=0,
                    label="Sparseを使う最小画像token数",
                    elem_id=prefix + "-minimum",
                )
                timeout = gr.Slider(0.5, 20, value=10, step=0.5, label="Jevの待ち時間上限（秒）")
                gr.Markdown(
                    "保持率は画像Attentionの計算範囲で、画質の保持率ではありません。"
                    "文章・参照画像のAttentionは保護します。初回のみでは生成全体で判定を再利用します。"
                    "層を再判定する設定では、各タイルでも指定した頻度で問い合わせます。"
                    "サンプラーによっては表示上の1stepでモデルを複数回評価します。"
                    "タイル配分では細部の少ない領域の再描画を省く場合があります。"
                    "Dense記録は通常のAttentionで比較ログだけを記録します。"
                )
            credential_controls(prefix)
        self.infotext_fields = [
            (mode, "Krea2 Sparse mode"),
            (keep, "Krea2 Sparse keep"),
            (minimum, "Krea2 Sparse min tokens"),
            (tile_mode, "Krea2 tile allocation"),
            (timeout, "Krea2 Sparse timeout"),
            (cadence, "Krea2 Sparse cadence"),
            (interval, "Krea2 Sparse interval"),
            (tile_cadence, "Krea2 tile cadence"),
        ]
        return [mode, keep, minimum, tile_mode, timeout, cadence, interval, tile_cadence]

    def process(self, p, *args):
        options = krea2_jobs.parse_options(*args)
        session = krea2_jobs.current_session()
        p._krea2_jev_owned = session is None
        p._krea2_jev_session = session or krea2_jobs.Session(options)
        p._krea2_jev_completed = False
        if options.mode != "off":
            p.extra_generation_params.update(
                {
                    "Krea2 Sparse mode": options.mode,
                    "Krea2 Sparse keep": options.keep_percent,
                    "Krea2 Sparse min tokens": options.min_tokens,
                    "Krea2 Sparse timeout": options.timeout,
                    "Krea2 Sparse cadence": options.decision_cadence,
                    "Krea2 Sparse interval": options.update_interval,
                }
            )

    def process_before_every_sampling(self, p, *args, **kwargs):
        session = getattr(p, "_krea2_jev_session", None)
        if session is None or session.options.mode == "off":
            return
        current = p.sd_model.forge_objects.unet
        if current is getattr(p, "_krea2_jev_patch", None):
            current = p._krea2_jev_base
        model = current.get_model_object("diffusion_model")
        if type(model).__module__ != "backend.nn.krea":
            p.extra_generation_params["Krea2 Sparse status"] = "not_krea2"
            return
        try:
            patched, session.run = krea2.attach(
                current,
                session.options,
                session.log_root,
                run=session.run,
                prompt=str(p.prompt),
                cancelled=krea2_jobs.cancelled,
            )
        except Exception as exc:
            # Forge catches script callback exceptions. Raise during sampling too,
            # so an explicitly requested experiment cannot silently become OFF.
            reason = type(exc).__name__
            patched = current.clone()

            def fail(_apply_model, _args):
                raise RuntimeError("Krea2 Jevを開始できません。キー・SDK・CUDA対応を確認してください: " + reason)

            patched.set_model_unet_function_wrapper(fail)
            p.extra_generation_params["Krea2 Sparse status"] = "not_active:" + reason
        else:
            p.extra_generation_params.update(
                {"Krea2 Sparse status": "active", "Krea2 Sparse log": str(session.run.log.path)}
            )
        p._krea2_jev_base, p._krea2_jev_patch = current, patched
        p.sd_model.forge_objects.unet = patched

    def postprocess(self, p, processed, *args):
        p._krea2_jev_completed = True

    def on_process_cleanup(self, p, *args):
        patched = getattr(p, "_krea2_jev_patch", None)
        if patched is not None and p.sd_model.forge_objects.unet is patched:
            p.sd_model.forge_objects.unet = p._krea2_jev_base
        p._krea2_jev_patch = None
        session = getattr(p, "_krea2_jev_session", None)
        if session is not None and getattr(p, "_krea2_jev_owned", False):
            status = "completed" if getattr(p, "_krea2_jev_completed", False) else "failed"
            session.close("cancelled" if krea2_jobs.cancelled() else status)
