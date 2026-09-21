"""Krea2 generation controls; secrets use the shared external credential store."""

from __future__ import annotations

import gradio as gr

from modules import scripts
from modules_forge.jev_sparse import krea2, krea2_jobs
from modules_forge.jev_sparse.ui import credential_controls


class Script(scripts.Script):
    sorting_priority = 96

    def title(self):
        return krea2_jobs.SCRIPT_TITLE

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        prefix = "krea2-jev-i2i" if is_img2img else "krea2-jev-t2i"
        with gr.Accordion("Krea2 · Jev高速化", open=False, elem_id=prefix):
            mode = gr.Dropdown(
                choices=[
                    ("OFF · 通常", "off"),
                    ("固定Sparse", "fixed"),
                    ("数値ルール", "rules"),
                    ("Jev · 層ごとに自動選択", "jev"),
                    ("Dense · 比較用の記録", "dense"),
                ],
                value="off",
                label="Attention",
                elem_id=prefix + "-mode",
            )
            keep = gr.Slider(
                1, 100, value=10, step=1, label="固定Sparseの保持率 %", visible=False, elem_id=prefix + "-keep"
            )
            mode.change(
                lambda value: gr.update(visible=value == "fixed"), inputs=mode, outputs=keep, api_visibility="private"
            )
            tile_mode = gr.Dropdown(
                choices=[("既存の配分", "off"), ("速度優先の数値ルール", "rules"), ("Jev · 細部量で配分", "jev")],
                value="off",
                label="4K/8Kタイルのstep配分（VRAM-Canvas）",
                visible=is_img2img,
                elem_id=prefix + "-tiles",
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
                    "文章・参照画像のAttentionは保護します。層の判定とタイル配分は各1回まで。"
                    "タイルごとに通信せず、同じ画像の処理中は選択を再利用します。"
                    "タイル配分では細部の少ない領域の再描画を省く場合があります。"
                )
            credential_controls(prefix)
        self.infotext_fields = [
            (mode, "Krea2 Sparse mode"),
            (keep, "Krea2 Sparse keep"),
            (minimum, "Krea2 Sparse min tokens"),
            (tile_mode, "Krea2 tile allocation"),
            (timeout, "Krea2 Sparse timeout"),
        ]
        return [mode, keep, minimum, tile_mode, timeout]

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
