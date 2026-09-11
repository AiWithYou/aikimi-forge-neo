"""長尺生成の入力と全体尺の表示。"""

from __future__ import annotations

import html

import gradio as gr

from modules_forge.minimax_h3_hybrid import H3Hybrid


def hybrid_summary(enabled, windows, overlap, switch_step, prompts, duration):
    try:
        option = H3Hybrid(enabled, windows, overlap, switch_step, prompts)
        option.validate()
        if not enabled:
            return ""
        frames = int(float(duration) * 24 + 0.5)
        frames += (5 - frames % 17) % 17
        total = option.frame_count(frames)
        return (
            '<div class="h3-settings-summary" role="status" aria-live="polite">'
            f"<strong>合計 {total / 24:.2f} 秒</strong><span>{int(windows)} 区間 · {total} フレーム</span></div>"
        )
    except (ValueError, TypeError, OverflowError) as exc:
        return html.escape(str(exc))


def create_hybrid_controls():
    with gr.Accordion("長尺生成", open=False, elem_id="h3-hybrid"):
        enabled = gr.Checkbox(False, label="長尺生成を使う", interactive=False, elem_id="h3-hybrid-enabled")
        windows = gr.Slider(2, 10, value=3, step=1, label="区間数", interactive=False, elem_id="h3-hybrid-windows")
        overlap = gr.Slider(
            5, 345, value=39, step=17, label="区間の重なり（フレーム）", interactive=False, elem_id="h3-hybrid-overlap"
        )
        switch = gr.Slider(
            1,
            100,
            value=16,
            step=1,
            label="全体の仕上げへ切り替えるステップ",
            interactive=False,
            elem_id="h3-hybrid-switch",
        )
        prompts = gr.Textbox(
            value="",
            lines=4,
            label="区間ごとの動き",
            info="1行に1区間。空欄なら全区間に共通プロンプトを使います。",
            interactive=False,
            elem_id="h3-hybrid-prompts",
        )
        summary = gr.HTML("", elem_id="h3-hybrid-summary")
    return [enabled, windows, overlap, switch, prompts], summary
