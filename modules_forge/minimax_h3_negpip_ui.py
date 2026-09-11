"""Shared, optional NegPiP controls for the H3 video and image tabs."""

from __future__ import annotations

import gradio as gr

from modules_forge.minimax_h3_negpip import H3NegPiP


def create_negpip_controls(*, prefix: str, interactive: bool = True, image_mode: bool = False) -> list:
    defaults = H3NegPiP()
    with gr.Accordion("NegPiP · プロンプト内の負の重み", open=False, elem_id=f"{prefix}-negpip"):
        enabled = gr.Checkbox(
            value=False, label="NegPiPを有効にする", interactive=interactive, elem_id=f"{prefix}-negpip-enabled"
        )
        gr.Markdown(
            "例: `An apple on the table. (red fruit:-1.0)`。負の語句を本文から分離し、AttentionのValueを反転します。"
            "通常のネガティブプロンプト欄とは別の機能です。強さは小さい値から比較してください。\n\n"
            "有効／無効の切替後は「選択設定で再起動」。有効時は同梱の固定版だけを選択先ComfyUIへコピー・許可します。"
            "Sparse Attentionとの併用は未検証のため無効です。"
        )
        strength = gr.Slider(
            0,
            8,
            value=defaults.value_strength,
            step=0.05,
            label="負の重みへの倍率（1: 指定どおり）",
            interactive=interactive,
        )
        positive = gr.Checkbox(value=True, label="正の重みもValueへ適用する", interactive=interactive)
        with gr.Row():
            start = gr.Number(
                value=0, minimum=0, maximum=999, precision=0, label="開始ブロック", interactive=interactive
            )
            end = gr.Number(
                value=999,
                minimum=0,
                maximum=999,
                precision=0,
                label="終了ブロック（999: 最後まで）",
                interactive=interactive,
            )
            stride = gr.Number(
                value=1, minimum=1, maximum=16, precision=0, label="ブロック間隔", interactive=interactive
            )
        protect = gr.Checkbox(
            value=False, label="テキスト行を保護（強い負の重みで逆効果になる場合）", interactive=interactive
        )
        measure = gr.Checkbox(value=False, label="Attention量を診断ログへ出力（追加負荷あり）", interactive=interactive)
        if image_mode:
            gr.Markdown("静止画は内部5フレームです。時間指定は使わず、通常の負の重みで比較してください。")
        else:
            gr.Markdown(
                "時間指定例: `(neon lights:-1.5@v2.5-4.0)`。`@v`は映像、`@a`は音声、指定なしは両方です。時間指定の効きは実験的です。"
            )
        gr.Markdown(
            "[原作者の説明](https://github.com/hako-mikan/comfyui-minimax-h3-negpip/blob/f725718b4c597fab92cdb1fe981dfe3544a75665/README_jp.md)"
        )
    return [enabled, strength, positive, start, end, stride, protect, measure]
