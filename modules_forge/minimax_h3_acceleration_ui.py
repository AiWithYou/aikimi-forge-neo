"""Gradio controls for independent, opt-in MiniMax H3 speed/quality trade-offs."""
from __future__ import annotations

import html

import gradio as gr

from modules_forge.minimax_h3_acceleration import H3Acceleration
from modules_forge.minimax_h3_hybrid_ui import create_hybrid_controls, hybrid_summary
from modules_forge.minimax_h3_negpip_ui import create_negpip_controls


def acceleration_note(*values) -> str:
    try:
        option = H3Acceleration.from_values(values)
    except ValueError as exc:
        return f'<div role="alert">{html.escape(str(exc))}</div>'
    notes = []
    if option.model_variant == "fused_turbo":
        notes.append("MATLOWAI版はTurboとMysticを焼き込んだ派生モデルです。動き・質感・音が標準と変わります。モデル選択だけではStepsを変えません。4/8 Stepsボタンで明示適用してください。")
    if option.video_vae == "int8":
        notes.append("INT8 ConvRot VAEは量子化による画質差やGPU依存の不具合があり得ます。黒画面・ノイズが出る場合はFP16で比較してください。")
    if option.decode_mode == "fast":
        notes.append("Fast VAEは実験的なタイル一括処理です。VRAMが増え、通常版より遅い場合もあります。node内部のOOM時はbatch 1で再試行されます。切替後はruntimeを再起動してください。")
    if option.attention != "dense":
        notes.append("Sparse Attentionは近似です。長い系列ほど有利ですが、短い系列（12288 tokens未満）と開始区間はdenseになります。音声・参照条件の行は保護しますが、全体の同一性は保証しません。")
        if option.attention == "sla":
            notes.append("SLAは特に実験的です。MATLOWAI作者の旧H3SLA nodeとは別実装で、同じ数値でも同じ結果にはなりません。VSA専用学習済み重みはこの選択には含みません。")
    if option.negpip.enabled:
        notes.append("NegPiPを有効にしています。プロンプト内の負の重みをValueへ適用します。切替後は選択設定で再起動してください。")
    if option.clip_cache != "off":
        notes.append("CLIPCachedは同じプロンプト・参照画像の条件をディスクへ保存し、再利用時のQwen3-VL読み込みを省きます。初回は通常の計算が必要です。切替後はruntimeを再起動してください。")
        if option.negpip.enabled:
            notes.append("NegPiPの重み・適用時間も専用キャッシュへ保存します。通常用とは分離し、NegPiP設定の変更時は再計算します。")
        if option.clip_cache == "refresh":
            notes.append("再計算モードを選択中です。生成するたびに条件を計算し直して保存します。再利用する場合は自動へ戻してください。")
    if not notes:
        notes.append("標準構成：従来モデル＋FP16 VAE＋通常デコード＋Kitchen dense。高速化による追加の近似は無効です。")
    return (
        '<div class="h3-settings-summary" role="status" aria-live="polite" '
        f'data-tone="{"warn" if len(notes) > 1 or option != H3Acceleration() else "ready"}">'
        '<strong>高速化の選択とトレードオフ</strong>'
        + ''.join(f'<p>{html.escape(note)}</p>' for note in notes)
        + '<small>解像度の用途プリセットとは独立した設定です。速度倍率の予測ではありません。'
        '選択後は「実行環境とモデル → 状態を再確認」で不足ファイル・nodeを確認できます。</small></div>'
    )


def _control_state(*values):
    try:
        option = H3Acceleration.from_values(values)
    except ValueError:
        return (acceleration_note(*values), *[gr.update() for _ in range(4)])
    return (acceleration_note(*values), gr.update(visible=option.decode_mode == "fast"), gr.update(visible=option.attention == "sol"), gr.update(visible=option.attention == "sla"), gr.update(visible=option.attention != "dense"))


def create_acceleration_controls(duration):
    defaults = H3Acceleration()
    with gr.Accordion("高速化 · 任意設定 / 画質・メモリとの比較", open=False, elem_id="h3-acceleration"):
        gr.Markdown("生成・デコードの高速化とCLIP条件の再利用を選べます。まず1つずつ同じSeedで比較してください。標準設定は自動変更しません。")
        model = gr.Dropdown(
            choices=[("標準 · モード別公式INT8モデル", "base"), ("MATLOWAI Fused Turbo · 派生モデル", "fused_turbo")],
            value=defaults.model_variant, label="1. 生成モデル", interactive=False,
            info="統合済みINT8 ConvRotモデルをmodels/diffusion_modelsへ配置します。配布元は下のリンクから確認できます。", elem_id="h3-model-variant",
        )
        with gr.Row():
            turbo4 = gr.Button("Turbo + 4 Stepsを適用", interactive=False, elem_id="h3-turbo-4")
            turbo8 = gr.Button("Turbo + 8 Stepsを適用", interactive=False, elem_id="h3-turbo-8")
        vae = gr.Dropdown(
            choices=[("FP16 · 標準", "fp16"), ("INT8 ConvRot · 量子化VAE", "int8")],
            value=defaults.video_vae, label="2. Video VAE", interactive=False,
            info="Kijai配布のINT8 ConvRot Video VAEをmodels/vaeへ配置します。", elem_id="h3-video-vae",
        )
        decode = gr.Dropdown(
            choices=[("VAEDecode · 標準", "standard"), ("Fast VAE Decode · 実験的", "fast")],
            value=defaults.decode_mode, label="3. Videoデコード", interactive=False,
            info="FastのみComfyUI-MiniMax-H3-MotionCacheを許可します。他のcustom nodeは無効のままです。",
            elem_id="h3-decode-mode",
        )
        batch = gr.Slider(1, 8, value=defaults.tile_batch_size, step=1, label="Fast VAE tile batch · 大きいほどVRAM増", visible=False, interactive=False, elem_id="h3-tile-batch")
        attention = gr.Dropdown(
            choices=[("Kitchen dense · 標準", "dense"), ("Sol-Attn · 適応しきい値 / 近似", "sol"), ("SLA top-k · 実験的 / 近似", "sla")],
            value=defaults.attention, label="4. Attention", interactive=False,
            info="公式BlockSparseAttention（2026-09-06追加）とcomfy-kitchen 0.2.33以上が必要です。",
            elem_id="h3-attention-mode",
        )
        tau = gr.Slider(0, 4, value=defaults.sparse_tau, step=0.05, label="Sol-Attn tau · 大きいほど疎", visible=False, interactive=False, elem_id="h3-sparse-tau")
        keep = gr.Slider(0.5, 95, value=defaults.sparse_keep_percent, step=0.5, label="SLA保持率 % · 大きいほどdenseに近い", visible=False, interactive=False, elem_id="h3-sparse-keep")
        start = gr.Slider(0, 1, value=defaults.sparse_start_percent, step=0.01, label="Sparse開始位置 · それ以前はdense", visible=False, interactive=False, elem_id="h3-sparse-start")
        clip_cache = gr.Dropdown(
            choices=[("オフ · 毎回通常処理", "off"), ("自動 · 保存した条件を再利用", "auto"), ("再計算して更新 · 毎回", "refresh")],
            value=defaults.clip_cache, label="5. CLIP条件キャッシュ", interactive=False, elem_id="h3-clip-cache",
            info="CLIPCachedが必要です。プロンプト・参照由来の条件やサムネイルを選択先ComfyUIのディスクに保存します。自動削除はありません。",
        )
        note = gr.HTML(acceleration_note(*defaults.values()), elem_id="h3-acceleration-note")
        reset = gr.Button("標準構成（高速化・キャッシュ・NegPiPオフ）+ 20 Stepsに戻す", interactive=False, elem_id="h3-acceleration-reset")
        gr.Markdown(
            "必要な重み・拡張は選択先のComfyUIへ別途導入します。この画面は自動ダウンロードしません。\n\n"
            "[INT8 VAE](https://huggingface.co/Kijai/MiniMax-H3-experimental) · "
            "[MATLOWAI Turbo](https://huggingface.co/MATLOWAI/minimax-h3-fused-turbo-int8-convrot) · "
            "[Fast VAE node](https://github.com/starsFriday/ComfyUI-MiniMax-H3-MotionCache) · "
            "[CLIPCached](https://github.com/Mu5hr00moO/ComfyUI-MiniMaxH3-CLIPCached) · "
            "[公式Sparse Attention](https://github.com/Comfy-Org/ComfyUI/commit/e308cc73b466584b0c17be695e5de1a17438bb40)"
        )
    negpip_controls = create_negpip_controls(prefix="h3", interactive=False)
    hybrid_controls, hybrid_note = create_hybrid_controls()
    controls = [model, vae, decode, batch, attention, tau, keep, start, *negpip_controls, clip_cache, *hybrid_controls]
    summary_inputs = hybrid_controls + [duration]
    for control in summary_inputs:
        control.change(hybrid_summary, inputs=summary_inputs, outputs=[hybrid_note], queue=False, show_progress="hidden")
    hybrid_controls[0].change(
        lambda enabled: gr.update(label="1区間の長さ（秒）" if enabled else "長さ（秒）"),
        inputs=[hybrid_controls[0]], outputs=[duration], queue=False, show_progress="hidden",
    )
    for control in controls:
        control.change(_control_state, inputs=controls, outputs=[note, batch, tau, keep, start], queue=False, show_progress="hidden")
    return controls, [turbo4, turbo8, reset]
