import json

import gradio as gr

from modules import scripts_postprocessing
from modules.grain_cleaner import GrainAnalysisCache, GrainSettings, clean_grain
from modules.ui_components import FormRow, InputAccordion


class ScriptPostprocessingGrainCleaner(scripts_postprocessing.ScriptPostprocessing):
    name = "Grain Cleaner"
    order = 1040

    def __init__(self):
        self.analysis_cache = GrainAnalysisCache()

    def image_changed(self):
        self.analysis_cache.clear()

    def ui(self):
        with InputAccordion(
            False, label="Grain Cleaner / 微細な粒状感を抑制", elem_id="extras_grain_cleaner"
        ) as enable:
            gr.Markdown(
                "原寸で細部を保護しながら粒を弱めます。自動推定の参考範囲が不足する画像は変更しません。単独で効果を見る場合は Color Flatten をオフにしてください。"
            )
            with FormRow():
                strength = gr.Slider(0, 1, value=0.5, step=0.01, label="全体強度")
                preserve_detail = gr.Slider(0, 1, value=0.75, step=0.01, label="細部保護")
            with FormRow():
                chroma_strength = gr.Slider(0, 1, value=0.6, step=0.01, label="色の粒の抑制")
                grain_scale = gr.Slider(0.5, 2, value=1, step=0.05, label="粒サイズ")
            with gr.Accordion("見本範囲・マスク・診断", open=False):
                mode = gr.Radio([("自動", "auto"), ("見本範囲", "sample")], value="auto", label="振幅の推定")
                with gr.Column(visible=False) as reference_group:
                    self.reference_preview = gr.Image(
                        label="見本の対角2点を選択（Upscaleオフ時）",
                        type="pil",
                        interactive=False,
                        format="png",
                        height=320,
                        buttons=[],
                    )
                    self.reference_note = gr.Markdown("画像を読み込んでください。")
                    self.reference_master = gr.State(None)
                    self.reference_size = gr.State(None)
                    self.reference_corner = gr.State(None)
                sample_roi = gr.Textbox(
                    label="見本範囲 x, y, width, height",
                    placeholder="例: 0, 0, 128, 128",
                    info="滑らかにしたい面だけを選びます。向き補正後の原寸座標、縦横32画素以上。",
                    visible=False,
                )
                mode.change(
                    lambda value: (gr.update(visible=value == "sample"), gr.update(visible=value == "sample")),
                    inputs=[mode],
                    outputs=[reference_group, sample_roi],
                    queue=False,
                    show_progress="hidden",
                )
                with FormRow():
                    apply_mask = gr.Image(
                        type="pil", image_mode="L", label="処理マスク（白＝許可、黒＝変更禁止）", sources=["upload"]
                    )
                    protect_mask = gr.Image(
                        type="pil", image_mode="L", label="保護マスク（白＝完全保護）", sources=["upload"]
                    )
                diagnostics = gr.Checkbox(False, label="差分×8・減衰率・保護・参考範囲を出力")
        return dict(
            enable=enable,
            strength=strength,
            preserve_detail=preserve_detail,
            chroma_strength=chroma_strength,
            grain_scale=grain_scale,
            mode=mode,
            sample_roi=sample_roi,
            apply_mask=apply_mask,
            protect_mask=protect_mask,
            diagnostics=diagnostics,
        )

    def process(
        self,
        pp,
        enable=False,
        strength=0.5,
        preserve_detail=0.75,
        chroma_strength=0.6,
        grain_scale=1.0,
        mode="auto",
        sample_roi="",
        apply_mask=None,
        protect_mask=None,
        diagnostics=False,
    ):
        if not enable:
            self.analysis_cache.clear()
            return
        try:
            roi = tuple(int(v.strip()) for v in sample_roi.split(",")) if mode == "sample" and sample_roi else None
            settings = GrainSettings(
                float(strength), float(preserve_detail), float(chroma_strength), float(grain_scale), mode
            )
            output, report, views = clean_grain(
                pp.image,
                settings,
                sample_roi=roi,
                apply_mask=apply_mask,
                protect_mask=protect_mask,
                diagnostics=diagnostics,
                cache=self.analysis_cache,
            )
        except (ValueError, TypeError) as exc:
            raise gr.Error(str(exc)) from exc
        pp.image = output
        pp.info["Grain Cleaner"] = json.dumps(report, ensure_ascii=False)
        if report["status"] == "INSUFFICIENT_REFERENCE":
            gr.Warning(
                "Grain Cleaner: 参考パッチが4個未満のため変更しませんでした。必要なら見本範囲を指定してください。"
            )
        for name, view in views.items():
            extra = pp.create_copy(view, nametags=["grain-" + name], disable_processing=True)
            extra.info.update(pp.info)
            pp.extra_images.append(extra)
