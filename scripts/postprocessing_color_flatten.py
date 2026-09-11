import gradio as gr

from modules import errors, scripts_postprocessing
from modules.color_flatten import (
    DEFAULT_GRADIENT_DETAIL_THRESHOLD,
    DEFAULT_GRADIENT_RADIUS,
    FAST_MODE,
    GRADIENT_MODE,
    MAX_GRADIENT_DETAIL_THRESHOLD,
    MAX_GRADIENT_RADIUS,
    SMART_MODE,
    SUPERPIXEL_MODE,
    color_flatten_pil,
)
from modules.krea2_quality import (
    adaptive_despeckle,
    smart_finish_image,
    smart_finish_summary,
)
from modules.ui_components import FormRow, InputAccordion


def mode_visibility(mode):
    return tuple(gr.update(visible=visible) for visible in (
        mode == SMART_MODE, mode == FAST_MODE, mode == SUPERPIXEL_MODE, mode == GRADIENT_MODE,
        mode != SMART_MODE,
    ))


class ScriptPostprocessingColorFlatten(scripts_postprocessing.ScriptPostprocessing):
    # Keep the public operation name stable: API clients and saved disable/order
    # preferences use this exact string.
    name = "Color Flatten"
    order = 1050

    def ui(self):
        with InputAccordion(
            True,
            label="Color Flatten / 色ムラ補正",
            elem_id="extras_color_flatten",
        ) as enable:
            gr.Markdown(
                "Smartはchroma色ムラだけを補正します。Smooth Gradient / AI Noiseは輝度を含む全面の微細模様を滑らかにし、強い輪郭はEdge Protectで保持します。テクスチャにも作用するため必要な画像だけで選択してください。約1K画像を強く均す目安はStrength 1.0 / Radius 12 / Detail Threshold 8です。"
            )
            with FormRow():
                mode = gr.Radio(
                    label="Mode",
                    choices=[SMART_MODE, FAST_MODE, SUPERPIXEL_MODE, GRADIENT_MODE],
                    value=SMART_MODE,
                    elem_id="extras_color_flatten_mode",
                )
                strength = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    label="Strength",
                    value=0.80,
                    elem_id="extras_color_flatten_strength",
                )
                with gr.Column(visible=False, min_width=130, scale=0) as edge_group:
                    edge_protect = gr.Checkbox(
                        label="Edge Protect",
                        value=True,
                        elem_id="extras_color_flatten_edge_protect",
                    )
                despeckle = gr.Checkbox(
                    label="孤立した白/黒粒を補修（雪・星ではOFF）",
                    value=False,
                    elem_id="extras_color_flatten_despeckle",
                )

            analysis_long_edge = gr.Slider(
                minimum=512,
                maximum=2048,
                step=256,
                label="Smart解析長辺上限",
                value=1536,
                elem_id="extras_color_flatten_analysis_long_edge",
            )

            with FormRow(visible=False) as mean_shift_group:
                mean_shift_sp = gr.Slider(
                    minimum=1,
                    maximum=80,
                    step=1,
                    label="Mean Shift Spatial",
                    value=12,
                    elem_id="extras_color_flatten_mean_shift_sp",
                )
                mean_shift_sr = gr.Slider(
                    minimum=1,
                    maximum=80,
                    step=1,
                    label="Mean Shift Color",
                    value=24,
                    elem_id="extras_color_flatten_mean_shift_sr",
                )

            with FormRow(visible=False) as superpixel_group:
                n_segments = gr.Slider(
                    minimum=100,
                    maximum=10000,
                    step=50,
                    label="SLIC Segments",
                    value=1200,
                    elem_id="extras_color_flatten_n_segments",
                )
                compactness = gr.Slider(
                    minimum=1.0,
                    maximum=40.0,
                    step=0.5,
                    label="SLIC Compactness",
                    value=12.0,
                    elem_id="extras_color_flatten_compactness",
                )

            with FormRow(visible=False) as gradient_group:
                gradient_radius = gr.Slider(
                    minimum=1.0,
                    maximum=MAX_GRADIENT_RADIUS,
                    step=1.0,
                    label="Smooth Gradient Radius (px)",
                    value=DEFAULT_GRADIENT_RADIUS,
                    elem_id="extras_color_flatten_gradient_radius",
                )
                gradient_detail_threshold = gr.Slider(
                    minimum=0.5,
                    maximum=MAX_GRADIENT_DETAIL_THRESHOLD,
                    step=0.5,
                    label=(
                        "Smooth Gradient Detail Threshold "
                        "(Lab ΔE, larger = smoother)"
                    ),
                    value=DEFAULT_GRADIENT_DETAIL_THRESHOLD,
                    elem_id="extras_color_flatten_gradient_detail_threshold",
                )

        gr.on(
            fn=mode_visibility, inputs=[mode],
            outputs=[analysis_long_edge, mean_shift_group, superpixel_group, gradient_group, edge_group],
            queue=False, show_progress="hidden",
        )

        return {
            "enable": enable,
            "mode": mode,
            "strength": strength,
            "edge_protect": edge_protect,
            "despeckle": despeckle,
            "analysis_long_edge": analysis_long_edge,
            "mean_shift_sp": mean_shift_sp,
            "mean_shift_sr": mean_shift_sr,
            "n_segments": n_segments,
            "compactness": compactness,
            "gradient_radius": gradient_radius,
            "gradient_detail_threshold": gradient_detail_threshold,
        }

    def process(
        self,
        pp: scripts_postprocessing.PostprocessedImage,
        enable=True,
        mode=SMART_MODE,
        strength=0.80,
        edge_protect=True,
        despeckle=False,
        analysis_long_edge=1536,
        mean_shift_sp=12,
        mean_shift_sr=24,
        n_segments=1200,
        compactness=12.0,
        gradient_radius=DEFAULT_GRADIENT_RADIUS,
        gradient_detail_threshold=DEFAULT_GRADIENT_DETAIL_THRESHOLD,
    ):
        if not enable:
            return

        try:
            strength = float(strength)
            edge_protect = bool(edge_protect)
            mean_shift_sp = int(mean_shift_sp)
            mean_shift_sr = int(mean_shift_sr)
            n_segments = int(n_segments)
            compactness = float(compactness)
            if mode == GRADIENT_MODE:
                gradient_radius = float(gradient_radius)
                gradient_detail_threshold = float(gradient_detail_threshold)

            if mode == SMART_MODE:
                pp.image, report = smart_finish_image(
                    pp.image,
                    color_strength=strength,
                    analysis_long_edge=int(analysis_long_edge),
                    despeckle=bool(despeckle),
                )
            else:
                if despeckle:
                    pp.image, speckle_report = adaptive_despeckle(pp.image)
                else:
                    speckle_report = {
                        "applied": False,
                        "masked_pixels": 0,
                        "masked_percent": 0.0,
                        "reason": "despeckle disabled",
                    }
                pp.image = color_flatten_pil(
                    pp.image,
                    mode,
                    strength,
                    edge_protect,
                    mean_shift_sp,
                    mean_shift_sr,
                    n_segments,
                    compactness,
                    gradient_radius,
                    gradient_detail_threshold,
                )
                report = {"speckle": speckle_report}
        except Exception:
            errors.report("Error running Color Flatten", exc_info=True)
            return

        if mode == GRADIENT_MODE:
            pp.info["Color Flatten"] = mode
            pp.info["Color Flatten strength"] = strength
            pp.info["Color Flatten edge protect"] = edge_protect
            pp.info["Smooth Gradient radius"] = gradient_radius
            pp.info["Smooth Gradient detail threshold"] = gradient_detail_threshold
            pp.info["Smooth Gradient despeckle"] = bool(despeckle)
            pp.info["Smooth Gradient despeckle applied"] = report["speckle"].get(
                "applied", False
            )
            pp.info["Smooth Gradient despeckle pixels"] = report["speckle"].get(
                "masked_pixels", 0
            )
            pp.info["Smooth Gradient despeckle reason"] = report["speckle"].get(
                "reason", "unknown"
            )
            return

        pp.info["Krea2 Smart Finish"] = mode
        pp.info["Krea2 Smart Finish strength"] = strength
        pp.info["Krea2 Smart Finish despeckle"] = bool(despeckle)
        if mode == SMART_MODE:
            pp.info["Krea2 Smart Finish summary"] = smart_finish_summary(report)
            chroma = report["chroma_mura"]
            pp.info["Krea2 Smart Finish chroma applied"] = chroma["applied"]
            pp.info["Krea2 Smart Finish chroma p95 before"] = (
                f"{chroma['before']['p95_chroma_delta']:.3f}"
            )
            pp.info["Krea2 Smart Finish chroma p95 after"] = (
                f"{chroma['after']['p95_chroma_delta']:.3f}"
            )
            pp.info["Krea2 Smart Finish speckle pixels"] = report["speckle"].get(
                "masked_pixels", 0
            )
        else:
            pp.info["Krea2 Smart Finish summary"] = (
                f"manual mode={mode}, "
                f"speckles={report['speckle'].get('masked_pixels', 0)}"
            )
            pp.info["Color Flatten"] = mode
            pp.info["Color Flatten strength"] = strength
            pp.info["Color Flatten edge protect"] = edge_protect
