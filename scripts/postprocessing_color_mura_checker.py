from __future__ import annotations

from typing import Optional

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from modules import scripts_postprocessing
from modules.krea2_quality import (
    ChromaMuraMetrics as MuraMetrics,
    ChromaMuraParams as MuraParams,
    analyze_chroma_mura,
)
from modules.ui_components import FormRow, InputAccordion


def compute_mura(
    image: Image.Image,
    params: Optional[MuraParams] = None,
    *,
    create_views: bool = True,
    requested_views: set[str] | None = None,
):
    """
    Return the original RGBA image, optional heatmap/overlay, chroma residual,
    and metrics. The score measures only Lab a/b variation. Luminance texture,
    shading, and monochrome noise are deliberately excluded from the color-mura
    score.
    """
    params = params or MuraParams()
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    delta_chroma, confidence, valid, metrics, _ = analyze_chroma_mura(image, params)
    delta_nan = delta_chroma.copy()
    delta_nan[~valid] = np.nan

    if not create_views:
        return rgba, None, None, delta_nan, metrics

    requested_views = requested_views or {"Overlay", "Heatmap"}

    heat_max = 12.0
    norm = np.clip(delta_chroma / heat_max * 255.0, 0, 255).astype(np.uint8)
    cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    heat_bgr = cv2.applyColorMap(norm, cmap)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    heat_rgba = None
    if "Heatmap" in requested_views:
        heat_rgba = np.zeros((*rgba.shape[:2], 4), dtype=np.uint8)
        heat_rgba[..., :3] = heat_rgb
        heat_rgba[..., 3] = np.where(
            valid, np.clip(np.rint(confidence * 255.0), 0, 255), 0
        ).astype(np.uint8)

    overlay_rgba = None
    if "Overlay" in requested_views:
        overlay = rgba.copy().astype(np.float32)
        overlay_alpha = 0.65 * confidence
        blend = overlay_alpha[..., None]
        overlay[..., :3] = np.where(
            valid[..., None],
            overlay[..., :3] * (1.0 - blend) + heat_rgb.astype(np.float32) * blend,
            overlay[..., :3],
        )
        overlay_rgba = np.clip(overlay, 0, 255).astype(np.uint8)
    return rgba, heat_rgba, overlay_rgba, delta_nan, metrics


def analysis_size_for(image: Image.Image, analysis_long_edge: int) -> tuple[int, int]:
    long_edge = max(image.size)
    if long_edge <= analysis_long_edge:
        return image.size
    scale = analysis_long_edge / long_edge
    return (
        max(1, int(round(image.width * scale))),
        max(1, int(round(image.height * scale))),
    )


def format_metrics(metrics: MuraMetrics) -> str:
    return (
        f"{metrics.rough_judgement} / "
        f"p95 chroma Δ={metrics.p95_chroma_delta:.2f}, "
        f">5 area={metrics.area_chroma_delta_gt_5_pct:.2f}%, "
        f">10 area={metrics.area_chroma_delta_gt_10_pct:.2f}%, "
        f"valid={metrics.valid_area_pct:.2f}%"
    )


def add_metrics_to_info(
    info: dict, metrics: MuraMetrics, analysis_size: tuple[int, int] | None = None
) -> None:
    info["Color mura metric"] = "Lab chroma-only delta (L excluded)"
    if analysis_size is not None:
        info["Color mura analysis size"] = f"{analysis_size[0]}x{analysis_size[1]}"
    info["Color mura judgement"] = metrics.rough_judgement
    info["Color mura valid area %"] = f"{metrics.valid_area_pct:.3f}"
    info["Color mura mean chroma delta"] = f"{metrics.mean_chroma_delta:.3f}"
    info["Color mura median chroma delta"] = f"{metrics.median_chroma_delta:.3f}"
    info["Color mura p90 chroma delta"] = f"{metrics.p90_chroma_delta:.3f}"
    info["Color mura p95 chroma delta"] = f"{metrics.p95_chroma_delta:.3f}"
    info["Color mura p99 chroma delta"] = f"{metrics.p99_chroma_delta:.3f}"
    info["Color mura max chroma delta"] = f"{metrics.max_chroma_delta:.3f}"
    info["Color mura area >2 chroma delta %"] = (
        f"{metrics.area_chroma_delta_gt_2_pct:.3f}"
    )
    info["Color mura area >5 chroma delta %"] = (
        f"{metrics.area_chroma_delta_gt_5_pct:.3f}"
    )
    info["Color mura area >10 chroma delta %"] = (
        f"{metrics.area_chroma_delta_gt_10_pct:.3f}"
    )


class ScriptPostprocessingColorMuraChecker(scripts_postprocessing.ScriptPostprocessing):
    name = "Color Mura Checker"
    order = 1100

    def ui(self):
        with InputAccordion(
            True,
            label="色むら確認 / Color Mura Checker",
            elem_id="extras_color_mura_checker",
        ) as enabled:
            gr.Markdown(
                "Labのa/bだけを縮小解析します。輝度の陰影・線・無彩色ノイズは色むらスコアに含めません。"
            )
            with FormRow():
                output_modes = gr.CheckboxGroup(
                    choices=["Overlay", "Heatmap"],
                    value=[],
                    label="確認画像（必要時のみ）",
                    elem_id="extras_color_mura_outputs",
                )
                add_metrics = gr.Checkbox(
                    label="結果をPNG infoへ追加",
                    value=True,
                    elem_id="extras_color_mura_add_metrics",
                )
                ignore_near_white_bg = gr.Checkbox(
                    label="白背景を無視",
                    value=False,
                    elem_id="extras_color_mura_ignore_white_bg",
                )

            with FormRow():
                analysis_long_edge = gr.Slider(
                    minimum=512,
                    maximum=2048,
                    step=256,
                    label="解析長辺上限",
                    value=1536,
                    elem_id="extras_color_mura_analysis_long_edge",
                )
                blur_sigma = gr.Slider(
                    minimum=3,
                    maximum=80,
                    step=1,
                    label="解析画像の基準色ぼかし sigma",
                    value=18,
                    elem_id="extras_color_mura_blur_sigma",
                )
                edge_percentile = gr.Slider(
                    minimum=50,
                    maximum=100,
                    step=1,
                    label="輪郭除外 percentile（100で無効）",
                    value=96,
                    elem_id="extras_color_mura_edge_percentile",
                )
                edge_dilate = gr.Slider(
                    minimum=0,
                    maximum=20,
                    step=1,
                    label="輪郭除外の太さ",
                    value=5,
                    elem_id="extras_color_mura_edge_dilate",
                )

        return {
            "color_mura_enabled": enabled,
            "color_mura_outputs": output_modes,
            "color_mura_add_metrics": add_metrics,
            "color_mura_ignore_near_white_bg": ignore_near_white_bg,
            "color_mura_analysis_long_edge": analysis_long_edge,
            "color_mura_blur_sigma": blur_sigma,
            "color_mura_edge_percentile": edge_percentile,
            "color_mura_edge_dilate": edge_dilate,
        }

    def process(
        self,
        pp: scripts_postprocessing.PostprocessedImage,
        color_mura_enabled=True,
        color_mura_outputs=None,
        color_mura_add_metrics=True,
        color_mura_ignore_near_white_bg=False,
        color_mura_analysis_long_edge=1536,
        color_mura_blur_sigma=18,
        color_mura_edge_percentile=96,
        color_mura_edge_dilate=5,
    ):
        if not color_mura_enabled:
            return

        outputs = set(color_mura_outputs or [])
        if not outputs and not color_mura_add_metrics:
            return
        params = MuraParams(
            analysis_long_edge=int(color_mura_analysis_long_edge),
            blur_sigma=float(color_mura_blur_sigma),
            edge_percentile=float(color_mura_edge_percentile),
            edge_dilate=int(color_mura_edge_dilate),
            ignore_near_white_bg=bool(color_mura_ignore_near_white_bg),
        )
        if outputs:
            _, heat_rgba, overlay_rgba, _, metrics = compute_mura(
                pp.image,
                params,
                create_views=True,
                requested_views=outputs,
            )
            analysis_size = analysis_size_for(pp.image, params.analysis_long_edge)
        else:
            _, _, _, metrics, analysis_size = analyze_chroma_mura(
                pp.image, params, full_resolution=False
            )

        if color_mura_add_metrics:
            add_metrics_to_info(pp.info, metrics, analysis_size)
            pp.info["Color mura summary"] = format_metrics(metrics)

        if not outputs:
            return

        if "Overlay" in outputs:
            overlay = Image.fromarray(overlay_rgba, mode="RGBA")
            overlay_pp = pp.create_copy(
                overlay, nametags=["mura-overlay"], disable_processing=True
            )
            add_metrics_to_info(overlay_pp.info, metrics, analysis_size)
            overlay_pp.info["Color mura view"] = "Overlay"
            pp.extra_images.append(overlay_pp)

        if "Heatmap" in outputs:
            heatmap = Image.fromarray(heat_rgba, mode="RGBA")
            heat_pp = pp.create_copy(
                heatmap, nametags=["mura-heatmap"], disable_processing=True
            )
            add_metrics_to_info(heat_pp.info, metrics, analysis_size)
            heat_pp.info["Color mura view"] = "Heatmap"
            pp.extra_images.append(heat_pp)
