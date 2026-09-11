import json
import shutil
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from backend import memory_management
import modules.scripts as scripts
from modules import devices, images, processing
from modules.krea2_quality import smart_finish_image, smart_finish_summary
from modules.shared import opts, state
from modules_forge.krea2_highres import (
    EXACT_IMG2IMG_STEPS,
    EXACT_IMG2IMG_STEPS_SCOPE,
    KREA2_PHASEWEAVE_PRODUCT_NAME,
    KREA2_PHASEWEAVE_PROFILE_KEY,
    ProcessingSnapshot,
    internal_exact_img2img_steps,
    krea2_detail_prompt,
    krea2_vram_canvas_profile,
)
from modules_forge.krea2_upscale import replace_infotext_size, target_size
from modules_forge.workflow_ui import (
    workflow_hero,
    workflow_section,
    workflow_summary,
)
from modules_forge.vram_canvas import (
    CONSENSUS_MERGE_MODE,
    DEFAULT_NOVEL_DETAIL_CONSENSUS_SIGMA,
    DEFAULT_NOVEL_DETAIL_CONSENSUS_STRENGTH,
    DEFAULT_NOVEL_DETAIL_INNER_RADIUS,
    DEFAULT_NOVEL_DETAIL_OUTER_RADIUS,
    DEFAULT_NOVEL_DETAIL_STRUCTURE_SIGMA,
    GIB,
    PHASE_WEAVE_CONTEXT_RADIUS,
    PHASE_WEAVE_MERGE_MODE,
    adaptive_step_count,
    balanced_virtual_axis_origin,
    consensus_gated_residual,
    coordinate_seed,
    detail_score,
    extract_tile_context,
    frequency_detail_delta,
    novel_detail_delta,
    phase_normalized_tile_weight,
    phase_weave_configuration,
    phase_weave_residual,
    phase_weight_normalizers,
    plan_tiles,
    progressive_stage_sizes,
    replace_infotext_seed,
    resolve_core_overlap,
    resolve_halo,
    resolve_tile_size,
    vram_canvas_work_bytes_per_pixel,
)

MAX_OUTPUT_PIXELS = 70_000_000
QUALITY_PROFILE_NAMES = {
    "Structure Safe": "structure_safe",
    "Krea2 Dense Detail 4K": "dense_detail_4k",
    "Krea2 Texture Rich 4K (Experimental)": "texture_rich_4k",
    "Krea2 PhaseWeave 4K (Experimental)": KREA2_PHASEWEAVE_PROFILE_KEY,
    "Krea2 Dense Detail 8K": "dense_detail_8k",
}
DEFAULT_QUALITY_PROFILE_LABEL = "Krea2 Dense Detail 4K"


class VRAMCanvasHighres(scripts.Script):
    def title(self):
        return "VRAM-Canvas 4K/8K Highres"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        default_profile = krea2_vram_canvas_profile(
            QUALITY_PROFILE_NAMES[DEFAULT_QUALITY_PROFILE_LABEL]
        )
        gr.HTML(
            workflow_hero(
                "VRAM-Canvas 4K / 8K",
                "入力の構図と色を基準に固定し、VRAM予算内のタイルを段階処理して局所ディテールを追加します。",
                badges=("通常img2img", "Batch 1 × 1", "Krea2推奨", "4Kから確認"),
                steps=(
                    "img2imgへ基準画像を入れる",
                    "まず「4K Smart」を適用する",
                    "4Kを確認後、その画像から8Kへ進む",
                ),
            ),
            elem_classes=["neo-workflow-hero-host"],
        )

        gr.HTML(
            workflow_section(
                1,
                "クイック設定",
                "推奨は4K Smart。実験プロファイルは通常版と別に比較してください。",
            ),
            elem_classes=["neo-workflow-section-host"],
        )
        with gr.Row(elem_classes=["neo-workflow-preset-grid"]):
            quick_4k = gr.Button(
                "4K Smart\n推奨・構図優先",
                variant="primary",
                elem_id=self.elem_id("quick_4k"),
                elem_classes=["neo-workflow-action"],
            )
            quick_phaseweave_4k = gr.Button(
                "PhaseWeave 4K\n細線・境界を比較",
                elem_id=self.elem_id("quick_phaseweave_4k"),
                elem_classes=["neo-workflow-action"],
            )
        with gr.Row(elem_classes=["neo-workflow-preset-grid"]):
            quick_texture_4k = gr.Button(
                "Texture Rich 4K\n書き込み強め・実験",
                elem_id=self.elem_id("quick_texture_4k"),
                elem_classes=["neo-workflow-action"],
            )
            quick_8k = gr.Button(
                "8K Smart\n承認済み4Kから正確に2倍",
                elem_id=self.elem_id("quick_8k"),
                elem_classes=["neo-workflow-action"],
            )
        with gr.Row(elem_classes=["neo-workflow-grid-2"]):
            quality_profile = gr.Dropdown(
                label="品質プロファイル",
                choices=list(QUALITY_PROFILE_NAMES),
                value=DEFAULT_QUALITY_PROFILE_LABEL,
                elem_id=self.elem_id("quality_profile"),
                tooltip="選択すると関連する品質設定をまとめて更新します。",
            )
            apply_quality_profile = gr.Button(
                "プロファイルを再適用",
                elem_id=self.elem_id("apply_quality_profile"),
                elem_classes=["neo-workflow-action"],
            )

        workflow_status = gr.HTML(
            self._workflow_summary_html(
                4096,
                0,
                0,
                DEFAULT_QUALITY_PROFILE_LABEL,
                0,
                0,
                default_profile["phase_count"],
                default_profile["minimum_steps"],
                default_profile["maximum_steps"],
                True,
                default_profile["merge_mode"],
            ),
            elem_classes=["neo-workflow-summary-host"],
        )

        gr.HTML(
            workflow_section(
                2,
                "納品サイズ",
                "幅・高さを0にすると入力比率を保ち、長辺を基準にします。",
            ),
            elem_classes=["neo-workflow-section-host"],
        )
        with gr.Row(elem_classes=["neo-workflow-grid-2"]):
            final_long_edge = gr.Slider(
                label="最終長辺",
                minimum=512,
                maximum=8192,
                step=64,
                value=4096,
                elem_id=self.elem_id("final_long_edge"),
                tooltip="8192は承認済み4Kを入力にした正確な2倍モードです。",
            )

        with gr.Row(elem_classes=["neo-workflow-grid-2"]):
            final_width = gr.Slider(
                label="最終幅（0 = 長辺指定）",
                minimum=0,
                maximum=8192,
                step=16,
                value=0,
                elem_id=self.elem_id("final_width"),
            )
            final_height = gr.Slider(
                label="最終高さ（0 = 長辺指定）",
                minimum=0,
                maximum=8192,
                step=16,
                value=0,
                elem_id=self.elem_id("final_height"),
            )

        gr.HTML(
            workflow_section(
                3,
                "品質とメモリ",
                "0は自動。RTX 3090ではまず自動設定のまま試せます。",
            ),
            elem_classes=["neo-workflow-section-host"],
        )
        with gr.Row(elem_classes=["neo-workflow-grid-3"]):
            vram_budget_gib = gr.Slider(
                label="VRAM予算 GiB（0 = 自動）",
                minimum=0,
                maximum=48,
                step=0.5,
                value=0,
                elem_id=self.elem_id("vram_budget_gib"),
                tooltip="GPUの総VRAMに合わせて自動計算する場合は0のままにします。",
            )
            model_reserve_gib = gr.Slider(
                label="モデル予約 GiB",
                minimum=0,
                maximum=16,
                step=0.25,
                value=5.5,
                elem_id=self.elem_id("model_reserve_gib"),
            )
            tile_size = gr.Dropdown(
                label="拡散タイル辺（0 = 自動）",
                choices=[0, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280],
                value=0,
                elem_id=self.elem_id("tile_size"),
            )

        with gr.Row(elem_classes=["neo-workflow-grid-2"]):
            phase_count = gr.Radio(
                label="グリッド位相数",
                choices=(1, 2),
                value=default_profile["phase_count"],
                elem_id=self.elem_id("phase_count"),
            )
            merge_mode = gr.Dropdown(
                label="位相マージ方式",
                choices=(CONSENSUS_MERGE_MODE, PHASE_WEAVE_MERGE_MODE),
                value=default_profile["merge_mode"],
                elem_id=self.elem_id("merge_mode"),
            )
        with gr.Row(elem_classes=["neo-workflow-grid-2"]):
            minimum_steps = gr.Slider(
                label="最小ステップ",
                minimum=1,
                maximum=12,
                step=1,
                value=default_profile["minimum_steps"],
                elem_id=self.elem_id("minimum_steps"),
            )
            maximum_steps = gr.Slider(
                label="最大ステップ",
                minimum=1,
                maximum=20,
                step=1,
                value=default_profile["maximum_steps"],
                elem_id=self.elem_id("maximum_steps"),
            )

        with gr.Row(elem_classes=["neo-workflow-grid-3"]):
            coarse_denoise = gr.Slider(
                label="粗段階 denoise",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=default_profile["coarse_denoise"],
                elem_id=self.elem_id("coarse_denoise"),
            )
            final_denoise = gr.Slider(
                label="最終 denoise",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=default_profile["denoise"],
                elem_id=self.elem_id("final_denoise"),
            )
            save_stages = gr.Checkbox(
                label="中間段階PNGを保存",
                value=False,
                elem_id=self.elem_id("save_stages"),
            )

        with gr.Accordion(
            "詳細設定 · 段階拡大と周波数マージ",
            open=False,
            elem_classes=["neo-workflow-accordion"],
        ):
            max_stage_scale = gr.Slider(
                label="1段階あたりの最大拡大率",
                minimum=1.25,
                maximum=2.0,
                step=0.05,
                value=2.0,
                elem_id=self.elem_id("max_stage_scale"),
            )
            with gr.Row():
                detail_knee = gr.Slider(
                    label="Adaptive Steps Detail Knee",
                    minimum=0.005,
                    maximum=0.10,
                    step=0.005,
                    value=default_profile["detail_knee"],
                    elem_id=self.elem_id("detail_knee"),
                )
                low_pass_radius = gr.Slider(
                    label="Low-pass Radius",
                    minimum=1,
                    maximum=48,
                    step=1,
                    value=default_profile["low_pass_radius"],
                    elem_id=self.elem_id("low_pass_radius"),
                )
            with gr.Row():
                detail_gain = gr.Slider(
                    label="Detail Gain",
                    minimum=0.1,
                    maximum=2.0,
                    step=0.05,
                    value=default_profile["detail_gain"],
                    elem_id=self.elem_id("detail_gain"),
                )
                max_detail_delta = gr.Slider(
                    label="Maximum Detail Delta",
                    minimum=1,
                    maximum=64,
                    step=1,
                    value=default_profile["max_detail_delta"],
                    elem_id=self.elem_id("max_detail_delta"),
                )
            with gr.Row():
                structure_sigma = gr.Slider(
                    label="Structure Gate Sigma",
                    minimum=1,
                    maximum=64,
                    step=1,
                    value=default_profile["structure_sigma"],
                    elem_id=self.elem_id("structure_sigma"),
                )
                base_detail_sigma = gr.Slider(
                    label="Base Detail Protection (0 = Off)",
                    minimum=0,
                    maximum=24,
                    step=0.5,
                    value=default_profile["base_detail_sigma"],
                    elem_id=self.elem_id("base_detail_sigma"),
                )
                consensus_sigma = gr.Slider(
                    label="Consensus Noise Floor (0 = Off)",
                    minimum=0,
                    maximum=32,
                    step=1,
                    value=default_profile["consensus_sigma"],
                    elem_id=self.elem_id("consensus_sigma"),
                )
            with gr.Row():
                novel_detail_gain = gr.Slider(
                    label="Cross-phase Novel Detail Gain",
                    minimum=0.0,
                    maximum=2.0,
                    step=0.05,
                    value=default_profile["novel_detail_gain"],
                    elem_id=self.elem_id("novel_detail_gain"),
                )
                novel_detail_max_delta = gr.Slider(
                    label="Novel Detail Maximum Delta",
                    minimum=1.0,
                    maximum=16.0,
                    step=0.5,
                    value=default_profile["novel_detail_max_delta"],
                    elem_id=self.elem_id("novel_detail_max_delta"),
                )
            with gr.Row():
                novel_detail_inner_radius = gr.Slider(
                    label="Novel Detail Inner Radius",
                    minimum=1,
                    maximum=4,
                    step=1,
                    value=default_profile["novel_detail_inner_radius"],
                    elem_id=self.elem_id("novel_detail_inner_radius"),
                )
                novel_detail_outer_radius = gr.Slider(
                    label="Novel Detail Outer Radius",
                    minimum=2,
                    maximum=12,
                    step=1,
                    value=default_profile["novel_detail_outer_radius"],
                    elem_id=self.elem_id("novel_detail_outer_radius"),
                )
                novel_detail_structure_sigma = gr.Slider(
                    label="Novel Detail Structure Sigma",
                    minimum=1.0,
                    maximum=24.0,
                    step=0.5,
                    value=default_profile["novel_detail_structure_sigma"],
                    elem_id=self.elem_id("novel_detail_structure_sigma"),
                )
            with gr.Row():
                novel_detail_consensus_sigma = gr.Slider(
                    label="Novel Detail Consensus Noise Floor",
                    minimum=0.1,
                    maximum=8.0,
                    step=0.1,
                    value=default_profile["novel_detail_consensus_sigma"],
                    elem_id=self.elem_id("novel_detail_consensus_sigma"),
                )
                novel_detail_consensus_strength = gr.Slider(
                    label="Novel Detail Consensus Strength",
                    minimum=0.1,
                    maximum=16.0,
                    step=0.1,
                    value=default_profile["novel_detail_consensus_strength"],
                    elem_id=self.elem_id("novel_detail_consensus_strength"),
                )

        with gr.Accordion(
            "仕上げ · Krea2 guidance / Smart Finish",
            open=False,
            elem_classes=["neo-workflow-accordion"],
        ):
            append_krea2_detail_prompt = gr.Checkbox(
                label="構図を守るKrea2ディテール指示を追加",
                value=True,
                elem_id=self.elem_id("append_krea2_detail_prompt"),
            )
            with gr.Row():
                smart_finish = gr.Checkbox(
                    label="Smart Finish（色ムラ + coherent detail）",
                    value=True,
                    elem_id=self.elem_id("smart_finish"),
                )
                detail_guard = gr.Checkbox(
                    label="ディテール保護",
                    value=True,
                    elem_id=self.elem_id("detail_guard"),
                )
                smart_color_strength = gr.Slider(
                    label="色補正強度",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.0,
                    elem_id=self.elem_id("smart_color_strength"),
                )
            with gr.Row():
                finish_detail_strength = gr.Slider(
                    label="Finish Detail Strength",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.75,
                    elem_id=self.elem_id("finish_detail_strength"),
                )
                finish_detail_radius = gr.Slider(
                    label="Finish Detail Radius",
                    minimum=0.5,
                    maximum=3.0,
                    step=0.1,
                    value=1.0,
                    elem_id=self.elem_id("finish_detail_radius"),
                )
                finish_detail_threshold = gr.Slider(
                    label="Finish Detail Threshold",
                    minimum=0.25,
                    maximum=4.0,
                    step=0.05,
                    value=0.6,
                    elem_id=self.elem_id("finish_detail_threshold"),
                )
                finish_max_detail_delta = gr.Slider(
                    label="Finish Maximum Detail Delta",
                    minimum=1.0,
                    maximum=12.0,
                    step=0.5,
                    value=5.0,
                    elem_id=self.elem_id("finish_max_detail_delta"),
                )

        profile_outputs = [
            phase_count,
            minimum_steps,
            maximum_steps,
            detail_knee,
            coarse_denoise,
            final_denoise,
            low_pass_radius,
            detail_gain,
            max_detail_delta,
            structure_sigma,
            base_detail_sigma,
            consensus_sigma,
            novel_detail_gain,
            novel_detail_max_delta,
            novel_detail_inner_radius,
            novel_detail_outer_radius,
            novel_detail_structure_sigma,
            novel_detail_consensus_sigma,
            novel_detail_consensus_strength,
            finish_detail_strength,
            finish_detail_radius,
            finish_detail_threshold,
            finish_max_detail_delta,
            append_krea2_detail_prompt,
            merge_mode,
        ]
        summary_inputs = [
            final_long_edge,
            final_width,
            final_height,
            quality_profile,
            vram_budget_gib,
            tile_size,
            phase_count,
            minimum_steps,
            maximum_steps,
            smart_finish,
            merge_mode,
        ]
        quick_4k.click(
            fn=lambda vram_budget, finish_enabled: self._quick_profile_values_with_summary(
                4096,
                "Krea2 Dense Detail 4K",
                vram_budget,
                finish_enabled,
            ),
            inputs=[vram_budget_gib, smart_finish],
            outputs=[
                final_long_edge,
                final_width,
                final_height,
                quality_profile,
                tile_size,
                *profile_outputs,
                workflow_status,
            ],
            show_progress="hidden",
        )
        quick_texture_4k.click(
            fn=lambda vram_budget, finish_enabled: self._quick_profile_values_with_summary(
                4096,
                "Krea2 Texture Rich 4K (Experimental)",
                vram_budget,
                finish_enabled,
            ),
            inputs=[vram_budget_gib, smart_finish],
            outputs=[
                final_long_edge,
                final_width,
                final_height,
                quality_profile,
                tile_size,
                *profile_outputs,
                workflow_status,
            ],
            show_progress="hidden",
        )
        quick_phaseweave_4k.click(
            fn=lambda vram_budget, finish_enabled: self._quick_profile_values_with_summary(
                4096,
                "Krea2 PhaseWeave 4K (Experimental)",
                vram_budget,
                finish_enabled,
            ),
            inputs=[vram_budget_gib, smart_finish],
            outputs=[
                final_long_edge,
                final_width,
                final_height,
                quality_profile,
                tile_size,
                *profile_outputs,
                workflow_status,
            ],
            show_progress="hidden",
        )
        quick_8k.click(
            fn=lambda vram_budget, finish_enabled: self._quick_profile_values_with_summary(
                8192,
                "Krea2 Dense Detail 8K",
                vram_budget,
                finish_enabled,
            ),
            inputs=[vram_budget_gib, smart_finish],
            outputs=[
                final_long_edge,
                final_width,
                final_height,
                quality_profile,
                tile_size,
                *profile_outputs,
                workflow_status,
            ],
            show_progress="hidden",
        )
        apply_quality_profile.click(
            fn=self._quality_profile_values_with_summary,
            inputs=[
                quality_profile,
                final_long_edge,
                final_width,
                final_height,
                vram_budget_gib,
                tile_size,
                smart_finish,
            ],
            outputs=[*profile_outputs, workflow_status],
            show_progress="hidden",
        )
        quality_profile.select(
            fn=self._quality_profile_values_with_summary,
            inputs=[
                quality_profile,
                final_long_edge,
                final_width,
                final_height,
                vram_budget_gib,
                tile_size,
                smart_finish,
            ],
            outputs=[*profile_outputs, workflow_status],
            show_progress="hidden",
        )

        for slider in (
            final_long_edge,
            final_width,
            final_height,
            vram_budget_gib,
            minimum_steps,
            maximum_steps,
        ):
            slider.input(
                fn=self._workflow_summary_html,
                inputs=summary_inputs,
                outputs=[workflow_status],
                show_progress="hidden",
            )
        for control in (tile_size, phase_count, smart_finish, merge_mode):
            control.select(
                fn=self._workflow_summary_html,
                inputs=summary_inputs,
                outputs=[workflow_status],
                show_progress="hidden",
            )

        return [
            final_long_edge,
            final_width,
            final_height,
            vram_budget_gib,
            tile_size,
            model_reserve_gib,
            max_stage_scale,
            phase_count,
            minimum_steps,
            maximum_steps,
            detail_knee,
            coarse_denoise,
            final_denoise,
            low_pass_radius,
            detail_gain,
            max_detail_delta,
            structure_sigma,
            base_detail_sigma,
            consensus_sigma,
            novel_detail_gain,
            novel_detail_max_delta,
            save_stages,
            append_krea2_detail_prompt,
            smart_finish,
            smart_color_strength,
            detail_guard,
            finish_detail_strength,
            finish_detail_radius,
            finish_detail_threshold,
            finish_max_detail_delta,
            novel_detail_inner_radius,
            novel_detail_outer_radius,
            novel_detail_structure_sigma,
            novel_detail_consensus_sigma,
            novel_detail_consensus_strength,
            merge_mode,
        ]

    @staticmethod
    def _workflow_summary_html(
        final_long_edge,
        final_width,
        final_height,
        quality_profile,
        vram_budget_gib,
        tile_size,
        phase_count,
        minimum_steps,
        maximum_steps,
        smart_finish,
        merge_mode,
    ) -> str:
        long_edge = int(float(final_long_edge or 0))
        width = int(float(final_width or 0))
        height = int(float(final_height or 0))
        budget = float(vram_budget_gib or 0)
        tile = int(float(tile_size or 0))
        phases = int(float(phase_count or 1))
        minimum = int(float(minimum_steps or 1))
        maximum = int(float(maximum_steps or minimum))
        profile_label = str(quality_profile or DEFAULT_QUALITY_PROFILE_LABEL)

        explicit_size_is_partial = (width > 0) != (height > 0)
        if width > 0 and height > 0:
            output_label = f"{width} × {height} px"
        else:
            output_label = f"長辺 {long_edge} px・入力比率を維持"

        status = "推奨設定"
        tone = "ready"
        note = (
            "通常img2img、Batch Count / Size = 1で実行します。"
            "CFG 1.0では重要な禁止条件もpositive promptへ含めてください。"
        )
        if "Experimental" in profile_label:
            status = "実験プロファイル"
            tone = "experimental"
            note = "通常の4K Smart出力と別に保存し、粒状感・細線・境界を比較してください。"
        if long_edge >= 8192:
            status = "4K承認後"
            tone = "caution"
            note = "入力には目視承認済みの4K画像を使います。native 8K生成ではありません。"
        if explicit_size_is_partial:
            status = "サイズ要確認"
            tone = "caution"
            note = "最終幅と最終高さは、両方を0にするか両方を1以上にしてください。"

        step_label = (
            f"{minimum} steps / tile"
            if minimum == maximum
            else f"{minimum}–{maximum} steps / tile"
        )
        return workflow_summary(
            profile_label,
            (
                ("納品", output_label),
                ("VRAM", "自動検出" if budget <= 0 else f"{budget:g} GiB"),
                ("Tile", "自動" if tile <= 0 else f"{tile} px"),
                ("処理", f"{phases} phase・{step_label}"),
                ("Merge", str(merge_mode)),
                ("仕上げ", "Smart Finish ON" if bool(smart_finish) else "補正なし"),
            ),
            status=status,
            note=note,
            tone=tone,
        )

    @staticmethod
    def _quality_profile_values(profile_label: str) -> tuple:
        try:
            profile_name = QUALITY_PROFILE_NAMES[profile_label]
        except KeyError as exc:
            raise ValueError(f"Unknown VRAM-Canvas quality profile: {profile_label}") from exc
        profile = krea2_vram_canvas_profile(profile_name)
        return (
            profile["phase_count"],
            profile["minimum_steps"],
            profile["maximum_steps"],
            profile["detail_knee"],
            profile["coarse_denoise"],
            profile["denoise"],
            profile["low_pass_radius"],
            profile["detail_gain"],
            profile["max_detail_delta"],
            profile["structure_sigma"],
            profile["base_detail_sigma"],
            profile["consensus_sigma"],
            profile["novel_detail_gain"],
            profile["novel_detail_max_delta"],
            profile["novel_detail_inner_radius"],
            profile["novel_detail_outer_radius"],
            profile["novel_detail_structure_sigma"],
            profile["novel_detail_consensus_sigma"],
            profile["novel_detail_consensus_strength"],
            profile["finish_detail_strength"],
            profile["finish_detail_radius"],
            profile["finish_detail_threshold"],
            profile["finish_max_detail_delta"],
            profile_name != "structure_safe",
            profile["merge_mode"],
        )

    @classmethod
    def _quality_profile_values_with_summary(
        cls,
        profile_label,
        final_long_edge,
        final_width,
        final_height,
        vram_budget_gib,
        tile_size,
        smart_finish,
    ) -> tuple:
        values = cls._quality_profile_values(str(profile_label))
        summary = cls._workflow_summary_html(
            final_long_edge,
            final_width,
            final_height,
            profile_label,
            vram_budget_gib,
            tile_size,
            values[0],
            values[1],
            values[2],
            smart_finish,
            values[-1],
        )
        return (*values, summary)

    @classmethod
    def _quick_profile_values(cls, long_edge: int, profile_label: str) -> tuple:
        if int(long_edge) not in (4096, 8192):
            raise ValueError("Smart target long edge must be 4096 or 8192.")
        return (
            int(long_edge),
            0,
            0,
            profile_label,
            (
                1024
                if int(long_edge) == 8192
                else 896
                if profile_label
                in (
                    "Krea2 Texture Rich 4K (Experimental)",
                    "Krea2 PhaseWeave 4K (Experimental)",
                )
                else 0
            ),
            *cls._quality_profile_values(profile_label),
        )

    @classmethod
    def _quick_profile_values_with_summary(
        cls,
        long_edge,
        profile_label,
        vram_budget_gib,
        smart_finish,
    ) -> tuple:
        values = cls._quick_profile_values(int(long_edge), str(profile_label))
        profile_values = values[5:]
        summary = cls._workflow_summary_html(
            values[0],
            values[1],
            values[2],
            values[3],
            vram_budget_gib,
            values[4],
            profile_values[0],
            profile_values[1],
            profile_values[2],
            smart_finish,
            profile_values[-1],
        )
        return (*values, summary)

    @staticmethod
    def _explicit_dimensions(width: int, height: int) -> tuple[int | None, int | None]:
        explicit_width = int(width) if int(width) > 0 else None
        explicit_height = int(height) if int(height) > 0 else None
        if (explicit_width is None) != (explicit_height is None):
            raise ValueError("Final Width and Final Height must both be 0 or both be > 0.")
        return explicit_width, explicit_height

    @staticmethod
    def _stage_denoise(stage_index: int, stage_count: int, coarse: float, final: float) -> float:
        if stage_count <= 1:
            return float(final)
        fraction = stage_index / (stage_count - 1)
        return float(coarse * (1.0 - fraction) + final * fraction)

    @staticmethod
    def _require_krea2_model(p) -> None:
        override_settings = getattr(p, "override_settings", None) or {}
        forbidden = sorted(
            key
            for key in ("sd_model_checkpoint", "sd_vae")
            if key in override_settings
        )
        if forbidden:
            raise ValueError(
                "Krea2 dense-detail mode does not allow per-run checkpoint/VAE "
                f"overrides ({', '.join(forbidden)}). Load Krea2 globally first."
            )
        model = getattr(p, "sd_model", None)
        model_config = getattr(model, "model_config", None)
        if type(model).__name__ != "Krea2" or type(model_config).__name__ != "Krea2":
            raise ValueError(
                "Krea2 dense-detail guidance requires a loaded Krea2 checkpoint with "
                "its Qwen Image VAE and Qwen3-VL text encoder."
            )

    @staticmethod
    def _close_memmap(value: np.memmap):
        value.flush()
        mapping = getattr(value, "_mmap", None)
        if mapping is not None:
            mapping.close()

    @classmethod
    def _finalize_stage(
        cls,
        base: Image.Image,
        accumulators: list[np.memmap],
        weight_sums: list[np.memmap],
        energy_sums: list[np.memmap],
        novel_accumulators: list[np.memmap],
        novel_energy_sums: list[np.memmap],
        work_dir: Path,
        stage_number: int,
        consensus_sigma: float,
        novel_consensus_sigma: float = DEFAULT_NOVEL_DETAIL_CONSENSUS_SIGMA,
        novel_consensus_strength: float = DEFAULT_NOVEL_DETAIL_CONSENSUS_STRENGTH,
        merge_mode: str = CONSENSUS_MERGE_MODE,
    ) -> tuple[Image.Image, dict[str, float | bool]]:
        width, height = base.size
        expected_sets = 2 if merge_mode == PHASE_WEAVE_MERGE_MODE else 1
        if not (
            len(accumulators)
            == len(weight_sums)
            == len(energy_sums)
            == expected_sets
        ):
            raise ValueError("VRAM-Canvas stage moment set count does not match merge mode.")
        if novel_accumulators and not (
            len(novel_accumulators)
            == len(novel_energy_sums)
            == expected_sets
        ):
            raise ValueError("VRAM-Canvas novel moment set count does not match merge mode.")
        result_path = work_dir / f"stage_{stage_number:02d}_result.uint8"
        preview_path = work_dir / f"stage_{stage_number:02d}.png"
        result = np.memmap(result_path, dtype=np.uint8, mode="w+", shape=(height, width, 3))
        covered_pixels = 0
        confidence_total = 0.0
        disagreement_total = 0.0
        phase0_pixels = 0
        phase1_pixels = 0
        input_rejected_pixels = 0
        both_unfaithful_pixels = 0
        uncertain_fused_pixels = 0
        boundary_pixels = 0
        confidence_gain_total = 0.0
        selected_fidelity_total = 0.0
        input_mix_total = 0.0
        support_weight_total = 0.0
        low_frequency_luma_gain_total = 0.0
        novel_evidence_pixels = 0
        novel_confidence_total = 0.0
        novel_abs_total = 0.0
        for y0 in range(0, height, 128):
            y1 = min(height, y0 + 128)
            base_stripe = np.asarray(base.crop((0, y0, width, y1)), dtype=np.float32)
            if merge_mode == PHASE_WEAVE_MERGE_MODE:
                padding = PHASE_WEAVE_CONTEXT_RADIUS
                read_y0 = max(0, y0 - padding)
                read_y1 = min(height, y1 + padding)
                base_read = np.asarray(
                    base.crop((0, read_y0, width, read_y1)),
                    dtype=np.float32,
                )
                normalized_read, diagnostics_read = phase_weave_residual(
                    np.asarray(accumulators[0][read_y0:read_y1], dtype=np.float32),
                    np.asarray(weight_sums[0][read_y0:read_y1], dtype=np.float32),
                    np.asarray(energy_sums[0][read_y0:read_y1], dtype=np.float32),
                    np.asarray(accumulators[1][read_y0:read_y1], dtype=np.float32),
                    np.asarray(weight_sums[1][read_y0:read_y1], dtype=np.float32),
                    np.asarray(energy_sums[1][read_y0:read_y1], dtype=np.float32),
                    base_rgb=base_read,
                    sigma=float(consensus_sigma),
                )
                local_y0 = y0 - read_y0
                local_y1 = local_y0 + (y1 - y0)
                normalized = normalized_read[local_y0:local_y1]
                diagnostics = {
                    key: value[local_y0:local_y1]
                    for key, value in diagnostics_read.items()
                }
                covered = diagnostics["covered"]
                confidence = diagnostics["support_confidence"]
                disagreement = diagnostics["cross_disagreement"]
                phase0_pixels += int(
                    np.count_nonzero(diagnostics["selected_phase"] == 0)
                )
                phase1_pixels += int(
                    np.count_nonzero(diagnostics["selected_phase"] == 1)
                )
                input_rejected_pixels += int(
                    np.count_nonzero(diagnostics["input_rejected"])
                )
                both_unfaithful_pixels += int(
                    np.count_nonzero(diagnostics["both_unfaithful"])
                )
                uncertain_fused_pixels += int(
                    np.count_nonzero(diagnostics["uncertain_fused"])
                )
                boundary_pixels += int(np.count_nonzero(diagnostics["boundary"]))
                confidence_gain_total += float(
                    np.sum(diagnostics["confidence_gain"][covered], dtype=np.float64)
                )
                selected_fidelity_total += float(
                    np.sum(
                        diagnostics["selected_fidelity"][covered],
                        dtype=np.float64,
                    )
                )
                input_mix_total += float(
                    np.sum(diagnostics["input_mix"][covered], dtype=np.float64)
                )
                support_weight_total += float(
                    np.sum(diagnostics["support_weight"][covered], dtype=np.float64)
                )
                low_frequency_luma_gain_total += float(
                    np.sum(
                        diagnostics["low_frequency_luma_gain"][covered],
                        dtype=np.float64,
                    )
                )
            else:
                weights = np.asarray(weight_sums[0][y0:y1], dtype=np.float32)
                normalized, confidence, disagreement = consensus_gated_residual(
                    np.asarray(accumulators[0][y0:y1], dtype=np.float32),
                    weights,
                    np.asarray(energy_sums[0][y0:y1], dtype=np.float32),
                    sigma=consensus_sigma,
                )
                covered = weights > 1e-8
            covered_pixels += int(np.count_nonzero(covered))
            confidence_total += float(np.sum(confidence[covered], dtype=np.float64))
            disagreement_total += float(
                np.sum(disagreement[covered], dtype=np.float64)
            )
            novel_normalized = np.zeros_like(normalized)
            if novel_accumulators:
                if merge_mode == PHASE_WEAVE_MERGE_MODE:
                    novel_read, novel_diagnostics_read = phase_weave_residual(
                        np.asarray(
                            novel_accumulators[0][read_y0:read_y1], dtype=np.float32
                        ),
                        np.asarray(
                            weight_sums[0][read_y0:read_y1], dtype=np.float32
                        ),
                        np.asarray(
                            novel_energy_sums[0][read_y0:read_y1], dtype=np.float32
                        ),
                        np.asarray(
                            novel_accumulators[1][read_y0:read_y1], dtype=np.float32
                        ),
                        np.asarray(
                            weight_sums[1][read_y0:read_y1], dtype=np.float32
                        ),
                        np.asarray(
                            novel_energy_sums[1][read_y0:read_y1], dtype=np.float32
                        ),
                        base_rgb=base_read,
                        sigma=float(novel_consensus_sigma),
                        strength=float(novel_consensus_strength),
                    )
                    novel_normalized = novel_read[local_y0:local_y1]
                    novel_diagnostics = {
                        key: value[local_y0:local_y1]
                        for key, value in novel_diagnostics_read.items()
                    }
                    independent_evidence = novel_diagnostics["both_covered"]
                    novel_confidence = novel_diagnostics["support_confidence"]
                else:
                    novel_normalized, novel_confidence, _ = consensus_gated_residual(
                        np.asarray(novel_accumulators[0][y0:y1], dtype=np.float32),
                        weights,
                        np.asarray(novel_energy_sums[0][y0:y1], dtype=np.float32),
                        sigma=float(novel_consensus_sigma),
                        strength=float(novel_consensus_strength),
                    )
                    independent_evidence = weights >= 1.5
                novel_normalized[~independent_evidence] = 0.0
                novel_evidence_pixels += int(np.count_nonzero(independent_evidence))
                novel_confidence_total += float(
                    np.sum(novel_confidence[independent_evidence], dtype=np.float64)
                )
                novel_abs_total += float(
                    np.sum(np.abs(novel_normalized), dtype=np.float64) / 3.0
                )
            result[y0:y1] = np.clip(
                np.rint(base_stripe + normalized + novel_normalized), 0, 255
            ).astype(np.uint8)
        result.flush()
        preview = Image.fromarray(np.asarray(result), mode="RGB")
        preview.save(preview_path, format="PNG")
        preview.close()
        cls._close_memmap(result)
        with Image.open(preview_path) as saved:
            image = saved.copy()
        covered_divisor = covered_pixels or 1
        novel_divisor = novel_evidence_pixels or 1
        return image, {
            "merge_mode": merge_mode,
            "mean_consensus_gate": confidence_total / covered_divisor,
            "mean_consensus_disagreement": disagreement_total / covered_divisor,
            "phaseweave_enabled": merge_mode == PHASE_WEAVE_MERGE_MODE,
            "phaseweave_phase0_selected_percent": (
                phase0_pixels * 100.0 / covered_divisor
            ),
            "phaseweave_phase1_selected_percent": phase1_pixels * 100.0 / covered_divisor,
            "phaseweave_input_rejected_percent": (
                input_rejected_pixels * 100.0 / covered_divisor
            ),
            "phaseweave_both_unfaithful_percent": (
                both_unfaithful_pixels * 100.0 / covered_divisor
            ),
            "phaseweave_uncertain_fused_percent": (
                uncertain_fused_pixels * 100.0 / covered_divisor
            ),
            "phaseweave_boundary_percent": boundary_pixels * 100.0 / covered_divisor,
            "phaseweave_mean_detail_gain": (
                confidence_gain_total / covered_divisor
                if merge_mode == PHASE_WEAVE_MERGE_MODE
                else 0.0
            ),
            "phaseweave_mean_selected_fidelity": (
                selected_fidelity_total / covered_divisor
                if merge_mode == PHASE_WEAVE_MERGE_MODE
                else 0.0
            ),
            "phaseweave_mean_input_mix": (
                input_mix_total / covered_divisor
                if merge_mode == PHASE_WEAVE_MERGE_MODE
                else 0.0
            ),
            "phaseweave_mean_support_weight": (
                support_weight_total / covered_divisor
                if merge_mode == PHASE_WEAVE_MERGE_MODE
                else 0.0
            ),
            "phaseweave_mean_low_frequency_luma_gain": (
                low_frequency_luma_gain_total / covered_divisor
                if merge_mode == PHASE_WEAVE_MERGE_MODE
                else 0.0
            ),
            "novel_detail_enabled": bool(novel_accumulators),
            "novel_evidence_percent": novel_evidence_pixels * 100.0 / (width * height),
            "mean_novel_consensus_gate": novel_confidence_total / novel_divisor,
            "mean_abs_novel_residual": novel_abs_total / novel_divisor,
        }

    @staticmethod
    def _validate_run(
        p,
        *,
        max_stage_scale: float,
        phase_count: int,
        minimum_steps: int,
        maximum_steps: int,
        detail_knee: float,
        coarse_denoise: float,
        final_denoise: float,
        low_pass_radius: int,
        detail_gain: float,
        max_detail_delta: float,
        structure_sigma: float,
        base_detail_sigma: float,
        consensus_sigma: float,
        novel_detail_gain: float,
        novel_detail_max_delta: float,
        smart_finish: bool = True,
        smart_color_strength: float = 0.0,
        detail_guard: bool = True,
        finish_detail_strength: float = 0.75,
        finish_detail_radius: float = 1.0,
        finish_detail_threshold: float = 0.6,
        finish_max_detail_delta: float = 5.0,
        novel_detail_inner_radius: int = DEFAULT_NOVEL_DETAIL_INNER_RADIUS,
        novel_detail_outer_radius: int = DEFAULT_NOVEL_DETAIL_OUTER_RADIUS,
        novel_detail_structure_sigma: float = DEFAULT_NOVEL_DETAIL_STRUCTURE_SIGMA,
        novel_detail_consensus_sigma: float = DEFAULT_NOVEL_DETAIL_CONSENSUS_SIGMA,
        novel_detail_consensus_strength: float = DEFAULT_NOVEL_DETAIL_CONSENSUS_STRENGTH,
        merge_mode: str = CONSENSUS_MERGE_MODE,
    ):
        if p.batch_size != 1 or p.n_iter != 1:
            raise ValueError("VRAM-Canvas supports Batch Count 1 and Batch Size 1 only.")
        if not p.init_images or p.init_images[0] is None:
            raise ValueError("VRAM-Canvas requires an img2img input image.")
        if p.image_mask is not None or getattr(p, "latent_mask", None) is not None:
            raise ValueError("VRAM-Canvas supports normal img2img only; inpaint masks are not supported.")
        if max_stage_scale <= 1:
            raise ValueError("Maximum Scale per Stage must be > 1.")
        if int(phase_count) not in (1, 2):
            raise ValueError("Grid Phases must be 1 or 2.")
        if str(merge_mode) not in (
            CONSENSUS_MERGE_MODE,
            PHASE_WEAVE_MERGE_MODE,
        ):
            raise ValueError(f"Unknown VRAM-Canvas merge mode: {merge_mode}")
        if (
            str(merge_mode) == PHASE_WEAVE_MERGE_MODE
            and int(phase_count) != 2
        ):
            raise ValueError("PhaseWeave requires Grid Phases 2.")
        if int(minimum_steps) <= 0 or int(maximum_steps) < int(minimum_steps):
            raise ValueError("Steps must satisfy 0 < Minimum Steps <= Maximum Steps.")
        if detail_knee <= 0 or low_pass_radius <= 0 or detail_gain <= 0 or max_detail_delta <= 0 or structure_sigma <= 0:
            raise ValueError("Advanced frequency merge values must be > 0.")
        if not np.isfinite(float(base_detail_sigma)) or float(base_detail_sigma) < 0:
            raise ValueError("Base Detail Protection must be finite and >= 0.")
        if not np.isfinite(float(consensus_sigma)) or float(consensus_sigma) < 0:
            raise ValueError("Consensus Noise Floor must be finite and >= 0.")
        if not np.isfinite(float(novel_detail_gain)) or float(novel_detail_gain) < 0:
            raise ValueError("Novel Detail Gain must be finite and >= 0.")
        if not np.isfinite(float(novel_detail_max_delta)) or float(novel_detail_max_delta) <= 0:
            raise ValueError("Novel Detail Maximum Delta must be finite and > 0.")
        if float(novel_detail_gain) > 0 and int(phase_count) < 2:
            raise ValueError("Novel detail requires Grid Phases 2 for independent evidence.")
        if float(novel_detail_gain) > 0 and float(base_detail_sigma) <= 0:
            raise ValueError("Novel detail requires Base Detail Protection > 0.")
        if int(novel_detail_inner_radius) <= 0 or int(novel_detail_outer_radius) <= 0:
            raise ValueError("Novel Detail radii must be > 0.")
        if int(novel_detail_inner_radius) >= int(novel_detail_outer_radius):
            raise ValueError("Novel Detail Inner Radius must be smaller than Outer Radius.")
        for value, name in (
            (novel_detail_structure_sigma, "Novel Detail Structure Sigma"),
            (novel_detail_consensus_sigma, "Novel Detail Consensus Noise Floor"),
            (novel_detail_consensus_strength, "Novel Detail Consensus Strength"),
        ):
            if not np.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and > 0.")
        if not 0 <= coarse_denoise <= 1 or not 0 <= final_denoise <= 1:
            raise ValueError("Denoising strengths must be between 0 and 1.")
        if smart_finish:
            if not np.isfinite(float(smart_color_strength)) or not 0 <= float(smart_color_strength) <= 1:
                raise ValueError("Smart Chroma Strength must be finite and between 0 and 1.")
            if not np.isfinite(float(finish_detail_strength)) or not 0 <= float(finish_detail_strength) <= 1:
                raise ValueError("Finish Detail Strength must be finite and between 0 and 1.")
            for value, name in (
                (finish_detail_radius, "Finish Detail Radius"),
                (finish_detail_threshold, "Finish Detail Threshold"),
                (finish_max_detail_delta, "Finish Maximum Detail Delta"),
            ):
                if not np.isfinite(float(value)) or float(value) <= 0:
                    raise ValueError(f"{name} must be finite and greater than 0.")

    def run(
        self,
        p,
        final_long_edge: int,
        final_width: int,
        final_height: int,
        vram_budget_gib: float,
        tile_size: int,
        model_reserve_gib: float,
        max_stage_scale: float,
        phase_count: int,
        minimum_steps: int,
        maximum_steps: int,
        detail_knee: float,
        coarse_denoise: float,
        final_denoise: float,
        low_pass_radius: int,
        detail_gain: float,
        max_detail_delta: float,
        structure_sigma: float,
        base_detail_sigma: float,
        consensus_sigma: float,
        novel_detail_gain: float,
        novel_detail_max_delta: float,
        save_stages: bool,
        append_krea2_detail_prompt: bool = False,
        smart_finish: bool = True,
        smart_color_strength: float = 0.0,
        detail_guard: bool = True,
        finish_detail_strength: float = 0.75,
        finish_detail_radius: float = 1.0,
        finish_detail_threshold: float = 0.6,
        finish_max_detail_delta: float = 5.0,
        novel_detail_inner_radius: int = DEFAULT_NOVEL_DETAIL_INNER_RADIUS,
        novel_detail_outer_radius: int = DEFAULT_NOVEL_DETAIL_OUTER_RADIUS,
        novel_detail_structure_sigma: float = DEFAULT_NOVEL_DETAIL_STRUCTURE_SIGMA,
        novel_detail_consensus_sigma: float = DEFAULT_NOVEL_DETAIL_CONSENSUS_SIGMA,
        novel_detail_consensus_strength: float = DEFAULT_NOVEL_DETAIL_CONSENSUS_STRENGTH,
        merge_mode: str = CONSENSUS_MERGE_MODE,
    ):
        self._validate_run(
            p,
            max_stage_scale=float(max_stage_scale),
            phase_count=int(phase_count),
            minimum_steps=int(minimum_steps),
            maximum_steps=int(maximum_steps),
            detail_knee=float(detail_knee),
            coarse_denoise=float(coarse_denoise),
            final_denoise=float(final_denoise),
            low_pass_radius=int(low_pass_radius),
            detail_gain=float(detail_gain),
            max_detail_delta=float(max_detail_delta),
            structure_sigma=float(structure_sigma),
            base_detail_sigma=float(base_detail_sigma),
            consensus_sigma=float(consensus_sigma),
            novel_detail_gain=float(novel_detail_gain),
            novel_detail_max_delta=float(novel_detail_max_delta),
            smart_finish=bool(smart_finish),
            smart_color_strength=float(smart_color_strength),
            detail_guard=bool(detail_guard),
            finish_detail_strength=float(finish_detail_strength),
            finish_detail_radius=float(finish_detail_radius),
            finish_detail_threshold=float(finish_detail_threshold),
            finish_max_detail_delta=float(finish_max_detail_delta),
            novel_detail_inner_radius=int(novel_detail_inner_radius),
            novel_detail_outer_radius=int(novel_detail_outer_radius),
            novel_detail_structure_sigma=float(novel_detail_structure_sigma),
            novel_detail_consensus_sigma=float(novel_detail_consensus_sigma),
            novel_detail_consensus_strength=float(novel_detail_consensus_strength),
            merge_mode=str(merge_mode),
        )
        if append_krea2_detail_prompt:
            self._require_krea2_model(p)
        global_seed = int(processing.get_fixed_seed(p.seed))
        global_subseed = int(processing.get_fixed_seed(p.subseed))
        original_prompt = p.prompt
        if append_krea2_detail_prompt:
            if not isinstance(original_prompt, str):
                raise ValueError("Krea2 dense-detail guidance requires one text prompt.")
            effective_prompt = krea2_detail_prompt(original_prompt)
        else:
            effective_prompt = original_prompt
        source = images.flatten(p.init_images[0], opts.img2img_background_color)
        explicit_width, explicit_height = self._explicit_dimensions(final_width, final_height)
        if (
            int(final_long_edge) == 8192
            and explicit_width is None
            and 3840 <= max(source.size) <= 4096
        ):
            target_w, target_h = source.width * 2, source.height * 2
        else:
            target_w, target_h = target_size(
                source.width,
                source.height,
                int(final_long_edge),
                explicit_width,
                explicit_height,
            )
        if max(target_w, target_h) > 4096:
            if not 3840 <= max(source.size) <= 4096:
                raise ValueError(
                    "8K mode requires a visually approved 4K input with long edge "
                    "3840-4096; run and inspect 4K first."
                )
            if (target_w, target_h) != (source.width * 2, source.height * 2):
                raise ValueError(
                    "8K mode must be the exact 2x dimensions of the approved 4K input."
                )
        if target_w * target_h > MAX_OUTPUT_PIXELS:
            raise ValueError(f"VRAM-Canvas target {target_w}x{target_h} exceeds {MAX_OUTPUT_PIXELS:,} pixels.")

        detected_vram_gib = memory_management.get_total_memory(devices.device) / GIB
        budget_gib = float(vram_budget_gib) if float(vram_budget_gib) > 0 else detected_vram_gib
        resolved_tile_size = resolve_tile_size(
            budget_gib,
            requested_tile_size=int(tile_size),
            model_reserve_gib=float(model_reserve_gib),
        )
        halo = resolve_halo(resolved_tile_size)
        core_size = resolved_tile_size - 2 * halo
        core_overlap = resolve_core_overlap(core_size, halo)
        stages = progressive_stage_sizes(
            source.width,
            source.height,
            target_w,
            target_h,
            max_stage_scale=float(max_stage_scale),
        )
        stage_plans = [
            plan_tiles(
                width,
                height,
                tile_size=resolved_tile_size,
                halo=halo,
                core_overlap=core_overlap,
                phase_count=int(phase_count),
                virtual_padding=str(merge_mode) == PHASE_WEAVE_MERGE_MODE,
            )
            for width, height in stages
        ]
        total_tiles = sum(len(plans) for plans in stage_plans)

        original_batch_size = p.batch_size
        original_n_iter = p.n_iter
        original_save_samples = p.save_samples()
        original_do_not_save_samples = p.do_not_save_samples
        original_do_not_save_grid = p.do_not_save_grid
        original_resize_mode = p.resize_mode
        original_width = p.width
        original_height = p.height
        original_denoising_strength = p.denoising_strength
        original_init_images = p.init_images
        original_seed = p.seed
        original_subseed = p.subseed
        original_steps = p.steps
        original_all_seeds = p.all_seeds
        original_all_subseeds = p.all_subseeds
        had_all_prompts = hasattr(p, "all_prompts")
        original_all_prompts = getattr(p, "all_prompts", None)
        had_all_negative_prompts = hasattr(p, "all_negative_prompts")
        original_all_negative_prompts = getattr(p, "all_negative_prompts", None)
        had_main_prompt = hasattr(p, "main_prompt")
        original_main_prompt = getattr(p, "main_prompt", None)
        had_prompts = hasattr(p, "prompts")
        original_prompts = getattr(p, "prompts", None)
        had_restore_faces = hasattr(p, "restore_faces")
        original_restore_faces = getattr(p, "restore_faces", None)
        had_tiling = hasattr(p, "tiling")
        original_tiling = getattr(p, "tiling", None)
        processing_snapshot = ProcessingSnapshot(p)

        current = source
        last_processed = None
        last_diffusion_size = source.size
        temp_root = Path(opts.temp_dir) if getattr(opts, "temp_dir", "") else None
        if temp_root is not None:
            temp_root.mkdir(parents=True, exist_ok=True)

        gui_manifest = {
            "format_version": 5,
            "algorithm": (
                KREA2_PHASEWEAVE_PRODUCT_NAME
                if str(merge_mode) == PHASE_WEAVE_MERGE_MODE
                else "VRAM-Canvas GUI"
            ),
            "source_size": [source.width, source.height],
            "target_size": [target_w, target_h],
            "stages": [list(size) for size in stages],
            "global_seed": global_seed,
            "base_prompt": original_prompt,
            "effective_prompt": effective_prompt,
            "vram_budget_gib": budget_gib,
            "tile_size": resolved_tile_size,
            "halo": halo,
            "core_overlap": core_overlap,
            "phase_count": int(phase_count),
            "merge_mode": str(merge_mode),
            "grid_layout": (
                "uniform_virtual_edge_balanced"
                if str(merge_mode) == PHASE_WEAVE_MERGE_MODE
                else "legacy_edge_anchored"
            ),
            "grid_stride": core_size - core_overlap,
            "grid_phase_offset": (
                (core_size - core_overlap) // int(phase_count)
                if int(phase_count) > 1
                else 0
            ),
            "grid_padding_mode": (
                "edge" if str(merge_mode) == PHASE_WEAVE_MERGE_MODE else "none"
            ),
            "grid_origin": (
                [
                    balanced_virtual_axis_origin(
                        target_w,
                        core_size,
                        core_overlap,
                        phase_count=int(phase_count),
                    ),
                    balanced_virtual_axis_origin(
                        target_h,
                        core_size,
                        core_overlap,
                        phase_count=int(phase_count),
                    ),
                ]
                if str(merge_mode) == PHASE_WEAVE_MERGE_MODE
                else [0, 0]
            ),
            "exact_img2img_steps": EXACT_IMG2IMG_STEPS,
            "exact_img2img_steps_scope": EXACT_IMG2IMG_STEPS_SCOPE,
            "phaseweave": {
                "enabled": str(merge_mode) == PHASE_WEAVE_MERGE_MODE,
                "product_name": KREA2_PHASEWEAVE_PRODUCT_NAME,
                "profile_key": KREA2_PHASEWEAVE_PROFILE_KEY,
                "grid_layout": (
                    "uniform_virtual_edge_balanced"
                    if str(merge_mode) == PHASE_WEAVE_MERGE_MODE
                    else "legacy_edge_anchored"
                ),
                **phase_weave_configuration(),
            },
            "settings": {
                "minimum_steps": int(minimum_steps),
                "maximum_steps": int(maximum_steps),
                "detail_knee": float(detail_knee),
                "coarse_denoise": float(coarse_denoise),
                "final_denoise": float(final_denoise),
                "low_pass_radius": int(low_pass_radius),
                "detail_gain": float(detail_gain),
                "max_detail_delta": float(max_detail_delta),
                "structure_sigma": float(structure_sigma),
                "base_detail_sigma": float(base_detail_sigma),
                "consensus_sigma": float(consensus_sigma),
                "novel_detail_gain": float(novel_detail_gain),
                "novel_detail_max_delta": float(novel_detail_max_delta),
                "novel_detail_inner_radius": int(novel_detail_inner_radius),
                "novel_detail_outer_radius": int(novel_detail_outer_radius),
                "novel_detail_structure_sigma": float(novel_detail_structure_sigma),
                "novel_detail_consensus_sigma": float(novel_detail_consensus_sigma),
                "novel_detail_consensus_strength": float(novel_detail_consensus_strength),
                "smart_finish": bool(smart_finish),
                "detail_guard": bool(detail_guard),
            },
            "stage_reports": [],
        }
        smart_finish_report = None

        try:
            p.extra_generation_params.update(
                {
                    "VRAM-Canvas": (
                        KREA2_PHASEWEAVE_PRODUCT_NAME
                        if str(merge_mode) == PHASE_WEAVE_MERGE_MODE
                        else "progressive base-detail-structure-consensus residual"
                    ),
                    "VRAM-Canvas target": f"{target_w}x{target_h}",
                    "VRAM-Canvas stages": " -> ".join(
                        f"{w}x{h}" for w, h in stages
                    ),
                    "VRAM-Canvas global seed": global_seed,
                    "VRAM-Canvas VRAM budget GiB": round(budget_gib, 2),
                    "VRAM-Canvas tile": resolved_tile_size,
                    "VRAM-Canvas halo": halo,
                    "VRAM-Canvas core overlap": core_overlap,
                    "VRAM-Canvas phases": int(phase_count),
                    "VRAM-Canvas merge mode": str(merge_mode),
                    "VRAM-Canvas grid layout": gui_manifest["grid_layout"],
                    "VRAM-Canvas exact img2img steps": EXACT_IMG2IMG_STEPS,
                    "VRAM-Canvas exact steps scope": EXACT_IMG2IMG_STEPS_SCOPE,
                    "VRAM-Canvas steps": (
                        f"{int(minimum_steps)}-{int(maximum_steps)} adaptive"
                    ),
                    "VRAM-Canvas denoise": (
                        f"{float(coarse_denoise):.2f}->{float(final_denoise):.2f}"
                    ),
                    "VRAM-Canvas low-pass radius": int(low_pass_radius),
                    "VRAM-Canvas max detail delta": float(max_detail_delta),
                    "VRAM-Canvas base detail protection": float(base_detail_sigma),
                    "VRAM-Canvas consensus floor": float(consensus_sigma),
                    "VRAM-Canvas novel detail gain": float(novel_detail_gain),
                    "VRAM-Canvas novel detail max delta": float(
                        novel_detail_max_delta
                    ),
                    "VRAM-Canvas novel detail band": (
                        f"{int(novel_detail_inner_radius)}-"
                        f"{int(novel_detail_outer_radius)} px"
                    ),
                    "VRAM-Canvas novel detail structure sigma": float(
                        novel_detail_structure_sigma
                    ),
                    "VRAM-Canvas novel consensus floor": float(
                        novel_detail_consensus_sigma
                    ),
                    "VRAM-Canvas novel consensus strength": float(
                        novel_detail_consensus_strength
                    ),
                    "VRAM-Canvas Krea2 detail prompt": bool(
                        append_krea2_detail_prompt
                    ),
                    "VRAM-Canvas Smart Finish": bool(smart_finish),
                    "VRAM-Canvas coherent detail guard": bool(detail_guard),
                    "VRAM-Canvas finish detail strength": float(
                        finish_detail_strength
                    ),
                }
            )
            p.batch_size = 1
            p.n_iter = 1
            p.do_not_save_samples = True
            p.do_not_save_grid = True
            p.resize_mode = 0
            p.prompt = effective_prompt
            p.seed = global_seed
            p.subseed = global_subseed
            if had_all_prompts:
                p.all_prompts = [
                    effective_prompt for _ in (original_all_prompts or [None])
                ]
            if had_prompts:
                p.prompts = [
                    effective_prompt for _ in (original_prompts or [None])
                ]
            if had_main_prompt:
                p.main_prompt = effective_prompt
            p.restore_faces = False
            p.tiling = False
            state.job_count = total_tiles + len(stages) + (1 if smart_finish else 0) + 1

            with tempfile.TemporaryDirectory(prefix="vram_canvas_", dir=temp_root) as temporary:
                work_dir = Path(temporary)
                work_bytes_per_pixel = vram_canvas_work_bytes_per_pixel(
                    phase_count=int(phase_count),
                    merge_mode=str(merge_mode),
                    novel_detail=float(novel_detail_gain) > 0,
                )
                required_work_bytes = sum(
                    width * height * work_bytes_per_pixel
                    for width, height in stages
                )
                free_disk_bytes = shutil.disk_usage(work_dir).free
                if free_disk_bytes < required_work_bytes:
                    raise RuntimeError(f"VRAM-Canvas needs about {required_work_bytes / GIB:.2f} GiB of temporary disk space; " f"only {free_disk_bytes / GIB:.2f} GiB is free.")

                completed_tiles = 0
                for stage_index, ((stage_w, stage_h), plans) in enumerate(zip(stages, stage_plans)):
                    stage_number = stage_index + 1
                    tile_records = []
                    base = current.resize((stage_w, stage_h), Image.Resampling.LANCZOS)
                    phase_normalizers = phase_weight_normalizers(plans, stage_w, stage_h)
                    denoise = self._stage_denoise(
                        stage_index,
                        len(stages),
                        float(coarse_denoise),
                        float(final_denoise),
                    )
                    moment_set_count = (
                        2 if str(merge_mode) == PHASE_WEAVE_MERGE_MODE else 1
                    )
                    accumulators: list[np.memmap] = []
                    weight_sums: list[np.memmap] = []
                    energy_sums: list[np.memmap] = []
                    novel_accumulators: list[np.memmap] = []
                    novel_energy_sums: list[np.memmap] = []
                    for phase_slot in range(moment_set_count):
                        suffix = (
                            f"_phase{phase_slot}"
                            if moment_set_count > 1
                            else ""
                        )
                        accumulators.append(
                            np.memmap(
                                work_dir
                                / f"stage_{stage_number:02d}{suffix}_delta.float32",
                                dtype=np.float32,
                                mode="w+",
                                shape=(stage_h, stage_w, 3),
                            )
                        )
                        weight_sums.append(
                            np.memmap(
                                work_dir
                                / f"stage_{stage_number:02d}{suffix}_weight.float32",
                                dtype=np.float32,
                                mode="w+",
                                shape=(stage_h, stage_w),
                            )
                        )
                        energy_sums.append(
                            np.memmap(
                                work_dir
                                / f"stage_{stage_number:02d}{suffix}_energy.float32",
                                dtype=np.float32,
                                mode="w+",
                                shape=(stage_h, stage_w),
                            )
                        )
                        if float(novel_detail_gain) > 0:
                            novel_accumulators.append(
                                np.memmap(
                                    work_dir
                                    / f"stage_{stage_number:02d}{suffix}_novel_delta.float32",
                                    dtype=np.float32,
                                    mode="w+",
                                    shape=(stage_h, stage_w, 3),
                                )
                            )
                            novel_energy_sums.append(
                                np.memmap(
                                    work_dir
                                    / f"stage_{stage_number:02d}{suffix}_novel_energy.float32",
                                    dtype=np.float32,
                                    mode="w+",
                                    shape=(stage_h, stage_w),
                                )
                            )
                    for value in (
                        accumulators
                        + weight_sums
                        + energy_sums
                        + novel_accumulators
                        + novel_energy_sums
                    ):
                        value[:] = 0

                    try:
                        for stage_tile_index, tile in enumerate(plans, start=1):
                            completed_tiles += 1
                            context = extract_tile_context(base, tile)
                            local_core = context.crop(tile.local_core_box)
                            score = detail_score(np.asarray(local_core, dtype=np.uint8))
                            steps = adaptive_step_count(
                                score,
                                int(minimum_steps),
                                int(maximum_steps),
                                knee=float(detail_knee),
                            )
                            seed = coordinate_seed(
                                global_seed,
                                tile.phase + stage_number * int(phase_count),
                                tile.grid_core_x0,
                                tile.grid_core_y0,
                            )
                            tile_records.append(
                                {
                                    "phase": tile.phase,
                                    "grid_core": [
                                        tile.grid_core_x0,
                                        tile.grid_core_y0,
                                        tile.grid_core_width,
                                        tile.grid_core_height,
                                    ],
                                    "core": [
                                        tile.core_x0,
                                        tile.core_y0,
                                        tile.core_width,
                                        tile.core_height,
                                    ],
                                    "context_padding": list(tile.context_padding),
                                    "seed": seed,
                                    "steps": steps,
                                    "detail_score": score,
                                }
                            )
                            diffusion_tile = self._pad_tile(context)
                            last_diffusion_size = diffusion_tile.size
                            p.width = diffusion_tile.width
                            p.height = diffusion_tile.height
                            p.steps = steps
                            p.denoising_strength = denoise
                            p.seed = seed
                            p.init_images = [diffusion_tile]
                            state.job = f"VRAM-Canvas Stage {stage_number}/{len(stages)} " f"Tile {stage_tile_index}/{len(plans)}"
                            state.textinfo = f"{stage_w}x{stage_h} | tile {completed_tiles}/{total_tiles} | " f"payload {diffusion_tile.width}x{diffusion_tile.height} | steps {steps}"

                            with internal_exact_img2img_steps(p):
                                last_processed = processing.process_images(p)
                            if last_processed is None or not last_processed.images:
                                raise RuntimeError(f"VRAM-Canvas stage {stage_number} tile {stage_tile_index} returned no image.")
                            if state.interrupted or state.skipped or state.stopping_generation:
                                raise RuntimeError(
                                    "VRAM-Canvas was interrupted before a complete image "
                                    "was produced; the unfinished tile was not returned."
                                )
                            refined = images.flatten(last_processed.images[0], opts.img2img_background_color)
                            if refined.size != diffusion_tile.size:
                                raise RuntimeError(f"VRAM-Canvas tile returned {refined.size}; expected {diffusion_tile.size}.")
                            refined = refined.crop((0, 0, context.width, context.height))
                            delta, _ = frequency_detail_delta(
                                np.asarray(refined, dtype=np.uint8),
                                np.asarray(context, dtype=np.uint8),
                                radius=int(low_pass_radius),
                                gain=float(detail_gain),
                                max_delta=float(max_detail_delta),
                                structure_sigma=float(structure_sigma),
                                base_detail_sigma=float(base_detail_sigma),
                            )
                            local_x0, local_y0, local_x1, local_y1 = tile.local_core_box
                            core_delta = delta[local_y0:local_y1, local_x0:local_x1]
                            mask = phase_normalized_tile_weight(tile, phase_normalizers)
                            canvas_slice = np.s_[
                                tile.core_y0 : tile.core_y1,
                                tile.core_x0 : tile.core_x1,
                            ]
                            phase_slot = (
                                tile.phase
                                if str(merge_mode) == PHASE_WEAVE_MERGE_MODE
                                else 0
                            )
                            accumulators[phase_slot][canvas_slice] += (
                                core_delta * mask[..., None]
                            )
                            weight_sums[phase_slot][canvas_slice] += mask
                            energy_sums[phase_slot][canvas_slice] += (
                                np.mean(
                                    np.square(core_delta),
                                    axis=2,
                                    dtype=np.float32,
                                )
                                * mask
                            )
                            if novel_accumulators:
                                novel_delta, _ = novel_detail_delta(
                                    np.asarray(refined, dtype=np.uint8),
                                    np.asarray(context, dtype=np.uint8),
                                    inner_radius=int(novel_detail_inner_radius),
                                    outer_radius=int(novel_detail_outer_radius),
                                    gain=float(novel_detail_gain),
                                    max_delta=float(novel_detail_max_delta),
                                    structure_sigma=float(novel_detail_structure_sigma),
                                    base_detail_sigma=float(base_detail_sigma),
                                )
                                novel_core_delta = novel_delta[
                                    local_y0:local_y1, local_x0:local_x1
                                ]
                                novel_accumulators[phase_slot][canvas_slice] += (
                                    novel_core_delta * mask[..., None]
                                )
                                novel_energy_sums[phase_slot][canvas_slice] += (
                                    np.mean(
                                        np.square(novel_core_delta),
                                        axis=2,
                                        dtype=np.float32,
                                    )
                                    * mask
                                )

                            p.latents_after_sampling.clear()
                            p.pixels_after_sampling.clear()
                            devices.torch_gc()

                        for value in (
                            accumulators
                            + weight_sums
                            + energy_sums
                            + novel_accumulators
                            + novel_energy_sums
                        ):
                            value.flush()
                        state.job = f"VRAM-Canvas Stage {stage_number}/{len(stages)} Merge"
                        state.textinfo = (
                            f"{stage_w}x{stage_h} | frequency and {str(merge_mode)} merge"
                        )
                        current, consensus_stats = self._finalize_stage(
                            base,
                            accumulators,
                            weight_sums,
                            energy_sums,
                            novel_accumulators,
                            novel_energy_sums,
                            work_dir,
                            stage_number,
                            float(consensus_sigma),
                            float(novel_detail_consensus_sigma),
                            float(novel_detail_consensus_strength),
                            str(merge_mode),
                        )
                        state.nextjob()
                    finally:
                        for value in (
                            accumulators
                            + weight_sums
                            + energy_sums
                            + novel_accumulators
                            + novel_energy_sums
                        ):
                            self._close_memmap(value)

                    gui_manifest["stage_reports"].append(
                        {
                            "stage": stage_number,
                            "size": [stage_w, stage_h],
                            "denoise": denoise,
                            "tile_count": len(plans),
                            "grid_origin": (
                                [
                                    balanced_virtual_axis_origin(
                                        stage_w,
                                        core_size,
                                        core_overlap,
                                        phase_count=int(phase_count),
                                    ),
                                    balanced_virtual_axis_origin(
                                        stage_h,
                                        core_size,
                                        core_overlap,
                                        phase_count=int(phase_count),
                                    ),
                                ]
                                if str(merge_mode) == PHASE_WEAVE_MERGE_MODE
                                else [0, 0]
                            ),
                            "consensus_stats": consensus_stats,
                            "tiles": tile_records,
                        }
                    )

                    if save_stages and stage_number < len(stages):
                        stage_info = f"VRAM-Canvas intermediate stage {stage_number}/{len(stages)}, " f"Size: {stage_w}x{stage_h}, Seed: {global_seed}"
                        images.save_image(
                            current,
                            p.outpath_samples,
                            f"vram_canvas_stage_{stage_number:02d}",
                            global_seed,
                            p.prompt,
                            opts.samples_format,
                            info=stage_info,
                            p=p,
                        )

            if last_processed is None:
                raise RuntimeError("VRAM-Canvas did not process any tile.")
            if smart_finish:
                state.job = "VRAM-Canvas Smart Finish"
                state.textinfo = (
                    f"{target_w}x{target_h} | adaptive chroma and coherent-detail gate"
                )
                current, smart_finish_report = smart_finish_image(
                    current,
                    color_strength=float(smart_color_strength),
                    detail_guard=bool(detail_guard),
                    detail_strength=float(finish_detail_strength),
                    detail_radius=float(finish_detail_radius),
                    detail_threshold=float(finish_detail_threshold),
                    max_detail_delta=float(finish_max_detail_delta),
                )
                gui_manifest["smart_finish_summary"] = smart_finish_summary(
                    smart_finish_report
                )
                state.nextjob()

            state.job = "VRAM-Canvas Finalize"
            state.textinfo = f"{target_w}x{target_h} | metadata and final save"
            final_info = last_processed.info or ""
            final_info = replace_infotext_size(
                final_info,
                last_diffusion_size[0],
                last_diffusion_size[1],
                target_w,
                target_h,
            )
            final_info = replace_infotext_seed(final_info, global_seed)
            last_processed.images = [current]
            last_processed.info = final_info
            last_processed.infotexts = [final_info]
            last_processed.seed = global_seed
            last_processed.all_seeds = [global_seed]
            last_processed.width = target_w
            last_processed.height = target_h
            if opts.enable_pnginfo:
                current.info["parameters"] = final_info
                current.info["vram_canvas"] = json.dumps(
                    gui_manifest, ensure_ascii=False, separators=(",", ":")
                )
                current.info["krea2_smart_highres"] = json.dumps(
                    {
                        "format_version": 2,
                        "target_size": [target_w, target_h],
                        "global_seed": global_seed,
                        "detail_prompt_appended": bool(append_krea2_detail_prompt),
                        "smart_finish": bool(smart_finish),
                        "merge_mode": str(merge_mode),
                        "grid_layout": gui_manifest["grid_layout"],
                        "grid_origin": gui_manifest["grid_origin"],
                        "exact_img2img_steps": EXACT_IMG2IMG_STEPS,
                        "exact_img2img_steps_scope": EXACT_IMG2IMG_STEPS_SCOPE,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                if str(merge_mode) == PHASE_WEAVE_MERGE_MODE:
                    current.info["krea2_phaseweave"] = json.dumps(
                        {
                            "format_version": 4,
                            "product_name": KREA2_PHASEWEAVE_PRODUCT_NAME,
                            "profile_key": KREA2_PHASEWEAVE_PROFILE_KEY,
                            "merge_mode": PHASE_WEAVE_MERGE_MODE,
                            "target_size": [target_w, target_h],
                            "phase_count": int(phase_count),
                            "grid_layout": gui_manifest["grid_layout"],
                            "grid_stride": gui_manifest["grid_stride"],
                            "grid_phase_offset": gui_manifest["grid_phase_offset"],
                            "grid_padding_mode": gui_manifest["grid_padding_mode"],
                            "grid_origin": gui_manifest["grid_origin"],
                            **phase_weave_configuration(),
                            "exact_img2img_steps": EXACT_IMG2IMG_STEPS,
                            "exact_img2img_steps_scope": EXACT_IMG2IMG_STEPS_SCOPE,
                            "stage_reports": gui_manifest["stage_reports"],
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                if smart_finish_report is not None:
                    current.info["krea2_smart_finish"] = json.dumps(
                        smart_finish_report,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )

            p.width = target_w
            p.height = target_h
            p.seed = global_seed
            if original_save_samples:
                images.save_image(
                    current,
                    p.outpath_samples,
                    "",
                    global_seed,
                    p.prompt,
                    opts.samples_format,
                    info=final_info,
                    p=p,
                    existing_info={
                        key: value
                        for key, value in current.info.items()
                        if isinstance(value, str)
                    },
                )
            state.nextjob()
            return last_processed
        finally:
            p.batch_size = original_batch_size
            p.n_iter = original_n_iter
            p.do_not_save_samples = original_do_not_save_samples
            p.do_not_save_grid = original_do_not_save_grid
            p.resize_mode = original_resize_mode
            p.width = original_width
            p.height = original_height
            p.denoising_strength = original_denoising_strength
            p.init_images = original_init_images
            p.seed = original_seed
            p.subseed = original_subseed
            p.steps = original_steps
            p.all_seeds = original_all_seeds
            p.all_subseeds = original_all_subseeds
            p.prompt = original_prompt
            if had_all_prompts:
                p.all_prompts = original_all_prompts
            elif hasattr(p, "all_prompts"):
                delattr(p, "all_prompts")
            if had_all_negative_prompts:
                p.all_negative_prompts = original_all_negative_prompts
            elif hasattr(p, "all_negative_prompts"):
                delattr(p, "all_negative_prompts")
            if had_main_prompt:
                p.main_prompt = original_main_prompt
            elif hasattr(p, "main_prompt"):
                delattr(p, "main_prompt")
            if had_prompts:
                p.prompts = original_prompts
            elif hasattr(p, "prompts"):
                delattr(p, "prompts")
            if had_restore_faces:
                p.restore_faces = original_restore_faces
            elif hasattr(p, "restore_faces"):
                delattr(p, "restore_faces")
            if had_tiling:
                p.tiling = original_tiling
            elif hasattr(p, "tiling"):
                delattr(p, "tiling")
            processing_snapshot.restore(
                p,
                preserve=("extra_generation_params",),
            )

    @staticmethod
    def _pad_tile(tile: Image.Image, alignment: int = 16) -> Image.Image:
        padded_width = max(alignment, ((tile.width + alignment - 1) // alignment) * alignment)
        padded_height = max(alignment, ((tile.height + alignment - 1) // alignment) * alignment)
        if (padded_width, padded_height) == tile.size:
            return tile.copy()
        values = np.asarray(tile.convert("RGB"), dtype=np.uint8)
        padded = np.pad(
            values,
            (
                (0, padded_height - tile.height),
                (0, padded_width - tile.width),
                (0, 0),
            ),
            mode="edge",
        )
        return Image.fromarray(padded, mode="RGB")
