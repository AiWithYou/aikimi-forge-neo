import gradio as gr
from PIL import Image

import modules.scripts as scripts
from modules import devices, images, processing
from modules.krea2_quality import smart_finish_image, smart_finish_summary
from modules.processing import Processed
from modules.shared import opts, state
from modules_forge.krea2_upscale import (
    KREA2_DEFAULT_DENOISE,
    KREA2_STAGE1_DENOISE,
    SAFE_DIFFUSION_LONG_EDGE,
    capped_diffusion_size,
    native_diffusion_long_edge,
    require_native_diffusion_size,
    require_safe_diffusion_size,
    replace_infotext_size,
    target_size,
    two_stage_sizes,
    validate_tile_geometry,
)
from modules_forge.workflow_ui import (
    workflow_hero,
    workflow_section,
    workflow_summary,
)


class KreaTwoStageUpscale(scripts.Script):
    def title(self):
        return "Krea2 2-Stage Upscale"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        gr.HTML(
            workflow_hero(
                "Krea2 2-Stage Upscale",
                "Krea2の対応解像度内でimg2imgし、最後に指定した納品サイズへ正確に拡大する安全側のワークフローです。",
                badges=("通常img2img", "Batch 1 × 1", "Turbo 2K proxy", "Raw 1K proxy"),
                steps=(
                    "基準画像とKrea2モデルを選ぶ",
                    "モデル種別と4K / 8Kを選ぶ",
                    "プラン表示を確認してGenerate",
                ),
            ),
            elem_classes=["neo-workflow-hero-host"],
        )

        gr.HTML(
            workflow_section(
                1,
                "クイック設定",
                "4Kは通常運用、8Kは大判納品向けです。どちらもnative 4K / 8K生成ではありません。",
            ),
            elem_classes=["neo-workflow-section-host"],
        )
        with gr.Row(elem_classes=["neo-workflow-preset-grid"]):
            quick_4k = gr.Button(
                "4K 納品\n長辺 4096",
                variant="primary",
                elem_classes=["neo-workflow-action"],
                elem_id=self.elem_id("quick_4k"),
            )
            quick_8k = gr.Button(
                "8K 納品\n長辺 8192",
                elem_classes=["neo-workflow-action"],
                elem_id=self.elem_id("quick_8k"),
            )

        workflow_status = gr.HTML(
            self._workflow_summary_html(
                4096,
                0,
                0,
                0,
                0,
                "custom",
                False,
                KREA2_STAGE1_DENOISE,
                KREA2_DEFAULT_DENOISE,
                768,
                768,
                96,
                True,
            ),
            elem_classes=["neo-workflow-summary-host"],
        )

        gr.HTML(
            workflow_section(
                2,
                "納品サイズとモデル",
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
            )
            model_profile = gr.Radio(
                label="Krea2モデル種別",
                choices=("custom", "turbo", "raw"),
                value="custom",
                elem_id=self.elem_id("model_profile"),
                tooltip="Turboは2K、Rawは1K、customは安全側に2Kを上限とします。",
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

        with gr.Accordion(
            "詳細設定 · proxy / タイル処理",
            open=False,
            elem_classes=["neo-workflow-accordion"],
        ):
            with gr.Row(elem_classes=["neo-workflow-grid-2"]):
                first_pass_long_edge = gr.Slider(
                    label="Stage 1 長辺（0 = 自動）",
                    minimum=0,
                    maximum=8192,
                    step=64,
                    value=0,
                    elem_id=self.elem_id("first_pass_long_edge"),
                )
                diffusion_long_edge_cap = gr.Slider(
                    label="最終拡散長辺上限（0 = モデル自動）",
                    minimum=0,
                    maximum=SAFE_DIFFUSION_LONG_EDGE,
                    step=64,
                    value=0,
                    elem_id=self.elem_id("diffusion_long_edge_cap"),
                )
            allow_non_native_diffusion = gr.Checkbox(
                label="モデル対応範囲を超える拡散を許可（非推奨）",
                value=False,
                elem_id=self.elem_id("allow_non_native_diffusion"),
            )
            with gr.Row(elem_classes=["neo-workflow-grid-2"]):
                first_pass_denoise = gr.Slider(
                    label="Stage 1 denoise",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=KREA2_STAGE1_DENOISE,
                    elem_id=self.elem_id("first_pass_denoise"),
                )
                final_denoise = gr.Slider(
                    label="最終拡散 denoise",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=KREA2_DEFAULT_DENOISE,
                    elem_id=self.elem_id("final_denoise"),
                )
            method = gr.Radio(
                label="MultiDiffusion方式",
                choices=("MultiDiffusion", "Mixture of Diffusers"),
                value="Mixture of Diffusers",
                elem_id=self.elem_id("method"),
            )
            with gr.Row(elem_classes=["neo-workflow-grid-2"]):
                tile_width = gr.Slider(
                    label="タイル幅",
                    minimum=256,
                    maximum=1280,
                    step=64,
                    value=768,
                    elem_id=self.elem_id("tile_width"),
                )
                tile_height = gr.Slider(
                    label="タイル高さ",
                    minimum=256,
                    maximum=1280,
                    step=64,
                    value=768,
                    elem_id=self.elem_id("tile_height"),
                )
            with gr.Row(elem_classes=["neo-workflow-grid-2"]):
                tile_overlap = gr.Slider(
                    label="タイル重なり",
                    minimum=0,
                    maximum=1024,
                    step=16,
                    value=96,
                    elem_id=self.elem_id("tile_overlap"),
                )
                tile_batch_size = gr.Number(
                    label="Tile Batch Size（固定）",
                    value=1,
                    precision=0,
                    interactive=False,
                    elem_id=self.elem_id("tile_batch_size"),
                )

        gr.HTML(
            workflow_section(
                3,
                "仕上げと保存",
                "Smart Finishは改善を確認できた補正だけを採用します。",
            ),
            elem_classes=["neo-workflow-section-host"],
        )
        save_stage1 = gr.Checkbox(
            label="Stage 1出力も保存",
            value=True,
            elem_id=self.elem_id("save_stage1"),
        )

        with gr.Row(elem_classes=["neo-workflow-grid-3"]):
            smart_finish = gr.Checkbox(
                label="Smart Finish",
                value=True,
                elem_id=self.elem_id("smart_finish"),
            )
            smart_despeckle = gr.Checkbox(
                label="孤立粒を補修",
                value=False,
                elem_id=self.elem_id("smart_despeckle"),
                tooltip="雪・星・粒子が重要な画像ではOFFにします。",
            )
            smart_color_strength = gr.Slider(
                label="色補正強度",
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=0.80,
                elem_id=self.elem_id("smart_color_strength"),
            )

        summary_inputs = [
            final_long_edge,
            first_pass_long_edge,
            diffusion_long_edge_cap,
            final_width,
            final_height,
            model_profile,
            allow_non_native_diffusion,
            first_pass_denoise,
            final_denoise,
            tile_width,
            tile_height,
            tile_overlap,
            smart_finish,
        ]
        quick_4k.click(
            fn=lambda profile_name, allow_non_native, stage1_denoise, stage2_denoise, tile_w, tile_h, overlap, finish_enabled: self._quick_target_values_with_summary(
                4096,
                profile_name,
                allow_non_native,
                stage1_denoise,
                stage2_denoise,
                tile_w,
                tile_h,
                overlap,
                finish_enabled,
            ),
            inputs=[
                model_profile,
                allow_non_native_diffusion,
                first_pass_denoise,
                final_denoise,
                tile_width,
                tile_height,
                tile_overlap,
                smart_finish,
            ],
            outputs=[
                final_long_edge,
                first_pass_long_edge,
                diffusion_long_edge_cap,
                final_width,
                final_height,
                workflow_status,
            ],
            show_progress="hidden",
        )
        quick_8k.click(
            fn=lambda profile_name, allow_non_native, stage1_denoise, stage2_denoise, tile_w, tile_h, overlap, finish_enabled: self._quick_target_values_with_summary(
                8192,
                profile_name,
                allow_non_native,
                stage1_denoise,
                stage2_denoise,
                tile_w,
                tile_h,
                overlap,
                finish_enabled,
            ),
            inputs=[
                model_profile,
                allow_non_native_diffusion,
                first_pass_denoise,
                final_denoise,
                tile_width,
                tile_height,
                tile_overlap,
                smart_finish,
            ],
            outputs=[
                final_long_edge,
                first_pass_long_edge,
                diffusion_long_edge_cap,
                final_width,
                final_height,
                workflow_status,
            ],
            show_progress="hidden",
        )

        for slider in (
            final_long_edge,
            first_pass_long_edge,
            diffusion_long_edge_cap,
            final_width,
            final_height,
            first_pass_denoise,
            final_denoise,
            tile_width,
            tile_height,
            tile_overlap,
        ):
            slider.input(
                fn=self._workflow_summary_html,
                inputs=summary_inputs,
                outputs=[workflow_status],
                show_progress="hidden",
            )
        for control in (model_profile, allow_non_native_diffusion, smart_finish):
            control.input(
                fn=self._workflow_summary_html,
                inputs=summary_inputs,
                outputs=[workflow_status],
                show_progress="hidden",
            )

        return [
            final_long_edge,
            first_pass_long_edge,
            diffusion_long_edge_cap,
            final_width,
            final_height,
            first_pass_denoise,
            final_denoise,
            method,
            tile_width,
            tile_height,
            tile_overlap,
            tile_batch_size,
            save_stage1,
            model_profile,
            allow_non_native_diffusion,
            smart_finish,
            smart_despeckle,
            smart_color_strength,
        ]

    @staticmethod
    def _quick_target_values(long_edge: int) -> tuple[int, int, int, int, int]:
        target = int(long_edge)
        if target not in (4096, 8192):
            raise ValueError("Krea2 quick target must be 4096 or 8192.")
        return target, 0, 0, 0, 0

    @classmethod
    def _quick_target_values_with_summary(
        cls,
        long_edge,
        model_profile,
        allow_non_native_diffusion,
        first_pass_denoise,
        final_denoise,
        tile_width,
        tile_height,
        tile_overlap,
        smart_finish,
    ) -> tuple:
        values = cls._quick_target_values(int(long_edge))
        summary = cls._workflow_summary_html(
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            model_profile,
            allow_non_native_diffusion,
            first_pass_denoise,
            final_denoise,
            tile_width,
            tile_height,
            tile_overlap,
            smart_finish,
        )
        return (*values, summary)

    @staticmethod
    def _workflow_summary_html(
        final_long_edge,
        first_pass_long_edge,
        diffusion_long_edge_cap,
        final_width,
        final_height,
        model_profile,
        allow_non_native_diffusion,
        first_pass_denoise,
        final_denoise,
        tile_width,
        tile_height,
        tile_overlap,
        smart_finish,
    ) -> str:
        long_edge = int(float(final_long_edge or 0))
        first_pass = int(float(first_pass_long_edge or 0))
        requested_cap = int(float(diffusion_long_edge_cap or 0))
        width = int(float(final_width or 0))
        height = int(float(final_height or 0))
        profile = str(model_profile or "custom")
        profile_cap = native_diffusion_long_edge(profile)
        effective_cap = requested_cap if requested_cap > 0 else profile_cap
        tile_w = int(float(tile_width or 0))
        tile_h = int(float(tile_height or 0))
        overlap = int(float(tile_overlap or 0))

        explicit_size_is_partial = (width > 0) != (height > 0)
        output_label = (
            f"{width} × {height} px"
            if width > 0 and height > 0
            else f"長辺 {long_edge} px・入力比率を維持"
        )
        status = "安全側プラン"
        tone = "ready"
        note = (
            "拡散はモデル上限内のproxyで行い、最後にLanczosで納品寸法へ合わせます。"
        )
        if long_edge >= 8192:
            status = "大判納品"
            tone = "caution"
            note = "8Kはproxy拡散後の納品拡大です。細部は4K出力も併せて確認してください。"
        if bool(allow_non_native_diffusion):
            status = "上限超過を許可"
            tone = "caution"
            note = "モデルの対応解像度を超える拡散が有効です。OOMと品質低下を個別に確認してください。"
        if explicit_size_is_partial:
            status = "サイズ要確認"
            tone = "caution"
            note = "最終幅と最終高さは、両方を0にするか両方を1以上にしてください。"

        return workflow_summary(
            f"{profile} · {output_label}",
            (
                ("Stage 1", "自動" if first_pass <= 0 else f"長辺 {first_pass} px"),
                ("拡散上限", f"長辺 {effective_cap} px"),
                (
                    "Denoise",
                    f"{float(first_pass_denoise):.2f} → {float(final_denoise):.2f}",
                ),
                ("Tile", f"{tile_w} × {tile_h} / overlap {overlap}"),
                ("仕上げ", "Smart Finish ON" if bool(smart_finish) else "補正なし"),
            ),
            status=status,
            note=note,
            tone=tone,
        )

    @staticmethod
    def _explicit_dimensions(width: int, height: int) -> tuple[int | None, int | None]:
        explicit_width = int(width) if int(width) > 0 else None
        explicit_height = int(height) if int(height) > 0 else None
        if (explicit_width is None) != (explicit_height is None):
            raise ValueError(
                "Final Width and Final Height must both be 0 or both be > 0."
            )
        return explicit_width, explicit_height

    @staticmethod
    def _effective_diffusion_cap(requested_cap: int, model_profile: str) -> int:
        requested_cap = int(requested_cap)
        if requested_cap < 0:
            raise ValueError("Final Diffusion Long Edge Cap must be >= 0.")
        if requested_cap == 0:
            return native_diffusion_long_edge(str(model_profile))
        return requested_cap

    @staticmethod
    def _enable_multidiffusion(
        p,
        method: str,
        tile_width: int,
        tile_height: int,
        tile_overlap: int,
        tile_batch_size: int,
    ):
        if p.scripts is None or p.script_args is None:
            raise RuntimeError("Krea2 2-Stage Upscale requires img2img script context.")

        multidiffusion = next(
            (
                script
                for script in p.scripts.alwayson_scripts
                if script.title() == "MultiDiffusion Integrated"
            ),
            None,
        )
        if multidiffusion is None:
            raise RuntimeError("MultiDiffusion Integrated script is not loaded.")

        replacement = [
            True,
            method,
            int(tile_width),
            int(tile_height),
            int(tile_overlap),
            int(tile_batch_size),
        ]
        if multidiffusion.args_to - multidiffusion.args_from != len(replacement):
            raise RuntimeError("MultiDiffusion Integrated argument layout changed.")

        script_args = list(p.script_args)
        script_args[multidiffusion.args_from : multidiffusion.args_to] = replacement
        p.script_args = script_args

    @staticmethod
    def _validate_run(
        p,
        first_pass_denoise: float,
        final_denoise: float,
        tile_width: int,
        tile_height: int,
        tile_overlap: int,
        tile_batch_size: int,
        smart_color_strength: float,
    ):
        if p.batch_size != 1 or p.n_iter != 1:
            raise ValueError(
                "Krea2 2-Stage Upscale supports Batch Count 1 and Batch Size 1 only."
            )
        if not p.init_images or p.init_images[0] is None:
            raise ValueError("Krea2 2-Stage Upscale requires an img2img input image.")
        if p.image_mask is not None or p.latent_mask is not None:
            raise ValueError(
                "Krea2 2-Stage Upscale supports normal img2img only; inpaint masks are not supported."
            )
        if not 0 <= first_pass_denoise <= 1:
            raise ValueError("Stage 1 Denoising Strength must be between 0 and 1.")
        if not 0 <= final_denoise <= 1:
            raise ValueError("Final Denoising Strength must be between 0 and 1.")
        if not 0 <= smart_color_strength <= 1:
            raise ValueError("Smart Chroma Strength must be between 0 and 1.")
        validate_tile_geometry(
            int(tile_width),
            int(tile_height),
            int(tile_overlap),
            int(tile_batch_size),
        )

    def run(
        self,
        p,
        final_long_edge: int,
        first_pass_long_edge: int,
        diffusion_long_edge_cap: int,
        final_width: int,
        final_height: int,
        first_pass_denoise: float,
        final_denoise: float,
        method: str,
        tile_width: int,
        tile_height: int,
        tile_overlap: int,
        tile_batch_size: int,
        save_stage1: bool,
        model_profile: str = "custom",
        allow_non_native_diffusion: bool = False,
        smart_finish: bool = True,
        smart_despeckle: bool = False,
        smart_color_strength: float = 0.80,
    ):
        self._validate_run(
            p,
            first_pass_denoise,
            final_denoise,
            tile_width,
            tile_height,
            tile_overlap,
            tile_batch_size,
            smart_color_strength,
        )
        processing.fix_seed(p)

        source = images.flatten(p.init_images[0], opts.img2img_background_color)
        explicit_width, explicit_height = self._explicit_dimensions(
            final_width, final_height
        )
        effective_diffusion_cap = self._effective_diffusion_cap(
            diffusion_long_edge_cap, model_profile
        )
        target_w, target_h = target_size(
            source.width,
            source.height,
            int(final_long_edge),
            explicit_width,
            explicit_height,
        )
        diffusion_w, diffusion_h = capped_diffusion_size(
            source.width,
            source.height,
            target_w,
            target_h,
            effective_diffusion_cap,
        )
        require_safe_diffusion_size(diffusion_w, diffusion_h)
        require_native_diffusion_size(
            diffusion_w,
            diffusion_h,
            str(model_profile),
            bool(allow_non_native_diffusion),
        )
        needs_final_resize = (diffusion_w, diffusion_h) != (target_w, target_h)
        if int(first_pass_long_edge) < 0:
            raise ValueError("Stage 1 Long Edge must be >= 0.")

        use_stage1 = max(diffusion_w, diffusion_h) >= max(source.size) + 128
        if use_stage1:
            (stage1_w, stage1_h), _ = two_stage_sizes(
                source.width,
                source.height,
                diffusion_w,
                diffusion_h,
                int(first_pass_long_edge),
            )
            stage1_long_edge = max(stage1_w, stage1_h)
            stage1_long_edge_label = (
                f"Auto ({stage1_long_edge})"
                if int(first_pass_long_edge) == 0
                else str(stage1_long_edge)
            )
            flow_label = "two-stage"
            stage1_status = "run"
        else:
            stage1_w, stage1_h = source.size
            stage1_long_edge_label = "Skipped (source near diffusion proxy)"
            flow_label = "single-stage"
            stage1_status = "skipped (source near diffusion proxy)"

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
        original_script_args = p.script_args

        try:
            self._enable_multidiffusion(
                p, method, tile_width, tile_height, tile_overlap, tile_batch_size
            )
            p.batch_size = 1
            p.n_iter = 1
            p.resize_mode = 0
            p.do_not_save_grid = True

            p.extra_generation_params.update(
                {
                    "Krea2 2-Stage Upscale": (
                        "Stage 1/2" if use_stage1 else "Single-stage diffusion"
                    ),
                    "Krea2 Upscale flow": flow_label,
                    "Krea2 Stage 1 status": stage1_status,
                    "Krea2 2-Stage Stage 1 long edge": stage1_long_edge_label,
                    "Krea2 2-Stage Stage 1 size": f"{stage1_w}x{stage1_h}",
                    "Krea2 2-Stage Final size": f"{target_w}x{target_h}",
                    "Krea2 2-Stage Diffusion size": f"{diffusion_w}x{diffusion_h}",
                    "Krea2 2-Stage Diffusion cap": effective_diffusion_cap,
                    "Krea2 Model profile": str(model_profile),
                    "Krea2 Smart Finish": bool(smart_finish),
                    "Krea2 Smart Finish strength": float(smart_color_strength),
                    "Krea2 Smart Finish despeckle": bool(smart_despeckle),
                    "Krea2 Smart Finish analysis long edge": 1536,
                }
            )
            if use_stage1:
                p.width = stage1_w
                p.height = stage1_h
                p.denoising_strength = float(first_pass_denoise)
                p.init_images = [
                    source.resize((stage1_w, stage1_h), Image.Resampling.LANCZOS)
                ]
                p.do_not_save_samples = original_do_not_save_samples or not save_stage1
                state.job = "Krea2 2-Stage Upscale: Stage 1/2"
                state.textinfo = f"Krea2 2-Stage Stage 1/2: {stage1_w}x{stage1_h}"

                stage1 = processing.process_images(p)
                if state.interrupted or state.skipped or state.stopping_generation:
                    return stage1
                if not stage1.images:
                    raise RuntimeError(
                        "Krea2 2-Stage Upscale Stage 1 returned no image."
                    )

                stage1_image = images.flatten(
                    stage1.images[0], opts.img2img_background_color
                )
                p.latents_after_sampling.clear()
                p.pixels_after_sampling.clear()
                devices.torch_gc()
            else:
                stage1_image = source

            p.extra_generation_params.update(
                {
                    "Krea2 2-Stage Upscale": (
                        "Stage 2/2" if use_stage1 else "Single-stage diffusion"
                    ),
                    "Krea2 Upscale flow": flow_label,
                    "Krea2 Stage 1 status": stage1_status,
                    "Krea2 2-Stage Stage 1 long edge": stage1_long_edge_label,
                    "Krea2 2-Stage Stage 1 size": f"{stage1_w}x{stage1_h}",
                    "Krea2 2-Stage Final size": f"{target_w}x{target_h}",
                    "Krea2 2-Stage Diffusion size": f"{diffusion_w}x{diffusion_h}",
                    "Krea2 2-Stage Diffusion cap": effective_diffusion_cap,
                    "Krea2 Model profile": str(model_profile),
                    "Krea2 Smart Finish": bool(smart_finish),
                    "Krea2 Smart Finish strength": float(smart_color_strength),
                    "Krea2 Smart Finish despeckle": bool(smart_despeckle),
                    "Krea2 Smart Finish analysis long edge": 1536,
                }
            )
            p.width = diffusion_w
            p.height = diffusion_h
            p.denoising_strength = float(final_denoise)
            p.init_images = [
                stage1_image.resize(
                    (diffusion_w, diffusion_h), Image.Resampling.LANCZOS
                )
            ]
            # Stage 2 must never save before deterministic resize and Smart Finish.
            p.do_not_save_samples = True
            p.do_not_save_grid = True
            if use_stage1:
                state.job = "Krea2 2-Stage Upscale: Stage 2/2"
                state.textinfo = f"Krea2 2-Stage Stage 2/2: {diffusion_w}x{diffusion_h}"
            else:
                state.job = "Krea2 2-Stage Upscale: Single diffusion stage"
                state.textinfo = (
                    "Krea2 single-stage diffusion "
                    f"(Stage 1 skipped): {diffusion_w}x{diffusion_h}"
                )

            final = processing.process_images(p)
            if final is None:
                return Processed(p, [], p.seed, "")
            if state.interrupted or state.skipped or state.stopping_generation:
                return final

            p.latents_after_sampling.clear()
            p.pixels_after_sampling.clear()
            devices.torch_gc()

            if needs_final_resize:
                final.info = replace_infotext_size(
                    final.info,
                    diffusion_w,
                    diffusion_h,
                    target_w,
                    target_h,
                )
                final.infotexts = [
                    replace_infotext_size(
                        text,
                        diffusion_w,
                        diffusion_h,
                        target_w,
                        target_h,
                    )
                    for text in final.infotexts
                ]

            delivery_images = []
            quality_summaries = []
            for image in final.images:
                delivery = images.flatten(image, opts.img2img_background_color)
                if delivery.size != (target_w, target_h):
                    delivery = delivery.resize(
                        (target_w, target_h), Image.Resampling.LANCZOS
                    )

                if smart_finish:
                    state.job = "Krea2 2-Stage Upscale: Smart Finish"
                    state.textinfo = f"Krea2 Smart Finish: {target_w}x{target_h} (CPU)"
                    delivery, quality_report = smart_finish_image(
                        delivery,
                        color_strength=float(smart_color_strength),
                        analysis_long_edge=1536,
                        despeckle=bool(smart_despeckle),
                    )
                    quality_summaries.append(smart_finish_summary(quality_report))
                else:
                    quality_summaries.append("disabled")
                delivery_images.append(delivery)

            for index, summary in enumerate(quality_summaries):
                if index < len(final.infotexts):
                    final.infotexts[index] = (
                        f"{final.infotexts[index]}, Krea2 Smart Finish result: {summary}"
                    )
            if final.infotexts:
                final.info = final.infotexts[0]

            if opts.enable_pnginfo:
                for index, image in enumerate(delivery_images):
                    info = (
                        final.infotexts[index]
                        if index < len(final.infotexts)
                        else final.info
                    )
                    image.info["parameters"] = info

            final.images = delivery_images
            final.width = target_w
            final.height = target_h
            p.width = target_w
            p.height = target_h

            if original_save_samples:
                first_image = final.index_of_first_image
                for index, image in enumerate(
                    delivery_images[first_image:], start=first_image
                ):
                    info = (
                        final.infotexts[index]
                        if index < len(final.infotexts)
                        else final.info
                    )
                    images.save_image(
                        image,
                        p.outpath_samples,
                        "",
                        p.seed,
                        p.prompt,
                        opts.samples_format,
                        info=info,
                        p=p,
                    )
            return final
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
            p.script_args = original_script_args
