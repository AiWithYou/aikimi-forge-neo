import json
import math
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from backend import memory_management
import modules.scripts as scripts
from modules import devices, images, processing
from modules.shared import opts, state
from modules_forge.krea2_highres import (
    EXACT_IMG2IMG_STEPS,
    EXACT_IMG2IMG_STEPS_SCOPE,
    ProcessingSnapshot,
    internal_exact_img2img_steps,
)
from modules_forge.krea2_local_supersample import (
    LOCAL_SUPERSAMPLE_PROFILES,
    MODE_FOCUSED_ROI_REWRITE,
    MODE_FULL_IMAGE_GRID,
    MODE_ROI_BOXES,
    PROFILE_FOCUSED_FACE_1536,
    PROFILE_ROI_ULTRA_2048,
    PROFILE_SAFE_1536,
    append_focused_face_guidance,
    append_local_detail_guidance,
    apply_canvas_residual,
    apply_canvas_residual_striped,
    build_axis_normalizers,
    candidate_seed,
    enforce_maximum_tile_count,
    estimate_temporary_bytes,
    evaluate_highres_candidate,
    extract_padded_payload,
    get_profile,
    lanczos_upscale,
    linear_to_uint8,
    normalized_tile_weight,
    parse_roi_boxes,
    plan_focused_rois,
    plan_local_tiles,
    rgb_sha256,
    roi_core_mask,
    select_candidate,
    select_tiles_for_rois,
    tile_composition_weights,
    validate_krea2_module_names,
    validate_request,
)
from modules_forge.krea2_upscale import replace_infotext_size
from modules_forge.vram_canvas import replace_infotext_seed
from modules_forge.workflow_ui import (
    workflow_hero,
    workflow_section,
    workflow_summary,
)

PROFILE_OUTPUT_KEYS = (
    "payload",
    "core",
    "overlap",
    "process_edge",
    "steps",
    "denoise",
    "candidates",
    "luma_cap",
    "chroma_cap",
    "low_frequency_reject_radius",
    "context_scale",
    "rewrite_feather",
)


class Krea2LocalSupersampleMemoryError(Exception):
    """A user-facing allocation failure that prevents Forge from hiding advice."""


class Krea2LocalSupersampleDetail(scripts.Script):
    def title(self):
        return "Krea2 Local Supersample Detail"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        defaults = get_profile(PROFILE_SAFE_1536)
        gr.HTML(
            workflow_hero(
                "Krea2 Local Supersample Detail",
                "画像全体または指定領域だけを高解像度で再評価し、安全なディテール残差だけを元画像へ戻します。",
                badges=("通常img2img", "Batch 1 × 1", "PNG metadata必須", "Krea2専用"),
                steps=(
                    "全体・ROI・顔の書き直しを選ぶ",
                    "目的に合うプロファイルを選ぶ",
                    "ROIが必要なモードは座標を入力する",
                ),
            ),
            elem_classes=["neo-workflow-hero-host"],
        )

        gr.HTML(
            workflow_section(
                1,
                "処理モード",
                "全体はSafe 1536、顔はFocused Face Rewrite 1536が基準です。",
            ),
            elem_classes=["neo-workflow-section-host"],
        )
        with gr.Row(elem_classes=["neo-workflow-grid-2"]):
            mode = gr.Radio(
                label="処理範囲",
                choices=[
                    MODE_FULL_IMAGE_GRID,
                    MODE_ROI_BOXES,
                    MODE_FOCUSED_ROI_REWRITE,
                ],
                value=MODE_FULL_IMAGE_GRID,
                elem_id=self.elem_id("mode"),
            )
            profile = gr.Dropdown(
                label="品質プロファイル",
                choices=list(LOCAL_SUPERSAMPLE_PROFILES),
                value=PROFILE_SAFE_1536,
                elem_id=self.elem_id("profile"),
                tooltip="選択すると関連する処理パラメータをまとめて更新します。",
            )
        apply_profile = gr.Button(
            "プロファイルを再適用",
            elem_id=self.elem_id("apply_profile"),
            elem_classes=["neo-workflow-action"],
        )

        with gr.Group(visible=False) as roi_input_group:
            gr.HTML(
                workflow_section(
                    2,
                    "対象ROI",
                    "1つ以上の left, top, right, bottom をセミコロンで区切ります。",
                ),
                elem_classes=["neo-workflow-section-host"],
            )
            roi_boxes = gr.Textbox(
                label="ROI座標 / Focus Targets",
                value="",
                placeholder="例: 120,80,420,380; 620,100,880,360",
                elem_id=self.elem_id("roi_boxes"),
            )

        with gr.Group(visible=False) as focused_settings_group:
            with gr.Row(elem_classes=["neo-workflow-grid-2"]):
                focused_context_scale = gr.Slider(
                    label="周辺context倍率",
                    minimum=1.0,
                    maximum=4.0,
                    step=0.1,
                    value=defaults["context_scale"],
                    elem_id=self.elem_id("focused_context_scale"),
                )
                focused_rewrite_feather = gr.Slider(
                    label="ROI境界フェザー（source px）",
                    minimum=0,
                    maximum=64,
                    step=1,
                    value=defaults["rewrite_feather"],
                    elem_id=self.elem_id("focused_rewrite_feather"),
                )

        workflow_status = gr.HTML(
            self._workflow_summary_html(
                MODE_FULL_IMAGE_GRID,
                PROFILE_SAFE_1536,
                "",
                defaults["process_edge"],
                defaults["steps"],
                defaults["denoise"],
                defaults["candidates"],
                defaults["payload"],
                defaults["core"],
                defaults["overlap"],
                False,
                256,
            ),
            elem_classes=["neo-workflow-summary-host"],
        )

        with gr.Accordion(
            "詳細設定 · タイル / 候補生成",
            open=False,
            elem_classes=["neo-workflow-accordion"],
        ):
            with gr.Row(elem_classes=["neo-workflow-grid-3"]):
                crop_payload = gr.Slider(
                    label="Crop payload",
                    minimum=256,
                    maximum=768,
                    step=16,
                    value=defaults["payload"],
                    elem_id=self.elem_id("crop_payload"),
                )
                core_size = gr.Slider(
                    label="Core size",
                    minimum=128,
                    maximum=640,
                    step=16,
                    value=defaults["core"],
                    elem_id=self.elem_id("core_size"),
                )
                core_overlap = gr.Slider(
                    label="Core overlap",
                    minimum=0,
                    maximum=256,
                    step=16,
                    value=defaults["overlap"],
                    elem_id=self.elem_id("core_overlap"),
                )

            with gr.Row(elem_classes=["neo-workflow-grid-2"]):
                process_edge = gr.Radio(
                    label="処理解像度",
                    choices=[1536, 2048],
                    value=defaults["process_edge"],
                    elem_id=self.elem_id("process_edge"),
                )
                steps = gr.Slider(
                    label="Exact steps",
                    minimum=1,
                    maximum=12,
                    step=1,
                    value=defaults["steps"],
                    elem_id=self.elem_id("steps"),
                )
            with gr.Row(elem_classes=["neo-workflow-grid-2"]):
                denoising_strength = gr.Slider(
                    label="Denoise",
                    minimum=0.0,
                    maximum=0.60,
                    step=0.01,
                    value=defaults["denoise"],
                    elem_id=self.elem_id("denoising_strength"),
                )
                candidate_count = gr.Radio(
                    label="候補数",
                    choices=[1, 2],
                    value=defaults["candidates"],
                    elem_id=self.elem_id("candidate_count"),
                )

        with gr.Accordion(
            "詳細設定 · 残差保護",
            open=False,
            elem_classes=["neo-workflow-accordion"],
        ):
            with gr.Row(elem_classes=["neo-workflow-grid-3"]):
                luma_residual_cap = gr.Slider(
                    label="輝度残差上限",
                    minimum=1,
                    maximum=32,
                    step=1,
                    value=defaults["luma_cap"],
                    elem_id=self.elem_id("luma_residual_cap"),
                )
                chroma_residual_cap = gr.Slider(
                    label="色差残差上限",
                    minimum=0.5,
                    maximum=12,
                    step=0.5,
                    value=defaults["chroma_cap"],
                    elem_id=self.elem_id("chroma_residual_cap"),
                )
                low_frequency_reject_radius = gr.Slider(
                    label="低周波reject半径",
                    minimum=2,
                    maximum=32,
                    step=1,
                    value=defaults["low_frequency_reject_radius"],
                    elem_id=self.elem_id("low_frequency_reject_radius"),
                )

        gr.HTML(
            workflow_section(
                3,
                "保護・保存",
                "全体2048は高負荷です。必要な場合だけ明示的に許可します。",
            ),
            elem_classes=["neo-workflow-section-host"],
        )
        with gr.Row(elem_classes=["neo-workflow-grid-3"]):
            strong_edge_protection = gr.Checkbox(
                label="強い輪郭を保護",
                value=True,
                elem_id=self.elem_id("strong_edge_protection"),
            )
            append_guidance = gr.Checkbox(
                label="モード別Krea2指示を追加",
                value=True,
                elem_id=self.elem_id("append_guidance"),
            )
            save_qa_crops = gr.Checkbox(
                label="QA cropを保存",
                value=False,
                elem_id=self.elem_id("save_qa_crops"),
            )

        with gr.Row(elem_classes=["neo-workflow-grid-2"]):
            allow_expensive_2048_full_grid = gr.Checkbox(
                label="高負荷な2048全体処理を許可",
                value=False,
                elem_id=self.elem_id("allow_expensive_2048_full_grid"),
            )
            maximum_tile_count = gr.Slider(
                label="最大タイル数",
                minimum=1,
                maximum=512,
                step=1,
                value=256,
                elem_id=self.elem_id("maximum_tile_count"),
            )

        profile_outputs = [
            crop_payload,
            core_size,
            core_overlap,
            process_edge,
            steps,
            denoising_strength,
            candidate_count,
            luma_residual_cap,
            chroma_residual_cap,
            low_frequency_reject_radius,
            focused_context_scale,
            focused_rewrite_feather,
        ]
        summary_inputs = [
            mode,
            profile,
            roi_boxes,
            process_edge,
            steps,
            denoising_strength,
            candidate_count,
            crop_payload,
            core_size,
            core_overlap,
            allow_expensive_2048_full_grid,
            maximum_tile_count,
        ]
        apply_profile.click(
            fn=self._profile_values_with_summary,
            inputs=[
                profile,
                mode,
                roi_boxes,
                allow_expensive_2048_full_grid,
                maximum_tile_count,
            ],
            outputs=[*profile_outputs, workflow_status],
            show_progress="hidden",
        )
        profile.input(
            fn=self._profile_values_with_summary,
            inputs=[
                profile,
                mode,
                roi_boxes,
                allow_expensive_2048_full_grid,
                maximum_tile_count,
            ],
            outputs=[*profile_outputs, workflow_status],
            show_progress="hidden",
        )

        mode.input(
            fn=self._mode_visibility_updates,
            inputs=[mode],
            outputs=[roi_input_group, focused_settings_group],
            show_progress="hidden",
        )
        mode.input(
            fn=self._workflow_summary_html,
            inputs=summary_inputs,
            outputs=[workflow_status],
            show_progress="hidden",
        )
        roi_boxes.blur(
            fn=self._workflow_summary_html,
            inputs=summary_inputs,
            outputs=[workflow_status],
            show_progress="hidden",
        )
        for slider in (
            steps,
            denoising_strength,
            crop_payload,
            core_size,
            core_overlap,
            maximum_tile_count,
        ):
            slider.input(
                fn=self._workflow_summary_html,
                inputs=summary_inputs,
                outputs=[workflow_status],
                show_progress="hidden",
            )
        for control in (
            process_edge,
            candidate_count,
            allow_expensive_2048_full_grid,
        ):
            control.input(
                fn=self._workflow_summary_html,
                inputs=summary_inputs,
                outputs=[workflow_status],
                show_progress="hidden",
            )

        return [
            mode,
            profile,
            roi_boxes,
            crop_payload,
            core_size,
            core_overlap,
            process_edge,
            steps,
            denoising_strength,
            candidate_count,
            luma_residual_cap,
            chroma_residual_cap,
            low_frequency_reject_radius,
            focused_context_scale,
            focused_rewrite_feather,
            strong_edge_protection,
            append_guidance,
            save_qa_crops,
            allow_expensive_2048_full_grid,
            maximum_tile_count,
        ]

    @staticmethod
    def _mode_visibility_updates(mode: str):
        selected_mode = str(mode)
        uses_roi = selected_mode in (MODE_ROI_BOXES, MODE_FOCUSED_ROI_REWRITE)
        uses_focused_settings = selected_mode == MODE_FOCUSED_ROI_REWRITE
        return (
            gr.update(visible=uses_roi),
            gr.update(visible=uses_focused_settings),
        )

    @staticmethod
    def _workflow_summary_html(
        mode,
        profile,
        roi_boxes,
        process_edge,
        steps,
        denoising_strength,
        candidate_count,
        crop_payload,
        core_size,
        core_overlap,
        allow_expensive_2048_full_grid,
        maximum_tile_count,
    ) -> str:
        selected_mode = str(mode or MODE_FULL_IMAGE_GRID)
        selected_profile = str(profile or PROFILE_SAFE_1536)
        roi_text = str(roi_boxes or "").strip()
        edge = int(float(process_edge or 0))
        step_count = int(float(steps or 0))
        candidates = int(float(candidate_count or 0))
        payload = int(float(crop_payload or 0))
        core = int(float(core_size or 0))
        overlap = int(float(core_overlap or 0))
        tile_limit = int(float(maximum_tile_count or 0))
        allow_full_2048 = bool(allow_expensive_2048_full_grid)

        mode_labels = {
            MODE_FULL_IMAGE_GRID: "画像全体",
            MODE_ROI_BOXES: "指定ROI",
            MODE_FOCUSED_ROI_REWRITE: "Focused ROI",
        }
        status = "準備完了"
        tone = "ready"
        note = "変更候補が改善しない領域は採用せず、元画像を保ちます。"

        requires_roi = selected_mode in (
            MODE_ROI_BOXES,
            MODE_FOCUSED_ROI_REWRITE,
        )
        profile_mismatch = (
            selected_mode == MODE_FOCUSED_ROI_REWRITE
            and selected_profile != PROFILE_FOCUSED_FACE_1536
        ) or (
            selected_mode != MODE_FOCUSED_ROI_REWRITE
            and selected_profile == PROFILE_FOCUSED_FACE_1536
        )
        roi_profile_mismatch = selected_profile == PROFILE_ROI_ULTRA_2048 and (
            selected_mode != MODE_ROI_BOXES or not roi_text
        )
        geometry_invalid = (
            payload < core
            or (payload - core) % 2 != 0
            or overlap < 0
            or overlap >= core
        )

        if requires_roi and not roi_text:
            status = "ROIを入力"
            tone = "caution"
            note = "このモードは1つ以上のROI座標が必要です。"
        elif profile_mismatch:
            status = "プロファイル不一致"
            tone = "caution"
            note = (
                "Focused ROI RewriteはFocused Face Rewrite 1536と組み合わせてください。"
            )
        elif roi_profile_mismatch:
            status = "ROI設定を確認"
            tone = "caution"
            note = "ROI Ultra 2048はROI BoxesモードとROI座標が必要です。"
        elif (
            selected_mode == MODE_FULL_IMAGE_GRID
            and edge == 2048
            and not allow_full_2048
        ):
            status = "2048全体処理は未許可"
            tone = "caution"
            note = "1536へ戻すか、高負荷な2048全体処理を明示的に許可してください。"
        elif geometry_invalid:
            status = "タイル形状を確認"
            tone = "caution"
            note = "Payload >= Core、差は偶数、0 <= overlap < Coreにしてください。"
        elif edge == 2048 or candidates > 1:
            status = "高品質・高負荷"
            tone = "experimental"
            note = "候補数と処理解像度に比例して時間とVRAM使用量が増えます。"

        roi_label = (
            f"{len([item for item in roi_text.split(';') if item.strip()])} box"
            if roi_text
            else "なし"
        )
        return workflow_summary(
            f"{mode_labels.get(selected_mode, selected_mode)} · {selected_profile}",
            (
                ("ROI", roi_label),
                ("処理", f"{edge} px / {step_count} steps"),
                ("候補", f"{candidates} / tile"),
                ("Denoise", f"{float(denoising_strength or 0):.2f}"),
                ("Tile", f"payload {payload} / core {core} / overlap {overlap}"),
                ("上限", f"{tile_limit} tiles"),
            ),
            status=status,
            note=note,
            tone=tone,
        )

    @staticmethod
    def _profile_values(profile: str) -> tuple:
        values = get_profile(profile)
        return tuple(values[key] for key in PROFILE_OUTPUT_KEYS)

    @classmethod
    def _profile_values_with_summary(
        cls,
        profile,
        mode,
        roi_boxes,
        allow_expensive_2048_full_grid,
        maximum_tile_count,
    ) -> tuple:
        values = cls._profile_values(str(profile))
        summary = cls._workflow_summary_html(
            mode,
            profile,
            roi_boxes,
            values[3],
            values[4],
            values[5],
            values[6],
            values[0],
            values[1],
            values[2],
            allow_expensive_2048_full_grid,
            maximum_tile_count,
        )
        return (*values, summary)

    @staticmethod
    def _validate_processing_input(p) -> None:
        if int(p.batch_size) != 1 or int(p.n_iter) != 1:
            raise ValueError("Krea2 Local Supersample Detail supports Batch Count 1 and Batch Size 1 only.")
        if not p.init_images or p.init_images[0] is None:
            raise ValueError("Krea2 Local Supersample Detail requires an img2img input image.")
        if getattr(p, "image_mask", None) is not None or getattr(p, "latent_mask", None) is not None:
            raise ValueError("Krea2 Local Supersample Detail supports normal img2img only; inpaint masks are not supported.")
        if not bool(getattr(opts, "enable_pnginfo", False)):
            raise ValueError("Krea2 Local Supersample Detail requires PNG metadata to be enabled before processing.")

    @staticmethod
    def _require_krea2_model(p) -> dict:
        override_settings = getattr(p, "override_settings", None) or {}
        forbidden = sorted(key for key in ("sd_model_checkpoint", "sd_vae", "forge_additional_modules") if key in override_settings)
        if forbidden:
            raise ValueError("Krea2 Local Supersample does not allow per-run model/module overrides " f"({', '.join(forbidden)}). Load Krea2 globally first.")

        configured = validate_krea2_module_names(
            getattr(opts, "sd_model_checkpoint", ""),
            getattr(opts, "forge_additional_modules", []),
        )
        configured_model = getattr(p, "sd_model", None)
        configured_model_config = getattr(configured_model, "model_config", None)
        if type(configured_model).__name__ != "Krea2" or type(configured_model_config).__name__ != "Krea2":
            loader = getattr(processing, "manage_model_and_prompt_cache", None)
            if not callable(loader):
                raise ValueError("Krea2 Local Supersample could not load the configured Krea2 checkpoint.")
            loader(p)

        model = getattr(p, "sd_model", None)
        model_config = getattr(model, "model_config", None)
        forge_objects = getattr(model, "forge_objects", None)
        text_engine = getattr(model, "text_processing_engine_qwen", None)
        vae = getattr(forge_objects, "vae", None)
        vae_model = getattr(vae, "first_stage_model", None)
        if type(model).__name__ != "Krea2" or type(model_config).__name__ != "Krea2":
            raise ValueError("Krea2 Local Supersample requires a loaded Krea2 checkpoint.")
        if type(text_engine).__name__ != "Qwen3VLTextProcessingEngine" or getattr(text_engine, "text_encoder", None) is None or getattr(text_engine, "tokenizer", None) is None:
            raise ValueError("Krea2 Local Supersample requires the loaded Qwen3-VL text encoder.")
        vae_ready = (
            vae_model is not None
            and bool(getattr(vae, "is_wan", False))
            and int(getattr(vae, "latent_channels", 0)) == 16
            and callable(getattr(vae, "encode", None))
            and callable(getattr(vae, "decode", None))
            and callable(getattr(vae_model, "process_in", None))
            and callable(getattr(vae_model, "process_out", None))
        )
        if not vae_ready:
            raise ValueError(
                "Krea2 Local Supersample requires the loaded Qwen Image VAE "
                f"(wrapper={type(vae).__name__}, model={type(vae_model).__name__}, "
                f"is_wan={bool(getattr(vae, 'is_wan', False))}, "
                f"latent_channels={int(getattr(vae, 'latent_channels', 0))})."
            )
        if getattr(forge_objects, "unet", None) is None or getattr(forge_objects, "clip", None) is None:
            raise ValueError("Krea2 model components are incomplete; reload the checkpoint and modules.")

        return configured

    @staticmethod
    def _check_interrupted() -> None:
        if bool(getattr(state, "interrupted", False)) or bool(getattr(state, "skipped", False)) or bool(getattr(state, "stopping_generation", False)):
            raise RuntimeError("Krea2 Local Supersample Detail was interrupted before a complete final image " "was produced; no unfinished tile was returned.")

    @staticmethod
    def _release_sampling_buffers(p) -> None:
        for name in ("latents_after_sampling", "pixels_after_sampling"):
            value = getattr(p, name, None)
            clear = getattr(value, "clear", None)
            if callable(clear):
                clear()
        for name in (
            "init_latent",
            "image_conditioning",
            "c",
            "uc",
            "rng",
            "sampler",
            "modified_noise",
        ):
            if hasattr(p, name):
                setattr(p, name, None)
        devices.torch_gc()

    @staticmethod
    def _prepare_candidate_processing(
        p,
        *,
        process_input: np.ndarray,
        effective_prompt: str,
        negative_prompt: str,
        seed: int,
        process_edge: int,
        steps: int,
        denoise: float,
    ) -> None:
        p.batch_size = 1
        p.n_iter = 1
        p.do_not_save_samples = True
        p.do_not_save_grid = True
        p.resize_mode = 0
        p.width = int(process_edge)
        p.height = int(process_edge)
        p.steps = int(steps)
        p.denoising_strength = float(denoise)

        p.prompt = effective_prompt
        p.negative_prompt = negative_prompt
        p.all_prompts = [effective_prompt]
        p.all_negative_prompts = [negative_prompt]
        p.main_prompt = effective_prompt
        p.main_negative_prompt = negative_prompt
        p.prompts = [effective_prompt]
        p.negative_prompts = [negative_prompt]

        p.seed = int(seed)
        p.subseed = int(seed)
        p.subseed_strength = 0.0
        p.seed_resize_from_h = 0
        p.seed_resize_from_w = 0
        p.all_seeds = [int(seed)]
        p.all_subseeds = [int(seed)]

        p.init_images = [Image.fromarray(process_input, mode="RGB")]
        for name in (
            "image_mask",
            "latent_mask",
            "mask",
            "nmask",
            "mask_for_overlay",
            "overlay_images",
            "paste_to",
        ):
            setattr(p, name, None)
        p.restore_faces = False
        p.tiling = False
        if hasattr(p, "refiner_checkpoint"):
            p.refiner_checkpoint = None

    @staticmethod
    def _close_memmap(value: np.memmap | None) -> None:
        if value is None:
            return
        value.flush()
        mapping = getattr(value, "_mmap", None)
        if mapping is not None:
            mapping.close()

    @staticmethod
    def _quality_aggregate(tile_records: list[dict]) -> dict[str, float | None]:
        candidate_stats = [stats for tile in tile_records for stats in tile.get("candidate_metrics", [])]
        agreements = [float(tile["agreement_coverage"]) for tile in tile_records if tile.get("agreement_coverage") is not None]
        if not candidate_stats:
            return {
                "agreement_coverage": None,
                "mean_low_frequency_drift": 0.0,
                "p95_low_frequency_drift": 0.0,
                "mean_residual": 0.0,
                "p95_residual": 0.0,
                "clipping_fraction": 0.0,
            }
        return {
            "agreement_coverage": float(np.mean(agreements)) if agreements else None,
            "mean_low_frequency_drift": float(np.mean([float(item["mean_low_frequency_drift"]) for item in candidate_stats])),
            "p95_low_frequency_drift": float(np.percentile([float(item["p95_low_frequency_drift"]) for item in candidate_stats], 95)),
            "mean_residual": float(np.mean([float(item["mean_residual"]) for item in candidate_stats])),
            "p95_residual": float(np.percentile([float(item["p95_residual"]) for item in candidate_stats], 95)),
            "clipping_fraction": float(np.mean([float(item["clipping_fraction"]) for item in candidate_stats])),
        }

    @staticmethod
    def _save_qa(qa: dict, output_root: Path) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir = output_root / "krea2_local_supersample_qa" / stamp
        output_dir.mkdir(parents=True, exist_ok=False)

        source_payload = qa["source_payload"]
        process_input = qa["process_input"]
        high_candidate = qa["high_candidate"]
        evaluation = qa["evaluation"]
        residual = qa["residual"]
        Image.fromarray(source_payload, mode="RGB").save(output_dir / "source_payload.png")
        Image.fromarray(process_input, mode="RGB").save(output_dir / "process_input.png")
        Image.fromarray(high_candidate, mode="RGB").save(output_dir / "high_resolution_candidate.png")
        Image.fromarray(linear_to_uint8(evaluation.c1_linear), mode="RGB").save(output_dir / "downsampled_candidate_c1.png")
        Image.fromarray(linear_to_uint8(evaluation.c0_linear), mode="RGB").save(output_dir / "roundtrip_baseline_c0.png")
        visualization = np.clip(np.rint(128.0 + residual * 255.0 * 8.0), 0, 255).astype(np.uint8)
        Image.fromarray(visualization, mode="RGB").save(output_dir / "residual_visualization.png")
        after_payload, _ = apply_canvas_residual(
            source_payload,
            residual,
            np.ones(source_payload.shape[:2], dtype=np.float32),
        )
        Image.fromarray(source_payload, mode="RGB").save(output_dir / "before_payload.png")
        Image.fromarray(after_payload, mode="RGB").save(output_dir / "after_payload.png")
        (output_dir / "qa_manifest.json").write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "tile_index": qa["tile_index"],
                    "core": qa["core"],
                    "payload_box": qa["payload_box"],
                    "effective_zoom": qa["effective_zoom"],
                    "candidate_seed": qa["candidate_seed"],
                    "selected_candidate": qa["selected_candidate"],
                    "rejection_reason": qa["rejection_reason"],
                    "quality_gate_override_reason": qa[
                        "quality_gate_override_reason"
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return output_dir

    def run(
        self,
        p,
        mode: str,
        profile: str,
        roi_boxes: str,
        crop_payload: int,
        core_size: int,
        core_overlap: int,
        process_edge: int,
        steps: int,
        denoising_strength: float,
        candidate_count: int,
        luma_residual_cap: float,
        chroma_residual_cap: float,
        low_frequency_reject_radius: float,
        focused_context_scale: float,
        focused_rewrite_feather: float,
        strong_edge_protection: bool,
        append_guidance: bool,
        save_qa_crops: bool,
        allow_expensive_2048_full_grid: bool,
        maximum_tile_count: int,
    ):
        self._validate_processing_input(p)
        source_input = p.init_images[0]
        source_info = {key: value for key, value in getattr(source_input, "info", {}).items() if isinstance(value, str) and key != "parameters"}
        source_image = images.flatten(source_input, opts.img2img_background_color)
        source_rgb = np.asarray(source_image.convert("RGB"), dtype=np.uint8)
        parsed_rois = parse_roi_boxes(
            roi_boxes,
            source_image.width,
            source_image.height,
        )
        validate_request(
            mode=str(mode),
            profile=str(profile),
            roi_boxes=parsed_rois,
            payload=int(crop_payload),
            core=int(core_size),
            overlap=int(core_overlap),
            process_edge=int(process_edge),
            steps=int(steps),
            denoise=float(denoising_strength),
            candidate_count=int(candidate_count),
            luma_cap=float(luma_residual_cap),
            chroma_cap=float(chroma_residual_cap),
            low_frequency_reject_radius=float(low_frequency_reject_radius),
            focused_context_scale=float(focused_context_scale),
            focused_rewrite_feather=float(focused_rewrite_feather),
            allow_expensive_2048_full_grid=bool(allow_expensive_2048_full_grid),
            maximum_tile_count=int(maximum_tile_count),
        )
        focused_rewrite = str(mode) == MODE_FOCUSED_ROI_REWRITE
        if focused_rewrite:
            selected_plans = plan_focused_rois(
                source_image.width,
                source_image.height,
                parsed_rois,
                context_scale=float(focused_context_scale),
            )
            all_plans = selected_plans
            for plan in selected_plans:
                payload_side = plan.payload_x1 - plan.payload_x0
                if payload_side >= int(process_edge):
                    raise ValueError(
                        "Focused ROI Rewrite must enlarge its context crop. "
                        f"ROI {plan.index} produces a {payload_side}px context crop, "
                        f"which is not smaller than Process Edge {int(process_edge)}. "
                        "Tighten the target ROI or lower Focused Context Scale."
                    )
        else:
            all_plans = plan_local_tiles(
                source_image.width,
                source_image.height,
                payload=int(crop_payload),
                core=int(core_size),
                overlap=int(core_overlap),
            )
            selected_plans = (
                all_plans
                if mode == MODE_FULL_IMAGE_GRID
                else select_tiles_for_rois(all_plans, parsed_rois)
            )
        if not selected_plans:
            raise ValueError("ROI Boxes do not intersect any writable local-detail core.")
        enforce_maximum_tile_count(selected_plans, int(maximum_tile_count))
        normalizers = (
            None
            if focused_rewrite
            else build_axis_normalizers(
                all_plans,
                source_image.width,
                source_image.height,
            )
        )
        self._require_krea2_model(p)

        snapshot = ProcessingSnapshot(p)
        original_save_samples = bool(p.save_samples())
        original_prompt = p.prompt
        original_negative_prompt = getattr(p, "negative_prompt", "")
        if not isinstance(original_prompt, str):
            raise ValueError("Krea2 Local Supersample Detail requires one text prompt.")
        if not isinstance(original_negative_prompt, str):
            raise ValueError("Krea2 Local Supersample Detail requires one negative prompt string.")
        if bool(append_guidance):
            effective_prompt = (
                append_focused_face_guidance(original_prompt)
                if focused_rewrite
                else append_local_detail_guidance(original_prompt)
            )
        else:
            effective_prompt = original_prompt
        global_seed = int(processing.get_fixed_seed(p.seed))
        global_subseed = int(processing.get_fixed_seed(p.subseed))
        input_sha256 = rgb_sha256(source_rgb)
        tile_records: list[dict] = []
        processed_tile_count = 0
        rejected_tile_count = 0
        last_processed = None
        qa_record = None
        residual_sum = None
        weight_sum = None

        try:
            if not isinstance(getattr(p, "extra_generation_params", None), dict):
                p.extra_generation_params = {}
            p.extra_generation_params.update(
                {
                    "Krea2 Local Supersample": (
                        "focused round-trip-compensated ROI rewrite"
                        if focused_rewrite
                        else "linear-light C1-C0 local residual"
                    ),
                    "Krea2 Local Supersample profile": str(profile),
                    "Krea2 Local Supersample mode": str(mode),
                    "Krea2 Local Supersample payload": int(crop_payload),
                    "Krea2 Local Supersample core": int(core_size),
                    "Krea2 Local Supersample overlap": int(core_overlap),
                    "Krea2 Local Supersample process edge": int(process_edge),
                    "Krea2 Local Supersample candidates": int(candidate_count),
                    "Krea2 Local Supersample focused context scale": float(
                        focused_context_scale
                    ),
                    "Krea2 Local Supersample rewrite feather": float(
                        focused_rewrite_feather
                    ),
                    "Krea2 Local Supersample exact img2img steps": EXACT_IMG2IMG_STEPS,
                    "Krea2 Local Supersample exact steps scope": EXACT_IMG2IMG_STEPS_SCOPE,
                }
            )
            state.job_count = len(selected_plans) * int(candidate_count) + 1
            self._check_interrupted()
            temp_root = Path(opts.temp_dir) if getattr(opts, "temp_dir", "") else None
            if temp_root is not None:
                temp_root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix="krea2_local_supersample_",
                dir=temp_root,
            ) as temporary:
                work_dir = Path(temporary)
                required_bytes = estimate_temporary_bytes(
                    source_image.width,
                    source_image.height,
                )
                free_bytes = shutil.disk_usage(work_dir).free
                if free_bytes < required_bytes:
                    raise RuntimeError("Krea2 Local Supersample needs about " f"{required_bytes / 1024**3:.2f} GiB of temporary disk space; " f"only {free_bytes / 1024**3:.2f} GiB is free.")

                residual_sum = np.memmap(
                    work_dir / "residual_sum.float32",
                    dtype=np.float32,
                    mode="w+",
                    shape=source_rgb.shape,
                )
                weight_sum = np.memmap(
                    work_dir / "weight_sum.float32",
                    dtype=np.float32,
                    mode="w+",
                    shape=source_rgb.shape[:2],
                )
                residual_sum[:] = 0
                weight_sum[:] = 0
                try:
                    for tile_number, tile in enumerate(selected_plans, start=1):
                        self._check_interrupted()
                        payload = extract_padded_payload(source_rgb, tile, pad_mode="edge")
                        process_input = lanczos_upscale(payload, int(process_edge))
                        evaluations = []
                        high_candidates = []
                        seeds = []
                        for candidate_index in range(int(candidate_count)):
                            self._check_interrupted()
                            seed = candidate_seed(
                                global_seed,
                                tile.core_x0,
                                tile.core_y0,
                                candidate_index,
                            )
                            seeds.append(seed)
                            self._prepare_candidate_processing(
                                p,
                                process_input=process_input,
                                effective_prompt=effective_prompt,
                                negative_prompt=original_negative_prompt,
                                seed=seed,
                                process_edge=int(process_edge),
                                steps=int(steps),
                                denoise=float(denoising_strength),
                            )
                            unit = "ROI" if focused_rewrite else "Tile"
                            state.job = "Krea2 Local Supersample " f"{unit} {tile_number}/{len(selected_plans)} " f"Candidate {candidate_index + 1}/{int(candidate_count)}"
                            state.textinfo = (
                                f"target {tile.core_x0},{tile.core_y0},{tile.core_x1},{tile.core_y1} | "
                                f"zoom {int(process_edge) / (tile.payload_x1 - tile.payload_x0):.2f}x | "
                                f"steps {int(steps)}"
                                if focused_rewrite
                                else f"core {tile.core_x0},{tile.core_y0} | "
                                f"process {int(process_edge)}x{int(process_edge)} | "
                                f"steps {int(steps)}"
                            )
                            try:
                                with internal_exact_img2img_steps(p):
                                    processed = processing.process_images(p)
                            except Exception as exc:
                                is_oom = getattr(memory_management, "is_oom", lambda _exc: False)
                                if is_oom(exc):
                                    if int(process_edge) > 1536:
                                        message = "Krea2 Local Supersample could not allocate GPU memory at " f"Process Edge {int(process_edge)}. No partial output was returned. " "Retry with Process Edge 1536 and limit 2048 to smaller ROI boxes."
                                    else:
                                        message = "Krea2 Local Supersample could not allocate GPU memory at the " "standard Process Edge 1536. No partial output was returned. " "Release other GPU workloads or use fewer/smaller ROI boxes."
                                    raise Krea2LocalSupersampleMemoryError(message) from exc
                                raise
                            finally:
                                self._release_sampling_buffers(p)
                            self._check_interrupted()
                            if processed is None or not getattr(processed, "images", None):
                                raise RuntimeError(f"Krea2 Local Supersample tile {tile_number} candidate " f"{candidate_index + 1} returned no image.")
                            refined_image = images.flatten(
                                processed.images[0],
                                opts.img2img_background_color,
                            )
                            if refined_image.size != (int(process_edge), int(process_edge)):
                                raise RuntimeError(f"Krea2 Local Supersample candidate returned {refined_image.size}; " f"expected {(int(process_edge), int(process_edge))}.")
                            refined = np.asarray(refined_image.convert("RGB"), dtype=np.uint8)
                            high_candidates.append(refined)
                            evaluations.append(
                                evaluate_highres_candidate(
                                    payload,
                                    process_input,
                                    refined,
                                    low_frequency_reject_radius=float(low_frequency_reject_radius),
                                    luma_cap=float(luma_residual_cap),
                                    chroma_cap=float(chroma_residual_cap),
                                    protect_strong_edges=bool(strong_edge_protection),
                                )
                            )
                            last_processed = processed

                        selection = select_candidate(
                            evaluations,
                            apply_full_rewrite=focused_rewrite,
                        )
                        local_x0, local_y0, local_x1, local_y1 = tile.local_core_box
                        core_residual = selection.residual[
                            local_y0:local_y1,
                            local_x0:local_x1,
                        ]
                        if focused_rewrite:
                            residual_weight = roi_core_mask(
                                tile,
                                [tile.core_box],
                                feather=float(focused_rewrite_feather),
                            )
                            normalization_weight = np.ones_like(
                                residual_weight,
                                dtype=np.float32,
                            )
                        elif mode == MODE_ROI_BOXES:
                            residual_weight, normalization_weight = (
                                tile_composition_weights(
                                    tile,
                                    normalizers,
                                    parsed_rois,
                                    feather=0.0,
                                )
                            )
                        else:
                            residual_weight = normalized_tile_weight(
                                tile,
                                normalizers,
                            )
                            normalization_weight = residual_weight
                        if (
                            not np.all(np.isfinite(residual_weight))
                            or not np.all(np.isfinite(normalization_weight))
                            or np.any(residual_weight < 0)
                            or np.any(normalization_weight < 0)
                        ):
                            raise RuntimeError("local supersample produced invalid canvas weights")
                        canvas_slice = np.s_[
                            tile.core_y0 : tile.core_y1,
                            tile.core_x0 : tile.core_x1,
                        ]
                        residual_sum[canvas_slice] += (
                            core_residual * residual_weight[..., None]
                        )
                        # In focused mode the denominator stays one so the target
                        # feather is not cancelled. Standard rejected/no-op tiles
                        # still contribute zero-residual normalized weight.
                        weight_sum[canvas_slice] += normalization_weight

                        candidate_metrics = [{key: (bool(value) if isinstance(value, (bool, np.bool_)) else float(value) if isinstance(value, (float, int, np.floating, np.integer)) else str(value)) for key, value in evaluation.stats.items()} for evaluation in evaluations]
                        record = {
                            "tile_index": int(tile.index),
                            "core": [
                                int(tile.core_x0),
                                int(tile.core_y0),
                                int(tile.core_x1),
                                int(tile.core_y1),
                            ],
                            "payload_box": [
                                int(tile.payload_x0),
                                int(tile.payload_y0),
                                int(tile.payload_x1),
                                int(tile.payload_y1),
                            ],
                            "payload_side": int(tile.payload_x1 - tile.payload_x0),
                            "effective_zoom": float(
                                int(process_edge) / (tile.payload_x1 - tile.payload_x0)
                            ),
                            "candidate_seed": [int(seed) for seed in seeds],
                            "selected_candidate": (int(selection.selected_index + 1) if selection.selected_index is not None else None),
                            "agreement_coverage": (
                                None
                                if focused_rewrite or int(candidate_count) != 2
                                else float(selection.agreement_coverage)
                            ),
                            "candidate_metrics": candidate_metrics,
                            "rejection_reason": selection.rejection_reason,
                            "quality_gate_override_reason": selection.quality_gate_override_reason,
                        }
                        tile_records.append(record)
                        processed_tile_count += 1
                        if selection.selected_index is None:
                            rejected_tile_count += 1

                        if qa_record is None and bool(save_qa_crops):
                            diagnostic_index = (
                                selection.selected_index
                                if selection.selected_index is not None
                                else min(
                                    range(len(evaluations)),
                                    key=lambda index: float(evaluations[index].stats.get("quality_score", math.inf)),
                                )
                            )
                            qa_record = {
                                "source_payload": payload.copy(),
                                "process_input": process_input.copy(),
                                "high_candidate": high_candidates[diagnostic_index].copy(),
                                "evaluation": evaluations[diagnostic_index],
                                "residual": selection.residual.copy(),
                                "tile_index": int(tile.index),
                                "core": record["core"],
                                "payload_box": record["payload_box"],
                                "effective_zoom": record["effective_zoom"],
                                "candidate_seed": int(seeds[diagnostic_index]),
                                "selected_candidate": record["selected_candidate"],
                                "rejection_reason": record["rejection_reason"],
                                "quality_gate_override_reason": record["quality_gate_override_reason"],
                            }

                    residual_sum.flush()
                    weight_sum.flush()
                    self._check_interrupted()
                    state.job = "Krea2 Local Supersample Finalize"
                    state.textinfo = (
                        f"{source_image.width}x{source_image.height} | "
                        + (
                            "blend coherent focused rewrites and preserve target exterior"
                            if focused_rewrite
                            else "normalize residual weights and preserve ROI exterior"
                        )
                    )
                    final_rgb, composition_stats = apply_canvas_residual_striped(
                        source_rgb,
                        residual_sum,
                        weight_sum,
                        stripe_height=256,
                    )
                finally:
                    self._close_memmap(residual_sum)
                    self._close_memmap(weight_sum)
                    residual_sum = None
                    weight_sum = None

            self._check_interrupted()
            if last_processed is None:
                raise RuntimeError("Krea2 Local Supersample did not process any candidate.")
            if bool(save_qa_crops) and qa_record is not None:
                self._save_qa(qa_record, Path(p.outpath_samples))

            final_image = Image.fromarray(final_rgb, mode="RGB")
            output_sha256 = rgb_sha256(final_rgb)
            aggregate = self._quality_aggregate(tile_records)
            manifest = {
                "format_version": 1,
                "algorithm": (
                    "focused single-context round-trip-compensated RGB rewrite"
                    if focused_rewrite
                    else "linear-light round-trip local residual"
                ),
                "input_size": [source_image.width, source_image.height],
                "output_size": [source_image.width, source_image.height],
                "profile": str(profile),
                "mode": str(mode),
                "global_seed": global_seed,
                "payload": None if focused_rewrite else int(crop_payload),
                "core": None if focused_rewrite else int(core_size),
                "overlap": None if focused_rewrite else int(core_overlap),
                "process_edge": int(process_edge),
                "steps": int(steps),
                "denoise": float(denoising_strength),
                "candidate_count": int(candidate_count),
                "exact_img2img_steps": EXACT_IMG2IMG_STEPS,
                "exact_img2img_steps_scope": EXACT_IMG2IMG_STEPS_SCOPE,
                "luma_cap": float(luma_residual_cap),
                "chroma_cap": float(chroma_residual_cap),
                "low_frequency_reject_radius": float(low_frequency_reject_radius),
                "strong_edge_protection": bool(strong_edge_protection),
                "focused_rewrite": focused_rewrite,
                "focused_context_scale": float(focused_context_scale),
                "focused_rewrite_feather": float(focused_rewrite_feather),
                "tile_count": len(selected_plans),
                "focused_region_count": (
                    len(selected_plans) if focused_rewrite else 0
                ),
                "processed_tile_count": processed_tile_count,
                "rejected_noop_tile_count": rejected_tile_count,
                "quality_gate_override_count": sum(
                    tile.get("quality_gate_override_reason") is not None
                    for tile in tile_records
                ),
                "agreement_coverage": (
                    None if focused_rewrite else aggregate["agreement_coverage"]
                ),
                "mean_low_frequency_drift": aggregate["mean_low_frequency_drift"],
                "p95_low_frequency_drift": aggregate["p95_low_frequency_drift"],
                "mean_residual": aggregate["mean_residual"],
                "p95_residual": aggregate["p95_residual"],
                "clipping_fraction": float(composition_stats["clipping_fraction"]),
                "candidate_clipping_fraction": aggregate["clipping_fraction"],
                "input_sha256": input_sha256,
                "output_sha256": output_sha256,
                "sha256_scope": "decoded RGB dimensions and pixel bytes",
                "tiles": tile_records,
            }

            final_info = replace_infotext_size(
                last_processed.info or "",
                int(process_edge),
                int(process_edge),
                source_image.width,
                source_image.height,
            )
            final_info = replace_infotext_seed(final_info, global_seed)
            png_info = dict(source_info)
            png_info["parameters"] = final_info
            png_info["krea2_local_supersample"] = json.dumps(
                manifest,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            final_image.info.update(png_info)

            last_processed.images = [final_image]
            last_processed.extra_images = []
            last_processed.index_of_first_image = 0
            last_processed.video_path = None
            last_processed.info = final_info
            last_processed.infotexts = [final_info]
            last_processed.seed = global_seed
            last_processed.all_seeds = [global_seed]
            last_processed.subseed = global_subseed
            last_processed.all_subseeds = [global_subseed]
            last_processed.width = source_image.width
            last_processed.height = source_image.height
            last_processed.batch_size = 1
            last_processed.prompt = effective_prompt
            last_processed.all_prompts = [effective_prompt]
            last_processed.negative_prompt = original_negative_prompt
            last_processed.all_negative_prompts = [original_negative_prompt]
            last_processed.extra_generation_params = dict(p.extra_generation_params)

            p.width = source_image.width
            p.height = source_image.height
            p.seed = global_seed
            p.subseed = global_subseed
            p.prompt = effective_prompt
            p.negative_prompt = original_negative_prompt
            if original_save_samples:
                images.save_image(
                    final_image,
                    p.outpath_samples,
                    "",
                    global_seed,
                    effective_prompt,
                    "png",
                    info=final_info,
                    p=p,
                    existing_info=png_info,
                )
            state.nextjob()
            return last_processed
        finally:
            self._close_memmap(residual_sum)
            self._close_memmap(weight_sum)
            snapshot.restore(p)
