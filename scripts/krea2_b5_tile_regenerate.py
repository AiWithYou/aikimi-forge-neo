from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import tempfile
import time

import gradio as gr
from PIL import Image

from backend import memory_management
import modules.scripts as scripts
from modules import devices, images, processing
from modules.krea2_quality import adaptive_detail_guard
from modules.shared import opts, state
from modules_forge.krea2_highres import (
    EXACT_IMG2IMG_STEPS,
    EXACT_IMG2IMG_STEPS_SCOPE,
    ProcessingSnapshot,
    internal_exact_img2img_steps,
    krea2_detail_prompt,
)
from modules_forge.krea2_local_supersample import (
    rgb_sha256,
    validate_krea2_module_names,
)
from modules_forge.krea2_upscale import replace_infotext_size
from modules_forge.vram_canvas import replace_infotext_seed
from modules_forge.workflow_ui import (
    workflow_hero,
    workflow_section,
    workflow_summary,
)
from tools.apply_krea2_identity_guard import (
    ProtectionEllipse,
    protection_mask,
)
from tools.krea2_b5_tile_regenerate import (
    B5_HEIGHT_MM,
    DEFAULT_TARGET_SIZE,
    LOCAL_TILE_PROMPT,
    apply_stage_protection,
    estimate_temporary_bytes,
    file_sha256,
    regenerate_stage,
    scale_protection_regions_xy,
    tile_origins,
    validate_args,
    validate_b5_aspect_ratio,
)


PROMPT_MODE_LOCAL = "safe_local"
PROMPT_MODE_IMG2IMG = "img2img"
PROMPT_MODE_GUIDED = "img2img_guided"

B5_GUI_ARGUMENTS = (
    "working_scale",
    "stages",
    "maximum_tile_count",
    "prompt_mode",
    "tile_size",
    "process_edge",
    "overlap",
    "steps",
    "denoising_strength",
    "merge_mode",
    "low_anchor_sigma",
    "detail_radius",
    "detail_gain",
    "max_tile_delta",
    "structure_sigma",
    "base_detail_sigma",
    "protection_rows",
    "print_detail_strength",
    "print_detail_radius",
    "print_detail_threshold",
    "print_max_detail_delta",
    "save_stage_images",
    "save_manifest",
)


class Krea2B5MemoryError(Exception):
    """A user-facing allocation failure that must not be hidden by Forge."""


class Krea2B5TileRegenerate(scripts.Script):
    def title(self):
        return "Krea2 B5 Whole-Tile Regeneration"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        gr.HTML(
            workflow_hero(
                "Krea2 B5 Whole-Tile Regeneration",
                "1K縦長画像を、重なり付きの完全タイル再生成でJIS B5 2896×4096へ仕上げます。",
                badges=(
                    "通常img2img",
                    "Krea2専用",
                    "Batch 1 × 1",
                    "JIS B5 2896 × 4096",
                    "長時間処理",
                ),
                steps=(
                    "Krea2と入力画像を選ぶ",
                    "必要なら顔・目の保護範囲を追加する",
                    "処理プランを確認してGenerate",
                ),
            ),
            elem_classes=["neo-workflow-hero-host"],
        )

        gr.HTML(
            workflow_section(
                1,
                "B5プラン",
                "推奨設定では1024×1448入力を165回の完全タイル処理で2896×4096へ仕上げます。",
            ),
            elem_classes=["neo-workflow-section-host"],
        )
        apply_recommended = gr.Button(
            "B5 4K 推奨設定を再適用",
            variant="primary",
            elem_id=self.elem_id("apply_recommended"),
            elem_classes=["neo-workflow-action"],
        )
        with gr.Row(elem_classes=["neo-workflow-grid-3"]):
            working_scale = gr.Radio(
                label="作業倍率",
                choices=[("2×（推奨）", 2), ("1×", 1)],
                value=2,
                elem_id=self.elem_id("working_scale"),
            )
            stages = gr.Radio(
                label="生成stage",
                choices=[1, 2],
                value=1,
                elem_id=self.elem_id("stages"),
            )
            maximum_tile_count = gr.Slider(
                label="最大タイル数",
                minimum=1,
                maximum=1024,
                step=1,
                value=256,
                elem_id=self.elem_id("maximum_tile_count"),
                info="計画がこの数を超える場合はモデル処理前に停止します。",
            )

        prompt_mode = gr.Radio(
            label="タイル用プロンプト",
            choices=[
                ("安全な局所復元ガイド（推奨）", PROMPT_MODE_LOCAL),
                ("上部のimg2img prompt", PROMPT_MODE_IMG2IMG),
                ("上部prompt + Krea2 detail guidance", PROMPT_MODE_GUIDED),
            ],
            value=PROMPT_MODE_LOCAL,
            elem_id=self.elem_id("prompt_mode"),
        )

        workflow_status = gr.HTML(
            self._workflow_summary_html(
                2,
                1,
                256,
                PROMPT_MODE_LOCAL,
                256,
                1024,
                64,
                6,
                0.35,
                "low_anchor",
                16.0,
                [],
            ),
            elem_classes=["neo-workflow-summary-host"],
        )

        gr.HTML(
            workflow_section(
                2,
                "タイル生成",
                "Sampler、Scheduler、CFG、Model shift、Seedは上部の通常img2img設定を使います。",
            ),
            elem_classes=["neo-workflow-section-host"],
        )
        with gr.Row(elem_classes=["neo-workflow-grid-3"]):
            steps = gr.Slider(
                label="Exact Steps",
                minimum=1,
                maximum=12,
                step=1,
                value=6,
                elem_id=self.elem_id("steps"),
            )
            denoising_strength = gr.Slider(
                label="Denoise",
                minimum=0.0,
                maximum=0.60,
                step=0.01,
                value=0.35,
                elem_id=self.elem_id("denoising_strength"),
            )
            merge_mode = gr.Radio(
                label="合成方式",
                choices=[
                    ("低周波を元画像へ固定（推奨）", "low_anchor"),
                    ("元画像に存在する高周波だけ採用", "source_gate"),
                ],
                value="low_anchor",
                elem_id=self.elem_id("merge_mode"),
            )

        with gr.Accordion(
            "詳細設定 · タイル / 残差",
            open=False,
            elem_classes=["neo-workflow-accordion"],
        ):
            with gr.Row(elem_classes=["neo-workflow-grid-3"]):
                tile_size = gr.Dropdown(
                    label="Source tile",
                    choices=[128, 256],
                    value=256,
                    elem_id=self.elem_id("tile_size"),
                )
                process_edge = gr.Dropdown(
                    label="Process edge",
                    choices=[768, 1024, 1536],
                    value=1024,
                    elem_id=self.elem_id("process_edge"),
                )
                overlap = gr.Slider(
                    label="Overlap",
                    minimum=0,
                    maximum=192,
                    step=16,
                    value=64,
                    elem_id=self.elem_id("overlap"),
                )
            with gr.Row(elem_classes=["neo-workflow-grid-3"]):
                low_anchor_sigma = gr.Slider(
                    label="Low-anchor sigma",
                    minimum=1.0,
                    maximum=64.0,
                    step=1.0,
                    value=16.0,
                    elem_id=self.elem_id("low_anchor_sigma"),
                )
                max_tile_delta = gr.Slider(
                    label="最大RGB変化",
                    minimum=1.0,
                    maximum=64.0,
                    step=1.0,
                    value=16.0,
                    elem_id=self.elem_id("max_tile_delta"),
                )
                detail_radius = gr.Slider(
                    label="Source-gate radius",
                    minimum=1,
                    maximum=32,
                    step=1,
                    value=8,
                    elem_id=self.elem_id("detail_radius"),
                )
            with gr.Row(elem_classes=["neo-workflow-grid-3"]):
                detail_gain = gr.Slider(
                    label="Source-gate gain",
                    minimum=0.1,
                    maximum=3.0,
                    step=0.05,
                    value=1.15,
                    elem_id=self.elem_id("detail_gain"),
                )
                structure_sigma = gr.Slider(
                    label="Structure sigma",
                    minimum=1.0,
                    maximum=32.0,
                    step=0.5,
                    value=12.0,
                    elem_id=self.elem_id("structure_sigma"),
                )
                base_detail_sigma = gr.Slider(
                    label="Base-detail sigma",
                    minimum=0.5,
                    maximum=16.0,
                    step=0.5,
                    value=2.5,
                    elem_id=self.elem_id("base_detail_sigma"),
                )

        gr.HTML(
            workflow_section(
                3,
                "顔・目などの保護",
                "元のimg2img入力画像のpixel座標で、楕円を1行ずつ指定します。空行は無視します。",
            ),
            elem_classes=["neo-workflow-section-host"],
        )
        protection_rows = gr.Dataframe(
            headers=["名前", "左", "上", "右", "下", "フェザー"],
            datatype=["str", "number", "number", "number", "number", "number"],
            type="array",
            row_count=2,
            column_count=6,
            value=[],
            interactive=True,
            max_height=190,
            label="保護楕円（元画像pixel座標）",
            elem_id=self.elem_id("protection_rows"),
        )

        with gr.Accordion(
            "詳細設定 · Print Finish / 保存",
            open=False,
            elem_classes=["neo-workflow-accordion"],
        ):
            with gr.Row(elem_classes=["neo-workflow-grid-2"]):
                save_stage_images = gr.Checkbox(
                    label="中間stage PNGも保存",
                    value=False,
                    elem_id=self.elem_id("save_stage_images"),
                )
                save_manifest = gr.Checkbox(
                    label="実行manifestを保存",
                    value=True,
                    elem_id=self.elem_id("save_manifest"),
                )
            with gr.Row(elem_classes=["neo-workflow-grid-2"]):
                print_detail_strength = gr.Slider(
                    label="Print detail strength",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.35,
                    elem_id=self.elem_id("print_detail_strength"),
                )
                print_detail_radius = gr.Slider(
                    label="Print detail radius",
                    minimum=0.1,
                    maximum=4.0,
                    step=0.1,
                    value=1.0,
                    elem_id=self.elem_id("print_detail_radius"),
                )
            with gr.Row(elem_classes=["neo-workflow-grid-2"]):
                print_detail_threshold = gr.Slider(
                    label="Print detail threshold",
                    minimum=0.1,
                    maximum=4.0,
                    step=0.1,
                    value=0.7,
                    elem_id=self.elem_id("print_detail_threshold"),
                )
                print_max_detail_delta = gr.Slider(
                    label="Print maximum detail delta",
                    minimum=0.1,
                    maximum=16.0,
                    step=0.1,
                    value=3.0,
                    elem_id=self.elem_id("print_max_detail_delta"),
                )

        summary_inputs = [
            working_scale,
            stages,
            maximum_tile_count,
            prompt_mode,
            tile_size,
            process_edge,
            overlap,
            steps,
            denoising_strength,
            merge_mode,
            low_anchor_sigma,
            protection_rows,
        ]
        apply_recommended.click(
            fn=self._recommended_values_with_summary,
            inputs=[],
            outputs=[
                working_scale,
                stages,
                maximum_tile_count,
                prompt_mode,
                tile_size,
                process_edge,
                overlap,
                steps,
                denoising_strength,
                merge_mode,
                low_anchor_sigma,
                max_tile_delta,
                workflow_status,
            ],
            show_progress="hidden",
        )
        for control in summary_inputs:
            control.input(
                fn=self._workflow_summary_html,
                inputs=summary_inputs,
                outputs=[workflow_status],
                show_progress="hidden",
            )

        return [
            working_scale,
            stages,
            maximum_tile_count,
            prompt_mode,
            tile_size,
            process_edge,
            overlap,
            steps,
            denoising_strength,
            merge_mode,
            low_anchor_sigma,
            detail_radius,
            detail_gain,
            max_tile_delta,
            structure_sigma,
            base_detail_sigma,
            protection_rows,
            print_detail_strength,
            print_detail_radius,
            print_detail_threshold,
            print_max_detail_delta,
            save_stage_images,
            save_manifest,
        ]

    @staticmethod
    def _recommended_values_with_summary() -> tuple:
        values = (
            2,
            1,
            256,
            PROMPT_MODE_LOCAL,
            256,
            1024,
            64,
            6,
            0.35,
            "low_anchor",
            16.0,
            16.0,
        )
        return (
            *values,
            Krea2B5TileRegenerate._workflow_summary_html(
                *values[:11],
                [],
            ),
        )

    @staticmethod
    def _workflow_summary_html(
        working_scale,
        stages,
        maximum_tile_count,
        prompt_mode,
        tile_size,
        process_edge,
        overlap,
        steps,
        denoising_strength,
        merge_mode,
        low_anchor_sigma,
        protection_rows,
    ) -> str:
        tile = int(float(tile_size or 0))
        process = int(float(process_edge or 0))
        overlap_value = int(float(overlap or 0))
        stage_count = int(float(stages or 0))
        scale = int(float(working_scale or 0))
        protected = Krea2B5TileRegenerate._protection_row_count(protection_rows)
        status = "推奨設定"
        tone = "ready"
        note = "入力1024×1448では165 tile passです。正確な計画はGenerate開始前に検査します。"
        if tile <= 0 or overlap_value < 0 or overlap_value >= tile:
            status = "タイル形状を確認"
            tone = "caution"
            note = "0 <= Overlap < Source tileにしてください。"
        elif process <= tile * 2 or process % 16 != 0:
            status = "処理解像度を確認"
            tone = "caution"
            note = "Process edgeは縮小後tileより大きく、16の倍数である必要があります。"
        elif process > 1024 or stage_count > 1:
            status = "高負荷設定"
            tone = "experimental"
            note = "Process edgeとstage数に比例してVRAM、時間、一時disk使用量が増えます。"

        prompt_labels = {
            PROMPT_MODE_LOCAL: "局所復元ガイド",
            PROMPT_MODE_IMG2IMG: "上部prompt",
            PROMPT_MODE_GUIDED: "上部prompt + guidance",
        }
        merge_labels = {
            "low_anchor": f"Low anchor σ{float(low_anchor_sigma or 0):g}",
            "source_gate": "Source gate",
        }
        return workflow_summary(
            "JIS B5 2896×4096",
            (
                ("作業", f"{scale}× / {stage_count} stage"),
                ("Tile", f"{tile} → {process} → {tile * 2}, overlap {overlap_value}"),
                ("生成", f"Exact {int(float(steps or 0))} steps / denoise {float(denoising_strength or 0):.2f}"),
                ("合成", merge_labels.get(str(merge_mode), str(merge_mode))),
                ("Prompt", prompt_labels.get(str(prompt_mode), str(prompt_mode))),
                ("保護", f"{protected} ellipse"),
                ("上限", f"{int(float(maximum_tile_count or 0))} tiles"),
            ),
            status=status,
            note=note,
            tone=tone,
        )

    @staticmethod
    def _rows_as_lists(rows) -> list[list]:
        if rows is None:
            return []
        if hasattr(rows, "values") and hasattr(rows.values, "tolist"):
            rows = rows.values.tolist()
        elif hasattr(rows, "tolist"):
            rows = rows.tolist()
        if not isinstance(rows, (list, tuple)):
            raise ValueError("保護楕円はGUIの表へ入力してください。")
        return [list(row) for row in rows if isinstance(row, (list, tuple))]

    @classmethod
    def _protection_row_count(cls, rows) -> int:
        try:
            return sum(
                not all(cls._cell_is_empty(cell) for cell in row)
                for row in cls._rows_as_lists(rows)
            )
        except ValueError:
            return 0

    @staticmethod
    def _cell_is_empty(value) -> bool:
        if value is None or (isinstance(value, str) and not value.strip()):
            return True
        try:
            return bool(value != value)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _integer_cell(value, *, row_number: int, field: str) -> int:
        if isinstance(value, bool) or Krea2B5TileRegenerate._cell_is_empty(value):
            raise ValueError(f"保護楕円 {row_number}行目の{field}を整数で入力してください。")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"保護楕円 {row_number}行目の{field}を整数で入力してください。"
            ) from exc
        if not math.isfinite(number) or not number.is_integer():
            raise ValueError(f"保護楕円 {row_number}行目の{field}を整数で入力してください。")
        return int(number)

    @classmethod
    def _parse_protection_rows(
        cls,
        rows,
        image_size: tuple[int, int],
    ) -> list[ProtectionEllipse]:
        regions: list[ProtectionEllipse] = []
        labels: set[str] = set()
        for row_number, row in enumerate(cls._rows_as_lists(rows), start=1):
            if all(cls._cell_is_empty(cell) for cell in row):
                continue
            if len(row) != 6:
                raise ValueError(
                    f"保護楕円 {row_number}行目は名前、左、上、右、下、フェザーの6列が必要です。"
                )
            label = str(row[0] or "").strip()
            if not label:
                raise ValueError(f"保護楕円 {row_number}行目の名前を入力してください。")
            if label in labels:
                raise ValueError(f"保護楕円の名前 {label!r} が重複しています。")
            x0 = cls._integer_cell(row[1], row_number=row_number, field="左")
            y0 = cls._integer_cell(row[2], row_number=row_number, field="上")
            x1 = cls._integer_cell(row[3], row_number=row_number, field="右")
            y1 = cls._integer_cell(row[4], row_number=row_number, field="下")
            feather = cls._integer_cell(row[5], row_number=row_number, field="フェザー")
            if x1 <= x0 or y1 <= y0:
                raise ValueError(
                    f"保護楕円 {row_number}行目は右 > 左、下 > 上にしてください。"
                )
            if feather <= 0:
                raise ValueError(
                    f"保護楕円 {row_number}行目のフェザーは1以上にしてください。"
                )
            labels.add(label)
            regions.append(ProtectionEllipse(label, (x0, y0, x1, y1), feather))
        if regions:
            protection_mask(image_size, regions)
        return regions

    @staticmethod
    def _validate_processing_input(p) -> None:
        if int(p.batch_size) != 1 or int(p.n_iter) != 1:
            raise ValueError(
                "Krea2 B5 Whole-Tile Regeneration supports Batch Count 1 and Batch Size 1 only."
            )
        if not p.init_images or p.init_images[0] is None:
            raise ValueError(
                "Krea2 B5 Whole-Tile Regeneration requires an img2img input image."
            )
        if getattr(p, "image_mask", None) is not None or getattr(
            p, "latent_mask", None
        ) is not None:
            raise ValueError(
                "Krea2 B5 Whole-Tile Regeneration supports normal img2img only; inpaint masks are not supported."
            )
        if not bool(getattr(opts, "enable_pnginfo", False)):
            raise ValueError(
                "Krea2 B5 Whole-Tile Regeneration requires PNG metadata to be enabled."
            )

    @staticmethod
    def _require_krea2_model(p) -> dict:
        override_settings = getattr(p, "override_settings", None) or {}
        forbidden = sorted(
            key
            for key in (
                "sd_model_checkpoint",
                "sd_vae",
                "forge_additional_modules",
            )
            if key in override_settings
        )
        if forbidden:
            raise ValueError(
                "Krea2 B5 regeneration does not allow per-run model/module overrides "
                f"({', '.join(forbidden)}). Load Krea2 globally first."
            )

        configured = validate_krea2_module_names(
            getattr(opts, "sd_model_checkpoint", ""),
            getattr(opts, "forge_additional_modules", []),
        )
        configured_model = getattr(p, "sd_model", None)
        configured_model_config = getattr(configured_model, "model_config", None)
        if (
            type(configured_model).__name__ != "Krea2"
            or type(configured_model_config).__name__ != "Krea2"
        ):
            loader = getattr(processing, "manage_model_and_prompt_cache", None)
            if not callable(loader):
                raise ValueError(
                    "Krea2 B5 regeneration could not load the configured Krea2 checkpoint."
                )
            loader(p)

        model = getattr(p, "sd_model", None)
        model_config = getattr(model, "model_config", None)
        forge_objects = getattr(model, "forge_objects", None)
        text_engine = getattr(model, "text_processing_engine_qwen", None)
        vae = getattr(forge_objects, "vae", None)
        vae_model = getattr(vae, "first_stage_model", None)
        if type(model).__name__ != "Krea2" or type(model_config).__name__ != "Krea2":
            raise ValueError("Krea2 B5 regeneration requires a loaded Krea2 checkpoint.")
        if (
            type(text_engine).__name__ != "Qwen3VLTextProcessingEngine"
            or getattr(text_engine, "text_encoder", None) is None
            or getattr(text_engine, "tokenizer", None) is None
        ):
            raise ValueError(
                "Krea2 B5 regeneration requires the loaded Qwen3-VL text encoder."
            )
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
                "Krea2 B5 regeneration requires the loaded Qwen Image VAE."
            )
        if getattr(forge_objects, "unet", None) is None or getattr(
            forge_objects, "clip", None
        ) is None:
            raise ValueError(
                "Krea2 model components are incomplete; reload the checkpoint and modules."
            )
        return configured

    @staticmethod
    def _check_interrupted() -> None:
        if (
            bool(getattr(state, "interrupted", False))
            or bool(getattr(state, "skipped", False))
            or bool(getattr(state, "stopping_generation", False))
        ):
            raise RuntimeError(
                "Krea2 B5 tile regeneration was interrupted before a complete final image was produced; no unfinished tile was returned."
            )

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
    def _prepare_tile_processing(
        p,
        *,
        process_input: Image.Image,
        prompt: str,
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
        p.prompt = prompt
        p.negative_prompt = negative_prompt
        p.all_prompts = [prompt]
        p.all_negative_prompts = [negative_prompt]
        p.main_prompt = prompt
        p.main_negative_prompt = negative_prompt
        p.prompts = [prompt]
        p.negative_prompts = [negative_prompt]
        p.seed = int(seed)
        p.subseed = int(seed)
        p.subseed_strength = 0.0
        p.seed_resize_from_h = 0
        p.seed_resize_from_w = 0
        p.all_seeds = [int(seed)]
        p.all_subseeds = [int(seed)]
        p.init_images = [process_input]
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
    def _total_tile_count(
        source_size: tuple[int, int],
        *,
        working_scale: int,
        stages: int,
        tile_size: int,
        overlap: int,
    ) -> int:
        width = int(source_size[0]) * int(working_scale)
        height = int(source_size[1]) * int(working_scale)
        total = 0
        for _ in range(int(stages)):
            total += len(tile_origins(width, tile_size, overlap)) * len(
                tile_origins(height, tile_size, overlap)
            )
            width *= 2
            height *= 2
        return total

    @staticmethod
    def _args(
        p,
        *,
        working_scale,
        stages,
        tile_size,
        process_edge,
        overlap,
        steps,
        denoising_strength,
        merge_mode,
        low_anchor_sigma,
        detail_radius,
        detail_gain,
        max_tile_delta,
        structure_sigma,
        base_detail_sigma,
        print_detail_strength,
        print_detail_radius,
        print_detail_threshold,
        print_max_detail_delta,
        protection_regions,
    ) -> Namespace:
        return Namespace(
            api="",
            target_width=int(DEFAULT_TARGET_SIZE[0]),
            target_height=int(DEFAULT_TARGET_SIZE[1]),
            tile_size=int(tile_size),
            process_edge=int(process_edge),
            overlap=int(overlap),
            working_scale=int(working_scale),
            stages=int(stages),
            steps=int(steps),
            denoise=float(denoising_strength),
            sampler=str(getattr(p, "sampler_name", "")),
            scheduler=str(getattr(p, "scheduler", "")),
            cfg=float(getattr(p, "cfg_scale", 1.0)),
            shift=float(getattr(p, "distilled_cfg_scale", 1.15)),
            seed=int(getattr(p, "seed", -1)),
            merge_mode=str(merge_mode),
            low_anchor_sigma=float(low_anchor_sigma),
            detail_radius=int(detail_radius),
            detail_gain=float(detail_gain),
            max_tile_delta=float(max_tile_delta),
            structure_sigma=float(structure_sigma),
            base_detail_sigma=float(base_detail_sigma),
            print_detail_strength=float(print_detail_strength),
            print_detail_radius=float(print_detail_radius),
            print_detail_threshold=float(print_detail_threshold),
            print_max_detail_delta=float(print_max_detail_delta),
            protect_ellipse=list(protection_regions),
            timeout=900.0,
        )

    def run(
        self,
        p,
        working_scale,
        stages,
        maximum_tile_count,
        prompt_mode,
        tile_size,
        process_edge,
        overlap,
        steps,
        denoising_strength,
        merge_mode,
        low_anchor_sigma,
        detail_radius,
        detail_gain,
        max_tile_delta,
        structure_sigma,
        base_detail_sigma,
        protection_rows,
        print_detail_strength,
        print_detail_radius,
        print_detail_threshold,
        print_max_detail_delta,
        save_stage_images,
        save_manifest,
    ):
        self._validate_processing_input(p)
        source_input = p.init_images[0]
        source_info = {
            key: value
            for key, value in getattr(source_input, "info", {}).items()
            if isinstance(value, str) and key != "parameters"
        }
        source = images.flatten(source_input, opts.img2img_background_color)
        source = source.convert("RGB")
        protection_regions = self._parse_protection_rows(protection_rows, source.size)
        args = self._args(
            p,
            working_scale=working_scale,
            stages=stages,
            tile_size=tile_size,
            process_edge=process_edge,
            overlap=overlap,
            steps=steps,
            denoising_strength=denoising_strength,
            merge_mode=merge_mode,
            low_anchor_sigma=low_anchor_sigma,
            detail_radius=detail_radius,
            detail_gain=detail_gain,
            max_tile_delta=max_tile_delta,
            structure_sigma=structure_sigma,
            base_detail_sigma=base_detail_sigma,
            print_detail_strength=print_detail_strength,
            print_detail_radius=print_detail_radius,
            print_detail_threshold=print_detail_threshold,
            print_max_detail_delta=print_max_detail_delta,
            protection_regions=protection_regions,
        )
        validate_args(args)
        if source.width < args.tile_size or source.height < args.tile_size:
            raise ValueError("input image is smaller than one complete source tile")
        validate_b5_aspect_ratio(source.size, DEFAULT_TARGET_SIZE)
        generated_size = (
            source.width * args.working_scale * (2**args.stages),
            source.height * args.working_scale * (2**args.stages),
        )
        if max(generated_size) > 8192:
            raise ValueError(
                f"generated working canvas {generated_size[0]}x{generated_size[1]} exceeds the guarded 8192 px long-edge limit"
            )
        if (
            generated_size[0] < args.target_width
            or generated_size[1] < args.target_height
        ):
            raise ValueError(
                f"generated working canvas {generated_size[0]}x{generated_size[1]} is smaller than target {args.target_width}x{args.target_height}; the final operation would enlarge instead of reduce"
            )
        total_tiles = self._total_tile_count(
            source.size,
            working_scale=args.working_scale,
            stages=args.stages,
            tile_size=args.tile_size,
            overlap=args.overlap,
        )
        if total_tiles > int(maximum_tile_count):
            raise ValueError(
                f"B5 plan needs {total_tiles} tile passes, exceeding the configured maximum of {int(maximum_tile_count)}"
            )

        original_prompt = getattr(p, "prompt", "")
        original_negative_prompt = getattr(p, "negative_prompt", "")
        if not isinstance(original_prompt, str) or not isinstance(
            original_negative_prompt, str
        ):
            raise ValueError("Krea2 B5 regeneration requires one prompt string.")
        if prompt_mode == PROMPT_MODE_LOCAL:
            effective_prompt = LOCAL_TILE_PROMPT
        elif prompt_mode == PROMPT_MODE_IMG2IMG:
            effective_prompt = original_prompt
        elif prompt_mode == PROMPT_MODE_GUIDED:
            effective_prompt = krea2_detail_prompt(original_prompt)
        else:
            raise ValueError(f"unknown B5 prompt mode: {prompt_mode!r}")

        global_seed = int(processing.get_fixed_seed(p.seed))
        global_subseed = int(processing.get_fixed_seed(p.subseed))
        args.seed = global_seed
        temp_root = Path(opts.temp_dir) if getattr(opts, "temp_dir", "") else None
        if temp_root is not None:
            temp_root.mkdir(parents=True, exist_ok=True)

        snapshot = ProcessingSnapshot(p)
        original_save_samples = bool(p.save_samples())
        last_processed = None
        saved_stage_paths: list[str] = []
        try:
            with tempfile.TemporaryDirectory(
                prefix="krea2_b5_",
                dir=temp_root,
            ) as temporary:
                work_dir = Path(temporary)
                required_bytes = estimate_temporary_bytes(
                    source.size,
                    working_scale=args.working_scale,
                    stages=args.stages,
                )
                free_bytes = shutil.disk_usage(work_dir).free
                if free_bytes < required_bytes:
                    raise RuntimeError(
                        "Krea2 B5 regeneration needs about "
                        f"{required_bytes / 1024**3:.2f} GiB of temporary disk space; "
                        f"only {free_bytes / 1024**3:.2f} GiB is free."
                    )
                configured = self._require_krea2_model(p)
                self._check_interrupted()
                state.job_count = total_tiles + 1

                if not isinstance(getattr(p, "extra_generation_params", None), dict):
                    p.extra_generation_params = {}
                p.extra_generation_params.update(
                    {
                        "Krea2 B5 Whole-Tile Regeneration": "complete-tile oversampled B5 regeneration",
                        "Krea2 B5 tile": args.tile_size,
                        "Krea2 B5 process edge": args.process_edge,
                        "Krea2 B5 overlap": args.overlap,
                        "Krea2 B5 merge": args.merge_mode,
                        "Krea2 B5 exact img2img steps": EXACT_IMG2IMG_STEPS,
                        "Krea2 B5 exact steps scope": EXACT_IMG2IMG_STEPS_SCOPE,
                    }
                )

                def progress_callback(event: dict) -> None:
                    state.job = (
                        f"Krea2 B5 Stage {event['stage']}/{event['stages']} · "
                        f"Tile {event['tile']}/{event['tiles']}"
                    )
                    state.textinfo = (
                        f"input {event['input_x']},{event['input_y']} · "
                        f"process {event['process_edge']} · exact steps {event['steps']} · "
                        f"denoise {event['denoise']:.2f}"
                    )

                def tile_regenerator(tile: Image.Image, **request):
                    nonlocal last_processed
                    self._check_interrupted()
                    process_input = tile.resize(
                        (int(request["process_edge"]), int(request["process_edge"])),
                        Image.Resampling.LANCZOS,
                    )
                    self._prepare_tile_processing(
                        p,
                        process_input=process_input,
                        prompt=str(request["prompt"]),
                        negative_prompt=str(request["negative_prompt"]),
                        seed=int(request["seed"]),
                        process_edge=int(request["process_edge"]),
                        steps=int(request["steps"]),
                        denoise=float(request["denoise"]),
                    )
                    started = time.perf_counter()
                    try:
                        with internal_exact_img2img_steps(p):
                            processed = processing.process_images(p)
                    except Exception as exc:
                        is_oom = getattr(memory_management, "is_oom", lambda _exc: False)
                        if is_oom(exc):
                            raise Krea2B5MemoryError(
                                "Krea2 B5 regeneration could not allocate GPU memory. No partial output was returned. Release other GPU workloads or lower Process edge."
                            ) from exc
                        raise
                    finally:
                        self._release_sampling_buffers(p)
                    self._check_interrupted()
                    if processed is None or not getattr(processed, "images", None):
                        raise RuntimeError("Krea2 B5 tile generation returned no image.")
                    refined = images.flatten(
                        processed.images[0],
                        opts.img2img_background_color,
                    ).convert("RGB")
                    expected = (int(request["process_edge"]),) * 2
                    if refined.size != expected:
                        raise RuntimeError(
                            f"Krea2 B5 tile generation returned {refined.size}; expected {expected}."
                        )
                    last_processed = processed
                    output_edge = int(request["output_edge"])
                    return (
                        refined.resize(
                            (output_edge, output_edge),
                            Image.Resampling.LANCZOS,
                        ),
                        time.perf_counter() - started,
                    )

                current = source.resize(
                    (
                        source.width * args.working_scale,
                        source.height * args.working_scale,
                    ),
                    Image.Resampling.LANCZOS,
                )
                stage_reports = []
                stage_paths: list[Path] = []
                run_started = time.perf_counter()
                for stage_index in range(args.stages):
                    current, report = regenerate_stage(
                        current,
                        args,
                        prompt=effective_prompt,
                        negative_prompt=original_negative_prompt,
                        global_seed=global_seed,
                        original_size=source.size,
                        original_regions=protection_regions,
                        stage_index=stage_index,
                        work_dir=work_dir,
                        output_dir=work_dir,
                        tile_regenerator=tile_regenerator,
                        progress_callback=progress_callback,
                        interrupted=lambda: bool(
                            getattr(state, "interrupted", False)
                            or getattr(state, "skipped", False)
                            or getattr(state, "stopping_generation", False)
                        ),
                        save_stage=bool(save_stage_images and original_save_samples),
                    )
                    if report.get("output"):
                        stage_paths.append(Path(report["output"]))
                    stage_reports.append(report)

                self._check_interrupted()
                state.job = "Krea2 B5 Finalize"
                state.textinfo = f"{args.target_width}×{args.target_height} · Print Finish · identity protection"
                final = current.resize(
                    (args.target_width, args.target_height),
                    Image.Resampling.LANCZOS,
                )
                final, print_finish = adaptive_detail_guard(
                    final,
                    strength=args.print_detail_strength,
                    radius=args.print_detail_radius,
                    detail_threshold=args.print_detail_threshold,
                    max_detail_delta=args.print_max_detail_delta,
                )
                final_base = source.resize(
                    (args.target_width, args.target_height),
                    Image.Resampling.LANCZOS,
                )
                final_regions = scale_protection_regions_xy(
                    protection_regions,
                    args.target_width / source.width,
                    args.target_height / source.height,
                )
                final, final_protection = apply_stage_protection(
                    final,
                    final_base,
                    final_regions,
                )
                if last_processed is None:
                    raise RuntimeError("Krea2 B5 regeneration did not process any tile.")

                dpi = args.target_height / (B5_HEIGHT_MM / 25.4)
                manifest = {
                    "format_version": 1,
                    "algorithm": "Krea2 complete-tile oversampled B5 regeneration",
                    "input_size": list(source.size),
                    "input_sha256": rgb_sha256(source),
                    "working_input_size": [
                        source.width * args.working_scale,
                        source.height * args.working_scale,
                    ],
                    "generated_canvas_size": list(generated_size),
                    "target_size": [args.target_width, args.target_height],
                    "tile_size": args.tile_size,
                    "process_edge": args.process_edge,
                    "reduced_tile_size": args.tile_size * 2,
                    "overlap": args.overlap,
                    "working_scale": args.working_scale,
                    "stages": args.stages,
                    "tile_count": total_tiles,
                    "steps": args.steps,
                    "denoise": args.denoise,
                    "exact_img2img_steps": EXACT_IMG2IMG_STEPS,
                    "exact_img2img_steps_scope": EXACT_IMG2IMG_STEPS_SCOPE,
                    "merge_mode": args.merge_mode,
                    "seed": global_seed,
                    "forge": configured,
                    "protection_regions": [asdict(region) for region in protection_regions],
                    "stage_reports": stage_reports,
                    "print_finish": print_finish,
                    "final_protection": final_protection,
                    "effective_dpi": dpi,
                    "elapsed_seconds": time.perf_counter() - run_started,
                }
                final_info = replace_infotext_size(
                    last_processed.info or "",
                    args.process_edge,
                    args.process_edge,
                    args.target_width,
                    args.target_height,
                )
                final_info = replace_infotext_seed(final_info, global_seed)
                summary = {
                    "format_version": 1,
                    "source_size": list(source.size),
                    "target_size": [args.target_width, args.target_height],
                    "tile_size": args.tile_size,
                    "process_edge": args.process_edge,
                    "stages": args.stages,
                    "tile_count": total_tiles,
                    "seed": global_seed,
                    "protected_region_count": len(protection_regions),
                    "effective_dpi": dpi,
                    "exact_img2img_steps": EXACT_IMG2IMG_STEPS,
                    "exact_img2img_steps_scope": EXACT_IMG2IMG_STEPS_SCOPE,
                }
                png_info = dict(source_info)
                png_info["parameters"] = final_info
                png_info["krea2_b5_tile_regenerate"] = json.dumps(
                    summary,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                final.info.update(png_info)

                p.width = args.target_width
                p.height = args.target_height
                p.seed = global_seed
                p.subseed = global_subseed
                p.prompt = effective_prompt
                p.negative_prompt = original_negative_prompt
                p.all_prompts = [effective_prompt]
                p.all_negative_prompts = [original_negative_prompt]
                p.main_prompt = effective_prompt
                p.main_negative_prompt = original_negative_prompt

                if original_save_samples:
                    final_path, _ = images.save_image(
                        final,
                        p.outpath_samples,
                        "",
                        global_seed,
                        effective_prompt,
                        "png",
                        info=final_info,
                        p=p,
                        existing_info=png_info,
                        suffix="-b5_4k",
                    )
                    manifest["output"] = {
                        "path": str(final_path),
                        "sha256": file_sha256(Path(final_path)),
                        "size": [args.target_width, args.target_height],
                    }
                    if bool(save_stage_images):
                        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        for stage_number, stage_path in enumerate(stage_paths, start=1):
                            with Image.open(stage_path) as opened:
                                stage_image = opened.convert("RGB")
                            saved_path, _ = images.save_image(
                                stage_image,
                                p.outpath_samples,
                                "",
                                global_seed,
                                effective_prompt,
                                "png",
                                info=final_info,
                                p=p,
                                existing_info=png_info,
                                forced_filename=f"krea2_b5_{stamp}_stage_{stage_number:02d}",
                                save_to_dirs=False,
                            )
                            saved_stage_paths.append(str(saved_path))
                        for report, saved_path in zip(
                            manifest["stage_reports"],
                            saved_stage_paths,
                        ):
                            report["output"] = saved_path
                    if bool(save_manifest):
                        manifest_path = Path(final_path).with_suffix(".json")
                        manifest_path.write_text(
                            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8",
                        )

                last_processed.images = [final]
                last_processed.extra_images = []
                last_processed.index_of_first_image = 0
                last_processed.video_path = None
                last_processed.info = final_info
                last_processed.infotexts = [final_info]
                last_processed.seed = global_seed
                last_processed.all_seeds = [global_seed]
                last_processed.subseed = global_subseed
                last_processed.all_subseeds = [global_subseed]
                last_processed.width = args.target_width
                last_processed.height = args.target_height
                last_processed.batch_size = 1
                last_processed.prompt = effective_prompt
                last_processed.all_prompts = [effective_prompt]
                last_processed.negative_prompt = original_negative_prompt
                last_processed.all_negative_prompts = [original_negative_prompt]
                last_processed.extra_generation_params = dict(
                    p.extra_generation_params
                )
                state.nextjob()
                return last_processed
        finally:
            snapshot.restore(p)
