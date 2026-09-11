"""Forge img2img Script entrypoint for HyperWeave 4K/8K."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image

import modules.scripts as scripts
from modules import devices, gradio_compat, images, processing
from modules.api import api
from modules.shared import opts, state
from modules_forge.workflow_ui import (
    workflow_hero,
    workflow_section,
    workflow_summary,
)

from hyperweave.config import (
    AccumulatorMode,
    ContentProfile,
    HYPERWEAVE_VERSION,
    PRESETS,
    HyperWeaveConfig,
    HyperWeavePreset,
    TargetMode,
    resolve_target_size,
)
from hyperweave.engine import (
    HyperWeaveEngine,
    HyperWeaveInterrupted,
    ProgressEvent,
)
from hyperweave.forge_adapter import ForgeGeneratorAdapter, ProcessingSnapshot
from hyperweave.noise import resolve_seed


logger = logging.getLogger("hyperweave")
HYPERWEAVE_ID = "hyper_weave"


UI_FIELDS = (
    "enabled",
    "target_mode",
    "custom_long_edge",
    "custom_width",
    "custom_height",
    "preset",
    "content_profile",
    "seed",
    "exact_steps",
    "overdraw_amount",
    "structural_lock",
    "low_frequency_lock",
    "tile_input_size",
    "core_size",
    "context_size",
    "stride",
    "accumulator_mode",
    "temp_directory",
    "maximum_ram_gib",
    "anchor_strength",
    "global_overdraw_strength",
    "face_strength",
    "hair_strength",
    "material_strength",
    "micro_strength",
    "global_candidates",
    "face_candidates",
    "hair_candidates",
    "material_candidates",
    "roi_final_pass_count",
    "enable_face_redraw",
    "enable_hair_redraw",
    "enable_material_redraw",
    "enable_micro_pass",
    "detector_provider",
    "detector_model_path",
    "minimum_face_size",
    "maximum_face_count",
    "identity_reference",
    "structure_conditioner",
    "protection_mask",
    "boost_mask",
    "manual_face_mask",
    "mask_channel",
    "boost_strength",
    "flat_region_detail",
    "face_structure_tolerance",
    "hair_flow_tolerance",
    "new_edge_tolerance",
    "color_drift_tolerance",
    "candidate_rejection_strictness",
    "enable_spatial_rescue",
    "spatial_decision_size",
    "spatial_transition_width",
    "spatial_score_margin",
    "spatial_fragmentation_limit",
    "spatial_minimum_component_cells",
    "roi_stages",
    "back_projection_iterations",
    "back_projection_beta",
    "append_prompt_suffixes",
    "common_suffix",
    "anchor_suffix",
    "global_suffix",
    "face_illustration_suffix",
    "face_photo_suffix",
    "hair_suffix",
    "material_suffix",
    "micro_suffix",
    "negative_suffix",
    "save_debug_images",
    "save_all_candidates",
    "save_maps",
    "save_roi_crops",
    "save_metrics_json",
    "save_metrics_csv",
    "debug_output_directory",
    "model_background",
    "share_anchor_noise_family",
    "oom_retry_smaller_tile",
    "candidate_score_margin",
)


def _normalize_script_image(value: Any) -> Image.Image | None:
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, str):
        return api.decode_base64_to_image(value)
    if isinstance(value, dict):
        candidate = value.get("composite") or value.get("background")
        if isinstance(candidate, Image.Image):
            return candidate
    raise TypeError(f"Unsupported HyperWeave image argument: {type(value).__name__}")


def _preset_updates(name: str):
    preset = HyperWeavePreset(name)
    if preset == HyperWeavePreset.CUSTOM:
        preset = HyperWeavePreset.OVERDRAW
    values = PRESETS[preset]
    return (
        values.overdraw_amount,
        values.structural_lock,
        values.low_frequency_lock,
        values.anchor_strength,
        values.global_overdraw_strength,
        values.face_strength,
        values.hair_strength,
        values.material_strength,
        values.micro_strength,
        values.global_candidates,
        values.face_candidates,
        values.hair_candidates,
        values.material_candidates,
        values.flat_region_detail,
    )


def _workflow_summary_html(
    enabled,
    target_mode,
    custom_long_edge,
    custom_width,
    custom_height,
    preset,
    content_profile,
    exact_steps,
    overdraw_amount,
    structural_lock,
    tile_input_size,
    core_size,
    stride,
    accumulator_mode,
    maximum_ram_gib,
) -> str:
    selected_target = str(target_mode or TargetMode.LONG_EDGE_4K.value)
    selected_preset = str(preset or HyperWeavePreset.OVERDRAW.value)
    long_edge = int(float(custom_long_edge or 0))
    width = int(float(custom_width or 0))
    height = int(float(custom_height or 0))
    tile = int(float(tile_input_size or 0))
    core = int(float(core_size or 0))
    tile_stride = int(float(stride or 0))

    target_label = selected_target
    if selected_target == TargetMode.CUSTOM_LONG_EDGE.value:
        target_label = f"長辺 {long_edge} px"
    elif selected_target == TargetMode.CUSTOM_SIZE.value:
        target_label = f"{width} × {height} px"

    status = "準備完了"
    tone = "ready"
    note = "入力を設計図として保持し、中～高周波の意味的ディテールを再作画します。"
    if not bool(enabled):
        status = "無効"
        tone = "caution"
        note = "HyperWeaveを使うには「Enable HyperWeave」をONにしてください。"
    elif (
        selected_target == TargetMode.CUSTOM_SIZE.value
        and (width <= 0 or height <= 0)
    ):
        status = "サイズ要確認"
        tone = "caution"
        note = "Custom width and heightでは幅と高さを両方1以上にしてください。"
    elif selected_target == TargetMode.LONG_EDGE_8K.value:
        status = "大判・高負荷"
        tone = "caution"
        note = "8Kは処理時間と一時領域が大きくなります。まず4Kで設定を確認してください。"
    elif selected_preset == HyperWeavePreset.MAX_OVERDRAW.value:
        status = "強い再作画"
        tone = "experimental"
        note = "Max Overdrawは構図逸脱や新規ディテールが増えるため、通常presetと比較してください。"

    return workflow_summary(
        f"{selected_preset} · {target_label}",
        (
            ("内容", str(content_profile)),
            ("Steps", f"{int(float(exact_steps or 0))} exact"),
            (
                "Lock",
                f"structure {float(structural_lock or 0):.2f} / overdraw {float(overdraw_amount or 0):.2f}",
            ),
            ("Tile", f"{tile} / core {core} / stride {tile_stride}"),
            (
                "Accumulator",
                f"{accumulator_mode} · RAM上限 {float(maximum_ram_gib or 0):g} GiB",
            ),
        ),
        status=status,
        note=note,
        tone=tone,
    )


def _target_visibility_updates(target_mode):
    selected_target = str(target_mode)
    return (
        gr.update(
            visible=gradio_compat.keep_hidden_component_mounted(
                selected_target == TargetMode.CUSTOM_LONG_EDGE.value
            )
        ),
        gr.update(
            visible=gradio_compat.keep_hidden_component_mounted(
                selected_target == TargetMode.CUSTOM_SIZE.value
            )
        ),
        gr.update(
            visible=gradio_compat.keep_hidden_component_mounted(
                selected_target == TargetMode.CUSTOM_SIZE.value
            )
        ),
    )


def _preset_updates_with_summary(
    preset,
    enabled,
    target_mode,
    custom_long_edge,
    custom_width,
    custom_height,
    content_profile,
    exact_steps,
    tile_input_size,
    core_size,
    stride,
    accumulator_mode,
    maximum_ram_gib,
):
    values = _preset_updates(str(preset))
    summary = _workflow_summary_html(
        enabled,
        target_mode,
        custom_long_edge,
        custom_width,
        custom_height,
        preset,
        content_profile,
        exact_steps,
        values[0],
        values[1],
        tile_input_size,
        core_size,
        stride,
        accumulator_mode,
        maximum_ram_gib,
    )
    return (*values, summary)


def _target_updates_with_summary(
    enabled,
    target_mode,
    custom_long_edge,
    custom_width,
    custom_height,
    preset,
    content_profile,
    exact_steps,
    overdraw_amount,
    structural_lock,
    tile_input_size,
    core_size,
    stride,
    accumulator_mode,
    maximum_ram_gib,
):
    visibility = _target_visibility_updates(target_mode)
    summary = _workflow_summary_html(
        enabled,
        target_mode,
        custom_long_edge,
        custom_width,
        custom_height,
        preset,
        content_profile,
        exact_steps,
        overdraw_amount,
        structural_lock,
        tile_input_size,
        core_size,
        stride,
        accumulator_mode,
        maximum_ram_gib,
    )
    return (*visibility, summary)


def _config_from_values(values: tuple[Any, ...]) -> HyperWeaveConfig:
    defaults = HyperWeaveConfig()
    data = {name: getattr(defaults, name) for name in UI_FIELDS}
    data.update(dict(zip(UI_FIELDS, values)))
    for field_name in (
        "identity_reference",
        "protection_mask",
        "boost_mask",
        "manual_face_mask",
    ):
        data[field_name] = _normalize_script_image(data[field_name])
    data["target_mode"] = TargetMode(data["target_mode"])
    data["preset"] = HyperWeavePreset(data["preset"])
    data["content_profile"] = ContentProfile(data["content_profile"])
    data["accumulator_mode"] = AccumulatorMode(data["accumulator_mode"])
    integer_fields = (
        "custom_long_edge",
        "custom_width",
        "custom_height",
        "seed",
        "exact_steps",
        "tile_input_size",
        "core_size",
        "context_size",
        "stride",
        "global_candidates",
        "face_candidates",
        "hair_candidates",
        "material_candidates",
        "roi_final_pass_count",
        "minimum_face_size",
        "maximum_face_count",
        "spatial_decision_size",
        "spatial_transition_width",
        "spatial_minimum_component_cells",
        "back_projection_iterations",
    )
    for name in integer_fields:
        data[name] = int(data[name])
    return HyperWeaveConfig(**data)


class HyperWeaveScript(scripts.Script):
    def title(self):
        return "HyperWeave 4K/8K"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        default = HyperWeaveConfig()
        gr.HTML(
            workflow_hero(
                "HyperWeave 4K / 8K",
                "入力を設計図として保持し、選択中の生成モデルで中～高周波の意味的ディテールを再作画します。",
                badges=("通常img2img", "生成的overdraw", "4Kから確認", f"v{HYPERWEAVE_VERSION}"),
                steps=(
                    "入力画像と納品サイズを決める",
                    "PresetとContent profileを選ぶ",
                    "顔passは必要に応じてManual Face Maskを指定",
                ),
            ),
            elem_classes=["neo-workflow-hero-host"],
        )

        gr.HTML(
            workflow_section(
                1,
                "基本プラン",
                "まず4K long edge / Overdrawで確認し、必要な場合だけ8Kへ進みます。",
            ),
            elem_classes=["neo-workflow-section-host"],
        )
        enabled = gr.Checkbox(
            label="HyperWeaveを有効化",
            value=True,
            elem_id=self.elem_id("enabled"),
        )
        with gr.Row(elem_classes=["neo-workflow-grid-2"]):
            target_mode = gr.Dropdown(
                label="納品サイズ",
                choices=[item.value for item in TargetMode],
                value=default.target_mode.value,
                elem_id=self.elem_id("target_mode"),
            )
            custom_long_edge = gr.Slider(
                label="Custom長辺",
                minimum=512,
                maximum=16384,
                step=8,
                value=4096,
                visible=gradio_compat.keep_hidden_component_mounted(False),
                elem_id=self.elem_id("custom_long_edge"),
            )
        with gr.Row(elem_classes=["neo-workflow-grid-2"]):
            custom_width = gr.Slider(
                label="Custom幅",
                minimum=0,
                maximum=16384,
                step=8,
                value=0,
                visible=gradio_compat.keep_hidden_component_mounted(False),
                elem_id=self.elem_id("custom_width"),
            )
            custom_height = gr.Slider(
                label="Custom高さ",
                minimum=0,
                maximum=16384,
                step=8,
                value=0,
                visible=gradio_compat.keep_hidden_component_mounted(False),
                elem_id=self.elem_id("custom_height"),
            )
        with gr.Row(elem_classes=["neo-workflow-grid-2"]):
            preset = gr.Dropdown(
                label="再作画Preset",
                choices=[item.value for item in HyperWeavePreset],
                value=default.preset.value,
                elem_id=self.elem_id("preset"),
            )
            content_profile = gr.Dropdown(
                label="内容Profile",
                choices=[item.value for item in ContentProfile],
                value=default.content_profile.value,
                elem_id=self.elem_id("content_profile"),
            )
        with gr.Row(elem_classes=["neo-workflow-grid-2"]):
            seed = gr.Number(
                label="HyperWeave seed (-1 = random once)",
                value=-1,
                precision=0,
                elem_id=self.elem_id("seed"),
            )
            exact_steps = gr.Slider(
                label="Exact Steps",
                minimum=1,
                maximum=30,
                step=1,
                value=6,
                elem_id=self.elem_id("exact_steps"),
            )
        with gr.Row(elem_classes=["neo-workflow-grid-3"]):
            overdraw_amount = gr.Slider(
                label="Overdraw Amount",
                minimum=0.0,
                maximum=2.0,
                step=0.05,
                value=default.overdraw_amount,
                elem_id=self.elem_id("overdraw_amount"),
            )
            structural_lock = gr.Slider(
                label="Structural Lock",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=default.structural_lock,
                elem_id=self.elem_id("structural_lock"),
            )
            low_frequency_lock = gr.Slider(
                label="Low Frequency Lock",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=default.low_frequency_lock,
                elem_id=self.elem_id("low_frequency_lock"),
            )

        with gr.Accordion(
            "詳細設定 · Tile / memory",
            open=False,
            elem_classes=["neo-workflow-accordion"],
        ):
            with gr.Row(elem_classes=["neo-workflow-grid-2"]):
                tile_input_size = gr.Slider(
                    label="Tile input size",
                    minimum=256,
                    maximum=2048,
                    step=8,
                    value=1280,
                    elem_id=self.elem_id("tile_input_size"),
                )
                core_size = gr.Slider(
                    label="Core size",
                    minimum=128,
                    maximum=1792,
                    step=8,
                    value=960,
                    elem_id=self.elem_id("core_size"),
                )
            with gr.Row(elem_classes=["neo-workflow-grid-2"]):
                context_size = gr.Slider(
                    label="Context size",
                    minimum=0,
                    maximum=512,
                    step=8,
                    value=160,
                    elem_id=self.elem_id("context_size"),
                )
                stride = gr.Slider(
                    label="Tile stride",
                    minimum=64,
                    maximum=1792,
                    step=8,
                    value=768,
                    elem_id=self.elem_id("stride"),
                )
            with gr.Row(elem_classes=["neo-workflow-grid-3"]):
                accumulator_mode = gr.Dropdown(
                    label="Accumulator mode",
                    choices=[item.value for item in AccumulatorMode],
                    value=default.accumulator_mode.value,
                    elem_id=self.elem_id("accumulator_mode"),
                )
                temp_directory = gr.Textbox(
                    label="HyperWeave temp directory",
                    value="",
                    elem_id=self.elem_id("temp_directory"),
                )
                maximum_ram_gib = gr.Slider(
                    label="Maximum RAM use (GiB)",
                    minimum=1,
                    maximum=48,
                    step=0.5,
                    value=8,
                    elem_id=self.elem_id("maximum_ram_gib"),
                )

        workflow_status = gr.HTML(
            _workflow_summary_html(
                True,
                default.target_mode.value,
                4096,
                0,
                0,
                default.preset.value,
                default.content_profile.value,
                6,
                default.overdraw_amount,
                default.structural_lock,
                1280,
                960,
                768,
                default.accumulator_mode.value,
                8,
            ),
            elem_classes=["neo-workflow-summary-host"],
        )

        with gr.Accordion(
            "詳細設定 · Pass strengths / candidates",
            open=False,
            elem_classes=["neo-workflow-accordion"],
        ):
            with gr.Row():
                anchor_strength = gr.Slider(
                    label="Anchor strength",
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    value=default.anchor_strength,
                )
                global_overdraw_strength = gr.Slider(
                    label="Global overdraw strength",
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    value=default.global_overdraw_strength,
                )
                face_strength = gr.Slider(
                    label="Face redraw strength",
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    value=default.face_strength,
                )
                hair_strength = gr.Slider(
                    label="Hair redraw strength",
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    value=default.hair_strength,
                )
                material_strength = gr.Slider(
                    label="Material redraw strength",
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    value=default.material_strength,
                )
                micro_strength = gr.Slider(
                    label="Micro detail strength",
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    value=default.micro_strength,
                )
            with gr.Row():
                global_candidates = gr.Slider(
                    label="Global candidate count",
                    minimum=1,
                    maximum=4,
                    step=1,
                    value=default.global_candidates,
                )
                face_candidates = gr.Slider(
                    label="Face candidate count",
                    minimum=1,
                    maximum=12,
                    step=1,
                    value=default.face_candidates,
                )
                hair_candidates = gr.Slider(
                    label="Hair candidate count",
                    minimum=1,
                    maximum=8,
                    step=1,
                    value=default.hair_candidates,
                )
                material_candidates = gr.Slider(
                    label="Material candidate count",
                    minimum=1,
                    maximum=4,
                    step=1,
                    value=default.material_candidates,
                )
                roi_final_pass_count = gr.Slider(
                    label="ROI final pass count",
                    minimum=1,
                    maximum=3,
                    step=1,
                    value=1,
                )

        with gr.Accordion("Semantic redraw and masks", open=False):
            with gr.Row():
                enable_face_redraw = gr.Checkbox(
                    label="Enable face/head redraw", value=True
                )
                enable_hair_redraw = gr.Checkbox(
                    label="Enable hair redraw", value=True
                )
                enable_material_redraw = gr.Checkbox(
                    label="Enable material redraw", value=True
                )
                enable_micro_pass = gr.Checkbox(
                    label="Enable micro pass", value=True
                )
            with gr.Row():
                detector_provider = gr.Dropdown(
                    label="Detector provider",
                    choices=[
                        "Auto (local only)",
                        "Manual ROI",
                        "OpenCV Haar (photo only)",
                    ],
                    value="Auto (local only)",
                )
                detector_model_path = gr.Textbox(
                    label="Optional Haar cascade XML path", value=""
                )
                minimum_face_size = gr.Slider(
                    label="Minimum source face size",
                    minimum=4,
                    maximum=256,
                    step=1,
                    value=12,
                )
                maximum_face_count = gr.Slider(
                    label="Maximum face count",
                    minimum=1,
                    maximum=32,
                    step=1,
                    value=12,
                )
            with gr.Row():
                identity_reference = gr.Image(
                    label="Identity Reference",
                    type="pil",
                    image_mode="RGB",
                    sources=["upload", "clipboard"],
                )
                structure_conditioner = gr.Dropdown(
                    label="Structure conditioner",
                    choices=["None"],
                    value="None",
                )
            gr.HTML(
                "<p><b>Manual Face Core Mask:</b> 顔そのものだけを塗ります。"
                "頭全体や広い背景は塗らず、context は自動拡張されます。"
                "複数人物は人物ごとに分離した連結成分を作ってください。</p>"
            )
            with gr.Row():
                protection_mask = gr.Image(
                    label="Structure Protection Mask",
                    type="pil",
                    image_mode="RGBA",
                    sources=["upload", "clipboard"],
                )
                boost_mask = gr.Image(
                    label="Overdraw Boost Mask",
                    type="pil",
                    image_mode="RGBA",
                    sources=["upload", "clipboard"],
                )
                manual_face_mask = gr.Image(
                    label="Manual Face Core Mask",
                    type="pil",
                    image_mode="RGBA",
                    sources=["upload", "clipboard"],
                )
            with gr.Row():
                mask_channel = gr.Radio(
                    label="Manual mask channel",
                    choices=["Luminance", "Alpha"],
                    value="Luminance",
                )
                boost_strength = gr.Slider(
                    label="Manual boost strength",
                    minimum=0,
                    maximum=2,
                    step=0.05,
                    value=0.75,
                )

        with gr.Accordion("Quality constraints", open=False):
            with gr.Row():
                flat_region_detail = gr.Slider(
                    label="Flat Region Detail",
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    value=default.flat_region_detail,
                )
                face_structure_tolerance = gr.Slider(
                    label="Face structure tolerance",
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    value=0.20,
                )
                hair_flow_tolerance = gr.Slider(
                    label="Hair flow tolerance",
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    value=0.30,
                )
                new_edge_tolerance = gr.Slider(
                    label="New edge tolerance",
                    minimum=0.01,
                    maximum=1,
                    step=0.01,
                    value=0.20,
                )
                color_drift_tolerance = gr.Slider(
                    label="Color drift tolerance",
                    minimum=0.01,
                    maximum=0.5,
                    step=0.01,
                    value=0.08,
                )
            with gr.Row():
                candidate_rejection_strictness = gr.Slider(
                    label="Candidate rejection strictness",
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    value=0.70,
                )
                candidate_score_margin = gr.Slider(
                    label="Candidate score margin over Anchor",
                    minimum=0,
                    maximum=0.5,
                    step=0.01,
                    value=default.candidate_score_margin,
                )
                enable_spatial_rescue = gr.Checkbox(
                    label="Spatial rescue after all whole-canvas candidates reject",
                    value=default.enable_spatial_rescue,
                )
                roi_stages = gr.Dropdown(
                    label="ROI stages",
                    choices=["Final stage only", "Last two stages", "Every stage"],
                    value="Last two stages",
                )
                back_projection_iterations = gr.Slider(
                    label="Back projection iterations",
                    minimum=0,
                    maximum=6,
                    step=1,
                    value=2,
                )
                back_projection_beta = gr.Slider(
                    label="Back projection beta",
                    minimum=0,
                    maximum=1,
                    step=0.05,
                    value=0.70,
                )
            with gr.Row():
                spatial_decision_size = gr.Slider(
                    label="Spatial rescue decision size",
                    minimum=32,
                    maximum=960,
                    step=8,
                    value=default.spatial_decision_size,
                )
                spatial_transition_width = gr.Slider(
                    label="Spatial rescue transition collar",
                    minimum=8,
                    maximum=128,
                    step=8,
                    value=default.spatial_transition_width,
                )
                spatial_score_margin = gr.Slider(
                    label="Spatial rescue score margin over Anchor",
                    minimum=0,
                    maximum=0.5,
                    step=0.01,
                    value=default.spatial_score_margin,
                )
                spatial_fragmentation_limit = gr.Slider(
                    label="Spatial rescue fragmentation limit",
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    value=default.spatial_fragmentation_limit,
                )
                spatial_minimum_component_cells = gr.Slider(
                    label="Spatial rescue minimum connected cells",
                    minimum=1,
                    maximum=8,
                    step=1,
                    value=default.spatial_minimum_component_cells,
                )

        with gr.Accordion("Prompt suffixes", open=False):
            append_prompt_suffixes = gr.Checkbox(
                label="Enable HyperWeave prompt suffixes", value=True
            )
            common_suffix = gr.Textbox(
                label="Common structure suffix", value=default.common_suffix, lines=2
            )
            anchor_suffix = gr.Textbox(
                label="Anchor suffix", value=default.anchor_suffix, lines=2
            )
            global_suffix = gr.Textbox(
                label="Global overdraw suffix", value=default.global_suffix, lines=2
            )
            face_illustration_suffix = gr.Textbox(
                label="Illustration face suffix",
                value=default.face_illustration_suffix,
                lines=2,
            )
            face_photo_suffix = gr.Textbox(
                label="Photo face suffix", value=default.face_photo_suffix, lines=2
            )
            hair_suffix = gr.Textbox(
                label="Hair suffix", value=default.hair_suffix, lines=2
            )
            material_suffix = gr.Textbox(
                label="Material suffix", value=default.material_suffix, lines=2
            )
            micro_suffix = gr.Textbox(
                label="Micro suffix", value=default.micro_suffix, lines=2
            )
            negative_suffix = gr.Textbox(
                label="HyperWeave negative suffix",
                value=default.negative_suffix,
                lines=3,
            )

        with gr.Accordion("Debug and advanced", open=False):
            with gr.Row():
                save_debug_images = gr.Checkbox(
                    label="Save debug images", value=False
                )
                save_all_candidates = gr.Checkbox(
                    label="Save all candidates", value=False
                )
                save_maps = gr.Checkbox(label="Save maps", value=False)
                save_roi_crops = gr.Checkbox(
                    label="Save ROI crops", value=False
                )
                save_metrics_json = gr.Checkbox(
                    label="Save metrics JSON", value=True
                )
                save_metrics_csv = gr.Checkbox(
                    label="Save metrics CSV", value=False
                )
            debug_output_directory = gr.Textbox(
                label="Debug output directory", value=""
            )
            with gr.Row():
                model_background = gr.Dropdown(
                    label="RGBA model background",
                    choices=["Auto edge color", "White", "Black"],
                    value="Auto edge color",
                )
                share_anchor_noise_family = gr.Checkbox(
                    label="Share Anchor/Overdraw noise family", value=True
                )
                oom_retry_smaller_tile = gr.Checkbox(
                    label="Retry once with smaller tiles after OOM", value=True
                )

        preset_outputs = [
            overdraw_amount,
            structural_lock,
            low_frequency_lock,
            anchor_strength,
            global_overdraw_strength,
            face_strength,
            hair_strength,
            material_strength,
            micro_strength,
            global_candidates,
            face_candidates,
            hair_candidates,
            material_candidates,
            flat_region_detail,
        ]
        summary_inputs = [
            enabled,
            target_mode,
            custom_long_edge,
            custom_width,
            custom_height,
            preset,
            content_profile,
            exact_steps,
            overdraw_amount,
            structural_lock,
            tile_input_size,
            core_size,
            stride,
            accumulator_mode,
            maximum_ram_gib,
        ]
        preset.input(
            fn=_preset_updates_with_summary,
            inputs=[
                preset,
                enabled,
                target_mode,
                custom_long_edge,
                custom_width,
                custom_height,
                content_profile,
                exact_steps,
                tile_input_size,
                core_size,
                stride,
                accumulator_mode,
                maximum_ram_gib,
            ],
            outputs=[*preset_outputs, workflow_status],
            queue=False,
            show_progress=False,
        )
        target_mode.input(
            fn=_target_updates_with_summary,
            inputs=summary_inputs,
            outputs=[
                custom_long_edge,
                custom_width,
                custom_height,
                workflow_status,
            ],
            queue=False,
            show_progress="hidden",
        )
        for slider in (
            custom_long_edge,
            custom_width,
            custom_height,
            exact_steps,
            overdraw_amount,
            structural_lock,
            tile_input_size,
            core_size,
            stride,
            maximum_ram_gib,
        ):
            slider.input(
                fn=_workflow_summary_html,
                inputs=summary_inputs,
                outputs=[workflow_status],
                queue=False,
                show_progress="hidden",
            )
        for control in (
            enabled,
            content_profile,
            accumulator_mode,
        ):
            control.input(
                fn=_workflow_summary_html,
                inputs=summary_inputs,
                outputs=[workflow_status],
                queue=False,
                show_progress="hidden",
            )

        controls = locals()
        result = [controls[name] for name in UI_FIELDS]
        self.infotext_fields = [
            (enabled, "HyperWeave enabled"),
            (target_mode, "HyperWeave target mode"),
            (preset, "HyperWeave preset"),
            (content_profile, "HyperWeave content profile"),
            (exact_steps, "HyperWeave Exact Steps"),
            (overdraw_amount, "HyperWeave Overdraw Amount"),
            (structural_lock, "HyperWeave Structural Lock"),
        ]
        return result

    @staticmethod
    def _progress_callback(event: ProgressEvent) -> None:
        if event.phase == "Plan":
            if state.job_no <= 0:
                state.job_count = max(1, event.total)
            else:
                state.job_count = max(state.job_count, state.job_no + event.total)
        stage = event.stage_index + 1
        state.job = (
            f"HyperWeave Stage {stage}/{event.stage_count}: {event.phase}"
        )
        counter = (
            f" {event.current}/{event.total}" if event.total else ""
        )
        state.textinfo = (
            f"HyperWeave Stage {stage}/{event.stage_count}: "
            f"{event.phase}{counter}"
            + (f" | {event.message}" if event.message else "")
        )

    @staticmethod
    def _interrupted() -> bool:
        return bool(
            state.interrupted or state.skipped or state.stopping_generation
        )

    def _run_one(
        self,
        p,
        source: Image.Image,
        config: HyperWeaveConfig,
        *,
        index: int,
    ):
        adapter = ForgeGeneratorAdapter(p)
        destination = (
            config.debug_output_directory
            or str(Path(p.outpath_samples) / "hyperweave_debug")
        )
        engine = HyperWeaveEngine(
            config,
            adapter,
            progress=self._progress_callback,
            interrupted=self._interrupted,
        )
        try:
            return engine.run(
                source,
                debug_stem=f"hyperweave_{index:02d}",
                debug_destination=destination,
            )
        except RuntimeError as exc:
            is_oom = "out of memory" in str(exc).lower()
            if not (
                is_oom
                and config.oom_retry_smaller_tile
                and config.tile_input_size > 1024
            ):
                raise
            logger.warning(
                "HyperWeave OOM: retrying once with 1024/768/128/640 tile geometry."
            )
            devices.torch_gc()
            retry = replace(
                config,
                tile_input_size=1024,
                core_size=768,
                context_size=128,
                stride=640,
                oom_retry_smaller_tile=False,
            )
            retry_engine = HyperWeaveEngine(
                retry,
                adapter,
                progress=self._progress_callback,
                interrupted=self._interrupted,
            )
            result = retry_engine.run(
                source,
                debug_stem=f"hyperweave_{index:02d}_oom_retry",
                debug_destination=destination,
            )
            result.messages.append(
                "OOM recovery changed tile geometry to 1024/768/128/640."
            )
            result.metadata["oom_retry_tile"] = [1024, 768, 128, 640]
            return result

    def run(self, p, *values):
        config = _config_from_values(values)
        if not config.enabled:
            return processing.process_images(p)
        if not getattr(p, "init_images", None):
            raise ValueError("HyperWeave requires at least one img2img input image.")

        sources = [image.copy() for image in p.init_images]
        concrete_seed = resolve_seed(config.seed)
        config = replace(config, seed=concrete_seed)
        snapshot = ProcessingSnapshot(p)
        original_save_samples = p.save_samples()
        original_prompt = p.prompt
        original_negative_prompt = p.negative_prompt
        original_subseed = getattr(p, "subseed", 0)
        results = []
        try:
            # One internal process_images call advances one job. A conservative count
            # keeps progress monotonic even when candidates reject early.
            state.job_count = max(state.job_count, 1)
            for index, source in enumerate(sources):
                results.append(
                    self._run_one(p, source, config, index=index)
                )
        except HyperWeaveInterrupted:
            raise
        finally:
            snapshot.restore(p)

        final_snapshot = ProcessingSnapshot(p)
        try:
            final_images: list[Image.Image] = []
            infotexts: list[str] = []
            target_sizes: list[tuple[int, int]] = []
            for result, source in zip(results, sources):
                target = resolve_target_size(source.size, config)
                target_sizes.append(target)
                extra = dict(getattr(p, "extra_generation_params", {}) or {})
                extra.update(
                    {
                        "Script": self.title(),
                        "HyperWeave enabled": True,
                        "HyperWeave version": HYPERWEAVE_VERSION,
                        "HyperWeave preset": config.preset.value,
                        "HyperWeave target mode": config.target_mode.value,
                        "HyperWeave target": f"{target[0]}x{target[1]}",
                        "HyperWeave seed": result.resolved_seed,
                        "HyperWeave Exact Steps": config.exact_steps,
                        "HyperWeave Overdraw Amount": config.overdraw_amount,
                        "HyperWeave Structural Lock": config.structural_lock,
                        "HyperWeave Low Frequency Lock": config.low_frequency_lock,
                        "HyperWeave candidates": (
                            f"G{config.global_candidates}/F{config.face_candidates}/"
                            f"H{config.hair_candidates}/M{config.material_candidates}"
                        ),
                        "HyperWeave tile": (
                            f"{config.tile_input_size}/{config.core_size}/"
                            f"{config.context_size}/{config.stride}"
                        ),
                        "HyperWeave detector": result.metadata[
                            "detector_provider"
                        ],
                        "HyperWeave stage plan": " -> ".join(
                            f"{item['target'][0]}x{item['target'][1]}"
                            for item in result.metadata["stage_plan"]
                        ),
                        "HyperWeave memmap": result.metadata["memmap_usage"],
                        "HyperWeave time": round(
                            float(result.metadata["processing_time_seconds"]), 3
                        ),
                    }
                )
                p.width, p.height = target
                p.steps = config.exact_steps
                p.seed = result.resolved_seed
                p.subseed = original_subseed
                p.prompt = original_prompt
                p.negative_prompt = original_negative_prompt
                prompt = (
                    original_prompt[0]
                    if isinstance(original_prompt, list)
                    else original_prompt
                )
                negative = (
                    original_negative_prompt[0]
                    if isinstance(original_negative_prompt, list)
                    else original_negative_prompt
                )
                p.all_prompts = [prompt]
                p.all_negative_prompts = [negative]
                p.all_seeds = [result.resolved_seed]
                p.all_subseeds = [int(original_subseed or 0)]
                p.main_prompt = prompt
                p.main_negative_prompt = negative
                p.extra_generation_params = extra
                info = processing.create_infotext(
                    p,
                    p.all_prompts,
                    p.all_seeds,
                    p.all_subseeds,
                    comments=[],
                )
                result.image.info["parameters"] = info
                result.image.info["hyperweave"] = json.dumps(
                    result.metadata,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                    allow_nan=False,
                )
                final_images.append(result.image)
                infotexts.append(info)
                if original_save_samples:
                    images.save_image(
                        result.image,
                        p.outpath_samples,
                        "",
                        result.resolved_seed,
                        prompt,
                        opts.samples_format,
                        info=info,
                        p=p,
                        existing_info={
                            "hyperweave": result.image.info["hyperweave"]
                        },
                    )

            last = results[-1].last_processed
            if last is None:
                last = processing.Processed(
                    p,
                    final_images,
                    seed=results[0].resolved_seed,
                    info=infotexts[0],
                    infotexts=infotexts,
                )
            last.images = final_images
            last.info = infotexts[0]
            last.infotexts = infotexts
            last.seed = results[0].resolved_seed
            last.subseed = int(original_subseed or 0)
            last.all_seeds = [item.resolved_seed for item in results]
            last.all_subseeds = [int(original_subseed or 0)] * len(results)
            last.all_prompts = [
                original_prompt[0]
                if isinstance(original_prompt, list)
                else original_prompt
            ] * len(results)
            last.all_negative_prompts = [
                original_negative_prompt[0]
                if isinstance(original_negative_prompt, list)
                else original_negative_prompt
            ] * len(results)
            last.width, last.height = target_sizes[0]
            last.steps = config.exact_steps
            last.extra_generation_params = p.extra_generation_params
            return last
        finally:
            final_snapshot.restore(p)
