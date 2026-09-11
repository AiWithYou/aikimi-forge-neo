"""Extrasの実行予定と結果を表示し、既存の操作項目へ接続する。"""

import html
import json

import gradio as gr

from modules.grain_selection import make_reference_preview, select_reference

ENABLE_KEYS = {
    "Upscale": "upscale_enabled",
    "Grain Cleaner": "enable",
    "Color Flatten": "enable",
    "Color Mura Checker": "color_mura_enabled",
}


def source_size(image):
    if image is None:
        return None
    return image.size[::-1] if image.getexif().get(274) in (5, 6, 7, 8) else image.size


def planned_size(size, values):
    if size is None or not values["upscale_enabled"] or values["upscaler_1_name"] in (None, "None"):
        return size
    width, height = size
    crop = values["upscale_mode"] == 1 and values["upscale_crop"]
    if values["upscale_mode"] == 1:
        scale = max(values["upscale_to_width"] / width, values["upscale_to_height"] / height)
    else:
        scale = values["upscale_by"]
        limit = values["max_side_length"]
        if limit and max(size) * scale > limit:
            target = (
                (int(limit * width / height), int(limit))
                if height > width
                else (int(limit), int(limit * height / width))
            )
            scale = max(target[0] / width, target[1] / height)
    if crop:
        return int(values["upscale_to_width"]), int(values["upscale_to_height"])
    return round(width * scale / 8) * 8, round(height * scale / 8) * 8


def plan_html(size, tab, operations):
    names = []
    for name, values in operations:
        if name not in ENABLE_KEYS:
            names.append(f"{name}（追加処理・各項目の設定を確認）")
            continue
        if not values[ENABLE_KEYS[name]]:
            continue
        if name == "Upscale":
            if values["upscaler_1_name"] in (None, "None"):
                continue
            size = planned_size(size, values)
        elif name == "Color Mura Checker" and not (values["color_mura_outputs"] or values["color_mura_add_metrics"]):
            continue
        names.append(name)
    steps = " → ".join(names) if names else "画像をそのまま保存"
    if tab == 0:
        dimensions = f"予定サイズ: {size[0]} × {size[1]} px" if size else "画像を読み込むと予定サイズを表示します"
    else:
        dimensions = "予定サイズ: 入力画像ごとに計算します" if tab in (1, 2) else "動画のフレームごとに処理します"
    return f'<div role="status"><strong>実行する処理</strong><br>{html.escape(steps)}<br>{dimensions}</div>'


def result_html(info):
    ordinary = {key: value for key, value in info.items() if key != "Grain Cleaner"}
    text = ", ".join(f"{key}: {value}" for key, value in ordinary.items())
    rendered = f"<p>{html.escape(text)}</p>" if text else ""
    if "Grain Cleaner" not in info:
        return rendered
    report = json.loads(info["Grain Cleaner"])
    statuses = {
        "APPLIED": "処理済み",
        "UNCHANGED": "無変更",
        "LOW_ACTIVITY": "粒状成分が弱いため無変更",
        "INSUFFICIENT_REFERENCE": "参考範囲が不足したため無変更",
    }
    summary = f"Grain Cleaner: {statuses[report['status']]}"
    if "unchanged_reason" in report:
        summary += f"（{report['unchanged_reason']}）"
    percent = 100 * report.get("changed_fraction", 0)
    changed_text = "0.01%未満" if 0 < percent < 0.01 else f"{percent:.2f}%"
    summary += f" ／ 変更画素 {changed_text} ／ {report['elapsed_seconds']:.2f}秒"
    if report.get("analysis_cache_hit"):
        summary += " ／ 解析を再利用"
    rendered += f'<p role="status">{html.escape(summary)}</p>'
    if "diagnostics_omitted" in report:
        rendered += f"<p>{html.escape(report['diagnostics_omitted'])}</p>"
    rendered += "<details><summary>Grain Cleanerの詳細JSON</summary><pre>"
    return rendered + html.escape(json.dumps(report, ensure_ascii=False, indent=2)) + "</pre></details>"


def bind_workflow(runner, image, tab_index, plan, solo_button):
    ordered = runner.scripts_in_preferred_order()
    controls = [component for script in ordered for component in script.controls.values()]
    dimensions = gr.State(None)

    def update_plan(size, tab, *values):
        operations = []
        offset = 0
        for script in ordered:
            keys = list(script.controls)
            operations.append((script.name, dict(zip(keys, values[offset : offset + len(keys)], strict=True))))
            offset += len(keys)
        return plan_html(size, tab, operations)

    # マスクや画像をスライダー操作のたびに再転送しない。
    summary_controls = []
    summary_defaults = []
    needed = set()
    for script in ordered:
        if script.name == "Upscale":
            needed.update(script.controls.values())
        elif script.name in ENABLE_KEYS:
            needed.add(script.controls[ENABLE_KEYS[script.name]])
            if script.name == "Color Mura Checker":
                needed.update(script.controls[key] for key in ("color_mura_outputs", "color_mura_add_metrics"))
    for component in controls:
        if component not in needed:
            summary_defaults.append(None)
        else:
            summary_defaults.append(len(summary_controls))
            summary_controls.append(component)

    def update_summary(size, tab, *values):
        return update_plan(size, tab, *(None if index is None else values[index] for index in summary_defaults))

    def image_summary(source, tab, *values):
        size = source_size(source)
        return size, update_summary(size, tab, *values)

    gr.on(
        fn=update_summary,
        inputs=[dimensions, tab_index, *summary_controls],
        outputs=[plan],
        queue=False,
        show_progress="hidden",
        trigger_mode="always_last",
    )
    plan.value = update_summary(None, 0, *(c.value for c in summary_controls))
    by_name = {script.name: script for script in ordered}
    if "Grain Cleaner" not in by_name:
        solo_button.visible = False

        def refresh_without_grain(source, tab, *values):
            runner.image_changed()
            return image_summary(source, tab, *values)

        image.change(
            refresh_without_grain,
            inputs=[image, tab_index, *summary_controls],
            outputs=[dimensions, plan],
            queue=False,
            show_progress="hidden",
        )
        return
    known = [script for script in ordered if script.name in ENABLE_KEYS]
    solo_button.interactive = len(known) == len(ordered)
    solo_button.click(
        lambda: tuple(script.name == "Grain Cleaner" for script in known),
        outputs=[script.controls[ENABLE_KEYS[script.name]] for script in known],
        queue=False,
        show_progress="hidden",
    )
    grain = by_name["Grain Cleaner"]
    outputs = [
        grain.reference_preview,
        grain.reference_master,
        grain.reference_size,
        grain.reference_corner,
        grain.reference_note,
        grain.controls["sample_roi"],
    ]

    def refresh_reference(source, mode):
        if mode != "sample":
            return None, None, None, None, "見本範囲モードで画像から選択できます。", ""
        return make_reference_preview(source)

    def refresh_source(source, mode, tab, *values):
        runner.image_changed()
        return (*image_summary(source, tab, *values), *refresh_reference(source, mode))

    image.change(
        refresh_source,
        inputs=[image, grain.controls["mode"], tab_index, *summary_controls],
        outputs=[dimensions, plan, *outputs],
        queue=False,
        show_progress="hidden",
    )

    def open_reference(source, mode, master):
        if mode != "sample" or master is not None:
            return tuple(gr.skip() for _ in outputs)
        return make_reference_preview(source)

    grain.controls["mode"].change(
        open_reference,
        inputs=[image, grain.controls["mode"], grain.reference_master],
        outputs=outputs,
        queue=False,
        show_progress="hidden",
    )
    upscale = by_name["Upscale"].controls["upscale_enabled"] if "Upscale" in by_name else gr.State(False)

    def selection_tab(tab):
        message = (
            "画像の対角2点を選択してください。" if tab == 0 else "バッチ・動画では処理後の座標を入力してください。"
        )
        return gr.update(visible=tab == 0), None, "", message

    tab_index.change(
        selection_tab,
        inputs=[tab_index],
        outputs=[grain.reference_preview, grain.reference_corner, grain.controls["sample_roi"], grain.reference_note],
        queue=False,
        show_progress="hidden",
    )
    if "Upscale" in by_name:
        geometry = [
            by_name["Upscale"].controls[key]
            for key in (
                "upscale_enabled",
                "upscale_mode",
                "upscale_by",
                "max_side_length",
                "upscale_to_width",
                "upscale_to_height",
                "upscale_crop",
                "upscaler_1_name",
            )
        ]
        for control in geometry:
            control.change(
                fn=lambda: (
                    "",
                    None,
                    "拡大設定が変わったため見本範囲を解除しました。再選択または処理後座標を入力してください。",
                ),
                outputs=[grain.controls["sample_roi"], grain.reference_corner, grain.reference_note],
                queue=False,
                show_progress="hidden",
            )
    grain.reference_preview.select(
        select_reference,
        inputs=[grain.reference_master, grain.reference_size, grain.reference_corner, upscale],
        outputs=[grain.reference_preview, grain.reference_corner, grain.controls["sample_roi"], grain.reference_note],
        queue=True,
        concurrency_limit=1,
        trigger_mode="multiple",
        show_progress="hidden",
    )
