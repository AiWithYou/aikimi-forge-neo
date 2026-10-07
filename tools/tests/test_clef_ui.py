"""Result labels, schema editing and comparisons must use fixed run inputs."""

import copy

from modules_forge.clef.core import TEMPLATES, ClefError
from modules_forge.clef.ui import (
    NEW_ITEM,
    _schema_outputs,
    _selected_index,
    detail_html,
    edit_fields,
    result_rows,
    run_heading,
    schema_update,
)


def test_installation_status_marks_normal_bundle_as_shared_by_both_gpu_profiles(monkeypatch):
    import modules_forge.clef.ui as ui

    def installed(root, profile):
        if profile != "clef-24gb":
            raise ClefError("missing")
        return "bundle", {}

    monkeypatch.setattr(ui, "bundle_manifest", installed)
    assert "24GB / 16GB共用" in ui.environment_status()


def sample():
    schema = {
        "safe": {"type": "noul", "instructions": "<script>evil</script>"},
        "quality": {"type": "score", "instructions": "品質", "criteria": ["低", "高"]},
    }
    return {
        "id": "run-1",
        "request": {
            "profile": "flash-int8",
            "questions": schema,
            "state": "evaluate",
            "max_pixels": 262144,
            "max_length": 4096,
        },
        "items": [
            {
                "name": "<bad>.png",
                "sha256": "a",
                "status": "done",
                "seconds": 1.2,
                "answers": {
                    "safe": {"type": "noul", "noul": 0.8},
                    "quality": {
                        "type": "score",
                        "score": 0.7,
                        "confidence": 0.7,
                        "probabilities": {"0": 0.3, "1": 0.7},
                    },
                },
            }
        ],
    }


def test_detail_escapes_user_text_and_shows_each_probability():
    rendered = detail_html(sample(), 0)
    assert "<script>" not in rendered and "&lt;script&gt;" in rendered
    assert "P(真)" in rendered and "0–1" in rendered
    assert "0.700" in rendered and "0.300" in rendered


def test_result_heading_keeps_executed_profile_after_change():
    run = sample()
    assert "Flash INT8" in run_heading(run, run["request"])
    changed = copy.deepcopy(run["request"])
    changed["profile"] = "clef-24gb"
    assert "Flash INT8" in run_heading(run, changed)
    assert "不一致" in run_heading(run, changed)


def test_missing_uploaded_file_does_not_break_the_executed_result(tmp_path):
    run = sample()
    run["items"][0]["path"] = "original.png"
    assert "不一致" in run_heading(run, None, [str(tmp_path / "removed.png")])
    assert "Flash INT8" in run_heading(run, None)


def test_editor_round_trip_keeps_official_schema():
    schema = TEMPLATES["画像の評価"]
    fields = edit_fields(schema, "style")
    assert fields[0] == "style" and fields[1] == "choice"
    updated = schema_update(schema, "style", *fields)
    assert updated == schema


def test_results_do_not_show_pending_as_probability_zero():
    run = sample()
    run["items"].append({"name": "waiting.png", "sha256": "b", "status": "pending"})
    rows = result_rows(run)
    assert rows[1][1] == "未処理" and rows[1][2] == ""


def test_new_question_uses_nonempty_dropdown_value():
    fields = edit_fields(TEMPLATES["画像の評価"], NEW_ITEM)
    assert fields == ("", "noul", "", "")
    schema = schema_update(TEMPLATES["画像の評価"], NEW_ITEM, "object", "noul", "人物が写っている。", "")
    assert "object" in schema and "style" in schema


def test_applied_question_stays_selected_for_review():
    schema = schema_update(TEMPLATES["画像の評価"], NEW_ITEM, "object", "noul", "人物が写っている。", "")
    assert _schema_outputs(schema, "object")[1]["value"] == "object"


def test_readonly_schema_summary_shows_every_question_and_escapes_text():
    schema = schema_update(TEMPLATES["画像の評価"], NEW_ITEM, "object", "noul", "<script>人物</script>", "")
    rendered = _schema_outputs(schema, "object")[2]
    assert isinstance(rendered, str)
    assert rendered.count("<tr>") == 5  # Header and all four questions.
    assert "object" in rendered and "&lt;script&gt;人物&lt;/script&gt;" in rendered
    assert "<script>" not in rendered


def test_partial_result_refresh_keeps_the_selected_image():
    run = sample()
    run["items"].append({"name": "second.png", "sha256": "b", "status": "done"})
    assert _selected_index(run, None) == 0
    assert _selected_index(run, 1) == 1
    run["items"][0]["status"] = "pending"
    assert _selected_index(run, None) == 1
    assert _selected_index(run, 0) == 0  # An explicit pending selection remains visible.


def test_comparison_shows_units_and_direction_without_internal_json():
    from modules_forge.clef.ui import comparison_rows

    left = sample()
    right = copy.deepcopy(left)
    right["items"][0]["answers"]["safe"]["noul"] = 0.4
    right["items"][0]["answers"]["quality"].update(score=0.9, probabilities={"0": 0.1, "1": 0.9})
    rows = comparison_rows(left, right)
    assert rows[0][:3] == ["<bad>.png", "あり", 0.4]
    assert "safe: 最大差 0.4000・ΔP(真) -0.4000" in rows[0][3]
    assert "quality: 最大差 0.2000・Δ期待値 +0.2000" in rows[0][3]
    assert "max_probability_delta" not in rows[0][3]


def test_forge_defaults_cannot_restore_unopened_history_or_unsaved_editor(tmp_path, monkeypatch):
    import sys
    from types import ModuleType

    import gradio as gr

    # The real widget loader only needs these types; avoid starting Forge's GPU stack.
    components = ModuleType("modules.ui_components")
    components.InputAccordionImpl = type("InputAccordionImpl", (), {})
    components.ToolButton = type("ToolButton", (), {})
    monkeypatch.setitem(sys.modules, "modules.ui_components", components)
    from modules.ui_loadsave import UiLoadsave
    from modules_forge.clef import ui

    key = "20261006T075747-82547d54"
    monkeypatch.setattr(ui, "_history", lambda: [("saved run", key)])
    monkeypatch.setattr(ui, "environment_status", lambda: "test environment")
    monkeypatch.setattr(ui, "OUTPUTS", tmp_path / "outputs")
    with gr.Blocks() as app:
        ui.build_ui()
    loader = UiLoadsave(str(tmp_path / "ui-config.json"))
    loader.ui_settings = {
        "clef_studio/保存した判定を開く/value": key,
        "clef_studio/比較対象の判定/value": key,
        "clef_studio/質問ID/value": "old-editor",
    }
    loader.add_block(app, "clef_studio")
    labelled = {getattr(c, "label", None): c for c in app.blocks.values()}
    assert labelled["保存した判定を開く"].value is None
    assert labelled["比較対象の判定"].value is None
    assert labelled["質問ID"].value == "style"


def test_cancelled_images_do_not_count_as_completed_progress():
    from modules_forge.clef.ui import progress_label

    run = sample()
    run["items"].extend(
        [
            {"status": "cancelled"},
            {"status": "cancelled"},
            {"status": "error"},
            {"status": "pending"},
        ]
    )
    assert progress_label(run, "停止しました。") == "停止しました。 · 1/5完了 · 2停止 · 1失敗"


def test_incompatible_comparison_clears_the_previous_table():
    from modules_forge.clef.ui import comparison_view

    left = sample()
    right = copy.deepcopy(left)
    rows, note = comparison_view(left, right)
    assert rows and "比較対象 − 表示中の判定" in note
    right["request"]["questions"]["safe"]["instructions"] = "別の質問"
    rows, note = comparison_view(left, right)
    assert rows == [] and "判断項目" in note and "一致" in note
