"""Use-case views use stable displayed records, without exposing source folders."""

import copy

import gradio as gr
import pytest

from modules_forge.clef.core import ClefError
from modules_forge.clef.ui import build_conditions, gallery_token, selected_token, table_view


def run():
    return {
        "id": "run-1",
        "request": {"questions": {"q": {"type": "noul", "instructions": "条件"}}},
        "items": [
            {"id": "0", "name": "same.webp", "status": "done", "answers": {"q": {"type": "noul", "noul": 0.8}}},
            {"id": "1", "name": "same.webp", "status": "done", "answers": {"q": {"type": "noul", "noul": 0.5}}},
        ],
    }


def test_filtered_table_carries_the_original_run_and_item_id():
    data = run()
    review = {"decisions": {"1": "hold"}}
    view = table_view(data, "曖昧な順", review, "保留", [])
    assert len(view["data"]) == 1
    assert selected_token(data, view["data"][0]) == ["run-1", "1"]
    assert view["metadata"]["display_value"][0][0] == "2"
    new_run = copy.deepcopy(data)
    new_run["id"] = "run-2"
    with pytest.raises(ClefError, match="表示"):
        selected_token(new_run, view["data"][0])


def test_quick_conditions_become_individual_fixed_questions():
    schema = build_conditions("青い服を着ている。\n\n背景に文字がある。")
    assert len(schema) == 2 and all(question["type"] == "noul" for question in schema.values())
    assert "青い服" in schema["condition_1"]["instructions"]
    with pytest.raises(ClefError):
        build_conditions("")


def test_gallery_selection_rejects_an_image_from_a_previous_run():
    data = run()
    assert gallery_token(data, "C:/cache/run-1__1.jpg") == ["run-1", "1"]
    with pytest.raises(ClefError, match="表示"):
        gallery_token(data, "C:/cache/old-run__1.jpg")


def test_selected_item_cannot_survive_outside_the_visible_results():
    from modules_forge.clef.ui import visible_selection

    data = run()
    assert visible_selection(data, ["run-1", "0"], ["1"])["id"] == "1"
    assert visible_selection(data, ["run-1", "0"], []) is None
    assert visible_selection(data, ["old", "1"], ["0", "1"])["id"] == "0"


def test_app_has_working_use_case_tabs_and_all_session_values_are_excluded(tmp_path, monkeypatch):
    import modules_forge.clef.ui as ui

    monkeypatch.setattr(ui, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(ui, "_history", lambda *args: [])
    monkeypatch.setattr(ui, "environment_status", lambda: "test")
    with gr.Blocks() as app:
        ui.build_ui()
    labels = {getattr(component, "label", "") for component in app.blocks.values()}
    assert {"画像", "文章", "JSONデータ"} <= labels
    assert not {"画像の仕分け", "文章・JSON"} & labels
    assert {"フォルダの絶対パス", "判定する文章", "判定するJSONデータ", "質問セットのJSON"} <= labels
    image_template = next(
        component
        for component in app.blocks.values()
        if getattr(component, "label", "") == "テンプレートから質問を読み込む"
    )
    assert image_template.value is None
    assert any(
        isinstance(component, gr.State)
        and isinstance(component.value, dict)
        and set(component.value) == {"style", "marks", "background"}
        for component in app.blocks.values()
    )
    limits = [
        component.value
        for component in app.blocks.values()
        if getattr(component, "label", "") == "総入力トークン上限 · 超過は切り捨てずエラー"
    ]
    assert limits and all(value >= 2828 for value in limits)
    buttons = {component.value for component in app.blocks.values() if isinstance(component, gr.Button)}
    assert {
        "フォルダを読み込む",
        "条件を判断項目に適用",
        "保存条件で未完了分を再開",
        "生成設定を読む",
        "テンプレートを適用",
    } <= buttons
    assert "現在の実行に戻る" in buttons
    downloads = {
        component.value if isinstance(component.value, str) else component.label
        for component in app.blocks.values()
        if isinstance(component, gr.DownloadButton)
    }
    assert {"workflow JSONを保存", "API prompt JSONを保存"} <= downloads
    assert {"表示中の結果 JSONを保存", "表示中の一覧 CSVを保存", "採用画像 ZIPを保存"} <= downloads
    gallery = next(
        component for component in app.blocks.values() if getattr(component, "elem_id", "") == "clef-result-gallery"
    )
    assert isinstance(gallery, gr.Gallery) and gallery.columns == 4
    assert any(isinstance(component, gr.JSON) and not component.visible for component in app.blocks.values())
    assert all(
        not component.interactive
        for component in app.blocks.values()
        if isinstance(component, gr.Button) and getattr(component, "elem_id", "") == "clef-judge"
    )
    assert all(dependency["api_visibility"] == "private" for dependency in app.get_config_file()["dependencies"])
    filter_ids = {
        component._id
        for component in app.blocks.values()
        if getattr(component, "label", "")
        in {
            "表示する項目",
            "一覧の順序",
            "絞り込む判断項目",
            "対象の選択肢・段階",
            "最低確率 · 正答率ではありません",
            "複数の真偽条件",
        }
    }
    assert not any(
        event == "change" and component_id in filter_ids
        for dependency in app.get_config_file()["dependencies"]
        for component_id, event in dependency["targets"]
    )
    for function in app.fns.values():
        if function.fn and function.fn.__name__ == "poll":
            assert isinstance(function.inputs[2], gr.State)
            assert isinstance(function.inputs[-1], gr.State)
            displayed_id = function.outputs[0]._id
            writers = [
                candidate
                for candidate in app.fns.values()
                if any(output._id == displayed_id for output in candidate.outputs)
            ]
            assert len({candidate.concurrency_id for candidate in writers}) == 1
            assert all(candidate.concurrency_limit == 1 for candidate in writers)
    studio = next(
        component for component in app.blocks.values() if getattr(component, "elem_id", None) == "clef-studio"
    )

    def visit(component):
        assert component.do_not_save_to_config
        for child in getattr(component, "children", []):
            visit(child)

    visit(studio)


def test_rendering_reads_latest_result_and_table_selection_changes_page(tmp_path, monkeypatch):
    from PIL import Image

    import modules_forge.clef.workspace_ui as workspace
    from modules_forge.clef.core import TEMPLATES, sha256

    path = tmp_path / "input.png"
    Image.new("RGB", (32, 32)).save(path)
    request = {
        "profile": "flash-int8",
        "state": "evaluate",
        "questions": TEMPLATES["顔・手の見直し（参考）"],
        "max_pixels": 262144,
        "max_length": 4096,
    }
    latest = {
        "id": "20261006T135224-12345678",
        "status": "partial",
        "request": request,
        "items": [
            {
                "id": str(index),
                "name": f"{index}.png",
                "kind": "image",
                "path": str(path),
                "sha256": sha256(path),
                "status": "cancelled",
            }
            for index in range(25)
        ],
    }
    stale = copy.deepcopy(latest)
    stale["status"] = "running"
    monkeypatch.setattr(workspace, "_read_run", lambda _: copy.deepcopy(latest))
    monkeypatch.setattr(workspace, "history_for", lambda: [])
    monkeypatch.setattr(workspace, "environment_status", lambda: "test")
    with gr.Blocks() as app:
        workspace.build(outputs=tmp_path / "outputs")
    select = next(
        function.fn
        for function in app.fns.values()
        if function.fn
        and function.fn.__name__ == "select"
        and getattr(function.inputs[2], "label", "") == "補足の文章・JSON"
    )
    current = ["flash-int8", "evaluate", request["questions"], 262144, 4096, {"items": latest["items"]}]
    view = ["入力順", "すべて", None, 1]
    event = gr.SelectData(None, {"index": [24, 0], "value": "25", "row_value": [latest["id"] + ":24"]})
    output = select(stale, *current, view, event)
    assert output[0]["status"] == "partial" and output[11]["visible"]
    assert output[8] == [latest["id"], "24"] and output[18]["value"] == 3
    record_input = [*current[:-1], {"source": "json", "items": [{"kind": "record", "sha256": "other"}]}]
    record_view = select(stale, *record_input, view, event)
    assert record_view[17]["visible"] and record_view[3]["value"] == str(path)
    assert "visible" not in record_view[25]
    assert record_view[25]["elem_classes"] == [] and "不一致" in record_view[5]
    review = next(
        function.fn
        for function in app.fns.values()
        if function.fn
        and function.fn.__name__ == "review"
        and getattr(function.inputs[4], "label", "") == "補足の文章・JSON"
    )
    output = review(stale, [latest["id"], "0"], [latest["id"], "24"], *current, output[-1])
    assert output[8] == [latest["id"], "24"]
    import json

    decisions = json.loads((tmp_path / "outputs" / latest["id"] / "review.json").read_text(encoding="utf-8"))[
        "decisions"
    ]
    assert decisions == {"0": "accepted"}
    import pytest

    exporter = next(
        function for function in app.fns.values() if function.fn and function.fn.__name__ == "export_images"
    )
    assert exporter.concurrency_id == next(
        function.concurrency_id for function in app.fns.values() if function.fn is select
    )
    with pytest.raises(gr.Error, match="表示する実行が変わりました"):
        exporter.fn(latest, "another-run · old heading", True)


def test_return_to_current_clears_history_so_the_same_run_can_be_reopened(tmp_path, monkeypatch):
    import modules_forge.clef.workspace_ui as workspace

    monkeypatch.setattr(workspace, "history_for", lambda: [])
    monkeypatch.setattr(workspace, "environment_status", lambda: "test")
    with gr.Blocks() as app:
        workspace.build(outputs=tmp_path / "outputs")
    buttons = {
        component._id
        for component in app.blocks.values()
        if isinstance(component, gr.Button) and component.value == "現在の実行に戻る"
    }
    callbacks = [
        function
        for function in app.fns.values()
        if function.fn and any(target[0] in buttons for target in function.targets)
    ]
    assert len(callbacks) == 1
    for callback in callbacks:
        history = [
            output
            for output in callback.outputs
            if getattr(output, "label", "") in {"保存した判定を開く", "文章の判定記録を開く"}
        ]
        assert len(history) == 1, "Returning to current must also clear the history selection"
        result = callback.fn(None, *[component.value for component in callback.inputs[1:]])
        assert result[callback.outputs.index(history[0])]["value"] is None


def test_history_labels_read_each_saved_run_once(tmp_path, monkeypatch):
    import modules_forge.clef.ui as ui
    import modules_forge.clef.workspace_ui as workspace
    from modules_forge.clef.core import atomic_json

    identifier = "20261011T120000-12345678"
    atomic_json(
        tmp_path / identifier / "result.json",
        {"status": "complete", "request": {"profile": "flash-int8"}, "items": [{"id": "0", "kind": "record"}]},
    )
    monkeypatch.setattr(ui, "OUTPUTS", tmp_path)
    calls = []
    original = ui._read_run

    def read(key):
        calls.append(key)
        return original(key)

    monkeypatch.setattr(ui, "_read_run", read)
    monkeypatch.setattr(workspace, "_read_run", read)
    choices = workspace.history_for()
    assert len(choices) == 1 and choices[0][0].startswith("文章・JSON · ")
    assert calls == [identifier]


def test_history_is_scanned_once_at_build_and_only_on_poll_completion(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import modules_forge.clef.workspace_ui as workspace

    calls = []
    monkeypatch.setattr(workspace, "history_for", lambda: calls.append(True) or [])
    monkeypatch.setattr(workspace, "environment_status", lambda: "test")
    with gr.Blocks() as app:
        workspace.build(outputs=tmp_path / "outputs")
    assert len(calls) == 1, "History and comparison use the same initial scan"
    poll = next(function.fn for function in app.fns.values() if function.fn and function.fn.__name__ == "poll")
    data = run()
    data["status"] = "running"
    current = ["flash-int8", "", data["request"]["questions"], 262144, 4096, {"items": []}]
    info = {"done": False, "result": data, "message": "running"}
    monkeypatch.setattr(workspace.STUDIO, "status", lambda *args: info)
    arguments = [data["id"], {"id": "viewing-history"}, None, *current, ["入力順", "すべて", None, 1]]
    request = SimpleNamespace(session_hash="owner")
    calls.clear()
    poll(*arguments, request)
    poll(*arguments, request)
    assert not calls, "Running polls must not read historical runs"
    info["done"] = True
    poll(*arguments, request)
    poll(*arguments, request)
    assert len(calls) == 1, "Repeated final polls must refresh history only once"


def test_result_downloads_aggregate_the_latest_running_decisions_on_click(tmp_path, monkeypatch):
    import json

    import modules_forge.clef.workspace_ui as workspace

    monkeypatch.setattr(workspace, "history_for", lambda: [])
    monkeypatch.setattr(workspace, "environment_status", lambda: "test")
    with gr.Blocks() as app:
        workspace.build(outputs=tmp_path / "outputs")
    latest = run()
    latest["status"] = "running"
    monkeypatch.setattr(workspace, "_read_run", lambda _: copy.deepcopy(latest))
    exports = {
        function.fn.__name__: function
        for function in app.fns.values()
        if function.fn and function.fn.__name__ in {"export_json", "export_csv_file"}
    }
    assert set(exports) == {"export_json", "export_csv_file"}
    saved_json = exports["export_json"].fn({"id": latest["id"]}, latest["id"] + " · heading")
    from pathlib import Path

    assert json.loads(Path(saved_json).read_text(encoding="utf-8")) == latest
    saved_csv = exports["export_csv_file"].fn({"id": latest["id"]}, latest["id"] + " · heading")
    assert "0.8" in Path(saved_csv).read_text(encoding="utf-8-sig")
    with pytest.raises(gr.Error, match="表示する実行が変わりました"):
        exports["export_json"].fn({"id": latest["id"]}, "old-run · heading")
