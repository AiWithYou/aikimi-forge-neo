"""Input drafts share questions and results; execution captures fresh input boundaries."""

import copy
import json
from types import SimpleNamespace

import gradio as gr
import pytest

from modules_forge.clef import workspace_ui as workspace
from modules_forge.clef.core import TEMPLATES, ClefError


def build(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "history_for", lambda: [])
    monkeypatch.setattr(workspace, "environment_status", lambda: "test")
    with gr.Blocks() as app:
        workspace.build(outputs=tmp_path / "outputs")
    return app


def callback(app, name):
    return next(function for function in app.fns.values() if function.fn and function.fn.__name__ == name)


def test_parent_result_visibility_cannot_override_independent_image_controls(tmp_path, monkeypatch):
    app = build(tmp_path, monkeypatch)
    parents = {
        component.elem_id: component
        for component in app.blocks.values()
        if getattr(component, "elem_id", "") in {"clef-result-content", "clef-image-exports"}
    }
    assert parents.keys() == {"clef-result-content", "clef-image-exports"}
    assert all(parent.visible is True for parent in parents.values())
    assert "clef-results-empty" in parents["clef-result-content"].elem_classes
    assert "clef-image-exports-hidden" in parents["clef-image-exports"].elem_classes

    run = saved_record(tmp_path)
    monkeypatch.setattr(workspace, "_read_run", lambda _: run)
    rendered = callback(app, "return_current").fn(
        run["id"],
        "flash-int8",
        "",
        run["request"]["questions"],
        262144,
        4096,
        {"source": "json", "items": []},
        ["入力順", "すべて", None, 1],
    )
    functions = callback(app, "return_current")
    updates = {
        component.elem_id: value
        for component, value in zip(functions.outputs, rendered, strict=True)
        if getattr(component, "elem_id", "") in parents
    }
    assert "visible" not in updates["clef-result-content"]
    assert "visible" not in updates["clef-image-exports"]
    assert updates["clef-result-content"]["elem_classes"] == []
    assert updates["clef-image-exports"]["elem_classes"] == ["clef-image-exports-hidden"]


def saved_record(tmp_path):
    directory = tmp_path / "outputs" / "20261007T000000-12345678"
    directory.mkdir(parents=True)
    path = directory / "input.json"
    path.write_text(json.dumps({"state": {"body": "保存した入力"}}, ensure_ascii=False), encoding="utf-8")
    request = {
        "profile": "flash-int8",
        "state": "",
        "questions": TEMPLATES["文章・JSONの分類"],
        "max_pixels": 262144,
        "max_length": 4096,
    }
    return {
        "id": directory.name,
        "status": "complete",
        "request": request,
        "items": [
            {
                "id": "0",
                "name": "JSON 1",
                "kind": "record",
                "path": str(path),
                "sha256": "record",
                "status": "done",
                "answers": {},
            }
        ],
    }


def test_tab_routing_is_immediate_and_never_replaces_questions_or_input_drafts(tmp_path, monkeypatch):
    app = build(tmp_path, monkeypatch)
    source = next(
        component for component in app.blocks.values() if getattr(component, "elem_id", "") == "clef-input-source"
    )
    assert isinstance(source, gr.Textbox) and source.value == "image" and not source.visible
    tabs = {component._id: component.label for component in app.blocks.values() if isinstance(component, gr.Tab)}
    switches = [
        dep
        for dep in app.get_config_file()["dependencies"]
        if any(cid in tabs for cid, event in dep["targets"] if event == "select")
    ]
    assert len(switches) == 3
    assert {dep["js"] for dep in switches} == {"() => ['image']", "() => ['text']", "() => ['json']"}
    assert all(not dep["backend_fn"] and dep["outputs"] == [source._id] for dep in switches)
    assert len([fn for fn in app.fns.values() if fn.fn and fn.fn.__name__ == "submit"]) == 1
    assert len([component for component in app.blocks.values() if isinstance(component, gr.Timer)]) == 1
    schemas = [
        component
        for component in app.blocks.values()
        if isinstance(component, gr.State) and isinstance(component.value, dict) and "style" in component.value
    ]
    assert len(schemas) == 1
    modes = {
        component.label: component.value
        for component in app.blocks.values()
        if getattr(component, "label", "") in {"文章の区切り", "JSONの区切り"}
    }
    assert modes == {"文章の区切り": "text", "JSONの区切り": "json"}


def test_preview_parses_the_active_source_and_reports_errors_without_inference(tmp_path, monkeypatch):
    app = build(tmp_path, monkeypatch)
    monkeypatch.setattr(workspace.STUDIO, "start", lambda *args: pytest.fail("Tab/preview must not infer"))
    route = callback(app, "preview_input")
    image = {
        "items": [{"kind": "image", "sha256": "image", "name": "one.png"}],
        "root": "H:/source",
        "bytes": 100,
        "skipped": [],
    }
    drafts = [image, "全文\n改行", "text", '{"body":"JSON"}', "json"]
    result = route.fn("text", *drafts)
    assert [item["state"] for item in result["items"]] == ["全文\n改行"]
    assert result["source"] == "text"
    assert route.fn("image", *drafts)["items"] == image["items"]
    assert route.fn("json", *drafts)["items"][0]["state"] == {"body": "JSON"}
    invalid = route.fn("json", image, "全文", "text", '{"broken":}', "json")
    assert not invalid["items"] and "JSON" in invalid["error"]
    assert all(isinstance(output, gr.State) for output in route.outputs), (
        "Preview must not overwrite drafts, source or saved results"
    )


@pytest.mark.parametrize(
    "source,body,mode,expected",
    [
        ("text", "新しい本文\n改行", "text", ["新しい本文\n改行"]),
        ("json", '{"body":"新しいJSON"}', "json", [{"body": "新しいJSON"}]),
        ("json", "[null,false,0]", "array", [None, False, 0]),
    ],
)
def test_submit_captures_client_source_and_fresh_draft_instead_of_preview(
    tmp_path, monkeypatch, source, body, mode, expected
):
    app = build(tmp_path, monkeypatch)
    captured = []
    monkeypatch.setattr(
        workspace.STUDIO,
        "start",
        lambda request, items, owner: captured.append((request, copy.deepcopy(items), owner)) or "new-run",
    )
    monkeypatch.setattr(workspace, "_read_run", lambda _: {"id": "new-run"})
    images = {"items": [], "root": "", "skipped": [], "bytes": 0}
    text, text_mode, data, json_mode = (
        (body, mode, '{"inactive":true}', "json") if source == "text" else ("inactive", "text", body, mode)
    )
    schema = copy.deepcopy(TEMPLATES["文章・JSONの分類"])
    result = callback(app, "submit").fn(
        source,
        images,
        text,
        text_mode,
        data,
        json_mode,
        "flash-int8",
        "画像専用の補足は混ぜない",
        schema,
        262144,
        4096,
        SimpleNamespace(session_hash="owner"),
    )
    request, items, owner = captured[0]
    assert request["state"] == "" and request["questions"] == schema and owner == "owner"
    assert [item["state"] for item in items] == expected
    assert result[-1]["items"] == items and result[-1]["source"] == source
    with pytest.raises(gr.Error):
        callback(app, "submit").fn(
            "json",
            images,
            "inactive",
            "text",
            "invalid",
            "json",
            "flash-int8",
            "",
            schema,
            262144,
            4096,
            SimpleNamespace(session_hash="owner"),
        )
    assert len(captured) == 1, "Invalid current data must fail before starting a job"


def test_resolver_keeps_image_context_separate_and_rejects_unknown_routes():
    empty = {"items": [], "bytes": 0, "skipped": [], "root": ""}
    with pytest.raises(ClefError):
        workspace.resolve_input("unknown", empty, "text", "text", "{}", "json")
    assert workspace.resolve_input("json", empty, "inactive", "text", "[]", "json")["items"][0]["state"] == []


def test_record_result_is_rendered_from_its_snapshot_even_on_image_input(tmp_path, monkeypatch):
    app = build(tmp_path, monkeypatch)
    run = saved_record(tmp_path)
    monkeypatch.setattr(workspace, "_read_run", lambda _: copy.deepcopy(run))
    current = [
        "flash-int8",
        "画像の補足",
        run["request"]["questions"],
        262144,
        4096,
        {"source": "image", "items": [{"kind": "image", "sha256": "different"}]},
    ]
    event = gr.SelectData(None, {"index": [0, 0], "value": "1", "row_value": [run["id"] + ":0"]})
    select = callback(app, "select")
    output = select.fn(run, *current, ["入力順", "すべて", None, 1], event)
    assert json.loads(output[4]["value"]) == {"body": "保存した入力"}
    assert not output[17]["visible"] and "不一致" in output[5]
    for component, value in zip(select.outputs, output, strict=True):
        if getattr(component, "elem_id", "") == "clef-image-exports":
            assert "visible" not in value
            assert value["elem_classes"] == ["clef-image-exports-hidden"]


def test_resume_rejects_a_late_click_from_another_run(tmp_path, monkeypatch):
    app = build(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(workspace.STUDIO, "resume", lambda *args: calls.append(args) or "saved-run")
    monkeypatch.setattr(workspace, "_read_run", lambda _: {"id": "saved-run"})
    resume = callback(app, "resume_job")
    with pytest.raises(gr.Error, match="表示する実行が変わりました"):
        resume.fn({"id": "saved-run"}, "old-run · old heading", False, SimpleNamespace(session_hash="owner"))
    assert not calls
    result = resume.fn(
        {"id": "saved-run"}, "saved-run · 現在の入力と不一致", True, SimpleNamespace(session_hash="owner")
    )
    assert calls == [("saved-run", "owner", True)] and result[0] == "saved-run"
    assert "保存した条件" in result[2]


def test_completed_poll_does_not_enable_invalid_input_or_take_over_history(tmp_path, monkeypatch):
    app = build(tmp_path, monkeypatch)
    run = saved_record(tmp_path)
    monkeypatch.setattr(workspace, "_read_run", lambda _: copy.deepcopy(run))
    monkeypatch.setattr(
        workspace.STUDIO, "status", lambda *args: {"done": True, "result": copy.deepcopy(run), "message": "完了"}
    )
    current = [
        "flash-int8",
        "画像の補足",
        run["request"]["questions"],
        1048576,
        4096,
        {"source": "json", "items": [], "error": "JSONの構文エラー"},
    ]
    poll = callback(app, "poll")
    result = poll.fn(
        run["id"], run, None, *current, ["入力順", "すべて", None, 1], SimpleNamespace(session_hash="owner")
    )
    judge_index = next(
        index for index, output in enumerate(poll.outputs) if getattr(output, "elem_id", "") == "clef-judge"
    )
    assert not result[judge_index]["interactive"]
    result = poll.fn(
        run["id"],
        {"id": "viewing-history"},
        None,
        *current,
        ["入力順", "すべて", None, 1],
        SimpleNamespace(session_hash="owner"),
    )
    assert result[0] == gr.update(), "Background completion must not replace the displayed history"
    assert not result[judge_index]["interactive"]


def test_record_metadata_never_opens_the_json_snapshot_as_an_image(tmp_path, monkeypatch):
    app = build(tmp_path, monkeypatch)
    run = saved_record(tmp_path)
    monkeypatch.setattr(workspace, "generation_metadata", lambda *args: pytest.fail("Record input is not an image"))
    metadata = callback(app, "image_settings")
    result = metadata.fn(run, [run["id"], "0"])
    assert all(not value["interactive"] and value["value"] is None for value in result[1:])
    assert metadata.concurrency_id == callback(app, "select").concurrency_id


def test_record_request_ignores_image_only_settings_and_identity_keeps_kind_and_order():
    schema = TEMPLATES["文章・JSONの分類"]
    request = workspace.current_request("flash-int8", "画像の補足", schema, 1048576, 4096, {"source": "json"})
    assert request["state"] == "" and request["max_pixels"] == 262144
    items = [{"kind": "record", "sha256": "a"}, {"kind": "record", "sha256": "b"}]
    assert workspace.same_input({"items": items}, {"source": "text", "items": items})
    assert not workspace.same_input({"items": items}, {"items": list(reversed(items))})
    assert not workspace.same_input({"items": items}, {"items": [{"kind": "image", "sha256": "a"}, items[1]]})


def test_examples_only_fill_empty_drafts_and_template_replacement_can_be_undone(tmp_path, monkeypatch):
    app = build(tmp_path, monkeypatch)
    example = callback(app, "text_example_value")
    assert example.fn("書きかけの文章") == "書きかけの文章"
    assert "請求" in example.fn("")
    json_example = callback(app, "json_example_value")
    assert json_example.fn("null") == "null"
    assert isinstance(json.loads(json_example.fn("")), dict)
    custom = {"custom": {"type": "noul", "instructions": "自分の質問"}}
    applied = callback(app, "load_template").fn("制作メモの整理", custom)
    assert applied[4] == custom and applied[5]["value"] is None
    assert applied[0] == workspace.RECORD_TEMPLATES["制作メモの整理"]
    restored = callback(app, "undo_template").fn(applied[4])
    assert restored[0] == custom and restored[4] is None and not restored[5]["interactive"]


def test_unapplied_question_edits_are_shown_near_execution(tmp_path, monkeypatch):
    app = build(tmp_path, monkeypatch)
    schema = {"question": {"type": "choice", "instructions": "適用した質問", "criteria": {"a": "A", "b": "B"}}}
    notice = callback(app, "question_notice")
    result = notice.fn(schema, "question", "question", "choice", "まだ確定していない質問", "a: A\nb: B")
    assert result["visible"] and "未適用" in result["value"]
    result = notice.fn(schema, "question", "question", "choice", "適用した質問", "a: A\nb: B")
    assert not result["visible"]
