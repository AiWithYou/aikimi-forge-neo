"""Mixed probability conditions keep independent values and ignore stale UI events."""

import copy

import gradio as gr

from modules_forge.clef.curation import matching_ids


def mixed_run():
    return {
        "id": "mixed-run",
        "request": {
            "questions": {
                "style": {
                    "type": "choice",
                    "instructions": "表現形式",
                    "criteria": {"photo": "写真", "illustration": "イラスト"},
                },
                "quality": {"type": "score", "instructions": "完成度", "criteria": ["0", "1", "2", "3"]},
                "text": {"type": "noul", "instructions": "文字がある"},
            }
        },
        "items": [
            {
                "id": str(index),
                "status": "done",
                "answers": {
                    "style": {"type": "choice", "probabilities": {"photo": photo, "illustration": 1 - photo}},
                    "quality": {"type": "score", "probabilities": dict(zip(map(str, range(4)), score, strict=True))},
                    "text": {"type": "noul", "noul": text},
                },
            }
            for index, (photo, score, text) in enumerate(
                [
                    (0.85, [0.05, 0.05, 0.4, 0.5], 0.9),
                    (0.2, [0.1, 0.2, 0.3, 0.4], 0.9),
                    (0.85, [0.2, 0.2, 0.3, 0.3], 0.1),
                ]
            )
        ],
    }


def test_choice_score_and_boolean_conditions_can_be_kept_together():
    from modules_forge.clef.filters import active_rules, filter_state

    run = mixed_run()
    state = filter_state(run)
    state["rows"][0].update(enabled=True, target="photo", minimum=0.7)
    state["rows"][1].update(enabled=True, target="2+", minimum=0.8)
    state = filter_state(run, state)
    assert matching_ids(run, active_rules(state), "すべて") == ["0"]
    assert matching_ids(run, active_rules(state), "いずれか") == ["0", "2"]
    state["rows"][1]["enabled"] = False
    state["rows"][2].update(enabled=True, target="false", minimum=0.8)
    state = filter_state(run, state)
    assert matching_ids(run, active_rules(state), "すべて") == ["2"]
    assert state["rows"][0]["target"] == "photo"
    assert state["rows"][1] == {"qid": "quality", "enabled": False, "target": "2+", "minimum": 0.8}


def test_photo_cannot_be_applied_to_a_score_row_or_silently_match_everything():
    from modules_forge.clef.filters import active_rules, filter_payload, filter_state

    run = mixed_run()
    state = filter_state(run)
    state["rows"][1].update(enabled=True, target="photo")
    state = filter_state(run, state)
    assert not state["rows"][1]["enabled"] and not active_rules(state)
    payload = filter_payload(run, state)
    assert [value for _, value in payload["questions"][1]["targets"]] == ["0+", "1+", "2+", "3+"]
    assert [value for _, value in payload["questions"][0]["targets"]] == ["photo", "illustration"]


def test_filter_events_ignore_older_changes_and_other_visits_to_the_same_run():
    from modules_forge.clef.filters import apply_filter_event, filter_state

    run = mixed_run()
    state = filter_state(run)
    event = copy.deepcopy(state)
    event["revision"] = 2
    event["rows"][0].update(enabled=True, target="illustration", minimum=0.9)
    updated = apply_filter_event(run, state, event)
    assert updated["rows"][0]["target"] == "illustration"
    event["revision"] = 1
    assert apply_filter_event(run, updated, event) is None
    event["revision"] = 3
    another_visit = filter_state(run)
    assert apply_filter_event(run, another_visit, event) is None
    other_run = copy.deepcopy(run)
    other_run["id"] = "another-run"
    assert apply_filter_event(other_run, updated, event) is None


def test_component_filter_script_is_not_also_loaded_globally_by_forge():
    from modules_forge.clef import workspace_ui as workspace

    extension = workspace.ROOT / "extensions-builtin/clef-studio"
    global_scripts = (extension / "javascript").glob("*.js")
    assert all(path.read_text(encoding="utf-8") != workspace.FILTER_JS for path in global_scripts), (
        "The component script requires Gradio's watch/props/trigger scope and cannot run in Forge's global head"
    )


def test_filter_callbacks_keep_controls_visible_and_do_not_start_inference(tmp_path, monkeypatch):
    from PIL import Image

    import modules_forge.clef.workspace_ui as workspace

    monkeypatch.setattr(workspace, "history_for", lambda: [])
    monkeypatch.setattr(workspace, "environment_status", lambda: "test")
    with gr.Blocks() as app:
        workspace.build(outputs=tmp_path / "outputs")
    panels = [
        component
        for component in app.blocks.values()
        if (getattr(component, "elem_id", None) or "").startswith("clef-filters-")
    ]
    assert len(panels) == 1 and all(isinstance(panel, gr.HTML) for panel in panels)
    assert all(
        dependency["show_progress"] == "hidden"
        for dependency in app.get_config_file()["dependencies"]
        if any(component_id in {panel._id for panel in panels} for component_id, _ in dependency["targets"])
    )
    callback = next(
        function
        for function in app.fns.values()
        if function.fn
        and function.fn.__name__ == "change_filters"
        and getattr(function.inputs[3], "label", "") == "補足の文章・JSON"
    )
    assert callback.trigger_mode == "multiple", "A submitted condition snapshot must not be dropped while busy"
    from modules_forge.clef.core import sha256
    from modules_forge.clef.filters import filter_state

    path = tmp_path / "image.png"
    Image.new("RGB", (16, 16)).save(path)
    run = mixed_run()
    run["status"] = "complete"
    run["request"].update(profile="flash-int8", state="evaluate", max_pixels=262144, max_length=4096)
    for index, item in enumerate(run["items"]):
        item.update(name=f"{index}.png", kind="image", path=str(path), sha256=sha256(path))
        item["answers"]["style"].update(choice="photo", confidence=0.85)
        item["answers"]["quality"]["score"] = 2.35
    monkeypatch.setattr(workspace, "_read_run", lambda _: copy.deepcopy(run))
    monkeypatch.setattr(
        workspace.STUDIO, "start", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("No inference"))
    )
    state = filter_state(run)
    event = copy.deepcopy(state)
    event["revision"] = 1
    event["rows"][0].update(enabled=True, target="photo", minimum=0.7)
    event["rows"][1].update(enabled=True, target="2+", minimum=0.8)
    current = ["flash-int8", "evaluate", run["request"]["questions"], 262144, 4096, {"items": run["items"]}]
    output = callback.fn(run, None, *current, ["入力順", "候補だけ", state, 1], gr.EventData(None, event))
    assert "候補 1" in output[9] and len(output[1]["data"]) == 1
    panel_index = next(
        index
        for index, component in enumerate(callback.outputs)
        if (getattr(component, "elem_id", None) or "").startswith("clef-filters-")
    )
    assert output[panel_index] == gr.skip(), "Editing conditions must preserve the native controls and focus"
    assert output[-1][2]["revision"] == 1
