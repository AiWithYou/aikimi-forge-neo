"""Clipboard additions share file imports and preserve folder input without GPU work."""

import copy

import gradio as gr
from PIL import Image

from modules_forge.clef import workspace_ui as workspace
from modules_forge.clef.collection import scan_folder


def build(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "history_for", lambda: [])
    monkeypatch.setattr(workspace, "environment_status", lambda: "test")
    monkeypatch.setattr(
        workspace.STUDIO, "start", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("No inference"))
    )
    with gr.Blocks() as app:
        workspace.build(outputs=tmp_path / "outputs")
    return app


def callback(app, name):
    return next(function for function in app.fns.values() if function.fn and function.fn.__name__ == name)


def image(path, color="blue"):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 16), color).save(path)
    return str(path)


def test_paste_is_scoped_to_the_existing_multiple_file_component(tmp_path, monkeypatch):
    app = build(tmp_path, monkeypatch)
    controls = {
        component.elem_id: component for component in app.blocks.values() if getattr(component, "elem_id", None)
    }
    assert isinstance(controls["clef-individual-images"], gr.Accordion)
    files = controls["clef-images"]
    assert isinstance(files, gr.File) and files.file_count == "multiple" and files.type == "filepath"
    paste = controls["clef-image-paste"]
    assert isinstance(paste, gr.HTML) and paste.js_on_load
    imports = callback(app, "load_files")
    assert imports.inputs[-1] is files
    assert imports.show_progress == "hidden" and imports.concurrency_limit == 1
    assert not any(isinstance(output, gr.Code) for output in imports.outputs)


def test_individual_additions_and_removal_preserve_the_folder(tmp_path, monkeypatch):
    app = build(tmp_path, monkeypatch)
    image(tmp_path / "folder" / "one.png")
    (tmp_path / "folder" / "notes.txt").write_text("skip", encoding="utf-8")
    added = image(tmp_path / "uploads" / "clipboard.png", "red")
    scanned = callback(app, "load_folder").fn(str(tmp_path / "folder"), False, [])
    base = scanned[0]
    frozen = copy.deepcopy(base)
    result = callback(app, "load_files").fn(base, [added])
    collection = result[0]
    assert [item["name"] for item in collection["items"]] == ["one.png", "clipboard.png"]
    assert [item["id"] for item in collection["items"]] == ["0", "1"]
    assert collection["root"] == base["root"] and collection["skipped"] == base["skipped"]
    assert collection["bytes"] == sum(item["bytes"] for item in collection["items"])
    assert "2件" in result[1] and len(result[2]["value"]) == 2
    assert base == frozen
    cleared = callback(app, "load_files").fn(base, [])
    assert cleared[0] == base and "1件" in cleared[1]


def test_loading_a_new_folder_keeps_individual_additions_and_reindexes(tmp_path, monkeypatch):
    app = build(tmp_path, monkeypatch)
    image(tmp_path / "folder" / "one.png")
    added = image(tmp_path / "uploads" / "clipboard.png", "red")
    result = callback(app, "load_folder").fn(str(tmp_path / "folder"), False, [added])
    assert [item["name"] for item in result[0]["items"]] == ["one.png"]
    assert [item["name"] for item in result[1]["items"]] == ["one.png", "clipboard.png"]


def test_same_path_is_not_added_twice_but_identical_images_at_other_paths_are_distinct(tmp_path, monkeypatch):
    app = build(tmp_path, monkeypatch)
    path = image(tmp_path / "folder" / "one.png")
    other = image(tmp_path / "uploads" / "same.png")
    base = scan_folder(str(tmp_path / "folder"))
    result = callback(app, "load_files").fn(base, [path, other, other])
    assert [item["name"] for item in result[0]["items"]] == ["one.png", "same.png"]
    assert [item["id"] for item in result[0]["items"]] == ["0", "1"]
