"""Folder and record inputs retain identity and never alter their sources."""

import json

import pytest
from PIL import Image

from modules_forge.clef.collection import generation_metadata, parse_records, scan_folder
from modules_forge.clef.core import ClefError, canonical_hash, sha256
from modules_forge.clef.service import snapshot_inputs


def image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 16), "blue").save(path)
    return path


def test_folder_scan_preserves_relative_names_and_explicit_recursion(tmp_path):
    image(tmp_path / "z.png")
    image(tmp_path / "nested" / "z.png")
    (tmp_path / "notes.txt").write_text("note")
    (tmp_path / "broken.png").write_text("broken")
    top = scan_folder(str(tmp_path))
    assert [item["name"] for item in top["items"]] == ["z.png"]
    assert len(top["skipped"]) == 2
    recursive = scan_folder(f'"{tmp_path}"', recursive=True)
    assert [item["name"] for item in recursive["items"]] == ["nested/z.png", "z.png"]
    assert len({item["id"] for item in recursive["items"]}) == 2


def test_scan_and_snapshot_do_not_silently_stop_at_64_images(tmp_path):
    folder = tmp_path / "source"
    for i in range(65):
        image(folder / f"{i:03d}.png")
    scanned = scan_folder(str(folder))
    assert len(scanned["items"]) == 65
    snapshots = snapshot_inputs(scanned["items"], tmp_path / "run")
    assert len(snapshots) == 65
    assert snapshots[-1]["name"] == "064.png"


@pytest.mark.parametrize("path", ["", ".", "relative/folder"])
def test_folder_path_must_be_explicit_and_absolute(path):
    with pytest.raises(ClefError, match="絶対"):
        scan_folder(path)


def test_folder_link_outside_root_is_reported(tmp_path):
    folder = tmp_path / "source"
    folder.mkdir()
    outside = image(tmp_path / "outside.png")
    try:
        (folder / "linked.png").symlink_to(outside)
    except OSError:
        pytest.skip("Windows symlink permission unavailable")
    scanned = scan_folder(str(folder))
    assert not scanned["items"] and "フォルダ外" in scanned["skipped"][0]["reason"]


def test_source_changed_after_scan_cannot_become_a_different_fixed_input(tmp_path):
    path = image(tmp_path / "source" / "one.png")
    scanned = scan_folder(str(path.parent))
    Image.new("RGB", (12, 16), "red").save(path)
    with pytest.raises(ClefError, match="変更"):
        snapshot_inputs(scanned["items"], tmp_path / "run")


def test_record_formats_are_explicit_and_have_individual_snapshots(tmp_path):
    rows = parse_records('[{"id":"A", "body":"請求が違います"}, {"body":"接続できない"}]', "array")
    snapshots = snapshot_inputs(rows, tmp_path / "run")
    assert len(snapshots) == 2 and all(item["kind"] == "record" for item in snapshots)
    assert json.loads(open(snapshots[0]["path"], encoding="utf-8").read())["state"] == rows[0]["state"]
    assert snapshots[0]["sha256"] == canonical_hash(rows[0]["state"])
    assert len(parse_records("一件目\n\n二件目", "lines")) == 2
    with pytest.raises(ClefError, match="2行"):
        parse_records('{"body":"ok"}\ninvalid', "jsonl")


def test_whole_text_keeps_newlines_and_only_splits_when_requested():
    body = "冒頭の文章。\n続く説明。\n \n別の段落。\n結び。"
    assert [item["state"] for item in parse_records(body, "text")] == [body]
    assert [item["state"] for item in parse_records(body, "lines")] == [
        "冒頭の文章。",
        "続く説明。",
        "別の段落。",
        "結び。",
    ]
    assert [item["state"] for item in parse_records(body, "paragraphs")] == [
        "冒頭の文章。\n続く説明。",
        "別の段落。\n結び。",
    ]
    assert len(parse_records(body.replace("\n", "\r\n"), "paragraphs")) == 2


@pytest.mark.parametrize(
    "value",
    [
        {"body": "任意のJSON", "nested": [1, False, None]},
        [1, {"body": "全文で1件"}],
        [],
        {},
        "",
        "文章",
        0,
        -1.5,
        False,
        None,
    ],
)
def test_any_json_value_can_be_one_record_without_automatic_array_splitting(value):
    items = parse_records(json.dumps(value, ensure_ascii=False), "json")
    assert len(items) == 1 and items[0]["state"] == value


def test_bulk_json_boundaries_are_explicit_and_invalid_data_is_rejected():
    value = '[{"id":"a"}, {"id":"b"}]'
    assert len(parse_records(value, "json")) == 1
    assert [item["name"] for item in parse_records(value, "array")] == ["a", "b"]
    assert [item["state"] for item in parse_records('null\n\nfalse\n{"id":"c"}', "jsonl")] == [None, False, {"id": "c"}]
    with pytest.raises(ClefError, match="配列"):
        parse_records('{"id":"a"}', "array")
    with pytest.raises(ClefError, match="ありません"):
        parse_records("[]", "array")
    for invalid in ('{"a":}', "NaN", "Infinity", "1e400"):
        with pytest.raises(ClefError):
            parse_records(invalid, "json")
    with pytest.raises(ClefError, match="60,000"):
        parse_records("a" * 60_001, "text")


def test_webp_comfyui_metadata_keeps_workflow_and_api_prompt_separate(tmp_path):
    prompt = {"12": {"class_type": "KSampler", "inputs": {"seed": 42, "steps": 20}}}
    workflow = {"nodes": [{"id": 12}], "links": [], "version": 0.4}
    path = tmp_path / "metadata.webp"
    exif = Image.Exif()
    exif[272] = "prompt:" + json.dumps(prompt)
    exif[271] = "workflow:" + json.dumps(workflow)
    Image.new("RGB", (12, 16)).save(path, exif=exif)
    before = sha256(path)
    metadata = generation_metadata(path)
    assert metadata["prompt"] == prompt and metadata["workflow"] == workflow
    assert metadata["settings"]["seed"] == "42"
    assert sha256(path) == before
    assert generation_metadata(image(tmp_path / "plain.png"))["workflow"] is None
