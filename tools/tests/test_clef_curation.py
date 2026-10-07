"""Model filters and human review remain separate, including during export."""

import copy
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL import Image

from modules_forge.clef.core import ClefError, atomic_json, sha256
from modules_forge.clef.curation import condition_probability, copy_accepted, matching_ids, read_review, set_decision


def run_fixture(tmp_path):
    directory = tmp_path / "run-1"
    (directory / "inputs").mkdir(parents=True)
    items = []
    for i in range(2):
        path = directory / "inputs" / f"{i}.png"
        Image.new("RGB", (12, 16), "blue").save(path)
        items.append(
            {
                "id": str(i),
                "kind": "image",
                "name": "same.png",
                "path": str(path),
                "sha256": sha256(path),
                "status": "done",
                "answers": {"one": {"type": "noul", "noul": 0.8}, "two": {"type": "noul", "noul": 0.75}},
            }
        )
    return directory, {"id": directory.name, "request": {"questions": {}}, "items": items}


def test_export_download_is_tied_to_the_run_and_current_decisions(tmp_path):
    import shutil

    from modules_forge.clef.curation import latest_export

    directory, run = run_fixture(tmp_path)
    set_decision(directory, run, [run["id"], "0"], "accepted")
    exported = copy_accepted(directory, run, False)
    archive = shutil.make_archive(str(exported), "zip", exported)
    atomic_json(directory / "last-export.json", {"directory": exported.relative_to(directory).as_posix()})
    assert latest_export(directory, run, {"0": "accepted"}) == (str(exported), archive)
    assert latest_export(directory, run, {"0": "accepted", "1": "accepted"}) is None
    different = {**run, "id": "run-2"}
    assert latest_export(directory, different, {"0": "accepted"}) is None
    atomic_json(directory / "last-export.json", {"directory": "../elsewhere"})
    assert latest_export(directory, run, {"0": "accepted"}) is None


def test_condition_thresholds_do_not_multiply_or_replace_missing_answers(tmp_path):
    _, run = run_fixture(tmp_path)
    run["items"][1]["answers"].pop("two")
    run["items"].append({"id": "2", "status": "error", "answers": {"one": {"type": "noul", "noul": 1.0}}})
    rules = [{"qid": "one", "target": "true", "minimum": 0.75}, {"qid": "two", "target": "true", "minimum": 0.75}]
    assert matching_ids(run, rules, "すべて") == ["0"]  # 0.8 * 0.75 would incorrectly reject it.
    assert matching_ids(run, rules, "いずれか") == ["0"]  # Missing answers remain unjudged.


def test_choice_and_score_conditions_use_the_requested_probability_mass():
    answer = {"type": "choice", "probabilities": {"a": 0.3, "b": 0.4, "c": 0.3}}
    assert condition_probability(answer, ["a", "b"]) == pytest.approx(0.7)
    answer = {"type": "score", "score": 1.5, "probabilities": {"0": 0.2, "1": 0.1, "2": 0.7}}
    assert condition_probability(answer, "1+") == pytest.approx(0.8)
    assert condition_probability({"type": "noul", "noul": 0.2}, "false") == pytest.approx(0.8)


@pytest.mark.parametrize(
    "probabilities,target,expected",
    [
        ({"0": 0.0001, "1": 0.3, "2": 0.4, "3": 0.3}, "0+", 1.0),
        ({"0": 0.0, "1": 0.2, "2": 0.3, "3": 0.4999}, "0+", 1.0),
        ({"0": 0.0, "1": 0.0001, "2": 0.5, "3": 0.5}, "1+", 1.0),
        ({"0": 0.2, "1": 0.3, "2": 0.4, "3": 0.3}, "0+", None),
    ],
)
def test_rounded_probability_mass_does_not_exclude_valid_images(probabilities, target, expected):
    assert condition_probability({"type": "score", "probabilities": probabilities}, target) == expected


def test_review_survives_worker_updates_and_uses_run_and_item_identity(tmp_path):
    directory, run = run_fixture(tmp_path)
    set_decision(directory, run, ["run-1", "1"], "accepted")
    atomic_json(directory / "result.json", run)
    assert read_review(directory, run)["decisions"]["1"] == "accepted"
    with pytest.raises(ClefError, match="選択"):
        set_decision(directory, run, ["different-run", "1"], "accepted")
    assert "0" not in read_review(directory, run)["decisions"]


def test_concurrent_review_changes_do_not_lose_other_items(tmp_path):
    directory, run = run_fixture(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda item: set_decision(directory, run, [run["id"], item], "accepted"), ["0", "1"]))
    assert len(read_review(directory, run)["decisions"]) == 2


def test_human_can_select_pending_images_while_model_loads(tmp_path):
    directory, run = run_fixture(tmp_path)
    run["items"][0]["status"] = "pending"
    set_decision(directory, run, [run["id"], "0"], "accepted")
    run["items"][0]["status"] = "done"
    atomic_json(directory / "result.json", run)
    assert read_review(directory, run)["decisions"]["0"] == "accepted"


def test_copy_preserves_bytes_and_same_names_without_source_changes(tmp_path):
    directory, run = run_fixture(tmp_path)
    for item in run["items"]:
        set_decision(directory, run, [run["id"], item["id"]], "accepted")
    target = copy_accepted(directory, run)
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert len(list(target.glob("*.png"))) == 2
    assert {row["name"] for row in manifest["items"]} == {"same.png"}
    assert all(sha256(target / row["file"]) == row["sha256"] for row in manifest["items"])
    assert all(sha256(item["path"]) == item["sha256"] for item in run["items"])
    assert copy_accepted(directory, run) != target


def test_copy_refuses_pending_or_modified_or_outside_snapshots(tmp_path):
    directory, run = run_fixture(tmp_path)
    set_decision(directory, run, [run["id"], "0"], "accepted")
    bad = copy.deepcopy(run)
    bad["items"][0]["path"] = str(tmp_path / "outside.png")
    with pytest.raises(ClefError, match="保存画像"):
        copy_accepted(directory, bad)
    bad = copy.deepcopy(run)
    bad["items"][0]["status"] = "pending"
    with pytest.raises(ClefError, match="判定済み"):
        copy_accepted(directory, bad)
    Image.new("RGB", (12, 16), "red").save(run["items"][0]["path"])
    with pytest.raises(ClefError, match="変更"):
        copy_accepted(directory, run)
    assert not list((directory / "exports").glob("selected-*"))
