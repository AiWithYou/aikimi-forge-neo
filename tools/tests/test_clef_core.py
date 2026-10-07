"""The decision contract must remain independent of torch/model downloads."""

import csv
import io
import json

import pytest

from modules_forge.clef.core import (
    PROFILES,
    TEMPLATES,
    ClefError,
    canonical_hash,
    compare_runs,
    edit_question,
    export_csv,
    fingerprint,
    parse_schema,
    summary,
    validate_request,
)


def test_profiles_preserve_normal_model_and_flash_identity():
    assert PROFILES["flash-int8"]["model"] == "clef-flash"
    assert PROFILES["flash-int8"]["precision"] == "int8"
    assert PROFILES["clef-24gb"]["model"] == PROFILES["clef-16gb"]["model"] == "clef"
    assert PROFILES["clef-24gb"]["precision"] == PROFILES["clef-16gb"]["precision"] == "nf4"
    assert PROFILES["clef-16gb"]["cpu_embeddings"]
    assert not PROFILES["clef-24gb"]["cpu_embeddings"]


def test_schema_round_trip_and_reject_duplicate_json_ids():
    schema = TEMPLATES["画像の評価"]
    assert parse_schema(json.dumps(schema, ensure_ascii=False)) == schema
    with pytest.raises(ClefError, match="重複"):
        parse_schema('{"a":{"type":"noul"},"a":{"type":"noul"}}')


@pytest.mark.parametrize(
    "schema",
    [
        {},
        {"q": {"type": "chat"}},
        {"q": {"type": "choice", "criteria": ["one", "two"]}},
        {"q": {"type": "score", "criteria": ["only"]}},
        {"q": {"type": "noul", "criteria": {"maybe": "unknown"}}},
        {"q": {"type": "choice", "criteria": {"a": "one"}}},
        {"q": {"type": "noul", "instructions": 12}},
    ],
)
def test_bad_schema_is_rejected_before_gpu(schema):
    with pytest.raises(ClefError):
        parse_schema(json.dumps(schema))


def test_editor_uses_one_question_and_keeps_other_items():
    base = {"other": {"type": "noul", "instructions": "other"}}
    result = edit_question(base, "", "style", "choice", "画像の種類", "photo: 写真\nart: イラスト")
    assert result["style"]["criteria"] == {"photo": "写真", "art": "イラスト"}
    assert result["other"] == base["other"]
    assert base.keys() == {"other"}
    result = edit_question(result, "style", "quality", "score", "品質", "破綻\n許容\n良好")
    assert "style" not in result
    assert result["quality"]["criteria"] == ["破綻", "許容", "良好"]
    with pytest.raises(ClefError, match="重複"):
        edit_question(result, "", "other", "noul", "another", "")
    with pytest.raises(ClefError, match="重複"):
        edit_question(base, "", "s", "choice", "style", "a: one\na: two")


def run(name="same.png", sha="abc", profile="flash-int8"):
    return {
        "request": {
            "profile": profile,
            "state": "evaluate",
            "questions": {"kind": {"type": "choice", "criteria": {"a": "A", "b": "B"}}},
            "max_pixels": 262144,
        },
        "items": [
            {
                "name": name,
                "sha256": sha,
                "status": "done",
                "answers": {
                    "kind": {"type": "choice", "choice": "a", "confidence": 0.6, "probabilities": {"a": 0.6, "b": 0.4}}
                },
            }
        ],
    }


def test_comparison_matches_image_content_and_keeps_all_probabilities():
    left, right = run("one.png"), run("renamed.png", profile="clef-24gb")
    right["items"][0]["answers"]["kind"].update(choice="b", confidence=0.8, probabilities={"a": 0.2, "b": 0.8})
    diff = compare_runs(left, right)
    assert len(diff) == 1 and diff[0]["sha256"] == "abc"
    assert diff[0]["changed"] and diff[0]["max_probability_delta"] == pytest.approx(0.4)
    assert compare_runs(left, run(sha="different")) == []


def test_comparison_does_not_reuse_one_duplicate_as_two_corresponding_records():
    import copy

    left, right = run(), run()
    left["items"].append(copy.deepcopy(left["items"][0]))
    assert len(compare_runs(left, right)) == 1
    left["items"][0]["kind"] = "record"
    assert len(compare_runs(left, right)) == 1  # Only the image copy can match.


@pytest.mark.parametrize(
    "field,value", [("state", "different"), ("max_pixels", 65536), ("questions", {"q": {"type": "noul"}})]
)
def test_comparison_refuses_changed_conditions(field, value):
    a, b = run(), run()
    b["request"][field] = value
    with pytest.raises(ClefError, match="一致"):
        compare_runs(a, b)


def test_comparison_refuses_different_image_preprocessing():
    a, b = run(), run()
    a["preprocessing"] = "exif-rgb-v1"
    b["preprocessing"] = "exif-rgb-size-v2"
    with pytest.raises(ClefError, match="前処理"):
        compare_runs(a, b)


def test_fingerprint_detects_conditions_and_images_but_not_filename():
    a, b = run(), run("renamed.png")
    assert fingerprint(a["request"], a["items"]) == fingerprint(b["request"], b["items"])
    b["request"]["profile"] = "clef-24gb"
    assert fingerprint(a["request"], a["items"]) != fingerprint(b["request"], b["items"])
    b = run(sha="other")
    assert fingerprint(a["request"], a["items"]) != fingerprint(b["request"], b["items"])
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})


def test_output_labels_preserve_true_probability_and_score_scale():
    assert "P(真)" in summary({"type": "noul", "noul": 0.82}, {"type": "noul"})
    text = summary(
        {"type": "score", "score": 1.3, "confidence": 0.7, "probabilities": {"0": 0.1, "1": 0.5, "2": 0.4}},
        {"type": "score", "criteria": ["悪い", "普通", "良い"]},
    )
    assert "0–2" in text and "1.30" in text and "100" not in text


def test_export_preserves_partial_rows_and_neutralizes_spreadsheet_formulas():
    data = run("=danger.png")
    data["items"].append({"name": "未処理.png", "sha256": "d", "status": "pending"})
    rows = list(csv.DictReader(io.StringIO(export_csv(data))))
    assert rows[0]["name"] == "'=danger.png"
    assert rows[1]["status"] == "pending" and rows[1]["answer:kind"] == ""


def test_csv_question_ids_do_not_replace_metadata_or_other_answers():
    data = run("original.png")
    data["request"]["questions"] = {key: {"type": "noul"} for key in ("name", "name_confidence")}
    data["items"][0]["answers"] = {
        "name": {"type": "noul", "noul": 0.8},
        "name_confidence": {"type": "noul", "noul": 0.4},
    }
    row = next(csv.DictReader(io.StringIO(export_csv(data))))
    assert row["name"] == "original.png"
    assert row["answer:name"] == "0.8"
    assert row["answer:name_confidence"] == "0.4"


def test_request_requires_installed_profile_and_finite_limits():
    with pytest.raises(ClefError):
        validate_request({"profile": "unknown", "questions": TEMPLATES["画像の評価"], "state": "x"})
    with pytest.raises(ClefError):
        validate_request(
            {"profile": "flash-int8", "questions": TEMPLATES["画像の評価"], "state": "x", "max_pixels": float("nan")}
        )
