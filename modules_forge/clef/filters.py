"""Independent conditions for a frozen run, with revisioned UI snapshots."""

from __future__ import annotations

import math
import uuid


def question_targets(question):
    if question["type"] == "noul":
        return [("真の確率", "true"), ("偽の確率", "false")]
    if question["type"] == "choice":
        return [(label, key) for key, label in question["criteria"].items()]
    return [(f"{index}以上 · {label}", f"{index}+") for index, label in enumerate(question["criteria"])]


def filter_state(run, state=None):
    if not run:
        return None
    state = state if isinstance(state, dict) and state.get("run_id") == run["id"] else {}
    previous = {
        row["qid"]: row for row in state.get("rows", []) if isinstance(row, dict) and isinstance(row.get("qid"), str)
    }
    rows = []
    for qid, question in run["request"]["questions"].items():
        old = previous.get(qid, {})
        targets = [key for _, key in question_targets(question)]
        target = old.get("target", targets[0])
        valid = target in targets
        try:
            minimum = float(old.get("minimum", 0.7))
            if not math.isfinite(minimum) or not 0 <= minimum <= 1:
                minimum = 0.7
        except (TypeError, ValueError):
            minimum = 0.7
        rows.append(
            {
                "qid": qid,
                "enabled": old.get("enabled") is True and valid,
                "target": target if valid else targets[0],
                "minimum": minimum,
            }
        )
    revision = state.get("revision", 0)
    return {
        "run_id": run["id"],
        "panel_id": state.get("panel_id") or uuid.uuid4().hex,
        "revision": revision if type(revision) is int and revision >= 0 else 0,
        "rows": rows,
        "mode": state.get("mode") if state.get("mode") in {"すべて", "いずれか"} else "すべて",
    }


def filter_payload(run, state):
    if not run:
        return None
    return {
        **state,
        "questions": [
            {"qid": key, "label": question.get("instructions") or key, "targets": question_targets(question)}
            for key, question in run["request"]["questions"].items()
        ],
    }


def active_rules(state):
    return [
        {"qid": row["qid"], "target": row["target"], "minimum": row["minimum"]}
        for row in (state or {}).get("rows", [])
        if row["enabled"]
    ]


def apply_filter_event(run, previous, event):
    state = filter_state(run, previous)
    if not state or not isinstance(event, dict):
        return None
    revision = event.get("revision")
    if (
        event.get("run_id") != state["run_id"]
        or event.get("panel_id") != state["panel_id"]
        or type(revision) is not int
        or revision <= state["revision"]
    ):
        return None
    return filter_state(run, event)
