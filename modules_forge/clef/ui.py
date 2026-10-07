"""A shared Gradio view for the Forge tab and a local standalone preview."""

from __future__ import annotations

import html
import json
import re
from functools import lru_cache
from pathlib import Path

import gradio as gr

from .bundle import bundle_manifest
from .core import (
    OUTPUTS,
    PROFILES,
    RUNTIME,
    ClefError,
    ambiguity,
    canonical_hash,
    compare_runs,
    edit_question,
    parse_schema,
    sha256,
    summary,
    validate_request,
)
from .curation import REVIEW_LABELS, matching_ids

PRIVATE = {"api_visibility": "private"}
STATUS = {"done": "完了", "pending": "未処理", "cancelled": "停止", "error": "エラー", "failed": "失敗"}
TYPE_LABELS = [("選択肢から選ぶ", "choice"), ("低い順の段階評価", "score"), ("命題の真偽", "noul")]
NEW_ITEM = "__clef_new_question__"
COMPARISON_NOTE = "差は **比較対象 − 表示中の判定**。最大確率差は絶対値です。"


@lru_cache(maxsize=256)
def _uploaded_hash(path, size, mtime):
    return sha256(path)


def edit_fields(schema, qid):
    if qid == NEW_ITEM:
        qid = ""
    question = schema.get(qid, {"type": "noul", "instructions": ""})
    kind = question["type"]
    criteria = question.get("criteria")
    options = (
        "\n".join(f"{key}: {value}" for key, value in criteria.items())
        if kind == "choice"
        else "\n".join(criteria)
        if kind == "score"
        else ""
    )
    return qid or "", kind, question.get("instructions", ""), options


def schema_update(schema, previous, *fields):
    return edit_question(schema, "" if previous == NEW_ITEM else previous, *fields)


def result_rows(run, order="入力順"):
    if not run:
        return []
    schema = run["request"]["questions"]
    rows = []
    for index, item in enumerate(run["items"]):
        answers = item.get("answers", {})
        score = max((ambiguity(a) for a in answers.values()), default=0)
        rows.append(
            (
                index,
                score,
                [
                    item["name"],
                    STATUS.get(item["status"], item["status"]),
                    " / ".join(f"{qid}: {summary(answer, schema[qid])}" for qid, answer in answers.items()),
                ],
            )
        )
    if order == "曖昧な順":
        rows.sort(key=lambda x: x[1], reverse=True)
    return [row[2] for row in rows]


def _selected_index(run, selected):
    if isinstance(selected, int) and 0 <= selected < len(run["items"]):
        return selected
    return next((i for i, item in enumerate(run["items"]) if item["status"] == "done"), 0)


def progress_label(run, message):
    counts = {key: sum(x["status"] == key for x in run["items"]) for key in STATUS}
    label = f"{message} · {counts['done']}/{len(run['items'])}完了"
    if counts["cancelled"]:
        label += f" · {counts['cancelled']}停止"
    failed = counts["error"] + counts["failed"]
    if failed:
        label += f" · {failed}失敗"
    return label


def comparison_rows(run, other):
    rows = []
    for item in compare_runs(run, other):
        details = []
        for qid, difference in item["differences"].items():
            text = f"{qid}: 最大差 {difference['max_probability_delta']:.4f}"
            kind = run["request"]["questions"][qid]["type"]
            if kind in ("noul", "score"):
                unit = "P(真)" if kind == "noul" else "期待値"
                text += f"・Δ{unit} {difference['delta']:+.4f}"
            details.append(text)
        rows.append(
            [
                item["name"],
                "あり" if item["changed"] else "なし",
                round(item["max_probability_delta"], 4),
                " / ".join(details),
            ]
        )
    return rows


def comparison_view(run, other):
    if not run or not other:
        return [], COMPARISON_NOTE
    try:
        rows = comparison_rows(run, other)
        return rows, COMPARISON_NOTE if rows else "共通の判定済み画像がありません。"
    except ClefError as exc:
        return [], str(exc)


def detail_html(run, index):
    if not run or index is None or index >= len(run["items"]):
        return '<p class="clef-empty">判定した入力を選ぶと、質問ごとの確率を確認できます。</p>'
    item = run["items"][index]
    esc = html.escape
    parts = [f'<div class="clef-detail"><h3>{esc(item["name"])}</h3>']
    if item["status"] != "done":
        return (
            "".join(parts)
            + f"<p>{esc(STATUS.get(item['status'], item['status']))} {esc(item.get('error', ''))}</p></div>"
        )
    schema = run["request"]["questions"]
    for qid, answer in item["answers"].items():
        question = schema[qid]
        parts.append(
            f'<section><div class="clef-question"><strong>{esc(question.get("instructions") or qid)}</strong><span>{esc(summary(answer, question))}</span></div>'
        )
        if answer["type"] == "noul":
            probabilities = {"true": answer["noul"], "false": 1 - answer["noul"]}
            labels = {"true": "真", "false": "偽"}
        elif answer["type"] == "choice":
            probabilities, labels = answer["probabilities"], question["criteria"]
        else:
            probabilities = answer["probabilities"]
            labels = {str(i): f"{i} · {value}" for i, value in enumerate(question["criteria"])}
        for key, p in probabilities.items():
            width = max(0, min(100, float(p) * 100))
            parts.append(
                f'<div class="clef-prob"><span>{esc(labels.get(key, key))}</span><div class="clef-track"><div style="width:{width:.3f}%"></div></div><code>{p:.3f}</code></div>'
            )
        parts.append("</section>")
    usage = item.get("usage", {})
    parts.append(
        f"<small>{item.get('seconds', 0):.2f}秒 · {usage.get('input_tokens', 0)} tokens · peak {item.get('peak_allocated_gib', 0):.2f} GiB</small></div>"
    )
    return "".join(parts)


def run_heading(run, current, paths=None):
    if not run:
        return "まだ判定していません"
    mismatch = canonical_hash(run["request"]) != canonical_hash(current)
    if paths is not None:
        try:
            hashes = [
                _uploaded_hash(str(path), Path(path).stat().st_size, Path(path).stat().st_mtime_ns) for path in paths
            ]
            original = [item["sha256"] for item in run["items"] if item["path"]]
            mismatch = mismatch or hashes != original
        except OSError:
            mismatch = True
    label = PROFILES[run["request"]["profile"]]["label"]
    done = sum(x["status"] == "done" for x in run["items"])
    unit = "件" if any(item.get("kind") == "record" for item in run["items"]) else "枚"
    kind = "文章・JSONの結果" if unit == "件" else "画像の結果"
    return f"{run['id']} · {kind} · {label} · {done}/{len(run['items'])}{unit}" + (
        " · 現在の入力と不一致" if mismatch else ""
    )


def _read_run(key):
    if not re.fullmatch(r"\d{8}T\d{6}-[a-f0-9]{8}", key or ""):
        raise ClefError("判定記録を選んでください。")
    directory = (OUTPUTS / key).resolve()
    if not directory.is_relative_to(OUTPUTS.resolve()):
        raise ClefError("判定記録の保存先が不正です。")
    return json.loads((directory / "result.json").read_text(encoding="utf-8"))


def _history():
    result = []
    if OUTPUTS.is_dir():
        for path in sorted(OUTPUTS.glob("*/result.json"), reverse=True)[:40]:
            try:
                run = _read_run(path.parent.name)
                label = f"{path.parent.name} · {PROFILES[run['request']['profile']]['label']}"
                if run["status"] == "running":
                    label += " · 処理中・中断記録"
                result.append((label, path.parent.name))
            except (OSError, ValueError, KeyError):
                continue
    return result


def _request(profile, state, schema, pixels, length):
    if (state or "").lstrip().startswith(("{", "[")):
        try:
            state = json.loads(state)
        except json.JSONDecodeError as exc:
            raise ClefError(f"補足JSONの{exc.lineno}行目を確認してください。") from exc
    return validate_request(
        {"profile": profile, "state": state or "", "questions": schema, "max_pixels": pixels, "max_length": length}
    )


def _schema_outputs(schema, selected=None):
    labels = {"choice": "選択", "score": "段階", "noul": "真偽"}
    rows = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(value)}</td>" for value in (qid, labels[q["type"]], q.get("instructions", qid)))
        + "</tr>"
        for qid, q in schema.items()
    )
    table = (
        '<div class="clef-schema"><table><thead><tr><th>ID</th><th>種類</th><th>判断すること</th></tr></thead><tbody>'
        + rows
        + "</tbody></table></div>"
    )
    key = selected if selected in schema else next(iter(schema))
    return (
        schema,
        gr.update(
            choices=[(q.get("instructions") or k, k) for k, q in schema.items()] + [("＋ 新しい項目", NEW_ITEM)],
            value=key,
        ),
        table,
        json.dumps(schema, ensure_ascii=False, indent=2),
    )


def environment_status():
    installed = []
    for profile in ("flash-int8", "clef-24gb", "flash-bf16"):
        try:
            bundle_manifest(RUNTIME, profile)
            installed.append("Clef NF4 · 24GB / 16GB共用" if profile == "clef-24gb" else PROFILES[profile]["label"])
        except ClefError:
            pass
    return (
        "導入済み: " + ", ".join(installed)
        if installed
        else "モデル未導入。aikimi-clef-setup.batを実行してから環境を再確認してください。"
    )


def build_conditions(text):
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return parse_schema(
        {
            f"condition_{index + 1}": {"type": "noul", "instructions": f"入力について次の条件が成り立つ。{line}"}
            for index, line in enumerate(lines)
        }
    )


def table_view(run, order="入力順", review=None, visibility="すべて", rules=None, combination="すべて"):
    headers = ["#", "入力", "状態", "人の判断", "判断結果"]
    rows, displays = [], []
    decisions = (review or {}).get("decisions", {})
    candidates = set(matching_ids(run, rules or [], combination)) if rules else set()
    items = list((run or {}).get("items", []))
    if order == "曖昧な順":
        items.sort(
            key=lambda item: max((ambiguity(answer) for answer in item.get("answers", {}).values()), default=0),
            reverse=True,
        )
    for item in items:
        decision = decisions.get(item["id"], "unreviewed")
        if visibility == "候補だけ" and item["id"] not in candidates:
            continue
        if visibility in REVIEW_LABELS.values() and REVIEW_LABELS[decision] != visibility:
            continue
        answers = item.get("answers", {})
        body = " / ".join(
            f"{qid}: {summary(answer, run['request']['questions'][qid])}" for qid, answer in answers.items()
        )
        row = [
            f"{run['id']}:{item['id']}",
            item["name"],
            STATUS.get(item["status"], item["status"]),
            REVIEW_LABELS[decision],
            body,
        ]
        rows.append(row)
        displays.append([str(int(item["id"]) + 1), *[html.escape(str(value)) for value in row[1:]]])
    return {"headers": headers, "data": rows, "metadata": {"display_value": displays}}


def selected_token(run, row):
    if not row or not run:
        raise ClefError("表示中の項目を選択してください。")
    identifier, separator, item_id = str(row[0]).partition(":")
    if not separator or identifier != run["id"] or not any(item["id"] == item_id for item in run["items"]):
        raise ClefError("表示が更新されています。現在の一覧から選び直してください。")
    return [identifier, item_id]


def gallery_token(run, path):
    identifier, separator, item_id = Path(path).stem.partition("__")
    return selected_token(run, [f"{identifier}:{item_id}"]) if separator else selected_token(run, [])


def visible_selection(run, token, visible_ids):
    items = {item["id"]: item for item in run["items"]}
    if token and token[0] == run["id"] and token[1] in visible_ids:
        return items[token[1]]
    return items[visible_ids[0]] if visible_ids else None


def build_ui():
    from .workspace_ui import build

    return build(outputs=OUTPUTS)
