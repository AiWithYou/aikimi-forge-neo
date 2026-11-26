"""Local prompt templates and reproducible seeded choices."""

import json
import random
import re
import secrets
import uuid
from datetime import UTC, datetime

from modules_forge.workspace_store import STATE_DIRECTORY, managed_path, read_json, state_transaction, write_json

LIMIT = 16384
MAX_ENTRIES = 512
ID = re.compile(r"[a-zA-Z0-9_-]{1,64}\Z")


def expand_wildcards(text: str, seed: int | None = None) -> tuple[str, int]:
    if not isinstance(text, str) or len(text) > LIMIT:
        raise ValueError("プロンプトが長すぎます。")
    seed = secrets.randbits(53) if seed is None else seed
    if type(seed) is not int or not -(2**63) <= seed < 2**63:
        raise ValueError("Seed が範囲外です。")
    rng = random.Random(seed)  # noqa: S311 - reproducible artistic choices
    stack = []
    parts = []
    choices = []
    index = groups = 0
    while index < len(text):
        char = text[index]
        index += 1
        if char == "\\" and index < len(text) and text[index] in r"{}|\\":
            parts.append(text[index])
            index += 1
        elif char == "{":
            if len(stack) >= 24 or groups >= 256:
                raise ValueError("選択肢の上限を超えています。")
            groups += 1
            stack.append((parts, choices))
            parts, choices = [], []
        elif stack and char == "|":
            choices.append("".join(parts))
            parts = []
        elif stack and char == "}":
            value = "".join(parts)
            value = rng.choice([*choices, value]) if choices else "{" + value + "}"
            parts, choices = stack.pop()
            parts.append(value)
        else:
            parts.append(char)
    if stack:
        raise ValueError("選択肢の閉じ括弧が必要です。")
    return "".join(parts), seed


def _string(value, limit):
    if not isinstance(value, str) or len(value) > limit or "\0" in value:
        raise ValueError("文字列の形式または長さが不正です。")
    return value


def validate(entry):
    fields = {"title", "prompt", "negative_prompt", "id", "studio", "tags", "updated_at", "json_fragment"}
    if not isinstance(entry, dict) or entry.keys() - fields:
        raise ValueError("テンプレート形式が不正です。")
    result = {
        key: _string(entry.get(key, ""), limit)
        for key, limit in (
            ("title", 160),
            ("prompt", LIMIT),
            ("negative_prompt", LIMIT),
            ("studio", 64),
            ("updated_at", 64),
        )
    }
    if not result["title"].strip() or not ID.fullmatch(_string(entry.get("id"), 64)):
        raise ValueError("タイトルと有効な ID が必要です。")
    tags = entry.get("tags", [])
    fragment = entry.get("json_fragment", {})
    if not isinstance(tags, list) or len(tags) > 32 or not isinstance(fragment, dict):
        raise ValueError("タグまたは JSON の形式が不正です。")
    try:
        _string(json.dumps(fragment, ensure_ascii=False, allow_nan=False), LIMIT)
    except RecursionError:
        raise ValueError("JSON が複雑すぎます。") from None
    return {**result, "tags": [_string(tag, 64) for tag in tags], "id": entry["id"], "json_fragment": fragment}


class PromptLibrary:
    def __init__(self, root):
        self.root = root

    @property
    def path(self):
        return managed_path(self.root, STATE_DIRECTORY + "/prompt_library.json")

    def entries(self):
        with state_transaction(self.root):
            return self._entries()

    def _entries(self):
        try:
            values = read_json(self.path)
        except FileNotFoundError:
            values = [
                validate(
                    {
                        "id": "landscape",
                        "title": "風景",
                        "prompt": "{朝|夕暮れ}の海",
                        "updated_at": datetime.now(UTC).isoformat(),
                    }
                )
            ]
            write_json(self.path, values)
        if not isinstance(values, list) or len(values) > MAX_ENTRIES:
            raise ValueError("プロンプト集の形式が不正です。")
        return [validate(value) for value in values]

    def save(self, entry):
        with state_transaction(self.root):
            updated = validate(
                {**entry, "id": entry.get("id", uuid.uuid4().hex), "updated_at": datetime.now(UTC).isoformat()}
            )
            values = [e for e in self._entries() if e["id"] != updated["id"]]
            if len(values) >= MAX_ENTRIES:
                raise ValueError("保存件数の上限です。")
            write_json(self.path, [*values, updated])
            return updated

    def delete(self, entry_id):
        if not ID.fullmatch(entry_id):
            raise ValueError("無効な ID です。")
        with state_transaction(self.root):
            values = self._entries()
            remaining = [entry for entry in values if entry["id"] != entry_id]
            if len(remaining) == len(values):
                raise ValueError("テンプレートが見つかりません。")
            write_json(self.path, remaining)
