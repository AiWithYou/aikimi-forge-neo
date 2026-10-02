"""Small visual framing surface; original pixels stay in the native pipeline."""

import base64
import io
import json
from html import escape

from PIL import Image

from .outpaint import normalize_image


def canvas_markup(source, left, top, right, bottom):
    if source is None:
        return ""
    try:
        original = normalize_image(source)
    except (OSError, ValueError):
        return ""
    thumb = original.copy()
    thumb.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    thumb.save(buffer, format="PNG")
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    pads = []
    for value in (left, top, right, bottom):
        try:
            pads.append(max(0, min(4096, int(value))))
        except (TypeError, ValueError, OverflowError):
            pads.append(0)
    layout = escape(json.dumps({"w": original.width, "h": original.height, "pads": pads}), quote=True)
    handles = "".join(
        f'<button type="button" class="qoc-h qoc-h-{key}" data-h="{key}" '
        f'aria-label="{label}をドラッグして広げる" title="{label}を広げる"></button>'
        for key, label in (
            ("n", "上"),
            ("s", "下"),
            ("w", "左"),
            ("e", "右"),
            ("nw", "左上"),
            ("ne", "右上"),
            ("sw", "左下"),
            ("se", "右下"),
        )
    )
    ratios = "".join(
        f'<button type="button" data-ratio="{ratio}" aria-label="完成画像を{label}にする">{label}</button>'
        for label, ratio in (("1:1", 1), ("4:3", 4 / 3), ("3:2", 1.5), ("16:9", 16 / 9), ("9:16", 9 / 16))
    )
    return (
        f'<div class="qoc" data-layout="{layout}">'
        f'<div class="qoc-toolbar" aria-label="完成画像の比率">{ratios}</div>'
        '<div class="qoc-viewport"><div class="qoc-frame">'
        f'<button type="button" class="qoc-picture" aria-label="元画像の位置を動かす" '
        f'title="ドラッグで余白の位置を変える"><img src="data:image/png;base64,{data}" '
        f'alt="元画像" draggable="false"></button>{handles}</div></div>'
        '<div class="qoc-toolbar qoc-bottom">'
        '<button type="button" data-action="grow">＋ 全体を広げる</button>'
        '<button type="button" data-action="center">画像を中央へ</button>'
        '<button type="button" data-action="reset">余白をリセット</button>'
        '<output class="qoc-size" aria-live="polite"></output></div>'
        '<p class="qoc-hint">辺・角をドラッグして広げる · 画像をドラッグして位置を調整</p></div>'
    )


def parse_canvas_commit(value, source):
    """Accept one atomic layout, rejecting stale source or malformed browser data."""
    if not isinstance(value, str) or len(value) > 1024:
        raise ValueError("範囲の指定を読み込めません。")
    original = normalize_image(source)
    try:
        payload = json.loads(value)
        if not isinstance(payload, dict) or (payload.get("w"), payload.get("h")) != original.size:
            raise ValueError("元画像が変わりました。範囲を指定し直してください。")
        pads = payload["pads"]
        if not isinstance(pads, list) or len(pads) != 4:
            raise ValueError("余白は4辺で指定してください。")
        if any(type(v) is not int or not 0 <= v <= 4096 for v in pads):
            raise ValueError("余白は0〜4096 pxの整数で指定してください。")
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("範囲の指定を読み込めません。") from exc
    return tuple(pads)
