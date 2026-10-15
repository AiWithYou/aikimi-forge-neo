# SPDX-License-Identifier: Apache-2.0
# Bounding-box and image preprocessing adapted from Black Forest Labs FLUX 3
# reference inference, revision 45e57c9fb06f9751cdced29a65c5d2171003be05.
# See LICENSE.bfl-reference and THIRD_PARTY_NOTICES.md in this directory.
"""Bounding boxes for the upsampler's layout mode: [y_min, x_min, y_max, x_max] in 0-1."""

import io
import json
import math
import re
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

SCALE = 1000
ID = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def to_canvas(boxes: list[dict]) -> list[dict]:
    """0-1 boxes -> upsampler rows on the 0-1000 canvas."""

    def grid(box):
        if not box:
            return None
        if len(box) != 4:
            raise ValueError(f"a box is [y_min, x_min, y_max, x_max], got {box}")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
            raise ValueError("box coordinates must be finite numbers")
        y1, x1, y2, x2 = box
        if not (0 <= y1 < y2 <= 1 and 0 <= x1 < x2 <= 1):
            raise ValueError("a normalized box must have ordered coordinates in 0..1")
        return [round(float(v) * SCALE) for v in box]

    rows = []
    for i, box in enumerate(boxes):
        row_id, desc = str(box.get("id") or f"obj_{i + 1}"), str(box.get("desc") or "")
        if box.get("bbox"):
            rows.append({"id": row_id, "bbox": grid(box["bbox"]), "desc": desc})
        elif box.get("src_bbox") or box.get("tgt_bbox"):
            rows.append(
                {
                    "id": row_id,
                    "from": str(box.get("from") or ""),
                    "src_bbox": grid(box.get("src_bbox")),
                    "tgt_bbox": grid(box.get("tgt_bbox")),
                    "desc": desc,
                }
            )
        else:
            raise ValueError(f"box {i} has neither bbox nor src_bbox/tgt_bbox")
    return rows


def detect(prompt: str) -> tuple[str, list[dict]] | None:
    """(prompt, rows) when the prompt already carries boxes on the 0-1000 canvas, else None. Two forms:
    "<caption> [rows]" (the DiT prompt format) and the upsampler's own "ORIGINAL PROMPT: ... LOCKED
    BOUNDING-BOX ANNOTATIONS ...:\\n[rows]" input. A trailing list that is not all box rows (e.g. a
    caption ending in "[SALE]") is not a box table."""
    text, rows = split(prompt)
    if not rows or not all(
        isinstance(r, dict)
        and isinstance(r.get("id"), str)
        and ID.fullmatch(r["id"])
        and any(valid_box(r.get(k)) for k in ("bbox", "src_bbox", "tgt_bbox"))
        for r in rows
    ):
        return None
    locked = re.fullmatch(r"ORIGINAL PROMPT:\n(.*?)\n\nLOCKED BOUNDING-BOX ANNOTATIONS .*", text, re.S)
    return (locked.group(1) if locked else text), rows


def is_finished(prompt: str) -> bool:
    """An upsampler answer, not a request: its box table is cited in full as <id>s in the caption."""
    found = detect(prompt)
    return found is not None and all(f"<{row['id']}>" in found[0] for row in found[1])


def plan(prompt: str, boxes: list[dict] | None, n_images: int) -> tuple[str, str, list[dict] | None]:
    """(system prompt key, user text, locked rows) for the upsampler. --boxes wins; otherwise boxes found in
    the prompt switch it to l3b by themselves."""
    task = "i2i" if n_images else "t2i"
    if boxes:
        locked = to_canvas(boxes)
    elif found := detect(prompt):
        prompt, locked = found
    else:
        return task, prompt, None
    return f"{task}_bbox", upsampling_input(prompt, locked), locked


def upsampling_input(prompt: str, rows: list[dict]) -> str:
    return (
        f"ORIGINAL PROMPT:\n{prompt.strip()}\n\n"
        "LOCKED BOUNDING-BOX ANNOTATIONS ([y_min, x_min, y_max, x_max] on a 0-1000 canvas, with (0, 0) at "
        "the top-left). This is an authoritative JSON list of dictionaries; preserve every supplied identity, "
        f"provenance value, null, and coordinate exactly:\n{json.dumps(rows, ensure_ascii=False)}"
    )


def split(text: str) -> tuple[str, list | None]:
    """caption, rows: the last '[' from which the rest parses as a JSON list. Not a regex: descs may
    contain brackets and escaped quotes."""
    for i in range(len(text) - 1, -1, -1):
        if text[i] == "[":
            try:
                rows = json.loads(text[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(rows, list):
                return text[:i].rstrip(), rows
    return text.strip(), None


def valid_box(box) -> list | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or abs(v) == float("inf") for v in box):
        return None
    y1, x1, y2, x2 = box
    # max <= 1 means 0-1 coordinates, not the 0-1000 canvas
    if y1 < 0 or x1 < 0 or y2 <= y1 or x2 <= x1 or y2 > SCALE or x2 > SCALE or max(box) <= 1:
        return None
    return list(box)


def valid_added(row, caption: str) -> dict | None:
    """A row the upsampler added: kept only if its id is cited as <id> and it is well formed."""
    if not isinstance(row, dict):
        return None
    row_id, desc = row.get("id"), row.get("desc")
    if (
        not isinstance(row_id, str)
        or not ID.fullmatch(row_id)
        or f"<{row_id}>" not in caption
        or not isinstance(desc, str)
        or not desc.strip()
    ):
        return None
    if (box := valid_box(row.get("bbox"))) is not None:
        return {"id": row_id, "bbox": box, "desc": desc.strip()}
    origin, src, tgt = row.get("from"), row.get("src_bbox"), row.get("tgt_bbox")
    src_box = valid_box(src) if src is not None else None
    tgt_box = valid_box(tgt) if tgt is not None else None
    if (
        not isinstance(origin, str)
        or not re.fullmatch(r"ref_image_\d+|target", origin)
        or (src is not None and src_box is None)
        or (tgt is not None and tgt_box is None)
        or (src_box is None and tgt_box is None)
        or (origin == "target") == (src_box is not None)
    ):
        return None
    return {"id": row_id, "from": origin, "src_bbox": src_box, "tgt_bbox": tgt_box, "desc": desc.strip()}


def iou(a: list, b: list) -> float:
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def reconcile(text: str, locked: list[dict]) -> str:
    """The upsampler's answer with the user's boxes re-imposed.

    1. A row re-planning a locked box under its own id (IoU >= 0.9, e.g. balloon_1 for obj_1) takes the
       locked id, in the rows and in the caption's <id> citations.
    2. Locked rows are kept verbatim, adopting the upsampler's desc; its own rows follow if valid.
    """
    caption, rows = split(text)
    rows = rows or []
    locked_ids = {r["id"] for r in locked}
    row_ids = {r.get("id") for r in rows if isinstance(r, dict)}
    renamed = {}
    for lock in locked:
        lock_box = lock.get("bbox") or lock.get("tgt_bbox") or lock.get("src_bbox")
        if lock["id"] in row_ids or lock_box is None:
            continue
        for row in rows:
            if not isinstance(row, dict) or row.get("id") in renamed or row.get("id") in locked_ids:
                continue
            box = valid_box(row.get("bbox") or row.get("tgt_bbox") or row.get("src_bbox"))
            if box is not None and iou(box, lock_box) >= 0.9:
                renamed[row["id"]] = lock["id"]
                break
    for row in rows:
        if isinstance(row, dict) and row.get("id") in renamed:
            row["id"] = renamed[row["id"]]
    for old, new in renamed.items():
        caption = caption.replace(f"<{old}>", f"<{new}>")

    by_id = {r.get("id"): r for r in rows if isinstance(r, dict) and isinstance(r.get("id"), str)}
    out = []
    for lock in locked:
        desc = by_id.get(lock["id"], {}).get("desc")
        out.append({**lock, "desc": desc.strip() if isinstance(desc, str) and desc.strip() else lock.get("desc", "")})
    for row in rows:
        added = valid_added(row, caption)
        if added is not None and added["id"] not in {r["id"] for r in out}:
            out.append(added)
    return f"{caption} {json.dumps(out, ensure_ascii=False)}"


MAX_REFERENCES = 10
MIN_SIDE, MAX_SIDE = 64, 4096
MAX_SINGLE_PIXELS = 4 * 1024 * 1024
MAX_MULTI_PIXELS = 1024 * 1024


@dataclass
class PreparedImages:
    references: list[Image.Image]
    width: int
    height: int
    crop: tuple[int, int, int, int] | None


def reference_canvas(width: int, height: int):
    """Fit one reference without stretching, counting its 16px padding in the budget."""
    scale = min(1.0, math.sqrt(MAX_SINGLE_PIXELS / (width * height)), MAX_SIDE / max(width, height))
    while True:
        fit = (max(1, round(width * scale)), max(1, round(height * scale)))
        canvas = tuple(min(MAX_SIDE, max(MIN_SIDE, -(-v // 16) * 16)) for v in fit)
        if canvas[0] * canvas[1] <= MAX_SINGLE_PIXELS:
            return fit, canvas
        scale *= 0.995


def prepare_reference(image: Image.Image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    fit, (width, height) = reference_canvas(*image.size)
    if fit != image.size:
        image = image.resize(fit, Image.Resampling.LANCZOS)
    if (width, height) == image.size:
        return image, None
    left, top = (width - image.width) // 2, (height - image.height) // 2
    padding = ((top, height - image.height - top), (left, width - image.width - left), (0, 0))
    mode = "reflect" if min(image.size) > 16 else "symmetric"
    canvas = Image.fromarray(np.pad(np.asarray(image), padding, mode=mode))
    return canvas, (left, top, left + image.width, top + image.height)


def prepare_multi_reference(image: Image.Image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if width * height > MAX_MULTI_PIXELS:
        scale = math.sqrt(MAX_MULTI_PIXELS / (width * height))
        width, height = max(1, int(width * scale)), max(1, int(height * scale))
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    width, height = width // 16 * 16, height // 16 * 16
    if min(width, height) < 16:
        raise ValueError("each multi-reference image must retain at least 16 pixels per side")
    left, top = (image.width - width) // 2, (image.height - height) // 2
    return image.crop((left, top, left + width, top + height))


def prepare_images(images, width=None, height=None):
    if len(images) > MAX_REFERENCES:
        raise ValueError("at most 10 reference images are supported")
    crop = None
    if len(images) > 1:
        references = [prepare_multi_reference(image) for image in images]
    elif images:
        reference, crop = prepare_reference(images[0])
        references = [reference]
        if width is not None or height is not None:
            crop = None
    else:
        references = []
    defaults = references[0].size if references else (1024, 1024)
    width = defaults[0] if width is None else width
    height = defaults[1] if height is None else height
    for side in (width, height):
        if isinstance(side, bool) or not isinstance(side, int) or not MIN_SIDE <= side <= MAX_SIDE or side % 16:
            raise ValueError("output sides must be 64..4096 and multiples of 16")
    return PreparedImages(references, width, height, crop)


def pup_reference_size(width, height, count):
    if count not in range(1, MAX_REFERENCES + 1):
        raise ValueError("PUP needs 1..10 references")
    if max(width, height) / min(width, height) > 200:
        raise ValueError("PUP reference aspect ratio exceeds 200:1")

    def floor(value):
        return max(32, int(math.floor(value / 32)) * 32)

    cap = 1024**2 if count == 1 else (724**2 if count == 2 else 512**2)
    w, h = max(32, round(width / 32) * 32), max(32, round(height / 32) * 32)
    if w * h > cap:
        beta = math.sqrt(width * height / cap)
        w, h = floor(width / beta), floor(height / beta)
    if w * h > cap:
        if w == 32:
            h = floor(cap / w)
        else:
            w = floor(cap / h)
    return w, h


def pup_reference(image, count):
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    image = image.convert("RGB")
    size = pup_reference_size(*image.size, count)
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    with Image.open(io.BytesIO(buffer.getvalue())) as compressed:
        return compressed.convert("RGB")


@dataclass
class PupRequest:
    system_key: str
    prompt: str
    images: list[Image.Image]
    locked: list[dict] | None
    presence_penalty: float
    max_tokens: int = 3584


def pup_request(prompt, images, boxes=None):
    if len(images) > MAX_REFERENCES:
        raise ValueError("PUP takes at most 10 references")
    key, text, locked = plan(prompt, boxes, len(images))
    references = [pup_reference(image, len(images)) for image in images]
    if images:
        if locked is None:
            text = (
                "Reference images follow in order (the first image, the second image, ...).\nUser request: "
                + text
                + "\nOutput the caption."
            )
        else:
            text = (
                "Reference images follow in attachment order: ref_image_0, ref_image_1, ...\nUser editing instruction: "
                + text
                + "\nDecide the edit, imagine the target image, and output the annotation record."
            )
    return PupRequest(key, text, references, locked, 0.0 if images else 1.5)


def pup_chat_text(spec, system_prompt):
    """The shipped Qwen3.5 PUP chat template, for its two-message inference mode.

    Comfy's generic template uses different empty-thinking whitespace. Supplying
    the rendered text also keeps every reference image in its original order.
    """
    vision = "<|vision_start|><|image_pad|><|vision_end|>" * len(spec.images)
    content = (vision + spec.prompt).strip()
    return (
        "<|im_start|>system\n" + system_prompt.strip() + "<|im_end|>\n"
        "<|im_start|>user\n" + content + "<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def bound_text_tokens(tokens):
    """Keep native 80-token padding while applying the reference's 4096-token cap."""
    key = "qwen3vl_4b"
    if key not in tokens or len(tokens[key]) != 1:
        raise ValueError("FLUX 3 text encoding requires one Qwen3-VL-4B prompt window")
    result = dict(tokens)
    result[key] = [list(tokens[key][0][:4096])]
    return result
