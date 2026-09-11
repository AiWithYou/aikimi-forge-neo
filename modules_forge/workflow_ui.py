"""Small, dependency-free helpers for Forge Neo workflow panels.

The helpers deliberately return plain HTML strings so img2img scripts can share
one visual language without importing Gradio (or coupling their processing
tests to Gradio internals).
"""

from __future__ import annotations

from html import escape
from typing import Iterable, Sequence


def workflow_hero(
    title: str,
    description: str,
    *,
    badges: Sequence[str] = (),
    steps: Sequence[str] = (),
    eyebrow: str = "AIKIMI NEO · IMG2IMG WORKFLOW",
) -> str:
    """Build the compact introduction shown above a workflow's controls."""

    badge_markup = "".join(f"<span>{escape(str(badge))}</span>" for badge in badges)
    steps_markup = "".join(
        (f"<li><span>{index:02d}</span><p>{escape(str(step))}</p></li>")
        for index, step in enumerate(steps, start=1)
    )
    return (
        f'<section class="neo-workflow-hero" aria-label="{escape(str(title))}">'
        f'<p class="neo-workflow-eyebrow">{escape(str(eyebrow))}</p>'
        f"<h3>{escape(str(title))}</h3>"
        f'<p class="neo-workflow-description">{escape(str(description))}</p>'
        f'<div class="neo-workflow-badges">{badge_markup}</div>'
        f'<ol class="neo-workflow-steps">{steps_markup}</ol>'
        "</section>"
    )


def workflow_section(index: int, title: str, description: str = "") -> str:
    """Build a numbered section heading for a dense settings panel."""

    description_markup = (
        f"<small>{escape(str(description))}</small>" if description else ""
    )
    return (
        '<div class="neo-workflow-section-heading">'
        f'<span aria-hidden="true">{int(index):02d}</span>'
        "<div>"
        f"<strong>{escape(str(title))}</strong>"
        f"{description_markup}"
        "</div>"
        "</div>"
    )


def workflow_summary(
    title: str,
    items: Iterable[tuple[str, object]],
    *,
    status: str = "準備完了",
    note: str = "",
    tone: str = "ready",
) -> str:
    """Build an accessible live summary for the currently selected workflow."""

    normalized_tone = tone if tone in {"ready", "caution", "experimental"} else "ready"
    item_markup = "".join(
        (f"<div><dt>{escape(str(label))}</dt><dd>{escape(str(value))}</dd></div>")
        for label, value in items
    )
    note_markup = f"<p>{escape(str(note))}</p>" if note else ""
    return (
        f'<section class="neo-workflow-summary is-{normalized_tone}" '
        'role="status" aria-live="polite" aria-atomic="true">'
        '<div class="neo-workflow-summary-title">'
        f"<span>{escape(str(status))}</span>"
        f"<strong>{escape(str(title))}</strong>"
        "</div>"
        f"<dl>{item_markup}</dl>"
        f"{note_markup}"
        "</section>"
    )
