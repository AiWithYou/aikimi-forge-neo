from __future__ import annotations

import math
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

NO_STANDARD_LORA = "None"


def _append_lora_tag(prompt: str, name: str, strength: float) -> str:
    existing = re.compile(
        rf"<lora:\s*{re.escape(name)}\s*:[^>]+>",
        flags=re.IGNORECASE,
    )
    if existing.search(prompt):
        return prompt
    separator = "" if not prompt or prompt[-1].isspace() else " "
    return f"{prompt}{separator}<lora:{name}:{strength:.12g}>"


def _append_to_prompt_value(value: str | list[str], name: str, strength: float):
    if isinstance(value, str):
        return _append_lora_tag(value, name, strength)
    if isinstance(value, list):
        return [_append_lora_tag(prompt, name, strength) for prompt in value]
    raise TypeError(f"Unsupported prompt type for Anima LoRA selection: {type(value)!r}")


def apply_standard_lora_selection(
    processing: Any,
    name: str | None,
    strength: float,
    available: Mapping[str, str],
) -> bool:
    """Inject one UI-selected LoRA into standard Forge prompt processing."""

    return apply_standard_lora_selections(
        processing,
        [(name, strength)],
        available,
    )


def apply_standard_lora_selections(
    processing: Any,
    selections: Iterable[tuple[str | None, float]],
    available: Mapping[str, str],
) -> bool:
    """Inject UI-selected LoRAs into standard Forge prompt processing."""

    validated: list[tuple[str, float]] = []
    seen: set[str] = set()
    paths: set[Path] = set()
    local_paths = set(getattr(processing, "_local_lora_paths", ()))
    own_paths = set(getattr(processing, "_anima_standard_lora_paths", ()))
    prompts = processing.prompt if isinstance(processing.prompt, list) else [processing.prompt]
    tags = re.findall(r"<lora:\s*([^:>]+?)\s*:", "\n".join([*prompts, *processing.all_prompts]))
    networks = sys.modules.get("networks")

    for name, strength in selections:
        if name in {None, "", NO_STANDARD_LORA} or name in seen:
            continue
        strength = float(strength)
        if not math.isfinite(strength):
            raise ValueError("Standard Anima LoRA strength must be finite.")
        if strength == 0:
            continue
        if name not in available:
            raise FileNotFoundError(f"Standard Anima LoRA '{name}' is unavailable. Refresh Forge and select it again.")
        path = Path(available[name]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Standard Anima LoRA '{name}' no longer exists at {path}.")

        if path in paths or path in local_paths:
            raise ValueError(f"Anima欄とLoRA選択欄に同じファイルがあります: {path.name}")
        for tag in tags:
            known = None
            if networks is not None:
                known = networks.available_networks.get(tag) or networks.available_network_aliases.get(tag)
            tagged_path = (
                Path(known.filename).resolve()
                if known
                else Path(available[tag]).resolve()
                if tag in available
                else None
            )
            if tagged_path == path and path not in own_paths:
                raise ValueError(f"プロンプトとAnima欄に同じファイルがあります: {path.name}")

        seen.add(name)
        paths.add(path)
        validated.append((name, strength))

    if not validated:
        return False

    prompt = processing.prompt
    all_prompts = processing.all_prompts
    for name, strength in validated:
        prompt = _append_to_prompt_value(prompt, name, strength)
        all_prompts = _append_to_prompt_value(all_prompts, name, strength)

    processing.prompt = prompt
    processing.all_prompts = all_prompts
    processing._anima_standard_lora_paths = tuple(paths)
    processing.extra_generation_params["Anima 3.8B standard LoRA"] = ", ".join(
        f"{name}:{strength:.12g}" for name, strength in validated
    )
    return True
