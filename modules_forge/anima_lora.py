"""Block-layout conversion helpers for Anima LoRAs.

Anima-2.9B expands the original 28 transformer blocks to 40 blocks, and
Anima-3.8B expands that layout to 52.  The inserted blocks were initialized
from specific source blocks, so a LoRA can be expanded without copying tensor
storage by exposing the same tensor under each corresponding target key.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import re
from typing import TypeVar


ANIMA_BASE_BLOCKS = 28
ANIMA_29B_BLOCKS = 40
ANIMA_38B_BLOCKS = 52

# Positions inserted by the published Anima-2.9B expansion manifest.
ANIMA_29B_INSERTED_BLOCKS = (2, 5, 8, 11, 14, 17, 21, 24, 27, 30, 33, 36)

# Each inserted block was initialized from this block in the 28-block model.
_ANIMA_29B_INSERTED_TO_BASE = {
    2: 1,
    5: 3,
    8: 5,
    11: 7,
    14: 9,
    17: 11,
    21: 14,
    24: 16,
    27: 18,
    30: 20,
    33: 22,
    36: 24,
}

ANIMA_BASE_TO_29B = tuple(block for block in range(ANIMA_29B_BLOCKS) if block not in ANIMA_29B_INSERTED_BLOCKS)
_ANIMA_29B_ORIGINAL_TO_BASE = {expanded: base for base, expanded in enumerate(ANIMA_BASE_TO_29B)}
ANIMA_29B_TO_BASE = tuple(
    _ANIMA_29B_INSERTED_TO_BASE.get(expanded, _ANIMA_29B_ORIGINAL_TO_BASE.get(expanded))
    for expanded in range(ANIMA_29B_BLOCKS)
)
# Positions and sources recorded by the published Anima-3.8B Pro52 manifest.
ANIMA_38B_INSERTED_BLOCKS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47)
_ANIMA_38B_INSERTED_TO_29B = {
    3: 2,
    7: 5,
    11: 8,
    15: 11,
    19: 14,
    23: 17,
    27: 20,
    31: 23,
    35: 26,
    39: 29,
    43: 32,
    47: 35,
}
ANIMA_29B_TO_38B = tuple(
    block
    for block in range(ANIMA_38B_BLOCKS)
    if block not in ANIMA_38B_INSERTED_BLOCKS
)
_ANIMA_38B_ORIGINAL_TO_29B = {
    expanded: source for source, expanded in enumerate(ANIMA_29B_TO_38B)
}
ANIMA_38B_TO_29B = tuple(
    _ANIMA_38B_INSERTED_TO_29B.get(
        expanded, _ANIMA_38B_ORIGINAL_TO_29B.get(expanded)
    )
    for expanded in range(ANIMA_38B_BLOCKS)
)


def _target_to_source_blocks(
    source_blocks: int, target_blocks: int
) -> tuple[int, ...]:
    if (source_blocks, target_blocks) == (ANIMA_BASE_BLOCKS, ANIMA_29B_BLOCKS):
        return ANIMA_29B_TO_BASE
    if (source_blocks, target_blocks) == (ANIMA_29B_BLOCKS, ANIMA_BASE_BLOCKS):
        return ANIMA_BASE_TO_29B
    if (source_blocks, target_blocks) == (ANIMA_29B_BLOCKS, ANIMA_38B_BLOCKS):
        return ANIMA_38B_TO_29B
    if (source_blocks, target_blocks) == (ANIMA_38B_BLOCKS, ANIMA_29B_BLOCKS):
        return ANIMA_29B_TO_38B
    if (source_blocks, target_blocks) == (ANIMA_BASE_BLOCKS, ANIMA_38B_BLOCKS):
        return tuple(ANIMA_29B_TO_BASE[block] for block in ANIMA_38B_TO_29B)
    if (source_blocks, target_blocks) == (ANIMA_38B_BLOCKS, ANIMA_BASE_BLOCKS):
        return tuple(
            ANIMA_29B_TO_38B[ANIMA_BASE_TO_29B[block]]
            for block in range(ANIMA_BASE_BLOCKS)
        )
    raise ValueError(
        f"Unsupported Anima LoRA layout conversion: {source_blocks} to {target_blocks}"
    )


# Kohya, Forge generic, PEFT-style, and bare Comfy-style block names.
_ANIMA_BLOCK_KEY_PATTERNS = (
    re.compile(r"^(?P<prefix>lora_unet_(?:model_)?(?:diffusion_model_)?blocks_)(?P<block>\d+)(?P<suffix>_.+)$"),
    re.compile(r"^(?P<prefix>(?:.+\.)?diffusion_model\.blocks\.)(?P<block>\d+)(?P<suffix>\..+)$"),
    re.compile(r"^(?P<prefix>blocks\.)(?P<block>\d+)(?P<suffix>\..+)$"),
)


@dataclass(frozen=True)
class AnimaLoraConversionReport:
    source_blocks: int | None
    target_blocks: int
    source_block_indices: tuple[int, ...]
    direction: str
    duplicated_entries: int = 0
    dropped_entries: int = 0

    @property
    def converted(self) -> bool:
        return self.direction not in {
            "native",
            "ambiguous_source",
            "unsupported_target",
        }


_Value = TypeVar("_Value")


def _parse_anima_block_key(key: str) -> tuple[int, str, str] | None:
    for pattern in _ANIMA_BLOCK_KEY_PATTERNS:
        match = pattern.match(key)
        if match is not None:
            return int(match.group("block")), match.group("prefix"), match.group("suffix")
    return None


def anima_lora_block_indices_from_keys(keys: Iterable[str]) -> tuple[int, ...]:
    """Return sorted transformer-block indices recognized in LoRA keys."""

    indices = set()
    for key in keys:
        parsed = _parse_anima_block_key(key)
        if parsed is not None:
            indices.add(parsed[0])
    return tuple(sorted(indices))


def anima_lora_block_indices(lora: dict[str, _Value]) -> tuple[int, ...]:
    """Return sorted transformer-block indices recognized in a LoRA state dict."""

    return anima_lora_block_indices_from_keys(lora.keys())


def detect_anima_lora_block_count_from_keys(keys: Iterable[str]) -> int | None:
    """Detect a complete supported layout without loading LoRA tensor data."""

    indices = anima_lora_block_indices_from_keys(keys)
    if indices == tuple(range(ANIMA_BASE_BLOCKS)):
        return ANIMA_BASE_BLOCKS
    if indices == tuple(range(ANIMA_29B_BLOCKS)):
        return ANIMA_29B_BLOCKS
    if indices == tuple(range(ANIMA_38B_BLOCKS)):
        return ANIMA_38B_BLOCKS
    return None


def detect_anima_lora_block_count(lora: dict[str, _Value]) -> int | None:
    """Detect only complete supported layouts to avoid unsafe guessing."""

    return detect_anima_lora_block_count_from_keys(lora.keys())


def has_anima_lora_signature(
    keys: Iterable[str],
    metadata: dict[str, str] | None = None,
) -> bool:
    """Reject unrelated architectures that happen to use 28/40/52 blocks."""

    metadata = metadata or {}
    metadata_fields = (
        "modelspec.architecture",
        "modelspec.tags",
        "ss_base_model_version",
        "ss_leco_model_type",
        "ss_network_module",
    )
    if any(
        "anima" in metadata.get(field, "").lower()
        for field in metadata_fields
    ):
        return True

    key_list = tuple(keys)
    has_adaln = any("adaln_modulation_" in key for key in key_list)
    has_cross_attention = any(
        re.search(r"cross_attn[._](?:q|k|v|output)_proj", key)
        for key in key_list
    )
    has_self_attention = any(
        re.search(r"self_attn[._](?:q|k|v|output)_proj", key)
        for key in key_list
    )
    has_anima_mlp = any(
        re.search(r"mlp[._]layer[12]", key)
        for key in key_list
    )
    return has_adaln and has_cross_attention and has_self_attention and has_anima_mlp


def convert_anima_lora_layout(
    lora: dict[str, _Value], target_blocks: int
) -> tuple[dict[str, _Value], AnimaLoraConversionReport]:
    """Convert a complete Anima LoRA between the 28-, 40-, and 52-block layouts.

    Conversion is deliberately skipped for sparse or otherwise ambiguous block
    coverage.  Expanding reuses tensor objects for inserted blocks, avoiding a
    second allocation of large LoRA tensors.  Collapsing a native 40-block LoRA
    drops entries for the 12 inserted blocks and is therefore lossy.
    """

    source_indices = anima_lora_block_indices(lora)
    source_blocks = detect_anima_lora_block_count(lora)

    supported_blocks = {
        ANIMA_BASE_BLOCKS,
        ANIMA_29B_BLOCKS,
        ANIMA_38B_BLOCKS,
    }
    if target_blocks not in supported_blocks:
        return lora, AnimaLoraConversionReport(source_blocks, target_blocks, source_indices, "unsupported_target")

    if source_blocks is None:
        return lora, AnimaLoraConversionReport(None, target_blocks, source_indices, "ambiguous_source")

    if source_blocks == target_blocks:
        return lora, AnimaLoraConversionReport(source_blocks, target_blocks, source_indices, "native")

    target_to_source = _target_to_source_blocks(source_blocks, target_blocks)
    targets_by_source = tuple(
        tuple(
            target
            for target, source in enumerate(target_to_source)
            if source == source_block
        )
        for source_block in range(source_blocks)
    )

    converted: dict[str, _Value] = {}
    source_for_output: dict[str, str] = {}
    duplicated_entries = 0
    dropped_entries = 0

    def add(output_key: str, value: _Value, source_key: str) -> None:
        if output_key in converted:
            previous = source_for_output[output_key]
            raise ValueError(f"Anima LoRA conversion collision for {output_key!r}: {previous!r} and {source_key!r}")
        converted[output_key] = value
        source_for_output[output_key] = source_key

    for key, value in lora.items():
        parsed = _parse_anima_block_key(key)
        if parsed is None:
            add(key, value, key)
            continue

        source_block, prefix, suffix = parsed
        targets = targets_by_source[source_block]
        if not targets:
            dropped_entries += 1
            continue
        for target_block in targets:
            add(f"{prefix}{target_block}{suffix}", value, key)
        duplicated_entries += len(targets) - 1

    direction = f"{source_blocks}_to_{target_blocks}"
    return converted, AnimaLoraConversionReport(
        source_blocks,
        target_blocks,
        source_indices,
        direction,
        duplicated_entries=duplicated_entries,
        dropped_entries=dropped_entries,
    )
