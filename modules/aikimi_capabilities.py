"""Lightweight built-in workflow capability checks for Aikimi diagnostics."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from modules.aikimi_diagnostics import (
    CheckState,
    DiagnosticCheck,
    DiagnosticPaths,
)

_SAFETENSORS_HEADER_LIMIT = 16 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


@dataclass(frozen=True)
class _FileContract:
    relative_path: str
    expected_bytes: int
    markers: tuple[str, ...] = ()


_KREA2_FILES = (
    _FileContract(
        "Stable-diffusion/krea2_turbo_int8_convrot.safetensors",
        13_492_686_496,
        ("blocks.0.attn.gate.weight_scale", "blocks.0.attn.gate.comfy_quant"),
    ),
    _FileContract(
        "text_encoder/qwen3vl_4b_fp8_scaled.safetensors",
        5_242_467_968,
        ("model.embed_tokens.weight", "model.visual.blocks.0.attn.qkv.weight"),
    ),
    _FileContract(
        "VAE/qwen_image_vae.safetensors",
        253_806_246,
        ("conv1.weight", "decoder.conv1.weight"),
    ),
)

_ANIMA38_FILES = (
    _FileContract(
        "Stable-diffusion/Anima-3.8B-v1.1-int8-convrot.safetensors",
        5_543_364_574,
        (
            "anima38_v11_main_attention_mlp_v1",
            "net.anima_v2_connector.semantic_resampler.query_tokens",
            "net.blocks.51.mlp.layer2.weight",
        ),
    ),
    _FileContract(
        "text_encoder/qwen35_4b.safetensors",
        4_779_016_600,
        ("embed_tokens.weight", "layers.31.input_layernorm.weight"),
    ),
    _FileContract("text_encoder/qwen_3_06b_base.safetensors", 1_192_135_096),
    _FileContract(
        "VAE/qwen_image_vae.safetensors",
        253_806_246,
        ("conv1.weight", "decoder.conv1.weight"),
    ),
)


def _safe_file_stat(path: Path) -> os.stat_result | None:
    try:
        return path.stat() if path.is_file() else None
    except OSError:
        return None


def _matches_file_contract(models_root: Path, contract: _FileContract) -> bool:
    """Check size and bounded safetensors markers without hashing model data."""

    path = models_root / contract.relative_path
    file_stat = _safe_file_stat(path)
    if file_stat is None or file_stat.st_size != contract.expected_bytes:
        return False
    if not contract.markers:
        return True
    try:
        with path.open("rb") as stream:
            header_size_raw = stream.read(8)
            if len(header_size_raw) != 8:
                return False
            header_size = int.from_bytes(header_size_raw, "little", signed=False)
            if not 2 <= header_size <= _SAFETENSORS_HEADER_LIMIT:
                return False
            header_raw = stream.read(header_size)
        if len(header_raw) != header_size:
            return False
        header_text = header_raw.decode("utf-8")
        if not isinstance(json.loads(header_text), dict):
            return False
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return False
    return all(marker in header_text for marker in contract.markers)


def _valid_sha256_sidecar(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size > 1024:
            return False
        token = path.read_text(encoding="utf-8-sig").strip().split()[0]
    except (OSError, UnicodeError, IndexError):
        return False
    return bool(_SHA256.fullmatch(token))


def _alternative_krea2_present(models_root: Path) -> bool:
    directory = models_root / "Stable-diffusion"
    try:
        return any(
            item.is_file()
            and item.suffix.casefold() == ".safetensors"
            and "krea2" in item.name.casefold()
            and item.stat().st_size > 1024 * 1024
            for item in directory.iterdir()
        )
    except OSError:
        return False


def _krea2_check(paths: DiagnosticPaths) -> DiagnosticCheck:
    verified = sum(_matches_file_contract(paths.models_root, contract) for contract in _KREA2_FILES)
    if verified == len(_KREA2_FILES):
        return DiagnosticCheck(
            "krea2",
            "Krea2",
            CheckState.READY,
            "Pinned Krea2 model, text encoder, and VAE files were detected.",
            "Use the model installer verify command before a release to recompute full hashes.",
            available=True,
        )
    if _alternative_krea2_present(paths.models_root):
        return DiagnosticCheck(
            "krea2",
            "Krea2",
            CheckState.WARNING,
            f"A Krea2 checkpoint was detected, but {verified}/{len(_KREA2_FILES)} pinned support files matched.",
            "Run download_krea2_int8_convrot_models.bat to install and verify the supported file set.",
            available=True,
        )
    return DiagnosticCheck(
        "krea2",
        "Krea2",
        CheckState.BLOCKED,
        "The supported Krea2 model set was not detected.",
        "Run download_krea2_int8_convrot_models.bat, review the model license, and retry the check.",
        available=False,
    )


def _anima38_check(paths: DiagnosticPaths) -> DiagnosticCheck:
    verified = sum(_matches_file_contract(paths.models_root, contract) for contract in _ANIMA38_FILES)
    checkpoint = paths.models_root / _ANIMA38_FILES[0].relative_path
    sidecar_valid = _valid_sha256_sidecar(Path(f"{checkpoint}.sha256"))
    if verified == len(_ANIMA38_FILES) and sidecar_valid:
        return DiagnosticCheck(
            "anima38",
            "Anima 3.8B",
            CheckState.READY,
            "The v1.1 52-block bundle, Semantic Connector v2, Qwen encoders, VAE, and integrity record were detected.",
            "No action is required. Re-run installer verification after moving model files.",
            available=True,
        )
    if verified:
        return DiagnosticCheck(
            "anima38",
            "Anima 3.8B",
            CheckState.WARNING,
            f"Anima support is partially installed ({verified}/{len(_ANIMA38_FILES)} files; integrity record {'present' if sidecar_valid else 'missing'}).",
            "Run download_anima38_v11_int8_convrot_models.bat to verify or repair the supported file set.",
            available=False,
        )
    return DiagnosticCheck(
        "anima38",
        "Anima 3.8B",
        CheckState.BLOCKED,
        "The supported Anima 3.8B model set was not detected.",
        "Run download_anima38_v11_int8_convrot_models.bat and review all upstream model licenses.",
        available=False,
    )


def _sensenova_check(paths: DiagnosticPaths) -> DiagnosticCheck:
    try:
        from modules_forge.sensenova_u15_bridge import (
            CONVROT_FILE_NAME,
            inspect_runtime,
        )

        status = inspect_runtime(
            paths.models_root / "SenseNova-U1" / "runtime-final",
            checkpoint=paths.models_root / "SenseNova-U1" / CONVROT_FILE_NAME,
        )
    except Exception:
        return DiagnosticCheck(
            "sensenova",
            "SenseNova U1.5",
            CheckState.BLOCKED,
            "SenseNova readiness could not be checked safely.",
            "Run download_sensenova_u15_int8.bat, then use the Studio status check.",
            available=False,
        )
    if status.ready:
        profile = "Quality and official 8-Step profiles" if status.lora_ready else "Quality profile"
        return DiagnosticCheck(
            "sensenova",
            "SenseNova U1.5",
            CheckState.READY,
            f"{profile} are available with the pinned runtime and checkpoint.",
            "No action is required.",
            available=True,
        )
    if status.partial_bytes:
        return DiagnosticCheck(
            "sensenova",
            "SenseNova U1.5",
            CheckState.WARNING,
            "A partial SenseNova download was detected.",
            "Run download_sensenova_u15_int8.bat again to resume and verify it.",
            available=False,
        )
    return DiagnosticCheck(
        "sensenova",
        "SenseNova U1.5",
        CheckState.BLOCKED,
        "The pinned SenseNova runtime or checkpoint is unavailable.",
        "Run download_sensenova_u15_int8.bat, review the model license, and retry the check.",
        available=False,
    )


def _implementation_check(
    paths: DiagnosticPaths,
    *,
    check_id: str,
    label: str,
    relative_files: Sequence[str],
    summary: str,
    missing_action: str,
) -> DiagnosticCheck:
    present = all((paths.root / relative).is_file() for relative in relative_files)
    if not present:
        return DiagnosticCheck(
            check_id,
            label,
            CheckState.BLOCKED,
            f"{label} implementation files are missing.",
            missing_action,
            available=False,
        )
    return DiagnosticCheck(
        check_id,
        label,
        CheckState.READY,
        summary,
        "No action is required.",
        available=True,
    )


def _minimax_h3_check(paths: DiagnosticPaths) -> DiagnosticCheck:
    implementation = (paths.root / "modules_forge/minimax_h3_bridge.py").is_file() and (
        paths.root / "extensions-builtin/minimax-h3-studio/scripts/minimax_h3_studio.py"
    ).is_file()
    if not implementation:
        return DiagnosticCheck(
            "minimax_h3",
            "MiniMax H3",
            CheckState.BLOCKED,
            "MiniMax H3 Studio implementation files are missing.",
            "Restore the built-in MiniMax H3 Studio files.",
            available=False,
        )
    try:
        from modules_forge.minimax_h3_bridge import (
            model_file_status,
        )
        from modules_forge.minimax_h3_runtime import installed_runtime_root

        runtime = installed_runtime_root(paths.root)
        files = model_file_status(runtime)
    except Exception:
        runtime = None
        files = {}
    if runtime is None:
        return DiagnosticCheck(
            "minimax_h3",
            "MiniMax H3",
            CheckState.WARNING,
            "H3 Studio is installed, but a compatible local ComfyUI runtime was not detected.",
            "Open H3 Studio and prepare the managed runtime and models.",
            available=False,
        )
    ready_files = sum(bool(value) for value in files.values())
    if ready_files != len(files):
        return DiagnosticCheck(
            "minimax_h3",
            "MiniMax H3",
            CheckState.WARNING,
            f"A local ComfyUI runtime was detected with {ready_files}/{len(files)} required H3 model files.",
            "Install the missing H3 files, then run the Studio status check.",
            available=False,
        )
    return DiagnosticCheck(
        "minimax_h3",
        "MiniMax H3",
        CheckState.WARNING,
        "A local ComfyUI runtime and all required H3 model files were detected; no live backend request was sent.",
        "Open H3 Studio and run its status check before generation.",
        available=True,
    )


def feature_checks(paths: DiagnosticPaths) -> tuple[DiagnosticCheck, ...]:
    """Return capability state without downloads, full hashes, or network calls."""

    return (
        _krea2_check(paths),
        _anima38_check(paths),
        _sensenova_check(paths),
        _minimax_h3_check(paths),
        _implementation_check(
            paths,
            check_id="forge_canvas",
            label="Forge Canvas",
            relative_files=(
                "modules_forge/forge_canvas/canvas.py",
                "modules_forge/forge_canvas/canvas.js",
            ),
            summary="Forge Canvas assets and integration code are installed.",
            missing_action="Restore the built-in Forge Canvas files.",
        ),
        _implementation_check(
            paths,
            check_id="hyperweave",
            label="HyperWeave",
            relative_files=(
                "extensions-builtin/hyperweave/hyperweave/engine.py",
                "extensions-builtin/hyperweave/scripts/hyperweave.py",
            ),
            summary="HyperWeave 4K/8K is installed and will validate the loaded model at run time.",
            missing_action="Restore the built-in HyperWeave extension files.",
        ),
    )


__all__ = ["feature_checks"]
