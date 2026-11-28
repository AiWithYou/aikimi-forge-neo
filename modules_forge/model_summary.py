"""Count weight files in managed checkpoint and LoRA directories without loading models."""

from pathlib import Path

from modules.aikimi_diagnostics import default_paths, feature_checks
from modules_forge.model_catalog import FORGE_FAMILIES, STUDIO_FAMILIES

FAMILIES = {
    "Anima": "anima38",
    "Krea2": "krea2",
    "SDXL": "forge",
    "Stable-diffusion": "forge",
    "Lora": "forge",
    "Qwen-Image-2.1": "qwen_image21",
    "Flux-3-Klein": "flux3",
    "MOSS-Image": "moss_image",
    "Ming-Image": "ming_image",
    "Clef": "clef",
    "MiniMax-H3": "minimax_h3",
    "YuE2": "yue2",
    "Irodori-TTS": "irodori",
}
EXCLUDED = {
    ".cache",
    ".git",
    ".venv",
    "venv",
    "worker-env",
    "site-packages",
    "repositories",
    "comfyui",
    "text_encoder",
    "text_encoder_2",
    "text_encoders",
    "vae",
    "tokenizer",
    "__pycache__",
}


def _counts():
    return {"checkpoints": 0, "loras": 0, "total_bytes": 0}


def inventory(root):
    totals = _counts()
    studios = {}
    for family in ("Stable-diffusion", "Lora", *FORGE_FAMILIES, *STUDIO_FAMILIES, "Irodori-TTS"):
        directory = Path(root) / family
        if not directory.is_dir() or directory.is_symlink() or directory.is_junction():
            continue
        for folder, dirs, files in directory.walk():
            dirs[:] = [d for d in dirs if d.casefold() not in EXCLUDED and not (folder / d).is_junction()]
            parts = {p.casefold() for p in folder.relative_to(root).parts}
            if parts & {"lora", "loras", "lycoris"}:
                kind = "loras"
            elif parts & {"checkpoints", "stable-diffusion", "diffusion_models", "transformer", "unet"}:
                kind = "checkpoints"
            else:
                continue
            for name in files:
                path = folder / name
                if path.suffix.lower() not in {".safetensors", ".ckpt", ".gguf", ".pt"} or path.is_symlink():
                    continue
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                studio = FAMILIES[family]
                if studio == "minimax_h3" and name.casefold().startswith(("p-h3-image", "minimax_h3_image")):
                    studio = "minimax_h3_image"
                counts = studios.setdefault(studio, _counts())
                counts[kind] += 1
                counts["total_bytes"] += size
                totals[kind] += 1
                totals["total_bytes"] += size
    return {**totals, "studios": studios}


def model_summary(paths=None):
    paths = paths or default_paths()
    counts = inventory(paths.models_root)
    features = {
        check.id: {**check.public_dict(), **counts["studios"].get(check.id, _counts())}
        for check in feature_checks(paths)
    }
    return {**counts, "features": features}
