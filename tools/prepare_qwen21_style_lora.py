"""Download one Hugging Face safetensors LoRA into Qwen's local library.

Run using models/Qwen-Image-2.1/worker-env/Scripts/python.exe.
Only weights are downloaded; optimizer checkpoints and repository code are not used.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from huggingface_hub import HfApi, hf_hub_download

    from modules_forge.qwen_image21.core import atomic_json
    from modules_forge.qwen_image21.style_lora import fingerprint, inspect

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", help="Hugging Face owner/repository")
    parser.add_argument("--file", help="Repository-relative .safetensors filename")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", args.repo):
        parser.error("repo must be owner/repository")
    info = HfApi().model_info(args.repo, revision=args.revision, files_metadata=True)
    candidates = [item for item in info.siblings if item.rfilename.endswith(".safetensors")]
    if args.file:
        candidates = [item for item in candidates if item.rfilename == args.file]
    if len(candidates) != 1:
        parser.error("Use --file to select one .safetensors file: " + ", ".join(item.rfilename for item in candidates))
    item = candidates[0]
    runtime = ROOT / "models/Qwen-Image-2.1"
    folder = runtime / "loras" / args.repo.replace("/", "--")
    destination = (folder / item.rfilename).resolve()
    if not destination.is_relative_to(folder.resolve()):
        parser.error("Invalid repository filename")
    # hf_hub_download uses an atomic completion and its normal authentication.
    path = Path(hf_hub_download(args.repo, item.rfilename, revision=info.sha, local_dir=folder))
    digest = fingerprint(path)
    expected = getattr(item.lfs, "sha256", None) if item.lfs else None
    if expected and digest != expected:
        raise ValueError("Downloaded LoRA SHA-256 does not match Hugging Face.")
    name = path.relative_to(runtime / "loras").as_posix()
    checked = inspect(runtime, name)
    receipt = {
        "repository": args.repo,
        "revision": info.sha,
        "file": item.rfilename,
        "sha256": digest,
        "size": path.stat().st_size,
        "linear_layers": checked["linear_layers"],
        "base_mismatch": checked["base_mismatch"],
    }
    atomic_json(path.with_suffix(".source.json"), receipt)
    sys.stdout.write(json.dumps({"installed": name, **receipt}, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
