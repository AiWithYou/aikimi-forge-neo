"""Explicit installer; never called automatically during app launch or generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules_forge.jev_sparse.h3_integration import (
    COMFY_REVISION,
    PACK,
    SPARSE_BLOBS,
    git_blob,
    pack_files,
)


def environment():
    # Downloads/builds do not need user API credentials or PYTHONPATH overrides.
    return {
        k: v
        for k, v in os.environ.items()
        if not any(word in k.upper() for word in ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "CREDENTIAL"))
        and k.upper() not in {"PYTHONPATH", "PYTHONHOME"}
    }


def run(args, cwd=None):
    subprocess.run([str(x) for x in args], cwd=cwd, env=environment(), check=True)  # noqa: S603 -- explicit installer arguments, no shell


def python_in(env):
    return env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def install_sdk():
    env = ROOT / "repositories/jev-sdk"
    python = python_in(env)
    if not python.is_file():
        if env.exists():
            raise RuntimeError("Incomplete SDK environment exists; inspect it before retrying")
        run([sys.executable, "-m", "venv", env])
    run([python, "-m", "pip", "install", "typesafe-sdk==0.7.0"])
    run([python, "-m", "pip", "check"])
    print("SDK Python:", python)


def install_pack(root: Path):
    root = root.resolve(strict=True)
    sparse = root / "comfy_extras/nodes_sparse_attention.py"
    if not sparse.is_file() or git_blob(sparse.read_bytes().replace(b"\r\n", b"\n")) not in SPARSE_BLOBS:
        raise RuntimeError(
            "ComfyUI internal API mismatch. Use --create-h3-runtime; the normal H3 runtime is not upgraded."
        )
    if not (root / "main.py").is_file() or not (root / "models").is_dir():
        raise RuntimeError("Not a ComfyUI root")
    custom = root / "custom_nodes"
    if custom.is_symlink():
        raise RuntimeError("Symlinked custom_nodes directory is not accepted")
    custom.mkdir(exist_ok=True)
    target = custom / PACK
    files = pack_files()
    license_file = ROOT / "extensions-builtin/jev-sparse-experiments/LICENSE.h3"
    if license_file.is_file():
        files["LICENSE"] = license_file.read_bytes()
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    if target.exists():
        if target.is_symlink() or not (target / "aikimi-install.json").is_file():
            raise RuntimeError("Existing custom node directory is not managed by this installer")
        prior = json.loads((target / "aikimi-install.json").read_text())
        for name, digest in prior.items():
            path = target / name
            if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError("Local node edits detected; they will not be overwritten")
        unknown = [
            p
            for p in target.rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and p.relative_to(target).as_posix() not in set(prior) | {"aikimi-install.json"}
        ]
        if unknown:
            raise RuntimeError("Unknown files in managed node directory; refusing replacement")
    stage = Path(tempfile.mkdtemp(prefix=".aikimi-jev-", dir=custom))
    backup = target.with_name(PACK + ".previous")
    try:
        for name, data in files.items():
            path = stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (stage / "aikimi-install.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if backup.exists():
            raise RuntimeError("Previous install backup exists; inspect before retrying")
        if target.exists():
            target.rename(backup)
        try:
            stage.rename(target)
        except BaseException:
            if backup.exists():
                backup.rename(target)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print("Installed:", target)


def create_runtime(models: Path):
    model_root = models.resolve(strict=True)
    if not model_root.is_dir():
        raise ValueError("Shared models must be an existing directory")
    parent = ROOT / "repositories/minimax-h3-jev"
    comfy = parent / "ComfyUI"
    if parent.exists():
        raise RuntimeError("Experiment runtime already exists. Use --h3 --comfy-root to refresh only the node.")
    parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--no-checkout", "https://github.com/Comfy-Org/ComfyUI.git", comfy])
    run(["git", "checkout", "--detach", COMFY_REVISION], cwd=comfy)
    if (
        git_blob((comfy / "comfy_extras/nodes_sparse_attention.py").read_bytes().replace(b"\r\n", b"\n"))
        not in SPARSE_BLOBS
    ):
        raise RuntimeError("Pinned runtime source verification failed")
    env = parent / ".venv"
    run([sys.executable, "-m", "venv", env])
    python = python_in(env)
    run(
        [
            python,
            "-m",
            "pip",
            "install",
            "torch==2.11.0",
            "torchvision==0.26.0",
            "torchaudio==2.11.0",
            "--index-url",
            "https://download.pytorch.org/whl/cu130",
        ]
    )
    run([python, "-m", "pip", "install", "-r", comfy / "requirements.txt"])
    run([python, "-m", "pip", "check"])
    entry = {
        "base_path": str(model_root),
        "is_default": True,
        **{name: name for name in ("diffusion_models", "text_encoders", "vae", "loras", "model_patches")},
    }
    (comfy / "extra_model_paths.yaml").write_text(json.dumps({"aikimi_h3": entry}, indent=2), encoding="utf-8")
    with (parent / "installed-packages.txt").open("w", encoding="utf-8") as stream:
        subprocess.run([str(python), "-m", "pip", "freeze"], stdout=stream, env=environment(), check=True)  # noqa: S603 -- owned virtualenv
    install_pack(comfy)
    print("H3 Studioの実行環境に次を指定し、選択設定で再起動してください:", comfy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", action="store_true")
    parser.add_argument("--h3", action="store_true")
    parser.add_argument("--create-h3-runtime", action="store_true")
    parser.add_argument("--comfy-root", type=Path, default=ROOT / "repositories/minimax-h3/ComfyUI")
    parser.add_argument("--models", type=Path, default=ROOT / "models/MiniMax-H3")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not any((args.sdk, args.h3, args.create_h3_runtime)):
        parser.error("Select --sdk, --h3 or --create-h3-runtime")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "sdk": args.sdk,
                    "h3_node": args.h3,
                    "create_h3_runtime": args.create_h3_runtime,
                    "comfy_revision": COMFY_REVISION,
                    "comfy_root": str(args.comfy_root),
                    "shared_models": str(args.models),
                    "api_calls": 0,
                },
                indent=2,
            )
        )
        return
    if args.sdk:
        install_sdk()
    if args.create_h3_runtime:
        create_runtime(args.models)
    elif args.h3:
        install_pack(args.comfy_root)


if __name__ == "__main__":
    main()
