"""Install the pinned official project and weights in a dedicated environment."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules_forge.irodori.core import (  # noqa: E402
    CODE_REVISION,
    CODEC_REPO,
    CODEC_REVISION,
    MODEL_REPO,
    MODEL_REVISION,
    MODELS,
    RUNTIME,
    WATERMARK_REPO,
    WATERMARK_REVISION,
    atomic_json,
    child_environment,
    python_path,
)


def run(command, *, offline=False, **kwargs):
    return subprocess.run(  # noqa: S603 -- Fixed tools and validated repository paths, no shell.
        [str(x) for x in command], check=True, cwd=ROOT, env=child_environment(offline=offline), **kwargs
    )


def sync_command(uv, source):
    return [uv, "sync", "--frozen", "--extra", "cu128", "--project", source, "--python", "3.11"]


def download_plan():
    return [
        {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "directory": "v4-Large",
            "files": [
                "model.safetensors",
                "tokenizer/tokenizer.json",
                "tokenizer/tokenizer_config.json",
                "README.md",
                "NOTICE",
                "GEMMA_TERMS_OF_USE.md",
                "GEMMA_PROHIBITED_USE_POLICY.md",
                "EMOJI_ANNOTATIONS.md",
            ],
        },
        {"repo": CODEC_REPO, "revision": CODEC_REVISION, "directory": "codec", "files": ["weights.pth", "README.md"]},
        {
            "repo": WATERMARK_REPO,
            "revision": WATERMARK_REVISION,
            "directory": "watermark",
            "files": [
                "44_1_khz/73999_iteration/enc_c.ckpt",
                "44_1_khz/73999_iteration/dec_c.ckpt",
                "44_1_khz/73999_iteration/dec_m_0.ckpt",
                "44_1_khz/73999_iteration/hparams.yaml",
            ],
        },
    ]


def uv_path(root):
    existing = shutil.which("uv")
    if existing:
        return Path(existing)
    bootstrap = Path(root) / "bootstrap"
    binary = bootstrap / "bin" / ("uv.exe" if os.name == "nt" else "uv")
    if not binary.is_file():
        run(
            [sys.executable, "-m", "pip", "install", "--target", bootstrap, "uv==0.13.1", "--disable-pip-version-check"]
        )
    return binary


def install(root=RUNTIME, models=MODELS):
    root, models = Path(root).resolve(), Path(models).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Remove the readiness marker before any mutation, including failed repairs.
    (root / "runtime.json").unlink(missing_ok=True)
    source = root / "source"
    if not source.exists():
        run(["git", "clone", "--filter=blob:none", "https://github.com/Aratako/Irodori-TTS.git", source])
        run(["git", "-C", source, "checkout", "--detach", CODE_REVISION])
    revision = run(["git", "-C", source, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if revision != CODE_REVISION:
        raise ValueError("Irodori公式ソースのcommitが一致しません。別の実行環境を指定してください。")
    changes = run(["git", "-C", source, "diff", "--name-only", "HEAD"], capture_output=True, text=True).stdout.strip()
    if changes:
        raise ValueError("Irodori公式ソースが変更されています。専用環境を確認してください。")
    uv = uv_path(root)
    run(sync_command(uv, source))
    # Use the installed hub library rather than importing inference dependencies
    # into the Forge environment. Verify remote sizes and LFS SHA-256 hashes.
    plan = download_plan()
    atomic_json(root / "download-plan.json", plan)
    download = """
import hashlib,json,sys
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download
root,models=map(Path,sys.argv[1:])
records=[]
for item in json.loads((root/'download-plan.json').read_text(encoding='utf-8')):
    info=HfApi().model_info(item['repo'],revision=item['revision'],files_metadata=True)
    for name in item['files']:
        path=Path(hf_hub_download(item['repo'],name,revision=item['revision'],local_dir=models/item['directory']))
        remote=next(x for x in info.siblings if x.rfilename==name)
        with path.open('rb') as stream:
            digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if path.stat().st_size!=remote.size or (remote.lfs and digest!=remote.lfs.sha256):
            raise ValueError('Downloaded file verification failed: '+name)
        records.append({'file':path.relative_to(models).as_posix(), 'size':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns,'sha256':digest})
(root/'model-files.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
"""
    run([python_path(root), "-c", download, root, models])
    # Verify local SilentCipher weights with the network disabled, using the
    # same adapter as inference rather than relying on a global HF cache.
    run(
        [
            python_path(root),
            "-c",
            "import sys; sys.path[:0]=sys.argv[1:3]; import torch; from tools.irodori_worker import local_watermark; from irodori_tts.watermark import SilentCipherWatermarker; assert torch.cuda.is_available(), 'CUDA GPU is required';\nwith local_watermark(sys.argv[3]):\n w=SilentCipherWatermarker(device='cpu'); assert w.ready, 'SilentCipher is unavailable'; print('SilentCipher ready:',w.ready)",
            source,
            ROOT,
            models / "watermark",
        ],
        offline=True,
    )
    run([uv, "pip", "check", "--python", python_path(root)])
    frozen = run([uv, "pip", "freeze", "--python", python_path(root)], capture_output=True, text=True).stdout
    (root / "installed-requirements.txt").write_text(frozen, encoding="utf-8")
    atomic_json(
        root / "runtime.json",
        {
            "code_revision": CODE_REVISION,
            "model_revision": MODEL_REVISION,
            "codec_revision": CODEC_REVISION,
            "watermark_revision": WATERMARK_REVISION,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RUNTIME)
    parser.add_argument("--models", type=Path, default=MODELS)
    args = parser.parse_args()
    install(args.root, args.models)


if __name__ == "__main__":
    main()
