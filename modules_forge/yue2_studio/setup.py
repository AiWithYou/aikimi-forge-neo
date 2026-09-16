"""Explicit, separate installer; never called by Forge startup or generation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import venv
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from modules_forge.yue2_studio.core import (  # noqa: E402
    GGUF_ID, GGUF_MODELS, MODEL_ID, SIDECARS, SOURCE_REVISION, VAE_ID,
    YuE2Error, atomic_json, read_json, required_cpp_files, runtime_lock, safe_environment,
)

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "extensions-builtin" / "yue2-studio" / "runtime"
TERMS = "https://huggingface.co/m-a-p/YuE2-3B/discussions/5"


def execute(args, **kwargs):
    print("+", subprocess.list2cmdline([str(a) for a in args]), flush=True)
    return subprocess.run([str(a) for a in args], check=True, **kwargs)


def environment(root: Path) -> Path:
    python = root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        venv.EnvBuilder(with_pip=True).create(root)
    # Do not resolve the venv interpreter symlink: doing so can bypass the venv on POSIX.
    return python.absolute()


def download(repo: str, destination: Path, patterns: list[str], python: Path) -> str:
    # The helper runs in the dedicated environment, never the Forge environment.
    code = (
        "import json,sys; from huggingface_hub import HfApi,snapshot_download; "
        "r,d,p=sys.argv[1:]; s=HfApi().model_info(r).sha; "
        "snapshot_download(r,revision=s,local_dir=d,allow_patterns=json.loads(p)); print(s)"
    )
    result = execute([python, "-c", code, repo, destination, json.dumps(patterns)],
                     stdout=subprocess.PIPE, text=True, encoding="utf-8")
    revision = result.stdout.strip().splitlines()[-1]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise YuE2Error("モデルのリビジョンを取得できませんでした。")
    return revision


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="YuE2の専用環境を導入します。重みは別途ダウンロードされます。")
    parser.add_argument("--engine", choices=["official", "cpp"])
    parser.add_argument("--accept-model-terms", action="store_true")
    parser.add_argument("--binary", type=Path, help="audio.cppのaudiocpp_cli.exeのパス（DLLも同じフォルダーへ）")
    parser.add_argument("--models", type=Path, help="既存GGUFディレクトリ。省略時は必要分を取得")
    parser.add_argument("--gguf", choices=list(GGUF_MODELS), default="q8_0")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        raise YuE2Error("セットアップはPython 3.12で実行してください。Windows: py -3.12 modules_forge/yue2_studio/setup.py")
    if args.engine is None:
        answer = input("1: 公式Python（標準） / 2: audio.cpp登録（GGUF） [1]: ").strip()
        if answer not in {"", "1", "2"}:
            raise YuE2Error("1または2を指定してください。")
        args.engine = "cpp" if answer == "2" else "official"
    print("モデル表示はCC BY-NC 4.0です。個人の出力収益化については公式回答を確認してください。")
    print(TERMS)
    print("企業の商用利用や、第三者の曲・追加モデルの権利まで自由になるものではありません。")
    print("専用環境の作成と数GB以上のモデル取得を行います。既存Forgeの依存関係は変更しません。")
    if not args.accept_model_terms:
        if input("利用条件を確認して導入しますか？ [y/N]: ").strip().lower() != "y":
            print("変更せず終了しました。")
            return 0
    RUNTIME.mkdir(parents=True, exist_ok=True)
    lock = runtime_lock(RUNTIME)
    try:
        # This OS lock is held for the complete install or inference lifetime.
        manifest_path = RUNTIME / "runtime.json"
        manifest = read_json(manifest_path) if manifest_path.exists() else {"schema": 1}
        if args.engine == "official":
            python = environment(RUNTIME / "official")
            execute([python, "-m", "pip", "install", "--upgrade", "pip"])
            execute([python, "-m", "pip", "install", "torch==2.10.0", "--index-url", "https://download.pytorch.org/whl/cu128"])
            execute([python, "-m", "pip", "install", f"git+https://github.com/multimodal-art-projection/YuE.git@{SOURCE_REVISION}"])
            execute([python, "-m", "pip", "check"])
            execute([python, "-c", "import torch; from yue2 import YuE2Pipeline; from yue2.modeling_yue2 import YuE2ForCausalLM; print('CUDA:',torch.cuda.is_available()); print('PyTorch:',torch.__version__)"], env=safe_environment())
            patterns = ["*.json", "*.safetensors", "qwen.tiktoken", "LICENSE*", "MODEL_LICENSE*", "THIRD_PARTY_NOTICES.md", "licenses/*"]
            model, vae = RUNTIME / "models" / "YuE2-3B", RUNTIME / "models" / "YuE2-Vae"
            model_rev = download(MODEL_ID, model, patterns, python)
            vae_rev = download(VAE_ID, vae, patterns, python)
            for directory in (model, vae):
                if not (directory / "config.json").is_file() or not list(directory.glob("*.safetensors")):
                    raise YuE2Error(f"モデルの取得が完了していません: {directory}")
            manifest["official"] = {"python": str(python), "model": str(model), "vae": str(vae),
                                    "source_revision": SOURCE_REVISION, "model_revision": model_rev, "vae_revision": vae_rev}
            frozen = execute([python, "-m", "pip", "freeze"], stdout=subprocess.PIPE, text=True, encoding="utf-8").stdout
            (RUNTIME / "official-requirements.lock.txt").write_text(frozen, encoding="utf-8")
        else:
            if args.binary is None:
                args.binary = Path(input("audiocpp_cli.exeのフルパス: ").strip().strip('"'))
            binary = args.binary.expanduser().resolve()
            if not binary.is_file() or binary.suffix.lower() in {".bat", ".cmd", ".ps1"}:
                raise YuE2Error("audio.cppのネイティブ実行ファイルを指定してください。")
            if args.models:
                models = args.models.expanduser().resolve()
                revision = "local-user-supplied"
            else:
                python = environment(RUNTIME / "downloader")
                execute([python, "-m", "pip", "install", "huggingface-hub==0.36.2"])
                models = RUNTIME / "models" / "YuE2-GGUF"
                patterns = [GGUF_MODELS[args.gguf], "yue2-vae-f16.gguf", "sidecars/*", "README.md", "LICENSE*", "THIRD_PARTY_NOTICES.md"]
                revision = download(GGUF_ID, models, patterns, python)
            missing = [p.name for p in required_cpp_files(models, args.gguf) if not p.is_file()]
            if missing:
                raise YuE2Error("GGUFの不足ファイル: " + ", ".join(missing))
            manifest["cpp"] = {"binary": str(binary), "binary_sha256": digest(binary), "models": str(models),
                               "model_revision": revision, "interface": "audio.cpp v0.8.0-compatible", "gguf": args.gguf}
        manifest["installed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["terms_acknowledged"] = TERMS
        atomic_json(manifest_path, manifest)
        print("登録しました。Forgeを起動し、YuE2 Musicタブで導入状態を確認してください。GPU生成成功を示すものではありません。")
    finally:
        lock.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (YuE2Error, OSError, subprocess.CalledProcessError) as exc:
        print(f"セットアップ未完了: {exc}", file=sys.stderr)
        raise SystemExit(1)
