"""Install the pinned Qwen Image 2.1 runtime without changing Forge dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_ID = "Qwen/Qwen-Image-2.1"
MODEL_REVISION = "b3179ad355be050328e483a9dfdd9e60cd62adfa"
DIFFUSERS_REVISION = "6256aa7666cedd47443adc8f82da9a10e110b09c"
RUNTIME = ROOT / "models" / "Qwen-Image-2.1"
REQUIREMENTS = ROOT / "tools" / "requirements-qwen-image21.txt"
TERMS = f"https://huggingface.co/{MODEL_ID}/blob/{MODEL_REVISION}/LICENSE"


def execute(arguments, **kwargs):
    print("+", subprocess.list2cmdline([str(item) for item in arguments]), flush=True)  # noqa: T201
    return subprocess.run([str(item) for item in arguments], check=True, **kwargs)  # noqa: S603


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def install_environment(root):
    environment = root / "worker-env"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        venv.EnvBuilder(with_pip=True).create(environment)
    uv = shutil.which("uv")
    prefix = [uv, "pip", "install", "--python", python] if uv else [python, "-m", "pip", "install"]
    execute([*prefix, "torch==2.13.0", "torchvision==0.28.0", "--index-url", "https://download.pytorch.org/whl/cu130"])
    execute([*prefix, "-r", REQUIREMENTS], cwd=ROOT)
    execute([python, "-m", "pip", "check"])
    # Import the actual new classes. A version string alone cannot detect an older checkout.
    execute(
        [
            python,
            "-c",
            "from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel, AutoencoderKLQwenImage21, GGUFQuantizationConfig, FlowMatchEulerDiscreteScheduler; from transformers import Qwen3VLForConditionalGeneration; import bitsandbytes, gguf; print('Qwen Image 2.1 / Turbo imports OK')",
        ]
    )
    frozen = execute([python, "-m", "pip", "freeze"], capture_output=True, text=True, encoding="utf-8").stdout
    (root / "installed-requirements.txt").write_text(frozen, encoding="utf-8")
    return python.absolute()


def download_model(root, python, *, shared_only=False):
    # Download through the dedicated interpreter: the main environment is untouched.
    execute(
        [
            python,
            Path(__file__).resolve(),
            "--download-only",
            "--root",
            root,
            *(["--shared-only"] if shared_only else []),
        ]
    )


def fetch_and_verify(root, *, include_transformer=True):
    from huggingface_hub import HfApi, snapshot_download

    destination = root / "model"
    info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION, files_metadata=True)
    if info.sha != MODEL_REVISION:
        raise RuntimeError("モデルの固定リビジョンを確認できません。")
    files = [
        item
        for item in info.siblings
        if item.rfilename != ".gitattributes" and (include_transformer or not item.rfilename.startswith("transformer/"))
    ]
    snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=destination,
        allow_patterns=[item.rfilename for item in files],
        max_workers=4,
    )
    records = []
    for item in files:
        path = destination / item.rfilename
        if not path.resolve().is_relative_to(destination.resolve()) or not path.is_file():
            raise RuntimeError(f"モデルファイルが不足しています: {item.rfilename}")
        if item.size is not None and path.stat().st_size != item.size:
            raise RuntimeError(f"モデルサイズが一致しません: {item.rfilename}")
        actual_hash = sha256(path)
        expected_hash = getattr(item.lfs, "sha256", None) if item.lfs else None
        if expected_hash and actual_hash != expected_hash:
            raise RuntimeError(f"モデルのSHA-256が一致しません: {item.rfilename}")
        records.append({"path": item.rfilename, "size": path.stat().st_size, "sha256": actual_hash})
    from modules_forge.qwen_image21.core import atomic_json

    atomic_json(root / "model-files.json", {"revision": MODEL_REVISION, "files": records})


def prepare_rewriter(root, *, editing=False):
    """Build a reusable NF4 checkpoint from the pinned official weights."""
    from huggingface_hub import HfApi, snapshot_download

    from modules_forge.qwen_image21 import prompt_rewriter as rewriter
    from modules_forge.qwen_image21.core import atomic_json, inside
    from modules_forge.qwen_image21_environment import validate_running_versions

    validate_running_versions()
    model_id, revision, directory = rewriter.profile(editing)
    label = "編集補助" if editing else "プロンプト書き換え"
    destination = inside(root, root / directory)
    if destination != root.resolve() / directory:
        raise RuntimeError("書き換えモデルの保存先が別のフォルダーを指しています。リンク先を確認してください。")
    try:
        rewriter.rewriter_manifest(root, verify_hashes=True, editing=editing)
    except ValueError:
        pass
    else:
        print(f"{label}: 導入済みNF4モデルのSHA-256を確認しました。")  # noqa: T201
        return
    # Keep verified/resumable source downloads until conversion succeeds. Only
    # this installer's marked staging directory is reclaimed after publication.
    source_name = "edit-prompt-rewriter-source" if editing else "prompt-rewriter-source"
    source = inside(root, root / source_name)
    if source != root.resolve() / source_name:
        raise RuntimeError("一時モデルの保存先が別のフォルダーを指しています。リンク先を確認してください。")
    marker = source / "aikimi-source.json"
    expected = {"model": model_id, "revision": revision}
    if source.exists() and (not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != expected):
        raise RuntimeError(f"未登録の作業フォルダーがあります。内容を確認してください: {source}")
    source.mkdir(parents=True, exist_ok=True)
    atomic_json(marker, expected)
    info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise RuntimeError("書き換えモデルの固定リビジョンが一致しません。")
    files = [item for item in info.siblings if item.rfilename != ".gitattributes"]
    print("公式の書き換えモデルを取得します。量子化完了後は4bit版だけを保持します。", flush=True)  # noqa: T201
    snapshot_download(
        model_id,
        revision=revision,
        local_dir=source,
        allow_patterns=[item.rfilename for item in files],
        max_workers=4,
    )
    print(f"{label}: 配布ファイルのサイズとSHA-256を確認中", flush=True)  # noqa: T201
    source_weights = []
    for item in files:
        path = inside(source, source / item.rfilename)
        if not path.is_file() or (item.size is not None and path.stat().st_size != item.size):
            raise RuntimeError(f"書き換えモデルのサイズ不一致: {item.rfilename}")
        expected_hash = getattr(item.lfs, "sha256", None) if item.lfs else None
        if expected_hash and sha256(path) != expected_hash:
            raise RuntimeError(f"書き換えモデルのSHA-256不一致: {item.rfilename}")
        if path.suffix == ".safetensors":
            if not expected_hash:
                raise RuntimeError(f"公式重みのSHA-256を確認できません: {item.rfilename}")
            source_weights.append({"path": item.rfilename, "size": path.stat().st_size, "sha256": expected_hash})
            print(f"SHA-256一致: {item.rfilename}", flush=True)  # noqa: T201
    import gc

    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("4bit量子化にはBF16対応のCUDA GPUが必要です。")
    print(f"{label}: テキスト部分と出力ヘッドを4bit NF4・二重量子化へ変換中", flush=True)  # noqa: T201
    with TemporaryDirectory(prefix=f".{directory}-", dir=root) as temporary:
        staging = inside(root, Path(temporary))
        build = staging / "model"
        model_class = AutoModelForImageTextToText if editing else AutoModelForCausalLM
        model = None
        try:
            model = model_class.from_pretrained(
                str(source),
                dtype=torch.bfloat16,
                quantization_config=rewriter.quantization_config(editing=editing),
                device_map={"": "cuda:0"},
                local_files_only=True,
                use_safetensors=True,
                trust_remote_code=False,
            ).eval()
            layers = rewriter.check_quantized_model(model, editing=editing)
            footprint = model.get_memory_footprint()
            model.save_pretrained(build, safe_serialization=True, max_shard_size="4GB")
            processor_class = AutoProcessor if editing else AutoTokenizer
            processor_class.from_pretrained(
                str(source), local_files_only=True, trust_remote_code=False
            ).save_pretrained(build)
            for name in ("system_prompt.txt", "LICENSE", "README.md"):
                shutil.copy2(source / name, build / name)
            model = None
            gc.collect()
            torch.cuda.empty_cache()
            # Validate the persisted checkpoint, not just its in-memory source.
            model = model_class.from_pretrained(
                str(build),
                dtype=torch.bfloat16,
                device_map={"": "cuda:0"},
                local_files_only=True,
                use_safetensors=True,
                trust_remote_code=False,
            ).eval()
            if rewriter.check_quantized_model(model, editing=editing) != layers:
                raise RuntimeError("保存後の書き換えモデルで4bit層数が変わりました。")
        finally:
            model = None
            gc.collect()
            torch.cuda.empty_cache()
        records = [
            {"path": str(path.relative_to(build)), "size": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(build.rglob("*"))
            if path.is_file()
        ]
        atomic_json(
            build / "rewriter-files.json",
            {
                "schema": 1,
                "model": model_id,
                "revision": revision,
                "quantization": "nf4-double",
                "linear4bit_layers": layers,
                "model_footprint_bytes": footprint,
                "text_only": not editing,
                "vision_precision": "bf16" if editing else None,
                "reload_verified": True,
                "source_weights": source_weights,
                "files": records,
            },
        )
        previous = staging / "previous"
        if destination.exists():
            destination.rename(previous)
        try:
            build.rename(destination)
            rewriter.rewriter_manifest(root, verify_hashes=True, editing=editing)
        except BaseException:
            if destination.exists():
                destination.rename(staging / "failed")
            if previous.exists():
                previous.rename(destination)
            raise
    # Both paths were resolved and checked inside the dedicated runtime. Never
    # remove an existing Hub cache or any source outside our marked directory.
    if (
        inside(root, source) != root.resolve() / source_name
        or not marker.is_file()
        or json.loads(marker.read_text(encoding="utf-8")) != expected
    ):
        raise RuntimeError("一時モデルの保存先が変わったため削除を中止しました。")
    shutil.rmtree(source)
    print(f"{label}NF4モデルの準備完了: {footprint / 2**30:.2f} GiB / {layers}層", flush=True)  # noqa: T201


def main(argv=None):
    parser = argparse.ArgumentParser(description="Qwen Image 2.1専用環境とモデルを準備")
    parser.add_argument("--root", type=Path, default=RUNTIME)
    parser.add_argument("--runtime-only", action="store_true", help="依存環境のみ導入。モデルは取得しません")
    parser.add_argument("--verify", action="store_true", help="既存ファイルのサイズとSHA-256を再検証")
    parser.add_argument("--dry-run", action="store_true", help="導入内容のみ表示")
    parser.add_argument("--download-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--shared-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--download-regular-gguf", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--download-turbo-profile", choices=["turbo_bf16", "turbo_q4_k_m"], help=argparse.SUPPRESS)
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--official-full", action="store_true", help="公式フルモデルを導入（INT8 / W4A8 / BF16用）"
    )
    model_group.add_argument("--turbo-bf16-only", action="store_true", help="Viggle Turbo BF16と共通部品を導入")
    model_group.add_argument(
        "--turbo-q4-only",
        "--turbo-q4-k-m-only",
        dest="turbo_q4_only",
        action="store_true",
        help="Viggle Turbo Q4_K_Mと共通部品を導入",
    )
    parser.add_argument(
        "--prompt-rewriter-only", action="store_true", help="任意のプロンプト書き換えモデルだけを4bitで追加"
    )
    parser.add_argument(
        "--with-prompt-rewriter", action="store_true", help="画像モデルに加えて4bit書き換えモデルも導入"
    )
    parser.add_argument(
        "--edit-prompt-rewriter-only", action="store_true", help="任意の画像編集補助モデルだけを4bitで追加"
    )
    parser.add_argument(
        "--with-edit-prompt-rewriter", action="store_true", help="画像モデルに加えて4bit画像編集補助モデルも導入"
    )
    parser.add_argument("--prepare-rewriter", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--prepare-edit-rewriter", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    turbo_profile = "turbo_bf16" if args.turbo_bf16_only else "turbo_q4_k_m" if args.turbo_q4_only else None
    only_rewriters = args.prompt_rewriter_only or args.edit_prompt_rewriter_only
    include_rewriter = args.prompt_rewriter_only or args.with_prompt_rewriter
    include_edit_rewriter = args.edit_prompt_rewriter_only or args.with_edit_prompt_rewriter
    if args.runtime_only and (include_rewriter or include_edit_rewriter):
        parser.error("--runtime-only と書き換えモデルの導入は同時に指定できません。")
    if args.runtime_only and args.official_full:
        parser.error("--runtime-only と --official-full は同時に指定できません。")
    if only_rewriters and args.official_full:
        parser.error("書き換えモデルだけの導入と --official-full は同時に指定できません。")
    if turbo_profile and (args.runtime_only or include_rewriter or include_edit_rewriter):
        parser.error("Turbo単独導入と他の導入オプションは同時に指定できません。")
    root = args.root.expanduser().absolute()
    if args.dry_run:
        from modules_forge.qwen_image21 import prompt_rewriter as rewriter

        selected_model, selected_revision, _ = rewriter.profile(args.edit_prompt_rewriter_only)
        if turbo_profile:
            from modules_forge.qwen_image21.turbo import PROFILES

            selected_model, selected_revision, _ = PROFILES[turbo_profile]
        elif not args.official_full and not only_rewriters:
            from modules_forge.qwen_image21.regular_gguf import MODEL_ID as GGUF_MODEL_ID
            from modules_forge.qwen_image21.regular_gguf import REVISION as GGUF_REVISION

            selected_model, selected_revision = GGUF_MODEL_ID, GGUF_REVISION
        print(  # noqa: T201 -- Explicit installer dry-run output.
            json.dumps(
                {
                    "runtime": str(root),
                    "model": selected_model if only_rewriters or turbo_profile or not args.official_full else MODEL_ID,
                    "model_revision": selected_revision
                    if only_rewriters or turbo_profile or not args.official_full
                    else MODEL_REVISION,
                    "diffusers_revision": DIFFUSERS_REVISION,
                    "model_bytes": (
                        None
                        if only_rewriters
                        else 14_230_284_584
                        if turbo_profile == "turbo_bf16"
                        else 4_189_346_592
                        if turbo_profile == "turbo_q4_k_m"
                        else 33_131_614_660
                        if args.official_full
                        else 4_199_565_024
                    ),
                    "shared_model_bytes": 18_901_299_599
                    if not only_rewriters and (turbo_profile or not args.official_full)
                    else None,
                    "precision": ["nf4-double"]
                    if only_rewriters
                    else [turbo_profile]
                    if turbo_profile
                    else ["int8", "w4a8", "bf16"]
                    if args.official_full
                    else ["base_q4_k_m"],
                    "license": (
                        f"https://huggingface.co/{selected_model}/blob/{selected_revision}/LICENSE"
                        if only_rewriters
                        else "https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo/blob/bafc91e4cc934f5fb1406b22496a0bed9b99c548/LICENSE"
                        if turbo_profile
                        else TERMS
                    ),
                    "prompt_rewriter": {
                        "model": rewriter.MODEL_ID,
                        "revision": rewriter.MODEL_REVISION,
                        "precision": "4bit NF4 + double quantization",
                        "source_download_gb_approx": 19,
                        "text_only": True,
                        "quantize_output_head": True,
                    }
                    if include_rewriter
                    else None,
                    "edit_prompt_rewriter": {
                        "model": rewriter.EDIT_MODEL_ID,
                        "revision": rewriter.EDIT_MODEL_REVISION,
                        "precision": "4bit NF4 + double quantization",
                        "source_download_gb_approx": 19,
                        "text_only": False,
                        "vision_precision": "bf16",
                        "quantize_output_head": True,
                    }
                    if include_edit_rewriter
                    else None,
                },
                indent=2,
            )
        )
        return 0
    if args.download_only:
        fetch_and_verify(root, include_transformer=not args.shared_only)
        return 0
    if args.download_regular_gguf:
        from modules_forge.qwen_image21.regular_gguf import download_regular

        download_regular(root)
        return 0
    if args.download_turbo_profile:
        from modules_forge.qwen_image21.turbo import download_turbo

        download_turbo(root, args.download_turbo_profile)
        return 0
    if args.prepare_rewriter or args.prepare_edit_rewriter:
        prepare_rewriter(root, editing=args.prepare_edit_rewriter)
        return 0
    from modules_forge.qwen_image21.core import QwenImage21Error, atomic_json, read_json, runtime_lock, runtime_manifest

    root.mkdir(parents=True, exist_ok=True)
    lock = runtime_lock(root)
    try:
        if only_rewriters:
            from modules_forge.qwen_image21.prompt_rewriter import rewriter_manifest
            from modules_forge.qwen_image21_environment import environment_status

            if args.verify:
                if include_rewriter:
                    rewriter_manifest(root, verify_hashes=True)
                if include_edit_rewriter:
                    rewriter_manifest(root, verify_hashes=True, editing=True)
                print("書き換え4bitモデルのSHA-256を確認しました。")  # noqa: T201
                return 0
            ready, _ = environment_status(root)
            python = root / "worker-env" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            if not ready:
                python = install_environment(root)
            for enabled, flag in (
                (include_rewriter, "--prepare-rewriter"),
                (include_edit_rewriter, "--prepare-edit-rewriter"),
            ):
                if enabled:
                    execute([python, "-X", "utf8", Path(__file__).resolve(), flag, "--root", root])
            return 0
        if args.verify:
            selected_precision = turbo_profile or ("int8" if args.official_full else "base_q4_k_m")
            runtime_manifest(root, selected_precision)
            record = read_json(root / "model-files.json")
            if record.get("revision") != MODEL_REVISION or not record.get("files"):
                raise RuntimeError("固定モデルの検証記録がありません。再セットアップしてください。")
            for item in record["files"]:
                path = root / "model" / item["path"]
                if not path.resolve().is_relative_to((root / "model").resolve()):
                    raise ValueError("検証記録に不正なパスがあります。")
                if not path.is_file() or path.stat().st_size != item["size"] or sha256(path) != item["sha256"]:
                    raise RuntimeError(f"モデル検証失敗: {item['path']}")
            if turbo_profile:
                from modules_forge.qwen_image21.turbo import turbo_manifest

                turbo_manifest(root, turbo_profile, verify_hashes=True)
            elif not args.official_full:
                from modules_forge.qwen_image21.regular_gguf import regular_manifest

                regular_manifest(root, verify_hashes=True)
            print("Qwen Image 2.1: 全モデルファイルのSHA-256が一致しました。")  # noqa: T201
            if include_rewriter or include_edit_rewriter:
                from modules_forge.qwen_image21.prompt_rewriter import rewriter_manifest

                if include_rewriter:
                    rewriter_manifest(root, verify_hashes=True)
                if include_edit_rewriter:
                    rewriter_manifest(root, verify_hashes=True, editing=True)
                print("書き換え4bitモデルのSHA-256を確認しました。")  # noqa: T201
            return 0
        print(f"Qwen Image 2.1の利用条件: {TERMS}")  # noqa: T201
        python = install_environment(root)
        if not args.runtime_only:
            if turbo_profile:
                for existing in ("int8", "base_q4_k_m", "turbo_bf16", "turbo_q4_k_m"):
                    try:
                        runtime_manifest(root, existing)
                    except QwenImage21Error:
                        continue
                    break
                else:
                    download_model(root, python, shared_only=True)
            elif args.official_full:
                download_model(root, python)
            else:
                for existing in ("int8", "base_q4_k_m", "turbo_bf16", "turbo_q4_k_m"):
                    try:
                        runtime_manifest(root, existing)
                    except QwenImage21Error:
                        continue
                    break
                else:
                    download_model(root, python, shared_only=True)
            atomic_json(
                root / "runtime.json",
                {
                    "schema": 1,
                    "python": str(python),
                    "model": str(root / "model"),
                    "model_revision": MODEL_REVISION,
                    "diffusers_revision": DIFFUSERS_REVISION,
                },
            )
            if turbo_profile:
                execute(
                    [
                        python,
                        "-X",
                        "utf8",
                        Path(__file__).resolve(),
                        "--download-turbo-profile",
                        turbo_profile,
                        "--root",
                        root,
                    ]
                )
                runtime_manifest(root, turbo_profile)
            elif not args.official_full:
                execute([python, "-X", "utf8", Path(__file__).resolve(), "--download-regular-gguf", "--root", root])
                runtime_manifest(root, "base_q4_k_m")
            print("準備完了。Neoを起動してQwen Image 2.1を開いてください。")  # noqa: T201
        for enabled, flag in (
            (include_rewriter, "--prepare-rewriter"),
            (include_edit_rewriter, "--prepare-edit-rewriter"),
        ):
            if enabled:
                execute([python, "-X", "utf8", Path(__file__).resolve(), flag, "--root", root])
        return 0
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
