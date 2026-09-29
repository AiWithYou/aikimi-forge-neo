"""Create an optional Ming DiT W4A8 checkpoint from the pinned BF16 original.

Run with Neo closed. The standard INT8 model is retained. No downloads on import.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import socket
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules_forge.ming_image_studio import W4A8_MODEL, W4A8_SOURCE_SHA256  # noqa: E402
from modules_forge.minimax_h3_runtime import setup_lock  # noqa: E402
from tools.aikimi_setup import ArtifactSpec, Installer, ProfileSpec, SetupError  # noqa: E402
from tools.setup_ming_image import runtime_ready  # noqa: E402

SOURCE_SHA256 = W4A8_SOURCE_SHA256
SOURCE_SIZE = 12309858664
WEIGHTS_REVISION = "53654871e47a5d2daed7b3a986cbf1010ef81c78"
QUANTIZER_REVISION = "d6797787e6bdb1a1fb0094d588a26f8e71a1c757"
QUANTIZER_SHA256 = "dd85ca376b985222719f052acbe47081c6fdf54742791ac1b44209923828146a"
QUANTIZER_SIZE = 24249
OUTPUT_NAME = W4A8_MODEL
CACHE = "repositories/ming-image/quantization"
ATTENTION_CONFIG = b'{"attention": "comfy_kitchen_int8"}'


class NativeMingReader:
    """Expose the BF16 source in Comfy's native namespace, fusing Q/K/V before quantization.

    Fusion happens per requested tensor; no extra 12.3 GB intermediate is written.
    The mapping comes from the pinned runtime; attention metadata follows the INT8 reference.
    """

    def __init__(self, path):
        from safetensors import safe_open

        sys.path.insert(0, str(ROOT / "repositories/ming-image/ComfyUI"))
        from comfy.utils import z_image_to_diffusers

        self.source = safe_open(str(path), framework="pt", device="cpu")
        mapping = z_image_to_diffusers({"n_layers": 30, "dim": 3840})
        self.parts = {}
        for key in self.source.keys():
            target = mapping.get(key, key)
            if isinstance(target, str):
                self.parts[target] = [(0, key)]
            else:
                destination, (axis, offset, length) = target
                if axis != 0 or self.source.get_slice(key).get_shape()[0] != length:
                    raise SetupError(f"未対応の重み結合です: {key}")
                self.parts.setdefault(destination, []).append((offset, key))
        self.attention = {
            key.removesuffix("qkv.weight") + "comfy_attention.config"
            for key in self.parts
            if key.endswith(".attention.qkv.weight")
        }

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.source.__exit__(*args)
        return False

    def metadata(self):
        return self.source.metadata()

    def keys(self):
        return list(self.parts) + sorted(self.attention)

    def get_slice(self, key):
        if key in self.attention:
            return SimpleNamespace(get_shape=lambda: [len(ATTENTION_CONFIG)], get_dtype=lambda: "U8")
        parts = sorted(self.parts[key])
        original = self.source.get_slice(parts[0][1])
        shape = original.get_shape()
        if len(parts) > 1:
            shape[0] = sum(self.source.get_slice(name).get_shape()[0] for _, name in parts)
        return SimpleNamespace(get_shape=lambda: shape, get_dtype=original.get_dtype)

    def get_tensor(self, key):
        import torch

        if key in self.attention:
            return torch.tensor(list(ATTENTION_CONFIG), dtype=torch.uint8)
        tensors = [self.source.get_tensor(name) for _, name in sorted(self.parts[key])]
        return torch.cat(tensors, dim=0) if len(tensors) > 1 else tensors[0]


def quantize_worker(quantizer: Path, source: Path, output: Path, report: Path) -> None:
    if file_hash(quantizer) != QUANTIZER_SHA256:
        raise SetupError("変換スクリプトのSHA-256が一致しません。")
    spec = importlib.util.spec_from_file_location("ming_pinned_quantizer", quantizer)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.open_model = NativeMingReader
    sys.argv = [str(quantizer), str(source), str(output), "--w4a8", "--verify-report", str(report)]
    module.main()


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def artifacts(*, download_source: bool) -> ProfileSpec:
    specs = [
        ArtifactSpec(
            "ming-w4a8-quantizer",
            f"{CACHE}/quant_int8_convrot.py",
            f"https://raw.githubusercontent.com/Comfy-Org/comfy-model-tools/{QUANTIZER_REVISION}/quant_int8_convrot.py",
            QUANTIZER_SIZE,
            QUANTIZER_SHA256,
            "https://github.com/Comfy-Org/comfy-model-tools",
        )
    ]
    if download_source:
        specs.append(
            ArtifactSpec(
                "ming-w4a8-source",
                f"{CACHE}/ming_image_0.1_design_bf16.safetensors",
                f"https://huggingface.co/Comfy-Org/Ming-Image/resolve/{WEIGHTS_REVISION}/diffusion_models/ming_image_0.1_design_bf16.safetensors",
                SOURCE_SIZE,
                SOURCE_SHA256,
                "https://huggingface.co/inclusionAI/Ming-Image-0.1-Design",
            )
        )
    return ProfileSpec("ming-w4a8", "Ming DiT W4A8 conversion", tuple(specs), (), 0)


def validate_source(source: Path) -> None:
    if source.is_symlink() or not source.is_file() or source.stat().st_size != SOURCE_SIZE:
        raise SetupError("固定版のMing DiT BF16原本を指定してください。この手順ではINT8から再量子化しません。")
    if file_hash(source) != SOURCE_SHA256:
        raise SetupError("BF16原本のSHA-256が一致しません。")


def validate_output(source: Path, output: Path) -> int:
    from safetensors import safe_open

    count = 0
    with NativeMingReader(source) as original, safe_open(str(output), framework="pt") as converted:
        if original.metadata() != converted.metadata():
            raise SetupError("モデル識別用のメタデータが失われています。")
        if not set(original.keys()).issubset(converted.keys()):
            raise SetupError("変換結果に必要な重みが不足しています。")
        if len(original.attention) != 34:
            raise SetupError("Attentionの構成が想定と異なります。")
        for key in original.attention:
            if bytes(converted.get_tensor(key).tolist()) != ATTENTION_CONFIG:
                raise SetupError("Attention設定がINT8の比較対象と一致しません。")
        for key in converted.keys():
            if not key.endswith(".comfy_quant"):
                continue
            config = json.loads(bytes(converted.get_tensor(key).tolist()))
            if config.get("format") != "asym_w4a8_int8" or config.get("group_size") != 16:
                raise SetupError(f"W4A8形式が一致しません: {key}")
            if config.get("convrot_groupsize") != 256:
                raise SetupError(f"回転のグループ幅が一致しません: {key}")
            weight = key.removesuffix(".comfy_quant") + ".weight"
            n, k = original.get_slice(weight).get_shape()
            actual = converted.get_slice(weight)
            if actual.get_shape() != [n, k // 2] or actual.get_dtype() != "I8":
                raise SetupError(f"W4A8重みの形状が一致しません: {weight}")
            count += 1
    if count != 202 or not 3_000_000_000 < output.stat().st_size < 4_000_000_000:
        raise SetupError(f"変換層数または出力サイズが想定と異なります: {count}層")
    return count


def convert(*, root: Path = ROOT, source: Path | None = None, dry_run: bool = False) -> dict:
    root = root.resolve()
    runtime = root / "repositories/ming-image/ComfyUI"
    destination = runtime / "models/diffusion_models" / OUTPUT_NAME
    receipt_path = destination.with_suffix(".json")
    source = source.resolve() if source is not None else None
    profile = artifacts(download_source=source is None)
    installer = Installer(root, {profile.name: profile})
    if dry_run:
        return {
            "downloads": installer.install(profile.name, dry_run=True, keep_source=True),
            "source": str(source) if source else f"{CACHE}/ming_image_0.1_design_bf16.safetensors",
            "output": str(destination),
            "output_bytes_approx": 3_490_000_000,
            "note": "Neoを閉じて実行。BF16原本約12.3GBと変換結果約3.49GBを保持します。",
        }
    if not runtime_ready(root):
        raise SetupError("先に tools/setup_ming_image.py でMing環境を準備してください。")
    with setup_lock(runtime):
        with socket.socket() as connection:
            connection.settimeout(1)
            if connection.connect_ex(("127.0.0.1", 8189)) == 0:
                raise SetupError("ComfyUIが起動中です。Neoを閉じ、モデルを解放してから実行してください。")
        if destination.exists() or receipt_path.exists():
            raise SetupError(f"変換先は既に存在します。既存のファイルは上書きしません: {destination}")
        plan = installer.install(profile.name, dry_run=True, keep_source=True)
        if shutil.disk_usage(root).free < int(plan["required_free_bytes"]) + 4_000_000_000:
            raise SetupError("原本と変換結果を保存する空き容量が不足しています。")
        if source is not None:
            sys.stdout.write("BF16原本の整合性を検証しています。\n")
            sys.stdout.flush()
            validate_source(source)
        installer.install(profile.name, dry_run=False, keep_source=True)
        source = source or root / CACHE / "ming_image_0.1_design_bf16.safetensors"
        python = runtime.parent / ".venv/Scripts/python.exe"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.stem}-{uuid.uuid4().hex}.partial.safetensors")
        report_path = root / CACHE / "layer-errors.txt"
        command = [
            str(python),
            "-u",
            "-X",
            "utf8",
            str(Path(__file__).resolve()),
            "--worker",
            str(root / CACHE / "quant_int8_convrot.py"),
            str(source),
            str(temporary),
            str(report_path),
        ]
        # Fixed, hash-verified upstream script in the dedicated CUDA environment.
        with subprocess.Popen(  # noqa: S603 -- hash-verified quantizer, fixed local worker arguments.
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ) as worker:
            for line in worker.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
            if worker.wait() != 0:
                raise SetupError("W4A8変換処理に失敗しました。直前のログを確認してください。")
        # Inspection uses safetensors from the same dedicated environment.
        inspection = subprocess.run(  # noqa: S603 -- own inspection command in the managed Python runtime.
            [str(python), "-X", "utf8", str(Path(__file__).resolve()), "--inspect", str(source), str(temporary)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if inspection.returncode:
            raise SetupError("変換結果の検証に失敗しました: " + inspection.stderr.strip())
        count = int(inspection.stdout.strip().splitlines()[-1])
        receipt = {
            "schema_version": 1,
            "format": "asym_w4a8_int8",
            "filename": OUTPUT_NAME,
            "source_sha256": SOURCE_SHA256,
            "source_revision": WEIGHTS_REVISION,
            "quantizer_revision": QUANTIZER_REVISION,
            "quantizer_sha256": QUANTIZER_SHA256,
            "quantized_layers": count,
            "native_layout": "ComfyUI ZImage/Ming QKV fusion",
            "attention": "comfy_kitchen_int8",
            "size": temporary.stat().st_size,
            "sha256": file_hash(temporary),
        }
        receipt_temp = temporary.with_suffix(".json")
        receipt_temp.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        temporary.rename(destination)
        receipt_temp.rename(receipt_path)
        return {"ok": True, "path": str(destination), **receipt}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="取得済みの固定BF16原本。省略時は検証付きで取得")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--inspect", nargs=2, type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker", nargs=4, type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.worker:
            quantize_worker(*args.worker)
        elif args.inspect:
            sys.stdout.write(str(validate_output(*args.inspect)) + "\n")
        else:
            sys.stdout.write(
                json.dumps(convert(source=args.source, dry_run=args.dry_run), ensure_ascii=False, indent=2) + "\n"
            )
        return 0
    except (SetupError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        sys.stderr.write(f"Ming W4A8の変換に失敗しました: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
