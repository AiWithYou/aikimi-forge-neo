"""Package verified conversion artifacts into a complete Hugging Face release."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules_forge.clef.bundle import PORTABLE_MODULES, SUPPORT_FILES, bundle_directory, read_bundle  # noqa: E402
from modules_forge.clef.cache import identity  # noqa: E402
from modules_forge.clef.core import (  # noqa: E402
    PREPROCESSING_VERSION,
    PROFILES,
    RUNTIME,
    ClefError,
    atomic_json,
    sha256,
    source_manifest,
)


def save_lexical(source, destination):
    import torch
    from safetensors.torch import save_file

    from modules_forge.clef.runtime import tensor_from_source

    weight = tensor_from_source(source, "lm_head.weight")
    if weight.dtype != torch.bfloat16:
        raise ClefError("出力語彙は元のBF16テンソルを保持してください。")
    save_file({"lm_head.weight": weight.contiguous()}, destination, metadata={"format": "pt"})
    return weight.numel() * weight.element_size()


def link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def auxiliary_assets(profile):
    assets = {"clef_runtime/" + name: (ROOT / "modules_forge/clef" / name).read_bytes() for name in PORTABLE_MODULES}
    assets.update(
        {
            "README.md": model_card(profile).encode("utf-8"),
            "inference.py": (ROOT / "tools/clef_infer.py").read_bytes(),
            "requirements.txt": (ROOT / "tools/requirements-clef.txt").read_bytes(),
        }
    )
    vendor = ROOT / "vendor/accelerate"
    files = [
        vendor / name for name in ("LICENSE", "AIKIMI-PATCH.md", "README.md", "setup.py", "setup.cfg", "pyproject.toml")
    ]
    files.extend(sorted((vendor / "src").rglob("*.py")))
    assets.update({"vendor/accelerate/" + path.relative_to(vendor).as_posix(): path.read_bytes() for path in files})
    return assets


def model_card(profile):
    model = PROFILES[profile]["model"]
    source = identity(profile)
    quant = "INT8" if source["precision"] == "int8" else "NF4 with double quantization"
    label = "Clef-Flash 9B INT8" if model == "clef-flash" else "Clef 27B NF4"
    example = "flash-int8" if model == "clef-flash" else "clef-24gb"
    return f"""---
license: apache-2.0
base_model: {source["repo"]}
base_model_relation: quantized
library_name: transformers
tags:
- clef
- bitsandbytes
- image-classification
- quantized
---
# {label}

Quantized from [{source["repo"]}](https://huggingface.co/{source["repo"]}/tree/{source["revision"]})
at `{source["revision"]}`. The original model, schema encoder and joint head are
by Cloudflare. This derived release packages {quant} decoder weights with the
original BF16 vision encoder, joint head and dense input/output vocabulary.
It includes every inference asset; the original BF16 checkpoint is not required.
The output vocabulary is in `lm-head.safetensors` and indexed separately.

Clef performs a single forward pass over a schema of choice, score and noul
(boolean) questions. It does not generate free-form text. Probabilities are
distributions over the supplied alternatives, not calibrated accuracy estimates.

## Usage

Use Forge Neo's Clef setup and GUI, or this repository's standalone CLI:

```powershell
python -m pip install pip==26.2.1 setuptools==83.0.0
python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python inference.py --profile {example} --image example.png
python inference.py --profile {example} --image example.png --questions questions.json
```

`questions.json` is the original Clef schema: each question ID maps to a `type`,
`instructions`, and, for choice/score, `criteria`. Use `--state` for context,
`--max-pixels` for image processing (default 262144), and `--max-length` for the
total input token cap (default 2048). Inputs exceeding the cap are refused.
An NVIDIA CUDA GPU with BF16 support is required. The standalone loader reads
weights locally and never downloads code during inference.

The 27B release is shared by `clef-24gb` and `clef-16gb`. The latter keeps the
input embedding on CPU. Both keep the dense output vocabulary on CPU and move
only requested rows to CUDA; decoder layers remain on CUDA. The 9B INT8 release
uses `flash-int8`. Packed lm_head replacement and whole-layer CPU offload are
not part of these profiles. The supplied loader is required for this placement.

## Validation and limits

Verified on Windows, RTX 3090 24GB, RAM 64GB, with Transformers 5.10.2,
bitsandbytes 0.50.2, safetensors 0.8.0 and PyTorch 2.13.0+cu130. We use SDPA,
batch=1, no KV cache, and safetensors `pread` to avoid persistent Windows
copy-on-write mappings of all shards.

Four images and a text-only request were checked with the same schemas.
27B 16GB and 24GB profiles produced identical answers and probabilities.
Peak image allocations were 13.38 GiB and 15.73 GiB respectively. The 16GB
profile passed with a 15 GiB PyTorch allocator budget on the 3090; an actual
16GB card was not available. The 9B INT8 peak was 9.82 GiB versus 16.26 GiB
for BF16, with no choice changes in those four images and a maximum probability
difference of 0.0423. This small comparison is not a general accuracy evaluation.
INT8 was slower than BF16 on this test; its benefit here is lower memory use.
Longer inputs or more image pixels require more memory.

The bundled loader uses image preprocessing `{PREPROCESSING_VERSION}`. It passes
the per-call pixel bounds through `images_kwargs.size` and reports the actual
processed image size and vision token count. Earlier releases passed the legacy
`max_pixels` option, which Transformers 5.10.2 ignored. The allocations and
probability comparisons above were measured with that earlier preprocessing,
at the original test images' sizes, rather than a verified 262144-pixel cap.
Do not compare old and new preprocessing as a quantization quality difference.

## Provenance and license

Apache-2.0; see `LICENSE` and the upstream model card. The original
`joint_schema_model.py`, head, vocabulary values and processor assets are
retained. Decoder linear layers were converted with bitsandbytes and saved
using Transformers. Vision and the judgment head remain BF16. The included
`clef_runtime` and `inference.py` add CPU row gathering, profile selection,
input validation and Windows `pread` loading. `complete.json` records the
upstream commit, quantization settings, file sizes and SHA-256 hashes.
`vendor/accelerate` includes Accelerate 1.15.0+aikimi.1, its Apache-2.0
`LICENSE` and `AIKIMI-PATCH.md`; the checkpoint loader rejects escaping shard
paths before reading weights. Install `requirements.txt` from this directory
to use the included patch instead of upstream Accelerate 1.15.0.
No user images, evaluation histories, local configuration or credentials are
included in this model release.
"""


def package_bundle(root, profile, backbone):
    root, backbone = Path(root), Path(backbone)
    target = bundle_directory(root, profile)
    if target is None:
        raise ClefError("BF16は量子化配布の対象ではありません。")
    if target.exists():
        read_bundle(target, profile)
        return target
    source, source_meta = source_manifest(root, PROFILES[profile]["model"])
    converted = json.loads((backbone / "complete.json").read_text(encoding="utf-8"))
    if any(converted.get(key) != value for key, value in identity(profile).items()):
        raise ClefError("変換物のモデル・精度・固定リビジョンが一致しません。")
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="." + target.name + "-", dir=target.parent) as temporary:
        staging, records = Path(temporary), []
        for item in converted["files"]:
            path = backbone / item["path"]
            if (
                not path.resolve().is_relative_to(backbone.resolve())
                or not path.is_file()
                or path.stat().st_size != item["size"]
            ):
                raise ClefError(f"変換物が不足・変更されています: {item['path']}")
            if item["path"] == "model.safetensors.index.json":
                continue  # This index gets its own copy before adding the dense head.
            link_or_copy(path, staging / item["path"])
            records.append(item)
        original_files = {x["path"]: x for x in source_meta["files"]}
        for name in SUPPORT_FILES:
            link_or_copy(source / name, staging / name)
            records.append(original_files[name])
        lexical_size = save_lexical(source, staging / "lm-head.safetensors")
        index = json.loads((backbone / "model.safetensors.index.json").read_text(encoding="utf-8"))
        index["weight_map"]["lm_head.weight"] = "lm-head.safetensors"
        index.setdefault("metadata", {})["total_size"] = index.get("metadata", {}).get("total_size", 0) + lexical_size
        atomic_json(staging / "model.safetensors.index.json", index)
        for name, data in auxiliary_assets(profile).items():
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        known = {x["path"] for x in records}
        for path in sorted(staging.rglob("*")):
            name = path.relative_to(staging).as_posix()
            if path.is_file() and name not in known:
                records.append({"path": name, "size": path.stat().st_size, "sha256": sha256(path)})
        atomic_json(
            staging / "complete.json",
            {
                **identity(profile),
                "bundle_format": 1,
                "preprocessing": PREPROCESSING_VERSION,
                "files": sorted(records, key=lambda x: x["path"]),
            },
        )
        read_bundle(staging, profile)
        staging.rename(target)
    return target


def refresh_bundle(root, profile):
    target, manifest = read_bundle(bundle_directory(root, profile), profile)
    assets = auxiliary_assets(profile)
    records = {item["path"]: item for item in manifest["files"]}
    # Stage both versions before changing any live file. Rollback only renames
    # existing files, so it also works if saving the final marker runs out of space.
    with TemporaryDirectory(prefix=".refresh-code-", dir=target.parent) as temporary:
        staging = Path(temporary)
        existing = [name for name in assets if (target / name).is_file()]
        for name in [*existing, "complete.json"]:
            backup = staging / "old" / name
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target / name, backup)
        for name, data in assets.items():
            prepared = staging / "new" / name
            prepared.parent.mkdir(parents=True, exist_ok=True)
            prepared.write_bytes(data)
            records[name] = {"path": name, "size": len(data), "sha256": sha256(prepared)}
        try:
            for name in assets:
                (target / name).parent.mkdir(parents=True, exist_ok=True)
                os.replace(staging / "new" / name, target / name)
            atomic_json(
                target / "complete.json",
                {
                    **manifest,
                    "preprocessing": PREPROCESSING_VERSION,
                    "files": sorted(records.values(), key=lambda item: item["path"]),
                },
            )
            read_bundle(target, profile)
        except BaseException:
            for name in assets.keys() - set(existing):
                (target / name).unlink(missing_ok=True)
            for name in [*existing, "complete.json"]:
                os.replace(staging / "old" / name, target / name)
            raise
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RUNTIME)
    parser.add_argument("--profile", choices=["flash-int8", "clef-24gb", "clef-16gb"], required=True)
    parser.add_argument("--backbone", type=Path, help="検証済み変換物のcomplete.jsonを含むディレクトリ")
    parser.add_argument("--refresh-code", action="store_true", help="停止中の既存配布の補助コードだけを更新する")
    args = parser.parse_args()
    if not args.refresh_code and args.backbone is None:
        parser.error("--backboneまたは--refresh-codeが必要です。")
    target = (
        refresh_bundle(args.root, args.profile)
        if args.refresh_code
        else package_bundle(args.root, args.profile, args.backbone)
    )
    print(target, flush=True)  # noqa: T201


if __name__ == "__main__":
    main()
