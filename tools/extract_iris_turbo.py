"""Compress an official Iris Turbo delta into self-contained local adapters."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules_forge import local_assets  # noqa: E402
from modules_forge.iris.core import RUNTIME, atomic_json, sha256  # noqa: E402
from modules_forge.iris.official_turbo import read_settings  # noqa: E402


def factorize(delta, rank, *, seed=0):
    """Balanced factors; small layers are padded to the adapter's global rank."""
    if delta.ndim != 2 or type(rank) is not int or rank <= 0 or not torch.isfinite(delta).all():
        raise ValueError("SVDには有限値の行列と正のrankが必要です。")
    delta = delta.float()
    dimension = min(delta.shape)
    effective = min(rank, dimension)
    if dimension <= rank + 64:
        u, singular, vh = torch.linalg.svd(delta, full_matrices=False)
        v = vh.T
    else:
        torch.manual_seed(seed)
        u, singular, v = torch.svd_lowrank(delta, q=min(dimension, rank + 64), niter=8)
    scale = singular[:effective].sqrt()
    down = delta.new_zeros((rank, delta.shape[1]))
    up = delta.new_zeros((delta.shape[0], rank))
    down[:effective] = scale[:, None] * v[:, :effective].T
    up[:, :effective] = u[:, :effective] * scale[None, :]
    return down, up


def export(root, turbo, output_dir, ranks=(64, 128), device="cuda"):
    from iris3b.config import inference_config
    from iris3b.models.dit import IrisDiT
    from omegaconf import OmegaConf

    root, turbo, destination = Path(root), Path(turbo).resolve(), Path(output_dir).resolve()
    ranks = sorted(set(ranks))
    if not ranks or any(type(rank) is not int or not 1 <= rank <= 2048 for rank in ranks):
        raise ValueError("rankは1〜2048の整数で指定してください。")
    base = (root / "official/model.safetensors").resolve()
    config = base.with_name("config.yaml")
    sampling = read_settings(turbo.with_name("distill.yaml"))
    paths = [destination / f"iris-turbo-r{rank}.safetensors" for rank in ranks]
    if any(path.exists() for path in [*paths, destination / "manifest.json"]):
        raise ValueError("出力が既にあります。別のフォルダーを指定してください。")
    source_files = (base, config, turbo, turbo.with_name("config.yaml"), turbo.with_name("distill.yaml"))
    identities = [local_assets.file_identity(path) for path in source_files]
    raw = OmegaConf.to_container(OmegaConf.load(config))
    raw.pop("task", None)
    cfg = inference_config(raw, [])
    with torch.device("meta"):
        model = IrisDiT(cfg.model)
    linear = {f"{name}.weight" for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
    shapes = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    del model
    outputs = {rank: {} for rank in ranks}
    records = []
    begin = time.perf_counter()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    with (
        safe_open(base, framework="pt", device="cpu") as original,
        safe_open(turbo, framework="pt", device="cpu") as teacher,
    ):
        if set(original.keys()) != set(shapes) or set(teacher.keys()) != set(shapes):
            raise ValueError("元モデル・Turbo・Iris本体のキーが一致しません。")
        for index, key in enumerate(sorted(shapes)):
            target = teacher.get_tensor(key)
            source = original.get_tensor(key)
            if source.dtype != torch.float32 or target.dtype != torch.bfloat16:
                raise ValueError("元モデルはFP32、公式TurboはBF16の重みを指定してください。")
            if tuple(target.shape) != shapes[key] or tuple(source.shape) != shapes[key]:
                raise ValueError(f"形状が一致しません: {key}")
            if not torch.isfinite(target).all() or not torch.isfinite(source).all():
                raise ValueError(f"モデルに非有限値があります: {key}")
            if key not in linear:
                for rank in ranks:
                    outputs[rank]["dense." + key] = target.bfloat16().contiguous()
                continue
            delta = (target.float() - source.float()).to(device)
            down, up = factorize(delta, max(ranks), seed=index)
            energy = float(delta.square().sum())
            errors = {}
            for rank in ranks:
                saved_down = down[:rank].bfloat16()
                saved_up = up[:, :rank].bfloat16()
                error = float((delta - saved_up.float() @ saved_down.float()).square().sum())
                errors[str(rank)] = {"squared_error": error, "retained_energy": 1 - error / energy if energy else 1.0}
                outputs[rank][key.removesuffix(".weight") + ".lora_down"] = saved_down.cpu().contiguous()
                outputs[rank][key.removesuffix(".weight") + ".lora_up"] = saved_up.cpu().contiguous()
            records.append({"name": key, "shape": list(shapes[key]), "delta_squared_norm": energy, "ranks": errors})
            del delta, down, up, saved_down, saved_up
            if len(records) % 10 == 0 or len(records) == len(linear):
                print(  # noqa: T201
                    json.dumps(
                        {
                            "linear": len(records),
                            "total": len(linear),
                            "seconds": round(time.perf_counter() - begin, 1),
                            "layer": key,
                        }
                    ),
                    flush=True,
                )  # noqa: T201
    if identities != [local_assets.file_identity(path) for path in source_files]:
        raise ValueError("抽出中に元ファイルが変更されました。")
    destination.mkdir(parents=True, exist_ok=True)
    files = []
    for rank, path in zip(ranks, paths, strict=True):
        temporary = path.with_suffix(".partial")
        metadata = {
            "format": "iris-turbo-svd-v1",
            "rank": str(rank),
            "alpha": str(rank),
            "base_sha256": identities[0]["sha256"],
            "base_config_sha256": identities[1]["sha256"],
            "target_sha256": identities[2]["sha256"],
            "sampling": json.dumps(sampling),
            "method": "raw-delta randomized SVD; oversample 64; 8 power iterations; balanced BF16 factors",
        }
        save_file(outputs[rank], temporary, metadata=metadata)
        with safe_open(temporary, framework="pt", device="cpu") as check:
            if set(check.keys()) != set(outputs[rank]):
                raise ValueError("保存したアダプターのキーが一致しません。")
            for key in check.keys():
                if not torch.equal(check.get_tensor(key), outputs[rank][key]):
                    raise ValueError(f"保存したアダプターの値が一致しません: {key}")
        digest = sha256(temporary)
        temporary.rename(path)
        files.append({"path": path.name, "size": path.stat().st_size, "sha256": digest, "rank": rank})
    manifest = {
        "format": "iris-turbo-svd-v1",
        "base_sha256": identities[0]["sha256"],
        "base_config_sha256": identities[1]["sha256"],
        "target_sha256": identities[2]["sha256"],
        "target_config_sha256": identities[3]["sha256"],
        "target_distill_sha256": identities[4]["sha256"],
        "sampling": sampling,
        "linear_count": len(linear),
        "dense_count": len(shapes) - len(linear),
        "seconds": time.perf_counter() - begin,
        "files": files,
        "layers": records,
    }
    atomic_json(destination / "manifest.json", manifest)
    print(json.dumps({"files": files, "seconds": manifest["seconds"]}), flush=True)  # noqa: T201
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RUNTIME)
    parser.add_argument("--turbo", type=Path, default=RUNTIME / "checkpoints/speridlabs-turbo/turbo/model.safetensors")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    export(args.root, args.turbo, args.output_dir, args.ranks, args.device)


if __name__ == "__main__":
    main()
