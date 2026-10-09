"""Pinned official and Viggle Turbo assets shared by the installer and worker."""

from __future__ import annotations

import json
from pathlib import Path

from .core import QwenImage21Error, atomic_json, inside, precision_label
from .gguf import load_transformer, sha256

VIGGLE_ID = "Viggle/Qwen-Image-2.1-viggle-turbo"
VIGGLE_REVISION = "bafc91e4cc934f5fb1406b22496a0bed9b99c548"
GGUF_ID = "Abiray/Qwen-Image-2.1-viggle-4-steps-turbo-GGUF"
GGUF_REVISION = "0016110b769f6625b92098ab9e3f4906dba367f6"
GGUF_NAME = "qwen_image_2.1_turbo_Q4_K_M.gguf"
OFFICIAL_ID = "Qwen/Qwen-Image-2.1-Turbo"
OFFICIAL_REVISION = "d65dbc9a7e8f6b5479e33dee6030eaab2a906509"
OFFICIAL_PRECISIONS = ("turbo_official_int8", "turbo_official_w4a8", "turbo_official_bf16")
OFFICIAL_SIGMAS = (1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568)
OFFICIAL_FILES = {
    "model_index.json": {
        "size": 578,
        "sha256": "97e2febeb19d93cb41fa44807cde8bd42757950b5dffeacdb0331fb2d390cf20",
    },
    "scheduler/scheduler_config.json": {
        "size": 486,
        "sha256": "16c948c71f9152a34cce6e3a2f309e97ad2dca32beecbe5e2fd3a7925b7eb4bb",
    },
    "transformer/config.json": {
        "size": 370,
        "sha256": "2c567038ca190824728844b8d06d94ae02360e668707afd9734e789a7eb58ce3",
    },
    "transformer/diffusion_pytorch_model.safetensors.index.json": {
        "size": 30283,
        "sha256": "17987f6623b1c814d0ef55a137d99142b7b3b040eb1bf241b5575dd35af803a2",
    },
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors": {
        "size": 9968332504,
        "sha256": "6cccd922767f01694461bdcf1f34ea7b771f3442e4d5e338ea9aa55277431cc0",
    },
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors": {
        "size": 4261951904,
        "sha256": "69f53ebb063d2f606bdaff7e22e0e2144f28d978d73dc75e6e4293fc5982eab5",
    },
}
PROFILES = {
    "turbo_bf16": (
        VIGGLE_ID,
        VIGGLE_REVISION,
        ("transformer/config.json", "transformer/diffusion_pytorch_model.safetensors"),
    ),
    "turbo_q4_k_m": (GGUF_ID, GGUF_REVISION, (GGUF_NAME,)),
    **{precision: (OFFICIAL_ID, OFFICIAL_REVISION, tuple(OFFICIAL_FILES)) for precision in OFFICIAL_PRECISIONS},
}
FOLDERS = {"turbo_bf16": "bf16", "turbo_q4_k_m": "gguf", **dict.fromkeys(OFFICIAL_PRECISIONS, "official")}
SCHEDULER = "scheduler/scheduler_config.json"


def transformer_directory(root: Path, precision: str) -> Path:
    if precision not in PROFILES:
        raise QwenImage21Error("Turboモデルの種類が不正です。")
    return inside(root, Path(root) / "turbo" / FOLDERS[precision] / "transformer")


def scheduler_directory(root: Path, precision: str) -> Path:
    if precision not in PROFILES:
        raise QwenImage21Error("Turboモデルの種類が不正です。")
    folder = "official" if precision in OFFICIAL_PRECISIONS else ""
    return inside(root, Path(root) / "turbo" / folder / "scheduler")


def _official_configuration(root: Path) -> list[float]:
    directory = inside(root, Path(root) / "turbo" / "official")
    index = json.loads((directory / "model_index.json").read_text(encoding="utf-8"))
    sigmas = index.get("sample_sigmas")
    if (
        index.get("_class_name") != "QwenImage21Pipeline"
        or not isinstance(sigmas, list)
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in sigmas)
        or tuple(sigmas) != OFFICIAL_SIGMAS
    ):
        raise ValueError("公式Turboの8-step sigma設定が一致しません。")
    scheduler = json.loads((directory / SCHEDULER).read_text(encoding="utf-8"))
    required = {
        "_class_name": "FlowMatchEulerDiscreteScheduler",
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,
        "use_dynamic_shifting": False,
        "invert_sigmas": False,
        "use_karras_sigmas": False,
        "use_exponential_sigmas": False,
        "use_beta_sigmas": False,
    }
    if any(key not in scheduler or scheduler[key] != value for key, value in required.items()):
        raise ValueError("公式Turboのscheduler設定が一致しません。")
    transformer = json.loads((directory / "transformer/config.json").read_text(encoding="utf-8"))
    if transformer.get("_class_name") != "QwenImage21Transformer2DModel" or transformer.get("num_layers") != 32:
        raise ValueError("公式TurboのTransformer設定が一致しません。")
    weights = json.loads(
        (directory / "transformer/diffusion_pytorch_model.safetensors.index.json").read_text(encoding="utf-8")
    )
    shards = {Path(name).name for name in OFFICIAL_FILES if name.endswith(".safetensors")}
    if not isinstance(weights.get("weight_map"), dict) or set(weights["weight_map"].values()) != shards:
        raise ValueError("公式Turboのshard構成が一致しません。")
    return list(sigmas)


def sampling_sigmas(root: Path, precision: str) -> list[float] | None:
    if precision not in PROFILES:
        raise QwenImage21Error("Turboモデルの種類が不正です。")
    if precision not in OFFICIAL_PRECISIONS:
        return None
    turbo_manifest(root, precision)
    return _official_configuration(root)


def turbo_manifest(root: Path, precision: str, *, verify_hashes: bool = False) -> dict:
    """Check a completed optional install without importing CUDA libraries."""
    if precision not in PROFILES:
        raise QwenImage21Error("Turboモデルの種類が不正です。")
    path = Path(root) / "turbo-files.json"
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, dict) or precision not in records:
            raise ValueError("導入記録がありません")
        record = records[precision]
        model_id, revision, names = PROFILES[precision]
        if record.get("model") != model_id or record.get("revision") != revision:
            raise ValueError("固定revisionが一致しません。")
        files = record["files"]
        official = precision in OFFICIAL_PRECISIONS
        expected = {f"{FOLDERS[precision]}/{name}" for name in names}
        if not official:
            expected.add(SCHEDULER)
        if set(files) != expected:
            raise ValueError("ファイル一覧が一致しません。")
        for name, entry in files.items():
            if official and entry != OFFICIAL_FILES[name.removeprefix("official/")]:
                raise ValueError(f"公式Turboの固定ファイル記録が一致しません: {name}")
            file = inside(root, Path(root) / "turbo" / name)
            if not file.is_file() or file.stat().st_size != entry["size"]:
                raise ValueError(f"不足またはサイズ不一致: {name}")
            if (verify_hashes or official and name.endswith(".json")) and sha256(file) != entry["sha256"]:
                raise ValueError(f"SHA-256不一致: {name}")
        if official:
            _official_configuration(root)
        return record
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        flag = (
            "--official-turbo-only"
            if precision in OFFICIAL_PRECISIONS
            else "--turbo-bf16-only"
            if precision == "turbo_bf16"
            else "--turbo-q4-only"
        )
        raise QwenImage21Error(
            f"Turboモデルが未導入または不完全です。aikimi-qwen-image21-setup.bat {flag} を実行してください（{exc}）。"
        ) from exc


def turbo_status(root: Path, precision: str) -> str:
    try:
        turbo_manifest(root, precision)
    except QwenImage21Error as exc:
        return str(exc)
    return f"{precision_label(precision)} · 導入済み。"


def download_turbo(root: Path, precision: str) -> None:
    """Download exact optional files and publish their inventory last."""
    from huggingface_hub import HfApi, hf_hub_download

    model_id, revision, names = PROFILES[precision]
    api = HfApi()
    official = precision in OFFICIAL_PRECISIONS
    specs = ((model_id, revision, names, FOLDERS[precision]),)
    if not official:
        specs += ((VIGGLE_ID, VIGGLE_REVISION, (SCHEDULER,), ""),)
    files = {}
    for repo, pinned, wanted, folder in specs:
        info = api.model_info(repo, revision=pinned, files_metadata=True)
        if info.sha != pinned:
            raise RuntimeError(f"{repo}の固定revisionを確認できません。")
        siblings = {item.rfilename: item for item in info.siblings}
        for name in wanted:
            item = siblings[name]
            destination = inside(root, Path(root) / "turbo" / folder)
            path = inside(root, destination / name)
            expected_file = OFFICIAL_FILES[name] if official else None
            remote_hash = getattr(item.lfs, "sha256", None) if item.lfs else None
            if official and (
                item.size != expected_file["size"] or remote_hash and remote_hash != expected_file["sha256"]
            ):
                raise RuntimeError(f"公式Turboの配布情報が固定値と一致しません: {name}")
            actual = (
                sha256(path) if official and path.is_file() and path.stat().st_size == expected_file["size"] else None
            )
            verified = official and actual == expected_file["sha256"]
            if not verified:
                path = Path(hf_hub_download(repo, filename=name, revision=pinned, local_dir=destination))
                actual = sha256(path)
            if not path.resolve().is_relative_to(destination.resolve()) or (
                item.size and path.stat().st_size != item.size
            ):
                raise RuntimeError(f"Turboモデルのファイルサイズが一致しません: {name}")
            expected = expected_file["sha256"] if official else remote_hash
            if expected and actual != expected:
                raise RuntimeError(f"TurboモデルのSHA-256が一致しません: {name}")
            files[f"{folder}/{name}".lstrip("/")] = {"size": path.stat().st_size, "sha256": actual}
    if official:
        _official_configuration(root)
    manifest = root / "turbo-files.json"
    try:
        inventory = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        inventory = {}
    record = {"model": model_id, "revision": revision, "files": files}
    for profile in OFFICIAL_PRECISIONS if official else (precision,):
        inventory[profile] = record
    atomic_json(manifest, inventory)
    turbo_manifest(root, precision)


def load_gguf_transformer(model_path: Path):
    return load_transformer(model_path, model_path.parent / "turbo" / "gguf" / GGUF_NAME)
