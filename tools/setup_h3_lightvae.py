"""Download optional pinned H3 VAEs without changing the runtime or base VAE."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules_forge.minimax_h3_runtime import configured_model_root  # noqa: E402 -- CLI root setup.
from tools.aikimi_setup import DOWNLOAD_HEADROOM_BYTES, ArtifactSpec, Installer, ProfileSpec, SetupError  # noqa: E402
from tools.h3_lightvae_convert import export_native  # noqa: E402

MANIFEST = Path(__file__).with_name("h3_lightvae_manifest.json")


def emit(message):
    sys.stdout.write(message + "\n")


def profile(variant="light_int8") -> ProfileSpec:
    value = json.loads(MANIFEST.read_text("utf-8"))
    artifacts = tuple(ArtifactSpec(**value["artifacts"][key]) for key in value["variants"][variant])
    return ProfileSpec(variant, "H3 LightVAE", artifacts,
                       tuple(dict.fromkeys(item.license_url for item in artifacts)), sum(item.size for item in artifacts))


def exports(variant):
    value = json.loads(MANIFEST.read_text("utf-8"))
    return [(value["artifacts"][key], value["native_exports"][key]) for key in value["variants"][variant]
            if key in value["native_exports"]]


def install_exports(installer, records):
    import torch

    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(min(threads, 4))
        with installer._mutation_lock():
            for source, native in records:
                export_native(installer._target(source["relative_path"]), installer._target(native["relative_path"]),
                              native["component"], source["sha256"], native["sha256"], native["size"])
    finally:
        torch.set_num_threads(threads)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("light_int8", "light_fp32", "lynnreal"), default="light_int8")
    parser.add_argument("--all", action="store_true", help="3種類すべてを取得")
    parser.add_argument("--dry-run", action="store_true", help="保存・通信せず必要量を表示")
    args = parser.parse_args(argv)
    profiles = [profile(name) for name in (("light_int8", "light_fp32", "lynnreal") if args.all else (args.variant,))]
    try:
        root = configured_model_root()
        for item in profiles:
            installer = Installer(root, {item.name: item})
            records = exports(item.name)
            extra_bytes = sum(native["size"] for _, native in records
                              if not installer._target(native["relative_path"]).is_file())
            if not args.dry_run:
                existing = next(path for path in (root, *root.parents) if path.exists())
                required = installer._profile_download_requirement(item, generated_ready=False) + extra_bytes + DOWNLOAD_HEADROOM_BYTES
                if shutil.disk_usage(existing).free < required:
                    raise SetupError(f"LightVAEの取得・変換に約{required}バイトの空き容量が必要です。")
            result = installer.install(item.name, dry_run=args.dry_run, keep_source=True)
            result["native_exports"] = [native for _, native in records]
            result["required_free_bytes"] = int(result.get("required_free_bytes", 0)) + extra_bytes
            if not args.dry_run and records:
                install_exports(installer, records)
            emit(json.dumps(result, ensure_ascii=False, indent=2))
    except (SetupError, OSError, ValueError) as error:
        emit(f"LightVAEを導入できませんでした: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
