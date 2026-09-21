"""Download optional H3 W4A8 weights into Neo; reuse the verified/resumable installer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.aikimi_setup import ArtifactSpec, Installer, ProfileSpec, SetupError  # noqa: E402

MANIFEST = ROOT / "tools/minimax_h3_w4a8_manifest.json"


def model_profile(mode="both"):
    if mode not in {"both", "fl2va", "ref2va"}:
        raise ValueError("mode must be both, fl2va or ref2va")
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    artifacts = tuple(
        ArtifactSpec(
            artifact_id=entry["mode"],
            relative_path=f"diffusion_models/{entry['filename']}",
            url=f"https://huggingface.co/{data['model_repository']}/resolve/{data['model_revision']}/{entry['filename']}",
            size=entry["size"],
            sha256=entry["sha256"],
            license_url=data["model_license"],
        )
        for entry in data["models"]
        if mode == "both" or entry["mode"] == mode
    )
    return ProfileSpec(
        "h3-w4a8", "MiniMax H3 W4A8", artifacts, (data["model_license"],), sum(x.size for x in artifacts)
    )


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("both", "fl2va", "ref2va"), default="both")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--dry-run", action="store_true")
    actions.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    models = (ROOT / "models/MiniMax-H3").resolve()
    try:
        if not models.is_relative_to(ROOT.resolve()):
            raise SetupError("Neo内のモデル保存先が外部へリンクされています。")
        profile = model_profile(args.mode)
        installer = Installer(models, {profile.name: profile})
        if args.verify:
            result = installer.verify([profile.name])
        else:
            result = installer.install(profile.name, dry_run=args.dry_run, keep_source=True)
        print(json.dumps(result, ensure_ascii=False, indent=2))  # noqa: T201
        return 0 if result.get("ok", True) else 1
    except (SetupError, ValueError, OSError) as exc:
        print(f"W4A8モデルの準備を完了できませんでした: {exc}", file=sys.stderr)  # noqa: T201
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
