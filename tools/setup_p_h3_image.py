"""Prepare P-H3 Image and the shared H3 runtime without downloading video models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules_forge.p_h3_image import MODEL, TEXT_ENCODER, UPSAMPLER, VAE  # noqa: E402
from tools.aikimi_setup import ArtifactSpec, Installer, ProfileSpec, SetupError  # noqa: E402
from tools.setup_minimax_h3 import RuntimeInstaller, emit, plan  # noqa: E402

MANIFEST = ROOT / "tools/p_h3_image_manifest.json"


def model_profile() -> ProfileSpec:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise SetupError("P-H3 Imageのモデル定義を確認できません。")
    entries = data["models"]
    names = {Path(item["path"]).name for item in entries}
    if len(entries) != 4 or names != {MODEL, TEXT_ENCODER, VAE, UPSAMPLER}:
        raise SetupError("P-H3 Imageのモデル構成が一致しません。")
    artifacts = tuple(
        ArtifactSpec(
            artifact_id=item["path"],
            relative_path=item["path"],
            url=f"https://huggingface.co/{data['model_repository']}/resolve/{data['model_revision']}/{item['path']}",
            size=item["size"],
            sha256=item["sha256"],
            license_url=data["model_license"],
        )
        for item in entries
    )
    return ProfileSpec("p_h3_image", "P-H3 Image", artifacts, (data["model_license"],), sum(x.size for x in artifacts))


def verify_models(models: Path, *, shared: bool, profile: ProfileSpec | None = None) -> dict:
    profile = profile or model_profile()
    installer = Installer(models, {profile.name: profile})
    if shared:
        result = installer.verify([profile.name])
        if not result["ok"]:
            failures = [item["path"] for item in result["profiles"][0]["artifacts"] if item["state"] != "ready"]
            raise SetupError("共有モデルが不足または破損しています: " + ", ".join(failures))
        return result
    return installer.install(profile.name, dry_run=False, keep_source=True)


def main(*, profile=model_profile, verify=verify_models, description=__doc__) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=description)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--share-models", type=Path, help="既存モデルフォルダーを読み取り専用で共有する")
    mode.add_argument("--models-only", action="store_true", help="選択したモデルのみ準備する")
    mode.add_argument("--runtime-only", action="store_true", help="モデルを取得せず共通実行環境のみ準備する")
    parser.add_argument("--dry-run", action="store_true", help="保存や通信をせず導入計画を表示する")
    args = parser.parse_args()
    try:
        if args.dry_run:
            emit(json.dumps(plan(shared_models=args.share_models, profile=profile()), ensure_ascii=False, indent=2))
        elif args.models_only:
            configuration = plan(profile=profile())
            verify(Path(configuration["models"]), shared=configuration["shared"])
            emit("モデルの準備ができました。")
        else:
            RuntimeInstaller().install(
                args.share_models, profile=profile, verify=verify, runtime_only=args.runtime_only
            )
    except (SetupError, ValueError, OSError) as error:
        emit(f"セットアップを完了できませんでした: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
