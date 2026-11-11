"""Prepare the official MiniMax-H3-Image models and shared H3 runtime."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.aikimi_setup import ArtifactSpec, ProfileSpec, SetupError  # noqa: E402
from tools.setup_p_h3_image import main, verify_models  # noqa: E402


def model_profile() -> ProfileSpec:
    data = json.loads((ROOT / "tools/minimax_h3_image_manifest.json").read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise SetupError("MiniMax-H3-Imageのモデル定義を確認できません。")
    license_url = data["model_license"]
    base = f"https://huggingface.co/{data['model_repository']}/resolve/{data['model_revision']}"
    artifacts = tuple(
        ArtifactSpec(
            artifact_id=item["path"], relative_path=item["path"], url=f"{base}/{item['path']}",
            size=item["size"], sha256=item["sha256"], license_url=license_url,
        )
        for item in data["models"]
    )
    return ProfileSpec("minimax_h3_image", "MiniMax-H3-Image", artifacts, (license_url,), sum(x.size for x in artifacts))


if __name__ == "__main__":
    raise SystemExit(main(description=__doc__, profile=model_profile, verify=lambda models, shared: verify_models(
        models, shared=shared, profile=model_profile()
    )))
