"""Evaluate four DetailWeave 4K runs against resize-only references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.compare_krea2_4k import candidate_metrics, load_rgb, read_json, sha256


DEFAULT_ROOT = REPO_ROOT / "output/detailweave_batch_20260722"
SCENES = (
    ("01", "botanical_library"),
    ("02", "moonlit_observatory"),
    ("03", "autumn_teahouse"),
    ("04", "snowy_workshop"),
)


def require_manifest_contract(manifest: dict, result: Image.Image) -> dict:
    expected = {
        "krea2_profile": "phaseweave_4k",
        "merge_mode": "phase_weave",
        "exact_img2img_steps": True,
        "exact_img2img_steps_scope": "internal_tiles_only",
    }
    mismatch = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatch:
        raise ValueError(f"manifest contract mismatch: {mismatch}")
    if manifest.get("target_size") != list(result.size):
        raise ValueError(
            f"manifest target {manifest.get('target_size')} differs from {result.size}"
        )
    reports = manifest.get("stage_reports") or []
    if not reports:
        raise ValueError("manifest has no stage reports")
    stage = reports[-1]
    if stage.get("size") != list(result.size):
        raise ValueError("final stage does not describe the result image")
    tiles = stage.get("tiles") or []
    if not tiles:
        raise ValueError("final stage has no tiles")
    if any(tile.get("skipped") for tile in tiles):
        raise ValueError("final stage unexpectedly contains skipped tiles")
    if any(tile.get("steps") != 6 for tile in tiles):
        raise ValueError("final stage does not use six exact steps per tile")
    phases = {int(tile["phase"]) for tile in tiles}
    if phases != {0, 1}:
        raise ValueError(f"final stage phases differ from {{0, 1}}: {phases}")
    stats = stage.get("consensus_stats") or {}
    if stats.get("merge_mode") != "phase_weave":
        raise ValueError("final merge statistics do not describe phase_weave")
    selection_total = sum(
        float(stats[key])
        for key in (
            "phaseweave_phase0_selected_percent",
            "phaseweave_phase1_selected_percent",
            "phaseweave_input_rejected_percent",
            "phaseweave_uncertain_fused_percent",
        )
    )
    if abs(selection_total - 100.0) > 0.01:
        raise ValueError(f"selection percentages do not sum to 100: {selection_total}")
    return stage


def require_png_contract(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        info = dict(image.info)
    required = ("parameters", "vram_canvas", "krea2_phaseweave")
    missing = [key for key in required if not info.get(key)]
    if missing:
        raise ValueError(f"PNG metadata missing from {path}: {missing}")
    vram_canvas = json.loads(info["vram_canvas"])
    phaseweave = json.loads(info["krea2_phaseweave"])
    expected_canvas = {
        "exact_img2img_steps": True,
        "exact_img2img_steps_scope": "internal_tiles_only",
        "merge_mode": "phase_weave",
    }
    mismatch = {
        key: (vram_canvas.get(key), value)
        for key, value in expected_canvas.items()
        if vram_canvas.get(key) != value
    }
    if mismatch:
        raise ValueError(f"PNG VRAM metadata mismatch: {mismatch}")
    if phaseweave.get("profile_key") != "phaseweave_4k":
        raise ValueError("PNG PhaseWeave metadata has the wrong profile key")
    return {
        "parameters_present": True,
        "exact_img2img_steps": True,
        "exact_img2img_steps_scope": "internal_tiles_only",
        "merge_mode": "phase_weave",
        "profile_key": "phaseweave_4k",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "deliverables/detailweave_4k_batch_qa.json"

    scene_reports = []
    for number, slug in SCENES:
        source_path = args.root / "sources" / f"{number}_{slug}_source.png"
        result_path = (
            args.root
            / "deliverables"
            / f"{number}_{slug}_detailweave_4k.png"
        )
        manifest_path = (
            args.root / "deliverables" / f"{number}_{slug}_manifest.json"
        )
        missing = [
            str(path)
            for path in (source_path, result_path, manifest_path)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(f"batch evaluation inputs missing: {missing}")
        source = load_rgb(source_path)
        result = load_rgb(result_path)
        manifest = read_json(manifest_path)
        stage = require_manifest_contract(manifest, result)
        metadata = require_png_contract(result_path)
        reference = source.resize(result.size, Image.Resampling.LANCZOS)
        metrics = candidate_metrics(reference, result, stage)
        stats = stage["consensus_stats"]
        scene_reports.append(
            {
                "number": number,
                "scene": slug,
                "source": str(source_path),
                "source_sha256": sha256(source_path),
                "result": str(result_path),
                "result_sha256": sha256(result_path),
                "manifest": str(manifest_path),
                "source_size": list(source.size),
                "target_size": list(result.size),
                "stage_count": len(manifest["stage_reports"]),
                "final_tile_count": len(stage["tiles"]),
                "metadata": metadata,
                "selection": {
                    "phase_a_percent": stats["phaseweave_phase0_selected_percent"],
                    "phase_b_percent": stats["phaseweave_phase1_selected_percent"],
                    "input_kept_percent": stats["phaseweave_input_rejected_percent"],
                    "weak_fusion_percent": stats["phaseweave_uncertain_fused_percent"],
                    "boundary_percent": stats["phaseweave_boundary_percent"],
                    "mean_support_weight": stats["phaseweave_mean_support_weight"],
                },
                "metrics_to_lanczos": metrics,
            }
        )

    report = {
        "format_version": 1,
        "method": "DetailWeave 4K",
        "comparison": "same source resized with Lanczos",
        "scene_count": len(scene_reports),
        "all_contract_checks_passed": True,
        "scenes": scene_reports,
        "notes": {
            "lanczos": "Resize-only reference, not ground truth.",
            "metrics": "Signal and input-preservation diagnostics do not prove that generated details are semantically correct.",
            "boundary": "Boundary ratios are evaluated at the final DetailWeave planned core boundaries.",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        scene["number"]: {
            "target_size": scene["target_size"],
            "highpass_ratio": scene["metrics_to_lanczos"]["highpass_mean_ratio"],
            "low_frequency_luma_drift": scene["metrics_to_lanczos"]["low_frequency_luma_drift_mean"],
            "boundary_ratio": scene["metrics_to_lanczos"]["residual_boundary_to_global_p95_ratio"],
        }
        for scene in scene_reports
    }
    print(json.dumps({"report": str(output), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
