"""Re-merge a completed PhaseWeave run from its retained disk moments."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.krea2_quality import adaptive_detail_guard
from modules_forge.vram_canvas import PHASE_WEAVE_MERGE_MODE, phase_weave_configuration
from tools.vram_canvas_highres import (
    _close_memmap,
    _save_stage_result,
    flatten_source_image,
    save_final_png,
)


def _open_moment(path: Path, shape: tuple[int, ...]) -> np.memmap:
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = int(np.prod(shape, dtype=np.int64)) * np.dtype(np.float32).itemsize
    if path.stat().st_size != expected:
        raise ValueError(
            f"moment size mismatch for {path}: {path.stat().st_size} != {expected}"
        )
    return np.memmap(path, dtype=np.float32, mode="r", shape=shape)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    source_manifest_path = args.run_dir / "run_manifest.json"
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("merge_mode") != PHASE_WEAVE_MERGE_MODE:
        raise ValueError("source run is not phase_weave")
    stages = manifest.get("stage_reports") or []
    if len(stages) != 1:
        raise ValueError("re-merge currently requires one completed delivery stage")
    stage = stages[0]
    if int(stage.get("processed_tile_count", -1)) != int(stage.get("tile_count", -2)):
        raise ValueError("source run did not complete every tile")
    width, height = map(int, manifest["target_size"])
    if list(stage.get("size") or []) != [width, height]:
        raise ValueError("source stage size differs from the delivery size")

    source_path = Path(manifest["input"])
    with Image.open(source_path) as image:
        parameters = str(image.info.get("parameters", ""))
        source = flatten_source_image(image)
    base = source.resize((width, height), Image.Resampling.LANCZOS)

    work_dir = args.run_dir / "work"
    prefix = "stage_01"
    accumulators = [
        _open_moment(
            work_dir / f"{prefix}_phase{phase}_delta.float32",
            (height, width, 3),
        )
        for phase in range(2)
    ]
    weight_sums = [
        _open_moment(
            work_dir / f"{prefix}_phase{phase}_weight.float32",
            (height, width),
        )
        for phase in range(2)
    ]
    energy_sums = [
        _open_moment(
            work_dir / f"{prefix}_phase{phase}_energy.float32",
            (height, width),
        )
        for phase in range(2)
    ]
    novel_accumulators: list[np.memmap] = []
    novel_energy_sums: list[np.memmap] = []
    if float(manifest.get("frequency_merge", {}).get("novel_detail_gain", 0.0)) > 0:
        novel_accumulators = [
            _open_moment(
                work_dir / f"{prefix}_phase{phase}_novel_delta.float32",
                (height, width, 3),
            )
            for phase in range(2)
        ]
        novel_energy_sums = [
            _open_moment(
                work_dir / f"{prefix}_phase{phase}_novel_energy.float32",
                (height, width),
            )
            for phase in range(2)
        ]

    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_work = args.output_dir / "work"
    output_work.mkdir()
    stage_path = args.output_dir / f"stage_01_{width}x{height}.png"
    phase_paths = (
        args.output_dir / f"stage_01_phase_a_{width}x{height}.png",
        args.output_dir / f"stage_01_phase_b_{width}x{height}.png",
    )
    selection_path = args.output_dir / f"stage_01_phase_selection_{width}x{height}.png"
    frequency = manifest["frequency_merge"]
    all_moments = (
        accumulators
        + weight_sums
        + energy_sums
        + novel_accumulators
        + novel_energy_sums
    )
    try:
        current, merge_stats = _save_stage_result(
            base,
            accumulators,
            weight_sums,
            energy_sums,
            novel_accumulators,
            novel_energy_sums,
            stage_path,
            output_work / "stage_01_result.uint8",
            stripe_height=128,
            consensus_sigma=float(frequency["consensus_sigma"]),
            novel_consensus_sigma=float(frequency["novel_detail_consensus_sigma"]),
            novel_consensus_strength=float(
                frequency["novel_detail_consensus_strength"]
            ),
            merge_mode=PHASE_WEAVE_MERGE_MODE,
            phase_candidate_paths=phase_paths,
            phase_selection_path=selection_path,
        )
    finally:
        for moment in all_moments:
            _close_memmap(moment)

    report = copy.deepcopy(manifest)
    report["format_version"] = max(5, int(report.get("format_version", 0)))
    report["save_phase_candidates"] = True
    report["phaseweave"] = {
        **(report.get("phaseweave") or {}),
        **phase_weave_configuration(),
    }
    report["remerge"] = {
        "source_manifest": str(source_manifest_path),
        "tile_generation_reused": True,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "reason": "absolute fidelity rejection applied to confident A/B seeds",
    }
    report["stage_reports"][0]["output"] = str(stage_path)
    report["stage_reports"][0]["consensus_stats"] = merge_stats

    finish = report.get("texture_finish") or {}
    if bool(finish.get("enabled")):
        current, finish_report = adaptive_detail_guard(
            current,
            strength=float(finish["detail_strength"]),
            radius=float(finish["detail_radius"]),
            detail_threshold=float(finish["detail_threshold"]),
            max_detail_delta=float(finish["max_detail_delta"]),
        )
        report["texture_finish"]["report"] = finish_report

    final_path = args.output_dir / "vram_canvas_highres.png"
    save_final_png(
        final_path,
        current,
        parameters=parameters,
        prompt=str(report.get("prompt", "")),
        negative_prompt=str(report.get("negative_prompt", "")),
        source_size=tuple(map(int, manifest["source_size"])),
        report=report,
    )
    manifest_path = args.output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_work / "stage_01_result.uint8").unlink(missing_ok=True)
    try:
        output_work.rmdir()
    except OSError:
        pass
    print(
        json.dumps(
            {
                "image": str(final_path),
                "manifest": str(manifest_path),
                "phase_a": str(phase_paths[0]),
                "phase_b": str(phase_paths[1]),
                "selection_map": str(selection_path),
                "selection": merge_stats,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
