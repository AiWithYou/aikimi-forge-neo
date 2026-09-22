"""CPU-only planning, provenance and reporting for sparse image benchmarks."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

CATEGORIES = ("text", "person", "fine_detail", "multi_reference", "rgba")
REPLAY_SOURCES = {
    "replay": "jev",
    "tiles-replay": "tiles-jev",
    "combined-replay": "combined",
    "fixed-tiles-replay": "fixed-tiles-jev",
}
LIVE_MODES = frozenset(REPLAY_SOURCES.values())


@dataclass(frozen=True)
class Case:
    id: str
    category: str
    prompt: str
    input_images: tuple[str, ...] = ()
    transparent: bool = False
    source: str | None = None


@dataclass(frozen=True)
class Variant:
    mode: str
    keep: float

    @property
    def key(self):
        return f"{self.mode}-{self.keep:g}" if self.mode.startswith("fixed") else self.mode

    @property
    def replay_key(self):
        source = REPLAY_SOURCES.get(self.mode)
        return Variant(source, self.keep).key if source else None

    @property
    def runtime_mode(self):
        return REPLAY_SOURCES.get(self.mode, self.mode)


def builtin_cases(reference_images=()):
    return [
        Case(
            "lettering",
            "text",
            "A realistic product photograph of a red ceramic teapot, a clear glass of tea and a blue linen cloth. "
            'A cream card clearly reads "TEA TIME" and a second card reads "OPEN 09:30". '
            "Keep the exact spelling and digits, natural window light, crisp readable printed letters.",
        ),
        Case(
            "portrait",
            "person",
            "An editorial photograph of one adult silver-haired astronomer in a navy coat on a stone observatory "
            "terrace. Both hands hold a brass telescope, natural skin texture, distinct fingers, individual hair "
            "strands, believable anatomy, distant mountains and soft evening light.",
        ),
        Case(
            "woven-detail",
            "fine_detail",
            "A macro product photograph of finely woven blue and ivory fabric beside an engraved brass watch, "
            "a woven wicker basket and fern leaves. Repeating thin stripes, regular tiny checks, clearly separated "
            "threads, small concentric metal engravings, consistent sharp texture and soft natural lighting.",
        ),
        Case(
            "two-references",
            "multi_reference",
            "Combine the main object from image 1 and the main object from image 2 in one studio photograph. "
            "Keep their distinct shapes, colors and visible markings. Place the first on the left and the second "
            "on the right with a clean light background and consistent soft lighting.",
            input_images=tuple(str(Path(p).resolve()) for p in reference_images),
        ),
        Case(
            "transparent-object",
            "rgba",
            "A single blue glass perfume bottle with a silver cap and a thin silk ribbon, isolated on a transparent "
            "background. Keep the complete silhouette, fine ribbon edges and small highlights, no ground plane.",
            transparent=True,
        ),
    ]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_cases(manifest=None, *, reference_images=(), selected=None, prompt=None, source=None):
    if prompt is not None and manifest is not None:
        raise ValueError("--prompt and --case-manifest cannot be combined")
    if manifest is None:
        cases = [Case("custom", "text", prompt)] if prompt is not None else builtin_cases(reference_images)
    else:
        manifest = Path(manifest).resolve()
        document = json.loads(manifest.read_text(encoding="utf-8-sig"))
        if not isinstance(document, dict) or document.get("schema") != 1 or not isinstance(document.get("cases"), list):
            raise ValueError("Case manifest must contain schema=1 and a cases array")
        cases = []
        for item in document["cases"]:
            if not isinstance(item, dict) or set(item) - set(Case.__dataclass_fields__):
                raise ValueError("Unknown case manifest fields")
            entry = dict(item)
            inputs = entry.get("input_images", [])
            if not isinstance(inputs, list) or any(not isinstance(p, str) or not p for p in inputs):
                raise ValueError("Case input_images must be an array of file paths")
            entry["input_images"] = tuple(str((manifest.parent / p).resolve()) for p in inputs)
            if entry.get("source") is not None:
                if not isinstance(entry["source"], str) or not entry["source"]:
                    raise ValueError("Case source must be a file path")
                entry["source"] = str((manifest.parent / entry["source"]).resolve())
            try:
                cases.append(Case(**entry))
            except TypeError as exc:
                raise ValueError("A case requires id, category and prompt") from exc
    ids = set()
    for case in cases:
        if not isinstance(case.id, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", case.id):
            raise ValueError("Case IDs must be short letters, digits, hyphens or underscores")
        if case.id in ids:
            raise ValueError(f"Duplicate case ID: {case.id}")
        ids.add(case.id)
        if case.category not in CATEGORIES or not isinstance(case.prompt, str) or not case.prompt.strip():
            raise ValueError("Cases require a supported category and a nonempty prompt")
        if type(case.transparent) is not bool or len(case.input_images) > 10:
            raise ValueError("transparent must be boolean; at most ten input images are supported")
        if case.category == "rgba" and not case.transparent:
            raise ValueError("An rgba case must set transparent=true")
    if not cases:
        raise ValueError("At least one case is required")
    if selected:
        unknown = set(selected) - ids
        if unknown:
            raise ValueError(f"Unknown case IDs: {', '.join(sorted(unknown))}")
        cases = [case for case in cases if case.id in selected]
    if source is not None:
        cases = [replace(case, source=str(Path(source).resolve())) for case in cases]
    return cases


def case_skip(case, model, stage):
    if model == "krea2" and case.transparent:
        return "unsupported: the Krea2 Forge benchmark path does not produce RGBA output"
    if model == "krea2" and (case.input_images or case.category == "multi_reference"):
        return "unsupported: this Forge API benchmark adapter has no multiple-reference input transport"
    if case.category == "multi_reference" and len(case.input_images) < 2:
        return "missing_input: a multiple-reference case requires at least two --reference-images or manifest paths"
    if model == "qwen21" and case.source is not None:
        return "unsupported: source is for Krea2 4k; use input_images for Qwen reference images"
    if model == "krea2" and stage == "native" and case.source is not None:
        return "unsupported: source requires the Krea2 4k stage"
    for filename in (*case.input_images, *((case.source,) if case.source else ())):
        if not Path(filename).is_file():
            return f"missing_input: file does not exist: {filename}"
    return None


def build_variants(modes, keeps, initial_keep):
    if not modes or len(set(modes)) != len(modes):
        raise ValueError("Select one or more distinct modes")
    if not keeps or len(set(keeps)) != len(keeps):
        raise ValueError("Select one or more distinct fixed keep percentages")
    if any(not math.isfinite(v) or not 1 <= v <= 100 for v in [initial_keep, *keeps]):
        raise ValueError("Keep percentages must be finite and in 1..100")
    variants = [
        Variant(mode, value) for mode in modes for value in (keeps if mode.startswith("fixed") else [initial_keep])
    ]
    keys = [v.key for v in variants]
    for variant in variants:
        if variant.replay_key in keys and keys.index(variant.replay_key) > keys.index(variant.key):
            raise ValueError("Put each live Jev mode before its replay mode; later repetitions reverse the full order")
    return variants


def build_plan(*, model, stage, cases, seeds, dimensions, variants, repeats, settings, replay_from=None):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 <= s < 2**63 for s in seeds):
        raise ValueError("Seeds must be distinct integers in 0..2^63-1")
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("Dimensions must be distinct")
    keys = {v.key for v in variants}
    if not replay_from and any(v.replay_key and v.replay_key not in keys for v in variants):
        raise ValueError("Replay needs the matching live mode or --replay-from benchmark.json")
    if replay_from:
        ReplayIndex(replay_from)  # Validate the saved report before importing a model runtime.
    plan = {
        "schema": 2,
        "model": model,
        "stage": stage,
        "settings": settings,
        "cases": [asdict(case) for case in cases],
        "seeds": seeds,
        "repeats": repeats,
        "variants": [dict(asdict(v), key=v.key, replay_of=v.replay_key) for v in variants],
        "replay_from": str(Path(replay_from).resolve()) if replay_from else None,
        "groups": [],
        "skipped_cases": [],
        "protocol": {
            "warmup": "one completed OFF image per case/seed/size, excluded from statistics",
            "order": "forward on odd repetitions, reversed on even repetitions",
            "timing": "completed image wall time including generation, API/controller wait and output persistence",
            "replay": "latest validated live log for identical input identity, or matching recorded benchmark",
            "quality": "not automatically scored; inspect each output for the case requirements",
        },
    }
    for case in cases:
        reason = case_skip(case, model, stage)
        if reason:
            plan["skipped_cases"].append(
                {"case_id": case.id, "category": case.category, "status": "skipped", "reason": reason}
            )
            continue
        input_hashes = [digest(p) for p in case.input_images]
        source_hash = digest(case.source) if case.source else None
        for seed in seeds:
            for width, height in dimensions:
                geometry = "4k" if stage == "4k" else f"{width}x{height}"
                group = f"{case.id}-s{seed}-{geometry}"
                identity = {
                    "model": model,
                    "stage": stage,
                    "case_id": case.id,
                    "prompt_sha256": hashlib.sha256(case.prompt.encode("utf-8")).hexdigest(),
                    "input_sha256": input_hashes,
                    "source_sha256": source_hash,
                    "transparent": case.transparent,
                    "seed": seed,
                    "width": width,
                    "height": height,
                    "settings": settings,
                }
                schedule = [
                    {
                        "label": f"{group}-warmup",
                        "variant": "off",
                        "mode": "off",
                        "keep": variants[0].keep,
                        "repetition": 0,
                    }
                ]
                for n in range(repeats):
                    ordered = variants if n % 2 == 0 else list(reversed(variants))
                    for variant in ordered:
                        schedule.append(
                            {
                                "label": f"{group}-{variant.key}-r{n + 1}",
                                "variant": variant.key,
                                "mode": variant.mode,
                                "keep": variant.keep,
                                "repetition": n + 1,
                                "replay_of": variant.replay_key,
                            }
                        )
                plan["groups"].append({"id": group, "case": asdict(case), "identity": identity, "schedule": schedule})
    plan["planned_runs"] = sum(len(group["schedule"]) for group in plan["groups"])
    plan["planned_live_generations"] = sum(
        run["mode"] in LIVE_MODES for group in plan["groups"] for run in group["schedule"]
    )
    return plan


def validate_output(path, *, transparent=False, expected_size=None):
    from PIL import Image

    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("Generation did not produce a complete image file")
    with Image.open(path) as image:
        image.load()
        if expected_size and image.size != tuple(expected_size):
            raise RuntimeError(f"Output dimensions {image.size} do not match {tuple(expected_size)}")
        if transparent and image.mode != "RGBA":
            raise RuntimeError("RGBA was requested but the output has no RGBA channels")
        with image.convert("RGBA") as rgba:
            pixels = {
                "mode": "RGBA",
                "width": rgba.width,
                "height": rgba.height,
                "sha256": hashlib.sha256(rgba.tobytes()).hexdigest(),
            }
        result = {
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "sha256": digest(path),
            "pixels": pixels,
        }
        if image.mode == "RGBA":
            result["alpha_extrema"] = list(image.getchannel("A").getextrema())
        return result


def read_log(path):
    path = Path(path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ends = [record for record in records if record.get("event") == "end"]
    if len(ends) != 1 or ends[0].get("status") != "completed" or records[-1] != ends[0]:
        raise RuntimeError("Experiment log is missing one completed terminal record")
    return {
        "log_path": str(path.resolve()),
        "end": ends[0],
        "decisions": [r for r in records if r.get("event") == "decision"],
        "records": records,
    }


def validate_decisions(log, *, replay, kind, cadence=None):
    records, end = log["records"], log["end"]
    failures = {"controller_failure", "nonfinite_statistics", "replay_failure"}
    if any(
        row.get("event") in failures
        or row.get("circuit_open")
        or row.get("source") in {"dense_fallback", "rules_fallback"}
        for row in records
    ):
        raise RuntimeError(f"{kind} controller fell back; this is not a successful Jev/replay result")
    if replay:
        if end.get("api_calls", 0) or not end.get("replay_calls", 0):
            raise RuntimeError(f"{kind} replay must consume recorded decisions with zero cloud calls")
    elif not end.get("api_calls", 0):
        raise RuntimeError(f"{kind} Jev did not make a successful live decision")
    decisions = log["decisions"]
    if not any(row.get("source") in {"jev", "replay"} or row.get("diagnostics") for row in decisions):
        raise RuntimeError(f"No valid {kind} Jev/replay decision was recorded")
    # Attention and tile allocation can share a job client, so its counters
    # include both controllers. Count this kind's decisions for cadence checks.
    if cadence == "once" and len(decisions) > 1:
        raise RuntimeError(f"{kind} exceeded its once-per-job decision budget")


def public_log(log):
    return {key: value for key, value in log.items() if key != "records"}


class ReplayIndex:
    """Reuse only completed live runs whose inputs and generation settings match."""

    def __init__(self, report_path=None):
        self.rows = []
        if report_path:
            report = json.loads(Path(report_path).read_text(encoding="utf-8"))
            if not isinstance(report, dict) or report.get("schema") != 2 or not isinstance(report.get("runs"), list):
                raise ValueError("Replay requires a schema=2 benchmark with recorded input identity")
            self.rows.extend(report.get("runs", []))

    def add(self, row):
        if row.get("status") == "completed" and row.get("mode") in LIVE_MODES:
            self.rows.append(row)

    def lookup(self, group, spec):
        for row in reversed(self.rows):
            if (
                row.get("status") == "completed"
                and row.get("variant") == spec["replay_of"]
                and row.get("identity") == group["identity"]
                and row.get("keep") == spec["keep"]
            ):
                logs = [log["log_path"] for log in row.get("logs", {}).values()]
                if not logs or any(not Path(path).is_file() for path in logs):
                    raise RuntimeError("Matching replay run has missing experiment logs")
                return {
                    "source_label": row["label"],
                    "logs": logs,
                    "wall_seconds": row.get("wall_seconds"),
                    "artifact_sha256": row.get("artifact", {}).get("sha256"),
                    "artifact_pixels": row.get("artifact", {}).get("pixels"),
                }
        raise RuntimeError("No completed live Jev run matches this case, seed, inputs, dimensions and settings")


def summarize(runs):
    measured = [row for row in runs if row.get("status") == "completed" and row.get("repetition", 0) > 0]
    groups = {}
    for row in measured:
        groups.setdefault(row["group_id"], {}).setdefault(row["variant"], []).append(row["wall_seconds"])
    summaries = []
    ratios = {}
    replay_sources = {row["variant"]: row["replay_of"] for row in measured if row.get("replay_of")}
    for group, variants in groups.items():
        baseline = statistics.median(variants["off"]) if "off" in variants else None
        for variant, times in variants.items():
            median = statistics.median(times)
            ratio = baseline / median if baseline is not None and median > 0 else None
            summaries.append(
                {
                    "group_id": group,
                    "variant": variant,
                    "samples": len(times),
                    "median_wall_seconds": median,
                    "min_wall_seconds": min(times),
                    "max_wall_seconds": max(times),
                    "speedup_vs_off": ratio,
                    "speedup_vs_live": statistics.median(variants[replay_sources[variant]]) / median
                    if median > 0 and replay_sources.get(variant) in variants
                    else None,
                }
            )
            if ratio is not None:
                ratios.setdefault(variant, []).append(ratio)
    return {
        "run_status_counts": dict(Counter(row.get("status", "unknown") for row in runs)),
        "measured_samples": len(measured),
        "groups": summaries,
        "paired_speedups": {
            key: {"groups": len(values), "median_speedup_vs_off": statistics.median(values)}
            for key, values in ratios.items()
        },
        "quality": "unscored; timing and pixel identity are not perceptual-quality evidence",
    }


def write_report(root, report):
    report["summary"] = summarize(report["runs"])
    atomic_json(Path(root) / "benchmark.json", report)
    columns = (
        "group_id",
        "variant",
        "samples",
        "median_wall_seconds",
        "min_wall_seconds",
        "max_wall_seconds",
        "speedup_vs_off",
        "speedup_vs_live",
    )
    temporary = Path(root) / "summary.csv.part"
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(report["summary"]["groups"])
    temporary.replace(Path(root) / "summary.csv")


def execute_plan(plan, root, runner, *, replay_index=None):
    root = Path(root)
    replay_index = replay_index or ReplayIndex(plan.get("replay_from"))
    report = {**plan, "status": "running", "runs": []}
    write_report(root, report)
    try:
        for group in plan["groups"]:
            runner.prepare(group)
            for spec in group["schedule"]:
                row = {**spec, "group_id": group["id"], "case_id": group["case"]["id"], "identity": group["identity"]}
                try:
                    directory = root / spec["label"]
                    directory.mkdir(exist_ok=False)
                    replay = replay_index.lookup(group, spec) if spec.get("replay_of") else None
                    print(f"BENCHMARK_START {spec['label']}", flush=True)  # noqa: T201
                    row.update(runner.generate(group, spec, directory, replay))
                    row["status"] = "completed"
                    if replay:
                        row["replay_source"] = replay
                        source_sha = replay.get("artifact_sha256")
                        current_sha = row.get("artifact", {}).get("sha256")
                        row["replay_file_identical"] = source_sha == current_sha if source_sha and current_sha else None
                        source_pixels = replay.get("artifact_pixels")
                        current_pixels = row.get("artifact", {}).get("pixels")
                        row["replay_pixel_identical"] = (
                            source_pixels == current_pixels if source_pixels and current_pixels else None
                        )
                    replay_index.add(row)
                except BaseException as exc:
                    row.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                    report["runs"].append(row)
                    write_report(root, report)
                    raise
                report["runs"].append(row)
                write_report(root, report)
                print(  # noqa: T201
                    "BENCHMARK_RESULT "
                    + json.dumps({key: row[key] for key in ("label", "status", "wall_seconds", "output_path")}),
                    flush=True,
                )
        report["status"] = "completed"
    except BaseException as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        write_report(root, report)
    return report


def save_plan(root, plan, *, dry_run):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "benchmark.json").exists() or (root / "plan.json").exists():
        raise ValueError("Use a new output directory so earlier benchmark evidence is not overwritten")
    atomic_json(root / "plan.json", plan)
    atomic_json(root / "cases.json", {"schema": 1, "cases": plan["cases"]})
    if dry_run:
        write_report(root, {**plan, "status": "dry_run", "runs": []})
    print(  # noqa: T201
        json.dumps(
            {
                "plan": str(root / "plan.json"),
                "dry_run": dry_run,
                "planned_runs": plan["planned_runs"],
                "planned_live_generations": plan["planned_live_generations"],
                "planned_source_generations": plan.get("planned_source_generations", 0),
                "skipped_cases": plan["skipped_cases"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return root
