import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageOps, PngImagePlugin

from modules.krea2_quality import smart_finish_image, smart_finish_summary


def emit(message: str):
    sys.stdout.write(f"{message}\n")
    sys.stdout.flush()


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_smart.png")


def save_png(
    path: Path,
    image: Image.Image,
    *,
    parameters: str,
    report: dict,
    overwrite: bool,
    preserved_info: dict | None = None,
):
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    pnginfo = PngImagePlugin.PngInfo()
    preserved_info = preserved_info or {}
    for key, value in preserved_info.items():
        if isinstance(value, str) and key not in {"parameters", "krea2_smart_finish"}:
            pnginfo.add_text(key, value)
    if parameters:
        pnginfo.add_text("parameters", parameters)
    pnginfo.add_text(
        "krea2_smart_finish",
        json.dumps(report, ensure_ascii=False, separators=(",", ":")),
    )
    save_kwargs = {"pnginfo": pnginfo}
    for key in ("icc_profile", "exif", "dpi"):
        if key in preserved_info:
            save_kwargs[key] = preserved_info[key]
    image.save(path, format="PNG", **save_kwargs)


def validate_paths(input_path: Path, output_path: Path, report_path: Path):
    resolved = [
        path.expanduser().resolve(strict=False)
        for path in (input_path, output_path, report_path)
    ]
    if len(set(resolved)) != len(resolved):
        raise ValueError("--input, --output, and --report must be distinct paths.")
    if output_path.suffix.lower() != ".png":
        raise ValueError("--output must use a .png extension.")
    if report_path.suffix.lower() != ".json":
        raise ValueError("--report must use a .json extension.")


def prepare_source_image(source: Image.Image) -> Image.Image:
    oriented = ImageOps.exif_transpose(source)
    has_alpha = "A" in oriented.getbands() or "transparency" in oriented.info
    return oriented.convert("RGBA" if has_alpha else "RGB")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Adaptive chroma-mura cleanup, optional isolated-speckle repair, and "
            "coherent source-detail protection."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--color-strength", type=float, default=0.80)
    parser.add_argument("--analysis-long-edge", type=int, default=1536)
    parser.add_argument(
        "--despeckle",
        action="store_true",
        help="Repair isolated bright/dark points. Leave off for snow, stars, freckles, or deliberate grain.",
    )
    parser.add_argument("--speckle-threshold", type=int, default=0)
    parser.add_argument("--max-speckle-percent", type=float, default=0.35)
    parser.add_argument(
        "--detail-guard",
        action="store_true",
        help=(
            "Increase only coherent microdetail already present in the image; "
            "flat regions and strong edges are protected."
        ),
    )
    parser.add_argument("--detail-strength", type=float, default=0.55)
    parser.add_argument("--detail-radius", type=float, default=1.0)
    parser.add_argument("--detail-threshold", type=float, default=1.0)
    parser.add_argument("--max-detail-delta", type=float, default=4.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_path = Path(args.output) if args.output else default_output_path(input_path)
    report_path = (
        Path(args.report) if args.report else output_path.with_suffix(".quality.json")
    )
    validate_paths(input_path, output_path, report_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Pass --overwrite to replace it."
        )
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{report_path} already exists. Pass --overwrite to replace it."
        )

    with Image.open(input_path) as source:
        parameters = str(source.info.get("parameters", ""))
        image = prepare_source_image(source)
        preserved_info = dict(image.info)

    result, report = smart_finish_image(
        image,
        color_strength=args.color_strength,
        analysis_long_edge=args.analysis_long_edge,
        despeckle=args.despeckle,
        speckle_threshold=args.speckle_threshold,
        max_speckle_percent=args.max_speckle_percent,
        detail_guard=args.detail_guard,
        detail_strength=args.detail_strength,
        detail_radius=args.detail_radius,
        detail_threshold=args.detail_threshold,
        max_detail_delta=args.max_detail_delta,
    )
    summary = smart_finish_summary(report)
    parameters = (
        f"{parameters}, Krea2 Smart Finish: {summary}"
        if parameters
        else f"Krea2 Smart Finish: {summary}"
    )
    save_png(
        output_path,
        result,
        parameters=parameters,
        report=report,
        overwrite=args.overwrite,
        preserved_info=preserved_info,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    emit(f"INPUT={input_path}")
    emit(f"OUTPUT={output_path}")
    emit(f"REPORT={report_path}")
    emit(f"SUMMARY={summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
