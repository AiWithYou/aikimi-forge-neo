"""Grain Cleaner の逐次処理。python -m tools.grain_cleaner INPUT OUTPUT_DIR"""

import argparse
import json
from pathlib import Path

from PIL import Image

from modules.grain_cleaner import GrainSettings, clean_grain


def main():
    parser = argparse.ArgumentParser(description="構造保護付き微細粒状感抑制")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--preserve-detail", type=float, default=0.75)
    parser.add_argument("--chroma-strength", type=float, default=0.6)
    parser.add_argument("--grain-scale", type=float, default=1)
    parser.add_argument("--sample-roi", type=int, nargs=4)
    parser.add_argument("--apply-mask", type=Path)
    parser.add_argument("--protect-mask", type=Path)
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    settings = GrainSettings(
        args.strength,
        args.preserve_detail,
        args.chroma_strength,
        args.grain_scale,
        "sample" if args.sample_roi else "auto",
    )
    settings.validate()
    sources = (
        sorted(p for p in args.input.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
        if args.input.is_dir()
        else [args.input]
    )
    if not sources:
        parser.error("入力画像がありません")
    names = set()
    for source in sources:
        if source.stem.casefold() in names:
            parser.error("同じ名前の入力があります。別々の出力ディレクトリを指定してください")
        names.add(source.stem.casefold())
    # 実行単位の専用ディレクトリにより既存成果物との衝突を防ぐ。
    args.output.mkdir(parents=True, exist_ok=False)
    masks = {}
    for key in ("apply_mask", "protect_mask"):
        path = getattr(args, key)
        if path:
            with Image.open(path) as mask:
                masks[key] = mask.copy()
    for source in sources:
        with Image.open(source) as image:
            output, report, views = clean_grain(
                image, settings, sample_roi=args.sample_roi, diagnostics=args.diagnostics, **masks
            )
        output.save(args.output / f"{source.stem}_clean.png")
        for name, view in views.items():
            view.save(args.output / f"{source.stem}_{name}.png")
        (args.output / f"{source.stem}_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
