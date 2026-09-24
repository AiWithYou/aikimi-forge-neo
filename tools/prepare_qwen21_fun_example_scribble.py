"""Convert our original ring lineart to a white-on-black Scribble guide."""

from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets/qwen-image21-fun-controlnet"


def main() -> None:
    with Image.open(ASSETS / "ring-observatory-lineart.png") as source:
        image = ImageOps.fit(
            source.convert("L"), (1024, 768), method=Image.Resampling.LANCZOS
        )
        # Keep darker architectural strokes, then thicken them slightly.
        image = image.point(lambda value: 255 if value < 128 else 0)
        image = image.filter(ImageFilter.MaxFilter(3))
        image.save(ASSETS / "ring-observatory-scribble.png")


if __name__ == "__main__":
    main()
