"""Build visible gallery and Lanczos-versus-DetailWeave crop sheets for four scenes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "output/detailweave_batch_20260722"


@dataclass(frozen=True)
class Scene:
    number: str
    title: str
    source_name: str
    result_name: str
    face_crop: tuple[float, float, float, float]
    slime_crop: tuple[float, float, float, float]


SCENES = (
    Scene(
        "01",
        "Botanical library",
        "01_botanical_library_source.png",
        "01_botanical_library_detailweave_4k.png",
        (0.04, 0.31, 0.96, 0.66),
        (0.16, 0.65, 0.84, 0.91),
    ),
    Scene(
        "02",
        "Moonlit observatory",
        "02_moonlit_observatory_source.png",
        "02_moonlit_observatory_detailweave_4k.png",
        (0.04, 0.22, 0.96, 0.58),
        (0.16, 0.62, 0.84, 0.87),
    ),
    Scene(
        "03",
        "Autumn tea house",
        "03_autumn_teahouse_source.png",
        "03_autumn_teahouse_detailweave_4k.png",
        (0.03, 0.18, 0.97, 0.58),
        (0.15, 0.55, 0.85, 0.84),
    ),
    Scene(
        "04",
        "Snowy workshop",
        "04_snowy_workshop_source.png",
        "04_snowy_workshop_detailweave_4k.png",
        (0.04, 0.18, 0.96, 0.49),
        (0.16, 0.45, 0.84, 0.70),
    ),
)


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def load_pair(root: Path, scene: Scene) -> tuple[Image.Image, Image.Image]:
    source_path = root / "sources" / scene.source_name
    result_path = root / "deliverables" / scene.result_name
    missing = [str(path) for path in (source_path, result_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"batch artifacts missing: {missing}")
    source = Image.open(source_path).convert("RGB")
    result = Image.open(result_path).convert("RGB")
    if max(result.size) != 4096:
        raise ValueError(f"{result_path} is not a 4K-long-edge image: {result.size}")
    expected_ratio = source.width / source.height
    actual_ratio = result.width / result.height
    if abs(expected_ratio - actual_ratio) > 0.001:
        raise ValueError(
            f"source/result aspect mismatch for {scene.number}: "
            f"{source.size} versus {result.size}"
        )
    return source, result


def paste_contained(
    canvas: Image.Image,
    image: Image.Image,
    box: tuple[int, int, int, int],
    *,
    border: tuple[int, int, int] = (99, 108, 127),
) -> None:
    x0, y0, x1, y1 = box
    inner = ImageOps.contain(image, (x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
    x = x0 + (x1 - x0 - inner.width) // 2
    y = y0 + (y1 - y0 - inner.height) // 2
    canvas.paste(inner, (x, y))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((x - 1, y - 1, x + inner.width, y + inner.height), outline=border, width=2)


def normalized_crop(
    image: Image.Image, crop: tuple[float, float, float, float]
) -> Image.Image:
    x0, y0, x1, y1 = crop
    box = (
        round(x0 * image.width),
        round(y0 * image.height),
        round(x1 * image.width),
        round(y1 * image.height),
    )
    return image.crop(box)


def lanczos_reference(source: Image.Image, result_size: tuple[int, int]) -> Image.Image:
    return source.resize(result_size, Image.Resampling.LANCZOS)


def build_gallery(root: Path, output: Path) -> None:
    width, height = 3200, 1900
    canvas = Image.new("RGB", (width, height), (22, 25, 33))
    draw = ImageDraw.Draw(canvas)
    title_font = font(54, bold=True)
    label_font = font(32, bold=True)
    note_font = font(25)
    draw.text((80, 45), "DetailWeave 4K — four independent scenes", font=title_font, fill=(245, 247, 252))
    draw.text(
        (82, 112),
        "Two characters + one slime in every image / long edge 4096 px",
        font=note_font,
        fill=(177, 185, 202),
    )
    gap = 38
    cell_width = (width - 2 * 70 - 3 * gap) // 4
    image_top, image_bottom = 205, 1800
    for index, scene in enumerate(SCENES):
        _, result = load_pair(root, scene)
        x0 = 70 + index * (cell_width + gap)
        x1 = x0 + cell_width
        paste_contained(canvas, result, (x0, image_top, x1, image_bottom))
        label = f"{scene.number}  {scene.title}"
        label_width = draw.textbbox((0, 0), label, font=label_font)[2]
        draw.text(
            (x0 + (cell_width - label_width) // 2, 1825),
            label,
            font=label_font,
            fill=(239, 241, 247),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def build_before_after(root: Path, output: Path) -> None:
    width, height = 3200, 3400
    canvas = Image.new("RGB", (width, height), (22, 25, 33))
    draw = ImageDraw.Draw(canvas)
    title_font = font(54, bold=True)
    row_font = font(34, bold=True)
    label_font = font(27, bold=True)
    note_font = font(24)
    draw.text(
        (80, 42),
        "Four scenes — native source to DetailWeave 4K",
        font=title_font,
        fill=(245, 247, 252),
    )
    draw.text(
        (82, 110),
        "Top: generated source / Bottom: long-edge 4096 px result",
        font=note_font,
        fill=(177, 185, 202),
    )
    gap = 38
    left = 70
    cell_width = (width - left - 70 - 3 * gap) // 4
    rows = (
        ("Generated source", 190, 245, 1685, False),
        ("DetailWeave 4K", 1790, 1845, 3285, True),
    )
    pairs = [(scene, *load_pair(root, scene)) for scene in SCENES]
    for row_label, label_y, y0, y1, use_result in rows:
        draw.text((70, label_y), row_label, font=row_font, fill=(218, 222, 233))
        for index, (scene, source, result) in enumerate(pairs):
            image = result if use_result else source
            x0 = left + index * (cell_width + gap)
            x1 = x0 + cell_width
            paste_contained(canvas, image, (x0, y0, x1, y1))
            label = f"{scene.number}  {scene.title}"
            label_width = draw.textbbox((0, 0), label, font=label_font)[2]
            draw.text(
                (x0 + (cell_width - label_width) // 2, y1 + 20),
                label,
                font=label_font,
                fill=(239, 241, 247),
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def build_crops(root: Path, output: Path) -> None:
    width = 3200
    header_height = 200
    row_height = 630
    height = header_height + row_height * len(SCENES) + 70
    canvas = Image.new("RGB", (width, height), (246, 246, 243))
    draw = ImageDraw.Draw(canvas)
    title_font = font(50, bold=True)
    header_font = font(28, bold=True)
    row_font = font(28, bold=True)
    small_font = font(23)
    draw.text((60, 35), "Lanczos reference vs. DetailWeave 4K — same-coordinate crops", font=title_font, fill=(24, 27, 32))
    draw.text(
        (62, 103),
        "Lanczos columns contain no generated detail. DetailWeave columns are the 4K outputs.",
        font=small_font,
        fill=(78, 82, 91),
    )
    left = 225
    gap = 28
    cell_width = 710
    labels = ("Lanczos / characters", "DetailWeave / characters", "Lanczos / slime", "DetailWeave / slime")
    for column, label in enumerate(labels):
        x = left + column * (cell_width + gap)
        draw.text((x, 155), label, font=header_font, fill=(45, 50, 60))
    colors = ((177, 83, 65), (54, 116, 154), (177, 83, 65), (54, 116, 154))
    for row, scene in enumerate(SCENES):
        source, result = load_pair(root, scene)
        reference = lanczos_reference(source, result.size)
        crops = (
            normalized_crop(reference, scene.face_crop),
            normalized_crop(result, scene.face_crop),
            normalized_crop(reference, scene.slime_crop),
            normalized_crop(result, scene.slime_crop),
        )
        y0 = header_height + row * row_height + 55
        y1 = y0 + 525
        draw.text((45, y0 + 5), scene.number, font=font(40, bold=True), fill=(28, 32, 40))
        wrapped = scene.title.replace(" ", "\n", 1)
        draw.multiline_text((45, y0 + 70), wrapped, font=row_font, fill=(68, 72, 82), spacing=5)
        for column, crop_image in enumerate(crops):
            x0 = left + column * (cell_width + gap)
            paste_contained(
                canvas,
                crop_image,
                (x0, y0, x0 + cell_width, y1),
                border=colors[column],
            )
        draw.line((45, y1 + 27, width - 55, y1 + 27), fill=(202, 202, 197), width=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.root / "deliverables"
    gallery = output_dir / "detailweave_4k_four_scene_gallery.png"
    before_after = output_dir / "detailweave_four_scene_before_after.png"
    crops = output_dir / "detailweave_4k_lanczos_comparison_crops.png"
    build_gallery(args.root, gallery)
    build_before_after(args.root, before_after)
    build_crops(args.root, crops)
    print(f"GALLERY={gallery}")
    print(f"BEFORE_AFTER={before_after}")
    print(f"CROPS={crops}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
