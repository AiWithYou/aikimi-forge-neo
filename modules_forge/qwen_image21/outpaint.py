"""CPU-only preparation/stitching for the ausboss Qwen 2.1 outpaint recipe.

This module does not load a LoRA, run inference, or translate ComfyUI weights.
Keep the prepared source and geometry together; do not resize model results.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from PIL import Image, ImageChops, ImageOps

SOURCE = "https://huggingface.co/ausboss/Qwen-Image-2.1-Outpaint-LoRA"
CHECKED_ON = "2026-09-26"
GRID = 32
MAX_CANVAS_PIXELS = 2_097_152
MAX_SOURCE_PIXELS = 40_000_000
MAX_SIDE = 4096
PROMPT = (
    "Outpaint the image: replace the solid gray areas with a seamless "
    "continuation of the scene, keeping the existing picture unchanged."
)
WEIGHTS = {
    "v1": "qwen-image-2.1-outpaint.safetensors",
    "v2": "qwen-image-2.1-outpaint-v2.safetensors",
}


def _integer(value, name: str, lower: int, upper: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not lower <= value <= upper
        or not math.isfinite(value)
        or int(value) != value
    ):
        raise ValueError(f"{name}は{lower}〜{upper}の整数で指定してください。")
    return int(value)


def normalize_image(image: Image.Image) -> Image.Image:
    """Snapshot one image, applying EXIF once and retaining alpha information."""
    if not isinstance(image, Image.Image):
        raise ValueError("静止画像をアップロードしてください。")
    if image.width * image.height > MAX_SOURCE_PIXELS:
        raise ValueError("入力画像は4000万画素以内にしてください。")
    if getattr(image, "n_frames", 1) != 1:
        raise ValueError("アニメーションではなく静止画像を指定してください。")
    if image.mode not in {"1", "L", "LA", "P", "PA", "RGB", "RGBA"}:
        raise ValueError("入力は8-bitのRGB／RGBA・グレースケール・パレット画像にしてください。16-bit・HDR・CMYKは変換せず拒否します。")
    try:
        result = ImageOps.exif_transpose(image)
        alpha = "A" in result.getbands() or "transparency" in result.info
        return result.convert("RGBA" if alpha else "RGB")
    except (OSError, SyntaxError) as exc:
        raise ValueError("画像を読み込めません。アップロードし直してください。") from exc


@dataclass(frozen=True)
class Plan:
    source_width: int
    source_height: int
    left: int
    top: int
    right: int
    bottom: int

    def __post_init__(self) -> None:
        for name in ("source_width", "source_height", "left", "top", "right", "bottom"):
            lower = 1 if name.startswith("source_") else 0
            object.__setattr__(self, name, _integer(getattr(self, name), name, lower, MAX_SIDE))

    @property
    def size(self) -> tuple[int, int]:
        return self.source_width + self.left + self.right, self.source_height + self.top + self.bottom

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.left + self.source_width, self.top + self.source_height

    def validate(self) -> None:
        for name in ("source_width", "source_height"):
            _integer(getattr(self, name), name, 1, MAX_SIDE)
        for name in ("left", "top", "right", "bottom"):
            _integer(getattr(self, name), name, 0, MAX_SIDE)
        if not any((self.left, self.top, self.right, self.bottom)):
            raise ValueError("少なくとも一方向に余白を指定してください。")
        width, height = self.size
        if not all(256 <= side <= MAX_SIDE and side % GRID == 0 for side in self.size):
            raise ValueError("キャンバスの各辺は256〜4096 px、32の倍数にしてください。")
        if width * height > MAX_CANVAS_PIXELS:
            raise ValueError("キャンバスは約2 MP以内にしてください。元画像を縮小するか余白を減らしてください。")


def _align_axis(size: int, before: int, after: int, label: str) -> tuple[int, int]:
    extra = (-size - before - after) % GRID
    # Never change aspect ratio or add padding to a side the user disabled.
    if extra and not (before or after):
        raise ValueError(f"{label}が32の倍数ではありません。この方向にも少量の余白を指定してください。")
    if before and after:
        before += extra // 2
        after += extra - extra // 2
    elif before:
        before += extra
    elif after:
        after += extra
    return before, after


def prepare(
    source: Image.Image,
    left: int = 128,
    top: int = 128,
    right: int = 128,
    bottom: int = 128,
) -> tuple[Image.Image, Image.Image, Plan]:
    """Return an independent original, gray RGB reference, and aligned geometry.

    Transparent source pixels are shown over gray *only* in the model reference.
    The untouched RGBA snapshot is used again when stitching.
    """
    pads = [_integer(value, name, 0, MAX_SIDE) for name, value in zip(
        ("左", "上", "右", "下"), (left, top, right, bottom), strict=True,
    )]
    source = normalize_image(source)
    left, top, right, bottom = pads
    left, right = _align_axis(source.width, left, right, "幅")
    top, bottom = _align_axis(source.height, top, bottom, "高さ")
    plan = Plan(source.width, source.height, left, top, right, bottom)
    plan.validate()
    canvas = Image.new("RGB", plan.size, (128, 128, 128))
    if source.mode == "RGBA":
        canvas.paste(source, (left, top), source.getchannel("A"))
    else:
        canvas.paste(source, (left, top))
    return source, canvas, plan


def recipe(plan: Plan, version: str = "v2", scene: str = "") -> dict:
    """Describe the external ComfyUI recipe; this is not an executable graph."""
    plan.validate()
    if not isinstance(version, str) or version not in WEIGHTS:
        raise ValueError("Outpaint LoRAはv1またはv2を選んでください。")
    if not isinstance(scene, str) or len(scene) > 10000:
        raise ValueError("場面の説明は10000文字以内で入力してください。")
    prompt = PROMPT + ("\nScene: " + scene.strip() if scene.strip() else "")
    warnings = ["これは前後処理用です。LoRAの読み込みと生成は別途ComfyUIで行います。"]
    if version == "v1" and math.prod(plan.size) > 1_048_576:
        warnings.append("v1の学習サイズは約1 MPです。このキャンバスはそれを超えます。")
    if plan.source_width * plan.source_height / math.prod(plan.size) < 0.15:
        warnings.append("元画像の面積が15%未満です。大幅な拡張では生成内容の変動が大きくなります。")
    return {
        "schema": "aikimi-qwen21-outpaint-recipe-v1",
        "source": SOURCE,
        "checked_on": CHECKED_ON,
        "base_model": "Qwen/Qwen-Image-2.1",
        "weights": WEIGHTS[version],
        "weight_format": "ComfyUI (not verified for Forge Diffusers)",
        "geometry": asdict(plan),
        "canvas_size": list(plan.size),
        "fill": "#808080",
        "prompt": prompt,
        "comfyui": {
            "reference_input": "image_1",
            "reference_resolution": 0,
            "latent_source": "Text Encode Qwen Image 2.1 latent output",
            "steps": 25,
            "cfg": 1.0,
            "sampler": "euler",
            "scheduler": "simple",
            "denoise": 1.0,
            "latent_noise_mask": False,
        },
        "warnings": warnings,
    }


def _feather_mask(plan: Plan, feather: int) -> Image.Image:
    width, height = plan.source_width, plan.source_height
    if not feather:
        return Image.new("L", (width, height), 255)

    def ramp(length: int, before: bool, after: bool) -> list[int]:
        return [round(255 * min(
            1.0,
            i / feather if before else 1.0,
            (length - 1 - i) / feather if after else 1.0,
        )) for i in range(length)]

    horizontal = Image.new("L", (width, 1))
    horizontal.putdata(ramp(width, bool(plan.left), bool(plan.right)))
    vertical = Image.new("L", (1, height))
    vertical.putdata(ramp(height, bool(plan.top), bool(plan.bottom)))
    return ImageChops.darker(
        horizontal.resize((width, height), Image.Resampling.NEAREST),
        vertical.resize((width, height), Image.Resampling.NEAREST),
    )


def stitch(source: Image.Image, generated: Image.Image, plan: Plan, feather: int = 32) -> Image.Image:
    """Restore the original; blend only inside edges that actually have padding.

    With feather=0 the entire original rectangle is copied byte-for-byte.
    With feather>0 only its border changes; its opaque-weight interior is exact,
    including hidden RGB under alpha=0. New areas are not modified.
    """
    if not isinstance(plan, Plan):
        raise ValueError("先に余白付き画像を準備してください。")
    plan.validate()
    feather = _integer(feather, "境界ぼかし", 0, min(256, (min(plan.source_width, plan.source_height) - 1) // 2))
    source = normalize_image(source)
    generated = normalize_image(generated)
    if source.size != (plan.source_width, plan.source_height):
        raise ValueError("元画像のサイズが準備時と違います。最初から準備し直してください。")
    if generated.size != plan.size:
        raise ValueError(f"生成結果は{plan.size[0]}×{plan.size[1]} pxである必要があります。自動リサイズは行いません。")
    mode = "RGBA" if source.mode == "RGBA" or generated.mode == "RGBA" else "RGB"
    source, result = source.convert(mode), generated.convert(mode)
    if not feather:
        result.paste(source, (plan.left, plan.top))
        return result
    mask = _feather_mask(plan, feather)
    crop = result.crop(plan.box)
    if mode == "RGBA":
        # Premultiplication avoids dark halos at transparent boundaries.
        blended = Image.composite(source.convert("RGBa"), crop.convert("RGBa"), mask).convert("RGBA")
        # Premultiplication roundoff must not change protected original pixels.
        exact = mask.point([255 if value == 255 else 0 for value in range(256)])
        blended = Image.composite(source, blended, exact)
        unchanged = mask.point([255 if value == 0 else 0 for value in range(256)])
        blended = Image.composite(crop, blended, unchanged)
    else:
        blended = Image.composite(source, crop, mask)
    result.paste(blended, (plan.left, plan.top))
    return result
