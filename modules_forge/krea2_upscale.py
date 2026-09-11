import math
import re


SAFE_DIFFUSION_LONG_EDGE = 3584
KREA2_TURBO_NATIVE_LONG_EDGE = 2048
KREA2_RAW_NATIVE_LONG_EDGE = 1024
KREA2_MAX_TILE_PIXELS = 1_638_400
KREA2_DIFFUSION_ALIGNMENT = 16
KREA2_MIN_TILE_EDGE = 256
KREA2_DEFAULT_SAMPLER = "DPM++ 2M SDE"
KREA2_DEFAULT_SCHEDULER = "Simple"
KREA2_T2I_STEPS = 4
KREA2_I2I_STEPS = 8
KREA2_LOCAL_REFINE_STEPS = 4
KREA2_DEFAULT_CFG = 1.0
KREA2_DEFAULT_SHIFT = 1.15
KREA2_STAGE1_DENOISE = 0.10
KREA2_DEFAULT_DENOISE = 0.12


def multiple_of(value: float, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def floor_multiple(value: int, multiple: int) -> int:
    require_positive_int(value, "value")
    require_positive_int(multiple, "multiple")
    if value < multiple:
        raise ValueError(f"value must be >= alignment ({multiple}).")
    return (int(value) // multiple) * multiple


def ceil_multiple(value: int, multiple: int) -> int:
    require_positive_int(value, "value")
    require_positive_int(multiple, "multiple")
    return int(math.ceil(value / multiple)) * multiple


def aligned_size(
    width: int, height: int, alignment: int = KREA2_DIFFUSION_ALIGNMENT
) -> tuple[int, int]:
    require_positive_int(width, "width")
    require_positive_int(height, "height")
    require_positive_int(alignment, "alignment")
    return multiple_of(width, alignment), multiple_of(height, alignment)


def require_positive_int(value: int | None, name: str):
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be > 0.")


def target_size(
    width: int,
    height: int,
    long_edge: int,
    explicit_width: int | None,
    explicit_height: int | None,
) -> tuple[int, int]:
    if (explicit_width is None) != (explicit_height is None):
        raise ValueError("--width and --height must be passed together.")
    require_positive_int(width, "source width")
    require_positive_int(height, "source height")
    require_positive_int(long_edge, "--long-edge")
    require_positive_int(explicit_width, "--width")
    require_positive_int(explicit_height, "--height")

    if explicit_width is not None and explicit_height is not None:
        return int(explicit_width), int(explicit_height)
    if width >= height:
        return int(long_edge), max(1, int(round(height * long_edge / width)))
    return max(1, int(round(width * long_edge / height))), int(long_edge)


def size_from_long_edge(
    width: int, height: int, long_edge: int, alignment: int = 64
) -> tuple[int, int]:
    require_positive_int(width, "width")
    require_positive_int(height, "height")
    require_positive_int(long_edge, "long edge")
    require_positive_int(alignment, "alignment")

    if width >= height:
        return multiple_of(long_edge, alignment), multiple_of(
            height * long_edge / width, alignment
        )
    return multiple_of(width * long_edge / height, alignment), multiple_of(
        long_edge, alignment
    )


def capped_diffusion_size(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    cap_long_edge: int,
) -> tuple[int, int]:
    if cap_long_edge < 0:
        raise ValueError("diffusion long edge cap must be >= 0.")
    if cap_long_edge == 0:
        return aligned_size(target_width, target_height)

    aligned_cap = floor_multiple(cap_long_edge, KREA2_DIFFUSION_ALIGNMENT)
    source_long_edge = max(source_width, source_height)

    if max(target_width, target_height) <= cap_long_edge:
        aligned_target = aligned_size(target_width, target_height)
        if (
            max(aligned_target) <= cap_long_edge
            and max(aligned_target) >= source_long_edge
        ):
            return aligned_target
        if aligned_cap < source_long_edge:
            raise ValueError(
                "aligned diffusion long edge cap must be >= source long edge."
            )
        return size_from_long_edge(
            target_width,
            target_height,
            aligned_cap,
            alignment=KREA2_DIFFUSION_ALIGNMENT,
        )

    if aligned_cap < source_long_edge:
        raise ValueError(
            "aligned diffusion long edge cap must be >= source long edge when it caps the diffusion pass."
        )
    return size_from_long_edge(
        target_width,
        target_height,
        aligned_cap,
        alignment=KREA2_DIFFUSION_ALIGNMENT,
    )


def native_diffusion_long_edge(profile: str) -> int:
    limits = {
        "raw": KREA2_RAW_NATIVE_LONG_EDGE,
        "turbo": KREA2_TURBO_NATIVE_LONG_EDGE,
        "custom": KREA2_TURBO_NATIVE_LONG_EDGE,
    }
    try:
        return limits[profile]
    except KeyError as exc:
        raise ValueError("Krea2 model profile must be raw, turbo, or custom.") from exc


def require_native_diffusion_size(
    width: int,
    height: int,
    profile: str,
    allow_non_native: bool = False,
):
    require_positive_int(width, "diffusion width")
    require_positive_int(height, "diffusion height")
    if allow_non_native:
        return

    limit = native_diffusion_long_edge(profile)
    long_edge = max(width, height)
    if long_edge > limit:
        raise ValueError(
            f"diffusion long edge {long_edge} exceeds the {profile} profile's "
            f"resolution guard {limit}. Use a smaller diffusion cap or explicitly "
            "allow non-native diffusion."
        )


def validate_tile_geometry(
    tile_width: int,
    tile_height: int,
    overlap: int,
    batch_size: int,
    *,
    max_tile_pixels: int = KREA2_MAX_TILE_PIXELS,
):
    require_positive_int(tile_width, "tile width")
    require_positive_int(tile_height, "tile height")
    require_positive_int(batch_size, "tile batch size")
    require_positive_int(max_tile_pixels, "max tile pixels")
    if tile_width < KREA2_MIN_TILE_EDGE or tile_height < KREA2_MIN_TILE_EDGE:
        raise ValueError(
            f"Krea2 tile dimensions must be >= {KREA2_MIN_TILE_EDGE} pixels."
        )
    if (
        tile_width % KREA2_DIFFUSION_ALIGNMENT != 0
        or tile_height % KREA2_DIFFUSION_ALIGNMENT != 0
    ):
        raise ValueError(
            f"Krea2 tile dimensions must be divisible by {KREA2_DIFFUSION_ALIGNMENT}."
        )
    if overlap < 0:
        raise ValueError("tile overlap must be >= 0.")
    if overlap and overlap % KREA2_DIFFUSION_ALIGNMENT != 0:
        raise ValueError(
            f"non-zero tile overlap must be divisible by {KREA2_DIFFUSION_ALIGNMENT}."
        )
    if overlap >= min(tile_width, tile_height):
        raise ValueError("tile overlap must be smaller than both tile dimensions.")
    tile_pixels = tile_width * tile_height
    if tile_pixels > max_tile_pixels:
        raise ValueError(
            f"tile area {tile_pixels} exceeds the Krea2 limit {max_tile_pixels}."
        )
    if batch_size != 1:
        raise ValueError("Krea2 safe high-resolution mode requires tile batch size 1.")


def require_safe_diffusion_size(width: int, height: int, allow_unsafe: bool = False):
    require_positive_int(width, "diffusion width")
    require_positive_int(height, "diffusion height")
    if allow_unsafe:
        return

    long_edge = max(width, height)
    if long_edge > SAFE_DIFFUSION_LONG_EDGE:
        raise ValueError(
            f"diffusion long edge {long_edge} exceeds safe limit {SAFE_DIFFUSION_LONG_EDGE}. "
            "Set a smaller diffusion cap or explicitly allow unsafe large diffusion."
        )


def replace_infotext_size(
    infotext: str,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> str:
    if not infotext:
        return infotext

    source = f"{source_width}x{source_height}"
    target = f"{target_width}x{target_height}"
    lines = infotext.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        if not lines[index].lstrip().startswith("Steps:"):
            continue
        pattern = rf"(?<!\w)Size:\s*{re.escape(source)}(?=,|$)"
        updated, replacements = re.subn(pattern, f"Size: {target}", lines[index])
        if replacements == 0:
            updated, _ = re.subn(
                r"(?<!\w)Size:\s*\d+x\d+(?=,|$)",
                f"Size: {target}",
                lines[index],
                count=1,
            )
        lines[index] = updated
        break
    return "\n".join(lines)


def auto_first_pass_long_edge(
    source_width: int, source_height: int, final_width: int, final_height: int
) -> int:
    require_positive_int(source_width, "source width")
    require_positive_int(source_height, "source height")
    require_positive_int(final_width, "final width")
    require_positive_int(final_height, "final height")

    source_long_edge = max(source_width, source_height)
    final_long_edge = max(final_width, final_height)
    if final_long_edge <= source_long_edge:
        raise ValueError("final long edge must be larger than source long edge.")

    first_pass_long_edge = multiple_of(
        math.sqrt(source_long_edge * final_long_edge), 64
    )
    first_pass_long_edge = max(first_pass_long_edge, multiple_of(source_long_edge, 64))
    first_pass_long_edge = min(first_pass_long_edge, final_long_edge - 64)
    if first_pass_long_edge < source_long_edge:
        raise ValueError(
            "final long edge is too close to source long edge for a two-stage upscale."
        )
    return first_pass_long_edge


def two_stage_sizes(
    source_width: int,
    source_height: int,
    final_width: int,
    final_height: int,
    first_pass_long_edge: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    require_positive_int(source_width, "source width")
    require_positive_int(source_height, "source height")
    require_positive_int(final_width, "final width")
    require_positive_int(final_height, "final height")
    if first_pass_long_edge < 0:
        raise ValueError("first pass long edge must be >= 0.")

    source_long_edge = max(source_width, source_height)
    final_long_edge = max(final_width, final_height)

    if final_long_edge <= source_long_edge:
        raise ValueError("final long edge must be larger than source long edge.")
    if first_pass_long_edge == 0:
        first_pass_long_edge = auto_first_pass_long_edge(
            source_width, source_height, final_width, final_height
        )
    if first_pass_long_edge < source_long_edge:
        raise ValueError("first pass long edge must be >= source long edge.")
    aligned_first_pass = ceil_multiple(first_pass_long_edge, 64)
    if aligned_first_pass >= final_long_edge:
        raise ValueError("first pass long edge must be < final long edge.")

    stage1_size = size_from_long_edge(
        final_width, final_height, aligned_first_pass, alignment=64
    )
    if max(stage1_size) < source_long_edge:
        raise ValueError("aligned first pass size must be >= source long edge.")
    return stage1_size, (final_width, final_height)
