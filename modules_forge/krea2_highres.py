"""Shared Krea2 high-resolution refinement profiles and prompt guidance."""

from __future__ import annotations

from contextlib import contextmanager


EXACT_IMG2IMG_STEPS = True
EXACT_IMG2IMG_STEPS_SCOPE = "internal_tiles_only"
KREA2_PHASEWEAVE_PRODUCT_NAME = "Krea2 PhaseWeave 4K"
KREA2_PHASEWEAVE_PROFILE_KEY = "phaseweave_4k"
KREA2_PHASEWEAVE_MERGE_MODE = "phase_weave"
LEGACY_CONSENSUS_MERGE_MODE = "consensus"


class ProcessingSnapshot:
    """Restore a processing object and the contents of its mutable containers."""

    def __init__(self, processing):
        self.values = dict(vars(processing))
        self.mutable_contents = {}
        for name, value in self.values.items():
            if isinstance(value, list):
                self.mutable_contents[name] = ("list", list(value))
            elif isinstance(value, dict):
                self.mutable_contents[name] = ("dict", dict(value))
            elif isinstance(value, set):
                self.mutable_contents[name] = ("set", set(value))

    def restore(self, processing, *, preserve: tuple[str, ...] = ()):
        preserve_names = set(preserve)
        preserved_values = {
            name: getattr(processing, name)
            for name in preserve_names
            if hasattr(processing, name)
        }
        for name, (kind, saved) in self.mutable_contents.items():
            if name in preserve_names:
                continue
            original = self.values[name]
            if kind == "list":
                original[:] = saved
            elif kind == "dict":
                original.clear()
                original.update(saved)
            else:
                original.clear()
                original.update(saved)
        current = vars(processing)
        current.clear()
        current.update(self.values)
        current.update(preserved_values)


@contextmanager
def internal_exact_img2img_steps(processing):
    """Force exact img2img steps for one internal call and restore the request.

    Forge applies ``processing.override_settings`` immediately around
    :func:`modules.processing.process_images` and restores the global option in its
    own ``finally`` block.  GUI and API generation are both serialized by Forge's
    generation queue, so using the request-local override avoids a long-lived global
    mutation.  Copying the override dictionary also makes nested internal calls safe:
    every scope restores the exact object and restore flag that it received.
    """

    had_overrides = hasattr(processing, "override_settings")
    original_overrides = getattr(processing, "override_settings", None)
    had_restore_flag = hasattr(processing, "override_settings_restore_afterwards")
    original_restore_flag = getattr(
        processing, "override_settings_restore_afterwards", None
    )

    overrides = dict(original_overrides or {})
    overrides["img2img_fix_steps"] = True
    processing.override_settings = overrides
    processing.override_settings_restore_afterwards = True
    try:
        yield processing
    finally:
        if had_overrides:
            processing.override_settings = original_overrides
        elif hasattr(processing, "override_settings"):
            delattr(processing, "override_settings")
        if had_restore_flag:
            processing.override_settings_restore_afterwards = original_restore_flag
        elif hasattr(processing, "override_settings_restore_afterwards"):
            delattr(processing, "override_settings_restore_afterwards")


KREA2_DETAIL_PROMPT_SUFFIX = (
    "Preserve the exact source image: every subject identity, facial proportions, age, "
    "expression, gaze, anatomy, pose, hand and finger count, hair flow, silhouette, "
    "camera, framing, depth of field, lighting, object count, readable text, graphics, "
    "and scene geometry. Add only coherent, non-repeating, location-specific fine "
    "detail already implied by the source. For people, resolve natural hair strands, "
    "eyelashes, clean eyelid lines, iris radial structure, and catchlights without "
    "changing identity or skin style. Where present, resolve material-appropriate "
    "fabric weave, seams, embroidery, hard-surface edges, restrained wear, wood grain, "
    "stone pores, foliage veins, transparent surfaces, liquid highlights, line art, "
    "and small depth-consistent reflections. Keep clean or stylized regions clean; do "
    "not force photographic pores or texture into illustration, anime, flat-color, or "
    "graphic-design areas. Every microdetail must follow the existing perspective, "
    "local geometry, material scale, and source style. Do not invent or remove facial "
    "features, people, limbs, hands, fingers, eyes, objects, text, logos, silhouettes, "
    "or repeated patterns. Do not introduce random grain, fake noise, oversharpening "
    "halos, doubled contours, changed crop edges, or tile seams."
)

KREA2_NATIVE_PROMPT_PROFILES = {
    "generic": (
        "Treat the preceding user prompt as authoritative. Create one coherent, "
        "high-quality image containing exactly the requested subjects, objects, text, "
        "style, setting, composition, camera, lighting, materials, and colors; do not "
        "invent a character, creature, prop, genre, or palette that the user did not "
        "request. Establish useful native-resolution detail with clean silhouettes, "
        "consistent perspective, readable forms, intentional edge hierarchy, and "
        "material- and style-appropriate local structure. Resolve fine detail only "
        "where it is supported by the prompt: for example natural strands and fabric "
        "construction on people, crisp manufactured edges on objects, or coherent "
        "foliage and surface variation in environments. Keep flat, minimalist, "
        "illustrative, graphic, painterly, and photographic regions faithful to their requested "
        "visual language instead of forcing one universal texture treatment. Avoid "
        "random grain, fake noise, plastic smoothing, oversharpening halos, doubled "
        "contours, accidental repetitions, malformed anatomy, unintended text, logos, "
        "signatures, and watermarks."
    ),
    "anime-slime-case-study": (
        "Create one exceptionally detailed, clean anime illustration from these subject "
        "attributes. One large cute translucent green slime is mandatory and must be "
        "clearly visible in the lower foreground; do not omit, replace, crop out, or hide "
        "it. Keep a readable face and silhouette, but establish dense intentional "
        "drawing at the native stage: many individually resolved layered hair locks and "
        "fine flyaway strands following the long wavy flow; faceted purple horn ridges and "
        "controlled glossy highlights; sharply constructed purple irises with radial fibers, "
        "concentric color variation, catchlights, crisp lashes, and clean eyelid lines; a "
        "complex dark gothic outfit with layered lace, embroidery, seams, woven texture, "
        "fasteners, ribbons, and small material-specific fold shadows. Frame the character "
        "from head to waist and keep one clearly visible cute translucent green slime with "
        "jig eyes in the lower foreground; give it thin specular edges, internal membranes, "
        "bubbles, filaments, and small depth-consistent caustics. Use precise line hierarchy "
        "and purposeful local microdetail rather than random grain. Avoid blur, plastic "
        "smoothing, fake noise, oversharpening halos, doubled contours, extra people, extra "
        "limbs, extra horns, text, logos, signatures, and watermarks."
    ),
}


KREA2_VRAM_CANVAS_PROFILES = {
    "structure_safe": {
        "merge_mode": LEGACY_CONSENSUS_MERGE_MODE,
        "phase_count": 1,
        "minimum_steps": 2,
        "maximum_steps": 4,
        "detail_knee": 0.035,
        "coarse_denoise": 0.12,
        "denoise": 0.08,
        "low_pass_radius": 12,
        "detail_gain": 1.0,
        "max_detail_delta": 32.0,
        "structure_sigma": 18.0,
        "base_detail_sigma": 6.0,
        "consensus_sigma": 8.0,
        "novel_detail_gain": 0.0,
        "novel_detail_max_delta": 8.0,
        "novel_detail_inner_radius": 1,
        "novel_detail_outer_radius": 4,
        "novel_detail_structure_sigma": 6.0,
        "novel_detail_consensus_sigma": 0.75,
        "novel_detail_consensus_strength": 8.0,
        "finish_detail_strength": 0.0,
        "finish_detail_radius": 1.0,
        "finish_detail_threshold": 1.0,
        "finish_max_detail_delta": 4.0,
    },
    "dense_detail_4k": {
        "merge_mode": LEGACY_CONSENSUS_MERGE_MODE,
        "phase_count": 2,
        "minimum_steps": 3,
        "maximum_steps": 4,
        "detail_knee": 0.035,
        "coarse_denoise": 0.16,
        "denoise": 0.13,
        "low_pass_radius": 12,
        "detail_gain": 1.25,
        "max_detail_delta": 32.0,
        "structure_sigma": 18.0,
        "base_detail_sigma": 2.5,
        "consensus_sigma": 8.0,
        "novel_detail_gain": 1.0,
        "novel_detail_max_delta": 8.0,
        "novel_detail_inner_radius": 1,
        "novel_detail_outer_radius": 4,
        "novel_detail_structure_sigma": 6.0,
        "novel_detail_consensus_sigma": 0.75,
        "novel_detail_consensus_strength": 8.0,
        "finish_detail_strength": 0.75,
        "finish_detail_radius": 1.0,
        "finish_detail_threshold": 0.6,
        "finish_max_detail_delta": 5.0,
    },
    "texture_rich_4k": {
        "merge_mode": LEGACY_CONSENSUS_MERGE_MODE,
        "phase_count": 2,
        "minimum_steps": 6,
        "maximum_steps": 6,
        "detail_knee": 0.025,
        "coarse_denoise": 0.22,
        "denoise": 0.18,
        "low_pass_radius": 10,
        "detail_gain": 1.55,
        "max_detail_delta": 40.0,
        "structure_sigma": 22.0,
        "base_detail_sigma": 1.5,
        "consensus_sigma": 12.0,
        "novel_detail_gain": 1.6,
        "novel_detail_max_delta": 12.0,
        "novel_detail_inner_radius": 1,
        "novel_detail_outer_radius": 5,
        "novel_detail_structure_sigma": 10.0,
        "novel_detail_consensus_sigma": 2.0,
        "novel_detail_consensus_strength": 4.0,
        "finish_detail_strength": 0.85,
        "finish_detail_radius": 1.4,
        "finish_detail_threshold": 0.4,
        "finish_max_detail_delta": 10.0,
    },
    KREA2_PHASEWEAVE_PROFILE_KEY: {
        "merge_mode": KREA2_PHASEWEAVE_MERGE_MODE,
        "phase_count": 2,
        "minimum_steps": 6,
        "maximum_steps": 6,
        "detail_knee": 0.025,
        "coarse_denoise": 0.20,
        "denoise": 0.16,
        "low_pass_radius": 10,
        "detail_gain": 1.55,
        "max_detail_delta": 40.0,
        "structure_sigma": 22.0,
        "base_detail_sigma": 1.5,
        "consensus_sigma": 10.0,
        "novel_detail_gain": 1.4,
        "novel_detail_max_delta": 10.0,
        "novel_detail_inner_radius": 1,
        "novel_detail_outer_radius": 5,
        "novel_detail_structure_sigma": 10.0,
        "novel_detail_consensus_sigma": 2.0,
        "novel_detail_consensus_strength": 4.0,
        "finish_detail_strength": 0.80,
        "finish_detail_radius": 1.2,
        "finish_detail_threshold": 0.45,
        "finish_max_detail_delta": 8.0,
    },
    "dense_detail_8k": {
        "merge_mode": LEGACY_CONSENSUS_MERGE_MODE,
        "phase_count": 2,
        "minimum_steps": 3,
        "maximum_steps": 4,
        "detail_knee": 0.035,
        "coarse_denoise": 0.12,
        "denoise": 0.11,
        "low_pass_radius": 12,
        "detail_gain": 1.25,
        "max_detail_delta": 32.0,
        "structure_sigma": 18.0,
        "base_detail_sigma": 2.0,
        "consensus_sigma": 8.0,
        "novel_detail_gain": 0.8,
        "novel_detail_max_delta": 6.0,
        "novel_detail_inner_radius": 1,
        "novel_detail_outer_radius": 4,
        "novel_detail_structure_sigma": 6.0,
        "novel_detail_consensus_sigma": 0.75,
        "novel_detail_consensus_strength": 8.0,
        "finish_detail_strength": 0.55,
        "finish_detail_radius": 1.0,
        "finish_detail_threshold": 0.8,
        "finish_max_detail_delta": 4.0,
    },
}


def uses_prompt_only_conditioning_cache(model: object) -> bool:
    """Return whether conditioning is independent of image shape and init latent."""

    return bool(getattr(model, "conditioning_cache_is_prompt_only", False))


def krea2_detail_prompt(base_prompt: str) -> str:
    """Append geometry-preserving detail guidance once without dropping the base."""

    base_prompt = str(base_prompt).strip()
    if not base_prompt:
        raise ValueError("base prompt must not be empty")
    if KREA2_DETAIL_PROMPT_SUFFIX in base_prompt:
        return base_prompt
    separator = " " if base_prompt.endswith((".", ",", ";", "!", "?")) else ". "
    return f"{base_prompt}{separator}{KREA2_DETAIL_PROMPT_SUFFIX}"


def krea2_native_detail_prompt(base_prompt: str, profile: str = "generic") -> str:
    """Append one explicit, subject-safe native-generation guidance profile."""

    base_prompt = str(base_prompt).strip()
    if not base_prompt:
        raise ValueError("base prompt must not be empty")
    try:
        suffix = KREA2_NATIVE_PROMPT_PROFILES[profile]
    except KeyError as exc:
        choices = ", ".join(sorted(KREA2_NATIVE_PROMPT_PROFILES))
        raise ValueError(
            f"unknown Krea2 native prompt profile {profile!r}; choose {choices}"
        ) from exc
    if suffix in base_prompt:
        return base_prompt
    separator = " " if base_prompt.endswith((".", ",", ";", "!", "?")) else ". "
    return f"{base_prompt}{separator}{suffix}"


def krea2_vram_canvas_profile(name: str) -> dict[str, float | int | str]:
    """Return an isolated profile copy so callers cannot mutate shared defaults."""

    try:
        return dict(KREA2_VRAM_CANVAS_PROFILES[name])
    except KeyError as exc:
        choices = ", ".join(sorted(KREA2_VRAM_CANVAS_PROFILES))
        raise ValueError(f"unknown Krea2 high-resolution profile {name!r}; choose {choices}") from exc
