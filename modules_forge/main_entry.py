import hashlib
import logging
import os.path
from collections import Counter, defaultdict

import gradio as gr
import torch
from gradio.context import Context
from rich import print_json

from backend import memory_management
from backend.args import dynamic_args
from backend.logging import setup_logger
from modules import (
    infotext_utils,
    paths,
    processing,
    sd_models,
    shared,
    shared_items,
    ui_common,
)
from modules_forge.presets import PresetArch, is_video, use_distill, use_shift

logger = logging.getLogger("ui_models")
setup_logger(logger)

ui_forge_preset: gr.Radio
ui_checkpoint: gr.Dropdown
ui_vae: gr.Dropdown
ui_forge_unet_dtype: gr.Radio

forge_unet_storage_dtype_options: dict[str, tuple[torch.dtype, bool]] = {
    "Automatic": (None, False),
    "Automatic (fp16 LoRA)": (None, True),
    "float8-e4m3fn": (torch.float8_e4m3fn, False),
    "float8-e4m3fn (fp16 LoRA)": (torch.float8_e4m3fn, True),
    "float8-e5m2": (torch.float8_e5m2, False),
    "float8-e5m2 (fp16 LoRA)": (torch.float8_e5m2, True),
}

if memory_management.bnb_enabled():
    forge_unet_storage_dtype_options.update(
        {
            "bnb-nf4": ("nf4", False),
            "bnb-nf4 (fp16 LoRA)": ("nf4", True),
            "bnb-fp4": ("fp4", False),
            "bnb-fp4 (fp16 LoRA)": ("fp4", True),
        }
    )


MODULE_FILE_EXTENSIONS = frozenset({".ckpt", ".pt", ".pth", ".bin", ".safetensors", ".sft", ".gguf"})
_UNRESOLVED_MODULE_PREFIX = "__forge_module_unresolved__:"

module_list: dict[str, os.PathLike] = {}
_module_path_to_selector: dict[str, str] = {}
_module_selector_to_kind: dict[str, str] = {}
_module_selector_to_infotext: dict[str, str] = {}
_module_basename_to_selectors: dict[str, tuple[str, ...]] = {}
_module_infotext_to_selectors: dict[str, tuple[str, ...]] = {}
_unresolved_module_choices: dict[str, str] = {}
_module_registry_initialized = False
_legacy_module_migration_complete = False


class ModuleResolutionError(ValueError):
    def __init__(self, value: object, reason: str, selectors: tuple[str, ...] = ()):
        name = portable_module_basename(value) or "module"
        if reason == "ambiguous":
            message = f"Additional module {name!r} matches multiple files; reselect one of: {', '.join(selectors)}"
        elif reason == "path":
            message = "Additional-module overrides must use a listed selector or unique filename, not a path"
        else:
            message = f"Additional module {name!r} is not available; refresh the model list and reselect it"
        super().__init__(message)
        self.reason = reason


def portable_module_basename(value: object) -> str:
    return str(value or "").replace("\\", "/").rsplit("/", 1)[-1]


def _canonical_module_path(value: os.PathLike | str) -> str:
    return os.path.realpath(os.path.abspath(os.fspath(value)))


def _normalize_module_path(value: os.PathLike | str) -> str:
    return os.path.normcase(_canonical_module_path(value))


def _normalized_extensions(extensions) -> set[str]:
    return {
        str(extension).casefold() if str(extension).startswith(".") else f".{str(extension).casefold()}"
        for extension in extensions
    }


def find_files_with_extensions(base_path: os.PathLike, extensions) -> list[str]:
    found_files = []
    allowed_extensions = _normalized_extensions(extensions)
    for root, directories, files in os.walk(os.path.abspath(base_path)):
        directories.sort(key=str.casefold)
        for file in sorted(files, key=str.casefold):
            if os.path.splitext(file)[1].casefold() in allowed_extensions:
                found_files.append(os.path.abspath(os.path.join(root, file)))
    return found_files


def _ordered_module_roots() -> list[tuple[str, str]]:
    candidates = [
        ("VAE", os.path.abspath(os.path.join(paths.models_path, "VAE"))),
        ("text_encoder", os.path.abspath(os.path.join(paths.models_path, "text_encoder"))),
    ]
    candidates.extend(
        ("VAE", os.fspath(path))
        for path in getattr(shared.cmd_opts, "vae_dirs", ()) or ()
    )
    candidates.extend(
        ("text_encoder", os.fspath(path))
        for path in getattr(shared.cmd_opts, "text_encoder_dirs", ()) or ()
    )

    roots = []
    seen = set()
    for label, path in candidates:
        absolute = os.path.abspath(path)
        normalized = _normalize_module_path(absolute)
        if normalized in seen:
            continue
        seen.add(normalized)
        roots.append((label, absolute))
    return roots


def _duplicate_module_selector(basename: str, root_label: str, identity: str, used: set[str]) -> str:
    stem, extension = os.path.splitext(basename)
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    for length in (8, 12, 16, len(digest)):
        selector = f"{stem} [{root_label}:{digest[:length]}]{extension}"
        if selector.casefold() not in used:
            return selector
    raise RuntimeError(f"Could not create a unique selector for additional module {basename!r}")


def _rebuild_module_registry(module_roots: list[tuple[str, str]], extensions=MODULE_FILE_EXTENSIONS) -> None:
    global _module_registry_initialized
    candidates: list[tuple[str, str, str, str]] = []
    seen_paths = set()
    for root_index, (root_label, root) in enumerate(module_roots):
        canonical_root = _canonical_module_path(root)
        for full_path in find_files_with_extensions(canonical_root, extensions):
            full_path = _canonical_module_path(full_path)
            normalized = _normalize_module_path(full_path)
            if normalized in seen_paths:
                continue
            seen_paths.add(normalized)
            relative = os.path.relpath(full_path, canonical_root).replace("\\", "/").casefold()
            identity = f"{root_index}:{root_label.casefold()}:{relative}"
            candidates.append((root_label, full_path, normalized, identity))

    candidates.sort(key=lambda candidate: (portable_module_basename(candidate[1]).casefold(), candidate[0].casefold(), candidate[2]))
    basename_counts = Counter(portable_module_basename(path).casefold() for _, path, _, _ in candidates)

    module_list.clear()
    _module_path_to_selector.clear()
    _module_selector_to_kind.clear()
    _module_selector_to_infotext.clear()
    _unresolved_module_choices.clear()
    basename_index: dict[str, list[str]] = defaultdict(list)
    used_selectors = set()

    for root_label, full_path, normalized, identity in candidates:
        basename = portable_module_basename(full_path)
        if basename_counts[basename.casefold()] == 1 and basename.casefold() not in used_selectors:
            selector = basename
        else:
            selector = _duplicate_module_selector(basename, root_label, identity, used_selectors)
        used_selectors.add(selector.casefold())
        module_list[selector] = full_path
        _module_path_to_selector[normalized] = selector
        _module_selector_to_kind[selector] = root_label
        basename_index[basename.casefold()].append(selector)

    _module_basename_to_selectors.clear()
    _module_basename_to_selectors.update(
        (name, tuple(dict.fromkeys(selectors))) for name, selectors in basename_index.items()
    )

    base_infotext_references = {
        selector: os.path.splitext(selector)[0] for selector in module_list
    }
    infotext_counts = Counter(
        reference.casefold() for reference in base_infotext_references.values()
    )
    infotext_index: dict[str, list[str]] = defaultdict(list)
    for selector, base_reference in base_infotext_references.items():
        reference = (
            base_reference
            if infotext_counts[base_reference.casefold()] == 1
            else selector
        )
        _module_selector_to_infotext[selector] = reference
        infotext_index[reference.casefold()].append(selector)
        basename = portable_module_basename(module_list[selector])
        infotext_index[os.path.splitext(basename)[0].casefold()].append(selector)

    _module_infotext_to_selectors.clear()
    _module_infotext_to_selectors.update(
        (name, tuple(dict.fromkeys(selectors))) for name, selectors in infotext_index.items()
    )
    _module_registry_initialized = True


def ensure_module_registry() -> None:
    """Build the module registry lazily without refreshing checkpoints or saving settings."""
    if not _module_registry_initialized:
        _rebuild_module_registry(_ordered_module_roots())


def resolve_module_value(value: object) -> str:
    text = str(value or "")
    if text in module_list:
        return os.fspath(module_list[text])
    if text.startswith(_UNRESOLVED_MODULE_PREFIX):
        raise ModuleResolutionError(text, "missing")

    looks_like_path = os.path.isabs(text) or "/" in text or "\\" in text
    if looks_like_path:
        selector = _module_path_to_selector.get(_normalize_module_path(text))
        if selector is not None:
            return os.fspath(module_list[selector])
        raise ModuleResolutionError(text, "missing")

    selectors = _module_basename_to_selectors.get(portable_module_basename(text).casefold(), ())
    if len(selectors) == 1:
        return os.fspath(module_list[selectors[0]])
    if len(selectors) > 1:
        raise ModuleResolutionError(text, "ambiguous", selectors)
    raise ModuleResolutionError(text, "missing")


def resolve_module_values(values) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, os.PathLike)):
        values = [values]
    resolved = [resolve_module_value(value) for value in values]
    return sorted(dict.fromkeys(resolved), key=_normalize_module_path)


def resolve_generation_module_values(values) -> list[str]:
    """Resolve only API-safe selectors or unique basenames for one generation request."""
    ensure_module_registry()
    for value in values:
        text = str(value or "")
        if "/" in text or "\\" in text:
            raise ModuleResolutionError(value, "path")
    return resolve_module_values(values)


def _register_unresolved_module_choice(value: object, reason: str) -> str:
    text = str(value or "")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    token = f"{_UNRESOLVED_MODULE_PREFIX}{digest}"
    basename = portable_module_basename(text) or "module"
    label = f"{basename} ({'ambiguous' if reason == 'ambiguous' else 'missing'}; reselect)"
    _unresolved_module_choices[token] = label
    return token


def module_values_to_ui_selectors(values) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, os.PathLike)):
        values = [values]

    selectors = []
    for value in values:
        try:
            resolved = resolve_module_value(value)
        except ModuleResolutionError as error:
            selectors.append(_register_unresolved_module_choice(value, error.reason))
        else:
            selectors.append(_module_path_to_selector[_normalize_module_path(resolved)])
    return selectors


def module_dropdown_choices() -> list[str | tuple[str, str]]:
    choices: list[str | tuple[str, str]] = sorted(module_list, key=str.casefold)
    choices.extend(
        (label, token)
        for token, label in sorted(_unresolved_module_choices.items(), key=lambda item: item[1].casefold())
    )
    return choices


def module_infotext_reference(value: object) -> str:
    try:
        path = resolve_module_value(value)
    except ModuleResolutionError:
        return os.path.splitext(portable_module_basename(value))[0]
    selector = _module_path_to_selector[_normalize_module_path(path)]
    return _module_selector_to_infotext[selector]


def module_kind(value: object) -> str | None:
    try:
        path = resolve_module_value(value)
    except ModuleResolutionError:
        return None
    selector = _module_path_to_selector[_normalize_module_path(path)]
    return _module_selector_to_kind.get(selector)


def module_selector_from_infotext(value: object) -> str | None:
    text = str(value or "")
    if text in module_list:
        return text
    selectors = _module_infotext_to_selectors.get(text.casefold(), ())
    return selectors[0] if len(selectors) == 1 else None


def _module_option_names() -> list[str]:
    return ["forge_additional_modules", *(f"forge_additional_modules_{preset}" for preset in PresetArch.choices())]


def _resolve_current_preset_legacy_modules(option_name: str, values: list) -> list[str] | None:
    current_preset_option = f"forge_additional_modules_{shared.opts.forge_preset}"
    if option_name != current_preset_option:
        return None

    try:
        active_modules = resolve_module_values(
            shared.opts.data.get("forge_additional_modules", [])
        )
    except ModuleResolutionError:
        return None
    if len(active_modules) != len(values):
        return None

    active_by_basename: dict[str, list[str]] = defaultdict(list)
    for path in active_modules:
        active_by_basename[portable_module_basename(path).casefold()].append(path)

    resolved = []
    for value in values:
        matches = active_by_basename.get(portable_module_basename(value).casefold(), [])
        if len(matches) != 1:
            return None
        resolved.append(matches.pop())

    if any(active_by_basename.values()):
        return None
    return sorted(resolved, key=_normalize_module_path)


def _migrate_legacy_module_options_once() -> None:
    global _legacy_module_migration_complete
    if _legacy_module_migration_complete:
        return
    _legacy_module_migration_complete = True
    if getattr(shared.cmd_opts, "freeze_settings", False):
        return

    changed: dict[str, list[str]] = {}
    originals: dict[str, list[str]] = {}
    for option_name in _module_option_names():
        values = shared.opts.data.get(option_name)
        if not isinstance(values, list) or not values:
            continue
        try:
            resolved = resolve_module_values(values)
        except ModuleResolutionError as error:
            resolved = _resolve_current_preset_legacy_modules(option_name, values)
            if resolved is None:
                logger.warning("Skipping additional-module migration for %s: %s", option_name, error)
                continue
            logger.info(
                "Migrating legacy additional modules for the active %s preset from its exact current selection",
                shared.opts.forge_preset,
            )
        if resolved != values:
            originals[option_name] = values
            changed[option_name] = resolved

    if not changed:
        return
    for option_name, values in changed.items():
        shared.opts.set(option_name, values, run_callbacks=False)
    try:
        shared.opts.save(shared.config_filename)
    except Exception:
        for option_name, values in originals.items():
            shared.opts.data[option_name] = values
        logger.exception("Could not save the additional-module identity migration; continuing with the original settings")


def _register_configured_module_choices() -> None:
    for option_name in _module_option_names():
        module_values_to_ui_selectors(shared.opts.data.get(option_name, []))


def make_checkpoint_manager_ui():
    global ui_forge_preset, ui_checkpoint, ui_vae, ui_forge_unet_dtype

    if shared.opts.sd_model_checkpoint in [None, "None", "none", ""]:
        if len(sd_models.checkpoints_list) == 0:
            sd_models.list_models()
        if len(sd_models.checkpoints_list) > 0:
            shared.opts.set("sd_model_checkpoint", next(iter(sd_models.checkpoints_list.values())).name)

    ckpt_list, vae_list = refresh_models()
    current_preset = shared.opts.forge_preset
    checkpoint_value = getattr(shared.opts, f"forge_checkpoint_{current_preset}", None) or shared.opts.sd_model_checkpoint
    module_value = module_values_to_ui_selectors(
        getattr(
            shared.opts,
            f"forge_additional_modules_{current_preset}",
            shared.opts.forge_additional_modules,
        )
    )

    ui_forge_preset = gr.Dropdown(label="UI Preset", value=shared.opts.forge_preset, choices=PresetArch.choices(), elem_id="forge_ui_preset")

    ui_checkpoint = gr.Dropdown(label="Checkpoint", value=checkpoint_value, choices=ckpt_list, elem_id="setting_sd_model_checkpoint", elem_classes=["model_selection"])

    ui_vae = gr.Dropdown(label="VAE / Text Encoder", value=module_value, choices=vae_list, multiselect=True, elem_id="setting_sd_modules", elem_classes=["model_selection"])

    def refresh_model_list():
        ckpt_list, vae_list = refresh_models()
        current_preset = shared.opts.forge_preset
        module_value = module_values_to_ui_selectors(
            getattr(
                shared.opts,
                f"forge_additional_modules_{current_preset}",
                shared.opts.forge_additional_modules,
            )
        )
        return [
            gr.update(choices=ckpt_list),
            gr.update(choices=vae_list, value=module_value),
        ]

    refresh_button = ui_common.ToolButton(value=ui_common.refresh_symbol, elem_id="forge_refresh_checkpoint", tooltip="Refresh")
    refresh_button.click(fn=refresh_model_list, outputs=[ui_checkpoint, ui_vae], queue=False)
    Context.root_block.load(fn=refresh_model_list, outputs=[ui_checkpoint, ui_vae], queue=False)

    ui_forge_unet_dtype = gr.Dropdown(label="Diffusion in Low Bits", value=None, choices=list(forge_unet_storage_dtype_options.keys()), elem_id="forge_ui_dtype")

    ui_checkpoint.input(checkpoint_change, inputs=[ui_checkpoint, ui_forge_preset], queue=False, show_progress=False)
    ui_vae.input(modules_change, inputs=[ui_vae, ui_forge_preset], queue=False, show_progress=False)
    ui_forge_unet_dtype.input(dtype_change, inputs=[ui_forge_unet_dtype, ui_forge_preset], queue=False, show_progress=False)


def refresh_models() -> tuple[list[os.PathLike], list[str | tuple[str, str]]]:
    shared_items.refresh_checkpoints()
    ckpt_list = shared_items.list_checkpoint_tiles(shared.opts.sd_checkpoint_dropdown_use_short)
    _rebuild_module_registry(_ordered_module_roots())
    _migrate_legacy_module_options_once()
    _register_configured_module_choices()
    return sorted(ckpt_list), module_dropdown_choices()


def refresh_model_loading_parameters(*, refresh: bool = True):
    if not refresh:
        return

    from modules.sd_models import model_data, select_checkpoint

    checkpoint_info = select_checkpoint()
    if checkpoint_info is None:
        logger.critical('You do not have any model... Please download models to "models/Stable-diffusion"')
        return

    unet_storage_dtype, lora_fp16 = forge_unet_storage_dtype_options.get(shared.opts.forge_unet_storage_dtype, (None, False))

    model_data.forge_loading_parameters = dict(checkpoint_info=checkpoint_info, additional_modules=shared.opts.forge_additional_modules, unet_storage_dtype=unet_storage_dtype)

    ckpt: str = checkpoint_info.filename
    modules: list[str] = [os.path.basename(x) for x in shared.opts.forge_additional_modules]
    dtype = str(unet_storage_dtype or [torch.float16, torch.bfloat16])

    logger.info("Model Selected:")
    print_json(data=dict(checkpoint=os.path.basename(ckpt), modules=modules, dtype=dtype))

    if ckpt.endswith(("gguf", "GGUF")) and not lora_fp16:
        logger.warning("GGUF requires fp16 LoRA ; overriding option")
        lora_fp16 = True

    dynamic_args.online_lora = lora_fp16
    logger.info(f"Patch LoRAs on-the-fly: {lora_fp16}")
    if not ckpt.endswith(("gguf", "GGUF")) and lora_fp16:
        logger.warning("on-the-fly WILL be slower ; enable only if you know what you are doing")

    processing.need_global_unload = True


def checkpoint_change(ckpt_name: str, preset: str, save=True, refresh=True) -> bool:
    """`ckpt_name` accepts valid aliases; returns `True` if checkpoint changed"""
    new_ckpt_info = sd_models.get_closet_checkpoint_match(ckpt_name)
    current_ckpt_info = sd_models.get_closet_checkpoint_match(getattr(shared.opts, "sd_model_checkpoint", ""))
    if new_ckpt_info == current_ckpt_info:
        return False

    shared.opts.set("sd_model_checkpoint", ckpt_name)
    if preset is not None:
        shared.opts.set(f"forge_checkpoint_{preset}", ckpt_name)

    if save:
        shared.opts.save(shared.config_filename)
    refresh_model_loading_parameters(refresh=refresh)
    return True


def modules_change(module_values: list, preset: str, save=True, refresh=True) -> bool:
    """Resolve every selector before atomically updating the active and preset module paths."""
    modules = resolve_module_values(module_values)
    current_modules = getattr(shared.opts, "forge_additional_modules", [])
    active_changed = modules != current_modules
    preset_name = f"forge_additional_modules_{preset}" if preset is not None else None
    preset_changed = preset_name is not None and modules != getattr(shared.opts, preset_name, [])
    if not active_changed and not preset_changed:
        return False

    if active_changed:
        shared.opts.set("forge_additional_modules", modules)
    if preset_changed:
        shared.opts.set(preset_name, modules)

    if save:
        shared.opts.save(shared.config_filename)
    if active_changed:
        refresh_model_loading_parameters(refresh=refresh)
    return True


def dtype_change(dtype: str, preset: str, save=True, refresh=True) -> bool:
    shared.opts.set("forge_unet_storage_dtype", dtype)
    if preset is not None:
        shared.opts.set(f"forge_unet_storage_dtype_{preset}", dtype)

    if save:
        shared.opts.save(shared.config_filename)
    refresh_model_loading_parameters(refresh=refresh)
    return True


def get_a1111_ui_component(tab: str, label: str) -> gr.components.Component:
    fields = infotext_utils.paste_fields[tab]["fields"]
    for f in fields:
        if f.label == label or f.api == label:
            return f.component


def forge_main_entry():
    ui_txt2img_steps = get_a1111_ui_component("txt2img", "Steps")
    ui_txt2img_hr_steps = get_a1111_ui_component("txt2img", "Hires steps")
    ui_img2img_steps = get_a1111_ui_component("img2img", "Steps")

    ui_txt2img_sampler = get_a1111_ui_component("txt2img", "sampler_name")
    ui_img2img_sampler = get_a1111_ui_component("img2img", "sampler_name")
    ui_txt2img_scheduler = get_a1111_ui_component("txt2img", "scheduler")
    ui_img2img_scheduler = get_a1111_ui_component("img2img", "scheduler")

    ui_txt2img_width = get_a1111_ui_component("txt2img", "Size-1")
    ui_img2img_width = get_a1111_ui_component("img2img", "Size-1")
    ui_txt2img_height = get_a1111_ui_component("txt2img", "Size-2")
    ui_img2img_height = get_a1111_ui_component("img2img", "Size-2")

    ui_txt2img_cfg = get_a1111_ui_component("txt2img", "CFG scale")
    ui_txt2img_hr_cfg = get_a1111_ui_component("txt2img", "Hires CFG Scale")
    ui_img2img_cfg = get_a1111_ui_component("img2img", "CFG scale")

    ui_txt2img_distilled_cfg = get_a1111_ui_component("txt2img", "Distilled CFG Scale")
    ui_txt2img_hr_distilled_cfg = get_a1111_ui_component("txt2img", "Hires Distilled CFG Scale")
    ui_img2img_distilled_cfg = get_a1111_ui_component("img2img", "Distilled CFG Scale")

    ui_txt2img_hr_denoise = get_a1111_ui_component("txt2img", "Denoising strength")
    ui_img2img_denoise = get_a1111_ui_component("img2img", "Denoising strength")

    ui_txt2img_batch_size = get_a1111_ui_component("txt2img", "Batch size")
    ui_img2img_batch_size = get_a1111_ui_component("img2img", "Batch size")

    output_targets = [
        ui_checkpoint,
        ui_vae,
        ui_forge_unet_dtype,
        ui_txt2img_steps,
        ui_txt2img_hr_steps,
        ui_img2img_steps,
        ui_txt2img_sampler,
        ui_img2img_sampler,
        ui_txt2img_scheduler,
        ui_img2img_scheduler,
        ui_txt2img_width,
        ui_img2img_width,
        ui_txt2img_height,
        ui_img2img_height,
        ui_txt2img_cfg,
        ui_txt2img_hr_cfg,
        ui_img2img_cfg,
        ui_txt2img_distilled_cfg,
        ui_txt2img_hr_distilled_cfg,
        ui_img2img_distilled_cfg,
        ui_txt2img_hr_denoise,
        ui_img2img_denoise,
        ui_txt2img_batch_size,
        ui_img2img_batch_size,
    ]

    preset_change = ui_forge_preset.change(
        on_preset_change,
        inputs=[ui_forge_preset],
        outputs=output_targets,
        queue=False,
        show_progress=False,
    )
    preset_load = preset_change.success(
        fn=_load_presets,
        inputs=[ui_checkpoint, ui_vae, ui_forge_unet_dtype, ui_forge_preset],
        queue=False,
        show_progress=False,
    )
    preset_load.success(js="clickLoraRefresh", fn=None, queue=False, show_progress=False)
    preset_load.failure(
        fn=recover_preset_ui,
        outputs=[ui_forge_preset, *output_targets],
        queue=False,
        show_progress=False,
    )
    Context.root_block.load(on_preset_change, inputs=[ui_forge_preset], outputs=output_targets, queue=False, show_progress=False)

    refresh_model_loading_parameters()


def _load_presets(ui_checkpoint: str, ui_vae: list[str], ui_forge_unet_dtype: str, ui_forge_preset: str):
    resolved_modules = resolve_module_values(ui_vae)
    if sd_models.get_closet_checkpoint_match(ui_checkpoint) is None:
        raise ValueError("The selected checkpoint is not available; refresh the model list and reselect it")

    shared.opts.set("forge_preset", ui_forge_preset)
    dtype_change(ui_forge_unet_dtype, ui_forge_preset, save=False, refresh=False)
    modules_change(resolved_modules, ui_forge_preset, save=False, refresh=False)
    checkpoint_change(ui_checkpoint, ui_forge_preset, save=False, refresh=False)
    shared.opts.save(shared.config_filename)
    refresh_model_loading_parameters()


def recover_preset_ui():
    """Restore the last committed preset after a failed preset load."""
    preset = shared.opts.forge_preset
    return [gr.update(value=preset), *on_preset_change(preset)]


def on_preset_change(preset: str):
    assert preset is not None

    if use_shift(preset):
        d_args = {"visible": getattr(shared.opts, f"{preset}_show_shift", True), "label": "Shift"}
    elif use_distill(preset):
        d_args = {"visible": True, "label": "Distilled CFG Scale"}
    else:
        d_args = {"visible": False}

    if (fps := is_video(preset)) > 1:
        batch_args_t2i = {"minimum": 1, "maximum": fps * 15 + 1, "step": fps, "label": "Frames", "value": getattr(shared.opts, f"{preset}_t2i_batch_size", 1)}
    else:
        batch_args_t2i = {"minimum": 1, "maximum": 8, "step": 1, "label": "Batch Size", "value": getattr(shared.opts, f"{preset}_t2i_batch_size", 1)}

    batch_args_i2i = batch_args_t2i.copy()
    batch_args_i2i["value"] = getattr(shared.opts, f"{preset}_i2i_batch_size", 1)

    return [
        # ui_checkpoint, ui_vae, ui_forge_unet_dtype
        gr.update(value=getattr(shared.opts, f"forge_checkpoint_{preset}", None) or shared.opts.sd_model_checkpoint),
        gr.update(
            value=module_values_to_ui_selectors(
                getattr(shared.opts, f"forge_additional_modules_{preset}", [])
            )
        ),
        gr.update(value=getattr(shared.opts, f"forge_unet_storage_dtype_{preset}", "Automatic")),
        # ui_txt2img_steps, ui_txt2img_hr_steps, ui_img2img_steps
        gr.update(value=v) if (v := getattr(shared.opts, f"{preset}_t2i_step", 20)) > 0 else gr.skip(),
        gr.update(value=v) if (v := getattr(shared.opts, f"{preset}_t2i_hr_step", 20)) > 0 else gr.skip(),
        gr.update(value=v) if (v := getattr(shared.opts, f"{preset}_i2i_step", 20)) > 0 else gr.skip(),
        # ui_txt2img_sampler, ui_img2img_sampler, ui_txt2img_scheduler, ui_img2img_scheduler
        gr.update(value=getattr(shared.opts, f"{preset}_t2i_sampler", "Euler")),
        gr.update(value=getattr(shared.opts, f"{preset}_i2i_sampler", "Euler")),
        gr.update(value=getattr(shared.opts, f"{preset}_t2i_scheduler", "Simple")),
        gr.update(value=getattr(shared.opts, f"{preset}_i2i_scheduler", "Simple")),
        # ui_txt2img_width, ui_img2img_width, ui_txt2img_height, ui_img2img_height
        gr.update(value=v) if (v := getattr(shared.opts, f"{preset}_t2i_width", 1024)) > 0 else gr.skip(),
        gr.update(value=v) if (v := getattr(shared.opts, f"{preset}_i2i_width", 1024)) > 0 else gr.skip(),
        gr.update(value=v) if (v := getattr(shared.opts, f"{preset}_t2i_height", 1024)) > 0 else gr.skip(),
        gr.update(value=v) if (v := getattr(shared.opts, f"{preset}_i2i_height", 1024)) > 0 else gr.skip(),
        # ui_txt2img_cfg, ui_txt2img_hr_cfg, ui_img2img_cfg
        gr.update(value=v) if (v := getattr(shared.opts, f"{preset}_t2i_cfg", 1.0)) > 0 else gr.skip(),
        gr.update(value=v) if (v := getattr(shared.opts, f"{preset}_t2i_hr_cfg", 1.0)) > 0 else gr.skip(),
        gr.update(value=v) if (v := getattr(shared.opts, f"{preset}_i2i_cfg", 1.0)) > 0 else gr.skip(),
        # ui_txt2img_distilled_cfg, ui_img2img_distilled_cfg, ui_txt2img_hr_distilled_cfg
        gr.update(value=getattr(shared.opts, f"{preset}_t2i_dcfg", 3.0), **d_args),
        gr.update(value=getattr(shared.opts, f"{preset}_t2i_hr_dcfg", 3.0), **d_args),
        gr.update(value=getattr(shared.opts, f"{preset}_i2i_dcfg", 3.0), **d_args),
        # ui_txt2img_hr_denoise, ui_img2img_denoise
        gr.update(
            value=getattr(shared.opts, f"{preset}_t2i_hr_denoise", 0.60),
            step=0.01 if preset == PresetArch.krea.name else 0.05,
        ),
        gr.update(value=getattr(shared.opts, f"{preset}_i2i_denoise", 0.60)),
        # ui_txt2img_batch_size, ui_img2img_batch_size
        gr.update(**batch_args_t2i),
        gr.update(**batch_args_i2i),
    ]
