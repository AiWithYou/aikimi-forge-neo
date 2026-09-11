"""Forge-specific internal img2img adapter and processing-state guard."""

from __future__ import annotations

from contextlib import contextmanager

from PIL import Image
import torch

from modules import devices, processing, shared

from .generator import GenerationRequest


class ProcessingSnapshot:
    """Restore processing attributes and top-level mutable container contents."""

    def __init__(self, request: object):
        self.values = dict(vars(request))
        self.mutable_contents: dict[str, tuple[str, object]] = {}
        for name, value in self.values.items():
            if isinstance(value, list):
                self.mutable_contents[name] = ("list", list(value))
            elif isinstance(value, dict):
                self.mutable_contents[name] = ("dict", dict(value))
            elif isinstance(value, set):
                self.mutable_contents[name] = ("set", set(value))

    def restore(self, request: object) -> None:
        for name, (kind, saved) in self.mutable_contents.items():
            original = self.values[name]
            if kind == "list":
                original[:] = saved
            elif kind == "dict":
                original.clear()
                original.update(saved)
            else:
                original.clear()
                original.update(saved)
        current = vars(request)
        current.clear()
        current.update(self.values)


@contextmanager
def exact_img2img_steps(request: object):
    had_overrides = hasattr(request, "override_settings")
    original_overrides = getattr(request, "override_settings", None)
    had_restore = hasattr(request, "override_settings_restore_afterwards")
    original_restore = getattr(
        request, "override_settings_restore_afterwards", None
    )
    overrides = dict(original_overrides or {})
    overrides["img2img_fix_steps"] = True
    request.override_settings = overrides
    request.override_settings_restore_afterwards = True
    try:
        yield
    finally:
        if had_overrides:
            request.override_settings = original_overrides
        elif hasattr(request, "override_settings"):
            delattr(request, "override_settings")
        if had_restore:
            request.override_settings_restore_afterwards = original_restore
        elif hasattr(request, "override_settings_restore_afterwards"):
            delattr(request, "override_settings_restore_afterwards")


def _first_text(value: str | list[str] | tuple[str, ...] | None) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    return str(value or "")


def _append_suffix(text: str, suffix: str) -> str:
    if not suffix:
        return text
    return f"{text.rstrip(' ,')}, {suffix.strip(' ,')}" if text else suffix


class ForgeGeneratorAdapter:
    """Run one model tile through Forge without loading a separate model or VAE."""

    latent_scale = int(getattr(processing, "opt_f", 8))

    def __init__(self, request: processing.StableDiffusionProcessingImg2Img):
        if not hasattr(shared.sd_model, "forge_objects"):
            # A selectable Script runs before the first normal process_images()
            # call, so Forge may still expose FakeInitialModel here. Use Forge's
            # own lazy-load entrypoint; this does not create a second model/VAE.
            processing.manage_model_and_prompt_cache(request)
        if not hasattr(shared.sd_model, "forge_objects"):
            raise RuntimeError(
                "Forge did not load the selected checkpoint before HyperWeave."
            )
        self.request = request
        self.base_prompt = _first_text(request.prompt)
        self.base_negative_prompt = _first_text(request.negative_prompt)
        self.base_subseed = int(_first_text(getattr(request, "subseed", 0)) or 0)
        self.latent_scale = int(getattr(processing, "opt_f", 8))
        self.latent_channels = int(
            shared.sd_model.forge_objects.vae.latent_channels
        )
        self.last_processed: processing.Processed | None = None
        self._calls = 0
        self._memory_device = devices.device
        self._memory_backend = torch.cuda if torch.cuda.is_available() else None
        if self._memory_backend is not None:
            try:
                # Resetting a peak keeps the current model allocation as the
                # baseline, then captures additional allocations made by the job.
                self._memory_backend.reset_peak_memory_stats(self._memory_device)
            except (RuntimeError, ValueError):
                self._memory_backend = None

    def generate(
        self, image: Image.Image, request: GenerationRequest
    ) -> Image.Image:
        p = self.request
        p.batch_size = 1
        p.n_iter = 1
        p.do_not_save_samples = True
        p.do_not_save_grid = True
        p.resize_mode = 0
        p.width = image.width
        p.height = image.height
        p.steps = int(request.steps)
        p.denoising_strength = float(request.strength)
        p.seed = int(request.seed)
        p.subseed = self.base_subseed
        p.init_images = [image.convert("RGB")]
        p.prompt = _append_suffix(self.base_prompt, request.prompt_suffix)
        p.negative_prompt = _append_suffix(
            self.base_negative_prompt, request.negative_suffix
        )
        p.restore_faces = False
        p.tiling = False
        p._hyperweave_noise_override = request.coordinate_noise
        p._hyperweave_noise_applied = False
        p._hyperweave_noise_error = None
        p._hyperweave_generation_request = request
        try:
            with exact_img2img_steps(p):
                result = processing.process_images(p)
            if (
                shared.state.interrupted
                or shared.state.skipped
                or shared.state.stopping_generation
            ):
                raise RuntimeError(
                    "HyperWeave generation was interrupted before the tile completed."
                )
            if getattr(p, "_hyperweave_noise_error", None):
                raise RuntimeError(str(p._hyperweave_noise_error))
            if not getattr(p, "_hyperweave_noise_applied", False):
                raise RuntimeError(
                    "HyperWeave coordinate-noise bridge did not run; refusing "
                    "tile-local noise fallback."
                )
            if result is None or not result.images:
                raise RuntimeError("Forge returned no image for a HyperWeave tile.")
            self.last_processed = result
            self._calls += 1
            return result.images[0].convert("RGB")
        finally:
            for name in (
                "_hyperweave_noise_override",
                "_hyperweave_noise_applied",
                "_hyperweave_noise_error",
                "_hyperweave_generation_request",
            ):
                if hasattr(p, name):
                    delattr(p, name)
            if hasattr(p, "latents_after_sampling"):
                p.latents_after_sampling.clear()
            if hasattr(p, "pixels_after_sampling"):
                p.pixels_after_sampling.clear()

    def model_metadata(self) -> dict[str, object]:
        p = self.request
        model = shared.sd_model
        return {
            "adapter": "ForgeGeneratorAdapter",
            "model": getattr(p, "sd_model_name", None)
            or getattr(getattr(model, "sd_checkpoint_info", None), "name", None),
            "model_hash": getattr(p, "sd_model_hash", None)
            or getattr(model, "sd_model_hash", None),
            "vae": getattr(p, "sd_vae_name", None),
            "sampler": getattr(p, "sampler_name", None),
            "scheduler": getattr(p, "scheduler", None),
            "cfg": getattr(p, "cfg_scale", None),
            "distilled_cfg": getattr(p, "distilled_cfg_scale", None),
            "latent_channels": self.latent_channels,
            "latent_scale": self.latent_scale,
            "internal_calls": self._calls,
            "always_visible_scripts": "Forge callback path preserved",
            "coordinate_noise_scope": "initial latent noise",
        }

    def runtime_metrics(self) -> dict[str, object]:
        backend = self._memory_backend
        if backend is None:
            return {"peak_vram_available": False}
        try:
            free_bytes, total_bytes = backend.mem_get_info(self._memory_device)
            return {
                "peak_vram_available": True,
                "peak_allocated_bytes": int(
                    backend.max_memory_allocated(self._memory_device)
                ),
                "peak_reserved_bytes": int(
                    backend.max_memory_reserved(self._memory_device)
                ),
                "current_allocated_bytes": int(
                    backend.memory_allocated(self._memory_device)
                ),
                "current_reserved_bytes": int(
                    backend.memory_reserved(self._memory_device)
                ),
                "device_free_bytes": int(free_bytes),
                "device_total_bytes": int(total_bytes),
            }
        except (RuntimeError, ValueError):
            return {"peak_vram_available": False}

    def pass_cleanup(self) -> None:
        devices.torch_gc()
