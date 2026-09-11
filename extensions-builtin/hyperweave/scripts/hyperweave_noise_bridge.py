"""AlwaysVisible coordinate-noise bridge for HyperWeave."""

from __future__ import annotations

import torch

import modules.scripts as scripts
from hyperweave.forge_noise import coordinate_noise_tensor


class HyperWeaveNoiseBridge(scripts.Script):
    """Normally inert bridge that installs image-coordinate initial noise."""

    create_group = False
    sorting_priority = 10000

    def title(self):
        return "HyperWeave Coordinate Noise Bridge"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return []

    def process_before_every_sampling(self, p, *args, **kwargs):
        override = getattr(p, "_hyperweave_noise_override", None)
        if override is None:
            return
        original = kwargs.get("noise")
        if original is None:
            p._hyperweave_noise_error = (
                "Forge sampling callback did not expose the initial noise tensor."
            )
            return
        try:
            multiplier = float(
                getattr(p, "initial_noise_multiplier", 1.0) or 1.0
            )
            prior_modified = getattr(p, "modified_noise", None)
            p.modified_noise = coordinate_noise_tensor(
                override,
                original,
                initial_noise_multiplier=multiplier,
                prior_modified=(
                    prior_modified
                    if isinstance(prior_modified, torch.Tensor)
                    else None
                ),
            )
            p._hyperweave_noise_applied = True
        except Exception as exc:
            p._hyperweave_noise_error = (
                f"HyperWeave coordinate-noise injection failed: {exc}"
            )
