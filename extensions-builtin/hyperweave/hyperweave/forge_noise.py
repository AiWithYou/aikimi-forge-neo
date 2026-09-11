"""Tensor adaptation for Forge's 4D and singleton-temporal image latents."""

from __future__ import annotations

import numpy as np
import torch


def coordinate_noise_tensor(
    override: np.ndarray,
    original: torch.Tensor,
    *,
    initial_noise_multiplier: float = 1.0,
    prior_modified: torch.Tensor | None = None,
) -> torch.Tensor:
    """Adapt C×H×W coordinate noise to the current Forge image latent."""
    noise = torch.from_numpy(
        np.ascontiguousarray(override, dtype=np.float32)
    ).unsqueeze(0)
    if original.ndim == 5:
        if original.shape[2] != 1:
            raise ValueError(
                "HyperWeave img2img supports only a singleton temporal latent "
                f"axis; Forge supplied shape {tuple(original.shape)}."
            )
        noise = noise.unsqueeze(2)
    if tuple(noise.shape) != tuple(original.shape):
        raise ValueError(
            f"Coordinate noise shape {tuple(noise.shape)} does not match "
            f"Forge noise shape {tuple(original.shape)}."
        )

    result = noise.to(original) * float(initial_noise_multiplier)
    if prior_modified is not None:
        if tuple(prior_modified.shape) != tuple(original.shape):
            raise ValueError(
                "An earlier AlwaysVisible script supplied modified noise with "
                f"shape {tuple(prior_modified.shape)}; expected "
                f"{tuple(original.shape)}."
            )
        # Replace only the random base. Preserve formal callback-path additions,
        # such as the latent offset from an inpaint ControlNet preprocessor.
        result = result + (prior_modified.to(original) - original)
    return result
