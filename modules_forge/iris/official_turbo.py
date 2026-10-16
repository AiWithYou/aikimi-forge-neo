"""Official deterministic few-step inference and lightweight checkpoint profiles.

Sampler source: speridlabs/iris-3b-turbo Space, revision
d1b6728b7e1d6fcbf4bf0d5cf2b6d755619c95c8, iris3b/sampling.py (Apache-2.0).
Adapted for Neo; see LICENSE.turbo for the license text.
"""

from __future__ import annotations

import math
from pathlib import Path

import yaml

SIZES = {
    "1024×1024": (1024, 1024),
    "1344×752": (1344, 752),
    "1344×896": (1344, 896),
    "1200×896": (1200, 896),
    "896×1200": (896, 1200),
    "896×1344": (896, 1344),
    "752×1344": (752, 1344),
}


def validate_settings(settings):
    sigmas = settings.get("sigmas") if isinstance(settings, dict) else None
    if (
        not isinstance(settings, dict)
        or settings.get("mode") != "official_turbo"
        or settings.get("noise_device") != "cpu"
        or not isinstance(sigmas, list)
        or not 1 <= len(sigmas) <= 200
        or any(type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1 for value in sigmas)
        or sigmas[0] != 1
        or any(a <= b for a, b in zip(sigmas, sigmas[1:], strict=False))
    ):
        raise ValueError("公式Turboのsigmasは1から降順の有限値・CPU初期ノイズを指定してください。")
    return {"mode": "official_turbo", "sigmas": [float(value) for value in sigmas], "noise_device": "cpu"}


def read_settings(path):
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        dmd = raw["dmd"]
        sigmas = dmd["sigmas"]
        if dmd.get("deterministic") is not True:
            raise ValueError("distill.yamlのdeterministic: trueが必要です。")
    except (OSError, UnicodeError, yaml.YAMLError, TypeError, KeyError, AttributeError) as exc:
        raise ValueError("distill.yamlの蒸留スケジュールを読み込めません。") from exc
    return validate_settings({"mode": "official_turbo", "sigmas": sigmas, "noise_device": "cpu"})


def checkpoint_settings(root, checkpoint):
    if checkpoint == "base":
        return None
    from .local_source import resolve

    return resolve(root, checkpoint).get("sampling")


def combine_settings(checkpoint, adapter):
    if checkpoint is None:
        return adapter
    if adapter["mode"] != "standard":
        raise ValueError("公式Turboと別のTurbo生成方式は併用できません。サンプリングを通常にしてください。")
    return checkpoint


def generation_sizes(root, checkpoint, *, sampling=None):
    from .core import GENERATION_SIZES

    sampling = sampling if sampling is not None else checkpoint_settings(root, checkpoint)
    return SIZES if sampling and sampling["mode"] == "official_turbo" else GENERATION_SIZES


def generate(
    model,
    text_encoder,
    prompts,
    height,
    width,
    sigmas,
    generator=None,
    device="cuda",
    *,
    num_train_timesteps=1000,
    prediction="v",
):
    import torch

    with torch.no_grad():
        dtype = next(model.parameters()).dtype
        encoding = text_encoder.encode(prompts)
        cond = encoding.embeddings.to(device=device, dtype=dtype)
        mask = encoding.mask.to(device=device)
        # Match the official Space: CPU noise is reproducible across GPU models.
        x = torch.randn(len(prompts), model.cfg.in_channels, height, width, generator=generator, device="cpu").to(
            device
        )
        sigma = torch.full((x.shape[0], 1, 1, 1), sigmas[0], device=device, dtype=torch.float32)
        for index in range(len(sigmas)):
            output = model(x.to(dtype), sigma.flatten() * num_train_timesteps, cond, y_mask=mask).x.float()
            if not torch.isfinite(output).all():
                raise ValueError("公式Turboのモデル出力に非有限値が含まれています。")
            clean = output if prediction == "x" else x - sigma * output
            if index + 1 < len(sigmas):
                implied_noise = (x - (1 - sigma) * clean) / sigma
                sigma = torch.full_like(sigma, sigmas[index + 1])
                x = (1 - sigma) * clean + sigma * implied_noise
        if not torch.isfinite(clean).all():
            raise ValueError("公式Turbo生成結果に非有限値が含まれています。")
        return clean.clamp(-1, 1)
