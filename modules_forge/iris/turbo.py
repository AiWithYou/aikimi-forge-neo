"""Iris Turbo inference, following pentaCoxian's shifted re-noising sampler.

Source: pentacoxian-dev/iris-3b-turbo Space, commit
5243039779e896b9dc884433e76b33cbe9f4f5e7, iris3b/distill.py (Apache-2.0).
Adapted for Neo with optional manual CFG and negative prompting. See LICENSE.turbo.
"""

import torch


@torch.no_grad()
def generate_turbo(
    model,
    text_encoder,
    prompts,
    height,
    width,
    steps,
    shift,
    generator=None,
    device="cuda",
    *,
    cfg_scale=1.0,
    negative_prompt="",
):
    dtype = next(model.parameters()).dtype
    encoding = text_encoder.encode(prompts)
    cond = encoding.embeddings.to(device=device, dtype=dtype)
    mask = encoding.mask.to(device=device)
    if cfg_scale != 1:
        negative = text_encoder.null(negative_prompt)
        uncond = negative.embeddings.to(device=device, dtype=dtype)
        uncond_mask = negative.mask.to(device=device)
    grid = 1.0 - torch.arange(steps, dtype=torch.float64) / steps
    times = (shift * grid / (1 + (shift - 1) * grid)).tolist()
    x = torch.randn(len(prompts), model.cfg.in_channels, height, width, generator=generator, device=device)
    for index, time in enumerate(times):
        t = torch.full((x.shape[0],), time, device=device)
        output = model(x.to(dtype), t * 1000.0, cond, y_mask=mask).x.float()
        if cfg_scale != 1:
            negative_output = model(x.to(dtype), t * 1000.0, uncond, y_mask=uncond_mask).x.float()
            output = negative_output + cfg_scale * (output - negative_output)
        x0 = x - time * output
        if index + 1 == len(times):
            if not torch.isfinite(x0).all():
                raise ValueError("Turbo生成結果に非有限値が含まれています。")
            return x0.clamp(-1, 1)
        fresh = torch.randn(x.shape, generator=generator, device=device)
        x = (1 - times[index + 1]) * x0 + times[index + 1] * fresh
    raise ValueError("Turboのステップ数が不正です。")
