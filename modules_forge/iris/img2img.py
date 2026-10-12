"""SDEdit with Iris's original shifted grid and DPM++ integration."""

import math
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


@dataclass
class ImageSchedule:
    requested_steps: int
    strength: float
    shift: float
    times: list[float]

    @property
    def steps(self):
        return len(self.times) - 1

    def metadata(self):
        return {
            "requested_steps": self.requested_steps,
            "executed_steps": self.steps,
            "strength": self.strength,
            "shift": self.shift,
            "start_sigma": self.times[0],
            "initialization": "noise" if self.strength == 1 else "image" if self.strength == 0 else "image_and_noise",
        }


def image_schedule(steps, strength, shift):
    if strength == 0:
        return ImageSchedule(steps, strength, shift, [0.0])
    from iris3b.flow.solver import FlowDPMSolver

    count = min(steps, max(1, math.floor(steps * strength + 1e-9)))
    grid = FlowDPMSolver.time_grid(steps, shift)
    if (
        len(grid) != steps + 1
        or grid[-1] != 0
        or not 0 < grid[0] < 1
        or not all(math.isfinite(value) for value in grid)
        or not all(a > b for a, b in zip(grid, grid[1:], strict=False))
    ):
        raise ValueError("img2imgのflow時間刻みが不正です。")
    times = grid[steps - count :]
    return ImageSchedule(steps, strength, shift, times)


def resized_input(path, width, height):
    with Image.open(path) as source:
        return source.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)


def image_pixels(image):
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1).unsqueeze(0)
    return pixels / 127.5 - 1


@torch.no_grad()
def generate_img2img(
    model,
    text_encoder,
    prompts,
    pixels,
    schedule,
    *,
    cfg_scale=3.0,
    cfg_interval=(0.0, 1.0),
    negative_prompt="",
    generator=None,
    device="cuda",
    num_train_timesteps=1000,
    prediction="v",
):
    if schedule.steps == 0:
        return pixels
    if schedule.strength == 1:
        from iris3b.sampling import generate

        return generate(
            model,
            text_encoder,
            prompts,
            height=pixels.shape[-2],
            width=pixels.shape[-1],
            steps=schedule.requested_steps,
            order=2,
            cfg_scale=cfg_scale,
            cfg_interval=cfg_interval,
            shift=schedule.shift,
            negative_prompt=negative_prompt,
            generator=generator,
            device=device,
            num_train_timesteps=num_train_timesteps,
            prediction=prediction,
        )
    from iris3b.flow.solver import FlowDPMSolver

    class ImageSolver(FlowDPMSolver):
        @staticmethod
        def time_grid(steps, shift):
            return schedule.times

    dtype = next(model.parameters()).dtype
    encoding = text_encoder.encode(prompts)
    cond = encoding.embeddings.to(device=device, dtype=dtype)
    cond_mask = encoding.mask.to(device=device)
    uncond, cfg_mask = None, None
    if cfg_scale != 1:
        negative = text_encoder.null(negative_prompt)
        uncond = negative.embeddings.to(device=device, dtype=dtype).expand(
            len(prompts), *([-1] * (negative.embeddings.ndim - 1))
        )
        null_mask = negative.mask.to(device=device).expand(len(prompts), -1)
        cfg_mask = torch.cat([null_mask, cond_mask], dim=0)

    def model_fn(x, t, y):
        mask = cond_mask if y is cond else cfg_mask
        return model(x.to(dtype), t, y, y_mask=mask).x.float()

    noise = torch.randn(pixels.shape, generator=generator, device=device)
    sigma = schedule.times[0]
    initial = (1 - sigma) * pixels.to(device=device, dtype=torch.float32) + sigma * noise
    solver = ImageSolver(
        model_fn,
        num_timesteps=num_train_timesteps,
        cfg_scale=cfg_scale,
        cfg_interval=cfg_interval,
        prediction=prediction,
    )
    output = solver.sample(initial, cond, uncond, steps=schedule.steps, order=2, shift=schedule.shift)
    if not torch.isfinite(output).all():
        raise ValueError("img2imgの生成結果に非有限値が含まれています。")
    return output.clamp(-1, 1)
