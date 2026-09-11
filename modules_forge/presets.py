from enum import Enum

from modules_forge.krea2_upscale import (
    KREA2_DEFAULT_CFG,
    KREA2_DEFAULT_DENOISE,
    KREA2_DEFAULT_SAMPLER,
    KREA2_DEFAULT_SCHEDULER,
    KREA2_DEFAULT_SHIFT,
    KREA2_I2I_STEPS,
    KREA2_T2I_STEPS,
)


class PresetArch(Enum):
    sd = 1  # SD1
    xl = 2  # SDXL
    flux = 3  # Flux.1
    klein = 4  # Flux.2
    qwen = 5  # Qwen-Image
    lumina = 6  # Lumina-Image-2.0
    zit = 7  # Z-Image-Turbo
    wan = 8  # Wan2.2
    anima = 9  # Anima
    ernie = 10  # Ernie-Image
    pid = 11  # PiD
    krea = 12  # Krea2

    @staticmethod
    def choices() -> list[str]:
        return [preset.name for preset in PresetArch]


DEFAULT_CHECKPOINTS = {
    PresetArch.krea: "krea2_turbo_int8_convrot.safetensors",
}

DEFAULT_ADDITIONAL_MODULES = {
    PresetArch.krea: (
        "qwen_image_vae.safetensors",
        "qwen3vl_4b_fp8_scaled.safetensors",
    ),
}

DEFAULT_UNET_STORAGE_DTYPES = {
    PresetArch.krea: "Automatic",
}


SAMPLERS = {
    PresetArch.sd: "Euler a",
    PresetArch.xl: "Euler a",
    PresetArch.flux: "Euler",
    PresetArch.klein: "Euler",
    PresetArch.qwen: "LCM",
    PresetArch.lumina: "Res Multistep",
    PresetArch.zit: "Euler",
    PresetArch.wan: "Euler",
    PresetArch.anima: "ER SDE",
    PresetArch.ernie: "Euler",
    PresetArch.pid: "LCM",
    PresetArch.krea: KREA2_DEFAULT_SAMPLER,
}

SCHEDULERS = {
    PresetArch.sd: "Automatic",
    PresetArch.xl: "Automatic",
    PresetArch.flux: "Beta",
    PresetArch.klein: "Beta",
    PresetArch.qwen: "Normal",
    PresetArch.lumina: "Simple",
    PresetArch.zit: "Beta",
    PresetArch.wan: "Simple",
    PresetArch.anima: "Beta",
    PresetArch.ernie: "Simple",
    PresetArch.pid: "Simple",
    PresetArch.krea: KREA2_DEFAULT_SCHEDULER,
}

STEPS = {
    PresetArch.sd: 32,
    PresetArch.xl: 24,
    PresetArch.flux: 20,
    PresetArch.klein: 4,
    PresetArch.qwen: 8,
    PresetArch.lumina: 32,
    PresetArch.zit: 9,
    PresetArch.wan: 4,
    PresetArch.anima: 32,
    PresetArch.ernie: 8,
    PresetArch.pid: 4,
    PresetArch.krea: KREA2_T2I_STEPS,
}

HIRES_STEPS = {
    PresetArch.krea: KREA2_I2I_STEPS,
}

I2I_STEPS = {
    PresetArch.krea: KREA2_I2I_STEPS,
}

CFG = {
    PresetArch.sd: 6.0,
    PresetArch.xl: 4.5,
    PresetArch.flux: 1.0,
    PresetArch.klein: 1.0,
    PresetArch.qwen: 1.0,
    PresetArch.lumina: 4.0,
    PresetArch.zit: 1.0,
    PresetArch.wan: 1.0,
    PresetArch.anima: 4.0,
    PresetArch.ernie: 1.0,
    PresetArch.pid: 1.0,
    PresetArch.krea: KREA2_DEFAULT_CFG,
}

DISTILL = {
    PresetArch.flux: 3.0,
}

SHIFT = {
    PresetArch.xl: -9.0,
    PresetArch.lumina: 6.0,
    PresetArch.zit: 9.0,
    PresetArch.wan: 5.0,
    PresetArch.anima: 3.0,
    PresetArch.ernie: 3.0,
    PresetArch.pid: -1.5,
    PresetArch.krea: -KREA2_DEFAULT_SHIFT,
}

HIRES_DENOISE = {arch: 0.60 for arch in PresetArch}
HIRES_DENOISE[PresetArch.krea] = KREA2_DEFAULT_DENOISE

I2I_DENOISE = {arch: 0.60 for arch in PresetArch}
I2I_DENOISE[PresetArch.krea] = KREA2_DEFAULT_DENOISE

T2I_DIMENSIONS = {
    PresetArch.krea: (1024, 1024),
}

FRAMES = {
    PresetArch.wan.name: 16,
}


def use_distill(arch: str) -> bool:
    return arch in [preset.name for preset in DISTILL.keys()]


def use_shift(arch: str) -> bool:
    return arch in [preset.name for preset in SHIFT.keys()]


def is_video(arch: str) -> int:
    return FRAMES.get(arch, 1)


def register(options_templates: dict):
    from gradio import Dropdown, Slider

    from modules.options import OptionInfo, OptionRow, options_section
    from modules.shared_items import list_samplers, list_schedulers

    for arch in PresetArch:
        name = arch.name

        options_templates.update(
            options_section(
                (None, "Forge Hidden Options"),
                {
                    f"forge_checkpoint_{name}": OptionInfo(DEFAULT_CHECKPOINTS.get(arch)),
                    f"forge_additional_modules_{name}": OptionInfo(list(DEFAULT_ADDITIONAL_MODULES.get(arch, ()))),
                    f"forge_unet_storage_dtype_{name}": OptionInfo(DEFAULT_UNET_STORAGE_DTYPES.get(arch, "Automatic")),
                },
            )
        )

        sampler, scheduler = SAMPLERS[arch], SCHEDULERS[arch]

        options_templates.update(
            options_section(
                (f"ui_{name}", name.upper(), "presets"),
                {
                    f"{name}_t2i_ss1": OptionRow(),
                    f"{name}_t2i_sampler": OptionInfo(sampler, "txt2img Sampler", Dropdown, lambda: {"choices": [x.name for x in list_samplers()]}),
                    f"{name}_t2i_scheduler": OptionInfo(scheduler, "txt2img Scheduler", Dropdown, lambda: {"choices": list_schedulers()}),
                    f"{name}_t2i_ss0": OptionRow(),
                    f"{name}_i2i_ss1": OptionRow(),
                    f"{name}_i2i_sampler": OptionInfo(sampler, "img2img Sampler", Dropdown, lambda: {"choices": [x.name for x in list_samplers()]}),
                    f"{name}_i2i_scheduler": OptionInfo(scheduler, "img2img Scheduler", Dropdown, lambda: {"choices": list_schedulers()}),
                    f"{name}_i2i_ss0": OptionRow(),
                },
            )
        )

        t2i_step = STEPS[arch]
        hires_step = HIRES_STEPS.get(arch, t2i_step)
        i2i_step = I2I_STEPS.get(arch, t2i_step)

        options_templates.update(
            options_section(
                (f"ui_{name}", name.upper(), "presets"),
                {
                    f"{name}_steps1": OptionRow(),
                    f"{name}_t2i_step": OptionInfo(t2i_step, "txt2img Steps", Slider, {"minimum": 0, "maximum": 150, "step": 1}),
                    f"{name}_t2i_hr_step": OptionInfo(hires_step, "txt2img Hires. Steps", Slider, {"minimum": 0, "maximum": 150, "step": 1}),
                    f"{name}_i2i_step": OptionInfo(i2i_step, "img2img Steps", Slider, {"minimum": 0, "maximum": 150, "step": 1}),
                    f"{name}_steps0": OptionRow(),
                },
            )
        )

        options_templates.update(
            options_section(
                (f"ui_{name}", name.upper(), "presets"),
                {
                    f"{name}_denoise1": OptionRow(),
                    f"{name}_t2i_hr_denoise": OptionInfo(
                        HIRES_DENOISE[arch],
                        "txt2img Hires. Denoising Strength",
                        Slider,
                        {"minimum": 0.0, "maximum": 1.0, "step": 0.01},
                    ),
                    f"{name}_i2i_denoise": OptionInfo(
                        I2I_DENOISE[arch],
                        "img2img Denoising Strength",
                        Slider,
                        {"minimum": 0.0, "maximum": 1.0, "step": 0.01},
                    ),
                    f"{name}_denoise0": OptionRow(),
                },
            )
        )

        cfg = CFG[arch]

        options_templates.update(
            options_section(
                (f"ui_{name}", name.upper(), "presets"),
                {
                    f"{name}_cfg1": OptionRow(),
                    f"{name}_t2i_cfg": OptionInfo(cfg, "txt2img CFG", Slider, {"minimum": 0, "maximum": 24, "step": 0.5}),
                    f"{name}_t2i_hr_cfg": OptionInfo(cfg, "txt2img Hires. CFG", Slider, {"minimum": 0, "maximum": 24, "step": 0.5}),
                    f"{name}_i2i_cfg": OptionInfo(cfg, "img2img CFG", Slider, {"minimum": 0, "maximum": 24, "step": 0.5}),
                    f"{name}_cfg0": OptionRow(),
                },
            )
        )

        if (distill := DISTILL.get(arch, None)) is not None:
            options_templates.update(
                options_section(
                    (f"ui_{name}", name.upper(), "presets"),
                    {
                        f"{name}_dcfg1": OptionRow(),
                        f"{name}_t2i_dcfg": OptionInfo(distill, "txt2img Distilled CFG", Slider, {"minimum": 1, "maximum": 24, "step": 0.5}),
                        f"{name}_t2i_hr_dcfg": OptionInfo(distill, "txt2img Hires. Distilled CFG", Slider, {"minimum": 1, "maximum": 24, "step": 0.5}),
                        f"{name}_i2i_dcfg": OptionInfo(distill, "img2img Distilled CFG", Slider, {"minimum": 1, "maximum": 24, "step": 0.5}),
                        f"{name}_dcfg0": OptionRow(),
                    },
                )
            )

        if (shift := SHIFT.get(arch, None)) is not None:
            options_templates.update(
                options_section(
                    (f"ui_{name}", name.upper(), "presets"),
                    {
                        f"{name}_show_shift": OptionInfo((shift > 0.0), "Display Shift Slider"),
                        f"{name}_dcfg1": OptionRow(),
                        f"{name}_t2i_dcfg": OptionInfo(abs(shift), "txt2img Shift", Slider, {"minimum": 1, "maximum": 24, "step": 0.5}),
                        f"{name}_t2i_hr_dcfg": OptionInfo(abs(shift), "txt2img Hires. Shift", Slider, {"minimum": 1, "maximum": 24, "step": 0.5}),
                        f"{name}_i2i_dcfg": OptionInfo(abs(shift), "img2img Shift", Slider, {"minimum": 1, "maximum": 24, "step": 0.5}),
                        f"{name}_dcfg0": OptionRow(),
                    },
                )
            )

        if (fps := FRAMES.get(arch.name, 1)) > 1:
            options_templates.update(
                options_section(
                    (f"ui_{name}", name.upper(), "presets"),
                    {
                        f"{name}_batch1": OptionRow(),
                        f"{name}_t2i_batch_size": OptionInfo(1, "txt2img Frames", Slider, {"minimum": 1, "maximum": fps * 15 + 1, "step": fps}),
                        f"{name}_i2i_batch_size": OptionInfo(1, "img2img Frames", Slider, {"minimum": 1, "maximum": fps * 15 + 1, "step": fps}),
                        f"{name}_batch0": OptionRow(),
                    },
                )
            )
        else:
            options_templates.update(
                options_section(
                    (f"ui_{name}", name.upper(), "presets"),
                    {
                        f"{name}_batch1": OptionRow(),
                        f"{name}_t2i_batch_size": OptionInfo(1, "txt2img Batch Size", Slider, {"minimum": 1, "maximum": 8, "step": 1}),
                        f"{name}_i2i_batch_size": OptionInfo(1, "img2img Batch Size", Slider, {"minimum": 1, "maximum": 8, "step": 1}),
                        f"{name}_batch0": OptionRow(),
                    },
                )
            )

        t2i_width, t2i_height = T2I_DIMENSIONS.get(arch, (0, 0))
        options_templates.update(
            options_section(
                (f"ui_{name}", name.upper(), "presets"),
                {
                    f"{name}_t2i_dim1": OptionRow(),
                    f"{name}_t2i_width": OptionInfo(t2i_width, "txt2img Width", Slider, {"minimum": 0, "maximum": 2048, "step": 64}),
                    f"{name}_i2i_width": OptionInfo(0, "img2img Width", Slider, {"minimum": 0, "maximum": 2048, "step": 64}),
                    f"{name}_t2i_dim0": OptionRow(),
                    f"{name}_i2i_dim1": OptionRow(),
                    f"{name}_t2i_height": OptionInfo(t2i_height, "txt2img Height", Slider, {"minimum": 0, "maximum": 2048, "step": 64}),
                    f"{name}_i2i_height": OptionInfo(0, "img2img Height", Slider, {"minimum": 0, "maximum": 2048, "step": 64}),
                    f"{name}_i2i_dim0": OptionRow(),
                },
            )
        )
