# reference: https://github.com/Comfy-Org/ComfyUI/blob/v0.27.1/comfy/supported_models.py

from enum import Enum

import torch

from . import diffusers_convert, latent, utils


class ModelType(Enum):
    EPS = 1
    V_PREDICTION = 2
    FLUX = 3
    FLOW = 4


class BASE:
    huggingface_repo = None
    unet_config = {}
    unet_extra_config = {
        "num_heads": -1,
        "num_head_channels": 64,
    }

    required_keys = {}

    clip_prefix = []
    clip_vision_prefix = None
    noise_aug_config = None
    sampling_settings = {}
    latent_format = latent.LatentFormat
    vae_key_prefix = ["first_stage_model."]
    text_encoder_key_prefix = ["cond_stage_model."]
    supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    memory_usage_factor = 2.0

    manual_cast_dtype = None
    unet_target = "unet"
    vae_target = "vae"

    @classmethod
    def matches(cls, unet_config, state_dict=None):
        for k in cls.unet_config:
            if k not in unet_config or cls.unet_config[k] != unet_config[k]:
                return False
        if state_dict is not None:
            for k in cls.required_keys:
                if k not in state_dict:
                    return False
        return True

    def model_type(self, state_dict):
        return ModelType.FLOW

    def clip_target(self, state_dict: dict):
        return {}

    def inpaint_model(self):
        return False

    def __init__(self, unet_config):
        self.unet_config = unet_config.copy()
        self.sampling_settings = self.sampling_settings.copy()
        self.latent_format = self.latent_format()
        for x in self.unet_extra_config:
            self.unet_config[x] = self.unet_extra_config[x]

    def process_clip_state_dict(self, state_dict):
        replace_prefix = {k: "" for k in self.text_encoder_key_prefix}
        return utils.state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=True)

    def process_unet_state_dict(self, state_dict):
        return state_dict

    def process_vae_state_dict(self, state_dict):
        return state_dict

    def process_clip_state_dict_for_saving(self, state_dict):
        replace_prefix = {"": self.text_encoder_key_prefix[0]}
        return utils.state_dict_prefix_replace(state_dict, replace_prefix)

    def process_clip_vision_state_dict_for_saving(self, state_dict):
        replace_prefix = {}
        if self.clip_vision_prefix is not None:
            replace_prefix[""] = self.clip_vision_prefix
        return utils.state_dict_prefix_replace(state_dict, replace_prefix)

    def process_unet_state_dict_for_saving(self, state_dict):
        replace_prefix = {"": "model.diffusion_model."}
        return utils.state_dict_prefix_replace(state_dict, replace_prefix)

    def process_vae_state_dict_for_saving(self, state_dict):
        replace_prefix = {"": self.vae_key_prefix[0]}
        return utils.state_dict_prefix_replace(state_dict, replace_prefix)


class SDXLRefiner(BASE):
    huggingface_repo = "stabilityai/stable-diffusion-xl-refiner-1.0"

    unet_config = {
        "model_channels": 384,
        "use_linear_in_transformer": True,
        "context_dim": 1280,
        "adm_in_channels": 2560,
        "transformer_depth": [0, 0, 4, 4, 4, 4, 0, 0],
        "use_temporal_attention": False,
    }

    latent_format = latent.SDXL
    memory_usage_factor = 1.0

    def model_type(self, state_dict: dict):
        return ModelType.EPS

    def process_clip_state_dict(self, state_dict):
        replace_prefix = {"conditioner.embedders.0.model.": "clip_g."}
        state_dict = utils.state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=True)
        return utils.clip_text_transformers_convert(state_dict, "clip_g.", "clip_g.transformer.")

    def process_clip_state_dict_for_saving(self, state_dict):
        state_dict_g = diffusers_convert.convert_text_enc_state_dict_v20(state_dict, "clip_g")
        if "clip_g.transformer.text_model.embeddings.position_ids" in state_dict_g:
            state_dict_g.pop("clip_g.transformer.text_model.embeddings.position_ids")
        replace_prefix = {"clip_g": "conditioner.embedders.0.model"}
        return utils.state_dict_prefix_replace(state_dict_g, replace_prefix)

    def clip_target(self, state_dict: dict):
        return {"clip_g": "text_encoder"}


class SDXL(BASE):
    huggingface_repo = "stabilityai/stable-diffusion-xl-base-1.0"

    unet_config = {
        "model_channels": 320,
        "out_channels": 4,
        "use_linear_in_transformer": True,
        "transformer_depth": [0, 0, 2, 2, 10, 10],
        "context_dim": 2048,
        "adm_in_channels": 2816,
        "use_temporal_attention": False,
    }

    latent_format = latent.SDXL
    memory_usage_factor = 0.8

    def inpaint_model(self):
        return self.unet_config.get("in_channels", -1) > 4

    def model_type(self, state_dict: dict):
        if "v_pred" in state_dict:
            return ModelType.V_PREDICTION
        else:
            return ModelType.EPS

    def process_clip_state_dict(self, state_dict):
        replace_prefix = {
            "conditioner.embedders.0.transformer.text_model": "clip_l.transformer.text_model",
            "conditioner.embedders.1.model.": "clip_g.",
        }
        state_dict = utils.state_dict_prefix_replace(state_dict, replace_prefix, filter_keys=True)
        return utils.clip_text_transformers_convert(state_dict, "clip_g.", "clip_g.transformer.")

    def process_clip_state_dict_for_saving(self, state_dict):
        state_dict_g = diffusers_convert.convert_text_enc_state_dict_v20(state_dict, "clip_g")
        for k in state_dict:
            if k.startswith("clip_l"):
                state_dict_g[k] = state_dict[k]

        state_dict_g["clip_l.transformer.text_model.embeddings.position_ids"] = torch.arange(77).expand((1, -1))
        pop_keys = ["clip_l.transformer.text_projection.weight", "clip_l.logit_scale"]
        for p in pop_keys:
            if p in state_dict_g:
                state_dict_g.pop(p)

        replace_prefix = {
            "clip_g": "conditioner.embedders.1.model",
            "clip_l": "conditioner.embedders.0",
        }
        return utils.state_dict_prefix_replace(state_dict_g, replace_prefix)

    def clip_target(self, state_dict: dict):
        return {"clip_l": "text_encoder", "clip_g": "text_encoder_2"}


class Anima(BASE):
    huggingface_repo = "circlestone-labs/Anima"

    unet_config = {
        "image_model": "anima",
    }

    sampling_settings = {
        "multiplier": 1.0,
        "shift": 3.0,
    }

    unet_extra_config = {}
    latent_format = latent.Wan21

    memory_usage_factor = 1.32

    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    unet_target = "transformer"

    def clip_target(self, state_dict={}):
        return {"qwen3_06b.transformer": "text_encoder"}


class Krea2(BASE):
    huggingface_repo = "krea/Krea-2-Raw"

    unet_config = {
        "image_model": "krea2",
    }

    sampling_settings = {
        "multiplier": 1.0,
        "shift": 1.15,
    }

    memory_usage_factor = 2.2

    latent_format = latent.Wan21

    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    unet_target = "transformer"

    def clip_target(self, state_dict={}):
        return {"qwen3vl_4b.transformer": "text_encoder"}

    def model_type(self, state_dict):
        return ModelType.FLUX


models = [SDXL, SDXLRefiner, Anima, Krea2]
