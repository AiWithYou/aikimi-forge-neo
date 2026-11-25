# https://github.com/Comfy-Org/ComfyUI/blob/v0.28.0/comfy/model_detection.py

import logging

from . import model_list


def count_blocks(state_dict_keys: list[str], prefix_string: str) -> int:
    count = 0
    while True:
        c = False
        for k in state_dict_keys:
            if k.startswith(prefix_string.format(count)):
                c = True
                break
        if c == False:
            break
        count += 1
    return count


def tensor_shape(state_dict: dict, key: str) -> tuple[int, ...]:
    quant_state_prefix = key + ".quant_state.bitsandbytes__"
    if any(k.startswith(quant_state_prefix) for k in state_dict):
        from bitsandbytes.nn.modules import QuantState

        stats_prefix = key + "."
        quant_state_dict = {k[len(stats_prefix) :]: v for k, v in state_dict.items() if k.startswith(stats_prefix)}
        return tuple(int(x) for x in QuantState.from_dict(quant_state_dict, device=state_dict[key].device).shape)

    return tuple(int(x) for x in state_dict[key].shape)


def calculate_transformer_depth(prefix, state_dict_keys, state_dict):
    context_dim = None
    use_linear_in_transformer = False

    transformer_prefix = prefix + "1.transformer_blocks."
    transformer_keys = sorted(list(filter(lambda a: a.startswith(transformer_prefix), state_dict_keys)))
    if len(transformer_keys) > 0:
        last_transformer_depth = count_blocks(state_dict_keys, transformer_prefix + "{}")
        context_dim = int(state_dict["{}0.attn2.to_k.weight".format(transformer_prefix)].shape[1])
        use_linear_in_transformer = len(state_dict["{}1.proj_in.weight".format(prefix)].shape) == 2
        time_stack = "{}1.time_stack.0.attn1.to_q.weight".format(prefix) in state_dict or "{}1.time_mix_blocks.0.attn1.to_q.weight".format(prefix) in state_dict
        time_stack_cross = "{}1.time_stack.0.attn2.to_q.weight".format(prefix) in state_dict or "{}1.time_mix_blocks.0.attn2.to_q.weight".format(prefix) in state_dict
        return last_transformer_depth, context_dim, use_linear_in_transformer, time_stack, time_stack_cross
    return None


def detect_unet_config(state_dict: dict, key_prefix: str) -> dict:
    state_dict_keys = list(state_dict.keys())

    if "{}blocks.0.mlp.layer1.weight".format(key_prefix) in state_dict_keys:  # Anima
        dit_config = {}
        assert "{}llm_adapter.blocks.0.cross_attn.q_proj.weight".format(key_prefix) in state_dict_keys
        dit_config["image_model"] = "anima"
        dit_config["in_channels"] = int(state_dict["{}x_embedder.proj.1.weight".format(key_prefix)].shape[1] / 4) - 1
        assert dit_config["in_channels"] == 16
        dit_config["out_channels"] = 16
        dit_config["patch_spatial"] = 2
        dit_config["patch_temporal"] = 1
        dit_config["model_channels"] = int(state_dict["{}x_embedder.proj.1.weight".format(key_prefix)].shape[0])
        assert dit_config["model_channels"] == 2048
        dit_config["concat_padding_mask"] = True
        dit_config["crossattn_emb_channels"] = 1024
        dit_config["adaln_lora_dim"] = 256
        dit_config["num_blocks"] = count_blocks(state_dict_keys, "{}blocks.".format(key_prefix) + "{}.")
        dit_config["num_heads"] = 16
        dit_config["rope_h_extrapolation_ratio"] = 4.0
        dit_config["rope_w_extrapolation_ratio"] = 4.0
        dit_config["rope_t_extrapolation_ratio"] = 1.0

        return dit_config

    if "{}txtfusion.projector.weight".format(key_prefix) in state_dict_keys:  # Krea 2
        dit_config = {}
        dit_config["image_model"] = "krea2"
        head_dim = 128
        first_w_shape = tensor_shape(state_dict, "{}first.weight".format(key_prefix))
        dit_config["features"] = first_w_shape[0]
        dit_config["channels"] = first_w_shape[1] // (2 * 2)
        dit_config["patch"] = 2
        dit_config["layers"] = count_blocks(state_dict_keys, "{}blocks.".format(key_prefix) + "{}.")
        dit_config["heads"] = tensor_shape(state_dict, "{}blocks.0.attn.wq.weight".format(key_prefix))[0] // head_dim
        dit_config["kvheads"] = tensor_shape(state_dict, "{}blocks.0.attn.wk.weight".format(key_prefix))[0] // head_dim
        dit_config["txtlayers"] = tensor_shape(state_dict, "{}txtfusion.projector.weight".format(key_prefix))[1]
        dit_config["txtdim"] = tensor_shape(state_dict, "{}txtfusion.layerwise_blocks.0.prenorm.scale".format(key_prefix))[0]
        return dit_config

    if "{}input_blocks.0.0.weight".format(key_prefix) not in state_dict_keys:
        return None

    unet_config = {
        "use_checkpoint": False,
        "image_size": 32,
        "use_spatial_transformer": True,
        "legacy": False,
    }

    y_input = "{}label_emb.0.0.weight".format(key_prefix)
    if y_input in state_dict_keys:
        unet_config["num_classes"] = "sequential"
        unet_config["adm_in_channels"] = int(state_dict[y_input].shape[1])
    else:
        unet_config["adm_in_channels"] = None

    model_channels = int(state_dict["{}input_blocks.0.0.weight".format(key_prefix)].shape[0])
    in_channels = int(state_dict["{}input_blocks.0.0.weight".format(key_prefix)].shape[1])

    out_key = "{}out.2.weight".format(key_prefix)
    if out_key in state_dict:
        out_channels = int(state_dict[out_key].shape[0])
    else:
        out_channels = 4

    num_res_blocks = []
    channel_mult = []
    transformer_depth = []
    transformer_depth_output = []
    context_dim = None
    use_linear_in_transformer = False

    video_model = False

    current_res = 1
    count = 0

    last_res_blocks = 0
    last_channel_mult = 0

    input_block_count = count_blocks(state_dict_keys, "{}input_blocks".format(key_prefix) + ".{}.")
    for count in range(input_block_count):
        prefix = "{}input_blocks.{}.".format(key_prefix, count)
        prefix_output = "{}output_blocks.{}.".format(key_prefix, input_block_count - count - 1)

        block_keys = sorted(list(filter(lambda a: a.startswith(prefix), state_dict_keys)))
        if len(block_keys) == 0:
            break

        block_keys_output = sorted(list(filter(lambda a: a.startswith(prefix_output), state_dict_keys)))

        if "{}0.op.weight".format(prefix) in block_keys:
            num_res_blocks.append(last_res_blocks)
            channel_mult.append(last_channel_mult)

            current_res *= 2
            last_res_blocks = 0
            last_channel_mult = 0
            out = calculate_transformer_depth(prefix_output, state_dict_keys, state_dict)
            if out is not None:
                transformer_depth_output.append(out[0])
            else:
                transformer_depth_output.append(0)
        else:
            res_block_prefix = "{}0.in_layers.0.weight".format(prefix)
            if res_block_prefix in block_keys:
                last_res_blocks += 1
                last_channel_mult = int(state_dict["{}0.out_layers.3.weight".format(prefix)].shape[0]) // model_channels

                out = calculate_transformer_depth(prefix, state_dict_keys, state_dict)
                if out is not None:
                    transformer_depth.append(out[0])
                    if context_dim is None:
                        context_dim = out[1]
                        use_linear_in_transformer = out[2]
                        video_model = out[3]
                else:
                    transformer_depth.append(0)

            res_block_prefix = "{}0.in_layers.0.weight".format(prefix_output)
            if res_block_prefix in block_keys_output:
                out = calculate_transformer_depth(prefix_output, state_dict_keys, state_dict)
                if out is not None:
                    transformer_depth_output.append(out[0])
                else:
                    transformer_depth_output.append(0)

    num_res_blocks.append(last_res_blocks)
    channel_mult.append(last_channel_mult)
    if "{}middle_block.1.proj_in.weight".format(key_prefix) in state_dict_keys:
        transformer_depth_middle = count_blocks(state_dict_keys, "{}middle_block.1.transformer_blocks.".format(key_prefix) + "{}")
    elif "{}middle_block.0.in_layers.0.weight".format(key_prefix) in state_dict_keys:
        transformer_depth_middle = -1
    else:
        transformer_depth_middle = -2

    unet_config["in_channels"] = in_channels
    unet_config["out_channels"] = out_channels
    unet_config["model_channels"] = model_channels
    unet_config["num_res_blocks"] = num_res_blocks
    unet_config["transformer_depth"] = transformer_depth
    unet_config["transformer_depth_output"] = transformer_depth_output
    unet_config["channel_mult"] = channel_mult
    unet_config["transformer_depth_middle"] = transformer_depth_middle
    unet_config["use_linear_in_transformer"] = use_linear_in_transformer
    unet_config["context_dim"] = context_dim

    assert not video_model
    unet_config["use_temporal_resblock"] = False
    unet_config["use_temporal_attention"] = False

    return unet_config


def model_config_from_unet_config(unet_config, state_dict=None):
    for model_config in model_list.models:
        if model_config.matches(unet_config, state_dict):
            return model_config(unet_config)

    logging.error("no match {}".format(unet_config))
    return None


def model_config_from_unet(state_dict, unet_key_prefix, use_base_if_no_match=False):
    unet_config = detect_unet_config(state_dict, unet_key_prefix)
    if unet_config is None:
        return None
    model_config = model_config_from_unet_config(unet_config, state_dict)
    if model_config is None and use_base_if_no_match:
        return model_list.BASE(unet_config)
    else:
        return model_config


def top_candidate(state_dict, candidates):
    counts = {k: 0 for k in candidates}
    for k in state_dict:
        for c in candidates:
            if k.startswith(c):
                counts[c] += 1
                break
    top = max(counts, key=counts.get)
    return top, counts[top]


def unet_prefix_from_state_dict(state_dict: dict) -> str:
    candidates = (
        "model.diffusion_model.",  # ldm/sgm models
        "net.",  # cosmos
    )
    for prefix in candidates:
        if sum(1 for k in state_dict if k.startswith(prefix)) > 5:
            return prefix
    return "model."  # etc.


def convert_config(unet_config):
    new_config = unet_config.copy()
    num_res_blocks = new_config.get("num_res_blocks", None)
    channel_mult = new_config.get("channel_mult", None)

    if isinstance(num_res_blocks, int):
        num_res_blocks = len(channel_mult) * [num_res_blocks]

    if "attention_resolutions" in new_config:
        attention_resolutions = new_config.pop("attention_resolutions")
        transformer_depth = new_config.get("transformer_depth", None)
        transformer_depth_middle = new_config.get("transformer_depth_middle", None)

        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        if transformer_depth_middle is None:
            transformer_depth_middle = transformer_depth[-1]
        t_in = []
        t_out = []
        s = 1
        for i in range(len(num_res_blocks)):
            res = num_res_blocks[i]
            d = 0
            if s in attention_resolutions:
                d = transformer_depth[i]

            t_in += [d] * res
            t_out += [d] * (res + 1)
            s *= 2
        transformer_depth = t_in
        new_config["transformer_depth"] = t_in
        new_config["transformer_depth_output"] = t_out
        new_config["transformer_depth_middle"] = transformer_depth_middle

    new_config["num_res_blocks"] = num_res_blocks
    return new_config


def unet_config_from_diffusers_unet(state_dict, dtype=None):
    match = {}
    transformer_depth = []

    attn_res = 1
    down_blocks = count_blocks(state_dict, "down_blocks.{}")
    for i in range(down_blocks):
        attn_blocks = count_blocks(state_dict, "down_blocks.{}.attentions.".format(i) + "{}")
        res_blocks = count_blocks(state_dict, "down_blocks.{}.resnets.".format(i) + "{}")
        for ab in range(attn_blocks):
            transformer_count = count_blocks(
                state_dict,
                "down_blocks.{}.attentions.{}.transformer_blocks.".format(i, ab) + "{}",
            )
            transformer_depth.append(transformer_count)
            if transformer_count > 0:
                match["context_dim"] = int(state_dict["down_blocks.{}.attentions.{}.transformer_blocks.0.attn2.to_k.weight".format(i, ab)].shape[1])

        attn_res *= 2
        if attn_blocks == 0:
            for i in range(res_blocks):
                transformer_depth.append(0)

    match["transformer_depth"] = transformer_depth

    match["model_channels"] = int(state_dict["conv_in.weight"].shape[0])
    match["in_channels"] = int(state_dict["conv_in.weight"].shape[1])
    match["adm_in_channels"] = None
    if "class_embedding.linear_1.weight" in state_dict:
        match["adm_in_channels"] = int(state_dict["class_embedding.linear_1.weight"].shape[1])
    elif "add_embedding.linear_1.weight" in state_dict:
        match["adm_in_channels"] = int(state_dict["add_embedding.linear_1.weight"].shape[1])

    SDXL = {
        "use_checkpoint": False,
        "image_size": 32,
        "out_channels": 4,
        "use_spatial_transformer": True,
        "legacy": False,
        "num_classes": "sequential",
        "adm_in_channels": 2816,
        "dtype": dtype,
        "in_channels": 4,
        "model_channels": 320,
        "num_res_blocks": [2, 2, 2],
        "transformer_depth": [0, 0, 2, 2, 10, 10],
        "channel_mult": [1, 2, 4],
        "transformer_depth_middle": 10,
        "use_linear_in_transformer": True,
        "context_dim": 2048,
        "num_head_channels": 64,
        "transformer_depth_output": [0, 0, 0, 2, 2, 2, 10, 10, 10],
        "use_temporal_attention": False,
        "use_temporal_resblock": False,
    }

    SDXL_refiner = {
        "use_checkpoint": False,
        "image_size": 32,
        "out_channels": 4,
        "use_spatial_transformer": True,
        "legacy": False,
        "num_classes": "sequential",
        "adm_in_channels": 2560,
        "dtype": dtype,
        "in_channels": 4,
        "model_channels": 384,
        "num_res_blocks": [2, 2, 2, 2],
        "transformer_depth": [0, 0, 4, 4, 4, 4, 0, 0],
        "channel_mult": [1, 2, 4, 4],
        "transformer_depth_middle": 4,
        "use_linear_in_transformer": True,
        "context_dim": 1280,
        "num_head_channels": 64,
        "transformer_depth_output": [0, 0, 0, 4, 4, 4, 4, 4, 4, 0, 0, 0],
        "use_temporal_attention": False,
        "use_temporal_resblock": False,
    }

    SDXL_mid_cnet = {
        "use_checkpoint": False,
        "image_size": 32,
        "out_channels": 4,
        "use_spatial_transformer": True,
        "legacy": False,
        "num_classes": "sequential",
        "adm_in_channels": 2816,
        "dtype": dtype,
        "in_channels": 4,
        "model_channels": 320,
        "num_res_blocks": [2, 2, 2],
        "transformer_depth": [0, 0, 0, 0, 1, 1],
        "channel_mult": [1, 2, 4],
        "transformer_depth_middle": 1,
        "use_linear_in_transformer": True,
        "context_dim": 2048,
        "num_head_channels": 64,
        "transformer_depth_output": [0, 0, 0, 0, 0, 0, 1, 1, 1],
        "use_temporal_attention": False,
        "use_temporal_resblock": False,
    }

    SDXL_small_cnet = {
        "use_checkpoint": False,
        "image_size": 32,
        "out_channels": 4,
        "use_spatial_transformer": True,
        "legacy": False,
        "num_classes": "sequential",
        "adm_in_channels": 2816,
        "dtype": dtype,
        "in_channels": 4,
        "model_channels": 320,
        "num_res_blocks": [2, 2, 2],
        "transformer_depth": [0, 0, 0, 0, 0, 0],
        "channel_mult": [1, 2, 4],
        "transformer_depth_middle": 0,
        "use_linear_in_transformer": True,
        "num_head_channels": 64,
        "context_dim": 1,
        "transformer_depth_output": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        "use_temporal_attention": False,
        "use_temporal_resblock": False,
    }

    SDXL_diffusers_inpaint = {
        "use_checkpoint": False,
        "image_size": 32,
        "out_channels": 4,
        "use_spatial_transformer": True,
        "legacy": False,
        "num_classes": "sequential",
        "adm_in_channels": 2816,
        "dtype": dtype,
        "in_channels": 9,
        "model_channels": 320,
        "num_res_blocks": [2, 2, 2],
        "transformer_depth": [0, 0, 2, 2, 10, 10],
        "channel_mult": [1, 2, 4],
        "transformer_depth_middle": 10,
        "use_linear_in_transformer": True,
        "context_dim": 2048,
        "num_head_channels": 64,
        "transformer_depth_output": [0, 0, 0, 2, 2, 2, 10, 10, 10],
        "use_temporal_attention": False,
        "use_temporal_resblock": False,
    }

    supported_models = [
        SDXL,
        SDXL_refiner,
        SDXL_mid_cnet,
        SDXL_small_cnet,
        SDXL_diffusers_inpaint,
    ]

    for unet_config in supported_models:
        matches = True
        for k in match:
            if match[k] != unet_config[k]:
                matches = False
                break
        if matches:
            return convert_config(unet_config)
    return None


def model_config_from_diffusers_unet(state_dict):
    unet_config = unet_config_from_diffusers_unet(state_dict)
    if unet_config is not None:
        return model_config_from_unet_config(unet_config)
    return None
