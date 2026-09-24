# Copyright 2025 HuggingFace Inc. team. All rights reserved.
# Modifications Copyright 2026 Aikimi Forge Neo contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. Full text: vendor/accelerate/LICENSE.
"""Adapt the pinned SenseNova backbone to Transformers 5 without removing paths.

Mask construction follows Transformers 4.57.6 (Apache-2.0), using the 5.10.4
mask interfaces. In particular, explicit cache_position remains authoritative.
Only SenseNova classes/module globals are adapted; Transformers is unchanged.
"""

from functools import wraps


def create_causal_mask(
    config,
    input_embeds,
    attention_mask,
    cache_position,
    past_key_values,
    position_ids=None,
    or_mask_function=None,
    and_mask_function=None,
):
    import torch
    from transformers import masking_utils as masks

    if isinstance(attention_mask, (torch.Tensor, masks.BlockMask)) and len(attention_mask.shape) == 4:
        return attention_mask
    if config._attn_implementation not in masks.ALL_MASK_ATTENTION_FUNCTIONS:
        return None
    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = attention_mask.to(device=input_embeds.device, dtype=torch.bool)
    layer_idx = 0
    if hasattr(past_key_values, "is_sliding") and False in past_key_values.is_sliding:
        layer_idx = past_key_values.is_sliding.index(False)
    q_length = cache_position.shape[0]
    if past_key_values is None:
        kv_length, kv_offset = input_embeds.shape[1], 0
    else:
        kv_length, kv_offset = past_key_values.get_mask_sizes(q_length, layer_idx)

    factory = masks.causal_mask_function
    allow_skip = not getattr(past_key_values, "is_compileable", False)
    use_vmap = False
    if or_mask_function is not None:
        factory = masks.or_masks(factory, or_mask_function)
        allow_skip, use_vmap = False, True
    if and_mask_function is not None:
        factory = masks.and_masks(factory, and_mask_function)
        allow_skip, use_vmap = False, True
    if position_ids is not None and attention_mask is None and past_key_values is None:
        if position_ids.shape[0] != input_embeds.shape[0]:
            position_ids = position_ids.expand(input_embeds.shape[0], -1)
        packed = masks.find_packed_sequence_indices(position_ids)
        if packed is not None:
            factory = masks.and_masks(factory, masks.packed_sequence_mask_function(packed))
            allow_skip = False
    return masks.ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation](
        batch_size=input_embeds.shape[0],
        q_length=q_length,
        kv_length=kv_length,
        q_offset=cache_position[0],
        kv_offset=kv_offset,
        mask_function=factory,
        attention_mask=attention_mask,
        allow_is_causal_skip=allow_skip,
        dtype=input_embeds.dtype,
        config=config,
        use_vmap=use_vmap,
        device=input_embeds.device,
    )


def _adapt_config(cls):
    if cls.__dict__.get("_aikimi_transformers5", False):
        return
    original = cls.__init__

    @wraps(original)
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.rope_scaling = self.rope_parameters

    # Keep the legacy spatial-axis assignments in sync with the Transformers 5
    # RoPE functions, which read rope_parameters instead of rope_theta.
    def set_theta(self, value):
        if not isinstance(getattr(self, "rope_parameters", None), dict):
            self.rope_parameters = {}
        self.rope_parameters["rope_theta"] = value

    def get_theta(self):
        return (getattr(self, "rope_parameters", None) or {}).get("rope_theta", self.default_theta)

    cls.rope_theta = property(get_theta, set_theta)
    cls.__init__ = initialize
    cls._aikimi_transformers5 = True


def install():
    from SenseNova.src.sensenova_u1.models.neo_unify import (
        configuration_neo_chat as configs,
    )
    from SenseNova.src.sensenova_u1.models.neo_unify import (
        modeling_qwen3 as dense,
    )
    from SenseNova.src.sensenova_u1.models.neo_unify import (
        modeling_qwen3_moe as moe,
    )

    for cls in (configs.NEOLLMConfig, configs.NEOMoELLMConfig):
        _adapt_config(cls)
    for module in (dense, moe):
        module.create_causal_mask = create_causal_mask

    # Transformers 5 initializes rotary buffers via this method. Reuse the
    # backbone's frequency-range preserving implementation, not a new formula.
    def compute_default(self, config, device=None, **kwargs):
        return self.rope_init_fn(config, device)

    dense.Qwen3RotaryEmbedding.compute_default_rope_parameters = compute_default
