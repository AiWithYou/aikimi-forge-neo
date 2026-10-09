"""Load a packed Qwen decoder while retaining the fixed Iris text encoder."""

from pathlib import Path

import torch


def initialize_encoder(encoder, cfg, tokenizer, decoder, prefix, suffix):
    """Populate the state expected by upstream Qwen3VLTextEncoder._run()."""
    encoder.cfg, encoder.dim, encoder.max_length = cfg, cfg.dim, cfg.max_length
    encoder.hidden_layers = tuple(cfg.hidden_layers)
    encoder.device, encoder.tokenizer, encoder.decoder = torch.device("cpu"), tokenizer, decoder
    if cfg.on_caption_overflow not in ("warn", "error", "silent"):
        raise ValueError("テキストのoverflow設定が一致しません。")
    if decoder.config.hidden_size != cfg.dim or any(
        index < 1 or index > decoder.config.num_hidden_layers for index in encoder.hidden_layers
    ):
        raise ValueError("テキストエンコーダーの寸法・hidden layerが一致しません。")
    encoder._prefix_ids = torch.tensor(tokenizer.encode(prefix, add_special_tokens=False), dtype=torch.long)
    encoder._suffix_ids = torch.tensor(tokenizer.encode(suffix, add_special_tokens=False), dtype=torch.long)
    encoder._prefix_len, encoder._suffix_len = len(encoder._prefix_ids), len(encoder._suffix_ids)
    encoder.caption_budget = cfg.max_length - encoder._suffix_len
    encoder._pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if not encoder._prefix_len or not encoder._suffix_len or encoder.caption_budget < 1 or encoder._pad_id is None:
        raise ValueError("テキストのテンプレート・token上限が一致しません。")
    encoder.calls = encoder.overflow_rows = encoder.overflow_tokens = encoder.max_caption_tokens = 0
    encoder._last_overflow_log = -100
    return encoder


def load_text_encoder(root, cfg):
    from accelerate import init_empty_weights
    from iris3b.text.qwen3_vl import _PROMPT_PREFIX, _PROMPT_SUFFIX, Qwen3VLTextEncoder
    from transformers import AutoConfig, AutoTokenizer, Qwen3VLTextModel

    from .w4a8 import load_w4a8

    folder = Path(root) / "w4a8/text-encoder"
    config = AutoConfig.from_pretrained(folder).text_config
    config._attn_implementation = cfg.attn_implementation
    with init_empty_weights(include_buffers=False):
        decoder = Qwen3VLTextModel(config)
    decoder = load_w4a8(decoder, folder / "model.safetensors")
    return initialize_encoder(
        Qwen3VLTextEncoder.__new__(Qwen3VLTextEncoder),
        cfg,
        AutoTokenizer.from_pretrained(folder),
        decoder,
        _PROMPT_PREFIX,
        _PROMPT_SUFFIX,
    )
