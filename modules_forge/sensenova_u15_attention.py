"""参照読込のAttentionを2ヘッドずつ計算し、一時行列の同時保持を避ける。"""

import torch


def headwise_eager_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    if module.training or dropout:
        raise RuntimeError("参照省メモリAttentionは推論専用です。")
    batch, heads, length, _ = query.shape
    result = torch.empty((batch, length, heads, value.shape[-1]), device=query.device, dtype=query.dtype)
    groups = module.num_key_value_groups
    if groups != 1:
        key = (
            key[:, :, None]
            .expand(batch, key.shape[1], groups, key.shape[2], key.shape[3])
            .reshape(batch, heads, key.shape[2], key.shape[3])
        )
        value = (
            value[:, :, None]
            .expand(batch, value.shape[1], groups, value.shape[2], value.shape[3])
            .reshape(batch, heads, value.shape[2], value.shape[3])
        )
    # 1ヘッドではCUDAが別の行列積カーネルを選び、BF16の丸めが変わる。
    # 固定モデルの偶数ヘッドを2つずつ渡し、元のbatched matmulを維持する。
    if heads % 2:
        raise RuntimeError("参照省メモリAttentionには偶数のヘッド数が必要です。")
    for head in range(0, heads, 2):
        weights = torch.matmul(query[:, head : head + 2], key[:, head : head + 2].transpose(2, 3)) * scaling
        if attention_mask is not None:
            mask = attention_mask if attention_mask.shape[1] == 1 else attention_mask[:, head : head + 2]
            weights = weights + mask[..., : key.shape[-2]]
        weights = torch.nn.functional.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
        output = torch.matmul(weights, value[:, head : head + 2])
        result[:, :, head : head + 2].copy_(output.transpose(1, 2))
        del weights, output
    return result, None
