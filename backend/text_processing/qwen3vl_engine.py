# https://github.com/Comfy-Org/ComfyUI/blob/v0.26.1/comfy/sd1_clip.py
# https://github.com/Comfy-Org/ComfyUI/blob/v0.26.1/comfy/text_encoders/krea2.py
# https://github.com/Comfy-Org/ComfyUI/blob/v0.26.1/comfy/text_encoders/qwen35.py
# https://github.com/Comfy-Org/ComfyUI/blob/v0.26.1/comfy/text_encoders/qwen3vl.py

import torch

from backend import memory_management
from backend.text_processing import emphasis

KREA2_TAP_LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]


class PromptChunk:
    def __init__(self):
        self.tokens = []
        self.multipliers = []


class Qwen3VLTextProcessingEngine:
    def __init__(self, text_encoder, tokenizer):
        # Qwen3-VL/Krea2 follows the upstream disable_weights behavior: prompt
        # emphasis syntax is passed to the language model as literal text.
        self.emphasis = emphasis.EmphasisNone()
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.max_length = 99999999
        self.min_length = 1
        self.id_pad = 151643
        self.id_template = 151644
        self.id_image = 151655
        self.llama_template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.vision_block = "<|vision_start|><|image_pad|><|vision_end|>"

    def tokenize(self, texts, images=None):
        images = images or []
        prompts = [
            f"{self.vision_block * len(images)}{str(text).strip()}"
            for text in texts
        ]
        llama_texts = [self.llama_template.format(prompt or " ") for prompt in prompts]
        return self.tokenizer(llama_texts)["input_ids"]

    def tokenize_line(self, line: str, images=None):
        images = images or []
        tokenized = self.tokenize([line], images)[0]
        chunk = PromptChunk()
        embed_count = 0
        for token in tokenized:
            if token == self.id_image:
                if embed_count >= len(images):
                    raise ValueError("Qwen3VL tokenizer produced more image tokens than supplied images")
                token = {"type": "image", "data": images[embed_count], "original_type": "image"}
                embed_count += 1
            chunk.tokens.append(token)
            chunk.multipliers.append(1.0)

        if embed_count != len(images):
            raise ValueError(
                "Qwen3VL tokenizer produced fewer image tokens than supplied images"
            )

        return [chunk]

    def __call__(self, texts, images=None):
        images = images or []
        self.emphasis = emphasis.EmphasisNone()

        zs = []
        cache = {}
        for line in texts:
            if line not in cache:
                line_z_values = []
                for chunk in self.tokenize_line(line, images):
                    z = self.process_tokens([chunk.tokens], [chunk.multipliers])
                    z = self.strip_template(z, chunk.tokens)
                    batch, sequence, fused = z.shape
                    expected_fused = len(KREA2_TAP_LAYERS) * 2560
                    if batch != 1 or fused != expected_fused:
                        raise ValueError(
                            "Krea2 expects one Qwen3-VL conditioning batch with "
                            f"{len(KREA2_TAP_LAYERS)}x2560={expected_fused} features; "
                            f"got {tuple(z.shape)}"
                        )
                    line_z_values.append(
                        z.reshape(batch * sequence, len(KREA2_TAP_LAYERS), 2560)
                    )
                cache[line] = line_z_values
            zs.extend(cache[line])
        return zs

    def strip_template(self, out, tokens):
        template_end = 0
        count_im_start = 0
        for i, value in enumerate(tokens):
            try:
                elem = int(value)
                if elem == self.id_template and count_im_start < 2:
                    template_end = i
                    count_im_start += 1
            except TypeError:
                continue

        if len(tokens) > template_end + 2 and out.shape[2] > template_end + 3:
            if int(tokens[template_end + 1]) == 872 and int(tokens[template_end + 2]) == 198:
                template_end += 3

        out = out[:, :, template_end:]
        batch, taps, sequence, hidden = out.shape
        return out.permute(0, 2, 1, 3).reshape(batch, sequence, taps * hidden)

    def process_embeds(self, batch_tokens):
        device = memory_management.text_encoder_device()
        embeds_out = []
        attention_masks = []
        num_tokens = []
        embeds_info = []

        for tokens in batch_tokens:
            attention_mask = []
            tokens_temp = []
            other_embeds = []
            eos = False

            for index, value in enumerate(tokens):
                try:
                    token = int(value)
                    attention_mask.append(0 if eos else 1)
                    tokens_temp.append(token)
                    if not eos and token == self.id_pad:
                        eos = True
                except TypeError:
                    other_embeds.append((index, value))

            tokens_embed = torch.tensor([tokens_temp], device=device, dtype=torch.long)
            tokens_embed = self.text_encoder.get_input_embeddings()(tokens_embed)
            index_offset = 0
            embeds_info = []

            for original_index, embed_value in other_embeds:
                emb, extra = self.text_encoder.preprocess_embed(embed_value, device=device)
                if emb is None:
                    index_offset -= 1
                    continue
                insert_at = index_offset + original_index
                emb = emb.view(1, -1, emb.shape[-1]).to(device=device, dtype=torch.float32)
                emb_shape = emb.shape[1]
                assert emb.shape[-1] == tokens_embed.shape[-1]
                tokens_embed = torch.cat([tokens_embed[:, :insert_at], emb, tokens_embed[:, insert_at:]], dim=1)
                attention_mask = attention_mask[:insert_at] + [1] * emb_shape + attention_mask[insert_at:]
                index_offset += emb_shape - 1
                embeds_info.append({"type": embed_value.get("type"), "index": insert_at, "size": emb_shape, "extra": extra})

            embeds_out.append(tokens_embed)
            attention_masks.append(attention_mask)
            num_tokens.append(sum(attention_mask))

        return torch.cat(embeds_out), torch.tensor(attention_masks, device=device, dtype=torch.long), num_tokens, embeds_info

    def process_tokens(self, batch_tokens, batch_multipliers):
        embeds, mask, count, info = self.process_embeds(batch_tokens)
        self.emphasis.tokens = batch_tokens
        self.emphasis.multipliers = torch.asarray(batch_multipliers).to(embeds)
        self.emphasis.z = embeds
        self.emphasis.after_transformers()
        embeds = self.emphasis.z
        _, z = self.text_encoder(None, embeds=embeds, attention_mask=mask, num_tokens=count, embeds_info=info, intermediate_output=KREA2_TAP_LAYERS, final_layer_norm_intermediate=False)
        return z
