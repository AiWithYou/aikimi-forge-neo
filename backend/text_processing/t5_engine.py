import torch

from backend import memory_management
from backend.args import dynamic_args
from backend.text_processing import emphasis, parsing
from modules.shared import opts


class PromptChunk:
    def __init__(self):
        self.tokens = []
        self.multipliers = []


class T5TextProcessingEngine:
    def __init__(self, text_encoder, tokenizer, min_length: int = 256, min_padding: int = -1):
        self.emphasis = emphasis.get_current_option(opts.emphasis)()

        self.text_encoder = text_encoder.transformer
        self.tokenizer = tokenizer

        self.min_length = min_length
        self.min_padding = min_padding
        self.id_end = 1
        self.id_pad = 0

    def tokenize(self, texts):
        tokenized = self.tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"]
        return tokenized

    def encode_with_transformers(self, tokens):
        device = memory_management.text_encoder_device()
        tokens = tokens.to(device)
        return self.text_encoder(input_ids=tokens)

    def tokenize_line(self, line):
        parsed = parsing.parse_prompt_attention(line, self.emphasis.name)

        tokenized = self.tokenize([text for text, _ in parsed])

        chunks = []
        chunk = PromptChunk()
        token_count = 0

        def next_chunk():
            nonlocal token_count
            nonlocal chunk

            chunk.tokens = chunk.tokens + [self.id_end]
            chunk.multipliers = chunk.multipliers + [1.0]

            if self.min_padding > 0:
                chunk.tokens += [self.id_pad] * self.min_padding
                chunk.multipliers += [1.0] * self.min_padding

            current_chunk_length = len(chunk.tokens)

            token_count += current_chunk_length
            remaining_count = self.min_length - current_chunk_length

            if self.min_length > 0 and remaining_count > 0:
                chunk.tokens += [self.id_pad] * remaining_count
                chunk.multipliers += [1.0] * remaining_count

            chunks.append(chunk)
            chunk = PromptChunk()

        for tokens, (text, weight) in zip(tokenized, parsed):
            if text == "BREAK" and weight == -1:
                next_chunk()
                continue

            position = 0
            while position < len(tokens):
                token = tokens[position]
                chunk.tokens.append(token)
                chunk.multipliers.append(weight)
                position += 1

        if chunk.tokens or not chunks:
            next_chunk()

        return chunks, token_count

    def __call__(self, texts):
        self.emphasis = emphasis.get_current_option(opts.emphasis)()
        if any(emphasis.uses_emphasis(x) for x in texts):
            dynamic_args.last_extra_generation_params["Emphasis"] = self.emphasis.name

        zs = []
        cache = {}

        batch_chunks = {}
        max_tokens = 0
        max_chunks = 0
        for line in texts:
            if line not in batch_chunks:
                chunks, _ = self.tokenize_line(line)
                batch_chunks[line] = chunks
                max_tokens = max(max_tokens, *(len(chunk.tokens) for chunk in chunks))
                max_chunks = max(max_chunks, len(chunks))
        empty_chunk = self.tokenize_line("")[0][0] if max_chunks > 1 else None

        for line in texts:
            if line in cache:
                line_z_values = cache[line]
            else:
                chunks = batch_chunks[line]
                if len(chunks) < max_chunks:
                    chunks = chunks + [empty_chunk] * (max_chunks - len(chunks))
                line_z_values = []

                for chunk in chunks:
                    tokens = chunk.tokens
                    multipliers = chunk.multipliers

                    remaining_count = max_tokens - len(tokens)
                    if remaining_count > 0:
                        tokens += [self.id_pad] * remaining_count
                        multipliers += [1.0] * remaining_count

                    z = self.process_tokens([tokens], [multipliers])[0]
                    line_z_values.append(z)
                line_z_values = torch.cat(line_z_values, dim=0)
                cache[line] = line_z_values

            zs.append(line_z_values)

        return torch.stack(zs)

    def process_tokens(self, batch_tokens, batch_multipliers):
        tokens = torch.asarray(batch_tokens)

        z = self.encode_with_transformers(tokens)

        self.emphasis.tokens = batch_tokens
        self.emphasis.multipliers = torch.asarray(batch_multipliers).to(z)
        self.emphasis.z = z
        self.emphasis.after_transformers()
        z = self.emphasis.z

        return z
