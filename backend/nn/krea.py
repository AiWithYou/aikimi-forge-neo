# https://github.com/Comfy-Org/ComfyUI/blob/v0.26.1/comfy/ldm/krea2/model.py
# https://github.com/lbouaraba/comfyui-krea2edit/blob/main/__init__.py

from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from backend.args import dynamic_args
from backend.attention import attention_function
from backend.memory_management import cast_to
from backend.misc.image_resize import adaptive_resize
from backend.nn.flux import EmbedND, timestep_embedding
from backend.quant_ops import ck
from backend.utils import pad_to_patch_size


class _TextConditioningCache:
    def __init__(self, max_bytes, weight_revision):
        self.max_bytes = max_bytes
        self.weight_revision = weight_revision
        self.hits = 0
        self.misses = 0
        self.bypasses = 0
        self.entries = []
        self.revision = None
        self.parameters = ()

    @property
    def retained_bytes(self):
        return sum(tensor.numel() * tensor.element_size() for entry in self.entries for tensor in entry)

    def clear(self):
        self.entries.clear()
        self.revision = None
        self.parameters = ()


def _imgids(bs: int, frame: int, h_: int, w_: int, device: torch.device) -> torch.Tensor:
    ids = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
    ids[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
    return ids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)


def _repeat_reference_to_batch(
    reference: torch.Tensor, batch_size: int
) -> torch.Tensor:
    reference_batch = int(reference.shape[0])
    if reference_batch <= 0 or batch_size <= 0:
        raise ValueError(
            "Krea2 reference and sampling batches must be positive; "
            f"got reference batch {reference_batch} and sampling batch {batch_size}"
        )
    if reference_batch == batch_size:
        return reference
    if batch_size % reference_batch != 0:
        raise ValueError(
            "Krea2 reference batch must divide the sampling batch; "
            f"got reference batch {reference_batch} and sampling batch {batch_size}"
        )
    repeats = [batch_size // reference_batch] + [1] * (reference.dim() - 1)
    return reference.repeat(*repeats)


def _validate_context_shape(
    context: torch.Tensor,
    batch_size: int,
    text_layers: int,
    text_dim: int,
) -> torch.Tensor:
    expected_tail = (text_layers, text_dim)
    if (
        context.ndim != 4
        or tuple(context.shape[-2:]) != expected_tail
        or int(context.shape[0]) != batch_size
    ):
        raise ValueError(
            "Krea2 expects conditioning shaped "
            f"[batch, sequence, {text_layers}, {text_dim}] from the "
            f"Qwen3-VL text encoder; got {tuple(context.shape)} for batch "
            f"{batch_size}. Load the text encoder with the Krea2 model "
            "configuration."
        )
    return context


class RMSNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.scale = nn.Parameter(torch.empty(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        weight = cast_to(self.scale, dtype=torch.float32, device=x.device) + 1.0
        return F.rms_norm(x.float(), (x.shape[-1],), weight=weight, eps=self.eps).to(dtype)


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.qnorm = RMSNorm(dim)
        self.knorm = RMSNorm(dim)

    def forward(self, q, k):
        return self.qnorm(q), self.knorm(k)


class SwiGLU(nn.Module):
    def __init__(self, features: int, multiplier: int, bias: bool = False, multiple: int = 128):
        super().__init__()

        mlpdim = int(2 * features / 3) * multiplier
        mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)
        self.gate = nn.Linear(features, mlpdim, bias=bias)
        self.up = nn.Linear(features, mlpdim, bias=bias)
        self.down = nn.Linear(mlpdim, features, bias=bias)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)).mul_(self.up(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, kvheads: Optional[int] = None, bias: bool = False):
        super().__init__()

        self.heads = heads
        self.kvheads = kvheads if kvheads is not None else heads
        self.headdim = dim // self.heads
        self.wq = nn.Linear(dim, self.headdim * self.heads, bias=bias)
        self.wk = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.wv = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.gate = nn.Linear(dim, dim, bias=bias)
        self.qknorm = QKNorm(self.headdim)
        self.wo = nn.Linear(dim, dim, bias=bias)

    def forward(self, x, freqs=None, mask=None, transformer_options={}):
        q, k, v, gate = self.wq(x), self.wk(x), self.wv(x), self.gate(x)
        q = rearrange(q, "B L (H D) -> B H L D", H=self.heads)
        k = rearrange(k, "B L (H D) -> B H L D", H=self.kvheads)
        v = rearrange(v, "B L (H D) -> B H L D", H=self.kvheads)
        q, k = self.qknorm(q, k)
        if freqs is not None:
            q, k = ck.apply_rope(q, k, freqs)
        if self.kvheads != self.heads:
            rep = self.heads // self.kvheads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        override = transformer_options.get("krea2_attention_override")
        if override is not None and "krea2_block_index" in transformer_options:
            out = override(q, k, v, self.heads, mask, transformer_options, attention_function)
        else:
            out = attention_function(q, k, v, self.heads, mask=mask, skip_reshape=True, transformer_options=transformer_options)
        return self.wo(out * F.sigmoid(gate))


class SimpleModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.lin = nn.Parameter(torch.empty(2, dim))

    def forward(self, vec):
        out = vec + cast_to(self.lin, dtype=vec.dtype, device=vec.device).unsqueeze(0)
        scale, shift = out.chunk(2, dim=1)
        return scale, shift


class DoubleSharedModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.lin = nn.Parameter(torch.empty(6 * dim))

    def forward(self, vec):
        out = vec + cast_to(self.lin, dtype=vec.dtype, device=vec.device)
        return out.chunk(6, dim=-1)


class TextFusionBlock(nn.Module):
    def __init__(self, features, heads, multiplier, bias=False, kvheads=None):
        super().__init__()

        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(features, heads, kvheads=kvheads, bias=bias)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x, mask=None, transformer_options={}):
        x.add_(self.attn(self.prenorm(x), mask=mask, transformer_options=transformer_options))
        x.add_(self.mlp(self.postnorm(x)))
        return x


class TextFusionTransformer(nn.Module):
    def __init__(self, num_txt_layers, txt_dim, heads, multiplier, bias=False, kvheads=None):
        super().__init__()

        self.layerwise_blocks = nn.ModuleList([TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)])
        self.projector = nn.Linear(num_txt_layers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList([TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)])

    def forward(self, x, mask=None, transformer_options={}):
        b, l, n, d = x.shape
        x = x.reshape(b * l, n, d)
        for block in self.layerwise_blocks:
            x = block(x.contiguous(), mask=None, transformer_options=transformer_options)
        x = rearrange(x, "(b l) n d -> b l d n", b=b, l=l)
        x = self.projector(x).squeeze(-1)
        for block in self.refiner_blocks:
            x = block(x, mask=mask, transformer_options=transformer_options)
        return x


class SingleStreamBlock(nn.Module):
    def __init__(self, features, heads, multiplier, bias=False, kvheads=None):
        super().__init__()

        self.mod = DoubleSharedModulation(features)
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(features, heads, kvheads=kvheads, bias=bias)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x, vec, freqs, mask=None, transformer_options={}):
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        x.addcmul_(pregate, self.attn(torch.addcmul(preshift, 1 + prescale, self.prenorm(x)), freqs, mask, transformer_options=transformer_options))
        x.addcmul_(postgate, self.mlp(torch.addcmul(postshift, 1 + postscale, self.postnorm(x))))
        return x


class LastLayer(nn.Module):
    def __init__(self, features, patch, channels):
        super().__init__()

        self.norm = RMSNorm(features)
        self.linear = nn.Linear(features, patch * patch * channels, bias=True)
        self.modulation = SimpleModulation(features)

    def forward(self, x, tvec):
        scale, shift = self.modulation(tvec)
        x = torch.addcmul(shift, 1 + scale, self.norm(x))
        return self.linear(x)


class SingleStreamDiT(nn.Module):
    def __init__(
        self,
        features=6144,
        tdim=256,
        txtdim=2560,
        heads=48,
        kvheads=12,
        multiplier=4,
        layers=28,
        patch=2,
        channels=16,
        bias=False,
        theta=1e3,
        txtlayers=12,
        txtheads=20,
        txtkvheads=20,
        **kwargs,
    ):
        super().__init__()

        self.patch = patch
        self.channels = channels
        self.tdim = tdim
        self.heads = heads
        self.txtdim = txtdim
        self.txtlayers = txtlayers
        self._text_conditioning_cache = None

        headdim = features // heads
        axes = [headdim - 12 * (headdim // 16), 6 * (headdim // 16), 6 * (headdim // 16)]
        assert sum(axes) == headdim, f"axes {axes} sum != headdim {headdim}"
        self.pe_embedder = EmbedND(dim=headdim, theta=int(theta), axes_dim=axes)

        self.first = nn.Linear(channels * patch**2, features, bias=True)
        self.blocks = nn.ModuleList([SingleStreamBlock(features, heads, multiplier, bias, kvheads) for _ in range(layers)])
        self.tmlp = nn.Sequential(
            nn.Linear(tdim, features),
            nn.GELU(approximate="tanh"),
            nn.Linear(features, features),
        )
        self.txtfusion = TextFusionTransformer(txtlayers, txtdim, txtheads, multiplier, bias, txtkvheads)
        self.txtmlp = nn.Sequential(
            RMSNorm(txtdim),
            nn.Linear(txtdim, features),
            nn.GELU(approximate="tanh"),
            nn.Linear(features, features),
        )
        self.last = LastLayer(features, patch, channels)
        self.tproj = nn.Sequential(
            nn.GELU(approximate="tanh"),
            nn.Linear(features, features * 6),
        )

    @contextmanager
    def text_conditioning_cache(self, max_bytes=64 * 1024 * 1024, *, weight_revision=None):
        previous = self._text_conditioning_cache
        if previous is not None:
            previous.clear()
        cache = _TextConditioningCache(max_bytes, weight_revision)
        self._text_conditioning_cache = cache
        try:
            yield cache
        finally:
            cache.clear()
            self._text_conditioning_cache = previous

    def _fused_text_context(self, context, transformer_options):
        def compute():
            fused = self.txtfusion(context, mask=None, transformer_options=transformer_options)
            return self.txtmlp(fused)

        cache = self._text_conditioning_cache
        if cache is None:
            return compute()

        modules = (*self.txtfusion.modules(), *self.txtmlp.modules())
        custom_text_attention = (
            transformer_options.get("krea2_attention_override") is not None
            and "krea2_block_index" in transformer_options
        )
        model_revision = cache.weight_revision() if cache.weight_revision is not None else ()
        if model_revision is None or self.training or torch.is_grad_enabled() or custom_text_attention or any(
            module.training or module._forward_hooks or module._forward_pre_hooks
            or getattr(module, "weight_function", ()) or getattr(module, "bias_function", ())
            for module in modules
        ):
            cache.clear()
            cache.bypasses += 1
            return compute()

        parameters = (*self.txtfusion.parameters(), *self.txtmlp.parameters())
        try:
            versions = tuple(parameter._version for parameter in parameters)
        except RuntimeError:
            # Forge constructs inference tensors; only its patch revision can
            # authorize reuse when those tensors have no mutation counters.
            if cache.weight_revision is None:
                cache.clear()
                cache.bypasses += 1
                return compute()
            versions = None
        revision = (
            model_revision, id(attention_function), versions,
            tuple((id(module), id(module.__dict__.get("forward", type(module).forward))) for module in modules),
            tuple((id(parameter), parameter.device, parameter.dtype) for parameter in parameters),
        )
        if cache.revision != revision:
            cache.clear()
            cache.revision = revision
            cache.parameters = parameters

        for index, (original, fused) in enumerate(cache.entries):
            if (
                original.shape == context.shape and original.dtype == context.dtype
                and original.device == context.device and torch.equal(original, context)
            ):
                cache.entries.append(cache.entries.pop(index))
                cache.hits += 1
                return fused

        estimated_bytes = context.numel() * context.element_size()
        estimated_bytes += context.shape[0] * context.shape[1] * self.first.out_features * context.element_size()
        if estimated_bytes > cache.max_bytes:
            cache.bypasses += 1
            return compute()
        while cache.entries and (len(cache.entries) >= 2 or cache.retained_bytes + estimated_bytes > cache.max_bytes):
            cache.entries.pop(0)

        # TextFusionBlock mutates its input, so compare against a separate copy.
        original = context.detach().clone()
        cache.misses += 1
        fused = compute()
        actual_bytes = original.numel() * original.element_size() + fused.numel() * fused.element_size()
        if actual_bytes <= cache.max_bytes:
            while cache.entries and cache.retained_bytes + actual_bytes > cache.max_bytes:
                cache.entries.pop(0)
            cache.entries.append((original, fused.detach()))
        return fused

    def forward(self, x, timesteps, context, attention_mask=None, transformer_options={}, **kwargs):
        x = x.squeeze(2)
        bs, c, H_orig, W_orig = x.shape
        patch = self.patch

        x = pad_to_patch_size(x, (patch, patch))
        H, W = x.shape[-2], x.shape[-1]
        h_, w_ = H // patch, W // patch

        ref_latents: list[torch.Tensor] = dynamic_args.ref_latents

        if (_edit := bool(ref_latents)) is True:
            refs = []
            ref_grids = []
            for ref in ref_latents:
                ref = _repeat_reference_to_batch(ref, int(x.shape[0]))
                if ref.shape != x.shape:
                    ref = adaptive_resize(ref, W, H, "area", "center")
                refs.append(pad_to_patch_size(ref.to(x), (patch, patch), padding_mode="replicate"))
                ref_grids.append((ref.shape[-2] // patch, ref.shape[-1] // patch))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
        img = self.first(img)

        if _edit:
            refs = [rearrange(r, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch) for r in refs]
            refs = [self.first(r) for r in refs]

        t = self.tmlp(timestep_embedding(timesteps, self.tdim).unsqueeze(1).to(img.dtype))
        tvec = self.tproj(t)

        context = _validate_context_shape(
            context,
            batch_size=bs,
            text_layers=self.txtlayers,
            text_dim=self.txtdim,
        )

        context = self._fused_text_context(context, transformer_options)

        txtlen, imglen = context.shape[1], img.shape[1]
        if _edit:
            reflen = sum(ref.shape[1] for ref in refs)
            combined = torch.cat((context, *refs, img), dim=1)
        else:
            reflen = 0
            combined = torch.cat((context, img), dim=1)

        device = combined.device
        txtpos = torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)

        if _edit:
            refids = [_imgids(bs, i + 1, gh, gw, device) for i, (gh, gw) in enumerate(ref_grids)]
            imgids = _imgids(bs, 0, h_, w_, device)
            pos = torch.cat((txtpos, *refids, imgids), dim=1)
        else:
            imgids = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
            imgids[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
            imgids[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
            imgpos = imgids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)
            pos = torch.cat((txtpos, imgpos), dim=1)

        freqs = self.pe_embedder(pos)

        block_options = transformer_options
        if transformer_options.get("krea2_attention_override") is not None:
            block_options = {
                **transformer_options,
                "krea2_text_tokens": txtlen,
                "krea2_reference_tokens": reflen,
                "krea2_image_tokens": imglen,
            }
        for index, block in enumerate(self.blocks):
            if block_options is not transformer_options:
                block_options["krea2_block_index"] = index
            combined = block(combined, tvec, freqs, None, transformer_options=block_options)

        final = self.last(combined, t)
        out = final[:, txtlen + reflen : txtlen + reflen + imglen, :]
        out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_, w=w_, ph=patch, pw=patch, c=self.channels)
        out = out[:, :, :H_orig, :W_orig]

        return out.unsqueeze(2)
