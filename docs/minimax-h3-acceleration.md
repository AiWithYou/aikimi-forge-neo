# MiniMax H3: optional acceleration

Verified against upstream sources on 2026-09-06. These controls are in **H3 Studio > 高速化**. They do not change the resolution presets or silently enable other optimizations.

## Acceleration and conditioning reuse

| Axis | Default | Opt-in alternative | Trade-off |
| --- | --- | --- | --- |
| Diffusion model | Mode-specific official INT8 ConvRot weights | MATLOWAI Fused Turbo | Baked-in Turbo and Mystic change the model and its visual/audio behavior. |
| Video VAE | FP16 | Kijai INT8 ConvRot | Quantization can change pixels and depends on GPU compatibility. |
| Video decoding | VAEDecode | MiniMax H3 Fast VAE Decode | Batched spatial tiles use more VRAM and can be slower. |
| Attention | Comfy Kitchen dense | Sol-Attn or SLA via official BlockSparseAttention | Approximate attention can change motion, detail and audio. |
| CLIP conditioning | Native encoder on each execution | CLIPCached auto / refresh | Stores conditioning and reference metadata on disk; repeated inputs can avoid the encoder. Incompatible with NegPiP. |

See the [CLIPCached integration guide](minimax-h3-clipcache.md) for the pinned installer, cache modes, storage and validation. The managed runtime only permits its selected node pack. The default graph remains unchanged when the cache is off.

Changing the model dropdown does **not** change Steps. The **Turbo + 4 Steps** and **Turbo + 8 Steps** buttons explicitly select the fused model, the named step count and `simple`. They retain the chosen VAE, attention and CLIP cache. The sampler remains `res_multistep` with `BasicGuider` (no CFG); the model's default video/audio shifts remain 12/3. The standard reset button restores all acceleration controls, including CLIP cache and NegPiP, to their defaults with 20 Steps and `simple`, without overwriting the prompt, aspect, resolution or duration.

Sparse attention uses the official DynamicCombo API, applies after Kitchen attention, keeps `extra_tokens=256`, `min_tokens=12288`, and protects conditioning/audio rows with `exact_kv_and_rows`. The default first 20% of the schedule and short sequences stay dense. The node's verbose log indicates why a run stays dense. Sol-Attn exposes tau; SLA exposes the exact-key-block keep percentage. VSA is deliberately not offered: it requires appropriate trained weights, which are not these defaults.

**SLA here is not the custom H3SLA implementation in the MATLOWAI example. Matching a keep percentage does not reproduce that recipe.** No author timing is a guarantee for an RTX 3090 or any other GPU.

## Install only what you select

Use the selected **ComfyUI runtime**, not Forge's Python environment. Model selection never downloads weights automatically. Standard settings still require only their original files/nodes; optional files are checked only when selected.

### INT8 Video VAE

Source: [Kijai/MiniMax-H3-experimental](https://huggingface.co/Kijai/MiniMax-H3-experimental/blob/7f5705937cc106963a9dd77c322f7631e3610e89/minimax_h3_video_vae_int8_convrot.safetensors).

Place `minimax_h3_video_vae_int8_convrot.safetensors` in `ComfyUI/models/vae/`. The original FP16 VAE is not additionally required for this selection. Keep FP16 available for A/B testing and recovery from black frames or numerical artifacts. Native support was merged in [ComfyUI PR #15334](https://github.com/Comfy-Org/ComfyUI/pull/15334), before the existing August 11 minimum core revision.

### MATLOWAI Fused Turbo

Source and model recipe: [model card at e6df1685](https://huggingface.co/MATLOWAI/minimax-h3-fused-turbo-int8-convrot/blob/e6df16857722fdd9b79f6ed6b63251119202a91f/README.md).

Place `minimax_h3_fused_refdelta_r1024_turbo8_mystic07_int8_convrot.safetensors` in `ComfyUI/models/diffusion_models/`. The source repository stores it under `diffusion_models/`. Select **MATLOWAI Fused Turbo** and explicitly apply 4 or 8 Steps. Its baked-in LoRAs are not added again at runtime. The same fused checkpoint is selected for the image/text and reference workflows; unused official FL2VA/Ref2VA files are not required by this choice.

This is a third-party derivative, not a new official MiniMax base release. Its baked-in style/acceleration weights cannot be turned off individually. Switching back to the standard model selects the original mode-specific weight files.

### Fast VAE Decode

Install [starsFriday/ComfyUI-MiniMax-H3-MotionCache](https://github.com/starsFriday/ComfyUI-MiniMax-H3-MotionCache) under the exact folder name `ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-MotionCache`, then select Fast VAE and use **実行環境とモデル > 選択設定で再起動**.

The managed command retains `--disable-all-custom-nodes --disable-api-nodes` and adds only `--whitelist-custom-nodes ComfyUI-MiniMax-H3-MotionCache`. Unknown packs, extra whitelist entries and manager activation remain rejected. The pack registers MotionCache as well, but this integration does **not** add a MotionCache node to the workflow. Nothing is installed or executed from the Internet by the GUI.

Switching back to standard decoding also requires a runtime restart to remove this exception. Externally launched processes are never automatically stopped. The existing empty-queue and in-flight-result checks still apply.

The inspected `fast_vae_decode.py` Git blob is `a3f5fb9cded5c30eacdbf9b6f890dadcda0f2805`. The node takes `samples`, `vae`, and integer `tile_batch_size` (1-8). Its **own** OOM handler retries batch 1 and logs the retry. Forge does not silently substitute another decoder. The author's controlled example was slower than standard decoding, so benchmark both on the actual GPU.

### Sparse Attention

Update ComfyUI to include [e308cc73 / PR #16072](https://github.com/Comfy-Org/ComfyUI/commit/e308cc73b466584b0c17be695e5de1a17438bb40), update that runtime's requirements, then restart it. That revision pins `comfy-kitchen==0.2.33`. The bridge requires at least 0.2.33 when sparse attention is selected and verifies the actual node input schema and core ancestry before submitting.

It does not remove the original core gate for standard runs or silently turn off a requested optimization. Missing models, nodes, incompatible schemas, old Kitchen versions and mismatched custom-node permissions produce actionable errors before generation.

## Reproducibility and limits

Generation JSON schema 2 records the selected acceleration settings, model filenames, decoder, requested attention backend, runtime profile, and ComfyUI/Kitchen versions. History restoration restores these settings but still clears all input media. Schema 1 histories without acceleration settings restore the standard defaults. Malformed settings are rejected rather than treated as defaults.

Recorded acceleration values are **requested settings**, not measurements of effective sparsity or the Fast VAE's final OOM fallback batch. Use backend logs for those details. Compare the same prompt, input media, seed, size, duration and sampler; compare motion, fine detail, speech and synchronized audio as well as elapsed time and peak memory.

No model weights or third-party implementation code are vendored. Follow the source models' and node pack's license terms. GPU output quality and end-to-end speed have not been measured by this integration's CPU/Gradio tests. Context-IR, 2K Regenerate, Fun Union and arbitrary-time guides are not added by this change; no paid API is invoked.
