---
language:
  - en
  - ja
license: other
license_name: qwen-research
license_link: https://huggingface.co/Aikimi/Forge-Neo-Image-2.1-Turbo-W4A8/blob/main/LICENSE
base_model: Qwen/Qwen-Image-2.1-Turbo
base_model_relation: quantized
pipeline_tag: text-to-image
tags:
  - qwen-image-2.1
  - image-editing
  - quantization
  - w4a8
  - forge-neo
---

# Forge Neo Image 2.1 Turbo W4A8

**Built with Qwen.** This is an unofficial W4A8 conversion of [Qwen Image 2.1 Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) for [Aikimi Forge Neo](https://github.com/AiWithYou/aikimi-forge-neo). Qwen supplies the source weights; Aikimi supplies this quantized conversion. No additional training was performed. Read [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md): use and redistribution are limited to non-commercial research or evaluation under the Qwen Research License.

## Contents and provenance

This repository contains the quantized generation Transformer and shared Qwen3-VL text encoder, plus an integrity manifest and importer. It is not a complete image pipeline. The VAE, processor, official sampling configuration, and original BF16 model files are supplied by Neo's setup.

- Transformer source: Qwen/Qwen-Image-2.1-Turbo revision d65dbc9a7e8f6b5479e33dee6030eaab2a906509.
- Shared encoder source: Qwen/Qwen-Image-2.1 revision b3179ad355be050328e483a9dfdd9e60cd62adfa. Its source tensor values match the encoder bundled with the official Turbo checkpoint.
- Sampling: the official eight sigma values, CFG 1, and FlowMatchEulerDiscreteScheduler with dynamic shifting disabled.

The files use Forge Neo's custom Comfy Kitchen AsymW4A8Int8Layout representation: packed 4-bit weights, 8-bit activation computation, group size 16, ConvRot size 256, and FP8 group scales. Packed tensors and their w4a8.json maps require Neo's W4A8 loader and matching CUDA kernels; these are not a generic Diffusers from_pretrained repository.

The conversion changes the selected Linear weights in transformer/ and text_encoder/. Config files are sanitized to remove machine-local paths. Each safetensors header and JSON component file carries a modification notice identifying Aikimi's conversion and its source revision. Adding these notices preserves every tensor descriptor, offset, and raw tensor byte from the saved conversion. The individual converted files, original source revisions and SHA-256 values, conversion recipes, runtime versions, and exported file hashes are listed in release_manifest.json. The original local complete.json and cache paths are not distributed.

## Install in Forge Neo

Neo currently requires its pinned official BF16 source files to remain installed. These files avoid GPU quantization before inference; they do not remove the source-model download or replace the original weights.

Close Neo and run from the Neo checkout root in PowerShell:

```powershell
.\aikimi-qwen-image21-setup.bat --official-turbo-only
& .\models\Qwen-Image-2.1\worker-env\Scripts\hf.exe download Aikimi/Forge-Neo-Image-2.1-Turbo-W4A8 --local-dir .\work\hf-models\official-turbo-w4a8
& .\models\Qwen-Image-2.1\worker-env\Scripts\python.exe -X utf8 tools\qwen21_hub_release.py install --precision turbo_official_w4a8 --release-dir .\work\hf-models\official-turbo-w4a8
```

Select **公式Turbo · W4A8 · 8 steps** in Qwen Image 2.1 Studio. The importer checks both components' source identities, recipes and runtime versions, then verifies each component's payload hashes before installing that component. It creates a local cache manifest bound to the receiving machine. An existing valid component is verified and retained. If runtime versions differ, prepare a conversion locally in that environment.

CUDA with BF16 support is required. The versions and quantization recipe recorded in the manifest are part of this release's format. Neo's isolated runtime provides its required loader and CPU-offload handling.

## Verification

The source W4A8 components used for this export passed CUDA text-to-image and cached reload runs at 768 by 768. Both runs recorded the eight official timesteps and produced identical pixel hashes. All saved component files passed SHA-256 verification before export. Reference editing and Outpaint were not included in this W4A8 check.

Tested on an NVIDIA RTX 3090 with 24 GB VRAM, driver 610.74, and PyTorch 2.13.0+cu130. The complete runtime versions are recorded in the manifest. These checks cover the tested settings, not general quality parity with BF16.

## Limits

Quantization changes numeric values and can affect composition, text, colors and details. This release does not claim pixel or quality parity with BF16. Additional LoRA, ControlNet, Outpaint and Sparse combinations have their own limits in [Neo's Turbo guide](https://github.com/AiWithYou/aikimi-forge-neo/blob/neo/docs/qwen21-official-turbo.md).

**日本語:** Neo向けの非公式量子化配布です。公式Turbo本体・共通部品・専用環境を先に導入し、保存済みTransformerとテキストエンコーダーを取り込むと、推論前のGPU変換を省けます。元BF16の保存は必要です。研究・評価用途のライセンスと、manifestに記録した専用環境の条件を確認してください。
