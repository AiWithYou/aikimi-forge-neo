---
language:
  - en
  - ja
license: other
license_name: qwen-research
license_link: https://huggingface.co/Aikimi/Neo-Image-2.1-W4A8/blob/main/LICENSE
base_model: Qwen/Qwen-Image-2.1
base_model_relation: quantized
pipeline_tag: text-to-image
tags:
  - qwen-image-2.1
  - image-editing
  - quantization
  - w4a8
  - convrot
  - forge-neo
---

# Neo Image 2.1 W4A8

**Built with Qwen.** This is an unofficial, weight-only W4A8 conversion of [Qwen Image 2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) for [Aikimi Studio Neo / Forge Neo](https://github.com/AiWithYou/aikimi-forge-neo). It is not trained, approved, or released by the Qwen team. The original weights and this conversion are subject to the [Qwen Research License](LICENSE); use and redistribution are limited to non-commercial research or evaluation under that license. Read [NOTICE.md](NOTICE.md) for the required attribution and change notice.

## Contents

This repository contains a 4.20 GB diffusion transformer and a 7.56 GB Qwen3-VL text encoder, both converted from upstream revision `b3179ad355be050328e483a9dfdd9e60cd62adfa`. The transformer has 224 W4A8 linear layers; the language component has 252. The remaining layers retain their source precision. The quantizer uses Comfy Kitchen's `AsymW4A8Int8Layout`: 4-bit packed weights, 8-bit activation computation, group size 16, ConvRot group size 256, and FP8 group scales. It requires the matching CUDA kernels.

`transformer/` and `text_encoder/` contain the saved packed tensors and their `w4a8.json` maps. `release_manifest.json` records source revision, conversion recipe, exact runtime versions, file sizes, and SHA-256 hashes. The VAE, processor, scheduler, and unconverted source files are not included.

## Use in Forge Neo

The current Neo loader needs the pinned official BF16 model installed first. This release avoids the local W4A8 conversion step; it does **not** currently remove the need to download and retain the official BF16 files. Install Neo's Qwen environment with `aikimi-qwen-image21-setup.bat`, then run from the Neo root in PowerShell:

```powershell
& .\venv\Scripts\hf.exe download Aikimi/Neo-Image-2.1-W4A8 --local-dir .\tmp\neo-image21-w4a8
& .\models\Qwen-Image-2.1\worker-env\Scripts\python.exe -X utf8 .\tmp\neo-image21-w4a8\install.py install --precision w4a8 --neo-root .
```

Select **W4A8** in the Qwen Image 2.1 tab. Neo checks the source revision and runtime versions before installing this release and verifies every downloaded file against `release_manifest.json`. A subsequent load should report `quantized_cache` status `hit` for both components. The installer does not replace an existing saved model. These files are not a standard Diffusers `from_pretrained()` repository; the W4A8 loader lives in Neo.

## Verification and limits

The saved W4A8 components were reloaded in another process and used for an actual 1024×1024 image edit on an RTX 3090. A separate new-image comparison with INT8 is [documented with images and measurements](https://github.com/AiWithYou/aikimi-forge-neo/blob/neo/docs/assets/qwen-image21-v1.3.0/README.md); it is one trial, not a general speed or image-quality guarantee. [Persistence and edit validation](https://github.com/AiWithYou/aikimi-forge-neo/blob/neo/docs/optimization-persistence-2026-09-22.md) and the [W4A8 implementation details](https://github.com/AiWithYou/aikimi-forge-neo/blob/neo/docs/w4a8.md) are available in Neo. Quantization can change composition and fine details. CUDA with BF16 support and the versions recorded in the manifest are required by this release's importer.

**日本語:** Neo向けの非公式変換です。公式のQwen Image 2.1一式を先に導入してから、上記の2コマンドで変換済み重みを取り込みます。再量子化を省けますが、現行のNeoでは元のBF16ファイルも必要です。非商用の研究・評価用途に限る元ライセンスと[変更・帰属表示](NOTICE.md)を確認してください。
