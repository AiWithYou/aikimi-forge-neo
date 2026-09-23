---
language:
  - en
  - ja
license: other
license_name: qwen-research
license_link: https://huggingface.co/Aikimi/Neo-Image-2.1-INT8/blob/main/LICENSE
base_model: Qwen/Qwen-Image-2.1
base_model_relation: quantized
pipeline_tag: text-to-image
tags:
  - qwen-image-2.1
  - image-editing
  - quantization
  - int8
  - bitsandbytes
  - forge-neo
---

# Neo Image 2.1 INT8

**Built with Qwen.** This is an unofficial, weight-only INT8 conversion of [Qwen Image 2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) for [Aikimi Studio Neo / Forge Neo](https://github.com/AiWithYou/aikimi-forge-neo). It is not trained, approved, or released by the Qwen team. The original weights and this conversion are subject to the [Qwen Research License](LICENSE); use and redistribution are limited to non-commercial research or evaluation under that license. Read [NOTICE.md](NOTICE.md) for the required attribution and change notice.

## Contents

This repository contains a 7.26 GB diffusion transformer and a 10.02 GB Qwen3-VL text encoder, converted from upstream revision `b3179ad355be050328e483a9dfdd9e60cd62adfa`. The transformer has 224 bitsandbytes INT8 linear layers; the text encoder has 368. Selected transformer input, modulation, output, and normalization modules retain their source precision. The saved weights use the Diffusers and Transformers `save_pretrained()` formats, with the quantization configuration recorded in each component.

`transformer/` and `text_encoder/` contain the saved tensors and configs. `release_manifest.json` records source revision, conversion recipe, exact runtime versions, file sizes, and SHA-256 hashes. The VAE, processor, scheduler, and unconverted source files are not included.

## Use in Forge Neo

The current Neo loader needs the pinned official BF16 model installed first. This release avoids the local INT8 conversion step; it does **not** currently remove the need to download and retain the official BF16 files. Install Neo's Qwen environment with `aikimi-qwen-image21-setup.bat`, then run from the Neo root in PowerShell:

```powershell
& .\venv\Scripts\hf.exe download Aikimi/Neo-Image-2.1-INT8 --local-dir .\tmp\neo-image21-int8
& .\models\Qwen-Image-2.1\worker-env\Scripts\python.exe -X utf8 .\tmp\neo-image21-int8\install.py install --precision int8 --neo-root .
```

Select **INT8** in the Qwen Image 2.1 tab. Neo checks the source revision and runtime versions before installing this release and verifies every downloaded file against `release_manifest.json`. A subsequent load should report `quantized_cache` status `hit` for both components. The installer does not replace an existing saved model. The Diffusers pipeline still needs Neo's isolated runtime and its CPU-offload fix.

## Verification and limits

Both saved INT8 components were reloaded after model release without another conversion. A new-image comparison with W4A8 is [documented with images and measurements](https://github.com/AiWithYou/aikimi-forge-neo/blob/neo/docs/assets/qwen-image21-v1.3.0/README.md); it is one trial, not a general speed or image-quality guarantee. [Persistence validation](https://github.com/AiWithYou/aikimi-forge-neo/blob/neo/docs/optimization-persistence-2026-09-22.md) and [conversion details](https://github.com/AiWithYou/aikimi-forge-neo/blob/neo/docs/w4a8.md) are available in Neo. Quantization can change composition and fine details. CUDA with BF16 support and the versions recorded in the manifest are required by this release's importer.

**日本語:** Neo向けの非公式変換です。公式のQwen Image 2.1一式を先に導入してから、上記の2コマンドで変換済み重みを取り込みます。再量子化を省けますが、現行のNeoでは元のBF16ファイルも必要です。非商用の研究・評価用途に限る元ライセンスと[変更・帰属表示](NOTICE.md)を確認してください。
