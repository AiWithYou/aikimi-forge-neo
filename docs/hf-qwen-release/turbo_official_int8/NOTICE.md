Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026 Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights Reserved.

Built with Qwen.

MODIFICATION NOTICE: The files in transformer/ and text_encoder/ are an unofficial INT8 conversion by Aikimi. The generation Transformer derives from Qwen/Qwen-Image-2.1-Turbo revision d65dbc9a7e8f6b5479e33dee6030eaab2a906509; the shared encoder derives from Qwen/Qwen-Image-2.1 revision b3179ad355be050328e483a9dfdd9e60cd62adfa. Selected Linear weights were quantized; other source weights retain their precision. Config metadata was sanitized to replace machine-local paths with the corresponding source model ID. No additional training was performed. See release_manifest.json for the individual modified files, source hashes, quantization recipe and exported hashes. This conversion does not imply endorsement by Qwen.

Source model and license: https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/d65dbc9a7e8f6b5479e33dee6030eaab2a906509/LICENSE

Every component safetensors header and JSON file carries its own modification notice. Export adds these notices without changing tensor descriptors, offsets, or raw tensor bytes from the saved quantized components.
Conversion implementation: https://github.com/AiWithYou/aikimi-forge-neo/blob/neo/tools/qwen_image21_worker.py
