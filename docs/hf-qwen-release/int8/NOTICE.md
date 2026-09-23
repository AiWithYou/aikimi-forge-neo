# Attribution and modification notice

Built with Qwen.

Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026 Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights Reserved.

This is an unofficial conversion of `Qwen/Qwen-Image-2.1` revision `b3179ad355be050328e483a9dfdd9e60cd62adfa` by Aikimi. The diffusion transformer and the Qwen3-VL text encoder have selected linear weights converted to bitsandbytes INT8. Other weights remain at their source precision. The saved transformer's `config.json` was also changed to replace a machine-local `_name_or_path` with the upstream model ID. No additional training was performed. The conversion does not imply endorsement by the Qwen team.

The original model and its license: https://huggingface.co/Qwen/Qwen-Image-2.1

The conversion implementation: https://github.com/AiWithYou/aikimi-forge-neo/blob/neo/tools/qwen_image21_worker.py
