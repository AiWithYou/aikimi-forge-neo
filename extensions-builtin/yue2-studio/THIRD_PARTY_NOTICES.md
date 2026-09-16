# YuE2 Music · Third-party notices

Checked 2026-09-16. No model weights, native binaries, or third-party UI assets are bundled by this extension.

## YuE2

- Authors: Multimodal Art Projection / YuE contributors.
- Source: https://github.com/multimodal-art-projection/YuE
- Reviewed source commit: `4d53bd5fc7e96a53cb907d3eb407a65df67a8b79`.
- Code license: Apache-2.0. The installer installs the upstream package with its LICENSE, MODEL_LICENSE and notices. These remain separate from this repository's license.
- Models: https://huggingface.co/m-a-p/YuE2-3B and https://huggingface.co/m-a-p/YuE2-Vae
- Model-card license: CC BY-NC 4.0. Preserve model notices with downloaded models.
- Author clarification on individual model/output use and monetization: https://huggingface.co/m-a-p/YuE2-3B/discussions/5 . The organization representative states that individual creators, musicians and researchers may monetize outputs, while companies need a commercial license. This clarification is not represented here as a replacement license for all redistributors, service providers or third-party adapters.
- API references: `docs/generation.md`, `docs/covers.md`, `src/yue2/pipeline.py`, `src/yue2/protocol.py`, `src/yue2/quantization.py` at the reviewed commit.
- CC license: https://creativecommons.org/licenses/by-nc/4.0/

The stage orchestration is an integration of the documented pipeline, not a new music generation model. Quantization/offload controls carry no quality or performance guarantee.

## audio.cpp

- Source: https://github.com/0xShug0/audio.cpp
- Referenced interface: https://github.com/0xShug0/audio.cpp/blob/v0.8.0/docs/models/yue2.md
- Models: https://huggingface.co/audio-cpp/Yue2-3B-GGUF
- Native binaries are user-installed and remain subject to the release's code and dependency licenses; they are not relicensed or redistributed here. The installer records the selected executable's SHA-256, not a claim that its publisher/signature has been independently audited.
- GGUF weights are model derivatives labeled CC BY-NC 4.0. The converter's code license does not remove model restrictions.

## Interoperability / evaluated alternatives

- FL YuE2: https://github.com/filliptm/ComfyUI-FL-YuE2 . ABC import/export interoperability only; no code, piano-roll assets or training package copied.
- SheetSage2: https://huggingface.co/m-a-p/SheetSage2 . ABC produced by a separately installed transcription environment can be supplied to YuE2. The transcriber and MERT2 are not installed by this extension.

A user must have appropriate rights to source audio, lyrics and scores. This extension does not grant rights to other people's compositions, recordings, voices, trademarks or Aikimi character assets.
