# Third-party notices

この文書は、Aikimi Forge Neoが利用または案内する主なthird-party成果物への索引です。各成果物のlicense本文と配布元の条件が優先されます。この文書は、モデルやassetの利用許諾を新たに与えるものではありません。

## コード基盤

| 成果物 | 用途 | licenseまたはnotice |
|---|---|---|
| AUTOMATIC1111 Stable Diffusion WebUI | WebUI基盤 | ルート[LICENSE](LICENSE)とupstream notice |
| Stable Diffusion WebUI Forge | Forge backend | ルート[LICENSE](LICENSE)とupstream notice |
| Stable Diffusion WebUI Forge - Neo | 現在のupstream | ルート[LICENSE](LICENSE)とupstream notice |
| Gradio 6.17.3 | WebUI frontend／backend | Apache-2.0。Aikimiは、PR #13509で修正された6.17.3のTabs reactive stormに対し、overflow計測を停止し、初期tab一覧をbatch同期する限定compat workaroundを利用。監査済みのversion、filename、SHA-256をすべて確認できた場合だけ、site-packagesを変更せず配信時に置換 |
| ComfyUI由来package | workflowとmodel処理 | [modules_forge/packages/comfy/LICENSE](modules_forge/packages/comfy/LICENSE) |
| GGUF package | GGUF読込 | [modules_forge/packages/gguf/LICENSE](modules_forge/packages/gguf/LICENSE) |
| built-in ControlNet／IP-Adapter | built-in extension | 各extension directoryの`LICENSE` |

ルートのコードはAGPL-3.0です。nested directoryに別のlicenseがある場合は、そのcopyright noticeと条件を保持します。

## Anima 3.8B extensionとtokenizer

`extensions-builtin/anima-3-8b`は、`GumGum10/forge-anima-3.8B`のcommit`59c27e5702f95c13dc5c08953637371d4749a034`を基にしています。extension codeのMIT Licenseは、[extensions-builtin/anima-3-8b/LICENSE](extensions-builtin/anima-3-8b/LICENSE)にあります。

同梱するQwen3.5 tokenizerの2ファイルは、`Qwen/Qwen3.5-4B`のrevision`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`由来です。出典は[extensions-builtin/anima-3-8b/THIRD_PARTY_NOTICES.md](extensions-builtin/anima-3-8b/THIRD_PARTY_NOTICES.md)にあります。配布元はApache-2.0と表示しています。release担当者は、配布物にApache-2.0本文が含まれることも確認してください。

## model installerが取得する成果物

モデルweightはGitリポジトリに含めません。`tools/aikimi_setup.py list`は、固定revisionとlicense URLを表示します。

| profile | 主な配布元 | 条件の確認先 |
|---|---|---|
| Krea2 | Comfy-Org/Krea-2、Qwen | 固定revisionのKrea 2 Community License PDFと各Qwen license |
| Anima 3.8B | lylogummy/Anima-3.8B、circlestone-labs/Anima、Qwen | 各model card、CircleStone Labs license、Qwen license |
| SenseNova U1.5 | SenseNova、starsFriday、joyfox | 各固定revisionのmodel cardとruntime LICENSE |
| MiniMax H3 | MiniMaxAI、ComfyUI、[KijaiのUnion 2.0変換済み重み](https://huggingface.co/Kijai/MiniMax-H3-experimental/tree/e042fe480f58806578713532b8ae4e3d47d1bd63/model_patches) | MiniMax H3 Community License、変換モデルの配布ページ、ComfyUI側のnotice。重みはGitに含めません。 |
| Qwen Image 2.1 | Qwen、Hugging Face Diffusers、bitsandbytes、[Unsloth通常版GGUF](https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF)、[Viggle Turbo](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo)、[Abiray Turbo GGUF](https://huggingface.co/Abiray/Qwen-Image-2.1-viggle-4-steps-turbo-GGUF) | [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/b3179ad355be050328e483a9dfdd9e60cd62adfa/LICENSE)（研究・評価目的の非商用利用。商用は別途許諾）、各変換モデルの配布ページ、Diffusers Apache-2.0、bitsandbytes MIT。重みはGitに含めません。 |
| Qwen Image 2.1のプロンプト補助 | Qwen PE-T2I、PE-I2I、Transformers、bitsandbytes | 各固定版のQwen Research License（[PE-T2I](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-T2I/blob/f3ed7985c788ad75b3ab7223e0c4c51e2a43545b/LICENSE)、[PE-I2I](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I/blob/72927bc08afc99b7888ceb7d7d51a12db3700bbd/LICENSE)）。導入先にLICENSEを保持し、量子化済み重みはGitに含めません。 |

Animaの配布repositoryは、upstream AnimaとNVIDIA由来条件の確認を求めています。条件を短く言い換えて断定せず、利用時点の原文を確認してください。Krea2にも独自のcommunity licenseがあります。

SenseNova runtimeは、固定commit`e6dfd45762eb46f805067fe079c14bcb643ccccd`から取得します。runtime directoryへApache-2.0の`LICENSE`も配置します。

## fontとUI asset

`modules/Roboto-Regular.ttf`は、font metadataでApache-2.0を示しています。`modules/web/fonts/sourcesanspro`にはSource Sans Proのwoff2が含まれます。release担当者は、fontの配布元と必要なlicense本文をrelease前に再確認してください。

ちびあいきみ画像の利用条件は[assets/aikimi/LICENSE.md](assets/aikimi/LICENSE.md)を参照してください。

docsのPDF、`html/ui.webp`、`html/card-no-preview.jpg`も配布assetです。新しいreleaseへ含める場合は、対応するsourceと利用条件をrelease checklistで確認してください。

## 外部サービス

MiniMax H3 Studioは、既定でローカルComfyUIだけを使います。H3のContext-IRと2K Regenerateは外部有料API向けですが、Aikimi Forge NeoのStudioは呼び出しません。将来外部APIを追加する場合は、送信データ、費用、利用規約、秘密情報の保存方法を別途明記してください。

## CD Tuner / MiniMax H3 NegPiP

- hako-mikan, [sd-webui-cd-tuner](https://github.com/hako-mikan/sd-webui-cd-tuner), revision `3685692b6a7001d3f34c5dfbec4756ac59884047`, AGPL-3.0. Adapted under `extensions-builtin/sd-webui-cd-tuner/`; upstream license and provenance are retained. Native Forge UI, rollback, validation and no-op optimizations differ from upstream.
- hako-mikan, [comfyui-minimax-h3-negpip](https://github.com/hako-mikan/comfyui-minimax-h3-negpip), revision `f725718b4c597fab92cdb1fe981dfe3544a75665`, AGPL-3.0. The original node and license are bundled unchanged under `extensions-builtin/minimax-h3-studio/comfyui_nodes/Aikimi-MiniMax-H3-NegPiP/`. Forge only adds policy, UI, graph wiring and opt-in local installation.

See [integration guide](docs/cd-tuner-negpip.md) for limitations and verification scope.

## MiniMax H3 CLIPCached

[Mu5hr00moO/ComfyUI-MiniMaxH3-CLIPCached](https://github.com/Mu5hr00moO/ComfyUI-MiniMaxH3-CLIPCached), revision `80ef7eb3b01565ff4190b519ce0db2f42b5e14e2`, MIT License. The optional installer retrieves and verifies the pinned runtime files, including the upstream LICENSE, into the selected ComfyUI custom_nodes directory. The implementation and model weights are not vendored in this repository. Forge adds settings, input mapping, installation checks and runtime permissions; see the [CLIPCached guide](docs/minimax-h3-clipcache.md).

## ComfyUI MiniMax H3 compiler patch

The patches under `patches/minimax-h3/` modify [ComfyUI](https://github.com/Comfy-Org/ComfyUI) revisions `efa6c8f804bff78b46a0fd458ebd2e47bba07a30` and `912fca4f39b875a0360f2c5170568176ea813ded`, distributed under GPL-3.0. They adjust the H3 compiler recording boundary and include regression tests. The separately installed official comfy-aimdo CI wheel is identified by source revision and SHA-256 in `runtime-provenance.json`; no wheel or model weights are bundled.
