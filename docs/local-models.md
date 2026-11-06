# モデルを選ぶ・配置する

対応する画面のモデル選択欄で本体モデルを選び、**一覧を更新** で再検索します。選んだ本体に合う精度・VRAM設定を確認してください。Irisの量子化版など保存形式が固定された本体では、精度はモデルから決まります。Forgeの `txt2img`／`img2img` は従来の上部Checkpointを使います。

FLUX 3は公式の蒸留版／Baseに加え、互換の独自本体も選べます。対応するローカルLoRAを追加することもできます。[導入と対応形式](flux3-klein.md)を参照してください。

MOSS ImageはMOSS-Image-SFTの画像生成本体とテキストエンコーダーを個別にINT8／BF16などの対応形式から選択します。`aikimi-moss-image-setup.bat`で導入し、互換の線形LoRAを追加できます。[使い方](moss-image.md)を参照してください。

## フォルダー構造

手元のモデルを追加する標準の置き場は `models/<系列>/checkpoints/` です。起動時に追加用フォルダーを用意します。Forge・Qwen・IrisのLoRAフォルダーも同様に用意し、下位フォルダーを作って整理できます。

```text
models/
├─ Stable-diffusion/          # Forgeの従来のCheckpoint置き場
├─ Anima/checkpoints/
├─ Krea2/checkpoints/
├─ Qwen-Image-2.1/checkpoints/
├─ Iris-3B/checkpoints/
├─ Iris-3B/loras/             # 画像生成用の互換LoRA（下位フォルダーで整理可能）
├─ Flux-3-Klein/checkpoints/
├─ MOSS-Image/checkpoints/
├─ MOSS-Image/loras/          # MOSS画像生成本体の互換LoRA（下位フォルダーで整理可能）
├─ Clef/checkpoints/
├─ MiniMax-H3/checkpoints/
├─ Irodori-TTS/v4-Large/checkpoints/  # v4-Large互換の単一本体
├─ YuE2/checkpoints/
├─ Lora/                      # Forgeの追加LoRA（下位フォルダーで整理可能）
├─ VAE/                       # Forgeの分離VAE
└─ text_encoder/              # Forgeの分離エンコーダー
```

既に導入した公式モデル・共有部品・専用実行環境は、現在の保存先をそのまま使います。移動やコピーは必要ありません。追加本体は上記の `checkpoints/` 配下に置き、**一覧を更新** を押します。Irodori TTSは記載したフォルダーの直下に置きます。Forgeは従来の `models/Stable-diffusion/` と、Anima・Krea2・SDXL・SD・Flux・Flux.2・Qwen-Image・Qwen-Image-Edit・Lumina-Image-2.0・Z-Image・Wan・Ernie-Image・PiDの `models/<系列>/checkpoints/` を使用します。

一覧は探索先からの相対パスで表示します。選択を保存する画面では、見つからなくなったモデルを別の本体に黙って置き換えません。探索先やファイルを直して更新するか、公式モデルを選び直してください。一覧の更新や選択だけではダウンロードしません。

## 系列ごとの形式

| 画面 | 選択する実体 | 対応条件 |
|---|---|---|
| Forge | safetensors・GGUFなど、既存ForgeローダーのCheckpoint | 分離モデルは **VAE / Text Encoder** も指定。Animaの参照・Krea2など既存の機能設定を使用 |
| Qwen Image 2.1 | 単一safetensors・GGUF、Transformerフォルダー、完全なDiffusersフォルダー | **2.1** と互換の構造。通常版から実行時INT8／W4A8へ変換できます。対応するINT8 ConvRot・GGUFも選択可能。旧Qwen Image／Image Editは対応Forgeプリセットを使用 |
| Iris | 隣接する `config.yaml` を持つ単一safetensors | 生成用Full FT、通常版BF16／FP32、Iris専用INT8／W4A8。深度・復元はそれぞれの公式モデルを使用。[詳細](iris-studio.md#ローカルで学習したチェックポイント) |
| FLUX 3 | 互換の生成本体のsafetensors | BF16、公式と同じINT8 ConvRot、対応するComfy W4A8。独自NVFP4は未対応です。[詳細](flux3-klein.md#独自のfull-ft本体) |
| MOSS Image | 互換の画像生成本体のsafetensors | BF16・INT8 ConvRot・対応するComfy形式のW4A8。テキストエンコーダーとVAEは共通部品を使用。[詳細](moss-image.md) |
| Clef | 互換のモデル一式のフォルダー | Clef-FlashのBF16／INT8、ClefのBF16／NF4。設定・tokenizer・画像処理設定・判断ヘッド・語彙行列・重みが必要です。[詳細](clef-studio.md#独自full-ftモデルの選択) |
| MiniMax H3・H3 Image（実験） | 互換の生成本体のsafetensors | FL2VA用とRef2VA用を別々に選択。FP16／BF16／FP32、INT8 Tensorwise、対応W4A8。用途は配布元で確認し、encoder・VAE・音声部品は共有します。[詳細](../extensions-builtin/minimax-h3-studio/README.md) |
| P-H3 Image | P-H3互換構造のBF16／INT8／W4A8本体 | 同じ `MiniMax-H3/checkpoints/` の下位フォルダーまたは画面で指定した絶対パスから選択。P-H3用のテキストエンコーダーとVAEが必要です。構造の一致は画質や互換性の保証ではありません。 |
| Irodori TTS | v4-Large互換の単一safetensors | 通常版と対応するtorchaoのINT8・INT4・FP8。公式tokenizer・codecを共有し、形式ごとにGPU条件が異なります。INT4 weight-onlyはW4A8ではありません。[詳細](irodori-tts.md) |
| YuE2 | officialのモデルフォルダー、またはGGUF一式のフォルダー | 選択エンジンに対応するモデル・tokenizer／sidecars・VAEが必要 |

モデル一式に対応するStudioでは、内部にある分割重み・encoder・VAEを本体とは別のCheckpointとして表示しません。同じ拡張子でも異なる系列・世代・量子化方式には互換性がありません。画面ごとに必要な設定やファイルを確認しますが、一覧への表示だけでは生成時の互換性や画質は保証されません。

## 標準本体を取得せずに使う

Qwen・Irisではローカル本体を選んでから、環境と不足する共通部品だけを準備できます。

```powershell
.\aikimi-qwen-image21-setup.bat --components-only
.\venv\Scripts\python.exe tools\setup_iris.py --components-only
```

Qwenの完全なDiffusersフォルダーを指定済みなら、`--runtime-only` で専用実行環境だけを準備できます。生成ボタンはモデルを取得しません。不足する部品や形式の不一致は準備状態・エラーで確認できます。

## LoRAとモデルの変更

LoRAの学習元モデル名や推奨設定は参考情報です。それだけを理由に、本体モデルを自動で切り替えたり、別の本体での使用を拒否したりしません。Baseで学習したLoRAを構造が対応する蒸留版で使う場合も、利用者が選んだ本体を維持します。適用可否は、実際の対象レイヤー・テンソル形状・対応する実行形式で検証します。推奨値と実行に必須の条件を区別します。

Forgeは `models/Lora/` と下位フォルダーのLoRAを「LoRAを組み合わせる」で複数選択し、表で各強度を−2〜2に設定します。初期値は1、0は読み込みを省略します。従来のLoRAカード・プロンプトの `<lora:名前:強度>` と、Animaの専用選択欄も利用できます。下位フォルダーの同名LoRAはフォルダー名を含む名前で区別します。同じLoRAを専用欄とプロンプトの両方で重ねて指定しないでください。選択と各強度は保存し、画面を開き直しても復元します。

Qwenは `models/Qwen-Image-2.1/loras/` とその下位フォルダーへ追加LoRAを置きます。追加LoRAを複数選択し、表で各強度を−2〜2に設定します。初期値は1、0は読み込みを省略します。フルパスの直接入力も利用できます。同じ実ファイルの重複指定は受け付けません。

ForgeでPNGの生成情報を読み込むと、LoRA選択欄は空に戻ります。保存されたプロンプトのLoRAタグを使用するため、前回の選択が二重に適用されません。

Irisは `models/Iris-3B/loras/` と下位フォルダーの互換LoRAを複数選択し、各強度を指定できます。通常版・INT8・W4A8と互換Full FTに対応します。コミュニティ製Turbo LoRAも通常LoRAとして使え、専用の少ステップ生成方式は詳細設定で明示的に選べます。[対応形式と導入](iris-studio.md#任意のloraとコミュニティ製turbo)。ほかの専用Studioでは、それぞれのLoRA機能と対応形式を使います。すべてのLoRA形式や異なるモデル間の組み合わせを一律に適用できるものではありません。

MOSSは `models/MOSS-Image/loras/` と下位フォルダーのLoRAを複数選択し、表で各強度を−2〜2に設定します。初期値は1、0は適用を省略します。選択は初期状態では空欄です。画像生成本体のBF16・INT8・対応W4A8で、実際の対象と形状が一致する線形LoRAを適用します。本体やSteps・CFGはLoRAの選択によって自動変更しません。[対応形式](moss-image.md#lora)を参照してください。

P-H3 Imageは共有モデル保存先の `loras/`（標準は `models/MiniMax-H3/loras/`）からLoRAを1件選び、強度を−2〜2に設定します。初期値は1で、0は適用を省略します。画像生成と画像編集に使用できます。対応するLoRAや推奨設定は配布元を確認してください。[使い方](../extensions-builtin/minimax-h3-studio/P_H3_IMAGE_GUIDE.md#loraを使う)

本体やLoRAの変更、同名ファイルの上書きの扱いは各Studioで異なります。P-H3 Imageでは同名LoRAの上書き後に再読み込みされるとは限らないため、変更版は別名で保存して選択してください。量子化キャッシュは元モデルと変換条件ごとに分離し、元ファイルには書き込みません。初回の本体識別ではSHA-256の計算に時間がかかる場合があります。生成情報にモデルのパスや識別情報が含まれるため、共有する際は保存場所を確認してください。

## Qwen Image 2.1 UC版を追加する例

[abenzerps/Qwen-Image-2.1-Uncensored-GGUF](https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF/tree/6b34e59458d3eb7ba6a6f86a116aed5253dc02c3)の `qwen-image-2.1-UC-Q4_K_M.gguf` または `qwen-image-2.1-UC-int8_convrot.safetensors` を `models/Qwen-Image-2.1/checkpoints/` の下へ配置します。共通部品は導入済みの `models/Qwen-Image-2.1/model` を使用できます。

**一覧を更新** して本体を選択します。GGUFは通常版Q4設定、INT8 ConvRotは対応量子化構成を使います。ファイル名だけで形式は判断しません。「Uncensored」は配布元の名称で、通常版との出力差を保証するものではありません。
