# モデルを選ぶ・配置する

各生成・評価画面の先頭にある **チェックポイント** で本体モデルを選び、**一覧を更新** で再検索します。選んだ本体に合う精度・VRAM設定を確認してください。Irisの量子化版など保存形式が固定された本体では、精度はモデルから決まります。Forgeの `txt2img`／`img2img` は従来の上部Checkpointを使います。

## フォルダー構造

手元のモデルを追加する標準の置き場は `models/<系列>/checkpoints/` です。起動時に追加用フォルダーを用意します。下位フォルダーを作って整理できます。

```text
models/
├─ Stable-diffusion/          # Forgeの従来のCheckpoint置き場
├─ Anima/checkpoints/
├─ Krea2/checkpoints/
├─ Qwen-Image-2.1/checkpoints/
├─ Ming-Image/checkpoints/
├─ Iris-3B/checkpoints/
├─ Iris-3B/loras/             # 画像生成用の互換LoRA（下位フォルダーで整理可能）
├─ Clef/checkpoints/
├─ SenseNova-U1/checkpoints/
├─ MiniMax-H3/checkpoints/
├─ Nanosaur2/checkpoints/
├─ YuE2/checkpoints/
├─ Lora/                      # Forgeの追加LoRA（下位フォルダーで整理可能）
├─ VAE/                       # Forgeの分離VAE
└─ text_encoder/              # Forgeの分離エンコーダー
```

既に導入した公式モデル・共有部品・専用実行環境は、現在の保存先をそのまま使います。移動やコピーは必要ありません。追加本体は上記の `checkpoints/` 配下に置き、**一覧を更新** を押します。下位フォルダーも探索するため、別の探索フォルダーを指定する必要はありません。Forgeは従来の `models/Stable-diffusion/` と、Anima・Krea2・SDXL・SD・Flux・Flux.2・Qwen-Image・Qwen-Image-Edit・Lumina-Image-2.0・Z-Image・Wan・Ernie-Image・PiDの `models/<系列>/checkpoints/` を使用します。

一覧は探索先からの相対パスで表示し、同名の候補は探索先も表示して区別します。選択内容は `models/.local-assets/library.json` に保存します。見つからなくなったモデルを別の本体に黙って置き換えません。探索先やファイルを直して更新するか、公式モデルを選び直してください。一覧の更新や選択だけではダウンロードしません。

## 系列ごとの形式

| 画面 | 選択する実体 | 対応条件 |
|---|---|---|
| Forge | safetensors・GGUFなど、既存ForgeローダーのCheckpoint | 分離モデルは **VAE / Text Encoder** も指定。Animaの参照・Krea2など既存の機能設定を使用 |
| Qwen Image 2.1 | 単一safetensors・GGUF、Transformerフォルダー、完全なDiffusersフォルダー | **2.1** と互換の構造。BF16、INT8 ConvRot、対応GGUF、フォルダーの実行時INT8/W4A8。旧Qwen Image／Image Editは対応Forgeプリセットを使用 |
| Ming Image | ComfyUI形式の単一safetensors | 互換の本体構造。テキスト・VAEは共通部品として個別指定可能 |
| Iris | 隣接する `config.yaml` を持つ単一safetensors | 生成用Full FT、通常版BF16／FP32、Iris専用INT8／W4A8。深度・復元はそれぞれの公式モデルを使用。[詳細](iris-studio.md#ローカルで学習したチェックポイント) |
| Clef | 対応するモデル一式のフォルダー | Clef／Clef-Flashの同版・同精度の設定・tokenizer・joint headを含む構成。量子化版は対応manifestとbundle形式 |
| SenseNova U1.5 | final／pruned INT8 ConvRot safetensors | U1.5の対応テンソル構造と量子化設定。通常BF16や別世代はこのローダーでは扱わない |
| MiniMax H3・H3 Image | 対応するFL2VA／Ref2VA本体 | 用途と実行プロファイルに合うモデル。text encoder・VAE・音声部品は導入済み共通部品を使用 |
| Nanosaur2 | 対応する本体safetensors | Nanosaur2の本体構造。共有text encoder・VAEを使用 |
| YuE2 | officialのモデルフォルダー、またはGGUF一式のフォルダー | 選択エンジンに対応するモデル・tokenizer／sidecars・VAEが必要 |

モデル一式に対応するStudioでは、内部にある分割重み・encoder・VAEを本体とは別のCheckpointとして表示しません。同じ拡張子でも異なる系列・世代・量子化方式には互換性がありません。各ローダーは用途、設定、テンソル構造、必要な共通部品を確認します。

## 標準本体を取得せずに使う

Qwen・Ming・Irisではローカル本体を選んでから、環境と不足する共通部品だけを準備できます。

```powershell
.\aikimi-qwen-image21-setup.bat --components-only
.\venv\Scripts\python.exe tools\setup_ming_image.py --components-only
.\venv\Scripts\python.exe tools\setup_iris.py --components-only
```

Qwenの完全なDiffusersフォルダーやMingの本体・text encoder・VAE一式を指定済みなら、`--runtime-only` で専用実行環境だけを準備できます。生成ボタンはモデルを取得しません。不足する部品や形式の不一致は準備状態・エラーで確認できます。

## LoRAとモデルの変更

Forgeは `models/Lora/` と下位フォルダーのLoRAを「LoRAを組み合わせる」で複数選択し、表で各強度を−2〜2に設定します。初期値は1、0は読み込みを省略します。従来のLoRAカード・プロンプトの `<lora:名前:強度>` と、Animaの専用選択欄も利用できます。下位フォルダーの同名LoRAはフォルダー名を含む名前で区別します。同じLoRAを専用欄とプロンプトの両方で重ねて指定しないでください。選択と各強度は保存し、画面を開き直しても復元します。

Qwenは `models/Qwen-Image-2.1/loras/`、Mingは `models/Ming-Image/loras/` とその下位フォルダーへ追加LoRAを置きます。追加LoRAを複数選択し、表で各強度を−2〜2に設定します。初期値は1、0は読み込みを省略します。フルパスの直接入力も利用できます。同じ実ファイルの重複指定は受け付けません。Mingの導入済み共有ランタイム内の `models/loras/` も利用できます。

ForgeでPNGの生成情報を読み込むと、LoRA選択欄は空に戻ります。保存されたプロンプトのLoRAタグを使用するため、前回の選択が二重に適用されません。

Irisは `models/Iris-3B/loras/` と下位フォルダーの互換LoRAを複数選択し、各強度を指定できます。通常版・INT8・W4A8と互換Full FTに対応します。Iris-3B Turboの蒸留LoRAでは専用の少ステップ生成を使います。[対応形式と導入](iris-studio.md#任意のloraとturbo)。ほかの専用Studioでは、それぞれのLoRA機能と対応形式を使います。すべてのLoRA形式や異なるモデル間の組み合わせを一律に適用できるものではありません。

本体やLoRAの変更、同名ファイルの上書きは必要に応じて再読み込みします。専用Studioでは受付時のLoRAを生成中も確認し、待機中や読み込み中にファイルが変わった場合は再実行を求めます。量子化キャッシュは元モデルと変換条件ごとに分離し、元ファイルには書き込みません。初回の本体識別ではSHA-256の計算に時間がかかる場合があります。生成情報にモデルのパスや識別情報が含まれるため、共有する際は保存場所を確認してください。

## Qwen Image 2.1 UC版を追加する例

[abenzerps/Qwen-Image-2.1-Uncensored-GGUF](https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF/tree/6b34e59458d3eb7ba6a6f86a116aed5253dc02c3)の `qwen-image-2.1-UC-Q4_K_M.gguf` または `qwen-image-2.1-UC-int8_convrot.safetensors` を `models/Qwen-Image-2.1/checkpoints/` の下へ配置します。共通部品は導入済みの `models/Qwen-Image-2.1/model` を使用できます。

**一覧を更新** して本体を選択します。GGUFは通常版Q4設定、INT8 ConvRotは対応量子化構成を使います。ファイル名だけで形式は判断しません。「Uncensored」は配布元の名称で、通常版との出力差を保証するものではありません。
