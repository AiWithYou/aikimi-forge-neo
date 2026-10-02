# 手元のモデルと複数LoRAを使う

v3.3.0から、既定モデルの代わりに手元の互換モデルを指定できます。元ファイルの移動・コピーは不要です。必要なテキストエンコーダーやVAEは用意しますが、標準の本体を先に取得する必要はありません。

## 画面での指定

| 使う画面 | 本体・共通部品 | LoRA |
|---|---|---|
| Forge `txt2img`／`img2img`（Anima 3.8B・SD系・その他対応モデル） | 上部の **Checkpoint** にフルパスを貼り付けてEnter。分離モデルの部品は **VAE / Text Encoder** へ指定 | **LoRAを組み合わせる** で複数選択し、表で強度を設定 |
| Qwen Image 2.1 | **本体モデル** でファイルまたはDiffusersフォルダーを指定。本体のみの場合は共通部品フォルダーも指定可能 | **LoRA** で複数選択・各強度を設定 |
| Ming Image | **モデル・LoRA** で本体を指定。**共通部品を指定** でテキスト・VAEも変更可能 | 同じ欄の **追加LoRA** で複数選択・各強度を設定 |

LoRAは`.safetensors`のフルパスを貼り付けても選べます。強度は−2〜2、初期値1、0は読み込み自体を省略します。×で外しても他のLoRAの強度は保持され、外したLoRAを追加し直すと1になります。同じ実ファイルを別名・相対パスで二重に指定することはできません。

使用したパスと選択は `models/.local-assets/library.json` に保存します。一覧の「更新」で再検索できます。Qwen／Mingでは、見つからなくなった選択を標準モデルへ自動で置き換えません。標準に戻す場合は **標準モデル** を選択してください。Forgeでは使用するCheckpointを選び直します。

Animaの既存4枠・プロンプトの`<lora:...>`も利用できます。同じLoRAを複数の方法で重ねて指定しないでください。Anima 3.8Bの派生モデルも52ブロック構成に対応した既存ローダーを使用します。Qwen3.5/v2モデルには従来どおり対応するエンコーダー／機能設定が必要です。

## 標準本体なしで準備する

**Qwen Image 2.1：** ローカル本体を指定し、**環境・共通部品を準備（標準本体なし）** を押します。共通部品が未指定なら公式のテキスト・VAE・processor・scheduler等を約18.9GB取得します。完全なDiffusersフォルダーか共通部品フォルダーを指定済みなら、専用実行環境だけを準備します。

```powershell
# 実行環境と共通部品。本体Transformer・GGUFはダウンロードしない
.\aikimi-qwen-image21-setup.bat --components-only
# 完全なパイプライン・全部品を持っている場合
.\aikimi-qwen-image21-setup.bat --runtime-only
```

**Ming Image：** **モデル・LoRA** で手元の本体を指定してから準備ボタンを押します。標準本体を取得せず、実行環境と標準テキスト・VAE（約13.1GB）を準備します。テキスト・VAEの両方も指定した場合は実行環境だけを準備します。

```powershell
.\venv\Scripts\python.exe tools\setup_ming_image.py --components-only
.\venv\Scripts\python.exe tools\setup_ming_image.py --runtime-only
```

生成ボタン自体はモデルをダウンロードしません。共通部品の不足や形式の不一致はエラーとして表示します。初回は大容量ファイルのSHA-256検証に時間がかかります。変更がないファイルの検証結果は再利用し、同名の上書きはWindowsの変更時刻も使って検出します。

## 対応する形式

| エンジン | 本体 | LoRA・制限 |
|---|---|---|
| Forge | 既存のForgeローダーが対応するアーキテクチャ／safetensors・GGUF | 既存のForge LoRAローダーを使用。すべての世代・量子化実装の互換を保証するものではない |
| Qwen Image **2.1** | 完全なDiffusersフォルダー、`config.json`と重みのあるTransformerフォルダー、Comfy／Diffusers系の単一safetensors、対応テンソル構成のGGUF | 線形A/B・down/up＋alpha。単一safetensorsはBF16、フォルダーはBF16／実行時INT8・W4A8、GGUFは通常版Q4の設定。外部の事前量子化safetensors／Diffusersフォルダーは未対応 |
| Ming Image | ComfyUI形式の単一safetensors（本体、テキスト、VAEを個別に指定） | 本体の線形A/B・down/up＋任意alpha。全ペアの適用先・寸法を検証。テキスト側LoRA・DoRA・LyCORIS・独自拡張キーは未対応 |

Qwen Image 2.1と旧Qwen Image／Image Editは別の構造です。旧世代は対応するForgeプリセットを使ってください。Qwenのローカル互換本体でもFun Acc・Outpaint・画風LoRAを併用できます。構造・テンソルの形状と必要な共通部品を検証します。標準Turboの選択と外部モデルの同時指定は受け付けません。追加LoRAには[学習時の量子化差と組み合わせ](../extensions-builtin/qwen-image21-studio/README.md#追加lora複数対応)の確認も適用します。

本体モデルやLoRAを変更すると必要に応じて再読み込みします。Qwenの量子化済みキャッシュは元ファイルの内容・精度ごとに分離し、元モデルには書き込みません。Ming／Qwenの結果には使用した絶対パス・SHA-256・強度を記録するため、生成条件を外部へ共有する際は保存場所も含まれます。

H3・YuE2など、他の専用Studioの任意モデル・LoRA指定はこの変更には含みません。外部で微調整された全モデルの画質や、異なるLoRA間の相性まで確認したものではありません。

## 検証記録（2026-09-30）

Windows 11・Python 3.13・RTX 3090 24GBで、既にある本体を「ローカル指定」の経路へ渡して確認しました。新しい本体のダウンロードは行っていません。LoRAは同名・別ディレクトリの検証用rank-1アダプター2個を作成し、強度0.7／−0.3で適用しました。

| 実機で確認した構成 | 結果 |
|---|---|
| Ming INT8本体・W4A8テキスト・BF16 VAEを個別指定＋2 LoRA | 256×256・2 stepsでRGBA PNG生成。実ファイルのSHA-256・強度を記録 |
| Ming W4A8本体＋同じ共通部品・2 LoRA | 同条件でRGBA PNG生成。圧縮された保存形状ではなく層の実寸でLoRAを検証 |
| Qwen Image 2.1 Q4_K_M GGUF＋外部2 LoRA | 同条件でRGBA PNG生成。両アダプター各1層の適用を記録 |
| Qwenから2 LoRAを外して続けて生成 | 適用記録が0件に戻り、同一seedの出力も変化 |

単体・回帰テストは320件中314件成功、任意の実機テスト6件はスキップ。標準本体がない構成、部品不足、異なるモデル世代、テンソル不一致、LoRAの重複、0による無効化、同名上書き、強度の保持、既存AnimaのLoRA経路を確認しています。新しい契約テストは次で再実行できます。

```powershell
.\venv\Scripts\python.exe tools\run_ci_tests.py --module tools.tests.test_local_model_sources
```

今回の実生成は読み込み・適用経路の動作検証です。外部配布の微調整モデル、Qwenの単一BF16／Diffusers本体、Anima 3.8Bの新しい派生モデルでの画質比較は行っていません。

UIの確認で、Gradio 6.17.3のDataframeが2件目以降の行数変更を描画しない問題も修正しました。固定した配布ファイルのSHA-256が一致する場合にだけ互換修正を配信し、インストール済みGradioのファイル自体は変更しません。
