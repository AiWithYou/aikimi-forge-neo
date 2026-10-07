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
| Qwen Image **2.1** | 完全なDiffusersフォルダー、`config.json`と重みのあるTransformerフォルダー、Comfy／Diffusers系の単一safetensors、対応テンソル構成のGGUF | 線形A/B・down/up＋alpha。単一safetensorsはBF16またはINT8 ConvRot、フォルダーはBF16／実行時INT8・W4A8、GGUFは通常版Q4の設定。ConvRot以外の外部事前量子化と事前量子化Diffusersフォルダーは未対応 |
| Ming Image | ComfyUI形式の単一safetensors（本体、テキスト、VAEを個別に指定） | 本体の線形A/B・down/up＋任意alpha。全ペアの適用先・寸法を検証。テキスト側LoRA・DoRA・LyCORIS・独自拡張キーは未対応 |

Qwen Image 2.1と旧Qwen Image／Image Editは別の構造です。旧世代は対応するForgeプリセットを使ってください。Qwenのローカル互換本体でもFun Acc・Outpaint・画風LoRAを併用できます。構造・テンソルの形状と必要な共通部品を検証します。標準Turboの選択と外部モデルの同時指定は受け付けません。追加LoRAには[学習時の量子化差と組み合わせ](../extensions-builtin/qwen-image21-studio/README.md#追加lora複数対応)の確認も適用します。

本体モデルやLoRAを変更すると必要に応じて再読み込みします。Qwenの量子化済みキャッシュは元ファイルの内容・精度ごとに分離し、元モデルには書き込みません。Ming／Qwenの結果には使用した絶対パス・SHA-256・強度を記録するため、生成条件を外部へ共有する際は保存場所も含まれます。

H3・YuE2など、他の専用Studioの任意モデル・LoRA指定はこの変更には含みません。外部で微調整された全モデルの画質や、異なるLoRA間の相性まで確認したものではありません。

## abenzerpsのQwen Image 2.1 UC版

[abenzerps/Qwen-Image-2.1-Uncensored-GGUF](https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF/tree/6b34e59458d3eb7ba6a6f86a116aed5253dc02c3)の`qwen-image-2.1-UC-Q4_K_M.gguf`と`qwen-image-2.1-UC-int8_convrot.safetensors`を、Qwen Studioのローカル本体として追加できます。既存の専用環境・テキストエンコーダー・VAE・processor・schedulerを共用します。

2026-10-07に配布元の固定revisionと`SHA256SUMS`を確認しました。取得対象は4,604,558,112 bytes（約4.60 GB）、SHA-256は`e79c8a009f2ecbdb6c70fd663d9aea9ee304a0d91f347e4169a756b8ad141b41`です。ライセンスは配布元記載のQwen Research Licenseです。「Uncensored」は配布元の名称であり、通常版との出力の差を保証するものではありません。

INT8 ConvRotは同じrevisionの7,256,796,840 bytes（約7.26 GB）で、HubのLFSメタデータに記載されたSHA-256は`5bc5a6c007eff1e0d4004344a24c4af966b9d2c83d142e613b0d456ccceb19ae`です。ファイル内の`comfy_quant`設定・INT8重み・FP32スケール・層の寸法を検証し、32ブロックの224線形層を`comfy-kitchen==0.2.31`のConvRot（groupsize 256）で実行します。正規化層と未量子化の線形層はBF16です。本体は配布時のINT8を保持し、テキストエンコーダーは従来のbitsandbytes INT8を使用します。

```powershell
.\models\Qwen-Image-2.1\worker-env\Scripts\hf.exe download abenzerps/Qwen-Image-2.1-Uncensored-GGUF qwen-image-2.1-UC-Q4_K_M.gguf README.md SHA256SUMS --revision 6b34e59458d3eb7ba6a6f86a116aed5253dc02c3 --local-dir models/Qwen-Image-2.1/checkpoints/abenzerps--Qwen-Image-2.1-Uncensored-GGUF
.\models\Qwen-Image-2.1\worker-env\Scripts\hf.exe download abenzerps/Qwen-Image-2.1-Uncensored-GGUF qwen-image-2.1-UC-int8_convrot.safetensors --revision 6b34e59458d3eb7ba6a6f86a116aed5253dc02c3 --local-dir models/Qwen-Image-2.1/checkpoints/abenzerps--Qwen-Image-2.1-Uncensored-GGUF
.\models\Qwen-Image-2.1\worker-env\Scripts\hf.exe cache verify abenzerps/Qwen-Image-2.1-Uncensored-GGUF --revision 6b34e59458d3eb7ba6a6f86a116aed5253dc02c3 --local-dir models/Qwen-Image-2.1/checkpoints/abenzerps--Qwen-Image-2.1-Uncensored-GGUF
```

Qwenタブの**本体モデル → 更新**で再検索し、`qwen-image-2.1-UC-Q4_K_M.gguf`を選びます。精度は通常版の`Q4_K_M`へ自動で切り替わります。共通部品フォルダーは空欄のままで、導入済みの`models/Qwen-Image-2.1/model`を使えます。標準モデルへ戻す場合は**標準モデル**を選択します。

INT8対応コードを反映するためNeoを再起動し、**本体モデル → 更新**から`qwen-image-2.1-UC-int8_convrot.safetensors`を選びます。ファイル内の量子化構成から精度を`INT8`へ切り替え、ConvRot専用ローダーで読み込みます。ファイル名だけでは判定しません。BF16／W4A8指定での読み込みや、ConvRot以外の事前量子化は受け付けません。生成情報の`convrot_int8`に方式と適用層数を記録します。学習元モデルが指定されたLoRAの組み合わせ確認は引き続き必要です。

GGUFのBF16正規化重みは、読み込み後に通常のBF16パラメーターへ復元します。Diffusers形式の名前を持つGGUFは変換コールバックが省略されるため、この復元は名前変換から独立して行います。線形層のGGUF重みは量子化されたまま使います。この処理はローカル本体・標準GGUF・Turbo GGUFで共通です。

2026-10-07にWindows・Python 3.13.14・RTX 3090 24GBでQ4_K_M版の導入を確認しました。実ファイルのサイズ・SHA-256、一覧への検出、既存共通部品での受け付けを確認し、通常版Q4設定・CPU退避・256×256・2 stepsでRGBA PNGを生成しました。生成記録の本体パス・SHA-256も配布版と一致しています。この実行では読み込み約420秒、生成約13.5秒でした。低ステップ数での動作確認であり、画質や通常版との出力差は比較していません。

INT8の配布ファイルは2026-10-07に取得し、サイズとファイル全体のSHA-256が上記の固定値に一致することを確認しました。同日にWindows・Python 3.13.14・RTX 3090 24GBで、専用ワーカーによる256×256・2 steps・seed 20261007・CPU退避の実生成が完了し、RGBA PNGを保存しました。実際のConvRot呼び出しは448回で、全224層のINT8重み・FP32スケールがCPUに戻ったことを確認しました。代表の`transformer_blocks.0.attn.to_q`はFP32スケールが元ファイルと完全一致しています。生成情報にも本体のSHA-256・224層・ConvRot方式が記録されています。

この実行の読み込みは293.239秒、生成処理は200.883秒でした。生成処理にはプロンプト処理やCPU退避からの転送も含まれます。PyTorchの確保メモリのピークは9,697.4 MiB、予約ピークは9,798.0 MiBです（モデル読み込みを含む、このワーカーの値。GPU総使用量ではありません）。生成後の重みを保持した状態でCPU退避を確認し、確認プロセスは終了しています。

生成後に実ファイルのAttention・MLPの3種類の線形層を単体で測定し、`comfy_kitchen.backends.cuda`が選ばれることと非有限値が出ないことを確認しました。入力はBF16・257トークンで、2回目以降の単体呼び出しは約0.30〜0.40 msでした。これは生成全体の速度測定ではなく、初回201秒の原因や通常ステップ数の所要時間を確定するものではありません。2 stepsの画像は赤い物体がぼけた状態で、画質・通常版との出力差は比較していません。

単体・回帰テストは84件中80件成功・GPU専用4件スキップに加え、ConvRotにLoRA残差を適用してINT8重み・FP32スケールを保つ追加ケース1件も成功しました（合計85件中81件成功・4件スキップ）。Ruff静的チェックも通過しています。実生成の結果・使用重み・メモリ・所要時間は`tmp/qwen21-uc-int8-smoke-20261007/result.json`、224層のCPU退避と代表スケールの一致は`gpu-verification.json`、単体CUDA測定は`kernel-probe.json`に保存しています。

Q4対応時の関連テストは65件中61件成功、GPU専用4件はスキップ。正規化層の実際の順伝播、量子化された線形層の保持、ローカル指定・標準GGUF・既存ワーカーを確認しています。

```powershell
.\venv\Scripts\python.exe tools\run_ci_tests.py --module tools.tests.test_qwen_gguf_normalization --module tools.tests.test_local_model_sources --module tools.tests.test_qwen_image21_gguf --module tools.tests.test_qwen_image21_worker
```

追加したINT8契約テストは次で確認できます。

```powershell
.\venv\Scripts\python.exe tools\run_ci_tests.py --module tools.tests.test_qwen_int8_convrot --module tools.tests.test_local_model_sources --module tools.tests.test_qwen_image21_worker --module tools.tests.test_qwen21_fun_controlnet --module tools.tests.test_qwen_gguf_normalization --module tools.tests.test_qwen_image21_gguf --module tools.tests.test_qwen21_style_lora
```

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
