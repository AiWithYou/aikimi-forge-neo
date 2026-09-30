# Ming Image Design

[inclusionAI/Ming-Image-0.1-Design](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design)の専用タブです。ポスター、UIの見本、インフォグラフィック、透過素材などを**1枚の画像**として生成します。編集可能なUIコードや分離レイヤーの出力ではありません。

## 導入

手元の互換モデルを使う場合は、**モデル・LoRA**で本体のフルパスを指定してから準備ボタンを押します。標準本体を取得せず、専用環境と共通テキスト・VAEだけを準備できます。共通部品も指定済みなら環境だけを準備します。同じ欄で複数の線形LoRAを選択し、それぞれの強度を−2〜2（0は無効）に設定できます。[ローカルモデル・LoRAの形式と手順](../../docs/local-models.md)

1. 標準モデルを使う場合は、Neoの **Ming Image → 詳細設定 → 標準モデルの精度** でINT8（標準）またはW4A8（省メモリ・試験版）を選び、**Ming Imageを準備** を押します。`aikimi-setup.bat`の **6 Ming Image** からも精度を選んで導入できます。
2. プロンプトを入力し、まず **1024×1024・12 steps** で「デザインを生成」を押します。
3. PNG原本・生成条件JSON・実際に使用したプロンプトを保存できます。出力先は `outputs/ming-image/` です。

取得済みのファイルはSHA-256で検証して再利用し、通常の「準備」では中断した取得を再開します。取得容量と検証状況は画面に表示します。「実行環境とモデル」内の修復は、破損ファイルと途中のファイルを退避して取得し直す操作です。専用Python 3.12と固定版ComfyUIを `repositories/ming-image/` に導入します。Forge、Qwen、H3のPython環境を上書きしません。Windows・NVIDIA CUDAが自動セットアップの対象です。

```powershell
# 導入予定だけ表示
.\venv\Scripts\python.exe tools\setup_ming_image.py --dry-run
# 導入／検証／破損ファイルの退避と再取得
.\venv\Scripts\python.exe tools\setup_ming_image.py
.\venv\Scripts\python.exe tools\setup_ming_image.py --verify
.\venv\Scripts\python.exe tools\setup_ming_image.py --repair
# 本体W4A8を直接ダウンロード（変換不要）
.\venv\Scripts\python.exe tools\setup_ming_image.py --precision w4a8
.\venv\Scripts\python.exe tools\setup_ming_image.py --precision w4a8 --verify
```

## 3090向けのモデル構成

| 部分 | 形式 | 配布ファイルの容量 |
|---|---|---:|
| 画像生成本体 | INT8 ConvRot | 6.18 GB |
| Ling-mini-2.0テキストエンコーダー・関連部品 | W4A8 | 12.81 GB |
| VAE | BF16 | 0.254 GB |
| 合計 | 本体INT8＋テキストW4A8 | 約19.2 GB（約17.9 GiB） |

この容量は保存する重みの合計で、必要VRAMとは異なります。計算中の一時メモリも必要で、ComfyUIがGPUとCPU間の配置を管理します。両方INT8ではテキスト側だけで19.51 GBになるため、この統合ではW4A8を選んでいます。

本体W4A8は[変換済み配布版](https://huggingface.co/Aikimi/Ming-Image-0.1-Design-W4A8)を直接取得できます。本体は約3.49GB、テキスト・VAEと合わせて約16.6GB（約15.4GiB）です。4bit表記だけで同じ形式や性能になるわけではありません。

## デザインを作る

自然文でも、公式のFigma風JSONでも入力できます。色、構図、文字の位置、各見出しの内容を具体的に書くと意図を伝えやすくなります。プロンプト例:

```text
A cream and forest green botanical poster. A large fern in the lower half,
elegant headline "BOTANICA", small subtitle "A quiet collection",
generous margins, minimal editorial design.
```

- **画像に載せる文字**は任意です。1行に1つ指定し、本文にすでにある文字は重複して追加しません。文字は生成モデルが描くため、誤字や欠落は画像で確認してください。
- **JSON**は構文を確認してそのまま送信します。JSON入力中は追加の文字欄・透過チェックを適用せず、値は保持します。文字・透過の指示はJSONに記述してください。
- **背景の透過を指示**では公式のRGBA指示を先頭に1つ付けます。透明背景になる保証はありません。透明に近い部分の面積を結果に表示し、透過を指定してもその面積が1%未満なら「ほぼ不透明」と知らせます。寸法は勝手に変えず、1024相当では公式推奨の2048相当を案内します。白・黒・チェックのプレビュー背景は保存PNGを変更しません。
- **この画像の条件をフォームに戻す**は表示中の結果の入力と確定Seedを復元します。フォームを編集しても、表示中の画像の条件は変わりません。
- **この画像の条件で W×H を再生成**は最後に成功した画像の入力・確定Seedを使い、縦横を2倍にして新しく生成します。拡大処理ではなく、構図や文字が変わることがあります。元の生成IDも条件JSONに記録します。最大2048×2048相当です。

寸法は16の倍数、各辺256〜4096、縦横比1:4〜4:1、総画素数2048²以下に対応します。Euler/simple・CFG 1を使用し、既定は12 stepsです。公式の推奨は2048相当ですが、まず1024相当で確認できます。

## モデル管理と制限

- Forge共通のGPU使用順・モデル保持・手動解放に参加します。停止後は処理の終了を確認してGPUの使用権を返します。H3・Nanosaur2と同じローカルポート8189を使い、別環境への誤接続を拒否します。
- 成功時だけPNG・JSON・prompt.txtをまとめて公開し、失敗時は前の結果を残します。PNGは再圧縮せず、RGBAを保持します。
- 自動プロンプト書き換え、参照画像編集、LoRA、別モデルのDesign-Layerはこのタブの対象外です。
- 生成はローカルです。入力プロンプトを外部の生成サービスへ送りません。

## 本体W4A8をダウンロードする（省メモリ・試験版）

Mingの **詳細設定 → 本体モデル → W4A8** を選び、**Ming Imageを準備** を押します。BF16原本の取得・変換は不要です。公開済みの本体W4A8と変換記録を取得し、固定revision・サイズ・SHA-256を検証します。共通のテキストエンコーダーとVAEは取得済みなら再利用します。既存のINT8版は残し、W4A8だけの新規導入ではINT8本体を取得しません。

コマンドで導入する場合は、Neoを終了して実行します。

```powershell
.\venv\Scripts\python.exe tools\setup_ming_image.py --precision w4a8 --dry-run
.\venv\Scripts\python.exe tools\setup_ming_image.py --precision w4a8
# セットアップランチャーから直接指定する場合
.\aikimi-setup.bat -Model ming-image -MingPrecision w4a8
```

既存のINT8構成に追加する場合の取得量は約3.49GBです。新規のW4A8構成は重み合計約16.6GBで、別途専用Python・CUDA依存パッケージの空き容量が必要です。テキストエンコーダーもW4A8なので、本体・テキストの両方がW4A8になります。VAEと量子化対象外の層はBF16です。

標準の選択はINT8で、W4A8だけを導入済みなら起動時にW4A8を選びます。選択した精度は生成条件に記録され、条件復元・高解像度再生成にも引き継がれます。「実行環境とモデル」の修復も選択中の精度が対象です。導入・生成中は精度を切り替えられません。

W4A8は対象層の重みを4bit、活性値を8bitに量子化する方式です。重みの容量が減っても生成時間が短くなるとは限りません。文字・細部・構図も変わるため、用途に合わない場合はINT8に戻してください。

### 変換を再現したい場合だけ

配布版の利用には不要ですが、`tools/quantize_ming_image.py`も残しています。Neo・生成用ComfyUIを終了して実行すると、固定版BF16原本と公式変換ツールのハッシュを検証し、量子化前のQKV結合・202層の形式と形状検査を行います。追加容量はBF16原本約12.3GB＋出力約3.49GBです。取得済み原本は `--source <パス>` で指定でき、既存の変換先は上書きしません。

```powershell
.\venv\Scripts\python.exe tools\quantize_ming_image.py --dry-run
.\venv\Scripts\python.exe tools\quantize_ming_image.py
```

## 検証

### v3.2.2の配布・導入確認

2026-09-29、公開リポジトリへログインなしでアクセスできることと、配布本体のサイズ・SHA-256を確認しました。手元のW4A8本体と変換記録を退避し、Windows・Chromeの画面でW4A8を選択→準備→実ダウンロード→ハッシュ検証→1024px生成まで確認しています。保存した条件のモデル名・変換記録も公開版と一致しました。既存のINT8・テキスト・VAEはサイズと更新日時が変わらず再利用されました。

未導入時に準備ボタンを再表示するGradio互換処理、W4A8だけを導入した場合の初期選択、選択した精度の取得・検証・修復を含むMingの23テストと、セットアップランチャーの5テストが通過しています。新規PCへの専用環境一式の導入は今回やり直していません。

### v3.2.1のINT8／W4A8比較

2026-09-29、RTX 3090 24GiB・RAM 64GB、同じ固定環境・プロンプト・Seed・12 stepsで、本体の精度だけを変えて実生成しました。テキスト側はどちらもW4A8です。

| 比較項目 | 本体INT8 | 本体W4A8 |
|---|---:|---:|
| 本体ファイル | 6.18 GB | 3.49 GB |
| 1024pxポスター、モデル保持後・2 Seed | 7.4〜7.6秒 | 8.5〜8.7秒 |
| 1024pxポスターのGPU使用量・標本最大 | 20.2 GiB | 17.9 GiB |
| 2048px透過素材、モデル保持後 | 53.2秒 | 57.8秒 |
| 2048px透過素材のGPU使用量・標本最大 | 21.2 GiB | 20.9 GiB |

1024pxでは余裕が増えましたが、今回の2048px生成ではピーク使用量の差は小さく、両サイズで生成時間は延びました。見出し・副題は両精度で読めましたが、W4A8のポスター1例には指定していない小さな文字列が加わりました。透過素材は両精度でRGBAを出力し、alpha16以下の面積はINT8約91.9%、W4A8約91.5%でした。形や細部は変わります。

GPU使用量は他アプリを含む全体値を約0.8秒＋計測処理時間ごとに標本化した最大です。所要時間はジョブ受付後で、環境起動・初回ハッシュ検証を含みません。少数例であり、速度・品質・必要VRAMの保証ではありません。[比較画像・生成条件・計測値](../../docs/assets/ming-image-w4a8/README.md)

画面からのW4A8生成、保存JSONの精度・変換記録、フォームを変更した後の精度・確定Seedの復元も実機で確認しました。

### v3.2.0のINT8実測

2026-09-29、Windows・RTX 3090 24GiB・RAM 64GB、専用Python 3.12.13 / PyTorch 2.11.0+cu130 / ComfyUI 0.37.0で確認しました。各12 steps・CFG 1の実測です。

| 操作 | ジョブ所要時間 | 結果 |
|---|---:|---|
| 1024×1024のポスター、初回読み込み | 約183秒 | 見出し・副題を含むPNGを保存 |
| モデル保持後、同じプロンプトでSeedを変更 | 約8秒 | 1024×1024の新しい画像を生成 |
| 表示中の画像の条件で2048×2048を再生成 | 約54秒 | 編集中のフォームに影響されず、元のプロンプト・Seedを使用 |
| 英語のイチョウ指示、1024×1024 | 約12秒 | RGBAだがほぼ不透明（alpha 252〜255） |
| 同じ指示・Seed、2048×2048 | 約50〜52秒 | 透明に近い部分が約92%のRGBA PNG |

時間はジョブ受付後で、専用環境の起動・初回のファイル整合性検証は別です。GPU全体の使用量は1秒間隔の標本最大で約21.9GiBでした。他アプリも含む値で、瞬間的な最大値の保証や他環境の性能予測ではありません。RAMの最小要件は未検証です。

透過はプロンプトと解像度に依存します。中国語の別の指示では1024でも透明になりましたが、少数例のため成功率や原因は断定していません。画面の「透明に近い部分」は **alpha 16以下** の面積です。背景の認識・切り抜きの正しさの判定ではなく、薄い色残りや輪郭は白・黒・チェック背景で確認してください。元のPNGをしきい値で加工することはありません。

ブラウザーで、自然文／JSON切り替え時の入力保持、不正JSONの送信阻止、条件復元、途中停止、停止・失敗時の前の結果の保持、モデル解放を確認しました。透明PNGのダウンロードは保存元とバイト単位で一致しました。デスクトップと390px幅の配置も確認しています。UI/UXはClaude Opus 5.5に3回相談し、試作と実生成の結果を反映しました。

入力、JSON保持、PNG原本・透明面積、Seed復元、読み込みグラフ、準備の進捗と再試行、W4A8の精度復元・変換記録・ハッシュ・上書き防止・量子化前のQKV結合、選択精度の直接取得・修復・操作中の固定などの自動テストを `tools/tests/test_ming_image_studio.py` に含め、関連するタブ・GPU管理・セットアップ・状態表示の回帰テストも実行しています。

[実生成の作例・条件と注意点](../../docs/assets/ming-image-v3.2.0/README.md)

```powershell
.\venv\Scripts\python.exe tools\run_ci_tests.py --module tools.tests.test_ming_image_studio
# Neo本体を終了し、モデル導入後に実際のGPU生成（通常のテストでは実行しません）
.\venv\Scripts\python.exe tools\test_ming_image_live.py
```

## 出典・固定版

確認日: 2026-09-29。

- [公式モデルとMITライセンス](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design)
- [公式の推奨設定・RGBA指示](https://github.com/inclusionAI/Ming-Image)
- [ComfyUI公式のMing実装・量子化MoE修正](https://github.com/Comfy-Org/ComfyUI/pull/16482)、固定コミット `3b4c0b0e457cf0a51cf3038e0a6750d8f96ce251`
- [Comfy-Org量子化重み](https://huggingface.co/Comfy-Org/Ming-Image/tree/53654871e47a5d2daed7b3a986cbf1010ef81c78)、全3ファイルのサイズ・SHA-256は [manifest](../../tools/ming_image_manifest.json)
- 依存パッケージは [専用lock](../../tools/requirements-ming-image.lock) で固定・ハッシュ検証します。モデルはこのGitリポジトリに含みません。
- 本体W4A8の変換は [Comfy-Org/comfy-model-tools](https://github.com/Comfy-Org/comfy-model-tools/blob/d6797787e6bdb1a1fb0094d588a26f8e71a1c757/quant_int8_convrot.py) の固定スクリプトと、導入済みのcomfy-kitchen 0.2.35を使います。
- [本体W4A8配布版](https://huggingface.co/Aikimi/Ming-Image-0.1-Design-W4A8)の取得先revision・サイズ・SHA-256は [W4A8 manifest](../../tools/ming_image_w4a8_manifest.json) で固定しています。配布先に元モデルのMITライセンス・変換記録・実測結果を含めています。
