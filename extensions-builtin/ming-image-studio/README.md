# Ming Image Design

[inclusionAI/Ming-Image-0.1-Design](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design)の専用タブです。ポスター、UIの見本、インフォグラフィック、透過素材などを**1枚の画像**として生成します。編集可能なUIコードや分離レイヤーの出力ではありません。

## 導入

1. `aikimi-setup.bat`で **6 Ming Image** を選ぶか、Neoの **Ming Image → Ming Imageを準備** を押します。
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
```

## 3090向けのモデル構成

| 部分 | 形式 | 配布ファイルの容量 |
|---|---|---:|
| 画像生成本体 | INT8 ConvRot | 6.18 GB |
| Ling-mini-2.0テキストエンコーダー・関連部品 | W4A8 | 12.81 GB |
| VAE | BF16 | 0.254 GB |
| 合計 | 本体INT8＋テキストW4A8 | 約19.3 GB（約17.9 GiB） |

この容量は保存する重みの合計で、必要VRAMとは異なります。計算中の一時メモリも必要で、ComfyUIがGPUとCPU間の配置を管理します。両方INT8ではテキスト側だけで19.51 GBになるため、この統合ではW4A8を選んでいます。

両方W4A8の構成は今回含みません。2026-09-29に確認した[Comfy-Orgの配布](https://huggingface.co/Comfy-Org/Ming-Image/tree/53654871e47a5d2daed7b3a986cbf1010ef81c78)には本体のW4A8ファイルがなく、自前変換の文字再現性・速度は未検証です。4bit表記だけで同じ形式や性能になるわけではありません。

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

## 検証

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

ブラウザーで、自然文／JSON切り替え時の入力保持、不正JSONの送信阻止、条件復元、途中停止、停止・失敗時の前の結果の保持、モデル解放を確認しました。透明PNGのダウンロードは保存元とバイト単位で一致しました。デスクトップと390px幅の配置も確認しています。

入力、JSON保持、PNG原本・透明面積、Seed復元、読み込みグラフ、準備の進捗と再試行など13件の自動テストを `tools/tests/test_ming_image_studio.py` に含め、関連するタブ・GPU管理・セットアップ・状態表示の回帰テストも実行しています。

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
