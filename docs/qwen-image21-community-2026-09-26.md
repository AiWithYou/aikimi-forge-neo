# Qwen Image 2.1: community features and Outpaint helper

調査日: **2026-09-26**。実装比較の基準: `neo` の `7211f19e7add9085374102d7c9870f2693e97945`。

## 今回の変更と境界

**追加するのは、Outpaint用の参照画像を準備し、外部生成後に元画像を復元するCPU処理と専用タブです。LoRAの読み込み・ダウンロード・画像生成は実装していません。** 別途Qwen Image 2.1対応ComfyUIとausbossのLoRAが必要です。設定JSONは説明用メモで、ComfyUIへ読み込めるワークフローではありません。

既存のQwen生成worker、INT8／GGUF／W4A8読み込み、GPUの占有・解放、モデルキャッシュ、依存関係、セットアップ、既存タブは変更しません。既存モデルを上書きしたり、調査したLoRAを自動で取得したりしません。

## 収集した知見と導入判断

以下は配布元のモデルカードと本リポジトリを照合した記録です。配布元の性能説明と、今回こちらで実行したテストを区別しています。すべてのQwen派生モデルを網羅する一覧ではありません。

| 機能・一次資料 | 確認した有用性と注意点 | 今回の扱い |
|---|---|---|
| [ausboss Outpaint v1/v2](https://huggingface.co/ausboss/Qwen-Image-2.1-Outpaint-LoRA) | 灰色の余白を画像の続きへ置換する専用LoRA。v1は大きな拡張、v2は日常的な小～中程度の拡張向けという作者の説明。配布重みはComfyUI形式。 | **前後処理UIを追加。推論は外部ComfyUI。** ForgeのDiffusers経路で読めると判断しない。 |
| [Viggle Turbo v0.2.1](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo) | モデルカードでは2026-09-24版。6-step LoRAと専用sigma列を推奨。旧4-stepの全体重み／LoRAとは区別が必要。複雑な編集・細かい文字などに限界を明記。 | **次の優先検証候補。** 現行の旧4-stepプロファイルは維持し、Stepsだけを6へ変更しない。 |
| [Alibaba PAI Fun Acc](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs) | 本リポジトリには通常版INT8、4-step、専用PDD設定の経路がすでにある。一般的なEuler＋任意Stepsへ置き換えるものではない。 | **既存対応を維持。** 二重に導入するコードは追加しない。 |
| [Alibaba PAI Fun ControlNet Union](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union) | 本リポジトリは8種類の制御とInpainting＋Controlを導入済み。Outpaint LoRAの灰色参照方式とは別。 | **既存対応を維持。** Outpaint用のlatent noise maskとControlNet用編集マスクを混同しない。 |
| [Pruna 5/8-step](https://huggingface.co/PrunaAI/Pruna-Qwen-Image-2.1) | 5-stepと8-stepは別アダプター・別sigma設定。作者は8-stepを品質優先として推奨し、v0.1はベースモデルの品質に未到達と明記。1K・最大3参照の学習範囲。 | **比較実験候補。** 品質の既定値にはせず、既存高速化と重ねて使わない。 |
| [Object Remover Bbox Turbo](https://huggingface.co/prithivMLmods/Qwen-Image-2.1-Object-Remover-Bbox-turbo) / [Object Mover Bbox Turbo](https://huggingface.co/prithivMLmods/Qwen-Image-2.1-Object-Mover-Bbox-turbo) | 赤い矩形で対象を指定する削除／移動用途。作者は実験的な小規模学習として公開している。一般的な注釈や白黒マスクと同じ入力仕様とは限らない。 | **用途限定の検証候補。** 既存の注釈機能に無条件で置き換えない。 |
| [Natural Exposure LoRA](https://huggingface.co/prithivMLmods/Qwen-Image-2.1-Natural-Exposure-LoRA) | 露出表現を調整する実験的アダプター。通常の非生成的な露出補正とは異なり、内容を変えない保証はない。 | **任意の表現調整候補。** 人物や文字の保持を評価してから採用する。 |
| [e-n-v-y Fix](https://huggingface.co/e-n-v-y/Qwen-Image-2.1-Fix) | 作者の説明は付属ワークフローとの併用を条件としている。名称だけで汎用的な品質改善・修正と判断できない。 | **保留。** ワークフローと同条件の比較を先に行う。 |
| [公式Qwen Image 2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) | 生成・参照編集・RGBAはベースモデルの機能。本リポジトリにも最大10参照、透過PNG、PE-T2I／PE-I2Iの経路がある。 | **既存対応を維持。** 蒸留LoRAの学習参照数と、ベースモデルの入力上限を区別する。 |

モデル重みの利用条件は配布元のLICENSE／NOTICEを別途確認してください。Forgeのコードライセンスが、そのまま追加モデルの利用条件になるわけではありません。今回は重みや第三者実装の再配布を行いません。

## Outpaint補助の使い方

1. Forgeを再起動し、**Qwen Outpaint 補助**タブを開きます。元画像、上下左右の余白、外部で使うv1／v2、任意の場面説明を指定します。
2. **余白付き画像と設定を準備**を押し、参照PNGをダウンロードします。元画像を拡縮せず、余白を灰色にしてキャンバスを32の倍数へ調整します。画面の幅・高さと実際の余白を確認します。
3. 別途ComfyUIに対応するベースモデルと選択したOutpaint LoRAを読み込みます。画面の指示文・設定を使い、準備したキャンバスと**同じ幅・高さ**で生成してください。作者配布の手順・ワークフローを参照してください。本タブはComfyUIへジョブを送信しません。
4. 生成した画像をこのタブへアップロードし、**元画像を復元してPNGを作成**を押します。境界ぼかし32 pxでは拡張側の内縁を合成します。0 pxでは元画像領域の全画素を復元します。結果PNGをダウンロードします。

元画像・余白・v1/v2選択・場面説明を変更すると、準備状態と以前の生成画像・復元結果を破棄します。ブラウザーを再読み込みした場合も準備からやり直してください。同じサイズの無関係な生成画像まで識別する機能はありません。対応する生成結果を選んでください。

### 配布元の設定と、ここで追加した処理

配布元は灰色 `#808080` の参照、`image_1`、参照の `resolution=0`、エンコーダーからのlatent、25 steps／CFG 1／Euler／Simple／denoise 1を案内しています。**Set Latent Noise Maskは使わない**手順です。通常のinpaintingの定石をそのまま足さないでください。作者は生成後の元画像の再合成を推奨しています。

追加したコードは、その前後処理をPillowで独立実装したもので、作者のノードやワークフローを移植したものではありません。作者のStitchノードとのビット単位の同一性や、継ぎ目の画質の同等性は主張しません。

| 項目 | この実装の動作 |
|---|---|
| 対応画像 | 8-bit RGB／RGBA、グレースケール、パレット静止画像。EXIF回転を適用。16-bit、浮動小数点HDR、CMYKは黙って変換せず拒否。 |
| 配置 | 指定された側にだけ余白を追加。32の倍数への端数調整も同じ側へ追加。拡張しない軸が32の倍数でなければエラー。元画像を勝手に伸縮しない。 |
| サイズ | 各辺256～4096 px、総画素数2,097,152以下。入力40,000,000画素以下。大きすぎるキャンバスは生成せずエラー。これは補助の上限で、ベースモデル全体の能力上限ではない。 |
| 参照の透過部分 | モデル用参照だけ灰色に合成し、元のRGBAスナップショットを別に保持。透過素材へのOutpaint品質は未評価。 |
| 元画素の復元 | ぼかし0なら正規化済み元画像の全領域を復元。ぼかしありでは拡張側の内縁のみ変更し、内部を保持。alpha=0の隠れたRGBも保護領域では保持。 |
| 合成 | 透過境界はpremultiplied alphaで合成し、完全保持／完全非変更の画素を戻す。拡張領域は触らない。入力は破壊しない。 |
| 不整合 | 生成サイズ違い、元画像サイズ違い、不正な余白・NaN・小数・境界幅などを拒否。サイズ違いを自動リサイズで隠さない。 |
| 注意表示 | v1で約1 MPを超える場合、元画像がキャンバス面積の15%未満の場合に注意を表示。画質を保証する閾値ではない。 |

## 高速化LoRAの互換性メモ

**同じ「高速化LoRA」でも、重み・sigma列・time shiftは交換可能ではありません。**

Viggle v0.2.1の6-step推奨列は `[1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]` です。モデルカードは解像度依存のシフトを保持し、`shift_terminal=None` とCFG 1を使うよう指定しています。既存の旧4-step全体重みを残したまま、このLoRAを上乗せする移行にはしません。

Prunaは逆に、指定sigmaへ追加シフトしない設定です。`use_dynamic_shifting=False`、`shift=1.0`、`shift_terminal=None` を指定し、5-step／8-stepそれぞれに対応するアダプターを使います。この違いを共通の「高速化ON」で隠す設計は避けます。

採用前に必要な確認は、固定した重み・対応scheduler・依存版での読み込み、ベースモデルとの同seed比較、人物・文字・参照順序・透過画像の退行評価、実際のVRAMと速度測定です。配布元のH100測定をRTX 3090や16 GB GPUの速度として転記しません。

## ネイティブLoRA生成を追加する際の実装課題

OutpaintのComfyUIキーがForgeのDiffusers transformerへ正しく対応するかを先に確認します。キー文字列の置換だけで互換性があると扱わず、対象層・shape・alpha／rankと量子化の扱いを調べます。bitsandbytes INT8、GGUF、ConvRotなど別の経路を一括で対応済みにしません。

そのうえで、アダプターとschedulerを既存workerの管理下に置き、切り替え時の解放、重みの識別を含むキャッシュキー、キャンセル時の状態破棄、既存GPU占有制御、設定・出典の記録を統合します。Fun Acc、Viggle、Prunaなど蒸留方式の無検証な重ね掛けは拒否する設計が必要です。これらは今回未実装です。

## 検証範囲

追加ファイル:

- `modules_forge/qwen_image21/outpaint.py`: モデル非依存の準備・設定・復元。
- `extensions-builtin/qwen-image21-studio/scripts/qwen_image21_outpaint.py`: 独立したGradioタブ。
- `tools/tests/test_qwen_image21_outpaint.py`: CPUテスト。
- この調査記録。

Python 3.13.5、Pillow 12.3.0、Gradio 6.5.1の環境で、追加テスト**37件**を実行。15通りの拡張方向、端数配置、RGB／RGBA、EXIF、透過パレット、不正値、サイズ違い、合成の非破壊性、状態破棄を検査します。実Gradioでタブを構築し、11個のイベントがprivateであることと、準備・復元のコールバックも検査します。Forgeのcallback登録部分だけテスト用に置き換えています。

```shell
# リポジトリルート、必要な既存依存を導入したPythonで実行
python -m unittest discover -s tools/tests -p "test_qwen_image21_outpaint.py" -v
```

**未検証:** Windows上のForge全体の起動・ブラウザー操作、実モデルのGPU生成、作者ComfyUIワークフローの実行、生成画質・速度・VRAM、本リポジトリの全回帰テスト。テストは追加ファイルの部分作業ツリーで実行しています。Gradio構築時にevent loopのResourceWarningが出る環境であり、警告を隠していません。構文検査も追加3 Pythonファイルに限定しています。

したがって、この変更はネイティブOutpaint推論の完了や本番GPU検証済みリリースではありません。レビューと実機確認用の変更として扱ってください。
