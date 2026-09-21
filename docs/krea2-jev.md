# Krea2 / Jev

Krea2の通常生成・img2img・アップスケールで、層別のSparse Attentionを使えます。VRAM-Canvasには、タイルの細部量に応じて拡散のstep数を配分する別の設定もあります。既定はどちらもOFFです。

## 設定

txt2img / img2imgの **Krea2 · Jev高速化** を開きます。

- **Attention**: OFF、固定Sparse、数値ルール、Jev、Dense比較記録。
- **固定Sparseの保持率**: 固定モードでだけ表示します。
- **4K/8Kタイルのstep配分**: img2imgに表示します。VRAM-Canvas専用で、既存の配分、速度優先の数値ルール、Jevを選べます。
- **Jev APIキー設定**: 他モデルと共通の保管先を使います。キーは画面へ読み戻しません。Windowsではユーザー領域へDPAPIで暗号化して保存し、Gitや生成メタデータに含めません。

Jevを選んだ機能だけが外部APIを使います。OFF・固定・数値ルールはAPIを呼びません。SDKの準備とキー登録は[共通ガイド](jev-sparse.md)を参照してください。

## 層別Sparse

Krea2生成本体の各層が対象です。標準の28層モデルと、その構造を維持する派生チェックポイントで使用します。Q/K正規化、位置情報、GQA、ゲート、出力投影、MLPは元の処理を使います。文章・参照画像のKVとquery領域は、Comfy-Kitchenの保護範囲で保持します。テキストエンコーダーとTextFusionは対象外です。

最初のモデル評価はDenseで行います。Jevは画像領域のAttention出力ノルムなどの集約統計から、各層の保持率を **1・3・5・10・25・50・100%** から選びます。画像・プロンプト・重みは送信しません。重要度は統計からの推定で、画質の測定値ではありません。有効な回答は信頼度だけを理由に100%へ戻しません。

Jevの層別判定は生成ジョブ全体で1回までです。Krea2 2-Stage Upscale、VRAM-Canvas、Local Supersample Detail、B5 Whole-Tile Regenerationでは、内部で複数回生成しても判定を再利用します。OFFへ戻すと通常のAttentionを使い、成功・失敗・中断の後にジョブの状態を閉じます。

既定では画像部分が4,096トークン未満の領域を通常処理に戻します。小さなタイルはSparseの選択処理が負担になることがあり、Sparseを選ぶだけで高速になるとは限りません。比較時には詳細設定の最小トークン数も揃えてください。マスク付きの未対応経路は通常処理を使います。APIの失敗後も通常処理へ戻し、その理由を記録します。

## タイルのstep配分

Attentionとは独立した機能です。既存の配分は従来の細部量に応じたstep数をそのまま使います。

速度比較と同じ設定を画面で使う場合は、img2imgのScriptを **VRAM-Canvas 4K/8K Highres** にして、品質プロファイルで **Krea2 速度優先 4K** を選びます。1位相・2〜4steps・denoise 0.13・novel detailなしの設定です。従来のプロファイルと既定値も残しています。

速度優先の数値ルールは、細部量が非常に小さい領域の拡散を省き、他の領域へ既存の最小・最大step数を割り当てます。Jevでは最初の拡大段階のタイルを細部量のグループに集約し、グループごとに **0・最小step数・最大step数** を一括選択します。0は拡大済みの入力をそのまま残す意味です。省いたタイルも合成用の重みには参加するため、重なり部分に穴を作りません。

タイル用のAPI呼び出しも1ジョブ1回までです。後続段階は同じ細部量グループの選択を再利用し、未観測グループやAPI失敗時は数値ルールを使います。層別JevとタイルJevを両方選んだ場合は、合計で最大2回です。タイルごとにAPIを呼びません。送信内容はグループごとの枚数・細部量などの集約統計だけです。

## 記録と比較

`outputs/jev-sparse/krea2-*.jsonl` に層別の実行回数・選択値・API待ち時間、`krea2-tiles-*.jsonl` にタイル配分を保存します。VRAM-CanvasのPNGメタデータには各タイルのstep数と拡散を省いたかどうかを記録します。全タイルが0の場合も、拡大画像として完了します。

ローカルAPIを有効にしてNeoを起動し、Krea2・Qwen3-VL 4B・Qwen Image VAEを選んでから実行します。

```powershell
# 1280 / 2048 pxの通常・固定Sparse・Jevを2回ずつ比較。
venv\Scripts\python.exe tools\benchmark_krea2_sparse.py --port 7864 --stage native --output outputs\krea2-native-comparison --allow-cloud

# 同じ入力で4K全体を比較。1位相・1280pxタイル・2〜4stepsの速度比較設定。
venv\Scripts\python.exe tools\benchmark_krea2_sparse.py --port 7864 --stage 4k --repeats 1 --modes off fixed jev tiles-rules tiles-jev combined fixed-rules fixed-tiles-jev --output outputs\krea2-4k-comparison --allow-cloud
```

比較ツールはAPIの実生成完了までを計り、API失敗による退避をJevの成功結果として扱いません。通常生成の初回はモデル準備用として分離します。4K比較は`--source`で既存の入力画像も指定できます。実行後は一時変更したモデル選択を復元します。
