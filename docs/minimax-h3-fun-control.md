# MiniMax H3 Fun ControlNet：INT8版

H3 Studioの「Fun ControlNet · INT8 / 動きと構図」で、動画の輪郭やDepth・Poseなどに沿った生成ができます。初期設定はオフです。

## モデルと容量

[Kijaiの配布モデル](https://huggingface.co/Kijai/MiniMax-H3-experimental/tree/f4cac997f880e93cf6940af61ee8d58ef31ff7f3/controlnet)から、pruned INT8 ConvRot版を使用します。

| 同じpruned構成での比較 | ファイル容量 |
| --- | ---: |
| BF16版 | 4,222,169,456 bytes（約4.22GB） |
| INT8 ConvRot版 | 2,296,635,360 bytes（約2.30GB） |

保存容量は約46%減り、約1.93GBの削減です。主要な行列はINT8、一部の投影・正規化などはBF16／FP32を維持する混合形式になっています。INT8化で画質や制御結果が変わる可能性はあり、BF16版との同一出力は保証しません。実際のVRAM使用量には中間データやオフロードも影響します。

配置先は、H3で使うComfyUIの次の場所です。

```text
models/model_patches/minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors
```

SHA256：`9c645c0a308c8af361efd43b409710f6f8fec0db297c29503e141a84991fed0c`

## 使い方

1. 通常どおり生成モード、プロンプト、解像度、長さを設定します。
2. 「Fun ControlNet · INT8 / 動きと構図」を開き、入力方式を選びます。
3. 制御動画を追加し、ControlNetの強さを指定して生成してください。

| 入力方式 | 渡す動画 |
| --- | --- |
| 元動画から輪郭を抽出 · Canny | 通常の動画。生成前に輪郭を抽出します。 |
| 前処理済み動画 | Depth、Pose、HED、MLSD、Cannyなどの制御用動画。追加の抽出処理は行いません。 |

DepthやPoseの自動抽出モデルは、この機能では追加しません。元動画の音声も制御には使用せず、音声はH3の生成プロンプトから作ります。

制御動画は先頭から必要な長さを24fpsで取り出し、中央を切り抜いて生成サイズへ合わせる仕様です。短い動画は最後のフレームで補います。元ファイルは変更せず、整形した一時動画は生成終了後に片付けます。

## 実行環境と組み合わせ

接続先はComfyUI標準の`ModelPatchLoader`と`MiniMaxH3FunControlNetApply`です。標準ノードやINT8モデルがない場合は、生成要求を送る前に理由を表示して停止します。共通H3実行環境ではComfy Compilerを無効にして起動します。

NegPiPとCLIPキャッシュを同時に選択できます。ControlNetは生成モデルに、CLIPキャッシュはプロンプト・参照由来の条件に作用する構成です。制御動画のVAE処理やControlNet本体の計算は、CLIPキャッシュでは省略されません。ComfyUIやエンコーダーの更新後は、古い条件キャッシュを再計算する場合があります。

生成履歴には入力方式と強さを保存します。履歴から設定を戻す際は、制御動画をもう一度指定してください。
