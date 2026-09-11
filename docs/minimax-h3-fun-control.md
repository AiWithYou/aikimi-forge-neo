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

接続先はComfyUI標準の`ModelPatchLoader`と`MiniMaxH3FunControlNetApply`です。標準ノードやINT8モデルがない場合は、生成要求を送る前に理由を表示して停止します。ComfyUIは`efa6c8f804bff78b46a0fd458ebd2e47bba07a30`（0.34.0）に後述の修正を適用し、Comfy Kitchen 0.2.33を使用しています。

ComfyUI側の`comfy/ldm/minimax/model.py`には、Comfy Compilerの記録範囲を生成本体の内側へ移す修正を適用しています。追加機能の準備が終わってからメモリ記録を開始し、生成本体から戻った時点で記録を終了する構成です。その後に追加機能の片付けを実行するため、VAE前処理やNegPiPの片付けが記録中に入りません。H3の起動時にコンパイラを無効にするオプションは付けません。修正前から起動している場合は「選択設定で再起動」で反映してください。

comfy-aimdo 0.5.2には、メモリ記録中の解放で失敗する不具合がありました。[公式修正 #109](https://github.com/Comfy-Org/comfy-aimdo/pull/109)を含む`f708e317bd2dedfb2d2a8518f7c568fee280bc69`の公式CIビルド、0.5.3.dev13を導入し、ComfyUIの`requirements.txt`も同版に固定しています。この版は検証時点でPyPI未公開です。

修正差分は[ComfyUI 0.34.0用パッチ](../patches/minimax-h3/comfyui-0.34.0-compiler.patch)、基準リビジョンとwheelのSHA-256は[実行環境の記録](../patches/minimax-h3/runtime-provenance.json)に同梱しています。パッチには回帰テストも含みます。モデルやwheel自体はリポジトリに含めません。

現在の[H3初回セットアップ](../extensions-builtin/minimax-h3-studio/README.md#初回セットアップ)は、この修正を含む正式版`comfy-aimdo==0.5.3`をSHA-256付きlockで導入します。新規導入でCI成果物を取得する必要はありません。上記の開発版の記録は、以前の外部ComfyUI環境を検証した際のものです。

基準リビジョンのComfyUIで、次のようにパッチの適用可否を確認してから適用します。パスは自分の配置先へ変更してください。

```powershell
git -C "H:\path\to\ComfyUI" apply --check "H:\path\to\forge-neo\patches\minimax-h3\comfyui-0.34.0-compiler.patch"
git -C "H:\path\to\ComfyUI" apply "H:\path\to\forge-neo\patches\minimax-h3\comfyui-0.34.0-compiler.patch"
```

適用後はComfyUIのPythonで`python -m unittest discover -s tests-unit/comfy_test -p test_minimax_compiler_wrappers.py`を実行します。検証環境では5件が通過しました。

NegPiPとCLIPキャッシュを同時に選択できます。ControlNetは生成モデルに、CLIPキャッシュはプロンプト・参照由来の条件に作用する構成です。制御動画のVAE処理やControlNet本体の計算は、CLIPキャッシュでは省略されません。ComfyUIやエンコーダーの更新後は、古い条件キャッシュを再計算する場合があります。

生成履歴には入力方式と強さを保存します。履歴から設定を戻す際は、制御動画をもう一度指定してください。

## 実機検証（2026-09-08）

RTX 3090で、Canny制御・強さ1.0・NegPiP・CLIPキャッシュ自動・Comfy Compilerを同時に使用しました。修正後の384×512、124フレーム、20 Stepsの生成は780.299秒で完了し、H.264映像とAAC音声を保存できています。前回のコンパイラ無効時の出力と比較した全フレームのRGB値と音声データは同一でした。単発の測定であり、速度改善を保証する結果ではありません。

この生成では条件キャッシュへのHIT、Fun ControlNet本体のロード、NegPiPの適用、Comfy Compilerの動作をログで確認しています。BF16版との画質比較や、Depth・Poseを使った実生成は今回の検証に含みません。修正後の記録と動画は`outputs/minimax-compiler-fixed-audit/`、比較元は`outputs/minimax-fun-control-audit/`に保存しています。
