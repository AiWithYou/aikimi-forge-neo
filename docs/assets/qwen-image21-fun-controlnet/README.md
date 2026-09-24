# Qwen Image 2.1 Fun ControlNet Union · INT8 オリジナル作例

構図を文章だけで指定しにくい二つの題材を、RTX 3090で実生成しました。2Dはユーザー提供画像の人物を**画像参照**として使い、別の**Lineart条件**で新しいポーズと傘・折り鶴の位置を指定します。3Dは輪の中を橋が通る建築を**Scribble条件**で指定します。制御あり／なしの比較は同じプロンプト・Seed `43`・40 stepsです。出力画像は生成されたPNGの全体です。

## 2D · 人物参照と別ポーズの組み合わせ

| 人物の参照画像（Image 1） | 新しい構図のLineart条件 |
|---|---|
| [原寸](anime-character-reference.jpg)<br><img src="anime-character-reference.jpg" width="250" alt="ユーザー提供の青系アニメ人物参照"> | [原寸](anime-pose-lineart.png)<br><img src="anime-pose-lineart.png" width="250" alt="片足立ちで傘と折り鶴を持つオリジナル線画"> |

| 画像参照のみ · ControlNetなし | 画像参照＋Lineart |
|---|---|
| [PNG](anime-reference-only.png)<br><img src="anime-reference-only.png" width="250" alt="画像参照のみのアニメ生成"> | [PNG](anime-reference-lineart.png)<br><img src="anime-reference-lineart.png" width="250" alt="画像参照とLineartを併用したアニメ生成"> |

両者は同一の参照画像・プロンプト・Seedです。制御なしでも人物と傘・折り鶴・足場は描けています。制御ありでは傘を持つ腕が頭上まで上がり、傘・折り鶴・足場がLineartの位置へより近づきました。画像参照は人物の特徴を渡し、Lineartは別の姿勢・物の位置を渡します。元絵の画素や人物の完全一致を保証する機能ではありません。

## 3D · 輪を貫く橋と離れた塔

| 元のオリジナルLineart | そこから作ったScribble条件 |
|---|---|
| [原寸](ring-observatory-lineart.png)<br><img src="ring-observatory-lineart.png" width="250" alt="輪の中を橋が通るオリジナル建築線画"> | [原寸](ring-observatory-scribble.png)<br><img src="ring-observatory-scribble.png" width="250" alt="オリジナル線画から前処理した白線・黒地のScribble条件"> |

| プロンプトのみ · ControlNetなし | Scribble · 石造 | Scribble · 銅製 |
|---|---|---|
| [PNG](ring-3d-prompt-only.png)<br><img src="ring-3d-prompt-only.png" width="220" alt="プロンプトのみの3D建築"> | [PNG](ring-3d-control.png)<br><img src="ring-3d-control.png" width="220" alt="Scribbleで構図を指定した石造3D建築"> | [PNG](ring-3d-copper.png)<br><img src="ring-3d-copper.png" width="220" alt="同じScribbleで生成した銅製3D建築"> |

石造の制御あり／なしは同一プロンプトです。制御なしでも輪と橋は描けますが、輪は中央寄り、橋はほぼ水平、船も中央付近です。Scribbleを使うと輪が左へ移り、橋が右の塔へ向かって上がり、船が右下に配置されました。銅製は材質と時間帯の指示を変え、同じScribbleからの別解を試しています。

## 実測

| 出力 | サイズ | 制御強度 | モデル読込 | 生成処理 | PyTorch GPU割当ピーク |
|---|---:|---:|---:|---:|---:|
| 2D · 画像参照のみ | 768×1024 | — | 284.1秒 | 49.7秒 | 12,403 MiB |
| 2D · 画像参照＋Lineart | 768×1024 | 0.8 | 再利用 | 120.3秒 | 13,555 MiB |
| 3D · プロンプトのみ | 1024×768 | — | 再利用 | 38.1秒 | 9,766 MiB |
| 3D · Scribble石造 | 1024×768 | 0.65 | 279.0秒 | 151.6秒 | 12,200 MiB |
| 3D · Scribble銅製 | 1024×768 | 0.65 | 285.5秒 | 151.4秒 | 12,203 MiB |

ControlNetの初回推論には各プロセスのウォームアップが含まれます。生成処理を単純に比較してControlNetの速度差とみなさないでください。

## 素材と再現条件

- [人物参照](anime-character-reference.jpg)はユーザーがこの作例用に提供した画像です。[人物Lineart](anime-pose-lineart.png)はそれを参照してImageGenで新たな片足立ちの構図を生成しました。[建築Lineart](ring-observatory-lineart.png)もImageGenで新規生成し、[Scribble条件](ring-observatory-scribble.png)を機械的に抽出しました。[素材の出自とSHA-256](sources.json)も記録しています。公式モデルの作例画像は同梱していません。
- Windows 11、RTX 3090 24GB、Qwen Image 2.1通常版INT8、CPU退避、Sparse OFF、プロンプト書き換えOFF。制御重みはKijaiのINT8 ConvRot（SHA-256 `07aa961570ac0e03d4ca936aecd76854d077a33cde69b5092399afba01b3715d`）。
- プロンプト、サイズ、強度、所要時間、PyTorchのGPU割当ピーク、出力PNGのSHA-256は[測定JSON](measurements.json)に記録しています。数値は各条件1回の測定で、平均や画質保証ではありません。PyTorch割当量はデスクトップやCUDAコンテキストを含むGPU全体の使用量ではありません。
- 条件画像はあらかじめ作成します。Qwenタブ内では元画像からのLineart／Scribble自動抽出を行いません。条件画像は出力の縦横比へ中央切り抜き・リサイズして使用します。
- 導入とUI操作は[Qwenガイド](../../../extensions-builtin/qwen-image21-studio/README.md#fun-controlnet-union--int8)を参照。2Dは入力画像を上の「参照画像」に、人物Lineartを「前処理済みの制御画像」に入れ、種類をLineartにします。3DはScribble条件画像を入れ、種類をScribbleにします。全作例の再生成コマンド：

```powershell
.\models\Qwen-Image-2.1\worker-env\Scripts\python.exe tools\prepare_qwen21_fun_example_scribble.py
.\models\Qwen-Image-2.1\worker-env\Scripts\python.exe tools\generate_qwen21_fun_examples.py
```

Qwen本体とControlNet重みの利用条件は[Qwen Research License](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union/blob/8a4702014d4dabb5f896fcba917e2ee0a961465f/LICENSE)を確認してください。
