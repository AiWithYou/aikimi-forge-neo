# Qwen Image 2.1 Fun ControlNet Union · INT8 オリジナル作例

文章だけでは位置関係を指定しにくい2D人物と3D建築を、RTX 3090で実生成しました。[公式Unionモデル](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union)が挙げるCanny、Depth、Gray、HED、Lineart、MLSD、Pose、Scribbleの8種類と、**Inpainting＋Control**を扱います。条件画像と参照画像はユーザー提供の人物画像と新規作成したオリジナル素材から用意し、公式作例画像は配布していません。出力画像は生成されたPNGの全体です。

## 2D · 人物参照と別ポーズの組み合わせ

| 人物の参照画像（Image 1） | 新しい構図のLineart条件 |
|---|---|
| [原寸](anime-character-reference.jpg)<br><img src="anime-character-reference.jpg" width="250" alt="ユーザー提供の青系アニメ人物参照"> | [原寸](anime-pose-lineart.png)<br><img src="anime-pose-lineart.png" width="250" alt="片足立ちで傘と折り鶴を持つオリジナル線画"> |

| 画像参照のみ · ControlNetなし | 画像参照＋Lineart |
|---|---|
| [PNG](anime-reference-only.png)<br><img src="anime-reference-only.png" width="250" alt="画像参照のみのアニメ生成"> | [PNG](anime-reference-lineart.png)<br><img src="anime-reference-lineart.png" width="250" alt="画像参照とLineartを併用したアニメ生成"> |

両者は同一の参照画像・プロンプト・Seedです。制御なしでも人物と傘・折り鶴・足場は描けています。制御ありでは傘を持つ腕が頭上まで上がり、傘・折り鶴・足場がLineartの位置へより近づきました。画像参照は人物の特徴を渡し、Lineartは別の姿勢・物の位置を渡します。元絵の画素や人物の完全一致を保証する機能ではありません。

Poseでは同じ人物参照に、オリジナル線画から手で起こした骨格だけを渡します。Lineartと違い、傘や折り鶴、髪、服の線は条件画像に含みません。

| 骨格条件 · Pose | 画像参照＋Poseの生成 |
|---|---|
| [PNG](anime-pose-guide.png)<br><img src="anime-pose-guide.png" width="250" alt="片足立ちで左腕を上げ右腕を伸ばす骨格条件"> | [PNG](anime-reference-pose.png)<br><img src="anime-reference-pose.png" width="250" alt="人物参照とPoseを併用したアニメ生成"> |

## 3D · 輪を貫く橋と離れた塔

| 元のオリジナルLineart | そこから作ったScribble条件 |
|---|---|
| [原寸](ring-observatory-lineart.png)<br><img src="ring-observatory-lineart.png" width="250" alt="輪の中を橋が通るオリジナル建築線画"> | [原寸](ring-observatory-scribble.png)<br><img src="ring-observatory-scribble.png" width="250" alt="オリジナル線画から前処理した白線・黒地のScribble条件"> |

| プロンプトのみ · ControlNetなし | Scribble · 石造 | Scribble · 銅製 |
|---|---|---|
| [PNG](ring-3d-prompt-only.png)<br><img src="ring-3d-prompt-only.png" width="220" alt="プロンプトのみの3D建築"> | [PNG](ring-3d-control.png)<br><img src="ring-3d-control.png" width="220" alt="Scribbleで構図を指定した石造3D建築"> | [PNG](ring-3d-copper.png)<br><img src="ring-3d-copper.png" width="220" alt="同じScribbleで生成した銅製3D建築"> |

石造の制御あり／なしは同一プロンプトです。制御なしでも輪と橋は描けますが、輪は中央寄り、橋はほぼ水平、船も中央付近です。Scribbleを使うと輪が左へ移り、橋が右の塔へ向かって上がり、船が右下に配置されました。銅製は材質と時間帯の指示を変え、同じScribbleからの別解を試しています。

## 3D · 同じ建築から作った6種類の条件

[新規作成した3D原画](ring-observatory-render-source.png)からCanny、Depth、Gray、HED、MLSDを用意しました。上のScribbleは別のオリジナル線画から作成しています。下の5例は同じ主題を、条件画像の情報量を変えて描かせたものです。種類の選択はUIと記録用で、モデルは共通のUnion重みを使います。

<img src="ring-observatory-render-source.png" width="430" alt="条件画像の元になったオリジナル3D建築原画">

| 方式 | 前処理済みの条件画像 | 生成結果 |
|---|---|---|
| Canny · 輪郭 | [PNG](ring-canny.png)<br><img src="ring-canny.png" width="220" alt="3D建築原画から抽出したCanny輪郭"> | [PNG](ring-3d-canny.png)<br><img src="ring-3d-canny.png" width="220" alt="Canny条件による3D建築"> |
| Depth · 奥行き | [PNG](ring-depth.png)<br><img src="ring-depth.png" width="220" alt="Depth Anything V2 Smallによる相対深度"> | [PNG](ring-3d-depth.png)<br><img src="ring-3d-depth.png" width="220" alt="Depth条件による3D建築"> |
| Gray · 明暗構図 | [PNG](ring-gray.png)<br><img src="ring-gray.png" width="220" alt="3D建築原画のグレースケール"> | [PNG](ring-3d-gray.png)<br><img src="ring-3d-gray.png" width="220" alt="Gray条件による3D建築"> |
| HED · 柔らかい輪郭 | [PNG](ring-hed.png)<br><img src="ring-hed.png" width="220" alt="HEDで抽出した3D建築の輪郭"> | [PNG](ring-3d-hed.png)<br><img src="ring-3d-hed.png" width="220" alt="HED条件による3D建築"> |
| MLSD · 直線構造 | [PNG](ring-mlsd.png)<br><img src="ring-mlsd.png" width="220" alt="MLSDで抽出した直線構造"> | [PNG](ring-3d-mlsd.png)<br><img src="ring-3d-mlsd.png" width="220" alt="MLSD条件による3D建築"> |

MLSDは直線を抽出するため、曲面の大きな輪は条件画像にほとんど残りません。出力には輪が残りますが、プロンプトでも輪を指定しているため、輪の再現をMLSDだけの効果とは言えません。橋の直線的な部分は残る一方、傾きや船の位置は他の条件画像より変わっています。

## Inpainting＋Control · 輪の材質だけを変更

石造の生成結果を編集元とし、輪と小ドームを白い編集マスクで指定します。同じ画像から抽出したCanny条件を併用して、建築の配置を保ちながら輪を古びた銅へ変える指示を与えます。これはモデルの129チャンネル制御枝に編集元とマスクを渡すInpaintingで、後処理の「マスク範囲外を元画像に固定」はOFFです。

| 編集元 | 白＝編集・黒＝保持 | Canny条件 | Inpainting＋Control |
|---|---|---|---|
| [PNG](ring-3d-control.png)<br><img src="ring-3d-control.png" width="205" alt="元になる石造の3D建築"> | [PNG](ring-inpaint-mask.png)<br><img src="ring-inpaint-mask.png" width="205" alt="輪と小ドームだけを白く塗った編集マスク"> | [PNG](ring-inpaint-canny.png)<br><img src="ring-inpaint-canny.png" width="205" alt="編集元から作ったCanny条件"> | [PNG](ring-3d-inpaint.png)<br><img src="ring-3d-inpaint.png" width="205" alt="輪の材質を変更したInpainting出力"> |

輪には銅色と青緑の古色が入りました。小ドームはほぼ元の材質のままで、指定が完全には反映されていません。マスク外は見た目には近いものの、元画像とのRGB平均絶対差は`3.46/255`で、画素の`99.98%`に何らかの差があります。マスク内の平均絶対差は`17.89/255`です。これは一例の画像比較であり、黒い範囲の画素を厳密に保つには別スイッチの**マスク範囲外を元画像に固定**を併用します。

## 実測

| 出力 | サイズ | 制御強度 | モデル読込 | 生成処理 | PyTorch GPU割当ピーク |
|---|---:|---:|---:|---:|---:|
| 2D · 画像参照のみ | 768×1024 | — | 284.1秒 | 49.7秒 | 12,403 MiB |
| 2D · 画像参照＋Lineart | 768×1024 | 0.8 | 再利用 | 120.3秒 | 13,555 MiB |
| 3D · プロンプトのみ | 1024×768 | — | 再利用 | 38.1秒 | 9,766 MiB |
| 3D · Scribble石造 | 1024×768 | 0.65 | 279.0秒 | 151.6秒 | 12,200 MiB |
| 3D · Scribble銅製 | 1024×768 | 0.65 | 285.5秒 | 151.4秒 | 12,203 MiB |
| 2D · 画像参照＋Pose | 768×1024 | 0.8 | 346.2秒 | 259.8秒 | 13,555 MiB |
| 3D · Canny | 1024×768 | 0.8 | 再利用 | 73.9秒 | 12,202 MiB |
| 3D · Depth | 1024×768 | 0.8 | 再利用 | 70.1秒 | 12,202 MiB |
| 3D · Gray | 1024×768 | 0.65 | 再利用 | 67.2秒 | 12,203 MiB |
| 3D · HED | 1024×768 | 0.8 | 再利用 | 72.2秒 | 12,203 MiB |
| 3D · MLSD | 1024×768 | 0.8 | 再利用 | 74.0秒 | 12,203 MiB |
| 3D · Inpainting＋Canny | 1024×768 | 0.8 | 再利用 | 168.4秒 | 13,540 MiB |

ControlNetの初回推論には各プロセスのウォームアップが含まれます。生成処理を単純に比較してControlNetの速度差とみなさないでください。

全作例の数値・プロンプト・Seed・出力SHA-256は[測定JSON](measurements.json)に記録しています。

## 素材と再現条件

- [人物参照](anime-character-reference.jpg)はユーザーがこの作例用に提供した画像です。[人物Lineart](anime-pose-lineart.png)はそれを参照してImageGenで新たな片足立ちの構図を生成しました。[建築Lineart](ring-observatory-lineart.png)と[3D原画](ring-observatory-render-source.png)もImageGenで新規生成しました。Scribble・Canny・Gray・Pose・Inpaintingマスクは元画像から抽出または手作業で作成し、Depth・HED・MLSDは各前処理モデルを適用しました。[素材の出自とSHA-256](sources.json)も記録しています。
- Windows 11、RTX 3090 24GB、Qwen Image 2.1通常版INT8、CPU退避、Sparse OFF、プロンプト書き換えOFF。制御重みはKijaiのINT8 ConvRot（SHA-256 `07aa961570ac0e03d4ca936aecd76854d077a33cde69b5092399afba01b3715d`）。
- プロンプト、サイズ、強度、所要時間、PyTorchのGPU割当ピーク、出力PNGのSHA-256は[測定JSON](measurements.json)に記録しています。数値は各条件1回の測定で、平均や画質保証ではありません。PyTorch割当量はデスクトップやCUDAコンテキストを含むGPU全体の使用量ではありません。
- 条件画像はあらかじめ作成します。Qwenタブ内では条件画像の自動抽出を行いません。条件画像は出力の縦横比へ中央切り抜き・リサイズして使用します。[前処理スクリプト](../../../tools/prepare_qwen21_fun_example_controls.py)はHED・MLSD・Depth Anything V2 Smallの重みを指定revisionから取得しますが、重みはGitへ追加しません。
- 導入とUI操作は[Qwenガイド](../../../extensions-builtin/qwen-image21-studio/README.md#fun-controlnet-union--int8)を参照。2Dは人物参照を「参照画像」へ、人物LineartまたはPoseを「前処理済みの制御画像」へ入れます。Inpaintingは編集元を参照に追加し、マスク欄とCanny条件を指定します。全作例の再生成コマンド：

```powershell
.\models\Qwen-Image-2.1\worker-env\Scripts\python.exe tools\prepare_qwen21_fun_example_scribble.py
.\venv\Scripts\python.exe tools\prepare_qwen21_fun_example_controls.py
.\models\Qwen-Image-2.1\worker-env\Scripts\python.exe tools\generate_qwen21_fun_examples.py
```

Qwen本体とControlNet重みの利用条件は[Qwen Research License](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union/blob/8a4702014d4dabb5f896fcba917e2ee0a961465f/LICENSE)を確認してください。
