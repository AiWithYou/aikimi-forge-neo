# Qwen Image 2.1 Fun ControlNet Union · INT8 オリジナル作例

文章だけでは位置関係を指定しにくい2D人物と3D建築を、RTX 3090で実生成しました。[公式Unionモデル](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union)が挙げるCanny、Depth、Gray、HED、Lineart、MLSD、Pose、Scribbleの8種類と、**Inpainting＋Control**を扱います。条件画像と参照画像はユーザー提供の人物画像と新規作成したオリジナル素材から用意し、公式作例画像は配布していません。出力画像は生成されたPNGの全体です。

## v2.2.1 · 同じ人物参照で8方式を比較

新たに提供されたカラフルなアニメ人物を**全方式の参照画像（Image 1）**に使います。ImageGenで別ポーズのガラス通路を描き、その新しい構図から8種類の条件画像を作りました。構図用原画そのものはQwenの参照画像には入れず、人物の特徴は左の提供画像から、配置や輪郭は方式別の条件画像から渡します。

| 人物の参照画像 · ユーザー提供 | 新しい構図の原画 · ImageGen | 参照だけ · ControlNetなし |
|---|---|---|
| [原寸](anime-v221-reference.png)<br><img src="anime-v221-reference.png" width="200" alt="金髪とピンクの髪、絵の具の付いた白いパーカーの人物参照"> | [原寸](anime-v221-layout-source.png)<br><img src="anime-v221-layout-source.png" width="200" alt="ガラス通路をローラースケートで進む新しい構図"> | [PNG](anime-v221-reference-only.png)<br><img src="anime-v221-reference-only.png" width="200" alt="新しい人物参照のみを使った比較画像"> |

| 方式 | 制御画像が渡す情報 | 前処理済みの制御画像 | 人物参照＋ControlNetの出力 |
|---|---|---|---|
| Canny | 細い輪郭とガラス通路 | [PNG](anime-v221-canny.png)<br><img src="anime-v221-canny.png" width="160" alt="Canny輪郭"> | [PNG](anime-v221-result-canny.png)<br><img src="anime-v221-result-canny.png" width="160" alt="Canny条件の出力"> |
| Depth | 人物と通路の相対的な奥行き | [PNG](anime-v221-depth.png)<br><img src="anime-v221-depth.png" width="160" alt="相対深度"> | [PNG](anime-v221-result-depth.png)<br><img src="anime-v221-result-depth.png" width="160" alt="Depth条件の出力"> |
| Gray | 明暗の塊 | [PNG](anime-v221-gray.png)<br><img src="anime-v221-gray.png" width="160" alt="グレースケール"> | [PNG](anime-v221-result-gray.png)<br><img src="anime-v221-result-gray.png" width="160" alt="Gray条件の出力"> |
| HED | 柔らかい輪郭 | [PNG](anime-v221-hed.png)<br><img src="anime-v221-hed.png" width="160" alt="HED輪郭"> | [PNG](anime-v221-result-hed.png)<br><img src="anime-v221-result-hed.png" width="160" alt="HED条件の出力"> |
| Lineart | 髪・服・手すりの線 | [PNG](anime-v221-lineart.png)<br><img src="anime-v221-lineart.png" width="160" alt="アニメ線画"> | [PNG](anime-v221-result-lineart.png)<br><img src="anime-v221-result-lineart.png" width="160" alt="Lineart条件の出力"> |
| MLSD | 通路やビルの直線 | [PNG](anime-v221-mlsd.png)<br><img src="anime-v221-mlsd.png" width="160" alt="MLSD直線検出"> | [PNG](anime-v221-result-mlsd.png)<br><img src="anime-v221-result-mlsd.png" width="160" alt="MLSD条件の出力"> |
| Pose | 手足と胴体の大まかな骨格 | [PNG](anime-v221-pose.png)<br><img src="anime-v221-pose.png" width="160" alt="手でトレースした骨格"> | [PNG](anime-v221-result-pose.png)<br><img src="anime-v221-result-pose.png" width="160" alt="Pose条件の出力"> |
| Scribble | 人物・星・斜め通路の長い輪郭線 | [PNG](anime-v221-scribble.png)<br><img src="anime-v221-scribble.png" width="160" alt="長い輪郭線を残したラフ画"> | [PNG](anime-v221-result-scribble.png)<br><img src="anime-v221-result-scribble.png" width="160" alt="Scribble条件の出力"> |

上表の8行は**同じ人物参照・プロンプト・Seed 43・40 steps・1152×1536・制御強度1.0**で生成します。MLSDには人物の輪郭がほとんど入らず、Poseには通路の線が入りません。方式名の選択だけで異なる重みを読み込む仕組みではなく、実際の制約は渡した画像の情報に依存します。

実画像では、Grayが人物だけでなく遠景の建物とガラス面も比較的細かく残しました。Lineartは人物の輪郭が明瞭な一方、街の背景を暗い屋内通路のように描き替えています。MLSDでは斜めの手すりが残っても腕の高さがずれ、Poseでは手足の方向が残っても星の大きさと背景が変わりました。**各方式1枚の観察**であり、品質の順位を示す比較ではありません。

Scribbleは最初の極端に疎い手描き案では人物が崩れたため、Canny画像から短い線を落とし、人物の外形・スケート・通路が読める長い線を残して再生成しました。修正版は全身と通路を描きましたが、条件画像中の小さな星は最終画像から消えています。

公式作例の条件は[Alibaba PAIのモデルカード](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union)の**40 steps・control_context_scale 1.0・Seed 43**です。[公式実行スクリプト](https://github.com/aigc-apps/VideoX-Fun/blob/main/examples/qwenimage21_fun/predict_t2i_control.py)は`FlowMatchEulerDiscreteScheduler`をQwen本体の`scheduler`ディレクトリから読み込みます。ローカルの`scheduler_config.json`は[Qwen本体の公式設定](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/scheduler/scheduler_config.json)とSHA-256 `5895f3a167c14a967fe9ac70c64924ae5acc79799e0679fd12907e594a713cd1`で一致しました。以前の作例は同じ40 stepsでも制御強度0.65〜0.8、人物出力768×1024でした。今回の画質評価では設定に加え、別の参照素材・プロンプト・構図も変わるため、改善要因を単独で特定した比較ではありません。

### v2.2.1 · Inpainting＋Pose

Canny結果を編集元にし、同じユーザー提供人物を再び参照画像へ渡します。白いマスクはパーカーの胸元だけを指定し、Pose条件は手足の大まかな位置を渡します。胸の柄をピンクのハートのワッペンへ変更する指示です。生成後の「マスク範囲外を元画像に固定」はOFFにし、モデルの生出力を掲載します。

| 編集元 · Canny結果 | マスク · 白が編集 | Pose条件 | Inpainting＋Pose結果 |
|---|---|---|---|
| [PNG](anime-v221-result-canny.png)<br><img src="anime-v221-result-canny.png" width="190" alt="編集元のCanny結果"> | [PNG](anime-v221-inpaint-mask.png)<br><img src="anime-v221-inpaint-mask.png" width="190" alt="パーカーの胸元を白くした編集マスク"> | [PNG](anime-v221-pose.png)<br><img src="anime-v221-pose.png" width="190" alt="大まかな骨格条件"> | [PNG](anime-v221-result-inpaint.png)<br><img src="anime-v221-result-inpaint.png" width="190" alt="InpaintingとPoseを併用した出力"> |

胸元には水色のハートとピンクの縁が入りました。指示した「ピンクの本体・水色の縁」と色の役割は逆です。マスク外も画素単位では固定されず、編集元とのRGB平均絶対差は`1.94/255`、少なくとも1チャンネルが異なる画素は`97.62%`でした。マスク内の平均絶対差は`35.40/255`、白い編集範囲は画面の`2.15%`です。厳密な範囲外保持には別機能の「マスク範囲外を元画像に固定」を使います。

### v2.2.1 · 実測

全例1152×1536・40 steps・Seed 43・INT8＋CPU退避です。制御ありは強度1.0、参照画像は全例同じ1枚。時間は秒、GPU割当はPyTorchのピークであり、GPU全体の使用量ではありません。

| 出力 | モデル読込 | 生成処理 | PyTorch GPU割当ピーク |
|---|---:|---:|---:|
| 参照のみ | 331.2 | 106.7 | 11,942 MiB |
| Canny | 279.8 | 306.6 | 14,788 MiB |
| Depth | 再利用 | 207.2 | 14,783 MiB |
| Gray | 再利用 | 207.7 | 14,783 MiB |
| HED | 再利用 | 207.3 | 14,784 MiB |
| Lineart | 再利用 | 207.6 | 14,783 MiB |
| MLSD | 再利用 | 207.0 | 14,783 MiB |
| Pose | 再利用 | 207.4 | 14,783 MiB |
| Scribble · 再生成 | 368.9 | 304.6 | 14,786 MiB |
| Inpainting＋Pose | 再利用 | 205.2 | 14,764 MiB |

CannyとScribbleは別プロセスで制御モデルを初回使用したため、生成処理にもウォームアップが含まれます。速度を方式の差として比較しないでください。全プロンプト・参照・制御画像SHA・出力SHA・スケジューラ設定SHAは[測定JSON](measurements.json)に記録しています。

## v2.2.0の人物作例 · 青系人物と別ポーズ

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

## v2.2.0 · 実測

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
- 導入とUI操作は[Qwenガイド](../../../extensions-builtin/qwen-image21-studio/README.md#fun-controlnet-union--int8)を参照。2Dは人物参照を「参照画像」へ、制御マップを「前処理済みの制御画像」へ入れます。Inpaintingでは編集元、白黒マスク、制御画像を追加します。下のコマンドはv2.2.1の人物作例を再生成します。以前の人物・3D作例も再生成する場合は、旧[制御画像の前処理](../../../tools/prepare_qwen21_fun_example_controls.py)と[手描きScribbleの準備](../../../tools/prepare_qwen21_fun_example_scribble.py)を実行してから、生成コマンドに`--all`を付けます。

```powershell
.\venv\Scripts\python.exe tools\prepare_qwen21_fun_anime_controls.py
.\models\Qwen-Image-2.1\worker-env\Scripts\python.exe tools\generate_qwen21_fun_examples.py
```

Qwen本体とControlNet重みの利用条件は[Qwen Research License](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union/blob/8a4702014d4dabb5f896fcba917e2ee0a961465f/LICENSE)を確認してください。
