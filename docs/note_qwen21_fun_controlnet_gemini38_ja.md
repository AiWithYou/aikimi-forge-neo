# 「この人物を、このポーズで」――Qwen Image 2.1のControlNetを8方式＋部分編集で試した

画像生成で「青い服のキャラクター」と指定するのは簡単です。しかし、人物の特徴を保ちながら、傘を高く掲げ、片足で立ち、右手の先に折り鶴を置く、といった細かな構図を文章だけで揃えるのは難しくなります。

Aikimi Forge NeoのQwen Image 2.1専用タブに、**Fun ControlNet UnionのINT8版**を追加しました。人物の特徴を伝える「参照画像」と、構図を指定する「条件画像」を別々に渡し、アニメ調の人物や3D調の建築を生成できます。

今回は、提供素材の人物画像と検証用に作成したオリジナル画像を使い、その実力を検証しました（公式配布画像の転載ではありません）。

## 人物の「見た目」と「ポーズ」を別々の画像から渡す

まずはアニメ調の人物で検証します。キャラクターの容姿を伝える「人物参照画像」と、ポーズや小物の位置を指定する「線画（Lineart）」を個別に用意しました。

![青系のアニメ人物の参照画像](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/anime-character-reference.jpg)

*人物の参照画像。淡い青の髪と衣装が特徴です（ユーザー提供素材）。*

![片足立ち、傘、折り鶴の位置を指定した線画](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/anime-pose-lineart.png)

*新たに作成したLineart条件。姿勢だけでなく、傘を上げる腕や浮遊する折り鶴の位置も描き込んでいます。*

同じ参照画像、プロンプト、Seed値を用い、「ControlNetなし（参照画像のみ）」と「Lineart併用」で生成結果を比較しました。

![画像参照だけで生成したアニメ人物](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/anime-reference-only.png)

*参照画像のみで生成。人物の特徴や小物は反映されていますが、傘を持つ腕の角度や小物の位置は線画の指定と異なります。*

![画像参照とLineartを併用したアニメ人物](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/anime-reference-lineart.png)

*参照画像＋Lineartで生成。腕が頭上へ上がり、傘や折り鶴、足場の位置が線画へ近づきました。*

**参照画像は人物の特徴、Lineartは位置と形を渡します。** この1例では併用した結果のほうが用意した線画の配置に近くなりました。元絵の画素や人物の完全一致を保証する機能ではありません。

さらに、線画から人物の骨格情報だけを手で起こした**Pose**条件も試しました。こちらには傘や折り鶴、髪や服の輪郭線は含まれていません。

![骨格だけを描いたPose条件画像](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/anime-pose-guide.png)

*骨格のみを指定したPose条件画像。*

![人物参照とPoseを組み合わせた生成結果](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/anime-reference-pose.png)

*人物参照＋Poseの生成結果。骨格に沿った片足立ちと腕の向きが出ています。条件画像にない服のシワや小物の位置は、Poseだけでは指定していません。*

## 複雑な3D建築の配置を線で渡す

背景や建築の構図指定でも威力を発揮します。今回は「海上の巨大な輪」「そこを貫く橋」「右側の塔」「手前の船」という複雑な空間配置を試しました。

文章だけでもそれぞれの要素は出せます。しかし、「輪を左に、橋を右上へ伸ばし、船を右下に収める」といった位置関係まで指定するのは難しくなります。

![文章だけで生成した輪、橋、塔の3D建築](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/ring-3d-prompt-only.png)

*文章のみで生成した結果。輪や橋はありますが、輪は中央寄り、橋はほぼ水平で、船も中央付近です。*

そこで、新たに作った建築線画から**Scribble**条件を作り、同じプロンプトで生成しました。この作例では輪が左へ移り、橋が右の塔へ向かって上がり、船も右下に配置されました。

![輪を通る橋、右側の塔、右下の船を指定したScribble条件](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/ring-observatory-scribble.png)

*Scribble条件。白い線で建築と船の配置を指定します。*

同じScribble条件から、材質と時間帯の指示を変えた別解も試しました。石造と銅製の2通りです。

![Scribbleで構図を指定した石造建築](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/ring-3d-control.png)

*Scribbleで構図を指定した石造建築。輪、橋、塔、船の大きな配置が条件画像に近づきました。*

![同じScribbleで材質を銅に変えた建築](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/ring-3d-copper.png)

*同じScribble条件から生成した銅製の別解。材質だけでなく、時間帯や光も変えています。*

## 8つの制御方式を試す

今回対応したのは、**Canny、Depth、Gray、HED、Lineart、MLSD、Pose、Scribble**の計8方式です。方式ごとに個別のモデルをロードし直す必要はなく、共通の「Union」重み1つに対して前処理済みの条件画像を渡す構成になっています。

前述のアニメ人物では[Lineartの結果](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/anime-reference-lineart.png)と[Poseの結果](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/anime-reference-pose.png)、建築では[Scribbleの結果](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/ring-3d-control.png)を使いました。残る5方式は、この検証用に作った別の3D原画から条件画像を抽出しています。

![5方式の条件画像の元になったオリジナル3D建築](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/ring-observatory-render-source.png)

*Canny、Depth、Gray、HED、MLSDに使った原画。上のScribble用の線画とは別素材です。*

**Canny｜輪郭**：[条件画像](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/ring-canny.png)

![Canny条件による3D建築の生成結果](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/ring-3d-canny.png)

**Depth｜奥行き**：[条件画像](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/ring-depth.png)

![Depth条件による3D建築の生成結果](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/ring-3d-depth.png)

**Gray｜明暗**：[条件画像](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/ring-gray.png)

![Gray条件による3D建築の生成結果](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/ring-3d-gray.png)

**HED｜柔らかい輪郭**：[条件画像](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/ring-hed.png)

![HED条件による3D建築の生成結果](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/ring-3d-hed.png)

**MLSD｜直線構造**：[条件画像](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/ring-mlsd.png)

![MLSD条件による3D建築の生成結果](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/ring-3d-mlsd.png)

MLSDの条件画像には、曲面である輪がほとんど残っていません。結果には輪がありますが、プロンプトでも輪を指定しているため、輪の再現をMLSDだけの効果とは言えません。

各方式の向き不向きを把握するには、**生成結果と条件画像の両方を見比べること**が大切です。すべての検証画像とプロンプトは[全画像と生成条件](https://github.com/AiWithYou/aikimi-forge-neo/blob/neo/docs/assets/qwen-image21-fun-controlnet/README.md)で公開しています。

## Inpainting＋Control：輪の材質を部分編集する

「全体の配置は変えずに、特定の部分だけ素材や色を変えたい」という場合には、Inpainting（部分再描画）とControlNetの併用が活躍します。

先ほどの石造建築から、「輪」と「輪の上の小さなドーム」を白く塗ったマスクを用意しました。白は再生成、黒は保持の指示です。さらに元画像から抽出した[Canny条件](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/ring-inpaint-canny.png)を同時に渡し、元の配置を参照しながら輪を古びた銅と青緑の古色へ変えます。

![輪と小さなドームを白く指定した編集マスク](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/ring-inpaint-mask.png)

*編集用マスク。輪と、その上の小ドームを再描画対象（白）に指定しています。*

![InpaintingとCanny条件で輪を銅色に変更した結果](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/ring-3d-inpaint.png)

*輪には銅色と青緑の古色が入りました。小ドームはほぼ元のままで、マスク内の指示も完全には反映されていません。*

この出力では、マスク外も元画像と画素単位では一致しません。元画像とのマスク外のRGB平均絶対差は**3.46/255**でした。外側を厳密に元画像の画素へ戻したい場合は、生成後の合成機能である**「マスク範囲外を元画像に固定」**を併用します。この作例では、その合成はOFFです。

## 試す前に知っておきたいこと

検証環境と生成パラメータは以下の通りです。

- **OS / GPU**：Windows 11、RTX 3090 24GB
- **モデル設定**：通常版Qwen Image 2.1、**INT8＋CPU退避**、Sparse Attention OFF
- **サンプリング**：40 steps
- **出力解像度**：3D建築は1024×768、アニメ人物は768×1024

各条件1回ずつの作例であり、生成速度や品質の平均値ではありません。Seedや計測値の詳細は[測定記録](https://github.com/AiWithYou/aikimi-forge-neo/blob/neo/docs/assets/qwen-image21-fun-controlnet/measurements.json)を参照してください。

### セットアップ手順

Neoを終了した状態で、`aikimi-qwen-image21-setup.bat --official-full`を実行して公式フルモデルを導入します。続けて、追加のControlNet重みを取得します。

```powershell
.\models\Qwen-Image-2.1\worker-env\Scripts\python.exe tools\prepare_qwen21_fun_controlnet.py --download
```
### 専用タブでの操作と重要な制約

Qwen Image 2.1タブで**通常・INT8**、**CPUへ退避**を選択し、Fun ControlNet欄で制御の種類、**前処理済みの制御画像**、強さを指定します。

- **前処理済み画像が必須**：タブ内にCannyやPoseの自動抽出機能はないため、条件画像は事前に作成して読み込ませる必要があります。
- **人物参照の併用**：キャラクターの見た目も引き継ぐ場合は、元画像を別の「参照画像」欄へセットします。
- **Inpaintingのサイズ指定**：編集元画像と同じ出力サイズに設定し、マスク画像と制御画像を併用してください。

操作の詳しい仕様は[機能ガイド](https://github.com/AiWithYou/aikimi-forge-neo/blob/neo/extensions-builtin/qwen-image21-studio/README.md#fun-controlnet-union--int8)にまとめています。

---

輪郭や骨格の画像を用意すれば、文章で何度も「もう少し左」「腕を上げて」と書いていた位置の指定を画像で渡せます。人物の同一性やマスク内の編集結果には揺らぎが残りますが、構図を先に作ってから生成に渡す選択肢が増えました。

モデル重みには非商用研究向けの**Qwen Research License**が適用されます。用途に合うか、利用前に[原文](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union/blob/8a4702014d4dabb5f896fcba917e2ee0a961465f/LICENSE)を確認してください（2026年9月24日確認）。

[Aikimi Forge Neo](https://github.com/AiWithYou/aikimi-forge-neo) ／ [公式Fun ControlNet Union](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union) ／ [INT8変換パッチ](https://huggingface.co/Kijai/QwenImage_experimental/tree/04987755e10002ff33e4ea307a811487dddd79d9/model_patches) ／ [利用ライセンス](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union/blob/8a4702014d4dabb5f896fcba917e2ee0a961465f/LICENSE)

#画像生成 #ControlNet #QwenImage #ローカルAI #AikimiForgeNeo
