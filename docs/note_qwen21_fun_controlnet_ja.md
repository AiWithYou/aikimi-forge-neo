# 「この人物を、このポーズで」――Qwen Image 2.1のControlNetを8方式＋部分編集で試した

画像生成で「青い服の人物」と書くのは簡単です。でも、同じ人物に片足で立ってもらい、傘を上げる腕と折り鶴の位置まで決めようとすると、文章だけでは難しくなります。

Aikimi Forge NeoのQwen Image 2.1専用タブに、**Fun ControlNet UnionのINT8版**を追加しました。人物の参照画像と構図の条件画像を別々に渡し、アニメ調の人物と3D調の建築で試しています。素材は提供された人物画像と、この検証用に作ったオリジナル画像です。公式作例の転載ではありません。

## 人物の見た目とポーズを、別々の画像から渡す

人物参照には、淡い青の髪と服のアニメ画像を使いました。新しいポーズ用には、傘を上げ、片足で立ち、右手を伸ばした線画を用意しました。

![青系のアニメ人物の参照画像](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/anime-character-reference.jpg)

*人物の参照画像。ユーザー提供素材。*

![片足立ち、傘、折り鶴の位置を指定した線画](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/anime-pose-lineart.png)

*新たに作ったLineart条件。人物だけでなく、傘や折り鶴の位置も描いています。*

同じ参照画像、プロンプト、Seedで、まずControlNetなし、次にLineartありを生成しました。

![画像参照だけで生成したアニメ人物](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/anime-reference-only.png)

*参照画像のみ。人物や小物は描けていますが、傘を持つ腕と小物の位置は線画と異なります。*

![画像参照とLineartを併用したアニメ人物](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/anime-reference-lineart.png)

*参照画像＋Lineart。腕が頭上へ上がり、傘、折り鶴、足場の位置が線画に近づきました。*

**参照画像は人物の特徴、Lineartは位置と形**を担います。完全に同じ人物を複製する機能ではありませんが、「誰を」「どこに」を分けて指定できるのが便利です。

線画から手で起こした骨格だけを使う**Pose**も試しました。こちらには傘、折り鶴、髪や服の線がありません。渡している情報が少ない分、姿勢を指定しつつ、小物や細部を生成側に任せられます。

![骨格だけを描いたPose条件画像](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/anime-pose-guide.png)

![人物参照とPoseを組み合わせた生成結果](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/anime-reference-pose.png)

## 輪を橋が貫く建築を、線で配置する

3D調では、海上の巨大な輪、そこを通る橋、右側の塔、手前の船という配置を試しました。文章だけでも要素は出ます。ただ、輪を左に、橋を右上へ、船を右下へ、といった関係まで揃えるのは別の話です。

![文章だけで生成した輪、橋、塔の3D建築](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/ring-3d-prompt-only.png)

*文章だけの結果。輪や橋はありますが、構図は指定線画から離れています。*

オリジナル線画から作った**Scribble**条件を加えると、輪が左へ移り、橋が右の塔へ向かって上がり、船も右下に収まりました。同じ線から材質の指示を変え、石造と銅製の二通りも生成しています。

![Scribbleで構図を指定した石造建築](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/ring-3d-control.png)

![同じScribbleで材質を銅に変えた建築](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/ring-3d-copper.png)

## 8方式を実際に使った

今回対応したのは**Canny、Depth、Gray、HED、Lineart、MLSD、Pose、Scribble**です。選択名ごとに別モデルを読み分ける方式ではなく、共通のUnion重みへ前処理済み画像を渡します。

アニメ人物には[Lineartの結果](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/anime-reference-lineart.png)と[Poseの結果](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/anime-reference-pose.png)、建築線画には[Scribbleの結果](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/ring-3d-control.png)を使いました。残る5方式は、別途作ったオリジナル3D原画から条件画像を作成して生成しました。

- **Canny**：輪郭を強く渡す。[条件画像](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/ring-canny.png)／[生成結果](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/ring-3d-canny.png)
- **Depth**：大まかな奥行きを渡す。[条件画像](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/ring-depth.png)／[生成結果](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/ring-3d-depth.png)
- **Gray**：明暗の配置を渡す。[条件画像](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/ring-gray.png)／[生成結果](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/ring-3d-gray.png)
- **HED**：柔らかい輪郭を渡す。[条件画像](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/ring-hed.png)／[生成結果](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/ring-3d-hed.png)
- **MLSD**：直線的な構造を渡す。[条件画像](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/ring-mlsd.png)／[生成結果](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/ring-3d-mlsd.png)

MLSDの条件画像には曲面の輪がほとんど残りません。結果に輪が描かれていても、プロンプトでも輪を指定しているため、MLSDだけの効果とは言えません。各方式の向き不向きは、**生成結果と条件画像の両方**を見ると分かりやすくなります。[全画像と生成条件](https://github.com/AiWithYou/aikimi-forge-neo/blob/neo/docs/assets/qwen-image21-fun-controlnet/README.md)を公開しています。

## Inpainting＋Control：輪の材質だけ変える

石造の建築画像から、輪と小さなドームを白く塗ったマスクを作りました。白は再生成、黒は残す指示です。さらに編集元から作ったCanny条件を渡し、建築の配置を保ちながら輪を古びた銅と青緑の古色へ変えます。

![輪と小さなドームを白く指定した編集マスク](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/ring-inpaint-mask.png)

![InpaintingとCanny条件で輪を銅色に変更した結果](https://raw.githubusercontent.com/AiWithYou/aikimi-forge-neo/neo/docs/assets/qwen-image21-fun-controlnet/ring-3d-inpaint.png)

*輪は変化しましたが、小ドームはほぼ元のまま。マスク内の指示も必ず完全に通るわけではありません。*

この出力では、マスク外も元画像と画素単位では一致しません。マスク外のRGB平均絶対差は**3.46/255**でした。外側を厳密に元画像へ戻したい場合は、別の**「マスク範囲外を元画像に固定」**を併用します。これは生成後の合成機能です。

## 試す前に知っておきたいこと

検証機はWindows 11、RTX 3090 24GB。通常版Qwen Image 2.1の**INT8＋CPU退避**、Sparse Attention OFF、40 stepsで生成しました。3DのCanny、Depth、Gray、HED、MLSDは1024×768、人物は768×1024です。各条件1回の作例で、速度や品質の平均値ではありません。計測値とSeedは[測定記録](https://github.com/AiWithYou/aikimi-forge-neo/blob/neo/docs/assets/qwen-image21-fun-controlnet/measurements.json)にあります。

Neoを終了して公式フルモデルを`aikimi-qwen-image21-setup.bat --official-full`で導入し、追加のControlNet重みを次のコマンドで取得します。

```powershell
.\models\Qwen-Image-2.1\worker-env\Scripts\python.exe tools\prepare_qwen21_fun_controlnet.py --download
```

Qwen Image 2.1タブで**通常・INT8**、**CPUへ退避**を選び、Fun ControlNet欄に種類、**前処理済みの制御画像**、強さを指定します。人物の見た目も使うなら、元画像を別の「参照画像」欄へ入れます。タブ内にCannyやPoseの自動抽出はないため、条件画像は事前に用意してください。Inpaintingでは編集元と同じ出力サイズにし、マスクと制御画像を併用します。操作の詳細は[機能ガイド](https://github.com/AiWithYou/aikimi-forge-neo/blob/neo/extensions-builtin/qwen-image21-studio/README.md#fun-controlnet-union--int8)にまとめました。

輪郭や骨格を渡すと、文章で何度も「もう少し左」「腕を上げて」と書いていた部分を画像で指定できます。今回の成果は、人物の同一性やマスクの厳密性が常に保証されたという意味ではありません。それでも、構図を自分で描いてから生成に渡せる選択肢が増えました。

[Aikimi Forge Neo](https://github.com/AiWithYou/aikimi-forge-neo) ／ [公式Fun ControlNet Union](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union) ／ [INT8変換パッチ](https://huggingface.co/Kijai/QwenImage_experimental/tree/04987755e10002ff33e4ea307a811487dddd79d9/model_patches) ／ [利用ライセンス](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union/blob/8a4702014d4dabb5f896fcba917e2ee0a961465f/LICENSE)

#画像生成 #ControlNet #QwenImage #ローカルAI #AikimiForgeNeo
