# MOSS-Image-SFT

[MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) は、OpenMOSSの5B画像生成モデルです。自然文、タグ列、JSONレイアウトから画像を生成できます。Forge Neoでは **MOSS Image** Studioを使用します。

## 導入

```powershell
.\aikimi-moss-image-setup.bat
```

既定ではSFTの画像生成本体とテキストエンコーダーをINT8 ConvRotで取得します。VAEは公式のFP32を保持します。モデルは`models/MOSS-Image/`、MOSSに標準対応した専用の固定ComfyUIは`repositories/moss-image/`へ導入します。NVIDIA CUDA GPUと、セットアップ済みのForge Neo環境が必要です。

画像生成本体はINT8／BF16、テキストエンコーダーはINT8／W4A8／BF16を個別に選択できます。量子化版はコミュニティ配布で、BF16版とは細部が変わる可能性があります。選んだ形式が未導入の場合、自動で別の形式へ切り替えません。

標準のINT8構成は約14.1GBです。テキストをW4A8にする場合は次のように導入し、Studioでも同じ形式を選択します。この構成の重みは約11.1GBです。これらはファイル容量で、生成中のVRAM使用量とは異なります。

```powershell
.\aikimi-moss-image-setup.bat --text-precision w4a8
# BF16本体とINT8テキストを導入する場合
.\aikimi-moss-image-setup.bat --precision bf16 --text-precision int8
```

INT8本体・エンコーダーは[Abiray版](https://huggingface.co/Abiray/MOSS-Image-SFT-INT8-convrot)、W4A8エンコーダーは[nif0版](https://huggingface.co/nif0/MOSS-Image-SFT-ComfyUI)を固定リビジョンとSHA-256で検証します。

## 生成

1. **MOSS Image** を開き、プロンプトを入力します。英語の自然文またはタグ列を利用できます。
2. 解像度、ステップ数、CFG、Seedを設定し、生成します。標準値は1024×1024、50ステップ、CFG 3、Shift 5、負条件側の22層スキップです。
3. 生成画像と条件を保存したPNGをダウンロードできます。中断ボタンで処理を停止できます。

このStudioでは幅と高さを512〜2048の16の倍数で指定します。負のプロンプトと負条件側の層スキップを変更できます。Seedを固定すると同じ条件で再生成できます。プロンプトは自動書き換えしません。公式のチャット形式を含めて4096トークンを超える入力は、途中で切り捨てずエラーにします。

JSONレイアウトもプロンプト欄にそのまま入力できます。公式形式は `summary.caption`、`items`、`bbox_format: "yxyx"`、`bbox_coord_space: "abs_centralized"` です。座標は画像の中心を原点とするピクセル値で、生成時の解像度に合わせます。

## 作例

<img src="assets/moss-image-v4.9.0/anime-portrait.png" alt="MOSS-Image-SFTで生成した赤いマフラーのアニメ少女" width="416">

832×1216、50 steps、CFG 3、Shift 5、負条件の22層スキップON、Seed 2026101701、負のプロンプト空欄、LoRAなし。本体・テキストはINT8 ConvRot、VAEはFP32です。元のPNGにも生成条件を保存しています。

```text
A single young woman with short black hair and a red scarf, half-body portrait, looking at the viewer with a gentle smile, soft daylight, refined Japanese anime illustration, clean line art, detailed blue eyes, warm gray background.
```

RTX 3090（24GB）、RAM 64GB、Windows 11、ドライバー610.74、専用PyTorch 2.13.0+cu130で確認しました。この作例の生成処理は約81秒、観測したGPU全体のVRAMピークは12,171MiBでした。初回のファイル整合性確認は別途時間がかかります。解像度・文章の長さ・LoRA・ほかのアプリによって必要量は変わります。実GPUで確認した保存形式はINT8本体＋INT8テキストです。

## LoRA

MOSSの画像生成本体と互換の線形LoRAを`models/MOSS-Image/loras/`へ置き、一覧を更新して選びます。最大16件の複数選択と個別の強度調整に対応します。通常の`lora_A/lora_B`または`lora_down/lora_up`形式を読み、対象と形状が一致することを検証します。学習元の名前だけで拒否したり、モデル・Steps・CFGを自動変更したりしません。

量子化した本体にもLoRAの差分を適用します。保存した量子化モデルは書き換えず、変更したLoRAの選択・強度は次の生成に反映します。PNGと生成条件JSONに実際のLoRA名・SHA-256・強度を記録します。MOSS以外のLoRAや未対応の追加方式は、対応する対象を確認できなければエラーになります。

LoRA適用時は重みを一時的に復元して計算するため、追加のメモリーが必要です。ComfyUIのサンプリングを使用するため、同じSeedと設定でも元のDiffusers出力との画素一致は保証しません。

このStudioはSFTのテキストからの画像生成を扱います。参照画像編集と蒸留版用の少ステップ設定は対応範囲に含みません。文字の正確さやレイアウトの再現は生成結果を確認してください。

モデルと公式推論コードはApache-2.0です。詳細は[公式ユーザーガイド](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md)を参照してください。
