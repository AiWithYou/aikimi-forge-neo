# Qwen Image 2.1 Consistency LoRA

参照画像の画風・色・照明などを変える際の構図のずれと、対象外の描き直しを抑える、ausbossの追加LoRAです。キャラクターの同一性やマスク外の完全な画素保持を保証するものではありません。

## 使い方

1. Qwen Image 2.1の **LoRA → LoRAの追加方法 → Consistency LoRAを準備（約160 MB）** を押します。作者推奨のstep 1500を取得し、既存の選択を保ったまま強度1.0で追加します。同じ実ファイルをフルパスで選択済みの場合も重複追加せず、既存の強度を保ちます。導入後は通常のLoRA一覧から選べます。自動で常時適用する機能ではありません。
2. 編集元を参照画像へ1枚追加し、**編集元と同じサイズ**を選びます。入力寸法は縦横とも32の倍数に揃えます（例: 768×1152）。通常版Q4_K_M・CPU退避・25 steps・強度1.0から試せます。同じ画像とSeedでLoRAなしと比較してください。作者の基準はINT8です。モデル・Steps・他のLoRAは準備ボタンで変更しません。
3. 色変更・画風変換・照明変更などの編集指示を入力して生成します。トリガーワードは不要です。外す場合はLoRAの×、無効にする場合は強度0を使います。

**新しいポーズや顔の向きへの変更を抑えてしまう場合があります。** 動かしたい編集ではLoRAを外して比較してください。step 2000は位置合わせを強める一方、絵の色が薄くなる場合があります。追加LoRAの併用、高解像度、Fun Acc・Turbo・Sparseとの組み合わせは画質比較が必要です。

## コマンドから準備・検証する

```powershell
# 通常版1500を取得・SHA-256検証
.\venv\Scripts\python.exe tools\prepare_qwen21_consistency_lora.py
# 取得済みファイルを再検証（ダウンロードしない）
.\venv\Scripts\python.exe tools\prepare_qwen21_consistency_lora.py --verify
# 強い位置合わせの2000を任意で追加
.\venv\Scripts\python.exe tools\prepare_qwen21_consistency_lora.py --version 2000
```

保存先は `models/Qwen-Image-2.1/loras/ausboss--Qwen-Image-2.1-Consistency-LoRA/` です。固定revisionの重みだけを取得し、生成時にはダウンロードしません。作者のコードやComfyUIノードは実行せず、既存の追加LoRAローダーで適用します。各ファイルは159,436,496 bytesです。ファイル名を変えたり外部パスから指定した場合も、配布版のSHA-256で識別します。

選択すると学習元・推奨条件とポーズ変更の制限を表示します。学習元のConvRot INT8や作者の推奨は本体の指定・使用条件ではなく、通常版の各精度やTurboで現在の設定のまま適用できます。実際の対象層・形状・ファイル内容は検証します。1枚の参照画像と出力が同じ寸法の場合は、参照画像のVAE処理もその面積に合わせ、固定1024px相当への拡縮を避けます。複数参照や異なる出力寸法で作者の位置保持効果を再現したものとは扱いません。

## 出典と検証範囲

確認日: 2026-10-05。配布元は [ausboss/Qwen-Image-2.1-Consistency-LoRA](https://huggingface.co/ausboss/Qwen-Image-2.1-Consistency-LoRA/blob/8f05b0fa027d517fa396fb31b71e0eaf48110e89/README.md)、固定revisionは `8f05b0fa027d517fa396fb31b71e0eaf48110e89` です。

| 版 | ファイル | SHA-256 |
|---|---|---|
| 1500 | `qwen-image-2.1-consistency.safetensors` | `4f44ada1be2189cc23b3d010f9603543403f48454e2f76842a3d30109b20bd63` |
| 2000 | `qwen-image-2.1-consistency-2000.safetensors` | `a0bf043edc4695b1661a0e768e49a7fdff9a656b7d4313203d590e0262b64beb` |

作者の検証はComfy-Org INT8 ConvRot・約1MP・25 steps・CFG 1・Euler/simpleです。本プロジェクトの標準INT8は実行時のbitsandbytes量子化方式です。ローカルINT8 ConvRot本体も指定できますが、作者と同じ重みや同じサンプラーでの再現とは扱いません。作者はTurbo・2MP・CFG 1超を未検証としています。ライセンスはQwen Research Licenseで、非商用の研究・評価用途に限られます。

step 2000の実機生成は未検証です。比較した作例ではLoRAの有無による差が小さい場合があり、効果や対象外領域の保持は保証できません。[比較図と条件](assets/qwen21-consistency-comparison/2026-10-06/index.html)を参照してください。
