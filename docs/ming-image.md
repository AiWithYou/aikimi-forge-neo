# Ming Image 0.2 Studio

Ming Image タブでデザイン画像を生成・編集します。プロンプトには自然文や Figma 風の構造化 JSON を入力できます。

## 使うモデル

**VRAM 24 GB 以上なら `vram24g_hq`、16 GB なら `vram16g_hq` を使います。** どちらも [inclusionAI の BF16 原本](https://huggingface.co/inclusionAI/Ming-Image-0.2-Design)（80 GB 級 GPU 向け）を Aikimi が量子化し、[Aikimi/Ming-Image-0.2-Design-HQ-Quant](https://huggingface.co/Aikimi/Ming-Image-0.2-Design-HQ-Quant) で公開したモデルです。Studio とセットアップの既定は `vram24g_hq` です。

```powershell
aikimi-ming-image-setup.bat --profile vram24g_hq   # 24 GB 以上（約 26.7 GB）
aikimi-ming-image-setup.bat --profile vram16g_hq   # 16 GB（約 18.2 GB）
```

セットアップメニューの 12 番でも選べます。Comfy-Org 配布の `comfy_org_w4a8`（約 19.2 GB）と `comfy_org_int8`（約 25.9 GB）は比較・互換用の選択肢です。

| プロファイル | 対象 GPU | DiT | テキストエンコーダー | VAE | 合計 |
|---|---|---|---|---|---:|
| `vram24g_hq`（推奨） | 24 GB 以上 | 8/16 bit・8.98 GB | MoE 5/6 bit・17.50 GB | BF16・0.26 GB | 約 26.7 GB |
| `vram16g_hq` | 16 GB | 8/16 bit・6.50 GB | MoE 3/4 bit・11.45 GB | BF16・0.26 GB | 約 18.2 GB |
| 参考：BF16 原本 | 80 GB 級 | 16 bit・12.3 GB | 16 bit・約 40 GB | BF16・0.25 GB | 約 52.9 GB |

### 原本との比較

512×512、12 steps、同じ seed で BF16 原本と生成を比べた結果です。PSNR・SSIM は高いほど、LPIPS は低いほど原本に近いことを示します。

| 評価例 | `vram24g_hq` PSNR / SSIM / LPIPS | `vram16g_hq` PSNR / SSIM / LPIPS |
|---|---|---|
| 表紙デザイン（日本語見出し） | 30.74 dB / 0.984 / 0.010 | 25.06 dB / 0.937 / 0.061 |
| 人物写真 | 21.33 dB / 0.876 / 0.087 | 19.12 dB / 0.783 / 0.128 |
| 透過ステッカー | 29.49 dB / 0.963 / 0.017 | 21.10 dB / 0.867 / 0.056 |
| 参照画像による編集 | 36.61 dB / 0.988 / 0.005 | 29.89 dB / 0.960 / 0.016 |

![表紙デザインの比較（左から BF16 原本、vram24g_hq、vram16g_hq）](https://huggingface.co/Aikimi/Ming-Image-0.2-Design-HQ-Quant/resolve/main/comparisons/final_vs_original_0.png)
![人物写真の比較（左から BF16 原本、vram24g_hq、vram16g_hq）](https://huggingface.co/Aikimi/Ming-Image-0.2-Design-HQ-Quant/resolve/main/comparisons/final_vs_original_1.png)

`vram24g_hq` は構図・色・文字配置が原本とほぼ一致します。ただし人物写真では右肩に原本にないギターの一部が写りました。`vram16g_hq` も構図と見出しは保ちますが、細部の質感に差が出ます。透過ステッカーのアルファ誤差（MAE）は 24 GB 版 0.0060、16 GB 版 0.0198 でした。4 例での比較であり、すべてのプロンプトでの画質を保証するものではありません。残りの比較画像と同じビット数の標準量子化との比較は [モデルカード](https://huggingface.co/Aikimi/Ming-Image-0.2-Design-HQ-Quant) にあります。

## 生成条件

標準は 2048×2048、12 steps、CFG 1.0 です。画面と REST API の CFG 指定範囲は 1.0〜20.0 です。横長には 2560×1440、2432×1824、縦長には 1664×2496、試作には 1024×1024 を選べます。幅と高さは 512〜3072 の16の倍数です。Euler と simple スケジューラーを使い、ModelSamplingFlux の max shift（既定 1.15）と base shift（0.5）を解像度に合わせて適用します。負のプロンプトは CFG が 1 を超える場合に使います。

編集では参照画像を使用順に最大8枚指定します。透過背景をオンにすると、デコードされたアルファを PNG に保存します。独自の Ming 用 safetensors は `models/Ming-Image/checkpoints/` に配置します。

モデルは MIT ライセンスです。[公式モデルカード](https://huggingface.co/inclusionAI/Ming-Image-0.2-Design) · [Aikimi HQ 量子化版](https://huggingface.co/Aikimi/Ming-Image-0.2-Design-HQ-Quant) · [Comfy-Org 配布モデル](https://huggingface.co/Comfy-Org/Ming-Image)

## モデルの容量と互換性

重みとは別に、専用の Python 環境とダウンロード用の空き容量も必要です。準備後はまず 1024×1024 で生成してください。解像度と参照枚数を増やすと、実行時に必要なメモリーが増えます。BF16 の 80 GiB GPU 構成については [inclusionAI/Ming-Image-0.2-Design の公式モデルカード](https://huggingface.co/inclusionAI/Ming-Image-0.2-Design) を参照してください。

旧 Studio と同じ PC では、専用 Python 3.13 と ComfyUI を別のサブフォルダーに導入します。セットアップ前の容量確認は `aikimi-ming-image-setup.bat --profile vram24g_hq --dry-run`、環境のみの準備は `--runtime-only` です。

編集するには **img2img** を選び、参照を順に追加します。直前の出力は **img2img へ送る** で再利用できます。参照は出力キャンバスの寸法に合わせてリサイズされます。透過スイッチはプロンプトに背景指定を加えます。透過指定時には RGBA 出力が必要です。アルファが全画素 255 の画像も保存できます。

生成条件は PNG と `result.json` に記録し、プロンプトの認証情報やローカルパスを伏せます。元の要求と参照画像は非公開の作業領域に一時保存します。中断に応答しない場合はワーカーを停止してから生成途中の画像を削除します。

## REST API

`POST /sdapi/v1/ming-image/jobs` に次の JSON を送るとジョブ ID が返ります。

```json
{"profile":"vram24g_hq","prompt":"A fern leaf","width":1024,"height":1024,"transparent":true}
```

`GET /sdapi/v1/ming-image/jobs/{id}` で状態を取得します。完了時の `image` は PNG の base64 です。中断には `POST /sdapi/v1/ming-image/jobs/{id}/cancel` を使います。編集時は `mode` に `img2img`、`images` に参照画像を base64 の配列で指定してください。Forge の API 認証設定が適用されます。

[ComfyUI の編集ノード](https://github.com/Comfy-Org/ComfyUI/blob/7c78b5641ab2d6cb7f7766a183a8a43a59de0a8b/comfy_extras/nodes_ming.py) · [Ming-Image 0.1 Design のワークフロー](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_ming_image_01_design_t2i.json)
