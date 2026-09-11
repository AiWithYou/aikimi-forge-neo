# Anima 3.8B for Aikimi Studio Neo

これは、Anima 3.8Bの52層DiTとQwen3.5 4B conditioningを、Aikimi Studio Neoの通常の`txt2img`と`img2img`で利用するためのbuilt-in Extensionです。最新版のv1.1はSemantic Connector v2をcheckpointに内包し、denoiseの各stepへ意味特徴を反映します。旧v1のprogressive cross-attention adapterにも引き続き対応します。

実装は[GumGum10/forge-anima-3.8B](https://github.com/GumGum10/forge-anima-3.8B)のcommit `59c27e5702f95c13dc5c08953637371d4749a034`を基にしています。Aikimi Studio Neoへの同梱にあたり、52層checkpointの必須検査、offline tokenizerの優先読込、INT8 ConvRotセットアップ、28↔40↔52 LoRA変換、省VRAM処理を統合しました。

## セットアップ

リポジトリ直下の次のファイルを実行してください。

```text
download_anima38_v11_int8_convrot_models.bat
```

セットアップは、[lylogummy/Anima-3.8B](https://huggingface.co/lylogummy/Anima-3.8B)のv1.1固定revisionから必要ファイルを取得します。一時的に取得したBF16 bundleのうち、DiT本体にある主要なattentionとMLPの520行列だけがINT8 ConvRotへの変換対象です。Semantic Connector v2、AdaLN、埋め込み、入出力、正規化層はBF16のまま保持されます。完成checkpointには、再検証用のSHA-256 sidecarが付きます。

| 役割 | ファイル | 精度 |
|---|---|---|
| v1.1 bundle | `models/Stable-diffusion/Anima-3.8B-v1.1-int8-convrot.safetensors` | DiT主要520行列がINT8 ConvRot、Connector v2を含む残りがBF16 |
| 意味encoder | `models/text_encoder/qwen35_4b.safetensors` | 配布済みのFP8/BF16混合 |
| Anima標準encoder | `models/text_encoder/qwen_3_06b_base.safetensors` | Anima共通 |
| VAE | `models/VAE/qwen_image_vae.safetensors` | Anima共通 |

旧v1を使う場合は、従来の`download_anima38_int8_convrot_models.bat`で`Anima-3.8B-int8-convrot.safetensors`と`Anima-3.8B-expanded_adapter.safetensors`を用意します。

Qwen3.5 tokenizerはextension内に同梱しています。生成時にHugging Face cacheや外部ネットワークへ接続しません。

## 使い方

Forgeを再起動し、次の値を選びます。

```text
Preset: anima
Checkpoint: Anima-3.8B-v1.1-int8-convrot.safetensors
VAE / Text Encoder:
  qwen_image_vae.safetensors
  qwen_3_06b_base.safetensors
Diffusion in Low Bits: Automatic
```

v1.1 bundleはmetadataから自動検出されるため、`Anima 3.8B (Qwen3.5 / v2)`を開かなくてもSemantic Connector v2が有効になります。v2のstrengthは学習時の`1.0`に固定され、旧v1用のadapter選択とstrengthは無視されます。

旧v1 checkpointでは、パネルを開いて有効にし、専用adapterを選びます。旧v1の`Adapter strength`は`1.0`が学習時の基準です。`0.0`を指定すると、Anima標準conditioningだけが使われます。

negative promptは既定でAnima標準encoderだけを使います。`Use adapter on negative prompt`を有効にすると、negative側のstrengthを個別に指定できます。

通常のAnima LoRAを使う場合は、同じパネルの`Standard Anima LoRA`で1件目を選択し、`Standard LoRA strength`を指定します。`Additional Standard Anima LoRAs`を開けば、さらに3件のLoRAと個別のstrengthを追加可能です。候補は`models/Lora`にある完全な28層・40層・52層のsafetensors LoRAに限定し、28層版と40層版は52層へ自動展開してからForge標準LoRA経路で適用します。起動後にLoRAを追加した場合は`Refresh standard Anima LoRAs`を押してください。5件以上を組み合わせる場合は、LoRAタブまたは`<lora:name:weight>`タグを利用します。

Qwen3.5の約1.2 GiBのembedding tableはCPUに保持し、promptで使うtoken行だけをGPUへ送ります。同じpromptでSeedだけを変えた場合は、ForgeのPersistent Cond Cacheがbundleまたはadapterのweight fingerprintを含む条件でconditioningを再利用します。

v1.1は、Anima標準encoderとQwen3.5を順番に実行し、conditioningをRAMへ移した後で両encoderをsampling前に解放します。Semantic Connector v2はdenoiseの各stepで動くため、Forgeのsampling modelとして管理されます。旧v1は通常、text encoderをGPUへ残す方が高速です。旧v1でVRAMを優先する場合は、`Low VRAM: offload text encoders before sampling`を有効にしてください。

最初の生成条件は次を目安にしてください。

```text
Resolution: 832x1216前後の約1MP
Sampler / Scheduler: res_multistep + Beta
Steps: 28-50
CFG: 4-7
Semantic Connector v2: 自動、strength 1.0固定
```

positive promptは、品質tagの後に`Description:`を置いて自然文を続けます。複数人物を指定する場合は代名詞を避け、人物名、位置、動作を文ごとに明示してください。

```text
masterpiece, best quality, high quality, newest,
Description:
A fox-girl jumps over a high fence.
```

## 読込と安全条件

- v1.1 bundleは、safetensors metadataの`anima_3_8b_semantic_connector_v2_bundle`とbundle formatを両方検査します。
- 旧v1では、safetensors metadataに`anima_progressive_qwen35_cross_adapter_v1`を持つadapterだけを検出対象とします。
- Qwen3.5の候補は、ファイル名に`qwen35_4b`、`qwen3.5-4b`、`qwen3_5_4b`のいずれかを含むものです。
- 52層以外のAnima checkpointにはadapterを適用せず、組み合わせが誤っていれば理由を表示して停止する設計です。
- Qwen3.5はAnima標準encoderに追加して読み込むため、conditioningの待ち時間とRAM使用量が増えます。
- Qwen3.5の未使用final projectionと中間tensor copyを省き、paddingなしpromptではPyTorchのcausal SDPA経路を使う実装です。
- 同一promptでも、adapter、Qwen3.5、strength、negative設定の変更を検出した時点でconditioning cacheを破棄します。
- 画像の生成parameterでは、bundle名、connector architecture、固定strength、任意のnegative strengthが記録対象です。旧v1の場合は、adapter名とstrengthを記録します。
- パネルで選んだ各Anima LoRAとstrengthは生成parameterへまとめ、Forge標準のLoRA hashも併記します。
- encoder offloadを有効にした場合、解放量も生成parameterへの記録対象です。
- 旧v1ではextensionを無効にすると、Forge標準のAnima処理を変更しません。v1.1 bundleはパネルを閉じた状態でも自動的に有効になります。

旧v1のRTX 3090実測では、832×1216、32 Steps、CFG 7.0、同一prompt・Seedのwall timeが29.770秒から25.010秒へ短縮しました。平均RGB差は5.19/255で、人物、透明レインコート、蝶、左右の花、じょうろ、温室の配置を目視で維持していることを確認しました。この数値はv1.1の性能値ではありません。

## ライセンスと由来

extensionコードのライセンスは[LICENSE](LICENSE)を参照してください。同梱tokenizerの由来は[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)に記録しています。モデルweightには、Anima、Anima-2.9B、Qwen3.5、配布repositoryが定める条件が別途適用されます。
