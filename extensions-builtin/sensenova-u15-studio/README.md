# SenseNova U1.5 Studio

SenseNova U1.5 Studioは、正式版の[sensenova/SenseNova-U1.5-8B-MoT](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT)をAikimi Studio Neoから実行する専用GUIです。テキスト生成と、単一または複数の参照画像を使った画像編集に対応します。

既定のweightは、正式版を基にしたコミュニティ配布のINT8 ConvRotです。SenseNova固有の画像token化、画像decoder、三分岐guidance、生成ループを保つため、Forgeの通常checkpointやKSamplerには接続しません。Forgeは入力検証、画面、保存、キャンセルを担当し、生成は隔離workerが専用ランタイムを使って実行します。

## セットアップ

Aikimi Studio Neoのリポジトリ直下にある次のファイルを実行してください。

```text
download_sensenova_u15_int8.bat
```

既定のセットアップ内容は次のとおりです。

| 項目 | 内容 |
|---|---|
| 対象モデル | 正式版`SenseNova-U1.5-8B-MoT` |
| ConvRotランタイム | `starsFriday/ComfyUI-SenseNova`の固定revision `e6dfd45762eb46f805067fe079c14bcb643ccccd` |
| checkpoint | `SenseNova-U1.5-8B-MoT-pruned-int8_convrot.safetensors` |
| 配布元 | `joyfox/SenseNova-U1.5-8B-MoT-FP8` |
| 配布元revision | `57de22ad4e2fc24c77f56dfe45dbb87a60dfebee` |
| ファイルサイズ | 17,734,813,848 bytes（約16.52 GiB） |
| SHA-256 | `cf6ed9ee3be516612b7fe083edfc7c9dd5d059cc759e300d2cf1f2726c0d250e` |
| 量子化構成 | `int8_tensorwise`、ConvRot group size 256、588 Linear層 |
| 公式高速LoRA | `SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors`、814,867,236 bytes |
| LoRA revision | `e909f4636d119d65fe4cba8770c19daff2ac102e` |
| LoRA SHA-256 | `3ef32180cdf1e30a870a83f4f136e897ea50b7ee467f863d75633464ebb25708` |

checkpointは32 MiB単位に分け、最大16接続で取得します。中断した場合は同じbatを再実行すると、完成済み部分と未完了chunkの先頭から続行できます。完成ファイルは、サイズ、safetensors内のConvRot署名、SHA-256の各検証値がすべて同じ場合だけ利用可能です。

ランタイムだけを準備する場合はPowerShell 7から次を実行します。

```powershell
.\download_sensenova_u15_int8.ps1 -RuntimeOnly
```

このコマンドは`models/SenseNova-U1/worker-env`へSenseNova専用のPython環境も用意します。固定ランタイムが使うTransformers 4.57.6とhuggingface-hub 0.36.2を、`tools/requirements-sensenova.txt`に従って導入する構成です。Forge本体は自身の`venv`を使います。モデルファイルは共通で、専用環境には依存パッケージ用のディスク容量が必要です。

専用Pythonや固定バージョンのパッケージが不足していると、画面に未準備と表示されます。上の`-RuntimeOnly`で導入・修復してください。worker自身も実行中のバージョンを確認してからモデルを読み込みます。

モデル資産のcheckpointと公式8-Step LoRAだけを取得または再検証する場合は、次を実行します。

```powershell
.\download_sensenova_u15_int8.ps1 -ModelOnly
```

## テキストから生成

1. `SenseNova U1.5`タブを開き、`テキストから生成`を選んでください。
2. `公式8-Step · 高速T2I`または`Quality 50-Step`を選び、解像度とSeedを確認します。
3. 初回は`1024 × 1024 · 動作確認`から始めると、読み込み確認を短時間で終えられます。
4. `画像を生成`を押すとForgeモデルが退避し、その後に専用workerが起動する流れです。

高速プロファイルでは、SenseNova公式の蒸留LoRA、8 Steps、CFG 1.0、Timestep Shift 3.0を一体で固定し、品質プロファイルでは基本モデルを50 Steps、CFG 4.0、Timestep Shift 3.0で実行します。公式8-Step LoRAはテキスト生成専用のため、画像編集へ切り替えた時点で品質プロファイルへ戻る設計です。

[公式8-Step LoRAの配布ページ](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-LoRAs)でも、少ステップ用LoRAはテキスト生成、画像編集は基本モデルの手順として案内されています。2026-09-08には、正式版INT8 ConvRot＋公式LoRAで2048×2048・8 Stepsの生成を実機確認しました。参照優先モードと従来の24GB SafeでPNGの画素は同一で、ピークVRAMは約2.27 GiBと約3.00 GiBでした。これは両VRAMモードの比較であり、50-Step基本モデルとの画質同等性を示すものではありません。

公式8-Step LoRAが未準備の場合は、Quality 50-Stepで起動します。`準備状況を再確認`は、現在使えるプロファイルとSteps・CFGの調整値を保持します。画像編集中に高速T2Iへ切り替わることはありません。LoRAの導入後に高速生成を使う場合は、テキスト生成で公式8-Stepを選んでください。

24GB Safeの出力上限は2048² pixelsで、2048×2048や同等画素数の公式解像度bucketを選べます。

## 複数参照画像を編集

1. まず`複数画像を編集`へ切り替えてください。
2. まとめて登録する場合は、`複数画像を一括選択`でファイルを選び、`選択ファイルを一括追加`を押してください。1枚ずつの追加・差し替えには`追加・差し替え画像`を使います。
3. サムネイルの番号と寸法を確認したうえで、対象画像を選び、`← 前へ`・`後ろへ →`で移動してください。選択位置に応じて移動・差し替え・削除ボタンが有効になります。
4. プロンプトの補助には`参照の使い方`を使ってください。被写体の保持、画風・色調、衣装、背景の用途を選んで`役割の文をプロンプトへ追記`を押すと、既存の指示に補助文が加わります。同じ文は重複して追加しません。
5. 並べ替えると画像番号も変わります。プロンプトの`Image-1`・`Image-2`が意図した画像を指すか確認してください。プロンプト本文は自動で書き換えません。
6. 出力解像度の`元の入力1枚目を基準 · 約1MP`は、縦横比を基準に小さく試すための設定です。従来の約4MPも選べます。入力画像予算を確認して生成してください。
7. 完成後に`結果をImage-1にして続ける`を押すと、生成結果が先頭の参照になり、画像編集モードへ移ります。Image-2以降とプロンプトは保持します。

参照の上限は64枚です。一括追加では登録できる分だけを読み込み、未追加ファイルを選択欄に保持します。追加数と残り枚数は画面の通知で確認してください。1枚の画素数は100MPまでで、上限を超えた画像は画素データを展開する前に拒否します。

モデルとGUIの上限は64枚で、参照画像は表示順を保ったまま`it2i_generate`へ渡されます。既定の`参照優先`は、参照8枚以下、出力2048² pixels以下、参照入力1枚あたり約1.05MPまでに対応します。`詳細設定`でVRAMモードを選び、`参照画像の情報量`で各画像の解像度を指定してください。初期値は各約0.26MPです。縦横比を保って縮小し、32pxグリッドの余白には画像端の画素を延長します。`24GB Safe`は従来の参照2枚・各約0.26MPの範囲を保ちます。これらの枠を超える実験には`Uncapped streaming`を明示的に選んでください。

参照優先では、参照KVを元の精度のままCPUに保持し、現在の層のGPUバッファだけを確保します。参照読込のAttentionも2ヘッドずつ計算します。重みのINT8 ConvRot、BF16演算、参照の順序、guidanceの計算式は共通です。CPU RAMとPCIe転送を使ってVRAMを減らす構成です。

RTX 3090での2026-09-08の比較では、2048×2048出力・2 Steps・CFG 4・Image CFG 1・同じseedで次の結果でした。GiBはTorchのpeak allocatedで、デスクトップ表示や他プロセスの使用量を含みません。2 Stepsはメモリと出力一致の検証条件です。

| 参照条件 | 従来のstreaming | 参照優先 | 出力の比較 |
|---|---:|---:|---|
| 2枚・各約0.26MP | 3.99 GiB | 2.21 GiB | PNG SHA-256一致 |
| 8枚・各約0.26MP | 5.08 GiB | 2.23 GiB | PNG SHA-256一致 |
| 8枚・各約1.05MP | GPU全体が約24GBに達し、10分以上進まないため中断 | 4.31 GiB | 参照優先で完了、従来版との一致は未確認 |

参照優先の3条件はOOM・allocation retryともに0でした。8枚・各約0.26MPのsamplingは20.508秒から26.017秒に増え、8枚・各約1.05MPは41.132秒でした。検証データはローカルの`outputs/sensenova-vram-audit/baseline.json`と`final.json`に保存しています。

通常生成は4枚・各約1.05MP・2432×1664・50 Stepsでも確認しました。人物と衣装、赤いスクーター、夕暮れの海岸、黒猫を別々の参照から指定し、4つの役割が完成画像に反映されていました。ピークは2.54 GiB、samplingは323.3秒、OOM・allocation retryは0です。完成画像と設定は`outputs/sensenova-vram-audit/quality_4refs_1024/`に保存しています。

参照画像を増やすほど、画像token列、CPU転送量、処理時間が増えます。RTX 3090で参照2枚、入力各512²、出力2048²、1 Stepを実測したところ、分岐転送の改善前はworker処理が28.933秒、改善後は19.580秒でした。新旧PNGのSHA-256は一致しています。改善後のTorch peakはallocated 4,202,799,104 bytes、reserved 4,573,888,512 bytesで、OOMとallocation retryは0でした。

## INT8 ConvRot

このcheckpointは、588個のLinear層を`int8_tensorwise`形式で保持し、各層をHadamard回転を戻しながらBF16へ復号する構成です。Lowモードでは、prefix処理に必要な理解分岐とdenoiseに必要な生成分岐を識別し、そのforwardで使うweightだけをGPUへ移して演算後にCPU参照へ戻します。activationと演算精度は変更しません。

正式版のモデル構成とtokenizerを使いますが、INT8変換weightと専用ローダーはコミュニティが管理しています。公式BF16との品質一致や数値一致は保証されません。このStudioは固定した配布ファイルだけを完全性確認し、任意のGGUF、bitsandbytes、別のsafetensorsを自動変換しません。

pruned checkpointはテキスト出力用の`language_model.lm_head`を削除しています。テキストからの画像生成と画像編集には使えますが、Think mode、VQAのテキスト回答、テキストと画像の交互出力には使えません。

## VRAMモード

| モード | 用途 |
|---|---|
| `24GB Safe` | Transformerを1層ずつ分岐別に転送し、出力2048² pixels以下、参照2枚以下、参照入力1枚あたり約0.26MPを強制するRTX 3090向け設定。被写体の縦横比は保護する。 |
| `Uncapped streaming` | 同じ層streamingを使いながら画素数制限を解除。大容量GPU向けの実験設定で、実行可能性は条件ごとに確認が必要。 |
| `Full GPU` | 量子化weight全体をGPUへ配置する実験設定。必要量は解像度と参照数で変わり、24GB GPUでは利用不可。 |

生成開始時は通常のForge画像モデルを退避してVRAMを確保し、SenseNova workerは1回の生成後にweightとactivationをOSへ返します。この設計によりメモリを確実に解放できますが、次回の生成時にはモデルを先頭から読み込みます。

## 保存とキャンセル

完成ファイルは次へ保存します。

```text
outputs/sensenova_u15/sensenova_u15_YYYYMMDD_HHMMSS_<job>.png
outputs/sensenova_u15/sensenova_u15_YYYYMMDD_HHMMSS_<job>.json
```

JSONにはモデルID、固定revision、checkpoint、量子化方式、生成プロファイル、LoRA完全性情報、ロードしたINT8層数、解像度、Steps、CFG、Seed、入力画像数、入力順、処理区間、Torch peak、分岐別転送量、出力SHA-256を記録します。参照画像本体は一時jobフォルダーへ複製し、生成終了またはキャンセル後に削除します。

キャンセルは隔離workerを停止します。モデル読み込み中とsampling中のどちらでも停止できますが、次回はモデルを先頭から読み込みます。

## 既知の制限

- checkpointだけで約16.52 GiBあり、CPU RAM、GPU activation、画像token、decoderにもメモリが必要です。
- 参照画像を増やすほど、1枚ごとの入力解像度は下がります。
- 多数の参照画像を使う場合は、保持対象と変更対象をプロンプト内で明示してください。
- `FlashAttention`を指定した場合、互換wheelがない環境では生成前に失敗します。通常は`自動`または`PyTorch SDPA`を使ってください。
- 公式8-Step LoRAはテキスト生成専用です。画像編集にはQuality 50-Stepを使ってください。
- Prompt Enhanceは外部モデル呼び出しを伴うため、自動実行しません。

## 出典とライセンス

- [SenseNova U1.5正式版モデル](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT)
- [SenseNova U1.5公式8-Step LoRA](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT-LoRAs)
- [INT8 ConvRot配布リポジトリ](https://huggingface.co/joyfox/SenseNova-U1.5-8B-MoT-FP8)
- [ConvRot対応ランタイム](https://github.com/starsFriday/ComfyUI-SenseNova)
- [OpenSenseNova/SenseNova-U1](https://github.com/OpenSenseNova/SenseNova-U1)

ランタイムコードはApache-2.0です。モデルweightは配布元に記載されたライセンスへ従ってください。
