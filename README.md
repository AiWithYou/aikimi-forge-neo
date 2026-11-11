# Aikimi Forge Neo

[English](README.en.md) · 日本語

**[v6.1.0](https://github.com/AiWithYou/aikimi-forge-neo/tree/v6.1.0)** · [最新リリース v6.1.0](https://github.com/AiWithYou/aikimi-forge-neo/releases/tag/v6.1.0) · [変更履歴](CHANGELOG.md)

<img src="assets/aikimi/pet.png" alt="ちびあいきみ" width="112" align="right">

**画像の生成・編集、音声付き動画の作成、作曲、画像や文章の評価をひとつのWebUIで扱える、Windows向けのForge Neo派生版です。** モデルの導入や高解像度化、画像の仕上げ機能も備えています。

同じ系列を追加学習したモデル本体（Full FT）も選べます。画像生成ではForgeの共通画面、P-H3 Image、MOSS Image、Qwen Image 2.1、Iris、FLUX 3に対応しています。H3動画、Clef、Irodori TTS、YuE2でも互換の本体を選べます。LoRAはモデル本体に重ねて使う追加データです。通常版・INT8・W4A8など、選べる保存形式や必要な共通部品は系列ごとに異なります。[モデル・LoRAの配置と対応形式](docs/local-models.md)

## できること

- **画像を作る・編集する**：MiniMax-H3-Image、P-H3 Image、MOSS Image、Qwen Image 2.1、Iris、FLUX 3などを目的に合わせて選択できます。P-H3は編集対象と参照画像を合わせて最大8枚まで使えます。
- **大きな画像を仕上げる**：Krea2の高解像度ワークフローやMOSSのHires.fix、各種アップスケーラーを使えます。
- **動画・音声を作る**：MiniMax H3で音声付き動画、YuE2で音楽、Irodori TTSで日本語音声を生成できます。
- **画像を整理する**：Clefで画像や文章の評価・仕分けを行えます。

P-H3の初回セットアップは、WebUIの **H3 Image → P-H3 Image → 準備を開く → 環境とモデルを準備** から行います。標準モデル4ファイルの合計サイズは約29.8 GBです。モデルおよび生成物には地域制限が適用されるため、SNSへ公開する前に利用条件を確認してください。[使い方と利用条件](extensions-builtin/minimax-h3-studio/P_H3_IMAGE_GUIDE.md)

SenseNova U1.5、Nanosaur2、Ming Image Designの画面とセットアップは本版から対象外です。

[セットアップ方法](#セットアップ方法) · [主な機能](#主な機能) · [LoRAの使い方](#手動でダウンロードしたloraを使う) · [更新方法](#更新方法) · [困ったとき](#トラブルシューティング)

## セットアップ方法

<a id="quick-start"></a>

### 必要な環境

先に次の環境を用意してください。以下のコマンドはPowerShell 7（`pwsh`）で実行します。

- Windows 11（主な対応環境）
- NVIDIA GPUとCUDA 13.0対応のドライバー
- Git、Python 3.13、PowerShell 7、[uv](https://docs.astral.sh/uv/getting-started/installation/)

必要なGPUメモリと、モデルの保存・変換に使う空き容量はモデルごとに異なります。**YuE2 Musicには別途Python 3.12が必要です。**

### はじめて使う場合

```powershell
git clone --branch neo https://github.com/AiWithYou/aikimi-forge-neo.git
cd aikimi-forge-neo
```

使いたいモデルを選び、下の表の方法で準備します。P-H3は先にWebUIを起動し、画面から準備します。**最初は使うモデルだけ導入すれば始められます。** モデル本体はリポジトリに含まれません。

<a id="model-setup"></a>

### モデルを選んで導入する

Krea2・Anima・H3動画は、次を実行してメニュー番号（1〜3）を選びます。P-H3は起動後の画面から準備できます。Qwen・YuE2・Irodori TTS・Clef・Iris・FLUX 3・MOSS Imageは表にある専用BATを実行してください。

```powershell
.\aikimi-setup.bat
```

| モデル | できること・使う画面 | セットアップ操作 |
|---|---|---|
| [Krea2](docs/krea2_local_supersample_detail_ja.md) | 画像生成・4K/8K処理。上部 **Krea2** から使用 | `.\aikimi-setup.bat` → **1** |
| [Anima 3.8B v1.1](extensions-builtin/anima-3-8b/README.md) | 画像生成。上部 **Anima** から使用 | `.\aikimi-setup.bat` → **2** |
| [MiniMax-H3-Image](extensions-builtin/minimax-h3-studio/OFFICIAL_IMAGE_GUIDE.md) | 約16 MPの直接生成と最大16枚の入力を使う画像編集 | **H3 Image → MiniMax-H3-Image → 環境とモデルを準備** |
| [P-H3 Image](extensions-builtin/minimax-h3-studio/P_H3_IMAGE_GUIDE.md) | 画像生成・最大8枚の画像を使う編集。**H3 Image** を使用 | 起動後に **P-H3 Image → 準備を開く** |
| [MiniMax H3](extensions-builtin/minimax-h3-studio/README.md) | 音声付き動画と写真一枚からの無音360°周回。**H3 Studio** を使用 | `.\aikimi-setup.bat` → **3** |
| [Qwen Image 2.1](extensions-builtin/qwen-image21-studio/README.md) | 画像生成・編集・透過PNG・拡張。専用画面を使用 | [`aikimi-qwen-image21-setup.bat`](aikimi-qwen-image21-setup.bat) |
| [YuE2 Music](extensions-builtin/yue2-studio/README.md) | 作曲・ABC楽譜編集。**YuE2 Music** タブを使用 | [`aikimi-yue2-setup.bat`](aikimi-yue2-setup.bat) → **1：公式Python** |
| [Irodori TTS](docs/irodori-tts.md) | 日本語読み上げ、声の説明によるVoice Design、参照音声からの声の再現。**Irodori TTS** タブを使用 | [`aikimi-irodori-setup.bat`](aikimi-irodori-setup.bat) |
| [Clef / Clef-Flash](extensions-builtin/clef-studio/README.md) | 画像・文章・JSONの評価と画像の仕分け。上部 **Clef** で判断項目ごとの確率を確認 | [`aikimi-clef-setup.bat`](aikimi-clef-setup.bat) |
| [Iris-3B](extensions-builtin/iris-studio/README.md) | 画像生成・img2img・相対深度推定・復元と4倍拡大。通常版／INT8／W4A8、互換Full FT・LoRAと4ステップTurbo | [`aikimi-iris-setup.bat`](aikimi-iris-setup.bat) |
| [FLUX 3 Image Klein](docs/flux3-klein.md) | 蒸留版／Baseの画像生成、最大10枚の参照編集、配置ボックス。PUPは任意の追加機能。**FLUX 3** タブを使用 | [`aikimi-flux3-klein-setup.bat`](aikimi-flux3-klein-setup.bat) |
| [MOSS-Image-SFT](docs/moss-image.md) | 自然文・タグ・JSONレイアウトから画像生成、Hires.fix、元画像からの再描画、編集用LoRAによる最大3枚の参照入力。**MOSS Image** タブを使用 | [`aikimi-moss-image-setup.bat`](aikimi-moss-image-setup.bat) |

初回はモデルと必要な実行環境のダウンロード・変換に時間がかかります。Qwenの既定モデルは通常版Q4_K_Mです。Irisは選択した用途の重みを取得し、INT8・W4A8はHugging Faceの変換済み配布を使います。W4A8は画像生成用のテキストエンコーダーも圧縮します。[Irisの導入・検証](docs/iris-studio.md)

AnimaはGPUでINT8へ変換し、検証成功後にBF16変換元を削除します。残す場合は `.\aikimi-setup.bat -Model anima38 -KeepSource` を実行してください。

BATで準備したモデルの追加や中断後の再開は、WebUIを終了して該当BATを再実行します。P-H3とIrisは画面から準備を再実行できます。取得済みのファイルは検証して再利用します。手元の互換モデルを使う場合は[ローカルモデルの使い方](docs/local-models.md)を参照してください。Qwenは標準本体を取得せず、実行環境と共通部品だけを準備できます。

<details>
<summary>モデルの個別取得・コマンド指定・追加設定</summary>

モデルを指定して実行する例です。`-DryRun -NoPause`を付けると、変更せずに実行予定を確認できます。

```powershell
.\aikimi-setup.bat -Model anima38
```

環境を導入済みでモデルだけ取得する場合は、次のBATを使えます。

- Krea2: [download_krea2_int8_convrot_models.bat](download_krea2_int8_convrot_models.bat)
- Anima: [download_anima38_v11_int8_convrot_models.bat](download_anima38_v11_int8_convrot_models.bat)（要GPU）
- MiniMax H3: [download_minimax_h3_models.bat](download_minimax_h3_models.bat) ／ [W4A8追加版](download_minimax_h3_w4a8_models.bat)

Animaの個別取得・変換には準備済みのNeo環境とNVIDIA GPUが必要です。H3の個別BATは専用実行環境を導入しません。

Qwen公式フルモデル（INT8／W4A8／BF16）は `.\aikimi-qwen-image21-setup.bat --official-full`、任意のプロンプト書き換えは `.\aikimi-qwen-image21-setup.bat --prompt-rewriter-only` で追加します。[Qwenガイド](extensions-builtin/qwen-image21-studio/README.md) · [量子化導入ガイド](docs/w4a8.md) · [モデル導入ガイド](docs/model-installation.md)

[公式Qwen Image 2.1 Turbo](docs/qwen21-official-turbo.md)は `.\aikimi-qwen-image21-setup.bat --official-turbo-only` で追加し、Qwen画面で**公式Turbo · W4A8／INT8／BF16**を選びます。公式Turboは8 steps、Viggle Turboは4 stepsを推奨しますが、Stepsは変更できます。ControlNetの併用や推奨範囲外の強度も設定でき、推奨事項は生成を止めない通知で表示します。

公式Turboの変換済み[INT8](https://huggingface.co/Aikimi/Forge-Neo-Image-2.1-Turbo-INT8)・[W4A8](https://huggingface.co/Aikimi/Forge-Neo-Image-2.1-Turbo-W4A8)はHugging Faceから導入できます。必要な共通部品・対応環境・導入コマンドは[公式Turboガイド](docs/qwen21-official-turbo.md)を参照してください。

</details>

### 起動して生成する

1. `aikimi-launch.bat`をダブルクリックします。普段の起動もこのBATを使います。P-H3は起動後に画面から準備します。
2. ログにURLが表示されたら、ブラウザーで [http://127.0.0.1:7861](http://127.0.0.1:7861) を開きます。
3. 使いたい画面を開き、生成条件を設定して生成します。詳しい操作や作例は、上の表のモデル名から確認できます。

通常の`LocalSafe`起動は自分のPC内だけで利用し、LANやインターネットへ自動公開しません。初回起動では本体の必要なライブラリも準備します。

<a id="起動profile"></a>

<details>
<summary>低VRAM・API・LAN利用などの起動設定</summary>

PowerShellからプロファイルを指定して起動できます。

```powershell
.\aikimi-launch.ps1 -Profile LowVRAM
```

| プロファイル | 用途 |
|---|---|
| `LocalSafe` | 通常利用（WebUI+API、PC内限定） |
| `LocalAPI` | API専用起動（PC内限定） |
| `LowVRAM` | 低GPUメモリ環境向け |
| `RTX3090Recommended` | RTX 3090向け |
| `Development` | 開発・UI確認用 |
| `LANAuthenticated` | LAN内別端末からの認証付き利用 |

LAN利用では、Git管理外の`secrets/gradio-auth.txt`と`secrets/api-auth.txt`に、各行を`username:password`形式で記述します。インターネットへ公開する場合はTLSとファイアウォールも必要です。[接続と認証の設定](docs/security-model.md)

`webui-user.bat`を使う場合の個人設定は、次のファイルを作って編集してください。

```powershell
Copy-Item .\webui-user.example.bat .\webui-user.local.bat
```

</details>

## 主な機能

### 画像の編集・拡張・仕上げ

| 目的 | 機能・参照ガイド |
|---|---|
| 周囲の描き足し | [Qwen Outpaint](extensions-builtin/qwen-image21-studio/README.md#outpaint補助)。辺や角をドラッグして範囲を指定。境界幅0では元画像の全画素を保持します |
| マスク・制御画像編集 | [Qwen Image 2.1](extensions-builtin/qwen-image21-studio/README.md)（囲み注釈、マスク、Fun ControlNet） |
| 入力画像から描き直す | [Iris img2img](docs/iris-studio.md#入力画像から描き直すimg2img)。互換Full FTを選び、変更強度0〜1を調整。通常サンプリングで使用します |
| 4K／8K再作画 | [Krea2高解像度処理](docs/krea2_local_supersample_detail_ja.md)・[HyperWeave](extensions-builtin/hyperweave/README.md)。細部はモデルが推定するため変化します。一部は実験機能です |
| 粒状感抑制 | Extras → [Grain Cleaner](docs/grain-cleaner.md)（追加モデル不要・CPU処理） |
| 背景透過PNG化 | Extras → [背景除去](docs/background-removal.md)（単画像・バッチ対応） |
| 明るさ・色むら補正 | ExtrasのColor Flatten・色むら確認、生成時の[CD Tuner](docs/cd-tuner-negpip.md) |

### 普段の操作

- Forgeの`txt2img`・`img2img`・Extrasを使えます。上部のモデルボタンから画像生成の系列を選び、共通の配置で生成・編集できます。LoRAやControlNetなどは上部ボタンで開閉します。[画像生成の共通画面](docs/image-workspace.md)
- **GPU・モデル保持** で「連続生成を優先」を選ぶと、次の生成までモデルを保持します。手動解放もできます。[モデル保持の使い方](docs/model-retention.md)
- **ちびあいきみ** をクリックすると進行状況や順番待ちを確認できます。ドラッグで移動し、上部のメニューで表示を切り替えられます。

<a id="手動でダウンロードしたloraを使う"></a>

### 手元のLoRAを使う

対応LoRAを下記フォルダーに配置します。画面によっては選択欄へのフルパス入力も使えます。

| 画面 | 本体フォルダー内の保存先 | 適用方法 |
|---|---|---|
| Forge `txt2img` / `img2img` | `models/Lora/` | **LoRAを組み合わせる** で選択、または `<lora:名前:0.8>` |
| P-H3 Image | `models/MiniMax-H3/loras/` | **LoRA → 一覧を更新 → 選択**。共有モデル保存先を変更した場合はその中の `loras/` を使用 |
| Qwen Image 2.1 | `models/Qwen-Image-2.1/loras/` | **LoRA → 一覧更新 → 複数選択** |
| Iris（画像生成） | `models/Iris-3B/loras/` | **LoRA → 一覧を更新 → 複数選択**。[対応形式・Turbo](docs/iris-studio.md#任意のloraとコミュニティ製turbo) |

強度は−2〜2で指定します。0は無効、×は選択解除です。同じLoRAを選択欄とプロンプトの両方へ指定しないでください。モデルの世代や形式が異なると使えません。[対応形式と外部モデルの指定方法](docs/local-models.md) · [Qwen追加LoRAガイド](extensions-builtin/qwen-image21-studio/README.md#追加lora)

Qwenの画風・色・照明の編集で構図を保ちたい場合は、[Consistency LoRA](docs/qwen21-consistency-lora.md)をLoRA欄から準備できます。通常版1500を取得し、既存の選択と強度を保って追加します。新しいポーズへの変更を抑える場合があります。

手元のQwen Image 2.1用GGUFやINT8 ConvRot本体は、**チェックポイント**から選択できます。配置・対応形式・共通部品は[ローカルモデルの使い方](docs/local-models.md)を参照してください。

<details>
<summary>タグ入力補助・高速化・詳細設定</summary>

Danbooruタグなどの入力候補を使う場合は[Tag Autocomplete](https://github.com/DominikDoom/a1111-sd-webui-tagcomplete)を追加し、Neoを再起動します。先頭候補はTab、矢印キーで選んだ候補はEnterで確定できます。

```powershell
git clone https://github.com/DominikDoom/a1111-sd-webui-tagcomplete.git extensions/tag-autocomplete
```

- **計算量の調整**：[Jev / Sparse Attention](docs/jev-sparse.md) · [Krea2設定例](docs/krea2-jev.md)。Jev自動判定は自分のAPIキーを使い、API回数・待ち時間の上限を設定できます。固定率ではAPIを呼びません。
- **Qwen拡張**: [Qwen Fun Acc（4-step）](docs/assets/qwen-image21-fun-acc/README.md) · [Fun ControlNet](docs/assets/qwen-image21-fun-controlnet/README.md)
- **MiniMax H3詳細**: [360°周回](docs/minimax-h3-orbit-lora.md) · [長尺生成](extensions-builtin/minimax-h3-studio/README.md#長尺生成) · [高速化](docs/minimax-h3-acceleration.md) · [CLIPキャッシュ](docs/minimax-h3-clipcache.md) · [ControlNet](docs/minimax-h3-fun-control.md)
- **量子化比較**: [INT8／W4A8量子化](docs/w4a8.md)
- **H3 Image（静止画）**：[MiniMax-H3-Imageガイド](extensions-builtin/minimax-h3-studio/OFFICIAL_IMAGE_GUIDE.md)で約16 MPの直接生成と最大16枚の入力を使う編集を案内しています。[P-H3 Imageガイド](extensions-builtin/minimax-h3-studio/P_H3_IMAGE_GUIDE.md)で画像生成と最大8枚を使う編集を案内しています。従来の[H3画像生成](extensions-builtin/minimax-h3-studio/IMAGE_GUIDE.md)も実験モードとして利用できます。

</details>

<a id="update"></a>

## 更新方法

WebUIを終了し、本体フォルダーでPowerShellを開いて次のコマンドを実行します。

```powershell
git switch neo
git pull --ff-only origin neo
```

古い取得元を使っている場合は、先に一度だけ `git remote set-url origin https://github.com/AiWithYou/aikimi-forge-neo.git` を実行してください。

**2026-09-24より前のQwen環境から更新する場合**は、起動前に該当するコマンドを実行して専用環境を更新します。モデルの再取得は不要です。

```powershell
# Qwen Image 2.1導入済み環境
.\aikimi-qwen-image21-setup.bat --runtime-only
```

必要な更新が終わったら`aikimi-launch.bat`で起動します。本体の依存環境は起動時に準備します。`TORCH_COMMAND`や`TORCH_INDEX_URL`を指定している場合は、その設定が優先されます。個別の注意事項は[変更履歴](CHANGELOG.md)を参照してください。

v3.6.0より前の開発版でClefを導入した場合は、WebUIを終了して`.\aikimi-clef-setup.bat --runtime-only`を実行し、専用環境を更新してください。取得済みの量子化モデルをそのまま使えます。

v4.9.0でMOSS Imageを導入した場合は、使用中の保存形式を指定してセットアップを一度実行し、ファイルの検証情報を更新してください。既定のINT8構成は`.\aikimi-moss-image-setup.bat`です。取得済みの同じ重みは検証して再利用します。BF16やW4A8を使用している場合は、[保存形式の指定方法](docs/moss-image.md#導入)を参照してください。

<a id="troubleshooting"></a>

## トラブルシューティング

起動できない、モデルを読み込めない、GPUメモリが足りない場合は[トラブルシューティング](docs/troubleshooting.md)を参照してください。Settingsの **Diagnostics** から環境状態を確認できます。

通常のtxt2img／img2imgでは、Settings → Optimizationsの **Batch Cond/Uncond** をOFFにすると、画像枚数を保ったまま正・負の条件を順に処理します。初期値はONです。

ログや生成条件を共有する前に、パスワード、個人用のフォルダーパス、プロンプトが含まれていないか確認してください。

<a id="forge-neoとの違い"></a>

<details>
<summary>Forge Neoとの違い・開発資料</summary>

[Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) を基盤とし、各モデルStudio・セットアップ支援・仕上げ機能・UIナビゲーションを追加しています。

Krea2・Animaの基本対応や量子化モデルの読み込みはForgeから引き継いでいます。既定ブランチは`neo`、同期基準は`0d0cb72951b059c8ea17861ba86db8d0f6098c28`です。その後の取り込みは[Forge Neo更新の確認記録](docs/upstream-sync.md)を参照してください。

<a id="documentation"></a>
<a id="test"></a>

- [開発環境・テスト手順](CONTRIBUTING.md)
- [アーキテクチャ](docs/architecture.md)
- [API・セキュリティモデル](docs/security-model.md)
- [セキュリティ問題の報告](SECURITY.md)
- [リリース確認項目](docs/release-checklist.md)

</details>

<a id="licenseと配布条件"></a>

## ライセンスと利用条件

コードのライセンスは[AGPL-3.0](LICENSE)です。モデル・VAE・テキストエンコーダー・LoRA・素材の利用条件は各配布元で確認してください。モデルやComfyUIは各開発元の成果です。出典と個別の条件は各機能ガイドと[Third-party notices](THIRD_PARTY_NOTICES.md)に記載しています。
