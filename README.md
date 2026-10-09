# Aikimi Forge Neo

**[v3.8.0](https://github.com/AiWithYou/aikimi-forge-neo/releases/tag/v3.8.0)** · [変更履歴](CHANGELOG.md)

<img src="assets/aikimi/pet.png" alt="ちびあいきみ" width="112" align="right">

**画像の生成・編集、音声付き動画、作曲、画像や文章の評価を、ひとつのWebUIから使えるWindows向けのForge Neo派生版です。** モデルの導入、高解像度処理、画像の仕上げも支援します。

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

使いたいモデルを選び、対応するセットアップBATを実行します。**最初は使うモデルだけ導入すれば始められます。** モデル本体はリポジトリに含まれません。

<a id="model-setup"></a>

### モデルを選んで導入する

Krea2・Anima・SenseNova・H3・Nanosaur2・Mingは、次を実行してメニュー番号（1〜6）を選びます。Qwen・YuE2・Clef・Irisは表にある専用BATを実行してください。

```powershell
.\aikimi-setup.bat
```

| モデル | できること・使う画面 | セットアップ操作 |
|---|---|---|
| [Krea2](docs/krea2_local_supersample_detail_ja.md) | 画像生成・4K/8K処理。上部 **Krea2** から使用 | `.\aikimi-setup.bat` → **1** |
| [Anima 3.8B v1.1](extensions-builtin/anima-3-8b/README.md) | 画像生成。上部 **Anima** から使用 | `.\aikimi-setup.bat` → **2** |
| [SenseNova U1.5](extensions-builtin/sensenova-u15-studio/README.md) | 画像生成・参照画像編集。専用Studioを使用 | `.\aikimi-setup.bat` → **3** |
| [MiniMax H3](extensions-builtin/minimax-h3-studio/README.md) | 音声付き動画生成。**H3 Studio** を使用 | `.\aikimi-setup.bat` → **4** |
| [Nanosaur2](extensions-builtin/nanosaur2-studio/README.md) | イラスト向け画像生成。専用タブを使用 | `.\aikimi-setup.bat` → **5** |
| [Ming Image Design](extensions-builtin/ming-image-studio/README.md) | ポスター・UI案・透過素材の生成。専用Studioを使用 | `.\aikimi-setup.bat` → **6**（INT8／W4A8選択） |
| [Qwen Image 2.1](extensions-builtin/qwen-image21-studio/README.md) | 画像生成・編集・透過PNG・拡張。専用画面を使用 | [`aikimi-qwen-image21-setup.bat`](aikimi-qwen-image21-setup.bat) |
| [YuE2 Music](extensions-builtin/yue2-studio/README.md) | 作曲・ABC楽譜編集。**YuE2 Music** タブを使用 | [`aikimi-yue2-setup.bat`](aikimi-yue2-setup.bat) → **1：公式Python** |
| [Clef / Clef-Flash](extensions-builtin/clef-studio/README.md) | 画像・文章・JSONの評価と画像の仕分け。上部 **Clef** で判断項目ごとの確率を確認 | [`aikimi-clef-setup.bat`](aikimi-clef-setup.bat) |
| [Iris-3B](extensions-builtin/iris-studio/README.md) | 画像生成・相対深度推定・復元と4倍拡大。上部 **Iris** で通常版／INT8を選択 | [`aikimi-iris-setup.bat`](aikimi-iris-setup.bat) |

初回はモデルと必要な実行環境のダウンロード・変換に時間がかかります。Qwenの既定モデルは通常版Q4_K_Mです。Irisは選択した用途の重みを取得し、INT8はHugging Faceの変換済み配布を使います。[Irisの導入・検証](docs/iris-studio.md)

AnimaはGPUでINT8へ変換し、検証成功後にBF16変換元を削除します。残す場合は `.\aikimi-setup.bat -Model anima38 -KeepSource` を実行してください。

モデルの追加や中断後の再開は、WebUIを終了して該当BATを再実行します。Irisは画面の**モデルを準備**から追加・再開できます。取得済みのファイルは検証して再利用します。手元の互換モデルを使う場合は[ローカルモデルの使い方](docs/local-models.md)を参照してください。Qwen・Mingは標準本体を取得せず、実行環境と共通部品だけを準備できます。

<details>
<summary>モデルの個別取得・コマンド指定・追加設定</summary>

モデルを指定して実行する例です。`-DryRun -NoPause`を付けると、変更せずに実行予定を確認できます。

```powershell
.\aikimi-setup.bat -Model anima38
```

環境を導入済みでモデルだけ取得する場合は、次のBATを使えます。

- Krea2: [download_krea2_int8_convrot_models.bat](download_krea2_int8_convrot_models.bat)
- Anima: [download_anima38_v11_int8_convrot_models.bat](download_anima38_v11_int8_convrot_models.bat)（要GPU）
- SenseNova: [download_sensenova_u15_models.bat](download_sensenova_u15_models.bat)
- MiniMax H3: [download_minimax_h3_models.bat](download_minimax_h3_models.bat) ／ [W4A8追加版](download_minimax_h3_w4a8_models.bat)

Animaの個別取得・変換には準備済みのNeo環境とNVIDIA GPUが必要です。SenseNovaとH3の個別BATは専用実行環境を導入しません。

Qwen公式フルモデル（INT8／W4A8／BF16）は `.\aikimi-qwen-image21-setup.bat --official-full`、任意のプロンプト書き換えは `.\aikimi-qwen-image21-setup.bat --prompt-rewriter-only` で追加します。[Qwenガイド](extensions-builtin/qwen-image21-studio/README.md) · [量子化導入ガイド](docs/w4a8.md) · [モデル導入ガイド](docs/model-installation.md)

</details>

### 起動して生成する

1. セットアップ完了後、`aikimi-launch.bat`をダブルクリックします。普段の起動もこのBATを使います。
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
| 4K／8K再作画 | [Krea2高解像度処理](docs/krea2_local_supersample_detail_ja.md)・[HyperWeave](extensions-builtin/hyperweave/README.md)。細部はモデルが推定するため変化します。一部は実験機能です |
| 粒状感抑制 | Extras → [Grain Cleaner](docs/grain-cleaner.md)（追加モデル不要・CPU処理） |
| 背景透過PNG化 | Extras → [背景除去](docs/background-removal.md)（単画像・バッチ対応） |
| 明るさ・色むら補正 | ExtrasのColor Flatten・色むら確認、生成時の[CD Tuner](docs/cd-tuner-negpip.md) |

### 普段の操作

- Forgeの`txt2img`・`img2img`・Extrasを使えます。上部のショートカットからモデル別の画面へ移動できます。
- **GPU・モデル保持** で「連続生成を優先」を選ぶと、次の生成までモデルを保持します。手動解放もできます。[モデル保持の使い方](docs/model-retention.md)
- **ちびあいきみ** をクリックすると進行状況や順番待ちを確認できます。ドラッグで移動し、上部のメニューで表示を切り替えられます。

<a id="手動でダウンロードしたloraを使う"></a>

### 手元のLoRAを使う

対応LoRAを下記フォルダーに配置するか、選択欄にフルパスを入力してEnterを押します。

| 画面 | 本体フォルダー内の保存先 | 適用方法 |
|---|---|---|
| Forge `txt2img` / `img2img` | `models/Lora/` | **LoRAを組み合わせる** で選択、または `<lora:名前:0.8>` |
| Qwen Image 2.1 | `models/Qwen-Image-2.1/loras/` | **LoRA → 一覧更新 → 複数選択** |
| Ming Image | `models/Lora/Ming/` | **モデル・LoRA** で選択 |

強度は−2〜2で指定します。0は無効、×は選択解除です。同じLoRAを選択欄とプロンプトの両方へ指定しないでください。モデルの世代や形式が異なると使えません。[対応形式と外部モデルの指定方法](docs/local-models.md) · [Qwen追加LoRAガイド](extensions-builtin/qwen-image21-studio/README.md#追加lora)

Qwenの画風・色・照明の編集で構図を保ちたい場合は、[Consistency LoRA](docs/qwen21-consistency-lora.md)をLoRA欄から準備できます。通常版1500を取得し、既存の選択と強度を保って追加します。新しいポーズへの変更を抑える場合があります。

手元のQwen Image 2.1用GGUFやINT8 ConvRot本体は、**本体モデル**から選択できます。対応形式・共通部品・実生成の確認範囲は[ローカルモデルの使い方](docs/local-models.md)を参照してください。

<details>
<summary>タグ入力補助・高速化・詳細設定</summary>

Danbooruタグなどの入力候補を使う場合は[Tag Autocomplete](https://github.com/DominikDoom/a1111-sd-webui-tagcomplete)を追加し、Neoを再起動します。先頭候補はTab、矢印キーで選んだ候補はEnterで確定できます。

```powershell
git clone https://github.com/DominikDoom/a1111-sd-webui-tagcomplete.git extensions/tag-autocomplete
```

- **計算量の調整**：[Jev / Sparse Attention](docs/jev-sparse.md) · [Krea2設定例](docs/krea2-jev.md)。Jev自動判定は自分のAPIキーを使い、API回数・待ち時間の上限を設定できます。固定率ではAPIを呼びません。
- **Qwen拡張**: [Qwen Fun Acc（4-step）](docs/assets/qwen-image21-fun-acc/README.md) · [Fun ControlNet](docs/assets/qwen-image21-fun-controlnet/README.md)
- **MiniMax H3詳細**: [長尺生成](extensions-builtin/minimax-h3-studio/README.md#長尺生成) · [高速化](docs/minimax-h3-acceleration.md) · [CLIPキャッシュ](docs/minimax-h3-clipcache.md) · [ControlNet](docs/minimax-h3-fun-control.md)
- **量子化比較**: [INT8／W4A8量子化](docs/w4a8.md)
- **H3 Image（静止画）**：[H3 Imageガイド](extensions-builtin/minimax-h3-studio/IMAGE_GUIDE.md)。実験機能です。実モデルでのGPU画像生成・画質・速度は未検証です。

</details>

<a id="update"></a>

## 更新方法

WebUIを終了し、本体フォルダーでPowerShellを開いて次を実行します。

```powershell
git switch neo
git pull --ff-only origin neo
```

古い取得元を使っている場合は、先に一度だけ `git remote set-url origin https://github.com/AiWithYou/aikimi-forge-neo.git` を実行してください。

**2026-09-24より前のSenseNova・Qwen環境から更新する場合**は、起動前に該当するコマンドを実行して専用環境を更新します。モデルの再取得は不要です。

```powershell
# SenseNova導入済み環境
.\download_sensenova_u15_int8.ps1 -RuntimeOnly

# Qwen Image 2.1導入済み環境
.\aikimi-qwen-image21-setup.bat --runtime-only
```

必要な更新が終わったら`aikimi-launch.bat`で起動します。本体の依存環境は起動時に準備します。`TORCH_COMMAND`や`TORCH_INDEX_URL`を指定している場合は、その設定が優先されます。個別の注意事項は[変更履歴](CHANGELOG.md)を参照してください。

v3.6.0より前の開発版でClefを導入した場合は、WebUIを終了して`.\aikimi-clef-setup.bat --runtime-only`を実行し、専用環境を更新してください。取得済みの量子化モデルをそのまま使えます。

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
