# Aikimi Studio Neo

**v1.0.0** · [変更履歴](CHANGELOG.md)

<img src="assets/aikimi/idle-still.webp" alt="ちびあいきみ" width="112" align="right">

**Krea2・Anima・SenseNovaの画像生成・編集と、MiniMax H3の音声付き動画を、Forge Neoの画面から使えるWindows向け派生版です。** モデルのセットアップ、4K／8K処理、画像の仕上げもまとめています。

[Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)を基盤にしています。主な対象はWindows 11・Python 3.13・NVIDIA GPUで、通常起動では自分のPC内だけで利用し、LANやインターネットへ自動公開しません。

[Forge Neoとの違い](#forge-neoとの違い) · [主な機能](#主な機能) · [セットアップ方法](#セットアップ方法) · [その他](#その他)

## Forge Neoとの違い

**Aikimi Studio Neoは、Forge Neoの生成基盤に、本ブランチ独自の仕上げ処理、専用UI、モデル導入支援、操作・効率の改善を加えたものです。** 本リポジトリの既定ブランチ`neo`で提供しています。

| 用途 | Forge Neoを基盤に、この派生版で加えたこと |
|---|---|
| **モデル対応** | Anima 3.8B v1.1の拡張、SenseNova専用Studio、MiniMax H3専用StudioとComfyUI連携。 |
| **導入** | モデルを選ぶだけのセットアップBAT。INT8モデルの取得・変換と、必要な専用環境の準備。 |
| **仕上げ** | HyperWeaveの高解像度再作画、Grain Cleaner、Color Flatten、CD Tunerの統合。 |
| **操作** | 追加機能へのショートカット、実行状態の表示、Extrasの処理順・予定サイズの表示、設定とジョブの復旧。 |

Forge NeoにもKrea2・Animaの基本対応、量子化モデルの読み込み、画像編集・動画生成があります。下の一覧では、**Forge継承**は基盤から引き継いだ機能、**独自追加**は本ブランチの追加処理、**独自統合**は外部モデルや実行環境を使うための専用UI・連携を指します。

比較の基準は[Forge Neoの同期元](https://github.com/Haoming02/sd-webui-forge-classic/tree/0d0cb72951b059c8ea17861ba86db8d0f6098c28)です。その後の更新は選んで取り込んでいます。[2026年9月12日の更新確認](docs/upstream-sync.md)に採用・見送りの内容を記載しています。

モデルやComfyUIそのものは各開発元の成果です。出典と利用条件は機能別ガイドと[Third-party notices](THIRD_PARTY_NOTICES.md)を参照してください。

## 主な機能

### モデル対応と導入

| 機能 | 区分 | できること |
|---|---|---|
| **かんたんセットアップ** | 独自追加 | BATでモデルを選び、本体と必要な専用環境をまとめて準備。取得済みのモデルは再利用できます。 |
| **Krea2** | Forge継承＋独自統合 | ForgeのKrea2対応に、INT8モデルの導入支援と4K／8K向けの追加処理を同梱。高解像度処理の一部は実験機能です。 |
| **Anima 3.8B** | Forge継承＋独自統合 | Qwen3.5を使う専用設定、v1.1のSemantic Connector v2、INT8変換・導入支援。v1／v1.1それぞれのモデル構成に対応します。 |
| **SenseNova U1.5 Studio** | 独自統合 | 画像生成と複数参照による編集。参照の順序変更・役割指定・生成結果からの継続編集に対応。テキスト生成は公式8-Step LoRA、参照編集はQuality 50-Stepを使います。 |
| **MiniMax H3 Studio** | 独自統合 | 音声付き動画を生成。専用ComfyUI・Python・標準INT8モデルのセットアップと、既存モデルの共有に対応します。 |
| **MiniMax H3 Image** | 独自統合・実験機能 | `H3 Image`タブで静止画生成と参照画像による編集。PNGと生成条件を保存します。実モデルでのGPU画像生成・画質・速度は未検証です。 |

モデル本体はリポジトリに含みません。[セットアップ方法](#セットアップ方法)で導入するモデルを選んでください。

### 画像生成と全体の操作

| 機能 | 区分 | できること |
|---|---|---|
| **Forge画像生成** | Forge継承 | `txt2img`、`img2img`、Extras、モデル読み込みなど、Forgeの基本機能を利用。 |
| **Aikimiナビゲーション** | 独自追加 | 画面上部のショートカットからKrea2・Anima・SenseNova・MiniMax H3へ移動。既存のForgeタブもそのまま使えます。 |
| **ちびあいきみ・状態表示** | 独自追加 | 生成状況、順番待ち、実行環境を確認。詳しい情報は展開して表示でき、あいきみの表示や動きはSettingsで調整できます。 |
| **Extrasの操作改善** | 独自追加 | 処理順と予定サイズの表示、Grain Cleaner単独設定、見本範囲の選択、結果の要約。同じ画像の再調整では解析結果を再利用します。 |
| **保存・ジョブ復旧** | 独自追加 | 設定や動画の保存中に失敗した場合の既存ファイル保護、H3ジョブの送信記録と再起動後の照合、Forge・SenseNova・H3間のGPU使用調整。 |
| **起動設定** | 独自追加 | 通常のローカル起動、低VRAM向け設定、API専用起動、認証付きLAN利用を選択。 |
| **Diagnostics** | 独自追加 | SettingsからPython・GPU・モデルの準備状況を確認。APIからも診断結果を取得できます。 |

### 拡張・仕上げと詳細設定

| 機能 | 区分 | できること |
|---|---|---|
| **HyperWeave 4K／8K** | 独自追加・実験機能 | 入力画像の構図を制約として、読み込み済みモデルで段階的に再作画。細部はモデルが推定するため、入力から変化します。 |
| **Grain Cleaner** | 独自追加 | 微細な粒状感を抑えながら細部を保護。自動推定・見本範囲指定・処理マスク・診断画像に対応。追加モデル不要のCPU処理です。 |
| **Color Flatten・色むら確認** | 独自追加 | 色差のムラ補正、Smooth Gradientによる平滑化、色むらの解析・可視化。 |
| **CD Tuner** | 独自統合 | `txt2img`／`img2img`のDetail・色・明るさ・彩度・Color Mapを調整。重みの直接編集は対応する浮動小数点層に限ります。 |
| **SenseNovaの参照優先モード** | 独自追加 | 参照キャッシュのCPU退避とAttentionの分割処理でVRAM使用量を削減。最大8枚・各約1MPの参照と約4MP出力に対応し、CPU RAMと転送時間を使用します。 |
| **H3の長尺生成** | 独自統合＋独自追加 | 共通プロンプトと区間ごとの指示から、複数区間をつないだ動画を生成。HybridWindowsを利用する方式も選べます。導入条件と併用できる設定は[長尺生成ガイド](extensions-builtin/minimax-h3-studio/README.md#長尺生成)を参照してください。 |
| **H3の高速化設定** | 独自統合 | Turbo・INT8 VAE・Fast Decode・Sparse Attentionを必要に応じて選択。画質・メモリ・速度とのトレードオフは[高速化ガイド](docs/minimax-h3-acceleration.md)に記載しています。 |
| **H3 NegPiP** | 独自統合 | H3 Studio／H3 Imageでプロンプト内の負の重みを使用。切替後は実行環境の再起動が必要で、Sparse Attentionとは併用できません。 |
| **H3 CLIP条件キャッシュ** | 独自統合＋独自追加 | 同じプロンプト・参照素材の条件を再利用し、Qwen3-VLの再ロードと再計算を省略。固定版CLIPCachedの導入が必要です。 |
| **H3 Fun ControlNet · INT8** | 独自統合 | 元動画のCannyや前処理済みのDepth・Pose動画で、動きと構図を制御。INT8制御モデルと対応ComfyUIが必要です。 |

### 機能別ガイド

- [Krea2の高解像度処理](docs/krea2_local_supersample_detail_ja.md)
- [Anima 3.8B](extensions-builtin/anima-3-8b/README.md)
- [SenseNova U1.5 Studio](extensions-builtin/sensenova-u15-studio/README.md)
- [MiniMax H3 Studio：動画生成](extensions-builtin/minimax-h3-studio/README.md)
- [MiniMax H3 Image：実験的な静止画生成](extensions-builtin/minimax-h3-studio/IMAGE_GUIDE.md)
- [MiniMax H3の任意の高速化設定](docs/minimax-h3-acceleration.md)
- [MiniMax H3のCLIPキャッシュとNegPiP併用](docs/minimax-h3-clipcache.md)
- [MiniMax H3 Fun ControlNetと実行環境の修正](docs/minimax-h3-fun-control.md)
- [HyperWeave](extensions-builtin/hyperweave/README.md)
- [Grain Cleanerガイド](docs/grain-cleaner.md)
- [CD Tuner・MiniMax H3 NegPiPガイド](docs/cd-tuner-negpip.md)

<details>
<summary>WebUIの操作と状態表示の詳細</summary>

<a id="webuiの操作境界"></a>

**Extrasの独自追加・改善**は次のとおりです。

- Generate付近に、現在有効な処理の順序と予定サイズを表示。
- **Grain Cleanerだけ実行する設定にする**ボタンで、Upscale・Color Flatten・色むら確認をオフに設定。
- Color Flattenは選択中の方式で使う設定だけ表示し、非表示にした値も保持。
- Grain Cleanerの見本範囲は、プレビュー上の対角2点から選択。拡大と併用する場合やバッチでは、処理後の座標を手入力。
- 同じ画像の強度・細部保護・マスクなどを変えた再実行では、直近1画像の解析を再利用。
- 結果は状態・変更画素率・所要時間を要約し、詳細JSONを展開式で表示。無変更や診断画像省略の理由も明記。
- バッチの入力画像は1枚ずつ読み込み、動画は処理したフレームから順に書き出して、全フレームをメモリへ蓄積する負荷を削減。
- エラーやキャンセル時も処理状態を終了し、使用中の動画ファイルや処理用の一時ファイルを片付ける仕組みを整備。

バッチで結果画像をすべて画面に表示する場合は、その画像分のメモリを使用します。多数の画像を処理する際は、`Batch from Directory`の`Show result images`をオフにすると表示用の保持を抑えられます。

操作方法、キャッシュの更新条件、CLIは[Grain Cleanerガイド](docs/grain-cleaner.md)を参照してください。`H3 Image`は静止画専用の独立タブです。上部の`MiniMax H3`ショートカットは動画の`H3 Studio`を開きます。

**生成中の調整にはCD TunerとH3 NegPiPを使います。** CD Tunerは通常の`txt2img`／`img2img`内の設定欄、H3 NegPiPは`H3 Studio`／`H3 Image`内の設定欄から有効にしてください。どちらも初期状態はオフです。原作者・導入条件・設定例は[CD Tuner・MiniMax H3 NegPiPガイド](docs/cd-tuner-negpip.md)にまとめています。

通常の`txt2img`、`img2img`、`Extras`、`Settings`に加え、画面上部から追加機能を開けます。

- **Krea2**：画像生成用のPreset`krea`への切り替え。
- **Anima**：Preset`anima`への切り替えと、専用設定欄の表示。
- **SenseNova・MiniMax H3**：それぞれの専用Studioの表示。

Krea2・Animaを選んでも、操作中の`txt2img`／`img2img`タブは維持されます。他のタブから選んだ場合は`txt2img`へ移動します。

ちびあいきみの状態表示から、進行状況や順番待ち、実行環境を確認できます。詳しい情報は展開して表示し、表示の有無・大きさ・動きはSettingsで調整してください。

</details>

## セットアップ方法

<a id="quick-start"></a>

### 必要な環境

先に次を用意してください。コマンドはPowerShell 7で実行します。

- Git
- Python 3.13
- PowerShell 7の`pwsh`
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- 対象PyTorch buildに対応したNVIDIA driver

### はじめて使う場合

```powershell
git clone --branch neo https://github.com/AiWithYou/aikimi-studio-neo.git
cd aikimi-studio-neo
.\aikimi-setup.bat
```

1. 表示された番号から使いたいモデルを選び、セットアップの完了を待ちます。
2. 完了後は、フォルダー内の`aikimi-launch.bat`をダブルクリックしてください。
3. 起動ログにURLが表示されたら、ブラウザーで<http://127.0.0.1:7861>を開きます。

**初回はセットアップBAT、普段は起動BATを使います。** 既存のモデルで始める場合は、セットアップを省いて`aikimi-launch.bat`で起動できます。モデルの配置先は各ガイドを参照してください。

`LocalSafe`は自分のPC内だけで利用する通常起動です。初回は必要なライブラリを自動導入します。既定のPyTorchは`2.11.0+cu130`、torchvisionは`0.26.0+cu130`なので、対応するNVIDIAドライバーを用意してください。

<a id="model-setup"></a>

### モデルの導入

セットアップメニューでは、次の4種類を選べます。モデルを追加するときもNeoを終了して`aikimi-setup.bat`を実行してください。

| 選択 | 自動で準備する内容 |
|---|---|
| 1：Krea2 | INT8 ConvRot配布版・エンコーダー・VAE |
| 2：Anima 3.8B v1.1 | BF16取得・INT8 ConvRot変換・エンコーダー・VAE |
| 3：SenseNova U1.5 | INT8 ConvRot配布版・8-Step LoRA・専用Python環境 |
| 4：MiniMax H3 | 標準INT8モデル一式・専用ComfyUI・Python環境 |

完了後は`aikimi-launch.bat`で起動します。中断時は同じモデルを選び直してください。取得済みのモデルは検証して再利用します。AnimaはNVIDIA GPUで自動変換し、検証に成功するとBF16変換元を削除します。

<details>
<summary>モデルを指定して実行する場合</summary>

```powershell
.\aikimi-setup.bat -Model krea2
.\aikimi-setup.bat -Model anima38
.\aikimi-setup.bat -Model sensenova
.\aikimi-setup.bat -Model h3
```

`-DryRun -NoPause`を付けると、変更せずに実行予定を確認できます。AnimaのBF16変換元を残す場合は`-KeepSource`を付けてください。

</details>

必要な空き容量、導入状況の確認、ファイルの修復は[モデル導入ガイド](docs/model-installation.md)にまとめています。

#### モデルだけ個別にダウンロード

一括セットアップを使わず、必要なモデルだけ取得する場合は次のBATをダブルクリックしてください。

| モデル | 個別ダウンロード |
|---|---|
| Krea2 | [download_krea2_int8_convrot_models.bat](download_krea2_int8_convrot_models.bat) |
| Anima 3.8B v1.1 | [download_anima38_v11_int8_convrot_models.bat](download_anima38_v11_int8_convrot_models.bat) |
| SenseNova U1.5 | [download_sensenova_u15_models.bat](download_sensenova_u15_models.bat) |
| MiniMax H3 | [download_minimax_h3_models.bat](download_minimax_h3_models.bat) |

AnimaはINT8変換にNeoのPython環境とNVIDIA GPUを使うため、先に通常起動を一度済ませてください。SenseNovaとH3の個別BATは専用実行環境を導入しません。

MiniMax H3の実行環境とモデルの準備は、[MiniMax H3 Studioガイド](extensions-builtin/minimax-h3-studio/README.md)を参照してください。

<a id="起動profile"></a>

### 起動プロファイル

通常は`LocalSafe`を使います。別の設定で起動する場合は、`-Profile`の後を次の表から選んでください。

```powershell
.\aikimi-launch.ps1 -Profile LocalSafe
```

| 起動設定 | 用途 | 接続できる範囲 |
|---|---|---|
| `LocalSafe` | 通常のWebUIとAPI | 自分のPC内 |
| `LocalAPI` | APIだけを起動 | 自分のPC内 |
| `Development` | 開発・画面確認用 | 自分のPC内 |
| `LowVRAM` | GPUメモリが少ない環境向け | 自分のPC内 |
| `RTX3090Recommended` | RTX 3090向け | 自分のPC内 |
| `LANAuthenticated` | 認証付きでLAN内の別端末から利用 | 認証必須。接続範囲はネットワーク設定による |

#### 認証付きLAN利用

`--listen`、loopback以外の`--server-name`、`--share`、`--ngrok`は、`--aikimi-remote`と認証がなければ起動前に失敗します。

`LANAuthenticated`を使う場合は、Git管理外の次の2ファイルを作成してください。

```text
secrets/gradio-auth.txt
secrets/api-auth.txt
```

各行は`username:password`形式です。

```powershell
.\aikimi-launch.ps1 -Profile LANAuthenticated
```

Basic認証だけでインターネットへ直接公開しないでください。TLS reverse proxyとfirewallを併用します。詳しくは[security model](docs/security-model.md)と[SECURITY.md](SECURITY.md)を参照してください。

### `webui-user.bat`のローカル設定

`webui-user.bat`で起動オプションを変更する場合は、設定例をコピーして`webui-user.local.bat`を作成し、そちらを編集してください。この個人用ファイルはGitの管理対象から除外されています。

```powershell
Copy-Item .\webui-user.example.bat .\webui-user.local.bat
```

## その他

### ベース・同期情報

| 項目 | 内容 |
|---|---|
| 既定ブランチ | `neo` |
| ベース | `Haoming02/sd-webui-forge-classic`の`neo` |
| 最終同期基準 | `0d0cb72951b059c8ea17861ba86db8d0f6098c28`（Forge Neo 2.29後の`arch`更新を含む） |
| 最新の確認先 | `76586f6a`（2026-09-12）。[選択取り込みの記録](docs/upstream-sync.md) |
| Aikimiの配布バージョン | `1.0.0`。Forgeのバージョンとは別に管理 |
| 主対象 | Windows 11、Python 3.13、NVIDIA GPU |
| コードのライセンス | AGPL-3.0。モデルとアセットには別条件が適用される場合があります。 |

### DiagnosticsとAPI

SettingsのDiagnosticsで、PCやモデルの準備状況を確認できます。状態をAPIから取得する場合は、次の読み取り専用URLを使います。

```text
GET /aikimi/api/v1/health
GET /aikimi/api/v1/status
GET /aikimi/api/v1/capabilities
```

`LocalSafe`での確認例：

```powershell
Invoke-RestMethod http://127.0.0.1:7861/aikimi/api/v1/health
Invoke-RestMethod http://127.0.0.1:7861/aikimi/api/v1/capabilities
```

LANなどの別端末から利用する場合は認証が必要です。認証情報や機密性の高いファイルパスは、APIの応答から除外します。

<a id="update"></a>

### 更新方法

WebUIを終了し、Aikimi Studio NeoをダウンロードしたフォルダーでPowerShellを開いて、次を実行してください。このリポジトリの`neo`ブランチを最新版に更新します。

```powershell
git switch neo
git pull --ff-only origin neo
```

更新が終わったら、`aikimi-launch.bat`をダブルクリックするか、普段使っている起動コマンドでWebUIを起動してください。導入済みのモデルや保存した画像を、ダウンロードし直す必要はありません。

<a id="test"></a>

### 開発・テスト

コードを変更する場合の環境構築とテスト手順は[開発ガイド](CONTRIBUTING.md)を参照してください。

<a id="troubleshooting"></a>

### トラブルシューティング

起動できない、モデルを読み込めない、認証で接続できないといった場合は[トラブルシューティング](docs/troubleshooting.md)を確認してください。SenseNovaとMiniMax H3の対処手順も記載しています。

ログや環境情報を共有する際は、パスワード、個人用のフォルダーパス、プロンプトなどが含まれていないか確認してください。

<a id="documentation"></a>

### ドキュメント

- [Security policy](SECURITY.md)
- [Security model](docs/security-model.md)
- [Architecture](docs/architecture.md)
- [Model installation](docs/model-installation.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Contributing](CONTRIBUTING.md)
- [変更履歴](CHANGELOG.md)
- [Forge Neo更新の確認記録](docs/upstream-sync.md)
- [Release checklist](docs/release-checklist.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

<a id="licenseと配布条件"></a>

### ライセンスと配布条件

コードのライセンスは[AGPL-3.0](LICENSE)です。個別のライセンスがあるライブラリや素材は、それぞれの条件に従ってください。

モデル・VAE・テキストエンコーダー・LoRAなどの利用条件は各配布元で確認できます。出典と個別のライセンス情報は、各機能のガイドと[Third-party notices](THIRD_PARTY_NOTICES.md)を参照してください。
