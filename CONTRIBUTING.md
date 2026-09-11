# Contributing to Aikimi Studio Neo

## 基本方針

Aikimi Studio Neoは、Windows向けの安全な既定値を保ちながら、Forge Neo由来の画像生成、動画生成、高解像度処理、Anima、SenseNova、MiniMax H3、Aikimi UIを維持します。変更は目的ごとに小さく分け、既存の公開APIとbat入口を理由なく削除しないでください。

## 開発環境

- Windows 11
- Python 3.13
- PowerShell 7の`pwsh`
- Git
- GPU機能を実機検証する場合は、対象PyTorch buildに対応したNVIDIA driver

取得方法は次のとおりです。

```powershell
git clone --branch neo https://github.com/AiWithYou/aikimi-studio-neo.git
cd aikimi-studio-neo
```

通常の起動と依存準備には、Local Safe profileを使います。

```powershell
.\aikimi-launch.ps1 -Profile LocalSafe
```

## 変更時の注意

- 認証情報を引数、ログ、fixture、commitへ入れないでください。
- モデル、dataset、checkpoint、生成物、cache、venvをcommitしないでください。
- 外部URL、ファイルパス、subprocess引数は、入口で検証してください。
- 必須設定にsilent fallbackを追加しないでください。
- GPU不要testを通すために、実機能を無効化しないでください。
- vendored codeやthird-party codeを変更する場合は、由来、固定revision、licenseを残してください。
- 日本語文書を変更した場合は、文書全体にstyle guardを実行してください。
- Forge由来の上部タブ、Quick Settings、`txt2img`、`img2img`、`Extras`、`Settings`は、Aikimi固有CSSから隠したり移動したりしないでください。
- Aikimi固有のナビゲーション、ちびあいきみ、status表示は`extensions-builtin/aikimi-ui`へ置き、Forge本体への変更面積を抑えてください。
- Krea2とAnimaのaliasは既存Forge controlを再利用します。生成設定、queue、model状態を別のUI stateへ複製しないでください。

## ローカル検証

CPU、offline、外部モデルdownloadなしのtest runnerは次です。

```powershell
uv pip install --python .\venv\Scripts\python.exe -r tools\requirements-test.txt
.\venv\Scripts\python.exe .\tools\run_ci_tests.py --verbosity 1
```

統一model setupだけを確認する場合は、次を使います。

```powershell
.\venv\Scripts\python.exe -m unittest -v tools.tests.test_aikimi_setup
.\venv\Scripts\python.exe -m unittest -v `
  tools.tests.test_krea2_download_script `
  tools.tests.test_anima38_download_script `
  tools.tests.test_sensenova_u15_download_script
```

変更したfirst-party codeはRuffでも確認します。

```powershell
.\venv\Scripts\python.exe -m ruff check <changed-python-paths>
.\venv\Scripts\python.exe -m ruff format --check <changed-python-paths>
```

PowerShell、security、dependencyのCI相当手順は[docs/release-checklist.md](docs/release-checklist.md)を参照してください。GPU live testは通常のunit testと分け、使用したモデル、revision、GPU、条件、実出力を記録します。

## upstream同期

通常利用者はAikimi Studio Neoの`origin/neo`だけを更新します。Forge Neo upstreamとの同期はmaintainer作業です。

```powershell
git remote add upstream https://github.com/Haoming02/sd-webui-forge-classic.git
git fetch upstream neo
```

同期基準を更新する場合は、upstream commit SHAをREADMEへ記録してください。自動rebaseや未確認の一括mergeで、Aikimi固有のsecurity guard、BnB/NF4/GGUF互換、built-in workflowを落とさないでください。`extensions-builtin/aikimi-ui`のfeature key、alias遷移先、SenseNovaとMiniMax H3のtab IDも確認します。

## Pull Request

Pull Requestには、目的、変更範囲、実行したtest、未実施のGPU test、互換性への影響を記載してください。セキュリティ上の未公開問題は通常のPull Requestへ書かず、[SECURITY.md](SECURITY.md)の非公開経路を使ってください。

## ライセンス

提出するコードは、リポジトリのAGPL-3.0と両立する条件で提供できるものに限ります。third-party codeやassetを含める場合は、著作権表示、license全文、固定した出典を追加してください。権利を確認できないassetを追加しないでください。
