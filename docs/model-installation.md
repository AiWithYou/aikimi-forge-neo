# Model installation

## 環境とモデルをまとめて導入

WindowsではNeoを終了して`aikimi-setup.bat`を実行し、モデルを選びます。Git、Python 3.13、PowerShell 7、uv、対応するNVIDIAドライバーを先に用意してください。

本体の仮想環境とGPUライブラリを準備してから、選んだモデルを導入します。AnimaはINT8へ自動変換し、SenseNovaは専用Python環境、H3は専用ComfyUIも準備します。既存の環境を使用する場合もPython 3.13であることを確認し、失敗した処理以降は実行しません。

```powershell
.\aikimi-setup.bat -Model krea2
.\aikimi-setup.bat -Model anima38
.\aikimi-setup.bat -Model sensenova
.\aikimi-setup.bat -Model h3
```

変更なしで手順を確認する場合は`-DryRun -NoPause`を付けます。AnimaのBF16変換元を残す場合は`-KeepSource`を指定してください。通常は変換と検証に成功した後に削除します。完了後は`aikimi-launch.bat`で起動します。

## モデルの確認・修復と個別導入

モデルだけ取得するBATも用意しています。本体の環境セットアップは実行しません。

| モデル | BAT | 取得内容 |
|---|---|---|
| Krea2 | `download_krea2_int8_convrot_models.bat` | INT8・エンコーダー・VAE |
| Anima 3.8B v1.1 | `download_anima38_v11_int8_convrot_models.bat` | BF16取得・INT8変換・エンコーダー・VAE |
| SenseNova U1.5 | `download_sensenova_u15_models.bat` | INT8・8-Step LoRA |
| MiniMax H3 | `download_minimax_h3_models.bat` | 標準INT8モデル・エンコーダー・映像／音声VAE |

Animaの変換には通常起動で準備済みのNeo環境とNVIDIA GPU、H3のダウンロードにはPython 3.12以上が必要です。H3の保存先は`models/MiniMax-H3`です。

SenseNovaの実行環境も導入する場合は既存の`download_sensenova_u15_int8.bat`を使えます。H3の実行環境はH3 Studio画面から準備できます。取得済みモデルは再利用します。

統一CLIはPython 3.13のstdlibだけで、profile一覧、導入、検証、修復を実行します。

```powershell
python .\tools\aikimi_setup.py list
python .\tools\aikimi_setup.py install krea2
python .\tools\aikimi_setup.py install anima38
python .\tools\aikimi_setup.py install sensenova
python .\tools\aikimi_setup.py verify
python .\tools\aikimi_setup.py repair anima38
```

計画だけを確認する場合は`--dry-run`、機械可読結果には`--json`を付けます。

```powershell
python .\tools\aikimi_setup.py install sensenova --dry-run --json
python .\tools\aikimi_setup.py verify krea2 --json
```

`--dry-run`はdirectoryを作らず、networkへ接続しません。計画ではpath、存在、sizeだけを確認し、SHA-256は読みません。`verify`は大きなmodel全体のSHA-256を読むため、完了まで時間がかかります。

## 安全性

CLIは次を固定manifestへ記録します。

- profileとartifact ID
- HTTPS URL
- immutable revision
- repositoryからの相対配置先
- byte size
- SHA-256
- SafeTensors marker
- license URL

配置先はrepository root配下へ限定します。absolute path、drive-qualified path、`..`、root外へ解決されるsymlinkやjunctionを拒否します。

downloadは正式名へ直接書かず、同じdirectoryの`.part`へstreamし、再開時にHTTP 206と`Content-Range`を検査します。serverがRangeを無視してHTTP 200を返した場合は、partialへの誤追記を防ぐため先頭から安全に書き直す設計です。size、SHA-256、SafeTensors markerが一致した後だけ`os.replace`で正式名へ移します。

CLIはdownload前に空き容量を検査し、中断時は`.part`を残して同じcommandから再開できます。installとrepairはrepository単位のnonblocking file lockを共有し、別processによるmodel fileの同時変更を防ぐ設計です。

## profile

### Krea2

取得物:

- Krea2 Turbo INT8 ConvRot checkpoint
- Qwen3-VL 4B encoder
- Qwen Image VAE

固定revisionは`Comfy-Org/Krea-2@8038ce89b91b042141541ad0fa51b985ca262c5f`です。新規環境の最終配置量は約17.68 GiBです。

```powershell
python .\tools\aikimi_setup.py install krea2
```

### Anima 3.8B

取得物:

- Anima 3.8B v1.1 BF16 bundle変換元
- Qwen3.5 4B encoder
- bundle内のSemantic Connector v2
- Anima標準Qwen3 0.6B encoder
- Qwen Image VAE

BF16 bundleは、DiT本体の主要520行列をINT8 ConvRotへ変換します。Semantic Connector v2はBF16のまま保持されます。セットアップBATは必要な`venv`とCUDA対応PyTorchも準備します。変換にはCUDA device 0を使用します。

```powershell
.\aikimi-setup.bat -Model anima38
```

変換と最終検証が成功すると、BF16変換元を削除します。変換元も残す場合は`--keep-source`を指定します。

```powershell
python .\tools\aikimi_setup.py install anima38 --keep-source
```

v1.1用PowerShell installerの一時peakは約17.82 GiBで、この値はAnima共通encoderとVAEがすでにある構成です。統一CLIが共通2ファイルも新規導入する場合は、約19.16 GiBに64 MiB以上の余裕を加えてください。

### SenseNova U1.5

取得物:

- 固定runtime source 27ファイル
- SenseNova U1.5 INT8 ConvRot checkpoint
- 公式8-Step LoRA
- runtime revision record

runtime revisionは`e6dfd45762eb46f805067fe079c14bcb643ccccd`です。各runtime fileもpath、size、SHA-256で固定しており、GitHub tree APIの件数だけを信用しません。

```powershell
python .\tools\aikimi_setup.py install sensenova
```

最終配置は約17.28 GiBです。統一CLIは単一`.part`を順番に取得します。既存PowerShell installerは32 MiB chunkを並列取得した後で単一modelへassembleするため、一時peakは約33.79 GiBです。

## 既存batとの互換

次の入口も残します。

```text
download_krea2_int8_convrot_models.bat
download_anima38_v11_int8_convrot_models.bat
download_anima38_int8_convrot_models.bat
download_sensenova_u15_int8.bat
```

Animaの旧ファイル名はv1専用の互換入口です。新規導入では`download_anima38_v11_int8_convrot_models.bat`を使います。

既存の自動化がPowerShell switchを使う場合も、`-KeepSource`、`-RuntimeOnly`、`-ModelOnly`を維持します。新しい導入では統一CLIを推奨します。

## verifyとrepair

`verify`は正式ファイルを変更しません。

```powershell
python .\tools\aikimi_setup.py verify anima38
```

破損した正式ファイルは、通常の`install`で上書きしません。明示的に`repair`を実行すると、破損した正式ファイルと残っている`.part`を次へ隔離し、先頭から再取得または再変換します。

```text
tmp/aikimi-setup/quarantine/<profile>/
```

```powershell
python .\tools\aikimi_setup.py repair krea2 --dry-run
python .\tools\aikimi_setup.py repair krea2
```

隔離は同じvolume内のrenameであり、空き容量を増やしません。再取得用の空き容量が不足する場合は、隔離内容を確認してから利用者自身が整理してください。

## 認証とlicense

現在の固定artifactはpublicで、Hugging Face tokenを必要としません。CLIはtoken引数を受け付けず、query string付きartifact URLも拒否します。tokenをcommand、URL、logへ書かないでください。

`list`とinstall結果はlicense URLを表示します。Krea2、Anima、SenseNova、Qwenには異なる条件があります。modelを利用または再配布する前に、固定revisionのmodel cardとlicense原文を確認してください。CLIの完全性検証は、利用許諾の判定ではありません。

## JSON結果

`--json`はschema version、command、profile、結果、相対path、license URLをstdoutへ出します。進捗はstderrへ分けるため、stdoutをJSON parserへ渡せます。認証値やsigned redirect URLは結果へ含めません。
