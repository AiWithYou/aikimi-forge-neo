# Troubleshooting

## Local Safeが起動しない

最初にPowerShell 7、Python、Gitを確認します。

```powershell
pwsh --version
python --version
git --version
```

Aikimi Studio Neoの対象はPython 3.13です。別のPythonで作成した`venv`を使い回さないでください。再構築前に必要なローカル設定を退避し、modelや生成物を削除対象へ含めないでください。

```powershell
.\aikimi-launch.ps1 -Profile LocalSafe
```

既定URLは`http://127.0.0.1:7861`です。

## remote exposureの起動エラー

`--listen`、loopback以外のserver name、share、ngrokだけを追加すると、安全policyが起動を止めます。LAN公開には次を使います。

```powershell
.\aikimi-launch.ps1 -Profile LANAuthenticated
```

次の両方を用意してください。

```text
secrets/gradio-auth.txt
secrets/api-auth.txt
```

各行は`username:password`です。空file、書式違反、64 KiB超過は拒否されます。認証情報をcommand lineへ移さないでください。

## containerをLANへ公開できない

containerはloopbackが既定です。外部bindには`AIKIMI_CONTAINER_REMOTE=1`を設定し、GradioとAPIのauth fileをread-onlyでmountします。

```text
AIKIMI_CONTAINER_REMOTE=1
COMMANDLINE_ARGS=--gradio-auth-path /run/secrets/gradio-auth.txt --api-auth-path /run/secrets/api-auth.txt
```

auth fileがない状態でremote modeを有効にすると、起動前に失敗します。host側のport公開、firewall、TLS reverse proxyも別途確認してください。

## APIへ接続できない

Local SafeはAPIもloopbackへ起動します。最小確認は次です。

```powershell
Invoke-RestMethod http://127.0.0.1:7861/aikimi/api/v1/health
Invoke-RestMethod http://127.0.0.1:7861/aikimi/api/v1/capabilities
```

`LocalAPI`は`--nowebui`を使うため、Gradio画面を表示しません。remote modeではBasic認証が必要です。

## DiagnosticsがWarningまたはBlockedになる

Diagnosticsは、Python、PyTorch、CUDA、GPU、RAM、disk、runtime、model、output書込、公開状態、認証状態を確認します。項目のdetailにある修復手順を先に試してください。

- model missing: model setupの`verify`を実行
- output not writable: repositoryとoutput directoryの権限を確認
- CUDA unavailable: NVIDIA driverとPyTorch buildを照合
- ComfyUI unavailable: H3 Studioのruntime pathとrevisionを確認
- remote authentication blocked: 2つのauth fileと書式を確認

## model setupが失敗する

計画と空き容量を先に確認します。

```powershell
python .\tools\aikimi_setup.py install sensenova --dry-run
python .\tools\aikimi_setup.py verify sensenova
```

hash mismatch、size mismatch、完成済みの不正partialがある場合は、通常installで上書きしません。

```powershell
python .\tools\aikimi_setup.py repair sensenova --dry-run
python .\tools\aikimi_setup.py repair sensenova
```

repairは破損fileを`tmp/aikimi-setup/quarantine`へ移します。隔離しても空き容量は増えません。

## downloadが途中で止まる

同じinstall commandを再実行してください。`.part`の長さからHTTP Rangeで再開します。proxyやCDNがRangeを無視した場合は、partialへ完全responseを追記せず、先頭から安全に再取得します。

長時間停止する場合は、firewall、HTTPS inspection、proxy、空き容量を確認してください。tokenをURL queryへ追加しないでください。現在の固定artifactはpublicです。

## Anima conversionが失敗する

Anima 3.8Bの変換には、準備済みForge venv、CUDA、対応GPU、RAM、diskが必要です。Local Safeを一度起動して依存関係を準備し、他のGPU処理を止めてから再実行してください。

変換中断後は、BF16 sourceと`.safetensors.part`を保持します。手動で正式名へ変更せず、repairの計画を確認してください。

```powershell
python .\tools\aikimi_setup.py repair anima38 --dry-run
```

## SenseNovaがmodel準備で止まる

別processがGPUを占有している場合、SenseNova workerはそのprocessのVRAMを解放できません。`nvidia-smi`とTask Managerで別の生成processを確認してください。

```powershell
nvidia-smi
```

24GB Safeは、参照2枚以下、各参照約512²、出力2048² pixels以下を基準にします。画像編集では公式8-Step LoRAを使わず、Quality 50-Stepを選びます。

## MiniMax H3 backendへ接続できない

H3 Studioはloopback ComfyUI、必要node、model、core revisionを検査します。UNC path、network drive、外部URL runtimeは拒否します。外部launcherが起動したComfyUIをAikimi Studio Neoが自動停止することはありません。

runtime cardで次を確認してください。

- ComfyUI directory
- core revision
- Comfy Kitchen version
- 5つのH3 model
- process identity
- RAMとOS commit余力

## unit testがmodel downloadを始める

共通runnerを使います。runnerはCPUとoffline環境変数をtest import前に設定します。

```powershell
.\venv\Scripts\python.exe .\tools\run_ci_tests.py --verbosity 1
```

live test用環境変数を個別に有効にしたshellでは実行しないでください。

## logやsysinfoを共有する

自動redaction後でも、認証情報、URL query、絶対path、prompt、入力basenameを目視してください。公開場所へ貼る前に[SECURITY.md](../SECURITY.md)を確認してください。
