# Aikimi Studio Neo security model

## 既定の境界

Aikimi Studio Neoは、信頼済みのWindows利用者が同じ端末のブラウザーから使う構成を既定とします。通常起動ではWebUIとAPIを`127.0.0.1`へbindし、LAN、Gradio share、ngrokへ公開しません。

推奨入口は次です。

```powershell
.\aikimi-launch.ps1 -Profile LocalSafe
```

`aikimi-launch.bat`を使う場合は、profile名を第1引数へ渡します。

```text
aikimi-launch.bat LocalSafe
```

## 起動profile

| profile | bindと用途 | 主な引数 |
|---|---|---|
| `LocalSafe` | loopback WebUIとAPI | `--server-name 127.0.0.1 --api` |
| `LocalAPI` | loopback APIのみ | `--nowebui --server-name 127.0.0.1` |
| `LANAuthenticated` | 認証付きLAN公開 | `--aikimi-remote --listen --api`と2つのauth file |
| `Development` | loopback UI debug | `--ui-debug-mode` |
| `LowVRAM` | loopback、低VRAM | `--lowvram --tiled-conv2d 64` |
| `RTX3090Recommended` | loopback、RTX 3090向け | BnB、tiled Conv2d、cudaMallocAsync |

`--listen`、loopback以外の`--server-name`、`--share`、`--ngrok`は、`--aikimi-remote`がなければ起動前に失敗します。WebUIを公開する場合はGradio認証、APIを公開する場合はAPI認証が必要です。両方を有効にする場合は、両方の認証を設定します。

## 認証ファイル

`LANAuthenticated`は次を読みます。

```text
secrets/gradio-auth.txt
secrets/api-auth.txt
```

各行は`username:password`形式です。複数利用者は1行ずつ記載できます。空のusername、空のpassword、64 KiBを超えるファイルは拒否します。`secrets/`はGit管理外です。

```powershell
New-Item -ItemType Directory -Force .\secrets | Out-Null
notepad .\secrets\gradio-auth.txt
notepad .\secrets\api-auth.txt
.\aikimi-launch.ps1 -Profile LANAuthenticated
```

認証値を`COMMANDLINE_ARGS`へ直接書かないでください。Basic認証情報を暗号化されていないHTTPでLAN外へ送らないでください。外部公開では、別のreverse proxyでTLSを終端し、firewallでも接続元を制限してください。

## container

containerも`127.0.0.1`が既定です。外部bindには`AIKIMI_CONTAINER_REMOTE=1`が必要です。さらに、`COMMANDLINE_ARGS`またはcontainer引数でGradioとAPIのauth fileを指定しなければ起動しません。

```text
AIKIMI_CONTAINER_REMOTE=1
--gradio-auth-path /run/secrets/gradio-auth.txt
--api-auth-path /run/secrets/api-auth.txt
```

secretはimageへ含めず、read-only volumeやcontainer secretとしてmountしてください。

## 信頼境界

```text
Browser
  -> Gradio / FastAPI authentication
  -> Forge request validation and options policy
  -> GPU model runtime
     -> SenseNova isolated worker
     -> local MiniMax H3 ComfyUI bridge
  -> managed outputs / temporary files
```

- Browser入力: API schema、OptionInfo policy、path policyによる検証
- API経由の設定変更: `restrict_api`と型の検査
- URL画像入力: safe fetch policyによるscheme、DNS、redirect、size、Content-Typeの確認
- Gradio公開path: 管理されたoutput、temporaryと、activeな静的assetの個別fileへの限定
- SenseNova: Forge UI、bridge、専用workerの責務分離
- MiniMax H3: loopback ComfyUIと選択runtime identityの検査
- model installer: 固定revision、size、SHA-256の確認後に正式名へ移動

管理されたoutput、temporary、Canvas assetがsymlinkやjunctionを経由して管理root外へ解決される場合は、起動を停止します。外部保存先を使う場合でも、WebUIの公開pathへ任意directoryを追加せず、生成後の別工程で移動してください。

### Gradio 6の静的asset

Gradio 6へ埋め込むJavaScriptとCSSは、現在のmount pathを保持できる相対URL `gradio_api/file=`から取得します。静的UI資産についてはparent directoryを`allowed_paths`へ加えず、起動時に使う個別fileだけを列挙します。対象はrootの`script.js`と`style.css`、activeなroot／extensionの`.js`と`.mjs`、active extensionの`style.css`、Forge Canvasの`canvas.js`と`canvas.css`、有効時の`notification.mp3`、card placeholderです。outputとtemporaryは、従来どおり管理directory単位で許可します。

extensionのPython、任意HTML、設定file、model、inactive extensionのasset、許可root外へ解決されるsymlinkは公開しません。`extensions`、`extensions-builtin`、`javascript`などのdirectory自体も静的assetのallowlistへ入りません。rootまたはsubpathへmountした実Gradio appを使い、許可したfileが取得でき、その他がHTTP 403になることを回帰テストで確認します。

Gradio 6.17.3のfile routeへ外部URLを渡した場合は、redirectやproxyを行いません。現行とdeprecatedのrouteを認証後にも再検査し、HTTP、HTTPS、protocol-relative、userinfo、複数回encodeされたURLをtarget非表示のHTTP 403で拒否します。local exact assetの配信だけを維持します。

## API

Local Safeでは次のread-only endpointをloopbackから確認できます。

```text
GET /aikimi/api/v1/health
GET /aikimi/api/v1/status
GET /aikimi/api/v1/capabilities
```

remote modeでは、これらを含むAPI routeが認証境界を通ります。healthは秘密値を返しません。status、capabilities、checkpoint、追加module、upscaler、face restorerの一覧も、認証情報やローカル絶対パスを返さず、安全なselectorまたはbasenameだけを公開します。

## ログとsysinfo

共通redactionは、password、token、secret、auth、Cookie、Authorization、URL userinfo、credentialに見えるquery parameterをマスクします。例外をclientへ返す場合は、長さを制限した安全な要約を使います。

redactionは最後の防御です。利用者は、共有前に[SECURITY.md](../SECURITY.md)の確認項目を目視してください。

## 依存関係監査の期限付き例外

2026年9月3日に、Diffusersを0.38.0、GitPythonを3.1.61、Transformersを5.10.4、huggingface-hubを1.5.0、PEFTを0.20.0へ更新しました。Diffusers 0.38.0は、`trust_remote_code`を回避する3件の脆弱性に対する公式修正版です。GitPython 3.1.61は、3.1.59未満に影響する`PYSEC-2026-3785`、`PYSEC-2026-3786`、`PYSEC-2026-3787`、`PYSEC-2026-3788`を修正済みです。Transformers 5.10.0はCVE-2026-9856の修正境界ですが、PyPIでyankされているため、同じ系列の非yank版である5.10.4を固定しています。PEFT 0.20.0は、Transformers 5で削除された`HybridCache`をPEFT 0.17.1が読み込んでDiffusersの起動を妨げる問題を避けるための固定です。

DiffusersとTransformersの旧版に対するadvisory IDは、CIの例外から削除しました。一方、公開済み修正版がない依存関係と、PyTorchの制約内で修正版を選べない依存関係については、次のadvisory IDを2026年9月30日までの期限付き例外として残しています。

- diskcache 5.6.3: `PYSEC-2026-2447`
- setuptools 81.0.0: `PYSEC-2026-3447`

diskcacheは、最新の5.6.3までが影響対象であり、監査時点では修正版が公開されていません。setuptoolsは83.0.0以降で修正済みですが、PyTorch 2.11が`setuptools<82`を要求するため、安全版との共通範囲がありません。CIはこの2件だけを除外し、新しいadvisory IDが追加された場合は即時に失敗します。期限までにdiskcacheの新規releaseとPyTorch 2.13以降への移行可否を再検証し、解除できない場合は理由と次回期限を改めて審査します。

## 非目標

- zero-trust gatewayは、このリポジトリの対象外です。
- ローカル管理者や同じWindows accountを制御した攻撃者は、想定するsecurity boundaryの外側です。
- third-party extension、model、custom runtimeは、導入者が個別に安全性を確認します。
- model licenseと生成物の権利は、利用者が配布元の原文から判断してください。
