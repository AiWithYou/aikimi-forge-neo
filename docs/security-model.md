# Aikimi Forge Neo security model

## 既定の境界

Aikimi Forge Neoは、信頼済みのWindows利用者が同じ端末のブラウザーから使う構成を既定とします。通常起動ではWebUIとAPIを`127.0.0.1`へbindし、LAN、Gradio share、ngrokへ公開しません。

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

## 2026年9月24日の依存関係修正

本体・SenseNova・QwenのCUDA環境をPyTorch 2.13.0、torchvision 0.28.0へ更新し、setuptools 83.0.0と両立させました。SenseNovaはTransformers 5.10.4へ移行し、専用の互換処理でRoPE設定、入力ID、明示的なキャッシュ位置を保持します。Transformers本体のAPIは変更しません。

Accelerateは上流1.15.0を基にした`1.15.0+aikimi.1`を同梱します。チェックポイント索引の各shardを読み込み前に検査し、ディレクトリ外への相対パス、絶対パス、Windowsドライブ指定、代替データストリーム、通常ファイルでない参照を拒否します。通常の読み込みAPI、サブディレクトリ、Hugging Faceのsnapshotからblobへのリンクは維持します。上流版へのバージョン変更だけを修正とは扱いません。由来・ライセンス・変更範囲は[vendor/accelerate/AIKIMI-PATCH.md](../vendor/accelerate/AIKIMI-PATCH.md)に記録しています。

ForgeのメタデータキャッシュはSQLiteとJSONへ移行しました。Pythonオブジェクトを復元するdiskcacheへの依存を除去しています。旧キャッシュを読み込まず、元ファイルから再計算して別名のDBに保存します。既存のモデル・設定・出力や旧キャッシュは削除しません。

期限付き例外と除外IDは撤去しました。CIはOSVの厳格監査を使い、固定Git版DiffusersとCUDA版PyTorchも監査対象に含めます。パッケージを照合できない場合も失敗します。Accelerateの修正は、監査データベースとは別に、実際の読み込みAPIを使う攻撃入力・正常入力の回帰テストで検証します。

既存環境では通常の依存準備で本体の旧既定CUDA版を更新します。`TORCH_COMMAND`または`TORCH_INDEX_URL`を指定している環境は、その指定を維持します。SenseNovaは`download_sensenova_u15_int8.ps1 -RuntimeOnly`、Qwenは専用セットアップで更新してください。旧環境に残るdiskcacheはForgeから使いません。外部拡張の依存関係を確認してからアンインストールできます。

以下の2026年9月3日・22日の記録は、修正前の履歴です。現在の依存定義・監査方針はこの節を参照してください。

## 依存関係監査の期限付き例外（修正前の記録）

2026年9月3日に、Diffusersを0.38.0、GitPythonを3.1.61、Transformersを5.10.4、huggingface-hubを1.5.0、PEFTを0.20.0へ更新しました。Diffusers 0.38.0は、`trust_remote_code`を回避する3件の脆弱性に対する公式修正版です。GitPython 3.1.61は、3.1.59未満に影響する`PYSEC-2026-3785`、`PYSEC-2026-3786`、`PYSEC-2026-3787`、`PYSEC-2026-3788`を修正済みです。Transformers 5.10.0はCVE-2026-9856の修正境界ですが、PyPIでyankされているため、同じ系列の非yank版である5.10.4を固定しています。PEFT 0.20.0は、Transformers 5で削除された`HybridCache`をPEFT 0.17.1が読み込んでDiffusersの起動を妨げる問題を避けるための固定です。

DiffusersとTransformersの旧版に対するadvisory IDは、CIの例外から削除しました。一方、公開済み修正版がない依存関係と、PyTorchの制約内で修正版を選べない依存関係については、次のadvisory IDを2026年9月30日までの期限付き例外として残しています。

- diskcache 5.6.3: `PYSEC-2026-2447`
- setuptools 81.0.0: `PYSEC-2026-3447`

diskcacheは、最新の5.6.3までが影響対象であり、監査時点では修正版が公開されていません。setuptoolsは83.0.0以降で修正済みですが、PyTorch 2.11が`setuptools<82`を要求するため、安全版との共通範囲がありません。CIはこの2件だけを除外し、新しいadvisory IDが追加された場合は即時に失敗します。期限までにdiskcacheの新規releaseとPyTorch 2.13以降への移行可否を再検証し、解除できない場合は理由と次回期限を改めて審査します。

## ローカル利用と依存監査

2026年9月12日、便利さと拡張のしやすさを優先し、v1.0.0で追加したAccelerateのAPI禁止と、SenseNovaの実行前ハッシュ検査・追加ファイル拒否・オフライン強制を撤回しました。Pythonのbytecode cacheも通常どおり利用します。

これらの制限を前提に追加した監査例外も取り消しました。AccelerateやSenseNova専用環境の依存関係で既知の問題が検出されると、GitHubの監査は失敗として報告します。監査の結果や期限を理由に、アプリの起動・生成・拡張を停止する処理はありません。検出内容は未修正として扱い、安全性を確認済みと表示しません。

## 2026年9月22日の互換性を優先した再確認

既存の機能・拡張API・生成の数値経路を保てる変更に限って確認しました。以下の依存監査の検出は未解消です。バージョン番号だけを上げたり、APIを禁止したりして監査を通す変更はしていません。

| 対象 | 現物・公式資料で確認したこと | 判断 |
|---|---|---|
| Accelerate 1.14.0 / `PYSEC-2026-3804` | 最新1.15.0でも対象のチェックポイント読み込み処理が同一。自作の一時ディレクトリだけを使い、両版の `load_checkpoint_in_model` と `load_checkpoint_and_dispatch` が、index内の相対・絶対パスで外側のsafetensorsを読むことを再現しました。 | 1.15.0への変更を脆弱性修正として扱いません。通常のForge・SenseNova経路ではこの2 APIの呼び出しを確認していませんが、拡張が使う可能性は残ります。APIの禁止や実行時の差し替えは追加しません。 |
| SenseNovaのTransformers 4.57.6 | 4.57系には修正済みの後続版がありません。監査を満たす5.10系はHub依存の更新に加え、Qwen3ConfigのRoPE設定と `create_causal_mask` の引数が現SenseNova runtimeと一致しません。 | 入力IDなどの経路を削って合わせる対応はしません。画像・入力ID・キャッシュ経路と、生成結果・速度の互換性を検証する移行として扱います。 |
| setuptools 81.0.0 / `PYSEC-2026-3447` | 修正版83以降は、現物のtorch 2.11.0の `setuptools<82` と両立しません。82では `pkg_resources` も削除されています。本件はsdistのUnicodeファイル除外に関する問題で、既知の生成経路にsdistの作成・公開処理は見つかりませんでした。 | 依存制約を無視した導入はしません。torch/torchvision/CUDA拡張をまとめて検証する更新時に再審査します。拡張のインストールや依存ビルド自体は禁止しません。 |

この確認では、autocropのOpenCV版判定にある `pkg_resources.parse_version` 依存だけを除去しました。元の名前は `packaging.version.Version` そのものであることを確認し、同じクラスを直接参照します。OpenCVの7種類の版判定とモデル選択が一致し、コーナー・エントロピーを使う3サイズの切り抜き・注釈計6画像も画素一致しました。顔検出モデルの取得・推論は行っていません。`pkg_resources` が使えない条件でのimportも成功しています。画像処理とモデルの計算を変えず、廃止APIへの直接依存を1つ減らす変更です。この変更によって上表の脆弱性が修正されたとは扱いません。

本体の期限付き例外とSenseNovaの独立監査を維持し、除外IDは増やしていません。Accelerateの「影響最終版1.14.0」というデータベース記載だけでは、1.15.0が修正済みだとは判断できません。

Transformers 5.10.0は修正境界の比較に使いましたが、PyPIでは公開内容の不備により撤回済みです。実際の移行候補には、現行本体が使う5.10.4など、撤回されていない版を個別に検証する必要があります。

一次資料：

- [Accelerateの報告と修正案](https://github.com/huggingface/accelerate/issues/4067)、[Hubキャッシュのリンクを維持する修正案](https://github.com/huggingface/accelerate/pull/4138)、[1.15.0の公開情報](https://github.com/huggingface/accelerate/releases/tag/v1.15.0)。修正案は確認時点で未マージです。
- [Transformers 5.10.0](https://github.com/huggingface/transformers/releases/tag/v5.10.0)。対象の報告IDは `PYSEC-2025-217`、`PYSEC-2026-2288`、`2289`、`2290`、`3929` です。
- [setuptoolsの公式advisory](https://github.com/pypa/setuptools/security/advisories/GHSA-h35f-9h28-mq5c)、[修正差分](https://github.com/pypa/setuptools/commit/dd9f436a36486b4cb8a4c70a2321548b0be09b8f)、[変更記録](https://setuptools.pypa.io/en/latest/history.html#v83-0-0)。

## 非目標

- zero-trust gatewayは、このリポジトリの対象外です。
- ローカル管理者や同じWindows accountを制御した攻撃者は、想定するsecurity boundaryの外側です。
- third-party extension、model、custom runtimeは、導入者が個別に安全性を確認します。
- model licenseと生成物の権利は、利用者が配布元の原文から判断してください。
