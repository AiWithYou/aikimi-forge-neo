# Architecture

## 全体像

Aikimi Studio Neoは、Forge Neo本体へすべてのruntimeを埋め込まず、機能ごとに責務を分けます。

```text
Windows launcher / Docker entrypoint
  -> startup security policy
  -> Forge WebUI and FastAPI
     -> standard txt2img / img2img / Extras
     -> Aikimi tab aliases
        -> Krea2 -> UI Preset krea / keep txt2img or img2img / other tabs -> txt2img
        -> Anima -> UI Preset anima / keep txt2img or img2img / other tabs -> txt2img / matching accordion
     -> Aikimi native tabs
        -> SenseNova bridge -> isolated worker
        -> MiniMax H3 bridge -> local ComfyUI process
     -> scoped Aikimi status
     -> Diagnostics and read-only API
  -> managed output and temporary directories
```

## 起動とsecurity policy

`launch.py`と`modules/initialize_util.py`は、modelを読み込む前に起動引数を検証します。`modules/aikimi_security`は、次の共通処理を持ちます。

- loopback既定とremote opt-in
- GradioとAPIの認証file
- APIへ返す設定の公開policy
- log、sysinfo、URL、例外のredaction
- 外部URL画像入力のfetch policy
- Gradioへ公開するpathの限定

`aikimi-launch.ps1`は、用途別profileを明示的な引数へ変換します。`webui-user.bat`は既存更新との互換を保つ入口です。新しい設定はGit管理外の`webui-user.local.bat`へ置きます。

## WebUIとAPI

`webui.py`がGradioとFastAPIを構築し、`modules/api/api.py`が標準APIとAikimi独自routeを登録します。APIの設定変更は`modules/options.py`の型と`restrict_api`を通ります。

Gradio UI callbackは、`api_name`と`api_visibility`をどちらも指定していない場合に限りprivateとして登録します。明示した`api_name`と`api_visibility="public"`は公開contractを維持します。この既定値は意図しないGradio API surfaceを減らすためのもので、remote modeの認証境界を置き換えません。

component constructorと通常のcallbackが返す`visible=False`は、lazyな既定動作を維持します。mounted hiddenは、DOM上の内部状態を必要とする`settings_json`とInputAccordion、MiniMax H3のmedia group、HyperWeaveのcustom size control、ControlNetのadvanced controlだけへ明示します。通常componentをすべて初期mountする方式には戻しません。

`modules/gradio_frontend_compat.py`は、固定したGradio 6.17.3のTabs reactive stormを避ける限定patchです。version、asset filename、元と置換後のSHA-256、三つの対象snippetを起動前に検証し、完全一致した場合だけ対象assetのexact routeを登録します。site-packagesと他のGradio assetは変更しません。

`modules/aikimi_diagnostics.py`はDiagnostics UIを構築し、次のread-only APIからJSONを返します。

```text
/aikimi/api/v1/health
/aikimi/api/v1/status
/aikimi/api/v1/capabilities
```

API routeは共通の認証処理を使います。healthは最小情報、statusとcapabilitiesはredaction済みの状態だけを返します。

## Aikimi UIとStatus

`extensions-builtin/aikimi-ui`は、Aikimi固有の小型ナビゲーション、ちびあいきみ、status用CSSを所有します。AikimiナビゲーションはGradioのタブ列へbuttonを挿入せず、`#tabs`直前の独立した1行として配置します。Forge由来の上部タブ、Quick Settings、`txt2img`、`img2img`、`Extras`、`Settings`はForge Neoの構成を維持し、Studio側CSSから変更しません。

`aikimi_tabs.js`は`window.AikimiTabs`と`aikimi:feature-tab-change`を公開します。feature keyとcapabilities APIは、`krea2`、`anima38`、`sensenova`、`minimax_h3`の識別子を共用します。Krea2はUI Presetの`krea`を選択し、現在のForgeタブが`txt2img`または`img2img`ならそのタブを維持するaliasです。別のタブから開いた場合は`txt2img`へ移動し、`Krea2 2-Stage Upscale`は自動選択しません。AnimaはUI Presetの`anima`を選択し、現在のForgeタブが`txt2img`または`img2img`ならそのタブを維持するaliasです。別のタブから開いた場合は`txt2img`へ移動し、選択したタブの`Anima 3.8B`設定欄を展開します。生成設定、queue、model状態を別のUI stateへ複製しません。

`modules/aikimi_status.py`が生成状態、queue、VRAM、model読込状態を要約し、`aikimiStatus.js`がactiveなAikimi操作領域の先頭へ一つのstatus要素をinline表示します。通常のForgeタブへ移動すると表示とstatus pollingを停止します。従来の全画面共通ヘッダーと固定オーバーレイは使用しません。技術logは残したまま、キャラクター向けmessageと詳細情報を分離しています。

assetは`assets/aikimi`から読みます。assetの読込失敗はUI全体の起動を止めません。animationを無効にした場合やreduced motion環境では、still assetを使います。UI再読込後もevent listenerとstatus要素を重複させず、keyboardとARIAの操作契約を維持します。

## Anima 3.8B

`extensions-builtin/anima-3-8b`は、通常のForge `txt2img`と`img2img`へQwen3.5 conditioningを追加するbuilt-in extensionです。v1.1では、checkpointに内包されたSemantic Connector v2がdenoiseの各stepで動作します。実装上は、52層checkpointの検査、encoder、connector、旧v1 adapter、UI callbackを別の責務として分離しています。

`modules_forge/anima_lora.py`は、28層、40層、52層のLoRA layoutを検査し、安全に特定できる完全なcoverageだけを変換します。sparseまたは曖昧なLoRAは自動変換しません。

INT8 ConvRot変換では、`tools/convert_anima38_int8_convrot.py`から共通streaming converterを呼び、v1.1のSemantic Connector v2をBF16のまま保持します。出力先は一時ファイルであり、tensor layoutとmetadataの検証を通過した場合だけ正式名へ移す設計です。

## SenseNova U1.5

SenseNovaは通常のForge samplerへ変換しません。境界は次のとおりです。

```text
extensions-builtin/sensenova-u15-studio
  -> modules_forge/sensenova_u15_bridge.py
     -> tools/sensenova_u15_worker.py
        -> pinned SenseNova runtime and checkpoint
```

Studioは入力とUI、bridgeはjob、VRAM解放、worker process、保存、cancel、workerは固定runtimeでの推論を担当します。workerを別processにすることで、終了時にweightとactivationをOSへ返します。

## MiniMax H3

MiniMax H3 Studioは、Aikimi Studio NeoへComfyUI runtime全体を組み込みません。

```text
extensions-builtin/minimax-h3-studio
  -> modules_forge/minimax_h3_bridge.py
     -> loopback ComfyUI runtime
```

bridgeは、runtime path、process identity、ComfyUI revision、必要node、model一覧を検査します。Forgeが起動したprocessだけを再起動し、外部launcherが所有するprocessを自動停止しません。

## model setup

`tools/aikimi_setup.py`はstdlibだけで動作し、固定manifest、download、resume、size、SHA-256、path containment、disk preflight、verify、repairを共通化します。Anima conversionだけは、準備済みForge venvとCUDAを使って既存converterを実行します。

既存のKrea2、Anima、SenseNova用batとPowerShell scriptは互換入口として残します。詳しい契約は[model-installation.md](model-installation.md)を参照してください。

## 保存先

- 通常画像はForgeの`output`または`outputs`
- SenseNovaは`outputs/sensenova_u15`
- MiniMax H3は`outputs/minimax_h3`
- setup partialは正式ファイルと同じdirectoryの`.part`
- repairで隔離したファイルは`tmp/aikimi-setup/quarantine`

管理対象外の任意pathをcleanup対象へ加えません。model、output、cache、log、secretはGit管理外です。
