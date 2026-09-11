# Gradio互換性方針

## 現在の固定構成

Aikimi Studio Neoは、WindowsとPython 3.13を対象に、次の組み合わせを直接依存として固定しています。

- Gradio `6.17.3`
- gradio-client `2.5.0`
- FastAPI `0.141.1`
- Starlette `1.6.0`
- Diffusers `0.38.0`
- huggingface-hub `1.5.0`
- PEFT `0.20.0`
- Transformers `5.10.4`

Gradioは`requirements.txt`から導入します。起動コードに別のGradioインストール指定はありません。PyTorchとtorchvisionはCUDA構成ごとにwheelが異なるため、従来どおり`TORCH_COMMAND`と`TORCH_INDEX_URL`を使って起動処理が管理します。

## 6.17.3を選んだ理由

旧版のGradio 4.40.0には、ファイル漏えい、CORS、SSRF、Windows上のパストラバーサルなど、Aikimi Studio Neoの利用方法に関係する既知の脆弱性が複数あります。特に、Python 3.13以降のWindowsでは、認証を有効にしていても任意ファイルを読み取られる可能性があるため、6.7.0未満は利用できません。

2026年8月26日に、Gradio 6.17.3をPython 3.13の隔離環境へ導入し、`pip-audit`でGradioに既知の脆弱性が検出されないことを確認しました。Gradio 6.17.3はhuggingface-hub 1.5.0も許容するため、CVE-2026-9856を修正したTransformers 5.10.4と同じ環境で利用できます。

Gradio 6.18.0以降では、複数の同種コンポーネントを同時に表示したときに画面が停止する問題が修正済みです。モデル依存関係はhuggingface-hub 1.5.0へ移行しましたが、Aikimi Studio Neoは6.17.3の固定assetに限定したcompat workaroundを使用しています。次回のGradio更新では、このworkaroundと静的asset routeのguardを再監査し、実WebUIで挙動を確認します。

参考資料:

- [GradioのSecurity Advisories](https://github.com/gradio-app/gradio/security/advisories)
- [WindowsとPython 3.13に関するGHSA-39mp-8hj3-5c49](https://github.com/gradio-app/gradio/security/advisories/GHSA-39mp-8hj3-5c49)
- [Tabsのreactive stormを修正したGradio PR #13509](https://github.com/gradio-app/gradio/pull/13509)
- [6.17.3のfile route open redirectを報告したIssue #13608](https://github.com/gradio-app/gradio/issues/13608)
- [Gradio 6 Migration Guide](https://www.gradio.app/guides/gradio-6-migration-guide)

## 6.17.3の既知制約

Gradio 6.6.0から6.17.xには、1回のイベントで複数の同種コンポーネントを`visible=False`から`visible=True`へ変更すると、ブラウザーが応答しなくなる既知の問題があります。Aikimi Studio Neoでは、次の方針で影響を抑えます。

- component constructorの`visible=False`を変更せず、通常の非表示componentはlazyを維持
- 通常のcallbackが返す`gr.update(visible=False)`もBoolean `False`のままにし、不要なsubtreeをmountしない
- DOM上の内部状態が必要な`settings_json`とInputAccordionを、明示的なmounted hiddenとして保持
- 複数controlを同時に初表示する既知経路だけ、`keep_hidden_component_mounted(False)`または明示的な`visible="hidden"`を指定
- 追加のmounted hiddenをMiniMax H3のmedia group、HyperWeaveのcustom size control、ControlNetのadvanced controlへ限定
- visibilityを変更するUIには実ブラウザーの応答性テストを追加
- UI追加時はPython側のBlocks構築テストと実ブラウザー検査を併用

短期固定版には、この機能上の制約と回避策が残ります。Gradio 6.18.0以降へ移行できる状態になった時点で、回避策を再評価します。

### Tabs overflowの限定compat workaround

Gradio 6.17.3には、多数の`TabItem`を登録した場合に、frontendのoverflow幅計測と1項目ごとのtab登録がreactive updateを繰り返す問題もあります。Aikimi Studio Neoでは、`modules/gradio_frontend_compat.py`が次の条件をすべて満たした場合だけ、該当処理を配信時に置き換えます。

- Gradioのversionが`6.17.3`
- frontend assetのfilenameが監査時の固定値と一致
- site-packagesから読み込んだ元assetのSHA-256が固定値と一致
- 対象となる3個のminified snippetが、それぞれ1回だけ存在
- 置換後assetのSHA-256が監査時の固定値と一致

workaroundはsite-packagesやwheelを変更せず、FastAPIの完全一致routeをGradioの汎用`/assets/{path}`より前へ登録します。このrouteがメモリ上で置き換えるのは該当assetだけであり、その他は従来のGradio routeが担当します。固定条件を確認できない場合は起動処理を止め、再監査が必要な状態を明示する設計です。

置換後はoverflow menuの計算を行わず、全tab buttonを表示すると同時に、初期tab一覧を1回のbatchで同期します。画面幅に収まらない場合のfallbackは、CSSの横スクロールです。このworkaroundは、Gradio 6.17.3のTabs mount stormへ対象を限定しています。PR #13509のSvelte 5移行全体は、依存関係を更新する段階で別途評価します。

固定されたtab構成の初期表示、複数回の切替、keyboard focusとactivation、`gr.render`によるtab追加、ブラウザーreloadはChromium回帰テストの対象です。一方、backendから既存`Tab.visible`だけを更新してtab buttonを増減する動作は、patchを適用しないGradio 6.17.3でも反映されません。Aikimi Studio Neoの固定tabはこの動作へ依存しませんが、extensionは既存tabの動的な表示切替を前提にしないでください。この制約は、Gradio 6.18.0以降へ移行する際に再評価します。

## Gradio 6の静的asset route

WebUIとForge Canvasがheadへ埋め込むJavaScriptとCSSは、Gradio 6の`API_PREFIX`から作る相対URL `gradio_api/file=`を使います。相対URLにより、WebUIをrootへ置いた場合もsubpathへmountした場合も、同じmount prefixのfile routeへ接続できます。

`allowed_paths`は、rootの`script.js`と`style.css`、activeなroot／extensionの`.js`と`.mjs`、active extensionの`style.css`、Forge Canvasの`canvas.js`と`canvas.css`、有効時の`notification.mp3`、card placeholderを個別fileとして登録します。これらのparent directoryは登録しません。extensionのPython、任意HTML、設定file、model、inactive extensionのassetは拒否します。outputとtemporaryだけは、管理directory単位の許可を維持します。

Gradio 6.17.3の`/gradio_api/file=`には、外部URLを302へ返すopen redirectがあります。そこでAikimi Studio Neoは、現行routeとdeprecated routeのGET／HEADを認証middlewareの内側で検査し、HTTP、HTTPS、protocol-relative、userinfo、encoded URLをtarget非表示のHTTP 403で拒否する設計です。local exact assetの配信は維持し、Gradioを後続修正版へ更新する段階でguardの撤去可否を再評価します。

## Gradio callbackのAPI visibility

UI eventへ`api_name`と`api_visibility`の指定がない場合は、`api_visibility="private"`を既定値として設定します。明示した`api_name`の公開APIと`api_visibility="public"`は変更しません。private既定は意図しないcallbackの公開面を減らしますが、remote modeで必要なGradio認証とAPI認証は引き続き適用します。

## 更新時の確認項目

GradioまたはFastAPI関連の固定版を変更する場合は、少なくとも次を確認します。

1. `uv pip install --dry-run -r requirements.txt`による依存関係の解決
2. `python -m pip check`の成功
3. `pip-audit`によるGradio関連の既知脆弱性検査
4. WindowsとPython 3.13でのWebUI構築
5. 画像アップロード、Gallery、動画出力、認証、`allowed_paths`、`blocked_paths`の動作
6. root／subpath mountでの相対`gradio_api/file=`解決、個別静的assetの取得、Python／HTML／設定file／modelのHTTP 403拒否
7. unnamed callbackのprivate既定と、明示したnamed／public API contractの維持
8. constructorと通常callbackのlazy `visible=False`と、`settings_json`、InputAccordion、H3、HyperWeave、ControlNetだけが使うmounted hiddenの分離
9. Krea2、Anima、SenseNova、MiniMax H3のUI構築と上部タブ順序
10. Aikimi aliasの移動契約
    - Krea2はUI Presetの`krea`を選択し、現在のForgeタブが`txt2img`または`img2img`ならそのタブを維持する。別のタブから開いた場合は`txt2img`へ移動し、`Krea2 2-Stage Upscale`は自動選択しない
    - AnimaはUI Presetの`anima`を選択し、現在のForgeタブが`txt2img`または`img2img`ならそのタブを維持する。別のタブから開いた場合は`txt2img`へ移動し、選択したタブのAnima設定欄を展開する
11. 通常のForgeタブへ戻った場合のAikimi Status非表示とstatus polling停止
12. UI再読込後のタブ、event listener、status要素の重複防止
13. 実ブラウザーでのselector、keyboard、ARIA、reduced motion、主要なvisibility変更後の応答性
14. GPU依存ライブラリー更新時の対象ワークフロー別実生成

ローカルの基本確認には次を使用します。

```powershell
uv pip install --dry-run --python .\venv\Scripts\python.exe -r requirements.txt
uv pip install --python .\venv\Scripts\python.exe -r tools\requirements-test.txt
.\venv\Scripts\python.exe -m pip check
.\venv\Scripts\python.exe -m unittest discover -s tools\tests -p "test_*.py" -q
```
