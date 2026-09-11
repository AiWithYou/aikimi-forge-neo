# Release checklist

## 1. 対象と履歴

- [ ] release対象repositoryが`AiWithYou/aikimi-studio-neo`である
- [ ] default branchが`neo`である
- [ ] READMEのclone URLと`cd`先が正しい
- [ ] upstream同期基準をcommit SHAで記録した
- [ ] unrelatedな変更とローカル生成物を含めていない
- [ ] model、dataset、checkpoint、output、cache、venvを追跡していない

## 2. セキュリティ

- [ ] 通常起動が`127.0.0.1`だけへbindする
- [ ] `--listen`、remote server name、share、ngrokが`--aikimi-remote`なしで失敗する
- [ ] remote WebUIへのGradio認証を確認した
- [ ] remote APIへのAPI認証を確認した
- [ ] auth fileをGit管理外へ配置した
- [ ] cmd flags、Options API、sysinfo、logのredaction testが通る
- [ ] URL画像fetcherのDNS、redirect、size、Content-Type testが通る
- [ ] Gradio allowed pathsがoutput、managed temporary、static assetに限定される
- [ ] Gitleaksで現在treeと全履歴を検査した
- [ ] GitHub Private Vulnerability Reportingを有効にした
- [ ] SECURITY.mdの非公開報告導線へlogin後に到達できることを確認した

Private Vulnerability Reportingを有効にできないreleaseは、第三者向け配布をBlockedとします。

## 3. 依存関係とCI

- [ ] Python 3.13のCPU／offline unit testが通る
- [ ] Windows smoke testが通る
- [ ] installer functional testが通る
- [ ] Ruff checkとformat checkが通る
- [ ] PowerShell parserがすべての`.ps1`を受理する
- [ ] actionlintがworkflowを受理する
- [ ] pip-auditのfindingを確認した
- [ ] `docs/security-model.md`の期限付きadvisory例外を再審査し、期限切れを残していない
- [ ] CodeQLのfindingを確認した
- [ ] dependency reviewを確認した
- [ ] GitHub Actionsの`uses`が確認済みcommit SHAへ固定されている

ローカルの中心command:

```powershell
.\venv\Scripts\python.exe .\tools\run_ci_tests.py --verbosity 1
.\venv\Scripts\python.exe -m unittest -v tools.tests.test_aikimi_setup
.\venv\Scripts\python.exe -m ruff check tools\aikimi_setup.py tools\tests\test_aikimi_setup.py
.\venv\Scripts\python.exe -m ruff format --check tools\aikimi_setup.py tools\tests\test_aikimi_setup.py
```

## 4. model installer

- [ ] `list`がKrea2、Anima 3.8B、SenseNovaを表示する
- [ ] `--dry-run`がnetworkとfilesystemを変更しない
- [ ] path traversal、absolute path、symlink escapeを拒否する
- [ ] HTTP Range、Range無視、途中切断、不正Content-Rangeをtestした
- [ ] size／SHA-256 mismatchで正式fileを作らない
- [ ] disk不足をdownload前に検出する
- [ ] repairが管理root外を変更しない
- [ ] JSON stdoutへtokenとsigned URLを出さない
- [ ] 固定revision、size、hashを配布元metadataと照合した
- [ ] model license URLを利用者へ表示する

## 5. 文書とlicense

- [ ] README、SECURITY、CONTRIBUTING、security model、architectureを更新した
- [ ] model installationとtroubleshootingを更新した
- [ ] rootとnested third-party noticesを確認した
- [ ] bundled tokenizerに必要なApache-2.0本文を含めた
- [ ] fontの出典とlicenseを確認した
- [ ] Aikimi画像assetの由来、権利保有者、再配布条件を権利者が確認した
- [ ] PDFとUI画像をreleaseへ含める権利を確認した
- [ ] model weightをrelease archiveへ含めていない

Aikimi画像assetの権利noticeを確認できないreleaseは、配布条件の監査を完了扱いにしません。

日本語文書の最終確認:

```powershell
& "$env:USERPROFILE\.codex\skills\enforce-japanese-style\scripts\japanese_style_guard.py" lint `
  README.md SECURITY.md CONTRIBUTING.md THIRD_PARTY_NOTICES.md docs
```

findingがある場合は、単語だけを置換せず、対象文全体を書き直します。

## 6. 実機確認

- [ ] Local SafeでWebUIとAPIを起動した
- [ ] browserから`127.0.0.1`だけで開ける
- [ ] Forge由来の上部タブとQuick Settingsが常に表示される
- [ ] Aikimiの4入口が`#tabs`直前の小型1行にあり、Gradio所有tablistのchild、順序、label、ARIAを変更しない
- [ ] `Krea2`がUI Presetの`krea`を選択し、現在のForgeタブが`txt2img`または`img2img`ならそのタブを維持する
- [ ] `Krea2`を別のタブから開くと`txt2img`へ移動し、`Krea2 2-Stage Upscale`を自動選択しない
- [ ] `Anima`がUI Presetの`anima`を選択し、現在のForgeタブが`txt2img`または`img2img`ならそのタブを維持する
- [ ] `Anima`を別のタブから開くと`txt2img`へ移動し、選択したタブのAnima 3.8B設定欄を展開する
- [ ] `SenseNova`と`MiniMax H3`が専用タブを直接開く
- [ ] 通常のForgeタブに全画面共通Aikimi headerや固定status overlayが表示されない
- [ ] ちびあいきみがAikimi機能内だけに表示され、通常のForgeタブではpollingを停止する
- [ ] Aikimiタブをkeyboardで選択でき、ARIA上で同じpanelに二つのselected tabが生じない
- [ ] UI再読込後もtab、event listener、status要素が重複しない
- [ ] Gradio Tabs compat assetの元と置換後SHA-256、exact route、root／subpath、ETagを確認した
- [ ] 現行とdeprecatedのGradio file routeが外部URLをLocationなしのHTTP 403で拒否する
- [ ] 実ブラウザーconsoleにlazy mount由来の未処理errorがない
- [ ] DiagnosticsのReady／Warning／Blockedと修復案を確認した
- [ ] capabilities APIが現在のmodel状態を返す
- [ ] remote modeが認証なしで失敗する
- [ ] 認証付きLAN profileを隔離networkで確認した
- [ ] Krea2、Anima、SenseNova、MiniMax H3の対象testを分離して記録した

GPU live testを実行していない機能は、成功と記載しません。GPU、driver、PyTorch、model revision、入力条件、出力確認方法をrelease noteへ残します。

## 7. Gitと公開確認

- [ ] `git diff --check`が成功する
- [ ] 対象fileだけをcommitした
- [ ] `neo`を対象remoteへpushした
- [ ] remote HEADとlocal HEADが同じcommit SHAを指す
- [ ] required workflowがすべて成功した
- [ ] 公開README、SECURITY、download commandをGitHub上で確認した
- [ ] release archiveにsecret、local config、model、outputがない
