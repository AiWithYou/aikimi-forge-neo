# Codexへの引き継ぎ：生成ジョブの安全性

更新: 2026-09-12。対象: `AiWithYou/aikimi-forge-neo` の `neo`。

`bb1a2c97` までfast-forwardしてから実装した。開始時の背景除去関連・H3 CSSなど11ファイルの既存差分は保持している。

## 実装済み

- Forge UI/API、SenseNova、H3のGPU所有権を共通のqueue_lockに統一した。`/sdapi/v1/unload-checkpoint` もこの排他を通る。
- SenseNovaはモデル退避前に所有権を取得する。停止未確認時は入力・worker登録・GPU所有権を保持し、再停止または自然終了後に回収する。待機中のキャンセルは生成ロックを取らない。
- H3は送信前にUUID・コピー済み素材・送信先process情報を永続化する。応答消失時は同じIDを照会し、再送しない。アプリ再起動後も、最初のGPU処理より先に未確定ジョブの所有権を復元する。
- H3のキャンセル意図と受付確認を区別した。意図だけの404やgenerator終了では入力を削除せず、所有権も返却しない。
- SenseNovaの依存監査を独立したCI jobに追加した。Windows CIに今回の回帰テストを追加した。CPU suiteの記録先は一時ディレクトリに隔離し、利用者の実ジョブに触れない。

詳細なAPI契約と監査結果: [generation-job-safety.md](generation-job-safety.md)。

## 確認結果

| 検証 | 結果 |
| --- | --- |
| CPU suite | 1,373件、失敗0件。skip 43、expected failure 1 |
| Windows向け選択テスト | `--preload modules.shared` 付きで44件合格 |
| Windowsの実CPU子process | 参照画像を開いたまま待機するworkerをキャンセルし、終了・入力削除・次回受付を確認 |
| Windows / RTX 3090 / SenseNova | Forge APIからの切替待機、512×512の実モデルキャンセル、次回生成成功、待機していたForge API生成成功 |
| 固定版ComfyUIの実HTTP | `efa6c8f8` を起動。正常完了・受付後の応答消失とアプリ再起動・実行中キャンセルが成功。投入は各1回、実行中の素材を保持し、終了後に回収 |
| Windows / RTX 3090 / H3実モデル | Forge APIからの切替待機、608×352・124フレーム・約5.17秒・ステレオ音声付きMP4の保存、待機していたForge API生成成功。省RAM・1ステップ |
| Ruff | 新規ファイルに指摘なし。変更した既存ファイルの指摘は増加なし |
| workflow / 文書 | actionlintと日本語文体検査が合格 |

実GPUの比較は速度や画質の評価ではない。通常UI入口の排他はCPUテスト、実モデルの往復切替は通常生成APIで確認した。

## セキュリティ検出

Gitleaksの合成検出コントロールは成功し、完全なGit履歴の走査で秘密値の検出は0件だった。`.gitleaksignore` は変更していない。

pip-auditは失敗を維持している。本体ではAccelerate・diskcache・setuptools、独立SenseNova環境ではAccelerate・Transformers・setuptoolsを検出した。既存の検出を無効化せず、依存バージョンも変更していない。CUDAローカル版のtorch・torchvisionはPyPI照合の対象外として報告された。

## 実行コマンドと記録

```powershell
venv/Scripts/python.exe tools/run_ci_tests.py --verbosity 1
venv/Scripts/python.exe tools/run_ci_tests.py --preload modules.shared --module tools.tests.test_ci_workflow_boundaries --module tools.tests.test_gpu_ownership --module tools.tests.test_minimax_h3_submission --module tools.tests.test_sensenova_worker_cleanup --module tools.tests.test_sensenova_u15_bridge --module tools.tests.test_run_ci_tests
uv tool run --python 3.13 pip-audit --path models/SenseNova-U1/worker-env/Lib/site-packages
```

このcheckoutの実検証スクリプト・結果・ログは `tmp/job-safety-validation/` に保持している。最終記録は `cpu-suite-isolated-final.log`、`windows-selected.log`、`gpu-report.json`、`h3-gpu-report-verified.json`、`contract-verified/report.json`。`verify_gpu.py` がForge/SenseNova、`verify_h3_gpu.py` がForge/H3、`verify_contract.py` が固定版ComfyUIのHTTP契約を検証する。日本語を標準出力へ出す実検証スクリプトは `python -X utf8` で実行する。

## 未実施

- GitHubへのpushとリモートCI。
- ブラウザーでの手操作による確認。
- 全品質・全高速化設定の実GPU検証。今回のH3実GPU検証は省RAM設定を対象とする。
