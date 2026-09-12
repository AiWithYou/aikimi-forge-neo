# 生成ジョブの所有権とH3送信記録

対象: Forge、SenseNova U1.5、MiniMax H3。確認日: 2026-09-12。

## GPUの排他

- `modules_forge/gpu_ownership.py` の `queue_lock` を通常UI・通常API・SenseNova・H3で共有する。
- 外部backendは同じロックを取得してからForgeモデルを退避する。待機中のキャンセルは生成ロックを取得しない。
- 明示的な `/sdapi/v1/unload-checkpoint` も同じロックで待機する。
- SenseNovaはworkerの終了を確認してから返却する。停止失敗時は登録・入力・所有権を保持し、再キャンセルまたは自然終了後に回収する。
- H3はgeneratorの終了だけでは返却しない。監視処理が実ジョブの完了・停止を確認するまで保持する。

## 固定版ComfyUIとの契約

対象リビジョン: `efa6c8f804bff78b46a0fd458ebd2e47bba07a30`。

| 操作 | 固定版の契約 | bridgeの扱い |
| --- | --- | --- |
| `POST /prompt` | クライアント指定の正規UUIDを受理する。同じIDの重複投入は防がない | 送信前にIDを記録し、一度だけ送信する |
| 入力検証のHTTP 400 | キュー投入前の拒否 | 入力と所有権を回収する |
| タイムアウト・応答破損 | 受付の有無を断定できない | 同じIDを照会し、再送しない |
| `GET /api/jobs/{id}` | キューと履歴を照会する。未知IDは404 | 単なる404は終了扱いにしない |
| `POST /api/jobs/{id}/cancel` | 実行中は割込み、待機中は削除。`cancelled: true` は受付であり、停止完了ではない | 終了状態、または受付確認後の404を待つ |

[server.py](https://github.com/Comfy-Org/ComfyUI/blob/efa6c8f804bff78b46a0fd458ebd2e47bba07a30/server.py)、[jobs.py](https://github.com/Comfy-Org/ComfyUI/blob/efa6c8f804bff78b46a0fd458ebd2e47bba07a30/comfy_execution/jobs.py)を確認した。

`cache/minimax-h3/pending/<UUID>.json` にID、送信先URL、送信先PIDと生成時刻、runtime、コピー済み入力一覧を記録する。書込みはflush・fsync後に置換する。キャンセル要求と受付確認も記録する。

再起動後は最初のGPU操作より先に記録を復元し、同じIDの照会が完了するまで排他を保持する。送信先processの終了も、PID再利用を区別して確認する。未確定記録に属する入力は、古くても削除しない。再起動前の完成動画はComfyUI側の出力に残り、既存の履歴から取得できる。

送信直前にアプリが落ち、サーバーにIDが見つからない場合も、未送信とは断定しない。記録を手動削除して排他を解除せず、送信先ComfyUIの停止を確認してから回収する。

## 依存監査

SenseNovaの監査は独立したCI jobで実行する。本体監査の失敗によって省略されない。`tools/requirements-sensenova.txt` をCUDA wheelのindexを含めて解決し、推移的依存も対象にする。ignoreは追加していない。

ローカルの導入済み環境をpip-audit 2.10.1で監査した結果:

| 環境 | パッケージ数 | 検出対象 |
| --- | ---: | --- |
| Forge | 154 | accelerate 1.14.0、diskcache 5.6.3、setuptools 81.0.0 |
| SenseNova worker | 31 | accelerate 1.14.0、transformers 4.57.6、setuptools 81.0.0 |

CUDAローカル版のtorch・torchvisionはPyPI監査サービスで照合できず、監査対象外として報告された。監査結果には同じ脆弱性の重複レコードが含まれる。

[Accelerateの既存検出](https://github.com/advisories/GHSA-4j2p-28q2-5m79)は修正版未掲載。固定SenseNovaの実ロード経路は `init_empty_weights` → `AutoModel.from_config` → 単一safetensorsの `load_file` → ConvRot loaderで、該当するsharded checkpoint loaderの呼出しは確認されなかった。この限定的な経路確認を理由に、依存全体の検出を無効にはしていない。依存バージョンも変更していない。

## 回帰確認

```powershell
venv/Scripts/python.exe tools/run_ci_tests.py --verbosity 1
```

`test_gpu_ownership.py`、`test_sensenova_worker_cleanup.py`、`test_sensenova_u15_bridge.py`、`test_minimax_h3_bridge.py`、`test_minimax_h3_submission.py` が、排他・停止未確認時の保持・キャンセル後の回復・実HTTPの応答消失・別processでの再起動照合を確認する。

実環境の検証結果は `docs/CODEX_HANDOFF.md` に記録する。
