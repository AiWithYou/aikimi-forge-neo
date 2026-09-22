# Krea2 / Qwen 2.1 Sparse 比較スイート

`tools/benchmark_krea2_sparse.py` と `tools/benchmark_qwen21_sparse.py` は、同じ題材・seed・生成設定で OFF、固定保持率、数値ルール、Jev live、記録済み判断の replay を比較します。既定は2 seed・2反復です。固定保持率は Krea2 が 3/10/25%、Qwen が 25/50/75% で、それぞれ独立した比較条件になります。

まず `--dry-run` で件数と除外理由を確認してください。dry-run はモデルを読み込まず、Forge API・Jev に接続しません。既定の全条件は多数の画像を生成するため、短い確認には `--cases`、`--seed`、`--fixed-keeps` で対象を絞ります。出力先は毎回新しいディレクトリにします。

```powershell
venv\Scripts\python.exe tools\benchmark_krea2_sparse.py --output outputs\bench-plan-krea --dry-run
models\Qwen-Image-2.1\worker-env\Scripts\python.exe tools\benchmark_qwen21_sparse.py --output outputs\bench-plan-qwen --dry-run
```

`plan.json` に全実行順、`cases.json` に題材、`benchmark.json` に状態を保存します。集計は表計算ソフトで開ける `summary.csv` にも出力します。`planned_runs` は warmup を含む比較生成件数、`planned_source_generations` は Krea2 4K 用の元画像生成件数です。`planned_live_generations` は Jev を使う生成の数で、API呼び出し回数や料金の見積もりではありません。再判定の頻度や attention / tile の併用で呼び出し回数は変わります。

| ケースID | 確認対象 | Qwen 2.1 | Krea2 のこのAPIベンチ |
| --- | --- | --- | --- |
| `lettering` | 文字・数字の正確さ | 実行 | 実行 |
| `portrait` | 人物・手・髪 | 実行 | 実行 |
| `woven-detail` | 細かい繰り返し模様 | 実行 | 実行 |
| `two-references` | 2枚以上の参照の保持・配置 | 入力指定時に実行 | 複数参照を運ぶAPIアダプター未実装のためskip |
| `transparent-object` | RGBA・細い輪郭 | 実行 | RGBA出力経路がないためskip |

未対応や参照不足は `skipped_cases` に理由を記録します。Krea2の複数参照skipは、このベンチの入力経路の制約です。モデル自体の能力判定ではありません。画像が生成されても、文字・人物・参照の保持を自動で合格判定しません。

## 実行例

外部接続を使わない Qwen 比較:

```powershell
models\Qwen-Image-2.1\worker-env\Scripts\python.exe tools\benchmark_qwen21_sparse.py --output outputs\bench-qwen-local --cases lettering portrait woven-detail --seeds 20260921 20260922 --fixed-keeps 25 50 75 --modes off dense fixed rules
```

参照画像2枚の比較:

```powershell
models\Qwen-Image-2.1\worker-env\Scripts\python.exe tools\benchmark_qwen21_sparse.py --output outputs\bench-qwen-references --cases two-references --reference-images H:\images\first.png H:\images\second.png --seed 20260921 --modes off fixed rules
```

Jev live と同じ判断列を使った replay の比較:

```powershell
models\Qwen-Image-2.1\worker-env\Scripts\python.exe tools\benchmark_qwen21_sparse.py --output outputs\bench-qwen-live --cases lettering --seed 20260921 --modes off fixed rules jev replay --allow-cloud
```

live は既存のJevキー設定が必要です。`--allow-cloud` なしでは実行を拒否します。replay はモデルの画像生成を実行し、判断だけをローカルの記録から読みます。live と replay は同じ Sparse 経路を使い、API待ちを除いたときの変化を比較できます。replay のクラウド呼び出しは0であることを検証します。

後日、同じ条件の保存結果から replay だけを実行する場合:

```powershell
models\Qwen-Image-2.1\worker-env\Scripts\python.exe tools\benchmark_qwen21_sparse.py --output outputs\bench-qwen-replay --cases lettering --seed 20260921 --modes off replay --replay-from outputs\bench-qwen-live\benchmark.json
```

プロンプト、参照画像のSHA-256、seed、解像度、生成設定、モデル・依存関係の識別情報が一致する完了済み live のみ再利用します。固定保持率・再判定間隔なども一致が必要です。新しい replay 契約を持たない旧ログは使用できません。同じ実行に live / replay を並べる場合、`--modes` の最初の並びでは live を replay より先に指定します。2周目は順序を反転し、すでに検証済みの live 記録を利用します。

Krea2 は更新後のForgeを起動し、Krea2チェックポイントを選択します。APIポートの既定は7864です。

```powershell
venv\Scripts\python.exe tools\benchmark_krea2_sparse.py --output outputs\bench-krea-native --cases lettering woven-detail --sizes 1280 2048 --seeds 20260921 20260922 --fixed-keeps 3 10 25 --modes off fixed rules jev replay --allow-cloud
venv\Scripts\python.exe tools\benchmark_krea2_sparse.py --output outputs\bench-krea-4k --stage 4k --cases woven-detail --source H:\images\source.png --seed 20260921 --modes off tiles-rules tiles-jev tiles-replay combined combined-replay --allow-cloud
```

Krea2のreplayモードは `replay`（attention）、`tiles-replay`、`combined-replay`、`fixed-tiles-replay` です。attentionとtileを併用する場合は同じジョブの両ログを一緒に再生します。両CLIの `--job-max-calls` / `--job-max-wait-seconds` はジョブ全体に追加する予算です。0は追加上限なしで、既存の再判定頻度・1回の待ち時間制限が適用されます。総呼び出し数・総待ち時間を制限するときは正の値を指定します。予算を使い切ってfallbackした結果はJevの成功結果として集計しません。

4Kは元画像の縦横比を保ち、長辺を4096pxにします。元画像を指定しなければ各題材・seedにつき1枚作り、以降すべての条件で共有します。この作成時間は比較値に含みません。別実行の `--replay-from` で4Kを再現するときは、元の `source_path` の画像を `--source` またはケースmanifestの `source` に指定して元画像自体を固定してください。

`--keep 25` のような従来指定は、`--fixed-keeps` を併用しない場合、固定保持率1種類の指定として維持しています。`--prompt` は任意の文章による1ケースです。

Qwenの `--query-batch-blocks`（既定4）と `--max-batch-workspace-mb`（既定256）は、Sparse計算のまとめ方とその作業メモリ上限です。値は生成設定とreplayの照合対象に保存します。attentionだけの小規模な計測は `tools/benchmark_qwen21_attention.py` を使用します。

## 任意の題材

`--case-manifest path\cases.json` で組み込み題材を置き換えます。ファイルパスはmanifestからの相対位置でも指定できます。IDは英数字・ハイフン・アンダースコアで一意にします。

```json
{
  "schema": 1,
  "cases": [
    {"id": "jp-sign", "category": "text", "prompt": "A shop sign that reads 喫茶店 and OPEN 09:30."},
    {"id": "pair", "category": "multi_reference", "prompt": "Place image 1 on the left and image 2 on the right.", "input_images": ["first.png", "second.png"]},
    {"id": "glass", "category": "rgba", "prompt": "One glass bottle on a transparent background.", "transparent": true}
  ]
}
```

4Kのケース別元画像は `source` で指定します。`input_images` はQwen参照入力です。指定した未存在ファイルはskip理由として残します。

## 計測と結果の読み方

- 各題材・seed・解像度の最初にOFF画像を最後まで生成し、warmupとして集計から除外します。奇数周は指定順、偶数周は逆順です。4Kも完全なOFFアップスケールをwarmupに使います。
- `wall_seconds` は画像が完成し、保存・読み込み検証が終わるまでの時間です。Jev待ち、APIの転送待ち、デコード・保存を含みます。ログの集計時間は含みません。
- `summary.groups` は同じ題材・seed・解像度ごとの中央値・最小・最大・標本数・OFF比です。replayには同じ実行のlive中央値との比も保存します。`paired_speedups` は対応する比較条件ごとの倍率の中央値です。
- `artifact.pixels` にRGBAへデコードした画素のSHA-256と幅・高さを保存し、`replay_pixel_identical` はその一致を比較します。PNGの生成情報や圧縮方法が違っても、RGBA画素と寸法が同じなら一致です。完全に透明な画素のRGBも比較に含みます。旧記録に画素hashがない場合は `null`（未判定）です。`artifact.sha256` は来歴確認用のPNGファイル全体のSHA-256として維持し、メタデータを含むファイルの一致は `replay_file_identical` に分けます。`replay_source` に再生元を記録します。画素が一致しないことだけで品質の劣化とは判断しません。
- Qwenの `timing_breakdown` は総所要、CPU側の評価・forward・制御時間、API待ち、CUDA event時間を分けます。CUDA eventはストリーム上の経過時間で、CPUからの投入待ちやoffloadの影響も含み得ます。純粋なGPUカーネル時間として扱いません。
- 画像ファイル欠落、要求解像度やRGBAの不一致、Sparse呼び出し0、Jev失敗後のfallback、replayの不一致・未消費は失敗です。失敗した実行を保存して停止し、速度の中央値へ混ぜません。固定100%とdenseはDenseの対照条件として扱います。

`benchmark.json` と各画像・生成設定・Jevログを一緒に保管し、画像の文字、人物、模様、参照の保持、アルファ境界を確認してください。速度とファイル一致の記録だけから見た目の同等性は結論づけません。
