# Jev / Sparse Attention

Krea2・H3・Anima・Qwen Image 2.1で、通常処理とSparse処理を切り替えて比較できます。既定はOFF（H3は「Kitchen dense」）です。SparseやJevを選ぶだけで必ず速くなる機能ではありません。[実測結果](jev-sparse-validation.md)も確認してください。Krea2の層別Sparseとタイル配分は[Krea2ガイド](krea2-jev.md)にまとめています。

## APIキーの登録

各モデルの次の場所に「Jev APIキー設定」があります。

- Qwen Image 2.1: **実行環境 → Jev APIキー設定**
- Krea2: **Krea2 · Jev高速化 → Jev APIキー設定**
- Anima: **Anima Self-Attention・実験 → Jev APIキー設定**
- H3: **高速化・任意設定 → Jev APIキー設定**

[TypeSafeのキー管理画面](https://console.typesafe.ai/settings/keys)で取得したキーを入力し、**保存してJevを準備**を押します。必要ならJev専用SDKを `repositories/jev-sdk` へ導入します。Torchや生成モデルはこの操作では変更しません。

キーはWindowsでは `%LOCALAPPDATA%\Aikimi\secrets\jev-api-key.dpapi` に、現在のWindowsユーザーのDPAPIで暗号化して保存します。Linux/macOSでは `$XDG_CONFIG_HOME/aikimi/secrets/jev-api-key`（未設定時は `~/.config/aikimi/secrets/jev-api-key`）に、ディレクトリー700・ファイル600で保存します。こちらは権限制限された平文です。

キーをブラウザーへ読み戻さず、保存後は入力欄を空にします。リポジトリの設定JSON、画像メタデータ、比較ログ、Gitへは書き込みません。不正な値の入力では以前のキーを置き換えません。4モデルで同じ保存済みキーを使用します。環境変数 `TYPESAFE_API_KEY` を設定している場合はそちらが優先です。

## 切り替え

| 対象 | 設定場所 | 選択肢 |
|---|---|---|
| Qwen 2.1 | 生成設定 → Sparse Attention | OFF / Dense計測 / 固定 / 数値ルール / Jev |
| Krea2 | Krea2 · Jev高速化 | OFF / 固定率（1〜100%スライダー） / Jev自動 / 数値ルール / Dense記録、VRAM-Canvasのタイル配分 |
| Anima | Anima Self-Attention・実験 | OFF / Dense計測 / 固定 / 数値ルール / Jev |
| H3 | 高速化・任意設定 → Attention | 通常Dense / 既存Sol・SLA / H3比較Dense / 固定5%・10% / Jev |

QwenとAnimaは次の生成から設定が変わります。Qwenで通常workerとSparse workerを切り替えるとモデルの再読み込みが発生する場合があります。H3は画面の案内に従い「選択設定で再起動」を使います。OFFと固定・数値ルールはJev APIを呼びません。

H3は **Fused Turbo・4 steps・res_multistep** の標準動画経路に限定します。通常H3環境を導入後、`aikimi-jev-setup.bat --h3` を実行して比較用ノードを追加してください。旧版を導入済みの場合も、実行環境を停止して同じコマンドでノードを更新してから利用します。[Fused Turboの配布元](https://huggingface.co/MATLOWAI/minimax-h3-fused-turbo-int8-convrot)の重みも別途必要です。Jev設定は通常の20-stepモデルや長尺・ControlNetへ自動適用しません。

H3の起動プロファイルには「RAM保持」を追加しています。64GB以上の物理RAMが必要です。重みをRAMに保持する従来の管理方式を使い、Pinned Memoryと非同期転送を無効にします。このプロファイルではCLIP条件キャッシュを「自動」にしてください。文章の処理後に大きなテキストモデルを解放し、次の生成用のメモリを確保します。未導入なら `venv\Scripts\python.exe tools\install_minimax_h3_clipcache.py --runtime-root repositories\minimax-h3\ComfyUI` で準備できます。通常の高速・省RAMプロファイルも残しています。

AnimaのSparse処理はcomfy-kitchen 0.2.33のCUDAカーネルを使用します。対応GPUとhead dimension 128が必要です。参照latentを使う編集は対象外です。Qwen 2.1は専用workerのblock-gather + SDPAを使用し、テキスト・参照画像のKVとpaddingを保持します。旧Qwen Image用の実装ではありません。

## 再判定頻度

Krea2・H3・Anima・Qwen Image 2.1の全4モデルで、Jevを選ぶと頻度の設定欄を表示します。

- **初回のみ**：最初の判定をその生成で使い続けます。既定値です。
- **指定間隔**：初回の判定後、スライダーで指定したN stepごとに統計を取り直して再判定します。1〜100を指定でき、2なら2stepごとです。
- **毎step**：初回の判定後、毎stepで再判定します。間隔1と同じ意味です。

H3は1step目を固定5%で計算して統計を集め、2step目に使う保持率から判定します。4step生成では、初回のみで1回、間隔2で2回（2・4step目用）、毎stepで3回（2・3・4step目用）の問い合わせです。各回で49層をまとめて判定し、層0は固定5%を維持します。全50層の計算は実行します。終了後の不要な問い合わせはしません。

Anima・Qwen・Krea2のstepはモデル評価単位です。最初に利用可能な統計が揃ってから初回判定し、以後は指定間隔を使います。サンプラーの内部評価や初回prefillにより、画面のSampling Stepsとモデル評価回数は異なる場合があります。画像モデルは25/50/75/100%（Krea2は1/3/5/10/25/50/100%）から選びます。

頻度を上げると途中の変化に追従できますが、観測の計算・API待ち・利用回数も増えます。最適な頻度を保証するものではありません。OFF・固定・数値ルールでは外部APIを呼ばず、API失敗後はその生成内で問い合わせを止めます。Krea2のタイル配分は別に初回のみ・拡大段階ごとを選べます。[Krea2の詳細](krea2-jev.md#api回数の決め方)

APIでは、H3の高速化設定に `jev_cadence`（`once` / `interval` / `step`）と `jev_interval`、Qwenのリクエストに `sparse_jev_cadence` と `sparse_jev_interval` を指定できます。Animaのalwayson scriptは従来の7項目の後に頻度を追加し、間隔は従来の5番目の引数です。Anima・Qwenで従来の上限回数を使うAPI設定は `decision_cadence="legacy"` を使用し、上限0はどの頻度でもAPI停止として保持します。

## 生成全体の上限と判定再生

再判定頻度とは別に、生成全体のAPI回数と累積待ち時間に上限を設定できます。この2つは **0＝上限なし** です。既存の低水準設定 `max_calls=0`（API停止）とは別の設定です。上限に達すると追加の問い合わせを止め、最後の有効な設定を使います。

Krea2では、層ごとの判定とタイルのstep配分が同じジョブ予算を共有します。再拡大やタイル分割で上限をリセットしません。Anima・Qwen・H3も生成単位で管理します。画面の回数見込みはモデル評価回数からの推定で、実際の回数・待ち時間はJSONLに記録します。

APIでは、H3の高速化設定に `jev_max_calls` と `jev_max_wait_seconds`、Qwenのリクエストに `sparse_jev_max_calls` と `sparse_jev_max_wait_seconds` を指定します。内部のSparse設定では `job_max_calls` と `job_max_wait_seconds` です。問い合わせ用の専用PythonとSDK接続はジョブ中に再利用し、終了・失敗・キャンセルで閉じます。

判定再生は、完了済みJevログの選択を通信なしで適用する比較機能です。入力・生成条件、判定順序、選択肢、予算、タイル配置を照合し、不一致や記録不足はエラーにします。旧ログのうち再生用の照合情報がないものは分析専用です。使い方と比較条件は [Sparse比較スイート](sparse-benchmark-suite.md) を参照してください。

## 外部通信と記録

Jevモードだけが、公式SDK 0.7.0・固定モデル `jev-1.13.0` で `https://api.typesafe.ai` へ問い合わせます。API利用料が発生する場合があります。

- Qwen / Anima: 層別の集約数値。画像・プロンプト・重みは送りません。
- Krea2: 層別の集約数値と、任意のタイル配分では細部量グループの集約数値。画像・プロンプト・重みは送りません。
- H3: 判定直前のステップの層別集約数値。プロンプト・生画像・音声・重みは送りません。

保持率は残す接続の割合です。重みの削減率ではありません。速度優先版ではJevに、弱い・安定した寄与は低い保持率、通常の寄与は中間、極端に強い寄与は高い保持率を選ぶ目的を伝えます。候補と再判定頻度は上記の設定に従います。初回のみでは以後の観測用計算も省きます。

画素一致は品質の合格条件にしていません。Jevの信頼度は選択肢の曖昧さを示す数値で、画質誤差ではないため、有効な回答を低信頼度だけで100%へ置き換えません。原回答・信頼度・適用値を別々に記録します。API失敗や不正回答への退避処理は維持しています。プロンプトへの適合、破綻の有無、見た目の品質と総時間を実画像・動画で比較してください。

Krea2・Anima・H3は `outputs/jev-sparse`、Qwenは各生成ジョブ内の `jev-sparse` にJSONLを保存します。実際のSparse回数、API呼び出し回数・待ち時間、モデル評価時間を記録します。全体時間とモデル評価時間は別の値です。

## 速度比較の再実行

通常の生成を終了し、GPUを空けてから実行します。各スクリプトはウォームアップを除き、順序を反転して2回ずつ測定します。Jevを含む比較は `--allow-cloud` が必要です。

```powershell
models\Qwen-Image-2.1\worker-env\Scripts\python.exe tools\benchmark_qwen21_sparse.py --output outputs\my-qwen-comparison --allow-cloud

# AnimaはローカルAPIを有効にしたNeoを起動してから。
venv\Scripts\python.exe tools\benchmark_anima_sparse.py --port 7861 --output outputs\my-anima-comparison --allow-cloud

# H3は導入済みの標準ComfyUIを自動起動して比較。
venv\Scripts\python.exe tools\benchmark_h3_sparse.py --output outputs\my-h3-comparison --allow-cloud

# H3を15秒の動画で比較する場合。
venv\Scripts\python.exe tools\benchmark_h3_sparse.py --duration 15 --output outputs\my-h3-15sec-comparison --allow-cloud
```

Animaの比較スクリプトには3.8B v1.1のモデル名・付属エンコーダーを指定しています。結果のPNG/動画と `benchmark.json` を同じフォルダーに残します。APIキーはコマンド引数へ渡しません。
