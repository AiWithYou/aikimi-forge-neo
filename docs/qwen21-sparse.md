# Qwen Image 2.1: Jev / block-sparse experiment

PR #5のH3・Anima実験を土台とする追加機能です。Qwen専用ブランチは `experiment/jev-sparse-qwen21`。通常の `neo` は変更しません。

以前の会話添付ZIPに入っていた `qwen_image.py` は、旧Qwenのdual-stream AttentionとForge側のUIを対象にしており、Qwen Image 2.1の専用workerへ接続できませんでした。そのZIPはこの実装には使いません。本実装は別ファイルの `qwen21.py` と専用workerで、Qwen Image 2.1のsingle-stream・prefix KV cache経路に接続します。以前の「103件成功」は旧構造のモックテストであり、2.1への接続確認にはなっていません。

## 起動と比較

通常のQwen Image 2.1セットアップとモデルを準備済みであることが前提です。Neoを終了し、リポジトリ直下で実行します。H3用ComfyUIや重みはQwenの検証には不要です。

```powershell
git fetch origin
git switch --track origin/experiment/jev-sparse-qwen21

# 1. 通常Attentionを計測する比較モード
powershell -NoProfile -ExecutionPolicy Bypass -File .\aikimi-qwen21-sparse-launch.ps1 -Mode dense

# 2. Neoを終了してから、固定75%で起動
powershell -NoProfile -ExecutionPolicy Bypass -File .\aikimi-qwen21-sparse-launch.ps1 -Mode fixed -Keep 75

# 3. Jevを使うときだけSDKを用意
py -3.12 tools/setup_jev_sparse.py --sdk
powershell -NoProfile -ExecutionPolicy Bypass -File .\aikimi-qwen21-sparse-launch.ps1 -Mode jev
```

起動後は既存の「Qwen Image 2.1」タブを使います。txt2img側に追加する機能ではありません。新規生成、参照画像、囲み編集、透過出力、INT8/BF16、CPU退避、モデル再利用は従来の専用workerを使います。設定は起動セッション単位で、変更時はNeoを終了・再起動します。GUI内でのモード切り替えは追加していません。通常の `aikimi-launch.bat` から起動すればこの追加経路は無効です。

選択肢は `off / dense / fixed / rules / jev`。保持率は固定モードのみ `-Keep` で指定し、rules/Jevは50/75/100%です。初期値は最初の1モデル評価をDense、制御更新は4評価ごと、Jevは最大4回・timeout 3秒。`-Warmup`、`-Interval`、`-MaxCalls`、`-Timeout`で変更できます。最初のprefix計算は設定に関係なく従来処理です。

## 計算と保護範囲

Diffusers `6256aa7666cedd47443adc8f82da9a10e110b09c` の `QwenImage21AttnProcessor` が対象です。初回のブロック因果Attentionとprefixキャッシュ作成は、そのまま公式Processorへ委譲します。以後の `cached` モードだけ、生成画像のQと「全prefix＋生成画像」のK/Vを扱います。QKV投影・QK正規化・RoPE・出力投影は既存処理です。

GPU依存の追加コンパイルを避けるため、この初版はPyTorchのblock-gather＋SDPAを使用します。Q/Kをブロック平均で比較して、queryブロック・head別にtargetのkeyブロックをtop-k選択します。選んだK/Vだけを取り出してからAttentionを計算するため、不要なtarget接続の行列演算を省略します。全prefixと選択targetは一つのsoftmaxで正規化し、prefixのpaddingマスクも保ちます。prefixとtargetを別々にsoftmaxして足す実装ではありません。

これはComfy-Kitchenのnative SLAではありません。ブロック選択・gather・複数SDPA呼び出しの負担で遅くなる場合もあります。高速化率・画質維持は未測定です。固定とJevを同一backendで比較してください。`-BlockSize 64/128/256/512`（既定256）、`-MinTokens`（既定1024）も全比較で揃えます。保持率はtarget keyブロックだけの割合で、総計算量の削減率ではありません。端数丸めで全ブロックが残る場合はDenseと記録します。

API失敗・不正回答・非有限統計の後はそのジョブの残りをDenseとし、再問い合わせしません。カーネルの実行エラーを黙って成功扱いにはしません。初回からDense時にも小さなtarget統計を収集するため、可変制御が観測待ちで永久に無効になることを防ぎます。終了・例外・キャンセル時は元のProcessorとhookを復元し、次のジョブに統計や制御を持ち越しません。

独自Processor、Flex Processor、分散推論は初版では拒否します。Qwen workerの内容がレビューした版と異なる場合も、通常生成へ黙って切り替えず、実験用起動のQwen生成をエラーで止めます。WindowsのCRLFは検査時にLFへ正規化します。

## 記録と外部送信

`outputs/qwen-image-2.1/<job>/request.json` に実験設定、`result.json` と `metadata.json` の `sparse_experiment` に実際のbackend、処理回数、API回数、時間とログパスを保存します。同じジョブ内の `jev-sparse/qwen21-*.jsonl` が層別の詳細です。

`attention_calls.sparse_target` が0なら、Sparse経路が動いた結果ではありません。全保持・短い系列・初回計算等の理由別にDense回数を記録します。`model_seconds` は計測用CUDA同期・統計収集・API待ちを含むTransformer評価時間の合計で、初回ロード・テキストエンコード・VAE・PNG保存込みの時間ではありません。元workerの `timings` と区別してください。生成条件・入力ファイルのSHA-256も詳細ログに記録し、条件不一致の結果から速度倍率を自動算出しません。

Jevだけは起動時に外部送信への同意とキー入力が必要です。送信先はTypeSafe API、SDK/モデルはPR #5と同じです。Qwenでは層別の集約ノルム・変化量のみを送信し、プロンプト本文・画像・重みは送りません。キーを引数・設定JSON・ログへ書きません。Dense/固定/rulesではキーを専用workerへ渡さず、APIを呼びません。モデル自体はローカル生成です。

## 検証済み範囲

```powershell
python -m pytest tests/jev_sparse -q
```

Linux / Python 3.13.5 / PyTorch 2.10.0+cpu / Gradio 6.5.1で、既存79件＋Qwen追加46件＝125件成功。QwenはCPUの実テンソルで、独立した密なマスク参照との数値一致、実際にSDPAへ渡すK/V長の短縮、非整列prefix、padding、初回cache一致、DenseからJev判断へ進むこと、終了復元、二連続ジョブ、メタデータ保存を検証しました。

モデルは小型のsingle-streamテストfixture、service・resident worker・Jev API境界はモックです。公式の実モデル、実CUDAカーネル、Windows/PowerShell、実ブラウザー、Jev実通信、リポジトリ全体の回帰テストは未実行です。この125件を実機動作・画質の確認として扱わないでください。検証用Draft PRです。

実装契約の一次資料: https://github.com/huggingface/diffusers/blob/6256aa7666cedd47443adc8f82da9a10e110b09c/src/diffusers/models/transformers/transformer_qwenimage21.py
