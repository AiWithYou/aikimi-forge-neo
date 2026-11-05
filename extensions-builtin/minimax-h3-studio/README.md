# MiniMax H3 Studio

追加本体から「標準モデル」へ戻すと、Neoが追加した探索先を外して実行環境を再起動します。外部起動のComfyUIは自動停止せず、取り除く起動引数を案内します。生成開始時に本体の内容を固定して確認し、待機中・実行中に同じファイルが上書きされた場合は生成をエラーにします。

画面先頭の「FL2VA本体」「Ref2VA本体」で用途ごとにH3互換本体を選べます。FL2VAはテキスト・キーフレーム・Orbit、Ref2VAは参照素材に使用します。両用途に対応する本体は、両方の欄で同じファイルを指定できます。対応用途は利用者が配布元の説明を確認して指定します。ファイル名やヘッダーから用途・Turboを推測しません。保存した選択はH3 Imageの実験モードとも共有し、画面を再読込すると反映します。

追加本体は`models/MiniMax-H3/checkpoints/`またはその下位フォルダーへ置き、「一覧を更新」で読み直します。テンソル構造と保存形式を確認してH3本体でないものを候補から除外しますが、構造検査は重みの品質や選択用途への適性を保証しません。FP16・BF16・FP32、INT8 Tensorwise、W4A8に対応し、未対応の量子化形式は受け入れません。実行する用途の本体のみを固定・検証し、もう片方の標準モデルが未導入でも使用できます。標準モデルは従来の保存先を使用し、VAE・テキストエンコーダー・LoRAも既存の共有フォルダーを使用します。Turboの少ないStepsは、本体の保存形式と独立した明示設定です。

全50ブロックの主要な線形層の有無と形状、量子化設定のある層の形式・scale・グループ設定を確認します。通常精度の重みにINT8の設定を付けたものや、scaleの欠落、同じ層に旧・新の量子化設定が重複したものは受け付けません。W4A8のscaleやcodebookは、固定された専用環境のローダーに対応するデータ型・形状が必要です。追加のW4A8本体にはComfy Kitchen 0.2.41以降が必要です。古い外部環境では専用環境を準備してから使用してください。

写真一枚から被写体の周りを一周する動画を作る場合は、**360°周回（写真1枚）**を選びます。[Orbit LoRAの導入・推奨設定](../../docs/minimax-h3-orbit-lora.md)を参照してください。追加重みは約148 MiBで、通常のH3モデルを再利用します。W4A8本体にOrbitを適用する場合、対象層のcodebookがあれば有限の16要素で昇順または同値であることを確認します。対象外の層にはこの条件を求めません。

標準W4A8本体とOrbitを併用する場合も、本体を受付時に固定し、送信前と結果保存前に変更がないことを確認します。

管理対象の実行環境では同梱のOrbitノードを導入し、許可したノードだけを有効にします。外部のComfyUIでは同梱ノードを導入し、再起動してください。

**v3.0.0の寸法指定：** 解像度プリセットに加えて幅・高さを直接入力でき、縦横入替も使えます。各辺256〜2048 px・32の倍数です。手入力中は比率プリセットを無効にし、解像度プリセットを選ぶと通常の比率指定へ戻ります。実際の寸法は負荷表示・生成履歴・ComfyUIへの書き出しにも引き継ぎます。大きい寸法や長い動画ではRAM／VRAM負荷が増えるため、既存の生成前チェックが引き続き適用されます。

Aikimi Forge Neo内で映像＋音声、または無音の周回動画を生成する専用GUIです。H3用のComfyUI・Python・モデルは、H3 Studioからまとめて準備できます。完成したMP4と生成条件は`outputs/minimax_h3`へ保存します。

モデルと生成物には[MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)の利用条件が適用されます。EU・英国・韓国・米国は適用地域から除外され、これらの地域でのモデルや生成物の利用・配布・表示などは、この契約では許諾されません。商用利用や配布にも条件があります。[主な条件](P_H3_IMAGE_GUIDE.md#はじめに)とライセンス原文を確認してください。

**画像生成・参照画像編集は`H3 Image`タブを使います。** [P-H3 Image](P_H3_IMAGE_GUIDE.md)と、標準H3の最小5フレームを使う[実験モード](IMAGE_GUIDE.md)を選べます。P-H3には専用のモデルが必要です。

## 初回セットアップ

1. `H3 Studio`の`実行環境とモデル`を開きます。
2. 既存モデルを使う場合だけ、`共有するモデルフォルダー`にComfyUIの`models`フォルダーなどを指定します。
3. `環境とモデルを準備`を押します。完了後は画像・動画の両タブで同じ環境を利用できます。

動画と標準H3画像の接続・生成などは、操作時点の管理対象環境を参照します。画面を開いたままP-H3画面やCLIから共通環境を準備した場合も、次の操作で切り替わります。共有モデルフォルダーと選択したチェックポイントは、それぞれの設定に従います。

Windows・NVIDIA CUDA対応GPU・Git・インターネット接続が必要です。実行環境に15 GiB以上、新規モデルに約59.1 GiBの空き容量を用意してください。ComfyUIを事前に導入する必要はありません。

実行環境は`repositories/minimax-h3-4ac1c173/ComfyUI`、専用Pythonはその親フォルダーの`python`と`.venv`、新規モデルは`models/MiniMax-H3`に保存します。既存のH3環境は残し、新環境の準備と確認が完了すると切り替わります。共有モデルはサイズとSHA-256を検証して参照し、移動・複製・修復しません。既存のComfyUI本体やPythonには依存せず、`forge_neo_model_paths.yaml`もH3の実行環境選択には使用しません。

導入する版は[manifest](../../tools/minimax_h3_runtime_manifest.json)と[依存ライブラリのlock](../../tools/requirements-minimax-h3.lock)で固定しています。標準のFL2VA・Ref2VA、Qwen3-VL encoder、映像・音声VAEの5ファイルを準備します。Turbo・INT8 VAE・Fun ControlNetなどの任意モデルは別途配置してください。途中で失敗した場合は同じ操作で再開できます。

H3は`127.0.0.1:8189`を使用します。起動中のH3に対するセットアップは、生成・保存とキューが空で、このNeoが起動したプロセスの場合だけ停止して進めます。CLIを使う場合はH3を停止してから`python tools/setup_minimax_h3.py`を実行してください。共有には`--share-models <モデルフォルダー>`、保存や通信をしない計画確認には`--dry-run`を指定します。

新しく起動したbackendが起動待ちの途中で失敗した場合は、このNeoが起動したプロセスと確認できる子プロセスを停止します。既に起動していたbackendや外部ランチャーのプロセスは停止しません。子プロセスの所有権や停止を確認できない場合は、管理情報を保持して安全な停止を優先します。

## 使い方

1. `H3 Studio`タブを開くと、画面が先に表示されます。backend・モデル・RAMの確認は、表示を妨げずタブ内で非同期に進みます。
2. `テキスト`、`キーフレーム`、`参照素材`、`360°周回（写真1枚）`から、目的に合うモードを選択してください。
3. 映像のショット、カメラ、台詞、効果音、音楽を一つのプロンプトへまとめてください。
4. Aspect、Quality、Durationを指定します。DurationはH3の`17k+5` frame gridへ自動的に揃う仕組みです。
5. `映像＋音声を生成`、周回モードでは`周回動画を生成（無音）`を押してください。backendが停止中でも設定済みのローカルruntimeを自動起動するため、事前起動は不要です。
6. 完成したMP4は`outputs/minimax_h3`に保存され、`最近の生成`から再表示できます。`設定を復元`で生成条件を戻せます。周回モードの強さも復元しますが、誤った素材の再利用を防ぐため、入力写真や参照素材は再指定してください。

入力中のプロンプトは300msの待ち時間を置いてブラウザー内だけに自動保存し、同じ端末で画面を再読み込みしたときに復元します。外部サービスやbackendへは生成を実行するまで送信しません。空欄に戻すと保存した下書きも削除されます。

通常モードの生成設定は、まず `動作確認`、通常は `標準`、完成版だけ `高品質` を選ぶと迷いません。3つともH3の公式20 Stepsを維持し、解像度だけで速度と品質を切り替えます。周回モードでは768×768・3秒（73 frames）・28 Steps・強さ1.0から始め、「推奨値に戻す」でこの条件へ戻せます。設定カードの「相対負荷」は `標準 / 5秒 / 20 Steps` を1.00倍とした比較値で、所要時間の予測ではありません。

## ComfyUIで続きを編集

H3 Studioの「ComfyUIで続きを編集」から「現在の設定をComfyUIで編集」、または「選択した履歴の生成をComfyUIで編集」を押します。専用ComfyUIが起動し、プロンプト、確定したSeed、モデル、参照素材と制御動画を編集可能なノードとして開きます。**読み込みだけではキューに送信されません。** 変更後の生成はComfyUIで明示的に実行してください。ブラウザーのポップアップが閉じられた場合は、画面に出るリンクから開けます。

現在の設定を書き出す時点でSeedが`-1`なら一度だけ確定し、その値を記録します。生成済み動画は、この版で保存したワークフロー記録のある履歴だけ開けます。旧履歴から設定・素材を推測して再構築しません。実行グラフと素材は専用ComfyUIの`input`へ、動画との対応記録は出力MP4の隣へ保存します。`input/aikimi_h3_workflows`と`input/aikimi_h3_...`の素材を削除すると履歴を開けなくなります。記録に含む生成環境・Seed・参照接続は再編集の出発点で、GPUでビット単位まで同じ動画が出る保証ではありません。素材の複製に必要な空き容量を確保してください。

引き継ぎパックはローカルの固定ファイルを検証して専用ComfyUIへ導入します。外部のComfyUIや別PCへのURL公開には対応しません。読み込みでノード仕様や選択中のモデルが合わない場合は、その内容を表示して停止します。ComfyUIの編集画面を壊さないよう、既存ワークフローを退避できる場合はJSONを保存できます。

## Union 2.0と更新版VAE

Fun ControlNetのUnion 1に加えて、Union 2.0のCanny、Gray、前処理済み動画を選べます。Cannyだけなら、重みが小さいUnion 1をまず選んでください（Union 1 INT8は約2.14 GiB、Union 2.0 INT8は約4.22 GiB）。Union 2.0はGrayなど追加の制御方式が必要な場合の任意モデルです。CannyとGrayは入力動画から生成します。Depth・Pose・HED・MLSD・Scribble・Layoutなどは事前に加工した動画を指定してください。映像を24fpsにそろえ、中央切り抜きと短い動画の末尾補完を適用します。元動画の音声は制御に使いません。Union 1の重みを名前だけ変更してUnion 2.0に使うことはできません。

通常のH3セットアップに対応CoreとVAE修正を含みます。H3の「環境とモデルを準備」、またはセットアップメニューの3で準備した後、必要なUnion 2.0の重みを取得してください。

```powershell
venv\Scripts\python.exe tools\prepare_minimax_h3_union2.py --list --precision int8
venv\Scripts\python.exe tools\prepare_minimax_h3_union2.py --download --precision int8 --repository Kijai/MiniMax-H3-experimental --filename model_patches/minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors
```

`--list`は配布リビジョン、モデル構造、サイズ、ハッシュが確認できる候補を表示します。候補が変わった場合は一覧の値を確認して、対象を明示してください。`--download`は約4.22GiBの重みを取得し、作業コピーを含めてモデルサイズの2倍＋1GiB以上の空き容量が必要です。既に取得済みの対応重みは`--source <ファイル>`で検証して導入できます。通常のH3セットアップや起動時にUnion 2.0の重みを自動ダウンロードしません。

既定のINT8候補は[Kijaiの変換済みUnion 2.0重み](https://huggingface.co/Kijai/MiniMax-H3-experimental/tree/e042fe480f58806578713532b8ae4e3d47d1bd63/model_patches)です。元の[Union 2.0モデル仕様](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0)とH3の利用条件も確認してください。モデル重みはGitリポジトリに含めません。

共通実行環境にはUnion 2.0に対応する固定版ComfyUIが含まれます。旧環境向けの更新スクリプトを実行する必要はありません。

VAEの「標準」は更新したCoreの処理を使います。「更新版 ＋ FP16積算」は比較用の明示的な選択で、使用時は起動引数との一致を確認します。既存の外部Fast VAE Decodeとは別の選択です。少数ステップの単発比較を一般的な画質・速度の優劣として扱わないでください。

共通実行環境はComfy Compilerを無効にして起動し、生成前に起動引数を確認します。動画、画像、ControlNet、NegPiPのいずれでも同じ設定です。

## 長尺生成

「長尺生成」を開いて有効にすると、長さの指定が1区間あたりの秒数になります。区間数と重なりから合計秒数を表示し、共通プロンプトに各行の区間指示を追加して連続生成します。区間指示が空欄なら共通プロンプトだけを使います。

- 標準モデルの設定例：1区間10秒、3区間、重なり39フレーム、20 Steps、切替16。出力は651フレーム・27.125秒です。
- 切替をStepsと同じ値にすると、最後の全体仕上げを行わない順次生成との比較ができます。
- 開始画像は最初の区間だけ、終了画像は各区間の共通ガイドに使います。参照画像は全区間で共有します。
- 有効・無効を切り替えた後、起動済みのbackendには「選択設定で再起動」を適用してください。区間設定は動画のJSONと履歴復元に含まれます。

導入先のComfyUIに、`python tools/install_minimax_h3_hybrid.py --runtime-root <ComfyUIフォルダー>`で[HybridWindows](https://github.com/Jalen-Brunson/ComfyUI-HybridWindows)の固定版を配置します。標準KSampler Advancedの2段階Euler経路を使い、MMH3Toolsは不要です。モデルやPDD LoRAは自動取得せず、標準モデルでは従来の20 Stepsを維持します。NegPiP、CLIPキャッシュ、Sparse Attention、Fun ControlNet、参照動画・音声との併用は生成前に拒否します。

長尺生成の処理時間や画質は、GPU、メモリ、区間数と設定により変わります。

## 高速・省メモリ構成

H3 Studioの専用runtimeは、生成品質とは独立した3つの起動profileを明示選択できます。

- `高速`: DynamicVRAM + Async Offload 2 streams + Pinned Memory
- `省RAM`: DynamicVRAMを使い、node cache・Pinned Memory・Async Offloadを無効化
- `RAM保持（64GB以上）`: DynamicVRAMを無効にし、classic node cacheを使用。Pinned Memory・Async Offloadは無効です。CLIP条件キャッシュは「自動」にしてください
- OS用VRAM 2 GiBを予約。高速・省RAMではDynamicVRAMのheadroomも2 GiBに設定し、RAM保持では追加headroomを設定しません
- USB SSDでは不利になる `--fast-disk` を使用しない
- H3のUNetだけに `ModelAttentionBackend = comfy kitchen attention` を適用
- 必要なH3用custom nodesだけを読み込み、preview・cloud API nodesは無効化

profileを変えただけでは接続中のprocessを書き換えません。キューが空の状態で `選択設定で再起動` を押すと、このForgeセッションが起動したbackendだけを安全に再起動します。外部ランチャーで起動したprocessは自動停止しません。生成前には選択profileと実際の引数を値・競合指定まで検査し、一致しなければ明示的に停止します。

生成の直前には、空き物理RAMに加えてWindowsのOS commit余力も確認し、少ない方が設定ごとの安全目安を下回る場合は送信前に停止します。Runtimeカードの `RAM余力` から現在の制限要因を確認できるため、不足時は他アプリを閉じる、`動作確認`へ下げる、または省RAM profileを明示選択してください。安全目安を自動的に緩めるfallbackは行いません。

ローカルAPIはHTTP接続を再利用し、短い生成の最初の60秒は2秒間隔、長時間生成は5秒間隔で状態を確認します。一時的なstatus/history取得失敗は回数を限定して再試行し、正常に進んでいる長時間ジョブを1回の瞬断だけで停止しません。結果をForge Neoへ保存し終わるまではruntime再起動を拒否します。完成動画とJSONは一時ファイルへ書いてから公開するため、コピー失敗時に途中のMP4を履歴へ残しません。

Comfy Kitchen INT8 attentionはH3ワークフロー内だけに限定しているため、同じComfyUIにある他モデルのattentionは変更しません。Kitchenが利用できない環境では標準attentionへ黙って切り替えず、生成前に更新方法を表示して停止します。生成JSONには、選択したattentionと起動profileに加え、ComfyUI revisionおよびComfyUI/Kitchen versionを記録します。

この構成には、公式coreのH3 video VAE chunked I/OとQ/K/V peak-memory修正も含まれます。起動前にローカルGit revisionが最低commit `62b3c94bd45154f6486c7abf1b9efcacee96ea69` を含むことを検査し、版番号だけではready扱いしません。標準構成は公式のINT8 ConvRot DiTとNVFP4-AWQ text encoderを使います。用途別に互換本体を選ぶこともできます。互換性未検証のthird-party INT4/GGUF、generic FP8、Torch Compileは自動適用しません。

参照素材では `<Picture 1>`、`<Video 1>`、`<Audio 1>` のように画面へ表示されたタグをプロンプトへ記述します。未使用・未知・表記違いのタグは生成前にエラーとして案内します。参照動画は2〜15秒・24fpsへ揃えてください。音声だけの参照は H3 の入力条件を満たさないため、画像または動画も追加してください。

ローカル公開 weight は H3 Base です。Context-IR と 2K Regenerate は MiniMax の外部有料 API 専用で、この extension は呼び出しません。

- [MiniMax H3 official model](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
- [Official ComfyUI guide](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)

## NegPiP（任意）

プロンプト内の負の重みを使用できます。既定はオフで、有効・無効の切替後は「選択設定で再起動」が必要です。[使い方と制約](../../docs/cd-tuner-negpip.md)を参照してください。
