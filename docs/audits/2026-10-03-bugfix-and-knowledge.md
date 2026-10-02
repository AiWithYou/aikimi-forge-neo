# Forge Neoのバグ修正と更新情報

2026年10月3日に、既存機能の不具合をCPU・オフラインの再現試験で確認し、v3.5.1として修正しました。新機能案はこの文書で提示するだけで、実装・モデル追加・依存関係の更新は行っていません。開始時のHEADは`4536cb1a`です。

## 修正した挙動

| 対象 | 再現した問題 | 修正後 |
|---|---|---|
| UIの既定値 | 複数選択を空にしても保存されない。数値の選択肢がindexと混同され、表示名と内部値が違う選択肢も復元できない。 | 空選択、数値、内部値を保存・復元。index形式も複数選択を扱う。 |
| 設定変更のレビュー | プロンプトや設定名のHTMLが表の一部として解釈される。 | 値を文字として表示し、表の閉じタグも補完。 |
| 設定変更の失敗 | 保存ファイルにない設定の変更callbackが失敗すると、既定値が`None`へ変わる。 | 既存値と設定の有無を復元。既定値と同じ変更でcallbackを再実行しない。 |
| Qwen Outpaint | 小さい元画像に対する境界幅を受理し、生成完了後の合成だけで失敗する。 | GPU取得前に同じ制約で検証し、UIの上限・値も元画像に合わせる。 |
| Qwen Outpaintの生成可否 | 履歴から消えたjob IDを参照すると、元画像や余白を変更しても生成可否の更新が例外になる。 | 失効した履歴の参照を処理して、現在の入力を検証する。 |
| Qwen Outpaintの透過境界 | alphaが1の画素が、8bitの乗算済み色変換で`(128,67,39,1)`から`(255,0,0,1)`へ変色する。 | 通常のマスク編集と同じ整数精度のRGBA合成を使う。 |
| Qwen LoRA設定 | 設定欄を再展開すると、ONにした「異なる量子化で試す」がOFFになる。 | 再展開では指定を保持し、選択変更時のリセットは維持。 |
| 常駐ワーカー | 起動・ログ準備・command保存の失敗で、子プロセスや制御ファイルが残る。 | 起動全体に終了確認と回収を接続。 |
| Windowsの常駐再利用 | 応答・command JSONの一時共有ロックで、正常なworkerのjobが失敗する。 | 最大1秒の再試行。worker終了と恒久エラーは報告。 |
| SenseNovaとYuE2 | 起動と終了の両方が失敗すると、実行中のプロセスを追跡せずGPUを返す。 | プロセスと所有権を終了確認まで保持。 |
| SenseNovaの完了・取消 | 結果検証中に受理した取消でも完了扱いになり、完了確定後の取消は待機モデルを停止する。 | 完了をlock内で確定し、確定前の取消を反映。確定後は取消を拒否。 |
| Qwen・Nanosaurの画像検証 | PNGの構造が正常でも画素が足りない画像を完成扱いにする。 | 通常・preserved画像とも画素をdecodeしてから公開。 |
| Refiner | 終了時にUNetが既に解放されていると、未定義の`idx`で復元が落ちる。 | まだ存在するUNetだけを解放し、元の重みの復元を継続。 |
| Llama系Embedding | 実装が対応していない`out_dtype`引数で、通常・Forge・INT8のEmbeddingが`TypeError`になる。 | Embeddingの出力後に指定dtypeへ変換。量子化した重みは保持。 |
| W4A8の保存 | `group_size`と`convrot_groupsize`を失い、再読込時に既定値へ戻る。不正なpacked shapeも読み込める。 | 保存条件を保持し、不正な重みの形状は読込時に拒否。 |
| checkpointの検査 | `.pt`内の文字列・数値metadataでdtype判定が落ちる。別componentのNF4判定がprefix指定を無視する。 | Tensorだけを検査し、対象prefixだけで判定。浮動小数点の重みがない場合は`None`を返す。 |
| H3履歴 | 出力外を指すsymlinkやMP4名のディレクトリを履歴へ採用し、cacheのsymlinkも信頼する。 | 解決後の出力範囲と通常ファイルを確認。cacheの外部実体は変更せず再作成。 |
| YuE2履歴と復元 | 出力外のmetadataを読む。壊れたsymlinkで全履歴が停止する。 | metadata・projectにも保存範囲を適用し、壊れた項目を除外。 |
| H3のSeedとSteps | ブラウザが大きいSeedを丸め、小数・booleanを整数として受理する。 | Seedを文字列として送受信し、UI・bridge・履歴で整数を検証。 |
| Mingの透明PNG | LA、パレット、RGB＋tRNSの透明PNGを不透明と記録する。 | それぞれのalphaを検出。保存PNGのbytesは維持。 |
| API・画像保存 | WebPへ変換すると透明部分を失う。LA・パレット画像のJPEG、大文字の拡張子、Path形式のEXIF保存も例外になる。 | WebPのalphaを保持。JPEGのmode、拡張子、ファイルパスを正規化。 |
| 画像の読み込み | 追加パラメータ付きdata URIで失敗し、途中切れや型が壊れたEXIFが有効なPNG生成情報の読込を阻害する。 | URIの画像部分を区切りで抽出し、任意metadataの破損を回収。 |
| EXIFの生成情報 | 壊れたIFD pointerでPNG情報の読込が例外になる。Orientation補正した正常なJPEGではUserCommentが整数tupleになり、日本語の生成情報を失う。 | 破損のTypeErrorを回収し、正当なbyte tupleを元のdecoderへ渡す。 |
| プレビュー・モデル情報の保存 | 隣接フォルダや外向きリンクを保存先として受理する。PNG・JSON名のリンクでモデル本体も上書きできる。ファイル名のHTML entityは記号へ変わる。 | 実体の保存範囲・拡張子を確認して読み書きし、ファイル名は文字として表示。明示登録した外部モデルのJSON保存は維持。 |
| 一時画像の削除 | 管理対象名のsymlinkを削除する際、リンク先の通常画像を削除する。 | 保存範囲を確認した後、リンク自身を削除。 |
| ComfyUIのモデルパス | 標準の複数行指定を1つのパスと解釈し、相対base_pathをNeoの作業位置から探す。 | 行ごとに読み、YAMLの所在から相対パスを解決。 |
| Attentionの代替処理 | xformers失敗時にheadとsequenceの軸を取り違え、指定した出力形式も失う。 | 軸をPyTorch形式へ変換し、trace経路も同じ出力形式を保持。 |
| Attentionのmask | 真偽値のmaskを0/1へ変え、遮断すべき箇所を通す。 | 0/−∞の加算maskへ変換し、成功・代替処理の両経路で意味を保持。 |
| Basic Attentionのmask | batch・head・queryの軸を混同し、broadcast maskで例外になる。全遮断行は一様な出力やNaNになる。 | PyTorchと同じ軸へ展開し、全遮断行は0で返す。既存のbatch別padding maskも保持。 |
| Attentionのバッチ分割 | 分割が発生した場合だけ、4次元の出力指定を無視して3次元で返す。 | 分割の有無によらず同じ形状・数値で返す。 |
| VAEの省メモリ処理 | 長さが分割数で割り切れないと、OOM後も元の大きさで再試行して失敗する。 | 端数を含めて縮小し、最小1要素まで分割できる。 |
| アップスケールのcache | 同じ画素数で縦横が違う画像や、パレット色だけ違う画像を取り違える。 | 寸法・mode・表示色を含めて画像を識別。 |
| 小画像のタイル処理 | 片辺がoverlap以下だとタイルが0枚になり、CPUアップスケールやSD Upscaleで例外になる。タイルより少し小さい画像の端も暗くなる。 | 各軸を最小1枚にし、先頭の負offsetはmaskを掛けず元offsetへ貼る。通常サイズとモデルへ渡す画像の配置を保持。 |
| Extras API | 中断・skipの空結果で例外。入力requestも変更し、無限大の拡大率を受理する。 | 空結果をnullable応答へ変換し、元requestを保持。非有限の拡大率を拒否。 |
| 生成中断 | Zero Terminal SNRの一時sigmasが残り、Refiner切替後は別predictorへ復元する。 | 成功・失敗・中断のいずれも元predictorへ復元。 |
| img2imgのbatch復元 | PNG解析の失敗・中断後に、全体のstyle設定がIgnoreのままになる。 | 解析の成否にかかわらず元設定へ復帰。 |
| Soft Inpaintingの連続処理 | 空maskで通常img2imgへ戻った次jobへ、前jobの画像とmaskが混入する。無効化や後のbatch失敗でも状態が残る。 | post_sampleごとにoverlay状態を初期化し、成功した現在のbatchだけで再構成。 |
| モデルの名前 | 設定したmodelsに隣接するmodels-extraを内側と誤判定し、Windowsの大文字・小文字でも失敗する。 | 相対パスで包含を判定し、別driveは次のrootへ進む。 |
| 追加VAEの生成情報 | 成功した追加VAEのpathを記録せず、name・hashが常に空になる。 | 読込成功後にidentityを記録し、元VAE復帰時に解除。 |
| モデル管理の寿命 | cloneのfinalizerが管理オブジェクトを強参照し、再利用のたびに不要な管理情報が残る。 | 初期化順序と弱参照callbackを修正し、多段の親への復帰も保持。 |
| VAEのclone | 拡大率・軸の式・出力channel等を失い、タイル処理がAttributeErrorになる。 | image・videoのタイル処理に必要な設定をコピー。 |
| VAEのOOM再試行 | 途中batchの入力・出力と全batchの出力bufferを保持したままタイル処理へ進む。 | cache回収の前に一時Tensorを解放。最初のload失敗でも再試行できる。 |
| Setupの管理先 | Ming・NanosaurとH3の一時ファイルが、外向きsymlinkを通じNeo外へ書き込む。 | 既存の管理対象範囲を実際の書込先・実行fileへ適用。 |
| Setupの修復失敗 | Ming・Nanosaurの古い完了記録が残り、未完成の環境を準備済みと判定する。 | 実更新前に完了記録を無効化し、成功時だけ再保存。 |
| DoRA | 更新前のweightのnormを使い、保存された大きさと異なる重みへ変換する。 | 更新後のweightで正規化。 |
| DoRAの入力軸 | Conv2dの出力channelが1のとき、保存された入力軸のscaleを出力軸と取り違える。 | 形状と要素数で軸を判定。 |
| LoKrの差分 | CP形式の再構成や転置保存の非連続TensorでKronecker積が失敗し、差分が適用されない。 | 積の前に両因子を連続化。 |
| LoRA metadataのcache | 同サイズ・同mtimeのatomic置換後も古いaliasを返す。 | ctime・inode・deviceも照合して置換を検出。 |
| Windowsのモデルcache | 同じサイズ・mtimeを保った直接書換えで、SHAとLoRA aliasが古いままになる。Windowsのctimeはこの変更を表さない。 | 通常ファイルはNTFS ChangeTimeも照合。取得できなければcacheを再利用・保存しない。ディレクトリと非Windowsの経路は維持。 |
| 拡張機能のGit情報 | worktreeのHEADを変更しても`.git`参照fileが変わらず、再起動後も古いcommit・branchを返す。branchなしHEADと同時初回読込も失敗する。 | `.git`が通常fileなら永続cacheを使わず実Git情報を取得。branchなしのcommitを保持し、lock後の完了情報を再利用。通常repositoryのcacheは維持。 |
| 生成情報・Prompt | Negative prompt本文中の同名文字列を削除し、不正な強調数値で解析が落ちる。 | 最初のprefixだけ除き、不正な数値は文字列として扱う。 |
| Seedの復元 | Seed resizeのAPI幅・高さが逆。Variation seed未使用時の再利用も−1になる。 | 幅・高さを正しく対応させ、未使用時は通常Seedへ復帰。 |
| Seed resizeのnoise | 奇数pixelの中央cropでslice長が合わず、動画の時間軸も落とす。 | 空間の末尾2軸だけをcropし、channel・時間軸と乱数の消費順序を保持。 |
| UMT5のtoken化 | Tokenizerへ渡す前に各区間の末尾1文字を切り、EOSも重複する。 | 元の文字列をtoken化した後にEOSを除去。 |
| T5・UMT5のbatch | 異なる長さのPromptでstackが失敗し、BREAKがbatch数を増やす。UMT5の補完token名も未定義。 | 長さと区間数を揃え、区間はsequenceへ連結。Prompt単位のbatchと重複cacheを保持。 |
| 強調の正規化 | 平均0のEmbeddingがNaNになる。fp16の小さい有限重みでも比がinf、乗算が全0になる。 | 平均0を回収。fp16・bf16は乗算から正規化までfp32で計算し、出力dtypeを戻す。 |
| ControlNet・T2I | CFG組数変更で参照順序やbatchがずれ、copyの明示deviceも失う。 | 元hintをcacheに保持し、現在のbatchへ展開。deviceもコピー。 |
| XL Adapterのcache | 8pixel刻みで指定した画像の寸法と内部の16pixel丸めを混同し、同じ入力でも毎step特徴量を計算する。 | resizeとcache比較に同じ丸め後の寸法を使う。 |
| ControlNetの終了処理 | T2Iの特徴Tensorが残り、未準備ControlLoraのcleanupや単独importも失敗する。 | 特徴を解放し、未準備時も回収。循環importを使用箇所へ移動。 |
| IPAdapter | cloneへ条件を追加すると元モデルも変わり、画像前処理が呼出元の乱数状態も変更する。 | 条件listをコピーして追加し、前処理後にCPU乱数状態を復元。 |
| 演算contextの終了 | 失敗・中断時にoffload streamの待合せを行わず、入れ子のdtype・device・cast設定も外側へ漏れる。 | 待合せと設定復元をfinallyで行い、setup失敗でも復帰。 |
| 量子化の再読込 | contextが入力のTE設定を削除し、2回目はfull precision設定を失う。 | 入力dictのコピーから演算設定を取得。 |
| 量子化の全精度演算 | 全精度指定でもQuantizedTensorのdispatchが入力を再量子化する。 | 全精度経路は重みを展開して計算し、通常の量子化経路は保持。 |
| OFT・BOFTの強度 | 回転の補間と差分への加算で強度を2回掛け、0.5が0.25相当になる。負値も符号が崩れる。 | DoRA・rescaleなしの既存形式では回転補間だけに強度を適用。BOFTのrescale指定時は強度0で元の重みを保持。 |
| Root層のLoRA | 最上位のLinearのweight名へ不要な`.`を付け、loadや省メモリpatchがAttributeErrorになる。 | PyTorchのstate dictと同じkey生成に統一。 |
| Root層の量子化LoRA | 最上位層だけ量子化の変換・再保存hookを使わず、差分が消える。 | 子層と同じhookを取得し、INT8の保存・復元まで保持。 |
| モデルの部分アンロード | offline・onlineのLoRAを併用すると、部分解放後に適用順序が逆転し、DoRAや倍率の結果が変わる。 | 通常のloadと同じoffline→onlineの順序を保つ。 |
| ControlNetの再使用 | samplingの準備がUNetの追加モデルlistを変更し、毎回参照を増やす。ControlNetを外しても前のモデルをloadする。 | 現在の準備用listをコピーして組み立てる。 |
| ライブプレビュー | 毎ステップ指定で同じstepを繰返しdecodeし、5step間隔も5・9・13へずれる。 | 完了stepで記録し、decode中に進んでも元のstepと対応。 |
| 進捗API | 実行中job IDが応答から消える。終了との競合で0除算し、完了時のETAも負になる。 | 動的IDと既存応答schemaを対応させ、分母等を一度取得。ETA計算前に完了値を補正。 |
| AND Promptのbatch | batch内のPrompt数が違うと最後の項を捨て、行ごとに違う強度も最後の行へ揃える。CFG 1では強度合計が1以外でもNegativeを省略する。 | 不足する項を強度0で補完し、行別の強度を保持。Negative省略は全行の合計が1の場合に限定。 |
| AND Promptの相殺 | `a:1 AND b:-1`がNaNになる。`0.1+0.2-0.3`も浮動小数点の丸めで異常値・Infになる。 | 相殺する行の局所CFG差分を保持。通常の平均条件は維持し、単独の小さい強度も変更しない。 |
| Referenceの条件記録 | CFG処理の変更後、CFG 1の条件記録にもNegativeを連結し、Attention・AdaINのbatch数が合わなくなる。 | 記録専用の処理にPositiveだけを渡す従来の契約を明示。通常のsampling条件は保持。 |
| Rescale CFG | 動画の5次元latentでstdの集計軸が不足し、一定値の予測では0除算してNaNになる。 | batch以外の全軸を集計し、guided stdが0のとき通常CFGを保持。画像の計算と有効なscaleは維持。 |
| 動画のSDE sampler | UIのフレーム数をlatentのbatch数と混同し、Brownian noiseのseed数が合わず例外になる。 | latentの実batch数でseedを選び、通常画像の順序を保持。 |
| DDIM・PLMSのstrength 0 | 固定stepsでは0除算し、通常stepsでもnoiseを混ぜて元画像を変える。 | strength 0は入力latentをそのまま返し、モデル準備と乱数消費を行わない。 |
| Canvasの画像読込 | 元画像や描画を素早く切り替えると、遅れてdecodeした古い画像が新しい画像を上書きする。 | 画像ごとの更新番号を照合し、古い完了を無効にする。同じ元画像への描画は保持。 |
| Gradioのタブ更新 | 未選択タブの表示更新が初期値を使い続け、選択済みのタブも一覧の表示・名称・操作可否へ反映されない。 | 明示した表示状態を保持し、変わった登録情報だけ一覧へ同期。再表示と動的追加・削除も反映。 |
| ローカルモデルの検査 | safetensorsのheaderだけ正常な途中取得ファイルを有効なモデルとして受理する。 | 開いたファイルの実サイズとTensorのdata_offsetsを照合して早期に拒否。 |
| Qwen共通部品の検査 | 外部TE・VAEはファイルの存在だけで受理し、途中取得の検査を通らない。 | 各weight fileに同じheader・データ範囲の検証を適用。検証失敗時は選択を保存しない。 |
| Qwen配布物の作成 | 完了記録を信頼し、同サイズ破損・コピー中の改変でも古いSHAをmanifestへ保存する。 | 保存先の実SHA・サイズを完了記録と照合。破損を完成扱いにしない。 |
| テストの隔離 | `sys.modules`全体の復元が初回importしたNumPy・Torch・Gradioも削除し、単独実行で再importとWindows異常終了を起こす。 | 差し替えた名前だけを復元。実メソッドの検証を保ち、読込順序への依存を減らす。 |

## 検証記録

検証環境はWindows、Python 3.13.14、PyTorch 2.13.0+cu130です。共通runnerのCPU・download禁止・一時記録先で実行しています。全体の再実行は`.\venv\Scripts\python.exe .\tools\run_ci_tests.py --verbosity 1`、個別回帰は`--module tools.tests.<module名>`で指定できます。

v3.5.1の最終回帰は2,211件を313.414秒で実行し、失敗なし・52件skip・expected failureなしでした。追加依頼で仕上げたGit拡張の情報更新、小画像grid、Soft Inpaintingの連続状態、最適化起動に依存しない保存先検証も含んでいます。記録は`tmp/bugfix-release-v3.5.1-2026-10-03.log`です。

初回の全体unittestは1,888件を実行し、52件skip・1件expected failure・1件失敗でした。失敗はFun Accの対応精度拡大後も「INT8へ変更して選択不可」を期待していたChromiumテストで、現行の「選択したモデルを維持して4 steps固定」へ修正し、実ブラウザで成功を確認しました。

1回目の修正後の全体unittestは1,953件を実行し、失敗なしでした（52件skip、1件expected failure）。追加修正後の2回目は2,032件を実行し、常駐ワーカーの再利用で1件失敗しました。Windowsで存在する応答ファイルの読み取りが共有ロックにより拒否される同じ境界を再現し、9件の回帰と失敗した再利用テストは修正後に成功しました。3回目は2,083件を実行し、失敗なしでした（52件skip、1件expected failure）。4回目は2,167件を291.251秒、5回目は2,194件を296.195秒、6回目は2,204件を307.338秒で実行し、いずれも失敗なし・52件skip・expected failureなしです。統合試験時のロックを保持した主体は未特定です。

Jev、YuE2、H3の追加pytestは378件成功、5件skip、19 subtests成功です。H3・OutpaintのNodeテストも20件成功しました。GPUモデル実生成と本サーバーの再起動は実施していません。

YuE2とH3の変更後の追加pytestは133件成功、5件skip、19 subtests成功です。独立レビューではAttentionのGQA・broadcast mask・指定scale、3段cloneのGC、実CPUモデルの切替、複数tileのVAE処理を検証しました。LoCon・LoHa・LoKrとDoRAの追加境界は21件成功。演算contextの追加回帰と関連Attentionは15件成功しました。最終変更までのRuff差分検査は157 Pythonファイル（新規テスト67、新規共通module 1）を開始時のHEADと比較し、新たな警告はありませんでした。

文字処理は、downloadを行わない実際の小型T5 encoderとメモリ内tokenizerを使い、T5・UMT5のBREAK・空Prompt・EOS mask、fp32・fp16・bf16、cacheの再計算を検証しました。新規と既存の20件が成功。プレビューの追加回帰と関連cacheは23件、進捗APIと関連経路は40件成功しました。CPU指定を消していた4つの既存fixtureも修正し、それぞれ単独で4・9・9・4件成功を確認しています。

演算contextの独立レビューは通常・GGUF層、ControlLora、小型CLIP Vision、実INT8、Online LoRA、転送後の失敗を検証。全精度指定の修正を含めた関連22件が成功し、GPU用2件はskipです。Root層のpatch・省メモリ移行とControlNet準備の追加検証は18件成功。SenseNova取消とStudio画素検証の新規7件・既存関連2件も成功しました。OFT・BOFTは2D・4D、強度0・0.5・1・−0.5、constraintとoffsetを検証。DoRA、OFTのrescale、BOFTのrescale指定時の非0強度は、保存形式とForgeの強度の対応が未確定で変更していません。

Basic Attentionの追加回帰と関連15件、Root量子化の新規・既存6件、AND Promptと関連sampling 66件が成功しました。ANDの検証は実Prompt parser・CFGDenoiser・bundled Eulerを接続しています。Canvasの読込順序7件と、実Chromiumで2回生成・編集・再送信する既存テストも成功。VAEのOOM回復とcloneは5件成功し、CPUでTensorの解放と実タイル処理の全batch出力を確認しました。GPUのVRAM削減量は未測定です。ローカルモデルの欠落データ・不正offset・有効な空Tensor等の回帰と既存選択は5件成功しました。

XL Adapter・IPAdapterの独立レビューは実小型Adapterとattention、公開apply、UnetPatcherのclone、条件のweakref解放、transform例外時のRNG復元を接続し、関連8件が成功。Qwen配布のintegrityは新規・既存6件、外部TE・VAE検査とsource変更・検査失敗時の選択保持は新規3件が成功しました。配布のstage処理には、重みの合計サイズを追加で1回読み出す費用があります。hardlinkは0回から1回、copyはコピーを含め1回から2回の読み出しになります。

ANDの相殺は実Euler、4D・5D、地域mask・area、実ContextHandlerの通常窓・分割窓、custom CFG、post CFGを含む関連25件が成功。部分アンロードの独立レビューは実CPU loaderで親子孫・通常/INT8・weight/bias・FP32/BF16・weakref解放を接続し、関連10件が成功しました。

画像metadataの追加回帰は実PNGとJPEGをAPI・Orientation補正へ通し、関連5件が成功。Referenceは実条件記録・Attention・AdaINを接続し、関連28件が成功しました。Rescale CFGは実Script登録・clone・samplingへ4Dと5Dを渡し、一定の予測と通常の数値を含む関連32件が成功。Seed resizeは実CPU・Philox乱数、batch・variation・eta deltaを使う新規4件が成功しました。SDEは実BrownianTreeとSDE/2M/3M samplerを使い、DDIM/PLMSのstrength 0と正のstrengthを含む新規5件が成功しました。

プレビュー・metadata保存は一時ファイル、実FastAPI配信・Pillow保存とHTML解析を使う新規8件と既存関連19件が成功。隣接フォルダ、外向きリンク、モデル重みを指すPNG/JSONリンクを拒否し、通常PNGの生成情報と日本語JSON・明示した外部モデルの保存を保持します。独立した追加3条件では、モデルとdirectoryのsymlink、内部JSON aliasの保存とcache更新も成功しました。5回目後にはPythonの`-O`でassert検証が消える7条件の失敗も再現し、明示した例外へ変更。同じ関連27件は通常起動・`-O`の両方で成功しました。

Windows cacheは実NTFSの同サイズ書換え、mtime復元、永続cache再読込、API失敗時の非cache化、読込中変更、旧cache移行を含む新規10件と既存15件が成功しました。実share=0の排他handleでstatだけ成功する場合も、追加openのOSErrorはcache無効化へ回収し、no_hashingの一覧取得を継続します。元のstat・実内容の読込エラーは維持します。通常ファイルのcache確認に追加のopen・file情報取得がありますが、payloadは読みません。ChangeTime取得失敗中は同サイズ・時刻保持の読込中変更を検出できませんが、その結果をcacheへ保存しません。

小画像のgridは実CPU nearest 2xとSD Upscaleの分割・生成呼出・再合成を接続した新規6件と既存20件が成功。tile直前寸法では修正前に約9.55％の画素が変わり、15×15画像・tile16・overlap4では左上の値が128から8へ落ちました。修正後はtiny・細長い画像、250/255/256/257/513と端数、overlap 0で元の画素と拡大結果を保持。通常gridのRGBA/P/LAとrow設定も確認しています。GPU upscalerの品質評価は実施していません。

Soft Inpaintingは実processing.init、CPUで計算するadaptive histogram、Pillowのapply_overlayを使う新規4件が成功しました。修正前は赤画像の処理後に青画像＋空maskを渡すと、次の出力pixelが青ではなく赤になりました。現在は空maskの通常img2imgへの切替、無効化、batch間の例外で前のoverlayを再利用せず、正常な次処理はその画像から再構成します。

停止時に未修正だったGit worktree拡張の情報更新は、追加依頼を受けて修正しました。一時repositoryの実commit・branch変更、永続cache再openと新instance、branchなしHEAD、同時初回読込、通常repositoryのcache再利用、既に初期化したinstance・builtin・存在しないrepositoryを使う新規7件が成功。修正前は3件失敗・同時読込1件例外でした。既存のGit操作とnative cacheを合わせた関連23件も成功しています。旧設定keyへの更新候補は、正当なcallerでは明示的に拒否されることを4件で確認し、仕様を変えていません。

Gradioは隔離したlocalhost fixtureで15件成功。49タブの起動・切替・keyboard・再読込、未選択/選択済みの非表示・再表示、名称と操作可否、gr.renderでの追加・削除を実Chromiumで確認しました。以前のexpected failureを通常の回帰試験へ変更。元の互換パッチを使った比較でも非表示が反映されないことを確認し、修正前後の画面を撮影して読みました。修正後にJavaScriptの例外・console errorはありません。site-packagesを書き換えず、固定版assetのSHAと置換箇所を検証して配信します。本サーバーでの見た目・GPU生成は別途未実施です。

RuffはCI対象の174ファイルと新規の68ファイルでcheck・formatとも成功しました。CI対象に残っていたMing準備判定の引数改行だけを整え、前後の構文木が同じことも確認しています。それ以外の既存ファイル全体に残る警告・整形不一致はHEADと比較し、対象外の一括整形は避けています。日本語style guardはCONTRIBUTINGが参照するローカル位置に存在せず、文章を読み直して確認しています。

公開前のGitleaks 8.30.0（CI固定版）は公式SHA-256を照合し、検出用の合成データを検出することを確認しました。リリース対象の差分と全8,236コミットを検査し、既存の明示した誤検出例外以外に検出はありませんでした。利用者用の未追跡資料、モデル、生成物、cache、試験logはコミット対象に含めていません。

初期のQwen単体テストは、利用者用の`models/.local-assets/library.json`へモデル選択を保存した可能性があります。確認時は本体の外部パスとLoRAが空、精度がBF16でした。変更前の値とバックアップは確認できていません。元の選択を推測で書き戻さず、以後の試験は共通runnerに加えて試験fixture内でも保存先を隔離しました。モデルの重みと生成済み画像は変更していません。

## 確認した更新情報

確認日はいずれも2026年10月3日です。公式資料と取得したソースを読み、現在の実装との関係を調べました。

- Forge Neoの`neo`を`97b26fb4`まで取得。Refinerの終了処理は[10月2日の上流修正](https://github.com/Haoming02/sd-webui-forge-classic/commit/4f0ee8ff1b63e7a4724f5df92e2746e1fe59b823)と同じ原因を確認して反映しました。W4A8の条件保持も[10月1日の上流修正](https://github.com/Haoming02/sd-webui-forge-classic/commit/89903ecbe26cae4791f742f1881063602a4f957d)と照合しています。テキスト処理の一括書換えやメモリ推定の変更は、今回のCPU再現から正否を判断できず取り込んでいません。
- [Gradio公式の変更履歴](https://gradio.app/changelog)では6.29.0を確認。現在の6.17.3には専用のfrontend互換パッチがあるため、自動更新は行っていません。
- [Gradio現行のTabs実装](https://raw.githubusercontent.com/gradio-app/gradio/main/js/tabs/shared/Tabs.svelte)は登録情報と表示用一覧を同期しています。今回の固定版修正は、現物のAppTree・Tabsを読んで未選択タブの初期表示とmutable state通知を切り分けたものです。現行版全体の移植は行っていません。
- [Pillow 12.3.0の公式release notes](https://pillow.readthedocs.io/en/stable/releasenotes/12.3.0.html)でWindowsViewer等の修正を確認。現在のrequirementsも12.3.0です。利用環境の`pip check`は成功しました。
- [PyTorchの最新SDPA資料](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)はquery・key・valueのhead/sequence軸と出力shapeを明記しています。Attentionの回帰試験は、導入済みのPyTorch 2.13.0で実際のCPU演算と比較しました。
- [HuggingFace PEFTのDoRA実装](https://raw.githubusercontent.com/huggingface/peft/main/src/peft/tuners/lora/dora.py)はLoRAを加えたweightからnormを計算しています。今回のDoRA修正もこの定義と照合しました。
- [LyCORISのLoKr保存形式](https://raw.githubusercontent.com/KohakuBlueleaf/LyCORIS/main/lycoris/modules/lokr.py)はDoRAの入力軸・出力軸のscaleを分けています。[LoKrの一次実装](https://raw.githubusercontent.com/KohakuBlueleaf/LyCORIS/main/lycoris/functional/lokr.py)とも再構成とKronecker積を照合しました。
- [ComfyUI現行の全精度経路](https://raw.githubusercontent.com/Comfy-Org/ComfyUI/master/comfy/ops.py)は、全精度指定時に量子化重みを展開してから演算します。固定版との差を確認し、同じ最小分岐を反映しました。
- [Python 3.13のstat資料](https://docs.python.org/3.13/library/os.html#os.stat_result.st_ctime)はWindowsのctimeが現在も作成時刻であることを説明しています。[MicrosoftのFILE_BASIC_INFO資料](https://learn.microsoft.com/en-us/windows/win32/api/winbase/ns-winbase-file_basic_info)はChangeTimeとLastWriteTimeを区別しています。実NTFSでmtimeを復元した書換えを確認し、cacheの識別にChangeTimeを追加しました。
- DDIM/PLMSのstep配列は[AUTOMATIC1111の現行実装](https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/modules/sd_samplers_timesteps_impl.py)と照合しました。[CompVis本家のDDIM](https://raw.githubusercontent.com/CompVis/stable-diffusion/main/ldm/models/diffusion/ddim.py)とはstepの契約が異なるため、全生成結果へ影響する配列変更は行っていません。今回の修正はstrength 0に限定しています。
- [Qwenの公式model card](https://huggingface.co/Qwen/Qwen-Image-2.1)は画像生成・編集・RGBAと最大10枚の参照を説明しています。これらは既存対応との照合に使い、新機能とは扱っていません。
- [Alibaba PAIのFun Acc model card](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs/blob/main/README.md)は4 NFEを説明し、小さい密な文字の劣化、編集結果のぼけ・暗さを制約として挙げています。速い候補生成と文字の最終仕上げを同じ評価だけで比較すべきではありません。
- [LanPaintの開発元README](https://github.com/scraed/LanPaint)はQwen Image 2.1の透明画像・マスク編集とH3の映像・音声の局所編集を説明しています。現在のForge Neoへの導入互換性と画質・速度は未検証です。

## 新機能案

以下は提案だけで、コードへ追加していません。

**独立した編集の良い変更だけを組み合わせる。** 同じ元画像に対して別々に生成した顔・服・背景の変更を差分として持ち、「Aの顔とBの服」を再生成せず採用する案です。今の完成画像・固定版の選択から、選択単位を変更領域へ移します。同じ元画像のhashと寸法、重ならないマスク、重なる場合の競合選択が成立条件です。最小検証は保存済みの元画像と2つの結果・マスクを合成し、両方の変更の採用と範囲外RGBA完全一致を確認することです。重なる影や反射は、機械的な合成では整合しない可能性があります。

**マスクを生成中の制約として使う局所編集。** 現在の範囲外固定に加えて、LanPaint型の条件付きサンプリングを比較する案です。描き直す領域の周囲を生成の各段階で参照することで、仕上げ時の合成だけでは解消できない境界の不整合を減らせる可能性があります。固定版Diffusers・RGBA latentとの互換性が条件で、計算回数と時間が増えます。最初は通常版Qwenで同じ元画像・マスク・Seedを使い、境界の破綻、保持画素、実行時間を比較します。TurboやFun Accへの適性は別に確認する必要があります。
