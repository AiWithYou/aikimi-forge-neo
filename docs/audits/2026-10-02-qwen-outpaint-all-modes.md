# Qwen Outpaint v3.5.0：モデル・画風LoRA・高速化・制御の共用

確認日：2026-10-02。Windows 11、RTX 3090 24GB、RAM 64GB、専用workerのPyTorch 2.13.0+cu130。UIはGradio 6.17.3。

## 変更と判断

「画像生成」のモデル・配置・複数LoRAと個別強度・Fun Acc・Sparse・ControlNet・互換ローカル本体を、Outpaintの生成開始時に直接取り込みます。入力はGradioのブラウザー別コンポーネント値で、サーバーのグローバルな設定辞書は使いません。元画像・余白・Steps・Seed・境界幅はOutpaint側を使い、通常編集の注釈や手描きマスクは持ち込みません。

Opus 5.5のUI相談とGPT-6 Astraの実装相談を実施しました。共通の編集元と適用条件の表示、Fun Acc → Outpaint → 画風LoRA → ControlNetの順序を採用。Gradioの実コールバックで別Blocks間の入力を渡せることを確認できたため、追加のセッション設定ストアや生成前の二度押し確認は導入していません。

単純な制限解除ではなく、PDDが元のLinear層を列挙した後にOutpaintと画風の差分を加えるように変更しました。GGUFの小さな`proj_out`だけをBF16に戻し、PDDが出力ヘッドを複製できるようにします。本体の残りの量子化層と保存済みモデルには書き込みません。

TurboはViggleの4-stepスケジューラ、Fun AccはPDDと配布sigma列・stepコールバックを使います。[Viggleの固定revisionの説明](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo/blob/bafc91e4cc934f5fb1406b22496a0bed9b99c548/README.md)も確認しました。配布Turbo本体へ別のTurbo LoRAを重ねる構成ではありません。Turbo＋Fun Accは排他、Sparse＋Fun Acc／ControlNetは現在のKVキャッシュ条件が合わず、生成前に拒否します。

## 実生成の条件と判定

元画像512×384、余白は左64・上32・右96・下64、完成672×480、Seed 42、境界幅0、Outpaint v2、CPU退避。画風は`celstk/20260926_193229_22397712_step_024000.safetensors`を0.65、学習時のConvRotとの量子化差を明示的に許可しました。ControlNetはCannyとして記録した手製の白い矩形の制御画像、強度0.5、外側Inpaint ONです。自動のエッジ抽出や制御追従の画質評価ではありません。

共通の指示文は公開JSONにそのまま残しています。元画像は青いティーポットですが、試験の追記には`blue flower pot`という表現が含まれています。全ケースで同じ文を使った実行試験です。

通常版4精度はFun Acc併用、Turbo 2精度はFun Accなしで、すべて4 steps。成功の判定はPNG保存、完成寸法、元画像領域のRGBA全画素一致、Outpaint 224層・画風LoRA 128層の読み込み、ControlNetの実使用、各スケジューラとKV条件です。速度や画質の優劣を決める試験ではありません。

上記の通常版4精度・Turbo 2精度はすべて実生成に成功しました。実測表・画像・SHA-256・設定は[公開証拠](../assets/qwen-outpaint-v3.5.0/README.md)と[検証JSON](../assets/qwen-outpaint-v3.5.0/verification.json)に掲載しています。読み込みにはキャッシュの状態や初回変換・保存の差があるため、単発の読み込み時間を精度の順位に使いません。

通常版Q4の同条件で画風強度を0へ変更すると、適用された画風層は0となりました。強度0.65との比較では追加領域125,952画素中113,855画素が異なり、RGB平均絶対差は3.179902でした。元画像部分は両方で一致しています。[強度0の比較JSON](../assets/qwen-outpaint-v3.5.0/style-zero-comparison.json)

さらに通常版Q4＋Outpaint v2＋画風0.65を、Fun Acc・ControlNet OFF、8 steps、固定Sparse 75%で実生成しました。Sparse対象のAttention呼び出しは224回、denseは32回、Jev API呼び出しは0回。PNG保存と元画像全画素一致も成功しました。この一件は他の4-step併用試験とは条件が異なるため、生成時間を直接比較しません。

## 回帰確認と修正

- 実際のPDD Linear、Outpaint、複数の画風LoRAの加算結果と元重みの不変を小型テンソルで確認。対応行列とGGUFの論理寸法・出力層だけの復元を含む51テストが成功しました。
- 主要11モジュールの回帰は234成功、7スキップ、93 subtests成功。最終UI配置変更後は86成功・63 subtests成功、画像・マスク境界とServiceの追加確認は90成功・23 subtests成功、開始時の適用条件表示を含む最終変更後は92成功・68 subtests成功。これらは重複を含む別実行で、件数は合算しません。変更したPythonコードのRuff確認も成功しました。
- Gradioの実`Request`注入を使い、開始時の新しいLoRA強度・設定がRequestへ渡ること、別ブラウザーの値と混ざらないことを確認しました。
- Chromeで通常版のOutpaint Stepsを31へ変更し、Turboで4に固定され、通常版へ戻すと31へ復帰することを確認。狭い画面では実測387 CSS pxでページのscrollWidthも387となり、共有条件の長いLoRA名と生成ボタンが横にはみ出さないことを確認しました。検証用viewport指定は解除しています。
- LoRA選択後に強度表と量子化差のチェック項目が現れない問題を修正。固定Sparseの保持率も非表示時にコンポーネントを保持します。
- ControlNetのマスク指定を通常編集の隠れた欄からControlNet欄へ移動。OFFに戻すとマスク指定も解除します。Outpaintでは元画像サイズの制御画像を元画像位置へ配置し、余白だけ白のマスクを作成します。
- 通常生成でもControlNet OFFなら以前のアップロード画像を有効な制御とみなさず、Sparseの衝突を誤判定しないようにしました。
- このOFF切替の回帰を追加した後、アダプター合成・対応行列・ControlNetの63テストが成功しました。
- 制御画像はジョブ内へコピーしてから拡張し、元のアップロードファイルへ書き込みません。参照・制御・外側マスクの寸法と保存先、マスク内容をworker側でも検証します。
- JPEGのEXIFで幅と高さが入れ替わる場合も、画面の寸法判定とジョブのコピー処理を一致させました。この追加修正後のOutpaint試験は67成功・60 subtests成功です。
- タイムラプスの確認で、停止後も前回画像のラベルに「新しい画像を生成中」が残る問題を発見。停止・失敗・ジョブ消失時は「前回の生成結果 · 開始時の設定」へ戻し、PNGと生成情報を保持します。追加した回帰を含むOutpaintの68テストが成功しました。

## 確認の限界

追加アダプターが読み込まれて実行できることと、すべてのLoRA・被写体で意図した画風や制御を得られることは別です。蒸留モデルと追加アダプターの画質は実験扱いです。互換ローカル本体の受付・構造検証は回帰試験で確認しましたが、未知の外部モデルを網羅したGPU試験ではありません。Sparseの数値ルール・Jevを含む全方式・全精度の画質や高速化率も評価していません。

## GUI実生成と最終表示の修正

Chromeで画像生成側の通常版Q4・画風0.65・Fun Acc・ControlNet Canny 0.5＋外側Inpaintを指定し、Outpaintの生成ボタンから実行しました。元画像・余白・Seed・境界幅は上記行列試験と同じで、描き足す内容は元画像に合わせて`Extend the blue teapot, house plants and the room naturally.`へ修正しました。

1回目の処理中に次回の共通設定をBF16・画風0.1へ変更しました。次回条件の表示は変わりましたが、進行中のRequest・結果metadataは開始時のQ4・0.65を保持。完成画面の「この結果の生成情報」にも開始時の条件が表示されました。PNGは672×480、元画像RGBA全画素一致、Outpaint 224層・画風128層。ブラウザーの保存ボタンで取得したPNGのSHA-256もジョブ出力と一致しました。

設定をQ4・0.65へ戻した2回目は、同じworker PIDとモデルを再利用。読み込み0秒、生成21.211秒で完了しました。1回目の読み込みは28.296秒・生成100.681秒ですが、配置済み状態などの差がある単発実測なので高速化率は出しません。3回目の処理中に停止し、`cancelled`で完成PNGを書かず、2回目の画像・保存・再拡張・次の生成を保持し、停止ボタンは無効に戻りました。終了後にモデルを解放し、ブラウザーconsoleのエラー0件を確認しました。

この停止画面の録画確認でラベルだけ「新しい画像を生成中」が残る問題を発見しました。修正後にサーバーを再起動し、同じモデル・画風・Fun Acc・ControlNet・Seed・余白で再生成と停止を実施。停止後は「前回の生成結果 · 開始時の設定」と表示され、誤った生成中ラベルはなく、保存・再拡張・次の生成が有効であることを実画面でも確認しました。再起動後の読み込み251.038秒・生成98.806秒、次の停止ジョブは完成PNGなしです。[最終ラベル修正の検証JSON](../assets/qwen-outpaint-v3.5.0/stopped-label-verification.json)。モデルを解放してworkerの終了とconsoleエラー0件も確認しました。

- [GUIジョブの条件・検証JSON](../assets/qwen-outpaint-v3.5.0/ui-job-verification.json) · [開始時と次回の設定が異なる完成画面](../assets/qwen-outpaint-v3.5.0/ui-complete.png) · [停止後の結果保持](../assets/qwen-outpaint-v3.5.0/ui-stopped.png)
- [今回のタイムラプス（MP4）](../assets/qwen-outpaint-v3.5.0/timelapse.mp4) · [GIF](../assets/qwen-outpaint-v3.5.0/timelapse.gif) · [フレームの取得時刻・SHA-256](../assets/qwen-outpaint-v3.5.0/timelapse-frames.json)

動画は約1,295秒の操作・進行表示・完了・停止・表示バグの発見と修正後確認の実画面28枚を34.4秒へ圧縮した記録です。間に修正作業と再起動の待機も含みます。待機中のすべての秒や拡散途中の画像を収録したものではありません。

ドラッグGUIと連続した再拡張の実生成は[前版の記録](2026-10-02-qwen-outpaint-gui-release.md)に残しています。

![開始時のQ4・画風0.65で完成し、次回条件だけBF16・0.1へ変更](../assets/qwen-outpaint-v3.5.0/ui-complete.png)
