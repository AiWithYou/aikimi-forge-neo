# MiniMax H3 360° Orbit LoRA

一枚の写真を開始・終了の両端に使い、被写体の周りをカメラが一周して戻る動画を作るための追加LoRAです。対象は作者の [pablodawson/MiniMax-H3-360-Orbit-LoRA](https://huggingface.co/pablodawson/MiniMax-H3-360-Orbit-LoRA)。Ref2VA向けのORB360とは別のモデルです。

## 導入契約

資材の確認日: 2026-10-11。重みはGitに含めず、H3の設定済みモデル保存先の `loras/` へ置きます。固定revision・サイズ・SHA-256を検証し、既存の異なるファイルは上書きしません。UIを開いただけでは取得しません。

| 項目 | 値 |
| --- | --- |
| ファイル | `minimax_h3_flf2v_lora_v1.safetensors` |
| revision | `5ddbc2dbbe95edbbdaf5017c3e934b1d01791697` |
| 容量 | 155,111,424 bytes（約148 MiB） |
| SHA-256 | `14f13e3effaf3e729fdc0c97680344aa63f473be0c55963f963d718b3db2a4d4` |

既存のH3本体・テキストエンコーダー・VAEを再利用し、ComfyUI標準のLoRA読み込みノードを使います。追加の学習ツールや外部ノードは不要です。利用条件は [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) に従います。

## 使い方

1. Forge Neoを再起動し、H3 Studioの「360°周回（写真1枚）」を選びます。
2. 「周回する写真」に一枚追加します。未導入なら「Orbit LoRAを準備」を押します。準備状態の確認と導入は「実行環境とモデル」で選択したComfyUIのモデル保存先を使います。
3. 被写体とシーンをプロンプトに書き、「周回の指示を追記」でカメラの一周を指示します。
4. 初期値の768×768・3秒・28 Steps・強さ1.0で「周回動画を生成（無音）」を押します。

CLIからの導入・ローカル検証も可能です。プロジェクト直下のPowerShellで実行します。

```powershell
.\venv\Scripts\python.exe tools\prepare_minimax_h3_orbit.py
.\venv\Scripts\python.exe tools\prepare_minimax_h3_orbit.py --verify
```

## 生成契約

作者の検証条件はpruned FL2VA INT8 ConvRot、768×768、73フレーム／24fps（約3.04秒）、28 Steps、LoRA強度1.0、同一画像を開始・終了へ、無音です。作者はモデルカードの指定全文プロンプトを変更せず使う条件で検証しています。Neoの「周回の指示を追記」は既存の文章を保つ独自の補助で、作者の全文プロンプトと同一ではありません。sampler／schedulerは作者が指定していないため、公式ComfyUI構成の初期値を使用します。CFG・negative promptは追加しません。通常のH3生成の5〜15秒という条件は維持し、Orbitの場合だけ3秒から使用できます。

非正方形の写真は生成サイズへ一度だけ中央切り抜き・拡縮し、同じ画像出力を両端へ接続します。H3標準ノードは開始画像をstretch、終了画像をcenter cropするため、この前処理なしでは同一写真でも両端の構図が異なります。

ComfyUI標準の `LoraLoaderModelOnly` をFL2VAロード直後に接続し、推論モデルとスケジューラーへ適用します。無音出力では音声VAE・音声デコード・動画への音声接続を省きます。H3標準ノードは内部で音声を含むlatentを作るため、音声推論そのものを停止する意味ではありません。

履歴とComfyUIへの書き出しにOrbitの選択・強度・モデル識別情報を残します。履歴から戻す際は入力写真をもう一度指定してください。完成動画は全フレーム・解像度・24fpsを検証し、Orbitだけ無音の動画を受け入れます。

生成完了の案内は通常モードで「音声付き動画」、Orbitで「無音動画」と表示します。モード共通の履歴にはどちらのMP4も並びます。

## 設定と制限

通常の三モードは設定を共有し、周回側は寸法・尺・Steps・scheduler・強さを別に保持します。モードの往復や周回の「推奨値に戻す」で、通常側の設定・開始／終了写真・プロンプトは変わりません。周回写真は専用の一枠です。

写真未指定、LoRA未導入・導入中、長尺生成との併用時は理由を表示して生成を停止します。キーボードの生成操作も同じ条件に従います。長尺生成は一周の条件を変えるため併用できません。ControlNet等の追加設定は利用できますが、併用品質は未検証です。

作者の学習対象は人物を中心とした正方形・73フレームの映像です。人物以外への一般的な品質、寸法や長さの変更、Turbo／W4A8／ControlNetとの併用品質は未検証です。入力に写らない背面はモデルが生成するため、正確な形状の復元、場面全体の360°整合性、完全な静止は保証しません。

必要なRAM・VRAMと生成時間は本体・VAE・解像度・尺・高速化設定によって変わります。メモリ設定は[H3 Studioガイド](../extensions-builtin/minimax-h3-studio/README.md)を参照してください。

構成の一次資料: [作者の固定版モデルカード](https://huggingface.co/pablodawson/MiniMax-H3-360-Orbit-LoRA/blob/5ddbc2dbbe95edbbdaf5017c3e934b1d01791697/README.md)、[公式H3ガイド](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)、[公式I2V graph](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_i2v.json)。
