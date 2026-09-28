# Qwen Image 2.1 追加LoRA（2026-09-28）

## 完了範囲

任意のQwen Image 2.1用線形LoRAをローカルの `models/Qwen-Image-2.1/loras/` から複数選択し、各−2〜2の強度で適用する。強度0はロードしない。比較スイッチ・比較生成・比較結果UIは製品から削除した。選択と強度はファイル名で対応させ、選択追加・削除で残った値を保持する。失敗するアダプターが混ざればモデルを部分変更しない。

## 配布元との互換性

- celstk/qwen2.1_lora revision: `296293875f8c0d25a4e15becbc09aed20dd4f2f1`
- weight: `20260926_193229_22397712_step_024000.safetensors`
- SHA-256: `863b0495905dfd13bb6cfe328e6bd9d394d6818a1ab2d5dadd4c06efe4313ab8`
- 538,025,676 bytes、128線形層。Hugging FaceのLFSハッシュと照合済み。
- SushiUI commit `69de838b18dbed8e8fc1e1294fb649a4f7dc5452` の `backend/core/pipeline_backends/qwen_image_21.py:55-78` を確認。通常の全面生成では学習補助global adapterの6テンソルを除外する。
- 元実装はConvRot INT8以外を拒否する。Forgeのbitsandbytes INT8/GGUFとの差は埋まっていない。ユーザーが「異なる量子化で試す」をONにした実験としてのみ適用し、生成情報に差を記録する。同一再現の主張ではない。

## 実行確認

- 9件の追加LoRAテストと65件の既存service/workerテスト: 74件、失敗0、skip4。UIの数値文字列修正後に9件、イベント順序を整理した最終版でLoRA/serviceの33件を再実行して成功。
- 2ファイルの同じ層への差分加算、alpha/rank、個別強度、非有限値・重複・パス逸脱、2つ目が不正でも元モデルを変更しないこと、0の読み込み省略、キャッシュ切替を検証。
- ブラウザー: 複数選択、0.75/0.25の入力、名前列の編集不可、先頭削除後の0.25保持、再追加時1.0、実験設定による生成可否、選択変更時の実験設定OFFを確認。`work/lora-20260928/multiple-ui-final.jpg`。
- 複数の実生成: `outputs/qwen-image-2.1/multiple-lora-smoke-20260928/`。Q4_K_M・256×320・2 steps。検証用に同じ重みを2つの名前で参照し、0.75と0.25で同時ロード。両方128層を適用したメタデータと1枚のPNGを確認。検証用の追加ファイルは削除済み。これは動作試験で、異なる2作品の画風の相性評価ではない。
- INT8とQ4で指定LoRA単独の実生成も確認。W4A8/BF16の追加LoRA実生成は今回未実施。
- Ruff・差分の空白チェックに合格。
- v3.1.0公開前にCIのCPU・オフライン設定でLoRA/service/worker、既存ForgeのLoRA一覧、CI設定、文書のローカルパス漏れの85件を実行。失敗0、skip4。変更したPython 9ファイルはRuff 0.15.20の検査・整形確認に合格。追加テストと取得補助を継続的なCIの対象にも含めた。

## 今回限りの画質比較

`outputs/qwen-image-2.1/celstk-portrait-comparison-20260928/` にLoRAなし/ありの960×1280 PNG、各メタデータ、`comparison.html` を保存。INT8・40 steps・Seed 92824000・強度1。プロンプト・Seed・サイズ・steps・スケジューラが一致し、なしはアダプター未接続で生成。異なる量子化での実験なので、学習時と同じ動作や幅広い品質改善を示す比較ではない。

## Opus相談

Claude Opus 5.5 / mediumで3回実施。方向性、試作画面のレビュー、ユーザーによる「比較機能は不要・複数LoRA読み込み」に合わせた設計を相談。`work/lora-20260928/opus-*-response.md` と `.meta.json` に原文・モデル確認を保存。複数選択＋名前を編集不可にした強度表、名前キーでの強度保持、未対応時の明示的な中止を採用。比較UIは最終成果に含まない。

公開前に同じモデル・推論設定でREADME構成と版数を追加相談。`work/lora-release-20260928/opus-release-response.md` と `.meta.json` に保存。従来ForgeとQwenの保存先・操作を2行の表で示す案、v3.1.0、実生成の確認範囲の明記を採用した。「導入済み」はローカル環境だけの状態なので公開文書から除去。互換性の根拠を追えるよう、検証したLoRAと固定版SushiUIへの出典は残した。
