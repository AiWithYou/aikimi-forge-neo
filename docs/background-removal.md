# Extrasの背景除去

Extrasで画像を読み込み、`Background Removal / 背景除去`をオンにして実行する。
拡大や色補正が不要なら、その項目はオフにする。
結果は透過PNGになり、`マスクも保存`をオンにすると白黒マスクも出力する。
単画像・画像バッチ・フォルダ一括処理に対応する。

| モデル | 推論解像度 | 用途 |
| --- | --- | --- |
| BiRefNet（初期選択） | 1024 × 1024 | 通常の切り抜き |
| BiRefNet HR | 2048 × 2048 | 高解像度の細部 |
| BiRefNet HR Matting | 2048 × 2048 | 髪や半透明部分 |

入力画像の寸法と元の透明部分を維持する。生成したマスクは二値化せずアルファとして適用する。
通常の保存形式がJPEGの場合でも、この処理の出力はPNGにする。
動画の背景除去は実行前にエラーにする。既存の動画出力では透過を保存できないため。

## モデルの取得

アプリ起動時やモデル選択時には取得しない。初回実行時に選んだモデルだけを
`models/background_removal/<モデルID>/<固定リビジョン>/`へ取得し、以後は通信せず再利用する。
重みは各モデル約444MB。取得に失敗した場合は再実行するとHubのキャッシュを使って再試行する。
推論には既存のPyTorch環境を使い、処理後はGPUからCPUへ戻す。CPUには直近の1モデルだけを保持する。

事前に取得する場合は次を実行する。

```bat
download_background_models.bat
download_background_models.bat --model birefnet-hr
download_background_models.bat --model birefnet-hr-matting
```

公式モデルのコードとsafetensorsを固定リビジョンで取得する。モデルの自動更新は行わない。
公式実装が使う`timm`はアプリのrequirementsに含める。

## モデル選定と確認

[参考記事](https://note.com/ai_image_journey/n/nf45afca51b00)の花火背景のイラストで3モデルを比較した。
通常版は細い髪の縁にギザつきがあるが、髪のハイライトを保った。
HRは縁が滑らかになる一方、一部のハイライトを透過した。
HR Mattingはさらに毛先を薄くした。この結果から通常版を初期選択とした。
画像ごとの優劣があるため、HRを一律に上位とは扱わない。

RTX 3090で3モデルの取得・実推論を確認した。Extrasの実画面から通常版で1200 × 1200画像を処理し、
透過PNGとマスクの保存まで約3.9秒、GPU使用量のピークは約1.6GBだった（ダウンロード済み）。
HRの単体推論テストではピーク約5.4GB。処理時間やメモリ量は画像・環境で変わる。

記事中のDepth Anythingによる奥行き推定やLaMaによる不要物消去は、この背景除去には含めない。

モデルと前処理の出典：

- [BiRefNet公式実装（MIT）](https://github.com/ZhengPeng7/BiRefNet)
- [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet)
- [BiRefNet HR](https://huggingface.co/ZhengPeng7/BiRefNet_HR)
- [BiRefNet HR Matting](https://huggingface.co/ZhengPeng7/BiRefNet_HR-matting)
