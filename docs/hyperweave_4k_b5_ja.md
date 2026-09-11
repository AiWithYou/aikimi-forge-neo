# HyperWeave 4K/8K：構造制約付き周波数選択によるタイル型生成アップスケール

**HyperWeave 4K/8K: Tiled Generative Upscaling with Structural Constraints and Frequency-Selective Residual Fusion**

**あいきみ**

## 要旨

生成型アップスケールは補間では得られない中周波・高周波の描写を加えられる一方、人物同一性、髪型、物体配置、色、タイル境界を変える危険がある。本稿は、既存の画像生成モデルを再学習せず、重複タイル、画像座標に固定した潜在ノイズ、低強度の基準再描画、全画面候補のハード棄却、周波数帯域別の残差採用、顔・頭部の局所再描画、低周波バックプロジェクションを組み合わせる HyperWeave 4K/8K を示す。1664×2353 のイラストを RTX 3090 上で 2897×4096 へ生成し、49回の内部生成を約662秒で完了した。Global Overdraw 候補は構造制約で棄却され、Anchor と Face/Head ROI を用いた安全側の出力になった。最終像は Lanczos より大きな中・高周波変化を持つ一方、入力への round-trip SSIM は 0.874 であり、忠実度では Lanczos の 0.994 を下回った。本結果は生成細部の正しさや優越性を証明せず、単一GPUでのフル4K実行可能性と棄却機構の作動を示す予備的事例である。

**キーワード：** 生成型超解像、潜在拡散、タイル生成、構造制約、周波数合成、4K

## 1. はじめに

Lanczos のような補間は入力画素から滑らかな高解像度画像を作り、構図と色を強く維持する。しかし、元画像に存在しない睫毛、髪内部の細線、布目、紙や木の微細構造は推定しない。Real-ESRGAN などの学習型復元や SR3 などの拡散型超解像は知覚的細部を生成できるが、単一の低解像度入力に対応する高解像度像は一意ではなく、生成細部は観測事実ではない。

HyperWeave は通常の補間型超解像ではなく、入力画像を構図・人物配置・顔向き・表情・髪型・衣装・物体・代表色・画風の設計図として扱う生成型再描画である。目的は画素一致の最大化だけではなく、「意味構造の制約を満たす候補の中で、自然で一貫した書き込み量を増やす」ことである。実装は Forge の独立した img2img Script であり、現在読み込まれている checkpoint、VAE、sampler、scheduler、CFG、LoRA、embedding、offload 方針を再利用する。

本稿の主な構成は次の四点である。

1. 出力全体をGPUへ常駐させない重複タイル生成と、重複位置で初期潜在ノイズを一致させる Coordinate Noise。
2. 低強度 Anchor、全画面候補、round-trip 構造評価、全候補棄却時の Anchor fallback。
3. 候補差分を六帯域へ分け、構造・方向・新規輪郭・タイル安定度・round-trip 信頼度を帯域別に適用する残差合成。
4. 顔・頭部などの ROI 再描画、タイル境界評価、低周波だけを入力へ戻す Back Projection。

## 2. 提案手法

### 2.1 段階計画と重複タイル

入力寸法を \((W_0,H_0)\)、目標を \((W_T,H_T)\) とする。各段階の倍率を 2.0 以下に制限し、最終段階だけ端数倍率を許す。内部処理解像度は潜在倍率8へ切り上げ、最後に指定寸法へ正確に戻す。今回の入力は長辺2353から4096への1.741倍であるため一段階となり、2897×4096を内部2904×4096で処理した。

既定タイルは1280×1280で、中央960×960を出力へ戻し、周囲160画素を文脈に使う。stride は768であり、通常の core overlap は192画素である。最終行・列は画像端へ一致させ、外周に接する raised-cosine window の重みを0にしない。タイル \(t\) の基準画像に対する生成差分を \(\Delta_t\)、窓を \(w_t\) とすると、全画面候補の差分は

```text
Δ = Σt wt Δt / max(Σt wt, ε)
```

で復元する。差分の二乗量も蓄積し、重複タイル間の分散から tile confidence を得る。候補画像を直接貼り付けないため、入力の低周波と候補の局所変化を分離して扱える。

### 2.2 画像座標に固定した潜在ノイズ

各タイルが同じ seed から独立ノイズを作ると、重複位置でノイズ位相が一致しない。HyperWeave は stage、pass、candidate、ROI を BLAKE2b で名前空間化し、PCG64 で出力全体に対応する CPU 上の潜在ノイズ canvas を作る。タイルは絶対潜在座標の crop を受け取る。

```text
seed' = BLAKE2b(base_seed, stage, pass, candidate, roi)
Nt = crop(Nglobal(seed'), absolute_latent_box(t))
```

したがって、同一候補内の重複領域では初期ノイズが bit-exact に一致する。保証範囲は sampler へ渡す初期 latent noise であり、SDE/Brownian sampler が sampling step 内で生成する二次ノイズまでは置換しない。

### 2.3 Anchor、候補生成、ハード棄却

各段階で、補間 base から低 strength の Global Anchor \(A\) を作る。続いて Anchor を入力として、候補 index ごとに一枚の Global Overdraw 候補 \(C_i\) を全タイルで完成させる。候補は次を評価する。

- source または前段へ縮小した round-trip SSIM、PSNR、低周波誤差、色差
- 輪郭位置、輪郭方向、連続する新規輪郭、二重線 proxy
- clipping、NaN/Inf、タイル境界 residual
- 中周波量、line continuity、material richness、style consistency、noise penalty

既定 strictness 0.70 では round-trip SSIM の下限は

```text
SSIMmin = 0.50 + 0.28 × strictness = 0.696
```

である。ハード制約を一つでも外れた候補はランキング対象にしない。全候補が外れた場合は Anchor へ戻し、「最も悪くない候補」を採らない。これは生成量を常に最大化する方式ではなく、危険な候補を捨てる fail-safe である。ただし Anchor 自身も生成像であり、fallback が入力画素との同一性を保証するわけではない。

### 2.4 周波数選択合成

候補残差 \(R=C-A\) を linear RGB で処理し、Gaussian blur \(G_\sigma\) から六帯域へ分ける。

```text
HIGH_0   = R - G1(R)
HIGH_1   = G1(R) - G2(R)
MID_HIGH = G2(R) - G4(R)
MID      = G4(R) - G8(R)
MID_LOW  = G8(R) - G16(R)
LOW      = G16(R)
```

各帯域は median、MAD、99.5百分位から求めた上限で tanh soft clipping する。輝度と色差を分け、色差 gain は輝度 gain の0.35倍を既定とする。出力は概念的に

```text
O = A + Σb gb Mb softclip(Rb)
```

である。\(M_b\) は structure protection、輪郭方向一致、新規輪郭 confidence、tile confidence、round-trip confidence、ROI、manual mask の積である。MID/MID_LOW/LOW ほど既存輪郭を強く守り、HIGH は既存輪郭に沿う細線を一定量許す。LOW は Low Frequency Lock により原則採用しない。

### 2.5 ROI、境界、Back Projection

顔検出は入力解像度で一度行い、各段階へ座標変換する。今回のイラストでは写真用検出器を用いず、左右人物へ Manual ROI を与えた。頭頂、前髪、側頭、首、肩、周辺背景まで拡張した二つの context が重なったため、一つの coherent ROI として処理した。ROI 候補にも round-trip と輪郭のハード制約を適用する。

タイル境界では、候補残差の境界勾配を同数の非境界位置と比較する。境界比が1.65を超えた場合だけ局所平滑化する。最後に出力を前段へ縮小し、Gaussian 低周波誤差だけを拡大して戻す。

```text
Ok+1 = clip(Ok + β U(Gσ(P - D(Ok))))
```

ここで \(P\) は前段、\(D\) と \(U\) は縮小・拡大、\(\beta=0.70\) である。誤差が増えた場合は係数を半減し、それでも悪化すれば rollback する。

![HyperWeave の処理フロー](../output/hyperweave_paper_20260724/assets/hyperweave_pipeline.png)

## 3. フル4K事例

### 3.1 条件

入力は1664×2353の不透明RGBAイラスト、出力は4K long edge の2897×4096である。Krea2_seedram5p2_style_int8_convrot checkpoint、DPM++ 2M SDE、Simple scheduler、CFG 1.0、distilled CFG 1.15、Structure Safe、Exact Steps 1、seed 976834651を用いた。GPUは24 GiBの NVIDIA GeForce RTX 3090 である。

| 項目 | 値 |
|---|---:|
| 入力 / 出力 | 1664×2353 / 2897×4096 |
| 内部 canvas | 2904×4096 |
| 段階数 | 1 |
| Tile geometry | input 1280, core 960, context 160, stride 768 |
| タイル数 | 24 / pass |
| 内部生成 | Anchor 24 + Global 24 + Face ROI 1 = 49 |
| strength | Anchor 0.12, Global 0.24, Face 0.22 |
| Global / Face candidates | 1 / 1 |
| Hair / Material / Micro | 無効 |
| Back Projection | 1回, β=0.70 |

### 3.2 候補選択と資源使用量

Anchor は round-trip SSIM 0.8382 で受理された。Global Overdraw 候補はハード制約で棄却され、`selected_global_candidate=null` となった。したがって Global 候補から採用された帯域 energy は0である。統合 Face/Head ROI の候補は round-trip SSIM 0.7575 で受理された。最終差分は Anchor、Face/Head ROI、境界処理、Back Projection に由来する。

処理時間は662.12秒、内部呼び出しを含む live test 全体は約700秒であった。PyTorch peak allocated は20,953,574,400 bytes、peak reserved は21,720,203,264 bytesであり、それぞれ約19.51 GiB、20.23 GiBである。CPU working RAM 見積りは約771 MiB、accumulator は約227 MiB、memmap は使用しなかった。CUDA OOM は0回であった。

計画境界8本に対する seam ratio は0.810であり、境界勾配は比較位置より小さかった。Back Projection は低周波誤差を0.00092294から0.00015321へ減らし、rollback は発生しなかった。

### 3.3 Lanczosとの比較

![入力、Lanczos、HyperWeave と左右人物の顔クロップ](../output/hyperweave_paper_20260724/assets/hyperweave_comparison.png)

Lanczos は入力忠実度の基準であり、正解高解像度画像ではない。各出力を入力解像度へ戻した比較を表に示す。

| 評価量 | Lanczos | HyperWeave |
|---|---:|---:|
| round-trip SSIM ↑ | 0.9943 | 0.8738 |
| round-trip PSNR ↑ | 40.37 dB | 27.86 dB |
| 低周波輝度誤差 ↓ | 5.14e-7 | 1.54e-4 |
| 色 drift ↓ | 8.41e-5 | 3.53e-3 |
| edge displacement ↓ | 0.0054 | 0.0440 |
| face structure ↑ | 0.9846 | 0.8633 |
| MID energy | 2.15e-8 | 3.63e-5 |
| MID_HIGH energy | 1.17e-7 | 1.19e-4 |
| HIGH energy | 1.76e-5 | 5.01e-4 |
| seam ratio | - | 0.8104 |

最終 HyperWeave は Lanczos に対して MID 約1688倍、MID_HIGH 約1016倍、HIGH 約28.5倍の変化 energy を持つ。ただし、この値は「正しい細部」の量ではなく、基準からの帯域別変化量である。round-trip SSIM、PSNR、face structure、hair-flow、coherent-line は Lanczos が上回った。元画像がすでに1664×2353で拡大率が1.741倍にとどまることも、補間基準に有利な条件である。

クロップでは顔、視線、髪型シルエットは概ね保たれるが、目、髪内部線、陰影はわずかに再描画されている。この一例だけでは、その差を真の復元または知覚品質向上とは判定できない。

## 4. 考察

本事例が示す最も明確な結果は、生成量の最大化ではなく安全機構の挙動である。Global 候補を全面採用する実装なら、構造違反を含む全画面変化が残る。HyperWeave は候補を棄却し、Anchor と局所 Face/Head ROI に留めた。Coordinate Noise と重複差分蓄積により、24タイルずつ処理しながらフル4Kを24 GiB GPUで完走し、出力解像度全体をGPUへ保持しなかった。

一方、Anchor fallback は「元画像へ戻る」ことと同義ではない。今回も Anchor の round-trip SSIM は0.838であり、最終像の入力忠実度は Lanczos より低い。より厳密な Structure Safe を目指すなら、Anchor の下限を上げる、Anchor 不合格時に補間 base へ戻す、ROI の顔構造しきい値を強める、といった追加アブレーションが必要である。

また、round-trip 指標は高解像度でだけ現れる細部を意図的に消すため、生成細部の自然さを直接測らない。主観評価、複数画像、複数 seed、複数モデル、既存復元法、顔同一性モデル、専門家によるペア比較が必要である。

## 5. 制限と結論

本実験は一画像、一モデル、一seed、Exact Steps 1の単一事例であり、統計的な性能評価ではない。高解像度 ground truth がなく、生成された睫毛、髪、布目、反射が元の被写体に実在したかを検証できない。Hair、Material、Micro pass と複数候補、4Kから8Kへの段階処理も本事例では評価していない。SDE sampler の二次ノイズはタイル座標で固定していない。イラストの顔は Manual ROI に依存する。

以上の制限の下で、HyperWeave は重複タイル、画像座標ノイズ、候補棄却、周波数選択、ROI、低周波 Back Projection を一つの Forge 実行経路へ統合し、RTX 3090 上で2897×4096の実モデル生成を完了した。Global 候補の棄却は fail-safe が実際に作動した証拠であるが、Lanczosより高品質であることの証明ではない。今後は入力へ戻る最終 fallback、候補数と Exact Steps のアブレーション、複数データと人手評価により、生成細部と構造忠実度の Pareto frontier を測る必要がある。

## 参考文献

1. J. Ho, A. Jain, and P. Abbeel, “[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239),” *NeurIPS*, 2020.
2. R. Rombach et al., “[High-Resolution Image Synthesis with Latent Diffusion Models](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html),” *CVPR*, pp. 10684–10695, 2022.
3. C. Meng et al., “[SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073),” *ICLR*, 2022.
4. C. Saharia et al., “[Image Super-Resolution via Iterative Refinement](https://arxiv.org/abs/2104.07636),” arXiv:2104.07636, 2021.
5. O. Bar-Tal et al., “[MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation](https://proceedings.mlr.press/v202/bar-tal23a.html),” *ICML*, PMLR 202, pp. 1737–1752, 2023.
6. L. Zhang, A. Rao, and M. Agrawala, “[Adding Conditional Control to Text-to-Image Diffusion Models](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html),” *ICCV*, pp. 3836–3847, 2023.
7. P. J. Burt and E. H. Adelson, “[The Laplacian Pyramid as a Compact Image Code](https://doi.org/10.1109/TCOM.1983.1095851),” *IEEE Transactions on Communications*, 31(4), pp. 532–540, 1983.
8. Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “[Image Quality Assessment: From Error Visibility to Structural Similarity](https://doi.org/10.1109/TIP.2003.819861),” *IEEE Transactions on Image Processing*, 13(4), pp. 600–612, 2004.
9. X. Wang et al., “[Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data](https://openaccess.thecvf.com/content/ICCV2021W/AIM/html/Wang_Real-ESRGAN_Training_Real-World_Blind_Super-Resolution_With_Pure_Synthetic_Data_ICCVW_2021_paper.html),” *ICCV Workshops*, pp. 1905–1914, 2021.

## 組版

JIS B5・2ページ版は次で生成する。

```powershell
& .\venv\Scripts\python.exe `
  .\tools\build_hyperweave_paper.py
```

出力先は `output/pdf/hyperweave_4k_b5_ja.pdf`、図版は `output/hyperweave_paper_20260724/assets/` である。
