# 承認済み4K画像に対する局所超解像残差のFail-Closed評価

**Fail-Closed Evaluation of Local Supersample Residual Refinement on an Approved 4K Krea2 Image**

AiWithYou — Technical Short Paper / 2026-07-13

## 要旨

本研究は、生成済み画像の構図・顔・色を保ったまま局所的な微細描写だけを追加することを目的として、Forge Neo に実装した Krea2 Local Supersample Detail を単一の承認済み縦長4K画像で評価した。512 px の局所 payload を 1536 px または 2048 px に拡大して低 denoise の Krea2 img2img を適用するが、候補画像自体は貼り戻さない。無処理往復像を $C_0$、Krea2 処理後の往復像を $C_1$ とし、linear-light で得た $C_1-C_0$ の帯域制限残差だけを品質 gate 通過時に中央 core へ合成する。

RTX 3090 上で、指定プロンプト・同一 seed による Safe 1536、Ultra Detail 1536、ROI Ultra 2048、および Safe 1536 全画面の4条件を実測した。顔領域の1536条件と目領域の2048条件は、全候補が detail-energy gate により不採用となり、出力画素は入力と完全一致した。全画面条件では117タイル中12タイルのみが採用され、変更画素率3.7205%、RGB code差の99 percentile 1、最大8、合成時 clipping 0、ピークVRAM 22,478 MiB、処理時間1,387.2 sであった。本単一事例は一般的な画質向上を証明しない一方、既に高密度な領域で候補が平滑化方向へ進んだときに変更を拒否できる fail-closed 特性を実証した。

**Keywords:** Krea2, local supersampling, residual refinement, fail-closed quality gate, 4K, RTX 3090

## 1. はじめに

拡散モデルを用いた高解像度化では、長辺を一度に処理するとVRAM制約を受け、分割処理では顔同一性の変化、低周波の色・明度ドリフト、タイル継ぎ目、偽テクスチャが問題になる。Latent Diffusion は潜在空間により高解像度生成の計算量を抑え、MultiDiffusion は複数の生成経路へ共有制約を与えて大きなキャンバスを扱う考え方を示した [2,3]。本研究の対象は再生成ではなく、承認後4K画像の一部にだけ保守的な残差を加える仕上げ工程である。

評価上の問いは「処理すれば常に細部が増えるか」ではない。むしろ、候補が元画像より細部を失う、または構図成分を変える場合に、元画像を壊さず何もしない結果へ戻れるかを検証する。

## 2. 手法

入力画像 $B$ から halo 付き512 px payloadを切り出し、Lanczosで処理辺 $s\in\{1536,2048\}$ へ拡大した像を $U(B)$ とする。同一の linear-light area downsample $D$ を使い、無処理往復と Krea2 refine $R$ の差を

$$
C_0=D(U(B)),\qquad C_1=D(R(U(B))),\qquad \Delta_{raw}=C_1-C_0
$$

と定義する。これにより、単なる拡大縮小の往復誤差を候補残差から相殺する。$\Delta_{raw}$ のluma低周波成分を Gaussian radius 12 px で除去し、元画像の構造と強エッジを保護する gate を掛け、Safeではluma/chromaを8/2 code、Ultraでは12/3 codeに制限する。次のいずれかに該当する候補はゼロ残差へ置換する。

- 平均残差が実質ゼロ
- $C_1$ の局所 detail energy が $C_0$ 以下
- 低周波 drift、RGB clipping、payload境界残差が閾値超過
- 2候補モードで同位置・同符号・局所相関の agreement coverage が5%未満

2候補は平均せず、品質 score が良い一方を代表とし、もう一方を支持証拠として agreement mask を掛ける。採用残差は384 px coreへ正規化重みで重畳する。すべて不採用なら、RGB画素列のSHA-256まで入力と一致する。

## 3. 実験条件

入力は既存4K preflightを通過した2896×4096 RGB PNG、seedは **3883506083**、base promptのSHA-256は **CFC5251C0001…** とした。ユーザー指定文字列を修正せず、そのまま使用した。

```text
light blue hair,long_wavy_hair,devil’s_horn,purple horn,purple_eyes,green_slime,jig eyes,smile,jitome,Expressionless,
```

使用 checkpoint は `turbo_gpt0630_krea2_final_forge_bnb_nf4.safetensors`、VAEはQwen Image VAE、text encoderはQwen3-VL 4B bf16、GPUはRTX 3090 24 GBである。VRAM、GPU利用率、温度、電力は `nvidia-smi` を1秒間隔で取得したため、プロセス専有値ではなくGPU全体の観測値である。

| ID | 範囲 / profile | tile | 候補 | steps / denoise |
|---|---|---:|---:|---:|
| A | 顔 ROI / Safe 1536 | 9 | 9 | 4 / 0.10 |
| B | 顔 ROI / Ultra Detail 1536 | 9 | 18 | 5 / 0.15 |
| C | 目 ROI / ROI Ultra 2048 | 1 | 2 | 5 / 0.14 |
| D | 全画面 / Safe 1536 | 117 | 117 | 4 / 0.10 |

顔ROIは `(1100,900,1700,1500)`、目ROIは `(1344,1024,1600,1280)` とした。Aの処理時間には初回model loadを含む。

## 4. 結果

| ID | 採用tile | 変更画素 | 時間 | peak VRAM | 判定 |
|---|---:|---:|---:|---:|---|
| A | 0 / 9 | 0% | 310.1 s | 21,479 MiB | 入力と完全一致 |
| B | 0 / 9 | 0% | 197.4 s | 21,617 MiB | 入力と完全一致 |
| C | 0 / 1 | 0% | 49.4 s | 22,600 MiB | 入力と完全一致 |
| D | 12 / 117 | 3.7205% | 1,387.2 s | 22,478 MiB | 微小残差のみ |

Dでは441,330画素が1 code以上変化したが、全RGB channelの平均絶対code差は0.02018、p95は0、p99は1、最大8であった。入力pixel hashは `241cba297e05…`、出力pixel hashは `5a5a3dc82e46…` である。採用tileは上端・左端と下部衣装の一部に偏り、顔中心では採用されなかった。候補全体のdetail-energy増分は平均−0.1733 codeで、多くの局所候補は元の4K画像より平滑化された。採用12 tileだけの平均増分は+0.0723 codeであった。

A–Cは候補がすべて `detail_energy_did_not_increase` を含む品質理由で拒否され、出力pixel hashが入力と同一になった。Cは2048処理自体には成功しOOMもなかったが、ピーク22,600 MiBで24 GB GPUの余裕は小さく、有用な残差も得られなかった。Dの合成metadataではclipping fraction 0、タイル境界に沿う連続的な差分帯は100% cropと増幅差分図で認めなかった。

## 5. 考察

本事例の主要な成果は「局所超解像が顔を改善した」ことではなく、改善の数値根拠がない顔・目候補を採用しなかったことである。既に細部を含む4K入力に低denoise img2imgを適用しても、モデル出力が局所帯域をわずかに平滑化する場合がある。正のdetail-energy、低周波drift、clipping、境界残差、候補間agreementを別々に検査する設計は、その状況で安全側のno-opを選択した。

一方、全画面では処理時間が23分を超えたにもかかわらず、変更は画面の3.72%かつp99で1 codeに留まり、顔には反映されなかった。この費用対効果から、本画像に対しては全画面実行よりSafe 1536のROI試行を先に行い、採用tileと100% cropを確認する運用が妥当である。ROI Ultra 2048は本GPUで動作したが、1536より一般に良いとはいえない。

## 6. 限界

本研究は画像1枚・prompt 1件・seed 1件のcase studyであり、外部ground truth、他方式との盲検比較、複数評価者による主観評価を含まない。detail-energy増加は知覚品質や意味的正しさを直接保証しない。GPU telemetryは全GPU値で、Aだけはmodel loadを含むため時間の単純比較もできない。閾値の一般性、異なる画風・顔サイズ・素材、量子化方式、GPUでの再現性は未検証である。

## 7. 結論

同一prompt・seedの承認済み4K画像に局所超解像残差を適用した結果、顔・目の全候補は安全にno-opとなり、全画面でも117 tile中12 tileだけが微小変更として採用された。本単一事例は普遍的な高精細化効果の証拠ではないが、悪化が疑われる候補を元画像へ戻すfail-closed実装の実証になった。実運用ではSafe 1536 ROI、100% crop確認、必要箇所だけの採用を基本とし、2048はVRAM余裕と実利を個別に確認すべきである。

## 参考文献

1. S. Lee et al., “Krea 2 Technical Report,” Krea, 2026. <https://www.krea.ai/blog/krea-2-technical-report>
2. R. Rombach et al., “High-Resolution Image Synthesis with Latent Diffusion Models,” CVPR, 2022. <https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html>
3. O. Bar-Tal et al., “MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation,” ICML, 2023. <https://proceedings.mlr.press/v202/bar-tal23a.html>

## 再現用成果物

- 実行器: `tools/run_krea2_local_supersample_experiment.py`
- 全画面manifest: `output/krea2_local_supersample_case_study/local_supersample_20260713_201930_238319/experiment_manifest.json`
- B5 PDF生成器: `tools/build_krea2_local_supersample_paper.py`
- 入力file SHA-256: `B303B89DEEB1C2FBD1FAEF72156465341F75177498F848BDB0D4D86F65048545`
- 全画面出力file SHA-256: `4C2BDBE5F8A4FFE0E610F1AB1BFF0C0E070EF2D85B67950AA8B30C5FE9033009`
