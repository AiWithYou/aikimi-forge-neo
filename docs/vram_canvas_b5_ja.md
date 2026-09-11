# Krea2 Smart 4K/8K：独立位相合意による高密度・高解像度リファイン

**Krea2 Smart 4K/8K: Dense High-Resolution Refinement with Independent-Phase Consensus**

AiWithYou — Algorithm Note / Single-case Validation — 2026-07-13

## 要旨

本稿はKrea2と24 GiB GPUで、4Kを主成果、8Kを承認済み4Kの条件付き2倍拡張として生成するtraining-free手法を示す。高画素化だけでは描写量は増えず、強いimg2imgは顔・角・髪流れ・透明物を壊し得る。そこで、密描写promptで基準像の情報量を先に確保し、halo付きtileから「基準に既存する細部を守るsafe残差」と「平坦域にも限定的な新規描線を許す2–8 px帯域残差」を分離する。後者は2つのshift位相の独立証拠と低分散合意がある画素だけ採用する。

指定prompt・seedの単一事例で2896×4096の4Kと5792×8192の8Kを全tile完了・skip 0で生成し、4096px正規化の多帯域metric、GPU telemetry、固定7地点の1024×1024原寸crop、SHA-256を保存した。主張は実装可能性と本1事例に限り、一般的な比較優位や意味的一致の保証ではない。

**キーワード:** Krea2、4K/8K、VRAM制約、band-pass residual、phase consensus、100% crop

## 1. 目的と品質境界

目的は、顔・瞳・角・髪流れ・スライム形状を保ちながら、zoom時に読める髪束、虹彩、角の面、レース、縫い目、透明材質の内部表現を増やすことである。入力tagは次を文字列prefixとして厳密に保持した。

    light blue hair,long_wavy_hair,devil’s_horn,purple horn,purple_eyes,green_slime,jig eyes,smile,jitome,Expressionless,

処理は次の品質境界を持つ。

1. Krea2 native段で髪束、虹彩、角、黒いgothic衣装、緑slimeを明示し、描写密度の高い全体基準像を作る。
2. 4Kを第一の納品点として生成し、顔、瞳、角数、髪流れ、衣装、slime、tile境界を原寸確認する。
3. 8Kは長辺3840–4096の承認済み4Kだけを受け付け、各辺を正確に2倍する。
4. 全tile、skip 0、寸法、細部保持の下限・上限、Smart Finish、平坦部、clippingをfail-closedに検査する。
5. 数値metricを人手QAの代用にしない。

## 2. 提案アルゴリズム

### 2.1 VRAM-Canvas

全画面をGPUへ載せず、1024角を上限とする局所payloadだけをKrea2 img2imgへ渡す。tile中央をcore、周辺をhaloとし、core間はsmoothstepで重ねる。GPU側のcanvas依存空間項を \(O(T^2)\) に制限し、全画面の一次・二次momentはCPU上のdisk memmapへ蓄積する。

段階 \(k\) の基準cropを \(B_i\)、Krea2候補を \(R_i\)、低域を \(L\)、高域を \(H=I-L\)、2–8 px帯域を \(BP_{2-8}\) とする。低周波構造整合度は

\[
a_i=\exp\left(-\frac{|Y(LR_i)-Y(LB_i)|}{\tau_s}\right)
\]

である。

### 2.2 基準細部を守るsafe枝

基準高域energy \(e_{Bi}\) からgateを作り、元から存在する局所細部だけを保存・強調する。

\[
\Delta_i^{\mathrm{safe}}=
\operatorname{clip}\left(
\gamma a_i\frac{e_{Bi}}{e_{Bi}+\tau_b}[H(R_i)-H(B_i)],
-d_s,d_s\right).
\]

生成tileそのものを貼らず高周波差だけを候補にするため、大域形状や低周波色調は直接出力へ入らない。この枝だけでは平坦な髪面・衣装面へ新しい描線を増やしにくい。

### 2.3 bounded novel-detail枝

safe枝の相補gate \(c_i=\tau_b/(e_{Bi}+\tau_b)\) を使い、基準細部が疎な場所に限って、候補が提案する輝度帯域差を取り出す。

\[
\Delta_i^{\mathrm{novel}}=
\operatorname{clip}\left(
\eta a_i c_i[BP_{2-8}(R_i)-BP_{2-8}(B_i)],
-d_n,d_n\right).
\]

色相、silhouette、大きな顔構造を直接描き直さないよう輝度だけを変更し、4Kでは \(d_n=8\)、8Kでは \(d_n=6\) intensity levelsに制限する。novel枝を有効にする場合は2 phasesを必須とする。

### 2.4 独立位相consensus

各shift位相のsmoothstep重みを個別に正規化し、端部のoverlap数で統計的質量が増えないようにする。全候補画像を保持せず、

\[
S_0=\sum_iw_i,\quad
S_1=\sum_iw_i\Delta_i,\quad
S_2=\sum_iw_i\frac{\|\Delta_i\|^2}{3}
\]

だけを蓄積する。

\[
\mu=S_1/S_0,\quad
E_2=S_2/S_0,\quad
V=\max(E_2-\|\mu\|^2/3,0)
\]

\[
g_{\mathrm{con}}=\exp\left(-\lambda\frac{V}{E_2+\kappa^2}\right).
\]

safe枝はこのconsensusで合成し、novel枝はさらに \(S_0\ge1.5\)、すなわち2つの独立shift位相から実質的な証拠がある画素だけ採用する。一致する微細線は残し、位相間で符号・位置が揺れるtextureを減衰する。ただし、全候補が同じ誤りを出す場合は検出できない。

### 2.5 Smart Finishと品質gate

Smart Finishは既存の輝度高域にtexture energy、structure-tensor coherence、強輪郭・暗部・白飛びguardを掛ける。意図した紫の角・瞳と緑slimeを守るため、Dense Detail workflowの色補正は既定0とする。detail候補は次を満たす場合だけ画像へ適用し、該当細部がなければbit-identical no-opを合格扱いにする。

- 方向整合領域の細部energy比が1.002以上。
- 平坦部の変更画素が0。
- clipping channel率が \(5\times10^{-4}\) 以下。

最終gateは1536px共通尺度のgradient/high-pass保持率に下限と1.8倍の上限を設け、情報消失だけでなくnoise・oversharpeningによる異常増加も拒否する。加えて長辺4096へ正規化した4周波数帯を記録する。

## 3. Forge Neo統合

img2imgのVRAM-Canvas GUIへ次を統合した。

- 4K/8K Smart buttonとprofile dropdownの即時反映。
- Structure Safe、Dense Detail 4K、Dense Detail 8K profile。
- novel detail gain / maximum delta。
- Krea2 checkpoint検査とper-run checkpoint/VAE override拒否。
- Krea2 dense-detail promptの冪等追加。
- 8Kは承認済み4Kの正確な2倍だけを許可。
- seed -1を実seedへ解決し、終了・例外時はsentinelを含む元状態へ復元。
- tile中のRestore faces / Tiling強制OFFと復元。
- 中断時に未完成tileを返さない。
- 必要memmap容量を全段合計し、開始前にdisk空き容量を検査。
- PNG info / JSON manifestへprompt、seed、stage、tile、consensus、Smart Finishを保存。

| profile | phase | steps | denoise | novel gain / cap |
|---|---:|---:|---:|---:|
| Structure Safe | 1 | 2–4 | 0.12→0.08 | 0 / — |
| Dense Detail 4K | 2 | 3–4 | 0.16→0.13 | 1.0 / ±8 |
| Dense Detail 8K | 2 | 3–4 | 0.12→0.11 | 0.8 / ±6 |

## 4. 単一事例評価

### 4.1 条件

- GPU: NVIDIA GeForce RTX 3090 24 GiB。
- checkpoint: turbo_gpt0630_krea2_final_forge_bnb_nf4.safetensors。
- seed: 3883506083。
- native基準像: 1024×1448。
- 4K: 2896×4096、tile 1280、2 phases。
- 8K: 5792×8192、tile 1024、2 phases。
- Smart Chroma Strength: 0。
- 内蔵ブラウザ・外部web検索: 不使用。

### 4.2 実行結果

| 段 | 寸法 | tile | skipped | wall time | VRAM |
|---|---:|---:|---:|---:|---:|
| 4K | 2896×4096 | 62 / 62 | 0 | 622.2 s | 連続計測なし |
| 8K | 5792×8192 | 225 / 225 | 0 | 1395.6 s | 22,722 MiB* |

\* 1秒間隔のGPU全体pollingで観測した最大値であり、Krea2 process単独値または真の瞬間peakを保証しない。1271 samplesで、GPU利用率最大100%、温度最大88 ℃だった。

### 4.3 品質gate

| metric | 4K | 8K |
|---|---:|---:|
| gradient p95 retention | 0.760 | 0.957 |
| high-pass p95 retention | 0.716 | 0.902 |
| 4096基準 sigma 0–1 p95 ratio | — | 0.826 |
| Smart Finish detail energy | 1.234× | 1.122× |
| flat-region changed pixels | 0 | 0 |
| clipped channel fraction | 0.000000 | 0.000000 |
| novel evidence coverage | 100.0% | 100.0% |
| mean novel consensus | 0.685（4K最終段） | 0.862 |
| mean accepted novel residual | 0.149 level（4K最終段） | 0.107 level |

8Kのgradient / high-pass p95は設定下限0.88を超え、1.8倍のnoise上限未満だった。4096基準の最細帯域比0.826は機械gateには使わず、8Kを4K表示尺度へ縮小したとき最細線energyが増えたとは主張しない。8K原寸では各辺2倍のsampling密度を持ち、対応する100% cropと全体像を目視した。

固定7地点の1024×1024原寸cropを4K/8Kで同一正規化中心から保存した。8K cropは4K cropの半分の画角である。顔、両目、角数、髪の大流れ、レース・縫い目、緑slimeの外形・透明感を維持し、明瞭なtile seam、二重輪郭、人物・角の増殖を認めなかった。ただし本1事例の人手観察である。

- 4K SHA-256: B303B89DEEB1C2FBD1FAEF72156465341F75177498F848BDB0D4D86F65048545
- 8K SHA-256: 8BAE32FA51D4C08C949121EB216AB59D8806EA14C4B50A17A7AA32E23759BD6B

## 5. 限界

評価は単一prompt・seed・portraitであり、他の画風、手指、文字、写実顔、一般的な比較優位を示さない。高周波metricは知覚品質や意味的一致の代理ではなく、noiseでも上昇し得る。consensusは共有誤りを検出できず、基準像の誤構造は高解像度でも残る。8Kは4Kより時間・熱・disk I/Oが大きい。観測温度88 ℃も運用上の熱負荷として残るため、4Kを既定の納品点とし、8Kは必要時のみ選ぶ。

## 6. 結論

密描写をnative段で確保し、base-anchored safe枝とbounded novel枝を分離し、2位相合意で後者をfail-closedに採用する手法をForge Neoへ実装した。指定条件で4Kと8Kを全tile完走し、数値gate、実測telemetry、原寸crop、hashを一体で保存した。

## PDF再構築

完走した4K/8K manifestと原寸cropを入力し、JIS B5（182×257 mm）2ページを生成する。ビルダーはstatus、4K入力SHA一致、tile完了、telemetry sample、ページ数、ページ寸法を検査する。

    .\venv\Scripts\python.exe .\tools\build_vram_canvas_paper.py --run-manifest '.\output\krea2_smart_dense_v2_8k\smart8k_20260713_052614_591407\smart8k_manifest.json' --preflight-manifest '.\output\krea2_smart_dense_v2\smart8k_20260713_045514_972374\smart8k_manifest.json'

## 参考文献

1. O. Bar-Tal et al., “MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation,” ICML, 2023.
2. R. Du et al., “DemoFusion: Democratising High-Resolution Image Generation With No $$$,” CVPR, 2024.
3. R. Rombach et al., “High-Resolution Image Synthesis with Latent Diffusion Models,” CVPR, 2022.
4. S. Lee et al., [“Krea 2 Technical Report,”](https://www.krea.ai/blog/krea-2-technical-report) Krea, 2026.
