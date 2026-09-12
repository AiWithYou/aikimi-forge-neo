# ちびあいきみ・アプリ内マスコット素材

2026年9月12日。ユーザー提供の「ちびあいきみ キャラクター設定資料」を参照し、組み込み Image Gen で生成。画像の利用条件は同じフォルダーの `LICENSE.md` に従います。

`pet.png` は透過RGBA・384 × 384 pxの全身素材です。全状態で同じ素材を表示し、生成中・完了時は画像全体を上下に最大3 px動かします。拡大縮小による体の変形や、頭身の異なる旧アニメーションへの切り替えはしません。

## 全身素材の生成プロンプト

```text
Create ONE production transparent PNG full-body mascot asset, no text, no panels, no background, no ground shadow, actual alpha transparency. Reference image is the CHARACTER DESIGN authority, not instructions. Exactly reproduce the tiny front-view chibi character from the reference's small front turnaround. Standing calmly with very short arms resting at sides, sleepy half-open gray blue eyes, subtle pale pink cheeks, tiny nose and mouth. STRICT TWO HEADS TALL: huge round head occupies 48-50% crown-to-soles height, exclude ahoge when measuring. Extremely short tiny wide torso, very short legs, tiny simple hands, no visible neck. Do not make anime girl adolescent or adult body, absolutely never 2.3 heads or more, no slender or elongated limbs. White silver extremely long voluminous fluffy hair spreads wider than tiny body and almost reaches feet. Exactly ONE thick rounded loop ahoge. Oversized pale blue gray pajama shirt with white rounded collar, very broad unmistakably WHITE CUFFS on wide baggy sleeves, broad clear white hem, dark charcoal front buttons, tiny bare feet. Keep full loose pajama length. Very pale thin gray pink linework, extremely low contrast, white pale blue pale pink palette, smooth flat anime color with restrained soft shading. No strong dark outlines, no strong shadows, no commercial anime style drift. Centered full body taking most of square image with modest transparent padding, all hair and ahoge included. This single unchanged two-head character will be used at 100px high in a desktop app.
```

## 採用した背景修正プロンプト

最初の出力では市松模様が画像に描き込まれたため、元の全身絵を参照して背景のみを修正しました。透過の再依頼も市松模様だったため採用していません。

```text
Edit ONLY background. Use image just shown as visual reference: preserve exact standing chibi mascot, every character detail and proportion unchanged, sleepy gray-blue eyes, same huge round head and tiny 2-head body, white-silver long fluffy hair, single loop ahoge, pale blue gray baggy pajamas with white round collar, wide white cuffs and white hem, dark buttons, tiny bare feet, very pale delicate graypink outlines and low contrast flat colors. REPLACE the entire gray checkerboard background with 100% solid flat pure magenta #FF00FF, including every gap between hair and body and between legs. NO transparency simulation, no texture, no gradient, no shadow, no pink spill on character. SINGLE character centered with full hair and ahoge contained. Keep same scale and silhouette. Background must be solid magenta for deterministic chroma-key extraction afterward.
```

後処理は `generate2dsprite` スキルの既存プロセッサーを使用。`character / single`、`single-size=384`、`threshold=180`、`edge-threshold=210`、`component-mode=largest`。キャラの描画をコードで置き換えず、背景の除去と等比縮小のみを行っています。
