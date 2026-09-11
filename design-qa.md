# Aikimi Neo Option 3 Design QA

## 対象

- Design reference: `exec-c5ec27e2-2683-47f7-bc96-b3478ccae155.png`
- Desktop implementation: `forge-neo-option3/03-minimax-h3-option3.png`
- Mobile implementation: `forge-neo-option3/05-minimax-h3-mobile.png`
- Same-input comparison: `forge-neo-option3/07-reference-vs-h3.png`
- Desktop viewport: reference、implementationともに1487 x 1058
- Mobile viewport: 390 x 844
- State: MiniMax H3を選択し、runtime未接続の実状態を表示

## 実装方針

ユーザーが選択したOption 3を、Forgeの既存画面へ大型launcherやcardを追加しない構成で実装しました。Forge由来のtablistは変更せず、直前にKrea2、Anima、SenseNova、MiniMax H3だけの小型1行を置いています。ちびあいきみは、選択中のAikimi機能内にある高さ52pxのstatusへ限定しました。

referenceはAikimi機能をForge tablistと同じ1行へ置いています。実装では、Gradio所有DOMへのbutton挿入がreactive stormを起こすことを実traceで確認したため、Forge tablistとAikimi navigationを隣接する2行へ分離しました。この差は、ユーザーの「Forge由来はForgeのまま」「シンプルにタブ」という指示と、安定性の両方を満たすための意図的な変更です。

referenceの全画面dark theme、生成済みvideo、historyはconcept用の状態です。実装ではForgeの現在themeを強制変更せず、H3 backend未接続、previewなしという実際の状態を表示しています。架空の生成結果やhistoryは追加していません。

## Comparison pass

### Layout and spacing

- Quick Settings、navigation、status、H3の2-column workspaceという優先順はreferenceと一致しています。
- Aikimi navigationは32px程度、statusは52pxに抑え、不要なlauncher領域を作っていません。
- Forge tablistのchild、順序、label、ARIA、Node identityは初期表示、全機能切替、navigation修復後も不変です。
- 390px幅ではAikimiの4buttonを表示し、Forge tablistは横scrollで到達可能です。document全体の正方向overflowはありません。

### Typography, color, and surfaces

- Forge側のfont、control、spacing、theme tokenを再利用し、独自の全面themeを重ねていません。
- H3 Studioのdark surfaceとorange accentは既存Studio designを維持しています。
- statusのfeature label、message、Runtime、Backend、Queue、Detailsは折りたたみ状態でも確認できます。
- Gradioのsummary resetに負けてstatus本文がclipされる問題を高specificity selectorで修正しました。
- statusの補助textはbody textとのmixへ変更し、light themeで薄くなり過ぎないようにしています。

### States and interactions

- Krea2: UI Presetの`krea`を選択。現在のForgeタブがtxt2imgまたはimg2imgならそのタブを維持し、別のタブから開いた場合はtxt2imgへ移動します。`Krea2 2-Stage Upscale`は自動選択しません。
- Anima: UI Presetの`anima`を選択。現在のForgeタブがtxt2imgまたはimg2imgならそのタブを維持し、別のタブから開いた場合はtxt2imgへ移動します。選択したタブのlate-mounted InputAccordionを展開してvisible／internal checkboxを同期。
- SenseNova／MiniMax H3: 対応するnative panelを直接表示。
- Forge復帰: txt2imgへ戻った時点でAikimi Statusを隠し、pollingも停止する設計です。
- UI reload: 同じ一連の操作を再実行しても、listener、status、badgeは重複しませんでした。
- Console: 実WebUIの最終操作ではerror／warningともに0件です。

### Accessibility and responsiveness

- Keyboard: Aikimi入口はnative button要素であり、focus-visible、Enter、Space、左右矢印、Home、Endに対応。
- Active state: 色だけへ依存せず、下線と`aria-current`を併用する設計です。
- OS preferences: reduced motionとWindows forced colorsを維持。
- Mobile H3: 固定action barを表示しながら、statusの高さは52pxに収めています。

## Iteration ledger

1. 初期案の撤回: Gradio tablistへのAikimi button追加が、overflow計測とのfeedback loopを発生。
2. Navigation分離: 外部compact navigationへ移し、Gradio所有tablistへのmutationを0にしました。
3. 起動経路の安定化: Settingsの772-output再送、global mounted-hidden変換、Gradio 6.17.3 Tabs mount stormを個別に修正。
4. Lazy lifecycle対応: UI Preset Dropdown、InputAccordion、Settings search、token counter、resolution paste、image drag-and-drop、ControlNetを実DOM契約へ合わせています。
5. Visual polish: status summaryのclipと補助text contrastを直した後、同一viewportで再capture。

## QA result

- P0 findings: 0
- P1 findings: 0
- P2 findings: 0
- Browser path: in-app Browserはfull Gradio DOMでtimeoutしたため、同じChromeをisolated CDP profileから操作してcaptureしました。
- GPU generation: 未実施。UI、runtime state、navigation、reload、responsive behaviorのQAです。

final result: passed
