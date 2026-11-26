# Japanese (ja) glossary review

Reviewed 2026-11-26 against `glossary/terms/ja.jsonl` (1,184 entries: 217
`verified_d2l`, 116 `verified_external`, 851 `proposed`). Every entry was read
against the sense in `concepts.jsonl`, not a sample. Corrections are in
`ja-corrections.jsonl` (188 lines). The conventions of `ja/style.md` are
followed where they are provisional (パラメータ-style long vowels, 訓練 for
"train", アテンション) and marked as pending in the affected lines.

## Verdict

The glossary is usable as a base, but its status labels cannot be trusted:
`verified` means "found in a source", and two of the three source families are
machine-translated. The verified entries are worse than the proposals.

| Status | Entries | To change | Of which not minor | Wrong sense or calque |
|---|---|---|---|---|
| verified_d2l | 217 | 58 (27%) | 49 | 21 |
| verified_external | 116 | 24 (21%) | 20 | 7 |
| proposed | 851 | 106 (12%) | 89 | 11 |
| all | 1,184 | 188 (16%) | 158 | 39 |

By severity: 14 wrong_sense, 25 calque_or_mt, 43 non_standard, 76 inconsistent,
30 minor. Of the 125 proposed entries with frequency ≥ 50, 28 change (11 of
them are names carrying an invented gloss). The verified entries to change
cover about 17,400 of the 81,200 counted occurrences of verified concepts
(21%), because the errors sit on head words: training, layer, variance,
kernel, action, feature, Adam, multilayer perceptron, SGD.

All 31 corrections of `ja/style.md` §7.2 are confirmed and included. One
caveat: 跡 for trace is the ja.wikipedia article title, so it is a legitimate
term; トレース is preferred for ML usage and is rated minor. The review adds
157 further corrections; the worst in verified entries that §7.2 missed:

- preconditioning → 学習 ("learning"; should be 前処理)
- network in network → ネットワーク (truncated to "network")
- condition number → 条件番号 ("serial number"; 条件数)
- region proposal network → 地域提案ネットワーク (地域 is a geographic region; 領域)
- dynamic programming → 動的プログラミング (動的計画法)
- matrix factorization → 行列因数分解 (因数分解 is for polynomials; 行列因子分解)
- change of variables → 変数の変更 (変数変換)
- bounding box → 境界ボックス (バウンディングボックス), intercept → インターセプト (切片)
- false positive → 誤検出 beside 偽陰性, 真陽性 (偽陽性)
- binary classification → バイナリ分類 (二値分類), projection → 投影 (射影)
- Jacobian → ヤコビアン (in Japanese usually the determinant; ヤコビ行列)

In proposed entries: GPU → グラフィックス処理装置 (frequency 1,195), forward/reverse
mode → 順伝播モード/逆伝播モード (conflated with the forward and backward pass),
soft maximum → ソフトマックス近似 (collides with softmax), Markov decision
process → マルコフ決定プロセス.

## Provenance

- **Earlier Japanese D2L edition** (`d2l-ai/d2l-ja`, commit f84fb22): machine
  translation. `config.ini` at that commit has `translator = aws` (checked
  directly), and the prose shows it (足を濡らすために for "to get our feet wet",
  グラデーション for gradient, 翻訳不変性). All 217 `verified_d2l` entries rest
  on it. The error rate depends on how the term was extracted:

  | Extraction (`notes`) | Entries | Not-minor errors |
  |---|---|---|
  | aligned_emphasis (italic term on the paired line) | 91 | 31 (34%) |
  | aligned_heading | 32 | 12 (38%) |
  | manual_body | 16 | 2 |
  | manual_context | 78 | 4 (5%) |

  Automatic alignment adds its own errors to the machine translation
  (preconditioning → 学習 and network in network → ネットワーク picked the
  wrong emphasised word). Treat `aligned_emphasis` and `aligned_heading`
  entries as proposals.
- **Google ML Glossary, `hl=ja`** (36 entries): the pages declare machine
  translation (`x-mtfrom-en`, noted in `sources.jsonl`). 11 of 36 are wrong:
  レイヤ, 一般化, アクション, バイナリ分類, 遅延, 収縮, グラウンド トゥルース,
  トレーニングの損失, エンベディング レイヤ, ワンホット エンコード, コンテキスト ウィンドウ.
- **Stanford CS229/CS230 cheatsheets in Japanese** (77 entries): volunteer
  human translation, the best of the three, 9 errors that are not minor (バリアンス, 和集合上界,
  確率標本, 文章の長さの正規化, 交差エントロピー誤差, 学習誤差, ビーム検索,
  Bleuスコア, 時間軸での誤差逆伝播法).
- **Proposals** (851): editorial, no sources. Mostly sound kanji terminology
  for mathematics, statistics and optimization; the errors are in naming
  conventions and consistency, not in sense.

No verified entry cites a Japanese textbook, a university course or
ja.wikipedia. The corrections cite ja.wikipedia article titles, redirects and
article text (checked through the MediaWiki API on 2026-11-26); 66 corrections
are `verified_external` on that basis, 122 stay `proposed`.

## Systematic problems

1. **One English head word, several renderings.**
   - training: 学習 / 訓練 / トレーニング; and 学習 is also "learning", so
     training-dynamics and learning-dynamics are both 学習ダイナミクス.
   - layer: レイヤ / レイヤー / 層. loss: 損失 / ロス / 誤差.
   - recurrent: リカレント / 回帰 / 再帰 (回帰 is also regression).
   - context: コンテキスト (window, in-context learning) / 文脈 (length, word,
     long-context, bidirectional, contextual bandit).
   - consistency: 整合性 / 一貫性. contrastive: 対照 / 対比的. forward
     (diffusion): 順方向 / 順. search: 検索 / 探索 / サーチ. -wise: ペアワイズ /
     深さ方向 / 点ごとの. expert: 専門家 / エキスパート.
   - gradient is consistent (勾配) once the two グラデーション entries are fixed.
2. **Sense collisions.** Correctly separated: bias / statistical bias, kernel
   (convolution) / kernel function, value (attention) / value function,
   bootstrap. Broken: forward mode vs forward pass, soft maximum vs softmax,
   weight tying vs weight sharing (same rendering), bootstrapping (RL) vs the
   bootstrap, 拡張 for both augmentation and dilation, 有効 for both effective
   and active, 前処理 for both preconditioning and preprocessing (unavoidable;
   note it). The concepts list has no entry for score (metric), sample,
   example, loss, cost, temperature or average; `style.md` §7.1 covers them,
   and its table agrees with this review.
3. **Names given a gloss or transliterated.** About 30 entries render a model,
   optimizer or acronym as katakana (アダム, アダグラッド, トランスフォーマー) or
   append an explanation that would be inserted at every occurrence
   (AdamW最適化アルゴリズム, Mamba選択的状態空間モデル, BERTモデル,
   Reflowによる軌道の再学習). DDPM, DDIM, RMSNorm and ALiBi were expanded into
   coined Japanese. All should be the Latin name alone.
4. **Script and spacing.** Spaces inside katakana compounds from the Google
   pages; エンコーダ-デコーダ with an ASCII hyphen; long-vowel exceptions
   (コンピュータービジョン, ルーター, データローダー); kanji and Roman numerals
   mixed (第1種 / 第II種). About 20 entries join a Latin name to Japanese
   without a space (KVキャッシュ, He初期化, Transformerブロック, Q学習, VC次元,
   p値, F1スコア) against `style.md` §3.2 (KV キャッシュ). These are not listed
   as corrections: normalise them mechanically once the spacing rule (§9.1) is
   decided.
5. **Wrong alternatives.** `batch-normalization` lists 正規化 as an
   alternative, `correlation` lists 相関関係, `vector` lists ベクター. If
   alternatives are used for matching, review them too.

## Decisions for the author or a native reviewer

Author (conventions):

1. 訓練 for training (12 lines depend on it), アテンション and 自己アテンション,
   パラメータ-style long vowels: all followed here, all pending.
2. context: コンテキスト for the LLM sense and 文脈 for word context, or one
   word throughout. ja.wikipedia attests コンテキスト内学習 and
   コンテキストウィンドウ; 文脈内学習 is common in Japanese LLM books.
3. generator / discriminator: 生成器 / 識別器 (proposed here) or
   ジェネレータ / ディスクリミネータ; not the present mix.
4. Whether named methods with a usable Japanese form keep it (スキップグラム,
   ラッソ回帰, 残差ネットワーク) or stay in Latin script.
5. Spacing between Latin names and Japanese inside glossary terms (point 4
   above).

Native reviewer (terminology, not confirmed by a source here):

1. underfitting: 学習不足 (machine-translated Google heading, kept by the style
   guide) against 未学習 or 過少適合 (ja.wikipedia).
2. off-policy / on-policy: オフポリシー (ja.wikipedia) or 方策オフ型 /
   方策オン型, which matches 方策 elsewhere.
3. translation invariance: 平行移動不変性 or 並進不変性.
4. statistical bias: 統計的バイアス or 偏り; rank: ランク or 階数; surprisal:
   サプライザル or 自己情報量; outer product: 外積 (also "cross product" in
   Japanese) or 直積.
5. early stopping: 早期停止, 早期終了 or 早期打ち切り; calibration: 較正 or
   キャリブレーション; importance sampling: 重要度サンプリング or 重点サンプリング.
6. Coinages for recent terms with no settled Japanese: 計算律速 / メモリ律速,
   演算密度 (arithmetic intensity), ノイズ球, 誘導ヘッド, 残差ストリーム,
   モード欠落, 最大平均値差異 (MMD), 祖先サンプリング, 短文ペナルティ, and the
   proposals in the corrections marked "native reviewer to confirm" (union
   bound, shrinkage, behavior cloning, weight tying, dilation, consistency
   model, loss landscape, minimizer, Itô calculus, style transfer, random
   sample, fully convolutional network).
7. trajectory is 軌跡 (RL) but 軌道 in 最適化軌道 and for ODE paths; accept as
   two senses or unify.

## Sources

- ja.wikipedia.org article titles, redirects and text, via the MediaWiki API
  (English-to-Japanese language links for about 260 concepts, title and
  redirect checks for about 330 Japanese candidates, text checks on about 40
  articles). Each correction lists the article it relies on.
- `d2l-ai/d2l-ja` `config.ini` at commit f84fb229198b49f2dc532ca72ce72b5611a124cb.
- `glossary/DISCOVERY.md`, `glossary/notes/east-asian.md`,
  `glossary/sources.jsonl`, `ja/style.md`.

Limits: no Japanese textbook was consulted directly (Sutton and Barto, Bishop
and Goodfellow et al. in translation are the references a native reviewer
should use for reinforcement learning, probability and deep learning terms).
Web search returned few usable Japanese teaching pages, so terms without a
ja.wikipedia attestation stay `proposed`.
