# Style guide: Japanese (ja)

Draft of 2026-11-26 for the author's review. Translators read this with every
unit (TRANSLATING.md §7–§8). Each rule has one reason and one example
(English → Japanese). Points that still need the author's ruling are collected
in §9.

Order of authority: the decisions log, then this guide, then the glossary. The
glossary's verified entries are binding except for the corrections in §7.2
(many `verified_d2l` entries come from the earlier machine-translated edition).

## 1. Register

| Where | Rule | Example |
|---|---|---|
| Body text, list items, tab notes, notes in the body | だ・である (常体)． である is the register of graduate monographs (the Japanese Goodfellow et al. and Bishop). | "Linear regression is a single artificial neuron." → 線形回帰は、1 個の人工ニューロンである。 |
| Headings, captions, slides, table cells | Noun phrases (体言止め) or plain form (常体); never です・ます (JTF 1.1.2, 1.1.5). See §6. | "Training the Model" → モデルの訓練 |
| Exercises | Polite requests: 〜してください; questions end in 〜ですか。 See §6. | "Show that …" → …を示してください。 |

**"We" and "you".** Japanese drops the subject. Do not write 私たちは or あなたは
for the book's "we" and "you".

| English | Japanese | Example |
|---|---|---|
| we + action in the text | no subject, 〜します | "We minimize the loss with SGD." → 損失は SGD で最小化します。 |
| "let us", "let's", "we now turn to" | 〜しましょう / 〜してみましょう (only where the English invites; not every sentence) | "Let us look at the gradient." → 勾配を見てみましょう。 |
| "in this section/chapter/book, we …" | 本節では / 本章では / 本書では + verb | "In this section, we formulate the model." → 本節では、モデルを定式化します。 |
| "our model", "our dataset" | この / その, or nothing | "our model overfits" → このモデルは過学習しています |
| "you can", "suppose you want" | no subject; 〜できます / 〜とします | "Suppose you want to estimate the price of a house." → 住宅の価格を見積もりたいとします。 |
| "we recommend", "we believe" (the authors' view) | 本書では〜 / 〜と考えられます | "We recommend AdamW." → 本書では AdamW を勧めます。 |
| "we" = people, the field | 私たち, 人間, 研究者 (here it is real content) | "We humans attend selectively." → 私たち人間は、選択的に注意を向けます。 |

Vary sentence endings (〜です, 〜ます, 〜でしょう, 〜ません, 〜からです) so that a
paragraph is not a run of 〜ます。 Do not mix in だ・である at sentence ends.

## 2. Punctuation and spacing

### 2.1 Punctuation

| Item | Rule | Example |
|---|---|---|
| Comma, full stop | Full-width 、 and 。; never ，． or half-width `,` `.` as Japanese punctuation (JTF 1.2.1). | "Here $b$ is the bias." → ここで、$b$ はバイアスです。 |
| Lists in a sentence | 、 between items, also between Latin items; no "and" comma. | "Adam, RMSProp, and SGD" → Adam、RMSProp、SGD |
| Parentheses | Full-width （） around Japanese text, an English gloss, or a reference; no space inside or outside (JTF 2.3.3, 3.3.1). Half-width `()` only where they belong to math, code, a Latin expression, or a panel marker `(a)`. | "(the workhorse optimizer of deep learning)" → （深層学習で最もよく使われるオプティマイザ） |
| Colon | Full-width ： only after a label (a bold lead-in, 例：); no space after it. A colon that introduces a clause, list or equation becomes a sentence ending (JTF 3.2.7). | "**Core assumption:** …" → **中心となる仮定：**… ; "We get the following:" → 次の結果が得られます。 |
| Semicolon, em dash | Not used. Split the sentence, or use 、 or （）. | "the bias—an offset—shifts the plane" → バイアス（オフセット）は平面を平行移動させます |
| Quotation marks | 「」 for quotations, scare quotes and words mentioned as words; 『』 for book titles and for quotes inside 「」. No “ ” in Japanese text. | the "attention" weights → 「アテンション」の重み ; *Deep Learning* (book) → 『Deep Learning』 |
| ? and ! | End body questions with 〜か。 and exclamations with 。 (JTF 3.2.1–3.2.2). ？ is allowed only in a heading or slide title that is a question. | "Why does this work?" → なぜこれでうまくいくのでしょうか。 |
| Coordinate compounds | ・ joins parallel items inside one term; no slash. | "queries, keys, and values" as one notion → クエリ・キー・値 |
| Emphasis | Keep `*…*` and `**…**` exactly on the corresponding word(s). Do not replace emphasis with 「」 and do not add any. The English gloss and particles stay outside the markers. | "*regression* predicts …" → *回帰*は…を予測します ; `*重み減衰*（weight decay）` |

### 2.2 Spacing: one rule

**Put one half-width space between Japanese characters (kanji, kana) and any
half-width run: a Latin word, a number, inline math `$…$`, inline code, or a
reference role (``:numref:`…` ``, ``:eqref:`…` ``, ``:cite:`…` ``, `§{…}`). Put
no space next to full-width punctuation or brackets (、。「」（）：・).**
Markup characters (`*`, `**`, `[`) do not count: look at the characters they
enclose.

Reasons: (1) it is required by the build: tested with the repository's pandoc,
a reference directly followed or preceded by kana or kanji is not recognised
(``:numref:`sec_sgd`で`` fails; ``:numref:`sec_sgd` で`` and
``（:numref:`sec_sgd`）`` work), and a closing `$` followed by a digit breaks
the math; (2) inline math set tight against kanji is hard to read; (3) one rule
for all half-width runs can be checked mechanically. This departs from JTF
2.3.1.1 (no space), which assumes a typesetter adds the gap.

| English | Japanese |
|---|---|
| "updates $\mathbf{w}$ after every minibatch of 256 examples" | 256 個の例からなるミニバッチごとに $\mathbf{w}$ を更新します |
| "ResNet appeared in 2015." | ResNet は 2015 年に登場しました。 |
| "as shown in :numref:`fig_x`, the loss falls" | :numref:`fig_x` に示すように、損失は減少します |
| "the KV cache (see :numref:`sec_kv`)" | KV キャッシュ（:numref:`sec_kv`）|
| "an accuracy of 92.3%" | 92.3% の正解率 |
| "`nn.Linear` is a *layer*" | `nn.Linear` は*層*です |

- Digits and Latin letters are always half-width (JTF 2.1.8–2.1.9); katakana is
  always full-width.
- Counters and suffixes follow the same rule: 3 つ, 2 番目, 10 エポック, 2 倍, 第 3 章,
  $d$ 次元.
- Do not write 図, 表, 式 or 節 before a reference role: the build prints the
  prefix. "in :numref:`fig_x`" → ``:numref:`fig_x` では``, not ``図 :numref:`fig_x` では``.
- A citation stands at the end of the clause it supports, before the 、 or 。:
  …を発表しました :cite:`Legendre.1805`。

### 2.3 Order of protected tokens

The check requires the protected tokens of a block in the English order, and
Japanese is verb-final. Use particles and topic-fronting to keep the order
before reordering; this is nearly always possible.

"We update $\mathbf{w}$ using the gradient $\mathbf{g}$ and the step size
$\eta$." → $\mathbf{w}$ は、勾配 $\mathbf{g}$ とステップサイズ $\eta$ を使って更新します。

- End the sentence before a display equation and start a new one after it:
  "…is given by $$…$$ where $n$ is…" → …は次の式で与えられます。 $$…$$ ここで、$n$ は…です。
- Math that contains English stays as it is; do not repeat its meaning in
  Japanese. "the $i^{\textrm{th}}$ house" → $i^{\textrm{th}}$ の住宅 (not 「番目」).
- If the English order would give wrong or unreadable Japanese, write correct
  Japanese and report the block in NOTES.

## 3. English terms

### 3.1 The English term in parentheses

Add the English once per page (one numbered section of the book, e.g. all of
"Linear Regression"), at the term's first occurrence in body text, in full-width
parentheses, lower case, as in the glossary. Check the English page in CONTEXT:
if an earlier unit of the page already introduced the term, do not repeat it.

- Gloss terms rendered in kanji or in a mixed form: *汎化*（generalization）,
  *重み減衰*（weight decay）, *方策*（policy）.
- Do not gloss katakana that transliterates the English (ミニバッチ, トークン,
  ドロップアウト), terms left in English, or school vocabulary (関数, 行列, 確率, 平均).
- With an acronym: 多層パーセプトロン（multilayer perceptron、MLP）; afterwards
  write what the English writes (MLP).
- Never in headings, captions, slides, exercises or table cells.

"*stochastic gradient descent* (SGD) updates …" → *確率的勾配降下法*（stochastic
gradient descent、SGD）は…を更新します。 Reason: readers go on to English papers
and documentation; once per page keeps the text clean.

### 3.2 Kanji, katakana or English

| Case | Rule | Examples |
|---|---|---|
| An established kanji term exists | Use it; never a katakana transliteration. | gradient descent → 勾配降下法 (not グラディエントディセント); weight → 重み (not ウェイト); layer → 層; loss → 損失 (not ロス); overfitting → 過学習 |
| Katakana is the established term | Use it; never coin kanji. | ミニバッチ, エポック, ドロップアウト, プーリング, パディング, ストライド, トークン, プロンプト, ファインチューニング, アテンション |
| Proper names of models, architectures, optimizers, libraries, datasets, metrics; acronyms; code | Leave in English. | Transformer, ResNet, BERT, Adam, AdamW, PyTorch, ImageNet, BLEU, ReLU, SGD, MLP, CNN, RNN, LSTM, GPU, KL |
| English name + common noun | English + space + Japanese. | KV キャッシュ, Transformer ブロック, ReLU 関数, Adam オプティマイザ, GPU メモリ, softmax → ソフトマックス (glossary) |

- Katakana compounds are written solid, without spaces or ・:
  ニューラルネットワーク, コンテキストウィンドウ (not コンテキスト ウィンドウ). Reason: this is
  the usage of Japanese ML books; JTF 2.1.7 leaves the choice to the document.
- Verbs: noun + する (正則化する, サンプリングする, 訓練する), not 〜を行う.
- Kana for function words (JTF 2.2.1): こと, とき, ため, もの, できる, たとえば, すなわち,
  および, または, さらに, ただし, なお, すべて, さまざま.

### 3.3 Long-vowel mark

**Drop the final ー of loanwords from English *-er*, *-or*, *-ar* and of the
*-y* words listed here: パラメータ, ハイパーパラメータ, エンコーダ, デコーダ, オプティマイザ,
トークナイザ, スケジューラ, ジェネレータ, フィルタ, コンピュータ, プロセッサ, アダプタ, バッファ, メモリ,
カテゴリ, ライブラリ, レイテンシ. Keep it in キー, エラー, コピー, スカラー, ユーザー, サーバー,
エントロピー, エネルギー.**

Reason: this is the spelling of the Japanese ML literature (パラメータ, レイヤ in
ゼロから作るDeep Learning; エンコーダ, デコーダ, パラメータ in 大規模言語モデル入門) and of
most glossary entries. JTF 2.1.6 prefers パラメーター but accepts field usage and
asks for consistency within a document. Respell glossary entries that differ
(コンピュータービジョン → コンピュータビジョン, フィルター → フィルタ).

## 4. Numbers and units

| Item | Rule | Example |
|---|---|---|
| Digits | Half-width Arabic digits for every quantity and ordinal, also where English spells the number out (JTF 2.2.2). Kanji numerals only in set words: 一方, 一般に, 一部, 唯一, 二乗, 二次関数, 一様分布. | "two hidden layers" → 2 つの隠れ層 ; "the third epoch" → 3 エポック目 |
| Decimal and thousands | `0.5` and `10,000`, as in the source (JTF 2.1.10). | "10,000 images" → 10,000 枚の画像 |
| Number words | 万, 億, 兆 for English number words; no comma with them. | "3 million parameters" → 300 万個のパラメータ ; "1.5 billion tokens" → 15 億トークン |
| Percent | Half-width `%`, attached to the number. | 92.3% |
| Units | Symbols as in the source, with the source's space; never katakana unit names. Time words written out in English become 秒, 分, 時間, 日, 年. | 24 GiB, 5 ms ; "3 hours" → 3 時間 |
| Ranges | 〜 (U+301C) without spaces, unit once at the end; not `-`, `–` or `~` (JTF 3.2.5). Math keeps its own notation. | "40–55" → 40〜55 ; "45–50%" → 45〜50% ; "2017–2024" → 2017〜2024 年 |
| Multipliers | 倍 in running prose; keep `×` in slides, tables, captions and math. | "about 2× faster" → 約 2 倍高速 ; slide: 2× 高速 |
| Approximation | 約 for "about", "roughly", "~"; "an order of magnitude" → 1 桁. | "roughly 90%" → 約 90% |

Quantities, signs and orders of magnitude stay exact; do not round or convert
units or currencies.

## 5. Names

- **Models, libraries, datasets, companies, hardware, benchmarks:** Latin
  script, exactly as in English. "Fashion-MNIST", "RTX 4090", "OpenAI".
- **People:** Latin script, no honorific (no 氏, 博士). "Gauss grounded least
  squares in probability theory." → Gauss は、最小二乗法を確率論で基礎づけました。 Reason:
  citations and searches use the Latin form, and transliterations of less-known
  names are unreliable.
- **Exception: terms named after people** follow the glossary, in katakana:
  ガウス分布, ベイズの定理, ヤコビアン, ヘッセ行列, マルコフ連鎖, ラグランジュ乗数, ランジュバン動力学.
- **The book:** *Dive into Deep Learning* stays in English; "this book" → 本書.
- **Languages and places:** Japanese (英語, 米国); programming languages stay
  (Python).
- **Structure words:** chapter 章, section 節, subsection 項, appendix 付録,
  Summary → まとめ, Exercises → 演習問題, Discussions → ディスカッション; Proposition 命題,
  Theorem 定理, Lemma 補題, Corollary 系, Definition 定義, Proof 証明.

## 6. Headings, captions, slides, exercises, notes

| Kind | Rule | Example |
|---|---|---|
| `heading` | Noun phrase, no 。, no English gloss. English "-ing + object" becomes object + の + noun. A question keeps its question form. | "The Model and the Loss" → モデルと損失 ; "Training the Model" → モデルの訓練 ; "Why Depth Helps" → なぜ深さが有効か |
| `caption` | Plain form or noun phrase, not です・ます. End with 。 exactly when the English caption ends with a full stop. Keep `(a)`, `(b)`. | "Masks compose by logical AND." → マスクは論理積で合成される。 |
| `slide_title`, `kicker` | Noun phrase, as short as the English; no 。 | `title="The Squared Error Loss Function"` → `title="二乗誤差損失関数"` ; `[Measuring Misfit]{.kicker}` → `[当てはまりの悪さを測る]{.kicker}` |
| `slide_text` | Plain form or 体言止め, one line where the English is one line; no 。 after a single-sentence bullet. Bold lead-ins keep their bold and take ：. | "**Absorbing the bias:** Appending a ones column reduces the affine map to a single matrix product." → **バイアスの吸収：**1 の列を追加すると、アフィン写像は 1 つの行列積になる |
| `exercise` | 〜してください; questions 〜ですか。 / なぜですか。 A bold title stays bold. Reason: 〜せよ and 〜しなさい belong to である-style exam books and clash with the body register. | "Show that the errors sum to zero. Why?" → 誤差の和が 0 になることを示してください。それはなぜですか。 |
| `tab_note` | Body register. Describe this framework only; never compare frameworks. API names stay in code. | "PyTorch's `nn.LazyLinear(1)` requires only the output dimension." → PyTorch の `nn.LazyLinear(1)` に必要なのは、出力次元だけです。 |
| `note` | The register of its surroundings: plain form inside a slide, です・ます in the body. Labels: **Answers:** → **解答：**, **Note:** → **補足：** | "**Answers:**" → **解答：** |
| `table_row` | Noun phrases; numbers and units as in §4. | "Training time" → 訓練時間 |

The cover kicker `[Dive into Deep Learning · §2.1]{.kicker}` is rewritten by the
build; copy it unchanged.

## 7. Known pitfalls

### 7.1 Words with more than one sense

| English | Sense → Japanese | Not |
|---|---|---|
| bias | parameter $b$ → バイアス; of an estimator → 統計的バイアス (then バイアス); inductive bias → 帰納バイアス | 偏見 (only for social bias) |
| kernel | convolution → 畳み込みカーネル, カーネル; covariance / kernel method → カーネル関数; GPU or OS code → カーネル; of a linear map → 核 (null space → 零空間) | 核 for the first three |
| score | $\nabla \log p$ → スコア関数; attention → アテンションスコア; evaluation → スコア | 得点, 楽譜 |
| return | RL → 収益; of a function → 戻り値, 返す | リターン for both |
| value | RL → 価値 (価値関数); attention → 値; a number → 値 | バリュー |
| action, policy, reward | RL → 行動, 方策, 報酬 | アクション, ポリシー, 褒美 |
| sample | a draw from a distribution → サンプル, サンプリングする; statistics → 標本 (標本平均) | 見本 |
| example | a data point → 例, 訓練例 (データ例 where 例 could mean "for example") | 実例 |
| feature | an input variable → 特徴量; in compounds and for learned features → 特徴 (特徴マップ, 特徴ベクトル) | 機能, 特長 |
| embedding | 埋め込み | エンベディング, 埋込 |
| attention | アテンション (自己アテンション, クロスアテンション, マルチヘッドアテンション). Reason: 注意 already means "note, caution" in this book's prose. | 注意, 注目 |
| token | トークン | 字句 |
| logit | ロジット | ログ, 対数 |
| temperature | 温度 | 気温, 体温 |
| loss, cost, objective, error, risk | Five words, five terms: 損失, コスト, 目的関数, 誤差, リスク. Never merge them. | 紛失, 費用 |
| generalization | ML → 汎化 (汎化誤差, 汎化ギャップ); of a formula or model class → 一般化 (一般化線形モデル) | 一般化 for the ML sense |
| overfitting, underfitting | 過学習, 学習不足 | オーバーフィッティング |
| regularization, normalization, standardization | 正則化, 正規化, 標準化; normal distribution → 正規分布 (ガウス分布 for "Gaussian") | mixing them |
| training, learning | train → 訓練 (訓練する, 訓練データ, 訓練損失, 訓練誤差); learn → 学習 (学習率, 教師あり学習); pretraining → 事前学習 (fixed) | トレーニング |
| inference | running a trained model → 推論; statistical inference → 推論; "the layer infers its input size" → 自動的に決まる; estimation → 推定; prediction → 予測 | 推論 for shape inference |
| translation | of an image or a function → 平行移動 (平行移動不変性); of language → 翻訳 | 翻訳不変性 |
| gradient, derivative | 勾配; 導関数 or 微分 | グラデーション, 派生 |
| variance, expectation | 分散, 期待値 | バリアンス, 期待 |
| likelihood, evidence | 尤度; Bayesian evidence → エビデンス (エビデンス下界); marginal likelihood → 周辺尤度 | 可能性, 証拠 |
| state | 状態 | 州 |
| rank, trace, norm | of a matrix → ランク, トレース, ノルム; ranking → ランキング | 順位, 跡, 規範 |
| layer, depth, head | 層, 深さ, ヘッド | レイヤ, 頭 |
| performance | of a model → 性能; of code → 速度 or 計算性能 | パフォーマンス |

### 7.2 Glossary corrections (applied)

These corrections are now in `glossary/terms/ja.jsonl`, with the full review's
188 (`glossary/review/ja.md`); the glossary is authoritative. The list stays here
as a record of the errors the earlier machine-translated edition introduced.

汎化 (not 一般化), 分散 (バリアンス), 期待値 (期待), 連鎖律 (チェーンルール), 平行移動不変性
(翻訳不変性), 相互情報量 (相互情報), 勾配消失 (消失するグラデーション), 勾配爆発 (グラデーションの爆発),
多層パーセプトロン (マルチレイヤパーセプトロン), 確率的勾配降下法 (確率的勾配降下), 層 (レイヤ), 層正規化
(レイヤー正規化), 埋め込み層 (エンベディング レイヤ), 語彙 (ボキャブラリー), 鞍点 (サドルポイント), トレース (跡),
正解 (グラウンド トゥルース), ビームサーチ (ビーム検索), 貪欲探索 (欲張り検索), レイテンシ (遅延), 順伝播
(フォワードパス), 逆伝播 (逆方向の計算過程), 自己アテンション (自己注意), 収益 (リターン), 行動 (アクション),
訓練損失 (トレーニングの損失), 訓練誤差 (学習誤差), ラベルシフト (ラベル shift), コンテキスト長 (文脈長),
Transformer (トランスフォーマー), Adam (アダム).

### 7.3 Translationese

| Pattern | Write instead | Example |
|---|---|---|
| "In this section, we will …" word for word | 本節では + verb, no subject | ✗ このセクションでは、私たちは損失関数を紹介するつもりです。 ✓ 本節では、損失関数を導入します。 |
| Three or more の in a row | A verb or a clause | "the shape of the output of the first layer of the network" ✗ ネットワークの最初の層の出力の形状 ✓ ネットワークの最初の層が出力する形状 |
| Passive carried over | Active with no subject | "The loss is computed and the gradients are then propagated backward." ✗ 損失が計算され、その後、勾配が逆方向に伝播されます。 ✓ 損失を計算し、続いて勾配を逆伝播します。 |
| これ, それ, それら chains | Repeat the noun, or drop the pronoun | "This means that it converges." ✗ これは、それが収束することを意味します。 ✓ つまり、この反復は収束します。 |
| Inanimate subject + transitive verb | 〜から / 〜により + わかる, なる | "This result tells us that depth helps." ✗ この結果は、深さが役立つことを私たちに教えます。 ✓ この結果から、深さが有効だとわかります。 |
| 〜することができます, 〜を行います | 〜できます, the verb itself | "we can compute" → 計算できます ; "performs the update" → 更新します |
| "Note that", "Recall that" every time as 〜に注意してください | なお、〜です。 / 〜でした。 | "Recall that $b$ is the bias." → $b$ はバイアスでした。 |
| Idioms translated by image | The meaning | "to get our feet wet" ✗ 足を濡らすために (the earlier edition) ✓ 手始めに |
| 彼ら, それら for models or data; plural 〜たち | The noun, singular | "they overfit" (models) → これらのモデルは過学習します |
| One long sentence with stacked relative clauses | Two or three sentences within the block | Split at "which", "where", "so that". |

Logical phrases are fixed: because → 〜ため / 〜からです; therefore → したがって;
however → しかし; only if → 〜の場合に限り; if and only if → 〜のとき、かつそのときに限り;
at most → 高々 (in mathematics), 多くても; at least → 少なくとも; "let $x$ be …" → $x$ を…とします; such that →
…を満たす; with respect to → …に関する; without loss of generality → 一般性を失うことなく.

## 8. Sources

- JTF 日本語標準スタイルガイド（翻訳用）第4.0版, 2026-07-25, 日本翻訳連盟
  (https://www.jtf.jp/pdf/jtf_style_guide.pdf): sections cited above, read in
  full for this draft.
- 斎藤康毅『ゼロから作るDeep Learning』, O'Reilly Japan
  (https://www.oreilly.co.jp/books/9784873117584/): table of contents checked
  for terms (パラメータ, レイヤ / 層, 過学習, 誤差逆伝播法, 損失関数, 勾配).
- 山田育矢 監修・著『大規模言語モデル入門』, 技術評論社
  (https://gihyo.jp/book/2023/978-4-297-13633-8): table of contents checked
  (Transformer in Latin script, エンコーダ, デコーダ, 事前学習, ファインチューニング, 埋め込み,
  層正規化; it writes 注意機構 where this guide writes アテンション).
- Goodfellow et al.『深層学習』(KADOKAWA) and Bishop『パターン認識と機械学習』(丸善出版):
  named as である-style monographs from the drafter's knowledge of the books;
  their text was not re-checked for this draft.
- The earlier ja.d2l.ai (`d2l-ai/d2l-ja`, machine-translated): used only as
  evidence of errors to avoid (ウェイト, グラデーション, 翻訳不変性, 足を濡らすために).
- Spacing and emphasis behaviour: tested with the pandoc bundled in d2l-neu's
  `.venv-build` on 2026-11-26.

## 9. For the author to confirm

1. **Spaces between Japanese and half-width runs** (§2.2). Required around
   references and after math by the current build; applied to all Latin words
   and digits for one rule. The alternative is JTF's no-space rule with spaces
   only around references and math.　- OK
2. **Long-vowel convention** (§3.3): ML usage (パラメータ) rather than JTF and
   Microsoft usage (パラメーター). - OK
3. **アテンション rather than 注意機構** (§7.1), and 自己アテンション against the glossary's
   自己注意. - OK
4. **訓練 for "train", 学習 for "learn"** (§7.1) against the glossary's
   training = 学習. - OK
5. **The glossary corrections in §7.2**, several against `verified` entries. - OK
6. **Exercises in 〜せよ (§6).
7. **People in Latin script** even for Gauss, Bayes, Newton (§5). - OK
8. **Checks:** whether `l10n check` may accept a different token order with a
   NOTE (§2.3), whether `N×` and en-dash ranges count as protected tokens (§4),
   and whether the Japanese build prints 図, 表, 式 before references (§2.2).  - OK
9. **Italics:** Japanese fonts have no italic; the Japanese build should style
   `*…*` as bold or with emphasis dots (a build setting, not a translator rule). - OK
