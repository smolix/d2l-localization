=== BRIEF ===

# Translating unit chapter_convolutional-neural-networks/conv-layer.md#u03 into 日本語 (ja)

Follow TRANSLATING.md. Return the UNIT part only (every marker line copied exactly, each <!--tr:...--> slot filled), then optional DECISIONS and NOTES.

## Style guide (ja/style.md)

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
| Body text, list items, tab notes, notes in the body | です・ます (敬体), throughout. Reason: it is the register of the Japanese introductions this book sits beside (ゼロから作るDeep Learning, 大規模言語モデル入門) and it carries the book's collegial tone; である is the register of graduate monographs (the Japanese Goodfellow et al. and Bishop). | "Linear regression is a single artificial neuron." → 線形回帰は、1 個の人工ニューロンです。 |
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
   only around references and math.
2. **Long-vowel convention** (§3.3): ML usage (パラメータ) rather than JTF and
   Microsoft usage (パラメーター).
3. **アテンション rather than 注意機構** (§7.1), and 自己アテンション against the glossary's
   自己注意.
4. **訓練 for "train", 学習 for "learn"** (§7.1) against the glossary's
   training = 学習.
5. **The glossary corrections in §7.2**, several against `verified` entries.
6. **Exercises in 〜してください** rather than 〜せよ (§6).
7. **People in Latin script** even for Gauss, Bayes, Newton (§5).
8. **Checks:** whether `l10n check` may accept a different token order with a
   NOTE (§2.3), whether `N×` and en-dash ranges count as protected tokens (§4),
   and whether the Japanese build prints 図, 表, 式 before references (§2.2).
9. **Italics:** Japanese fonts have no italic; the Japanese build should style
   `*…*` as bold or with emphasis dots (a build setting, not a translator rule).

## Decisions log (ja/decisions.md)

(no decisions recorded yet)

## Glossary entries for this unit (ja)

| English | Sense | Term | Status | Alternatives |
|---|---|---|---|---|
| automatic differentiation | The systematic computation of derivatives by composing elementary operations. | 自動微分 | verified_d2l |  |
| backpropagation | An algorithm that propagates derivatives backward through a computation. | 誤差逆伝播法 | verified_external | バックプロパゲーション |
| backward pass | The computation that propagates derivatives from outputs toward inputs. | 逆伝播 | verified_external |  |
| bias | An additive offset in a model or prediction. | バイアス | verified_d2l |  |
| bias (estimator bias) | The difference between an estimator’s expected value and the true parameter. | 統計的バイアス | verified_d2l | 偏り(Bias) |
| channel | A feature dimension in a structured representation such as an image. | チャネル | verified_d2l |  |
| convolution | An operation that combines local input values with shared kernel weights. | 畳み込み | verified_d2l |  |
| convolutional layer | A layer that applies learned convolutional filters to its input. | 畳み込み層 | verified_d2l |  |
| cross-correlation | An operation that combines a local input window with a kernel without reversing the kernel. | 相互相関 | verified_d2l |  |
| derivative | The rate of change of a function with respect to an input. | 導関数 | proposed |  |
| finite difference | An approximation to a derivative based on changes over a finite input interval. | 有限差分 | proposed |  |
| fully connected layer | A layer in which each output unit receives a learned connection from every input unit. | 全結合層 | proposed |  |
| Gaussian distribution (normal distribution) | A continuous distribution characterized by a mean and variance. | ガウス分布 | verified_d2l |  |
| gradient | The vector of partial derivatives of a scalar function. | 勾配 | verified_d2l |  |
| kernel (convolutional kernel) | A set of weights applied locally in a convolution. | カーネル | verified_external |  |
| layer | A stage of a neural network that transforms its input. | 層 | verified_external |  |
| loss | The value of the loss function for a prediction, an example or a minibatch; the quantity training minimizes. | 損失 | verified_external |  |
| matrix | A rectangular array of numerical entries. | 行列 | verified_external |  |
| mean (average) | A measure of central tendency given by an average or expectation. | 平均 | verified_d2l |  |
| neural network | A model composed of interconnected parameterized transformations. | ニューラルネットワーク | proposed |  |
| nonlinearity | A transformation that is not linear in its input. | 非線形性 | verified_d2l |  |
| orthogonal | Having zero inner product, or preserving inner products for a transformation. | 直交 | verified_d2l |  |
| padding | Extra values added around an input to control an operation's boundary or output size. | パディング | verified_d2l |  |
| pixel | An individual sample in a digital image. | ピクセル | verified_d2l |  |
| receptive field | The region of an input that can influence a particular activation. | 受容野 | verified_external | 受容場 |
| scalar | A single numerical quantity. | スカラー | verified_d2l |  |
| second derivative | The derivative of a first derivative, measuring how a slope changes. | 二階導関数 | proposed |  |
| shape | The sizes of the axes of an array or tensor. | 形状 | proposed |  |
| sparsity | The condition in which most entries or interactions in a representation are absent or zero. | スパース性 | verified_external |  |
| standard deviation | The square root of variance. | 標準偏差 | verified_external | 標準偏差 |
| step size | The magnitude of a step taken by an iterative method. | ステップサイズ | proposed |  |
| tensor | A multidimensional array generalizing scalars, vectors, and matrices. | テンソル | verified_d2l |  |
| translation equivariance | The property that translating an input translates its representation correspondingly. | 平行移動同変性 | proposed |  |
| value | A representation combined according to attention weights. | 値 | verified_d2l |  |
| vector | An ordered collection of numerical components. | ベクトル | verified_external | ベクター |
| weight | A coefficient that scales an input or connection in a model. | 重み | verified_d2l |  |

=== CONTEXT (read, do not translate) ===

## Chapter outline (English | translation)

- `chapter_convolutional-neural-networks/index.md`
  - Convolutional Neural Networks
    - Resources and Further Reading {.unnumbered}
- `chapter_convolutional-neural-networks/why-conv.md`
  - From Fully Connected Layers to Convolutions
    - Equivariance and Invariance
    - Constraints on the Fully Connected Layer
      - Translation Equivariance
      - Locality
    - Convolution and Cross-Correlation
    - Channels
    - Summary
    - Exercises
- `chapter_convolutional-neural-networks/conv-layer.md`  <- this page
  - Convolutions for Images
    - Cross-Correlation
      - The Operation
      - Implementation
    - Convolutional Layers
      - The Layer
      - Edge Detection
      - Learning a Kernel
    - Convolution as Matrix Multiplication
    - Feature Maps and Receptive Fields
      - Stacked Layers
      - Effective Receptive Fields
    - Summary
    - Exercises
- `chapter_convolutional-neural-networks/padding-and-strides.md`
  - Padding, Stride, and Dilation
    - Padding
      - Boundary Pixels
      - Zero Padding
      - Padding Values
    - Stride
      - Strided Convolution
      - Stride and the Receptive Field
    - Dilation
      - Dilated Convolution
      - Dilation and the Receptive Field
    - Summary
    - Exercises
- `chapter_convolutional-neural-networks/channels.md`
  - Multiple Input and Multiple Output Channels
    - Multiple Input Channels
    - Multiple Output Channels
      - Parameters and Cost
    - The $1\times 1$ Convolution
    - Summary
    - Exercises
- `chapter_convolutional-neural-networks/pooling.md`
  - Pooling
    - Maximum Pooling and Average Pooling
    - Tolerance to Small Shifts
    - Global Pooling
    - Padding and Stride
    - Multiple Channels
    - Summary
    - Exercises
- `chapter_convolutional-neural-networks/lenet.md`
  - LeNet
    - The LeNet Architecture
    - Layer Shapes
    - Parameters and Multiply-Adds
    - Training
      - The Learned Features
      - Shuffled Pixels
    - Summary
    - Exercises

## The whole English page (chapter_convolutional-neural-networks/conv-layer.md at 370f7b3a15d0557b44a27c52e06ab7f5b30584e1)

````markdown
```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Convolutions for Images
:label:`sec_conv_layer`

In :numref:`sec_why-conv`, we derived the convolutional layer on paper from two
geometric principles: translation equivariance and locality. Now we see what
that formula actually computes on an image, and whether gradient descent can
discover useful kernels automatically from data.

Consider locating vertical edges where a bright region meets a dark one.
Subtracting each pixel's right-hand neighbor ($[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$)
yields zero inside uniform regions, $+1$ where white transitions to black, and
$-1$ where black returns to white. Sweeping the two-weight detector $(1, -1)$
across every valid image location produces an *edge map*. In this section, we
implement two-dimensional cross-correlation, test our hand-crafted edge
detector, let gradient descent *learn* $(1, -1)$ directly from data, show how
a convolution becomes a single matrix multiplication (*im2col*), and
derive how stacking layers widens a unit's receptive field.
[mxnet variant of code cell ```{.python .input #conv-layer-convolutions-for-images} omitted]

```{.python .input #conv-layer-convolutions-for-images}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

[tensorflow variant of code cell ```{.python .input #conv-layer-convolutions-for-images} omitted]

[jax variant of code cell ```{.python .input #conv-layer-convolutions-for-images} omitted]

## Cross-Correlation

### The Operation

Leaving channels aside for the moment, let the input $\mathbf{X}$ be an
$n_\textrm{h} \times n_\textrm{w}$ matrix and the kernel $\mathbf{K}$ be a
$k_\textrm{h} \times k_\textrm{w}$ matrix. The kernel defines a sliding
*window* of shape $k_\textrm{h} \times k_\textrm{w}$. Placing the window's
upper-left corner at $(i, j)$, we multiply the covered input patch elementwise
with $\mathbf{K}$ and sum the products to produce the output entry at $(i, j)$:

$$[\mathbf{Y}]_{i, j} = \sum_{a=0}^{k_\textrm{h}-1} \sum_{b=0}^{k_\textrm{w}-1} [\mathbf{K}]_{a, b}\, [\mathbf{X}]_{i+a, j+b}.$$
:eqlabel:`eq_corr2d`

This is the *two-dimensional cross-correlation* of $\mathbf{X}$ with
$\mathbf{K}$, written $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ with
zero-based indices matching code. Whereas :numref:`sec_why-conv` centered
offsets symmetrically from $-\Delta$ to $\Delta$, anchoring at the upper-left
corner differs only by a fixed shift of $\Delta$ while naturally handling
even-sized kernels (such as a $2 \times 2$ kernel or our $1 \times 2$ edge
detector) that have no center pixel.

![Two-dimensional cross-correlation. The shaded portions are the first output element and the input and kernel elements used to compute it: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
:label:`fig_correlation`

In :numref:`fig_correlation`, we slide a $2 \times 2$ kernel across a
$3 \times 3$ input one pixel at a time, left to right and top to bottom. The
window fits cleanly at four positions, yielding the $2 \times 2$ output:

$$
\begin{aligned}
0\times0+1\times1+3\times2+4\times3 &= 19,\\
1\times0+2\times1+4\times2+5\times3 &= 25,\\
3\times0+4\times1+6\times2+7\times3 &= 37,\\
4\times0+5\times1+7\times2+8\times3 &= 43.
\end{aligned}
$$

Notice that the output shrinks slightly along each axis because the kernel must
stay wholly inside the input. Fitting a width-$k$ window into length $n$ leaves
$n - k + 1$ valid positions, so the output shape is

$$(n_\textrm{h}-k_\textrm{h}+1) \times (n_\textrm{w}-k_\textrm{w}+1).$$

Thus a $3 \times 3$ input with a $2 \times 2$ kernel yields $2 \times 2$, a
$6 \times 8$ image with a $1 \times 2$ kernel yields $6 \times 7$, and a
$28 \times 28$ Fashion-MNIST image with a $5 \times 5$ kernel yields
$24 \times 24$. Whenever we want to preserve spatial dimensions, we surround the
input with a border of zeros (*padding*), which we examine alongside strides and
dilation in :numref:`sec_padding`.

### Implementation

We implement :eqref:`eq_corr2d` directly in `corr2d` using a pair of loops over
every output coordinate.

[mxnet variant of code cell ```{.python .input #conv-layer-the-cross-correlation-operation-1} omitted]

```{.python .input #conv-layer-the-cross-correlation-operation-1}
%%tab pytorch
def corr2d(X, K):  #@save
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = d2l.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = d2l.reduce_sum((X[i: i + h, j: j + w] * K))
    return Y
```

[tensorflow variant of code cell ```{.python .input #conv-layer-the-cross-correlation-operation-1} omitted]

[jax variant of code cell ```{.python .input #conv-layer-the-cross-correlation-operation-1} omitted]

To verify `corr2d` against our hand calculation in :numref:`fig_correlation`,
we construct the $3 \times 3$ input tensor `X` and $2 \times 2$ kernel `K` and
evaluate `corr2d(X, K)`.

```{.python .input #conv-layer-the-cross-correlation-operation-2}
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = d2l.tensor([[0.0, 1.0], [2.0, 3.0]])
corr2d(X, K)
```

The output reproduces :numref:`fig_correlation` entry for entry (and as shown in
:numref:`sec_why-conv`, strict mathematical convolution with a flipped kernel
$\mathbf{K}'$ learns the exact same family of layers, so we follow standard
deep learning terminology and call cross-correlation "convolution").


## Convolutional Layers

### The Layer

A two-dimensional convolutional layer cross-correlates its input with a learned
kernel and adds a scalar bias. Just like our MLP weights in
:numref:`chap_perceptrons`, we initialize the kernel from
$\mathcal{N}(0, 0.01^2)$ and the bias to zero. Below, `Conv2D` registers both
parameters and invokes `corr2d` in its forward pass; a layer with a kernel of
height $h$ and width $w$ is called an $h \times w$ convolutional layer.

[mxnet variant of code cell ```{.python .input #conv-layer-convolutional-layers} omitted]

```{.python .input #conv-layer-convolutional-layers}
%%tab pytorch
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(0.01 * torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```

[tensorflow variant of code cell ```{.python .input #conv-layer-convolutional-layers} omitted]

[jax variant of code cell ```{.python .input #conv-layer-convolutional-layers} omitted]

Unlike a dense layer, a convolutional layer is not locked to a fixed input
resolution because its parameters depend only on the kernel size. The exact same
$2 \times 2$ layer (5 parameters total) processes both our $3 \times 3$ tensor
`X` and a $6 \times 8$ image, returning outputs of shapes $2 \times 2$ and
$5 \times 7$:

[mxnet variant of code cell ```{.python .input #conv-layer-the-layer-1} omitted]

```{.python .input #conv-layer-the-layer-1}
%%tab pytorch, tensorflow, jax
layer = Conv2D((2, 2))
layer(X).shape, layer(d2l.ones((6, 8))).shape
```

### Edge Detection
:label:`subsec_edge_detection`

We now test the vertical edge detector from the introduction on a
$6 \times 8$ synthetic image whose middle four columns are black (`0`) and outer
columns are white (`1`).

```{.python .input #conv-layer-object-edge-detection-in-images-1}
%%tab mxnet, pytorch
X = d2l.ones((6, 8))
X[:, 2:6] = 0
X
```

[tensorflow variant of code cell ```{.python .input #conv-layer-object-edge-detection-in-images-1} omitted]

[jax variant of code cell ```{.python .input #conv-layer-object-edge-detection-in-images-1} omitted]

We define the $1 \times 2$ kernel

$$\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}.$$
:eqlabel:`eq_edge_kernel`

At every position $(i, j)$, this computes the horizontal *finite difference*
$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, a discrete approximation (up to sign)
of the horizontal derivative of brightness, which vanishes across flat regions
and spikes wherever intensity jumps.

```{.python .input #conv-layer-object-edge-detection-in-images-2}
K = d2l.tensor([[1.0, -1.0]])
```

Cross-correlating the $6 \times 8$ stripe image `X` with the $1 \times 2$
horizontal-difference kernel `K` via `corr2d(X, K)` produces the $6 \times 7$
edge map `Y`:

```{.python .input #conv-layer-object-edge-detection-in-images-3}
Y = corr2d(X, K)
Y
```

Sure enough, `Y` fires `+1` along the white-to-black boundary, `-1` along the
black-to-white boundary, and `0` everywhere else. Applying the exact same kernel
to the transposed image (turning the vertical stripe into a horizontal one)
yields all zeros:

```{.python .input #conv-layer-object-edge-detection-in-images-4}
corr2d(d2l.transpose(X), K)
```

Clearly `[1, -1]` responds strictly to vertical edges. Detecting horizontal
edges, diagonals, corners, or textures requires different kernels altogether.

### Learning a Kernel
:label:`subsec_learning_kernel`

Hand-crafting `[1, -1]` is easy when we know the exact feature we want, yet no
engineer can specify thousands of intermediate filters across a deep network by
hand. Can gradient descent recover the kernel `[1, -1]` from the input–output
pair `(X, Y)` alone? We test this using each framework's built-in 2D
convolutional layer (with bias omitted and weights initialized from
$\mathcal{N}(0, 0.01^2)$), minimizing squared error summed across all 42 output
entries over 10 gradient steps at learning rate $0.03$. Because built-in layers
expect 4D tensors including batch and channel axes (:numref:`sec_channels`), we
reshape `X` and `Y` accordingly.

[mxnet variant of code cell ```{.python .input #conv-layer-learning-a-kernel-1} omitted]

```{.python .input #conv-layer-learning-a-kernel-1}
%%tab pytorch
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
nn.init.normal_(conv2d.weight, std=0.01)
# Layers take (example, channel, height, width): one example, one channel
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2  # Learning rate

for i in range(10):
    l = ((conv2d(X) - Y) ** 2).sum()
    conv2d.zero_grad()
    l.backward()
    with torch.no_grad():
        conv2d.weight -= lr * conv2d.weight.grad
    if i % 3 == 0:
        print(f'step {i}, loss {l:.3f}')
```

[tensorflow variant of code cell ```{.python .input #conv-layer-learning-a-kernel-1} omitted]

[jax variant of code cell ```{.python .input #conv-layer-learning-a-kernel-1} omitted]

Initially the loss sits near 12 (the near-zero random kernel misses all 12
$\pm 1$ edge pixels by roughly 1), and within just nine steps it drops below
0.01. Inspecting the learned kernel confirms what it found:

[mxnet variant of code cell ```{.python .input #conv-layer-learning-a-kernel-2} omitted]

```{.python .input #conv-layer-learning-a-kernel-2}
%%tab pytorch
d2l.reshape(conv2d.weight.data, (1, 2))
```

[tensorflow variant of code cell ```{.python .input #conv-layer-learning-a-kernel-2} omitted]

[jax variant of code cell ```{.python .input #conv-layer-learning-a-kernel-2} omitted]

Both entries land within roughly 0.02 of `[1, -1]`. Nothing in this loop knew
anything about edges: given a loss signal, gradient descent discovered the
filter on its own. Here we supervised the single layer directly with `Y`; in a
deep CNN, backpropagation learns every layer's kernels end to end from the final
task loss alone.


## Convolution as Matrix Multiplication
:label:`subsec_conv_matmul`

While the nested loops in `corr2d` make the math transparent, visiting one
output pixel at a time in Python is far too slow for real workloads. A classic
remedy, and still one of the algorithms that libraries choose from, is to recast
cross-correlation as a single large matrix multiplication, the exact operation
GPUs execute at peak speed. Notice that every
output entry in :numref:`fig_correlation` is simply the dot product of the
flattened $k_\textrm{h} k_\textrm{w}$-element kernel with the corresponding
flattened input window. Extracting all
$(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ windows and
stacking them as rows yields a patch matrix; multiplying it by the flattened
kernel computes the entire cross-correlation in one shot. This standard layout
transformation is called *im2col* ("image to columns", or rows here).
:numref:`fig_conv_im2col` carries it out on the example of
:numref:`fig_correlation`.

![The cross-correlation of :numref:`fig_correlation` as one matrix product. Each of the four $2 \times 2$ windows of the input becomes a row of the patch matrix, and multiplying by the flattened kernel gives the flattened output. The shaded cells follow the first window to its row and to the output 19. The center pixel 4 lies in all four windows, so it appears in every row.](../img/conv-im2col.svg)
:label:`fig_conv_im2col`

The trade-off is extra workspace memory, since each interior pixel is duplicated
into up to $k_\textrm{h} k_\textrm{w}$ patches. Below we verify on
:numref:`fig_correlation` that this single matrix multiply matches `corr2d`
exactly:

```{.python .input #conv-layer-convolution-as-matrix-multiplication-1}
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = d2l.tensor([[0.0, 1.0], [2.0, 3.0]])
h, w = K.shape
h_o, w_o = X.shape[0] - h + 1, X.shape[1] - w + 1
patches = d2l.stack([d2l.reshape(X[i:i + h, j:j + w], (-1,))
                     for i in range(h_o) for j in range(w_o)])
Y_mat = d2l.reshape(d2l.matmul(patches, d2l.reshape(K, (-1, 1))), (h_o, w_o))
Y_mat, d2l.reduce_sum(d2l.abs(Y_mat - corr2d(X, K)))
```

The discrepancy is identically zero: one matrix product reproduces `corr2d`
perfectly.

Equivalently, we can unroll the *kernel* instead of the input into a sparse
banded matrix $\mathbf{W}$ multiplying the flattened image—the exact 2D analog
of panel (c) in :numref:`fig_why_conv_weight_sharing`, where every row repeats
the same handful of kernel weights. Seeing a convolution as a structured dense
layer brings us full circle to the derivation of :numref:`sec_why-conv` (and the
exercises build $\mathbf{W}$ explicitly).

## Feature Maps and Receptive Fields

A single convolutional layer sees only a patch the size of its kernel, whereas
recognizing whole objects demands integrating context across much wider areas.
Recall from :numref:`subsec_why-conv-channels` that a convolutional layer's
output is called a *feature map* (like our vertical edge map `Y`). For any
element $x$ in a feature map, its *receptive field* is the region of the input
image (or of an earlier layer) whose values can influence $x$ during forward
propagation—borrowing the neurophysiology term from :numref:`field_visual`.

### Stacked Layers

In :numref:`fig_correlation`, the shaded output `19` has a $2 \times 2$ receptive
field covering the top-left four pixels of $\mathbf{X}$. Now stack a *second*
$2 \times 2$ convolutional layer on top of the $2 \times 2$ output $\mathbf{Y}$
to produce a single scalar $z$: $z$ depends on all four elements of $\mathbf{Y}$,
which together depend on all nine elements of $\mathbf{X}$. Stacking layers
expands the receptive field smoothly with depth.

How quickly does it grow? Each $k \times k$ convolutional layer reaches $k - 1$
pixels farther than a single element of its input, so stacking $L$ layers with
kernel sizes $k_1, \ldots, k_L$ yields a receptive field of side length

$$r = 1 + \sum_{i=1}^{L} (k_i - 1).$$

For a stack of $L$ consecutive $3 \times 3$ layers, each layer adds 2 pixels,
covering $(2L + 1) \times (2L + 1)$ inputs: $5 \times 5$ after two layers and
$7 \times 7$ after three (:numref:`fig_conv_receptive_field`). This linear
growth is actually a blessing rather than a reason to reach for huge kernels:
three stacked $3 \times 3$ layers achieve the same $7 \times 7$ receptive field
as a single $7 \times 7$ layer using far fewer parameters ($3 \times 9 = 27$
weights vs. 49 per channel pair) while interleaving three nonlinearities instead
of one.

![Receptive field of one unit after three one-dimensional layers with kernel size 3. Shaded cells are the units that the top unit depends on, and the lines connect each of them to its inputs. Each layer widens the receptive field by 2, to 7 inputs.](../img/conv-receptive-field.svg)
:label:`fig_conv_receptive_field`

### Effective Receptive Fields

Keep in mind that $r$ measures the *theoretical* receptive field (every input
pixel with *any* path to an output unit) without saying how strongly each pixel
actually contributes. Measuring input gradients directly,
:citet:`Luo.Li.Urtasun.ea.2016` showed that the *effective* receptive field
decays sharply away from the center in a roughly Gaussian bell curve, in both
untrained and trained networks. Deep units can theoretically see far across the
image, but their sensitivity concentrates heavily near the center.

Counting paths explains the bell shape. An input at the edge of the receptive
field reaches the output unit along a single chain of connections, whereas the
center input can take many routes through the layers in between.
:numref:`fig_conv_rf_paths` counts them for the tower of
:numref:`fig_conv_receptive_field`: the center input has seven paths to the top
unit and each edge input has one. If all weights have a similar magnitude, the
influence of an input grows with its number of paths.

![Number of paths from each unit to the top unit of :numref:`fig_conv_receptive_field` (three one-dimensional layers with kernel size 3). Each count is the sum of the counts of the units that the cell feeds. Darker cells have more paths than the other cells of their layer, and the bars repeat the counts of the input row: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)
:label:`fig_conv_rf_paths`

Two practical gaps remain in our basic layer: every unpadded convolution shrinks
spatial dimensions while under-sampling border pixels, and a single kernel
detects only one pattern. We address spatial sizing and borders next in
:numref:`sec_padding`, and multi-channel feature maps in :numref:`sec_channels`.

## Summary

A convolutional layer applies two-dimensional cross-correlation with a learned
kernel plus a scalar bias, sliding the kernel across every valid window of the
input. A simple $1 \times 2$ finite-difference kernel `[1, -1]` detects vertical
edges anywhere in an image, and gradient descent recovers those exact weights
from input–output data alone, just as deep CNNs learn entire hierarchies of
filters end to end. Mathematical convolution flips the kernel, making no
difference once kernels are learned. A convolution can be evaluated as a single
matrix multiplication (*im2col*), and stacking $L$ convolutional
layers widens each unit's receptive field linearly by $k_i - 1$ per layer.

## Exercises

1. [code] **Hand-designed kernels.** Design some kernels manually.
    1. Given a directional vector $\mathbf{v} = (v_1, v_2)$, derive an
       edge-detection kernel that detects edges orthogonal to $\mathbf{v}$,
       i.e., edges in the direction $(v_2, -v_1)$.
    1. Derive a finite difference operator for the second derivative.
       Which structures in images respond most strongly to it?
    1. What is the minimum size of a kernel to obtain a derivative of
       order $d$?
    1. How would you design a blur kernel? Why might you want to use such
       a kernel?
    1. Load a photograph, convert it to one channel, and downsample it to
       about $100 \times 100$ pixels (`corr2d` loops in Python). Apply the
       edge detector, the blur kernel, and the sharpening kernel "twice
       the identity minus the blur", and display each result next to the
       original. What does the edge map of a photograph show that the
       edge map of the black band could not?

    *Adapted from Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    ConvolutionalNetworks notebook.*

2. [code] **Convolution as a banded matrix.** The patch matrix of
   :numref:`subsec_conv_matmul` unfolds the input. The opposite
   construction unrolls the kernel. Let $\mathbf{X}$ be a $4 \times 4$
   input, $\mathbf{K}$ a $3 \times 3$ kernel, and $\textrm{vec}(\cdot)$
   the row-by-row flattening.
    1. Construct the $4 \times 16$ matrix $\mathbf{W}$ with
       $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$,
       where $\mathbf{Y}$ is the cross-correlation of $\mathbf{X}$ with
       $\mathbf{K}$. How many nonzero entries does each row have, and how
       many distinct values appear in $\mathbf{W}$?
    1. Build $\mathbf{W}$ in code from `K` and confirm that the product
       reproduces `corr2d(X, K)`.
    1. For an $n \times n$ input and a $k \times k$ kernel, count the
       entries of $\mathbf{W}$ stored as a dense array, the nonzero
       entries of $\mathbf{W}$, and the entries of the patch matrix. How
       does each count grow with $n$?
    1. Keep the sparsity pattern of $\mathbf{W}$ but give every row its
       own $k^2$ values. The result is a locally connected layer: the
       fourth-order tensor $\mathsf{V}$ of :numref:`sec_why-conv` in
       $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$,
       restricted to a $k \times k$ window of offsets $(a, b)$ before
       translation equivariance ties the weights. For an $n \times n$
       input, count the parameters of this layer, of a fully connected
       layer with the same input and output shapes, and of the
       convolution.

    *Adapted from CMU 11-785,
    [Homework 2 Part 1](https://deeplearning.cs.cmu.edu/S21/index.html).*

3. [code] **Effective receptive field.** ● The theoretical receptive
   field $r$ of this section bounds which inputs *can* influence an
   output; the effective receptive field measures how much
   they actually do :cite:`Luo.Li.Urtasun.ea.2016`. Build a stack of $L$
   single-channel $3 \times 3$ convolutional layers with no
   padding, no bias, and no nonlinearity on a $32 \times 32$ input, using
   the built-in layer of :numref:`subsec_learning_kernel`. Draw every
   kernel entry independently from a standard normal distribution.
    1. For $L = 3$ and a single draw of the kernels, backpropagate a unit
       gradient from an output unit near the center of the last layer to
       the input and plot the magnitude of the input gradient. Compare its
       support with the $7 \times 7$ theoretical receptive field, and say
       where the magnitude is largest.
    1. Average the squared input gradient over 500 draws of the kernels,
       for $L = 3$ and for $L = 10$. Sum each averaged map over its rows
       and normalize the resulting column profile to sum to one. How does
       the ratio of the profile's standard deviation to the theoretical
       side length $2L + 1$ change with depth?
    1. :numref:`fig_conv_rf_paths` counts the paths for $L = 3$ in one
       dimension. Show that for general $L$ the expected squared input
       gradient, summed over rows, is proportional to the convolution of
       $L$ copies of the vector $(1, 1, 1)$. Derive how the standard
       deviation of the normalized profile grows with $L$.

4. [code] **Backward pass of a convolution.** ● Let $\mathbf{Y}$ be the
   single-channel cross-correlation of an input $\mathbf{X}$ with a kernel
   $\mathbf{K}$, as computed by `corr2d`, and let $\ell$ be a scalar loss
   that depends on $\mathbf{X}$ and $\mathbf{K}$ only through $\mathbf{Y}$.
    1. Derive $\partial \ell/\partial \mathbf{K}$ and
       $\partial \ell/\partial \mathbf{X}$ in terms of
       $\partial \ell/\partial \mathbf{Y}$, in the style of the
       backpropagation equations of :numref:`sec_backprop`.
    1. Implement your expression for $\partial \ell/\partial \mathbf{K}$
       without automatic differentiation for the loss
       $\ell = \sum_{i,j} G_{ij} Y_{ij}$, where $\mathbf{G}$ is a fixed
       random matrix of the shape of $\mathbf{Y}$. Compare it with a
       centered finite-difference gradient with step size $10^{-2}$ on a
       small random `X` and `K`, and report the largest relative error.
    1. Check your expression for $\partial \ell/\partial \mathbf{K}$
       against the kernel gradient that automatic differentiation
       computes in the experiment of :numref:`subsec_learning_kernel`.

    *Adapted from Michael Nielsen,
    [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html),
    Chapter 6, and Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    ConvolutionalNetworks notebook.*

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/65)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/66)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/271)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/17996)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §6.2]{.kicker}

Convolutions for Images<br>**2D cross-correlation · edge detection · learned kernels · im2col · receptive fields**
:::
:::

::: {.slide title="Detecting Local Edges Across an Image"}
[Motivation & Core Mental Model]{.kicker}

::: {.cols .vc}
::: {.col}
- **Synthetic $6 \times 8$ test image:** White outer columns (`1`) surrounding a 4-column black vertical band (`0`).
- **Horizontal finite difference:** Sweeping $\mathbf{K} = [1, -1]$ computes $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, firing `+1` at white-to-black and `-1` at black-to-white edges.
- **Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent.
:::

::: {.col .fig .big}
@fig:ch06-cnn-vertical-edge-motiv
:::
:::
:::

::: {.slide title="The 2D Cross-Correlation Operation"}
[Sliding-Window Inner Products]{.kicker}

Given an $n_\textrm{h} \times n_\textrm{w}$ input matrix $\mathbf{X}$ and a $k_\textrm{h} \times k_\textrm{w}$ kernel $\mathbf{K}$, the 2D cross-correlation $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ places the window's top-left corner at $(i, j)$ and sums elementwise products:

::: {.cols .vc}
::: {.col}
$$[\mathbf{Y}]_{i, j} = \sum_{a=0}^{k_\textrm{h}-1} \sum_{b=0}^{k_\textrm{w}-1} [\mathbf{K}]_{a, b}\, [\mathbf{X}]_{i+a, j+b}$$

- **Top-left anchor:** Zero-based indices $a, b \ge 0$ match array indexing in code and handle even-sized kernels (like $2 \times 2$ or $1 \times 2$) cleanly.
- **Shaded top-left output $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.
:::

::: {.col .fig}
![](../img/correlation.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Valid Window Positions and Output Spatial Shape"}
[Why Unpadded Convolutions Shrink the Grid]{.kicker}

::: {.cols .vc}
::: {.col}
$$\textrm{Output shape} = (n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$$

- **Valid window placements:** A $k_\textrm{h} \times k_\textrm{w}$ window fits inside an $n_\textrm{h} \times n_\textrm{w}$ grid at $n - k + 1$ positions per axis.
- **Shape examples:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.
:::

::: {.col .fig}
@fig:ch06-cnn-valid-window-shrinkage
:::
:::
:::

::: {.slide title="Implementing 2D Cross-Correlation in Code"}
[Direct Nested-Loop Implementation]{.kicker}

The function `corr2d` allocates the $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ output tensor and computes each entry as the sum of `X[i:i+h, j:j+w] * K`:

@conv-layer-the-cross-correlation-operation-1

::: {.d2l-note}
Each iteration slices a $k_\textrm{h} \times k_\textrm{w}$ sub-window of `X`, multiplies elementwise with `K`, and reduces via `sum`.
:::
:::

::: {.slide title="Verifying corr2d and the Flipped-Kernel Equivalence"}
[Cross-Correlation vs. Mathematical Convolution]{.kicker}

::: {.cols .vc}
::: {.col}
@conv-layer-the-cross-correlation-operation-2

- **180° flipped kernel $\mathbf{K}'$:** Cross-correlating with $\mathbf{K}$ equals convolving with $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.
- **Identical learned class:** Frameworks compute cross-correlation and call it convolution.
:::

::: {.col .fig}
@fig:ch06-cnn-flipped-kernel-equiv
:::
:::
:::

::: {.slide title="Building a 2D Convolutional Layer Module"}
[Learnable Kernel Weights Plus Scalar Bias]{.kicker}

A convolutional layer wraps `corr2d` with a learnable $k_\textrm{h} \times k_\textrm{w}$ weight parameter (initialized from $\mathcal{N}(0, 0.01^2)$) and a scalar bias (initialized to $0$):

@conv-layer-convolutional-layers
:::

::: {.slide title="Resolution Independence of Convolutional Layers"}
[Fixed Parameters Across Arbitrary Input Sizes]{.kicker}

::: {.cols .vc}
::: {.col}
@conv-layer-the-layer-1

- **Decoupled from $H \times W$:** The exact same 5 parameters ($2 \times 2$ kernel + 1 bias) process $3 \times 3$, $6 \times 8$, or megapixel inputs without reshaping weights.
:::

::: {.col .fig}
@fig:ch06-cnn-resolution-independence
:::
:::
:::

::: {.slide title="Constructing a Synthetic Vertical-Edge Image"}
[A 6×8 Binary Test Pattern]{.kicker}

We construct the $6 \times 8$ test image `X` with ones on the left and right borders and zeros across the middle four columns (`X[:, 2:6] = 0`):

@conv-layer-object-edge-detection-in-images-1

- **Piecewise-constant columns:** Intensity is constant (`1` or `0`) within columns $0\text{--}1$, $2\text{--}5$, and $6\text{--}7$, jumping only between columns $1 \to 2$ and $5 \to 6$.
:::

::: {.slide title="Applying the Horizontal Finite-Difference Kernel"}
[Detecting Vertical Edges with K = [1, -1]]{.kicker}

Cross-correlating `X` with the $1 \times 2$ kernel $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ computes $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ at every horizontal pair:

@conv-layer-object-edge-detection-in-images-2

@conv-layer-object-edge-detection-in-images-3

- **Signed edge map:** Column 1 fires `+1` (white-to-black transition), column 5 fires `-1` (black-to-white transition), and all constant interior columns evaluate to `0`.
:::

::: {.slide title="Orientation Selectivity of Edge Kernels"}
[Why One Kernel Cannot Detect All Orientations]{.kicker}

::: {.cols .vc}
::: {.col}
@conv-layer-object-edge-detection-in-images-4

- **Blind to horizontal edges:** Transposing `X` (`X.T`) makes horizontal neighbors identical, so $[1, -1]$ outputs all zeros.
- **Filter banks (§{sec_channels}):** Vision layers learn parallel channels for horizontal, vertical, and diagonal orientations.
:::

::: {.col .fig}
@fig:ch06-cnn-orientation-selectivity
:::
:::
:::

::: {.slide title="Learning an Edge Detector via Gradient Descent"}
[Supervising a Built-In Convolutional Layer on (X, Y)]{.kicker}

::: {.cols .vc}
::: {.col}
@conv-layer-learning-a-kernel-1
:::

::: {.col .fig .narrow}
@fig:ch06-cnn-learning-edge-loop

- **4D framework tensors:** Reshape `X` and `Y` to 4D tensors (one example, one channel) and minimize $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ at $\eta = 0.03$.
:::

:::
:::

::: {.slide title="Inspecting the Learned Kernel Weights"}
[Why Initial Loss Is 12 and Converges to [1, -1]]{.kicker}

::: {.cols .vc}
::: {.col}
@conv-layer-learning-a-kernel-2

- **Why loss starts at $12.0$:** Near-zero weights predict $\hat{Y} \approx 0$, missing the twelve $\pm 1$ edge pixels in `Y` ($12 \times 1^2 = 12$).
- **Convergence in 9 steps:** SGD recovers $[0.99, -0.99]$ automatically from data.
:::

::: {.col .fig}
@fig:ch06-cnn-kernel-gd-convergence
:::
:::
:::

::: {.slide title="Fast Execution: Convolution via Matrix Multiplication"}
[The im2col (Image-to-Columns) Transformation]{.kicker}

::: {.cols .vc}
::: {.col}
@conv-layer-convolution-as-matrix-multiplication-1

- **Single dense GEMM:** Flatten each $k_\textrm{h} \times k_\textrm{w}$ window into a row of `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ and multiply by $\textrm{vec}(\mathbf{K})$.
:::

::: {.col .fig}
@fig:ch06-cnn-im2col-gemm

- **Trade-off:** Peak GPU Tensor Core speed at the cost of $k_\textrm{h} k_\textrm{w}\times$ patch workspace duplication.
:::
:::
:::

::: {.slide title="Dual View: Unrolling the Kernel into a Banded Matrix"}
[Toeplitz Weight Matrix Multiplication]{.kicker}

Instead of unfolding the input image into duplicated patches, we can unroll the *kernel* into a sparse banded matrix $\mathbf{W}$ that multiplies the flattened input vector $\textrm{vec}(\mathbf{X})$:

![](../img/why-conv-weight-sharing.svg){.nostretch width=46% fig-align="center"}

- **Matrix–vector form:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, where $\mathbf{W}$ has $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ rows and $n_\textrm{h} n_\textrm{w}$ columns.
- **Connection to §{sec_why-conv} and backprop:** Every row of $\mathbf{W}$ holds the same $k_\textrm{h} k_\textrm{w}$ kernel entries shifted across columns (panel c), making the backward pass with respect to $\mathbf{X}$ simply multiplication by $\mathbf{W}^\top$ (a transposed convolution).
:::

::: {.slide title="Feature Maps and Receptive Fields Across Depth"}
[Linear Receptive-Field Growth in Stacked Layers]{.kicker}

A unit's **receptive field** is the region of the original input image whose pixels can influence that unit's activation during forward propagation:

::: {.cols .vc}
::: {.col}
- **Single layer ($L=1$):** A $k_1 \times k_1$ kernel sees a $k_1 \times k_1$ input patch.
- **Stacking $L$ stride-1 layers:** Layer $i$ with kernel size $k_i$ reaches $k_i - 1$ pixels farther along each axis than a single element of layer $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$
- **$L$ stacked $3 \times 3$ layers:** Each layer adds $3 - 1 = 2$ pixels, giving side length $r = 2L + 1$ ($3 \to 5 \to 7$).
:::

::: {.col .fig}
![](../img/conv-receptive-field.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Why Deep Stacks of 3×3 Kernels Beat One Large Kernel"}
[Parameter Efficiency & Nonlinear Expressivity]{.kicker}

::: {.cols .vc}
::: {.col}
- **Matched $7 \times 7$ receptive field:** Three stacked $3 \times 3$ layers ($r = 1 + 2 + 2 + 2 = 7$) see the exact same $7 \times 7$ patch as one $7 \times 7$ layer.
- **45% fewer parameters:** $3 \times 9 c^2 = 27 c^2$ weights instead of $49 c^2$ across $c$ channels.
- **Richer expressivity:** Compounds **3 nonlinear activations** instead of 1 (the core VGG design rule in §{sec_blocks}).
:::

::: {.col .fig .big}
@fig:ch06-cnn-stacked-3x3-vs-7x7
:::
:::
:::

::: {.slide title="Theoretical vs. Effective Receptive Fields"}
[Gaussian Decay of Sensitivity Away from the Center]{.kicker}

::: {.cols .vc}
::: {.col}
- **Theoretical RF ($r = 2L+1$):** Bounding box of all pixels with *at least one* path to the output unit.
- **Central Limit Theorem:** Center pixels connect via exponentially many paths, whereas corner pixels connect via only **1 path**.
- **Effective RF (Luo et al., 2016):** Input gradient magnitude $\partial y / \partial \mathbf{X}$ decays in a Gaussian bell curve with std $\sigma \propto \sqrt{L}$.
:::

::: {.col .fig .big}
@fig:ch06-cnn-effective-rf-gaussian
:::
:::
:::

::: {.slide title="Concept Check: Cross-Correlation & Receptive Fields"}
[Concept Check]{.kicker}

Test your understanding of output shapes, `im2col`, and receptive-field arithmetic:

1. **im2col matrix dimensions:** You apply an unpadded $3 \times 3$ single-channel kernel to a $10 \times 10$ image via `im2col`. What is the output spatial shape, and what are the exact dimensions of the `patches` matrix and flattened kernel vector?
2. **Depth to span Fashion-MNIST:** At stride 1, how many stacked $3 \times 3$ layers are required so a single unit's theoretical receptive field spans an entire $28 \times 28$ image? How many with $5 \times 5$ layers?
3. **Edge-loss boundary shift:** In our $6 \times 8$ vertical-edge learning experiment, why did the loss start at $\approx 12$, and what would the initial loss be if the black band extended all the way to the right edge (`X[:, 2:8] = 0`)?

::: {.d2l-note .fragment .fade-up}
**Answers:**
(1) Output shape is $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ outputs). The `patches` matrix has shape $(64, 9)$ and multiplies the $(9, 1)$ kernel vector.
(2) For $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ layers. For $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ layers.
(3) Each of the 6 rows had two $\pm 1$ edges ($6 \times 2 \times 1^2 = 12$). Extending the black band to the right border eliminates the right-hand black-to-white edge, leaving one edge per row and halving the initial loss to $\approx 6$.
:::
:::

::: {.slide title="Summary & Key Takeaways"}
[Section Synthesis]{.kicker}

::: {.cols .vc}
::: {.col}
- **2D Cross-Correlation & Edge Learning:** Slides $\mathbf{K}$ across valid windows ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); SGD learns $[1, -1]$ in 9 steps.
- **im2col Matrix Multiply:** Trades $k_\textrm{h} k_\textrm{w}\times$ patch duplication for peak GPU GEMM throughput.
- **Stacked Receptive Fields:** $r = 1 + \sum (k_i - 1) = 2L+1$ for $3 \times 3$ stacks, with Gaussian center sensitivity.
:::

::: {.col .fig .big}
@fig:ch06-cnn-conv-layer-summary
:::
:::
:::
````

## Translation of the previous unit

(the previous unit, chapter_convolutional-neural-networks/conv-layer.md#u02, is not translated yet)

=== UNIT ===

<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u03 lang=ja rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:2294243a kind=heading-->
## Exercises
<!--tr:2294243a-->

<!--en:6f4ddec6 kind=exercise-->
1. [code] **Hand-designed kernels.** Design some kernels manually.
    1. Given a directional vector $\mathbf{v} = (v_1, v_2)$, derive an
       edge-detection kernel that detects edges orthogonal to $\mathbf{v}$,
       i.e., edges in the direction $(v_2, -v_1)$.
    1. Derive a finite difference operator for the second derivative.
       Which structures in images respond most strongly to it?
    1. What is the minimum size of a kernel to obtain a derivative of
       order $d$?
    1. How would you design a blur kernel? Why might you want to use such
       a kernel?
    1. Load a photograph, convert it to one channel, and downsample it to
       about $100 \times 100$ pixels (`corr2d` loops in Python). Apply the
       edge detector, the blur kernel, and the sharpening kernel "twice
       the identity minus the blur", and display each result next to the
       original. What does the edge map of a photograph show that the
       edge map of the black band could not?

    *Adapted from Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    ConvolutionalNetworks notebook.*
<!--tr:6f4ddec6-->

<!--en:ec8621f8 kind=exercise-->
2. [code] **Convolution as a banded matrix.** The patch matrix of
   :numref:`subsec_conv_matmul` unfolds the input. The opposite
   construction unrolls the kernel. Let $\mathbf{X}$ be a $4 \times 4$
   input, $\mathbf{K}$ a $3 \times 3$ kernel, and $\textrm{vec}(\cdot)$
   the row-by-row flattening.
    1. Construct the $4 \times 16$ matrix $\mathbf{W}$ with
       $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$,
       where $\mathbf{Y}$ is the cross-correlation of $\mathbf{X}$ with
       $\mathbf{K}$. How many nonzero entries does each row have, and how
       many distinct values appear in $\mathbf{W}$?
    1. Build $\mathbf{W}$ in code from `K` and confirm that the product
       reproduces `corr2d(X, K)`.
    1. For an $n \times n$ input and a $k \times k$ kernel, count the
       entries of $\mathbf{W}$ stored as a dense array, the nonzero
       entries of $\mathbf{W}$, and the entries of the patch matrix. How
       does each count grow with $n$?
    1. Keep the sparsity pattern of $\mathbf{W}$ but give every row its
       own $k^2$ values. The result is a locally connected layer: the
       fourth-order tensor $\mathsf{V}$ of :numref:`sec_why-conv` in
       $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$,
       restricted to a $k \times k$ window of offsets $(a, b)$ before
       translation equivariance ties the weights. For an $n \times n$
       input, count the parameters of this layer, of a fully connected
       layer with the same input and output shapes, and of the
       convolution.

    *Adapted from CMU 11-785,
    [Homework 2 Part 1](https://deeplearning.cs.cmu.edu/S21/index.html).*
<!--tr:ec8621f8-->

<!--en:54bcee26 kind=exercise-->
3. [code] **Effective receptive field.** ● The theoretical receptive
   field $r$ of this section bounds which inputs *can* influence an
   output; the effective receptive field measures how much
   they actually do :cite:`Luo.Li.Urtasun.ea.2016`. Build a stack of $L$
   single-channel $3 \times 3$ convolutional layers with no
   padding, no bias, and no nonlinearity on a $32 \times 32$ input, using
   the built-in layer of :numref:`subsec_learning_kernel`. Draw every
   kernel entry independently from a standard normal distribution.
    1. For $L = 3$ and a single draw of the kernels, backpropagate a unit
       gradient from an output unit near the center of the last layer to
       the input and plot the magnitude of the input gradient. Compare its
       support with the $7 \times 7$ theoretical receptive field, and say
       where the magnitude is largest.
    1. Average the squared input gradient over 500 draws of the kernels,
       for $L = 3$ and for $L = 10$. Sum each averaged map over its rows
       and normalize the resulting column profile to sum to one. How does
       the ratio of the profile's standard deviation to the theoretical
       side length $2L + 1$ change with depth?
    1. :numref:`fig_conv_rf_paths` counts the paths for $L = 3$ in one
       dimension. Show that for general $L$ the expected squared input
       gradient, summed over rows, is proportional to the convolution of
       $L$ copies of the vector $(1, 1, 1)$. Derive how the standard
       deviation of the normalized profile grows with $L$.
<!--tr:54bcee26-->

<!--en:6ab5b67c kind=exercise-->
4. [code] **Backward pass of a convolution.** ● Let $\mathbf{Y}$ be the
   single-channel cross-correlation of an input $\mathbf{X}$ with a kernel
   $\mathbf{K}$, as computed by `corr2d`, and let $\ell$ be a scalar loss
   that depends on $\mathbf{X}$ and $\mathbf{K}$ only through $\mathbf{Y}$.
    1. Derive $\partial \ell/\partial \mathbf{K}$ and
       $\partial \ell/\partial \mathbf{X}$ in terms of
       $\partial \ell/\partial \mathbf{Y}$, in the style of the
       backpropagation equations of :numref:`sec_backprop`.
    1. Implement your expression for $\partial \ell/\partial \mathbf{K}$
       without automatic differentiation for the loss
       $\ell = \sum_{i,j} G_{ij} Y_{ij}$, where $\mathbf{G}$ is a fixed
       random matrix of the shape of $\mathbf{Y}$. Compare it with a
       centered finite-difference gradient with step size $10^{-2}$ on a
       small random `X` and `K`, and report the largest relative error.
    1. Check your expression for $\partial \ell/\partial \mathbf{K}$
       against the kernel gradient that automatic differentiation
       computes in the experiment of :numref:`subsec_learning_kernel`.

    *Adapted from Michael Nielsen,
    [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html),
    Chapter 6, and Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    ConvolutionalNetworks notebook.*
<!--tr:6ab5b67c-->

<!--en:657a396a kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/65)
<!--tr:657a396a-->

<!--en:63c44b48 kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/66)
<!--tr:63c44b48-->

<!--en:00b73d0b kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/271)
<!--tr:00b73d0b-->

<!--en:b79af65b kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/17996)
<!--tr:b79af65b-->
