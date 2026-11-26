# Style guide: Russian (ru)

Draft of 2026-11-26 for the author's review. Translators read this with every
unit (TRANSLATING.md §7–§8). Each rule has one reason and one example
(English → Russian, or ✗ wrong / ✓ right). Examples marked *(test)* are real
sentences from the translation test `evals/ru-conv-layer/`. Rules that rest on a
build setting name it (§11); points for a native reviewer are in §12.

Order of authority: the decisions log, then this guide, then the glossary. The
glossary says *what* a term is; this guide says *how to choose and inflect*.
Every entry of the Russian glossary is still `proposed`
(`glossary/notes/ru-review.md`), so TRANSLATING.md §7 lets a translator replace
an entry that is wrong in context; record each such case in DECISIONS.

## 1. Register

**Decision: authorial «мы», the reader as «вы» in lower case, never «ты» and
never «Вы».** Reason: this is the register of Russian university textbooks; «мы»
carries the book's "we work through this together" tone (TRANSLATING.md §8).
Capitalised «Вы» belongs to a letter addressed to one person, and «ты» to blogs.
Address the reader only where the English really does; most "you" sentences are
general statements and become impersonal.

| English | Russian | Example |
|---|---|---|
| we + action in the text | 1st person plural, usually without the pronoun | "We minimize the loss with SGD." → Минимизируем потери с помощью SGD. |
| "let us", "let's", "consider" | perfective future: Рассмотрим …, Посмотрим …, Вычислим … | "Let us look at the gradient." → Посмотрим на градиент. ✗ Давайте посмотрим … |
| "we will see", "we will return to" | perfective future | "We will return to this in …" → К этому мы вернёмся в … |
| "we" where the authors are the point | «мы» written out | "We recommend AdamW." → Мы рекомендуем AdamW. |
| "you" = the reader (advice, a question, exercises) | «вы», «ваш», lower case | "Suppose you want to estimate the price of a house." → Предположим, вы хотите оценить стоимость дома. |
| "you can", "one can" = anyone | «можно» + infinitive, or the verb itself | "You can verify this numerically." → Это можно проверить численно. ✗ Вы можете проверить это численно. |
| "our model", "our dataset" | «наша модель» at most once per paragraph; then «модель», «эта модель» | |
| "we" = people, the field | «мы, люди», «исследователи», «в литературе» | "We humans attend selectively." → Мы, люди, направляем внимание избирательно. |
| "Note that", "Recall that" | Заметим, что … ; Напомним, что … (not for every sentence, §9.4) | "Recall that $b$ is the bias." → Напомним, что $b$ — смещение. |

People in general take the plural or the grammatical masculine (исследователи,
практики, читатель). Do not write feminitives or paired forms
(✗ исследователь(ница)); a generic "she" in the English becomes a plural. A named
person keeps the pronoun of the source. Technical roles are not people: агент,
критик, генератор, дискриминатор, учитель, оптимизатор.

**Sentence shape** (each point is checkable):

- **«X — это Y», not «X является Y».** «является» only where a dash is
  impossible (after a negation or a modal). ✗ Свёртка является линейной
  операцией. ✓ Свёртка — линейная операция.
- **Verbs, not verbal nouns.** No sentence whose only verb is *осуществляется*,
  *производится*, *выполняется*, *имеет место* or *происходит*. ✗ Вычисление
  градиента осуществляется при помощи цепного правила. ✓ Градиент вычисляем по
  цепному правилу.
- **At most two genitives in a row.** ✗ форма выхода первого слоя сети ✓ форма,
  которую выдаёт первый слой сети.
- **A деепричастный оборот has the subject of its main verb.** ✗ Используя
  цепное правило, градиент вычисляется за один проход. ✓ По цепному правилу
  градиент вычисляется за один проход. / Используя цепное правило, вычислим
  градиент за один проход.
- **One participial phrase or one «который» clause per noun**, and no «который»
  inside a «который» clause. Split English sentences at "which", "where", "so
  that" (TRANSLATING.md §3 allows splitting within a block).
- **No канцелярит.** «этот», not «данный»; «чтобы», not «в целях» or «с целью»;
  «с помощью» or the instrumental case, not «посредством»; no «в рамках», «на
  сегодняшний день», «представляет собой» (every time).
- **Known first, new last.** Russian word order carries the emphasis that
  English gives with articles and stress. ✗ Мы затем вычисляем … Мы затем
  обновляем … ✓ Затем вычисляем … После этого обновляем …
- **No slang loans** (§5.1): never лосс, фича, таргет, трейн, скор, бэкпроп,
  файнтюнинг, лернинг рейт, батчсайз, зафитить.

Spelling follows the current academic norm (Правила 1956, the 2006 handbook
edited by Lopatin). **The letter ё is always written**: свёртка, обучён, всё, ещё, её, трёх,
четырёх, расчёт, приближённый, смещённый (but вложенный, разреженный, which
have no ё). Reason: the glossary writes ё, a missing ё changes meaning (все / всё, небо /
нёбо) and it makes the glossary check fail (`свертка` does not match the term
`свёртка`); all three test translators chose the same.

## 2. Punctuation and typography

| Item | Rule | Example |
|---|---|---|
| Quotation marks | «…» (U+00AB, U+00BB); inside them „…“ (U+201E, U+201C). Never "…", “…” or '…' in Russian text. Titles of books and papers take «…», no italics added. | the "attention" weights → веса «внимания» ; «статья „Attention Is All You Need“ вышла в 2017 году» |
| Quotes in attributes | A slide `title="…"` is delimited by ASCII `"`; inside it only «…». | `title="Что значит «глубокий»?"` |
| Dash | The em dash `—` with a space on each side, for the English em dash and wherever Russian grammar wants a dash: between a noun subject and a noun predicate, before «это», in an apposition. Never a hyphen or an en dash with spaces. | "the bias—an offset—shifts the plane" → смещение — постоянный сдвиг — переносит плоскость ; "$b$ is the bias" → $b$ — смещение |
| Dash, limits | No dash after a pronoun subject (✗ Она — линейная. ✓ Она линейна.) and none before «не». At most one pair of dashes per sentence; otherwise commas or parentheses. | |
| En dash | Only in ranges of numbers, without spaces (§6), and between two surnames in a name (метод Эйлера–Маруямы, размерность Вапника–Червоненкиса). | 40–55 ; 2017–2024 |
| Hyphen | In compounds and with particles (кросс-энтропия, мини-батч, $k$-мерный, из-за, какой-либо). Never as a dash. | |
| Pair of equal nouns | An em dash with spaces, the pair in «…» (the rule for «человек — машина»). | "the input–output pair" → ✓ (test, Fable, Sonnet) паре «вход — выход» `(X, Y)` ✗ (test, Opus) паре вход–выход |
| Comma | Russian rules, not English ones: a comma before «что», «чтобы», «который», «если», «когда», «так как», «то есть»; around a participial phrase that follows its noun and around every деепричастный оборот; around вводные слова (например, во-первых, следовательно, таким образом, наконец, конечно). No comma after a fronted adverbial, none after «однако» that opens a sentence, none before «и» / «или» in a list. | ✗ В этом разделе, мы … ✓ В этом разделе мы … ; "Adam, RMSProp, and SGD" → Adam, RMSProp и SGD |
| Colon | Lower case after a colon, also after a bold lead-in that ends in a colon. A capital only for a name or after a lead-in that ends in a full stop. | **Основное предположение:** ошибки независимы. ; **Свёртка как ленточная матрица.** Матрица фрагментов … |
| Semicolon | Allowed between two main clauses and between long list items; do not copy every English semicolon. | |
| Abbreviations | In body text write out «то есть», «например», «так как», «так называемый». «и т. д.», «и т. п.», «и др.» may close a list anywhere; «см.», «ср.» only in parentheses. Tables and slides may also use т. е., напр., тыс., млн, млрд (млн and млрд without a full stop). "i.e." → то есть; "e.g." → например; "cf." → ср.; "etc." → и т. д.; "vs." → и, против, по сравнению с. | ✗ т.е., т.к., и т.д. ✓ и т. д. |
| Spaces | Type ordinary spaces everywhere (`и т. д.`, `10 000`, `24 GiB`, before and after `—`). Do not type no-break or narrow spaces or soft hyphens (TRANSLATING.md §8); the build adds them (§11). No space before `%` (§6), none before `,` `.` `:` `;` `?` `!`. | ✓ `92.3%` ✗ `92.3 %` |
| Apostrophe | Never: no ending is attached to a Latin word (§5.3). | ✗ Adam'а, ✗ GPU'шки |
| Emphasis | Keep `*…*` and `**…**` on the corresponding word(s), around the whole inflected word; add none. The English gloss stays outside the markers. | "*regression* predicts …" → *регрессия* предсказывает … ; `*взаимная корреляция* (англ. cross-correlation)` |
| Headline case | Russian has no title case: capitalise the first word and proper names only. | "The Model and the Loss" → Модель и функция потерь |
| ? and ! | A question stays a question. Drop the English "!" except in a real exclamation. | "Why does this work?" → Почему это работает? |

## 3. Cross-references and citations

### 3.1 What the build prints

A translator sees ``:numref:`fig_x` `` and must know the printed words, because
Russian prepositions govern case and an attribute must agree in gender. The
build takes the words from Quarto's Russian language file, overridden by
`quarto.*` keys in `ru/strings.yml`. **This guide requires the overrides of
§11, which make every printed prefix an invariable lower-case abbreviation:**

| Label starts with | Printed | Preposition for "in" | Gender (avoid agreement, §3.2) |
|---|---|---|---|
| `fig_`, `img_` | рис. 7.2 | на | m |
| `tab_`, `table_` | табл. 7.1 | в | f |
| `sec_`, `subsec_` | разд. 7.2 | в | m |
| `chap_` | гл. 7 | в | f |
| `:eqref:` (`eq_`) | (7.2.1), the number only | write the noun yourself | |
| `§{sec_x}` (slides) | §7.2 | в | m |
| `§{num:sec_x}` | 7.2, the number only | | |

Reason: «рис.», «табл.», «разд.» and «гл.» are the standard abbreviations of
Russian scientific publishing and do not change with case, so «на рис. 7.2», «из
разд. 7.1», «в гл. 5» are correct with any preposition. Without the overrides
Quarto prints «Рисунок 7.2», «Глава 7.1» (its Russian word for a *section* is
wrong) and «Уравнение 7.1», capitalised and in the nominative, which no
preposition can precede. Of the three test translators, Fable assumed the
abbreviations, Opus assumed nominative words and avoided every preposition
(«Ранее (:numref:`sec_why-conv`) мы вывели …»), and Sonnet wrote its own word
(«на рис. :numref:`fig_correlation`»), which would print «на рис. рис. 7.2».

A label without one of these prefixes (`field_visual`) is looked up in the
English page in CONTEXT to see what it names.

### 3.2 Rules

1. **Write no prefix of your own.** ✗ (test, Sonnet) В разделе
   :numref:`sec_why-conv` мы вывели … ✓ (test, Fable) В :numref:`sec_why-conv`
   мы вывели свёрточный слой на бумаге …
2. **Choose the preposition by the label** (table above): «на» for figures, «в»
   for everything else; «из», «с», «по», «см.» as the sentence needs.
   ✓ (test, Fable) На :numref:`fig_correlation` мы сдвигаем ядро $2 \times 2$ по
   входу … ; … мы займёмся в :numref:`sec_padding`.
3. **Never start a sentence, a caption or a list item with a reference**: the
   printed abbreviation is lower case. ✗ (test, Opus) :numref:`fig_correlation`
   показывает ядро … ✓ На :numref:`fig_correlation` показано ядро …
4. **Never make a reference the subject and never attach an agreeing word to
   it** (adjective, participle, pronoun, past-tense verb). ✗ этот
   :numref:`fig_x` ; ✗ :numref:`tab_x` показал … ✓ В :numref:`tab_x` показано …
   ; «тот же пример, что и на :numref:`fig_x`».
5. **Parentheses are always safe** and are the usual Russian form for a passing
   pointer: … после трёх слоёв (:numref:`fig_conv_receptive_field`). "see" →
   «см.» inside parentheses: (см. :numref:`sec_sgd`).
6. **Equations: write the noun, then the reference.** The noun says what the
   line is (формула, уравнение, неравенство, определение, тождество, разложение)
   and takes the case. "We implement :eqref:`eq_corr2d` directly" → ✓ Реализуем
   формулу :eqref:`eq_corr2d` напрямую … ; "substituting :eqref:`a` into
   :eqref:`b`" → подставив :eqref:`a` в :eqref:`b` (bare numbers are fine once
   the noun is clear).
7. **Several references**: repeat nothing, the build prints each prefix. "Figures
   :numref:`a` and :numref:`b`" → на :numref:`a` и :numref:`b`.
8. **Slides** use `§{sec_x}` as the English does: «(основное правило VGG,
   §{sec_blocks})» or «в §{sec_blocks}».

### 3.3 Citations

- `:cite:` stands at the end of the clause it supports, before the comma or full
  stop, as in English.
- `:citet:` prints the authors' names in Latin script. The key shows how many
  there are: several names or `.ea.` take a plural verb, one name takes a
  construction without gender. ✓ (test, all three) :citet:`Luo.Li.Urtasun.ea.2016`
  показали, что … ; one author: В работе :citet:`Hochreiter.1991` показано, что
  … ✗ :citet:`Hochreiter.1991` показал …
- No case ending and no possessive on a citation: ✓ модель из работы
  :citet:`Vaswani.Shazeer.Parmar.ea.2017`.

## 4. Math and code in a sentence

- **A symbol or code span that needs a case gets a noun**: с градиентом
  $\mathbf{g}$, норма вектора $\mathbf{w}$, в функции `corr2d`. A symbol alone
  may stand as a subject, after a preposition, or after its noun.
- **Russian order: noun, symbol, then size or value.** "an $n \times m$ matrix
  $\mathbf{X}$" → матрица $\mathbf{X}$ размера $n \times m$. This reorders
  protected tokens, which `l10n check` reports as a warning; **reorder whenever
  the English order would give unnatural Russian, and name the block in NOTES**
  (one line: «8d45aae8: порядок формул изменён»). Do not bend the sentence to
  keep the English order. ✓ (test, Fable) Для входной матрицы $\mathbf{X}$
  размера $n_\textrm{h} \times n_\textrm{w}$ и ядра $\mathbf{K}$ размера
  $k_\textrm{h} \times k_\textrm{w}$ … ✗ (test, Opus) Для входной $n_\textrm{h}
  \times n_\textrm{w}$-матрицы $\mathbf{X}$ и $k_\textrm{h} \times
  k_\textrm{w}$-ядра $\mathbf{K}$ … ✗ (test, Sonnet) с ядром $1 \times 2$
  $\mathbf{K} = …$ (two formulas side by side).
- A size directly after its noun needs no word when no symbol stands between:
  ядро $2 \times 2$, вход $3 \times 3$, слой $3 \times 3$. With a symbol:
  «размера».
- **Compounds with a symbol or number take a hyphen**: $d$-мерный, $k$-й,
  $n$-грамма, $\ell_2$-норма, 32-битный. The hyphenated form is for adjectives
  and fixed terms, not for sizes (see above).
- **Protected math is copied as it is, even when it contains English.** "the
  $i^{\textrm{th}}$ layer" → $i^{\textrm{th}}$ слой. Do not change it and do not
  report it in NOTES (the German ruling, de/style.md §10.16).
- **End the sentence before a display equation if a verb would have to follow
  it**: "… is given by $$…$$ where $n$ is …" → … задаётся формулой $$…$$ Здесь
  $n$ — …. After a display equation «где $n$ — число примеров» continues the
  sentence in lower case, as in Russian mathematical texts.
- Inline code stays English and is never inflected or hyphenated to an ending:
  ✗ `corr2d`-ом ✓ функцией `corr2d`.
- A closing `$` must not be followed directly by a digit (TRANSLATING.md and
  `l10n check`): keep a space or a word between.

## 5. Terminology

### 5.1 Russian term, Cyrillic loan, or Latin script

**Use the Russian term where mathematics, statistics or computer-science
teaching has one. Use the Cyrillic loan where it is the written norm of current
Russian courses and no Russian term exists. Keep Latin script for names and for
the few terms Russian prose really keeps in English. Never use a spoken-language
loan.** The glossary decides each case; for a term without an entry apply this
rule, and record the choice in DECISIONS.

| Case | Rule | Examples |
|---|---|---|
| Established Russian term | Use it. | слой, вес, свёртка, признак, функция потерь, скорость обучения, градиентный спуск, переобучение, регуляризация, состояние, награда, стратегия, математическое ожидание, набор данных, обучающая выборка, внимание, дообучение, предобучение, обратное распространение ошибки, ранняя остановка |
| Cyrillic loan is the written norm | Use it, lower case, declined (§5.2). | батч, мини-батч, токен, эмбеддинг, дропаут, пулинг, промпт, логит, энкодер, декодер, трансформер, фреймворк, инференс (running a model), кэш, патч |
| Slang or spoken loan | Never, even if a glossary entry lists it as an alternative. | ✗ лосс, фича, таргет, трейн, скор, датасет, паддинг, страйд, бэкпроп, файнтюнинг |
| Names of models, architectures, optimizers, libraries, datasets, metrics; acronyms; code | Latin script, exactly as in English; never translated in part. | ResNet, BERT, Adam, AdamW, PyTorch, ImageNet, BLEU, ReLU, SGD, MLP, LSTM, GPU, im2col |
| Terms the glossary keeps in Latin script | As the glossary. | softmax, teacher forcing, seq2seq, score matching, flow matching, off-policy, log-sum-exp |
| English acronym | Keep the English acronym (CNN, MLP, SGD, KL). No Russian-made acronyms: an ad hoc English abbreviation is written out. Russian acronyms only where the glossary has them (ОДУ, ИИ). | "RF" on a slide → рецептивное поле ✗ (test, Opus, Sonnet) РП |

«Transformer» in Latin script is the original model of 2017 only; the
architecture is «трансформер» (glossary).

### 5.2 Cyrillic loans: declension

Cyrillic loans are ordinary Russian nouns: lower case, with gender by ending and
full declension. All of the list in §5.1 are masculine: батча, батчу, батчем, в
батче; батчи, батчей. Genitive plural: токенов, эмбеддингов, логитов, промптов,
патчей, батчей, энкодеров. Compounds follow the glossary: мини-батч,
батч-нормализация, макс-пулинг, автоэнкодер.

### 5.3 Latin-script words: no endings, a noun for the case, a fixed gender

**A word in Latin script is never declined and never takes a Russian ending**,
neither with an apostrophe nor with a hyphen (✗ Adam'ом, ✗ GPU-шка, ✗ LSTM-ы,
✗ softmax'а). Reason: such forms are spoken Russian, and they break search and
the glossary check.

- **Where the case must show, add the generic noun, which carries it**:
  оптимизатор Adam, функция softmax, функция активации ReLU, слой ReLU, сеть
  LSTM, модель BERT, библиотека PyTorch, набор данных ImageNet, метрика BLEU.
  ✗ обучаем модель Adam ✓ обучаем модель оптимизатором Adam.
- **A bare Latin word is fine** as a subject with a present-tense verb, after a
  preposition, and after its noun: Adam сходится быстрее; в PyTorch; с помощью
  SGD; на GPU; по сравнению с ResNet-50.
- **Plural**: no -s, no ending; number shows in the numeral or the noun: два
  GPU, на четырёх GPU, несколько GPU, сети LSTM. ✗ GPUs, ✗ GPU-шки.
- **Agreement** (adjective, past tense, «который») follows the fixed gender
  below. Where it would look odd, add the generic noun instead.

| Gender | Words | Head noun | Example |
|---|---|---|---|
| masculine | GPU, CPU, TPU | (графический) процессор | один GPU, мощный GPU, GPU простаивал |
| masculine | Adam, AdamW, SGD, RMSProp, AdaGrad | оптимизатор, алгоритм, спуск | Adam сошёлся за 10 эпох |
| masculine | MLP | перцептрон | двухслойный MLP |
| masculine | softmax, im2col, seq2seq, API | (indeclinable; prefer «функция softmax», «операция softmax») | softmax нормирует логиты |
| feminine | CNN, RNN, GAN, LSTM, GRU | сеть | глубокая CNN, LSTM обучена |
| feminine | ReLU, GELU | функция | ReLU обнуляет отрицательные значения |
| feminine | BERT, GPT-2, ResNet-18, VGG and other model names | модель, сеть | prefer «модель BERT обучена»; bare: BERT использует … |

### 5.4 Mixed-script compounds

The rule of the glossary review (`ru-review.md` §4), stated for translators:

| Case | Rule | Examples |
|---|---|---|
| Latin word names or specifies a Russian noun | Postposed, no hyphen. This is the default. | функция softmax, модель roofline, маршрутизация top-k, слой ReLU, оптимизатор Adam, память GPU |
| Preposed Latin element | Hyphen, and only in the forms the glossary has. | softmax-регрессия, KV-кэш, score-функция, one-hot-кодирование, TD-ошибка, JIT-компиляция |
| Single letter, symbol or number + noun or adjective | Hyphen. | Q-обучение, n-грамма, p-значение, F1-мера, $k$-мерный |
| Never | A preposed Latin word with a space; a code span in a compound. | ✗ GPU память, ✗ top-k маршрутизация, ✗ `nn.Linear`-слой ✓ слой `nn.Linear` |

### 5.5 Verbs

Russian verbs for actions; a loan verb only where it is the written norm.

| English | Russian | Not |
|---|---|---|
| train, pretrain, fine-tune | обучать; предобучать (предобученная модель); дообучать | тренировать, файнтюнить, тюнить |
| learn (of a model) | обучаться; "learns a kernel" → выучивает ядро, находит ядро | учит ядро |
| overfit, underfit | переобучаться, недообучаться | оверфититься |
| sample | сэмплировать (from a model); statistics: извлекать выборку, брать наблюдение | семплить |
| update, initialize, normalize, regularize, tokenize, quantize, parallelize | обновлять, инициализировать, нормализовать (to unit norm: нормировать), регуляризовать, токенизировать, квантовать, распараллеливать | апдейтить |
| pad, pool, clip, mask, flatten | дополнять; применять пулинг; отсекать (градиент); маскировать; разворачивать в вектор | паддить, клипать |
| cache, shuffle, log, plot, debug, deploy | кэшировать; перемешивать; записывать в журнал; строить график; отлаживать; развёртывать | логать, шаффлить, деплоить |
| fit | обучать (a model); подгонять (a curve); "fits in memory" → помещается в память | фитить |
| stack (layers) | накладывать слои, ставить слой поверх; a stack → стопка слоёв | стекать |

### 5.6 The English term in parentheses

Add the English once per page (one numbered section of the book), at the first
occurrence of a key term in body text, as `(англ. …)`, lower case as in the
glossary, outside the emphasis markers. Check the English page in CONTEXT: if an
earlier unit of the page introduced the term, do not repeat it.

- Gloss Russian terms whose English differs: *взаимная корреляция* (англ.
  cross-correlation), *рецептивное поле* (англ. receptive field), *дополнение*
  (англ. padding), *переобучение* (англ. overfitting), *стратегия* (англ.
  policy).
- Do not gloss loans that transliterate the English (батч, токен, дропаут),
  cognates (регуляризация, градиент, тензор) or school vocabulary (производная,
  вероятность, среднее).
- With an acronym: *стохастический градиентный спуск* (англ. stochastic gradient
  descent, SGD); afterwards write what the English writes (SGD).
- Never in headings, captions, slides, exercises or table cells.

Reason: readers go on to English papers, documentation and code; «англ.» is the
form of Russian reference works, and once per page keeps the text clean. The
test translators wrote the bare English word («(cross-correlation)»); add
«англ.».

## 6. Numbers and units

**Decision (to confirm, §12.1): the decimal point everywhere, a space as
thousands separator.** Numerals are copied from the English (0.5, 92.3%), in
prose as in math, code, code output and figures. Reason: inline and display math
are protected and keep the point, and outputs and figure axes show it; a decimal
comma in prose would put two conventions into one sentence (✗ со скоростью
обучения $0.03$ потери падают до 0,01) and would need a rule for every
identifier (GPT-3.5, Python 3.12). `l10n check` also compares numerals as
written: `0,5` for `0.5` raises "numbers not found". This departs from ГОСТ and
school usage, as the German edition departs from DIN by the author's ruling;
all three test translators kept the point. With the point as decimal sign, a
point or comma must never group thousands.

| Item | Rule | Example |
|---|---|---|
| Decimal | As in the source, with a point. | "a dropout rate of 0.5" → вероятность дропаута 0.5 |
| Thousands | A space from five digits on; none in four-digit numbers and years. Never `10,000` or `10.000`. | "10,000 images" → 10 000 изображений ; 2048 ; 60 000 |
| Number words | million → миллион, **billion → миллиард**, **trillion → триллион**; тыс., млн, млрд only in tables and slides. | "3 billion parameters" → 3 миллиарда параметров |
| Digits or words | As the English: a number spelled out in English is spelled out in Russian, a digit stays a digit (the check compares digits). | "two hidden layers" → два скрытых слоя ; "10 epochs" → 10 эпох ; ✗ (test, Sonnet) "1 path" → одним путём |
| Noun after a numeral | Russian agreement: 1, 21 слой; 2–4, 22 слоя; 5–20, 100 слоёв; after a decimal fraction the genitive singular. | 3 нелинейные активации ; 42 элемента ; 0.5 эпохи ; 2.5 секунды |
| Endings on digits | None on a cardinal (✗ в 5-ти слоях ✓ в 5 слоях). An ordinal takes a hyphen and one letter after a vowel, two after a consonant: 5-й, 5-я, 5-е, 5-го, 5-му, 5-х. | "the 2010s" → 2010-е годы ; "in the 1990s" → в 1990-х годах |
| Number + adjective | In words up to ten (двумерный, трёхслойный, четырёхмерный), then digit and hyphen. "2D", "3D" as words. | "a 4D tensor" → четырёхмерный тензор ; 32-битный ; 100-слойная сеть |
| Ordinals | Written out up to «десятый»; otherwise digit with ending. | "the third epoch" → третья эпоха |
| Percent | No space before `%`, as in the source (TRANSLATING.md §4). Adjective: 5-процентный. "percentage points" → процентных пунктов. | "an accuracy of 92.3%" → доля правильных ответов 92.3% ✗ 92.3 % |
| Units | Symbols as in the source (Latin), with a space; time words written out become секунды, минуты, часы, дни. | 24 GiB, 5 ms ; "3 hours" → 3 часа |
| Ranges | En dash without spaces, unit once; after «от» write «до». | "40–55" → 40–55 ; "45–50%" → 45–50% ; ✗ от 40–55 ✓ от 40 до 55 |
| Multipliers | «в N раз(а)» in prose and slides, keeping the digit; `×` stays in tables, captions and math. "an order of magnitude" → на порядок. | "about 2× faster" → примерно в 2 раза быстрее ; "10× fewer" → в 10 раз меньше ✗ вдвое (drops the digit) |
| Approximation | примерно, около, порядка; «≈» and «~» only in tables and slides. | "roughly 90%" → около 90% |
| Years | "in 2017" → в 2017 году; a bare year in parentheses and tables. | ResNet появилась в 2015 году. |
| "for 10 epochs" | Accusative of duration, or «в течение». | обучаем 10 эпох ✗ обучаем для 10 эпох |
| Money | Amount, space, currency word, in the source's currency. | "\$2.50" → 2.50 доллара США |

Quantities, signs and orders of magnitude stay exact; do not round or convert
units or currencies. The only change to a numeral is the thousands grouping
(`10,000` → `10 000`).

## 7. Names

- **Models, libraries, datasets, companies, hardware, benchmarks, courses:**
  Latin script, exactly as in English: Fashion-MNIST, RTX 4090, OpenAI, GPT-2,
  Stanford CS231n.
- **People named in a glossary term:** Cyrillic, as the glossary: теорема
  Байеса, матрица Якоби, метод Ньютона, множитель Лагранжа, уравнение Беллмана,
  динамика Ланжевена, расстояние Вассерштейна, инициализация Хэ.
- **People in prose:** Cyrillic for scientists whose Russian form is settled in
  reference works, which is everyone the glossary names and the classical canon
  (Гаусс, Лежандр, Лаплас, Ньютон, Эйлер, Байес, Марков, Колмогоров, Чебышёв,
  Ляпунов, Шеннон, Тьюринг, Беллман). Latin script, undeclined, for contemporary
  researchers, as the citations print them (Hinton, LeCun, Vaswani); give them a
  noun where the case must show: «группа Hinton», «в работе LeCun и соавторов».
  Reason: a classical name in Latin script inside Russian text is wrong (Markov,
  Kolmogorov and Chebyshev are Russian names), and a home-made transliteration
  of a living author cannot be checked against the bibliography. (to confirm,
  §12.6)
- **Terms named after people:** the noun first and the name in the genitive
  (распределение Бернулли, норма Фробениуса, неравенство Йенсена, метод
  Монте-Карло), or the settled adjective, lower case (гауссовское распределение,
  марковская цепь, евклидова норма, байесовский вывод, липшицева функция,
  тёплицева матрица). Two surnames take an en dash (§2).
- **The book:** *Dive into Deep Learning* stays in English; "this book" → эта
  книга.
- **Languages and places:** Russian (английский язык, США); programming
  languages stay (Python).
- **Structure words:** chapter глава, section раздел, subsection подраздел,
  appendix приложение; Summary → Итоги, Exercises → Упражнения, Discussions →
  Обсуждение; Theorem → Теорема, Proposition → Утверждение, Lemma → Лемма,
  Corollary → Следствие, Definition → Определение, Proof → Доказательство,
  Remark → Замечание, Example → Пример. «Упражнения» and «Обсуждение» must
  equal the values in `ru/strings.yml` (§11), because the build looks for them.

## 8. Headings, captions, slides, exercises, notes

| Kind | Rule | Example |
|---|---|---|
| `heading` | Noun phrase, no full stop, no English gloss, sentence case. English "-ing + object" becomes a verbal noun + genitive. A question stays a question. | "Learning a Kernel" → Обучение ядра ; "Convolution as Matrix Multiplication" → Свёртка как умножение матриц ; "Why Depth Helps" → Почему помогает глубина |
| `caption` | Body register without «мы» / «вы»; a full stop exactly when the English caption has one. Keep `(a)`, `(b)` in Latin letters, as on the figure. No reference at the start (§3.2). | "Masks compose by logical AND." → Маски комбинируются логическим И. |
| `slide_title`, `kicker` | Noun phrase, as short as the English, no full stop. | `title="The Squared Error Loss Function"` → `title="Квадратичная функция потерь"` ; `[Measuring Misfit]{.kicker}` → `[Как измерить ошибку]{.kicker}` |
| `slide_text` | Short statements or noun phrases; no «мы», no «вы»; instructions as infinitives; one line where the English is one line; no full stop after a single-phrase bullet. A bold lead-in keeps its bold and colon, and the text after it starts in lower case (§2). | "**Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent." → **Обучение вместо ручного задания:** глубокие свёрточные сети сами находят тысячи таких ядер градиентным спуском ; "Lower the learning rate gradually" → Постепенно снижать скорость обучения |
| `exercise` | Imperative plural (the «вы» form): Покажите, Постройте, Выведите, Сравните; direct questions. A bold exercise title is translated and keeps its full stop. `[code]` and `●` are copied. Reason: it is the imperative of the body register; the infinitive (Показать, что …) is the style of exam sheets. | "Show that the errors sum to zero. Why?" → Покажите, что сумма ошибок равна нулю. Почему? |
| `tab_note` | Body register. Describe this framework only; never compare frameworks. API names stay in code. | "PyTorch's `nn.LazyLinear(1)` requires only the output dimension." → В PyTorch слою `nn.LazyLinear(1)` достаточно указать размерность выхода. |
| `note` | Register of its surroundings. Labels: **Note:** → **Примечание:**, **Warning:** → **Внимание:**, **Tip:** → **Совет:**, **Answers:** → **Ответы:**, "Concept Check" → Проверка понимания, "Trade-off" → Компромисс | |
| `table_row` | Noun phrases; numbers and units as in §6. | "Training time" → Время обучения |
| Source lines of exercises | "Adapted from X, …" → По материалам X, … ; course, assignment and book titles stay in English; "Chapter 6" → глава 6, "Homework 2" → домашнее задание 2. | |

The cover kicker `[Dive into Deep Learning · §2.1]{.kicker}` is rewritten by the
build; copy it unchanged.

## 9. Known pitfalls

### 9.1 Words with more than one sense

| English | Sense → Russian | Not |
|---|---|---|
| bias | parameter $b$ → смещение; of an estimator → смещение (смещённая / несмещённая оценка); regression intercept → свободный член; social → предвзятость | сдвиг for $b$; предубеждение |
| offset, shift | index or position offset, displacement → **сдвиг** (сдвиг распределения, эквивариантность относительно сдвига). Never «смещение», which is the bias. | ✗ (test, Fable, Opus) "centered offsets symmetrically" → смещения располагались симметрично, two sentences from «слой без смещения» ✓ сдвиги |
| kernel | convolution → ядро свёртки, then ядро (размер ядра); kernel methods, covariance → ядерная функция (матрица ядра); Markov → переходное ядро; of a matrix → ядро матрицы; GPU code → ядро (запуск ядра) | фильтр for a kernel (фильтр = filter); кернел |
| step, stride | convolution stride → шаг (шаг свёртки next to an optimization step); optimization → шаг, размер шага; in a sequence → временной шаг | страйд |
| padding | → дополнение (дополнение нулями); a set complement is also «дополнение»: near one write «дополнение нулями» | паддинг, набивка |
| return | RL → отдача; of a function → возвращаемое значение, возвращать | возврат for RL |
| score | attention, ranking → оценка; $\nabla \log p$ → score-функция; next to an estimate («оценка») write «балл» | скор, счёт |
| estimate, estimator | оценка (точечная оценка, несмещённая оценка); to estimate → оценивать; policy *evaluation* → оценивание стратегии | эстиматор |
| sample | a draw from a model → сэмпл; statistics → выборка (a set), наблюдение (one element); sample mean → выборочное среднее; to sample → сэмплировать | образец, семпл |
| example | a data point → пример (обучающий пример); "for example" → например | образец, экземпляр |
| feature | → признак (карта признаков, вектор признаков); of a library or product → возможность | фича, особенность, черта |
| policy, reward, action, state, value, environment | стратегия, награда, действие, состояние, ценность (функция ценности), среда | политика, вознаграждение, значение for RL value, окружение |
| loss, cost, objective, error, risk | Five words, five terms: потери (plural: потери падают) and функция потерь, затраты, целевая функция, ошибка, риск. | лосс; merging them |
| accuracy, precision, recall | доля правильных ответов, точность, полнота; floating-point *precision* → точность; numerical accuracy → точность | точность for classification accuracy |
| generalization | ML → обобщающая способность, обобщение (ошибка обобщения); *generalized* linear model → обобщённая линейная модель | генерализация |
| normalization | of activations or data → нормализация (батч-нормализация); to unit norm or sum → нормировка, нормировать (нормировочная константа); normal distribution → нормальное распределение; Gaussian → гауссовский | mixing them |
| training, learning | both → обучение (обучающая выборка, скорость обучения); the model «обучается», we «обучаем» | тренировка |
| inference | running a trained model → инференс; statistics → вывод (байесовский вывод); "the layer infers its input size" → определяет автоматически | вывод for model inference (it also means derivation and output) |
| output, derivation, conclusion | output of a layer → выход; program output → вывод; derivation → вывод (формулы); conclusion → вывод. Next to «выход» avoid «вывод» for the derivation: «мы получили формулу». | |
| translation | of an image or function → сдвиг (инвариантность относительно сдвига); of language → перевод | трансляция; инвариантность к переводу |
| hidden, latent | hidden layer, hidden state → скрытый; latent variable, latent space → латентный | скрытая переменная |
| gate, filter | LSTM gate → вентиль; convolution filter → фильтр | гейт |
| gradient, derivative | градиент; производная, "with respect to $x$" → по $x$ | деривация; градиент относительно |
| likelihood, evidence | правдоподобие; Bayesian model evidence → обоснованность (ELBO → вариационная нижняя оценка, as the glossary); "empirical evidence" → данные, свидетельства | вероятность; доказательство |
| expectation, mean, average | математическое ожидание ("in expectation" → в среднем, по математическому ожиданию); среднее; to average → усреднять | ожидание alone |
| variance, standard deviation | дисперсия; стандартное отклонение | вариация, разброс as a term |
| rank, trace, norm, span, support | ранг, след, норма, линейная оболочка, носитель; the verb "span" → охватывать | трасса, спан |
| performance | of a model → качество; of code or hardware → производительность, скорость | перформанс; производительность модели for accuracy |
| efficient, effective | efficient → эффективный, экономный; effective (receptive field, sample size) → эффективный; "effectively" → по сути, фактически | эффективно for "in effect" |
| label | → метка; of an axis → подпись оси | лейбл, ярлык |
| number | a count → число, количество; an identifier → номер | номер параметров |
| edge | in an image → граница (выделение границ, детектор границ); border of an image or window → край; of a graph → ребро | кромка |
| unit | of a layer → нейрон, элемент (of a feature map); of measurement → единица | юнит |
| patch, window | фрагмент (of the input in a convolution; патч for ViT image patches, as the glossary), окно | заплатка |
| converge | сходится **к** $x^*$ | конвергирует |
| smooth, convex, bounded, continuous | гладкий, выпуклый, ограниченный, непрерывный | плавный for smooth functions |

### 9.2 False friends and calques

| English | Russian | Not (because) |
|---|---|---|
| accurate | точный | аккуратный (= neat) |
| actual, actually | фактический, на самом деле | актуальный (= topical) |
| argument (of a discussion) | довод, рассуждение | аргумент only for a function's argument |
| data (plural) | данные | дата |
| decade | десятилетие | декада (= ten days) |
| dramatically | резко, значительно | драматически |
| eventually | в конце концов, со временем | эвентуально |
| expertise | опыт, квалификация | экспертиза (= examination) |
| figure | рисунок; a number → цифра, показатель | фигура |
| intelligent | интеллектуальный; AI → искусственный интеллект | интеллигентный |
| list (verb) | перечислять | листать |
| original | исходный | оригинальный (= unusual) |
| pattern | закономерность, шаблон, структура | паттерн, except in glossary terms (паттерн внимания) |
| principal (component) | главный | принципиальный |
| realize | понять; реализовать only "to implement" | |
| resolution | разрешение; of a problem → решение | резолюция |
| silicon | кремний | силикон |
| technique | метод, приём | техника (every time) |
| billion | миллиард | биллион |
| "makes sense" | имеет смысл, разумно | делает смысл |
| "in terms of" | через, в виде, с точки зрения (once) | в терминах (every time) |
| "it turns out" | оказывается | выясняется (every time) |
| arbitrary | произвольный | арбитражный |
| consistent | согласованный, единообразный; *consistent estimator* → состоятельная оценка | консистентный |
| robust, stable | робастный (робастность); устойчивый (численная устойчивость) | устойчивый for robust |

### 9.3 Fixed logical and mathematical phrases

because → потому что, так как, поскольку; therefore → поэтому, следовательно;
however → однако; only if → только если; if and only if → тогда и только тогда,
когда; at most → не более, не больше; at least → не менее, по крайней мере; "let
$x$ be …" → пусть $x$ — …; such that → такой, что; with respect to → по
(производная по $x$), относительно; without loss of generality → без
ограничения общности; for all → для всех, для любого; there exists → существует;
respectively → соответственно; almost surely → почти наверное; i.i.d. →
независимые и одинаково распределённые; bounded → ограниченный; dot product →
скалярное произведение; elementwise → поэлементный; "up to a constant" → с
точностью до константы; "it follows that" → отсюда следует, что; "holds" →
выполняется; "trivially" → очевидно; $\blacksquare$ stays.

### 9.4 Translationese

| Pattern | Write instead | Example |
|---|---|---|
| "Note that", "It is important to note that" every time | Drop it, or Заметим, что … once per paragraph | ✗ Важно отметить, что смещение не регуляризуется. ✓ Смещение при этом не регуляризуется. |
| Passive with an agent | Active | ✗ Нами было показано, что … ✓ Мы показали, что … |
| "we can" every time as «мы можем» | «можно», or the verb | "we can compute" → можно вычислить ; вычислим |
| "there is / there are" as «существует», «имеется» | A full verb, or «есть» | "There are three reasons." → На это три причины. |
| "is used to", "allows us to" | The verb; «позволяет» once | "This allows us to compute …" ✗ Это позволяет нам вычислить … ✓ Так мы вычислим … |
| Inanimate subject + "tells us", "sees", "wants" | Отсюда видно, Из этого следует; a model «получает на вход», «выдаёт» | "This result tells us that depth helps." ✗ Этот результат говорит нам, что … ✓ Отсюда видно, что глубина помогает. |
| "based on" as «основанный на» every time | на основе, по, исходя из | внимание на основе скалярного произведения |
| "using X" as «используя X» / «с использованием X» | с помощью X, the instrumental, or «по» | "using the chain rule" → по цепному правилу |
| Possessive chains, "its", "their" | Drop the possessive; Russian needs it only for contrast | "the layer and its weights" → слой и его веса ; "the model updates its parameters" → модель обновляет параметры |
| English order: subject first in every sentence | Known first, new last (§1) | |
| "simply", "just", "really", "very" word for word | Drop, or «достаточно» + infinitive | "we simply add" → достаточно прибавить ; прибавим |
| "different" for "various" | разные, различные; «другой» only for "another" | |
| "one of the most widely used" | один из самых распространённых; относится к самым распространённым | ✗ один из наиболее широко используемых |
| Idioms translated by image | The meaning | "under the hood" → внутри ; "from scratch" → с нуля ; "out of the box" → без настройки ; "sanity check" → проверка на правдоподобие ; "rule of thumb" → практическое правило |
| Title Case, English plural | Russian forms | ✗ Функция Потерь ✗ токенс |

## 10. Sources

Read for this draft, 2026-11-26:

- This repository: TRANSLATING.md, DESIGN.md, `tools/l10n/protect.py`,
  `glossary.py`, `config.py`; `glossary/terms/ru.jsonl` and
  `glossary/notes/ru-review.md`; `evals/ru-conv-layer/` (RESULTS.md and the
  answers of Fable, Opus and Sonnet); `de/style.md`, `ja/style.md`,
  `de/strings.yml`.
- Quarto's Russian language file in d2l-neu's build environment
  (`quarto_cli/share/language/_language-ru.yml`): `crossref-fig-title:
  Рисунок`, `crossref-tbl-title: Таблица`, `crossref-ch-prefix: Глава`,
  `crossref-sec-prefix: Глава`, `crossref-eq-prefix: Уравнение`,
  `section-title-references: использованная литература`; and Quarto's filter
  (`refPrefix`), which reads `crossref-<type>-prefix` from the language keys.
- d2l-neu `tools/i18n.py` (`quarto.<key>` overrides in the locale file),
  `tools/fix_crossref_numbers.py` (it matches the printed words by regular
  expression), `tools/d2l_preprocess.py` (a reference range is written with a
  literal English "to").

Not opened; stated from the drafter's knowledge and to be checked by a native
reviewer:

- The orthographic and typographic norms cited: Правила русской орфографии и
  пунктуации (1956), the 2006 academic handbook edited by V. V. Lopatin,
  Мильчин and Чельцова, *Справочник издателя и автора* (endings on numerals,
  abbreviations рис., табл., гл., разд., dashes), ГОСТ 7.0.12 and ГОСТ 8.417.
- Russian ML texts (the Russian editions of Goodfellow et al. and
  Sutton–Barto, Vorontsov's lectures, the Yandex ML handbook): no page was
  read. Term choices rest on the glossary review, which states the same limit.
- No Russian build was run: §3.1 describes what the overrides of §11 will
  print, derived from the code above, not an observed page.

## 11. Settings

`languages.yml`, existing keys:

- `check: {thousands: space}`: accepts `10,000` → `10 000` and warns on a kept
  English comma.
- `typography: {nbsp: true}`: the build puts no-break spaces into grouped
  numbers (10 000) and between a number and a Latin unit (24 GiB).
- `check: {order: warning}` (the default) stays: a reorder of math tokens is
  acknowledged by a NOTE on the block (§4).

`ru/strings.yml`, existing mechanism (`quarto.*` overrides, `book.*` keys):

- `quarto.crossref-fig-prefix: "рис."`, `quarto.crossref-tbl-prefix: "табл."`,
  `quarto.crossref-sec-prefix: "разд."`, `quarto.crossref-ch-prefix: "гл."`.
  §3 depends on these four. The caption words stay Quarto's («Рисунок 7.2»,
  «Таблица 7.1», keys `crossref-fig-title`, `crossref-tbl-title`).
- `quarto.section-title-references: "Литература"` (Quarto has
  «использованная литература», lower case); if the book uses Quarto callouts,
  `quarto.callout-note-title: "Примечание"` (Quarto has «Уведомление»).
- `book.exercises_heading: "Упражнения"`, `book.discussions_link:
  "Обсуждение"` (§7).
- `pdf.lang: "ru"`. The PDF text fonts (Source Serif 4, Source Sans 3) contain
  Cyrillic; the mono font Inconsolata in `static/fonts/` does not, which matters
  only if Cyrillic ever appears in a code span.

Settings that need new code or a test:

1. **Equation references as a bare number** (§3.2 rule 6): an empty
   `quarto.crossref-eq-prefix`. `fix_crossref_numbers.py` builds its pattern
   from the printed word; with an empty word the pattern `(?:)&nbsp;(\d+)…`
   would match any number after a no-break space, so the script needs a case
   for an empty prefix, and Quarto's output for an empty prefix (a leading
   no-break space) must be looked at in HTML and PDF.
2. **Reference ranges**: `d2l_preprocess.py` turns ``:numref:`a`--:numref:`b` ``
   into "@a to @b" with an English "to" (7 places in the book); it should come
   from the string catalog (Russian: «–» or «и далее до»).
3. **No-break spaces for Russian**: the `nbsp` rules know Latin abbreviations
   only. Russian needs a no-break space inside «т. е.», «и т. д.», «и т. п.»,
   «и др.», before an em dash, and after one-letter words (в, к, с, о, у, и,
   а).
4. **Decimal sign**: if the author rules for the decimal comma (§12.1), the
   check needs `check: {decimal: comma}` so that `0,5` matches `0.5`.
5. **Glossary matching of short words**: `glossary._present` lets a word of up
   to four letters only *add* letters, so «слой» does not match «слоя» or
   «слоёв», nor «ядро» → «ядер», «окно» → «окон». No effect today (every
   Russian entry is `proposed` and unenforced); needed before entries become
   `verified`.

## 12. For the author to confirm

1. **Decimal sign** (§6). Proposed: the point everywhere, as in the German
   ruling. The alternative is the Russian norm, a comma in prose (0,5; 92,3%)
   with the point kept in math and code; it needs setting 4.
2. **Reference prefixes as abbreviations** (§3.1): рис., табл., разд., гл.
   through `quarto.*` overrides, so that any preposition may precede a
   reference. The alternative is Opus's rule without settings beyond
   `crossref-sec-prefix: "Раздел"`: nominative words, references only in
   parentheses or as the subject, which prints a capitalised «Рисунок 7.2» in
   mid-sentence.
3. **Equation references** (§3.2 rule 6): the translator writes «формула»,
   «уравнение», «неравенство» and the build prints «(7.2.1)» only; needs
   setting 1. Fallback without new code: `quarto.crossref-eq-prefix:
   "уравнение"` and references to equations only as a direct object or after
   «в» + accusative (подставим уравнение 7.1 в …).
4. **Reader as «вы», lower case; exercises in the imperative plural** (§1,
   §8), against the infinitive of exam sheets (Показать, что …).
5. **ё always** (§1).
6. **People's names** (§7): Cyrillic for names in glossary terms and for the
   classical canon, Latin script for contemporary researchers. The German and
   Japanese guides keep every name as in English; Russian cannot do that for
   Markov, Kolmogorov or Chebyshev. A native reviewer should say whether
   contemporary names should also be transliterated (Хинтон, Лекун).
7. **Gender of Latin-script words** (§5.3): GPU masculine (процессор; spoken
   Russian also has it feminine, after «видеокарта»), networks feminine (CNN,
   LSTM), model names feminine with the head noun preferred (модель BERT
   обучена; usage also has «BERT обучен»), softmax masculine.
8. **English gloss as `(англ. …)`** (§5.6), where the three test translators
   wrote the bare word.
9. **Math token order** (§4): Russian order with a NOTE, as Fable did. If the
   NOTES become noise, `check: {order: off}` for `ru` removes the warning.
10. **Structure words** (§7): «Итоги» for Summary (Fable wrote «Резюме», Opus
    «Итоги»), «Обсуждение» in the singular for the forum link (all three wrote
    «Обсуждения»), «Утверждение» for Proposition (also «Предложение»).
11. **Lower case after a bold lead-in with a colon** (§2, §8), against Fable's
    capital copied from the English slides.
12. **Loans** (§5.1): «инференс» for running a model (the glossary has it only
    in «стоимость инференса»), «фреймворк», «ноутбук» (Jupyter); «датасет»,
    «паддинг», «страйд» rejected for «набор данных», «дополнение», «шаг».
13. **Percent without a space** (92.3%), as TRANSLATING.md §4 and the German
    ruling; ГОСТ 8.417 asks for a space.
14. **The glossary's open items** (`ru-review.md` §7), above all «отдача» or
    «доход» for the RL return and «сэмпл» in statistical passages; this guide
    follows the glossary as it stands. «обоснованность» for Bayesian evidence
    (§9.1) has no entry of its own yet.

## 13. Proposed glossary corrections

None of these was applied; the glossary is unchanged.

1. **`bias` alternatives: remove «сдвиг».** The guide reserves «сдвиг» for
   offset, shift and translation (сдвиг распределения, инвариантность
   относительно сдвига); as an alternative for the layer's bias it brings back
   the collision seen in the test (§9.1). Likewise `intercept`: drop the
   alternative «сдвиг».
2. **Add concepts `offset` and `shift` (positional) → «сдвиг»**, with the note
   "never «смещение»".
3. **Add `inference` (running a trained model) → «инференс»**, alternative
   «применение модели»; today only «стоимость инференса» carries the word,
   while the four statistical entries use «вывод».
4. **`max-pooling` «макс-пулинг» next to `average-pooling` «усредняющий
   пулинг»**: one pattern for the pair. Proposed: «пулинг по максимуму» and
   «пулинг по среднему» as preferred, or keep «макс-пулинг» and note the
   asymmetry; a native reviewer should choose.
5. **Slang alternatives** («лосс», «фича», «таргет», «трейн», «скор»,
   «бэкпроп», «файнтюнинг», «датасет», «паддинг», «страйд»): mark them
   `avoid` or move them to notes. As alternatives they count as acceptable
   forms for the glossary check, and §5.1 forbids them.
6. **Add the frequent terms without an entry** that the test translators had
   to decide: framework → фреймворк; notebook → ноутбук; edge (image) →
   граница; patch (of a convolution input) → фрагмент; unit (of a feature map)
   → элемент; dense layer → полносвязный слой (as an alias of the
   existing `fully connected layer`); banded matrix
   → ленточная матрица; Toeplitz matrix → тёплицева матрица; locally connected
   layer → локально связный слой (Fable «локально-связный», Opus «локально
   связанный», Sonnet «локально связный»; the adverb + adjective pair is
   written apart).
7. **`floating-point`**: «плавающая запятая» stands beside this guide's
   decimal point (§6). Keep the entry, which is the standard term, but if the
   author confirms the decimal point, add the note that the term does not
   describe the book's notation.
