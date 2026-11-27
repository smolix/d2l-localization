# Russian (`ru`): evidence from the 2021 stabuev/d2l-ru translation

Source id `d2l-ru-stabuev` ([stabuev/d2l-ru](https://github.com/stabuev/d2l-ru), branch
`main`, pinned at commit `3d5548ab83f46e3f13e9df8791c08f1df748b8ec`, accessed 2026-11-27).
Mined 2026-11-27. Only `terms/ru.jsonl` and this source were touched.

## What the source covers

The repository imports the English book in its first commit (`5d9e696`, 2021-09-23; the
README names v0.17.0) and then replaces English text in place, 45 commits in all up to
2021-10-31. The English originals were taken from that first commit and aligned paragraph
by paragraph with the HEAD text; the alignment is one-to-one except for two English
paragraphs the translator split in two.

Translated passages:

- `chapter_introduction/index.md`: the first 68 English paragraphs (about 5,400 English
  words): the opening, A Motivating Example, Key Components (Data, Models, Objective
  Functions, Optimization Algorithms), and Kinds of Machine Learning Problems up to the
  Regression, Classification and Tagging openings. From line 166 on the file is English.
- `chapter_appendix-mathematics-for-deep-learning/index.md`: the whole index page (about
  400 words).
- `chapter_linear-networks/index.md`: the one-paragraph chapter introduction.
- `chapter_linear-networks/linear-regression.md`: title and first paragraph only.
- `README.md`, the book title in `index.md`, and the code-of-conduct paragraph; these hold
  no glossary terms beyond those listed below.

Everything else, about 840 files, is untouched English. The source is useful only for
early-chapter vocabulary.

## Human or machine translation

There are two hands, and they differ.

**Introduction and appendix index (Soslan Tabuev / `stabuev`): machine draft, post-edited
by hand.** 29 commits, all "Update index.md", each replacing one to four paragraphs
through the GitHub web editor over five weeks. The paragraphs carry artifacts that a
machine translation engine produces and a person does not type:

- spaces injected into markup and URLs: `` : numref: `fig_wake_word` `` (L23, L36, L140,
  L149), `[приз Netflix в 1 миллион долларов] (https: //en.wikipedia.org/...)` (L120),
  `[Линнею] (https://...)` (L151);
- decimal commas inside LaTeX: `$0,2 \times \infty + 0,8 \times 0 = \infty$` (L149);
- a sentence split leaving a capital after a comma: «…в течение следующего года?), В
  некоторых случаях…» (L107);
- word-for-word calques: «Но мы еще не совсем там» ("we are not quite there yet",
  appendix L8), «Взятые вместе» ("Taken together", appendix L22), «Приземлим описание на
  конкретный пример» ("To ground this description", L97), «в широкие отрасли» ("to broad
  industries", README);
- one leftover default rendering of a term the editor fixed elsewhere: «задачах
  контролируемого обучения» (L59) next to «обучением с учителем» (L50).

A human then edited the drafts: typing slips no engine makes («Напишисать» three times at
L11–13, «ионял», «чатсти», «анлогичного», «развовьем», «результатя»,
«пытаемсяминимизировать»), two paragraphs split, examples localized ("Apricot" →
«Абрикос», "garter" → «ужа», "Hey Siri" → «Привет, Сири»), and field terms chosen over literal ones (*eigen-decompositions* →
«спектральных разложений», *features (or covariates)* → «признаками (или ковариатами)»).
Evidence from these passages was added but the entries **stay `proposed`**.

**Linear-networks index and linear-regression opening (Denis Abrashin, PR #1, commits
"Первичный перевод на русский", "Переведен параграф Linear Regression"): human
translation.** The text is rewritten freely, with additions ("the entire training process"
→ «весь типовой процесс обучения»; "defining simple neural network architectures" →
«построение архитектуры нейронной сети на примере несложных моделей») and human misreadings
an engine would not make ("stocks" → «имущества»; "softmax regression" → «логистическая
регрессия»). There are no markup artifacts. Terms that match the glossary here were set to
**`verified_d2l`**.

## Results

- verified_d2l: 7 (from the human pages);
- attested, kept `proposed`: 34 (preferred term matches, post-edited passages);
- alternative: 3 (`backpropagation` gains «обратное распространение»; `gaussian-noise`
  «гауссов шум» and `sample` «образец» were already alternatives and gain evidence and a
  note);
- skipped: 13 rows below;
- conflicts: 0 (no Russian entry was verified before this pass).

Concepts matched in English only by an unrelated sense were not recorded: *key* (key
properties), *value* (numerical values), *state* (user's state), *action* (program
action), *attention* ("deserves attention"), *span* ("in the span of"), *shape* (image
shape), *advantage*, *cost*, *score* (exam scores), *mean* (verb), *sampling* ("a sampling
of problems"), *hypothesis* (NLI sense vs. "hypothesis tests"), and *likelihood* (only
inside "maximum likelihood", credited to `maximum-likelihood`).

| Concept | Glossary preferred | Source rendering | Action | Locator |
| --- | --- | --- | --- | --- |
| `neural-network` | нейронная сеть | нейронных сетей | verified | [linear-networks index L4](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_linear-networks/index.md#L4) |
| `linear-regression` | линейная регрессия | Линейная регрессия | verified | [linear-regression L6](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_linear-networks/linear-regression.md#L6) |
| `regression` | регрессия | регрессии | verified | [linear-regression L9](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_linear-networks/linear-regression.md#L9) |
| `classification` | классификация | задачи классификации | verified | [linear-regression L9](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_linear-networks/linear-regression.md#L9) |
| `loss-function` | функция потерь | функции потерь | verified | [linear-networks index L4](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_linear-networks/index.md#L4) |
| `model` | модель | обучения модели | verified | [linear-networks index L4](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_linear-networks/index.md#L4) |
| `training` | обучение | процесс обучения | verified | [linear-networks index L4](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_linear-networks/index.md#L4) |
| `feature` | признак | признаками | attested, kept proposed | [intro L59](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_introduction/index.md#L59) |
| `label` | метка | метка | attested, kept proposed | [intro L59](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_introduction/index.md#L59) |
| `dataset` | набор данных | набор данных | attested, kept proposed | [intro L28](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_introduction/index.md#L28) |
| `training-set` | обучающая выборка | обучающую выборку | attested, kept proposed | [intro L83](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_introduction/index.md#L83) |
| `test-set` | тестовая выборка | тестовую выборку | attested, kept proposed | [intro L83](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_introduction/index.md#L83) |
| `model-parameter` | параметр модели | параметров модели | attested, kept proposed | [intro L83](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_introduction/index.md#L83) |
| `prediction` | предсказание | предсказанием | attested, kept proposed | [intro L23](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_introduction/index.md#L23) |
| `objective-function` | целевая функция | целевыми функциями | attested, kept proposed | [intro L79](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_introduction/index.md#L79) |
| `squared-error` | квадратичная ошибка | квадратичную ошибку | attested, kept proposed | [intro L81](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_introduction/index.md#L81) |
| `gradient-descent` | градиентный спуск | градиентный спуск | attested, kept proposed | [intro L87](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_introduction/index.md#L87) |
| `optimization` | оптимизация | оптимизации | attested, kept proposed | [intro L79](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_introduction/index.md#L79) |
| `gradient` | градиент | градиент | attested, kept proposed | [math-appendix index L14](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_appendix-mathematics-for-deep-learning/index.md#L14) |
| `vector` | вектор | векторов | attested, kept proposed | [intro L63](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_introduction/index.md#L63) |
| `linear-algebra` | линейная алгебра | линейной алгебры | attested, kept proposed | [math-appendix index L12](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_appendix-mathematics-for-deep-learning/index.md#L12) |
| `probability` | вероятность | вероятность | attested, kept proposed | [intro L136](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_introduction/index.md#L136) |
| `random-variable` | случайная величина | случайных величин | attested, kept proposed | [math-appendix index L16](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_appendix-mathematics-for-deep-learning/index.md#L16) |
| `maximum-likelihood` | максимальное правдоподобие | максимальным правдоподобием | attested, kept proposed | [math-appendix index L6](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_appendix-mathematics-for-deep-learning/index.md#L6) |
| `entropy` | энтропия | энтропия | attested, kept proposed | [math-appendix index L10](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_appendix-mathematics-for-deep-learning/index.md#L10) |
| `cross-entropy` | кросс-энтропия | кросс-энтропия | attested, kept proposed | [intro L138](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_introduction/index.md#L138) |
| `overfitting` | переобучение | переобучена | attested, kept proposed | [intro L83](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_introduction/index.md#L83) |
| `channel` | канал | каналов | attested, kept proposed | [intro L61](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_introduction/index.md#L61) |
| `computer-vision` | компьютерное зрение | компьютерное зрение | attested, kept proposed | [intro L17](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_introduction/index.md#L17) |
| `sequence` | последовательность | последовательностей | attested, kept proposed | [intro L105](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_introduction/index.md#L105) |
| `confidence-interval` | доверительный интервал | доверительных интервалов | attested, kept proposed | [math-appendix index L18](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_appendix-mathematics-for-deep-learning/index.md#L18) |
| `supervised-learning` | обучение с учителем | обучением с учителем | attested, kept proposed | [intro L50](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_introduction/index.md#L50) |
| `pixel` | пиксель | пикселя | attested, kept proposed | [intro L61](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_introduction/index.md#L61) |
| `image-recognition` | распознавание изображений | распознавании изображений | attested, kept proposed | [intro L15](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_introduction/index.md#L15) |
| `observation` | наблюдение | наблюдение | attested, kept proposed | [intro L97](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_introduction/index.md#L97) |
| `binary-classification` | бинарная классификация | бинарной классификацией | attested, kept proposed | [intro L134](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_introduction/index.md#L134) |
| `multiclass-classification` | многоклассовая классификация | многоклассовой классификацией | attested, kept proposed | [intro L138](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_introduction/index.md#L138) |
| `feature-vector` | вектор признаков | вектор признаков | attested, kept proposed | [intro L116](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_introduction/index.md#L116) |
| `information-theory` | теория информации | теории информации | attested, kept proposed | [math-appendix index L20](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_appendix-mathematics-for-deep-learning/index.md#L20) |
| `eigendecomposition` | спектральное разложение | спектральных разложений | attested, kept proposed | [math-appendix index L12](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_appendix-mathematics-for-deep-learning/index.md#L12) |
| `loss` | потери | потери | attested, kept proposed | [intro L83](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_introduction/index.md#L83) |
| `backpropagation` | обратное распространение ошибки | обратное распространение | alternative | [math-appendix index L14](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_appendix-mathematics-for-deep-learning/index.md#L14) |
| `gaussian-noise` | гауссовский шум | гауссовым шумом | alternative | [intro L128](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_introduction/index.md#L128) |
| `sample` | сэмпл | образец | alternative | [intro L59](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_introduction/index.md#L59) |
| `softmax-regression` | softmax-регрессия | логистическая регрессия | skipped: wrong sense («linear and softmax regression»; binary logistic regression is not the multiclass model) | [linear-networks index L4](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_linear-networks/index.md#L4) |
| `softmax` | softmax | (inside «логистическая регрессия») | skipped: same occurrence, no rendering of softmax | [linear-networks index L4](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_linear-networks/index.md#L4) |
| `statistical-learning` | статистическое обучение | статистической теории обучения | skipped: names a different concept (statistical learning theory) | [linear-networks index L4](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_linear-networks/index.md#L4) |
| `multi-label-classification` | классификация с несколькими метками | классификация позволяющая присвоить несколько классов одному элементу | skipped: paraphrase, not a term | [intro L151](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_introduction/index.md#L151) |
| `conditional-probability` | условная вероятность | вероятности метки при условии входных характеристик | skipped: paraphrase, the term itself is not used | [intro L99](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_introduction/index.md#L99) |
| `ground-truth` | истинная разметка | реальным значением / с истиной / верная метка / истинные метки | skipped: varying paraphrases, no stable term | [intro L81](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_introduction/index.md#L81) |
| `variance` | дисперсия | некоторая разница | skipped: mistranslation («some of the variance owes to…») | [intro L127](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_introduction/index.md#L127) |
| `gradient-flow` | градиентный поток | движение градиента | skipped: different sense (gradient signal through a network, not the continuous-time flow) | [math-appendix index L10](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_appendix-mathematics-for-deep-learning/index.md#L10) |
| `hypothesis-testing` | проверка гипотез | тестирований гипотез | skipped: calque of «hypothesis tests» | [math-appendix index L18](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_appendix-mathematics-for-deep-learning/index.md#L18) |
| `dataset` | набор данных | датасете | skipped as alternative: colloquial loan removed by the 2026-11-26 style guide (§5.1); preferred is attested at L28 | [intro L116](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_introduction/index.md#L116) |
| `sample` | сэмпл | сэмплов | skipped: audio samples (amplitude measurements), a different sense from one draw of a distribution | [intro L23](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_introduction/index.md#L23) |
| `environment` | среда | окружающей средой | skipped: everyday phrase «окружающая среда» in a general sentence, not a term occurrence | [intro L17](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_introduction/index.md#L17) |
| `supervised-learning` | обучение с учителем | контролируемого обучения | skipped: one MT-residue occurrence; preferred is attested at L50 | [intro L59](https://github.com/stabuev/d2l-ru/blob/3d5548ab83f46e3f13e9df8791c08f1df748b8ec/chapter_introduction/index.md#L59) |

## Candidates (no concept in the catalog)

| English | Source rendering | Comment |
| --- | --- | --- |
| machine learning | машинное обучение | consistent throughout |
| deep learning | глубокое обучение | consistent throughout |
| example / data point / data instance | пример / точка данных / экземпляр данных | intro L59 |
| covariate | ковариата | intro L59 |
| target (of a label) | целевое значение | intro L59 |
| learning algorithm | алгоритм обучения | intro L30 |
| model family | семейство моделей | intro L30 |
| surrogate objective | суррогатная целевая функция | intro L81 |
| error rate | частота ошибок, then коэффициент ошибок | intro L81, two renderings in one paragraph |
| dimensionality | размерность | intro L63 |
| classifier / regressor | классификатор / регрессор | intro L134 |
| hierarchical classification | иерархическая классификация | intro L151 |
| tagging | Разметка | intro L155; wrong sense (annotation, not multi-label tagging) |
| wake word | слово пробуждения / пробуждающее слово | intro L23, L50; inconsistent |
| estimator | оценщик | appendix L18; calque, statistics uses «оценка» |
| steepest descent | наибыстрейший спуск | appendix L14 |
| naive Bayes classifier | наивный байесовский классификатор | appendix L16 |
| natural language processing | обработка естественного языка | intro L17, README |

`rating-prediction` (предсказание рейтингов) is close: L120 renders "Predicting the rating
that a user will assign" as «Предсказание рейтинга», but the English term itself does not
occur, so no evidence was added.
