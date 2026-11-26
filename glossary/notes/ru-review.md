# Russian (`ru`) glossary: terminology review

Reviewed 2026-11-26 by Fable (Russian terminology reviewer) on the Opus draft of
`terms/ru.jsonl`. All 1,191 entries were read against the concept's English term and
`sense`. 141 entries changed: 73 got a new preferred term, 68 changed only in
alternatives or notes. 1,050 entries stand as drafted.

**Evidence status.** No source was opened during this review. Every entry stays
`proposed` with empty `source_ids` and `evidence`. Statements below such as "the Russian
Sutton–Barto edition uses …" or "course usage" are the reviewer's recollection of the
literature (Russian editions of Goodfellow et al. and Sutton–Barto, Vorontsov's lectures,
the Yandex ML handbook, Russian Bayesian-methods and RL lecture notes). They are reasons
for a choice, not attestations, and none of them has been checked against a page.

The corrections are applied by a script from the untouched draft, so they can be
re-derived: draft + script = current file.

## 1. Family consistency

| Family | Decision | Why |
| --- | --- | --- |
| batch | **батч** everywhere: батч, мини-батч, размер батча, **батч-нормализация** (was пакетная нормализация), объединение в батчи (was батчинг), градиентный спуск по мини-батчам | The draft mixed «батч» and «пакет». «батч» is the written norm of current courses; «пакет…» forms stay as alternatives. |
| cross- | **кросс-энтропия**, кросс-энтропийная функция потерь, бинарная кросс-энтропия, кросс-валидация | One morpheme for the two terms the draft had split; «кросс-» also gives the adjective «кросс-энтропийный». Two reasoned exceptions: **взаимная корреляция** (established signal-processing term) and **перекрёстное внимание** (translated in parallel with «самовнимание»); both carry the «кросс-» form as an alternative. |
| encoder | энкодер, декодер, **автоэнкодер**, **вариационный автоэнкодер** (were автокодировщик) | The draft had «энкодер» next to «автокодировщик». |
| gates | **вентиль забывания, входной/выходной вентиль, вентиль обновления, вентиль сброса** (were «фильтр …») | «фильтр» is the convolution filter in this book; «вентиль» is the textbook word for a gate, «гейт» the colloquial loan. |
| latent | **латентная переменная**, модель с латентными переменными, латентное пространство, латентный код/фактор | Keeps «скрытый» for hidden layers and hidden states. |
| denoising | **расшумление**, целевая функция расшумления, условное расшумление, член расшумления; денойзер | «шумоподавление» is the audio/DSP term (noise suppression); «расшумление» pairs with «зашумление». |
| decay | **затухание** in all four: весов, скорости обучения (was уменьшение), косинусное, экспоненциальное | Same word for the same idea. |
| sampling | **сэмплирование** in every compound, including **сэмплирование по значимости** and **сэмплирование с отклонением** (were «выборка …»); сэмплирование по расписанию | «выборка» is kept for a set of observations (обучающая выборка). The classical «выборка по значимости / с отклонением» are alternatives. |
| policy (RL) | стратегия; **градиент стратегии** (was градиент по стратегии), **оценивание стратегии** (was оценка) | Genitive as in «теорема о градиенте стратегии»; «оценивание» avoids the overloaded «оценка». |
| reward | награда; **модель награды** (was модель вознаграждения) | «марковский процесс вознаграждения» stays as a fixed name, noted as the exception. |
| invariance | **инвариантность / эквивариантность относительно сдвига** (were «к сдвигу») | Normative government; the shorter «к сдвигу» is an alternative. |
| bottleneck | **бутылочное горлышко**, блок с бутылочным горлышком, информационное бутылочное горлышко | The draft had «узкое место», «узкое горлышко» and «бутылочное горлышко» for one image. |
| skip connection | **обходная связь** (was пропускающая связь), тождественная обходная связь | The draft already used «обходная» in the identity-shortcut entry. |
| rating | **предсказание рейтингов**, **матрица рейтингов** (were «оценок») | «оценка» already means estimate and score. |
| guidance | **управление с помощью классификатора**, **управление без классификатора**, коэффициент управления | One root, with a preposition that cannot be misread. |
| patch | **патч изображения** (was фрагмент), эмбеддинг патча | |
| region proposal | **регион-кандидат**, сеть генерации регионов-кандидатов | |
| checkpointing | сохранение контрольных точек, **сохранение контрольных точек активаций** (was чекпоинтинг активаций) | |

## 2. Wrong sense or misleading form

- `independent-samples`: «независимые выборки» means independent datasets; now **независимые наблюдения**.
- `scaled-dot-product-attention`, `dot-product-attention`: «скалярное внимание» reads as "scalar attention"; now **внимание на основе (масштабированного) скалярного произведения**.
- `attention-pooling`: «пулинг внимания» reads as "pooling of attention"; now **пулинг на основе внимания**.
- `classifier-guidance`: «управление классификатором» reads as "controlling the classifier".
- `region-proposal`: the concept is one candidate region, not the act of proposing.
- `bottleneck`: «узкое место» is the performance sense; the concept is a narrow representation.
- `maximum-entropy`, `maximum-a-posteriori`: the concepts are principles; now **принцип максимума энтропии** and **максимум апостериорной вероятности**.
- `robustness`: **робастность**, because «устойчивость» is stability elsewhere in the catalog.
- `gradient-clipping`: **отсечение градиента**, because «обрезка» is image cropping.
- Alternatives removed as different concepts: «полносвязная сеть» (feedforward network), «энграмма» (n-gram), «буфер воспроизведения» (experience replay), «неявная регуляризация» (implicit bias), «правило Армихо» (backtracking), «достаточное условие экстремума» (second-order condition), «принудительное обучение» (teacher forcing), «функция оценки» (score function), «шумоподавитель» (denoiser).

## 3. Better standard term

- `backtracking-line-search`: **дробление шага**, the term of Russian optimization textbooks.
- `entropy-rate`: **энтропийная скорость** (the draft had a paraphrase).
- `matrix-factorization`: **матричная факторизация** (recommender-systems usage).
- `concept-drift`: **дрейф концепта**; `covariate-shift`: **сдвиг ковариат**.
- `cycle-consistency`: **циклическая согласованность**.
- `gradient-penalty`: **градиентный штраф**; `warmup`: **прогрев**; `time-step`: **временной шаг**.
- `temporal-difference-error`: **TD-ошибка**.
- `proposal-distribution`: **предложное распределение**.
- `multivariate-gaussian`: **многомерное гауссовское распределение** (Gaussian → гауссовский, normal → нормальный).
- `training-throughput`: **производительность обучения**; `training-compute`: **вычислительные затраты на обучение**.
- `multi-fidelity-optimization`: **оптимизация с несколькими уровнями точности** (the draft «многоточностная» was a coinage).
- `byte-pair-encoding`: **кодирование пар байтов**; `grouped-query-attention`: **внимание с группировкой запросов**; `class-conditioning`: **обусловливание по классу**; `decoupled-weight-decay`: **отделённое затухание весов**.

## 4. Grammar, lemma, spelling

- `minibatch-gradient-descent`: «мини-батч градиентный спуск» is not grammatical; now **градиентный спуск по мини-батчам**.
- `monte-carlo`: noun lemma **метод Монте-Карло**. `tokens-per-second`: nominative **токены в секунду**.
- `lipschitz-continuous`: the draft listed the feminine «липшицева» as an alternative to «липшицев»; replaced by real synonyms.
- `he-initialization`: **инициализация Хэ** (the Chinese surname He is «Хэ»; «Хе» is a common misspelling).
- `dilated-convolution`: alternative respelled «разреженная свёртка» (the adjective has no ё).
- `truncated-backpropagation-through-time`: now parallel to the untruncated entry (…распространение **ошибки** во времени).
- Mixed-script compounds: Latin element postposed (**маршрутизация top-k**, like «модель roofline», «нормализация softmax») or hyphenated when preposed (multi-label-классификация, multi-fidelity-оптимизация, KV-кэш, softmax-регрессия). The unhyphenated «top-k маршрутизация», «bias-variance разложение», «image-to-image трансляция» are gone.
- ё was checked across all terms; no missing ё found.

## 5. Latin-script retention

The 45 Latin-only preferred terms were reviewed and kept: they are names of models,
optimizers, metrics and primitives (Adam, BERT, DDPM, FID, all-reduce, RMSNorm, …) or
terms Russian prose really keeps (softmax, ReLU, teacher forcing, off-policy/on-policy,
score matching, flow matching, log-sum-exp). One was added: `sequence-to-sequence` →
**seq2seq** (the draft's hyphenated «последовательность-в-последовательность» is not
Russian typography; the spelled-out form in quotation marks is the alternative). One was
moved the other way: `prefill` → **предзаполнение**, with the Latin form as alternative.

## 6. Decisions on the drafter's uncertain list

1. **Batch normalization**: батч-нормализация; whole family on «батч». Changed.
2. **Accuracy**: доля правильных ответов, «точность» reserved for precision. Kept.
3. **Cross-entropy**: кросс-энтропия, consistent with кросс-валидация. Changed.
4. **Encoder/decoder**: энкодер/декодер kept; autoencoder → автоэнкодер. Changed.
5. **Embedding**: эмбеддинг. Kept; «вложение» noted as the mathematical embedding.
6. **Dropout**: дропаут. Kept; «прореживание» is avoided because it renders subsampling.
7. **Weight decay**: затухание весов. Kept; decay family aligned to it.
8. **Policy**: стратегия. Kept; two compounds regularized.
9. **Return**: отдача. Kept, flagged (see below).
10. **Fine-tuning**: дообучение. Kept.
11. **Self-supervised learning**: обучение с самоконтролем. Kept; «самоконтролируемое обучение» added as first alternative; flagged.
12. **Sample / sampling**: сэмпл / сэмплирование. Kept, and the family extended to importance and rejection sampling.
13. **Gates**: вентиль. Changed (was фильтр).
14. **Padding / pooling**: дополнение / пулинг. Kept.
15. **Chain rule**: цепное правило. Kept.

Also tentative in the draft: **teacher forcing** kept in Latin script; **skip connection** →
обходная связь; **seq2seq** → Latin short form; **object detection** детекция объектов kept
(pairs with «детектор»); **acquisition function** функция приобретения kept, flagged.

Sense splits confirmed as drafted: bias / statistical bias both «смещение», intercept
«свободный член»; the five kernels (ядро свёртки, ядерная функция, переходное ядро, ядро
матрицы, запуск ядра); матрица Якоби vs якобиан; импульс; исчезновение порядка; бутстрэп vs
бутстрэппинг; оценка (score) vs score-функция. Shared preferred terms confirmed:
dot product / inner product, critic (RL) / critic (GAN), training dynamics / learning
dynamics. The `score` note now says what to write when an estimate («оценка») is nearby.

## 7. Entries that most need a native-speaker check

- `return`: отдача vs доход (the Russian Sutton–Barto edition, as recalled, has «доход»).
- `self-supervised-learning`: обучение с самоконтролем vs самоконтролируемое обучение vs самообучение.
- `classifier-guidance`, `classifier-free-guidance`, `guidance-scale`: «управление» vs «наведение» vs keeping the English name.
- `proposal-distribution`: предложное vs пробное распределение.
- `importance-sampling`, `rejection-sampling`: «сэмплирование …» (family consistency) vs the classical «выборка …».
- `prefill`: предзаполнение vs prefill.
- `acquisition-function`: функция приобретения.
- `control-variate`: контрольная переменная.
- `parallel-scan`, `selective-scan`: «сканирование» for scan.
- `exposure-bias`: смещение экспозиции.
- `sample`: «сэмпл» as a preferred written term in statistical passages.
- `score`: «оценка» against «балл» where estimates are discussed.
- `memory-bound`, `compute-bound`: ограниченный памятью / вычислениями.
- `multi-query-attention`: многозапросное внимание.
- `wasserstein-distance`: Вассерштейна (dominant) vs Васерштейна (the mathematician's own spelling).
- `hinge-loss`: кусочно-линейная функция потерь vs шарнирная / hinge.
