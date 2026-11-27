# Spanish: evidence from `acca3003/d2l-es` (2021) and the Datitos course

Mined 2026-11-27. Locale file: `glossary/terms/es.jsonl`. Sources: `glossary/sources/d2l-es-acca3003.jsonl`,
`glossary/sources/datitos-curso-ap.jsonl`. Read with `glossary/review/es.md`: the earlier Spanish `verified_external`
entries rest on the machine-translated Google glossary or the crowd-translated Stanford cheatsheets.

## What `acca3003/d2l-es` covers

Repository `acca3003/d2l-es`, branch `master`, pinned at `1d65b3d1e8b9789f1d9d4b00bc2a0fbaf1c9cd9a` (2021-08-23).
It is the full d2l-en history up to `25fe659e` (2021-08-11, "Merge pull request #1880") followed by 11 commits by
one translator, Alfonso Carabantes Alamo, all titled "Update <file>.md" (GitHub web editor), 2021-08-12 to 08-23.
`git diff 25fe659e HEAD` touches six files and about 2,200 words:

| file | translated |
|---|---|
| `chapter_notation/index.md` | all (about 630 words): the notation list for numbers, sets, functions, calculus, probability |
| `chapter_installation/index.md` | all (about 660 words) |
| `chapter_preface/index.md` | the title, the first two paragraphs, two headings; the rest English |
| `README.md` | most prose and headings; endorsements and paper list English |
| `chapter_introduction/index.md` | headings only (30 headings); all body text English |
| `chapter_preliminaries/linear-algebra.md` | the opening paragraph and the headings; body English |

Every other chapter is the unmodified English book. Translated passages were aligned with the English at `25fe659e`
(the translator replaced the English in place, so the diff pairs them line by line).

## Human or machine

Human, by one translator, and unrevised. The evidence:

- Keystroke typos that machine translation does not make: «Introduccción», «resperesenta», «isntala» / «isntalado»
  (three times), «paqeute», «tensro», «vecotres», «cojunto», «entropyia», «procesameinto», «servidor con GPR»,
  «PRoductos Dot».
- English left beside the Spanish where the translator typed over the original: «Este libro de código abierto
  resperesenta nuestro intento de hacer el deep learning This open-source book represents our attempt to make deep
  learning accesible»; «Libro Interactivo de Deep Learning Book con Código Multi-Framework Code»; «una función de
  densidad (PDF) associated with distribution P»; «ejecuta:run `conda deactivate`».
- Errors of a person writing quickly, not of an MT system: «Secuencia de Aprendizaje» for *Sequence Learning*
  (word order, so "a sequence of learning"); «$\ell_p$ norma» (English order); «la KL-divergencia»; redundant
  subject pronouns copied from English («Tu deberías», «tu descargarías»); literal idioms («disciplina académica de
  cielos despejados» for *blue skies*, «una gentil introducción» for *gentle*).
- Commit pattern: small web-editor commits on consecutive days, file by file.

The register is Peninsular Spanish (`ordenador`, `fichero`, `portátil`, `visión por ordenador`), and the translator
leaves *deep learning*, *machine learning*, *dataset*, *framework* and *notebook* in English. Being human, matching
renderings in the intended sense were set to `verified_d2l`; the poor proofreading is a reason to weigh it as one
person's usage, not as an edited edition.

## Results for `d2l-es-acca3003`

Counts: 26 `proposed` entries set to `verified_d2l`; 15 already-verified entries given agreeing evidence; 3
alternatives added (on 3 entries, one of them also verified); 2 conflicts; 7 skipped; 18 candidate terms with no
concept. Locators are blob URLs at the pinned commit (prefix
`https://github.com/acca3003/d2l-es/blob/1d65b3d1e8b9789f1d9d4b00bc2a0fbaf1c9cd9a/`, shortened below).

| concept | glossary preferred | source rendering | action | locator |
|---|---|---|---|---|
| scalar | escalar | escalar | verified | `chapter_notation/index.md#L16` |
| vector | vector | vector | verified | `chapter_notation/index.md#L17` |
| tensor | tensor | tensores | verified | `chapter_preliminaries/linear-algebra.md#L4` |
| concatenation | concatenación | concatenación | verified | `chapter_notation/index.md#L54` |
| dot-product | producto escalar | producto escalar/punto | verified; alternative «producto punto» added | `chapter_notation/index.md#L57` |
| derivative | derivada | derivada | verified | `chapter_notation/index.md#L66` |
| partial-derivative | derivada parcial | derivada parcial | verified | `chapter_notation/index.md#L67` |
| definite-integral | integral definida | integral definida | verified | `chapter_notation/index.md#L69` |
| indefinite-integral | integral indefinida | integral indefinida | verified | `chapter_notation/index.md#L70` |
| probability | probabilidad | Probabilidad (heading) | verified | `chapter_notation/index.md#L74` |
| information-theory | teoría de la información | Teoría de la Información (heading) | verified | `chapter_notation/index.md#L74` |
| probability-distribution | distribución de probabilidad | distribución de probabilidad | verified | `chapter_notation/index.md#L77` |
| relative-entropy | entropía relativa | entropia relativa (no accent) | verified | `chapter_notation/index.md#L90` |
| linear-algebra | álgebra lineal | álgebra lineal | verified | `chapter_preliminaries/linear-algebra.md#L7` |
| matrix-multiplication | multiplicación de matrices | multiplicación de matrices | verified | `chapter_preliminaries/linear-algebra.md#L10` |
| model | modelo | Modelos (heading, Key Components) | verified | `chapter_introduction/index.md#L358` |
| optimization | optimización | Algoritmos de Optimización (heading) | verified | `chapter_introduction/index.md#L443` |
| supervised-learning | aprendizaje supervisado | Aprendizaje Supervisado (heading) | verified | `chapter_introduction/index.md#L471` |
| classification | clasificación | Clasificación (heading) | verified | `chapter_introduction/index.md#L619` |
| recommender-system | sistema de recomendación | Sistemas de Recomendación (heading) | verified | `chapter_introduction/index.md#L823` |
| unsupervised-learning | aprendizaje no supervisado | Aprendizaje No-Supervisado (heading, hyphenated) | verified | `chapter_introduction/index.md#L992` |
| self-supervised-learning | aprendizaje autosupervisado | Auto-Supervisado (heading, hyphenated) | verified | `chapter_introduction/index.md#L992` |
| environment | entorno | Entorno (heading, agent sense) | verified | `chapter_introduction/index.md#L1080` |
| reinforcement-learning | aprendizaje por refuerzo | aprendizaje por refuerzo | verified | `chapter_preface/index.md#L19` |
| agent | agente | agentes de software (Go-playing agents) | verified | `chapter_preface/index.md#L26` |
| gpu | GPU | GPU | verified | `chapter_installation/index.md#L72` |
| matrix | matriz | matriz | evidence added (already verified) | `chapter_notation/index.md#L18` |
| identity-matrix | matriz identidad | matriz identidad | evidence added | `chapter_notation/index.md#L20` |
| norm | norma | Normas (heading) | evidence added | `chapter_preliminaries/linear-algebra.md#L907` |
| gradient | gradiente | gradiente | evidence added | `chapter_notation/index.md#L68` |
| random-variable | variable aleatoria | variable aleatoria | evidence added | `chapter_notation/index.md#L76` |
| conditional-probability | probabilidad condicional | distribución de la probabilidad condicional | evidence added | `chapter_notation/index.md#L80` |
| expectation | esperanza | esperanza | evidence added | `chapter_notation/index.md#L82` |
| standard-deviation | desviación estándar | desviación estándar | evidence added | `chapter_notation/index.md#L85` |
| variance | varianza | varianza | evidence added | `chapter_notation/index.md#L86` |
| covariance | covarianza | covarianza | evidence added | `chapter_notation/index.md#L87` |
| correlation | correlación | coeficiente de correlación de Pearson | evidence added | `chapter_notation/index.md#L88` |
| entropy | entropía | entropyia (typo) | evidence added | `chapter_notation/index.md#L89` |
| objective-function | función objetivo | Funciones Objetivo (heading) | evidence added | `chapter_introduction/index.md#L381` |
| regression | regresión | Regresión (heading) | evidence added | `chapter_introduction/index.md#L539` |
| neural-network | red neuronal | redes neuronales | evidence added | `chapter_preface/index.md#L14` |
| dataset | conjunto de datos | datasets (loanword) | alternative «dataset» | `chapter_preliminaries/linear-algebra.md#L4` |
| probability-density | densidad de probabilidad | función de densidad (PDF) | alternative «función de densidad» | `chapter_notation/index.md#L81` |
| transpose | transpuesta | transposición (the operation, for "transpose of a vector or a matrix") | conflict | `chapter_notation/index.md#L51` |
| computer-vision | visión artificial | visión por ordenador | conflict (Peninsular form; see review table) | `chapter_preface/index.md#L11`, `#L18` |
| hadamard-product | producto de Hadamard | producto Hadamard (por elementos) | skipped: English noun-adjunct order, missing «de» | `chapter_notation/index.md#L53` |
| dot-product | producto escalar | PRoductos Dot (heading) | skipped: calque; the notation line is used instead | `chapter_preliminaries/linear-algebra.md#L645` |
| norm | norma | $\ell_p$ norma | skipped: English word order; the heading is used instead | `chapter_notation/index.md#L55` |
| independence / conditional-independence | independencia / independencia condicional | son (codicionalmente) independientes | skipped: only the adjective occurs | `chapter_notation/index.md#L83-L84` |
| environment | entorno | entorno (conda environment) | skipped: software-environment sense, not the RL concept | `chapter_installation/index.md#L4` |
| (sequence learning) | — | Secuencia de Aprendizaje | skipped: wrong word order, no concept | `chapter_introduction/index.md#L885` |
| (KL divergence) | — | KL-divergencia | skipped: calque of English order (usual: divergencia KL), no concept | `chapter_notation/index.md#L90` |

Candidates (no concept in `concepts.jsonl`): *machine learning* and *deep learning* (left in English throughout),
*natural language processing* «procesamiento del lenguaje natural» (P#L18), *speech recognition* «reconocimiento de
voz» (P#L11), *matrix inverse* «inversa de una matriz» (N#L52), *indicator function* «función indicador» (N#L49;
standard is «función indicadora»), *natural logarithm* «logaritmo neperiano/natural» (N#L46), *exponential function*
«función exponencial» (N#L48), *cardinality* «cardinalidad» (N#L35), *set theory* «teoría de conjuntos» (N#L26),
*Pearson correlation coefficient* «coeficiente de correlación de Pearson» (N#L88), *matrix-vector product*
«producto matriz-vector» (L#L708), *reduction* «reducción» (L#L445), *tagging* «etiquetado» (I#L734),
*deep learning framework* «framework de deep learning» (S#L71), *CPU* (S#L107), *KL divergence* (above).

## Datitos course (`datitos-curso-ap`)

`datitos/curso-aprendizaje-profundo`, pinned at `3bc0fd2d396ab3e85c967e901319b2fe11f16deb` (2022-04-07). The site
of a free course run in Mendoza, Argentina, in 2021 and 2022, that reads D2L chapter by chapter. It is not a
translation and has no parallel English, so it was treated as an external teaching source: evidence only where a
syllabus topic, schedule row or assignment names a glossary concept unambiguously. About 1,600 words of
human-written Argentine Spanish (voseo: «tenés», «necesitás»; inclusive «otres»): `web/2022/01_CONTENIDOS.md`
(syllabus by unit; identical to 2021 except «Prerrequisitos»), `README.md` (2021 class schedule, terse rows with
typos such as «Foward Propagation», «recomencación», «Segmentación semática»), `Trabajos.md`,
`ProyectosFinales.md`, and the markdown cells of `tp2/Trabajo práctico 2.ipynb`. The schedule keeps many English
loanwords (Pooling, Padding, Dropout, Batch Normalization, Bounding Boxes, Fine-Tuning). Locators below are relative
to `https://github.com/datitos/curso-aprendizaje-profundo/blob/3bc0fd2d.../`; C = `web/2022/01_CONTENIDOS.md`,
R = `README.md`, W = `Trabajos.md`, NB = `tp2/Trabajo%20pr%C3%A1ctico%202.ipynb`.

Counts: 31 `proposed` entries set to `verified_external`; 15 entries given agreeing evidence only (8 already
`verified_external`, 7 just set to `verified_d2l` from d2l-es); 1 matching entry kept `proposed` (self-attention:
the authors flagged the name «Autoatención» as "revisar nombre"); 11 new alternatives on 10 entries, plus agreeing evidence for «dataset» (added from d2l-es); 4 conflicts;
10 skipped; 8 candidates.

| concept | glossary preferred | source rendering | action | locator |
|---|---|---|---|---|
| gaussian-distribution | distribución gaussiana | distribución gaussiana | verified | C#L29 |
| multilayer-perceptron | perceptrón multicapa | perceptrón multicapa | verified | C#L45 |
| automatic-differentiation | diferenciación automática | diferenciación automática | verified | C#L46 |
| convolutional-neural-network | red neuronal convolucional | redes neuronales convolucionales (CNNs) | verified | C#L53 |
| recurrent-neural-network | red neuronal recurrente | redes neuronales recurrentes (RNNs) | verified | C#L64 |
| attention | atención | mecanismos de atención | verified | C#L67 |
| matrix-factorization | factorización de matrices | factorización de matrices | verified | C#L74 |
| layer | capa | Capas | verified | R#L31 |
| initialization | inicialización | Inicialización (de parámetros) | verified | R#L33 |
| momentum | momentum | Momentum | verified | R#L40 |
| model-selection | selección de modelos | Selección de modelos | verified | R#L46 |
| dropout | dropout | Dropout | verified | R#L50 |
| convolution | convolución | Convoluciones | verified | R#L53 |
| pooling | pooling | Pooling | verified | R#L54 |
| residual-connection | conexión residual | Conexiones residuales | verified | R#L65 |
| style-transfer | transferencia de estilo | Transferencia de estilos | verified | R#L71 |
| object-detection | detección de objetos | Detección de Objetos | verified | R#L73 |
| semantic-segmentation | segmentación semántica | Segmentación semática (typo) | verified | R#L80 |
| generative-model | modelo generativo | Modelos generativos | verified | R#L81 |
| perplexity | perplejidad | Perplejidad | verified | R#L86 |
| likelihood | verosimilitud | verosimilitud | verified | R#L86 |
| sentiment-analysis | análisis de sentimientos | Analisis de sentimiento | verified | R#L96 |
| sequence-to-sequence | secuencia a secuencia | secuencia a secuecia (typo) | verified | R#L101 |
| word-embedding | embedding de palabras | Embeddings de palabras | verified | R#L102 |
| embedding | embedding | Embedding | verified | W#L59 |
| layer-normalization | normalización por capas | Normalización por capas | verified | R#L119 |
| positional-encoding | codificación posicional | Codificación posicional | verified | R#L121 |
| negative-sampling | muestreo negativo | Muestreo negativo | verified | R#L141 |
| factorization-machine | máquina de factorización | Máquinas de factorización | verified | R#L146 |
| transposed-convolution | convolución transpuesta | Convolución Transpuesta (README: «traspuesta») | verified | W#L41 |
| target-variable | variable objetivo | variable objetivo | verified | NB#L283 |
| linear-algebra, probability, optimization, gpu, model, recommender-system, tensor | (as above) | álgebra lineal, probabilidad, optimización, GPUs, modelos, sistemas de recomendación, tensores | evidence added (verified_d2l kept) | C#L23, C#L27, C#L36, C#L36, C#L38, C#L73, NB#L350 |
| standard-deviation, linear-regression, loss-function, regularization, gradient-descent, softmax-regression, entropy, machine-translation | (as preferred) | same terms | evidence added (already verified) | C#L29, C#L45 ×3, R#L38, R#L19, R#L86, R#L100 |
| self-attention | autoatención | Autoatención «<-- revisar nombre» | evidence only, kept proposed | R#L115 |
| stochastic-gradient-descent | descenso de gradiente estocástico | descenso por el gradiente estocástico; Descenso estocástico de gradiente | 2 alternatives | C#L46, R#L39 |
| transfer-learning | aprendizaje por transferencia | _transfer learning_ | alternative | C#L57 |
| fine-tuning | ajuste fino | _fine-tuning_ | alternative | C#L57 |
| underfitting | subajuste | Underfitting | alternative | W#L19 |
| overfitting | sobreajuste | Overfitting | alternative | W#L20 |
| padding | relleno | Padding | alternative | R#L55 |
| bounding-box | cuadro delimitador | Bounding Boxes | alternative | R#L74 |
| attention-pooling | pooling por atención | Pooling de atención | alternative | R#L109 |
| masked-language-modeling | modelado de lenguaje enmascarado | Modelado del lenguaje con máscaras | alternative | R#L126 |
| ranking | ordenación | ranking | alternative | R#L140 |
| dataset | conjunto de datos | dataset | alternative (same as d2l-es) | NB#L343 |
| mean | media | promedio (also «promedios», NB#L219) | conflict | C#L29 |
| backpropagation | retropropagación | _backpropagation_ (English, italic) | conflict | C#L46 |
| computer-vision | visión artificial | Visión por computadora | conflict (Latin-American form) | C#L49 |
| batch-normalization | normalización por lotes | Batch Normalization | conflict (English kept) | R#L64 |
| forward-pass / backward-pass | pasada hacia adelante / atrás | Foward Propagation / Backward Propagation | skipped: English, typo | R#L25-L26 |
| stride | stride | striding | skipped: English verb form | R#L55 |
| weight-decay | decaimiento de pesos | Regularización de los pesos | skipped: description, not a term | R#L49 |
| pretraining | preentrenamiento | Redes preentrenadas | skipped: adjective only | R#L69 |
| backpropagation-through-time | retropropagación a través del tiempo | Backpropagation con dependencia temporal | skipped: paraphrase | R#L87 |
| gated-recurrent-unit / long-short-term-memory | unidad recurrente con puertas / memoria larga a corto plazo | Recurrencias con compuertas: GRU / Recurrencia con memoria: LSTM | skipped: section paraphrases | R#L92-L93 |
| multi-head-attention | atención multicabeza | Multiples cabezales | skipped: incomplete term | R#L114 |
| next-sentence-prediction | predicción de la siguiente oración | Predicción de próxima oraciones | skipped: ungrammatical | R#L129 |
| natural-language-inference | inferencia en lenguaje natural | Inferencia del lenguaje | skipped: truncated | R#L132 |
| latent-variable / hidden-state | variable latente / estado oculto | Variables ocultas | skipped: ambiguous | R#L85 |

Candidates (no concept): *deep learning* «aprendizaje profundo» (C#L4), *machine learning* «aprendizaje automático»
(C#L5; NB also «aprendizaje de máquinas»), *natural language processing* «procesamiento de lenguaje natural»
(C#L61), *time series* «series temporales» (C#L65), *generative adversarial networks* «redes generativas antagónicas
(GANs)» (C#L59), *standardization* «estandarización» (NB#L219), *optimization algorithm* «algoritmos de
optimización» (C#L46), *vector calculus* «cálculo vectorial» (C#L23).

## Observations for the native reviewer

- Both sources contradict the reviewed `visión artificial`: the translator writes «visión por ordenador» (Spain), the
  course «visión por computadora» (Argentina). The regional split in `review/es.md` is real in D2L-adjacent usage.
- Both use the loanword *dataset*, and the course keeps *transfer learning*, *fine-tuning*, *overfitting*,
  *padding*, *bounding box*, *batch normalization* and *backpropagation* in English; the glossary translates all of
  them. Whether Spanish D2L should follow practitioner loanwords is the same open question as *embedding* and
  *pooling* in the review.
- `notes` of entries whose status changed had their phrase "not yet independently attested in this sense" replaced
  by "attested in" plus the source, so the note no longer contradicts the status.
