# Spanish (es) glossary review

Reviewed 2026-11-26. Scope: all 1,184 entries of `glossary/terms/es.jsonl` were read against the senses in
`concepts.jsonl` (69 `verified_external`, 1,115 `proposed`; the brief asked for a sample of the low-frequency
proposals, but the file is small enough to read whole). About 75 terms were checked against es.wikipedia article
titles and redirects, the pinned Stanford cheatsheet translation, the Spanish Python documentation and the live
Google glossary. Corrections: `glossary/review/es-corrections.jsonl` (110 lines).

## Verdict

The Spanish glossary is in much better shape than the Japanese one. The mathematical, statistical and
optimization vocabulary (about three quarters of the file) is correct, idiomatic Spanish, and several hard
choices are already right: `exactitud` / `precisión` / `exhaustividad` for accuracy / precision / recall,
`equivarianza` / `invariancia a traslaciones` (not *traducción*), `desvanecimiento del gradiente`,
`regla de la cadena`, `perceptrón multicapa`, Adam and the other optimizer names left alone.

The errors are concentrated in three places: the two highest-frequency verified entries taken from Google,
a layer of invented descriptive paraphrases for deep-learning and systems jargon, and inconsistencies between
head terms and their compounds.

| | entries | changed | share |
|---|---|---|---|
| `verified_external` | 69 | 12 | 17% |
| `proposed`, frequency ≥ 50 | 242 | 35 | 14% |
| `proposed`, frequency < 50 | 873 | 63 | 7% |
| **all** | 1,184 | 110 | 9% |

By severity: wrong sense 4, calque / machine translation 7, non-standard 44, inconsistent 41, minor 14.
Of the 12 verified changes, 2 are substantive MT renderings (`descenso de gradientes`, `propagación inversa`;
461 and 172 occurrences), 4 are non-standard or inconsistent (`media de la muestra`, `varianza de la muestra`,
`función valor`, `valor esperado` vs. `esperanza`), 5 are capitalisation artefacts copied from
headings (`variable Aleatoria`, `probabilidad Condicional`, `muestra Aleatoria`, `teorema del Límite Central`,
`regresión Softmax`) and 1 is a minor number change (`selección de modelo`). 19 corrections carry a cited
source and are marked `verified_external`; the other 91 stay `proposed`.

## Provenance

There is no Spanish D2L edition (`es.d2l.ai` does not resolve; the `acca3003/d2l-es` fork is English with a
translated preface). The 69 verified entries come from two sources of unequal quality:

- **Google for Developers ML glossary, `hl=es` (25 entries).** Machine-translated. The live page
  (`?hl=es-419`, fetched 2026-11-26) is flagged as translated in its page metadata and shows plain MT errors:
  *feature* → «función», *neural network* → «neuronal prealimentada», *precision* and *recall* left in
  English, *bias* → «ordenada al origen». Its `descenso de gradientes` and `propagación inversa` are the two
  wrong verified entries. Treat this source as attestation that a string exists, never as evidence of correct usage.
- **Stanford CS229/CS230 cheatsheets, crowd-translated (`shervinea/cheatsheet-translation`, 44 entries).**
  Volunteer translation (not machine output, by its prose); terminology is mostly sound, Latin-American in
  flavour (`desviación estándar`, `costo`). Defects are heading capitalisation and a few literal renderings
  (`media de la muestra`, `función valor`, LSTM as «memoria de corto y largo plazo»). It contradicts the Google
  entries: it has «Retropropagación», «Descenso de gradiente», «Pérdida de bisagra», «Q-learning», «puerta».

"Verified" therefore means "seen in one of these two", as in the Japanese case. The 1,115 proposals have no
source at all; they read as one careful model-written pass.

## Systematic problems

1. **Sense glosses stored as the preferred term.** To separate English homonyms the builder wrote the
   disambiguation into the term: `núcleo de convolución` (kernel, 1,058 occurrences), `crítico de valor` /
   `crítico generativo`, `término de sesgo` / `sesgo estadístico`, `retorno acumulado`, `incrustación vectorial`,
   `función de puntuación de densidad`, `remuestreo bootstrap`, `clasificación por relevancia`. A translator
   applying the glossary would paste the gloss into every occurrence. The term should be the bare word
   (`núcleo`, `crítico`, `sesgo`, `retorno`) with the sense kept in `notes`.
2. **Definitions in place of terms.** Jargon with no Spanish equivalent was replaced by a paraphrase of its
   definition: `procesamiento inicial del contexto` (prefill), `recopilación colectiva` / `reducción colectiva` /
   `reducción con distribución` (all-gather / all-reduce / reduce-scatter), `modelo de techo de rendimiento`
   (roofline), `actualización por estimaciones` (bootstrapping), `desactivación aleatoria` (dropout),
   `forzado del profesor` (teacher forcing), `ajuste de flujos` / `ajuste de puntuaciones` (flow / score
   matching), `caché de claves y valores`, and the expansions used for GPU (1,195 occurrences), ReLU and ROC.
   Spanish technical writing keeps these as loanwords, names or acronyms.
3. **New collisions created by translation.** `difusión de dimensiones` (broadcasting) vs. diffusion models;
   `momento` for momentum vs. statistical moments, side by side in Adam; `paso` for stride vs. step;
   `rendimiento` for throughput vs. performance; `entorno de ruido` (noise ball) vs. the RL `entorno`;
   `dispersión` for sparsity vs. statistical spread; `precisión media` (half precision) = average precision;
   `submuestreo` for both downsampling and subsampling (left as is, noted).
4. **Head term and compounds disagree.** `distribución a priori / a posteriori` but 16 compounds with
   `previa / posterior / hiperprior`; `valor esperado` but `esperanza condicional`; `descenso de gradientes` but
   `descenso de gradiente estocástico`; `propagación inversa` but `retropropagación a través del tiempo`;
   `puerta de olvido` but `compuerta de actualización`; `cuadro delimitador` but `caja de anclaje`;
   `clasificación por relevancia` but `ordenación por pares`; `función valor` but `función de valor de acción`;
   `contraste de hipótesis` but `estadístico de prueba`; `anchura del modelo` but `ancho del haz`;
   `logaritmo de la …` for every log- compound against the usual `log-verosimilitud`.
5. **No alternatives recorded.** `alternatives` is empty in all 1,184 entries, so accepted variants
   (`valor esperado`, `autovalor`, `kernel`, `embedding`, regional forms) are invisible to translators and QA.

## Sense collisions checked (brief item 3)

Consistent after the corrections: gradient (singular `gradiente` in every compound except the accepted plurals
`recorte / acumulación de gradientes`), label, state, action, reward, policy, value, token, logit,
temperature (`temperatura de softmax`), attention, feature (`característica`), generalization, regularization,
loss (`pérdida`; there is no cost-function concept, `costo` appears only as inference cost), sample / sampling
(`muestra` / `muestreo`), kernel (`núcleo` for convolution, covariance and transition kernels; `kernel` only for
a GPU kernel launch). Inherent to Spanish and left alone: `sesgo` covers both biases, `objetivo` covers both
objective and target (use `función objetivo` where ambiguous), `media` / `promedio` are used interchangeably
for mean / average, `puntuación` covers attention score and F1 once the density score becomes `score`.

## Decisions for the author or a native reviewer

Loanword or translation (each affects many entries; the corrections file records my recommendation, all as `proposed`):

- **embedding**: `embedding` (es.wikipedia, Google, most course material) or `incrustación`; 11 compounds follow.
- **prompt**: `prompt` or `instrucción` (es.wikipedia «Ingeniería de instrucciones»); the latter collides with *instruction*.
- **Transformer**: the file has `transformador` (es.wikipedia title), giving `bloque transformador`,
  `codificador transformador`, `transformador de visión`. Practitioners write «Transformer», «Vision Transformer». Not changed; needs a ruling.
- **pooling**: `agrupación` (file) or `pooling`; `agrupación` sits close to `agrupada` (grouped convolution / attention) and to clustering. Not changed.
- **momentum, stride, throughput, score, dropout, sparsity (noun), surprisal, ground truth, odds**: see corrections.
- **Monte Carlo** vs. RAE / Wikipedia «Montecarlo»; **Kolmogórov** with accent (kept) alongside Itô, Fréchet.
- **gate**: `puerta` (Spain, cheatsheet) vs. `compuerta` (Latin America). **box**: `cuadro` vs. `caja`.

Regional splits where the file picks one side (no correction filed unless inconsistent):

| concept | in file | other region |
|---|---|---|
| computer vision | visión por computadora (LatAm) | visión por ordenador (Spain); neutral: visión artificial |
| floating point | coma flotante (Spain) | punto flotante (LatAm) |
| standard deviation / error | desviación / error estándar (LatAm) | desviación / error típico (Spain) |
| hypothesis testing | contraste de hipótesis (Spain) | prueba de hipótesis (LatAm) |
| statistical significance | significación (Spain) | significancia (LatAm) |
| multivariate, Euclidean | multivariante, euclidiana | multivariada (LatAm), euclídea (Spain) |
| cost | costo (LatAm) | coste (Spain) |
| intercept | intercepto (LatAm) | ordenada en el origen (Spain) |
| random walk | paseo aleatorio (Spain) | caminata aleatoria (LatAm) |
| boundary condition | condición de contorno (Spain) | condición de frontera (LatAm) |
| rating | calificación (LatAm) | valoración (Spain) |
| eigenvalue | valor propio | autovalor (both regions, by school) |

The file mixes Peninsular and Latin-American choices; a native reviewer should fix one policy (e.g. Latin-American
forms where the split is binary, since most readers are there) and record the other form in `alternatives`.

## Limits of this review

Checks relied on es.wikipedia titles and redirects, the cheatsheet text, docs.python.org/es and the Google page;
no Spanish textbook or university syllabus was read in full, and Wikipedia titles show that a term is in use,
not that it is the best one. Terms judged from my own knowledge of Spanish technical usage are marked
`proposed`. Low-frequency entries not listed in the corrections file were read once and looked right; they were
not individually sourced.

Sources: es.wikipedia.org (Descenso del gradiente, Retropropagación, Esperanza (matemática), Red neuronal
prealimentada, Memoria larga a corto plazo, Formato en coma flotante de media precisión, Media cuadrática,
Bandido multibrazo, Visión artificial, Aliasing, Q-learning, Modelo de lenguaje de gran tamaño, Odds ratio,
Curva ROC, Contraste de hipótesis, Ingeniería de instrucciones, Word embedding);
github.com/shervinea/cheatsheet-translation at 99f45b6 (`es/cs-229-*.md`);
docs.python.org/es/3/glossary.html; developers.google.com/machine-learning/glossary/fundamentals?hl=es-419.
