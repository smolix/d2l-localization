# Style guide: Spanish (es)

Draft of 2026-11-26 for the author's review. Translators read this with every
unit (TRANSLATING.md §7–§8). Each rule has one reason and one example
(English → Spanish, or ✗ wrong / ✓ right). Rules marked "(to confirm)" are
decided here and listed again in §11 for a native reviewer. Examples marked
*(test)* are sentences from the translation test `evals/es-conv-layer/`.

Order of authority: the decisions log, then this guide, then the glossary. The
glossary says *what* a term is; this guide says *how to choose and inflect*.
Of the 1,191 Spanish glossary entries 76 are verified and the rest `proposed`;
where this guide names a different term (§12), the guide applies.

## 1. Variety and register

### 1.1 Which Spanish

**Decision: one neutral international Spanish (*español neutro*), readable in
Spain and in all of Latin America, in the orthography of the RAE/ASALE
*Ortografía* of 2010.** Reason: the book has one Spanish edition (`es`, not
`es-ES` or `es-MX`), and about nine in ten Spanish speakers live outside Spain. Where the
regions differ, take the form that no region finds wrong or comic; where both
forms are equally understood, the table fixes one so that they do not
alternate.

| Point | Write | Not | Reason |
|---|---|---|---|
| Second person plural | (not needed; see §1.2) | vosotros, *os*, *habéis visto*, *fijaos* | Spain only |
| Voseo | never | *vos tenés*, *fijate* | Río de la Plata and Central America only |
| Past of earlier sections | pretérito simple: *En … vimos*, *dedujimos* | *hemos visto* for a section read days ago | the simple past is right in every region |
| Object pronouns | *lo/la* for direct objects: *el modelo, lo entrenamos* | leísmo: *le entrenamos* | Spain only |
| "to take, pick" | tomar, elegir, extraer | coger | vulgar in much of Latin America |
| computer | la computadora | el ordenador (Spain), el computador | understood everywhere |
| file | el archivo | el fichero | |
| cost | el costo (glossary) | el coste | (to confirm) |
| video, pixel | el video; el píxel, los píxeles | el vídeo; el pixel | RAE lists both pairs; fixed here |
| to run code | ejecutar | correr (un programa) | calque |
| loop | el bucle | el ciclo, el lazo, el loop | all three test translators chose it |
| library | la biblioteca | la librería (= bookshop) | false friend |
| array (in code) | el array, los arrays | el arreglo (Latin America), el vector/la matriz for a code object of any rank | see §3.3 |
| to plot | representar (gráficamente), trazar | plotear, graficar | *graficar* is American only |
| to check | comprobar, verificar | chequear, checar | regional |
| here, there | aquí, allí | acá, allá | |
| floating point | punto flotante | coma flotante | fits the decimal point of §4 (to confirm) |

### 1.2 We, you, and the impersonal

**Decision: authorial *nosotros* (verb ending only), the reader as *usted* in
the rare cases where the English really addresses the reader, never *tú*, *vos*
or *vosotros* (to confirm).** Reason: first person plural is the voice of
Spanish university textbooks and carries the book's "we work through this
together" tone. For direct address, *usted* is the only form that is correct
and unmarked in every Spanish-speaking country, and it is the form of the
Spanish-language university textbooks (*Demuestre que …*, *Calcule …*); *tú* is
the register of tutorials and reads as foreign where *vos* or *usted* is the
everyday form. Two of the three test translators chose *tú*, one avoided
address altogether; the guide follows the second for body text and fixes
*usted* for what remains.

| English | Spanish | Example |
|---|---|---|
| we + action in the text | first person plural, **no pronoun** | "We minimize the loss with SGD." → Minimizamos la pérdida con SGD. ✗ Nosotros minimizamos … |
| "let us", "let's" | present subjunctive: Veamos, Consideremos, Supongamos | "Let us look at the gradient." → Veamos el gradiente. ✗ Dejemos que …, ✗ Vamos a ver … (every time) |
| "we will see", "we will use" | simple future or present; *vamos a* + infinitive at most once per page | "We will return to this in …" → Volveremos sobre esto en … |
| "note that", "recall that" | *nosotros*: Observemos que …, Recordemos que … ; or drop it (§9.4) | ✗ Observa que la salida se encoge un poco *(test)* ✓ Observemos que la salida se reduce ligeramente en cada eje *(test)* |
| "you" = the reader (advice, a question to the reader, exercises) | *usted* forms, the pronoun itself only where needed for clarity; lower case *usted*, *su* | "Suppose you want to estimate the price of a house." → Suponga que quiere estimar el precio de una casa. (or: Supongamos que queremos …) |
| "you can", "one can" = anyone | *se* + verb, or *nosotros* | "You can verify this numerically." → Esto se puede comprobar numéricamente. ✗ Puedes comprobar … |
| "your model", "our model" | the article | "our model overfits" → el modelo se sobreajusta |
| "we recommend", "we believe" | *nosotros* | "We recommend AdamW." → Recomendamos AdamW. |
| "we" = people, the field | los seres humanos, la comunidad, *se* | "We humans attend selectively." → Los seres humanos atendemos de forma selectiva. |

Most English "you" sentences are general statements: translate them with
*nosotros* or *se*, and keep *usted* for exercises (§8) and for the few
sentences that give the reader advice.

### 1.3 Sentence shape

Each point is checkable.

- **No subject pronoun** where the verb ending shows the person (✗ *Nosotros
  calculamos*, ✗ *Usted puede ver*). Write *nosotros* or *usted* only for
  contrast.
- **Active or *se*, not *ser* + participle.** The English passive becomes
  first person plural or the *pasiva refleja*; *ser* + participle only where
  the agent is named. ✗ El gradiente es calculado mediante la regla de la
  cadena. ✓ Calculamos el gradiente con la regla de la cadena. ✓ El gradiente
  se calcula con la regla de la cadena. Never *puede ser calculado*: *se puede
  calcular*.
- **Gerund only for an action simultaneous with the main verb and with the
  same subject.** No gerund for a result or a later action, none as an
  adjective. ✗ El modelo converge, obteniendo una exactitud del 92% ✓ El modelo
  converge y alcanza una exactitud del 92%. ✗ una matriz conteniendo los pesos
  ✓ una matriz que contiene los pesos.
- **Verbs, not noun chains.** No sentence whose only verb is *realizar*,
  *llevar a cabo*, *efectuar* or *proceder a*. ✗ Realizamos la actualización de
  los pesos. ✓ Actualizamos los pesos.
- **At most three *de* phrases in a row.** ✗ la forma de la salida de la
  primera capa de la red ✓ la forma que produce la primera capa de la red.
- **One main clause and at most two subordinate clauses** per sentence; split
  English sentences at "which", "where", "so that" (TRANSLATING.md §3 allows
  splitting within a block).
- **No two adverbs in *-mente* in a row**, and no literal "simply"/"just"/
  "basically": drop it or write *solo*, *basta con*.
- **Subjunctive where Spanish requires it:** *para que el modelo converja*,
  *cuando el gradiente sea cero* (future reference), *supongamos que … es* (the
  indicative after *suponer que*).
- **No anglicisms beyond the loan list in §3.** In particular never *layer*,
  *loss*, *weights*, *feature*, *learning rate*, *dataset*, *batch*, *input*,
  *output*, *performance*, *accuracy*, *shape*, *training* as Spanish words:
  capa, pérdida, pesos, característica, tasa de aprendizaje, conjunto de datos,
  lote, entrada, salida, rendimiento, exactitud, forma, entrenamiento.

### 1.4 Orthography points that matter here (RAE 2010)

| Point | Rule | Example |
|---|---|---|
| *solo*, demonstratives | No accent, ever. | ✓ solo depende del núcleo ; este, ese, aquel ✗ sólo, éste |
| *o* between numerals | No accent. | 2 o 3 capas ✗ 2 ó 3 |
| *y* → *e*, *o* → *u* | Before a word, numeral or symbol **as it is read**: *e* before the sound /i/, *u* before /o/. | `X` e `Y` ; $j$ e $i$ ; entrada e inicialización ; 7 u 8 ; 10 u 11 ; $m$ u $o$ ✗ `X` y `Y` *(test, two translators)* |
| Prefixes | Joined, no hyphen: preentrenamiento, autoatención, hiperparámetro, sobreajuste, subajuste, multicapa, minilote, retropropagación, autorregresivo (double *r*), semisupervisado, antisimétrico. Hyphen only before a capital, an acronym or a numeral: pre-BERT, anti-GAN, sub-10. | ✗ pre-entrenamiento, ✗ auto-atención, ✗ hiper parámetro |
| Coordinated pairs | A hyphen joins two nouns or names of equal rank: codificador-decodificador, actor-crítico, Cauchy-Schwarz. | la arquitectura codificador-decodificador |
| Acronyms (siglas) | Capitals, no full stops, **no plural *-s***; the article shows the number. Gender in §3.3. | las GPU, los MLP, las CNN *(test)* ✗ las GPUs, ✗ los MLP's |
| Lower case | Languages, months, days, adjectives from names, and the generic word in a name. | inglés, enero, gaussiano, el teorema de Bayes, la ecuación de Bellman |
| *porque / por qué / porqué*, *sino / si no* | Keep them apart: *porque* gives the cause, *por qué* asks, *el porqué* is a noun. | ¿Por qué converge? Porque la función es convexa. |
| *de que* | *asegurarse de que*, *el hecho de que*, *darse cuenta de que*; but *pensamos que*, *resulta que*. | ✗ nos aseguramos que ✗ pensamos de que |
| Adapted loans take Spanish accents | píxel, clúster, búfer, caché, perceptrón, bucle, estándar, escáner. | los clústeres, los búferes |

## 2. Punctuation and typography

| Item | Rule | Example |
|---|---|---|
| Opening ¿ and ¡ | Always, in body text, headings, captions, slides and `title="…"` attributes. The opening mark stands where the question begins, which may be mid-sentence; what follows a comma is lower case. After `?` or `!` no full stop. Avoid exclamations. | Si la banda negra llegara al borde, ¿cuál sería la pérdida inicial? ; `title="¿Por qué ayuda la profundidad?"` ✗ Por qué converge? |
| Several questions | Each has its own marks. | ¿Cuál es la forma de la salida? ¿Y con capas de $5 \times 5$? |
| Quotation marks | «…» (U+00AB, U+00BB); inside them “…”, then ‘…’. Never straight `"…"` in prose. The full stop or comma goes **after** the closing mark. Book titles in italics as in the source. | the "attention" weights → los pesos de «atención» ; llamamos «convolución» a la correlación cruzada *(test)* ✗ el núcleo de nitidez "el doble de la identidad menos el desenfoque" *(test)* |
| Quotes in attributes | A slide `title="…"` is delimited by ASCII `"`; inside it use «…» only. | `title="¿Qué significa «profundo»?"` |
| Dash | Do not copy the English em dash `—`. Use a comma, a colon, a semicolon or parentheses. The Spanish raya for an aside (—así—, attached to the words it encloses) is allowed at most once per paragraph. Never a hyphen with spaces. | "the bias—an offset—shifts the plane" → el sesgo (un desplazamiento) traslada el plano |
| Comma | Spanish rules: no comma between subject and verb, however long the subject; no comma before *y*, *e*, *o*, *u* in a list; a comma after a fronted connector (Sin embargo, Por tanto, Es decir, En cambio, Además,) and around *es decir*, *por ejemplo*; a comma after a long fronted adverbial or subordinate clause. | "Adam, RMSProp, and SGD" → Adam, RMSProp y SGD ; ✗ La capa que definimos antes, calcula … ✓ Sin embargo, la pérdida aumenta. |
| Colon | Lower case after a colon in running text, also before a list or an explanation. Capital only after a bold lead-in label (below) and before a quoted full sentence. No colon between a preposition or verb and its list (✗ *formado por: A y B*). | Hay tres hiperparámetros: la tasa de aprendizaje, el tamaño del lote y el número de épocas. |
| Bold lead-in | `**Label:**` keeps the bold and the colon inside it, and the text after it starts with a **capital**, as in English (RAE: text after the colon of an epigraph). (to confirm) | **Supuesto central:** Los errores son independientes. |
| Full stop | After a closing parenthesis, bracket or quotation mark, never before it. None after headings, slide titles, kickers. | … (véase la sección anterior). |
| Abbreviations | In body text write them out: por ejemplo, es decir; *etc.* is allowed at the end of a list. In parentheses, captions, slides and tables: p. ej., etc., aprox., cap., fig., n.º, EE. UU. (space inside). "i.e." → es decir; "e.g." → por ejemplo; "cf." → véase; "vs." → frente a; "et al." stays, roman. | "learned vs. hand-designed" → **Aprender frente a diseñar a mano:** *(test)* ✗ i.e., ✗ vs. |
| Spaces | Type ordinary spaces in `10 000`, `24 GiB`, `p. ej.`; the build makes them no-break (§10). Do not type no-break or narrow spaces yourself (TRANSLATING.md §8). No space before `%` (§4). | |
| Apostrophe | Spanish has none: no `'s`, no plural apostrophe. | ✗ los MLP's ✗ la tasa de Adam's ✓ la tasa de aprendizaje de Adam |
| Emphasis | Keep `*…*` and `**…**` on the corresponding word(s), around the whole inflected word; add none. In particular **do not italicise loans** (stride, dropout, token), although the RAE asks for italics on unadapted foreign words: in this book `*…*` marks a term being defined. | "*regression* predicts …" → la *regresión* predice … ; ✗ el *stride* (where the English has no emphasis) |
| Headline case | Spanish has no title case: capital on the first word and on proper names only. | "The Model and the Loss" → El modelo y la pérdida ✗ El Modelo y la Pérdida |

### 2.1 References

The build turns each reference into a word and a number. With Quarto's Spanish
language file the word depends only on the label's prefix, so its gender is
known:

| Label | Printed | Gender |
|---|---|---|
| `sec_…`, `subsec_…` | Sección 7.2, Sección 7.2.1 | feminine |
| `fig_…`, `img_…` | Figura 7.2.1 | feminine |
| `tab_…`, `table_…` | Tabla 7.2.1 | feminine |
| ``:eqref:`…` `` (any label) | Ecuación 7.2.1 | feminine |
| `chap_…` | Capítulo 7 | **masculine** |
| a label with no such prefix (`field_visual`) | usually Figura; not known from the label | unknown |

**Decision: write no noun of your own, but do write the article the printed
word needs: *la* for every reference except `chap_`, *el* (so *del*, *al*)
for `chap_` (to confirm).** Reason: unlike German or English, Spanish does not
drop the article here: ✗ *En Sección 7.1 dedujimos* and ✗ *Implementamos
Ecuación 7.2* are telegraphic, and all three test translators, lacking this
table, wrote them. Only for a label without a known prefix use a form that
needs no article: parentheses or *véase*.

| English | Spanish |
|---|---|
| "In :numref:`sec_why-conv` we derived …" | En la :numref:`sec_why-conv` dedujimos … ✗ En :numref:`sec_why-conv` dedujimos … *(test)* |
| "as shown in :numref:`fig_x`, the loss falls" | Como muestra la :numref:`fig_x`, la pérdida disminuye. |
| "We implement :eqref:`eq_corr2d` directly" | Implementamos la :eqref:`eq_corr2d` directamente … ✗ Implementamos :eqref:`eq_corr2d` *(test)* |
| "the MLPs of :numref:`chap_perceptrons`" | los MLP del :numref:`chap_perceptrons` ✗ de la :numref:`chap_perceptrons` |
| "see :numref:`tab_x`" | véase la :numref:`tab_x` ; (:numref:`tab_x`) |
| "the term from :numref:`field_visual`" | el término (véase :numref:`field_visual`) |
| ":numref:`sec_a` and :numref:`sec_b`" | la :numref:`sec_a` y la :numref:`sec_b` (repeat the article; never *las* + two references) |
| ":citet:`Vaswani.2017` introduced …" | :citet:`Vaswani.2017` introdujeron … (plural verb for "et al." names; ✓ el modelo de :citet:`…`) |

- Never write Sección, Figura, Tabla, Capítulo or Ecuación before a reference:
  the build prints it. ✗ la figura :numref:`fig_x`.
- Do not start a sentence with a bare reference; start with its article (La
  :numref:`fig_x` muestra …).
- `§{sec_x}` prints "§7.3" and occurs on slides: copy it where the English has
  it, without an article: (§{sec_channels}).
- A citation `:cite:` stands at the end of the clause it supports, before the
  comma or full stop, as in English.

### 2.2 Math and code in a sentence

- **Give a symbol or code name its noun and article** where the sentence needs
  gender or number: la matriz $\mathbf{X}$, el núcleo `K`, la tasa de
  aprendizaje $\eta$. A bare symbol is masculine singular: $n$ es par.
- **Sizes follow the noun, with *de*:** una matriz de $m \times n$, un núcleo de
  $2 \times 2$, una capa convolucional de $3 \times 3$, una imagen de
  $28 \times 28$ *(test)*. ✗ un $2 \times 2$ núcleo, ✗ un núcleo
  $2 \times 2$.
- **Name before size:** "a $k_\textrm{h} \times k_\textrm{w}$ kernel
  $\mathbf{K}$" → un núcleo $\mathbf{K}$ de $k_\textrm{h} \times k_\textrm{w}$.
  This reorders two math tokens, which `l10n check` reports as a warning: write
  the Spanish order and name the block in NOTES (TRANSLATING.md §4). Do not
  write the calque ✗ *el núcleo de $1 \times 2$ $\mathbf{K}$* to keep the order.
- **No Spanish ending on a symbol or code**, and no article in the plural
  before one: ✗ las `X` ✓ los tensores `X`. A hyphen joins a symbol to a word
  only in fixed forms ($k$-ésimo, $n$-grama); otherwise use *de*: "a
  $d$-dimensional vector" → un vector de dimensión $d$.
- **Protected math is copied as it is**, even when it contains English: "the
  $i^{\textrm{th}}$ layer" → la capa $i^{\textrm{th}}$ (symbol after the noun).
  Do not change it and do not report it in NOTES.
- **A closing `$` must not be followed by a digit** (the check rejects it):
  with stride $s$ and 2 layers → con stride $s$ y 2 capas, never `$s$2`.
- **Display equations:** end the clause before the equation with a colon or
  nothing, as the English does, and continue after it in lower case (*donde
  $n$ es …*). If the Spanish verb would have to follow the equation, split:
  "… is given by $$…$$ where $n$ is …" → … viene dada por $$…$$ donde $n$ es ….

## 3. Terminology: Spanish term or English loan

### 3.1 How to choose

**Use the Spanish term wherever mathematics, statistics or computer-science
teaching in Spanish has one. Use the English loan only where Spanish practice
has none or only a calque that readers do not use. Never coin a Spanish word
for an English term, and never leave an English word where a Spanish term is
established.** The glossary decides each case; for a term without an entry,
apply the rule, check Spanish Wikipedia or a Spanish-language textbook, and
record the choice in DECISIONS.

| Case | Rule | Examples |
|---|---|---|
| Established Spanish term | Use it. | capa, peso, sesgo, pérdida, núcleo (✗ kernel for a convolution), convolución, relleno (✗ padding), característica (✗ feature), tasa de aprendizaje, descenso de gradiente, sobreajuste, regularización, retropropagación, lote, minilote, época, entrenamiento, preentrenamiento, ajuste fino, atención, autoatención, codificador, decodificador, recompensa, política, verosimilitud, esperanza, conjunto de datos, exactitud, aprendizaje profundo (✗ deep learning as a common noun), aprendizaje automático (✗ aprendizaje de máquina), red neuronal (✗ red neural) |
| No Spanish term, or only a calque | English loan, lower case, roman, with gender and plural (§3.3). | dropout, stride, token, prompt, embedding, logit, softmax, momentum, bootstrap, framework, broadcasting, pooling (§12), teacher forcing, score matching, throughput, benchmark, checkpoint, pipeline |
| Names of models, architectures, optimizers, libraries, datasets, metrics; acronyms; code | Leave in English; never translate a name in part. | Transformer (§12), ResNet, BERT, Adam, AdamW, PyTorch, ImageNet, BLEU, ReLU, SGD, MLP, LSTM, GPU; Q-learning; Proximal Policy Optimization (✗ optimización de política proximal as the name); Tensor Core, im2col, GEMM |
| Name or loan + Spanish noun | Spanish noun first, then *de* or apposition; no hyphen. | el optimizador Adam, la activación ReLU, la función softmax, la tasa de dropout, la caché KV, un bloque Transformer, la memoria de la GPU, la capa de pooling |

The acronym is the English one even when the term is translated: *red neuronal
convolucional* (CNN), *descenso de gradiente estocástico* (SGD), *perceptrón
multicapa* (MLP), *modelo de lenguaje de gran tamaño* (LLM). ✗ RNC, ✗ DGE,
✗ PMC. A slide does not invent Spanish abbreviations either: ✗ **CR teórico**
*(test)* ✓ **Campo receptivo teórico** *(test)*.

### 3.2 Frequent head words, with gender

el modelo · el entrenamiento · el gradiente · la media · la capa · la matriz
(las matrices) · el estado · la probabilidad · la varianza · el valor · el peso
· la política · el vector · la entropía · la norma · el núcleo · la exactitud
(accuracy) · el crítico · el sesgo · la secuencia · la clave (key) · la consulta
(query) · la característica · el rango (rank) · la predicción · el perceptrón
multicapa (MLP) · la distribución a posteriori · el residuo · la forma (shape) ·
la acción · el conjunto de datos · el tensor · la recompensa · la optimización ·
el retorno (RL return) · la precisión · el descenso de gradiente estocástico
(SGD) · la derivada · la exhaustividad (recall) · el aprendizaje por refuerzo ·
el canal · la activación · la tasa de aprendizaje · la época · la matriz
jacobiana · la convolución · la entropía cruzada · la distribución a priori ·
la esperanza · la máscara · el tamaño de paso · el tamaño del lote · la
clasificación · la generalización · el entorno · el agente · el vocabulario ·
la pasada hacia adelante, la pasada hacia atrás · el decaimiento de pesos · el
modelo de lenguaje · el preentrenamiento · la capa oculta · la red neuronal ·
la red neuronal convolucional (CNN) · el parámetro · el hiperparámetro · la
función de pérdida · la función objetivo · el mapa de características · el
campo receptivo · el relleno · el valor propio, el vector propio · el punto de
silla · la muestra · el muestreo · la etiqueta · la traza · el producto escalar.

Adjectives agree in gender and number and follow the noun: *redes neuronales
convolucionales*, *capas ocultas*, *gradientes estocásticos*. In *X de Y*
terms the plural goes on the head: *tasas de aprendizaje*, *funciones de
pérdida*, *mapas de características*, *conjuntos de datos*.

### 3.3 Loan nouns: gender, plural, spelling

Loans are written in lower case, roman (§2), with a Spanish article. Gender:
the Spanish word the loan stands for decides if there is an obvious one;
otherwise **masculine**. Plural: add *-s* (or *-es* to an adapted loan ending
in a consonant); never invariable, never an English plural with an apostrophe.
An acronym takes the gender of its Spanish head noun and no plural ending.

| Loan | Plural | Note |
|---|---|---|
| el stride | los strides | all three test translators; el stride 2, con stride 1 |
| el dropout | — | la tasa de dropout |
| el token | los tokens | |
| el prompt | los prompts | |
| el embedding | los embeddings | (§12) capa de embedding, matriz de embeddings |
| el logit | los logits | |
| la función softmax; softmax as a name without article | — | ✗ el softmax, ✗ la softmax: write *la función softmax*, *la capa softmax*, *la regresión softmax* |
| el momentum | — | not *momento* (= moment of a distribution), not *impulso* |
| el pooling | — | (§12) el max pooling, la capa de pooling |
| el framework | los frameworks | the deep learning frameworks of the tabs; a *biblioteca* is any library |
| el array | los arrays | the code object; mathematical objects are *vector*, *matriz*, *tensor* |
| el bootstrap, el broadcasting, el teacher forcing, el score matching, el throughput | — | |
| el score | los scores | only $\nabla \log p$ (función score, red de score); a rating is *la puntuación* |
| el benchmark, el checkpoint, el pipeline, el kernel (GPU or OS code only) | -s | *el pipeline*, not *la pipeline* |
| la caché | las cachés | adapted; la caché KV; verb *almacenar en caché* |
| el píxel, el clúster, el búfer, el bit, el byte, el hardware, el software, el script | píxeles, clústeres, búferes, bits, bytes, scripts | adapted forms carry Spanish accents |
| la GPU, la CPU, la API, la RAM | las GPU | ← unidad, interfaz, memoria |
| la CNN, la RNN, la GAN, la LSTM, la GRU | las CNN | ← red, memoria, unidad |
| el MLP, el SGD, el LLM, el MDP | los MLP, los LLM | ← perceptrón, descenso, modelo, proceso |
| el Transformer | los Transformers | a name used as a common noun for the architecture; capital kept |

Model names used as names take no article: BERT usa …, los pesos de BERT.
Model families used as countable nouns are feminine when they are networks and
masculine when they are models: una ResNet-18, una VGG; un GPT, un modelo BERT.

A loan not listed here: choose by the rule above and record it in DECISIONS
with its gender.

### 3.4 Verbs

Spanish verbs for actions. A verb built on an English stem is allowed only
where the dictionary (DLE) or settled technical usage has it.

| English | Spanish | Not |
|---|---|---|
| train, pretrain, fine-tune | entrenar; preentrenar (un modelo preentrenado); ajustar (ajuste fino: *aplicar un ajuste fino a*, *un modelo ajustado*) | finetunear, tunear, afinar |
| sample (from a distribution), draw | muestrear; extraer (una muestra) de | samplear |
| overfit, underfit | sobreajustar(se): el modelo se sobreajusta; subajustar | overfitear |
| update, compute, run | actualizar; calcular; ejecutar | updatear; computar; correr |
| pad, pool, clip, mask, flatten, stack, slide | rellenar; aplicar pooling; recortar; enmascarar; aplanar; apilar; deslizar *(test)* | padear, poolear, clipear |
| reshape | cambiar la forma de *(test)* | reformar, redimensionar (= resize) |
| cache, shuffle, map, fit, plot | almacenar en caché; barajar, mezclar; aplicar, asignar (map $x$ to $y$: *lleva $x$ a $y$*); ajustar; representar | cachear, shufflear, mapear, fitear, plotear |
| tokenize, initialize, normalize, regularize, quantize, parallelize, optimize, encode, decode, index, vectorize | tokenizar, inicializar, normalizar, regularizar, cuantizar, paralelizar, optimizar, codificar, decodificar, indexar, vectorizar | encodear; descodificar (fixed: *decodificar*) |
| backpropagate, cross-correlate | retropropagar; calcular la correlación cruzada de … con … *(test)* | correlacionar cruzadamente |
| deploy, benchmark, log, download, debug, test | desplegar; evaluar, comparar; registrar; descargar; depurar; probar, evaluar | deployar, loguear, debuguear, testear |
| assume, support (a feature), remove | suponer; admitir, ser compatible con; eliminar, quitar | asumir; soportar; remover |

### 3.5 The English term in parentheses

Add the English once per page (one numbered section of the book), at the first
occurrence of a key term in body text, as `(en inglés, …)`, lower case as in
the glossary, roman, outside the emphasis markers. Check the English page in
CONTEXT: if an earlier unit of the page introduced the term, do not repeat it.

- Gloss Spanish terms whose English form the reader meets in code and papers
  and would not guess: *relleno* (en inglés, padding), *núcleo* (en inglés,
  kernel), *sobreajuste* (en inglés, overfitting), *ajuste fino* (en inglés,
  fine-tuning), *política* (en inglés, policy), *lote* (en inglés, batch).
- Do not gloss loans (dropout, token), cognates (regularización, gradiente,
  tensor, matriz, convolución, correlación cruzada), or school vocabulary
  (derivada, probabilidad, media). ✗ correlación cruzada bidimensional
  (two-dimensional cross-correlation) *(test)*.
- With an acronym: *descenso de gradiente estocástico* (en inglés, stochastic
  gradient descent, SGD); afterwards write what the English writes (SGD).
- Where the English itself names the term in emphasis, the emphasis goes on
  the Spanish term: "a border of zeros (*padding*)" → un marco de ceros
  (*relleno*; en inglés, padding) *(test)*.
- Never in headings, captions, slides, exercises or table cells.

Reason: readers go on to English papers, documentation and code; once per page
keeps the text clean.

## 4. Numbers and units

**Decision: the decimal point everywhere, a space as thousands separator.**
Numerals are copied from the English (0.5, 92.3%), in prose as in math, code,
code output and figures. Reasons: (1) inline and display math are protected and
keep the point, and outputs and figure axes show it; a decimal comma in prose
would put two conventions into one sentence (✗ con $\eta = 0.03$ la pérdida cae
por debajo de 0,01); (2) the Spanish-speaking world is itself split (comma in
Spain and South America, point in Mexico, Central America and the Caribbean),
and the *Ortografía* of 2010 accepts both and recommends the point for
unification; (3) `l10n check` compares numerals as written: `0,5` for `0.5`
is reported as "numbers not found". All three test translators kept the point.
With the point as decimal sign, a point or comma must never group thousands.

| Item | Rule | Example |
|---|---|---|
| Decimal | As in the source, with a point. | "a dropout rate of 0.5" → una tasa de dropout de 0.5 ✗ 0,5 |
| Thousands | A space from five digits on; none in four-digit numbers and years (RAE, ISO). Never `10,000` or `10.000`. | "10,000 images" → 10 000 imágenes ; 2048 ; 60 000 |
| Number words | million → millón (un millón **de** parámetros, 3 millones de …); **billion → mil millones**; **trillion → billón**. | "3 billion parameters" → 3000 millones de parámetros ✗ 3 billones ; "1.5 billion tokens" → 1500 millones de tokens |
| Digits or words | As the English: a number spelled out in English is spelled out in Spanish; digits with units, symbols and comparisons. *uno* agrees: una capa, un canal, veintiuna capas. | "two hidden layers" → dos capas ocultas ; 10 épocas |
| Percent | No space before `%`, as in the source (TRANSLATING.md §4) (to confirm). A percentage takes an article (*el*, *un*). | "45% fewer parameters" → un 45% menos de parámetros *(test)* ✗ 45% menos de parámetros *(test)* ; una exactitud del 92.3% |
| Units | Symbols as in the source, with a space (not `%`, not `°`); time words written out become segundos, minutos, horas, días. | 24 GiB, 5 ms ; "3 hours" → 3 horas |
| Ranges | In prose *de … a …* or *entre … y …*; in tables, slides and captions a hyphen without spaces, unit once. Not the English en dash. | "40–55" → de 40 a 55 ; 40-55 ; "45–50%" → entre el 45% y el 50% ; 45-50% |
| Multipliers | In prose *el doble de*, *N veces más*; keep `×` in slides, tables, captions and math. | "about 2× faster" → aproximadamente el doble de rápido ; "3× larger" → 3 veces mayor ; slide: 2× más rápido |
| Approximation | aproximadamente, unos/unas, cerca de; *aprox.* only in tables and slides; "an order of magnitude" → un orden de magnitud. | "roughly 90%" → aproximadamente el 90% ; "about 100 pixels" → unos 100 píxeles |
| Ordinals | Written out up to *décimo*, agreeing (la tercera época, el primer paso); beyond that the cardinal after the noun. No `3ª`, no `3.º` in body text. | "the 12th layer" → la capa 12 |
| Decades, years | "the 2010s" → la década de 2010; "in 2017" → en 2017. | ✗ los 2010s, ✗ los años 2010's |
| Money | Amount, space, currency name, in the source's currency. | "\$2.50" → 2.50 dólares |

Quantities, signs and orders of magnitude stay exact; do not round or convert
units or currencies. The only change to a numeral is the thousands grouping
(`10,000` → `10 000`); `l10n check` accepts it once `es` has
`check: {thousands: space}` (§10). Two cases make the check report a numeral as
missing although the translation is right; name the block in NOTES: a digit
that Spanish must write as a word ("1 path" → un solo camino *(test)*), and
"billion" ("3 billion" → 3000 millones).

## 5. Gender-inclusive language

**One rule: neutral wording where it is natural, the generic masculine
otherwise, no doubled forms and no invented endings.** No `@`, `x` or `e`
endings (✗ l@s usuari@s, lxs, les usuaries) and no slash or doubled forms
(✗ los/las investigadores/as, ✗ los usuarios y las usuarias). Reason: the RAE
does not admit the invented endings, they cannot be read aloud or by screen
readers, and doubling breaks the short sentences this book needs.

- People in general: collective or relative forms. "the practitioner … she" →
  quien entrena modelos … ; la comunidad, el equipo, el personal, las personas
  que ….
- "the reader", "the user": *usted* or *nosotros* (§1.2); *quien lee*.
- Technical roles are not people and keep their grammatical gender: el agente,
  el actor, el crítico, el generador, el discriminador, el optimizador, el
  maestro (in distillation; *teacher forcing* stays).
- Named people keep the pronoun of the source.

## 6. Names

- **Models, libraries, datasets, companies, hardware, benchmarks:** exactly as
  in English: Fashion-MNIST, RTX 4090, OpenAI, GPT-2. Product names take no
  plural: los Tensor Core *(test)* ✗ los Tensor Cores.
- **People:** as in the English source, without title; this includes
  transliterated names (Markov, Kolmogorov, Chebyshev, Lyapunov). Reason:
  citations, searches and code use these forms, and one rule can be checked.
  This departs from the RAE's transliterations (✗ Márkov, Chebyshov,
  Kolmogórov); the glossary has *cadena de Markov*.
- **Terms named after people:** Spanish noun + *de* + name, the name
  unchanged: teorema de Bayes, ecuación de Bellman, método de Newton, serie de
  Taylor, cadena de Markov, desigualdad de Chebyshev, desigualdad de
  Cauchy-Schwarz, divergencia de Kullback-Leibler, estimador de Monte Carlo,
  constante de Lipschitz, multiplicador de Lagrange.
- **Adjectives from names:** lower case, and only where the glossary or
  mathematical usage has the adjective: gaussiano (ruido gaussiano,
  distribución gaussiana), bayesiano, euclidiano (not *euclídeo*), markoviano,
  lagrangiano, jacobiana, hessiana, lipschitziano, booleano, laplaciano.
  ✗ Gaussiano, ✗ ruido Gaussiano, ✗ Bayesiano.
- **The book:** *Dive into Deep Learning* stays in English; "this book" → este
  libro.
- **Titles of English-language material** (courses, assignments, books without
  a Spanish edition) stay in English as link text, so that the reader can find
  them: [Assignment 2](…), *Neural Networks and Deep Learning* *(test, all
  three)*. Around them: "Adapted from" → Adaptado de; "notebook" → cuaderno;
  "Chapter 6" of another book → capítulo 6.
- **Languages and places:** Spanish, lower case for languages (inglés; Estados
  Unidos, in tables EE. UU.); programming languages stay (Python).
- **Structure words:** chapter capítulo, section sección, subsection
  subsección, appendix apéndice, Summary → Resumen, Exercises → Ejercicios,
  Discussions → Discusiones, Concept Check → Repaso de conceptos; Theorem →
  Teorema, Proposition → Proposición, Lemma → Lema, Corollary → Corolario,
  Definition → Definición, **Proof → Demostración** (✗ Prueba), Remark →
  Observación, Example → Ejemplo.

## 7. Senses that share a word

Spanish has one word for several English ones here; the translator must keep
the senses apart in the sentence.

| Spanish word | English senses | Rule |
|---|---|---|
| entrada | input; entry of a matrix; (dictionary) entry | *entrada* = input only. An entry of a matrix, tensor or output is **el elemento**. ✗ sumamos los productos para obtener la entrada $(i, j)$ de la salida *(test)* ✓ … el elemento $(i, j)$ de la salida |
| borde | edge (image feature); border of an image | *borde* = edge (detector de bordes, mapa de bordes). The outer frame of an image is *el margen* or *el borde de la imagen* in full; a border of zeros is *un marco de ceros*; a boundary between regions is *la frontera* *(test)* |
| rango | rank; range | *rango* = rank of a matrix. A range of values is *intervalo*; the range of a map is *imagen* or *recorrido* |
| sesgo | bias parameter $b$; bias of an estimator; social bias | one word for all three; add the noun where both occur: *el término de sesgo $b$*, *el sesgo del estimador*; unbiased → insesgado |
| muestra | sample (one draw); sample (a set of draws); generated sample | *muestra* for all; a data point is *el ejemplo* (ejemplo de entrenamiento) |
| precisión | precision (metric); numerical precision | both *precisión* (precisión mixta, doble precisión); **accuracy is never *precisión***: *exactitud* |
| paso | step (of training); step size; pass | step → *paso* (paso de gradiente, paso temporal); pass → *pasada* (pasada hacia adelante); stride → *stride* |
| forma | shape; form | *forma* = shape of a tensor (✗ dimensión, ✗ tamaño for shape); "closed form" → forma cerrada |

## 8. Headings, captions, slides, exercises, notes

| Kind | Rule | Example |
|---|---|---|
| `heading` | Noun phrase, sentence case, no full stop, no English gloss. English "-ing + object" becomes a noun + *de*, not a gerund or infinitive. A question stays a question, with ¿…?. | "Learning a Kernel" → Aprendizaje de un núcleo *(test)* ✗ Aprendiendo un núcleo ; "Edge Detection" → Detección de bordes ; "Why Depth Helps" → Por qué ayuda la profundidad ; "Why Does Depth Help?" → ¿Por qué ayuda la profundidad? |
| `caption` | Body register without *nosotros*/*usted*; a full stop exactly when the English caption has one. Keep `(a)`, `(b)`. | "Masks compose by logical AND." → Las máscaras se combinan mediante un Y lógico. |
| `slide_title`, `kicker` | Noun phrase, sentence case, as short as the English, no full stop. Drop articles where Spanish headlines allow. | `title="Implementing 2D Cross-Correlation in Code"` → `title="Implementación de la correlación cruzada 2D"` ; `[Concept Check]{.kicker}` → `[Repaso de conceptos]{.kicker}` |
| `slide_text` | Short statements with *se* or noun phrases; no *nosotros*, no *usted*, no *tú*; instructions as infinitives; one line where the English is one line; no full stop after a single-phrase bullet. Bold lead-ins keep bold and colon; a capital follows (§2). | "**4D framework tensors:** Reshape `X` and `Y` to 4D tensors … and minimize …" → **Tensores 4D del framework:** Cambiar la forma de `X` e `Y` a tensores 4D … y minimizar … ✗ … damos a `X` y `Y` forma de tensores 4D … y minimizamos *(test)* |
| Concept-check slides | Questions to the audience are impersonal: *se aplica*, *¿cuál es …?*. | "You apply an unpadded $3 \times 3$ kernel … What is the output shape?" → Se aplica un núcleo de $3 \times 3$ sin relleno … ¿Cuál es la forma de la salida? ✗ aplicas un núcleo … *(test)* |
| `exercise` | *usted* imperative for instructions (Demuestre, Calcule, Deduzca, Diseñe, Implemente, Compare, Explique); direct questions with ¿…?, conditional for "how would you": *¿Cómo diseñaría …?*. Reason: the form of Spanish-language university textbooks, and the imperative of the body register (§1.2). The bare infinitive (*Diseñar …*) is kept for slides. A bold exercise title stays bold, with its full stop. `[code]` and ● are copied. (to confirm) | "1. [code] **Hand-designed kernels.** Design some kernels manually." → 1. [code] **Núcleos diseñados a mano.** Diseñe algunos núcleos manualmente. ✗ Diseña … *(test)* ✗ Diseñar … *(test)* ; "Show that the errors sum to zero. Why?" → Demuestre que los errores suman cero. ¿Por qué? |
| `tab_note` | Body register. Describe this framework only; never compare frameworks. API names stay in code. | "PyTorch's `nn.LazyLinear(1)` requires only the output dimension." → En PyTorch, `nn.LazyLinear(1)` solo necesita la dimensión de salida. |
| `note` | Register of its surroundings. Labels: **Note:** → **Nota:**, **Warning:** → **Advertencia:**, **Answers:** → **Respuestas:**, **Tip:** → **Consejo:** | "**Answers:**" → **Respuestas:** *(test)* |
| `table_row` | Noun phrases; numbers and units as in §4. | "Training time" → Tiempo de entrenamiento |

The cover kicker `[Dive into Deep Learning · §2.1]{.kicker}` is rewritten by the
build; copy it unchanged.

## 9. Known pitfalls

### 9.1 English words with more than one sense

| English | Sense → Spanish | Not |
|---|---|---|
| kernel | convolution → el núcleo (tamaño del núcleo); kernel methods, covariance → la función núcleo, la matriz de núcleo; of a linear map → el núcleo (null space → espacio nulo); GPU or OS code → el kernel | kernel for a convolution |
| return | RL → el retorno; of a function → valor devuelto, devolver | regreso, rendimiento; retornar for code |
| score | $\nabla \log p$ → el score, la función score; attention, rating → la puntuación; a metric → the metric's name (F1, BLEU) | partitura, marcador |
| feature | → la característica (vector de características, mapa de características); of a library or product → la función, la funcionalidad | rasgo, feature |
| policy | RL → la política | póliza, directiva |
| loss, cost, objective, error, risk | Five words, five terms: pérdida, costo, función objetivo, error, riesgo. | merging them |
| accuracy, precision, recall | exactitud, precisión, exhaustividad | precisión for accuracy; recuerdo |
| recall (verb), return (verb), span (verb) | recordar, devolver, abarcar: ordinary verbs, not the glossary terms; if the check warns, name the false match in NOTES | |
| normalization | of activations or data → normalización (normalización por lotes); normal distribution → distribución normal; *regularization* → regularización; *standardization* → estandarización | mixing them |
| normal, Gaussian | "normal distribution", "standard normal" → distribución normal, normal estándar *(test)*; "Gaussian" → gaussiana | distribución gaussiana estándar |
| step size, stride | el tamaño de paso (optimization); el stride (convolution) | paso for both |
| mean, average | la media; to average → promediar; "on average" → en promedio | el medio |
| sequence | la secuencia (ML data); in mathematics → la sucesión (sucesión convergente) | secuencia for $(x_n)$ |
| map, mapping | → la aplicación, la función; feature map → mapa de características; attention map → mapa de atención | mapeo |
| inference | running a trained model, statistics → la inferencia; "the layer infers its input size" → determina automáticamente | |
| translation | of an image or function → traslación (equivarianza a traslaciones, invariancia a traslaciones) *(test)*; of language → traducción | invariancia a la traducción |
| gradient, derivative | el gradiente; la derivada, "with respect to $x$" → respecto a $x$ | la gradiente; derivado |
| likelihood, evidence | la verosimilitud; Bayesian → la evidencia; "empirical evidence" → indicios, pruebas, resultados | probabilidad; evidencia for "proof" |
| expectation | la esperanza (glossary); "in expectation" → en esperanza; "expected squared gradient" → valor esperado del cuadrado del gradiente | expectativa |
| performance | of a model → rendimiento, calidad; of code → rendimiento, velocidad | performance, desempeño |
| patch | of an image → el parche *(test, two of three)* | fragmento, trozo |
| flip, flatten, unroll, unfold | voltear (núcleo volteado; *invertido* suggests an inverse); aplanar; desenrollar; desplegar | |
| banded, sparse, dense | matriz banda; dispersa (✗ escasa, ✗ rala); densa | matriz de bandas |
| trade-off | compromiso, contrapartida; bias–variance trade-off as in the glossary | trade-off, intercambio |
| built-in, hand-crafted | integrado (la capa integrada); diseñado a mano *(test)* | incorporado, empotrado; artesanal |
| end to end | de extremo a extremo | punta a punta, end-to-end |
| converge | converge **a** $x^*$ | converge hacia, en |

### 9.2 False friends

| English | Spanish | Not (because) |
|---|---|---|
| eventually | finalmente, con el tiempo, al final | eventualmente (= possibly, occasionally) |
| actual, actually | real, verdadero; en realidad, de hecho | actual, actualmente (= current, currently) |
| consistent | coherente, uniforme; *consistent with* → acorde con; only *consistent estimator* → estimador consistente | consistente (= solid) |
| sensible | sensato, razonable | sensible (= sensitive) |
| sensitive | sensible (sensible a la inicialización) | sensitivo |
| assume | suponer | asumir (= to take on) |
| support (a feature) | admitir, ser compatible con | soportar (= to endure, to bear a load) |
| library | biblioteca | librería (= bookshop) |
| remove | eliminar, quitar | remover (= to stir) |
| realize | darse cuenta de, comprobar | realizar (= to carry out) |
| resume | reanudar | resumir (= to summarize) |
| billion, trillion | mil millones, billón | billón, trillón |
| "makes sense" | tiene sentido | hace sentido |
| significant | considerable, notable; *significativo* only in the statistical sense | significativo for "large" |
| ultimately | en última instancia, al final | últimamente (= lately) |
| discussion | análisis, debate | discusión (also = quarrel) outside the fixed heading |
| naive | ingenuo (Bayes ingenuo), directo (implementación directa) | naíf, nativo |
| random | aleatorio | randómico, random |
| topic | tema | tópico (= cliché) |
| typically | por lo general, normalmente | típicamente (every time) |
| "in order to" | para | en orden a, con el fin de (every time) |

### 9.3 Fixed logical and mathematical phrases

because → porque, ya que; therefore → por tanto, por consiguiente; however →
sin embargo, no obstante; only if → solo si; if and only if → si y solo si; at
most → como máximo, a lo sumo; at least → al menos, como mínimo; "let $x$ be
…" → Sea $x$ … (plural *Sean*); such that → tal que (plural *tales que*);
with respect to → respecto a, con respecto a; without loss of generality → sin
pérdida de generalidad; for all → para todo (para toda, para todos: agrees);
there exists → existe; respectively → respectivamente; almost surely → casi
seguramente; i.i.d. → independientes e idénticamente distribuidas (then i.i.d.,
as the English); bounded → acotado; smooth → suave; elementwise → elemento a
elemento; dot product → producto escalar; inner product → producto interno; up
to (a sign, a constant) → salvo (salvo el signo); "it follows that" → se sigue
que, de aquí que; "the following" + colon → as in English, never ✗ *lo
siguiente:* before a display equation: end the clause naturally.

### 9.4 Translationese

| Pattern | Write instead | Example |
|---|---|---|
| "It is important to note that", "Note that" word for word | Drop it, or Observemos que … once per page | ✗ Es importante notar que el sesgo no se regulariza. ✓ El sesgo no se regulariza. |
| "Recall that", "Recall from …" | Recordemos que … ; Como vimos en … | ✗ Recuerda de :numref:`subsec_why-conv-channels` que … *(test)* ✓ Recordemos de la :numref:`subsec_why-conv-channels` que … |
| "based on" as *basado en* every time; *en base a* | según, a partir de, con base en; or the verb *se basa en* | ✗ en base a los datos ✓ a partir de los datos |
| Gerund clauses | A clause with *al* + infinitive, *si*, *cuando*, *con* | "By minimizing the loss, we obtain …" ✗ Minimizando la pérdida obtenemos … ✓ Al minimizar la pérdida obtenemos … ; "Using the chain rule, …" → Con la regla de la cadena, … |
| English possessives | The article; *su* only where the owner is unambiguous | "its kernel and its bias" → el núcleo y el sesgo ; ✗ su entrada y su salida y su forma |
| *el mismo / la misma* as a pronoun | A pronoun, a demonstrative, or nothing | ✗ la capa y los parámetros de la misma ✓ la capa y sus parámetros |
| Subject first in every sentence | Put what is known first, the new information last | ✗ Calculamos luego … Actualizamos luego … ✓ A continuación calculamos … Con ello actualizamos … |
| Inanimate subject + "tells us", "allows us to" | De aquí se deduce, Esto muestra que, Así podemos | "This result tells us that depth helps." ✗ Este resultado nos dice que … ✓ Este resultado muestra que la profundidad ayuda. ; ✗ Esto nos permite calcular (every time) ✓ Así podemos calcular |
| "we can" every time as *podemos* | The verb itself, or *se puede* | "we can compute" → calculamos |
| "there is / there are" as *hay* every time | A full verb | "There are three reasons." → Tres razones lo explican. |
| "for 10 epochs" | durante 10 épocas | ✗ por 10 épocas |
| "X times faster/larger than" | N veces más rápido que; el doble de | ✗ 2 veces tan rápido como |
| "in terms of" | en función de, respecto a, expresado mediante | ✗ en términos de (every time) |
| *y/o*, *a nivel de*, *es por eso que*, *jugar un papel* | o ; en ; por eso ; desempeñar un papel | ✗ a nivel de píxel ✓ píxel a píxel *(test)* |
| Idioms translated by image | The meaning | "under the hood" → internamente ; "from scratch" → desde cero ; "out of the box" → sin configuración adicional ; "sanity check" → comprobación básica ; "rule of thumb" → regla práctica |
| Compressed English noun stacks (slide labels) | The meaning, as a Spanish noun phrase | "Edge-loss boundary shift" ✗ Desplazamiento de la frontera y pérdida por bordes *(test)* ✓ La pérdida si la banda llega al borde de la imagen *(test)* |
| Title Case, English plural or genitive | Spanish forms | ✗ La Función de Pérdida ✗ las GPUs ✗ Adam's |

## 10. Settings

In `languages.yml`, entry `es` (existing keys of `tools/l10n/config.py` and
`protect.CheckOptions`):

- `check: {thousands: space}`: `10,000` → `10 000` and `1,000` → `1000` pass
  the number check (§4). Without it every regrouped number is a warning.
- `typography: {nbsp: true}`: the build puts no-break spaces into `10 000`
  and between a number and its unit (`24 GiB`); translators type ordinary
  spaces (§2).
- Unchanged defaults: `dir: ltr`; `check.cjk_adjacent` (irrelevant);
  `check.order: warning` (name-before-size reorders, §2.2, are acknowledged in
  NOTES); `check.numbers: warning`.

In `es/strings.yml` (the existing `quarto.<key>` override mechanism):

- `pdf.lang: "es"`.
- Optional, after a build test (§11, question 4): `quarto.crossref-sec-prefix:
  "sección"`, `quarto.crossref-fig-prefix: "figura"`,
  `quarto.crossref-tbl-prefix: "tabla"`, `quarto.crossref-ch-prefix:
  "capítulo"`, `quarto.crossref-eq-prefix: "ecuación"`, leaving the caption
  titles (`crossref-fig-title`, `crossref-tbl-title`) capitalised.
- `quarto.callout-tip-title: "Consejo"` (Quarto's Spanish file has "Tip").

Would need new code:

- **No-break rules for Spanish abbreviations.** The `nbsp` rule for
  abbreviations matches single letters only (z. B.); `p. ej.`, `EE. UU.` and
  `n.º 3` are not covered.
- **PDF: babel's Spanish module must not rewrite math.** With `lang: es`,
  babel-spanish by default prints the decimal point in math as a comma and
  accents operator names (lím, máx, mín), and it names tables "Cuadro". The PDF
  build needs the options `es-nodecimaldot` and `es-tabla` and
  `\unaccentedoperators` (or their equivalent) so that the PDF matches the
  HTML. Stated from the drafter's knowledge of babel-spanish; not tested in
  this build.
- **A check for `¿`/`¡`:** a translated block that contains `?` without `¿`
  (outside protected tokens) could be a warning; no such switch exists.
- **A check for the reference article** (§2.1): `del`/`el` before a non-`chap_`
  reference, or `la` before a `chap_` one, could be a warning.

## 11. For the author to confirm

1. **Reader address** (§1.2, §8): *usted* for the few direct addresses and for
   exercises (*Demuestre que …*). Alternative: *tú* (*Demuestra que …*), which
   two of the three test translators chose and which Spain's tutorials use.
   Proposed: *usted*, as the one form that is unmarked in every country.
2. **Decimal point and thousands space** (§4): 0.5 and 10 000 in prose.
   Proposed: yes, as ruled for German, and in line with the RAE's
   recommendation of the point.
3. **Percent without a space** (§4): 92.3%, against the RAE's `92.3 %`.
   Proposed: no space, as in the source and the German ruling.
4. **Articles before references** (§2.1): *la* :numref: for sections, figures,
   tables and equations, *el/del/al* for `chap_`. This departs from the German
   ruling (no article), because Spanish grammar needs the article and the
   gender follows from the label prefix. Also: should the build print the
   in-text prefixes in lower case (la figura 7.3, as the RAE writes) or keep
   Quarto's capitals (la Figura 7.3)? Proposed: articles yes; lower case after
   a build test of the `quarto.crossref-*-prefix` overrides (§10).
5. **Capital after a bold lead-in** (§2): **Supuesto central:** Los errores ….
   The test translators wrote lower case. Proposed: capital, as in English and
   the RAE rule for epigraphs.
6. **Regional vocabulary** (§1.1): computadora, archivo, costo, video, píxel,
   bucle, punto flotante. Proposed: as listed; the Spain-only alternatives
   (ordenador, fichero, coste, vídeo, coma flotante) are excluded.
7. **Loans kept** (§3.1, §3.3): dropout, stride, token, prompt, embedding,
   logit, softmax, momentum, framework, pooling, Transformer; Spanish kept for
   relleno, núcleo, lote, minilote, ajuste fino, preentrenamiento, atención,
   retropropagación, parada temprana, decaimiento de pesos. Proposed: as
   listed.
8. **No italics on loans** (§2), against the RAE rule for unadapted foreign
   words. Proposed: none, because `*…*` marks defined terms.
9. **English gloss as `(en inglés, …)`** once per page (§3.5). Alternative:
   the bare English word in parentheses, (padding). Proposed: with *en inglés*.
10. **People's names as in English** (§6): Markov, Chebyshev, Kolmogorov.
    Proposed: yes, as ruled for German.
11. **Matrix entry = *elemento***, never *entrada* (§7); **acronyms without
    plural *-s*** (las GPU) but loans with it (los tokens, los Transformers)
    (§3.3). Proposed: as written.
12. **Exercise and slide forms** (§8): *usted* imperative in exercises, bare
    infinitive in slide instructions, impersonal *se* in concept checks.
    Proposed: as written.
13. **Post-training** has no glossary entry. Proposed: *el posentrenamiento*
    (RAE: prefix *pos-* before a consonant or vowel), with the verb phrase
    *ajustar tras el preentrenamiento*; a native reviewer should say whether
    *postentrenamiento* reads better.

## 12. Proposed glossary corrections

All of these entries are `proposed` in `glossary/terms/es.jsonl`; none is
verified. Until the glossary changes, translators follow this guide and record
the term in DECISIONS.

| Concept | Glossary now | Proposed | Reason |
|---|---|---|---|
| transformer, transformer block, transformer encoder/decoder, vision transformer, diffusion transformer | transformador, bloque transformador, codificador transformador, transformador de visión, transformador de difusión | Transformer; bloque Transformer; codificador / decodificador del Transformer; Vision Transformer (ViT); Diffusion Transformer (DiT) | An architecture name, kept in English like ResNet (TRANSLATING.md §7; same ruling in German and Japanese); *transformador* is the electrical device. |
| word / position / rotary position / patch / user / item embedding, embedding layer, matrix, table, dimension | incrustación de palabras, incrustación posicional, incrustación posicional rotatoria, incrustación de parche, capa de incrustación, matriz / tabla de incrustaciones, dimensión de la incrustación | embedding de palabras, embedding posicional, embedding posicional rotatorio, embedding de parches, capa de embedding, matriz / tabla de embeddings, dimensión del embedding | The head entry `embedding` is already the loan *embedding*; the compounds must agree with it. |
| pooling, max / average / global / global average / attention pooling, pooling layer, pooling window | agrupación, agrupación máxima, agrupación por promedio, …, capa de agrupación | pooling, max pooling, average pooling (pooling por promedio), pooling global, capa de pooling, ventana de pooling | *agrupación* / *agrupamiento* is also clustering; Spanish-language texts on CNNs write *pooling*, as the glossary already does for stride and dropout. |
| floating point, floating-point precision, floating-point format | coma flotante, precisión de coma flotante, formato de coma flotante | punto flotante, precisión de punto flotante, formato de punto flotante | Agrees with the decimal point of §4; majority usage in Latin America. |
| temperature | temperatura de softmax | temperatura (keep *temperatura de softmax* for "softmax temperature") | The book uses the bare word for sampling temperature. |
| score function | función score | keep; add alternative *función de score* | Both occur; the check should accept either. |
| cost | costo (alternative coste) | costo; remove the alternative *coste* | §1.1: one form. |
| matrix-entry sense of "entry", "patch", "edge", "edge map", "flatten", "banded matrix", "built-in" | no entries | elemento, parche, borde, mapa de bordes, aplanar, matriz banda, integrado | Choices from the translation test that later units must share (§7, §9.1). |
| deep learning, machine learning, framework, array, optimizer, post-training, standard normal distribution | no entries found | aprendizaje profundo, aprendizaje automático, framework (m.), array (m.), optimizador, posentrenamiento, distribución normal estándar | Head words of the book without an entry. |

## 13. Sources

Read for this draft, in this repository, on 2026-11-26:

- `TRANSLATING.md`, `DESIGN.md` §10, `tools/l10n/protect.py` (number matching:
  `\d+(?:[.,]\d+)*`, compared as written; thousands regrouping only with
  `thousands: space`; token order a warning), `tools/l10n/glossary.py`
  (inflected forms of a glossary term are accepted: *matriz* in *matrices*),
  `tools/l10n/config.py`, `languages.yml`.
- `glossary/terms/es.jsonl` (1,191 entries: 76 `verified_external`, 1,115
  `proposed`) and `glossary/notes/es-fr-pt.md` (sources: localized Google for
  Developers glossaries and paired Stanford CS229/CS230 notes).
- `evals/es-conv-layer/`: RESULTS.md and the nine answer files of Fable, Opus
  and Sonnet with their DECISIONS and NOTES. Examples marked *(test)* are
  quoted from them.
- The English book's build: `tools/d2l_preprocess.py` (`convert_label_id`:
  label prefix → reference kind), `tools/i18n.py` and Quarto's
  `_language-es.yml` (Figura, Tabla, Sección, Capítulo, Ecuación), and the
  finished guides `de/style.md` and `ja/style.md`.

Not checked (stated from the drafter's knowledge; no web source or book was
opened): the RAE/ASALE *Ortografía* (2010) and *Diccionario panhispánico de
dudas* for the points of §1.4, §2 and §4; the usage of Spanish-language
textbooks (*usted* imperative in exercises; *pooling*, *dropout*, *stride*
kept in Spanish books on deep learning); the regional distribution of the
word pairs in §1.1 and of the decimal sign; the babel-spanish defaults (§10).
