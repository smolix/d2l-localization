=== BRIEF ===

# Translating unit chapter_convolutional-neural-networks/conv-layer.md#u01 into Deutsch (de)

Follow TRANSLATING.md. Return the UNIT part only (every marker line copied exactly, each <!--tr:...--> slot filled), then optional DECISIONS and NOTES.

## Style guide (de/style.md)

# Style guide: German (de)

Confirmed by the author on 2026-11-26 (§10 lists the rulings). Translators
read this with every unit (TRANSLATING.md §7–§8). Each rule has one reason and
one example (English → German, or ✗ wrong / ✓ right).

Order of authority: the decisions log, then this guide, then the glossary. The
glossary says *what* a term is; this guide says *how to choose and inflect*.
The glossary carries the author's term rulings (§8.2), so the two agree.

## 1. Register

**Decision: authorial *wir*, the reader as *Sie*, never *du*.** Reason: this is
the register of German university textbooks and of the German editions of
technical books; *wir* carries the book's "we work through this together" tone
(TRANSLATING.md §8), and *du* is the tone of blogs and tutorials, which dates
quickly and does not fit proofs. Use *Sie* only where the English really
addresses the reader; most "you" sentences are general statements.

| English | German | Example |
|---|---|---|
| we + action in the text | *wir* + present tense | "We minimize the loss with SGD." → Wir minimieren den Verlust mit SGD. |
| "let us", "let's" | inverted *wir*: Betrachten wir …, Sehen wir uns … an | "Let us look at the gradient." → Betrachten wir den Gradienten. ✗ Lasst uns … / Lassen Sie uns … |
| "we will see", "we will use" | present tense; *werden* only for a real contrast with now | "We will return to this in …" → Darauf kommen wir in … zurück. |
| "you" = the reader (advice, a question to the reader, exercises) | *Sie*, *Ihr*, *Ihnen* (capitalised) | "Suppose you want to estimate the price of a house." → Angenommen, Sie möchten den Preis eines Hauses schätzen. |
| "you can", "one can" = anyone | *lässt sich*, *man*, or *wir* | "You can verify this numerically." → Das lässt sich numerisch nachprüfen. |
| "we recommend", "we believe" | *wir* | "We recommend AdamW." → Wir empfehlen AdamW. |
| "we" = people, the field | *wir Menschen*, *man*, *die Forschung* | "We humans attend selectively." → Wir Menschen richten unsere Aufmerksamkeit selektiv aus. |

**Sentence shape** (each point is checkable):

- **Verbs, not noun chains.** No sentence whose only verb is *erfolgen*,
  *stattfinden*, *durchführen*, *vornehmen* or *zur Anwendung kommen*.
  ✗ Die Berechnung des Gradienten erfolgt durch Anwendung der Kettenregel.
  ✓ Den Gradienten berechnen wir mit der Kettenregel.
- **Active with *wir*.** Passive only where the agent does not matter (Das
  Modell wurde auf ImageNet vortrainiert). No stacked passive (✗ … trainiert
  werden können wird); prefer *lässt sich* + infinitive to *kann … werden*.
- **Short verb bracket.** At most about twelve words between the finite verb
  and the part that closes the clause (participle, prefix, infinitive). If more
  are needed, split the sentence; TRANSLATING.md §3 allows splitting within a
  block. ✗ Wir werten das Modell, das wir im vorigen Abschnitt über 10 Epochen
  mit einer Lernrate von 0.1 auf Fashion-MNIST trainiert haben, aus. ✓ Wir
  werten nun das Modell aus dem vorigen Abschnitt aus. Trainiert haben wir es …
- **One main clause and at most two subordinate clauses** per sentence. Split
  English sentences at "which", "where", "so that".
- **No more than two genitives in a row, no compound of more than three parts.**
  ✗ Lernratenanpassungsverfahren ✓ Verfahren zur Anpassung der Lernrate.
- **No filler particles** (*ja*, *halt*, *eben*, *eigentlich*, *quasi*) and no
  literal "simply"/"just": drop it or write *nur*, *direkt*.
- **No Denglisch beyond the loan list in §3.** In particular never Layer, Loss,
  Weights, Feature, Learning Rate, Dataset, Task, Setting, Input, Output,
  Performance, Accuracy, Shape as German words: write Schicht, Verlust,
  Gewichte, Merkmal, Lernrate, Datensatz, Aufgabe, Einstellung, Eingabe,
  Ausgabe, Güte or Geschwindigkeit, Genauigkeit, Form.

Spelling: current official orthography (Duden), German standard (ß, not Swiss
ss). Technical spellings: Differentialgleichung, Potential (not -zial);
Kodierung, kodieren, dekodieren (but Encoder, Decoder, Code); Perzeptron; Gauß.

## 2. Punctuation and typography

| Item | Rule | Example |
|---|---|---|
| Quotation marks | „…“ (U+201E, U+201C); inside them ‚…‘. Never "…", ''…'' or »…«. Book titles in italics as in the source. | the "attention" weights → die „Attention“-Gewichte |
| Quotes in attributes | A slide `title="…"` is delimited by ASCII `"`; inside it use „…“ only. | `title="Was heißt „tief“?"` |
| Dash | The English em dash `—` becomes an en dash with spaces ( – ), at most once per sentence pair; prefer a comma, colon or parentheses. Never `—`, never a hyphen with spaces. | "the bias—an offset—shifts the plane" → der Bias (ein Offset) verschiebt die Ebene |
| Comma | German rules, not English ones: a comma before every subordinate clause (*dass*, *weil*, *wenn*, relative clauses) and before an infinitive group with *um/ohne/statt … zu*; none after a fronted adverbial; none before *und*/*oder* in a list. | ✗ In diesem Abschnitt, betrachten wir … ✓ In diesem Abschnitt betrachten wir … ; "Adam, RMSProp, and SGD" → Adam, RMSProp und SGD |
| Colon | Lower case after a colon unless a full sentence follows (then a capital). | **Kernannahme:** Die Fehler sind unabhängig. ; drei Größen: Gewicht, Bias und Lernrate |
| Semicolon | Allowed between two main clauses; do not copy every English semicolon, a full stop is usually better. | |
| Abbreviations | In body text write them out: zum Beispiel, das heißt, unter anderem, beziehungsweise, und so weiter. Abbreviations (z. B., d. h., u. a., bzw., usw., ca., vgl., Mio., Mrd.) only inside parentheses, captions, slides and tables, with a space inside multi-part ones. "i.e." → das heißt; "e.g." → zum Beispiel; "cf." → vgl.; "etc." → usw.; "et al." stays. | ✗ z.B., d.h. ✓ (z. B. Adam) |
| Spaces | Type ordinary spaces in `z. B.`, `24 GiB`, `10 000`; the German build turns them into no-break spaces. Do not type no-break or narrow spaces or soft hyphens yourself (TRANSLATING.md §8, no invisible characters). No space before `%` (§4). | ✓ `z. B.`, `24 GiB`, `92.3%` ✗ `92.3 %` |
| Apostrophe | No genitive apostrophe (✗ Adam's Lernrate ✓ die Lernrate von Adam); no plural apostrophe (✗ GPU's ✓ GPUs). | |
| Emphasis | Keep `*…*` and `**…**` on the corresponding word(s), around the whole inflected word; add none. The English gloss stays outside the markers. | "*regression* predicts …" → *Regression* sagt … vorher ; `*Überanpassung* (engl. overfitting)` |
| Headline case | German has no title case: capitalise the first word and nouns only. | "The Model and the Loss" → Das Modell und der Verlust |

**References.** The German build prints the prefix: Kapitel 3, Abschnitt 5.1,
Abbildung 4, Tabelle 2. Do not write a prefix of your own, and never put an
article or a contracted preposition before a reference: the translator does not
see which noun, and so which gender, will be printed.

| English | German |
|---|---|
| "as shown in :numref:`fig_x`, the loss falls" | Wie :numref:`fig_x` zeigt, fällt der Verlust. |
| "see :numref:`sec_sgd`" | siehe :numref:`sec_sgd` ; ✗ siehe den :numref:`sec_sgd`, ✗ im :numref:`sec_sgd` ✓ in :numref:`sec_sgd` |
| ":citet:`Vaswani.2017` introduced …" | :citet:`Vaswani.2017` führten … ein (plural verb for "et al." names; no genitive on a citation: ✓ das Modell von :citet:`…`) |

A citation `:cite:` stands at the end of the clause it supports, before the
comma or full stop, as in English.

**Math and code in a sentence.**

- Give a symbol its noun where case or gender must show: mit dem Gradienten
  $\mathbf{g}$, die Lernrate $\eta$. No genitive ending on symbols or code:
  ✓ die Norm von $\mathbf{w}$.
- Compounds with a symbol, code or number take a hyphen: $d$-dimensional,
  $k$-fach, `nn.Linear`-Schicht, 3-Gramm, $\ell_2$-Norm.
- Protected math is copied as it is, even when it contains English: "the
  $i^{\textrm{th}}$ layer" → die $i^{\textrm{th}}$ Schicht. The math stays as
  the author wrote it; do not change it and do not report it in NOTES.
- Token order: German main clauses rarely need a reorder; where a subordinate
  clause would move a token, restructure (two main clauses) before reordering.
  End the sentence before a display equation if the verb would otherwise have
  to follow it: "… is given by $$…$$ where $n$ is …" → … ist gegeben durch
  $$…$$ Dabei ist $n$ ….

## 3. Terminology: German term or English loan

### 3.1 How to choose

**Use the German term wherever mathematics, statistics or computer-science
teaching has an established one. Use the English loan where German practice has
none or only a calque. Never coin a German compound for an English term, and
never use an English word where a German term is established.** The glossary
decides each case; for a term without an entry, apply the rule, check German
Wikipedia or a German textbook, and record the choice in DECISIONS.

| Case | Rule | Examples |
|---|---|---|
| Established German term | Use it. | Schicht (✗ Layer), Gewicht, Faltung (✗ Konvolution), Merkmal (✗ Feature), Verlust (✗ Loss), Einbettung, Lernrate, Gradientenabstieg, Überanpassung, Regularisierung, Zustand, Belohnung, Strategie, Erwartungswert, Datensatz, Genauigkeit, Schrittweite, Ableitung, Kettenregel |
| No German term, or only a calque | English loan, capitalised, with gender (§3.3). | Dropout (✗ Aussetzen), Batch (✗ Stapel, Charge), Token, Prompt, Pooling (✗ Bündelung), Padding (✗ Auffüllung as a term), Stride, Momentum (✗ Impuls), Backpropagation (✗ Fehlerrückführung), Fine-Tuning, Pre-Training (✗ Vortraining), Post-Training, Attention (✗ Aufmerksamkeit), Early Stopping, Logit, Checkpoint, Sampling |
| Names of models, architectures, optimizers, libraries, datasets, metrics; acronyms; code | Leave in English; never translate a name in part. | Transformer, ResNet, BERT, Adam, AdamW, PyTorch, ImageNet, BLEU, ReLU, SGD, MLP, LSTM, GPU; Q-Learning (✗ Q-Lernen); Proximal Policy Optimization (✗ Proximale Policy-Optimierung); Long Short-Term Memory |
| Name or loan + German noun | Hyphenated compound (§3.4). | Adam-Optimierer, ReLU-Aktivierung, Softmax-Funktion, KV-Cache, Transformer-Block, GPU-Speicher |

**Convolution has two word families: the network family is *konvolutionär*;
the operation and its parts are *Faltung*.** The network is *das
konvolutionäre neuronale Netz*, short *konvolutionäres Netz*, acronym
*CNN*; likewise *vollständig konvolutionäres Netz*. The operation is *die
Faltung*, and layers, blocks and kernels are named after it: Faltungsschicht,
Faltungsblock, Faltungskern, transponierte Faltung. A network is always *das Netz*
(neuronales Netz, konvolutionäres Netz, rekurrentes neuronales Netz,
Residualnetz, Zielnetz), never *Netzwerk*.

| English | German | Not |
|---|---|---|
| "Convolutional Neural Networks" (heading) | Konvolutionäre Netze | Faltungsnetze, Konvolutionäre Netzwerke |
| "a convolutional network", "a convnet" | ein konvolutionäres Netz, ein CNN | Faltungsnetz, Convnet, konvolutionäres Netzwerk |
| "the convolutional layers of a CNN" | die Faltungsschichten eines CNN | konvolutionäre Schichten, Konvolutionsschichten |
| "the convolution of $f$ and $g$" | die Faltung von $f$ und $g$ | Konvolution |

### 3.2 Frequent head words

German terms, by frequency in the English book (`concept-frequency.tsv`), with
article and the forms that are easy to get wrong:

das Modell · das Training · der Gradient (weak: des Gradienten, den Gradienten,
die Gradienten) · der Mittelwert · die Schicht · die Matrix (die Matrizen) · der
Zustand · die Wahrscheinlichkeit · die Varianz · der Wert · das Gewicht · die
Strategie (policy) · der Vektor (die Vektoren) · die Entropie · die Norm · der
Faltungskern · die Genauigkeit (accuracy) · der Kritiker · die Verzerrung
(statistical bias) · die Sequenz · der Schlüssel (key) · das Merkmal · der Rang
· die Vorhersage · das mehrschichtige Perzeptron (MLP) · die
A-posteriori-Verteilung · das Residuum (die Residuen) · die Form (shape) · die
Aktion · der Datensatz · der Tensor (die Tensoren) · die Belohnung · die
Optimierung · der Ertrag (RL return) · die Präzision · der stochastische
Gradientenabstieg (SGD) · die Ableitung · die Trefferquote (recall) · das
bestärkende Lernen · der Kanal · die Aktivierung · die Lernrate · die Epoche ·
die Abfrage (query) · die Jacobi-Matrix · die Faltung · die Kreuzentropie · die
A-priori-Verteilung · die Einbettung · der Erwartungswert · die Maske · die
Schrittweite (step size) · die Batchgröße · die Klassifikation · die
Generalisierung · die Umgebung · der Agent (weak: des Agenten, den Agenten) ·
das Vokabular · der Vorwärtsdurchlauf, der Rückwärtsdurchlauf · der
Gewichtszerfall · das Sprachmodell · das Pre-Training · die versteckte Schicht ·
das neuronale Netz (✗ neuronales Netzwerk) · das konvolutionäre neuronale
Netz (CNN; ✗ Faltungsnetz, §3.1) · der Parameter (die Parameter, no -s) ·
die Verlustfunktion · die Zielfunktion.

Adjectives in a term are lower case in running text (stochastischer
Gradientenabstieg, neuronales Netz, konvolutionäres Netz, bestärkendes
Lernen) and are inflected: *mit dem stochastischen Gradientenabstieg*, *eines
neuronalen Netzes*, *in konvolutionären Netzen*.

### 3.3 Loan nouns: gender, genitive, plural

Loan nouns are capitalised and inflected as German nouns. Gender: the Duden
entry if there is one; otherwise *-ing* → neuter, *-er* → masculine, an acronym
takes the gender of its German head noun (die GPU ← Einheit, das LSTM, das CNN,
das RNN, das MLP ← Netz, Netz, Perzeptron). Where Duden gives two genders (Token,
Batch, Bias), the one below is fixed.

| Loan | Genitive, plural | Note |
|---|---|---|
| das Token | des Tokens, die Token | dative plural *den Token*; ✗ der Token |
| der Prompt | des Prompts, die Prompts | |
| der Batch, der Minibatch | des Batchs, die Batches | ✗ das Batch |
| der Bias | des Bias, plural via Bias-Terme, Bias-Vektoren | layer parameter only (§8.1) |
| das Dropout, das Padding, das Pooling, das Sampling, das Fine-Tuning, das Pre-Training, das Post-Training, das Q-Learning, das Batching, das Checkpointing, das Downsampling, das Ranking, das Retrieval, das Alignment | des Dropouts, des Poolings, des Fine-Tunings, des Pre-Trainings; no plural | |
| die Attention | der Attention; no plural | die Self-Attention, die Cross-Attention, die Multi-Head-Attention |
| die Backpropagation | der Backpropagation | |
| der Stride | des Strides, die Strides | |
| das Momentum | des Momentums | |
| der Transformer, der Encoder, der Decoder, der Autoencoder, der Tokenizer | des Transformers, die Transformer (no -s in the plural) | *Transformer* is a common noun for the architecture: ein Transformer, die Schichten des Transformers |
| das Sample | des Samples, die Samples | generated output only (§8.1) |
| das Logit, das Label | des Logits, die Logits; des Labels, die Labels | |
| der Score | des Scores, die Scores | |
| der Kernel | des Kernels, die Kernel | GPU/OS code and kernel methods (§8.1) |
| der Checkpoint, der Cache, der Benchmark | -s, die Checkpoints, Caches, Benchmarks | |
| die Pipeline, die Baseline, die Roofline, die Likelihood, die Ground Truth, die Beam-Search | unchanged; die Pipelines, die Baselines | |
| das Early Stopping, das Deep Learning, das Update, das Embedding | des Early Stoppings, des Updates | *Embedding* only inside English names (Rotary Position Embedding); the term is *Einbettung* |
| die Softmax-Funktion, die ReLU | | write *Softmax-Funktion* where an article is needed |
| die GPU, die CPU; das LSTM, das CNN, das RNN, das MLP, das GAN | genitive without -s (der GPU, des LSTM), plural with -s (die GPUs, die LSTMs) | |

Model families used as countable nouns take the gender of *Netz*/*Modell*: ein
ResNet-18, das LSTM. Model names used as names take no article and no ending:
BERT verwendet …, die Gewichte von BERT (✗ BERTs Gewichte).

A loan noun not listed here: choose by the rule above and record it in
DECISIONS with its gender.

### 3.4 Compounds and hyphens

Duden D 41/D 42 leave a choice in most cases; these rules fix it.

| Case | Rule | Examples |
|---|---|---|
| Never a space inside a compound | A German compound is one word or hyphenated throughout. | ✗ Batch Größe, ✗ Attention Mechanismus, ✗ Deep Learning Modell ✓ Batchgröße, Attention-Mechanismus, Deep-Learning-Modell |
| Assimilated loan + German noun: closed | Only these first parts: Batch, Minibatch, Token, Label, Pixel, Tensor, Training(s), Test, Modell, Code. | Batchgröße, Tokenfolge, Labelverteilung, Trainingsverlust, Testfehler, Modellgröße |
| Any other loan + German noun: hyphen | Also every name, acronym, symbol or number. | Attention-Gewicht, Pooling-Schicht, Dropout-Rate, Padding-Maske, Prompt-Länge, Stride-Wert, Momentum-Term, Sampling-Verfahren, Kernel-Trick, GPU-Speicher, Adam-Optimierer |
| English noun + noun | Hyphen, both capitalised. | Score-Matching, Label-Smoothing, Teacher-Forcing, Flow-Matching, Max-Pooling |
| English adjective or prefix + noun | Open, both capitalised; hyphen where the hyphenated form is the established one, and always after a prefix or a letter. | Deep Learning, Early Stopping, Ground Truth, Soft Actor-Critic; Fine-Tuning, Pre-Training, Post-Training, Self-Attention, Cross-Attention, Q-Learning |
| A multi-word loan inside a compound | Hyphens throughout (Durchkopplung). | Deep-Learning-Modell, Early-Stopping-Kriterium, Fine-Tuning-Datensatz, Pre-Training-Daten, Multi-Head-Attention, One-Hot-Kodierung, Few-Shot-Lernen, In-Context-Lernen, End-to-End-Training |
| Linking element | As in the glossary; by analogy otherwise. | Merkmalsvektor, Einbettungsschicht, Faltungsschicht, Gewichtsmatrix, Trainingsdaten; Verlustfunktion, Lernrate (no -s-) |
| German prefix + verb or noun | Closed. The training stages are not formed this way: their nouns are the loans. | vortrainieren, vortrainiert, Hyperparameter, Subwort; but das Pre-Training, das Post-Training (✗ Vortraining, ✗ Nachtraining) |

### 3.5 Verbs

German verbs for actions; English stems only where a German-inflected form is
established. Never an English participle (✗ gefinetunt, ✗ gepoolt, ✗ gesampelt,
✗ getuned).

| English | German | Not |
|---|---|---|
| train | trainieren | ausbilden, schulen |
| pretrain | vortrainieren (trainiert … vor, vortrainiert: ein vortrainiertes Modell); noun *das Pre-Training* | pretrainen; *das Vortraining* as the noun |
| post-train | no verb; use the noun *das Post-Training*: im Post-Training anpassen, nach dem Pre-Training weitertrainieren | nachtrainieren (in German usage it means to train an existing model further, that is fine-tuning or retraining, not this stage); posttrainieren |
| fine-tune | feinabstimmen (stimmt … fein ab, feinabgestimmt); noun *das Fine-Tuning* | finetunen, feintunen |
| sample (from a distribution) | ziehen: Wir ziehen $x$ aus $p$. | sampeln |
| overfit, underfit | Das Modell ist überangepasst / unterangepasst; es kommt zur Überanpassung | overfitten |
| update | aktualisieren | updaten |
| pad, pool, clip, mask | mit Padding auffüllen; Pooling anwenden; (Gradienten) begrenzen; maskieren | padden, poolen, clippen |
| cache, shuffle, map, fit | zwischenspeichern; mischen; abbilden; anpassen | cachen, shufflen, mappen, fitten |
| tokenize, initialize, normalize, regularize, quantize, parallelize, optimize, encode, decode | tokenisieren, initialisieren, normalisieren, regularisieren, quantisieren, parallelisieren, optimieren, kodieren, dekodieren | |
| plot, debug | plotten, debuggen (both in Duden) | |
| deploy, benchmark, log, download | bereitstellen; vermessen, vergleichen; protokollieren; herunterladen | deployen, benchmarken, loggen, downloaden |

### 3.6 The English term in parentheses

Add the English once per page (one numbered section of the book), at the first
occurrence of a key term in body text, as `(engl. …)`, lower case as in the
glossary, outside the emphasis markers. Check the English page in CONTEXT: if
an earlier unit of the page introduced the term, do not repeat it.

- Gloss German terms whose English differs: *Überanpassung* (engl.
  overfitting), *Faltung* (engl. convolution), *Strategie* (engl. policy),
  *Verlustfunktion* (engl. loss function).
- Do not gloss loans (Dropout, Token), cognates a reader recognises
  (Regularisierung, Gradient, Tensor, Matrix), or school vocabulary (Ableitung,
  Wahrscheinlichkeit, Mittelwert).
- With an acronym: *stochastischer Gradientenabstieg* (engl. stochastic
  gradient descent, SGD), *konvolutionäres neuronales Netz* (engl.
  convolutional neural network, CNN); afterwards write what the English writes
  (SGD, CNN).
- Never in headings, captions, slides, exercises or table cells.

Reason: readers go on to English papers, documentation and code; once per page
keeps the text clean.

## 4. Numbers and units

**Decision: the decimal point everywhere, the space as thousands separator.**
Numerals are copied from the English (0.5, 92.3%), in prose as in math, code,
code output and figures. Reason: inline and display math are protected and keep
the point, and outputs and figure axes show it; a decimal comma in prose would
put two conventions into one sentence (✗ mit $\eta = 0.1$ erreichen wir
92,3%), would need a rule for every identifier (GPT-3.5, Python 3.12), and a
German reader of a programming book reads the point without effort. This
departs from DIN 5008/Duden in prose, by the author's ruling. With the point as
decimal sign, a point or comma must never group thousands.

| Item | Rule | Example |
|---|---|---|
| Decimal | As in the source, with a point. | "a dropout rate of 0.5" → eine Dropout-Rate von 0.5 |
| Thousands | A space from five digits on; none in four-digit numbers and years. Never `10,000` or `10.000`. | "10,000 images" → 10 000 Bilder ; 2048 ; 60 000 |
| Number words | million → Million(en), **billion → Milliarde(n)**, **trillion → Billion(en)**; Mio., Mrd. only in tables and slides. | "3 billion parameters" → 3 Milliarden Parameter ✗ 3 Billionen |
| Digits or words | As the English: a number spelled out in English is spelled out in German; digits with units, symbols and comparisons. | "two hidden layers" → zwei versteckte Schichten ; 10 Epochen |
| Percent | No space before `%`, as in the source (TRANSLATING.md §4). Adjective: 5-prozentig. | "an accuracy of 92.3%" → eine Genauigkeit von 92.3% ✗ 92.3 % |
| Units | Symbols as in the source, with a space (not `%`); time words written out become Sekunden, Minuten, Stunden, Tage. | 24 GiB, 5 ms ; "3 hours" → 3 Stunden |
| Ranges | En dash without spaces, unit once; after *von* write *bis*. | "40–55" → 40–55 ; "45–50%" → 45–50% ; ✗ von 40–55 ✓ von 40 bis 55 |
| Multipliers | In prose *doppelt so* / *N-mal so* + adjective; keep `×` in slides, tables, captions and math. | "about 2× faster" → etwa doppelt so schnell ; "3× larger" → 3-mal so groß ; slide: 2× schneller |
| Approximation | etwa, rund, ungefähr; *ca.* only in tables and slides; "an order of magnitude" → eine Größenordnung. | "roughly 90%" → etwa 90% |
| Ordinals | Written out up to *zwölfte*; otherwise digit + point, not at the end of a sentence. | "the third epoch" → die dritte Epoche |
| Decades, years | "the 2010s" → die 2010er-Jahre; "in 2017" → 2017 or im Jahr 2017 (✗ in 2017). | ResNet erschien 2015. |
| Money | Amount, space, currency, as in the source's currency. | "\$2.50" → 2.50 US-Dollar |

Quantities, signs and orders of magnitude stay exact; do not round or convert
units or currencies. TRANSLATING.md §4 lists results such as `92.3%` among the
tokens to copy; in German they are copied unchanged, and the only change to a
numeral is the thousands grouping (`10,000` → `10 000`). `l10n check` accepts
this for `de` (`check: {thousands: space}` in `languages.yml`) and warns if a
German number keeps the English comma (`10,000`).

## 5. Gender-inclusive language

**One rule: neutral wording where it is natural, the plain grammatical form
otherwise, and no word-internal signs.** No Genderstern, colon, underscore,
Binnen-I or slash forms (✗ Nutzer*innen, Forscher:innen, LeserInnen). Reason:
the Rat für deutsche Rechtschreibung confirmed in 2023 that these signs are not
part of the official orthography; they also break compounds, the emphasis
markup (`*`) and screen readers.

- People in general: plural or participle forms. "the practitioner … she" →
  wer Modelle trainiert, … ; Studierende, Lehrende, Fachleute, alle, die ….
- "the reader", "the user": *Sie* (§1), or die Leserinnen und Leser at most
  once per page.
- Technical roles are not people and keep their grammatical gender: der Agent,
  der Akteur, der Kritiker, der Generator, der Diskriminator, der Lehrer (in
  Teacher-Forcing, distillation), der Optimierer.
- Named people keep the pronoun of the source.

## 6. Names

- **Models, libraries, datasets, companies, hardware, benchmarks:** exactly as
  in English: Fashion-MNIST, RTX 4090, OpenAI, GPT-2.
- **People:** as in the English source, without title; this includes
  transliterated names (Markov, Kolmogorov, Chebyshev, Lyapunov). Reason:
  citations, searches and code use these forms, and one rule can be checked.
  This departs from Duden and German Wikipedia (✗ Markow, Kolmogorow,
  Tschebyscheff); the glossary has the same forms (Markov-Kette,
  Markov-Entscheidungsprozess, Kolmogorov-Vorwärtsgleichung).
- **Terms named after people:** name + hyphen + noun, the name unchanged:
  Gauß-Verteilung, Jacobi-Matrix, Hesse-Matrix, Lagrange-Funktion,
  Newton-Verfahren, Bellman-Gleichung, Taylor-Entwicklung, Lipschitz-stetig,
  Markov-Kette, Chebyshev-Ungleichung,
  Cauchy-Schwarz-Ungleichung, Kullback-Leibler-Divergenz, Monte-Carlo-Schätzer.
  With *Satz*: Satz von Bayes (✗ Bayes' Theorem, ✗ Bayes-Theorem).
- **Adjectives from names** (Duden D 91): lower case, no apostrophe, and only
  where the glossary has the adjective or no compound exists: gaußsches
  Rauschen, bayessche Inferenz, euklidische Norm, boolesche Ungleichung.
  ✗ Gauß'sches, ✗ Bayes'sche, ✗ Gaußsches (mid-sentence).
- **German spelling of the name inside German terms:** Gauß (not Gauss), Hesse
  (Hesse-Matrix, not Hessian), Jacobi.
- **The book:** *Dive into Deep Learning* stays in English; "this book" →
  dieses Buch.
- **Languages and places:** German (Englisch, die USA); programming languages
  stay (Python).
- **Structure words:** chapter Kapitel, section Abschnitt, subsection
  Unterabschnitt, appendix Anhang, Summary → Zusammenfassung, Exercises →
  Übungen, Discussions → Diskussion; Theorem → Satz, Proposition → Proposition,
  Lemma → Lemma, Corollary → Korollar, Definition → Definition, Proof → Beweis,
  Remark → Bemerkung, Example → Beispiel.

## 7. Headings, captions, slides, exercises, notes

| Kind | Rule | Example |
|---|---|---|
| `heading` | Noun phrase, no full stop, no English gloss, no title case. English "-ing + object" becomes a noun + genitive or *von*. A question stays a question. | "Training the Model" → Training des Modells ; "The Model and the Loss" → Das Modell und der Verlust ; "Why Depth Helps" → Warum Tiefe hilft |
| `caption` | Body register without *wir*/*Sie*; a full stop exactly when the English caption has one. Keep `(a)`, `(b)`. | "Masks compose by logical AND." → Masken werden durch logisches UND verknüpft. |
| `slide_title`, `kicker` | Noun phrase, as short as the English, no full stop. | `title="The Squared Error Loss Function"` → `title="Die quadratische Verlustfunktion"` ; `[Measuring Misfit]{.kicker}` → `[Die Abweichung messen]{.kicker}` |
| `slide_text` | Short statements or noun phrases; no *wir*, no *Sie*; instructions as infinitives; one line where the English is one line; no full stop after a single-phrase bullet. Bold lead-ins keep their bold and colon. | "**Absorbing the bias:** Appending a ones column reduces the affine map to a single matrix product." → **Bias einbeziehen:** Eine Einserspalte macht aus der affinen Abbildung ein einziges Matrixprodukt ; "Lower the learning rate gradually" → Lernrate schrittweise senken |
| `exercise` | *Sie* imperative; direct questions. Reason: it is the imperative of the body register; the bare infinitive (Zeigen, dass …) reads as a note, *du* breaks the register. | "Show that the errors sum to zero. Why?" → Zeigen Sie, dass sich die Fehler zu null summieren. Warum ist das so? |
| `tab_note` | Body register. Describe this framework only; never compare frameworks. API names stay in code. | "PyTorch's `nn.LazyLinear(1)` requires only the output dimension." → `nn.LazyLinear(1)` in PyTorch benötigt nur die Ausgabedimension. |
| `note` | Register of its surroundings. Labels: **Note:** → **Hinweis:**, **Warning:** → **Achtung:**, **Answers:** → **Lösungen:**, **Tip:** → **Tipp:** | |
| `table_row` | Noun phrases; numbers and units as in §4. | "Training time" → Trainingszeit |

The cover kicker `[Dive into Deep Learning · §2.1]{.kicker}` is rewritten by the
build; copy it unchanged.

## 8. Known pitfalls

### 8.1 Words with more than one sense

| English | Sense → German | Not |
|---|---|---|
| bias | parameter $b$ → der Bias (Bias-Vektor); of an estimator → die Verzerrung, *unbiased* → erwartungstreu; fixed compounds as in the glossary (Bias-Varianz-Zerlegung, induktiver Bias); social → Voreingenommenheit | Vorurteil, Vorspannung; Verzerrung for $b$ |
| kernel | convolution → der Faltungskern (Kerngröße); kernel methods and covariance functions → der Kernel (Kernel-Funktion, Kernel-Matrix, Kernel-Trick, Kernel-Methode); GPU or OS code → der Kernel; of a linear map → der Kern (null space → Nullraum); fixed mathematical terms keep *Kern*: Kerndichteschätzer, Übergangskern | Kernel for a convolution; Kernfunktion, Kernmatrix for kernel methods |
| return | RL → der Ertrag; of a function → Rückgabewert, zurückgeben | Rückkehr, Rendite, Return |
| score | $\nabla \log p$ → die Score-Funktion, der Score; attention → der Attention-Score; a metric → der Wert (BLEU-Wert, F1-Wert) | Punktestand, Partitur |
| sample | statistics → die Stichprobe (Stichprobenmittelwert, Zufallsstichprobe); a generated output → das Sample (Sample-Qualität); a data point → das Beispiel; to sample → ziehen | Probe, Muster; Stichprobe for a generated image |
| example | a data point → das Beispiel (Trainingsbeispiel); "for example" → zum Beispiel | Exemplar |
| feature | → das Merkmal (Merkmalsvektor, Merkmalskarte); of a library or product → die Funktion | Feature, Eigenschaft |
| policy | RL → die Strategie; English inside fixed names (Policy-Gradient, On-Policy, Off-Policy) | Politik, Richtlinie |
| state | → der Zustand (versteckter Zustand, Zustandsraum); "state of the art" → Stand der Technik | Staat, Status |
| value | RL → der Wert (Wertfunktion); attention → der Wert; a number → der Wert | Value |
| action, reward, environment | die Aktion, die Belohnung, die Umgebung | Handlung, Lohn, Umwelt |
| loss, cost, objective, error, risk | Five words, five terms: Verlust, Kosten, Zielfunktion, Fehler, Risiko. | merging them |
| accuracy, precision, recall | Genauigkeit, Präzision, Trefferquote; floating-point *precision* → Genauigkeit (doppelte Genauigkeit) | Präzision for float formats |
| generalization | ML → die Generalisierung; of a theorem → die Verallgemeinerung; *generalized* linear model → verallgemeinertes lineares Modell | generalisiertes Modell |
| normalization | of activations or data → Normalisierung (Batchnormalisierung); to unit norm → normieren; normal distribution → Normalverteilung; *regularization* → Regularisierung; *standardization* → Standardisierung | mixing them |
| training, learning | train → trainieren, das Training; learn → lernen (Lernrate, überwachtes Lernen) | Ausbildung, Schulung |
| network, net | → das Netz throughout (neuronales Netz, konvolutionäres Netz, Zielnetz, Residualnetz); computer network → das Netzwerk | neuronales Netzwerk; konvolutionäres Netzwerk |
| convolution, convolutional | the operation and its parts → die Faltung (Faltungsschicht, Faltungsblock, Faltungskern); the network family → konvolutionär (konvolutionäres Netz, vollständig konvolutionäres Netz) | Faltungsnetz; konvolutionäre Schicht; Konvolution |
| step size, stride | die Schrittweite (optimization); der Stride (convolution) | Schrittweite for both |
| mean, average | der Mittelwert; to average → mitteln; "on average" → im Mittel | Durchschnitt (also means intersection of sets) |
| sequence | ML data → die Sequenz; in mathematics → die Folge (konvergente Folge) | Sequenz for $(x_n)$ |
| continuous | function → stetig; random variable → stetig; time, relaxation → kontinuierlich | kontinuierliche Funktion |
| map, mapping | → die Abbildung; the build also prints *Abbildung* before figure references, so near one write Funktion or Zuordnung; feature map → Merkmalskarte | Karte, Mapping |
| inference | running a trained model → die Inferenz; statistics → Inferenz (bayessche Inferenz); "the layer infers its input size" → bestimmt … automatisch | Schlussfolgerung |
| translation | of an image or function → Verschiebung (but Translationsinvarianz); of language → Übersetzung | Übersetzungsinvarianz |
| gradient, derivative | der Gradient (weak declension); die Ableitung, "with respect to $x$" → nach $x$ | Gefälle, Farbverlauf, Derivat |
| likelihood, evidence | die Likelihood; Bayesian → die Evidenz; "empirical evidence" → Belege, Hinweise | Wahrscheinlichkeit, Beweis |
| expectation | der Erwartungswert; "in expectation" → im Erwartungswert | Erwartung |
| rank, trace, norm | der Rang, die Spur, die Norm; ranking → das Ranking | Rangfolge for matrix rank |
| memory | hardware → der Speicher; LSTM, cognition → das Gedächtnis | |
| domain | of a function → Definitionsbereich; of data → die Domäne | |
| order | second-order method → Verfahren zweiter Ordnung; sequence order → Reihenfolge; order of magnitude → Größenordnung | |
| performance | of a model → Güte, Leistung; of code → Geschwindigkeit, Rechenleistung | Performance, Performanz |
| label | → das Label; of an axis → Achsenbeschriftung | Etikett |
| number | a count → die Anzahl; a number → die Zahl; an identifier → die Nummer | die Nummer der Parameter |
| converge | konvergiert **gegen** $x^*$ | konvergiert zu |

### 8.2 Author rulings on terms

Settled by the author on 2026-11-26; the glossary has these forms. Where an
older text or a reference work differs, these apply.

| Term | Ruling | Not |
|---|---|---|
| People's names | As in English: Markov-Kette, Markov-Entscheidungsprozess, Kolmogorov, Chebyshev (§6). | Markow-Kette, Tschebyscheff |
| Kernel methods, covariance functions | Kernel: Kernel-Funktion, Kernel-Matrix, Kernel-Trick. Reason: *Kern* is the null space of a linear map in the linear-algebra appendix. *Faltungskern* and *Kerndichteschätzer* stay (§8.1). | Kernfunktion, Kernmatrix |
| CNN | konvolutionäres neuronales Netz, short konvolutionäres Netz, CNN; the operation and its parts stay *Faltung* (§3.1). | Faltungsnetz, konvolutionäre Schicht |
| Training stages | Nouns: Fine-Tuning, Pre-Training, Post-Training. Verbs: feinabstimmen, vortrainieren (vortrainiert); none for post-training (§3.5). | Vortraining, Feinabstimmung as the nouns; nachtrainieren |
| Loans kept | Attention, Backpropagation, Early Stopping, Q-Learning. | Aufmerksamkeit, Fehlerrückführung, Q-Lernen |
| German kept | Einbettung; versteckte Schicht; bestärkendes Lernen; Strategie (policy); Ertrag (return); Abfrage, Schlüssel, Wert; Form (shape); Gewichtszerfall; Lernratenplan; Präzision, Genauigkeit, Trefferquote; Sequenz (ML) and Folge (mathematics); Kodierung; Differentialgleichung. | Embedding; verborgen, verdeckt; Weight Decay; Lernraten-Schedule |
| Token | das Token, plural die Token. Duden also allows *die Tokens*; fixed so that the two do not alternate. | der Token, die Tokens |

### 8.3 False friends

| English | German | Not (because) |
|---|---|---|
| eventually | schließlich, am Ende | eventuell (= possibly) |
| actual, actually | tatsächlich, eigentlich | aktuell (= current) |
| consistent | einheitlich, durchgängig, stimmig; *consistent with* → im Einklang mit; only *consistent estimator* → konsistenter Schätzer | konsistent for "uniform" |
| sensible | sinnvoll, vernünftig | sensibel (= sensitive) |
| sensitive | empfindlich (Sensitivität only as the statistical term) | sensitiv |
| control | steuern, regeln | kontrollieren (= to check) |
| realize | erkennen, feststellen | realisieren (= to implement) |
| "makes sense" | ist sinnvoll, ergibt Sinn | macht Sinn |
| billion, trillion | Milliarde, Billion | Billion, Trillion |
| significant | deutlich, erheblich; *signifikant* only in the statistical sense | signifikant for "large" |
| arbitrary | beliebig | willkürlich, arbiträr |
| effectively | faktisch, praktisch | effektiv (= effective) |
| novel | neu, neuartig | Novelle |
| concurrent | gleichzeitig, nebenläufig | konkurrierend |
| silicon | Silizium | Silikon |
| decade | Jahrzehnt | Dekade |
| evidence (non-Bayesian) | Belege, Hinweise | Evidenz |
| technically | streng genommen | technisch |
| dramatically | deutlich, stark | dramatisch |
| "we argue" | wir begründen, wir zeigen | wir argumentieren (every time) |
| typically | in der Regel, üblicherweise, meist | typischerweise (every time) |

### 8.4 Translationese

| Pattern | Write instead | Example |
|---|---|---|
| "It is important to note that", "Note that" word for word | Drop it, or Dabei …, Wichtig ist: … ; *Beachten Sie, dass* at most once per page | ✗ Es ist wichtig zu beachten, dass der Bias nicht regularisiert wird. ✓ Der Bias wird dabei nicht regularisiert. |
| "Recall that" | Zur Erinnerung: … ; Wie in … gesehen, … | ✗ Erinnern Sie sich, dass $b$ der Bias ist. ✓ Zur Erinnerung: $b$ ist der Bias. |
| "based on" as *basierend auf* | anhand, aus, auf der Grundlage von, or the verb *beruht auf* | "a model based on attention" ✗ ein auf Attention basierendes Modell (every time) ✓ ein Modell, das auf Attention beruht |
| Gerund and participle clauses | A *wenn*/*indem* clause or a preposition | "By minimizing the loss, we obtain …" ✗ Durch das Minimieren des Verlusts erhalten wir … ✓ Minimieren wir den Verlust, so erhalten wir … ; "Using the chain rule, …" ✗ Die Kettenregel verwendend / Unter Verwendung der Kettenregel ✓ Mit der Kettenregel |
| Possessive and *of* chains | A relative clause or a compound | "the shape of the output of the first layer of the network" ✗ die Form der Ausgabe der ersten Schicht des Netzes ✓ die Form, die die erste Schicht des Netzes ausgibt |
| English word order: subject first in every sentence | Put what is known first, the new information last; vary the first position | ✗ Wir berechnen dann … Wir aktualisieren dann … ✓ Anschließend berechnen wir … Damit aktualisieren wir … |
| Comma after a fronted adverb | None; or move the adverb inside | ✗ Jedoch, das Modell … ✗ Zudem, … ✓ Das Modell … jedoch … ; Allerdings ist … |
| Inanimate subject + "tells us", "allows us to" | Daran sehen wir, Daraus folgt, Damit können wir | "This result tells us that depth helps." ✗ Dieses Ergebnis sagt uns, dass Tiefe hilft. ✓ Daran sehen wir, dass Tiefe hilft. ; ✗ Dies erlaubt es uns, … zu berechnen ✓ Damit können wir … berechnen |
| "we can" every time as *wir können* | The verb itself, or *lässt sich* | "we can compute" → wir berechnen ; … lässt sich berechnen |
| "there is / there are" as *es gibt* every time | A full verb | "There are three reasons." → Dafür sprechen drei Gründe. |
| "for 10 epochs" | 10 Epochen lang, über 10 Epochen | ✗ für 10 Epochen trainieren |
| "one of the most widely used" | zu den … gehören | ✗ einer der am meisten verwendeten Optimierer ✓ gehört zu den meistverwendeten Optimierern |
| "in terms of" | hinsichtlich, gemessen an, ausgedrückt durch | ✗ in Bezug auf (every time), ✗ in Termen von |
| "this means" | das heißt, das bedeutet | ✗ das meint |
| Idioms translated by image | The meaning | "under the hood" → intern ; "from scratch" → von Grund auf ; "out of the box" → ohne weitere Anpassung ; "sanity check" → Plausibilitätsprüfung |
| Title Case, English plural or genitive | German forms | ✗ Die Verlust-Funktion ✗ die Parameters ✗ Adam's |

Logical and mathematical phrases are fixed: because → weil, da; therefore →
daher, folglich, also; however → jedoch, allerdings; only if → nur wenn; if and
only if → genau dann, wenn; at most → höchstens; at least → mindestens; "let
$x$ be …" → Sei $x$ …; such that → sodass, mit; with respect to → bezüglich
(derivative: nach); without loss of generality → ohne Beschränkung der
Allgemeinheit; for all → für alle; there exists → es gibt, es existiert;
respectively → jeweils, beziehungsweise; almost surely → fast sicher; i.i.d. →
unabhängig und identisch verteilt (then i.i.d., as the English); bounded →
beschränkt; smooth → glatt; dot product → Skalarprodukt.

## 9. Sources

Checked on 2026-11-26:

- Duden online, entries Token (das oder der; des Tokens; die Token[s]), Batch
  (der oder das; des Batchs; die Batches), Prompt (der; des Prompts; die
  Prompts), Bias (der oder das; des Bias; die Bias):
  https://www.duden.de/rechtschreibung/Token, …/Batch, …/Prompt, …/Bias.
- Duden, Rechtschreibregeln, D 91 (adjectives from personal names: *die
  darwinsche oder Darwin'sche Evolutionstheorie*):
  https://www.duden.de/sprachwissen/rechtschreibregeln/gross-und-kleinschreibung
- Duden, Rechtschreibregeln, D 41 and D 42 (compounds from English; mixed
  compounds follow the general rules: Blackboxverfahren, Black-Box-Verfahren):
  https://www.duden.de/sprachwissen/rechtschreibregeln/fremdwoerter
- Rat für deutsche Rechtschreibung, statement of 2023-12-15 confirming the
  decision of 2023-07-14 not to take the Genderstern, underscore, colon or
  other word-internal signs into the official rules:
  https://www.rechtschreibrat.com/geschlechtergerechte-schreibung-erlaeuterungen-begruendung-und-kriterien-vom-15-12-2023/
- German Wikipedia: *Schreibweise von Zahlen* (decimal comma; thousands grouped
  with spaces under DIN 1333, DIN 5008 and ISO 80000-1; four-digit numbers
  usually ungrouped); *Markow-Kette* (title and variants); *Transformer
  (Maschinelles Lernen)* (das Token, Schicht, Abfragevektor, Schlüsselvektor,
  Wertevektor, Positionskodierung, feinabgestimmt; it writes
  Aufmerksamkeitsmechanismus and Kodierer where this guide writes Attention and
  Encoder); *Künstliches neuronales Netz* (Netz far more often than Netzwerk;
  versteckte and verdeckte Schicht; Überanpassung; Lernrate; bestärkendes
  Lernen; Backpropagation beside Fehlerrückführung).
- `glossary/review/de.md`, `de-corrections.jsonl`, `concept-frequency.tsv` and
  `glossary/terms/de.jsonl` of this repository.

Not checked:

- Goodfellow, Bengio, Courville, *Deep Learning. Das umfassende Handbuch*
  (mitp, 2018): the catalogue record was found, the table of contents and text
  could not be retrieved. The loan decisions in §3 and the adjective for
  "hidden" rest on the author's rulings (§10), not on this book.
- German university course material: none was read.
- DIN 5008 itself (space before %, spaces in z. B.): stated from the
  drafter's knowledge and from the Wikipedia summary above.
- The existence of a Duden entry *feintunen* was seen in search results only.
- *nachtrainieren*: a web search on 2026-11-26 found no dictionary entry and
  no use for the post-training stage; German ML texts use the word for training
  an existing model further (fine-tuning, retraining). §3.5 therefore gives
  post-training no verb.

## 10. Author rulings

Confirmed by the author on 2026-11-26. One line per ruling; the rule itself is
in the section named.

1. **Decimal sign:** the point everywhere, in prose as in math and code (§4).
2. **Thousands and percent:** a space from five digits on, none in four-digit
   numbers (10 000, 2048); `l10n check` accepts this for `de`. No space
   before `%` (92.3%), as in TRANSLATING.md §4; the adjective is 5-prozentig
   (§4).
3. **No-break spaces:** the build makes them in `z. B.`, `10 000`, `24 GiB`;
   translators type ordinary spaces (§2).
4. **Reader:** *Sie*, never *du* (§1).
5. **Exercises:** *Sie* imperative (§7).
6. **Gender-inclusive language:** neutral forms where natural, grammatical
   gender otherwise, no word-internal signs (§5).
7. **People's names:** as in English: Markov, Kolmogorov, Chebyshev (§6).
8. **Adjectives from names:** lower case without apostrophe (gaußsches
   Rauschen); name compounds preferred (Gauß-Verteilung, Satz von Bayes) (§6).
9. **Terms:** Attention, Einbettung, Backpropagation, Early Stopping,
   Q-Learning (§3.1, §8.2).
10. **Training stages:** the nouns are Fine-Tuning, Pre-Training and
    Post-Training (not Vortraining). Verbs: feinabstimmen, vortrainieren /
    vortrainiert; post-training has no verb and is phrased with the noun
    (§3.4, §3.5).
11. **Kernel:** Kernel for kernel methods and covariance functions
    (Kernel-Funktion, Kernel-Matrix), Faltungskern for convolutions, Kern for
    the null space (§8.1).
12. **CNN:** konvolutionäres neuronales Netz, heading "Konvolutionäre
    Netze" (not Faltungsnetz); *Netz* for every network or net (revised
    the same day: not *Netzwerk*); the operation and its parts
    stay Faltung, Faltungsschicht, Faltungsblock, Faltungskern (§3.1).
13. **Loan genders and plurals:** as in the table of §3.3.
14. **Compounds:** closed only for the short list of first parts, a hyphen for
    every other loan (§3.4).
15. **Open glossary choices:** all as the glossary has them; listed in §8.2.
16. **Math that contains English** (`$i^{\textrm{th}}$`): stays as it is;
    translators copy it and do not report it (§2).
17. **Reference prefixes:** the German build prints Kapitel 3, Tabelle 2,
    Abbildung 4, Abschnitt 5.1; translators write no prefix and no article (§2).
18. **Abbreviations:** written out in body text; z. B., d. h. only in
    parentheses, captions, slides and tables (§2).
19. **English gloss:** `(engl. …)` once per page, only for German terms whose
    English differs (§3.6).
20. **Follow-up the same day:** *das Netz* for every network or net, including
    the convolutional family (konvolutionäres neuronales Netz); *Kern* stays in
    Kerndichteschätzer and Übergangskern; *Abbildung* serves for both a figure
    and a mapping.

## Decisions log (de/decisions.md)

(no decisions recorded yet)

## Glossary entries for this unit (de)

| English | Sense | Term | Status | Alternatives |
|---|---|---|---|---|
| backpropagation | An algorithm that propagates derivatives backward through a computation. | Backpropagation | verified_external |  |
| batch | A set of examples processed together in one forward/backward pass (the full dataset in batch gradient descent, or a minibatch). | Batch | proposed |  |
| bias | An additive offset in a model or prediction. | Bias | proposed |  |
| bias (estimator bias) | The difference between an estimator’s expected value and the true parameter. | Verzerrung | verified_external |  |
| channel | A feature dimension in a structured representation such as an image. | Kanal | proposed |  |
| convergence | The approach of a sequence or process toward a limiting value or behavior. | Konvergenz | verified_external |  |
| convolution | An operation that combines local input values with shared kernel weights. | Faltung | proposed |  |
| convolutional layer | A layer that applies learned convolutional filters to its input. | Faltungsschicht | proposed |  |
| convolutional neural network (CNN) | A neural network using convolutional layers to process structured data. | konvolutionäres neuronales Netz | verified_external | konvolutionäres Netz, CNN |
| cross-correlation | An operation that combines a local input window with a kernel without reversing the kernel. | Kreuzkorrelation | proposed |  |
| derivative | The rate of change of a function with respect to an input. | Ableitung | proposed |  |
| dilation | Spacing inserted between sampled positions of a convolutional kernel. | Dilatation | proposed |  |
| feature | An observed input variable used by a model. | Merkmal | proposed |  |
| filter | A learned set of local weights used to extract a pattern. | Filter | proposed |  |
| finite difference | An approximation to a derivative based on changes over a finite input interval. | endliche Differenz | proposed |  |
| forward pass (forward propagation) | The computation that evaluates a model from inputs to outputs. | Vorwärtsdurchlauf | proposed |  |
| gradient | The vector of partial derivatives of a scalar function. | Gradient | proposed |  |
| gradient descent | An optimization method that moves parameters against the objective gradient. | Gradientenabstieg | verified_external |  |
| inner product | An operation generalizing the dot product and inducing notions of length and angle. | inneres Produkt | proposed |  |
| kernel (convolutional kernel) | A set of weights applied locally in a convolution. | Faltungskern | verified_external |  |
| kernel size | The spatial dimensions of a convolution or pooling kernel. | Kerngröße | proposed |  |
| layer | A stage of a neural network that transforms its input. | Schicht | proposed |  |
| learning rate | The step-size setting used in an optimization update. | Lernrate | verified_external |  |
| loss | The value of the loss function for a prediction, an example or a minibatch; the quantity training minimizes. | Verlust | verified_external |  |
| matrix | A rectangular array of numerical entries. | Matrix | proposed |  |
| matrix multiplication | A product combining rows of one matrix with columns of another. | Matrizenmultiplikation | proposed |  |
| model | A mathematical or computational representation used to describe data or make predictions. | Modell | verified_external |  |
| multilayer perceptron (MLP) | A feedforward neural network with multiple layers of learned transformations. | mehrschichtiges Perzeptron | proposed |  |
| padding | Extra values added around an input to control an operation's boundary or output size. | Padding | proposed |  |
| pixel | An individual sample in a digital image. | Pixel | proposed |  |
| receptive field | The region of an input that can influence a particular activation. | rezeptives Feld | proposed |  |
| return | The accumulated future reward from a point in a trajectory. | Ertrag | verified_external |  |
| scalar | A single numerical quantity. | Skalar | proposed |  |
| shape | The sizes of the axes of an array or tensor. | Form | verified_external |  |
| slice | A selected portion of an array or sequence. | Slice | proposed |  |
| squared error | The square of the difference between a prediction and its target. | quadratischer Fehler | proposed |  |
| stochastic gradient descent (SGD) | A gradient method using a randomly sampled estimate of the gradient. | stochastischer Gradientenabstieg | proposed |  |
| stride | The spacing between successive positions of an operation on an input. | Stride | proposed |  |
| tensor | A multidimensional array generalizing scalars, vectors, and matrices. | Tensor | proposed |  |
| translation equivariance | The property that translating an input translates its representation correspondingly. | Translationsäquivarianz | proposed |  |
| weight | A coefficient that scales an input or connection in a model. | Gewicht | verified_external |  |

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

(the previous unit, chapter_convolutional-neural-networks/why-conv.md#u02, is not translated yet)

=== UNIT ===

<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u01 lang=de rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:33d3b96b kind=heading-->
# Convolutions for Images
<!--tr:33d3b96b-->

<!--en:852f5868 kind=paragraph-->
In :numref:`sec_why-conv`, we derived the convolutional layer on paper from two
geometric principles: translation equivariance and locality. Now we see what
that formula actually computes on an image, and whether gradient descent can
discover useful kernels automatically from data.
<!--tr:852f5868-->

<!--en:55f0a41b kind=paragraph-->
Consider locating vertical edges where a bright region meets a dark one.
Subtracting each pixel's right-hand neighbor ($[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$)
yields zero inside uniform regions, $+1$ where white transitions to black, and
$-1$ where black returns to white. Sweeping the two-weight detector $(1, -1)$
across every valid image location produces an *edge map*. In this section, we
implement two-dimensional cross-correlation, test our hand-crafted edge
detector, let gradient descent *learn* $(1, -1)$ directly from data, show how
a convolution becomes a single matrix multiplication (*im2col*), and
derive how stacking layers widens a unit's receptive field.
<!--tr:55f0a41b-->

<!--en:47958ed8 kind=heading-->
## Cross-Correlation
<!--tr:47958ed8-->

<!--en:5e4296fe kind=heading-->
### The Operation
<!--tr:5e4296fe-->

<!--en:ae7e906d kind=paragraph-->
Leaving channels aside for the moment, let the input $\mathbf{X}$ be an
$n_\textrm{h} \times n_\textrm{w}$ matrix and the kernel $\mathbf{K}$ be a
$k_\textrm{h} \times k_\textrm{w}$ matrix. The kernel defines a sliding
*window* of shape $k_\textrm{h} \times k_\textrm{w}$. Placing the window's
upper-left corner at $(i, j)$, we multiply the covered input patch elementwise
with $\mathbf{K}$ and sum the products to produce the output entry at $(i, j)$:
<!--tr:ae7e906d-->

<!--en:3688b38b kind=paragraph-->
This is the *two-dimensional cross-correlation* of $\mathbf{X}$ with
$\mathbf{K}$, written $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ with
zero-based indices matching code. Whereas :numref:`sec_why-conv` centered
offsets symmetrically from $-\Delta$ to $\Delta$, anchoring at the upper-left
corner differs only by a fixed shift of $\Delta$ while naturally handling
even-sized kernels (such as a $2 \times 2$ kernel or our $1 \times 2$ edge
detector) that have no center pixel.
<!--tr:3688b38b-->

<!--en:fa1b7da9 kind=caption-->
![Two-dimensional cross-correlation. The shaded portions are the first output element and the input and kernel elements used to compute it: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
<!--tr:fa1b7da9-->

<!--en:2f8e5deb kind=paragraph-->
In :numref:`fig_correlation`, we slide a $2 \times 2$ kernel across a
$3 \times 3$ input one pixel at a time, left to right and top to bottom. The
window fits cleanly at four positions, yielding the $2 \times 2$ output:
<!--tr:2f8e5deb-->

<!--en:4741e012 kind=paragraph-->
Notice that the output shrinks slightly along each axis because the kernel must
stay wholly inside the input. Fitting a width-$k$ window into length $n$ leaves
$n - k + 1$ valid positions, so the output shape is
<!--tr:4741e012-->

<!--en:12d81c14 kind=paragraph-->
Thus a $3 \times 3$ input with a $2 \times 2$ kernel yields $2 \times 2$, a
$6 \times 8$ image with a $1 \times 2$ kernel yields $6 \times 7$, and a
$28 \times 28$ Fashion-MNIST image with a $5 \times 5$ kernel yields
$24 \times 24$. Whenever we want to preserve spatial dimensions, we surround the
input with a border of zeros (*padding*), which we examine alongside strides and
dilation in :numref:`sec_padding`.
<!--tr:12d81c14-->

<!--en:869d223f kind=heading-->
### Implementation
<!--tr:869d223f-->

<!--en:6a869f7c kind=paragraph-->
We implement :eqref:`eq_corr2d` directly in `corr2d` using a pair of loops over
every output coordinate.
<!--tr:6a869f7c-->

<!--en:81e1f383 kind=paragraph-->
To verify `corr2d` against our hand calculation in :numref:`fig_correlation`,
we construct the $3 \times 3$ input tensor `X` and $2 \times 2$ kernel `K` and
evaluate `corr2d(X, K)`.
<!--tr:81e1f383-->

<!--en:f567e1b2 kind=paragraph-->
The output reproduces :numref:`fig_correlation` entry for entry (and as shown in
:numref:`sec_why-conv`, strict mathematical convolution with a flipped kernel
$\mathbf{K}'$ learns the exact same family of layers, so we follow standard
deep learning terminology and call cross-correlation "convolution").
<!--tr:f567e1b2-->

<!--en:bfe11c62 kind=heading-->
## Convolutional Layers
<!--tr:bfe11c62-->

<!--en:2cdc020d kind=heading-->
### The Layer
<!--tr:2cdc020d-->

<!--en:c6371ece kind=paragraph-->
A two-dimensional convolutional layer cross-correlates its input with a learned
kernel and adds a scalar bias. Just like our MLP weights in
:numref:`chap_perceptrons`, we initialize the kernel from
$\mathcal{N}(0, 0.01^2)$ and the bias to zero. Below, `Conv2D` registers both
parameters and invokes `corr2d` in its forward pass; a layer with a kernel of
height $h$ and width $w$ is called an $h \times w$ convolutional layer.
<!--tr:c6371ece-->

<!--en:29cd9061 kind=paragraph-->
Unlike a dense layer, a convolutional layer is not locked to a fixed input
resolution because its parameters depend only on the kernel size. The exact same
$2 \times 2$ layer (5 parameters total) processes both our $3 \times 3$ tensor
`X` and a $6 \times 8$ image, returning outputs of shapes $2 \times 2$ and
$5 \times 7$:
<!--tr:29cd9061-->

<!--en:c9e5aa07 kind=heading-->
### Edge Detection
<!--tr:c9e5aa07-->

<!--en:3d9a357b kind=paragraph-->
We now test the vertical edge detector from the introduction on a
$6 \times 8$ synthetic image whose middle four columns are black (`0`) and outer
columns are white (`1`).
<!--tr:3d9a357b-->

<!--en:68513ff0 kind=paragraph-->
We define the $1 \times 2$ kernel
<!--tr:68513ff0-->

<!--en:eb45803b kind=paragraph-->
At every position $(i, j)$, this computes the horizontal *finite difference*
$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, a discrete approximation (up to sign)
of the horizontal derivative of brightness, which vanishes across flat regions
and spikes wherever intensity jumps.
<!--tr:eb45803b-->

<!--en:e810b3a9 kind=paragraph-->
Cross-correlating the $6 \times 8$ stripe image `X` with the $1 \times 2$
horizontal-difference kernel `K` via `corr2d(X, K)` produces the $6 \times 7$
edge map `Y`:
<!--tr:e810b3a9-->

<!--en:568358e4 kind=paragraph-->
Sure enough, `Y` fires `+1` along the white-to-black boundary, `-1` along the
black-to-white boundary, and `0` everywhere else. Applying the exact same kernel
to the transposed image (turning the vertical stripe into a horizontal one)
yields all zeros:
<!--tr:568358e4-->

<!--en:71c591a4 kind=paragraph-->
Clearly `[1, -1]` responds strictly to vertical edges. Detecting horizontal
edges, diagonals, corners, or textures requires different kernels altogether.
<!--tr:71c591a4-->

<!--en:f0af4887 kind=heading-->
### Learning a Kernel
<!--tr:f0af4887-->

<!--en:61cacfaf kind=paragraph-->
Hand-crafting `[1, -1]` is easy when we know the exact feature we want, yet no
engineer can specify thousands of intermediate filters across a deep network by
hand. Can gradient descent recover the kernel `[1, -1]` from the input–output
pair `(X, Y)` alone? We test this using each framework's built-in 2D
convolutional layer (with bias omitted and weights initialized from
$\mathcal{N}(0, 0.01^2)$), minimizing squared error summed across all 42 output
entries over 10 gradient steps at learning rate $0.03$. Because built-in layers
expect 4D tensors including batch and channel axes (:numref:`sec_channels`), we
reshape `X` and `Y` accordingly.
<!--tr:61cacfaf-->

<!--en:c29c5d61 kind=paragraph-->
Initially the loss sits near 12 (the near-zero random kernel misses all 12
$\pm 1$ edge pixels by roughly 1), and within just nine steps it drops below
0.01. Inspecting the learned kernel confirms what it found:
<!--tr:c29c5d61-->

<!--en:67d25b0b kind=paragraph-->
Both entries land within roughly 0.02 of `[1, -1]`. Nothing in this loop knew
anything about edges: given a loss signal, gradient descent discovered the
filter on its own. Here we supervised the single layer directly with `Y`; in a
deep CNN, backpropagation learns every layer's kernels end to end from the final
task loss alone.
<!--tr:67d25b0b-->

<!--en:91b0010d kind=slide_text-->
Convolutions for Images<br>**2D cross-correlation · edge detection · learned kernels · im2col · receptive fields**
<!--tr:91b0010d-->

<!--en:1d94295a kind=slide_title-->
::: {.slide title="Detecting Local Edges Across an Image"}
<!--tr:1d94295a-->

<!--en:687ed302 kind=kicker-->
[Motivation & Core Mental Model]{.kicker}
<!--tr:687ed302-->

<!--en:05fceed0 kind=slide_text-->
- **Synthetic $6 \times 8$ test image:** White outer columns (`1`) surrounding a 4-column black vertical band (`0`).
<!--tr:05fceed0-->

<!--en:71e700b3 kind=slide_text-->
- **Horizontal finite difference:** Sweeping $\mathbf{K} = [1, -1]$ computes $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, firing `+1` at white-to-black and `-1` at black-to-white edges.
<!--tr:71e700b3-->

<!--en:957dc033 kind=slide_text-->
- **Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent.
<!--tr:957dc033-->

<!--en:cb1045e1 kind=slide_title-->
::: {.slide title="The 2D Cross-Correlation Operation"}
<!--tr:cb1045e1-->

<!--en:17bcece5 kind=kicker-->
[Sliding-Window Inner Products]{.kicker}
<!--tr:17bcece5-->

<!--en:8d45aae8 kind=slide_text-->
Given an $n_\textrm{h} \times n_\textrm{w}$ input matrix $\mathbf{X}$ and a $k_\textrm{h} \times k_\textrm{w}$ kernel $\mathbf{K}$, the 2D cross-correlation $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ places the window's top-left corner at $(i, j)$ and sums elementwise products:
<!--tr:8d45aae8-->

<!--en:e484380c kind=slide_text-->
- **Top-left anchor:** Zero-based indices $a, b \ge 0$ match array indexing in code and handle even-sized kernels (like $2 \times 2$ or $1 \times 2$) cleanly.
<!--tr:e484380c-->

<!--en:137c0980 kind=slide_text-->
- **Shaded top-left output $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.
<!--tr:137c0980-->

<!--en:575c4aea kind=slide_title-->
::: {.slide title="Valid Window Positions and Output Spatial Shape"}
<!--tr:575c4aea-->

<!--en:7eda4531 kind=kicker-->
[Why Unpadded Convolutions Shrink the Grid]{.kicker}
<!--tr:7eda4531-->

<!--en:891243ae kind=slide_text-->
- **Valid window placements:** A $k_\textrm{h} \times k_\textrm{w}$ window fits inside an $n_\textrm{h} \times n_\textrm{w}$ grid at $n - k + 1$ positions per axis.
<!--tr:891243ae-->

<!--en:55cc8c32 kind=slide_text-->
- **Shape examples:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.
<!--tr:55cc8c32-->

<!--en:1e3463da kind=slide_title-->
::: {.slide title="Implementing 2D Cross-Correlation in Code"}
<!--tr:1e3463da-->

<!--en:07566992 kind=kicker-->
[Direct Nested-Loop Implementation]{.kicker}
<!--tr:07566992-->

<!--en:202aa5d3 kind=slide_text-->
The function `corr2d` allocates the $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ output tensor and computes each entry as the sum of `X[i:i+h, j:j+w] * K`:
<!--tr:202aa5d3-->

<!--en:bef130ba kind=slide_text-->
Each iteration slices a $k_\textrm{h} \times k_\textrm{w}$ sub-window of `X`, multiplies elementwise with `K`, and reduces via `sum`.
<!--tr:bef130ba-->

<!--en:4ee9f6b1 kind=slide_title-->
::: {.slide title="Verifying corr2d and the Flipped-Kernel Equivalence"}
<!--tr:4ee9f6b1-->

<!--en:2b0be423 kind=kicker-->
[Cross-Correlation vs. Mathematical Convolution]{.kicker}
<!--tr:2b0be423-->

<!--en:74b9b745 kind=slide_text-->
- **180° flipped kernel $\mathbf{K}'$:** Cross-correlating with $\mathbf{K}$ equals convolving with $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.
<!--tr:74b9b745-->

<!--en:5c8af7cd kind=slide_text-->
- **Identical learned class:** Frameworks compute cross-correlation and call it convolution.
<!--tr:5c8af7cd-->

<!--en:a69e29b5 kind=slide_title-->
::: {.slide title="Building a 2D Convolutional Layer Module"}
<!--tr:a69e29b5-->

<!--en:55763603 kind=kicker-->
[Learnable Kernel Weights Plus Scalar Bias]{.kicker}
<!--tr:55763603-->

<!--en:5b8bad24 kind=slide_text-->
A convolutional layer wraps `corr2d` with a learnable $k_\textrm{h} \times k_\textrm{w}$ weight parameter (initialized from $\mathcal{N}(0, 0.01^2)$) and a scalar bias (initialized to $0$):
<!--tr:5b8bad24-->

<!--en:c16d97d6 kind=slide_title-->
::: {.slide title="Resolution Independence of Convolutional Layers"}
<!--tr:c16d97d6-->

<!--en:a7d1e48f kind=kicker-->
[Fixed Parameters Across Arbitrary Input Sizes]{.kicker}
<!--tr:a7d1e48f-->

<!--en:53a2bce7 kind=slide_text-->
- **Decoupled from $H \times W$:** The exact same 5 parameters ($2 \times 2$ kernel + 1 bias) process $3 \times 3$, $6 \times 8$, or megapixel inputs without reshaping weights.
<!--tr:53a2bce7-->

<!--en:017fec24 kind=slide_title-->
::: {.slide title="Constructing a Synthetic Vertical-Edge Image"}
<!--tr:017fec24-->

<!--en:896e641c kind=kicker-->
[A 6×8 Binary Test Pattern]{.kicker}
<!--tr:896e641c-->

<!--en:0a4b456b kind=slide_text-->
We construct the $6 \times 8$ test image `X` with ones on the left and right borders and zeros across the middle four columns (`X[:, 2:6] = 0`):
<!--tr:0a4b456b-->

<!--en:dfc7e3e3 kind=slide_text-->
- **Piecewise-constant columns:** Intensity is constant (`1` or `0`) within columns $0\text{--}1$, $2\text{--}5$, and $6\text{--}7$, jumping only between columns $1 \to 2$ and $5 \to 6$.
<!--tr:dfc7e3e3-->

<!--en:3ce31cbe kind=slide_title-->
::: {.slide title="Applying the Horizontal Finite-Difference Kernel"}
<!--tr:3ce31cbe-->

<!--en:b37a1029 kind=kicker-->
[Detecting Vertical Edges with K = [1, -1]]{.kicker}
<!--tr:b37a1029-->

<!--en:9be85019 kind=slide_text-->
Cross-correlating `X` with the $1 \times 2$ kernel $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ computes $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ at every horizontal pair:
<!--tr:9be85019-->

<!--en:f649cfae kind=slide_text-->
- **Signed edge map:** Column 1 fires `+1` (white-to-black transition), column 5 fires `-1` (black-to-white transition), and all constant interior columns evaluate to `0`.
<!--tr:f649cfae-->

<!--en:e6d155be kind=slide_title-->
::: {.slide title="Orientation Selectivity of Edge Kernels"}
<!--tr:e6d155be-->

<!--en:b525cde4 kind=kicker-->
[Why One Kernel Cannot Detect All Orientations]{.kicker}
<!--tr:b525cde4-->

<!--en:38159de6 kind=slide_text-->
- **Blind to horizontal edges:** Transposing `X` (`X.T`) makes horizontal neighbors identical, so $[1, -1]$ outputs all zeros.
<!--tr:38159de6-->

<!--en:f0f30f76 kind=slide_text-->
- **Filter banks (§{sec_channels}):** Vision layers learn parallel channels for horizontal, vertical, and diagonal orientations.
<!--tr:f0f30f76-->

<!--en:b755e31b kind=slide_title-->
::: {.slide title="Learning an Edge Detector via Gradient Descent"}
<!--tr:b755e31b-->

<!--en:86bbeea7 kind=kicker-->
[Supervising a Built-In Convolutional Layer on (X, Y)]{.kicker}
<!--tr:86bbeea7-->

<!--en:a569780c kind=slide_text-->
- **4D framework tensors:** Reshape `X` and `Y` to 4D tensors (one example, one channel) and minimize $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ at $\eta = 0.03$.
<!--tr:a569780c-->

<!--en:2489ce6e kind=slide_title-->
::: {.slide title="Inspecting the Learned Kernel Weights"}
<!--tr:2489ce6e-->

<!--en:7e4ef44b kind=kicker-->
[Why Initial Loss Is 12 and Converges to [1, -1]]{.kicker}
<!--tr:7e4ef44b-->

<!--en:9cd1c645 kind=slide_text-->
- **Why loss starts at $12.0$:** Near-zero weights predict $\hat{Y} \approx 0$, missing the twelve $\pm 1$ edge pixels in `Y` ($12 \times 1^2 = 12$).
<!--tr:9cd1c645-->

<!--en:456670c5 kind=slide_text-->
- **Convergence in 9 steps:** SGD recovers $[0.99, -0.99]$ automatically from data.
<!--tr:456670c5-->
