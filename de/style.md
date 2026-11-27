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
| Names of models, architectures, optimizers, libraries, datasets, metrics; acronyms; code | Use the English name unless the glossary specifies a German form. | Transformer, ResNet, BERT, Adam, AdamW, PyTorch, ImageNet, BLEU, ReLU, SGD, MLP, LSTM, GPU, Q-Learning, Long Short-Term Memory |
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
Reinforcement Learning · der Kanal · die Aktivierung · die Lernrate · die Epoche ·
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
Gradientenabstieg, neuronales Netz, konvolutionäres Netz) and are inflected:
*mit dem stochastischen Gradientenabstieg*, *eines
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
| Reinforcement learning | Reinforcement Learning, Deep Reinforcement Learning, Offline Reinforcement Learning. Keep the source abbreviations RL, Deep RL, and Offline RL. Phrase around inflection when necessary to retain the full term unchanged. | |
| Proximal policy optimization | Proximal Policy Optimierung. Keep PPO where the source uses the abbreviation. | |
| German kept | Einbettung; versteckte Schicht; Strategie (policy); Ertrag (return); Abfrage, Schlüssel, Wert; Form (shape); Gewichtszerfall; Lernratenplan; Präzision, Genauigkeit, Trefferquote; Sequenz (ML) and Folge (mathematics); Kodierung; Differentialgleichung. | Embedding; verborgen, verdeckt; Weight Decay; Lernraten-Schedule |
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
21. **Reinforcement Learning:** Retain the English term, including in Deep
    Reinforcement Learning and Offline Reinforcement Learning (§8.2).
22. **Proximal policy optimization:** Use Proximal Policy Optimierung (§8.2).
