# Style guide: French (fr)

Draft of 2026-11-26 (Fable 5.2) for native-speaker review. Translators read this with every
unit (TRANSLATING.md §7–§8). Each rule has one reason and one example
(English → French, or ✗ wrong / ✓ right). Sentences marked *fable*, *opus* or
*sonnet* are quoted from the translation test `evals/fr-conv-layer/`. Points
that still need the author's ruling are marked "(to confirm)" and collected in
§11.

Order of authority: the decisions log, then this guide, then the glossary. The
glossary says *what* a term is; this guide says *how to choose, inflect and
punctuate*. Only 106 of the 1,191 French glossary entries are verified; the
rest are `proposed`, so §3 applies to them in full (TRANSLATING.md §7).

Variety: the French of France (*apprentissage automatique*, not *apprentissage
machine*; *courriel* and other Quebec forms only where the glossary has them).
Spelling: the traditional one, as in the glossary (entraînement, chaîne, coût,
connaître, not the 1990 forms entrainement, chaine, cout). Capital letters keep
their accents (À, É, Équation, État); œ is typed as one letter (cœur, nœud).

## 1. Register

**Decision: authorial *nous*, the reader as *vous*, never *tu*.** Reason: it is
the register of French university textbooks, and *nous* carries the book's "we
work through this together" tone (TRANSLATING.md §8); *tu* is the tone of
tutorials and blogs. All three test translators chose it.

| English | French | Example |
|---|---|---|
| we + action in the text | *nous* + present tense | "We minimize the loss with SGD." → Nous minimisons la perte avec la SGD. |
| "let us", "let's", "now we see" | first-person plural imperative: Voyons, Considérons, Supposons | "Now we see what that formula computes." → Voyons maintenant ce que cette formule calcule. (*fable*) ✗ Laissez-nous … / Laissons-nous … |
| what an earlier section did | passé composé | "In …, we derived the layer." → …, nous avons établi la couche. |
| "we will see", "we will return to" | futur simple, only for a later place in the book | "We return to this in …" → Nous y reviendrons à la … |
| "you" = the reader (advice, a question, exercises) | *vous*, *votre* | "Suppose you want to estimate the price of a house." → Supposons que vous vouliez estimer le prix d'une maison. |
| "you can", "one can" = anyone | *on*, or *nous* | "You can verify this numerically." → On le vérifie numériquement. |
| "we recommend", "we believe" | *nous* | "We recommend AdamW." → Nous recommandons AdamW. |
| "we" = people, the field | *nous* + noun, *on*, *les chercheurs* | "We humans attend selectively." → Nous, humains, portons une attention sélective. |

- ***Nous* is a real plural here** (four authors and the reader): participles
  and adjectives agree in the masculine plural. ✓ nous sommes partis de …,
  nous sommes convaincus ✗ nous sommes parti.
- ***On* is only the generic "one"** (on obtient, on dit que). Do not use *on*
  for the authors, and do not switch between *nous* and *on* for the same
  subject inside one sentence.
- **Tenses:** present for what the text does, passé composé for what was done,
  futur for what comes later. Never the passé simple; no *nous allons* +
  infinitive in every sentence.

**Sentence shape** (each point is checkable):

- **Active with *nous* or *on*.** Passive only where the agent does not matter
  (Le modèle a été préentraîné sur ImageNet). Prefer the pronominal form to
  *être* + participle: ✗ Le gradient est calculé par la règle de la chaîne.
  ✓ Le gradient se calcule par la règle de la chaîne. / Nous calculons le
  gradient par la règle de la chaîne.
- **A participle or gerund must have the subject of its main clause.**
  ✗ En utilisant la règle de la chaîne, le gradient vaut … ✓ La règle de la
  chaîne donne … / En appliquant la règle de la chaîne, nous obtenons …
- **At most three nouns chained by *de*.** ✗ la forme de la sortie de la
  première couche du réseau ✓ la forme que produit la première couche du réseau.
- **One main clause and at most two subordinate clauses** per sentence. Split
  English sentences at "which", "where", "so that"; a semicolon may join the
  halves (§2).
- **Pronouns agree with the French noun**, not with English "it": la couche …
  elle ; le noyau … il. Repeat the noun when two candidates have the same
  gender.
- **No English word inflected as French and no English order of words**:
  ✗ fine-tuner, ✗ les patches, ✗ Adam optimiseur, ✗ GPU mémoire (§3.4, §3.5).
- **No literal "simply", "just", "actually"**: drop the word or write
  *seulement*, *directement*, *en réalité* (§8.2).

## 2. Punctuation and typography

| Item | Rule | Example |
|---|---|---|
| Space before `:` `;` `?` `!` | Always one space before, one after. Type an ordinary space; the build must make it no-break (§10). No space before `,` `.` `)` and none after `(`. | ✓ deux principes géométriques : l'équivariance par translation et la localité ✗ géométriques: l'équivariance |
| Quotation marks | « … » with a space inside each mark; inside them “…”. Never "…" or ''…''. Book titles in italics as in the source. | the "attention" weights → les poids d'« attention » ; appelons « convolution » la corrélation croisée (*fable*) |
| Quotes in attributes | A slide `title="…"` is delimited by ASCII `"`; inside it use « … » only. | `title="Que veut dire « profond » ?"` |
| Apostrophe | Type the straight `'` (the build prints ’). Reason: the glossary and its check use `'`; a typed ’ makes every term with an apostrophe (taux d'apprentissage) fail the check. | ✓ `l'entrée` ✗ `l’entrée` |
| Dash | The English em dash `—` becomes a comma, a colon, a semicolon or parentheses; where an aside needs dashes, an en dash with spaces ( – ), one pair per sentence at most. Never `—`, never a hyphen with spaces. | "…during forward propagation—borrowing the neurophysiology term from …" → … pendant la passe avant ; le terme, issu de la neurophysiologie, est repris de la … (*fable*) ✗ … pendant la passe avant — terme emprunté … (*sonnet*) |
| Hyphen between paired nouns | A hyphen, also where the English has an en dash. | "input–output pair" → paire entrée-sortie ; encodeur-décodeur ; compromis biais-variance |
| Comma | French rules: no comma before *et*/*ou* in a list; no comma before *que*; commas around an inserted clause. | "Adam, RMSProp, and SGD" → Adam, RMSProp et SGD |
| Colon | Lower case after a colon, also when a full sentence follows and after a bold lead-in. At most one colon per sentence. | **Hypothèse centrale :** les erreurs sont indépendantes. |
| Semicolon | Allowed, and often the right rendering of an English "so", "and" or dash between two main clauses. | … laisse $n - k + 1$ positions valides ; la forme de la sortie est donc (*fable*) |
| Abbreviations | In body text write them out: par exemple, c'est-à-dire, environ, et ainsi de suite. Only in parentheses, captions, slides and tables: p. ex., c.-à-d., env., cf., etc. (one full stop, never « etc… »). "i.e." → c'est-à-dire; "e.g." → par exemple; "vs." → et, contre, face à; "et al." stays. | "Learning vs. hand-crafting" → Apprendre plutôt que concevoir à la main (*fable*) |
| Invisible characters | Do not type no-break or narrow spaces or soft hyphens (TRANSLATING.md §8). | ✓ `45 %`, `10 000`, `24 GiB` with ordinary spaces |
| Emphasis | Keep `*…*` and `**…**` on the corresponding word(s), around the whole inflected word; add none. The English gloss stays outside the markers. In a bold lead-in the space and colon stay inside the bold, as the colon does in English. | "*regression* predicts …" → la *régression* prédit … ; `*remplissage* (angl. padding)` ; `**Convergence en 9 pas :** la SGD retrouve …` |
| Capitals | French has no title case: a capital on the first word and on proper nouns only, in headings, slide titles, captions and table headers. Names of theorems, laws, languages, days and months are lower case. | "Convolutions for Images" → Convolutions pour les images ; le théorème de Bayes ; l'anglais |

### 2.1 References

The French build prints the word before the number, capitalised: **Chapitre 4**
(masculine), **Section 6.1** (feminine), **Figure 6.2.1** (feminine),
**Tableau 6.2.1** (masculine; needs the setting in §10, Quarto's own French
file prints *Table*), **Équation 6.2.1** (feminine, begins with a vowel). The
kind, and so the gender, can be read from the label:

| Label | Printed | Write | Example |
|---|---|---|---|
| `sec_…`, `subsec_…` | Section 6.1 | la, à la, de la | "In :numref:`sec_why-conv`, we derived …" → À la :numref:`sec_why-conv`, nous avons établi … (*fable*) |
| `chap_…` | Chapitre 4 | le, au, du | "our MLP weights in :numref:`chap_perceptrons`" → les poids de nos MLP au :numref:`chap_perceptrons` (*fable*) |
| `fig_…`, `img_…` (and `field_visual`, `asha`, `distributed_scheduling`, `synchronous_sh`) | Figure 6.2.1 | la, dans la, de la | "In :numref:`fig_correlation`, we slide …" → Dans la :numref:`fig_correlation`, nous faisons glisser … (*fable*) |
| `tab_…`, `table_…` | Tableau 6.2.1 | le, dans le, du | "as :numref:`tab_x` shows" → comme le montre le :numref:`tab_x` |
| `:eqref:` (any label) | Équation 6.2.1 | l', de l', dans l' (elided, no space) | "We implement :eqref:`eq_corr2d` directly" → Nous implémentons l':eqref:`eq_corr2d` directement |

- **Always write the article, never the noun.** Reason: without the article
  the printed sentence is not French. ✗ Dans :numref:`sec_why-conv`, nous avons
  dérivé … (*opus*, *sonnet*; prints "Dans Section 6.1") ✗ à la section
  :numref:`sec_x` (prints "à la section Section 6.1").
- **Inside parentheses, no article:** (:numref:`sec_channels`), (voir
  :numref:`fig_x`).
- **Two references:** repeat the article and the preposition, because each
  prints its own singular word: à la :numref:`sec_a` et à la :numref:`sec_b`
  ✗ aux :numref:`sec_a` et :numref:`sec_b`.
- **A reference never opens a sentence bare:** La :numref:`fig_x` montre …
- The elision before `:eqref:` is safe: tested with the repository's pandoc on
  2026-11-26, `l'` and `d'` set directly against a reference are recognised.
- A label that points at an unnumbered page (the preface, a part) prints that
  page's title, not a number: phrase the sentence so that a title fits and
  name the block in NOTES.
- **Slides:** `§{sec_x}` prints "§9.3" and `§{num:sec_x}` prints "9.3". Write
  (§{sec_x}) or voir §{sec_x}, without an article; "Chapter §{num:chap_x}" →
  chapitre §{num:chap_x}.
- **Citations.** `:cite:` stands at the end of the clause it supports, before
  the comma or full stop, as in English. `:citet:` prints the authors' names:
  a plural verb for several authors (:citet:`Vaswani.2017` ont introduit …),
  and never *de*, *que* or *le* directly before it, because the name may begin
  with a vowel: write selon :citet:`…`, par :citet:`…`, or make it the subject.
  ✓ le modèle proposé par :citet:`…` ✗ le modèle de :citet:`…`

### 2.2 Math and code in a sentence

- **Give a symbol or an identifier its noun**, so that gender and number show:
  la matrice $\mathbf{X}$, le noyau `K`, la fonction `corr2d`, le taux
  d'apprentissage $\eta$.
- **No elision before math or code:** de $x$, que `X`, le $i$. Reason: the
  reader cannot know how the symbol is read aloud. (References are different,
  §2.1: their printed word is known.)
- **Size after the noun, name last:** "a $2 \times 2$ kernel" → un noyau
  $2 \times 2$ ; "the $3 \times 3$ input tensor `X`" → le tenseur d'entrée
  $3 \times 3$ `X`.
- **Number or symbol + noun:** a preposition, not a hyphenated compound:
  "$d$-dimensional" → de dimension $d$ ; "$k$-class" → à $k$ classes ;
  "a width-$k$ window" → une fenêtre de largeur $k$. Fixed terms keep their
  hyphen: $n$-gramme, $k$-ième.
- **Protected math is copied as it is, even when it contains English:** "the
  $i^{\textrm{th}}$ layer" → la $i^{\textrm{th}}$ couche. Do not change it and
  do not report it in NOTES.
- **Token order.** The check compares the order of protected tokens class by
  class (math with math, code with code). French seldom needs a reorder; where
  an adjective phrase would swap two math spans, restructure first: "an
  $n_\textrm{h} \times n_\textrm{w}$ input matrix $\mathbf{X}$" → une matrice
  d'entrée $n_\textrm{h} \times n_\textrm{w}$ notée $\mathbf{X}$ (*fable*). If
  a reorder is unavoidable, name the block in NOTES.
- **Before a display equation** the sentence may end with a colon (with its
  space). After it, "where $n$ is …" → où $n$ désigne …

## 3. Terminology: French term or English loan

### 3.1 How to choose

**Use the French term wherever mathematics, statistics or computer-science
teaching has one, and wherever the glossary gives one. Use an English loan only
where the glossary keeps it or French practice has nothing else. Never coin a
French word for an English term, and never leave an English word where a French
term is established.** For a term without an entry, apply the rule, check
French Wikipedia or a French textbook, and record the choice in DECISIONS.

| Case | Rule | Examples |
|---|---|---|
| Established French term | Use it. | couche (✗ layer), poids, noyau (✗ kernel), perte (✗ loss), lot, mini-lot (✗ batch), taux d'apprentissage, jeu de données (✗ dataset), caractéristique (✗ feature), plongement (✗ embedding), jeton (✗ token), réglage fin (✗ fine-tuning), surapprentissage (✗ overfitting), remplissage (✗ padding), pas (✗ stride), rétropropagation, descente de gradient, décroissance des poids, arrêt anticipé, apprentissage profond (✗ deep learning), apprentissage automatique (✗ machine learning) |
| Loan kept by the glossary | Use it as a French noun, lower case, with the gender of §3.3. | dropout, pooling, max-pooling, prompt, softmax, softplus, momentum, logit, broadcasting, bootstrapping, Q-learning, score matching, flow matching, skip-gram, encodage one-hot, perte hinge, régression ridge, all-reduce |
| Loan without an entry, standard in French practice | Allowed for these; any other goes to DECISIONS. | framework, notebook, patch, pipeline, cache, benchmark |
| Names of models, architectures, optimizers, libraries, datasets, metrics; acronyms; code | Leave in English; never translate a name in part. | ResNet, BERT, Adam, AdamW, PyTorch, ImageNet, BLEU, ReLU, GELU, SGD, MLP, CNN, LSTM, GPU, Tensor Core, Network in Network |
| The architecture as a common noun | *transformeur*, lower case, as the glossary and French Wikipedia have it; the paper's and the library's names stay. | "a transformer block" → un bloc transformeur ; la bibliothèque Transformers |

Compound words follow the glossary's spelling: préentraînement, autoencodeur,
autorégressif, hyperparamètre, multicouche, surapprentissage,
sous-apprentissage, mini-lot, auto-attention, multi-têtes, pseudo-inverse,
log-vraisemblance. The adjective is *non linéaire* (no hyphen), the noun
*non-linéarité*. The adjective for convolution is *convolutif*, *convolutive*
(✗ convolutionnel).

### 3.2 Frequent head words

French terms, by frequency in the English book, with gender and the forms that
are easy to get wrong:

le modèle · l'entraînement (m.) · le gradient · la moyenne · la couche · la
matrice · l'état (m.) · la probabilité · la variance · la valeur · le poids
(invariable: les poids) · la politique (policy) · le vecteur · l'entropie (f.)
· la norme · le noyau (les noyaux) · l'exactitude (f., accuracy) · le critique
(RL critic, masculine) · le biais (invariable) · la séquence · la clé · la
caractéristique · le rang · la prédiction · le perceptron multicouche (MLP) ·
la distribution a posteriori · le résidu · la forme (shape) · l'action (f.) ·
le jeu de données (les jeux de données) · le tenseur · la récompense ·
l'optimisation (f.) · le retour (RL return) · la précision · la descente de
gradient stochastique (SGD) · la dérivée · le rappel · l'apprentissage par
renforcement (m.) · le canal (les canaux) · l'activation (f.) · le taux
d'apprentissage (invariable) · l'époque (f.) · la requête · la matrice
jacobienne · la convolution · l'entropie croisée · la distribution a priori ·
le plongement · l'espérance (f.) · le masque · la taille du pas · la taille du
lot · la généralisation · l'environnement (m.) · l'agent (m.) · le vocabulaire
· la passe avant, la passe arrière · le modèle de langage · le
préentraînement · la couche cachée · le réseau de neurones (les réseaux de
neurones) · le réseau de neurones convolutif (CNN) · le paramètre · la
fonction de perte · la fonction objectif · le jeton · le mini-lot · le réglage
fin.

- *a priori* and *a posteriori*: no accent, no italics, invariable.
- Adjectives in a term agree: des couches convolutives, une matrice définie
  positive, des fonctions lipschitziennes, des variables indépendantes et
  identiquement distribuées, des échantillons indépendants et identiquement
  distribués. The glossary gives one form; inflect it.
- Named laws are *loi* (loi normale, loi de Poisson, loi normale centrée
  réduite for "standard normal"); the generic word is *distribution*
  (distribution a priori, distribution des données), as in the glossary.

### 3.3 Loan nouns and acronyms: gender and plural

Gender: the entry in Larousse or Le Robert if there is one; otherwise the
gender of the French noun the word stands for; otherwise masculine. Plural:
loan nouns take a French *-s* (never *-es*); acronyms are invariable.

| Word | Gender, plural | Note |
|---|---|---|
| dropout, pooling, max-pooling, broadcasting, bootstrapping, momentum, Q-learning, score matching, flow matching | masculine; no plural | le dropout, un taux de dropout de 0.5 |
| prompt, logit, framework, notebook, pipeline, cache, benchmark | masculine; les prompts, les logits, les frameworks | |
| patch | masculine; les patchs | ✗ les patches ; ✗ bloc for an image patch (*sonnet*) |
| softmax | masculine; *la fonction softmax* where a noun helps | la température du softmax |
| GPU, CPU | masculine (processeur); les GPU | ✗ la GPU, ✗ les GPUs |
| SGD | feminine (descente); no plural | la SGD retrouve … (*fable*) |
| CNN, RNN, MLP, GAN, LSTM, GRU, VAE | masculine (réseau, perceptron, autoencodeur); invariable | un CNN profond, les CNN (*fable*) ; LSTM and GRU (to confirm) |
| ReLU | feminine (unité, fonction) | la ReLU ; better: la fonction ReLU, l'activation ReLU |
| API | feminine (interface); les API | |
| KL, BLEU, ROC, FID | no gender of their own: use the head noun | la divergence KL, le score BLEU, la courbe ROC |

- **No elision before an acronym or a symbol spelled letter by letter:** le
  MLP, de MLP, le LSTM, que SGD. **Elision before a name that is read as a
  word and begins with a vowel:** d'Adam, d'ImageNet, l'API, qu'AdamW.
- **Model names used as names take no article:** BERT utilise …, les poids de
  BERT (✗ les poids BERT, ✗ BERT's). Used as countable nouns they are
  masculine: un ResNet-18, un transformeur.
- A loan not listed here: choose by the rule above and record it in DECISIONS
  with its gender.

### 3.4 Compounds

English stacks nouns; French puts the head noun first and joins with *de*, *à*,
*par*, or by apposition. Never keep the English order and never join a stack
with hyphens.

| Case | Rule | Examples |
|---|---|---|
| noun + noun | head + *de* + complement, no article unless the sense needs one | couche de pooling, taux de dropout, masque de remplissage, taille du lot, carte de caractéristiques, noyau de flou |
| name or acronym + noun | head noun, then the name in apposition | l'optimiseur Adam (✗ Adam optimiseur), la fonction ReLU, le cache KV, la mémoire GPU, la divergence KL, une image Fashion-MNIST, un bloc transformeur |
| method or mode | *par*, *à*, *en* | normalisation par lots, apprentissage par renforcement, attention multi-têtes, convolution à pas supérieur à un, recherche en faisceau |
| two coordinated nouns | hyphen | encodeur-décodeur, acteur-critique, état-action, entrée-sortie |
| long English stack | a relative clause or two phrases | "hand-crafted edge detector" → détecteur de contours conçu à la main ; "end-to-end training" → entraînement de bout en bout |

### 3.5 Verbs

French verbs for actions. Never an English stem with a French ending
(✗ fine-tuner, ✗ pooler, ✗ padder, ✗ shuffler, ✗ overfitter, ✗ plotter).

| English | French | Not |
|---|---|---|
| train, pretrain | entraîner, préentraîner (un modèle préentraîné) | former, pré-entrainer |
| fine-tune | affiner (un modèle affiné); noun *le réglage fin* | fine-tuner, régler finement |
| sample (from a distribution) | tirer, échantillonner: nous tirons $x$ selon $p$ | sampler |
| overfit, underfit | le modèle surapprend ; le modèle est en surapprentissage, en sous-apprentissage | overfitter, surajuster |
| update | mettre à jour | updater, actualiser |
| pad, pool, clip, mask | compléter par des zéros, appliquer un remplissage ; appliquer un pooling ; écrêter ; masquer | padder, pooler, clipper |
| tokenize | segmenter en jetons | tokeniser (to confirm, §11) |
| cache, shuffle, fit, flatten, reshape, stack | mettre en cache ; mélanger ; ajuster ; aplatir ; redimensionner ; empiler | cacher, remodeler |
| derive (a formula, a result) | établir, obtenir, déduire ; "derive … in terms of" → exprimer … en fonction de | dériver (= to differentiate) |
| differentiate | dériver (par rapport à $x$) | différencier |
| compute, run, return (a value) | calculer ; exécuter ; renvoyer | computer, rouler, retourner |
| implement, initialize, normalize, regularize, optimize, encode, decode, parallelize | implémenter, initialiser, normaliser, régulariser, optimiser, encoder, décoder, paralléliser | |
| quantize | quantifier (la quantification) | quantiser |
| plot, debug, deploy, download, log | tracer ; déboguer ; déployer ; télécharger ; journaliser, enregistrer | plotter, débugger, logger |
| support, assume, process, perform | prendre en charge ; supposer ; traiter ; effectuer | supporter, assumer, processer, performer |

### 3.6 The English term in parentheses

Add the English once per page (one numbered section of the book), at the first
occurrence of a key term in body text, as `(angl. …)`, lower case as in the
glossary, in roman, outside the emphasis markers. Check the English page in
CONTEXT: if an earlier unit of the page introduced the term, do not repeat it.

- Gloss French terms that practitioners know under an English name that looks
  different: *remplissage* (angl. padding), *pas* (angl. stride), *plongement*
  (angl. embedding), *jeton* (angl. token), *réglage fin* (angl. fine-tuning),
  *surapprentissage* (angl. overfitting), *mini-lot* (angl. minibatch),
  *décroissance des poids* (angl. weight decay), *arrêt anticipé* (angl. early
  stopping), *carte de caractéristiques* (angl. feature map), *champ
  récepteur* (angl. receptive field), *politique* (angl. policy), *retour*
  (angl. return), *recherche en faisceau* (angl. beam search).
- Do not gloss loans (dropout, pooling), transparent cognates (régularisation,
  gradient, tenseur, convolution, attention, corrélation croisée, entropie
  croisée), or school vocabulary (dérivée, probabilité, moyenne).
- With an acronym: *descente de gradient stochastique* (angl. stochastic
  gradient descent, SGD), *réseau de neurones convolutif* (angl. convolutional
  neural network, CNN); afterwards write what the English writes (SGD, CNN).
- Never in headings, captions, slides, exercises or table cells.
- If the English sentence already has a parenthesis at that place, keep one of
  the two as an apposition; do not stack parentheses.

"we surround the input with a border of zeros (*padding*)" → nous entourons
l'entrée d'une bordure de zéros, le *remplissage* (angl. padding).
✗ (le *remplissage*, padding en anglais) ✗ (*remplissage*) with no English
(*opus*: the reader loses the word used in every paper and API).

Reason: readers go on to English papers, documentation and code; one fixed
form, once per page, keeps the text clean and can be checked.

## 4. Numbers and units

**Decision: the decimal point everywhere, a space as thousands separator, a
space before `%`.** Numerals are copied from the English (0.5, 0.01), in prose
as in math, code, code output and figures. Reason: inline and display math are
protected and keep the point, and outputs and figure axes show it; a decimal
comma in prose would put two conventions into one sentence (✗ avec $\eta =
0.03$, la perte passe sous 0,01) and would need a rule for every identifier
(GPT-3.5, Python 3.12). The check also compares numerals as written: `0,5` for
`0.5` raises "numbers not found". All three test translators kept the point.
This departs from French typographic usage in prose, as the German edition
does by the author's ruling (to confirm for French, §11).

| Item | Rule | Example |
|---|---|---|
| Decimal | As in the source, with a point. Never a comma as decimal sign. | "a dropout rate of 0.5" → un taux de dropout de 0.5 |
| Thousands | An ordinary space from five digits on; none in four-digit numbers and years. Never `10,000` or `10.000`. | "10,000 images" → 10 000 images ; 2048 ; 60 000 |
| Number words | million → million(s), **billion → milliard(s)**, **trillion → billion(s)** (or mille milliards); followed by *de*; singular below 2. | "3 billion parameters" → 3 milliards de paramètres ✗ 3 billions ; "1.5 million tokens" → 1.5 million de jetons |
| Digits or words | As the English: a number spelled out in English is spelled out in French; digits with units, symbols and comparisons. | "two geometric principles" → deux principes géométriques ; 10 pas de gradient ; les 42 éléments de sortie |
| Percent | A space before `%`. | "45% fewer parameters" → 45 % de paramètres en moins (*fable*, *sonnet*) ✗ 45% (*opus*) |
| Units | Symbols as in the source, with a space; time words written out become secondes, minutes, heures, jours. | 24 GiB, 5 ms ; "3 hours" → 3 heures |
| Ranges | In running text *de … à …*; an en dash without spaces in parentheses, tables and slides; the unit once. | "40–55" → de 40 à 55 ; (40–55) ; 45–50 % |
| Multipliers | In prose *N fois plus* + adjective, the number as the English has it; keep `×` in slides, tables, captions and math. | "about 2× faster" → environ 2 fois plus rapide ; slide: 2× plus rapide |
| Approximation | environ, près de, à peu près; *env.* only in tables and slides; "an order of magnitude" → un ordre de grandeur. | "roughly 90%" → environ 90 % |
| Ordinals | Written out up to *dixième*; otherwise 1er, 1re, 2e, 21e (✗ 2ème, ✗ 1ère). | "the third epoch" → la troisième époque |
| Decades, years | "the 2010s" → les années 2010; "in 2017" → en 2017. | ResNet est paru en 2015. |
| Money | Amount, space, currency name; never the `$` sign in prose (it opens math). | "\$2.50" → 2.50 dollars |

Quantities, signs and orders of magnitude stay exact; do not round or convert
units or currencies. In French mathematics *positif* means ≥ 0: "positive" →
strictement positif, "nonnegative" → positif (ou nul), "increasing" →
strictement croissant where the English means strictly, "greater than" →
strictement supérieur à.

What the check does: with `check: {thousands: space}` it removes the English
grouping comma and the French grouping space before comparing, so `10,000` →
`10 000` and `1,024` → `1024` pass; without that setting `10 000` raises a
warning. `45 %` for `45%` passes. `N×` is not compared, so *2 fois* and *deux
fois* both pass; keep the digit anyway.

## 5. Gender-inclusive writing

**Decision: none.** No point médian, no doubled forms with a slash, hyphen or
parentheses, no *iel* (✗ les chercheur·euse·s, ✗ les utilisateur/trice/s,
✗ lecteur(trice)s). Reason: the Académie française (declaration of 26 October
2017) and the Prime Minister's circular of 21 November 2017 keep these signs
out of standard written French; they also collide with the book's markup (`·`
separates items on slides, `*` is emphasis) and with screen readers.

- People in general: the plural, an epicene noun, or a clause. "the
  practitioner … she" → les spécialistes, quiconque entraîne un modèle, on.
- "the reader", "the user": *vous* (§1); otherwise le lecteur, l'utilisateur.
- Technical roles are not people and keep their grammatical gender: l'agent,
  l'acteur, le critique, le générateur, le discriminateur, l'enseignant
  (forçage par enseignant), l'optimiseur.
- Named people keep the pronoun of the source.

## 6. Names

- **Models, libraries, datasets, companies, hardware, benchmarks:** exactly as
  in English: Fashion-MNIST, RTX 4090, OpenAI, GPT-2, GEMM, `im2col`.
- **People:** as in the English source, without title (Markov, Kolmogorov,
  Chebyshev, Lyapunov), with the diacritics the glossary shows (Fréchet, Itô).
  Reason: citations, searches and code use these forms. This departs from the
  French transliteration *Tchebychev* (to confirm, §11).
- **Terms named after people:** noun + *de* + name: théorème de Bayes, méthode
  de Newton, loi de Poisson, inégalité de Jensen, équation de Bellman,
  développement de Taylor, multiplicateur de Lagrange, distance de Wasserstein,
  matrice de Gram, norme de Frobenius. Elide before a vowel (méthode d'Euler,
  calcul d'Itô); no elision before H (produit de Hadamard, perte de Huber,
  initialisation de He). With two names, a hyphen: méthode d'Euler-Maruyama.
- **Adjectives from names:** lower case, and only those the glossary or French
  mathematics has: gaussien, bayésien, markovien, lipschitzien, euclidien,
  brownien, la hessienne, la matrice jacobienne, le lagrangien.
- **The book:** *Dive into Deep Learning* stays in English; "this book" → ce
  livre. Titles of external books, courses and assignments stay in English as
  link text (*Neural Networks and Deep Learning*, Assignment 2).
- **Languages and places:** French, lower case for languages (l'anglais, les
  États-Unis); programming languages stay (Python).
- **Structure words:** chapter chapitre, section section, subsection
  sous-section, appendix annexe, Summary → Résumé, Exercises → Exercices,
  Discussions → Discussions, Concept Check → Vérification des acquis; Theorem →
  Théorème, Proposition → Proposition, Lemma → Lemme, Corollary → Corollaire,
  Definition → Définition, Proof → Démonstration (✗ Preuve), Remark → Remarque,
  Example → Exemple; "panel (c)" → panneau (c).

## 7. Headings, captions, slides, exercises, notes

| Kind | Rule | Example |
|---|---|---|
| `heading` | Noun phrase, sentence case, no full stop, no English gloss. English "-ing + object" becomes a noun + *de*. Keep the article where the English has one, add none. A question stays a question. | "Learning a Kernel" → Apprentissage d'un noyau (*fable*) ; "The Operation" → L'opération ; "Convolutional Layers" → Couches convolutives ; "Why Depth Helps" → Pourquoi la profondeur aide |
| `caption` | Body register without *nous*/*vous*; a full stop exactly when the English caption has one. Keep `(a)`, `(b)`. | "Two-dimensional cross-correlation. The shaded portions are …" → Corrélation croisée bidimensionnelle. Les parties grisées sont … |
| `slide_title`, `kicker` | As short as the English, sentence case, no full stop. An English "-ing" title becomes an infinitive (the slide does something); any other title a noun phrase. | `title="Implementing 2D Cross-Correlation in Code"` → `title="Implémenter la corrélation croisée 2D en code"` ; `[Sliding-Window Inner Products]{.kicker}` → `[Produits scalaires sur fenêtre glissante]{.kicker}` |
| `slide_text` | Short statements or noun phrases; no *nous*, no *vous*; instructions as infinitives; one line where the English is one line; final punctuation as in the English bullet. Bold lead-ins keep their bold, take « : » inside the bold, and are followed by lower case. Abbreviations the English bullet uses stay (SGD, CNN, 2D). | "**Convergence in 9 steps:** SGD recovers $[0.99, -0.99]$ automatically from data." → **Convergence en 9 pas :** la SGD retrouve $[0.99, -0.99]$ automatiquement à partir des données. (*fable*) ✗ … la descente de gradient stochastique retrouve … (*opus*: longer than the English line) |
| `exercise` | *Vous* imperative; direct questions. Bold titles keep their bold, with the full stop inside. `[code]` and `●` are copied. Reason: it is the imperative of the body register; the infinitive (Montrer que …) is the style of exam sheets. | "Design some kernels manually." → Concevez quelques noyaux à la main. ; "How would you design a blur kernel?" → Comment concevriez-vous un noyau de flou ? ; "derive an edge-detection kernel" → construisez un noyau de détection de contours (*fable*) ✗ dérivez un noyau (*opus*) |
| `tab_note` | Body register. Describe this framework only; never compare frameworks. API names stay in code. | "PyTorch's `nn.LazyLinear(1)` requires only the output dimension." → Dans PyTorch, `nn.LazyLinear(1)` ne demande que la dimension de sortie. |
| `note` | Register of its surroundings. Labels: **Note:** → **Remarque :**, **Warning:** → **Attention :**, **Tip:** → **Conseil :**, **Answers:** → **Réponses :** | |
| `table_row` | Noun phrases, sentence case; numbers and units as in §4. | "Training time" → Temps d'entraînement |

Attribution lines: "*Adapted from …*" → *Adapté de …*. The cover kicker `[Dive
into Deep Learning · §2.1]{.kicker}` is rewritten by the build; copy it
unchanged. Where an English kicker or title is elliptical to the point of
being wrong, translate what it evidently means and name it in NOTES: "[Why
Initial Loss Is 12 and Converges to [1, -1]]{.kicker}" → [Pourquoi la perte
initiale vaut 12 et le noyau converge vers [1, -1]]{.kicker} (*fable*, *opus*).

## 8. Known pitfalls

### 8.1 Words with more than one sense

| English | Sense → French | Not |
|---|---|---|
| step, stride | convolution stride → le pas; optimizer step size → la taille du pas; time step → le pas de temps; one update → un pas de gradient, une itération; a stage → une étape. Write *pas* alone only for the stride. | pas for all of them |
| kernel | convolution → le noyau (taille du noyau); kernel methods → fonction noyau, matrice de noyau; of a linear map → le noyau de $\mathbf{A}$; GPU or OS code → le noyau. One word in French: add the complement where two senses meet on a page. | kernel |
| bias | parameter $b$ → le biais; of an estimator → le biais (sans biais for "unbiased"); social → préjugé, partialité | |
| return | RL → le retour; of a function → la valeur de retour, renvoyer | retourner une valeur |
| sample | statistics and generated outputs → l'échantillon; a data point → l'exemple; to sample → tirer, échantillonner | |
| feature | → la caractéristique (vecteur de caractéristiques, carte de caractéristiques); of a library or product → la fonctionnalité | feature, trait |
| entry, element | of a matrix, tensor or array → l'élément; "elementwise" → élément par élément (*fable*). *Coefficient* is reserved for model coefficients. | coefficient par coefficient (*opus*) |
| sparse, sparsity | matrix, array → creux, creuse (matrice creuse, structure creuse for "sparsity pattern"); model, solution, attention → parcimonieux, la parcimonie | motif de parcimonie (*sonnet*) |
| edge | in an image → le contour (détection de contours); of an image → le bord; of a graph → l'arête | bord for an image edge |
| patch | → le patch (matrice de patchs, patch d'image) | bloc, pièce |
| window | → la fenêtre (fenêtre glissante, fenêtre de contexte) | |
| policy, state, action, reward, value | la politique, l'état, l'action, la récompense, la valeur; "state of the art" → l'état de l'art | police, statut |
| loss, cost, objective, error, risk | Five words, five terms: perte, coût, objectif (fonction objectif), erreur, risque. | merging them |
| accuracy, precision, recall | exactitude, précision, rappel; floating-point *precision* → précision (double précision); "recall that" is the verb: rappelons que | précision for accuracy |
| normalization | → normalisation (normalisation par lots); normal distribution → loi normale; *regularization* → régularisation; *smoothness* → régularité; *standardize* → centrer et réduire | mixing them |
| training, learning | train → entraîner, l'entraînement (jeu d'entraînement, perte d'entraînement); learn → apprendre, l'apprentissage (taux d'apprentissage, apprentissage supervisé) | formation |
| network, net | → le réseau (réseau de neurones, réseau résiduel, réseau cible) | réseau neuronal, filet |
| translation | of an image or function → la translation (équivariance par translation); of language → la traduction | traduction d'une image |
| gradient, derivative | le gradient; la dérivée, "with respect to $x$" → par rapport à $x$; differentiable → dérivable (one variable), différentiable | dérivative, dégradé |
| likelihood, evidence | la vraisemblance; Bayesian → l'évidence (borne inférieure de l'évidence); "empirical evidence" → résultats, indices | probabilité, évidence empirique |
| expectation, mean | l'espérance, "in expectation" → en espérance; la moyenne, "on average" → en moyenne | attente |
| sequence | ML data → la séquence; in mathematics → la suite (suite convergente) | séquence for $(x_n)$ |
| map, mapping | function → l'application; feature map → la carte de caractéristiques; "maps $x$ to $y$" → associe $y$ à $x$ | mapping, mappage |
| inference | running a trained model, statistics → l'inférence; "the layer infers its input size" → déduit automatiquement | |
| shape, size, dimension | la forme (of a tensor), la taille, la dimension | shape |
| memory | hardware → la mémoire; "workspace memory" → mémoire de travail | |
| domain | of a function → l'ensemble de définition; of data → le domaine | |
| order | second-order method → méthode du second ordre; tensor order → tenseur d'ordre quatre; sequence order → l'ordre; order of magnitude → ordre de grandeur | |
| performance | of a model → les performances; of code → la vitesse | la performance du code |
| label | → l'étiquette; of an axis → le titre de l'axe | label |
| number, digit, figure | a count → le nombre (de); an identifier → le numéro; a digit → le chiffre; a picture → la figure | digit, numéro de paramètres |
| built-in, hand-crafted | intégré (couche intégrée); conçu à la main | natif (*sonnet*) |
| library, framework | la bibliothèque; le framework | librairie (= bookshop) |
| converge | converger **vers** $x^*$ | converger à |

### 8.2 False friends

| English | French | Not (because) |
|---|---|---|
| to derive | établir, obtenir, déduire | dériver (= to differentiate) |
| actually, actual | en réalité, réellement ; réel | actuellement (= currently) |
| eventually | finalement, à terme | éventuellement (= possibly) |
| to assume | supposer | assumer (= to take on) |
| to support | prendre en charge, permettre | supporter (= to bear) |
| consistent | cohérent ; *consistent estimator* → estimateur convergent | consistant (= thick, substantial) |
| sensible | raisonnable, judicieux | sensible (= sensitive) |
| efficient | efficace | efficient |
| definitely | sans aucun doute, clairement | définitivement (= for good) |
| dramatically | nettement, fortement | dramatiquement |
| significant | important, notable ; *significatif* only in the statistical sense | significatif for "large" |
| arbitrary | quelconque (in mathematics) ; arbitraire (chosen freely) | arbitraire for "any" |
| to address | traiter, aborder | adresser |
| to complete | terminer, achever | compléter (= to add to) |
| location | position, emplacement | location (= rental) |
| opportunity | occasion, possibilité | opportunité |
| alternatively | sinon, on peut aussi | alternativement (= in turns) |
| digital | numérique | digital (= of fingers) |
| billion, trillion | milliard, billion | billion, trillion |
| silicon | silicium | silicone |
| decade | décennie | décade (= ten days) |
| concurrent | simultané | concurrent (= competing) |
| versatile | polyvalent | versatile (= fickle) |
| "makes sense" | a du sens, est logique | fait du sens |
| typically | en général, le plus souvent | typiquement (every time) |
| trivial | immédiat, évident ; *trivial* only in mathematics | |

### 8.3 Translationese

| Pattern | Write instead | Example |
|---|---|---|
| "It is important to note that", "Note that" word for word | Drop it, or Notons que …, Remarquons que … ; *Notez que* at most once per page | "Notice that the output shrinks slightly." → Remarquons que la sortie rétrécit légèrement. (*fable*) |
| "Recall that" | Rappelons que … ; Comme nous l'avons vu à la … | ✗ Rappelez-vous que $b$ est le biais. ✗ Rappelons-nous … (*sonnet*) ✓ Rappelons que $b$ est le biais. |
| "based on" as *basé sur* | fondé sur, à partir de, qui repose sur | "a model based on attention" → un modèle fondé sur l'attention |
| "using X" as *en utilisant X* every time | avec, à l'aide de, par | "using a pair of loops" → à l'aide de deux boucles (*fable*) |
| "for 10 epochs", "over 10 steps" | pendant 10 époques, en 10 pas | ✗ pour 10 époques |
| "in order to" as *afin de* every time | pour | |
| "in terms of" | en fonction de (a formula), du point de vue de, pour ce qui est de | ✗ en termes de (every time) |
| Inanimate subject + "tells us", "allows us to" | On en déduit que, Il en résulte que, … permet de | "This result tells us that depth helps." ✗ Ce résultat nous dit que … ✓ On en déduit que la profondeur aide. |
| "we can" every time as *nous pouvons* | The verb itself, or *on peut* once | "we can compute" → nous calculons |
| "there is / there are" as *il y a* every time | il existe, on compte, or a full verb | "There are three reasons." → Trois raisons l'expliquent. |
| "this", "it" alone as subject | cela, or repeat the noun | "This means that it converges." ✗ Ceci signifie qu'il converge. ✓ Autrement dit, l'itération converge. |
| "a pair of", "a couple of" | deux | "a pair of loops" → deux boucles ✗ une paire de boucles (*sonnet*) |
| "Sure enough", "Clearly" | Comme prévu, Manifestement | |
| Idioms translated by image | The meaning | "under the hood" → en interne ; "from scratch" → à partir de zéro ; "out of the box" → tel quel ; "sanity check" → vérification de cohérence ; "rule of thumb" → règle empirique ; "trade-off" → compromis ; "end to end" → de bout en bout ; "at the cost of" → au prix de |
| Title Case, English plural or genitive | French forms | ✗ La Fonction de Perte ✗ les patches ✗ les poids d'Adam's |

Logical and mathematical phrases are fixed: because → car, parce que;
therefore → donc, par conséquent; however → cependant, toutefois; only if →
seulement si; if and only if → si et seulement si; at most → au plus; at least
→ au moins; "let $x$ be …" → Soit $x$ … (Soient $x$ et $y$ …); such that → tel
que (telle que, tels que); with respect to → par rapport à; without loss of
generality → sans perte de généralité; for all → pour tout; there exists → il
existe; respectively → respectivement; almost surely → presque sûrement;
"up to a constant", "up to sign" → à une constante près, au signe près; i.i.d.
→ indépendants et identiquement distribués (then i.i.d., as the English);
bounded → borné; smooth → lisse, régulier; dot product → produit scalaire;
namely → à savoir; "holds" → est vérifiée.

## 9. Sources

Checked on 2026-11-26:

- This repository: `glossary/terms/fr.jsonl` (all 1,191 rows),
  `glossary/notes/es-fr-pt.md`, the three answer sets of
  `evals/fr-conv-layer/` with their DECISIONS and NOTES, `tools/l10n/protect.py`,
  `glossary.py`, `config.py`, `build.py`, `_wt_helper.py`.
- What the build prints for references: Quarto's `_language-fr.yml` in
  d2l-neu's `.venv-build` (Chapitre, Section, Figure, Table, Équation), and the
  rendered English page in `_build/xx/` ("Equation 6.2.1", "Chapter 4",
  "Figure 6.1", "Section 6.1").
- The check, run on French test sentences: `0,5` for `0.5` raises "numbers not
  found"; `10 000` passes only with `thousands: space`; `45 %` passes; the
  plural of a term with an apostrophe (jeux d'entraînement, fonctions
  d'activation) and a typed ’ are reported as missing; `nbsp()` joins `45 %`,
  `10 000`, `24 GiB` but leaves the spaces before `: ; ?` and inside « ».
- pandoc (d2l-neu `.venv-build`): `l'@eq-x`, `d'@eq-x`, `(@fig-x)`, `« @fig-x »`
  are all parsed as references.
- French Wikipedia, *Transformeur*: un transformeur (lower case, masculine),
  plongements lexicaux (embeddings), encodeur-décodeur, tête d'attention,
  fonction softmax, « affinés … (fine-tuning) » linked to *Réglage fin*; it
  writes *lexèmes (tokens en anglais)* where the glossary has *jeton*, and both
  *pré-entraînement* and *préentraînés*.
- Search-result excerpts of A. Géron, *Deep Learning avec Keras et TensorFlow*
  (Dunod, trad. H. Soulard): taux d'apprentissage, mini-lots, jeu
  d'entraînement, époque, optimiseur; it writes *surajuster* where the glossary
  has *surapprentissage*.

Not checked:

- The *Vocabulaire de l'intelligence artificielle* of the Commission
  d'enrichissement de la langue française (Journal officiel, 6 September 2024):
  the page could not be retrieved (HTTP 403). Search results show that it has
  entries for *jeton textuel*, *instruction générative* (prompt) and
  *transformeur*; their text was not read.
- The Académie française declaration and the circular of 2017 (§5), the
  *Lexique des règles typographiques en usage à l'Imprimerie nationale* (§2,
  §4) and the dictionary genders in §3.3: stated from the drafter's knowledge.
- No French university course material and no printed French ML textbook was
  read in full.
- Whether babel-french, loaded by `pdf.lang: "fr"`, changes list markers,
  caption separators or punctuation spacing in the PDFs.

## 10. Settings

In `languages.yml`, on the `fr` entry (existing keys, `tools/l10n/config.py`
and `protect.py`):

- `check: {thousands: space}`: needed, or every `10 000` raises a number
  warning (§4).
- `typography: {nbsp: true}`: makes the build put no-break spaces into
  `10 000`, `24 GiB` and `45 %`.

In `fr/strings.yml` (existing keys; the file does not exist yet, `l10n strings
fr` creates it):

- `quarto.crossref-tbl-title: "Tableau"` and `quarto.crossref-tbl-prefix:
  "Tableau"`: §2.1 writes *le*, *au*, *du* before table references. Without
  this the build prints "Table" (feminine) and those sentences are wrong.
- `pdf.lang: "fr"`, `book.exercises_heading: "Exercices"`,
  `book.discussions_link: "Discussions"`.

Needs new code:

- **French punctuation spacing.** `nbsp()` has no rule for it. Needed: outside
  protected tokens, turn the ordinary space before `:` `;` `?` `!` `»` and
  after `«` into a no-break space (proposed: U+202F before `;` `?` `!`, U+00A0
  before `:` and inside « »), e.g. as `typography: {nbsp: true, punctuation:
  french}`. The rule must skip table alignment rows (`| :--- |`), div fences
  (`:::`), and lines that begin with `:`; it must require a letter, digit,
  closing bracket or token before the space.
- **Glossary check and apostrophes.** `glossary._present` accepts inflected
  forms only for terms made of letters; a term with `'` or `-` must match
  exactly, so *jeux d'entraînement* and *fonctions d'activation* are reported
  as missing *jeu d'entraînement* and *fonction d'activation*. Splitting words
  at `'` and treating ’ as `'` would remove these false warnings. Until then,
  translators name them in NOTES.
- **Abbreviations:** the no-break rule for `z. B.` covers only one-letter
  parts; `p. ex.` is not joined. Minor, since §2 keeps abbreviations out of
  body text.

## 11. For the author to confirm

1. **Decimal sign** (§4): the point everywhere, as ruled for German. The
   alternative is the French comma in prose (0,5) with the point in math and
   code; it would need `check: {numbers: off}` or new code.
2. **Space before `%`** (§4): `45 %`, as French typography requires and two of
   three test translators wrote. The German ruling is no space (92.3%).
   Proposed: keep the space for French, because it belongs to the same system
   as the spaces before `: ; ? !`.
3. **No-break spaces** (§2, §10): translators type ordinary spaces and the
   build converts them; this needs the new punctuation rule. Proposed widths:
   narrow (U+202F) before `; ? !`, normal (U+00A0) before `:` and inside « ».
4. **Thousands** (§4): a space from five digits on, none in four-digit numbers
   (2048), as for German. French printing also groups four digits (2 048).
5. **Reference words** (§2.1): "Tableau" instead of Quarto's "Table"; and
   whether the words should stay capitalised mid-sentence (à la Section 6.1) or
   be set in lower case through `quarto.crossref-*-prefix` (à la section 6.1,
   the form French printing prefers; captions would keep "Figure"). Proposed:
   Tableau; keep the capitals until the lower-case override is tested.
6. **Article before references** (§2.1): translators write *à la*, *au*, *l'*
   by the label's prefix. This ties the text to the printed words; a change of
   a word's gender in `strings.yml` would need a pass over the translation.
7. **English gloss** (§3.6): `(angl. …)` once per page, parallel to the German
   `(engl. …)`. The alternative is a bare parenthesis, as the test translators
   wrote: *carte de caractéristiques* (feature map).
8. **jeton, plongement, réglage fin, transformeur** (§3.1): the glossary's
   French terms rather than the loans token, embedding, fine-tuning,
   Transformer, which many French papers use. Verb for "tokenize": segmenter
   en jetons, not tokeniser.
9. **Loans kept:** dropout, pooling, prompt, momentum, softmax (glossary), and
   framework, notebook, patch, pipeline, cache, benchmark (§3.1).
10. **Genders** (§3.3): le GPU, la SGD, la ReLU, le softmax; le LSTM and le
    GRU (by *réseau*, although the glossary expands them as *mémoire* and
    *unité*).
11. **People's names** (§6): as in English (Chebyshev, not Tchebychev), as
    ruled for German.
12. **Inclusive writing** (§5): none.
13. **Slide titles** (§7): an infinitive for English "-ing" titles
    (Implémenter …), a noun in book headings (Implémentation).
14. **Proof → Démonstration, Summary → Résumé, Concept Check → Vérification
    des acquis** (§6).
15. **1990 spelling** (top): not used (entraînement, chaîne, coût).

## 12. Proposed glossary corrections

None of these is applied; the glossary file was not edited.

| Concept | Now | Proposed | Reason |
|---|---|---|---|
| `sparsity` | parcimonie | add the sense "of a matrix or array": creux, creuse; "sparsity pattern" → structure creuse | *parcimonie* names a property of models and solutions; French numerical linear algebra says *matrice creuse* (§8.1; *fable* NOTE on block ec8621f8). |
| `long-short-term-memory` | mémoire à long et court terme | LSTM as the term; expansion *longue mémoire à court terme* at first use | The English means a short-term memory that lasts long, not "long and short"; French texts use the acronym. |
| `on-policy`, `off-policy` | sur politique, hors politique | keep *hors politique*; replace *sur politique* (proposal: *sous la politique courante*, or the loan *on-policy* as an invariable adjective) | *sur politique* is not attested French and reads as "about politics". |
| `independent-and-identically-distributed`, `identically-distributed`, `lipschitz-continuous`, `positive-definite`, `positive-semidefinite` | feminine forms only | add the masculine and plural forms as alternatives | The term agrees with its noun (§3.2); the check should accept *indépendants et identiquement distribués*, *lipschitzien*, *défini positif*. |
| `null-space` | noyau d'une application linéaire | noyau (with a note: complement by the map, *noyau de $\mathbf{A}$*) | The entry is a description, not a term that can stand in a sentence. |
| `tokenization` | segmentation en jetons | keep; add the verb *segmenter en jetons* and the agent noun for "tokenizer" (proposal: *segmenteur*, to confirm) | §3.5 needs the verb; "tokenizer" has no entry. |
| `standard normal distribution` (no entry) | none | loi normale centrée réduite | The fixed French name; all three test translators chose it. |
| `cross-correlation` | corrélation croisée | keep; note that signal processing says *intercorrélation* | All three translators and French deep-learning texts use *corrélation croisée*. |
