# French (fr) glossary review

Reviewed 2026-11-26 against `glossary/terms/fr.jsonl` (1,184 entries: 91
`verified_external`, 1,093 `proposed`, no `verified_d2l`). Corrections are in
`fr-corrections.jsonl` (102 lines). All 91 verified entries and all 237
proposed entries with frequency ≥ 50 were checked; the other 856 proposed
entries were all read once (not sampled), with source checks only where a term
looked wrong.

## Verdict

The French glossary has no Japanese-style failure: there is no earlier French
D2L edition, so no machine-translated prose was promoted to "verified", and
the classic traps are rendered correctly (invariance par translation,
disparition du gradient, règle de la chaîne, perceptron multicouche, Adam,
généralisation, variance). The proposals read as written by someone who knows
French mathematics (théorème des accroissements finis, fonction de
répartition, échantillonnage préférentiel, estimateur convergent, repliement
spectral). The problems are of a different kind: over-translation of terms
that French practitioners keep in English, descriptive paraphrases where a
term is needed, and head words that change between an entry and its compounds.

| Status | Entries | Flagged | Not minor | Wrong sense or calque |
|---|---|---|---|---|
| `verified_external` | 91 | 18 | 11 | 0 |
| `proposed`, frequency ≥ 50 | 237 | 32 | 31 | 2 |
| `proposed`, frequency < 50 | 856 | 52 | 37 | 7 |

By severity: 1 wrong sense, 8 calque/MT, 43 non-standard, 27 inconsistent,
23 minor. About 9% of the glossary changes; the share is higher among the
frequent terms (13%), and several of those occur hundreds of times (GPU 1,195,
kernel 1,058, bias 959, prompt 586, return 571, pooling 435, forward pass 355,
embedding 352, dropout 237, warmup 212).

## Provenance of the 91 verified entries

- **24 from Google's ML glossary (`?hl=fr`).** The page declares itself
  machine-translated: its `<html>` carries `lang="fr-x-mtfrom-en"` (checked
  2026-11-26). `sources.jsonl` records this marker for the ar/ja/ko/vi/zh
  Google sources but not for the three French ones; add it. These 24 are basic
  terms and MT got them right, except that `ensemble d'entraînement` /
  `ensemble de validation` conflict with `jeu de données` (2 flagged).
- **67 from the Stanford CS221/229/230 cheatsheet translations**
  (shervinea/cheatsheet-translation). These are volunteer human translations,
  not MT, but uneven: the same set writes « Hinge loss » in one file and
  « Fonction objectif de Hinge » in another, and leaves Pooling, Stride,
  Padding and feature map in English. The harvest took headings as they came:
  16 of the 67 are flagged, among them `analyse des composantes principales`,
  `théorème de la limite centrale`, `fonction de valeurs`, `fonction objectif
  de Hinge`, and the stray capital in `intersection sur Union`.
- `julien-blanchon/d2l-fr` is English prose under a French name
  (DISCOVERY.md); it contributed nothing, correctly.

So "verified" here means "appears as a heading in one MT page or one volunteer
cheatsheet". None of the 91 is wrong in sense; 11 are non-standard or
inconsistent with the rest of the glossary.

## Systematic problems

1. **Over-translation.** French ML teaching keeps a set of English terms, and
   the glossary translates them into words nobody uses: `regroupement` for
   pooling (it means clustering), `abandon aléatoire` for dropout,
   `processeur graphique` for GPU, `apprentissage Q` for Q-learning,
   `diffusion des dimensions` for broadcasting (collides with diffusion
   models), `ajustement de flot` / `ajustement du score` for flow/score
   matching, `collecte collective` for all-gather, `vecteurs globaux` for
   GloVe, `réseau dans le réseau` for NiN. CNAM's deep learning course, the
   French cheatsheets and fr.wikipedia all write pooling, dropout, GPU,
   Q-learning.
2. **Paraphrase instead of term.** `phase de montée progressive du taux
   d'apprentissage` (warmup, 212 occurrences), `traitement initial du contexte`
   (prefill), `modèle de plafond de performance` (roofline), `logarithme de
   somme d'exponentielles`, `cache de clés et de valeurs`, `mise à jour par
   estimations` (bootstrapping), `courbe caractéristique de fonctionnement du
   récepteur` (ROC).
3. **Head word differs from its compounds.** `plongement vectoriel` vs.
   `table de plongements`; `espérance mathématique` vs. `espérance
   conditionnelle`; `noyau de convolution` vs. `taille du noyau`; `retour
   cumulé` vs. `retour de Monte-Carlo`; `critique de valeur` vs. `perte du
   critique`; `terme de biais` / `biais statistique` vs. `estimateur sans
   biais`; `algorithme du gradient stochastique` vs. `descente de gradient`;
   `erreur d'apprentissage` vs. `perte d'entraînement`; `jeu de données` vs.
   `ensemble d'entraînement`.
4. **Two concepts, one rendering.** cross-entropy and cross-entropy loss;
   squared error and squared loss; weight sharing and weight tying; retrieval
   and search (`recherche`); subsampling and downsampling
   (`sous-échantillonnage`, acceptable).
5. **Calques.** `produit extérieur` for outer product (that is the wedge
   product; the one wrong sense found), `produit intérieur`, `mode avant`,
   `forme fermée`, `racine carrée moyenne` (RMS is `moyenne quadratique`),
   `adversarial` (French: `antagoniste`), `postérieure`, `balle lourde`.
   One Portuguese typo: `mau conditionnement`.
6. **Standard French names missed.** `méthode de la puissance itérée`,
   `bilan détaillé`, `soupassement`, `théorème central limite`, `analyse en
   composantes principales`.

Sense collisions checked and found sound: accuracy `exactitude` vs. precision
`précision`; loss `perte` vs. objective `objectif`; intercept vs. bias;
quantization and uncertainty quantification (both `quantification` in French);
RL policy/value/state/action/reward; label `étiquette`; mean `moyenne` vs.
expectation `espérance`; logit, softmax and Adam kept; `noyau` for convolution
kernel, covariance kernel, transition kernel and null space (French uses one
word for all four, as it should).

## Official terminology (FranceTerme) versus practice

Checked against the Commission d'enrichissement's *50 termes clés de
l'intelligence artificielle* (terms published in the Journal officiel, most on
6 Sept 2024).

| English | FranceTerme | Practitioners | Glossary now | Recommendation |
|---|---|---|---|---|
| prompt | instruction (générative); « invite » is "déconseillé dans ce sens" | prompt | invite | prompt; never « invite » |
| token | jeton textuel | token | jeton | author's ruling (below) |
| transformer | transformeur | Transformer / transformeur | transformeur | keep |
| LLM | grand modèle de langage (GML) | LLM, grand modèle de langage | grand modèle de langage | keep |
| diffusion model | modèle à bruit statistique | modèle de diffusion | modèle de diffusion | keep; the official term is unused |
| adversarial | antagoniste | adversarial / antagoniste | adversarial | antagoniste |
| self-supervised | autosupervisé | auto-supervisé | auto-supervisé | autosupervisé |
| training set | jeu d'entraînement | jeu / ensemble d'entraînement | ensemble d'entraînement | jeu d'entraînement |
| deep learning, RL, transfer learning | apprentissage profond / par renforcement / par transfert | same | same | keep |

FranceTerme has no entry for fine-tuning, embedding, dropout, pooling or batch
in this list.

## Decisions for the author or a native reviewer

1. **token**: `jeton` (official, and what the glossary's 11 other token entries
   use) or `token` (what most French NLP practitioners write, 2,117
   occurrences). Either is defensible; it must be one. With `jeton`, keep
   `segmentation en jetons` (FranceTerme's own wording) for tokenization.
2. **How much English to keep.** I recommend keeping pooling, dropout,
   Q-learning, broadcasting, flow matching, score matching and collective-op
   names. If the author prefers a fully French text, the alternatives are
   `mise en commun`, `abandon`, `appariement de flots`; they need a native
   reviewer, and `regroupement` should go either way.
3. **Algorithm names with French expansions**: `optimisation proximale de
   politique`, `acteur-critique souple`, `réseau Q profond`, `estimation
   généralisée de l'avantage`, `mémoire à long et court terme`. Suggest the
   acronym (PPO, SAC, DQN, GAE, LSTM) as the working term and the French
   expansion once per chapter.
4. **RL return**: `retour` (recommended, matches the compounds), `gain`
   (fr.wikipedia) or `utilité` (cheatsheet). Also `sur politique` /
   `hors politique`: attested in the cheatsheet, while fr.wikipedia keeps
   on-policy / off-policy.
5. **minibatch / batch**: `mini-lot` / `lot` come from Google's MT page;
   French courses mostly write « batch » and « mini-batch ». The French forms
   are understood and consistent across 6 entries; keep unless the reviewer
   objects.
6. **loi vs. distribution**: named families are `loi` (loi de Poisson, loi
   uniforme), generic ones `distribution`. This matches French practice;
   confirm `loi gaussienne` vs. `loi normale`, and `jointe` vs. `conjointe`
   (joint-distribution, joint-probability).
7. **Smaller native checks**: `calendrier` for schedule (vs. `planification`),
   `rétroaction` for recommender feedback (vs. `retour`), `connexion de saut`
   (skip connection), `acuité` (sharpness), `surprise` (surprisal),
   `prix d'ombre`, `hyperprior`, `remplissage` vs. `marge` for padding,
   `arrêt anticipé` vs. `arrêt précoce`, `taux d'apprentissage` vs.
   `pas d'apprentissage` (CNAM uses the latter).
8. **Abbreviations**: the long forms `équation différentielle ordinaire /
   stochastique` appear in 5 entries; rule that EDO and EDS are used after
   first mention.
9. **Hyphenation**: FranceTerme and the glossary write `préentraînement`,
   `autosupervisé` without hyphen; fr.wikipedia writes `pré-entraîné`,
   `auto-encodeur`. Adopt the unhyphenated (1990 reform) forms throughout,
   with `auto-attention` the only exception (vowel clash).

## Sources

- Commission d'enrichissement de la langue française, *50 termes clés de
  l'intelligence artificielle* (2025), read from
  `https://datafranca.org/wiki/images/50_termes_de_l'IA-2025.pdf`; publication
  page on culture.gouv.fr.
- CNAM, cours US3362 « Apprentissage profond » (Pons, Breloy, Audebert),
  `https://cedric.cnam.fr/vertigo/Cours/US3362-Apprentissage/slides/coursDeep{1,2,3}.pdf`:
  passe avant / passe arrière, descente de gradient stochastique, pooling,
  dropout, fine-tuning, biais, inertie (momentum), pas d'apprentissage.
- S. Airiau (Paris-Dauphine), reinforcement learning slides,
  `https://www.lamsade.dauphine.fr/~airiau/Teaching/AOL/aol-02.pdf`: itération
  sur les valeurs, fonction de valeur, glouton.
- fr.wikipedia article titles and lead text (ACP, Théorème central limite,
  Réglage fin, Dropout, Q-learning, Processus de décision markovien, Bilan
  détaillé, Soupassement arithmétique, Méthode de la puissance itérée, Moyenne
  quadratique, Produit dyadique, Courbe ROC, Réseaux antagonistes génératifs,
  Réseau neuronal convolutif, Transformeur, Apprentissage par renforcement).
- The Stanford cheatsheet translations at the pinned commit, read as evidence
  of volunteer usage, not as authority.

Not consulted: the French translation of Goodfellow et al. (*L'apprentissage
profond*, Massot/Quantmetry 2018) is not available online, and scikit-learn has
no official French documentation. A reviewer with the book should check
`mini-lot`, `normalisation par lots`, `réglage fin` and `rétropropagation`
against it. Corrections marked `proposed` (69 of 102) rest on my judgement and
on consistency arguments, not on a cited source.
