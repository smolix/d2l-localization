# Brazilian Portuguese (pt-BR) glossary review

Reviewed 2026-11-26 against `glossary/terms/pt-BR.jsonl` (1,184 entries: 17
`verified_d2l`, 67 `verified_external`, 1,100 `proposed`). All 1,184 entries
were read against the sense in `concepts.jsonl`, not only the sample the brief
asked for. Corrections: `glossary/review/pt-BR-corrections.jsonl` (124 lines).

## Verdict

The glossary is in much better shape than the Japanese one. The mathematical
and statistical vocabulary (about two thirds of the entries) is correct,
idiomatic Brazilian Portuguese: verossimilhança, autovalor, posto, acurácia,
revocação, desvanecimento do gradiente, equivariância à translação, condição de
contorno, política gulosa. Nothing like 翻訳不変性 survived: the worst
machine-translation errors of the earlier Portuguese edition ("invariância da
tradução", "recurso" for feature) were not carried over.

The problems are of three kinds: a few attested-but-wrong "verified" entries,
editorial over-translation of terms Brazilians keep in English, and internal
inconsistency.

| Status | Entries | Would change | Of which |
|---|---|---|---|
| `verified_d2l` | 17 | 3 (18%) | 1 calque (`expectation`), 1 inconsistent, 1 minor |
| `verified_external` | 67 | 13 (19%) | 2 calques, 5 inconsistent, 1 non-standard, 5 minor (capitals) |
| `proposed` | 1,100 | 108 (10%) | 2 wrong sense, 1 typo, 31 non-standard, 44 inconsistent, 30 minor |
| `proposed`, frequency ≥ 50 | 229 | 35 (15%) | |

By severity: wrong_sense 2, calque_or_mt 4, non_standard 32, inconsistent 50,
minor 36. Only about ten changes are errors a reader would call wrong; the
rest follow from three or four policy decisions (below), so the count drops to
a handful once those are made.

## Provenance

- **`verified_d2l` (17): machine translation, lightly post-edited.**
  `d2l-ai/d2l-pt` has `translator = aws` in `config.ini` (same pipeline as the
  Japanese and Korean editions). The repository was started in July 2021 by
  three people at the Universidade do Oeste de Santa Catarina (Dalcimar
  Casanova, Lucas V. Hendges, Fabio H. Kurpel, listed as "Portuguese
  Translators"); the chapters arrived in one bulk commit and later commits
  touch a few files. English passages remain (the opening of
  `linear-regression.md`), and the text has MT errors: "invariância da
  tradução" (translation invariance), "recurso" for feature (273 occurrences),
  "Na expectativa, $E[x'] = x$". It also mixes variants freely: taxa de
  aprendizado 82 / aprendizagem 67; gradiente descendente 42 / descida do
  gradiente 22; overfitting 49 / sobreajuste 10. Its evidence shows a term
  occurs, not that it is right. It is still useful as a usage count for what
  the post-editors left in English (dropout 294, embedding 220, kernel 415,
  pooling 164, stride 199, minibatch 262).
- **`verified_external` (67), two sources:**
  - 46 from `shervinea/cheatsheet-translation` (`pt/`), volunteer human
    translations of the Stanford CS229/CS230 cheat sheets. The CNN sheet names
    a translator and a reviewer; other files still have the "Traduzido por X, Y
    e Z" placeholder, i.e. unreviewed. Quality is uneven: "porta esquecida"
    (forget gate), "estimador imparcial" (unbiased), "mapa de recursos"
    (feature map), "probabilidade máxima estimada" (maximum likelihood). The
    repository targets Portuguese in general, not pt-BR. Term headings were
    copied with their capitals (rede Neural, teorema do Limite Central).
  - 21 from Google's pt-BR ML glossary topic pages, which are auto-localized.
    Two of its terms are wrong ("função objetiva", "camada escondida"); Google's
    own pt-BR crash course says "camada oculta".
- **`proposed` (1,100):** editorial suggestions by a model. Competent, with a
  recognisable habit of translating everything, including names.

## Systematic problems

1. **Over-translation into invented paraphrases.** Terms every Brazilian
   practitioner says in English were replaced by descriptions nobody uses:
   dropout → "desativação aleatória", prompt → "instrução", embedding →
   "incorporação vetorial", roofline → "modelo de teto de desempenho",
   broadcasting → "transmissão de dimensões", teacher forcing → "forçamento
   pelo professor", flow/score matching → "ajuste de fluxos/escores",
   Q-learning → "aprendizado Q", off-policy → "fora da política", all-reduce →
   "redução coletiva", OV circuit → "circuito valor-saída".
2. **Disambiguators baked into the term.** The sense note was written into the
   word: "núcleo de convolução", "termo de viés", "viés estatístico", "crítico
   de valor", "crítico generativo", "retorno acumulado", "função escore de
   densidade", "reamostragem bootstrap". The running-text term is the bare
   word; the qualifier belongs in `notes`.
3. **Expansions as preferred terms for acronyms.** GPU ("unidade de
   processamento gráfico", 1,195 occurrences), ReLU, LSTM, GRU, KV cache,
   log-SNR. The term is the acronym; the expansion is a first-mention gloss.
4. **Sense collisions created by translating.** momentum → "momento" collides
   with primeiro/segundo momento in the Adam sections; stride → "passo"
   collides with step (tamanho do passo, passo de tempo); pooling →
   "agrupamento" is the Brazilian word for clustering; kernel → "núcleo" is the
   linear-algebra kernel (null space); "ajuste" is fit, fine-tuning and now
   matching; "escalonamento" is both scheduling and scaling; "recorte" is both
   clipping and crop. The collisions the brief listed are otherwise handled
   well: escore (density) vs pontuação (metric, attention), esperança in every
   compound except the head entry, perda throughout, política/valor/estado/
   ação/recompensa consistent, característica for feature.
5. **Inconsistent head words.** descent: "gradiente descendente" vs "descida"
   in seven compounds; prior/posterior: "a priori", "prévia", "prior", "a
   posteriori", "posterior" across 20 entries; hidden: "camada escondida" vs
   "estado oculto"; Hessian: "hessiano" vs "autovalor da hessiana"; sample:
   "média da amostra" vs "variância amostral".
6. **Mechanical errors.** Agreement ("descida de gradiente estocástico/
   projetado"), a typo ("distribuição marginale"), stray capitals from source
   headings (6 entries), missing hyphens (rede alvo, caixa âncora),
   "logaritmo da verossimilhança" for log-verossimilhança.

Names are otherwise handled correctly: Adam, AdamW, BERT, Mamba, Muon,
RMSNorm, DDPM, softmax, skip-gram, word2vec stay in Latin script and unchanged.

## Recommended policy for English terms

One rule, applied everywhere: **translate when Brazilian university teaching
has a settled Portuguese term; otherwise keep the English word, set it in
italics with a short Portuguese gloss at first mention, and use it unchanged
afterwards. Never coin a paraphrase to avoid an English word.**

- Translate (settled): retropropagação, sobreajuste, ajuste fino, taxa de
  aprendizado, função de perda, camada oculta, lote, época, rótulo,
  característica, codificador/decodificador, autoatenção, preenchimento,
  parada antecipada, aprendizado por reforço, política, recompensa.
- Keep English: dropout, embedding, prompt, token, kernel, pooling, stride,
  momentum, softmax, logit, transformer, checkpoint(ing), broadcasting,
  bootstrap(ping), Q-learning, on-/off-policy, teacher forcing, flow matching,
  score matching, roofline, underflow, pipeline, cache, and all algorithm,
  model and primitive names.
- Acronyms keep their English letters (GPU, LSTM, MLP, CNN, SGD, PPO, ELBO),
  with the Portuguese expansion once.

## Decisions for the author or a native reviewer

1. **The policy above**, in particular pooling, kernel, stride and momentum,
   where a Portuguese word exists but collides (36 of the corrections follow
   from it).
2. **gradient descent:** "descida de gradiente" (consistent with all other
   compounds; recommended) or "gradiente descendente" (very common in Brazil,
   but a word-order calque). Either way one form; fix the agreement in the SGD
   entry.
3. **prediction:** "previsão" (current; Google) or "predição". Academic ML
   prefers predição and keeps previsão for forecasting. Left unchanged; seven
   compounds depend on it.
4. **minibatch:** "minilote" (current, consistent with lote) or "minibatch"
   (what d2l-pt and many courses use). Left unchanged.
5. **ODE/SDE:** Portuguese mathematics writes EDO and EDE. Recommended, as the
   one exception to "acronyms stay English"; PCA, SVD, MSE stay.
6. **Algorithm expansions** (otimização proximal de política, ator-crítico
   suave, rede Q profunda): gloss once, then PPO, SAC, DQN.
7. **denoising** → "remoção de ruído" (six compounds) is correct but heavy;
   "denoiser" has no good equivalent.
8. **schedule** → "agendamento" (proposed) vs "cronograma" (current).
9. **exploration** is "exploração", and so is exploitation in ordinary
   Portuguese. The book's exploitation must be "explotação" (or
   "aproveitamento"); add a concept entry.
10. **bound:** "limite" (current: limite da união, limite inferior da
    evidência) vs "cota"; limite also means limit. Left unchanged.
11. **multi-armed bandit** ("bandido de múltiplos braços") and **statistical
    bias** ("viés"; older Brazilian statistics says "vício", "estimador não
    viciado"): confirm with a native reviewer.

## European Portuguese variants (do not use; add as `alternatives` if a pt-PT edition is planned)

| pt-BR (glossary) | pt-PT |
|---|---|
| aprendizado (de máquina, por reforço, profundo) | aprendizagem (automática, por reforço, profunda) |
| treinamento | treino |
| posto (rank) | característica |
| autovalor, autovetor | valor próprio, vetor próprio |
| ponto flutuante | vírgula flutuante |
| neurônio, epistêmica (ô, ê) | neurónio, epistémica (ó, é) |
| verossimilhança | verosimilhança |
| condição de contorno | condição de fronteira |
| decodificador | descodificador |
| fatoração | fatorização |
| acurácia, revocação | exatidão, abrangência/sensibilidade |
| faixa dinâmica | gama dinâmica |
| interseção | intersecção |
| mapa, transformação | aplicação |
| conjunto de dados; tela, arquivo (prose) | conjunto de dados; ecrã, ficheiro |

Two current entries lean European: "aplicação de fluxo" (flow map) and the
Stanford-sourced terms, whose repository is not Brazil-specific.

## Sources relied on

- `d2l-ai/d2l-pt` at `ee8a9f0` (`config.ini`, `static/frontpage/frontpage.html`,
  git history, corpus counts).
- `shervinea/cheatsheet-translation` at `99f45b6`, `pt/*.md` (read in full for
  the CNN, deep-learning, probability and linear-algebra sheets).
- pt.wikipedia.org articles (titles and lead sentences via the API): Valor
  esperado, Camada oculta, Análise de componentes principais, Hessiano, Posto
  matricial, Média e covariância amostrais, Word embedding, Engenharia de
  prompts, Q-learning, Sobreajuste, Método do gradiente, Otimização, Rede
  neural convolucional, Formato de ponto flutuante de meia precisão, Valor-p,
  Teorema central do limite, Problema do desaparecimento do gradiente.
  Wikipedia ML articles are often translated from English and mix variants;
  used for attestation, not as the sole authority.
- Hugging Face LLM course in Portuguese (chapter 2.3): checkpoint, tokenizer
  kept in English.
- University material found by search (not all opened): L. T. Rolla,
  *Introdução à Probabilidade* (IME-USP), "esperança"; UFMG reinforcement
  learning syllabus (L. Chaimowicz), "Métodos On-Policy x Off-Policy"; Unicamp
  IA353 and ICMC-USP deep-learning slides.

Not checked: the Brazilian editions of Géron (*Mãos à Obra*), Faceli et al.
(*Inteligência Artificial: Uma Abordagem de Aprendizado de Máquina*) and
Russell and Norvig were not available online; a native reviewer with those
books should settle decisions 2 to 4.
