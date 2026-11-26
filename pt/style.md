# Style guide: Brazilian Portuguese (pt, locale pt-BR)

Draft of 2026-11-26 for the author's review. Translators read this with every
unit (TRANSLATING.md §7–§8). Each rule has one reason and one example
(English → Portuguese, or ✗ wrong / ✓ right). Examples marked F, O or S are
sentences from the translation test `evals/pt-conv-layer/` (Fable, Opus,
Sonnet). Points that still need a ruling are marked "(to confirm)" and
collected in §11.

Order of authority: the decisions log, then this guide, then the glossary. The
glossary says *what* a term is; this guide says *how to choose and inflect*.
The glossary already carries the corrections of the review of 2026-11-26
(`glossary/review/pt-BR.md`); §12 lists the few entries this guide would still
change.

## 1. Variety and register

### 1.1 Brazilian, not European, Portuguese

**Decision: the written standard of Brazil (Acordo Ortográfico de 1990 as
applied in Brazil, VOLP of the Academia Brasileira de Letras), in spelling,
vocabulary and syntax.** Reason: the locale is pt-BR and the readers are
Brazilian students and engineers; European forms read as foreign and several
are different technical terms.

| Point | Write (pt-BR) | Not (pt-PT) |
|---|---|---|
| The two head nouns of the book | aprendizado (de máquina, profundo, por reforço), treinamento | aprendizagem (automática, profunda), treino |
| Linear algebra | autovalor, autovetor, posto (rank), fatoração | valor próprio, vetor próprio, característica, fatorização |
| Other terms | ponto flutuante, verossimilhança, decodificador, acurácia, revocação, interseção, sequência (also for $(x_n)$), condição de contorno | vírgula flutuante, verosimilhança, descodificador, exatidão, abrangência, intersecção, sucessão, condição de fronteira |
| Spelling: ô/ê before m, n | neurônio, polinômio, binômio, fenômeno, econômico, epistêmica, gênero | neurónio, polinómio, fenómeno, económico |
| Spelling: kept consonants | detectar, detecção, aspecto, perspectiva, recepção, caractere (a character), fato, contato, seção, registrar, controle, equipe | detetar, deteção, aspeto, perspetiva, receção, carácter, facto, contacto, secção, registar, controlo, equipa |
| Numbers | dezesseis, dezessete, dezenove; bilhão, trilhão ($10^9$, $10^{12}$) | dezasseis, dezassete, dezanove; mil milhões, bilião |
| Computing words | arquivo, tela, usuário, baixar, salvar, acessar, senha, aplicativo, banco de dados, gerenciar, pixel/pixels, mouse | ficheiro, ecrã, utilizador, descarregar, guardar, aceder, palavra-passe, aplicação, base de dados, gerir, píxel/píxeis, rato |
| Progressive | estar + gerund: o modelo está convergindo | estar a + infinitive: está a convergir |
| Object pronouns | before the verb inside a sentence (o que nos permite, a perda se anula); never first in a sentence (✓ Observa-se que … ✗ Se observa que …); no mesoclisis (✗ calcular-se-á → será calculado) | enclisis by default (permite-nos, anula-se) |
| Possessives | no article: nosso modelo, sua entrada | o nosso modelo, a sua entrada |
| Contractions | em um, em uma, de um | num, numa, dum |
| Address | você | tu, vós, o leitor (third person) |

✓ F: "testamos nosso detector de bordas feito à mão" ✗ F (same page): "o nosso
detector de bordas $1 \times 2$". ✗ S: "numa curva em sino" ✓ "em uma curva em
sino".

### 1.2 Authorial *nós*, the reader as *você*

**Decision: first person plural with the pronoun left out; the reader is
*você*; never *tu*, *a gente*, or the impersonal third person as a habit.**
Reason: this is the register of Brazilian university textbooks and of the
Brazilian editions of technical books, and the verb ending already says "we"
(TRANSLATING.md §8). All three test translators chose it.

| English | Portuguese | Example |
|---|---|---|
| we + action in the text | verb in *-mos*, present tense, no *nós* | "We minimize the loss with SGD." → Minimizamos a perda com o SGD. ✗ Nós minimizamos … (every sentence) |
| "let us", "let's" | *Vamos* + infinitive, or Considere / Vejamos | "Let us look at the gradient." → Vamos examinar o gradiente. ✗ Deixe-nos …, ✗ Olhemos |
| "we will see", "we will use" | present, or *vamos* + infinitive; the simple future (veremos) only for a forward reference | "Now we see what the formula computes." → Agora vamos ver o que essa fórmula calcula (F). "We return to this in …" → Voltaremos a isso na … |
| "you" = the reader (advice, a question, exercises) | *você*, *seu/sua*; imperative in the third person: Considere, Observe, Suponha | "Suppose you want to estimate the price of a house." → Suponha que você queira estimar o preço de uma casa. |
| "you can", "one can" = anyone | *podemos*, *é possível*, or the *se* passive | "You can verify this numerically." → Isso pode ser verificado numericamente. / Podemos verificar isso numericamente. |
| "our model", "our dataset" | *nosso/nossa* without article, or the plain article | "our MLP weights" → os pesos do nosso MLP |
| "we" = people, the field | nós, seres humanos; a área; os pesquisadores | "We humans attend selectively." → Nós, seres humanos, prestamos atenção de forma seletiva. |

**Sentence shape** (each point is checkable):

- **Verbs, not noun chains.** No sentence whose only verb is *realizar*,
  *efetuar*, *ocorrer* or *dar-se*. ✗ O cálculo do gradiente é realizado por
  meio da aplicação da regra da cadeia. ✓ Calculamos o gradiente com a regra
  da cadeia.
- **Active voice with *-mos*.** Passive only where the agent does not matter
  (O modelo foi pré-treinado no ImageNet); then prefer the *se* passive to
  *ser* + participle when it is shorter (calcula-se, obtém-se).
- **No *gerundismo*.** ✗ vamos estar calculando ✓ vamos calcular / calculamos.
  A plain gerund is fine: "Using the chain rule, …" → Usando a regra da
  cadeia, ….
- **At most two subordinate clauses per sentence.** Split English sentences at
  "which", "where", "so that"; TRANSLATING.md §3 allows it inside a block.
- **No more than three *de* in a row.** ✗ a forma da saída da primeira camada
  da rede ✓ a forma que a primeira camada da rede produz.
- ***este* / *esse*.** *Este, esta, isto* for what is here or comes next
  (nesta seção, neste livro, a seguinte equação … esta); *esse, essa, isso* for
  what was just said. ✓ F: "Essa é a *correlação cruzada bidimensional* de …"
  ✗ O: "Esta é a *correlação cruzada bidimensional* …".
- **"where" after a formula is *em que*, never *onde*.** *Onde* is for places.
  ✓ F: "…, em que $\mathbf{Y}$ é a correlação cruzada de $\mathbf{X}$ com
  $\mathbf{K}$" ✗ onde $\mathbf{Y}$ é ….
- ***em vez de*** for "instead of" (✗ ao invés de, which means "contrary to");
  ***por meio de***, *com* or *por* for a means (✗ através de, ✗ via in every
  sentence). ✗ O: "automaticamente via descida de gradiente" ✓ F:
  "automaticamente por descida de gradiente".
- **No anglicisms beyond the loan list in §3.** In particular never dataset,
  feature, layer, loss, learning rate, batch, performance, default, input,
  output, shape as Portuguese words: write conjunto de dados, característica,
  camada, perda, taxa de aprendizado, lote, desempenho, padrão, entrada,
  saída, forma.

## 2. Punctuation and typography

| Item | Rule | Example |
|---|---|---|
| Quotation marks | “…” (U+201C, U+201D); inside them ‘…’. Never «…» (European) and no straight "…" in prose. | 'call cross-correlation "convolution"' → ✓ O: chamamos a correlação cruzada de “convolução” |
| Quotes in attributes | A slide `title="…"` is delimited by ASCII `"`; inside it use “…” only. | `title="O que significa “profundo”?"` |
| Dash | The English em dash `—` becomes a comma, a colon or parentheses; where a dash is really needed, an em dash with a space on each side ( — ), at most once per paragraph. Never an unspaced `—`, never a hyphen with spaces. | "the bias—an offset—shifts the plane" → o viés (um deslocamento) translada o plano |
| Hyphen between paired terms | A hyphen, as in the glossary (par estado-ação, codificador-decodificador), not an en dash. | "input–output pair" → par entrada-saída ✗ F: par entrada–saída |
| Comma | Never between subject and verb; after a fronted adverbial of three words or more; before *mas*, *porém*, *pois*, *portanto*; none before *e* in a list. A comma before *e* is allowed between two clauses with different subjects. | "Adam, RMSProp, and SGD" → Adam, RMSProp e SGD ; Nesta seção, implementamos … |
| Colon | Lower case after a colon in running text. After a bold lead-in (`**Rótulo:**`) the text starts with a capital, as in the English (to confirm). | três grandezas: peso, viés e taxa de aprendizado ; ✓ F: **Diferença finita horizontal:** Varrer a imagem com … |
| Semicolon | Allowed between two main clauses; do not copy every English semicolon, a full stop is usually better. | |
| Abbreviations | In body text write them out: por exemplo, isto é, ou seja. "e.g." → por exemplo; "i.e." → isto é; "vs." → versus (prose) or vs. (slides, kickers, tables); "cf." → cf.; "etc." → etc., after a comma and without *e*; "et al." stays. *p. ex.* and *aprox.* only inside parentheses, captions, slides and tables. | ✓ F: bordas ortogonais a $\mathbf{v}$, isto é, bordas na direção … ✗ i.e., ✗ ex.: |
| Spaces | Type ordinary spaces in `10 000`, `24 GiB`; the build makes them no-break (§10). Do not type no-break or narrow spaces or soft hyphens (TRANSLATING.md §8). No space before `%`, `:`, `;`, `?`, `!`. | ✓ `45%` ✗ `45 %` |
| Apostrophe | Portuguese has no possessive apostrophe and no plural apostrophe. | ✗ a taxa de aprendizado do Adam's ✓ a taxa de aprendizado do Adam ; ✗ GPU's ✓ GPUs |
| Emphasis | Keep `*…*` and `**…**` on the corresponding word(s); add none. Loan words are **not** set in italics (§3.3). | "an *edge map*" → um *mapa de bordas* |
| Headline case | Sentence case: capital on the first word and on proper names only. | "The Model and the Loss" → O modelo e a perda ✗ O Modelo e a Perda |
| Capitals | Lower case for languages, months, days and fields (inglês, novembro, aprendizado profundo). Capital for the numbered word the build prints (Seção 7.2); lower case in prose (nesta seção, neste capítulo). | |

### 2.1 References: article, contraction and crase

The build prints the noun before the number: **Seção 7.2, Figura 7.2.1, Tabela
3.1, Equação (7.2.1)** are feminine, **Capítulo 7** is masculine. The label
tells which one will be printed, so the translator writes the article and the
contraction that Portuguese requires. Reason: *em Seção 7.2* and *de Figura 3*
are not Portuguese; the contraction is obligatory before a numbered noun
(ABNT-style academic prose writes *na Seção 2*, *da Figura 3*).

| Token in the source | Printed | Gender | Write |
|---|---|---|---|
| ``:numref:`sec_…` ``, ``:numref:`subsec_…` `` | Seção 7.2 | f. | a, na, da, à, pela :numref:`sec_x` |
| ``:numref:`fig_…` ``, ``:numref:`img_…` ``, a label with no prefix | Figura 7.2.1 | f. | a, na, da, à, pela :numref:`fig_x` |
| ``:numref:`tab_…` ``, ``:numref:`table_…` `` | Tabela 3.1 | f. | a, na, da, à, pela :numref:`tab_x` |
| ``:eqref:`eq_…` `` | Equação (7.2.1) | f. | a, na, da, à, pela :eqref:`eq_x` |
| ``:numref:`chap_…` `` | Capítulo 7 | m. | o, no, do, ao, pelo :numref:`chap_x` |
| `§{sec_…}` (slides) | §7.2 | — | no article: em §{sec_x}, (§{sec_x}) |
| ``:citet:`…` `` | Luo et al. (2016) | — | no article; plural verb for "et al." names |

| English | Portuguese |
|---|---|
| "In :numref:`sec_why-conv`, we derived …" | ✓ Na :numref:`sec_why-conv`, deduzimos … ✗ F, O, S: Em :numref:`sec_why-conv`, deduzimos … (prints "Em Seção 7.1") |
| "as shown in :numref:`fig_x`, the loss falls" | Como mostra a :numref:`fig_x`, a perda cai. |
| "Just like our MLP weights in :numref:`chap_perceptrons`" | Assim como os pesos do nosso MLP no :numref:`chap_perceptrons` |
| "We implement :eqref:`eq_corr2d` directly" | Implementamos a :eqref:`eq_corr2d` diretamente |
| "The output reproduces :numref:`fig_correlation`" | A saída reproduz a :numref:`fig_correlation` |
| "we return to :numref:`sec_padding`" | voltamos à :numref:`sec_padding` (crase: *a* + *a Seção*) ; voltamos ao :numref:`chap_x` |
| ":numref:`fig_x` counts the paths" (sentence start) | A :numref:`fig_x` conta os caminhos |
| "see :numref:`sec_sgd`", "(:numref:`sec_channels`)" | veja a :numref:`sec_sgd` ; in parentheses no article: (:numref:`sec_channels`), (ver :numref:`sec_sgd`) |
| "in :numref:`sec_a` and :numref:`sec_b`" | na :numref:`sec_a` e na :numref:`sec_b` (repeat preposition and article; ✗ nas :numref:`sec_a` e :numref:`sec_b`, which prints "nas Seção …") |
| ":citet:`Luo.Li.Urtasun.ea.2016` showed that …" | ✓ F: :citet:`Luo.Li.Urtasun.ea.2016` mostraram que … ; possession with *de*: o modelo de :citet:`…` |

- Do not write a noun of your own before a reference (✗ na Seção
  :numref:`sec_x` prints "na Seção Seção 7.2").
- One label breaks the pattern: `chap_classification_generalization` names a
  section page and prints "Seção", so it takes *na*, *da*. If the chapter
  outline in CONTEXT shows that a `chap_` label belongs to a section page,
  use the feminine and name the block in NOTES.
- A reference into a chapter that is not translated yet prints the same noun
  with an "(em inglês)" marker, so the same article applies.
- A citation `:cite:` stands at the end of the clause it supports, before the
  comma or full stop, as in English.

**Crase elsewhere.** *à* = preposition *a* + article *a*; it needs a feminine
noun with an article. Write it in equivariância à translação, em relação à
entrada, igual à soma, devido à curvatura, feito à mão, à medida que. Never
before a masculine noun, a verb, a bare symbol or code, or between repeated
words: em relação a $x$, igual a `Y`, proporcional a $\sqrt{L}$, de $-\Delta$ a
$\Delta$, a partir de, elemento a elemento, de ponta a ponta, passo a passo.
With the noun, the crase returns: proporcional à variância $\sigma^2$.

### 2.2 Math and code in a sentence

- **A bare symbol or code name takes no article and no contraction**: de
  $\mathbf{X}$, em `corr2d`, com $\mathbf{K}$ (✗ do $\mathbf{X}$, ✗ na
  `corr2d`). Give it its noun where gender or number must show: o kernel
  $\mathbf{K}$, a matriz $\mathbf{W}$, a função `corr2d`. Agreement follows
  that noun: $\mathbf{W}$ armazenada (a matriz).
- **A size follows its noun, without *de***: "a $2 \times 2$ kernel" → um
  kernel $2 \times 2$; "a $3 \times 3$ input" → uma entrada $3 \times 3$;
  "a width-$k$ window" → uma janela de largura $k$; "stride-1 layers" →
  camadas de stride 1.
- **Never two math spans side by side.** English "the $4 \times 16$ matrix
  $\mathbf{W}$" has size and name around the noun; Portuguese puts both after
  it. Keep the English order and set the name off with commas: ✓ a matriz
  $4 \times 16$, $\mathbf{W}$, tal que … ✗ F: "Construa a matriz $4 \times 16$
  $\mathbf{W}$ tal que". If that reads badly, reorder (a matriz $\mathbf{W}$,
  de dimensão $4 \times 16$) and name the block in NOTES: the checker warns
  when tokens of one class change order (✓ O did this). Math and code are
  different classes, so "the $3 \times 3$ tensor `X`" → o tensor `X`,
  $3 \times 3$, needs no note.
- **Noun before symbol in compounds**: norma $\ell_2$, vetor de dimensão $d$,
  os $k$ maiores valores; fixed forms keep the hyphen: $n$-grama, top-$k$.
- **Protected math is copied as it is**, even when it contains English: "the
  $i^{\textrm{th}}$ layer" → a $i^{\textrm{th}}$ camada. Do not add *-ésima*
  and do not report it in NOTES (to confirm).
- **A closing `$` must not be followed by a digit** (the checker rejects it):
  keep a word or punctuation between a formula and a number.
- End the sentence before a display equation if the verb would otherwise
  follow it: "… is given by $$…$$ where $n$ is …" → … é dada por $$…$$ em que
  $n$ é ….

## 3. Terminology: Portuguese term or English loan

### 3.1 How to choose

**Use the Portuguese term wherever Brazilian university teaching has a settled
one. Keep the English word where Brazilian practice keeps it. Never coin a
paraphrase to avoid an English word, and never use an English word where a
Portuguese term is settled.** The glossary decides each case; for a term
without an entry, apply the rule, check a Brazilian textbook or course, and
record the choice in DECISIONS.

| Case | Rule | Examples |
|---|---|---|
| Settled Portuguese term | Use it. | camada (✗ layer), peso, viés, perda (✗ loss), característica (✗ feature, ✗ recurso), rótulo, lote, minilote, época, taxa de aprendizado, descida de gradiente (✗ gradiente descendente), retropropagação, sobreajuste (✗ overfitting), subajuste, regularização, ajuste fino, pré-treinamento, preenchimento (padding), camada oculta (✗ escondida), codificador, decodificador, atenção, autoatenção, parada antecipada, decaimento de pesos, conjunto de dados, acurácia, esperança, verossimilhança, política, recompensa, retorno |
| Brazilian practice keeps the English | English word, lower case, with gender (§3.3). | kernel (✗ núcleo), stride (✗ passo), pooling (✗ agrupamento), dropout, embedding (✗ incorporação), token, prompt (✗ instrução), momentum (✗ momento), softmax, logit, transformer, checkpoint, cache, pipeline, patch, framework, array, broadcasting, bootstrap, teacher forcing, score matching, flow matching, roofline, ground truth, underflow |
| Names of models, architectures, optimizers, libraries, datasets, metrics; acronyms; code | Leave in English; never translate a name in part. | ResNet, BERT, Adam, AdamW, PyTorch, ImageNet, BLEU, ReLU, SGD, MLP, LSTM, GPU; Q-learning (✗ aprendizado Q); on-policy, off-policy (✗ fora da política) |
| Name or loan + Portuguese noun | Portuguese noun first, then *de* or the bare modifier, as in the glossary. | camada de pooling, tamanho do kernel, matriz de embeddings, função kernel, bloco transformer, codificação one-hot, roteamento top-k, cache KV, memória da GPU, otimizador Adam |

Reasons for the four loans that have a Portuguese word: *núcleo* is the null
space side of linear algebra, *passo* is a step (tamanho do passo, passo de
tempo), *agrupamento* is clustering, and *momento* is a statistical moment
(primeiro e segundo momentos, in the Adam sections). The earlier Portuguese
edition (pt.d2l.ai, machine-translated and lightly post-edited) left the same
words in English: kernel 415 times, dropout 294, minibatch 262, embedding 220,
stride 199, pooling 164 (`glossary/review/pt-BR.md`). Its other habits are not
a model: it wrote *recurso* for feature and *invariância da tradução* for
translation invariance, and mixed aprendizado/aprendizagem and
overfitting/sobreajuste.

**Acronyms stay English** (GPU, CNN, RNN, LSTM, MLP, SGD, PCA, SVD, MSE, KL,
ELBO, PPO), with the Portuguese expansion at the first mention on a page.
Exception: differential equations take the Portuguese acronyms **EDO, EDE,
EDP** for ODE, SDE, PDE (to confirm). **Never coin a Portuguese acronym.**
Where an English slide abbreviates a phrase that has no acronym of its own,
write it out: "Theoretical RF" → ✓ F: Campo receptivo teórico ✗ O, S: CR
teórico.

### 3.2 Frequent head words

Portuguese terms by frequency in the English book
(`glossary/review/concept-frequency.tsv`), with gender and the forms that are
easy to get wrong:

o modelo · o treinamento · o gradiente · o rótulo · a média · a camada · a
matriz (as matrizes) · o token · o estado · a probabilidade · a variância · o
valor · o peso · a atenção · a política · o vetor · a GPU · a entropia · a
norma · o softmax · o kernel · a acurácia · o crítico · o viés (os vieses) · a
regressão · a sequência · a chave · a característica · o posto · o minilote ·
a previsão · o perceptron multicamadas (MLP) · a distribuição a posteriori · a
verossimilhança · o resíduo · a forma (shape) · a ação · o conjunto de dados ·
o tensor · a recompensa · o prompt · a otimização · o retorno · a amostragem ·
a precisão · o gerador · a descida de gradiente estocástica (SGD) · a derivada
· a revocação · o aprendizado por reforço · o canal · a ativação · a taxa de
aprendizado · a época · a consulta (query) · o pixel · o pooling · a matriz
jacobiana · a convolução · a entropia cruzada · a distribuição a priori · o
codificador · a passagem direta, a passagem reversa · o embedding · a rede
neural recorrente · a esperança · a máscara · o logit · o desvio padrão · o
momentum · o stride · a rede neural convolucional (CNN) · o tamanho do passo ·
o tamanho do lote · o preenchimento · o transformer · a hessiana · o autovalor
· o dropout · a perplexidade · o corpus (os corpora) · o decaimento de pesos ·
o modelo de linguagem · o aquecimento (warmup) · a retropropagação · a rede
neural (✗ rede neuronal) · a função de perda · a função objetivo (✗ objetiva).

Adjectives agree with the head noun, also inside a term: descida de gradiente
estocástica (✗ estocástico), camadas convolucionais, redes neurais profundas,
distribuição a posteriori gaussiana. *A priori* and *a posteriori* are
invariable and take no accent and no italics.

### 3.3 Loan nouns: gender, plural, typography

Loan nouns are written in lower case, **without italics** and without quotes,
and behave as Portuguese nouns. Reason for no italics: `*…*` marks a term being
defined in this book, and a page of the convolution chapter has *kernel* forty
times (to confirm; ABNT sets foreign words in italics).

**Gender:** an English loan is masculine, unless the Portuguese word it stands
for in the reader's mind is feminine. An acronym takes the gender of its
Portuguese head noun. **Plural:** add *-s* (never an apostrophe, never the
European *-eis*).

| Loan | Gender and plural | Note |
|---|---|---|
| kernel, stride, pooling, dropout, embedding, token, prompt, logit, patch, checkpoint, benchmark, framework, array, buffer, pipeline, cache, cluster, bootstrap, momentum, softmax, transformer, autoencoder, notebook, pixel, bit, byte, hardware, software, backbone, baseline | o; os kernels, strides, embeddings, tokens, prompts, logits, patches, checkpoints, pixels | no plural for pooling, dropout, momentum; *o cache* (✗ a cache); *o baseline* (to confirm) |
| GPU, CPU, CNN, RNN, LSTM, GRU, GAN, DQN, ReLU, SVD, PCA, EDO, EDE | a (unidade, rede, função, decomposição, análise, equação); as GPUs, as CNNs | ✓ F: em uma CNN profunda |
| MLP, MSE | o (perceptron, erro); os MLPs | KL only with its noun: a divergência KL |
| SGD, Adam, AdamW, RMSProp, PPO, SAC and other optimizers and algorithms | o (o otimizador, o algoritmo) | ✓ F: O SGD recupera … ✗ O, S: a SGD recupera … (to confirm) |
| Model and dataset names | with the article, masculine unless a *rede*: o BERT, o GPT-2, a ResNet-18, a VGG, o ImageNet, o Fashion-MNIST | os pesos do BERT, treinado no ImageNet ✗ os pesos de BERT |
| Libraries and languages | o PyTorch, o JAX, o TensorFlow, o MXNet, o NumPy | no article after *em* for the language of an implementation: implementado em PyTorch, laços em Python |

A loan noun not listed here: choose by the rule and record it in DECISIONS
with its gender.

### 3.4 Prefixes and hyphens (Acordo Ortográfico de 1990)

| Case | Rule | Examples |
|---|---|---|
| Prefix + word | Closed; double *r* or *s* after a vowel. | autoatenção, autorregressivo, autossupervisionado, multirrótulo, multiescala, hiperparâmetro, subamostragem, sobreajuste, minilote, retropropagação, semidefinida, reamostragem, coocorrência, antissimétrica |
| Hyphen: same vowel, *h*, or *r* after *hiper-, inter-, super-, sub-* | Hyphen. | micro-ondas, anti-horário, inter-relação, sub-rede, super-resolução |
| *pré-, pós-, pró-* (stressed) | Always a hyphen. | pré-treinamento, pré-treinado, pós-treinamento, pós-processamento, pré-ativação |
| *não* + adjective or noun | No hyphen. | não linear, não linearidade, não supervisionado, não nulo ✗ não-linear |
| Two nouns of equal rank | Hyphen. | codificador-decodificador, par estado-ação, ator-crítico, viés-variância, rede-alvo, variável-alvo, caixa-âncora |
| Portuguese noun + English modifier | Open; the English keeps its own hyphen. | codificação one-hot, aprendizado few-shot, métodos off-policy, treinamento de ponta a ponta (end-to-end) |
| *log-* | Hyphen, as in the glossary. | log-verossimilhança, log-probabilidade |

### 3.5 Verbs

Portuguese verbs for actions; an English stem only where the Portuguese verb
in *-izar*/*-ar* is in the dictionaries. Never an English verb with a
Portuguese ending for ordinary actions (✗ printar, setar, startar, deletar,
performar, rodar for "run" in prose).

| English | Portuguese | Not |
|---|---|---|
| train, pretrain | treinar; pré-treinar (um modelo pré-treinado) | treinar previamente |
| fine-tune | fazer o ajuste fino de; ajustar (um modelo ajustado na tarefa) | fine-tunar, afinar |
| sample, draw | amostrar $x$ de $p$; sortear (a single random draw: ✓ F, O: "Sorteie cada elemento …") | samplear |
| overfit, underfit | o modelo sofre sobreajuste / subajuste; ajusta-se demais aos dados | overfitar |
| update, fit | atualizar; ajustar (um modelo aos dados) | fitar |
| pad, pool, clip, mask, crop | preencher (com zeros); aplicar pooling; limitar (a norma do gradiente); mascarar; recortar | padar, clipar |
| flatten, reshape, stack, slice | achatar; mudar a forma de (✗ redimensionar, which is to resize: ✗ S "redimensione `X` e `Y`"); empilhar; fatiar | flatar |
| cache, shuffle, map | armazenar em cache; embaralhar; mapear | cachear |
| tokenize, initialize, normalize, regularize, quantize, parallelize, optimize, encode, decode, backpropagate | tokenizar, inicializar, normalizar, regularizar, quantizar, paralelizar, otimizar, codificar, decodificar, retropropagar | |
| plot | traçar o gráfico de; exibir (to confirm: *plotar*) | printar |
| run, debug, deploy, log, download, return (a function) | executar; depurar; implantar; registrar; baixar; retornar | rodar, debugar, deployar, logar |
| derive (a formula) | deduzir; noun *dedução* (✓ F, S: "deduzimos a camada convolucional") | derivar, which is to differentiate (✗ O: "derivamos a camada convolucional no papel") |

### 3.6 The English term in parentheses

Add the English once per page (one numbered section of the book), at the first
occurrence of a key term in body text, as `(em inglês, …)`, lower case, no
italics, outside the emphasis markers. Check the English page in CONTEXT: if
an earlier unit of the page introduced the term, do not repeat it.

- Gloss Portuguese terms whose English differs and that the reader meets in
  code and papers: *sobreajuste* (em inglês, overfitting), *preenchimento* (em
  inglês, padding), *taxa de aprendizado* (em inglês, learning rate),
  *ajuste fino* (em inglês, fine-tuning).
- Do not gloss loans (kernel, dropout), cognates a reader recognises
  (regularização, gradiente, tensor, convolução), descriptive phrases (✗ O:
  "um *mapa de bordas* (edge map)", "*correlação cruzada bidimensional*
  (two-dimensional cross-correlation)"), or school vocabulary (derivada,
  probabilidade, média).
- With an acronym: *rede neural convolucional* (em inglês, convolutional
  neural network, CNN); afterwards write what the English writes (CNN).
- An English name that abbreviates English words keeps them and adds the
  meaning: im2col (de "image to columns", imagem para colunas).
- Never in headings, captions, slides, exercises or table cells.

✓ "(*preenchimento*; em inglês, padding)" follows the rule when the term sits
inside a parenthesis already. ✗ F: "(*preenchimento*, do inglês padding)" (*do
inglês* gives an etymology) ✗ S: "(*preenchimento*, ou *padding*)" (adds
emphasis and a second term).

## 4. Numbers and units

**Decision: the decimal point everywhere, the space as thousands separator**
(to confirm; it departs from the Brazilian decimal comma of ABNT and Inmetro in
prose). Numerals are copied from the English (0.5, 92.3%), in prose as in math,
code, code output and figures. Reason: inline and display math are protected
and keep the point, and outputs and figure axes show it; a decimal comma in
prose would put two conventions into one sentence (✗ com $\eta = 0.03$ a perda
cai abaixo de 0,01) and would need a rule for every identifier (GPT-3.5,
Python 3.12). The German edition has the same ruling, and all three test
translators chose it independently. With the point as decimal sign, **a point
must never group thousands** (✗ 10.000 imagens).

What the checker does (`tools/l10n/protect.py`): it compares the numerals of
each block with the English as strings of digits with `.` and `,`. So `0,01`
for `0.01` gives the warning "numbers not found: 0.01"; with
`check: {thousands: space}` (§10) the English `10,000` and the Portuguese
`10 000` or `1000` match; `10.000` and "10 mil" do not. A digit turned into a
word ("2×" → "duas vezes") also warns. Extra numerals in the translation and
numerals inside math or code are not compared.

| Item | Rule | Example |
|---|---|---|
| Decimal | As in the source, with a point. | "drops below 0.01" → ✓ F: cai abaixo de 0.01 |
| Thousands | A space from five digits on; none in four-digit numbers and years. Never `10,000` or `10.000`. | "10,000 images" → 10 000 imagens ; 2048 ; 60 000 |
| Number words | million → milhão (milhões), billion → bilhão (bilhões), trillion → trilhão; with *de* before a noun. Keep digits as digits (✗ 10 mil for 10,000). | "3 billion parameters" → 3 bilhões de parâmetros ✗ 3 mil milhões |
| Digits or words | As the English: a number spelled out in English is spelled out in Portuguese (with gender: duas camadas, um canal); digits with units, symbols and comparisons. | "two hidden layers" → duas camadas ocultas ; "within nine steps" → em nove passos ; 10 épocas |
| Percent | No space before `%`, as in the source. *por cento* only for a spelled-out number. | "45% fewer parameters" → ✓ F: 45% menos parâmetros |
| Units | Symbols as in the source, with a space; time words written out become segundos, minutos, horas, dias. | 24 GiB, 5 ms ; "3 hours" → 3 horas |
| Ranges | En dash without spaces, unit once; after *de* write *a*, after *entre* write *e*. | "40–55" → 40–55 ; ✗ de 40–55 ✓ de 40 a 55 ; entre 40 e 55 |
| Multipliers | Keep the digit: *N vezes* + comparative in prose; keep `×` in slides, tables, captions and math. | "about 2× faster" → cerca de 2 vezes mais rápido ; slide: 2× mais rápido |
| Approximation | cerca de, aproximadamente, por volta de; "an order of magnitude" → uma ordem de grandeza. | "roughly 90%" → cerca de 90% |
| Ordinals | Written out up to *décimo*, agreeing in gender; otherwise digit + º/ª. | "the third epoch" → a terceira época ; a 12ª camada |
| Decades, years | "the 2010s" → a década de 2010; "in 2017" → em 2017. | A ResNet surgiu em 2015. |
| Money | Currency sign, space, amount, in the source's currency. | "\$2.50" → US\$ 2.50 |

Quantities, signs and orders of magnitude stay exact; do not round or convert
units or currencies.

## 5. Gender-inclusive language

**One rule: neutral wording where it is natural, the plain grammatical form
otherwise, and no invented endings or signs.** No *x*, *@* or *-e* forms and
no slash or parenthesis doublets (✗ todxs, tod@s, todes, o(a) leitor(a),
pesquisadores/as). Reason: these forms are outside the VOLP and the official
orthography, and `@`, `*` and `/` collide with the book's markup.

- People in general: a collective or a relative clause. "the practitioner …
  she" → quem treina modelos … ; a equipe, as pessoas, a comunidade.
- "the reader", "the user": *você* (§1.2).
- Technical roles are not people and keep their grammatical gender: o agente,
  o ator, o crítico, o gerador, o discriminador, o professor and o aluno
  (distillation), o otimizador.
- Named people keep the pronoun of the source.

## 6. Names

- **Models, libraries, datasets, companies, hardware, benchmarks:** exactly as
  in English: Fashion-MNIST, RTX 4090, OpenAI, GPT-2.
- **People:** as in the English source, without title, including
  transliterated names (Markov, Kolmogorov, Chebyshev, Lyapunov). Reason:
  citations, searches and code use these forms (✗ Tchebychev, Liapunov).
- **Terms named after people:** noun + *de* + name, the name unchanged and
  capitalised: cadeia de Markov, teorema de Bayes, método de Newton, equação
  de Bellman, expansão de Taylor, multiplicador de Lagrange, constante de
  Lipschitz, distância de Wasserstein, desigualdade de Cauchy-Schwarz,
  divergência de Kullback-Leibler, estimativa de Monte Carlo.
- **Adjectives from names:** lower case, agreeing: distribuição gaussiana,
  ruído gaussiano, inferência bayesiana, norma euclidiana, movimento
  browniano, a hessiana, a matriz jacobiana, o lagrangiano, gradiente
  lipschitziano. Follow the English word: "Gaussian" → gaussiana, "normal" →
  normal ("standard normal distribution" → distribuição normal padrão).
- **The book:** *Dive into Deep Learning* stays in English; "this book" →
  este livro.
- **Titles of English courses, assignments, papers and books** stay in
  English, also as link text: [Assignment 2](…), *Neural Networks and Deep
  Learning*; "Chapter 6" of such a book → Capítulo 6.
- **Languages and places:** Portuguese (inglês, os Estados Unidos);
  programming languages stay (Python).
- **Structure words:** chapter capítulo, section seção, subsection subseção,
  appendix apêndice, Summary → Resumo, Exercises → Exercícios, Discussions →
  Discussões; Theorem → Teorema, Proposition → Proposição, Lemma → Lema,
  Corollary → Corolário, Definition → Definição, Proof → Demonstração, Remark
  → Observação, Example → Exemplo.

## 7. Headings, captions, slides, exercises, notes

| Kind | Rule | Example |
|---|---|---|
| `heading` | Noun phrase in sentence case, no full stop, no English gloss. English "-ing + object" becomes a noun + *de*, never a gerund. A question stays a question. | "Learning a Kernel" → ✓ F: Aprendizado de um kernel ✗ O, S: Aprendendo um kernel ; "Training the Model" → Treinamento do modelo ; "Why Depth Helps" → Por que a profundidade ajuda |
| `caption` | Body register without *nós*/*você*; a full stop exactly when the English caption has one. Keep `(a)`, `(b)`. | ✓ F: Correlação cruzada bidimensional. As partes sombreadas são … |
| `slide_title`, `kicker` | Noun phrase, as short as the English, no full stop; "&" → *e*; "vs." stays. | `title="Detecting Local Edges Across an Image"` → ✓ F: `title="Detecção de bordas locais em uma imagem"` ✗ O, S: `title="Detectando bordas locais em uma imagem"` |
| `slide_text` | Short statements or noun phrases; no *nós*; instructions in the *você* imperative; one line where the English is one line; no full stop after a single-phrase bullet. Bold lead-ins keep their bold and colon. No coined abbreviations (§3.1). | "Reshape `X` and `Y` to 4D tensors" → ✓ F: Mude a forma de `X` e `Y` para tensores 4D ; "Lower the learning rate gradually" → Reduza a taxa de aprendizado aos poucos |
| `exercise` | *Você* imperative (Mostre, Deduza, Implemente, Compare); direct questions, with *você* only where the English asks "you". Keep `[code]`, ● and `:exlabel:`. "Adapted from …" → Adaptado de …. | "Show that the errors sum to zero. Why?" → Mostre que a soma dos erros é zero. Por quê? ; ✓ F: Como você projetaria um kernel de desfoque? |
| `tab_note` | Body register. Describe this framework only; never compare frameworks. API names stay in code. | "PyTorch's `nn.LazyLinear(1)` requires only the output dimension." → No PyTorch, `nn.LazyLinear(1)` exige apenas a dimensão de saída. |
| `note` | Register of its surroundings. Labels: **Note:** → **Nota:**, **Warning:** → **Atenção:**, **Tip:** → **Dica:**, **Answers:** → **Respostas:**; "Concept Check" → Verificação de conceitos; "Key Takeaways" → Principais conclusões. | |
| `table_row` | Noun phrases; numbers and units as in §4. | "Training time" → Tempo de treinamento |

*Por que* (question, two words), *porque* (answer), *por quê* (before a full
stop or question mark), *o porquê* (noun): ✓ F: "Por que a perda inicial é 12
…".

The cover kicker `[Dive into Deep Learning · §2.1]{.kicker}` is rewritten by
the build; copy it unchanged.

## 8. Known pitfalls

### 8.1 Words with more than one sense

| English | Sense → Portuguese | Not |
|---|---|---|
| bias | parameter $b$ → o viés; of an estimator → o viés, *unbiased* → não viesado; social → viés, preconceito | bias, vício, tendência, polarização |
| kernel | convolution, kernel methods, GPU or OS code → o kernel (tamanho do kernel, função kernel, matriz de kernel); of a linear map → o núcleo, null space → espaço nulo | núcleo de convolução |
| return | RL → o retorno; of a function → retornar, valor de retorno | devolução, rendimento |
| score | $\nabla \log p$ → o escore (função escore, rede de escore); attention, a metric → a pontuação (pontuação de atenção, pontuação BLEU); names stay (score matching) | placar, partitura |
| sample | → a amostra (média amostral, amostra aleatória; also a generated output: qualidade das amostras); a data point → o exemplo; to sample → amostrar | exemplar, espécime |
| feature | → a característica (vetor de características, mapa de características); of a library or product → o recurso, a funcionalidade | recurso for a model feature; feature |
| edge | in an image → a borda (detector de bordas, mapa de bordas); the border of an image or figure → a margem; of a graph → a aresta; between regions → a fronteira | beira, canto (= corner) |
| entry | of a matrix, tensor or output → o elemento | entrada, which is "input" (✗ S: "todas as 42 entradas da saída") |
| shape | → a forma (✓ F: "Qual é a forma espacial da saída"); reshape → mudar a forma de | formato (✗ S), shape, redimensionar |
| built-in | → embutido(a), as in Python's Brazilian documentation (funções embutidas, a camada embutida do framework) | interno, nativo (to confirm; F wrote *nativa*) |
| patch | → o patch (matriz de patches, patch de imagem, as in the glossary) | recorte (F; it is "crop" in the glossary: recorte aleatório), trecho (S) |
| policy | RL → a política; English inside fixed names (on-policy, off-policy) | diretriz, apólice |
| state | → o estado (estado oculto, espaço de estados); "state of the art" → estado da arte | status |
| loss, cost, objective, error, risk | Five words, five terms: perda, custo, objetivo (função objetivo), erro, risco. | merging them; função objetiva |
| accuracy, precision, recall | acurácia, precisão, revocação; floating-point *precision* → precisão (precisão dupla, meia precisão, precisão mista) | exatidão; precisão for accuracy; "Recall that" is not the metric |
| normalization | → normalização (normalização em lote, por camada); *standardization* → padronização; *regularization* → regularização; normal distribution → distribuição normal | mixing them |
| training, learning | train → treinar, o treinamento; learn → aprender, o aprendizado | treino, aprendizagem (pt-PT) |
| network, net | → a rede (rede neural, rede residual, rede-alvo) | rede neuronal |
| step, step size, stride | o passo (passo de gradiente, passo de tempo); o tamanho do passo; o stride (convolution: convolução com stride 2) | passo for stride |
| mean, average | a média; to average → calcular a média de; "on average" → em média | |
| expectation | → a esperança; "expected value" → valor esperado; "in expectation" → em esperança | expectativa (the old edition's "Na expectativa") |
| likelihood, evidence | a verossimilhança; Bayesian → a evidência; "empirical evidence" → evidências empíricas, indícios | probabilidade for likelihood |
| translation | of an image or function → a translação (equivariância à translação, invariância à translação); of language → a tradução | invariância da tradução |
| map, mapping | feature map → mapa; a function → mapeamento, função; linear map → transformação linear | aplicação (pt-PT) |
| inference | running a trained model → a inferência; statistics → inferência (inferência bayesiana); "the layer infers its input size" → determina automaticamente | |
| gradient, derivative | o gradiente; a derivada; "with respect to $x$" → em relação a $x$ | derivativo (finance) |
| rank, trace, norm | o posto, o traço, a norma; ranking → o ranqueamento | rank, classificação for matrix rank |
| exploration, exploitation | a exploração; a explotação (to confirm; both are "exploração" in everyday Portuguese) | exploração for both |
| bound | limite superior, limite inferior (as in the glossary); bounded → limitado | cota in one place and limite in another |
| trade-off | → o compromisso (compromisso entre viés e variância); "at the cost of" → ao custo de (to confirm) | trade-off, troca |
| forward / backward | forward pass → passagem direta; backward pass → passagem reversa; forward propagation → propagação direta | passagem para frente |
| performance | of a model → o desempenho; of code → desempenho, velocidade | performance |
| default, standard | → padrão (valor padrão, desvio padrão: invariable, no hyphen) | default |
| label | → o rótulo; of an axis → rótulo do eixo | etiqueta, label |
| converge | converge **para** $x^*$ | converge a |
| library | → a biblioteca | livraria (= bookshop) |
| fires, spikes (a detector) | vale `+1`, responde com; dispara for "spikes" | atira |

### 8.2 False friends

| English | Portuguese | Not (because) |
|---|---|---|
| eventually | por fim, com o tempo | eventualmente (= occasionally) |
| actual, actually | real; na verdade, de fato | atual, atualmente (= current) |
| assume | supor | assumir (= to take on) |
| realize | perceber, notar | realizar (= to carry out) |
| support (a feature) | aceitar, oferecer suporte a | suportar (= to endure) |
| address (a problem) | tratar, resolver | endereçar |
| resume | retomar | resumir (= to summarize) |
| consistent | coerente, uniforme; *consistent with* → compatível com; only *consistent estimator* → estimador consistente | consistente for "uniform" |
| sensible | sensato, razoável | sensível (= sensitive) |
| large | grande | largo (= wide) |
| comprehensive | abrangente | compreensivo (= understanding) |
| novel | novo, inédito | novela |
| significant | considerável, expressivo; *significativo* only in the statistical sense | significante |
| randomly | aleatoriamente, ao acaso | randomicamente |
| billion, trillion | bilhão, trilhão | bilião, mil milhões (pt-PT) |
| technically | a rigor | tecnicamente |
| dramatically | drasticamente, muito | dramaticamente |
| leverage | aproveitar | alavancar |

### 8.3 Translationese

| Pattern | Write instead | Example |
|---|---|---|
| "It is important to note that", "Note that" word for word | Drop it, or Observe que … at most once per page | ✗ É importante notar que o viés não é regularizado. ✓ O viés não é regularizado. |
| "Recall that", "Recall from …" | Como vimos na …, ; Lembre-se de que … (*lembrar-se de*) | ✗ S: "Lembre de :numref:`subsec_why-conv-channels` que a saída …" ✓ Como vimos na :numref:`subsec_why-conv-channels`, a saída … |
| Subject pronoun in every sentence | The verb ending | ✗ Nós calculamos … Nós atualizamos … ✓ Calculamos … Em seguida, atualizamos … |
| Possessive for every "its", "their" | The article, or *dele/dela* where *seu* is ambiguous | "a unit's receptive field" → o campo receptivo de uma unidade ; "its forward pass" → a passagem direta |
| Passive with *por* | Active voice | ✗ O gradiente é calculado pela retropropagação. ✓ A retropropagação calcula o gradiente. |
| Inanimate subject + "tells us", "allows us to" | Daí concluímos, Isso mostra que, Com isso podemos | "This result tells us that depth helps." ✗ Este resultado nos diz que … ✓ Isso mostra que a profundidade ajuda. |
| "we can" every time as *podemos* | The verb itself | "we can compute" → calculamos |
| "there is / there are" as *tem* | *há*, *existem*, or a full verb | "There are three reasons." → Há três razões. |
| "based on" as *baseado em* (invariable) | com base em, a partir de; or agree: um modelo baseado em atenção, uma rede baseada em | ✗ Baseado nisso, definimos … ✓ Com base nisso, definimos … |
| *o mesmo* as a pronoun, *sendo que*, *a nível de*, *onde* for "in which" | A pronoun, *e*, *em*, *em que* | ✗ aplicamos o kernel e somamos o mesmo ✓ … e o somamos |
| "in terms of" | em função de, quanto a | ✓ F: "Deduza $\partial \ell/\partial \mathbf{K}$ … em função de $\partial \ell/\partial \mathbf{Y}$" |
| "simply", "just", "basically" | Drop, or *apenas*, *basta* | ✗ simplesmente chamamos ✓ basta chamar |
| Idioms translated by image | The meaning | "under the hood" → internamente ; "from scratch" → do zero ; "out of the box" → sem configuração adicional ; "rule of thumb" → regra prática ; "end to end" → de ponta a ponta ; "hand-crafted" → feito à mão ; "sure enough" → como esperado |
| Title Case, English plural or possessive | Portuguese forms | ✗ A Função de Perda ✗ os parameters ✗ Adam's |

Logical and mathematical phrases are fixed: because → porque, pois; therefore
→ portanto, logo; however → no entanto, porém; only if → somente se; if and
only if → se, e somente se,; at most → no máximo; at least → pelo menos; "let
$x$ be …" → Seja $x$ … (Sejam for several); such that → tal que; with respect
to → em relação a; without loss of generality → sem perda de generalidade; for
all → para todo; there exists → existe; respectively → respectivamente; almost
surely → quase certamente; i.i.d. → independentes e identicamente distribuídas
(then i.i.d.); bounded → limitado; smooth → suave; dot product → produto
escalar; inner product → produto interno; "up to sign" → a menos do sinal;
"up to a constant" → a menos de uma constante; "it follows that" → segue que;
"that is" → isto é.

## 9. Sources

Read for this guide (2026-11-26):

- `TRANSLATING.md`, `DESIGN.md`, `tools/l10n/protect.py` (number and token
  checks, `nbsp`), `tools/l10n/config.py`, `languages.yml`, `de/style.md`,
  `ja/style.md`.
- `glossary/terms/pt-BR.jsonl`, `glossary/notes/es-fr-pt.md`,
  `glossary/review/pt-BR.md` and `pt-BR-corrections.jsonl` (the review records
  what the earlier Portuguese edition, `d2l-ai/d2l-pt`, attests: machine
  translation, the counts quoted in §3.1, its errors), and
  `glossary/review/concept-frequency.tsv`.
- `evals/pt-conv-layer/`: RESULTS.md and the answers of Fable, Opus and
  Sonnet with their DECISIONS and NOTES.
- The English build: `tools/d2l_preprocess.py` (label prefixes →
  `sec-`/`fig-`/`tbl-`/`eq-`), `tools/fix_crossref_numbers.py` and
  `tools/i18n.py` (which word is printed), and Quarto's
  `_language-pt-BR.yml` (Seção, Figura, Tabela, Equação; no chapter word).
  In the English book all 45 chapter index pages carry a `chap_` label and no
  other; one `chap_` label sits on a section page (§2.1).

Not checked (stated from the drafter's knowledge of Brazilian usage):

- The Acordo Ortográfico rules in §3.4 and the VOLP entries were not looked up
  online; neither were ABNT NBR 6023/14724 or Inmetro's SI rules (decimal
  comma, space as thousands separator, italics for foreign words).
- The Python documentation in pt-BR ("funções embutidas") and the Brazilian
  editions of Géron, Goodfellow et al. and Faceli et al. were not opened.
- pt.d2l.ai itself was not read; everything said about it comes from the
  glossary review.
- No native speaker has read this guide or the test translations
  (RESULTS.md: "Verdict: pending a native reader").

## 10. Settings

Existing keys (`tools/l10n/config.py`, `protect.py`); to be set in the `pt`
entry of `languages.yml`:

- `check: {thousands: space}`: accept `10,000` → `10 000` and `1,000` → `1000`
  (§4). Without it every regrouped number warns.
- `typography: {nbsp: true}`: the build makes the spaces in `10 000` and
  `24 GiB` no-break. Its abbreviation rule was written for German (`z. B.`); it
  only touches single letters followed by a full stop, which Portuguese prose
  does not use, so it is harmless.
- `dir: ltr`, `locale: pt-BR`, `forum: english`: as they are.

In `pt/strings.yml` (generated by `l10n strings pt`, existing keys):

- `lang: "pt-BR"` and `pdf.lang: "pt-BR"` (babel *brazilian* hyphenation).
- `quarto.crossref-ch-prefix: "Capítulo"`: Quarto's `_language-pt-BR.yml` has
  no chapter prefix (its `_language-pt.yml` has), and `tools/i18n.py` reads
  only the first file it finds, so without the override chapter references
  print "Chapter 7".
- `book.exercises_heading: "Exercícios"`, `book.discussions_link:
  "Discussões"` (§6), so the build recognises the translated headings.

Needs new code:

- A check that the article before a reference agrees with the noun the
  build will print (masculine only before a chapter; `em`/`de` + reference
  flagged), or the printed noun shown next to each reference in the work
  file. §2.1 is otherwise enforced by reviewers only.
- A warning for a decimal comma in prose (`\d,\d` where the English has
  `\d.\d`) with a message that names §4; today it appears only as "numbers
  not found".
- If the decimal comma is chosen after all (§11, question 1): a
  `check: {decimal: comma}` option that maps `0.5` ↔ `0,5` outside math and
  code, and a rule for identifiers (GPT-3.5).

## 11. For the author to confirm

A native reviewer should settle these; the proposed answer is the rule in the
section named.

1. **Decimal sign.** Point everywhere, as in the German edition, against the
   Brazilian decimal comma (§4)? Proposed: point.
2. **Thousands.** A space from five digits on (10 000; SI and Inmetro allow
   it), never the Brazilian point (10.000), because the point is the decimal
   sign here (§4)? Proposed: yes.
3. **References take the article and contraction** (na :numref:`sec_x`, no
   :numref:`chap_x`, à :numref:`fig_x`), chosen from the label prefix (§2.1).
   This differs from the German rule (no article) and from what all three
   test translators wrote ("Em :numref:`…`"). Proposed: yes; it depends on
   `quarto.crossref-ch-prefix` being set and on the build keeping the nouns
   Seção, Figura, Tabela, Equação.
4. **Loan words without italics** (§3.3), against ABNT practice? Proposed:
   no italics.
5. **Gender of SGD and other optimizers:** o SGD, o Adam (the optimizer), or
   a SGD (a descida) as Opus and Sonnet wrote (§3.3)? Proposed: o SGD.
6. **EDO, EDE, EDP** for ODE, SDE, PDE, as the only translated acronyms
   (§3.1)? Proposed: yes (the glossary review recommends it).
7. **patch** stays English (o patch, matriz de patches) as in the glossary,
   not *recorte* or *trecho* (§8.1)? Proposed: patch.
8. **built-in** → embutido (Python's Brazilian documentation) or nativo
   (§8.1)? Proposed: embutido.
9. **plot** → traçar o gráfico de, or the common loan verb *plotar* (§3.5)?
   Proposed: traçar o gráfico de.
10. **trade-off** → compromisso, or keep the loan (§8.1)? Proposed:
    compromisso.
11. **exploitation** → explotação, to keep it apart from exploração
    (exploration) (§8.1)? Proposed: explotação; the alternative is
    *aproveitamento*.
12. **The English gloss** as `(em inglês, …)`, once per page, only for
    Portuguese terms whose English differs (§3.6)? Proposed: yes.
13. **Capital after a bold lead-in** on slides (`**Rótulo:** Texto …`), lower
    case after every other colon (§2)? Proposed: yes.
14. **Math that contains English** (`$i^{\textrm{th}}$`): copied unchanged,
    as ruled for German (§2.2)? Proposed: yes.
15. **Headings as noun phrases** (Treinamento do modelo), never the gerund
    (Treinando o modelo) that many Brazilian translations use (§7)? Proposed:
    noun phrases.
16. **Glossary choices the review left open** and this guide follows as they
    stand: previsão (not predição) for prediction, minilote (not minibatch),
    viés for both senses of bias, limite for bound, bandido de múltiplos
    braços, o baseline. Proposed: keep all, pending a native reviewer with
    the Brazilian textbooks named in §9.

## 12. Proposed glossary corrections

The glossary agrees with this guide except for these points:

| Concept | Glossary now | Proposed | Reason |
|---|---|---|---|
| `strided-convolution` | convolução com passo | convolução com stride | `stride` is *stride*; *passo* is a step (§8.1) |
| `ordinary-differential-equation`, `stochastic-differential-equation` | full forms only | add the acronyms EDO, EDE as `alternatives` | §3.1, if question 6 is confirmed |
| `ode-solver`, `neural-ode`, `probability-flow-ode`, `reverse-time-sde` | solucionador de equações diferenciais ordinárias; equação diferencial ordinária neural; … | solucionador de EDOs; EDO neural; EDO de fluxo de probabilidade; EDE em tempo reverso | The written-out forms are too long to repeat in running text |
| `exploitation` | no entry | explotação | Collides with `exploration` = exploração (§8.1; review, decision 9) |
| `patch` | no entry (only `image-patch` = patch de imagem) | patch (m.; os patches) | The three test translators chose three different words |
| `trade-off` / `bias-variance-tradeoff` | no entry | compromisso; compromisso entre viés e variância | §8.1, if question 10 is confirmed |
| `deep-learning`, `machine-learning` | no entry | aprendizado profundo; aprendizado de máquina | Used on every page; all three translators chose them |
| `built-in`, `entry` (of a matrix), `edge` (image) | no entry | embutido; elemento; borda | Recurring words with a wrong obvious translation (§8.1) |
| `soft-actor-critic`, `proximal-policy-optimization`, `deep-q-network` | ator-crítico suave; otimização proximal de política; rede Q profunda | keep as the first-mention gloss; preferred term SAC, PPO, DQN | Acronyms stay English (§3.1; review, decision 6) |
