# Style guide: Korean (ko)

Draft of 2026-11-26 for the author's review. Translators read this with every
unit (TRANSLATING.md §7–§8). Each rule has one reason and one example
(English → Korean, or ✗ wrong / ✓ right). Rules marked "(to confirm)" and the
other points that need the author's or a native reviewer's ruling are collected
in §11. Examples marked *fable*, *opus*, *sonnet* are sentences from the
translation test `evals/ko-conv-layer/`.

Order of authority: the decisions log, then this guide, then the glossary. The
glossary says *what* a term is; this guide says *how to choose, spell, space and
attach particles to* terms. Where the two differ, §12 lists the entry.

## 1. Register

**Decision: 합니다체 (formal polite, sentences ending in -ㅂ니다/-습니다) for
all running text; no subject for the book's "we" and "you".** Reason: it is the
register of the Korean introductions this book sits beside (the translated
*밑바닥부터 시작하는 딥러닝* and *핸즈온 머신러닝*) and of the earlier ko.d2l.ai,
and it carries the book's collegial tone. 한다체 (-다) is the register of
graduate monographs and papers; 해요체 (-해요, -이에요) is the register of blogs.
All three test translators chose 합니다체 independently.

| Where | Rule | Example |
|---|---|---|
| Body text, list items, captions that are sentences, tab notes, notes in the body | 합니다체 throughout | "Linear regression is a single artificial neuron." → 선형 회귀는 인공 뉴런 하나입니다. |
| Headings, kickers, slide titles, table cells | Noun phrases (§8) | "Training the Model" → 모델 훈련 |
| Slide text | 한다체 or a noun ending; never 합니다체 (§8) | "Lower the learning rate gradually" → 학습률을 점차 낮춘다 |
| Exercises | -하세요 for instructions, -ㅂ니까? for questions (§8) | "Show that …" → …임을 보이세요. |

**"We" and "you".** Korean drops the subject. Do not write 우리는, 우리의 or
당신은 for the book's "we" and "you".

| English | Korean | Example |
|---|---|---|
| we + action in the text | no subject, -ㅂ니다 | "We minimize the loss with SGD." → 손실은 SGD로 최소화합니다. |
| "let us", "let's", "consider" | -ㅂ시다, -아/어 봅시다 (only where the English invites) | "Let us look at the gradient." → 기울기를 살펴봅시다. |
| "in this section/chapter/book, we …" | 이 절에서는 / 이 장에서는 / 이 책에서는 + verb | "In this section, we formulate the model." → 이 절에서는 모델을 정식화합니다. |
| "we will see", "we will use" | present tense; -겠습니다 only where the authors announce a decision | "we will call it convolution" → 합성곱이라고 부르겠습니다 ; "We will return to this in …" → 이 문제는 …에서 다시 다룹니다. |
| "our model", "our dataset" | 이 / 앞의, or nothing | "our model overfits" → 이 모델은 과적합됩니다 |
| "you can", "suppose you want" | no subject; -ㄹ 수 있습니다, -다고 합시다 | "Suppose you want to estimate the price of a house." → 집값을 추정하려 한다고 합시다. |
| "we recommend", "we believe" | 이 책에서는 … | "We recommend AdamW." → 이 책에서는 AdamW를 권장합니다. |
| "we" = people, the field | 우리, 사람, 연구자들 (here it is real content) | "We humans attend selectively." → 우리 인간은 주의를 선택적으로 기울입니다. |
| a question put to the reader in the body | -ㄹ까요? | 경사 하강법은 커널 `[1, -1]`을 입출력 쌍 `(X, Y)`만으로 복원할 수 있을까요? (*fable*) |

✗ …같은 계열의 층을 학습하므로, 우리는 딥러닝의 표준 용어를 따라 교차 상관을 "합성곱"이라고 부릅니다 (*sonnet*)
✓ …같은 층의 모임을 학습하므로, 딥러닝의 표준 용어를 따라 교차 상관을 ‘합성곱’이라고 부르겠습니다 (*fable*, with the quotation marks of §4)

**Sentence shape** (each point is checkable):

- **One register per block.** No 해요체 and no 한다체 sentence ending inside
  body text; no 합니다체 inside a slide. Vary the endings (-ㅂ니다, -입니다,
  -ㅂ시다, -기 때문입니다, -ㄹ까요?) so that a paragraph is not a run of -합니다.
- **Reader-directed imperatives in the body are -ㅂ시다 or are dropped;**
  -하세요 is for exercises and for real instructions (install, run).
  ✗ 출력이 조금씩 줄어든다는 점에 주목하세요 (*opus*, *sonnet*)
  ✓ 출력이 … 조금씩 줄어든다는 점에 주목합시다 (*fable*)
- **Active, with the object as topic.** Passive only where the agent does not
  matter (모델은 ImageNet으로 사전 훈련되었습니다). No double passive
  (✗ 계산되어집니다, ✗ 보여집니다) and no -에 의해 -되다 where an active verb
  exists. ✗ 손실이 계산되고 그 후 기울기가 역방향으로 전파됩니다. ✓ 손실을
  계산한 다음 기울기를 역전파합니다.
- **At most two 의 in one noun phrase**, and at most one -에 대한 / -를 위한 /
  -에 의한 per sentence. Use a verb clause instead (§9.3).
- **No pronoun chains.** 그것, 이것들, 그들 for models, layers or data: repeat
  the noun or drop the pronoun.
- **No plural -들 after a numeral or a quantity word, or on inanimate nouns
  whose number is clear.** ✗ 세 개의 층들, ✗ 여러 파라미터들 ✓ 세 층, 여러
  파라미터.
- **At most two connective endings (-고, -며, -지만, -므로, -는데) in a row.**
  Split English sentences at "which", "where", "so that"; TRANSLATING.md §3
  allows splitting within a block.
- **No 및 in body text**; write 와/과 or 그리고. 및 is for headings, slides and
  table cells. ✗ 그 계산에 쓰인 입력 및 커널 원소입니다 (*opus*) ✓ 입력 원소와
  커널 원소입니다.

Spelling: 한글 맞춤법 and 표준어 규정. 사이시옷 as the glossary has it: 고윳값,
특잇값, 최솟값, 최댓값, 기댓값, 함숫값, 꼭짓점, 자릿수.

## 2. Word spacing (띄어쓰기)

Korean has word spaces, and most errors in machine-translated Korean are
spacing errors. The rules below fix every case that 한글 맞춤법 leaves open.

### 2.1 Hangul text

| Case | Rule | Example |
|---|---|---|
| Particles (조사) | Attached to the word before them, always (맞춤법 제41항). | 손실을, 층에서는, 커널보다, 모델만으로도 |
| Dependent nouns (의존 명사) | Spaced: 수, 것, 때, 데, 뿐, 만큼 (after a verb form), 등. | 계산할 수 있습니다 ; 이동만 다를 뿐입니다 ✗ 계산할수 있습니다 |
| Auxiliary verbs | Spaced, except the fixed forms -아/어지다, -아/어하다. | 살펴봅니다 (one word) ; 시험해 봅니다 ; 제쳐 둡니다 ; 떨어집니다 |
| Technical terms of several words | Spaced, word by word (맞춤법 제50항, the principle; the permitted solid form is not used). The glossary's spelling of a term is binding **including its spaces**: the glossary check looks for the exact string. | 손실 함수, 경사 하강법, 교차 엔트로피, 활성화 함수, 확률 변수, 완전 연결층 ✗ 손실함수, 경사하강법 |
| One-syllable suffix nouns | Attached to the word before: -층, -법, -률, -값, -점, -식, -화, -성, -기, -형, -량. | 합성곱층, 은닉층, 풀링층, 정규화층, 학습률, 최솟값, 안장점, 정칙화, 볼록성, 분류기 |
| Sentences | One space after `.`, `?`, `!`, `,` and `:`; none before. | ✗ 계산합니다.이제 … (the earlier ko.d2l.ai) ✓ 계산합니다. 이제 … |

The suffix rule and the glossary: 합성곱층, 은닉층 and 풀링층 are closed, and
so is every new compound in -층. A term with a two-syllable head noun stays
open (합성곱 신경망, 잔차 블록, 특징 맵, 패치 행렬).

### 2.2 Latin words, digits, code and counters

| Next to Hangul | Rule | Example |
|---|---|---|
| Latin word or acronym + particle | Attached. | CNN은, GPU를, Adam으로, PyTorch의, Fashion-MNIST에서 |
| Latin word or acronym + Korean noun | Spaced. | Adam 옵티마이저, ReLU 함수, GPU 메모리, KV 캐시, Tensor Core 속도 |
| Inline code + particle | Attached. Code + noun: spaced. | `corr2d`는, `X`와, `[1, -1]`을 ; `Conv2D` 층 ✗ `corr2d` 를 (*sonnet*) |
| Digits + particle | Attached. | 0이 됩니다, 0으로 초기화합니다, 12 근처 (근처 is a noun: spaced) |
| Digits + counter or Korean unit word | Attached (맞춤법 제43항 permits it; one rule). | 5개, 10번, 3층, 2차원, 4차 텐서, 10에포크, 9스텝, 2배, 6장, 2015년, 2010년대 |
| Digits + unit symbol in Latin letters | As in the source, with its space; `%` attached. | 24 GiB, 5 ms, 92.3% |
| Native numeral word + counter | Spaced. | 두 개, 세 번, 네 곳, 한 번에 한 픽셀씩 |
| Parenthesis | The opening `(` is attached to the word before it; the particle follows the closing `)` directly. | 수용 영역(receptive field)이 ; 합성곱층(편향은 생략합니다)으로 ✗ 분산 (Variance) (the earlier ko.d2l.ai) |
| Emphasis markers | `*…*` and `**…**` enclose the Korean word without its particle; nothing changes in the spacing. | *유한 차분*을, **비선형 활성화 3개**를 |

### 2.3 Inline math and reference roles: always a space

**Put one space between Hangul and inline math `$…$`, a reference role
(``:numref:`…` ``, ``:eqref:`…` ``, ``:cite:`…` ``, ``:citet:`…` ``) or
`§{…}`, on both sides, also before a particle or counter.** No space is needed
next to a parenthesis or other punctuation.

Reason: `l10n check` rejects a role or inline math that touches a Hangul
letter. For roles the build needs it: tested with the repository's pandoc,
``:numref:`fig_x`를`` becomes the unknown reference `fig-x를`. For math pandoc
does not need the space; the checker asks for it today, and the rule is the same
as for Japanese. This departs from Korean typesetting, where the particle is
attached (the printed page shows "$\mathbf{X}$ 를" with a gap); §10 asks for a
build step that removes the gap.

| English | Korean |
|---|---|
| "let the input $\mathbf{X}$ be an $n_\textrm{h} \times n_\textrm{w}$ matrix" | 입력 $\mathbf{X}$ 를 $n_\textrm{h} \times n_\textrm{w}$ 행렬이라고 합시다 |
| "leaves $n - k + 1$ valid positions" | 유효한 위치가 $n - k + 1$ 개 남습니다 |
| "in :numref:`sec_padding`" | ``:numref:`sec_padding` 에서`` |
| "subtracting its right neighbor ($[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$) gives" | 오른쪽 이웃을 빼면($[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$) … |
| "expect 4D tensors (:numref:`sec_channels`)" | 4차원 텐서를 기대하므로(:numref:`sec_channels`), … |

Never let a digit follow a closing `$` (`$k$2` breaks the math); the space rule
already prevents it.

## 3. Particles after Latin terms, math, numbers and references

A Korean particle changes with the last sound of the word before it (은/는,
이/가, 을/를, 과/와, 으로/로, 이라고/라고, 이다 after a consonant or vowel).

**Decision: choose the particle by the way the token is read aloud in Korean.
Never write a double form (✗ 을(를), ✗ 이(가), ✗ (으)로).** Reason: double forms
are form-letter Korean, and a reader says the symbol aloud when reading.

### 3.1 Readings

| Token | Read as | Final consonant (은, 이, 을, 과, 으로) | Vowel or ㄹ (는, 가, 를, 와, 로) |
|---|---|---|---|
| Latin letters | 에이, 비, 시, 디, 이, 에프, 지, 에이치, 아이, 제이, 케이, 엘, 엠, 엔, 오, 피, 큐, 알, 에스, 티, 유, 브이, 더블유, 엑스, 와이, 제트 | L, M, N, R: $n$ 은, $m$ 이, $L$ 을, $r$ 은 (R: to confirm, §11) | all others: $k$ 는, $\mathbf{X}$ 를 |
| Greek letters | 알파, 베타, 감마, 델타, 엡실론, 에타, 세타, 람다, 뮤, 시그마, 타우, 파이, 로, 오메가 | ε only: $\epsilon$ 은 | all others: $\eta$ 는, $\sigma$ 를 |
| Digits (Sino-Korean) | 영, 일, 이, 삼, 사, 오, 육, 칠, 팔, 구, 십, 백, 천, 만 | 0, 1, 3, 6, 7, 8, 10: 0이, 1을, 3은 | 2, 4, 5, 9: 2가, 5를 |
| Marks | prime 프라임, hat 햇, squared 제곱; bar 바, star 스타, transpose 트랜스포즈 | $\mathbf{K}'$ 을, $\hat{y}$ 은, $n^2$ 은 | $\bar{x}$ 는, $\mathbf{X}^\top$ 를 |
| Acronyms read by letter | last letter | CNN은, RNN을, LSTM이, KL은 | GPU를, CPU가, MLP는, SGD로, GRU는 |
| Names read as words | Korean pronunciation | Adam은 (아담), ResNet은 (레즈넷), ImageNet을, Python은, GAN은 (갠), MXNet은 | ReLU는 (렐루), BERT는 (버트), PyTorch는, JAX는, NumPy를, BLEU는 |
| Percent, units | 퍼센트, 기비바이트, 밀리초 | | 92.3%는, 24 GiB를, 5 ms가 |

- **으로/로:** 로 after a vowel **and after ㄹ**: 1로, 7로, 8로, $L$ 로, $r$ 로, KL로,
  im2col로; 으로 after any other consonant: 0으로, 3으로, $n$ 으로, CNN으로.
- **A formula is read to its last spoken element:** $x_i$ 는 (아이),
  $f(x)$ 는 (엑스), $[\mathbf{X}]_{i,j+1}$ 을 (일), $2 \times 2$ 와 (이),
  $6 \times 7$ 이 (칠), $(1, -1)$ 을 (일), $\mathcal{N}(0, 0.01^2)$ 에서.
- **Code is read to its last spoken element too:** `corr2d`는 (디),
  `corr2d(X, K)`를 (케이), `[1, -1]`은 (일), `0`을 (영).
- **A term with an English gloss takes the particle of the Korean term**, not of
  the English: *에지 맵*(edge map)이 (맵 → 이), 수용 영역(receptive field)이.
- **Where the reading is not obvious** (a long expression, one that ends in an
  operator or a bracket, an unfamiliar name), restructure so that a particle
  with one form follows (의, 에, 에서, 도, 만, 처럼, 까지, 부터, 보다), or put
  the Korean noun last: ✓ $\textrm{vec}(\cdot)$ 연산은 … ; ✓ 커널
  $\mathbf{K}'$ 에서 ….

✗ 뒤집은 커널 $\mathbf{K}'$ 를 쓰는 엄밀한 수학적 합성곱 (*opus*, *sonnet*: 프라임 ends in ㅁ)
✓ 뒤집힌 커널 $\mathbf{K}'$ 을 쓰는 엄밀한 수학적 합성곱 (*fable*)

### 3.2 After a reference: only particles that have one form

The build prints the word and the number: ``:numref:`fig_x` `` → 그림 7.2.1,
``:numref:`tab_x` `` → 표 7.1, ``:eqref:`eq_x` `` → 식 7.2.1, a section → its
number with the section word, `§{sec_x}` → §7.2, ``:citet:`…` `` → the authors
and year. The translator does not see the number, so the last sound is unknown.

**Decision: after a reference role or `§{…}` write only a particle that has a
single form — 에, 에서, 에는, 에서는, 에도, 의, 도, 만, 처럼, 까지, 부터, 보다 —
or a noun (참조). Never 은/는, 이/가, 을/를, 과/와, 으로/로, 이라고/라고 or
입니다 directly after a reference.** Do not write 그림, 표, 식, 절 or 장 before a
role: the build prints the word.

| English | ✗ | ✓ |
|---|---|---|
| ":numref:`fig_x` shows the loss" | ``:numref:`fig_x` 는 손실을 보여 줍니다`` | ``:numref:`fig_x` 에는 손실이 나와 있습니다`` |
| "The output reproduces :numref:`fig_correlation` entry for entry" | ``출력은 :numref:`fig_correlation` 을 원소 하나하나까지 재현합니다`` (*fable*, *opus*) | ``출력은 :numref:`fig_correlation` 의 결과를 원소 하나하나까지 재현합니다`` |
| "implement :eqref:`eq_corr2d` literally" | ``:eqref:`eq_corr2d` 를 … 그대로 구현합니다`` (*all three*) | ``:eqref:`eq_corr2d` 의 정의를 … 그대로 구현합니다`` |
| "as shown in :numref:`fig_x`, the loss falls" | | ``:numref:`fig_x` 에서 보듯이 손실은 줄어듭니다`` |
| "see :numref:`sec_sgd`" | ``:numref:`sec_sgd` 를 보세요`` | ``(:numref:`sec_sgd` 참조)`` or ``(:numref:`sec_sgd`)`` |
| ":numref:`sec_a` and :numref:`sec_b`" | ``:numref:`sec_a` 와 :numref:`sec_b` `` | ``:numref:`sec_a`, :numref:`sec_b` 에서`` |
| ":citet:`Luo.2016` showed that …" | ``:citet:`Luo.2016` 은 …을 보였습니다`` (*fable*, *opus*) | ``:citet:`Luo.2016` 의 연구에 따르면 …입니다`` |

A citation ``:cite:`…` `` stands at the end of the clause it supports, before
the comma or full stop, with a space before it: …를 발표했습니다
:cite:`Legendre.1805`.

### 3.3 Order of protected tokens

The check compares the order of the protected tokens with the English, and
Korean is verb-final. Topic-fronting and particles keep the English order in
most sentences; use them before reordering.

"We update $\mathbf{w}$ using the gradient $\mathbf{g}$ and the step size
$\eta$." → $\mathbf{w}$ 는 기울기 $\mathbf{g}$ 와 스텝 크기 $\eta$ 를 써서
갱신합니다.

- End the sentence before a display equation or a code cell with a full stop,
  never a colon, and start a new sentence after it: "… is given by $$…$$ where
  $n$ is …" → …은 다음과 같습니다. $$…$$ 여기서 $n$ 은 …입니다.
  ✗ $2 \times 2$ 출력이 나옵니다: (*opus*) ✓ 그 결과 $2 \times 2$ 출력이
  나옵니다. (*fable*)
- Math that contains English stays as it is; do not repeat it in Korean: "the
  $i^{\textrm{th}}$ layer" → $i^{\textrm{th}}$ 층 (not 번째).
- If the English order would give wrong or clumsy Korean, write correct Korean
  and name the block in NOTES; the reorder is then a warning that is accepted.

## 4. Punctuation

| Item | Rule | Example |
|---|---|---|
| Full stop, comma, `?`, `!` | Half-width, as in English; a space after, none before. Full-width forms (。、，) are never used. Fewer commas than English: none after a short fronted adverbial, none before 그리고. | "Adam, RMSProp, and SGD" → Adam, RMSProp, SGD |
| Parentheses | Half-width `( )`, attached (§2.2). | 다층 퍼셉트론(MLP) |
| Quotation marks | Type the curly marks themselves: ‘…’ for a word mentioned as a word and for scare quotes, “…” for quoted speech or text. Never ASCII `'…'` (tested: pandoc turns `'합성곱'이라고` into ’합성곱’, both marks closing), never 「」. Book titles stay in italics as in the source. | call it "convolution" → ‘합성곱’이라고 부르겠습니다 |
| Quotes in attributes | A slide `title="…"` is delimited by ASCII `"`; inside it use ‘…’ only. | `title="‘깊다’는 무슨 뜻인가"` |
| Colon | Only after a label (a bold lead-in, 예:), attached to the label, a space after. A colon that introduces a clause, list, equation or code becomes a sentence ending. | "**Core assumption:** …" → **핵심 가정:** … ; "We get the following:" → 다음 결과를 얻습니다. |
| Semicolon, em dash | Not used. Split the sentence, or use a comma or parentheses. | "the bias—an offset—shifts the plane" → 편향(오프셋)은 평면을 평행이동시킵니다 |
| Ranges | The en dash of the source, no spaces, the unit once at the end. Reason: 문장 부호 규정 gives the tilde `~` as the range mark and permits a dash; `~` is pandoc's subscript marker and would have to be escaped (to confirm, §11). | "40–55" → 40–55 ; "45–50%" → 45–50% ; "2017–2024" → 2017–2024년 |
| Middle dot | `·` joins parallel items inside one term or a slide line, as in the source. | 2차원 교차 상관 · 에지 검출 · 학습된 커널 |
| "A vs. B" | 대 | "Learning vs. hand-crafting" → 학습 대 수작업 설계 (*fable*) ✗ 학습 vs. 수작업 설계 (*opus*) |
| "e.g.", "i.e.", "etc.", "cf." | 예를 들어, 즉, 등, 참조; never the Latin abbreviation. "et al." in running text stays, to match the printed citations (to confirm, §11). | "(e.g., Adam)" → (예: Adam) |
| Emphasis | Keep `*…*` and `**…**` on the corresponding word(s), without the particle. Add none; do not replace emphasis with quotation marks. | "*regression* predicts …" → *회귀*는 …를 예측합니다 |

No invisible characters and no full-width spaces (TRANSLATING.md §8).

## 5. Terms: Korean, Hangul transliteration or English

### 5.1 How to choose

The glossary decides each term. For a term without an entry, apply the table,
check Korean Wikipedia or a Korean textbook, and record the choice in DECISIONS.

| Case | Rule | Examples |
|---|---|---|
| An established Sino-Korean or native term exists | Use it; never a transliteration. | 합성곱 (✗ 컨볼루션), 층 (✗ 레이어, 계층), 기울기 (✗ 그래디언트), 가중치, 편향, 손실, 은닉층, 과적합, 가능도, 잡음 (✗ 노이즈), 교차 엔트로피 (✗ 크로스 엔트로피), 정책, 보상, 학습률 |
| The transliteration is the established term | Use it, spelled by 외래어 표기법 (§5.2); never coin a Sino-Korean word. | 미니배치, 에포크, 드롭아웃, 풀링, 패딩, 스트라이드, 토큰, 프롬프트, 어텐션 (✗ 주의), 임베딩, 커널, 모멘텀, 로짓, 텐서, 트랜스포머, 인코더, 디코더, 옵티마이저, 하이퍼파라미터 |
| Proper names of models, optimizers, libraries, datasets, metrics, hardware; acronyms; code | Latin script, as in English; never a Hangul spelling. | ResNet, BERT, GPT-2, Adam, AdamW, RMSProp, PyTorch, ImageNet, Fashion-MNIST, BLEU, ReLU, GELU, SGD, MLP, CNN, RNN, LSTM, GPU, KL, im2col, Tensor Core ✗ 아담, 렐루, 버트 |
| Name or acronym + common noun | Latin + space + Korean (§2.2). | Adam 옵티마이저, ReLU 함수, Xavier 초기화, He 초기화, VC 차원, KV 캐시 |
| Glossary terms that are Hangul although a name lies behind them | As the glossary has them. | 트랜스포머 (the architecture, a common noun), 소프트맥스, 시그모이드, Q-러닝 |

- **딥러닝** is one word (as in the earlier ko.d2l.ai and Korean publishing);
  "machine learning" → 머신러닝 (to confirm, §11). "Deep neural network" → 심층
  신경망; "a deeper network" → 더 깊은 신경망.
- **Verbs:** noun + 하다 (정칙화하다, 샘플링하다, 초기화하다, 평탄화하다,
  훈련하다), not 을/를 수행하다 or 을/를 진행하다. "train" → 훈련하다 (the
  model is the object); "learn" → 학습하다 (the model is the subject).
- **Parameter** is 파라미터, with 하이퍼파라미터 (to confirm, §11 and §12).
  매개변수 is kept for the parameter of a function or a curve.

### 5.2 Spelling of transliterations

Follow 외래어 표기법 (국립국어원), which the glossary already does:

| Rule | ✓ | ✗ |
|---|---|---|
| No 우 for a final /ou/ | 윈도, 플로, 섀도, 언더플로, 오버플로 | 윈도우 (*opus*), 플로우 |
| No ㅅ받침 for /dʒ/, /tʃ/; no tense consonant | 에지, 배치, 타깃, 버퍼, 메서드 | 엣지, 타겟, 메소드 |
| Fixed spellings of the glossary | 레이블, 데이터셋, 알고리즘, 스케줄, 콘텐츠, 캐시, 벤치마크, 체크포인팅, 가우시안, 마르코프, 푸아송, 라그랑주, 립시츠 | 라벨, 데이타셋, 알고리듬, 컨텐츠, 포아송 |
| Transliterated compounds | Spaced where English has a space, solid where English has a hyphen or one word. | 타임 스텝, 피드포워드 네트워크, 멀티헤드 어텐션, 미니배치, 오토인코더 |

### 5.3 The English term in parentheses

Add the English once per page (one numbered section of the book), at the first
occurrence of a key term in body text, as `용어(english)`: half-width
parentheses attached to the Korean term, lower case as in the glossary, outside
the emphasis markers. Check the English page in CONTEXT: if an earlier unit of
the page introduced the term, do not repeat it.

- **Gloss Sino-Korean and native terms whose English a reader cannot
  recover:** *2차원 교차 상관*(two-dimensional cross-correlation) (*opus*),
  유닛의 수용 영역(receptive field) (*fable*), *유한 차분*(finite difference),
  가중치 감쇠(weight decay), 과적합(overfitting).
- **Do not gloss transliterations** (the Hangul already spells the English),
  terms left in English, or school vocabulary (함수, 행렬, 확률, 평균, 벡터).
  ✗ 각 픽셀(pixel)에서 ; 잠시 채널(channel)은 제쳐 두고 ; 스칼라 편향(bias)을
  (*sonnet*, which glossed about thirty terms on one page).
- **With an acronym:** 합성곱 신경망(convolutional neural network, CNN) at the
  first occurrence on the page; if the English gives only the acronym there,
  다층 퍼셉트론(MLP). Afterwards write what the English writes (CNN, MLP).
- **Never in headings, captions, slides, exercises or table cells.** Slides
  keep the acronym the English slide uses (SGD, CNN, RF).
- No capital letter in the gloss and no space before it (✗ 분산 (Variance),
  the earlier ko.d2l.ai).

Reason: readers go on to English papers, documentation and code; once per page
keeps the text clean.

## 6. Numbers and units

| Item | Rule | Example |
|---|---|---|
| Decimal and thousands | `0.5` and `10,000`, as in the source; Korean uses the same signs. | "10,000 images" → 이미지 10,000장 |
| Digits or words | As the English: digits stay digits. A number the English spells out (one to nine) becomes a native numeral with its counter (한, 두, 세, 네, 다섯, 여섯, 일곱, 여덟, 아홉); larger spelled-out numbers become digits. Always digits with units, symbols, dimensions and comparisons. | "two hidden layers" → 은닉층 두 개 ; "in just nine steps" → 단 아홉 스텝 만에 ; "10 gradient steps" → 경사 하강 스텝 10번 ; "4D tensor" → 4차원 텐서 |
| Counter | Every counted noun takes a counter; the order is noun + number + counter, or number + counter + 의 + noun when the noun phrase is long. | "42 output entries" → 출력 원소 42개 ; "thousands of filters" → 필터 수천 개 ; "5 parameters" → 파라미터 5개 |
| Which counter | 개 (things, parameters, tokens, layers), 번 (times), 번째 (ordinal), 장 (images), 대 (GPUs, machines), 가지 (kinds), 곳 (positions), 쌍, 배, 차원, 명 (people). A unit noun counts itself: 10에포크, 9스텝, 3층, 100픽셀. | "4 GPUs" → GPU 4대 ; "3 epochs" → 3에포크 |
| Number words | 만, 억, 조 with digits, no comma inside. The check then reports the English numeral as missing: name the block in NOTES. | "3 million parameters" → 파라미터 300만 개 ; "1.5 billion tokens" → 토큰 15억 개 ; "a trillion" → 1조 |
| Ordinals | 첫 번째, 두 번째, 세 번째 …; with digits 10번째. "First, … Second, …" → 첫째, … 둘째, …. | "the third epoch" → 세 번째 에포크 |
| Percent | `%` attached, as in the source; in prose read 퍼센트 (§3.1). "Percentage points" → 퍼센트포인트. | "an accuracy of 92.3%" → 92.3%의 정확도 |
| Units | Symbols as in the source, with the source's space; never Hangul unit names for symbols (✗ 24 기비바이트). Time words written out become 초, 분, 시간, 일, 년, attached to digits. | 24 GiB, 5 ms ; "3 hours" → 3시간 |
| Ranges | §4: 40–55, 45–50%; with words, …에서 …까지. | "from 0 to 1" → 0에서 1까지 |
| Multipliers | 배 in prose and slides; `×` only inside math and in tables. | "about 2× faster" → 약 2배 빠릅니다 ; "3× larger" → 3배 큽니다 |
| Approximation | 약 before the number; "roughly", "about", "~" all become 약. "An order of magnitude" → 약 10배; "two orders of magnitude" → 약 100배; "orders of magnitude" → 몇 자릿수. | "roughly 90%" → 약 90% |
| Years, decades | 2015년, 2010년대; "in 2017" → 2017년에. | ResNet은 2015년에 나왔습니다. |
| Money | Amount + 달러, as in the source's currency; do not convert. | "\$2.50" → 2.50달러 |

Quantities, signs and orders of magnitude stay exact; do not round or convert
units. Digits attached to a counter or particle (5개, 0이) are matched by
`l10n check`; a number written as a Korean word where the English has digits is
reported as missing, so keep the English digits.

## 7. Names

- **Models, libraries, datasets, companies, hardware, benchmarks:** Latin
  script, exactly as in English: Fashion-MNIST, RTX 4090, OpenAI, GPT-2.
- **People:** Latin script as in the English source, no title (no 박사, 교수,
  씨). "Gauss grounded least squares in probability theory." → Gauss는
  최소제곱법의 근거를 확률론에서 찾았습니다. Reason: citations and searches use
  the Latin form, and Hangul spellings of less-known names vary (to confirm,
  §11).
- **Terms named after people** follow the glossary, in Hangul: 가우시안 분포,
  베이즈 정리, 야코비 행렬, 헤시안, 마르코프 연쇄, 라그랑주 승수, 립시츠 연속,
  테일러 전개, 푸아송 분포, 옌센 부등식, 랑주뱅 동역학, 벨만 최적 방정식.
- **Course, assignment, paper and book titles** in a link text or in italics
  stay in English: *Neural Networks and Deep Learning*, Assignment 2. The
  glossary check may report such a title as a missing term; name the false match
  in NOTES.
- **The book:** *Dive into Deep Learning* stays in English; "this book" → 이 책.
- **Languages and places:** Korean (영어, 미국); programming languages stay in
  Latin script (Python).
- **Structure words:** chapter 장, section 절, subsection 소절, appendix 부록,
  Summary → 요약, Exercises → 연습 문제, Discussions → 토론; Theorem 정리,
  Proposition 명제, Lemma 보조정리, Corollary 따름정리, Definition 정의, Proof
  증명, Remark 참고, Example 예.

## 8. Headings, captions, slides, exercises, notes

| Kind | Rule | Example |
|---|---|---|
| `heading` | Noun phrase, no full stop, no English gloss, no 합니다체. English "-ing + object" becomes object + verbal noun; -기 only where no verbal noun exists. A question becomes -는가 / -ㄹ까. | "Training the Model" → 모델 훈련 ; "Learning a Kernel" → 커널 학습 ; "The Model and the Loss" → 모델과 손실 ; "Why Depth Helps" → 깊이는 왜 도움이 되는가 |
| `caption` | A noun phrase where the English is one; a sentence in 합니다체. A full stop exactly when the English caption has one. Keep `(a)`, `(b)`; "panel (c)" → 패널 (c). | "Two-dimensional cross-correlation. The shaded portions are …" → 2차원 교차 상관. 음영 부분은 …입니다. |
| `slide_title`, `kicker` | Noun phrase, as short as the English, no full stop, no -하기 where a verbal noun exists. | `title="The 2D Cross-Correlation Operation"` → `title="2차원 교차 상관 연산"` ; `[Sliding-Window Inner Products]{.kicker}` → `[슬라이딩 윈도 내적]{.kicker}` |
| `slide_text` | 한다체 (-다) or a noun ending; one line where the English is one line; a full stop after a sentence in -다, none after a noun phrase. Never 합니다체 and never the report endings -함, -임, -음. Bold lead-ins keep their bold and colon. Instructions end in -ㄴ다 as well. (to confirm, §11) | "**Synthetic $6 \times 8$ test image:** White outer columns (`1`) surrounding a 4-column black vertical band (`0`)." → **$6 \times 8$ 합성 테스트 이미지:** 흰색 바깥 열(`1`)이 열 4개 너비의 검은 수직 띠(`0`)를 둘러싼다. (*fable*) ✗ …를 둘러쌈. (*opus*) ✗ …를 둘러싸고 있습니다. (*sonnet*) |
| `exercise` | Instructions in -하세요; questions in -ㅂ니까? / -입니까?; a set-up in -다고 합시다. A bold title stays bold. `[code]` and `●` are copied. Reason: -하세요 is what current Korean textbooks use beside a 합니다체 body; -하라 and -하시오 are exam-paper style. | "Derive a finite-difference operator for the second derivative. Which image structures respond most strongly to it?" → 이계 도함수에 대한 유한 차분 연산자를 유도하세요. 이미지의 어떤 구조가 이 연산자에 가장 강하게 반응합니까? (*opus*) |
| `tab_note` | Body register. Describe this framework only; never compare frameworks. API names stay in code. | "PyTorch's `nn.LazyLinear(1)` requires only the output dimension." → PyTorch의 `nn.LazyLinear(1)`에는 출력 차원만 지정하면 됩니다. |
| `note` | The register of its surroundings: 한다체 inside a slide, 합니다체 in the body. Labels: **Note:** → **참고:**, **Warning:** → **주의:**, **Tip:** → **팁:**, **Answers:** → **정답:**, "Concept Check" → 개념 확인. | "**Answers:**" → **정답:** |
| `table_row` | Noun phrases; numbers and units as in §6. | "Training time" → 훈련 시간 |

An exercise source line "Adapted from X" → X에서 각색. with the names in X left
in English. The cover kicker `[Dive into Deep Learning · §2.1]{.kicker}` is
rewritten by the build; copy it unchanged.

## 9. Known pitfalls

### 9.1 Words with more than one sense

| English | Sense → Korean | Not |
|---|---|---|
| bias | parameter $b$ → 편향; of an estimator → 통계적 편향 (then 편향); inductive bias → 귀납적 편향; social → 편견 | 바이어스 |
| kernel | convolution, kernel methods, covariance function, GPU or OS code → 커널 (커널 함수, 전이 커널); of a linear map → 핵 (null space → 영공간) | 핵 for the first four; 알맹이 |
| return | RL → 수익; of a function → 반환값, 반환하다 | 리턴; 반환값 for RL |
| score | $\nabla \log p$ → 스코어 (스코어 함수, 스코어 매칭); attention and metrics → 점수 (어텐션 점수, F1 점수) | 득점 |
| sample | a draw from a distribution, a generated output → 샘플, 샘플링하다; statistics → 표본 (표본 평균, 표본 분산) | 견본 |
| example | a data point → 예제 (훈련 예제); "for example" → 예를 들어 | 실례, 본보기 |
| feature | → 특징 (특징 맵, 특징 벡터); of a library or product → 기능 | 특성, 피처 |
| value | RL → 가치 (가치 함수); attention, a number → 값 | 밸류 |
| policy, action, reward, state | 정책, 행동, 보상, 상태 | 폴리시, 액션, 작업 |
| exploration, search, retrieval | RL → 탐험; of an algorithm → 탐색 (빔 탐색, 그리드 탐색); of documents → 검색 | 검색 for beam search |
| loss, cost, objective, error, risk | Five words, five terms: 손실, 비용, 목적 함수, 오차, 위험. | merging them |
| error | a numerical quantity → 오차 (훈련 오차, 반올림 오차); a mistake, a bug, type I/II error → 오류 | 에러 |
| accuracy, precision, recall | 정확도, 정밀도, 재현율; floating-point precision → 정밀도 | 정확성 |
| regularization, normalization, standardization | 정칙화, 정규화, 표준화; normal distribution → 정규 분포; "Gaussian" → 가우시안 | 정규화 for regularization |
| training, learning | train → 훈련 (훈련 손실, 훈련 예제, 사전 훈련); learn → 학습 (학습률, 지도 학습, 학습된 커널) | 트레이닝; 학습 for both |
| network | neural → 신경망 (합성곱 신경망, 심층 신경망); in the glossary's fixed terms → 네트워크 (타깃 네트워크, 피드포워드 네트워크); computers → 네트워크 | 망 alone |
| layer | → 층 (합성곱층, 완전 연결층). "Dense layer" and "fully connected layer" are one concept: 완전 연결층. | 레이어, 계층; 밀집층 |
| gradient, derivative | 기울기; fixed terms keep 경사 (경사 하강법, 정책 경사, "gradient step" → 경사 하강 스텝); derivative → 도함수, to differentiate → 미분하다, "with respect to $x$" → $x$ 에 대한 | 그래디언트, 구배 |
| step, stride | optimization or time step → 스텝 (스텝 크기, 타임 스텝); convolution → 스트라이드; a stage of a procedure → 단계 | 보폭; 단계 for a gradient step |
| edge | of an image feature → 에지 (에지 검출, 에지 맵); the border of an image → 가장자리, 테두리; of a graph → 간선 | 엣지; 가장자리 검출 (*sonnet*) |
| window | sliding, pooling → 윈도 (슬라이딩 윈도); context window → 문맥 창 (glossary) | 윈도우 |
| forward / backward pass | 순전파, 역전파 (§12) | 포워드 패스 |
| entry, element | of a matrix or tensor → 원소 | 항목, 엔트리 |
| shape, dimension, rank | 모양; 차원 (2차원, "fourth-order tensor" → 4차 텐서); rank of a matrix → 랭크 | 형태; 계수 for rank |
| sequence | ML data → 시퀀스; in mathematics → 수열 | 순서 |
| map, mapping | in mathematics → 사상; feature map → 특징 맵; "maps $x$ to $y$" → $x$ 를 $y$ 로 보냅니다 | 지도 |
| translation | of an image or function → 평행이동 (평행이동 불변성); of language → 번역 | 번역 불변성 (the earlier ko.d2l.ai) |
| inference | running a trained model, statistics → 추론; "the layer infers its input size" → 자동으로 정해집니다 | 추론 for shape inference |
| likelihood, evidence | 가능도; Bayesian → 증거 (증거 하한); "empirical evidence" → 근거 | 우도, 확률 |
| mean, expectation, variance | 평균, 기댓값, 분산 | 기대, 기대치 |
| memory | hardware → 메모리; LSTM → 장단기 메모리 | 기억 장치 |
| domain | of a function → 정의역; of data → 도메인 | |
| order | second-order method → 이차 방법; sequence order → 순서; of magnitude → §6 | |
| performance | of a model → 성능; of code → 속도, 처리 성능 | 퍼포먼스 |
| number | a count → 개수, 수; a number → 수; an identifier → 번호 | 파라미터의 번호 |
| converge | $x^*$ 로 수렴합니다 | …에 수렴 |
| support (of a function) | 지지 집합 | 지원 |
| trade-off | 절충; in prose often 대가 ("at the cost of") | 트레이드오프 |

### 9.2 Errors of the earlier edition

The chapter text of the earlier ko.d2l.ai is machine translation
(`glossary/review/ko.md`); its human-edited term list is older. Keep from it
the 합니다체 register, 딥러닝 as one word, and the `용어(english)` gloss. Do not
carry over: 컨볼루션 / 컨벌루션, 레이어 / 계층, 그래디언트, 주의 / 주의력 (for
attention), 우도, 번역 불변성, 관절 확률, 규범 (for norm), 아담, 렐루, a missing
space after a full stop, a space before a parenthesis, capitalised glosses.

### 9.3 Translationese

| Pattern | Write instead | Example |
|---|---|---|
| "we", "our" as 우리는, 우리의 | No subject (§1) | ✗ 우리는 손실을 최소화합니다. ✓ 손실을 최소화합니다. |
| "have" with an inanimate subject as 가지다 | 이다, 있다, or the number as predicate | "The layer has 5 parameters." ✗ 이 층은 5개의 파라미터를 가집니다. ✓ 이 층의 파라미터는 5개입니다. |
| Chains of 의, -에 대한, -를 위한, -에 의한, -를 통해 | A verb clause | "the shape of the output of the first layer of the network" ✗ 신경망의 첫 번째 층의 출력의 모양 ✓ 신경망의 첫 번째 층이 출력하는 모양 |
| Inanimate subject + "tells us", "allows us to", "shows" | -에서 … 알 수 있습니다, -(으)면 … 할 수 있습니다 | "This result tells us that depth helps." ✗ 이 결과는 우리에게 깊이가 도움이 된다는 것을 말해 줍니다. ✓ 이 결과에서 깊이가 도움이 됨을 알 수 있습니다. |
| 그것, 이것은, 그들 | Repeat the noun or drop it | "This means that it converges." ✗ 이것은 그것이 수렴한다는 것을 의미합니다. ✓ 즉, 이 반복은 수렴합니다. |
| -하는 것이 가능합니다, -할 수 있습니다 every time | The verb itself | "we can compute" → 계산합니다 |
| -을/를 수행합니다, -을/를 진행합니다 | The verb | "performs the update" → 갱신합니다 |
| "Note that", "Recall that" every time as …에 유의하세요, …을 기억하세요 | Drop it, or 이때 …, 앞에서 보았듯이 … | "Recall that $b$ is the bias." → 앞에서 보았듯이 $b$ 는 편향입니다. |
| "one of the most widely used" | 가장 널리 쓰이는 … 가운데 하나 only once; else 널리 쓰이는 | ✗ 가장 널리 쓰이는 옵티마이저들 중 하나입니다 |
| "for 10 epochs" | 10에포크 동안 | ✗ 10에포크를 위해 |
| "by …ing", "using …" | -(으)면, -아/어, -(으)로 | "By minimizing the loss, we obtain …" ✗ 손실을 최소화함으로써 우리는 …를 얻습니다. ✓ 손실을 최소화하면 …를 얻습니다. |
| Long modifier clause before a noun | Split into two sentences | Split at "which", "where", "so that". |
| Idioms translated by image | The meaning | "under the hood" → 내부적으로 ; "from scratch" → 처음부터 직접 ; "out of the box" → 별도 설정 없이 ; "derived on paper" → 수식으로 유도했습니다 ✗ 종이 위에서 유도했습니다 (*all three*) |
| "simply", "just", "very" word for word | Drop it, or 단, -만 | ✗ 단순히 …하기만 하면 됩니다 (every time) |

Logical and mathematical phrases are fixed: because → -기 때문에, -(으)므로;
therefore → 따라서; however → 그러나, 하지만; only if → -일 때만; if and only if
→ -일 때, 그리고 그때에만; at most → 많아야, 최대; at least → 적어도, 최소;
"let $x$ be …" → $x$ 를 …라고 합시다; such that → …를 만족하는; with respect to
→ …에 대한 (derivative), …에 관해; without loss of generality → 일반성을 잃지
않고; for all → 모든 …에 대해; there exists → …가 존재합니다; respectively →
각각; almost surely → 거의 확실하게; i.i.d. → 독립 동일 분포 (then i.i.d., as
the English); bounded → 유계; smooth → 매끄러운; elementwise → 원소별.

## 10. Settings

In `languages.yml`, the entry for `ko` (existing keys, `tools/l10n/config.py`
and `protect.py`):

- `dir: ltr`, `forum: english`: as now.
- `check:` nothing to set. The defaults fit this guide: `cjk_adjacent: letters`
  (a role or inline math touching Hangul is an error, §2.3), `order: warning`
  (§3.3), `numbers: warning`, `thousands: keep` (10,000 stays, §6).
- `typography: {nbsp: …}`: not set. Its rules are German (z. B., 10 000).

In `ko/strings.yml` (existing keys, read by d2l-neu's `tools/i18n.py`):

- `quarto.crossref-eq-prefix: "식"`. Quarto's Korean default is 방정식, which is
  "equation to solve", not "numbered formula".
- Figures and tables: Quarto's Korean `crossref-fig-title` 그림 and
  `crossref-tbl-title` 표 are right; no override.
- `quarto.crossref-sec-prefix` and `quarto.crossref-ch-prefix`: Quarto's Korean
  defaults print 섹션 7.2 and 장 7. Neither is Korean book usage, and no
  prefix-only value is (see the first item below). Until that exists:
  `quarto.crossref-sec-prefix: "§"`; the chapter prefix has no acceptable value.
- `book.exercises_heading: "연습 문제"`, `book.discussions_link: "토론"`.
- Fonts: the defaults for `ko` in `tools/l10n/build.py` (Noto Sans KR, Noto
  Serif CJK KR) are right.

Needing new code:

1. **Number-first reference words.** Korean writes 7.2절 and 7장 (the word after
   the number), but 그림 7.2.1, 표 7.1, 식 7.2.1. The build has only prefixes.
2. **Attaching particles to references and math.** A build step for `ko` that
   deletes the space between a reference or inline math and a following
   particle or counter (a fixed list: 은 는 이 가 을 를 의 에 에서 에는 와 과 로
   으로 도 만 처럼 까지 부터 보다 인 개 …, matched as the start of the next
   word), so that the page shows 그림 7.2.1에서 and $\mathbf{X}$를. For
   references the pandoc form `[@fig-x]에서` works (tested). Translators keep
   typing the space (§2.3).
3. **Particle agreement after references.** With the number known at build time,
   the same step could pick 은/는, 이/가, 을/를, 과/와, 으로/로 from the last
   digit (as koTeX's automatic particles do). §3.2 could then be dropped.
4. **A `cjk_adjacent: refs` mode** (roles only), if the author prefers attached
   math in the sources instead of item 2 for math.
5. **Emphasis style.** Hangul fonts have no italic; `*…*` renders as a slanted
   face. The Korean build should style `em` as bold or with emphasis dots
   (a CSS and LaTeX setting, not a translator rule).

## 11. For the author to confirm

1. **Register** (§1): 합니다체 for the body, captions and notes; 한다체 or noun
   endings on slides; -하세요 and -ㅂ니까? in exercises. Proposed: yes. The
   alternative for the body is 한다체, as in Korean graduate texts.
2. **Slide endings** (§8): -다 and noun endings, never -함/-임. Proposed: yes;
   the three test translators each chose differently.
3. **Space before a particle after math** (§2.3) and Settings items 2 and 4.
   Proposed: keep the space in the sources and remove it in the build.
4. **Particles after references** (§3.2): only single-form particles until the
   build can choose (Settings item 3). Proposed: yes.
5. **Reference words** (§10): 식 for equations; 7.2절 and 7장 by new code; "§" for
   sections until then. Proposed: yes.
6. **Reading of R** (§3.1): the standard letter name is 아르 ($r$ 는); speakers
   say 알 ($r$ 은), and both test translators who met $r$ wrote 은. Proposed: 알.
7. **Parameter** (§5.1): 파라미터 beside 하이퍼파라미터, against the glossary's
   매개변수. Proposed: 파라미터 (two of three test translators chose it, and
   매개변수 beside 하이퍼파라미터 splits one word family).
8. **Machine learning:** 머신러닝 beside 딥러닝, or 기계 학습 (Korean Wikipedia,
   and the glossary's 기계 번역). Proposed: 머신러닝.
9. **Forward and backward pass:** 순전파 and 역전파, against the glossary's
   순방향 패스 and 역방향 패스 (§12). Proposed: 순전파, 역전파.
10. **Range mark** (§4): the en dash of the source rather than the tilde `~` that
    문장 부호 규정 names first. Proposed: en dash.
11. **People's names in Latin script** (§7), also Gauss, Bayes, Newton; "et al."
    kept in running text (§4) rather than 등 or 외. Proposed: yes to both.
12. **Glossary rulings left open by `glossary/review/ko.md`:** 기울기 (not
    그래디언트), 가능도 (not 우도), 정칙화 (not 규제), 훈련 for training and 사전
    훈련 for pretraining (not 사전 학습), 특징 (not 특성), 수익 for the RL return,
    탐험 for exploration, 사영 for projection. This guide follows the glossary
    on all of them.
13. **Dense layer** as 완전 연결층 (§9.1), one term for "dense" and "fully
    connected". Proposed: yes.
14. **"On paper", "by hand"** and similar images (§9.3): translated by meaning.
    Proposed: yes.
15. **Native review.** This guide was not written by a native speaker. The
    particle table (§3.1), the counters (§6) and the example sentences should be
    read by one before the first release.

## 12. Proposed glossary corrections

Entries of `glossary/terms/ko.jsonl` that this guide's rules contradict. The
glossary was not edited.

| Concept | Now | Proposed | Rule |
|---|---|---|---|
| model-parameter, parameter-count, parameter-update, active-parameters, natural-parameter | 모델 매개변수, 매개변수 수, 매개변수 갱신, 활성 매개변수 수, 자연 매개변수 | 모델 파라미터, 파라미터 수, 파라미터 갱신, 활성 파라미터 수, 자연 파라미터 | §5.1; one family with 하이퍼파라미터 |
| forward-pass, backward-pass | 순방향 패스, 역방향 패스 | 순전파, 역전파 | §9.1; the established pair (flagged by *fable*) |
| regularization-strength, regularization-path, regularized-objective | 규제 강도, 정규화 경로, 정규화된 목적함수 | 정칙화 강도, 정칙화 경로, 정칙화된 목적 함수 | §9.1; regularization is 정칙화, and 정규화 is normalization |
| normal-distribution, exponential-distribution | 정규분포, 지수분포 | 정규 분포, 지수 분포 | §2.1; spaced like 가우시안 분포, 균등 분포, 베타 분포 |
| probability-density, latent-variable, convex-function | 확률밀도, 잠재변수, 볼록함수 | 확률 밀도, 잠재 변수, 볼록 함수 | §2.1; the glossary itself has 확률 질량 함수, 잠재 변수 모델, 볼록 집합 |
| conjugate-prior, forward-posterior, variational-objective | 켤레 사전분포, 순방향 사후분포, 변분 목적함수 | 켤레 사전 분포, 순방향 사후 분포, 변분 목적 함수 | §2.1; as 사전 분포, 사후 분포, 목적 함수 |
| backtracking-line-search | 역추적 선탐색 | 역추적 선 탐색 | §2.1; as 선 탐색 |
| pooling-window | 풀링 창 | 풀링 윈도 | §9.1; one word for the sliding window of convolution and pooling |
| newton-step | 뉴턴 단계 | 뉴턴 스텝 | §9.1; an optimization step is 스텝 |
| residual-network, fully-convolutional-network | 잔차 네트워크, 완전 합성곱 네트워크 | 잔차 신경망, 완전 합성곱 신경망 | §9.1; a neural network is 신경망 |
| soft-thresholding | 소프트 임계값 처리 | 소프트 임곗값 처리 | §1; 사이시옷 as in 고윳값, 최솟값 |
| gaussian-distribution (check only) | matches "normal distribution" | "standard normal distribution" should accept 표준 정규 분포 | *fable* and *opus* wrote 표준 가우시안 분포 only to satisfy the check, and said so in NOTES |

## 13. Sources

Checked on 2026-11-26:

- This repository: `glossary/terms/ko.jsonl`, `glossary/review/ko.md` and
  `ko-corrections.jsonl`, `glossary/notes/east-asian.md`;
  `evals/ko-conv-layer/` (RESULTS.md and the answer sets `answers-fable`,
  `answers-opus`, `answers-sonnet`, with their DECISIONS and NOTES);
  `tools/l10n/protect.py`, `glossary.py`, `config.py`, `build.py`.
- Quarto's Korean language file in d2l-neu's `.venv-build`
  (`_language-ko.yml`: 그림, 표, 섹션, 장, 방정식) and d2l-neu's
  `tools/i18n.py`, `tools/d2l_preprocess.py` (how a role becomes `@fig-x`).
- Pandoc behaviour, tested with the pandoc bundled in d2l-neu's `.venv-build`:
  inline math directly followed by Hangul parses as math; `@fig-x를` is read as
  the citation key `fig-x를`, `[@fig-x]를` is not; emphasis and bold directly
  followed by a particle work; ASCII single quotes around Hangul come out as two
  closing marks, ASCII double quotes come out right.

Not checked (stated from the drafter's knowledge):

- 한글 맞춤법 (제41항 particles, 제43항 unit nouns and digits, 제50항 technical
  terms), its 문장 부호 appendix (2015 revision: range marks, quotation marks)
  and 외래어 표기법: the texts were not re-read for this draft.
- The Korean editions of *밑바닥부터 시작하는 딥러닝* and *핸즈온 머신러닝*,
  named as 합니다체 introductions: their text was not re-checked.
- The term lists of the Korean Mathematical Society and the Korean Statistical
  Society (they would settle 가능도 / 우도 and 사영 / 투영).
- The live ko.d2l.ai pages: only what the glossary notes and the review record
  of them was used.
