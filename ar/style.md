# Style guide: Arabic (ar)

Draft of 2026-11-26 (Fable 5.2); the author decided its open points on
2026-11-27 (§12). No native reader has confirmed it yet. Translators read this
with every unit (TRANSLATING.md §7–§8). Each rule has one reason and one
example (English → Arabic, or ✗ wrong / ✓ right). §12 lists the decisions and
the evidence behind each.

Order of authority: the decisions log, then this guide, then the glossary. The
glossary says *what* a term is; this guide says *how to choose, spell and
inflect*. 163 of the 1,203 Arabic glossary entries are attested; the rest are
proposals. §13 lists the corrections applied to the glossary on 2026-11-27.

The site is right-to-left (`dir: rtl`). Code, math, Latin names and numbers
stay left-to-right inside it. §3 says how to write sentences so that the mix
reads well without direction marks, which the checker rejects.

## 1. Register

**Decision: Modern Standard Arabic (الفصحى المعاصرة), the authors and the reader
together as نحن, expressed by the verb alone.** Reason: it is the register of
Arabic university textbooks and is read the same way from Morocco to Iraq; it
carries the book's "we work through this together" tone (TRANSLATING.md §8).
No dialect words, no regional vocabulary where a pan-Arab word exists (§5.1).

| English | Arabic | Example |
|---|---|---|
| we + action in the text | first person plural verb, no pronoun | "We minimize the loss with SGD." → نصغر الخسارة بخوارزمية SGD. ✗ نحن نصغر … |
| "let us", "consider", "suppose" | لـ + jussive: لنأخذ، لنفترض، لننظر; never دعونا | "Consider locating vertical edges." → لنفترض أننا نريد تحديد مواضع الحواف العمودية. ✗ دعونا نفترض … |
| "let $x$ be …" | ليكن / لتكن + the noun + the symbol | "Let $\mathbf{X}$ be a matrix." → لتكن $\mathbf{X}$ مصفوفة. |
| "we will see", "we will use" | present tense; سـ only for a later section; never سوف | "In this section, we implement …" → في هذا القسم ننفذ … ; "We will return to this in …" → سنعود إلى ذلك في … |
| "you" = anyone ("you can verify") | نحن, or an impersonal verb (يمكن، يكفي) | "You can verify this numerically." → يمكن التحقق من ذلك عدديًا. |
| "notice", "recall", "keep in mind" | masculine singular imperative, as often as the English and no more: لاحظ أن، تذكر أن; or نحن | "Keep in mind that $r$ measures …" → لا يغب عن بالنا أن $r$ يقيس … |
| "we recommend", "we believe" | نحن | "We recommend AdamW." → نوصي بخوارزمية AdamW. |
| "we" = people, the field | نحن البشر، الباحثون | "We humans attend selectively." → نحن البشر ننتبه انتباهًا انتقائيًا. |
| exercises | masculine singular imperative (§8) | "Show that …" → بيّن أن … |

The masculine singular is the generic form of address in Arabic textbooks. Do
not write double forms (✗ اشتق/اشتقي، ✗ القارئ/ـة). Technical roles keep their
grammatical gender: الوكيل، الناقد، الفاعل، المولد، المميز.

**Sentence shape** (each point is checkable):

- **Verb first.** The default sentence is verbal (verb, subject, object). A
  nominal sentence is for contrast, for a topic picked up from the previous
  sentence, and for slides (§8). Reason: English subject-first order copied
  into every sentence is the main mark of translated Arabic; a leading verb
  also starts the line with an Arabic word (§3.2).
  "A convolutional layer cross-correlates its input with a learned kernel." →
  ✓ تحسب الطبقة الالتفافية الارتباط المتقاطع بين مدخلاتها ونواة متعلَّمة.
- **No إن as a sentence opener by habit.** At most once per paragraph, where
  the sentence states a claim. ✗ إن طرح الجار الأيمن … يعطي صفرًا ✓ طرح الجار
  الأيمن … يعطي صفرًا.
- **No "تم / يتم / قام بـ" + verbal noun.** Use the verb, active with نحن or a
  true passive. ✗ تم تدريب النموذج على ImageNet ✗ قمنا بتدريب النموذج ✗ يتم
  حساب الخسارة ✓ درّبنا النموذج على ImageNet ✓ دُرّب النموذج على ImageNet
  ✓ نحسب الخسارة. Taima's machine-translated edition has يتم 782 times and
  قمنا / نقوم بـ 396 times; these are the patterns to expect in drafts.
- **No agent with a passive.** ✗ … تُحسب من قِبل الطبقة / بواسطة الطبقة
  ✓ تحسبها الطبقة.
- **Generic nouns take the article.** English bare plurals and mass nouns that
  mean the kind are definite in Arabic. "Gradient descent finds kernels" →
  يجد النزول التدرجي النوى. "Convolutional layers are …" → الطبقات
  الالتفافية ….
- **Non-human plurals agree in the feminine singular.** الطبقات الالتفافية
  تتعلم، النماذج الكبيرة، هذه المصفوفات. ✗ الطبقات يتعلمون.
- **At most three nouns in a genitive chain (إضافة).** Break a longer one
  with لـ, في or a relative clause. "the shape of the output of the first layer
  of the network" ✗ شكل مخرجات طبقة الشبكة الأولى ✓ شكل المخرجات التي تنتجها
  الطبقة الأولى في الشبكة.
- **Lists repeat و.** Short items are joined by و alone; a comma plus و only
  between clauses. "the weight, the bias, and the learning rate" → الوزن
  والانحياز ومعدل التعلم ✗ الوزن، الانحياز، ومعدل التعلم.
- **One main clause and at most two subordinate clauses** per sentence. Split
  English sentences at "which", "where", "so that" (TRANSLATING.md §3 allows
  splitting within a block).

## 2. Spelling and marks

**Decision (author, §12.2): unvocalized text. A mark is written only where the
bare spelling reads as another word, in nouns and verbs alike; the tanwin of
the accusative is always written, before the alif.** Reasons: Arabic
textbooks, newspapers, Arabic Wikipedia, SDAIA and Hsoub print without vowels,
and SDAIA and MBZUAI mark only sporadically (a shadda on 9% and 22% of Form II
and V verbs); nouns and terms are what a reader searches for, without marks,
and what `l10n check` compares as strings (معدّل ≠ معدل). A mark is needed
where the bare word has a second common reading in the same place: بعد
"after" / بُعد "dimension", المعلمة "the teacher" / المعلَمة "the parameter",
يحدث "happens" / يحدّث "updates".

| Item | Rule | Example |
|---|---|---|
| Tanwin of the accusative | Always written, on the letter before the alif (ICAIRE 1150:9, Hsoub 9243:275, Google 9571:1355, Wikipedia 1086:399). | صفرًا، تلقائيًا، مباشرةً ✗ صفرا ✗ صفراً |
| Passive verbs | A damma on the first letter when the active reading also fits. | يُعد، يُكتب، تُسمى، دُرّب، أُعطي |
| Nouns with a second reading | The mark that settles it. | بُعد (dimension; بُعد التضمين), المعلَمة، المعلَمات الفائقة (parameter, not المعلّمة "the teacher") |
| Verbs with a second reading | The shadda that settles it. | يحدّث (updates; يحدث "happens"), يطبّع (normalizes; يطبع "prints"), يعلّم (teaches; يعلم "knows"), بيّن (show; بين "between"), درّبنا (we trained; دربنا "our path") |
| Participles with a second reading | Shadda with its vowel where the active participle is also a word: مدرِّب "trainer", متعلِّم "learner", مصمِّم "designer". | نواة متعلَّمة (learned), نموذج مدرَّب (trained), مصمَّم يدويًا |
| Everything else | No shadda and no short vowels: nouns, adjectives, verbal nouns, terms, and Form II and V verbs whose bare spelling cannot mislead. | ننفذ، نصغر، يتعلم، تذكر، التعلم العميق، معدل التعلم، الخطي، دالة، المحول، المشفر، طبقات مكدسة ✗ ننفّذ ✗ التعلّم ✗ الخطّي ✗ دالّة |
| Hamza | Write hamzat al-qat' and its seat. | إلى، أن، إن، أو، الإزاحة ✗ الى، ان، او |
| ى and ي, ة and ه | Never interchanged. | في، على، مصفوفة ✗ فى ✗ مصفوفه |
| Tatweel (ـ) | Only in the carriers الـ، بـ، لـ، كـ (§3.3); never to stretch a word. | |

The glossary follows this rule since 2026-11-27 (§13): its terms carry no
optional marks except معلَمة, بُعد, حالّ (ODE solver) and مُشرِّط
(preconditioner), and where a source spelled a term with marks (معدّل التعلّم,
دالّة التفعيل) that spelling stays an alternative, so the check accepts both.

## 3. Right-to-left text with left-to-right parts

### 3.1 What stays in Latin script

Inline code, math, reference roles, URLs, and the names and acronyms of models,
libraries, datasets, optimizers, metrics and hardware (PyTorch, ResNet, BERT,
Adam, ReLU, softmax, GPU, Fashion-MNIST, im2col) stay as in English
(TRANSLATING.md §4, §7). Never write them in Arabic letters (✗ بايثون،
✗ تينسرفلو، ✗ باي تورش، ✗ جوبيتر، ✗ ريزنت، ✗ جي بي يو), although SDAIA does
(سوفت ماكس، تينسرفلو). Terms follow §5; companies and people follow §7.

### 3.2 Order: how to write a sentence that displays correctly

The browser orders each line by the Unicode bidirectional algorithm. Arabic
letters set the direction; Latin letters form left-to-right runs; digits,
spaces and punctuation take the direction of their neighbours. Never insert
direction marks or other invisible characters (U+200E, U+200F, U+202A–U+202E,
U+2066–U+2069, U+061C): `l10n check` rejects the unit. Arrange the words
instead:

| Rule | Reason | Example |
|---|---|---|
| **Start every block with an Arabic word** (a verb, a preposition, or the class noun of the thing named): paragraphs, list items, captions, headings, slide titles. | A block that starts with a Latin letter is shown left-to-right by every tool that guesses the direction from the first letter (editors, diffs, exported slides). A class noun (الدالة، المكتبة، الشبكة، النموذج، الصنف) is grammar, not added content. Exception (author, 2026-11-27): a symbol list (the notation page, `* $x$: عدد قياسي`) starts with the symbol, as in English. | "`Conv2D` registers both parameters" → يسجل الصنف `Conv2D` المعلَمتين كلتيهما ; "PyTorch's `nn.LazyLinear(1)` requires …" → تحتاج الطبقة `nn.LazyLinear(1)` في PyTorch إلى … |
| **Put an Arabic word between two left-to-right items** (code, a Latin name, a number with its unit). Never leave only a space, a comma or a bracket between them. | Two Latin runs with only neutral characters between them join into one left-to-right run, which then reads in the opposite order to the sentence. Exception (author, 2026-11-27): an all-Latin list of names or credits (contributors and GitHub IDs, image credits) is copied as one Latin run with its Latin punctuation; only the final "and" becomes و. | "the input tensor `X` and kernel `K`" → موتر المدخلات `X` والنواة `K` ; "`X` and `Y`" → `X` و `Y` ; "Adam, RMSProp, and SGD" → Adam و RMSProp و SGD ✗ Adam، RMSProp، SGD |
| **Name the thing, then give its symbol.** Where two math spans of one noun follow each other, join them with هو / هي, ذات, بحجم or المسمى. | The noun carries case, gender and the article, which a symbol cannot; and the protected tokens keep the English order, so the check stays silent. | "a $3 \times 3$ input tensor `X`" → موتر مدخلات $3 \times 3$ هو `X` ; "the kernel $\mathbf{K}$ be a $k_\textrm{h} \times k_\textrm{w}$ matrix" → النواة $\mathbf{K}$ مصفوفة أبعادها $k_\textrm{h} \times k_\textrm{w}$ |
| **Keep a number with its Latin unit or name in the English order**, as one piece. | It is one left-to-right run and displays as written. | 24 GiB, 5 ms, Python 3.12, GPT-2, RTX 4090 |
| **Write ranges in prose with words.** Keep the en dash only in slides, tables and captions. | In right-to-left text `40–55` is drawn with 40 on the right, `40-55` with 40 on the left; من … إلى … cannot be misread. | "40–55 epochs" → من 40 إلى 55 حقبة ; table cell: 40–55 |
| **Arrows in Arabic text point in the reading direction** (author, 2026-11-27): an arrow that means "next", "leads to" or "becomes" between Arabic words, or after an Arabic link text, is ← (`&larr;`); one that means "back" is →. Arrows inside math, code, or a Latin run (a menu path such as **File → Save a copy in Drive**) stay as written. | Unicode arrows are not mirrored by the bidirectional algorithm, and the site's style sheet does not mirror them either, so an English → would point backwards in right-to-left text. | "Start with the Introduction&nbsp;&rarr;" → ابدأ بالمقدمة&nbsp;&larr; ; "state → action → reward" → الحالة ← الفعل ← المكافأة |
| **Geometry is copied, not mirrored.** Left, right, top-left, "left to right" in the English describe figures, matrices, equations and code, which are not mirrored. | The figure the reader sees is the English one. | "the upper-left corner" → الركن العلوي الأيسر ; "left to right" → من اليسار إلى اليمين |

If the English order of protected tokens gives wrong Arabic, write correct
Arabic and name the block in NOTES (the check then accepts the reorder). A
reorder between classes (a math span and a code span changing places) needs no
note: the check compares the order within each class only.

### 3.3 Spacing and attached particles

**One space on each side of every protected token and every Latin word or
number, except next to punctuation. Nothing is attached to them.**

| Case | Rule | Example |
|---|---|---|
| Reference roles (``:numref:``, ``:eqref:``, ``:cite:``, ``:citet:``, `§{…}`) | A space before and after, or punctuation. Required by the build: tested with the book's pandoc, a role directly after an Arabic letter is not recognised, and an Arabic letter directly after it becomes part of the label. | ✓ ``في :numref:`sec_sgd` نرى`` ✓ ``(:numref:`sec_sgd`)`` ✓ ``:numref:`sec_sgd`، ثم`` ✗ ``ب:numref:`sec_sgd` `` ✗ ``:numref:`sec_sgd`و`` |
| و before a token, a Latin word or digits | Written alone, with a space on both sides (author, §12.4). The more common practice in Arabic ML writing, and the build needs it: pandoc takes an attached و into a label (``@sec-xو``). | `X` و `Y` ; و $-1$ حيث … ; PyTorch و JAX ; 3 و 4 ✗ و$-1$ ✗ و`0` ✗ وPyTorch |
| و، ف، ب، ل، ك and ال before an Arabic word | Attached as always, no space. | والانحياز، بالنزول التدرجي، للطبقة ✗ و الانحياز |
| ب، ل، ك before a token or Latin name | First choice: attach the particle to a class noun, or use a free-standing preposition (في، من، إلى، على، عن، مع، بين، عبر، مثل، باستخدام). MBZUAI does the same (24 class nouns against 6 carriers before Latin). | "for GPT" → لنموذج GPT ; "with `corr2d`" → بالدالة `corr2d` ; "in PyTorch" → في PyTorch ; "of $\mathbf{X}$ with $\mathbf{K}$" → بين $\mathbf{X}$ و $\mathbf{K}$ ; "to $\mathbf{w}$" → إلى المتجه $\mathbf{w}$ |
| Carriers بـ، لـ، كـ | Allowed before Latin or digits, with tatweel and a space, where no class noun and no free preposition fits (author, §12.5). | لـ $\mathbf{X}$ ; بـ 3 طبقات ✗ لGPT ✗ ب`X` |
| The article with a Latin name or acronym | None. A class noun, the genitive or the context makes it definite. | "the GPU" → وحدة GPU ; "GPUs" → وحدات GPU ; "the Transformer block" → كتلة المحول ✗ الـ GPU ✗ الـGPU |
| The article with a number written in digits | الـ + space, for a definite counted noun. | "all 42 output entries" → عناصر المخرجات الـ 42 كلها |
| Punctuation | No space before ، ؛ ؟ : . ! and before a closing bracket; one space after. | $(i, j)$، ثم … |
| Emphasis markers | Around the whole written word, with its article, attached particle and suffix; never inside a word. | *الحشو* ; *بالنزول التدرجي* ; *مجال استقباله* ✗ ب*النزول* ✗ *استقبال*ه |

## 4. Punctuation

| Item | Rule | Example |
|---|---|---|
| Comma, semicolon, question mark | Arabic ، ؛ ؟ in Arabic text. The Latin `,` `;` `?` only inside math, code, numbers (10,000), an all-Latin parenthesis and an all-Latin list of names or credits (§3.2). | هل يستطيع النزول التدرجي أن يستعيد النواة؟ ; (convolutional neural network, CNN) |
| Full stop, colon, exclamation mark, parentheses | `.` `:` `!` `( )`. Brackets are typed in logical order; the display mirrors them. | |
| Comma after a fronted phrase | None. Do not copy the English comma after "In this section," or "In :numref:`x`,". | ✓ ``في :numref:`sec_why-conv` اشتققنا الطبقة الالتفافية`` ✗ ``في :numref:`sec_why-conv`، اشتققنا …`` |
| Comma before و | None between words; allowed between clauses. | ننفذ الارتباط المتقاطع، ونختبر كاشف الحواف، ونبين … |
| Quotation marks | Curly “…”, typed in reading order (“ first); inside them ‘…’. No space inside the marks. Never straight "…" and never «…» in Arabic text (author: MBZUAI writes “…” 1,479 times and «…» 39). | "we call cross-correlation "convolution"" → نسمي الارتباط المتقاطع “التفافًا” ✗ «التفافًا» ✗ “ التفافًا ” |
| Quotes in attributes | A slide `title="…"` is delimited by ASCII `"`; inside it use “…” only. | `title="ما معنى “عميق”؟"` |
| Dash | The English em dash `—` becomes a comma, a colon or parentheses. No `—` and no spaced hyphen. | "… during forward propagation—borrowing the neurophysiology term" → … أثناء التمرير الأمامي، وهو مصطلح مستعار من فسيولوجيا الأعصاب |
| Colon before a display equation | Keep it where the English block ends with one. | … لنحصل على عنصر المخرجات عند $(i, j)$: |
| Abbreviations | Written out: "e.g." → مثل، مثلًا; "i.e." → أي; "etc." → وغيرها، إلخ; "vs." → مقابل; "cf." → قارن بـ. "et al." stays inside a Latin citation: (Luo et al., 2016). | "i.e., edges in the direction …" → أي الحواف في الاتجاه … |
| Emphasis | Keep `*…*` and `**…**` on the corresponding word(s); add none. The English gloss stays outside the markers. | `*الفرق المحدود* (finite difference)` |
| Bold lead-ins | Keep the bold and the colon inside it. | `**الفرق المحدود الأفقي:** تمرير …` |

## 5. Terminology

### 5.1 Arabic term, Latin name or loanword

**Use the Arabic term the glossary gives. Keep names and acronyms in Latin
script. Never transliterate an English term into Arabic letters on your own.**
Reason: Arabic ML terminology differs by country and by author; one term per
concept, with the English beside it once (§5.4), is what lets a reader follow.

| Case | Rule | Examples |
|---|---|---|
| Glossary term | Use it; inflect it as grammar requires (§5.2). | الطبقة، الوزن، التدرج، النزول التدرجي، دالة الخسارة، معدل التعلم، الانتشار العكسي، فرط الملاءمة، التضمين، الانتباه، التجميع، الحشو |
| Loanwords the glossary writes in Arabic letters | Use them as Arabic nouns. | بكسل (بكسلات)، إنتروبيا، لوجيت، بيرسبترون، خوارزمية، ديناميكيات |
| No glossary entry | The term of the SDAIA/ICAIRE glossary (icaire.org/ar/glossary), Arabic Wikipedia or Hsoub Academy, where it is the common one; record it in DECISIONS. If there is none, an Arabic description with the English in parentheses; not a transliteration. | ✗ كيرنل، ✗ باتش، ✗ دروب آوت، ✗ توكن، ✗ ترانسفورمر |
| Names of models, architectures, optimizers, libraries, datasets, metrics; acronyms; code | Latin script; never translated in part. | ResNet, BERT, Adam, AdamW, PyTorch, ImageNet, BLEU, ReLU, softmax, SGD, MLP, CNN, LSTM, GPU |
| Latin name + Arabic noun | The Arabic class noun comes first, without the article (it is definite by the genitive); an adjective follows the name. | خوارزمية Adam، دالة ReLU، مكتبة PyTorch، مجموعة بيانات ImageNet، نموذج BERT، درجة BLEU، تعلم Q ; "a deep CNN" → شبكة CNN عميقة |
| Plural or dual of an acronym | On the class noun; never a Latin plural in prose. | "GPUs" → وحدات GPU ; "two GPUs" → وحدتا GPU ; "CNNs" → شبكات CNN ✗ CNNs ✗ GPUs |
| "Transformer" | The architecture is the glossary's المحول (a common noun); product and model names keep Latin (Vision Transformer (ViT) on first mention, then ViT). Taima's edition writes المحول 217 times, SDAIA مُحوِّل. | مشفر المحول، كتلة المحول |

Pan-Arab words, not regional ones: حاسوب (✗ كمبيوتر، ✗ الكمبيوتر), دالة
(✗ تابع), شبكة عصبية (✗ عصبونية), مبرهنة for a theorem (نظرية is a theory),
ملف، مكتبة، إطار عمل، دفتر (notebook: دفتر Jupyter). Program code is الشيفرة
(author, §12.12); the glossary also accepts الكود (SDAIA) and التعليمات
البرمجية (Google, IBM), but write الشيفرة.

### 5.2 Inflecting a glossary term

- **The article is free.** Glossary entries are lemmas without ال (since
  2026-11-27, as in the SDAIA glossary); make each definite or indefinite by
  grammar: وزن → الوزن، بالوزن، أوزان الشبكة.
- **Plurals are the natural ones**, broken where the word has one: نموذج →
  نماذج، دالة → دوال، وزن → أوزان، خريطة → خرائط، شكل → أشكال، نواة → نوى.
  Never a made-up sound plural to match the glossary (✗ نموذجات).
- **نواة الالتفاف, then النواة.** The full term at the first mention in a
  unit, then the short form; plural نوى (author, §12.9). Taima's edition and
  Arabic Wikipedia write نوى for kernels; أنوية is Taima's word for CPU cores.
  Where نوى could be read as the verb, write النوى or a genitive (نوى
  الالتفاف).
- **Number grammar applies to terms.** After 3–10 the plural in the genitive
  (10 خطوات); after 11–99 and after a symbol the singular in the accusative
  (42 عنصرًا، $L$ طبقة، $n - k + 1$ موضعًا); after 100 and above the singular
  in the genitive (100 طبقة).
- **Compound adjectives follow the noun without their own article:** الارتباط
  المتقاطع ثنائي الأبعاد، البيرسبترون متعدد الطبقات، موتر رباعي الأبعاد
  (author, §12.13; formal grammar and SDAIA's definitions write الثنائي
  الأبعاد, Taima's edition the bare form 161 times against 4).
- **A symbol agrees through its noun.** Give the noun where gender or case must
  show: المصفوفة $\mathbf{W}$ متناثرة ; لتكن $\mathbf{X}$ مصفوفة. A bare symbol
  takes the gender of what it denotes.

### 5.3 Verbs

Arabic verbs from the root of the glossary noun; never an English stem.

| English | Arabic | Not |
|---|---|---|
| train, pretrain, fine-tune | يدرب; يدرب مسبقًا (نموذج مدرَّب مسبقًا); يضبط ضبطًا دقيقًا | يعمل تدريب، يفاين-تيون |
| learn (a model learns a kernel) | يتعلم | يدرس |
| sample (from a distribution) | يسحب عينة من؛ noun أخذ العينات | يعين |
| initialize, update, normalize, regularize | يهيئ، يحدّث، يطبّع، ينظم | |
| minimize, maximize, optimize | يصغر، يعظم، يحسن | يقلل for "minimize" (it means "reduce") |
| compute, evaluate (an expression) | يحسب | يقيّم (it means "assess") |
| implement, run, call | ينفذ، يشغل، يستدعي | يطبق for "implement" (it means "apply") |
| flatten, stack, reshape, pad | يسطح، يكدس (noun تكديس، طبقات مكدسة)، يعيد تشكيل، يحشو | |
| return (a function) | يعيد، تعيد الدالة | يرجع بـ |
| backpropagate | ينشر عكسيًا; noun الانتشار العكسي | |

### 5.4 The English term in parentheses

Add the English once per page (one numbered section of the book), at the first
occurrence of a key term in body text, in ASCII parentheses, lower case as in
the glossary, after the Arabic term and outside the emphasis markers. Check the
English page in CONTEXT: if an earlier unit of the page introduced the term, do
not repeat it.

- **Gloss** the terms the page introduces or defines (the English usually sets
  them in `*…*`) and ML terms whose Arabic is not uniform across the Arab
  world: *الارتباط المتقاطع ثنائي الأبعاد* (two-dimensional cross-correlation),
  *مجال الاستقبال* (receptive field), الانتشار العكسي (backpropagation), فرط
  الملاءمة (overfitting).
- **Do not gloss** school mathematics (مصفوفة، متجه، مشتق، احتمال، دالة),
  everyday words (صورة، قناة), names and acronyms, or a term the page only uses
  in passing. ✗ لنفترض أن المدخل $\mathbf{X}$ مصفوفة (matrix) (Sonnet glossed
  20 terms in the first unit of the test page, Fable 4 and Opus 2.)
- **With an acronym:** الشبكة العصبية الالتفافية (convolutional neural
  network, CNN), with a Latin comma inside the parenthesis; afterwards write
  what the English writes (CNN, MLP, SGD), with a class noun where grammar
  needs one (§5.1).
- **A term the English already has in parentheses:** "a border of zeros
  (*padding*)" → إطار من الأصفار (*الحشو*، padding).
- **Never** in headings, captions, slides, exercises or table cells, and never
  with a label such as بالإنجليزية.
  An acronym that the page never introduces with its gloss is written out in
  Arabic, even where the English uses it (author, 2026-11-27): "reverse-mode
  AD" → الاشتقاق التلقائي بالنمط العكسي ; "95% CI" → فترة ثقة 95%.

Reason: readers go on to English papers, documentation and code, and the Latin
script already says that the word is English.

## 6. Numbers and units

**Decision: Western digits 0–9, the decimal point, and the English thousands
comma, everywhere; numerals are copied from the English.** Reasons: math, code,
code output and figure axes are protected and show 0–9 and the point, and a
second digit set in the prose would put two conventions into one sentence;
North Africa writes 0–9 in any case and Arabic Wikipedia uses them; and
`l10n check` compares digit strings, so ٣ for 3 or ٠٫٥ for 0.5 is reported as a
missing number. Never ٠–٩, never the Arabic decimal sign ٫, thousands sign ٬ or
percent sign ٪, and never the Arabic comma ، inside a number.

| Item | Rule | Example |
|---|---|---|
| Digits or words | As the English: digits stay digits, a number spelled out in English is spelled out in Arabic. | "10 gradient steps" → 10 خطوات تدرج ; "within just nine steps" → في تسع خطوات فقط |
| One and two of a counted noun | The singular with واحد, or the dual, without the digit. The check then reports the number as missing: name the block in NOTES. In a table cell, or where the sentence compares it with other digits, keep the digit and the singular noun. | "each layer adds 2 pixels" → تضيف كل طبقة بكسلين ; "only **1 path**" → **مسار واحد** فقط ; table: 2 بكسل |
| Order | The number before its noun; the noun's form by §5.2. | 42 عنصرًا، 100 طبقة |
| Decimal, thousands | As in the source. | "learning rate 0.03" → معدل تعلم 0.03 ; "10,000 images" → 10,000 صورة |
| Number words | million → مليون, billion → مليار, trillion → تريليون. | "3 billion parameters" → 3 مليارات معلَمة |
| Percent | `%` after the number, no space, as in the source; never ٪ (author; SDAIA 287:0, MBZUAI 192:9). | "45% fewer parameters" → معلَمات أقل بنسبة 45% ✗ 45٪ |
| Units | Symbols as in the source, with the source's space; time words written out become ثانية، دقيقة، ساعة، يوم. | 24 GiB, 5 ms ; "3 hours" → 3 ساعات |
| Ranges | In prose من … إلى …, the unit once; the en dash stays in slides, tables and captions (§3.2). | "45–50%" → من 45% إلى 50% |
| Multipliers | In prose مرة / مرات / ضعف; keep `×` in slides, tables, captions and math. | "about 2× faster" → أسرع بنحو مرتين ; "3× larger" → أكبر 3 مرات ; slide: أسرع 2× |
| Approximation | نحو، قرابة، تقريبًا; "an order of magnitude" → مرتبة مقدار واحدة. | "roughly 0.02" → 0.02 تقريبًا |
| Ordinals | Written out: الأولى، الثانية، العاشرة; above ten with the noun and رقم or the digit. | "the third epoch" → الحقبة الثالثة |
| Years, decades | Gregorian, digits, with عام where it reads better; decades in words. | "in 2017" → عام 2017 ; "the 1990s" → تسعينيات القرن العشرين |
| Money | Amount, then the currency in words, in the source's currency. | "\$2.50" → 2.50 دولار |
| Dimensions | In math as written; in prose with ثنائي / ثلاثي / رباعي الأبعاد. | "2D", "4D tensors" → ثنائي الأبعاد، موترات رباعية الأبعاد |

Quantities, signs and orders of magnitude stay exact; do not round or convert
units or currencies.

## 7. Names

- **Models, libraries, tools, datasets, hardware, benchmarks, courses:** Latin
  script, exactly as in English, also for versioned model names: Fashion-MNIST,
  RTX 4090, GPT-2, GPT-4o, Llama-8B, PyTorch, Python, Jupyter, Stanford
  CS231n (✗ بايثون، ✗ تينسرفلو، ✗ باي تورش، ✗ جوبيتر).
- **Companies and products** (author, 2026-11-27): the same rule as for
  people. They stay in Latin script, as in the English: أعلنت Nvidia …،
  خدمات Amazon … (✗ إنفيديا، ✗ جوجل، ✗ مايكروسوفت). SDAIA and MBZUAI write
  them in Arabic letters (مايكروسوفت (Microsoft)); the book does not, so that
  company, product, model and library names all read the same way.
- **People** (author, §12.7): a person whose name is part of a glossary term
  is written in Arabic letters, with the glossary's spelling, everywhere: غاوس،
  بايز، ماركوف، نيوتن، لاغرانج، تايلور، فورييه، برنولي، بواسون، ليبشيتز،
  كولموغوروف، فيشر، بولتزمان. Cited authors and every other person stay in
  Latin script, as in the English and in the printed citation: اقترح Hinton ….
  Both kinds may meet in one sentence. Reason: Arabic writing transliterates
  the classical names, and a reader expects غاوس beside التوزيع الغاوسي; for
  living researchers the transliteration is not fixed (Arabic Wikipedia has
  both يان لوكون and يان ليكون), and citations print the Latin form. No titles
  (✗ الدكتور، ✗ البروفيسور).
- **Transliteration, where a person's name is in Arabic letters:** /g/ is غ
  (غاوس، لاغرانج، غرام); /v/ is ف; /p/ is ب; no ڤ، پ، گ. This is the
  glossary's practice.
- **Terms named after people:** as the glossary has them: التوزيع الغاوسي،
  مبرهنة بايز، سلسلة ماركوف، مضاعف لاغرانج، طريقة نيوتن، مفكوك تايلور، ثابت
  ليبشيتز، المصفوفة الهيسية، المصفوفة اليعقوبية.
- **The book:** *Dive into Deep Learning* stays in English; "this book" → هذا
  الكتاب.
- **Languages and places:** Arabic (الإنجليزية، الولايات المتحدة); programming
  languages stay in Latin (Python, ✗ بايثون).
- **Structure words:** chapter الفصل, section القسم, subsection القسم الفرعي,
  appendix الملحق, part الجزء, the Attic part → الأرشيف (author, 2026-11-27; not the literal العلية); Summary → الملخص, Exercises → تمارين,
  Discussions → النقاشات; Theorem → مبرهنة, Proposition → قضية, Lemma →
  توطئة, Corollary → نتيجة, Definition → تعريف, Proof → برهان, Remark →
  ملاحظة, Example → مثال. The heading "تمارين" and the link text "النقاشات" are
  matched by the build (§11): write them exactly so.

## 8. Headings, captions, slides, exercises, notes

| Kind | Rule | Example |
|---|---|---|
| `heading` | Noun phrase, no full stop, no English gloss. A topic takes the article or a genitive; English "-ing + object" becomes a verbal noun (مصدر) + genitive; a generic English plural becomes the definite singular or plural. A question stays a question. | "Convolutions for Images" → الالتفاف على الصور ; "The Operation" → العملية ; "Edge Detection" → كشف الحواف ; "Learning a Kernel" → تعلم نواة ; "Why Depth Helps" → لماذا يفيد العمق؟ |
| `caption` | Body register without نحن; a full stop exactly when the English caption has one. Keep `(a)`, `(b)`. | "Two-dimensional cross-correlation. The shaded portions are …" → الارتباط المتقاطع ثنائي الأبعاد. الأجزاء المظللة هي … |
| `slide_title`, `kicker` | Noun phrase, as short as the English, no full stop; starts with an Arabic word (§3.2). | `title="Implementing 2D Cross-Correlation in Code"` → `title="تنفيذ الارتباط المتقاطع ثنائي الأبعاد في الشيفرة"` ; `[Motivation & Core Mental Model]{.kicker}` → `[الدافع والنموذج الذهني الأساسي]{.kicker}` |
| `slide_text` | Short nominal sentences or noun phrases; no نحن; instructions as verbal nouns, not imperatives; one line where the English is one line; no full stop after a single-phrase bullet. Bold lead-ins keep their bold and colon. Acronyms as the English writes them. | "**Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent." → **التعلم مقابل التصميم اليدوي:** الشبكات الالتفافية العميقة تكتشف آلافًا من هذه النوى تلقائيًا بالنزول التدرجي. ; "Lower the learning rate gradually" → خفض معدل التعلم تدريجيًا |
| `exercise` | Masculine singular imperative of the verb itself, never قم بـ + verbal noun; direct questions with ؟. A bold title stays bold with its full stop inside. The tag `[code]` and the mark ● are copied. | "Implement the edge detector." → نفذ كاشف الحواف. ✗ قم بتنفيذ كاشف الحواف. ; "Derive a finite difference operator for the second derivative. Which structures in images respond most strongly to it?" → اشتق مؤثر فروق محدودة للمشتق الثاني. أي البنى في الصور تستجيب له بأقوى ما يكون؟ |
| `tab_note` | Body register. Describe this framework only; never compare frameworks. API names stay in code, after a class noun. | "PyTorch's `nn.LazyLinear(1)` requires only the output dimension." → تحتاج الطبقة `nn.LazyLinear(1)` في PyTorch إلى بُعد المخرجات فقط. |
| `note` | Register of its surroundings. Labels: **Note:** → **ملاحظة:**, **Warning:** → **تنبيه:**, **Tip:** → **نصيحة:**, **Answers:** → **الإجابات:**, Concept Check → اختبار الفهم. | |
| `table_row` | Noun phrases; numbers and units as in §6. | "Training time" → زمن التدريب |

The cover kicker `[Dive into Deep Learning · §2.1]{.kicker}` is rewritten by the
build; copy it unchanged. "Adapted from …" → مقتبس بتصرف من …, with course and
book titles in English.

## 9. References, math and code in a sentence

### 9.1 Cross-references

The Arabic build is to print the prefix with the article: الفصل 3، القسم 5.1،
الشكل 4، الجدول 2، المعادلة 3.1, as Arab journal papers do (§11; until
`ar/strings.yml` sets them the build prints the English words, because Quarto
ships no Arabic language file). Whether equation references print المعادلة is
checked in the first Arabic build (§12.16).
Write no prefix of your own and attach nothing to a reference (§3.3).

| English | Arabic |
|---|---|
| "In :numref:`fig_correlation`, we slide a kernel …" | ``في :numref:`fig_correlation` نزلق نواة …`` (prints: في الشكل 7.2 نزلق) |
| "as shown in :numref:`fig_x`, the loss falls" | ``كما يبين :numref:`fig_x`، تنخفض الخسارة`` |
| "see :numref:`sec_sgd`" | ``انظر :numref:`sec_sgd` `` ✗ ``انظر القسم :numref:`sec_sgd` `` ✗ ``انظر في ال:numref:`sec_sgd` `` |
| "(:numref:`sec_channels`)" | ``(:numref:`sec_channels`)`` |
| "We implement :eqref:`eq_corr2d` directly" | ``ننفذ :eqref:`eq_corr2d` مباشرة`` |
| ":citet:`Luo.Li.Urtasun.ea.2016` showed that …" | ``بيّن :citet:`Luo.Li.Urtasun.ea.2016` أن …`` |

- **Prepositions before a reference are free-standing:** في، من، إلى، على،
  حسب، مثل. Not ب، ل، ك.
- **Agreement:** the printed nouns الفصل، القسم، الشكل، الجدول are masculine,
  المعادلة is feminine. A verb or adjective that agrees with a ``:numref:`` is
  masculine, with an ``:eqref:`` feminine: ``يبين :numref:`fig_x` ``,
  ``تعطي :eqref:`eq_x` ``. Prefer the reference after a preposition, where
  nothing has to agree.
- **Citations:** put the verb before ``:citet:`` and keep it masculine
  singular; a verb before its subject does not change for "et al.". No
  possessive on a citation: النموذج الذي اقترحه ``:citet:`…` ``. A ``:cite:``
  stands at the end of the clause it supports, after a space and before the
  ، or the full stop, as in English.
- `§{sec_x}` (slides) prints "§9.3"; copy it and attach nothing to it.

### 9.2 Math and code

- Protected math and code are copied as they are, also when they contain
  English ("the $i^{\textrm{th}}$ layer" → الطبقة $i^{\textrm{th}}$). Do not
  add a math span or backticks that the English does not have.
- Code needs its class noun where the sentence would otherwise begin with it or
  a particle would attach to it: الدالة `corr2d`، الصنف `Conv2D`، الموتر `X`.
  The noun is not needed after a free preposition: نستدعي `corr2d` في `forward`.
- A symbolic count takes the singular accusative: $c$ قناة، $L$ طبقة.
- End the sentence before a display equation if the Arabic would otherwise have
  to continue after it: "… is given by $$…$$ where $n$ is …" → … تُعطى
  بالعلاقة $$…$$ حيث $n$ ….
- A name such as `[1, -1]` or `-1` in backticks, and a signed number outside
  math, are copied; their display is the build's task (§11).

### 9.3 Fixed logical and mathematical phrases

because → لأن; therefore → لذلك، ومن ثم; however → لكن، غير أن; only if → فقط
إذا; if and only if → إذا وفقط إذا; at most → على الأكثر; at least → على
الأقل; such that → بحيث; with respect to → بالنسبة إلى; without loss of
generality → دون فقدان العمومية; for all → لكل; there exists → يوجد;
respectively → على الترتيب; up to (a sign, a constant) → بصرف النظر عن;
i.i.d. → مستقلة ومتماثلة التوزيع (then i.i.d.); elementwise → عنصرًا بعنصر;
row by row → صفًا بصف; end to end → من طرف إلى طرف; from scratch → من الصفر;
dot product → الجداء النقطي; "the following" → ما يلي; "as follows" → كما يلي.

### 9.4 What the checks accept

`l10n check` accepts a glossary term with the article, with one or two attached
particles (و ف ب ل ك), in the dual, in a sound plural and with a suffix. It
recognises a broken plural or a marked spelling only where the glossary lists
it as an alternative (it lists نماذج، دوال الخسارة، خرائط الميزات، أشكال، نوى and the
attested marked spellings). Any other broken plural, or a spelling that differs
in marks (§2), is a false match: write the correct Arabic and say so in NOTES,
once per unit, as TRANSLATING.md §9 requires. The same holds for the number warning after a dual (§6) and for the
English verbs "return", "recall" and "span" taken for their noun senses.

## 10. Known pitfalls

### 10.1 Words with more than one sense

| English | Sense → Arabic | Not |
|---|---|---|
| vertical, column, orthogonal | vertical → عمودي (الحواف العمودية، المحور العمودي); column → عمود (أعمدة); orthogonal, perpendicular → متعامد | رأسي (author, §12.11); عمودي for "perpendicular" |
| input, output | the data of a layer, model or operation → المدخلات، المخرجات (one item: المدخل، المخرج); the act of reading or writing → الإدخال، الإخراج, also in fixed terms (طبقة الإدخال، بوابة الإخراج) | الدخل، الخرج (author, §12.10); الإدخال، الإخراج for the data |
| parameter, coefficient | parameter → معلَمة (معلَمات); coefficient → معامل (معاملات) | معامل for "parameter" (SDAIA and MBZUAI write it, and so did all three test translations; the glossary keeps the two apart) |
| kernel | convolution → نواة الالتفاف, then النواة, plural النوى; kernel methods → دالة النواة، مصفوفة النواة; GPU or OS code → النواة; of a linear map → الفضاء الصفري | أنوية for kernels (it is the plural for CPU cores) |
| convolution | → الالتفاف; convolutional → التفافي (طبقة التفافية) | التلافيف، تلافيفية; ترشيح (SDAIA's term, an alternative in the glossary) in prose |
| bias | parameter $b$ → الانحياز; of an estimator → الانحياز (مقدر غير منحاز); social → التحيز | الميل; التحيز for the bias term |
| scalar | noun → عدد قياسي; adjective → قياسي (انحياز قياسي، خسارة قياسية) | عددي (it means "numerical") |
| accuracy, precision, resolution | accuracy → الدقة; precision (metric) → الإحكام; floating-point precision → الدقة (دقة مزدوجة); resolution → دقة الصورة، دقة المدخلات, always with its noun | الدقة alone for "resolution" |
| translation, shift, offset | of an image or function → الإزاحة (تساوق الإزاحة، ثبات الإزاحة); distribution shift → الانزياح; an index offset → الإزاحة; of language → الترجمة | ترجمة for a shift |
| step size, stride, time step | حجم الخطوة; خطوة الانزلاق; خطوة زمنية | الخطوة alone for a stride |
| shape, figure | tensor shape → الشكل (شكل المخرجات); the build also prints الشكل before figure references, so near one write أبعاد or شكل الموتر | |
| network, grid | network → شبكة; grid of pixels or points → شبكة نقطية | شبكة for both in one sentence |
| edge | in an image → حافة (حواف); of a region or a receptive field → طرف; of a graph → ضلع | |
| stack | stacking layers or tensors → التكديس; verb → يكدس; stacked → مكدسة; the data structure → المكدس | كدسة، كومة |
| batch, "in one shot" | batch → دفعة; "in one shot", "at once" → في مرة واحدة | دفعة واحدة for "at once" |
| feature | → ميزة (ميزات); feature map → خريطة الميزات | سمة and ميزة mixed; خاصية (SDAIA) in prose; خريطة المعالم، خريطة الخصائص |
| return | RL → العائد; of a function → القيمة المعادة، يعيد | |
| sample | → عينة; to sample → يسحب عينة; a draw → سحبة | |
| loss, cost, objective, error, risk | Five words, five terms: خسارة، كلفة، دالة الهدف، خطأ، خطر. | merging them |
| regularization, normalization, standardization, calibration | التنظيم; التطبيع (تطبيع الدفعات); التوحيد المعياري; المعايرة; normal distribution → التوزيع الطبيعي, Gaussian → الغاوسي | mixing them |
| training, learning | train → التدريب; learn → التعلم (معدل التعلم، التعلم العميق) | |
| sequence | ML data → تسلسل; in mathematics → متتالية | |
| gradient, derivative | التدرج; المشتق, "with respect to $x$" → بالنسبة إلى $x$ | الميل، الانحدار (it is "regression"); التدرج الاشتقاقي |
| backpropagation | الانتشار العكسي | الانتشار الخلفي (Arabic Wikipedia's title) |
| likelihood, probability | الإمكان (الإمكان الأعظم، تقدير الإمكان الأعظم); الاحتمال | الاحتمالية for both; الأرجحية, which is the glossary's word for odds (SDAIA's تقدير الأرجحية الأعلى is only an alternative) |
| token, symbol, encoding | token → الرمز المميز (الرموز المميزة), in fixed compounds the short الرمز (رمز مجهول، التنبؤ بالرمز التالي); mathematical symbol → رمز, so write الرمز الرياضي where both occur; encoding → ترميز | وحدة (SDAIA; it reads as "unit") |
| code | program text → الشيفرة; a code in coding theory → شفرة (شفرة بادئة); encoder → المشفر, decoder → مفكك التشفير; encoding → ترميز | رمز for program code; الكود in body text |
| performance | of a model → الأداء; of code → السرعة، الأداء الحسابي | |
| unit | of a network → وحدة; "unit gradient" → تدرج مقداره واحد | وحدوي |
| support (of a function) | الحامل | الدعم |

The ✗ forms above that come from earlier Arabic texts are the ones readers
have already seen, mostly in Taima's machine-translated edition: التلافيف
(254 times), التدرج الاشتقاقي, خريطة المعالم / الخصائص, الانتشار الخلفي,
التحيز for the bias term, الكمبيوتر (§5.1). A draft that uses them is
corrected, not glossed.

### 10.2 Translationese

| Pattern | Write instead | Example |
|---|---|---|
| Subject first in every sentence | Verb first (§1) | ✗ الطبقة الالتفافية تحسب … المخرجات تنكمش … ✓ تحسب الطبقة الالتفافية … فتنكمش المخرجات … |
| "It is important to note that", "Note that" word for word | Drop it, or لاحظ أن once in a while | ✗ من المهم أن نلاحظ أن الانحياز لا يخضع للتنظيم. ✓ الانحياز لا يخضع للتنظيم. |
| "Sure enough", "Clearly", "Simply" as fillers | وبالفعل، واضح أن; drop "simply" | "is simply the dot product" → ليس إلا الجداء النقطي |
| "using X", "via X", "based on X" as باستخدام every time | بـ on a class noun, or عبر، وفق، من | "using a pair of loops" → بحلقتين |
| "which …" as والذي / والتي after every noun | A new sentence or وهو / وهي | "… (*padding*), which we examine in …" → …، وسندرسه في … |
| Passive with an agent, تم / يتم, قام بـ / نقوم بـ | Active (§1) | ✗ يتم حساب الخسارة ✓ نحسب الخسارة |
| "let us" as دعونا | لنـ + jussive (§1) | ✗ دعونا نحسب ✓ لنحسب |
| Exercise "do X" as قم بـ + verbal noun | The imperative of the verb (§8) | ✗ قم بتنفيذ ✓ نفذ |
| English plural acronyms (CNNs, GPUs) | Class noun + acronym (§5.1) | ✗ تدرب CNNs ✓ تدرب شبكات CNN |
| "there is / there are" as يوجد / هناك every time | A full verb, or لـ | "kernels that have no center pixel" → نوى ليس لها بكسل مركزي |
| "we can" as نستطيع أن / يمكننا أن every time | The verb itself | "we can unroll the kernel" → نفرد النواة |
| Inanimate subject + "tells us", "allows us to" | من ذلك نرى، بذلك نستطيع | "This result tells us that depth helps." ✓ من هذه النتيجة نرى أن العمق مفيد. |
| "our model", "our edge detector" as نموذجنا every time | The noun with the article or a relative clause | "our hand-crafted edge detector" → كاشف الحواف الذي صممناه يدويًا |
| "it" whose noun is unclear in Arabic | Repeat the noun | "confirms what it found" → يؤكد ما عثر عليه النزول التدرجي |
| Idioms translated by image | The meaning | "under the hood" → داخليًا ; "brings us full circle" → يعيدنا إلى نقطة البداية ; "in one shot" → في مرة واحدة |
| English adverb order | The adverb after the verb phrase | ✗ تلقائيًا تكتشف ✓ تكتشف … تلقائيًا |

## 11. Settings

Existing keys (`languages.yml`, and `ar/strings.yml`, which `l10n strings ar`
creates):

- `languages.yml`: `dir: rtl` (set). No `check:` entry: the defaults are what
  this guide needs (`thousands: keep`, `numbers: warning`, `order: warning`).
  No `typography: {nbsp: true}`: its rules are written for German.
- `quarto.crossref-ch-prefix: "الفصل"`, `quarto.crossref-sec-prefix: "القسم"`,
  `quarto.crossref-fig-prefix` and `-fig-title: "الشكل"`,
  `quarto.crossref-tbl-prefix` and `-tbl-title: "الجدول"`,
  `quarto.crossref-eq-prefix: "المعادلة"`,
  `quarto.crossref-apx-prefix: "الملحق"`,
  `quarto.section-title-references: "المراجع"`. The installed Quarto (1.9.37)
  has no `_language-ar.yml`, so every one of these prints in English unless set.
- `book.exercises_heading: "تمارين"`, `book.discussions_link: "النقاشات"`
  (§7).
- `fonts.html.sans` with an Arabic family first and `fonts.html.google` to
  load it (for example `Noto Sans Arabic:wght@400;500;700`); `pdf.lang: null`
  until the right-to-left PDF exists.

Settings that need new code:

1. **Isolation of left-to-right inlines** in the RTL style sheet: `code`,
   `kbd`, `pre`, math containers `{direction: ltr; unicode-bidi: isolate}`.
   Without it `[1, -1]` in backticks is drawn as `[-1 ,1]` and adjacent code
   spans merge (§3.2).
2. **Emphasis without italics:** Arabic fonts have no italic and browsers slant
   the letters. Style `em` in the Arabic build upright (heavier weight or a
   colour).
3. **Checker, glossary match:** remove Arabic marks (U+064B–U+0652, U+0670) and
   tatweel from both sides before comparing (§2, §9.4). The glossary is
   normalized, but a translation that writes an optional mark (التعلّم) still
   fails the match today.
4. **Checker, digits:** an error for ٠–٩, ۰–۹, ٫, ٬ and ٪ in a translation
   (§6). Today `٣` for `3` only gives "numbers not found".
5. **Checker, adjacency:** an error for a reference role touching an Arabic
   letter (the build loses the reference; `cjk_adjacent: all` cannot be used,
   it also flags `$x$،`), and a warning for math or code touching one (§3.3).
6. **No-break space** after a free-standing و and after الـ، بـ، لـ، كـ before
   a token, so that a line does not end with the particle.
7. **PDF, slides, PowerPoint and Beamer in right-to-left** with Arabic fonts
   (DESIGN.md §7 names this as open work); no font key for Arabic PDF exists.
8. **Plain `3×3` outside math** (slide titles) is drawn right-to-left; the
   build could wrap such runs as it wraps code.

## 12. Decisions (author, 2026-11-27)

The author decided the open points of the draft of 2026-11-26 after three
evidence reports (§14): the two earlier Arabic editions of D2L, Arabic ML
writing on the web with the SDAIA/ICAIRE glossary as the primary reference, and
the prose of SDAIA and MBZUAI publications. Two rules govern the rest:
**established practice wins over SDAIA**, whose form then becomes an
alternative in the glossary; and **Taima's edition is weak evidence**, because
it reads as lightly post-edited machine translation. All rules and examples in
this guide follow the decisions. One point is open (16).

1. **Digits** (§6): accepted. Western 0–9, the decimal point and the
   thousands comma for the whole book. Every source agrees: no Eastern digits
   in Google (2768:0), Wikipedia (3911:0), SDAIA, MBZUAI or either edition.
2. **Marks** (§2): changed. A mark only where a word is ambiguous, in nouns
   and verbs alike (بُعد، معلَمة، يُعد، يحدّث); no systematic shadda on Form II
   and V verbs. SDAIA marks 9% of those verbs and MBZUAI 22%, neither
   systematically.
3. **Tanwin before the alif** (§2): accepted (صفرًا). Technical and teaching
   writing puts it there: ICAIRE 1150:9, Hsoub 9243:275, Google 9571:1355,
   Wikipedia 1086:399, Taima 3508:85. SDAIA's PDFs (847:6), MBZUAI news and
   the journals write it after.
4. **و before code, math, Latin words and digits** (§3.3): accepted, written
   separately. The more common form (Hsoub 152:30, Wikipedia 34:24, Taima
   160:118), although SDAIA and MBZUAI attach it; it also keeps pandoc from
   taking و into a label (``@sec-xو``).
5. **Carriers** (§3.3): changed. A class noun first (لنموذج GPT), as MBZUAI
   writes it 24 times against 6 carriers; بـ، لـ، كـ + space allowed before Latin
   or digits; never الـ + Latin (✗ الـ GPU); الـ + digit allowed.
6. **parameter → معلَمة, coefficient → معامل** (§10.1): accepted; the fatha is
   an ambiguity mark (rule 2). Google writes مَعلمة, IBM معلمات 75, d2l-ar
   المعلمات. SDAIA
   (مُعامِل) and MBZUAI (معامل 45:2) use one word for both; it is recorded in
   the glossary notes, not as an alternative, because the check would then
   accept the coefficient word for "parameter".
7. **Names** (§7): the two-tier rule for people stands (Arabic letters for
   names inside glossary terms, Latin for cited authors; mixing is fine).
   Companies and products follow the same rule as people, so they stay in
   Latin (Nvidia, Google), although SDAIA and MBZUAI write them in Arabic
   letters (author, 2026-11-27). Versioned model names (GPT-4o, Llama-8B),
   libraries and tools stay in Latin.
8. **المحول for the Transformer architecture** (§5.1): accepted. Taima
   writes المحول 217 times, SDAIA مُحوِّل, Wikipedia محول 101.
9. **Kernel plural** (§5.2): changed to نوى. Taima writes نوى for kernels and
   أنوية only for CPU cores; Google, IBM and Wikipedia write نوى.
10. **input, output** (§10.1): changed to المدخلات، المخرجات (SDAIA: بيانات
    مُدْخَلة (أو مُدْخَلات); Google, IBM, d2l-ar), not الدخل، الخرج, a Levantine
    engineering usage on which Wikipedia and Hsoub are split.
11. **vertical** (§10.1): changed to عمودي (Wikipedia 3:1 against رأسي,
    MBZUAI, SDAIA توسيع عمودي). The evidence is thin on both sides.
12. **code → الشيفرة** (§5.1): accepted, with الكود (SDAIA, Taima 120) and
    التعليمات البرمجية (IBM 9, MBZUAI 23) as glossary alternatives. Hsoub
    writes الشيفرة 200 times beside الكود 185, MBZUAI 8.
13. **Compound adjectives without the article** (§5.2): accepted. Taima 161
    bare against 4; SDAIA's definitions keep the article.
14. **English gloss** (§5.4): accepted, plain parentheses once per page.
    Wikipedia (433), Hsoub (269) and SDAIA print Arabic (English) the same way.
15. **Exercises** (§1, §8): accepted, masculine singular imperative; no
    source uses a double form (Hsoub 0, Google 0). Added: ✗ قم بـ.
16. **Reference prefixes** (§9.1): open. The prefixes with the article
    (الشكل 4، القسم 5.1) stand, as the Arab journal papers write them; whether
    equation references print المعادلة, and whether ``:citet:`` prints
    "et al." in Latin, is checked in the first Arabic build.
17. **token** (§10.1): changed to الرمز المميز (IBM 71, Google 24); the short
    الرمز stays in fixed compounds. SDAIA's وحدة is not used: it reads as
    "unit".
18. **precision (metric) → الإحكام** (§10.1): accepted; SDAIA has إحكام beside
    دقة (accuracy) and استدعاء (recall).
19. **Quotation marks** (§4): changed to “…”. MBZUAI writes “…” 1,479 times and
    «…» 39; SDAIA mixes them.
20. **Percent** (§6): `%` only, never ٪ (SDAIA 287:0, MBZUAI 192:9).
21. **encoder, encoding** (§10.1): المشفر / مفكك التشفير for encoder and
    decoder, ترميز for encoding, as the glossary has them (IBM مشفر 23, فك
    التشفير 28). SDAIA and Google use the ترميز root for both (مُرمِّز); not
    used, since Hsoub's المرمز is the tokenizer.
22. **Maximum likelihood** (§10.1): الإمكان الأعظم, one form.
23. **stack** (§10.1): تكديس for stacking layers, مكدس for the data structure.
    Neither SDAIA nor MBZUAI has the word; accepted by the author with the
    other thinly attested new entries (2026-11-27).
24. **Additions from the editions report** (§1, §5.1, §7, §10): ✗ دعونا
    (Taima 213×) → لنـ; ✗ قم بـ in exercises (140×); fewer يتم (782×); no
    Latin plural acronyms in prose (✗ CNNs); library and tool names in Latin
    (✗ بايثون، تينسرفلو، باي تورش، جوبيتر); a glossary term for notebook
    (دفتر); and a list of the ✗ variants readers have seen (§10.1).
25. **Conflicts from the log compaction** (author, 2026-11-27): symbol lists
    start with the symbol (§3.2); an all-Latin list of names or credits stays
    one Latin run (§3.2, §4); slide titles quote with “…” only (§4); headings
    that name kinds are definite (§8: الطبقات ذات المعلَمات); no emphasis added,
    not even on an English gloss (§4); names follow §7 unchanged (Arabic letters
    inside a glossary term: إنتروبيا شانون، جداء كرونيكر; otherwise Latin:
    متباينة Cauchy–Schwarz); an acronym the page never introduces is written
    out in Arabic (§5.4).

## 13. Glossary corrections (applied 2026-11-27)

The corrections proposed in the draft were applied to
`glossary/terms/ar.jsonl`, in two passes (2026-11-26 and 2026-11-27); details
and lists are in `glossary/notes/ar-style-guide-corrections.md`. Arabic now has
163 attested entries and 1,040 proposals among 1,203 concepts.

1. **Marks:** the optional marks are gone from every term (11 more entries on
   2026-11-27, for example معدل التعلم، دالة التفعيل، محول). Kept: معلَمة,
   بُعد, حالّ, مُشرِّط. Where a source spelled the term with marks, that
   spelling is an alternative.
2. **معلَمة with its fatha** in all parameter entries.
3. **Tanwin before the alif** (جزئيًا، حسابيًا، ليبشيتزيًا).
4. **Acronyms out of `preferred`** (SVD, LSTM, ViT, JIT, SAC, DQN, VAE, MMD,
   PPO); the acronym is in the note.
5. **Maximum likelihood:** الإمكان الأعظم throughout (تقدير / مقدر الإمكان
   الأعظم).
6. **Broken plurals as alternatives** (نماذج، دوال الخسارة، خرائط الميزات،
   أشكال، خسائر، قيود، نوى).
7. **New concepts:** input مدخلات، output مخرجات، parameter معلَمة، window
   نافذة، patch رقعة، edge حافة، stack تكديس، framework إطار عمل، code شيفرة،
   dense layer طبقة كثيفة، vertical عمودي، notebook دفتر. All except dense
   layer are common English words and are not enforced by the check.
8. **Encoder and encoding:** kept as they are (المشفر، مفكك التشفير; ترميز)
   by decision 21.
9. **Headwords without the article** (149 entries); التفاف keeps its ال, which
   belongs to the word.
10. **Decided terms:** token رمز مميز, precision إحكام, LSTM ذاكرة قصيرة المدى
    مطولة (SDAIA and Wikipedia; accepted by the author, 2026-11-27), kernel
    plural نوى (أنوية removed).
11. **SDAIA forms as alternatives** where practice won: معامل ضبط
    (hyperparameter), خاصية (feature), خلية عصبية (neuron), ترشيح
    (convolution), تقدير الأرجحية الأعلى (MLE).
12. **Taima's edition re-checked:** of the 89 entries it alone attested, 71
    are attested by another source (SDAIA/ICAIRE 17, Hsoub 21, Wikipedia 17,
    the CS 229 sheets 11, IBM 5) and 18 are proposals again.

## 14. Sources

Checked on 2026-11-26:

- The three test translations in `evals/ar-conv-layer/` (Fable, Opus, Sonnet),
  with their DECISIONS and NOTES; examples marked by name come from them.
- `tools/l10n/protect.py` and `glossary.py`, run on Arabic samples: `٣` is not
  matched to `3`; a glossary form with a shadda matches only its exact
  spelling; the article, attached particles and sound plurals are matched.
- The book's pandoc (`.venv-build`): ``ب@sec-x`` is not a reference and
  ``@sec-xو`` takes the و into the label; ``@sec-x،``, ``(@sec-x)``,
  ``«@sec-x»`` and (2026-11-27) ``“@sec-x”`` work; math, code and emphasis
  directly against an Arabic letter parse.
- Quarto 1.9.37 language files in `.venv-build`: none for Arabic.

Checked on 2026-11-27, for the decisions in §12:

- **SDAIA/ICAIRE glossary**, معجم الذكاء الاصطناعي والبيانات
  (icaire.org/ar/glossary; 1,444 headwords; with ICESCO and the King Salman
  Global Academy for Arabic). The primary reference for terms.
- **SDAIA and MBZUAI prose:** nine SDAIA documents (about 79,600 words) and
  109 MBZUAI Arabic news articles, counted for marks, tanwin, و, quotation
  marks, percent signs, names and terms.
- **Arabic Wikipedia:** 79 articles linked from 90 English ML articles.
- **Hsoub Academy:** 103 AI articles, including its Arabic Hugging Face and
  PyTorch documentation; a professional house style.
- **Google's Arabic ML glossary** (machine-translated, marked `x-mtfrom-en`)
  and **IBM Think** (nine topic pages, vendor localization): used for counts,
  with low weight.
- **Arab journal papers** (Tishreen, Latakia, Al-Balqa, Algiers) and the
  Syrian *معجم المصطلحات المعلوماتية* (2000): register, reference prefixes,
  older terms.
- **The Arabic CS 229 cheat sheets** (shervinea/cheatsheet-translation): a
  community bilingual translation.
- **The two Arabic editions of D2L.** Mohamed Ali's d2l-ar (GitHub, 2021,
  about 6,300 words): handwritten but idiosyncratic, heavily vocalized; its
  رمز means program code. Alaa Taima's *التعمق في التعلم العميق* (dlarabic,
  three PDF volumes, about 170,000 words): it reads as lightly post-edited
  machine translation (يتم 782, دعونا 213, قم بـ 140, التلافيف 254), so it
  counts as weak evidence. It is good for counting what an Arabic ML reader
  has seen, but it is not an authority on spelling or word choice.
  dlarabic.com's glossary is not imported.
- `glossary/terms/ar.jsonl` and `glossary/notes/ar-tr-de.md` of this
  repository.

Not checked:

- No Arabic university textbook was read. The register, number grammar and
  punctuation rules are stated from Modern Standard Arabic, with the sources
  above as the check.
- The UAE AI Office and QCRI were not reached (their sites timed out).
- The bidirectional display described in §3.2 and §11 follows the Unicode
  algorithm (UAX #9); it was reasoned, not rendered in a browser, and the
  Arabic build does not exist yet.
- Whether the site search ignores Arabic marks was not tested.

## 15. Conventions from the translated chapters

Rules from the decisions log, front matter to the attention chapter. They add
to §1–§10; chapter checks override units. Single terms are in the glossary and the lexicon (ar/decisions.md).

### 15.1 Headings and recurring titles

| English | Arabic | Note |
|---|---|---|
| Further Reading (page title); Recommended Resources; Visual introductions | قراءات إضافية; مصادر موصى بها; مقدمات بصرية | |
| Implementation; … from Scratch; Concise … | التنفيذ; التنفيذ من الصفر; التنفيذ الموجز | also as kickers |
| Preface; Introduction (parts of the book); Advanced (part overview) | التمهيد; المقدمة; المواضيع المتقدمة | |
| "the Basics part", "Parts 3 & 4", "Part 1: …" | جزء *الأساسيات*; الجزءان 3 و 4; الجزء 1: … | part names keep the English italics |
| Legacy … Library (build-only, unlisted) | مكتبة … القديمة (للبناء فقط، غير مدرجة) | `*-lib.md` pages |
| Legacy … Shims (build-only, unlisted) | دوال التوافق القديمة لـ … (للبناء فقط، غير مدرجة) | "Legacy Tokenizer Shims" → دوال التوافق القديمة للمجزئ |
| *Name, date* (signature line) | *بقلم Name، date* | starts with an Arabic word; months transliterated: نوفمبر، يوليو |

- **Heading patterns.** "X as Y" → X بوصفه / بوصفها Y (المربعات الصغرى بوصفها
  إسقاطًا); "X under squared loss" → X في حالة الخسارة التربيعية; "The
  Mechanics of X" → آلية عمل X; "Extensions to X" → التعميم على X; "Beyond X" →
  ما بعد X; "Limits of X" (shortcomings) → قصور X; "Taxonomy of X" → أصناف X;
  "X at a Glance" → X في لمحة.
- **A heading that defines a Latin-named term** puts the class noun first:
  "Masked Softmax" → عملية *softmax المقنعة*; in running text دالة softmax
  المقنعة.
- **Quoted section and page titles** are copied as the chapter index gives them;
  a list of them is joined by و (الانتباه والمحولات ونماذج فضاء الحالات).

### 15.2 Slides, kickers and table headers

These recur on every page; write them identically.

| English | Arabic |
|---|---|
| Check Your Understanding (kicker); [Check]; Concept Check: … (title); (answers) | التحقق من الفهم; [التحقق]; اختبار الفهم: …; (الإجابات) |
| Key Takeaways; Takeaway; Summary: X; Summary & Key Takeaways; Summary: Key Takeaways from X | أهم الخلاصات; الخلاصة; الملخص: X; الملخص وأهم الخلاصات; الملخص: أهم الخلاصات من X |
| Section Summary; Section Takeaways; Section Synthesis; Section & Chapter Synthesis; Check & Synthesis | ملخص القسم; خلاصات القسم; خلاصة القسم; خلاصة القسم والفصل; التحقق والتوليف |
| Motivation; Worked Example; (cont.); Test your understanding of X; Solution preview | الدافع; مثال محلول; (تابع); اختبار فهم X; لمحة عن الحل |
| Pitfall:; Pitfall 1: …; Diagnostic Pitfall; Fix:; Hint:; Measured:; Predict, then verify: | مزلق:; المزلق 1: …; مزلق تشخيصي; الحل:; تلميح:; بالقياس:; توقع ثم تحقق: |
| Next (§{…}):; Up next (§{…} Title):; Setup; Experimental Setup | التالي (§{…}):; التالي (§{…} العنوان):; الإعداد; إعداد التجربة |
| Code Implementation; Module Implementation; High-Level API; Model Definition; Shape Verification | التنفيذ في الشيفرة (✗ تنفيذ الشيفرة); تنفيذ Module; واجهة API عالية المستوى; تعريف النموذج; التحقق من الأشكال |
| Experimental Evidence; Empirical Probe; Open Questions; Macro Architecture; Optimization & Regularization | الدليل التجريبي; مسبار تجريبي; أسئلة مفتوحة; المعمارية الكلية; التحسين والتنظيم |
| Resource Accounting; Resource Ledger; Systems Bottleneck; X Trade-Offs; Decision Guide; Systems-Level Comparison | محاسبة الموارد; سجل الموارد; عنق زجاجة في الأنظمة; المقايضات المعمارية; دليل القرار المعماري; مقارنة على مستوى الأنظمة |
| Chapter Architecture; Chapter Study Strategy; Architecture and Roadmap of Chapter N; Fast track to Chapter N | معمارية الفصل; استراتيجية دراسة الفصل; معمارية الفصل N وخارطة طريقه; المسار السريع إلى الفصل N |
| Subtlety 1; Snippet A; (default); (result); tag [extended] | النقطة الدقيقة 1; المقطع A; (القيمة الافتراضية); (النتيجة); copied |
| Table: Model / Input / Params / MACs / Peak / Val Acc; Model / Metric; part / shape / parameters / total | النموذج / المدخلات / المعلَمات / عمليات MAC / ذروة / دقة التحقق; النموذج / المقياس; الجزء / الشكل / المعلَمات / المجموع |
| Table: Concept / Rule; Mathematical Form; Mathematical Expression; Key Property; Key Capability / Bottleneck; Practical Guideline | المفهوم / القاعدة; العبارة الرياضية; التعبير الرياضي; الخاصية الأساسية; القدرة الأساسية / عنق الزجاجة; الإرشاد العملي |
| Table: Role & Practical Takeaway; Role & Systems Takeaway; Technique | الدور والخلاصة العملية; الدور والخلاصة من منظور الأنظمة; التقنية |

- **Slash and ampersand in headers.** A slash stays a slash; & becomes و.
  "A / B" with a shared head noun shares it in Arabic: "high- / low-frequency"
  → عالي / منخفض التردد.
- **Enumerated titles.** "Pillar 1 — X", "Route 1 — X" → الركيزة 1، X (dash →
  comma; a colon stays); Roman series stay Latin (الاستعادة I / II); "Step 1 & 2"
  → الخطوتان 1 و 2; "Unit 1 / Unit 2" → الوحدة الأولى / الوحدة الثانية.
- **"Why …" titles** are questions with ؟, a lead-in "Why never …:" → سبب …:.
  Verb-led titles become noun phrases: "Notice the flaw:" → العيب الظاهر:;
  "Replacing A by B" → الاستعاضة عن A بـ B (the English order of A and B kept).
- **Tasks on slides** take المطلوب. With a colon in the English: "Work out …
  for each scenario:" → المطلوب: حساب … في كل حالة:; as a sentence without a
  colon: "Work out these 3 problems." → المطلوب حل هذه المسائل الثلاث.
  "Predict …" → المطلوب: توقع …; "You want …" → المطلوب …; "You must **lower**
  …" → الواجب هو **خفض** …. Drop "our" on slides: "our valley" → الوادي نفسه.
- **Slide enumerators** "(a)", "(b)" → في الحالة (a); "Model A / Model B" →
  النموذج A / النموذج B. Free-standing slide labels "Hypothesis A / B — …" take
  Arabic letters: الفرضية أ / ب، ….
- **Lead words before math at a block start** (§3.2): "**$k = 1$:**" →
  **عند $k = 1$:**; a formula answer takes its noun (النسبة $16 : 1$، الرتبة $\le d_h$);
  $\eta$ → المعدل $\eta$; $A \to B$ → الانتقال $A \to B$; "$n = …$" → العرض $n = …$.
- **A kicker or title starting with a number** is reordered: "[7,850 Learnable
  Parameters]" → [المعلَمات القابلة للتعلم: 7,850]; "[40 Steps vs. 139 Steps]" →
  [الخطوات: 40 مقابل 139]; "1,000 Valley Runs" → جولات الوادي الـ 1,000.
- **Slide shorthand is written out** (LR → معدل التعلم، RF → مجال الاستقبال،
  params → معلَمات، std → انحراف معياري); acronyms that are names (BN, GAP, MAC,
  FLOP) stay Latin with a class noun.

### 15.3 Exercises

- **"Exercise :exref:`…`" and "problem :exref:`…`"** → التمرين :exref:`…`,
  also in parentheses; "(Exercise 1)" → (التمرين 1); "Problems 29–30" →
  المسألتان 29 و 30. A bare reference takes the noun too: "the 70B model of
  :exref:`…`" → نموذج 70B من التمرين :exref:`…`.
- **Parts.** "part (a)" → الجزء (a); "from (b)", "your answer to (b)" → من
  الجزء (b)، إجابتك عن الجزء (b); "the previous part" → الجزء السابق; a part
  letter after a reference is copied (:exref:`…`a). (i) … (v) and (a) … (d) keep
  Latin markers, with a free-standing و before the last. "(Optional.)" →
  (اختياري.).
- **Exercise titles** are noun phrases with the English full stop: "Where the
  savings come from." → مصدر التوفير.; "Why reshuffle." → لماذا نعيد الخلط. (no
  ؟); "X ablation" → استئصال X.
- **"Suppose you …"** → افترض أنك …; "Prove" → برهن; أثبت only for a cited
  result in body text: أثبت :citet:`…` أن ….
- **"Replace A with B"** as an instruction → ضع مكان A B (✗ استبدل بـ, which
  readers take in the reverse sense): "Replace the body of `f` by X" → ضع مكان
  جسم الدالة `f` التعبير X.
- **Imperatives with a second reading carry their mark** (§2): فُكّ، عُدّ، ثبّت،
  وصِف، عمّم، ركّب، كامِل ("integrate"; كامل "complete"); none on أعط، اكتب، قس.
- **Commands in body text** (installation steps, procedure lists) are masculine
  singular imperatives (ثبّت، نزّل، **قدّر:**); on slides the same steps are
  verbal nouns (تقدير، ضبط). Statements about the book keep نحن.

### 15.4 Credit lines and reading lists

Credit lines are italic when they stand alone; titles stay English; an Arabic
word separates the source's Latin name from its Latin link text.

| Source | Arabic |
|---|---|
| Kaggle Learn course | *مقتبس بتصرف من دورة [Course](url) في Kaggle Learn، التمرين N.* |
| Course assignment | *مقتبس بتصرف من مقرر CMU 10-714، الواجب [Homework 2](url)، السؤال 2.* ; a generic link text is translated: مقرر Stanford CS224N، [الواجب 4](url)، المسألتان 1(b) و 1(c); MIT 18.05، [مجموعة المسائل 7](url)، المسألة 4 |
| Book | *مقتبس بتصرف من كتاب [Title](url) لمؤلفه Author، الفصل N.* ; several authors: لمؤلفيه A و B; a draft: مسودة الطبعة الثالثة |
| Project, repository | مشروع [micrograd](url) لمؤلفه Andrej Karpathy، الملف `file` |
| Collection | مجموعة [Title](url) التي أعدها Author، المسألة N |
| Documentation | وثائق PyTorch، صفحة [title](url) |
| Tutorial section; two sources | درس [title](url) الذي أعده Author، قسم “section”; … من دليل PyTorch التعليمي [title](url) ومن دفتر JAX المعنون [title](url) |

- **Reading-list items start with a class noun** before the author or link:
  كتاب، مقرر، سلسلة، مقالة (blog post)، دراسة / ورقة (paper)، درس (video
  lesson)، أداة (web tool)، دليل، صفحة; a named title takes المعنون / المعنونة:
  مقالة Gabriel Goh المعنونة *[Title](url)*. "(paywalled)" → (الوصول مدفوع).
- **A :citet: with its English title** is the subject: يُعد :citet:`…`
  (*Title*) المرجع …; "A and :citet:`…`" → … وكذلك :citet:`…`; an acronym and
  its citation in one parenthesis: (**L-BFGS**، وقد اقترحها :citet:`…`).
- **Slide citations** stay Latin: (Bernstein & Newhouse, 2024). After a math span
  or as a clause, add وفق / لدى: (وفق Yang et al., 2022); "Polyak's method
  (1964)" → الطريقة لدى Polyak (1964); "Wen et al. (2024) explain" → تفسير Wen et
  al. (2024) لـ …. A kicker that is only a citation takes بحث: [بحث Loshchilov
  و Hutter (2019)].
  A lead-in made of two cited groups takes the dual: "Cho et al. & Sutskever et
  al. (2014):" → بحثا Cho et al. و Sutskever et al. (2014):.
- **A book named in body text with a Latin citation** takes كتاب before its
  link, and the citation is opened so that an Arabic word separates the two
  Latin runs: "[Title](url) (Austin et al., 2025)" → كتاب [Title](url) لمؤلفيه
  Austin et al. (2025); an organization: الصادر عن Hugging Face (2025).
- **A Latin name directly before :citet:** gets an Arabic word between them:
  (اختصار *Momentum Orthogonalized by Newton–Schulz*، وقد اقترحه :citet:`…`).

### 15.5 Figures, captions and tables

- **Panels:** "(left) / (middle) / (right)" → (يسارًا) / (في الوسط) / (يمينًا);
  "panel (a)" → اللوحة (a). **Colours** with بـ: (بالأسود)، (بالألوان); line
  styles: (الخط المتصل)، (الخط المتقطع)، (الأحمر المتقطع).
- **Labels printed in a figure or by code** (Fashion-MNIST classes, heads/tails)
  are translated in the text with the English once per page: القميص قصير الكمين
  (T-shirt), حذاءً رياضيًا (sneaker، الفئة 7), الصورة (heads); a class named in
  quotes as the code prints it stays Latin after الفئة.
- **Table cells:** a model or method name in the first cell takes its class
  noun, outside the bold (المحسن **AdamW**، وحدة GPU 0); a code label is copied
  (`VGG-style, narrow`، `d2l.Module`). "**Framework** `layer`" → الطبقة `layer`
  في **Framework**; "A + B" between code → A مع B; "None" → لا شيء; "Standard
  for" → الخيار المعياري لـ; "predicted sick" → التنبؤ: مريض; "$3\times 3$ conv,
  64 to 128, stride 2" → التفاف $3\times 3$، من 64 إلى 128، بخطوة انزلاق 2.
- **Architecture comparison tables** (model families by row): every cell names
  its class noun (انتباه GQA، طبقة RMSNorm، ترميز RoPE، طريقة YaRN، تطبيع
  QK-norm); a bare name stays only under a header that carries the noun (SwiGLU
  under شبكة FFN). "GQA $32{:}8$" → انتباه GQA بنسبة $32{:}8$; "window 4096" →
  نافذة عرضها 4096; "pre" / "pre and post" → تطبيع مسبق / مسبق ولاحق; "128
  experts, 8 active" → 128 خبيرًا، النشطون منهم 8. Parameter counts in cells
  (46.7B / 12.9B) are copied, not turned into مليار.
- **Hardware and communication tables.** A cell that names a link takes the
  class noun وصلة and بمعدل between the name and the rate: "NVLink 4, 450 GB/s"
  → وصلة NVLink 4 بمعدل 450 GB/s. A cell that counts collectives keeps the
  digit before the class noun: "2 All-Gathers + 1 Reduce-Scatter" → 2 عملية
  all-gather + 1 عملية reduce-scatter; slide shorthand (RS) is written out.

### 15.6 Class nouns and names

| Kind | Class noun | Example |
|---|---|---|
| CNN architectures | شبكة (feminine) | شبكة LeNet، شبكة على نمط VGG ("VGG-style") |
| Language models | نموذج (masculine) | نموذج GPT-2 small، النموذج Kimi K2 |
| Optimizers | محسن (masculine); SGD → خوارزمية; AMSGrad, Yogi, SVRG → طريقة | محسنا Adam و AdamW، بالمحسن AdamW |
| Layers, activations | طبقة; دالة | طبقة LayerNorm، دالتا التفعيل ReLU و GELU |
| Recurrent and state space mixers | طبقة for the layer; نموذج for the trained model or an experiment arm; المازج for the mixer of a block; شبكة for LSTM and GRU; كتلة for a block; معمارية for a published design (Based, xLSTM, TTT, Titans); خلية mLSTM; إطار HiPPO; طريقة Longhorn | طبقة minGRU، طبقة S4D، طبقة Mamba، طبقة Gated DeltaNet، النموذج `MinGRU`، المازج `GatedDeltaNet`، شبكتا LSTM و GRU ✗ وحدة GRU (reads beside وحدة GPU) ✗ شبكة minGRU |
| Techniques, methods | طريقة in prose; خوارزمية for FlashAttention on slides and for online softmax | طريقة Mixup، طريقة im2col، تنعيم Kneser–Ney |
| Positional schemes | ترميز; NoPE → أسلوب | ترميزا RoPE و ALiBi |
| Python class (also a page-defined "module"), function, decorator | الصنف، الدالة، المزخرف; وحدة only for a Python module or a network unit | الصنف `Trainer`، "the Trainer" → الصنف Trainer |
| Number formats | الصيغة / بصيغة; in backticks النوع | في الصيغة fp16، بصيغة bf16، بالنوع `float32` |
| Hardware, memory | وحدة، بطاقة، ذاكرة | بطاقة RTX 4090، ذاكرة HBM |
| Theorems, inequalities | مبرهنة، متباينة، توطئة | مبرهنة Pythagoras، متباينة Hoeffding |
| Quoted example word; a letter | الكلمة; الرمز | الكلمة *gave*، الكلمتان *Alice* و *Bob* |
| Books, novels | كتاب، رواية | رواية *The Time Machine* للكاتب H. G. Wells |
| Games, datasets | لعبة; بيانات / مجموعة بيانات | لعبة Go، بيانات Fashion-MNIST |

- **Agreement follows the class noun**, also for the bare name later (يحقق Muon،
  خوارزمية BPTT المقتطعة). The noun is needed at the start of a block, clause or
  cell, before a particle and between Latin runs; not after a free preposition
  or on a later mention (في Muon، في fp32).
- **Write elliptic lists out:** "ResNet-50, -101, and -152" → ResNet-50 و
  ResNet-101 و ResNet-152 (a bare "-101" reads wrongly in right-to-left text).
- **Compound person names keep the source dash or double hyphen:**
  Kullback–Leibler، Nadaraya--Watson، Black--Scholes; a pair term in an equation
  label takes an en dash between Arabic words (حد المحتوى–الموضع); body text
  the descriptive form (الحدان التقاطعيان بين المحتوى والموضع).
- **A product named with its maker's possessive** puts the class noun and the
  product first, then من and the maker: "NVIDIA's Nsight Systems" → أداة Nsight
  Systems من NVIDIA; "Apple's M3 Ultra" → الجهاز M3 Ultra من Apple.

### 15.7 Numbers and units

- **K, M, B, T after a number in prose become words:** "8B parameters" → 8
  مليارات معلَمة; "43M-param" → 43 مليون معلَمة; "~1.5B" → نحو 1.5 مليار; "276M
  multiply-adds" → 276 مليون عملية ضرب وجمع. Reason: B reads as bytes next to GB.
  Model names keep the Latin (Llama 3 8B); math spans are copied
  ($542\text{K}$، $3.38\textrm{M}$).
- **A unit written as a word becomes an Arabic word; a symbol is copied:** "40
  gigabytes" → 40 غيغابايت, "40 GB" → 40 GB; "milliseconds" → ميلي ثانية; "hertz"
  → الهرتز, Hz → Hz. "per" → لكل: "2.1 bits per byte" → 2.1 بت لكل بايت; "/param"
  → لكل معلَمة; on slides the slash stays: رمز مميز/كلمة.
- **Ranges between math spans** in prose: بين $5$ و $10$ or من … إلى …; "135K
  to 141M parameters" → من 135 ألف إلى 141 مليون معلَمة; chapter ranges
  "(:numref:`a`--:numref:`b`)" → (من :numref:`a` إلى :numref:`b`).
- **Quantity phrases with a math span** take a noun: "a $64\times$ range" →
  مدى قدره $64\times$; "capped at $2.5$" → بسقف مقداره $2.5$.
- **A multiplier in a math span after a comparative** in prose takes بمقدار or
  بنحو: "$24\times$ fewer wires" → عدد الأسلاك أقل بمقدار $24\times$; "~$3\times$
  more" → أكبر بنحو $3\times$; a speedup: تسريع قدره $1.6\times$.
- **Time units written as words keep the prefix as a separate word**, like ميلي
  ثانية: ميكرو ثانية (ميكرو ثوان)، بيكو ثانية (✗ ميكروثانية); the symbols ms, µs,
  ns are copied.
- **A compound number above 100 governs its noun by its last two digits**
  (this refines §5.2): 03–10 the plural in the genitive (103 خطوات), 11–99 the
  singular in the accusative (128 بايتًا، 512 رمزًا مميزًا، 1,024 بتًا، 4,096
  رمزًا مالئًا); only round hundreds, thousands and millions take the singular
  in the genitive (200 رمز مميز، 1,000 خطوة). A following adjective agrees
  (1,024 تسلسل اختبار جديدًا). Units written as Arabic words inflect like any
  noun (✗ نوافذ من 128 بايت).
- **"256K-token" and "a 256K context length"** in prose and captions → 256 ألف
  رمز مميز، طول سياق قدره 256 ألف رمز مميز (✗ 256K in body text); a count
  written with a space in a table cell ("4.3 M") is copied, and two of them are
  joined by و.

### 15.8 Glosses, parentheses and quotations

Additions to §5.4:

- **Keep a per-page list** of glossed terms; a term defined on an earlier page
  is not glossed again; a chapter index glosses only its headline terms.
- **English plural acronyms in a gloss become singular:** (multilayer
  perceptron, MLP); a method name's expansion: Arabic, then English (*الانتباه
  بانحيازات خطية*، Attention with Linear Biases).
- **A synonym in the source** ("the *label* (or *target*)") → *الوسم* (label)،
  ويسمى أيضًا *الهدف* (target), so that two parentheses do not touch.
- **Inside one parenthesis**, Arabic commas join a gloss to more glosses or to
  a reference: (*التنبؤ الأدائي*، performative prediction، *وحلقات التغذية
  الراجعة*، feedback loops); (reinforcement learning، :numref:`…`); an
  abbreviation: (maximum pooling، واختصاره *max-pooling*).
- **Gloss an Arabic term that could be misread** even if it is common:
  الانزياح السلبي (passive shift), so سلبي is not read as "negative".
- **Quoted English text stays English** inside “…” where code, a figure or the
  argument depends on its words (a prompt, a dataset sentence, “Hey Siri”,
  *“Alice gave Bob the book”*); a sentence that only illustrates is translated:
  "dog bites man" → “الكلب يعض الرجل”. A well-known phrase is translated with
  the English once: “علينا أن نتعمق أكثر” (we need to go deeper). Words printed
  by code stay Latin in quotes: الحشو “same”.
- **An illustrative English sentence about an invented person** (*"Mira is the
  ship's engineer"*) is translated inside “…” with the person's name kept in
  Latin script: *“Mira مهندسة السفينة”*. Place names in such an example are
  written in Arabic (Paris → باريس، Rome → روما).

### 15.9 Notation lists, math and code in prose

- **Notation entries** (`* $x$: عدد قياسي`): several symbols before the colon
  are joined by a free-standing و; English "a" / "the" become an indefinite /
  definite noun ("a scalar" → عدد قياسي, "the identity matrix" → مصفوفة الوحدة).
  Openers: "introduced in" → ترد هذه الرموز أول مرة في …; "these symbols belong
  to" → تخص هذه الرموز ….
- **Symbol-modified adjectives** become a noun phrase:

| English | Arabic |
|---|---|
| $n$-dimensional vector; $d$-dimensional subspace | متجه بُعده $n$; فضاء جزئي بُعده $d$ |
| $k$-th order tensor | موتر من المرتبة $k^\textrm{th}$ |
| degree-$k$ polynomial | كثير حدود من الدرجة $k$ |
| $K$-fold cross-validation | التحقق المتقاطع بـ $K$ طية |
| the $i$-th convolution (no math superscript) | الالتفاف رقم $i$ |
| $d'$-wide layer; a 256-to-256 layer | طبقة عرضها $d'$; الطبقة التي تنقل 256 قناة إلى 256 قناة |
| the $K$ held-out losses; the first `num_train` windows | الخسائر المستبعدة وعددها $K$; النوافذ الأولى وعددها `num_train` |

- **Hyphenated compounds** become adjective + noun: "fixed-input model" → نموذج
  ثابت المدخلات; "heavy-tailed" → ثقيل الذيل; "noise-limited" → محدود بالضوضاء.
- **A display equation followed by a counted noun:** the block after the
  equation starts with the noun (معلَمة قابلة للتعلم.); a following "and
  concatenates …" → ثم يربط ….
- **Code named in prose** takes its class noun as the object: "sweep
  `d2l.lr_sweep`" → نجري المسح بالدالة `d2l.lr_sweep`; "save `X`" → يحفظ الموتر
  `X`; "a fresh `X`" → نموذج `X` جديد; "toggling `norm`" → تبديل قيمة `norm`.
  Tab note "In JAX it is `x`" → في JAX هذا … هو `x`.
- **A keyword option or an experiment label in backticks** is copied after its
  noun: "`pos='learned'`:" as a lead-in or a named option → الخيار
  `pos='learned'`; arm names (`no balancing`, `frozen router`, `dense`) → الإعداد
  / الطرف / النموذج `…`; a plot colour → المنحنى `C0`.
- **`if`--`else`** and a run of names joined by slashes (RandAugment/Mixup/CutMix)
  are copied as one left-to-right run; "`a` / `b`" → `a` أو `b`.

### 15.10 Consistency

- **Reuse the earlier page's rendering** wherever the English recurs; the
  chapter index decides the chapter's terms, a chapter check corrects earlier
  units in place, and may keep a chapter-specific term (تمديد السياق in
  attention, إطالة السياق in the RNN chapter).
- **Agreement inside a fixed term stays the same in the plural:** the
  adjectives of المسح agree with المسح, also after عمليات (عمليات المسح
  التجميعي المتوازي ✗ عمليات المسح التجميعية المتوازية), so that the page title,
  the slides and the chapter index print one form.
- **Marks that drafts keep adding** and §2 does not allow: كلًا، أيًا (✗ كلًّا،
  ✗ أيًّا), المكون (✗ المكوّن), وصلة تخط (✗ تخطٍ), دوال (✗ دوالَّ).
- **Source defects** (a math span cut across a paragraph, "Theorem" on a slide
  against "Proposition" in the body) are translated as they stand; the English
  is fixed, not the Arabic.
