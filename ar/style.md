# Style guide: Arabic (ar)

Draft of 2026-11-26 for the author's review; no native reader has confirmed it
yet. Translators read this with every unit (TRANSLATING.md §7–§8). Each rule
has one reason and one example (English → Arabic, or ✗ wrong / ✓ right). Rules
marked "(to confirm)" are decided here and listed again in §12.

Order of authority: the decisions log, then this guide, then the glossary. The
glossary says *what* a term is; this guide says *how to choose, spell and
inflect*. Only 167 of the 1,191 Arabic glossary entries are verified; the rest
are proposals, and §13 lists the entries this guide's rules contradict.

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
| we + action in the text | first person plural verb, no pronoun | "We minimize the loss with SGD." → نصغّر الخسارة بخوارزمية SGD. ✗ نحن نصغّر … |
| "let us", "consider", "suppose" | لـ + jussive: لنأخذ، لنفترض، لننظر | "Consider locating vertical edges." → لنفترض أننا نريد تحديد مواضع الحواف الرأسية. |
| "let $x$ be …" | ليكن / لتكن + the noun + the symbol | "Let $\mathbf{X}$ be a matrix." → لتكن $\mathbf{X}$ مصفوفة. |
| "we will see", "we will use" | present tense; سـ only for a later section; never سوف | "In this section, we implement …" → في هذا القسم ننفّذ … ; "We will return to this in …" → سنعود إلى ذلك في … |
| "you" = anyone ("you can verify") | نحن, or an impersonal verb (يمكن، يكفي) | "You can verify this numerically." → يمكن التحقق من ذلك عدديًا. |
| "notice", "recall", "keep in mind" | masculine singular imperative, as often as the English and no more: لاحظ أن، تذكّر أن; or نحن | "Keep in mind that $r$ measures …" → لا يغب عن بالنا أن $r$ يقيس … |
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
  ✓ تحسب الطبقة الالتفافية الارتباط المتقاطع بين دخلها ونواة متعلَّمة.
- **No إن as a sentence opener by habit.** At most once per paragraph, where
  the sentence states a claim. ✗ إن طرح الجار الأيمن … يعطي صفرًا ✓ طرح الجار
  الأيمن … يعطي صفرًا.
- **No "تم / قام بـ" + verbal noun.** Use the verb, active with نحن or a true
  passive. ✗ تم تدريب النموذج على ImageNet ✗ قمنا بتدريب النموذج ✓ درّبنا
  النموذج على ImageNet ✓ دُرّب النموذج على ImageNet.
- **No agent with a passive.** ✗ … تُحسب من قِبل الطبقة / بواسطة الطبقة
  ✓ تحسبها الطبقة.
- **Generic nouns take the article.** English bare plurals and mass nouns that
  mean the kind are definite in Arabic. "Gradient descent finds kernels" →
  يجد النزول التدرجي الأنوية. "Convolutional layers are …" → الطبقات
  الالتفافية ….
- **Non-human plurals agree in the feminine singular.** الطبقات الالتفافية
  تتعلّم، النماذج الكبيرة، هذه المصفوفات. ✗ الطبقات يتعلّمون.
- **At most three nouns in a genitive chain (إضافة).** Break a longer one
  with لـ, في or a relative clause. "the shape of the output of the first layer
  of the network" ✗ شكل خرج طبقة الشبكة الأولى ✓ شكل الخرج الذي تنتجه الطبقة
  الأولى في الشبكة.
- **Lists repeat و.** Short items are joined by و alone; a comma plus و only
  between clauses. "the weight, the bias, and the learning rate" → الوزن
  والانحياز ومعدل التعلم ✗ الوزن، الانحياز، ومعدل التعلم.
- **One main clause and at most two subordinate clauses** per sentence. Split
  English sentences at "which", "where", "so that" (TRANSLATING.md §3 allows
  splitting within a block).

## 2. Spelling and marks

**Decision: unvocalized text. Nouns and terms carry no marks; verbs of Form II
and Form V carry their shadda; a few fixed cases are marked against misreading
(to confirm).** Reasons: Arabic textbooks, newspapers and Arabic Wikipedia are
printed without vowels; verbs are where the bare spelling is another word
(يحدث "happens" / يحدّث "updates", يطبع "prints" / يطبّع "normalizes", يعلم
"knows" / يعلّم "teaches"); nouns and terms are what a reader searches for,
without marks, and what `l10n check` compares as strings (معدّل ≠ معدل). The
three test translations already wrote the shadda on most verbs and on few
nouns, each a little differently; this rule fixes the line.

| Item | Rule | Example |
|---|---|---|
| Tanwin of the accusative | Always written, on the letter before the alif. | صفرًا، تلقائيًا، مباشرةً ✗ صفرا ✗ صفراً |
| Passive verbs | A damma on the first letter when the active reading also fits. | يُكتب، تُسمّى، دُرّب، أُعطي |
| Passive participles of Form II and V | Shadda with its fatha where the active participle is also a word: مدرِّب "trainer", متعلِّم "learner", مصمِّم "designer". | نواة متعلَّمة (learned), نموذج مدرَّب (trained), مصمَّم يدويًا |
| Verbs of Form II and Form V | Shadda on the doubled consonant, in every finite form and in the imperative. | ننفّذ، نصغّر، يبيّن، يحدّث، يتعلّم، تذكّر، صمّمناه |
| المعلَمة | Always with the fatha (parameter, not المعلّمة "the teacher"). | المعلَمات الفائقة |
| Nouns, adjectives, verbal nouns, terms | No shadda and no short vowels, also where the word is a participle used as a noun or adjective. | التعلم العميق، معدل التعلم، الخطي، دالة، المحول، المشفر، المولد، طبقات مكدسة ✗ التعلّم ✗ الخطّي ✗ دالّة |
| Hamza | Write hamzat al-qat' and its seat. | إلى، أن، إن، أو، الإزاحة ✗ الى، ان، او |
| ى and ي, ة and ه | Never interchanged. | في، على، مصفوفة ✗ فى ✗ مصفوفه |
| Tatweel (ـ) | Only in the carriers الـ، بـ، لـ، كـ (§3.3); never to stretch a word. | |

A verified glossary entry spelled with optional marks (معدّل التعلّم, دالّة
التفعيل, الانحدار الخطّي) is written by this rule, without them. The check then
reports the term as missing; name that once per unit in NOTES (§9.4) until the
glossary is normalized (§13).

## 3. Right-to-left text with left-to-right parts

### 3.1 What stays in Latin script

Inline code, math, reference roles, URLs, and the names and acronyms of models,
libraries, datasets, optimizers, metrics and hardware (PyTorch, ResNet, BERT,
Adam, ReLU, softmax, GPU, Fashion-MNIST, im2col) stay as in English
(TRANSLATING.md §4, §7). Never write them in Arabic letters (✗ باي تورش،
✗ ريزنت، ✗ جي بي يو). Terms follow §5.

### 3.2 Order: how to write a sentence that displays correctly

The browser orders each line by the Unicode bidirectional algorithm. Arabic
letters set the direction; Latin letters form left-to-right runs; digits,
spaces and punctuation take the direction of their neighbours. Never insert
direction marks or other invisible characters (U+200E, U+200F, U+202A–U+202E,
U+2066–U+2069, U+061C): `l10n check` rejects the unit. Arrange the words
instead:

| Rule | Reason | Example |
|---|---|---|
| **Start every block with an Arabic word** (a verb, a preposition, or the class noun of the thing named): paragraphs, list items, captions, headings, slide titles. | A block that starts with a Latin letter is shown left-to-right by every tool that guesses the direction from the first letter (editors, diffs, exported slides). A class noun (الدالة، المكتبة، الشبكة، النموذج، الصنف) is grammar, not added content. | "`Conv2D` registers both parameters" → يسجّل الصنف `Conv2D` المعلَمتين كلتيهما ; "PyTorch's `nn.LazyLinear(1)` requires …" → تحتاج الطبقة `nn.LazyLinear(1)` في PyTorch إلى … |
| **Put an Arabic word between two left-to-right items** (code, a Latin name, a number with its unit). Never leave only a space, a comma or a bracket between them. | Two Latin runs with only neutral characters between them join into one left-to-right run, which then reads in the opposite order to the sentence. | "the input tensor `X` and kernel `K`" → موتر الدخل `X` والنواة `K` ; "`X` and `Y`" → `X` و `Y` ; "Adam, RMSProp, and SGD" → Adam و RMSProp و SGD ✗ Adam، RMSProp، SGD |
| **Name the thing, then give its symbol.** Where two math spans of one noun follow each other, join them with هو / هي, ذات, بحجم or المسمى. | The noun carries case, gender and the article, which a symbol cannot; and the protected tokens keep the English order, so the check stays silent. | "a $3 \times 3$ input tensor `X`" → موتر دخل $3 \times 3$ هو `X` ; "the kernel $\mathbf{K}$ be a $k_\textrm{h} \times k_\textrm{w}$ matrix" → النواة $\mathbf{K}$ مصفوفة أبعادها $k_\textrm{h} \times k_\textrm{w}$ |
| **Keep a number with its Latin unit or name in the English order**, as one piece. | It is one left-to-right run and displays as written. | 24 GiB, 5 ms, Python 3.12, GPT-2, RTX 4090 |
| **Write ranges in prose with words.** Keep the en dash only in slides, tables and captions. | In right-to-left text `40–55` is drawn with 40 on the right, `40-55` with 40 on the left; من … إلى … cannot be misread. | "40–55 epochs" → من 40 إلى 55 حقبة ; table cell: 40–55 |
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
| و before a token or Latin word | Written alone, with a space on both sides. | `X` و `Y` ; و $-1$ حيث … ; PyTorch و JAX ✗ و$-1$ ✗ و`0` |
| و، ف، ب، ل، ك and ال before an Arabic word | Attached as always, no space. | والانحياز، بالنزول التدرجي، للطبقة ✗ و الانحياز |
| ب، ل، ك before a token or Latin name | Attach the particle to a class noun; or use a free-standing preposition (في، من، إلى، على، عن، مع، بين، عبر، مثل، باستخدام). | "with `corr2d`" → بالدالة `corr2d` ; "in PyTorch" → في PyTorch ; "of $\mathbf{X}$ with $\mathbf{K}$" → بين $\mathbf{X}$ و $\mathbf{K}$ ; "to $\mathbf{w}$" → إلى المتجه $\mathbf{w}$ |
| Last resort (to confirm) | The carrier with tatweel and a space: بـ، لـ، كـ. Only where no class noun and no free preposition fits. | لـ $\mathbf{X}$ |
| The article with a Latin name or acronym | None. A class noun, the genitive or the context makes it definite. | "the GPU" → وحدة GPU ; "GPUs" → وحدات GPU ; "the Transformer block" → كتلة المحول ✗ الـ GPU ✗ الـGPU |
| The article with a number written in digits | الـ + space, for a definite counted noun. | "all 42 output entries" → عناصر الخرج الـ 42 كلها |
| Punctuation | No space before ، ؛ ؟ : . ! and before a closing bracket; one space after. | $(i, j)$، ثم … |
| Emphasis markers | Around the whole written word, with its article, attached particle and suffix; never inside a word. | *الحشو* ; *بالنزول التدرجي* ; *مجال استقباله* ✗ ب*النزول* ✗ *استقبال*ه |

## 4. Punctuation

| Item | Rule | Example |
|---|---|---|
| Comma, semicolon, question mark | Arabic ، ؛ ؟ in Arabic text. The Latin `,` `;` `?` only inside math, code, numbers (10,000) and an all-Latin parenthesis. | هل يستطيع النزول التدرجي أن يستعيد النواة؟ ; (convolutional neural network, CNN) |
| Full stop, colon, exclamation mark, parentheses | `.` `:` `!` `( )`. Brackets are typed in logical order; the display mirrors them. | |
| Comma after a fronted phrase | None. Do not copy the English comma after "In this section," or "In :numref:`x`,". | ✓ ``في :numref:`sec_why-conv` اشتققنا الطبقة الالتفافية`` ✗ ``في :numref:`sec_why-conv`، اشتققنا …`` |
| Comma before و | None between words; allowed between clauses. | ننفّذ الارتباط المتقاطع، ونختبر كاشف الحواف، ونبيّن … |
| Quotation marks | «…»; inside them “…”. Never "…" in Arabic text. | "we call cross-correlation "convolution"" → نسمّي الارتباط المتقاطع «التفافًا» |
| Quotes in attributes | A slide `title="…"` is delimited by ASCII `"`; inside it use «…» only. | `title="ما معنى «عميق»؟"` |
| Dash | The English em dash `—` becomes a comma, a colon or parentheses. No `—` and no spaced hyphen. | "… during forward propagation—borrowing the neurophysiology term" → … أثناء التمرير الأمامي، وهو مصطلح مستعار من فسيولوجيا الأعصاب |
| Colon before a display equation | Keep it where the English block ends with one. | … لنحصل على عنصر الخرج عند $(i, j)$: |
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
| No glossary entry | The term of Arabic university textbooks or Arabic Wikipedia; record it in DECISIONS. If there is none, an Arabic description with the English in parentheses; not a transliteration. | ✗ كيرنل، ✗ باتش، ✗ دروب آوت، ✗ توكن، ✗ ترانسفورمر |
| Names of models, architectures, optimizers, libraries, datasets, metrics; acronyms; code | Latin script; never translated in part. | ResNet, BERT, Adam, AdamW, PyTorch, ImageNet, BLEU, ReLU, softmax, SGD, MLP, CNN, LSTM, GPU |
| Latin name + Arabic noun | The Arabic class noun comes first, without the article (it is definite by the genitive); an adjective follows the name. | خوارزمية Adam، دالة ReLU، مكتبة PyTorch، مجموعة بيانات ImageNet، نموذج BERT، درجة BLEU، تعلم Q ; "a deep CNN" → شبكة CNN عميقة |
| Plural or dual of an acronym | On the class noun; never a Latin `s`. | "GPUs" → وحدات GPU ; "two GPUs" → وحدتا GPU ; "CNNs" → شبكات CNN |
| "Transformer" | The architecture is the glossary's المحول (a common noun); product and model names keep Latin (Vision Transformer (ViT) on first mention, then ViT). (to confirm) | مشفر المحول، كتلة المحول |

Pan-Arab words, not regional ones: حاسوب (✗ كمبيوتر), دالة (✗ تابع), شبكة
عصبية (✗ عصبونية), مبرهنة for a theorem (نظرية is a theory), الشيفرة
(البرمجية) for program code (to confirm; ✗ كود), ملف، مكتبة، إطار عمل.

### 5.2 Inflecting a glossary term

- **The article is free.** Glossary entries come with or without ال; treat
  every entry as a lemma and make it definite or indefinite by grammar: الوزن →
  وزن، بالوزن، أوزان الشبكة.
- **Plurals are the natural ones**, broken where the word has one: نموذج →
  نماذج، دالة → دوال، وزن → أوزان، خريطة → خرائط، شكل → أشكال، نواة → أنوية.
  Never a made-up sound plural to match the glossary (✗ نموذجات).
- **نواة الالتفاف, then النواة.** The full term at the first mention in a
  unit, then the short form; plural أنوية. Reason for أنوية over نوى: it
  shows its case and cannot be read as the verb نوى.
- **Number grammar applies to terms.** After 3–10 the plural in the genitive
  (10 خطوات); after 11–99 and after a symbol the singular in the accusative
  (42 عنصرًا، $L$ طبقة، $n - k + 1$ موضعًا); after 100 and above the singular
  in the genitive (100 طبقة).
- **Compound adjectives follow the noun without their own article:** الارتباط
  المتقاطع ثنائي الأبعاد، البيرسبترون متعدد الطبقات، موتر رباعي الأبعاد
  (to confirm; formal grammar asks for الثنائي الأبعاد).
- **A symbol agrees through its noun.** Give the noun where gender or case must
  show: المصفوفة $\mathbf{W}$ متناثرة ; لتكن $\mathbf{X}$ مصفوفة. A bare symbol
  takes the gender of what it denotes.

### 5.3 Verbs

Arabic verbs from the root of the glossary noun; never an English stem.

| English | Arabic | Not |
|---|---|---|
| train, pretrain, fine-tune | يدرّب; يدرّب مسبقًا (نموذج مدرَّب مسبقًا); يضبط ضبطًا دقيقًا | يعمل تدريب، يفاين-تيون |
| learn (a model learns a kernel) | يتعلّم | يدرس |
| sample (from a distribution) | يسحب عينة من؛ noun أخذ العينات | يعيّن |
| initialize, update, normalize, regularize | يهيّئ، يحدّث، يطبّع، ينظّم | |
| minimize, maximize, optimize | يصغّر، يعظّم، يحسّن | يقلّل for "minimize" (it means "reduce") |
| compute, evaluate (an expression) | يحسب | يقيّم (it means "assess") |
| implement, run, call | ينفّذ، يشغّل، يستدعي | يطبّق for "implement" (it means "apply") |
| flatten, stack, reshape, pad | يسطّح، يكدّس، يعيد تشكيل، يحشو | |
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
| Percent | `%` after the number, no space, as in the source. | "45% fewer parameters" → معلَمات أقل بنسبة 45% |
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

- **Models, libraries, datasets, companies, hardware, benchmarks, courses:**
  Latin script, exactly as in English: Fashion-MNIST, RTX 4090, OpenAI, GPT-2,
  Stanford CS231n.
- **People** (to confirm): a person whose name is part of a glossary term is
  written in Arabic letters, with the glossary's spelling, everywhere: غاوس،
  بايز، ماركوف، نيوتن، لاغرانج، تايلور، فورييه، برنولي، بواسون، ليبشيتز،
  كولموغوروف، فيشر، بولتزمان. Every other person stays in Latin script, as in
  the English and in the printed citation: اقترح Hinton …. Reason: Arabic
  writing transliterates the classical names, and a reader expects غاوس beside
  التوزيع الغاوسي; for living researchers the transliteration is not fixed
  (Arabic Wikipedia has both يان لوكون and يان ليكون), and citations print the
  Latin form. No titles (✗ الدكتور، ✗ البروفيسور).
- **Transliteration, where a name is in Arabic letters:** /g/ is غ (غاوس،
  لاغرانج، غرام); /v/ is ف; /p/ is ب; no ڤ، پ، گ. This is the glossary's
  practice.
- **Terms named after people:** as the glossary has them: التوزيع الغاوسي،
  مبرهنة بايز، سلسلة ماركوف، مضاعف لاغرانج، طريقة نيوتن، مفكوك تايلور، ثابت
  ليبشيتز، المصفوفة الهيسية، المصفوفة اليعقوبية.
- **The book:** *Dive into Deep Learning* stays in English; "this book" → هذا
  الكتاب.
- **Languages and places:** Arabic (الإنجليزية، الولايات المتحدة); programming
  languages stay (Python).
- **Structure words:** chapter الفصل, section القسم, subsection القسم الفرعي,
  appendix الملحق, part الجزء; Summary → الملخص, Exercises → تمارين,
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
| `slide_text` | Short nominal sentences or noun phrases; no نحن; instructions as verbal nouns, not imperatives; one line where the English is one line; no full stop after a single-phrase bullet. Bold lead-ins keep their bold and colon. Acronyms as the English writes them. | "**Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent." → **التعلم مقابل التصميم اليدوي:** الشبكات الالتفافية العميقة تكتشف آلافًا من هذه الأنوية تلقائيًا بالنزول التدرجي. ; "Lower the learning rate gradually" → خفض معدل التعلم تدريجيًا |
| `exercise` | Masculine singular imperative; direct questions with ؟. A bold title stays bold with its full stop inside. The tag `[code]` and the mark ● are copied. | "Derive a finite difference operator for the second derivative. Which structures in images respond most strongly to it?" → اشتق مؤثر فروق محدودة للمشتق الثاني. أي البنى في الصور تستجيب له بأقوى ما يكون؟ |
| `tab_note` | Body register. Describe this framework only; never compare frameworks. API names stay in code, after a class noun. | "PyTorch's `nn.LazyLinear(1)` requires only the output dimension." → تحتاج الطبقة `nn.LazyLinear(1)` في PyTorch إلى بعد الخرج فقط. |
| `note` | Register of its surroundings. Labels: **Note:** → **ملاحظة:**, **Warning:** → **تنبيه:**, **Tip:** → **نصيحة:**, **Answers:** → **الإجابات:**, Concept Check → اختبار الفهم. | |
| `table_row` | Noun phrases; numbers and units as in §6. | "Training time" → زمن التدريب |

The cover kicker `[Dive into Deep Learning · §2.1]{.kicker}` is rewritten by the
build; copy it unchanged. "Adapted from …" → مقتبس بتصرف من …, with course and
book titles in English.

## 9. References, math and code in a sentence

### 9.1 Cross-references

The Arabic build is to print the prefix with the article: الفصل 3، القسم 5.1،
الشكل 4، الجدول 2، المعادلة 3.1 (§11; until `ar/strings.yml` sets them the
build prints the English words, because Quarto ships no Arabic language file).
Write no prefix of your own and attach nothing to a reference (§3.3).

| English | Arabic |
|---|---|
| "In :numref:`fig_correlation`, we slide a kernel …" | ``في :numref:`fig_correlation` نزلق نواة …`` (prints: في الشكل 7.2 نزلق) |
| "as shown in :numref:`fig_x`, the loss falls" | ``كما يبيّن :numref:`fig_x`، تنخفض الخسارة`` |
| "see :numref:`sec_sgd`" | ``انظر :numref:`sec_sgd` `` ✗ ``انظر القسم :numref:`sec_sgd` `` ✗ ``انظر في ال:numref:`sec_sgd` `` |
| "(:numref:`sec_channels`)" | ``(:numref:`sec_channels`)`` |
| "We implement :eqref:`eq_corr2d` directly" | ``ننفّذ :eqref:`eq_corr2d` مباشرة`` |
| ":citet:`Luo.Li.Urtasun.ea.2016` showed that …" | ``بيّن :citet:`Luo.Li.Urtasun.ea.2016` أن …`` |

- **Prepositions before a reference are free-standing:** في، من، إلى، على،
  حسب، مثل. Not ب، ل، ك.
- **Agreement:** the printed nouns الفصل، القسم، الشكل، الجدول are masculine,
  المعادلة is feminine. A verb or adjective that agrees with a ``:numref:`` is
  masculine, with an ``:eqref:`` feminine: ``يبيّن :numref:`fig_x` ``,
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
particles (و ف ب ل ك), in the dual, in a sound plural and with a suffix. It does
not recognise broken plurals (نماذج for نموذج, دوال for دالة, خرائط for خريطة)
or a spelling that differs in marks (§2). Both are false matches: write the
correct Arabic and say so in NOTES, once per unit, as TRANSLATING.md §9
requires. The same holds for the number warning after a dual (§6) and for the
English verbs "return", "recall" and "span" taken for their noun senses.

## 10. Known pitfalls

### 10.1 Words with more than one sense

| English | Sense → Arabic | Not |
|---|---|---|
| vertical, column, orthogonal | vertical → رأسي (الحواف الرأسية); column → عمود (أعمدة); orthogonal, perpendicular → متعامد | عمودي for "vertical" (it is read as "column-wise" or "perpendicular") |
| input, output | of a layer or operation → الدخل، الخرج; countable → مدخلات، مخرجات | المدخل and الدخل mixed in one page |
| parameter, coefficient | parameter → معلَمة (معلَمات); coefficient → معامل (معاملات) | معامل for "parameter" (all three test translations wrote it; the glossary keeps the two apart) |
| kernel | convolution → نواة الالتفاف, then النواة; kernel methods → دالة النواة، مصفوفة النواة; GPU or OS code → النواة; of a linear map → الفضاء الصفري | |
| bias | parameter $b$ → الانحياز; of an estimator → الانحياز (مقدر غير منحاز); social → التحيز | الميل |
| scalar | noun → عدد قياسي; adjective → قياسي (انحياز قياسي، خسارة قياسية) | عددي (it means "numerical") |
| accuracy, precision, resolution | accuracy → الدقة; precision (metric) → الإحكام; floating-point precision → الدقة (دقة مزدوجة); resolution → دقة الصورة، دقة الدخل, always with its noun | الدقة alone for "resolution" |
| translation, shift, offset | of an image or function → الإزاحة (تساوق الإزاحة، ثبات الإزاحة); distribution shift → الانزياح; an index offset → الإزاحة; of language → الترجمة | ترجمة for a shift |
| step size, stride, time step | حجم الخطوة; خطوة الانزلاق; خطوة زمنية | الخطوة alone for a stride |
| shape, figure | tensor shape → الشكل (شكل الخرج); the build also prints الشكل before figure references, so near one write أبعاد or شكل الموتر | |
| network, grid | network → شبكة; grid of pixels or points → شبكة نقطية | شبكة for both in one sentence |
| edge | in an image → حافة (حواف); of a region or a receptive field → طرف; of a graph → ضلع | |
| stack | noun (of layers) → كدسة (كدسات); verb → يكدّس; stacked → مكدسة | كومة and مكدس in the same book |
| batch, "in one shot" | batch → دفعة; "in one shot", "at once" → في مرة واحدة | دفعة واحدة for "at once" |
| feature | → ميزة (ميزات); feature map → خريطة الميزات | سمة and ميزة mixed |
| return | RL → العائد; of a function → القيمة المعادة، يعيد | |
| sample | → عينة; to sample → يسحب عينة; a draw → سحبة | |
| loss, cost, objective, error, risk | Five words, five terms: خسارة، كلفة، دالة الهدف، خطأ، خطر. | merging them |
| regularization, normalization, standardization, calibration | التنظيم; التطبيع (تطبيع الدفعات); التوحيد المعياري; المعايرة; normal distribution → التوزيع الطبيعي, Gaussian → الغاوسي | mixing them |
| training, learning | train → التدريب; learn → التعلم (معدل التعلم، التعلم العميق) | |
| sequence | ML data → تسلسل; in mathematics → متتالية | |
| gradient, derivative | التدرج; المشتق, "with respect to $x$" → بالنسبة إلى $x$ | الميل، الانحدار (it is "regression") |
| likelihood, probability | الإمكان; الاحتمال | الاحتمالية for both |
| token, symbol, encoding | token → رمز; mathematical symbol → رمز, so write الرمز الرياضي where both occur; encoding → ترميز | |
| code | program text → الشيفرة; a code in coding theory → شفرة (شفرة بادئة); encoder → المشفر | |
| performance | of a model → الأداء; of code → السرعة، الأداء الحسابي | |
| unit | of a network → وحدة; "unit gradient" → تدرج مقداره واحد | وحدوي |
| support (of a function) | الحامل | الدعم |

### 10.2 Translationese

| Pattern | Write instead | Example |
|---|---|---|
| Subject first in every sentence | Verb first (§1) | ✗ الطبقة الالتفافية تحسب … الخرج ينكمش … ✓ تحسب الطبقة الالتفافية … فينكمش الخرج … |
| "It is important to note that", "Note that" word for word | Drop it, or لاحظ أن once in a while | ✗ من المهم أن نلاحظ أن الانحياز لا يخضع للتنظيم. ✓ الانحياز لا يخضع للتنظيم. |
| "Sure enough", "Clearly", "Simply" as fillers | وبالفعل، واضح أن; drop "simply" | "is simply the dot product" → ليس إلا الجداء النقطي |
| "using X", "via X", "based on X" as باستخدام every time | بـ on a class noun, or عبر، وفق، من | "using a pair of loops" → بحلقتين |
| "which …" as والذي / والتي after every noun | A new sentence or وهو / وهي | "… (*padding*), which we examine in …" → …، وسندرسه في … |
| Passive with an agent, تم, قام بـ | Active (§1) | |
| "there is / there are" as يوجد / هناك every time | A full verb, or لـ | "kernels that have no center pixel" → أنوية ليس لها بكسل مركزي |
| "we can" as نستطيع أن / يمكننا أن every time | The verb itself | "we can unroll the kernel" → نفرد النواة |
| Inanimate subject + "tells us", "allows us to" | من ذلك نرى، بذلك نستطيع | "This result tells us that depth helps." ✓ من هذه النتيجة نرى أن العمق مفيد. |
| "our model", "our edge detector" as نموذجنا every time | The noun with the article or a relative clause | "our hand-crafted edge detector" → كاشف الحواف الذي صمّمناه يدويًا |
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
   tatweel from both sides before comparing (§2, §9.4).
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

## 12. For the author to confirm

1. **Digits:** Western 0–9 with the decimal point and the comma for thousands
   (§6). Proposed: yes, for the whole book; no Eastern Arabic digits.
2. **Marks:** unvocalized text; the tanwin of the accusative always; the
   shadda on Form II and V verbs and not on nouns and terms (§2). The
   alternatives are no shadda at all, or a shadda on every doubled consonant.
   Proposed: the rule of §2.
3. **Tanwin placement:** before the alif (صفرًا), as the three test
   translations wrote; Arabic Wikipedia writes it after (نوعاً). Proposed:
   before.
4. **و before code, math and Latin words:** free-standing with spaces (§3.3).
   Arabic orthography attaches و; two of three translators separated it.
   Proposed: separate.
5. **Carriers بـ، لـ، كـ، الـ:** last resort for particles, الـ only before
   digits (§3.3). Proposed: as written; a native reviewer may prefer to forbid
   لـ $x$ altogether.
6. **parameter → معلَمة, coefficient → معامل** (§10.1), against the three test
   translations, which wrote معامل for both. Alternative: بارامتر. Proposed:
   معلَمة, written with its fatha.
7. **People's names:** Arabic letters for the names inside glossary terms,
   Latin for all others (§7). Alternative: Latin for all, as in the German and
   Japanese guides. Proposed: the two-tier rule.
8. **المحول for the Transformer architecture** (verified glossary entry,
   spelled المحوّل there) rather than the Latin name (§5.1). Proposed: keep it, gloss
   (Transformer) once per page.
9. **Kernel plural أنوية** (Fable) rather than نوى (Opus, Sonnet) (§5.2).
   Proposed: أنوية.
10. **الدخل / الخرج** for input and output (Fable, Opus; also Arabic
    Wikipedia) rather than المدخل / المخرج (§10.1). Proposed: الدخل، الخرج.
11. **رأسي for "vertical"** rather than عمودي (§10.1). Proposed: رأسي.
12. **الشيفرة for program code** (all three translators) rather than الكود;
    it shares a root with المشفر (encoder) (§5.1). Proposed: الشيفرة.
13. **Compound adjectives without the article** (الارتباط المتقاطع ثنائي
    الأبعاد) (§5.2). Proposed: accept the modern usage.
14. **English gloss** in plain parentheses, once per page, only for introduced
    or non-uniform terms (§5.4). Proposed: yes.
15. **Exercises and address:** masculine singular imperative, no double forms
    (§1, §8). Proposed: yes.
16. **Reference prefixes** with the article (الشكل 4، القسم 5.1، المعادلة
    3.1), all masculine except المعادلة (§9.1), and the headings تمارين and
    النقاشات. To check in a build: whether equation references print the word,
    and whether ``:citet:`` prints "et al." and "and" in Latin.
17. **token → رمز** while "symbol" is also رمز and "encoding" is ترميز
    (§10.1). Alternative: وحدة نصية. Proposed: keep رمز, review with a native
    NLP reader.
18. **precision (metric) → الإحكام** beside accuracy → الدقة and
    floating-point precision → الدقة. Proposed: keep; a native reader should
    confirm الإحكام.

## 13. Proposed glossary corrections

Not applied; the glossary is unchanged.

1. **Marks on nouns and terms** in about 45 entries, against §2, and inconsistent within
   the glossary (مشفّر المحول beside مفكك تشفير المحوّل; معدّل التعلّم beside
   جدول معدل التعلم). Verified entries: الانحدار الخطّي → الانحدار الخطي;
   مجموعة التحقّق → مجموعة التحقق; معدّل التعلّم → معدل التعلم; دالّة التفعيل →
   دالة التفعيل; المحوّل → المحول; التعلّم داخل السياق → التعلم داخل السياق;
   مقدّر غير منحاز → مقدر غير منحاز. The proposed entries likewise (متّجه
   الميزات، مشفّر تلقائي، موجّه، اللّاغرانجي، العائلة الأُسيّة، …).
2. **معلمة without its fatha:** model-parameter, parameter-count,
   parameter-update, natural-parameter, active-parameters,
   hyperparameter-optimization, hyperparameter-configuration → معلَمة،
   معلَمات, as in المعلَمة الفائقة.
3. **Tanwin after the alif:** مرصود جزئياً → جزئيًا; أمثل حسابياً → حسابيًا;
   مستمر ليبشيتزياً → ليبشيتزيًا.
4. **Acronym inside `preferred`:** the term can then only match with the
   parenthesis. Move the acronym to a note: تحليل القيم المفردة (SVD)، الذاكرة
   قصيرة المدى الطويلة (LSTM)، محوّل الرؤية (ViT)، التصريف في أثناء التنفيذ
   (JIT) (verified)، الفاعل–الناقد الليّن (SAC)، شبكة Q العميقة (DQN)، مشفّر
   تلقائي تغيّري (VAE)، أقصى تفاوت في المتوسط (MMD)، بُعد VC
   (فابنيك–تشيرفوننكيس)، Reflow (إعادة التدفق). PPO (تحسين السياسة التقاربي)
   has the two parts in the opposite order.
5. **Two words for "maximum likelihood":** تقدير الإمكان الأعظم and مقدّر
   الإمكان الأقصى → one of them (proposed: الأعظم).
6. **Broken plurals as alternatives,** so that the check accepts them: نماذج
   (model), دوال (function entries), خرائط الميزات (feature-map), أشكال
   (shape), خسائر (loss), قيود (constraint), عصبونات is sound and needs none.
   `weight` and `kernel` already list theirs.
7. **Missing entries the test page needed:** input (الدخل), output (الخرج),
   parameter (معلَمة), window (نافذة), patch (رقعة), edge (حافة), stack
   (كدسة), framework (إطار عمل), code (الشيفرة), dense layer (طبقة كثيفة),
   vertical (رأسي).
8. **Two roots for "encode":** المشفر، مفكك التشفير، فك التشفير beside ترميز
   موضعي، الترميز الأحادي، ترميز أزواج البايت. Both sets include verified
   entries; a native reviewer should choose one root for encoder and encoding.
9. **Article:** 156 entries carry ال and the rest do not. Harmless for the
   check; a uniform article-free lemma would be clearer.

## 14. Sources

Checked on 2026-11-26:

- The three test translations in `evals/ar-conv-layer/` (Fable, Opus, Sonnet),
  with their DECISIONS and NOTES; examples marked by name come from them.
- `tools/l10n/protect.py` and `glossary.py`, run on Arabic samples: `٣` is not
  matched to `3`; a glossary form with a shadda matches only its exact
  spelling; broken plurals are not matched; the article, attached particles and
  sound plurals are.
- The book's pandoc (`.venv-build`): ``ب@sec-x`` is not a reference and
  ``@sec-xو`` takes the و into the label; ``@sec-x،``, ``(@sec-x)`` and
  ``«@sec-x»`` work; math, code and emphasis directly against an Arabic letter
  parse.
- Quarto 1.9.37 language files in `.venv-build`: none for Arabic.
- Arabic Wikipedia, شبكة عصبونية التفافية: Western digits (عام 1998م); دخل and
  خرج; unvocalized text with tanwin (نوعاً خاصاً); people's names in Arabic
  letters with two spellings of one name; English terms after the Arabic one.
  It writes عصبونية and الانتشار نحو الخلف where the glossary has عصبية and
  الانتشار العكسي.
- `glossary/terms/ar.jsonl` and `glossary/notes/ar-tr-de.md` of this
  repository.

Not checked:

- No Arabic university textbook or translated ML book was read; the register,
  number grammar and punctuation rules are stated from the drafter's knowledge
  of Modern Standard Arabic.
- Google's Arabic ML glossary could not be retrieved (the page came back in
  English).
- The bidirectional display described in §3.2 and §11 follows the Unicode
  algorithm (UAX #9); it was reasoned, not rendered in a browser, and the
  Arabic build does not exist yet.
- Whether the site search ignores Arabic marks was not tested.
