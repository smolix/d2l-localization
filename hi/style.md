# Style guide: Hindi (hi)

Draft of 2026-11-26 for the author's review. Translators read this with every
unit (TRANSLATING.md §7–§8). Each rule has one reason and one example
(English → Hindi, or ✗ wrong / ✓ right). Sentences marked *fable*, *opus* or
*sonnet* are quoted from the translation test `evals/hi-conv-layer/`. Points
that need a native reviewer are marked "(to confirm)" and collected in §12.

Order of authority: the decisions log, then this guide, then the glossary. The
glossary says *what* a term is; this guide says *how to choose, spell and
inflect*. Where the two disagree today, §13 lists the glossary entry to change.

The reader is an Indian student or engineer who studied school mathematics in
Hindi or in English, learns computer science mostly from English material, and
will go on to English papers, documentation and code. The text must read as
Hindi written by a Hindi-speaking engineer: neither the Sanskritised Hindi of
government glossaries nor the Hinglish of chat.

## 1. Register

**Decision: the authors are हम, the reader is आप, never तुम or तू.** Reason:
हम carries the book's "we work through this together" tone and is the voice of
Hindi textbooks; तुम is the voice of a teacher talking down to a class.

| English | Hindi | Example |
|---|---|---|
| we + action in the text | हम + habitual present (करते हैं) | "We minimize the loss with SGD." → हम SGD से लॉस न्यूनतम करते हैं। |
| "In this section/chapter, we …" (a roadmap) | future (करेंगे) | "In this section, we implement cross-correlation." → इस अनुभाग में हम क्रॉस-सहसंबंध लागू करेंगे। |
| "we will see", "we will return to" | future | "We will return to this in …" → इस पर हम :numref:`sec_x` में लौटेंगे। |
| "let us", "let's" | आइए, … + subjunctive (देखें) | "Let us look at the gradient." → आइए, ग्रेडिएंट को देखें। |
| "you" = the reader (advice, a question, exercises) | आप + polite forms | "Suppose you want to estimate the price of a house." → मान लें कि आप किसी मकान की कीमत का आकलन करना चाहते हैं। |
| "you can", "one can" = anyone | passive of ability, or हम; no आप | "You can verify this numerically." → इसे संख्यात्मक रूप से जाँचा जा सकता है। |
| "we recommend", "we believe" | हम | "We recommend AdamW." → हम AdamW की सलाह देते हैं। |
| "we" = people, the field | हम मनुष्य, शोधकर्ता | "We humans attend selectively." → हम मनुष्य चुनिंदा चीज़ों पर ही ध्यान देते हैं। |
| "Note that", "Recall that" | ध्यान दें कि …, याद करें कि … | "Recall that $b$ is the bias." → याद करें कि $b$ बायस है। |
| "let $x$ be", "suppose", "consider" | मान लें कि … | "Let the input be a matrix." → मान लें कि इनपुट एक आव्यूह है। |

- **One imperative form: the -एँ/-ें form** (करें, देखें, दिखाएँ, बनाएँ, मान लें),
  in body text, slides and exercises. Not the -इए form (कीजिए, दीजिए, मान
  लीजिए), not the bare infinitive (करना), never करो/कर. Reason: one form for
  the whole book; it is the polite form of current technical Hindi and is
  gender-neutral. All three test translators mixed मान लीजिए into text that
  otherwise uses करें; *fable*'s exercise has the form to use: ✓ मान लें
  $\mathbf{X}$ एक $4 \times 4$ इनपुट है. (to confirm, §12.1)
- **हम takes the masculine plural** (हम देखते हैं), आप likewise (आप देखेंगे).
  This is the unmarked form of Hindi; no slash forms (✗ देखेंगे/देखेंगी).
- **People are referred to with the honorific plural** where a verb shows it:
  Hinton और उनके सहयोगी … मानते हैं. With ने the question does not arise
  (§4, citations).

**Sentence shape** (each point is checkable):

- **Verb last, and soon.** At most about 25 words in a sentence; split English
  sentences at "which", "where", "so that" (TRANSLATING.md §3 allows splitting
  within a block). A relative clause comes before its noun or as a जो … वह
  pair, not after the noun with जो कि.
  ✗ कर्नेल जो कि डेटा से सीखा गया है, किनारों को पकड़ता है।
  ✓ डेटा से सीखा गया कर्नेल किनारों को पकड़ता है।
- **Active with हम.** Passive (किया जाता है) only where the agent does not
  matter; never the agent with के द्वारा.
  ✗ ग्रेडिएंट की गणना हमारे द्वारा शृंखला नियम के उपयोग से की जाती है।
  ✓ ग्रेडिएंट हम शृंखला नियम से निकालते हैं।
- **No article.** एक only where the English means "one" or "a certain".
  ✗ दिखाएँगे कि एक कन्वोल्यूशन कैसे एक ही आव्यूह गुणन … बन जाता है (*opus*)
  ✓ दिखाएँगे कि कन्वोल्यूशन कैसे एक ही आव्यूह गुणन … बन जाता है (*fable*)
- **को only on a specific object that needs it.** Inanimate objects usually
  take none. ✗ `Conv2D` दोनों पैरामीटरों को दर्ज करता है (*opus*) ✓ `Conv2D`
  दोनों पैरामीटर दर्ज करता है (*sonnet*).
- **At most two of का/की/के in a row.** ✗ नेटवर्क की पहली लेयर के आउटपुट का
  शेप ✓ नेटवर्क की पहली लेयर जो आउटपुट देती है, उसका शेप.
- **Agreement in the perfective.** With ने the verb agrees with the object:
  हमने लेयर बनाई, हमने कर्नेल बनाया; with को on the object it is masculine
  singular: हमने लेयर को बदला. ✓ … से कागज़ पर कन्वोल्यूशनल लेयर … व्युत्पन्न की
  थी (*opus*).
- **Simple tenses.** English simple present → habitual (करता है), not the
  progressive (✗ कर रहा है) unless the English is progressive.
- **Everyday connectives and words**, not officialese: और (तथा only to avoid a
  second और in one phrase), लेकिन, इसलिए, क्योंकि, हालाँकि, यानी, से, सिर्फ़ or
  केवल, ज़रूरी, तरीका. Not एवं, किंतु, परंतु, अतएव, अर्थात्, के द्वारा, के माध्यम से
  for plain "by/with", उपरोक्त, तत्पश्चात्. (अतः is allowed in proofs.)
- **No Hinglish for ordinary words.** ✗ यूज़ करना, इम्पॉर्टेंट, सिंपल, डिफ़रेंट,
  रिज़ल्ट, प्रॉब्लम, चेक करना ✓ उपयोग करना, महत्वपूर्ण, सरल, अलग, परिणाम, समस्या,
  जाँचना. English enters only through the term classes of §5.

## 2. Script and spelling

One spelling per word. The standard is the Central Hindi Directorate's
*देवनागरी लिपि तथा हिंदी वर्तनी का मानकीकरण*; the points below are those that
vary in practice.

| Item | Rule | Example |
|---|---|---|
| Unicode form | NFC. Nukta letters are the base letter + U+093C, which is what NFC produces; never the precomposed U+0958–U+095F. Reason: the glossary and all three test translations are in this form, and the term check compares code points. | ज़ = ज + ़ ; ड़ = ड + ़ |
| Invisible characters | None: no ZWJ, ZWNJ, no-break space or soft hyphen (TRANSLATING.md §8). Conjuncts are the font's business. | क्वेरी, सॉफ़्टमैक्स typed plainly |
| Nukta | Always ज़ and फ़ (in English and Perso-Arabic words) and the native ड़, ढ़. Never क़, ख़, ग़. Reason: z and f are contrasts every reader hears and the glossary writes them; the other three are dropped in standard modern Hindi, and one rule can be checked. (to confirm, §12.4) | ✓ ज़रूरी, सफ़ेद, तेज़, सिर्फ़, फ़ीचर, ऑप्टिमाइज़र, बड़ा, पढ़ना ; ✓ कीमत, कागज़, खुद, बाकी ✗ क़ीमत, काग़ज़, ख़ुद |
| Anusvara for a nasal before a stop | ं, not the half nasal, in Hindi words and in loans. | ✓ संबंध, अंक, हिंदी, टेंसर, एंट्रॉपी, एंबेडिंग, सैंपलिंग ✗ सम्बन्ध, अङ्क, टेन्सर, एम्बेडिंग |
| …except | the half nasal stays (a) before another nasal, य, व, ह (अन्य, सम्मान, जन्म); (b) in loans after ऑ/ॉ (कॉन्टेक्स्ट, हॉरिज़ॉन्टल, फ़ॉन्ट); (c) in loans where n meets क, ग, फ़, व at a prefix boundary (एन्कोडर, इन्फ़रेंस, कन्वोल्यूशन, कन्फ़्यूज़न). The glossary spelling decides each term. | ✓ कन्वोल्यूशन ✗ कंवोल्यूशन |
| Chandrabindu | ँ for a nasal vowel when no vowel sign rises above the headline; ं when one does. In loans only ं. Reason: the standard rule; it keeps हँस/हंस apart. | ✓ जहाँ, पाँच, ऊँचाई, बाएँ, दिखाएँ, हूँ ; में, हैं, नहीं, करें, लेयरों ✗ जहां, पांच, दिखाएं |
| ए, not ये, in verb and plural endings | | ✓ गए, लिए, चाहिए, नई, आएगा ✗ गये, लिये, चाहिये, नयी |
| English vowels | ऑ/ॉ for the vowel of "loss", "model"; ऐ/ै for the vowel of "batch", "map". No ऍ/ॅ. | लॉस, मॉडल, ऑप्टिमाइज़र, कॉलम ; बैच, पैडिंग, ऐक्टिवेशन |
| English consonants | t, d → ट, ड; th → थ; f → फ़; z and voiced s → ज़; v, w → व; -tion → -शन; -sion → -ज़न; -ing → -िंग. | टोकन, डेटा, थ्रेशोल्ड, फ़िल्टर, नॉर्मलाइज़ेशन, विंडो, कन्फ़्यूज़न, पूलिंग |
| Fixed spellings | शृंखला (✗ श्रृंखला), महत्त्व (as the glossary; ✗ महत्व inside terms), अंग्रेज़ी, हिंदी (the language menu's हिन्दी is a name and stays). | शृंखला नियम, महत्त्व सैंपलिंग |
| Word division | Postpositions are separate words after nouns and attached to pronouns; -कर is attached; वाला is separate; ही, भी, तो, तक are separate. | मॉडल में, GPU पर ; उसका, इसमें, हमने ; रखकर ; दो भारों वाला डिटेक्टर |
| Visarga | ः only inside Sanskrit words (अतः, क्रमशः, प्रायः). It is not a colon and a colon is not a visarga. | ✗ उत्तरः ✓ उत्तर: |

## 3. Punctuation and spacing

| Item | Rule | Example |
|---|---|---|
| Sentence end | The danda । (U+0964), set against the last character, one space after. Never a full stop, never the pipe `\|`, never ॥. Reason: it is the Hindi sentence mark and all three test translations use it. | "The window fits at four positions." → विंडो चार स्थितियों पर समाती है। |
| ? and ! | As in English; they replace the danda. | क्या ग्रेडिएंट डिसेंट कर्नेल को फिर से पा सकता है? |
| Full stop | Only inside numbers (0.5), Latin abbreviations (et al., vs. inside a name), file names and code. | |
| Danda after a token | Directly after math, code or a reference; tested with the build's pandoc. | … वहाँ $-1$। ; … :numref:`sec_padding` में देखेंगे। ; … करते हैं :cite:`Luo.Li.Urtasun.ea.2016`। |
| Comma | Between items of a list and between clauses where Hindi pauses (before लेकिन, क्योंकि, a जो clause); no comma before और/या in a list. | "Adam, RMSProp, and SGD" → Adam, RMSProp और SGD |
| Colon | As in English, after a lead-in and before a list or a display equation; no space before, one after. | **मुख्य मान्यता:** त्रुटियाँ स्वतंत्र हैं। |
| Semicolon | Only where the English has one and the two halves stay one thought; otherwise a danda. Do not add semicolons to join split sentences. | |
| Dash | The English em dash `—` becomes a comma, a colon or parentheses; where a dash is the only fit, an em dash with a space on each side, at most one pair per sentence. Ranges and pairs keep the en dash of the source (इनपुट–आउटपुट जोड़ा, 40–55). | "the bias—an offset—shifts the plane" → बायस (एक ऑफ़सेट) समतल को खिसकाता है। |
| Hyphen | In repeated and paired Hindi words (एक-एक, अलग-अलग, धीरे-धीरे, प्रविष्टि-दर-प्रविष्टि), in loans the glossary hyphenates (क्रॉस-एंट्रॉपी, फ़ाइन-ट्यूनिंग, मिनी-बैच), and between a symbol or letter and a suffix-like word (table below). No hyphen between an adjective and its noun. | ✗ रैखिक-प्रतिगमन |
| Quotation marks | Type ASCII `"…"`; the build turns them into “…” (tested). Inside them '…'. In a slide `title="…"` the ASCII `"` delimits the attribute, so type “…” (U+201C, U+201D) inside it. | हम क्रॉस-सहसंबंध को ही "कन्वोल्यूशन" कहते हैं। |
| Parentheses | ASCII `( )`, a space outside, none inside. | ग्रहणशील क्षेत्र (receptive field) |
| Abbreviations | Write them out: "e.g." → जैसे, उदाहरण के लिए; "i.e." → यानी; "etc." → आदि; "vs." → बनाम; "cf." → देखें; "std" → मानक विचलन. No Hindi abbreviations with a dot (✗ उदा., इ.). "et al." stays in Latin. English acronyms stay (§5). | |
| Emphasis | Keep `*…*` and `**…**` on the corresponding word(s), around the whole inflected word, never splitting off an ending; add none. The English gloss stays outside the markers. | ✓ *लेयरों* ✗ *लेयर*ों ; `*परिमित अंतर* (finite difference)` |

**Spacing: one ordinary space between Devanagari text and any Latin word,
number, inline math, inline code or reference.** No space before । , ; : ? ! )
or after (. Reason: Hindi separates words with spaces, and the build needs it:
tested with the repository's pandoc, a reference followed directly by a
Devanagari letter is not recognised (``:numref:`sec_x`में`` is read as the
label `sec-xम`; ``:numref:`sec_x` में`` and ``:numref:`sec_x`।`` work).

**Postpositions after Latin terms, acronyms, math, code and references** are
separate words, never attached and never hyphenated.

| Case | Rule | Example |
|---|---|---|
| का / की / के | The form follows the noun *after* it (the thing possessed), never the token before it. | $\mathbf{X}$ का शेप, $\mathbf{X}$ की पंक्तियाँ, $\mathbf{X}$ के अवयव ; `corr2d` का आउटपुट ; :numref:`fig_correlation` की गणना |
| में, से, पर, को, तक, ने | Unchanged after any token. | GPU पर, PyTorch में, `X` को, $-\Delta$ से $\Delta$ तक, :numref:`sec_why-conv` में |
| Token as subject of a gendered verb or adjective | Put the token's noun before it and agree with the noun. | आव्यूह $\mathbf{W}$ विरल है। कर्नेल $\mathbf{K}$ खिसकता है। लेयर $i$ … पहुँचती है। |
| Bare symbol, code identifier or model name as subject | Masculine singular. Do not turn the sentence passive to avoid the choice. | $n$ बढ़ता है। BERT … सीखता है। ✓ नीचे `Conv2D` दोनों पैरामीटर दर्ज करता है (*sonnet*) |
| Acronyms and Latin names | No English plural -s, no Hindi ending, no apostrophe. Show the plural with a numeral, a quantifier or a head noun. | दो GPU, सभी GPU पर, CNN मॉडलों में ✗ GPUs, GPUओं, CNN्स, Adam's |
| Symbol or number + suffix-like word | Hyphen after math or a letter; ordinals with digits are written solid. | $d$-विमीय, $k$-गुना, Q-लर्निंग, n-ग्राम, p-मान ; 10वाँ, 21वीं |
| Latin name + Hindi noun | A space, no hyphen. | Adam ऑप्टिमाइज़र, ReLU ऐक्टिवेशन, GPU मेमोरी, KV कैश, Fashion-MNIST छवि |

**Order of protected tokens.** Hindi is verb-final and postpositional, so two
tokens of a class sometimes change places. First try the English order (Hindi
word order is free enough for most cases: front the topic, keep the verb
last). If natural Hindi needs another order, write natural Hindi and name the
block in NOTES; the check then accepts it. Never distort a sentence to silence
the warning. Around a display equation, close the clause before it and start a
new one after it: "… so the output shape is $$…$$ where $n$ is …" → … इसलिए
आउटपुट का शेप है $$…$$ यहाँ $n$ … है। Math that contains English stays as it is:
"the $i^{\textrm{th}}$ layer" → $i^{\textrm{th}}$ लेयर.

## 4. Cross-references and citations

The Hindi build prints the prefix and the number: अध्याय 7, अनुभाग 7.2,
चित्र 7.2.1, तालिका 7.2.1, समीकरण 7.2.1 (the words are set in
`hi/strings.yml`, §11). Do not write a prefix of your own. A slide's
`§{sec_x}` prints §7.2 and `§{num:sec_x}` the bare number.

| English | Hindi |
|---|---|
| "In :numref:`sec_why-conv`, we derived …" | :numref:`sec_why-conv` में हमने … व्युत्पन्न किया था। |
| "as shown in :numref:`fig_x`, the loss falls" | जैसा :numref:`fig_x` में दिखाया गया है, लॉस घटता है। |
| "see :numref:`sec_sgd`" | :numref:`sec_sgd` देखें ; in parentheses (:numref:`sec_sgd`) |
| ":numref:`fig_x` shows the kernel" | :numref:`fig_x` में कर्नेल दिखाया गया है। |
| "the experiment of :numref:`subsec_x`" | :numref:`subsec_x` का प्रयोग |
| "We implement :eqref:`eq_corr2d` in `corr2d`." | हम :eqref:`eq_corr2d` को `corr2d` में लागू करते हैं। |
| ":citet:`Vaswani.2017` introduced …" | :citet:`Vaswani.2017` ने … प्रस्तुत किया। |
| "… do :cite:`Luo.2016`." | … करते हैं :cite:`Luo.2016`। |

- ✗ चित्र :numref:`fig_x` में, ✗ अनुभाग :numref:`sec_x` देखें: the prefix would
  print twice.
- **Do not make a reference the subject of a verb that shows gender.** Use में
  and let the verb agree with the thing shown. Reason: the printed nouns are
  masculine (अध्याय, अनुभाग, चित्र, समीकरण) except तालिका, and labels do not
  always show their kind. If a reference must be the subject, agree with the
  printed noun: ✓ :numref:`tab_x` … दिखाती है.
- A citation `:citet:` prints Latin names ("Luo et al. (2016)"). Follow it
  with ने and a perfective verb, which agrees with the object and so needs no
  gender or number for the authors; in the present tense use the honorific
  plural (… दिखाते हैं). `:cite:` stands at the end of its clause, a space
  before it, the danda directly after.
- A reference into a chapter outside the translation scope is copied as it is;
  the build adds the "(English)" marker.

## 5. Terminology: Hindi, Devanagari loan, or Latin

### 5.1 How to choose

**Decide each word by its class and render it the same way in every compound.**
The glossary applies this rule (`glossary/review/hi.md`); for a term without
an entry, find its class, check how Hindi teaching material writes it, and
record the choice in DECISIONS.

| Class | Rule | Examples |
|---|---|---|
| 1. School and first-degree mathematics and statistics | The Hindi term in its NCERT/IGNOU form. | आव्यूह, सदिश, अदिश, सारणिक, अवकलज, अवकलन, समाकल, फलन, प्रायिकता, प्रसरण, माध्य, माध्यिका, मानक विचलन, वितरण, प्रतिगमन, प्रत्याशा, संभाविता, अभिसरण, इष्टतमीकरण, विमा, पंक्ति, स्तंभ, अवयव, समष्टि |
| 2. Everyday words that are also terms | Hindi. | प्रशिक्षण, भार, शोर, त्रुटि, मान, नीति, अवस्था, क्रिया, पुरस्कार, परिवेश, गहराई, छवि, नमूना, उदाहरण, चरण |
| 3. Terms of art of machine learning and computing (first met in an ML course, a paper or framework documentation) | The English word in Devanagari, in the glossary's spelling. Never a coinage. | ग्रेडिएंट, लॉस, लेयर, बायस, फ़ीचर, कर्नेल, टेंसर, टोकन, बैच, एपॉक, ऐक्टिवेशन, नॉर्मलाइज़ेशन, रेगुलराइज़ेशन, कन्वोल्यूशन, पूलिंग, पैडिंग, स्ट्राइड, अटेंशन, एंबेडिंग, ड्रॉपआउट, सैंपलिंग, इन्फ़रेंस, लर्निंग रेट, ओवरफ़िटिंग, बैकप्रॉपैगेशन, न्यूरल नेटवर्क, डीप लर्निंग |
| 4. General computing words long settled in Hindi | Devanagari loan. | कंप्यूटर, कोड, फ़ाइल, डेटा, प्रोग्राम, लूप, इंडेक्स, फ़्रेमवर्क, लाइब्रेरी, मेमोरी, इनपुट, आउटपुट, पैरामीटर, फ़ंक्शन (in code), क्लास, इंजीनियर |
| 5. Names: models, architectures by name, optimizers, libraries, datasets, benchmarks, companies, hardware; acronyms; anything in code | Latin script, exactly as in English; never transliterated, never translated. | Adam, AdamW, ResNet, BERT, PyTorch, ImageNet, Fashion-MNIST, RTX 4090, ReLU, GELU, SGD, MLP, CNN, LSTM, GPU, KL, im2col ✗ एडम, पाइटॉर्च, रेसनेट, दीर्घ अल्पकालिक स्मृति |
| 6. Government-glossary coinages that classrooms do not use | Never. | ✗ प्रवणता अवरोहण, गहन अधिगम, यंत्र अधिगम, तंत्रिका जाल, संवलन, प्रदिश, आच्छद, पृच्छा, कूटलेखक, स्तर (for layer) |

Reasons. Engineering in India is taught in English, and Hindi ML teaching
keeps the English technical nouns and uses Hindi for grammar and for school
mathematics; a coined term (class 6) has to be translated back before the
reader can use a paper or an API. School mathematics is the opposite case: its
Hindi vocabulary is stable and known to every Hindi-medium student.
✗ … इसलिए हम गहन अधिगम की मानक शब्दावली अपनाकर … (*sonnet*)
✓ … इसलिए हम डीप लर्निंग की प्रचलित शब्दावली के अनुसार … (*fable*)

- **Mixed compounds are normal Hindi and follow from the rule:** लॉस फलन,
  फ़ीचर सदिश, ग्रेडिएंट का नॉर्म, रैखिक प्रतिगमन, बैच नॉर्मलाइज़ेशन, पैच आव्यूह,
  क्रॉस-सहसंबंध.
- **Architecture words that the glossary writes in Devanagari are common
  nouns:** ट्रांसफ़ॉर्मर, न्यूरल नेटवर्क, सॉफ़्टमैक्स, सिग्मॉइड, एन्कोडर. Named
  models built from them stay in Latin (Vision Transformer, ResNet-18).
- **No Latin-script English words inside a Hindi sentence** apart from class 5
  and the gloss of §5.5. ✗ kernel का size ✓ कर्नेल का आकार.
- **डीप only in fixed names** (डीप लर्निंग, डीप रीइन्फ़ोर्समेंट लर्निंग); a deep
  network is गहरा नेटवर्क, its depth गहराई.
- **A mathematical function is फलन; a function in code is फ़ंक्शन.** लॉस फलन,
  ऐक्टिवेशन फलन; `corr2d` फ़ंक्शन.

### 5.2 Gender of loan nouns

Every Hindi noun has a gender, and verbs and adjectives show it, so a loan's
gender must be fixed. Rule for a loan not listed: the gender of its nearest
Hindi equivalent (लेयर ← परत, विंडो ← खिड़की, मेमोरी ← स्मृति, क्लास ← कक्षा);
otherwise -इंग and -ई/-ी words are feminine, -शन, -मेंट and -र words masculine,
and the default is masculine. A compound takes the gender of its last word.
Record the choice in DECISIONS.

| Gender | Loans |
|---|---|
| Masculine | मॉडल, नेटवर्क, कर्नेल, टेंसर, ग्रेडिएंट, लॉस, बायस, फ़ीचर, फ़िल्टर, चैनल, पिक्सेल, टोकन, बैच, एपॉक, पैरामीटर, इनपुट, आउटपुट, डेटा, डेटासेट, लेबल, स्ट्राइड, मास्क, प्रॉम्प्ट, नॉर्म, स्कोर, लॉजिट, शेप, पैच, कैश, ग्राफ़, एजेंट, ट्रांसफ़ॉर्मर, एन्कोडर, डीकोडर, ऑप्टिमाइज़र, जनरेटर, अटेंशन, ड्रॉपआउट, सॉफ़्टमैक्स, लर्निंग रेट, शेड्यूल, चेकपॉइंट, फ़ंक्शन; all -शन words (ऐक्टिवेशन, कन्वोल्यूशन, नॉर्मलाइज़ेशन, रेगुलराइज़ेशन, बैकप्रॉपैगेशन) |
| Feminine | लेयर, विंडो, क्लास, यूनिट, रैंक, क्वेरी, मेमोरी, लाइब्रेरी, एंट्रॉपी, लेटेंसी, ट्रैजेक्टरी, परप्लेक्सिटी, पाइपलाइन, बेसलाइन; all -इंग words (पूलिंग, पैडिंग, सैंपलिंग, एंबेडिंग, लर्निंग, प्री-ट्रेनिंग, फ़ाइन-ट्यूनिंग, क्लिपिंग) |

Hindi terms keep their own gender; the frequent ones: आव्यूह, सदिश, फलन,
अवकलज, प्रसरण, माध्य, वितरण, भार, मान, पुरस्कार, प्रतिफल, स्तंभ, अवयव, शोर,
प्रशिक्षण, समीकरण are masculine; प्रायिकता, संभाविता, प्रत्याशा, सटीकता, त्रुटि,
अवस्था, नीति, क्रिया, पंक्ति, प्रविष्टि, छवि, गहराई, विमा, समष्टि, शृंखला, कुंजी
are feminine.
✓ कन्वोल्यूशनल लेयर … क्रॉस-सहसंबंध निकालती है और उसमें एक अदिश बायस जोड़ती है।
(*fable*)

### 5.3 Plural and oblique forms of loans

Loans inflect as Hindi nouns. Never the English plural.

| Noun | Direct plural | Oblique plural | Not |
|---|---|---|---|
| Masculine loan (कर्नेल, मॉडल, टोकन, पैरामीटर) | unchanged: दो कर्नेल | -ों: कर्नेलों में, मॉडलों को | ✗ कर्नेल्स, टोकन्स, पैरामीटर्स |
| Feminine loan in a consonant (लेयर, क्लास) | -ें: लेयरें | -ों: लेयरों में | ✗ लेयर्स |
| Feminine loan in -ई (क्वेरी, लाइब्रेरी) | -इयाँ: क्वेरियाँ | -इयों: क्वेरियों में | ✗ क्वेरीज़ |
| Loan in -ओ (विंडो) | unchanged: चारों विंडो | unchanged: विंडो में | |
| डेटा | singular mass noun: डेटा उपलब्ध है; oblique unchanged: डेटा से | | ✗ डेटाओं, डेटे |

### 5.4 Verbs

Everyday actions keep Hindi verbs (सीखना, चलाना, बदलना, जोड़ना, घटाना, गुणा
करना, निकालना, जाँचना). A technical action is a noun or adjective + करना. An
English stem is used only where the term itself is a class-3 loan; never an
English participle (✗ ट्रेन्ड मॉडल, ✗ लर्न करता है).

| English | Hindi | Not |
|---|---|---|
| train | प्रशिक्षित करना; a trained model प्रशिक्षित मॉडल | ट्रेन करना |
| pretrain, fine-tune | प्री-ट्रेन करना, फ़ाइन-ट्यून करना; प्री-ट्रेन किया हुआ मॉडल | पूर्व-प्रशिक्षित |
| learn | सीखना; learned kernel सीखा हुआ कर्नेल; learnable सीखने योग्य | लर्न करना |
| compute, evaluate | निकालना, गणना करना; का मान निकालना | कंप्यूट करना |
| implement | लागू करना (noun: कार्यान्वयन) | इम्प्लीमेंट करना |
| initialize | इनिशियलाइज़ करना (noun: as the glossary, §13.5) | आरंभ करना (= to begin) |
| minimize, maximize, optimize | न्यूनतम करना, अधिकतम करना, ऑप्टिमाइज़ करना | |
| normalize, regularize, quantize | नॉर्मलाइज़ करना, रेगुलराइज़ करना, क्वांटाइज़ करना | सामान्यीकृत करना (that is "generalize") |
| differentiate, derive (a formula) | अवकलन करना; व्युत्पन्न करना | |
| sample (from a distribution) | $p$ से $x$ का नमूना लेना | सैंपल करना |
| update, encode, decode, mask, clip, cache, backpropagate | अपडेट करना, एन्कोड करना, डीकोड करना, मास्क करना, क्लिप करना, कैश करना, बैकप्रॉपैगेट करना | |
| pad, pool | पैडिंग लगाना, पूलिंग करना | पैड करना |
| flatten, stack, reshape | समतल करना; एक के ऊपर एक रखना (noun: स्टैक); शेप बदलना | चपटा करना |
| transpose | का परिवर्त लेना (noun: परिवर्त); fixed name ट्रांसपोज़्ड कन्वोल्यूशन | |
| overfit, underfit | मॉडल ओवरफ़िट करता है / अंडरफ़िट करता है | |

### 5.5 The English term in parentheses

**Add the English once per page (one numbered section of the book), at the
first occurrence in body text, after every Hindi technical term of class 1 or
2 whose everyday meaning does not already give it: `हिंदी पद (english term)`,
lower case, ASCII parentheses, outside the emphasis markers.** Reason: many
readers know these concepts by their English names only, and all go on to
English sources; once per page keeps the text clean. Check the English page in
CONTEXT: if an earlier unit of the page introduced the term, do not repeat it.

- Gloss: आव्यूह (matrix), अवकलज (derivative), प्रसरण (variance), संभाविता
  (likelihood), ग्रहणशील क्षेत्र (receptive field), स्थानांतरण समचरता (translation
  equivariance), *परिमित अंतर* (finite difference).
- Never gloss a loan: its sound is the English.
  ✗ शुरू में लॉस (loss) लगभग 12 पर रहता है (*sonnet*)
  ✗ … डेटा से अपने-आप उपयोगी कर्नेल (kernel) खोज सकता है (*opus*)
  ✓ … क्या ग्रेडिएंट डिसेंट डेटा से उपयोगी कर्नेल अपने आप खोज सकता है (*fable*)
- Do not gloss words whose plain meaning is the technical one: भार, शोर,
  त्रुटि, मान, योग, गहराई, उदाहरण, चरण, पंक्ति.
- With an acronym, the first use gives the full term and the acronym as the
  glossary has them: स्टोकेस्टिक ग्रेडिएंट डिसेंट (SGD), LSTM (long short-term
  memory); afterwards write what the English writes (SGD, LSTM).
- Never in headings, captions, slides or table cells. In an exercise only for
  a term the page's body does not contain.

## 6. Numbers and units

**Decision: Latin digits 0–9 everywhere, never Devanagari digits ०–९; the
decimal point and the thousands comma exactly as in the English.** Reasons:
these are the "international form of Indian numerals" that Hindi textbooks
and official Hindi use; math, code, outputs and figures are protected and show
Latin digits, so Devanagari digits would put two systems into one sentence;
and `l10n check` compares numerals as written (tested: ३ for 3 and 1,00,000
for 100,000 both raise "numbers not found").

| Item | Rule | Example |
|---|---|---|
| Decimal, thousands | As in the source. No Indian grouping. | "10,000 images" → 10,000 छवियाँ ; 100,000 ✗ 1,00,000 |
| Number words | hundred सौ, thousand हज़ार, million मिलियन, billion बिलियन, trillion ट्रिलियन, with the English numeral unchanged. No conversion to लाख, करोड़, अरब. Reason: the numeral stays equal to the English, to code output and to model names (7B). (to confirm, §12.3) | "3 million parameters" → 3 मिलियन पैरामीटर ✗ 30 लाख |
| Digits or words | As the English: a number spelled out in English is a Hindi word (दो, तीन, दस); a digit stays a digit. Never spell out a digit. | "two hidden layers" → दो हिडन लेयरें ; 10 एपॉक |
| Ordinals | पहला, दूसरा, तीसरा, चौथा, पाँचवाँ … दसवाँ, agreeing with the noun (तीसरी लेयर); above ten, digits + वाँ/वीं/वें. | "the third epoch" → तीसरा एपॉक |
| Percent | `%` attached, as in the source; प्रतिशत only where the English writes "percent". | "45% fewer parameters" → 45% कम पैरामीटर |
| Units | Symbols in Latin as in the source, with its space; time words written out become सेकंड, मिनट, घंटे, दिन. | 24 GiB, 5 ms ; "3 hours" → 3 घंटे |
| Ranges | The source's en dash without spaces; "from a to b" → a से b तक. | 40–55 ; "from $-\Delta$ to $\Delta$" → $-\Delta$ से $\Delta$ तक |
| Multipliers | In prose N गुना with the digit; "twice" (a word) → दोगुना; keep `×` in slides, tables, captions and math. | "about 2× faster" → लगभग 2 गुना तेज़ ; slide: 2× तेज़ |
| Approximation | लगभग for "about", "roughly", "~"; "an order of magnitude" → परिमाण की एक कोटि. | "roughly 90%" → लगभग 90% |
| Years, decades | "in 2017" → 2017 में; "the 2010s" → 2010 के दशक में. | ResNet 2015 में आया। |
| Money | Amount, space, currency word, in the source's currency. | "\$2.50" → 2.50 डॉलर |
| Counted nouns | After a numeral the direct plural, the oblique before a postposition. | 10 चरण ; 10 चरणों में ; "for 10 epochs" → 10 एपॉक तक |

Quantities, signs and orders of magnitude stay exact; do not round or convert
units or currencies.

## 7. Names

- **Models, libraries, datasets, companies, hardware, benchmarks:** Latin, as
  in English (§5.1 class 5).
- **People, when they act in the text:** Latin script, no title (no श्री, डॉ.,
  प्रो.). "Gauss grounded least squares in probability theory." → Gauss ने
  न्यूनतम वर्ग को प्रायिकता सिद्धांत पर आधारित किया। Reason: citations print in
  Latin and searches use the Latin form.
- **Terms named after people** follow the glossary. Devanagari only for the
  names with a settled Hindi school spelling: न्यूटन, गाउस (गाउसियन), बेज़
  (बेज़ियन), मार्कोव, टेलर, लाग्रांज (लाग्रांजियन), ऑयलर, यूक्लिड (यूक्लिडीय),
  बर्नूली, पॉइसन, फूरिये, लाप्लास, बेलमैन, जेनसेन, ब्राउनी गति, जैकोबियन, हेसियन,
  मोंटे कार्लो. Every other name stays Latin + space + Hindi noun: Fisher सूचना,
  Wasserstein दूरी, Cholesky अपघटन, Kronecker गुणनफल, Nesterov संवेग, Huber
  लॉस, Xavier आरंभीकरण. (to confirm, §12.6)
- **The book:** *Dive into Deep Learning* stays in English; "this book" → यह
  पुस्तक.
- **Languages and places:** Hindi (अंग्रेज़ी, अमेरिका); programming languages
  stay (Python). Titles of courses, books and assignments in a source credit
  stay in English.
- **Structure words:** part भाग, chapter अध्याय, section अनुभाग, subsection
  उप-अनुभाग, appendix परिशिष्ट, figure चित्र, table तालिका, equation समीकरण;
  Summary → सारांश, Exercises → अभ्यास, Discussions → चर्चा; Theorem → प्रमेय,
  Proposition → साध्य, Lemma → लेम्मा, Corollary → उपप्रमेय, Definition →
  परिभाषा, Proof → उपपत्ति, Remark → टिप्पणी, Example → उदाहरण. The test
  translators wrote अनुभाग, खंड and सेक्शन for "section"; अनुभाग is fixed.

## 8. Headings, captions, slides, exercises, notes

| Kind | Rule | Example |
|---|---|---|
| `heading` | Noun phrase or verbal noun, no danda, no English gloss. English "-ing + object" becomes object + का + noun, or object + verbal noun. A question keeps its ?. | "Convolutions for Images" → छवियों के लिए कन्वोल्यूशन ; "Training the Model" → मॉडल का प्रशिक्षण ; "Learning a Kernel" → कर्नेल सीखना ; "Why Depth Helps" → गहराई क्यों मदद करती है? |
| `caption` | Body register without हम and आप; a danda exactly where the English caption has a full stop. Keep `(a)`, `(b)`. | "Masks compose by logical AND." → मास्क तार्किक AND से जुड़ते हैं। |
| `slide_title`, `kicker` | Noun phrase, as short as the English, no danda; "vs." → बनाम, "&" → और. | `title="Theoretical vs. Effective Receptive Fields"` → `title="सैद्धांतिक बनाम प्रभावी ग्रहणशील क्षेत्र"` (*fable*) ; `[Concept Check]{.kicker}` → `[समझ की जाँच]{.kicker}` |
| `slide_text` | Short statements or noun phrases; no हम, no आप; instructions in the -एँ form; one line where the English is one line; a danda exactly where the English bullet ends in a full stop. Bold lead-ins keep their bold and colon. | "- **45% fewer parameters:** $27 c^2$ weights instead of $49 c^2$." → - **45% कम पैरामीटर:** $27 c^2$ भार, न कि $49 c^2$। ; "Lower the learning rate gradually" → लर्निंग रेट धीरे-धीरे घटाएँ |
| `exercise` | The -एँ imperative; direct questions with ?. A bold title keeps its bold, with the danda inside it where the English has the full stop inside. Copy `[code]`, ● and `:exlabel:` unchanged. | "**Hand-designed kernels.** Design some kernels manually. Why?" → **हाथ से बनाए कर्नेल।** कुछ कर्नेल हाथ से डिज़ाइन करें। क्यों? |
| Source credits | "Adapted from X." → X से रूपांतरित। The link moves before से रूपांतरित; name the block in NOTES if tokens change order. | *Stanford CS231n, [Assignment 2](…) से रूपांतरित।* |
| `tab_note` | Body register. Describe this framework only; never compare frameworks. API names stay in code. | "PyTorch's `nn.LazyLinear(1)` requires only the output dimension." → PyTorch के `nn.LazyLinear(1)` को केवल आउटपुट की विमा चाहिए। |
| `note` | Register of its surroundings. Labels: **Note:** → **टिप्पणी:**, **Warning:** → **चेतावनी:**, **Tip:** → **सुझाव:**, **Answers:** → **उत्तर:** | |
| `table_row` | Noun phrases; numbers and units as in §6. | "Training time" → प्रशिक्षण का समय |

The cover kicker `[Dive into Deep Learning · §2.1]{.kicker}` is rewritten by the
build; copy it unchanged.

## 9. Known pitfalls

### 9.1 Words with more than one sense

| English | Sense → Hindi | Not |
|---|---|---|
| bias | parameter $b$ → बायस; of an estimator → अभिनति (unbiased → अनभिनत); fixed compounds as in the glossary (इंडक्टिव बायस, चयन बायस); social → पूर्वाग्रह | पूर्वाग्रह for $b$ |
| dimension | → विमा; two-dimensional → द्विविमीय, $d$-dimensional → $d$-विमीय; "2D" stays 2D. All three test translators wrote आयाम; in Hindi physics आयाम is amplitude. | आयाम, द्वि-आयामी, द्विआयामी |
| shape, size, rank, order | tensor shape → शेप; size → आकार (कर्नेल का आकार); matrix rank → रैंक; order of a derivative or tensor → कोटि; sequence order → क्रम | आकार for shape; कोटि for rank |
| learning, training | learn → सीखना; in terms → लर्निंग (लर्निंग रेट, पर्यवेक्षित लर्निंग); train → प्रशिक्षण; the stage names → प्री-ट्रेनिंग, फ़ाइन-ट्यूनिंग | अधिगम, ट्रेनिंग alone |
| generalization, normalization | generalization → सामान्यीकरण; normalization → नॉर्मलाइज़ेशन; normal distribution → प्रसामान्य वितरण; regularization → रेगुलराइज़ेशन | सामान्यीकरण for normalization |
| probability, likelihood | प्रायिकता; संभाविता; "likely", "chance" in plain prose → संभावना | संभावना for either term |
| prediction, estimate, inference | पूर्वानुमान; आकलन; इन्फ़रेंस | भविष्यवाणी, अनुमान |
| mean, average, expectation | माध्य; औसत (औसत पूलिंग); प्रत्याशा, expected → प्रत्याशित | |
| variance, deviation, divergence | प्रसरण; विचलन; डाइवर्जेंस | विचलन for divergence |
| loss, cost, objective, error, risk | Five words, five terms: लॉस, लागत, उद्देश्य फलन, त्रुटि, जोखिम. | हानि; merging them |
| accuracy, precision, recall | सटीकता; the metric → प्रीसिज़न, floating-point → परिशुद्धता; रीकॉल; "Recall that" → याद करें कि | |
| kernel | → कर्नेल in every sense (convolution, कर्नेल फलन, GPU or OS code) | गुठली, अष्टि |
| return | RL → प्रतिफल; of a function → लौटाना, लौटाया गया मान | प्रतिफल for a function's result |
| value | a number, RL, attention → मान (मान फलन; कुंजी, मान) | मूल्य |
| state, action, reward, environment, policy | अवस्था, क्रिया, पुरस्कार, परिवेश, नीति; English inside ऑन-पॉलिसी, ऑफ़-पॉलिसी | राज्य, स्थिति; ऐक्शन; इनाम; पर्यावरण; पॉलिसी alone |
| sample, example | नमूना (नमूना माध्य); the act → सैंपलिंग; a data point → उदाहरण (प्रशिक्षण उदाहरण) | |
| unit | of a network → यूनिट (हिडन यूनिट); a unit quantity → इकाई (इकाई सदिश) | |
| class, square | class label, Python class → क्लास; square → वर्ग (वर्ग त्रुटि); परिकल्पना वर्ग and वर्गीकरण as in the glossary | वर्ग for a label class |
| step, stride | चरण (चरण आकार, समय चरण); स्ट्राइड | |
| image, edge | छवि (छवियाँ, छवियों); an edge in an image → किनारा, the border of an array or region → सीमा | तस्वीर |
| weight | भार | वज़न, वेट |
| derivative, gradient | अवकलज; ग्रेडिएंट; "with respect to $x$" → $x$ के सापेक्ष | डेरिवेटिव, प्रवणता |
| translation | of an image or function → स्थानांतरण; of language → अनुवाद | अनुवाद निश्चरता |
| vocabulary | of tokens → शब्दावली; "terminology" → पारिभाषिक शब्द, नाम | शब्दावली for both in one passage |
| space | → समष्टि (सदिश समष्टि); in statistics समष्टि is also "population" (समष्टि माध्य): name the sense when both occur | अंतरिक्ष |
| use, experiment | use → उपयोग; experiment → प्रयोग | प्रयोग for "use" |
| test | test set → परीक्षण सेट; to test an idea → परखना, जाँचना | |
| performance | of a model → प्रदर्शन; of code → गति | परफ़ॉर्मेंस |
| random, stochastic | यादृच्छिक; स्टोकेस्टिक | |
| memory | hardware → मेमोरी; LSTM stays a name | स्मृति for RAM |
| index, row, column, element, entry | इंडेक्स; पंक्ति; स्तंभ; अवयव; प्रविष्टि | कॉलम, सूचकांक |

### 9.2 Translationese

| Pattern | Write instead | Example |
|---|---|---|
| "using X", "with X", "via X" as का उपयोग करके every time | से | "Using the chain rule, …" ✗ शृंखला नियम का उपयोग करके ✓ शृंखला नियम से |
| "It is important to note that" word for word | Drop it, or ध्यान दें कि … at most once per page | ✗ यह ध्यान रखना महत्वपूर्ण है कि बायस रेगुलराइज़ नहीं होता। ✓ बायस रेगुलराइज़ नहीं होता। |
| "we can" every time as हम … कर सकते हैं | The verb itself, or … किया जा सकता है | "we can compute" → हम निकालते हैं |
| Inanimate subject + "tells us", "allows us to" | इससे पता चलता है कि, इससे हम … कर पाते हैं | ✗ यह परिणाम हमें बताता है कि गहराई मदद करती है। ✓ इससे पता चलता है कि गहराई मदद करती है। |
| "in terms of" as के संदर्भ में, के मामले में | के रूप में, की दृष्टि से, or the verb | "derive … in terms of $\partial \ell/\partial \mathbf{Y}$" → … को $\partial \ell/\partial \mathbf{Y}$ के रूप में व्युत्पन्न करें |
| "for 10 epochs" as के लिए | तक | ✗ 10 एपॉक के लिए ✓ 10 एपॉक तक |
| "if …" without तो | यदि/अगर … तो … | यदि $k$ सम है, तो कोई केंद्रीय पिक्सेल नहीं होता। |
| यह, वह, वे chains for models and data | Repeat the noun | "they overfit" (models) → ये मॉडल ओवरफ़िट करते हैं |
| "there is / there are" as वहाँ … है | A plain है/होता है sentence | "There are three reasons." → इसके तीन कारण हैं। |
| "one of the most widely used" | सबसे अधिक प्रचलित … में से एक | |
| "respectively" | क्रमशः, once, before the first item | … क्रमशः $2 \times 2$ तथा $5 \times 7$ शेप के आउटपुट देती है (*fable*) |
| Idioms translated by image | The meaning | "under the hood" → भीतर ; "from scratch" → शुरू से ; "out of the box" → बिना किसी बदलाव के ; "sanity check" → सरसरी जाँच ; "rule of thumb" → मोटा नियम |
| English plural, genitive or participle on a loan | Hindi forms (§5.3, §5.4) | ✗ लेयर्स, Adam's, ट्रेन्ड |
| A glossary word matched as an everyday verb ("returning outputs", "at the cost of", "spans the image") | Translate the plain meaning and name the false match in NOTES | आउटपुट देती है ; की कीमत पर ; ढक लेता है |

Logical and mathematical phrases are fixed: because → क्योंकि; since …, … →
चूँकि …, इसलिए …; therefore → इसलिए (in proofs अतः); thus → इस प्रकार; however →
लेकिन, हालाँकि; if … then → यदि … तो; only if → केवल तभी जब; if and only if →
यदि और केवल यदि; at most → अधिक से अधिक; at least → कम से कम; "let $x$ be …"
→ मान लें कि $x$ … है; such that → इस प्रकार कि, जिसके लिए; with respect to →
के सापेक्ष; without loss of generality → व्यापकता खोए बिना; for all → सभी … के
लिए; there exists → ऐसा … है कि; almost surely → लगभग निश्चित रूप से; i.i.d. →
स्वतंत्र और समान रूप से वितरित (then i.i.d., as the English); proportional to
→ के समानुपाती; up to sign → चिह्न को छोड़कर; nonzero → शून्येतर.

## 10. Sources

Read or tested for this draft (2026-11-26):

- This repository: `glossary/terms/hi.jsonl`, `glossary/review/hi.md` (the
  three-band policy, taken over in §5.1), `glossary/notes/hi.md`,
  `glossary/notes/hindi-source-leads.md`, `glossary/review/concept-frequency.tsv`.
- The translation test `evals/hi-conv-layer/` (*fable*, *opus*, *sonnet*): all
  DECISIONS and NOTES, and the translations compared block by block. All three
  agree on हम, the danda, Latin digits, NFC with decomposed nukta, and ँ/ं by
  the standard rule; they differ on the English gloss (§5.5), "image",
  "section", "dimension", articles and को (§1), and the gender of code names.
- The build's pandoc (`.venv-build`): a reference followed by a Devanagari
  letter fails, followed by a danda, comma or parenthesis works; ASCII quotes
  become “…” ; `*…*` closed before an ending splits the word.
- `tools/l10n/protect.py` run on Hindi samples: Devanagari digits and Indian
  digit grouping raise the numbers warning; `cjk_adjacent: all` rejects a danda
  after math; `glossary._present` does not match inflected Devanagari forms
  (नमूना in नमूनों, बैच का आकार in बैच के आकार).
- Quarto's language directory in `.venv-build`: it has no `_language-hi.yml`.

Not checked (stated from the drafter's knowledge of them):

- The Central Hindi Directorate spelling standard, NCERT mathematics and
  physics textbooks in Hindi (विमा, आयाम, कोटि, परिवर्त, उपपत्ति, के सापेक्ष,
  यदि और केवल यदि, परिमाण की कोटि) and IGNOU material: none was retrieved for
  this draft.
- The CSTT AI/ML glossary: not reachable (see `glossary/review/hi.md`).
- Hindi ML course material (NPTEL/SWAYAM) and Hindi editions of technical
  books: none was read.

## 11. Settings

`languages.yml`, existing keys (config.py, protect.py): the `hi` entry needs
no change.

- `dir: ltr`, `forum: english`: as they are.
- `check.thousands`: `keep` (the default). Numbers keep the English grouping.
- `check.numbers`: `warning` (the default).
- `check.order`: `warning` (the default); a NOTE acknowledges a reorder (§3).
- `check.cjk_adjacent`: `letters` (the default). Do not set `all`: it rejects
  a danda directly after math or a reference, which this guide prescribes.
- `typography.nbsp`: not set. Its rules are German (z. B., 10 000).

`hi/strings.yml` (existing mechanism, to be filled when the file is created):

- `quarto.crossref-ch-prefix: "अध्याय"`, `quarto.crossref-sec-prefix:
  "अनुभाग"`, `quarto.crossref-fig-prefix` and `-fig-title: "चित्र"`,
  `quarto.crossref-tbl-prefix` and `-tbl-title: "तालिका"`,
  `quarto.crossref-eq-prefix: "समीकरण"`, `quarto.crossref-apx-prefix:
  "परिशिष्ट"`, `quarto.section-title-references: "संदर्भ"`. All are needed:
  Quarto ships no Hindi language file, so every prefix would print in English.
- `book.exercises_heading: "अभ्यास"`, `book.discussions_link: "चर्चा"`.
- `fonts.html.google` with a Devanagari family (for example "Noto Sans
  Devanagari:wght@400;500;700") and that family in `fonts.html.sans`.

Needs new code:

1. **Check: a reference role touching a Devanagari letter or vowel sign**
   (U+0900–U+097F except the danda U+0964) should be an error, as it is for
   CJK; the build drops such a reference silently.
2. **Check: Hindi lint**, as warnings: Devanagari digits ०–९; precomposed
   nukta letters U+0958–U+095F or text that is not NFC; ZWJ, ZWNJ and no-break
   spaces; a full stop or `|` ending a Devanagari sentence.
3. **Glossary check: inflection for Devanagari.** `_present` treats vowel
   signs as non-letters, so oblique forms (नमूनों, के आकार) count as missing.
   Until this is fixed, such warnings are false matches for NOTES.
4. **PDF fonts:** `fonts.pdf.*` has a slot for CJK only; XeLaTeX needs a
   Devanagari font with `Script=Devanagari`, and `pdf.lang: "hi"` must be
   tested with it.
5. **Emphasis:** Devanagari fonts have no italic; the Hindi build should style
   `*…*` in another way (a build setting, not a translator rule).

## 12. For the author to confirm

1. **Imperatives** (§1, §8). Proposed: the -एँ form everywhere (करें, दिखाएँ,
   मान लें). The alternative is the -इए form of NCERT exercises (सिद्ध कीजिए,
   मान लीजिए).
2. **The class line of §5.1**, above all ग्रेडिएंट, लॉस, लेयर, कन्वोल्यूशन and
   न्यूरल नेटवर्क as loans against प्रवणता, हानि, परत, संवलन and तंत्रिका
   नेटवर्क. Proposed: as the glossary has them now.
3. **Number words** (§6). Proposed: मिलियन, बिलियन with the English numeral.
   The alternative is लाख, करोड़, अरब with converted numerals (30 लाख), which
   reads more naturally and breaks the number check.
4. **Nukta** (§2). Proposed: ज़ and फ़ always, क़, ख़, ग़ never.
5. **The English gloss** (§5.5). Proposed: after every class-1 and class-2
   term, school mathematics included (आव्यूह (matrix)), once per page. The
   alternative, as in the German guide, leaves school vocabulary unglossed.
6. **People's names** (§7). Proposed: Latin for people in the text; in terms,
   Devanagari for the listed names only. The alternative is Latin for all, as
   in the German ruling.
7. **"image"** (§9.1). Proposed: छवि, as the glossary (nine entries, one from
   the CBSE manual). *fable* and *opus* chose इमेज to match पिक्सेल, चैनल and
   फ़ीचर मैप.
8. **"dimension"** (§9.1). Proposed: विमा, द्विविमीय (the glossary, NCERT)
   against the आयाम that all three test translators wrote.
9. **Structure words** (§7): अनुभाग for section, साध्य for proposition, उपपत्ति
   for proof (the alternative is प्रमाण), लेम्मा for lemma.
10. **Reference prefixes** (§4, §11): अध्याय, अनुभाग, चित्र, तालिका, समीकरण.
11. **Gender of loans** (§5.2), in particular शेप, लर्निंग रेट and अटेंशन as
    masculine, रैंक, क्लास and यूनिट as feminine, and masculine singular for
    bare symbols, code names and model names.
12. **इंडेक्स** for an array index (§9.1) against the सूचकांक of *fable* and
    *opus*; **यूनिट** for a network unit against the इकाई of *fable* and
    *sonnet*.
13. **Digits and grouping** (§6): Latin digits and 100,000, never ०–९ or
    1,00,000.
14. **Dash and quotes** (§3): the em dash only as a last resort, with spaces;
    ASCII quotes turned into “…” by the build.

## 13. Proposed glossary corrections

All entries are `proposed` in `glossary/terms/hi.jsonl`; the guide's rules
contradict them as they stand.

1. `normal distribution`: प्रसामान्य बंटन → **प्रसामान्य वितरण**. Every other
   distribution in the file (31 entries) is वितरण.
2. `computational graph`: कम्प्यूटेशनल ग्राफ़ → **कंप्यूटेशनल ग्राफ़** (anusvara
   rule; the file has कंप्यूट-सीमित, कंपाइलेशन).
3. `computer vision`: कंप्यूटर दृष्टि → **कंप्यूटर विज़न**; `vision
   transformer`: दृष्टि ट्रांसफ़ॉर्मर → **विज़न ट्रांसफ़ॉर्मर (ViT)**. A field name
   and a model name are class 3 and class 5, and half-translated compounds
   are the fault the review removed elsewhere.
4. `backtracking line search`: बैकट्रैकिंग लाइन सर्च → **बैकट्रैकिंग रेखा खोज**
   (`line search` is रेखा खोज).
5. `initialization` and its four compounds: आरंभीकरण → **इनिशियलाइज़ेशन**, so
   that the noun matches the verb इनिशियलाइज़ करना and the other -ization terms
   of class 3; likewise `tokenization`: टोकनीकरण → **टोकनाइज़ेशन**.
6. Names outside the list of §7 written in Devanagari: `Frobenius norm`
   फ़्रोबेनियस नॉर्म → **Frobenius नॉर्म**; `Lipschitz continuity` लिप्शित्स
   सातत्य → **Lipschitz सातत्य**; `Langevin sampling` लांजेविन सैंपलिंग →
   **Langevin सैंपलिंग**. And the reverse: `Beta distribution` Beta वितरण →
   **बीटा वितरण** (a Greek letter, not a name).
7. `expected reward`, `expected improvement`: अपेक्षित … → **प्रत्याशित
   पुरस्कार**, **प्रत्याशित सुधार** (expectation is प्रत्याशा).
8. Hyphens in pair compounds: `encoder-decoder` एन्कोडर डीकोडर → **एन्कोडर-
   डीकोडर**; `signal-to-noise ratio` संकेत शोर अनुपात → **संकेत-शोर अनुपात**;
   `rate-distortion` दर विरूपण → **दर-विरूपण**; `scaled dot-product attention`
   स्केल्ड डॉट-गुणनफल अटेंशन → **स्केल्ड डॉट गुणनफल अटेंशन** (as `dot product`).
9. To add, since the test showed three renderings each: `dimension` विमा (with
   द्विविमीय), `image` छवि, `index` इंडेक्स, `section` अनुभाग, `deep learning`
   डीप लर्निंग, `machine learning` मशीन लर्निंग, `unit` (network) यूनिट,
   `dense layer` डेंस लेयर.
