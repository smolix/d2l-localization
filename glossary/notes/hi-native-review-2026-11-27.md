# Hindi: glossary changes from the native-speaker review (2026-11-27)

A native Hindi speaker who learned computer science in English read the Hindi
test translations (`evals/hi-conv-layer/`). Sanskritised coinages for terms of
art were unreadable to them (स्थानांतरण समचरता, स्थानांतरण अपरिवर्तनीयता,
स्थानीयता for translation equivariance, invariance and locality), and छवि for
"image" was obscure. The author approved the feedback; the rulings are in
`hi/style.md` §14 and are registered as source `author-ruling-hi-2026-11-27`
(`sources/author-rulings.jsonl`).

**Why it happened.** The test translations were made before `hi/style.md`
existed. The glossary's `proposed` Hindi entries were generated before the
guide's class-3 rule (§5.1: terms of art of machine learning and computing are
the English word in Devanagari, never a coinage) and were never checked
against it. TRANSLATING.md §7 tells translators to use `proposed` entries, so
the translators used the coinages. Ruling 3 now says a `proposed` entry never
overrides the class-3 rule.

**What was done.** All 1,203 Hindi entries were read against `hi/style.md` §5.1
(classes 1–6) and the rulings. 77 entries changed and 2 concepts were added:

| Kind | Entries |
|---|--:|
| Named by the reviewer (`translation-equivariance`, `translation-invariance`) | 2 |
| `image` compounds (ruling 2) | 9 |
| Coinage or paraphrase for a term of art replaced by the loan (ruling 1) | 34 |
| Compound made consistent with a changed head word | 32 |
| New concepts (`image`, `locality`; `unresolved` in the other twelve languages) | 2 |

Status: the two reviewer terms and the two new concepts are `verified_external`
with the author ruling as evidence. Every other changed entry is `proposed`.
Only one changed entry had evidence (`image-recognition`, CBSE manual: छवि
पहचान); its attested form stays as an alternative, and its status went from
`verified_external` to `proposed` because nothing attests इमेज पहचान yet. No
other old form was kept as an alternative: none had evidence. Each entry's
`notes` records the replaced form and the reason.

Spelling follows `hi/style.md` §2: nukta on ज़ and फ़ (डिफ़्यूज़न, रिसेप्टिव
फ़ील्ड, कॉन्फ़िगरेशन), ऐ for the vowel of "batch" (ऐडवर्सेरियल), the half nasal
after ऑ (कॉन्फ़िगरेशन), `-sion` → -ज़न (डिफ़्यूज़न). Hindi words of classes 1
and 2 stay in compounds (इमेज वर्गीकरण, कॉज़ल भाषा मॉडल, बायस-प्रसरण अपघटन,
डीकपल्ड भार क्षय, ऐडवर्सेरियल प्रशिक्षण).

## Changed entries

| Concept | Old preferred | New preferred | Kind | Reason | Status |
|---|---|---|---|---|---|
| `translation-equivariance` | स्थानांतरण समचरता | ट्रांसलेशन इक्विवेरिएंस | named by the reviewer | reviewer's form; समचरता is a coinage | proposed → verified_external |
| `translation-invariance` | स्थानांतरण निश्चरता | ट्रांसलेशन इनवेरिएंस | named by the reviewer | reviewer wrote ट्रांसलेशनल इनवेरिएंस (kept as alternative); निश्चरता is a coinage; ट्रांसलेशनल इनवेरिएंस added as alternative | proposed → verified_external |
| `image` | (new concept) | इमेज | new concept | Ruling 2: image is इमेज, also in every compound (इमेज वर्गीकरण, इमेज पैच). The reviewer found छवि obscure; छवि is not used for this concept. | verified_external |
| `locality` | (new concept) | लोकैलिटी | new concept | The native reviewer's sample for the CNN chapter writes लोकैलिटी (Locality) for this principle; the author approved it (ruling 1). स्थानीयता is a coinage readers do not know. | verified_external |
| `image-augmentation` | छवि संवर्धन | इमेज ऑगमेंटेशन | image → इमेज | image → इमेज; augmentation is a term of art (संवर्धन a coinage) | proposed |
| `image-classification` | छवि वर्गीकरण | इमेज वर्गीकरण | image → इमेज | image → इमेज; classification stays वर्गीकरण | proposed |
| `image-classifier` | छवि क्लासिफ़ायर | इमेज क्लासिफ़ायर | image → इमेज | image → इमेज | proposed |
| `image-generation` | छवि जनरेशन | इमेज जनरेशन | image → इमेज | image → इमेज | proposed |
| `image-patch` | छवि पैच | इमेज पैच | image → इमेज | image → इमेज | proposed |
| `image-recognition` | छवि पहचान | इमेज पहचान | image → इमेज | image → इमेज; the CBSE form छवि पहचान stays as an alternative; old form kept as alternative (attested) | verified_external → proposed |
| `image-resolution` | छवि रिज़ॉल्यूशन | इमेज रिज़ॉल्यूशन | image → इमेज | image → इमेज | proposed |
| `image-segmentation` | छवि सेगमेंटेशन | इमेज सेगमेंटेशन | image → इमेज | image → इमेज | proposed |
| `image-to-image-translation` | एक छवि से दूसरी छवि में रूपांतरण | इमेज-टू-इमेज ट्रांसलेशन | image → इमेज | image → इमेज; the paraphrase एक छवि से दूसरी छवि में रूपांतरण was not a term; GAN term of art transliterated | proposed |
| `adversarial-training` | प्रतिद्वंद्वी प्रशिक्षण | ऐडवर्सेरियल प्रशिक्षण | coinage for a term of art | adversarial is a term of art (GAN); प्रतिद्वंद्वी is a calque; train stays प्रशिक्षण | proposed |
| `alignment` | संरेखण | अलाइनमेंट | coinage for a term of art | term of art (attention alignment, model alignment); संरेखण is a coinage here | proposed |
| `ancestral-sampling` | पूर्वज आधारित सैंपलिंग | एंसेस्ट्रल सैंपलिंग | coinage for a term of art | term of art; पूर्वज आधारित is a coinage | proposed |
| `attention-pattern` | अटेंशन प्रतिरूप | अटेंशन पैटर्न | coinage for a term of art | pattern as the field says it; प्रतिरूप is a coinage in this sense | proposed |
| `bidirectional-context` | द्विदिश कॉन्टेक्स्ट | बाइडायरेक्शनल कॉन्टेक्स्ट | coinage for a term of art | term of art (bidirectional RNN, BERT); द्विदिश is a coinage | proposed |
| `causal-attention` | कारणात्मक अटेंशन | कॉज़ल अटेंशन | coinage for a term of art | term of art; कारणात्मक is a coinage here (causality in statistics stays कारणता) | proposed |
| `co-occurrence` | सह-घटन | को-ऑकरेंस | coinage for a term of art | NLP term of art; सह-घटन is a coinage | proposed |
| `color-jitter` | रंग परिवर्तन | कलर जिटर | coinage for a term of art | augmentation name; रंग परिवर्तन loses the meaning | proposed |
| `configuration-space` | विन्यास समष्टि | कॉन्फ़िगरेशन समष्टि | coinage for a term of art | computing term of art; विन्यास is a coinage | proposed |
| `consistency-distillation` | संगति आसवन | कंसिस्टेंसी डिस्टिलेशन | coinage for a term of art | method name; संगति आसवन is a coinage | proposed |
| `consistency-model` | संगति मॉडल | कंसिस्टेंसी मॉडल | coinage for a term of art | model family name; संगति is a coinage in this sense | proposed |
| `corpus` | पाठ संग्रह | कॉर्पस | coinage for a term of art | NLP term of art; पाठ संग्रह is a paraphrase | proposed |
| `data-augmentation` | डेटा संवर्धन | डेटा ऑगमेंटेशन | coinage for a term of art | augmentation as image-augmentation; संवर्धन is a coinage | proposed |
| `decoupled-weight-decay` | पृथक्कृत भार क्षय | डीकपल्ड भार क्षय | coinage for a term of art | decoupled is the term of art (AdamW); पृथक्कृत is a coinage | proposed |
| `diffusion-model` | विसरण मॉडल | डिफ़्यूज़न मॉडल | coinage for a term of art | ML term of art; विसरण (school physics) is not what the field calls it | proposed |
| `discriminative-model` | विभेदक मॉडल | डिस्क्रिमिनेटिव मॉडल | coinage for a term of art | term of art; विभेदक is a coinage; discriminator is डिस्क्रिमिनेटर | proposed |
| `downstream-task` | अनुवर्ती कार्य | डाउनस्ट्रीम कार्य | coinage for a term of art | downstream is a term of art; अनुवर्ती is a coinage; task stays कार्य | proposed |
| `dynamic-range` | गतिशील परास | डायनेमिक रेंज | coinage for a term of art | number-format term of art; गतिशील परास is a coinage | proposed |
| `identity-shortcut` | तत्समक संक्षिप्त मार्ग | आइडेंटिटी शॉर्टकट | coinage for a term of art | ResNet term of art; तत्समक संक्षिप्त मार्ग is a coinage (identity matrix stays तत्समक आव्यूह) | proposed |
| `knowledge-distillation` | ज्ञान आसवन | नॉलेज डिस्टिलेशन | coinage for a term of art | term of art; ज्ञान आसवन is a coinage | proposed |
| `local-receptive-field` | स्थानीय ग्रहणशील क्षेत्र | लोकल रिसेप्टिव फ़ील्ड | coinage for a term of art | local in the CNN sense as locality (लोकैलिटी); receptive field as above | proposed |
| `named-entity-recognition` | नामित सत्ता पहचान | नेम्ड एंटिटी रिकग्निशन | coinage for a term of art | NLP task name; नामित सत्ता पहचान is a coinage | proposed |
| `parallelism` | समांतरता | पैरेललिज़्म | coinage for a term of art | computing term of art; समांतरता is a coinage in this sense | proposed |
| `perceptual-loss` | बोधात्मक लॉस | पर्सेप्चुअल लॉस | coinage for a term of art | term of art; बोधात्मक is a coinage | proposed |
| `receptive-field` | ग्रहणशील क्षेत्र | रिसेप्टिव फ़ील्ड | coinage for a term of art | CNN term of art; ग्रहणशील क्षेत्र is a coinage | proposed |
| `retrieval` | पुनर्प्राप्ति | रिट्रीवल | coinage for a term of art | term of art (retrieval-augmented generation); पुनर्प्राप्ति is a coinage | proposed |
| `scaling-law` | पैमाना परिवर्तन नियम | स्केलिंग लॉ | coinage for a term of art | term of art; पैमाना परिवर्तन नियम is a paraphrase | proposed |
| `self-supervised-learning` | स्व पर्यवेक्षित लर्निंग | सेल्फ़-सुपरवाइज़्ड लर्निंग | coinage for a term of art | as supervised-learning; self- as in सेल्फ़-अटेंशन | proposed |
| `spatial-invariance` | स्थानिक निश्चरता | स्पेशियल इनवेरिएंस | coinage for a term of art | invariance as translation-invariance; spatial transliterated with it (CNN term of art) | proposed |
| `statistical-bias` | अभिनति | बायस | coinage for a term of art | bias in every sense is बायस; अभिनति is a coinage readers do not know | proposed |
| `subword` | उपशब्द | सबवर्ड | coinage for a term of art | NLP term of art; उपशब्द is a coinage | proposed |
| `supervised-learning` | पर्यवेक्षित लर्निंग | सुपरवाइज़्ड लर्निंग | coinage for a term of art | learning paradigm, a term of art; पर्यवेक्षित is a calque | proposed |
| `textual-entailment` | पाठीय निहितार्थ | टेक्स्चुअल एंटेलमेंट | coinage for a term of art | NLP task name; पाठीय निहितार्थ is a coinage | proposed |
| `unsupervised-learning` | अपर्यवेक्षित लर्निंग | अनसुपरवाइज़्ड लर्निंग | coinage for a term of art | as supervised-learning | proposed |
| `adversarial-objective` | प्रतिद्वंद्वी उद्देश्य | ऐडवर्सेरियल उद्देश्य | compound made consistent | as adversarial-training | proposed |
| `asynchronous-execution` | असिंक्रोनस निष्पादन | असिंक्रोनस एक्ज़ीक्यूशन | compound made consistent | execution as in ईगर एक्ज़ीक्यूशन (verified); निष्पादन is a coinage | proposed |
| `bias-correction` | अभिनति सुधार | बायस सुधार | compound made consistent | as statistical-bias | proposed |
| `bias-variance-decomposition` | अभिनति-प्रसरण अपघटन | बायस-प्रसरण अपघटन | compound made consistent | as statistical-bias; variance stays प्रसरण (school statistics) | proposed |
| `causal-language-model` | कारणात्मक भाषा मॉडल | कॉज़ल भाषा मॉडल | compound made consistent | as causal-attention | proposed |
| `causal-mask` | कारणात्मक मास्क | कॉज़ल मास्क | compound made consistent | as causal-attention | proposed |
| `causal-masking` | कारणात्मक मास्किंग | कॉज़ल मास्किंग | compound made consistent | as causal-attention | proposed |
| `computational-complexity` | संगणकीय जटिलता | कंप्यूटेशनल जटिलता | compound made consistent | computational as in कंप्यूटेशनल ग्राफ़; संगणकीय is a coinage | proposed |
| `computational-efficiency` | संगणकीय दक्षता | कंप्यूटेशनल दक्षता | compound made consistent | as computational-complexity | proposed |
| `consistency-training` | संगति प्रशिक्षण | कंसिस्टेंसी प्रशिक्षण | compound made consistent | as consistency-model | proposed |
| `content-loss` | सामग्री लॉस | कंटेंट लॉस | compound made consistent | pairs with style-loss in style transfer | proposed |
| `data-parallelism` | डेटा समांतरता | डेटा पैरेललिज़्म | compound made consistent | parallelism as above | proposed |
| `diffusion-coefficient` | विसरण गुणांक | डिफ़्यूज़न गुणांक | compound made consistent | diffusion as diffusion-model (doubtful: SDE math, school physics has विसरण) | proposed |
| `diffusion-process` | विसरण प्रक्रिया | डिफ़्यूज़न प्रक्रिया | compound made consistent | diffusion as diffusion-model | proposed |
| `diffusion-transformer` | विसरण ट्रांसफ़ॉर्मर | डिफ़्यूज़न ट्रांसफ़ॉर्मर | compound made consistent | diffusion as diffusion-model | proposed |
| `expert-parallelism` | विशेषज्ञ समांतरता | विशेषज्ञ पैरेललिज़्म | compound made consistent | parallelism as above; expert left as विशेषज्ञ (see doubtful cases) | proposed |
| `forward-diffusion` | फ़ॉरवर्ड विसरण | फ़ॉरवर्ड डिफ़्यूज़न | compound made consistent | diffusion as diffusion-model | proposed |
| `hyperparameter-configuration` | हाइपरपैरामीटर विन्यास | हाइपरपैरामीटर कॉन्फ़िगरेशन | compound made consistent | as configuration-space | proposed |
| `latent-diffusion` | लेटेंट विसरण | लेटेंट डिफ़्यूज़न | compound made consistent | diffusion as diffusion-model | proposed |
| `parallel-scan` | समांतर स्कैन | पैरेलल स्कैन | compound made consistent | parallel as parallelism | proposed |
| `parameter-update` | पैरामीटर अद्यतन | पैरामीटर अपडेट | compound made consistent | update as the verb अपडेट करना (style §5.4); अद्यतन is a coinage | proposed |
| `perceptual-compression` | बोधात्मक संपीड़न | पर्सेप्चुअल संपीड़न | compound made consistent | as perceptual-loss | proposed |
| `pipeline-parallelism` | पाइपलाइन समांतरता | पाइपलाइन पैरेललिज़्म | compound made consistent | parallelism as above | proposed |
| `residual-block` | अवशिष्ट ब्लॉक | रेज़िडुअल ब्लॉक | compound made consistent | as residual-network | proposed |
| `residual-network` | अवशिष्ट नेटवर्क | रेज़िडुअल नेटवर्क | compound made consistent | residual connection, residual stream are रेज़िडुअल; अवशिष्ट stays for the regression residual | proposed |
| `semantic-compression` | अर्थगत संपीड़न | सिमेंटिक संपीड़न | compound made consistent | as semantic-similarity | proposed |
| `semantic-similarity` | अर्थगत समानता | सिमेंटिक समानता | compound made consistent | semantic as in सिमेंटिक सेगमेंटेशन; अर्थगत is a coinage | proposed |
| `spatial-resolution` | स्थानिक रिज़ॉल्यूशन | स्पेशियल रिज़ॉल्यूशन | compound made consistent | spatial as in spatial-invariance | proposed |
| `style-loss` | शैली लॉस | स्टाइल लॉस | compound made consistent | style as in स्टाइल ट्रांसफ़र | proposed |
| `subword-tokenization` | उपशब्द टोकनाइज़ेशन | सबवर्ड टोकनाइज़ेशन | compound made consistent | as subword | proposed |
| `tensor-parallelism` | टेंसर समांतरता | टेंसर पैरेललिज़्म | compound made consistent | parallelism as above | proposed |
| `unbiased-estimator` | अनभिनत आकलक | बायस-रहित आकलक | compound made consistent | as statistical-bias; रहित is plain Hindi (cf. हानिरहित) | proposed |

## Judgement calls made (the reviewer may reverse them)

- **Statistical bias** (`statistical-bias`, `bias-variance-decomposition`,
  `bias-correction`, `unbiased-estimator`): अभिनति → बायस, अनभिनत आकलक →
  बायस-रहित आकलक. अभिनति is the IGNOU statistics term (class 1), but readers
  meet "bias" as one English word in ML; variance stays प्रसरण. The two `bias`
  concepts now share one Hindi form, as they share one English word.
- **Learning paradigms**: पर्यवेक्षित / अपर्यवेक्षित / स्व पर्यवेक्षित लर्निंग →
  सुपरवाइज़्ड / अनसुपरवाइज़्ड / सेल्फ़-सुपरवाइज़्ड लर्निंग, against the
  2026-11-26 review, which had kept पर्यवेक्षित as transparent.
- **diffusion** → डिफ़्यूज़न in all six entries, including the SDE
  `diffusion-coefficient` (school physics says विसरण गुणांक).
- **spatial** → स्पेशियल in `spatial-invariance` and `spatial-resolution`
  (स्थानिक before).
- **parallel, parallelism** → पैरेलल, पैरेललिज़्म in six entries (समांतर is
  also the school-geometry word for parallel lines).
- **local** → लोकल only in `local-receptive-field`; `local-minimum`,
  `local-maximum` and `local-truncation-error` keep स्थानीय (school
  mathematics and numerical analysis).

## Doubtful cases left unchanged

These are Hindi words (not coinages in the narrow sense) or school-mathematics
terms used for a term of art. The guide's rows do not settle them; they are
left as they are for the reviewer.

- **decay**: भार क्षय (weight decay), लर्निंग रेट क्षय, कोज्या क्षय (cosine
  decay), चरघातांकी क्षय. Practitioners say वेट डिके, but भार (weight) is a
  class-2 word by the guide.
- **floating-point precision**: परिशुद्धता in `mixed-precision`,
  `half-precision`, `single-precision`, `double-precision`,
  `floating-point-precision`, against the loan प्रीसिज़न (the metric). The
  guide's §9.1 keeps परिशुद्धता; open as `hi/style.md` §12.16.
- **vanishing / exploding gradient**: लुप्त होता ग्रेडिएंट, विस्फोटक ग्रेडिएंट
  (descriptive Hindi; वैनिशिंग / एक्सप्लोडिंग ग्रेडिएंट is what the field says).
- **greedy**: लालची नीति, लालची डीकोडिंग, लालची खोज (ग्रीडी is the algorithms
  term).
- **global**: वैश्विक पूलिंग, वैश्विक औसत पूलिंग (ग्लोबल पूलिंग); `global-minimum`
  and `global-maximum` are school mathematics and stay.
- **Recommender systems**: अनुशंसा प्रणाली, अनुशंसा, सहयोगी फ़िल्टरिंग (and
  न्यूरल सहयोगी फ़िल्टरिंग), अप्रत्यक्ष / स्पष्ट / नकारात्मक प्रतिक्रिया,
  गुणनखंडन मशीन.
- **NLP tasks in plain Hindi**: भाव विश्लेषण, भाव वर्गीकरण (sentiment), पाठ
  वर्गीकरण, पाठ जनरेशन (text), संक्षिप्तता दंड (BLEU brevity penalty).
- **Mixture of experts**: विशेषज्ञों का मिश्रण (alternative एक्सपर्ट का मिक्सचर),
  साझा विशेषज्ञ, and `expert-parallelism` (now विशेषज्ञ पैरेललिज़्म).
- **Optimization and training**: भार साझाकरण (weight sharing), लॉस परिदृश्य and
  इष्टतमीकरण परिदृश्य (landscape), सार्वत्रिक सन्निकटन (universal
  approximation), तीक्ष्णता (sharpness), विश्वास क्षेत्र (trust region),
  ग्रेडिएंट संचय (gradient accumulation), चरघातांकी औसतीकरण, अनुकूली चरण आकार
  (adaptive; elsewhere एडैप्टिव), न्यूनतमकारी / अधिकतमकारी बिंदु, अर्ध न्यूटन
  (quasi-Newton).
- **Numerics**: छंटाई त्रुटि, स्थानीय छंटाई त्रुटि (truncation; ट्रंकेटेड BPTT
  elsewhere), विमीयता न्यूनीकरण (dimensionality reduction; dimension is विमा by
  the guide).
- **Generative models**: प्रसरण-संरक्षी (VP), प्रसरण-विस्फोटी (VE), चक्र संगति
  (cycle consistency), पुनर्निर्माण त्रुटि.
- **RL**: व्यवहार अनुकरण (behavior cloning), आंशिकतः प्रेक्षणीय (partially
  observable).
- **Uncertainty**: ज्ञानगत अनिश्चितता (epistemic), यादृच्छिक अनिश्चितता
  (aleatoric), अनिश्चितता का परिमाणन.
- **Hardware and interpretability**: अंकगणितीय तीव्रता (arithmetic intensity),
  QK / OV / इंडक्शन परिपथ (circuit), पुनरुत्पादकता (reproducibility).
- **Other**: क्रॉस-सहसंबंध (cross-correlation; correlation is class 1), रुचि का
  क्षेत्र (region of interest), आव्यूह पूर्णन (matrix completion), क्रमिक
  अर्धीकरण (successive halving), वैलिडेशन मापदंड (metric).
- **School mathematics kept by class 1**, though practitioners often say the
  English: अभिलक्षणिक मान / सदिश (eigen-), अतिपरवलयिक स्पर्शज्या (tanh),
  कोज्या समानता (cosine similarity), प्रसरण (variance).
- **Conflicts with the guide's own rules**: `weight-tying` is वेट टाइइंग (भार
  बाँधना), but the guide has weight = भार, ✗ वेट.
- **Spelling of convolution**: the reviewer wrote कॉन्वोल्यूशनल; the glossary
  keeps कन्वोल्यूशन (`hi/style.md` §12.15).
- **Unresolved Hindi entries**: `input`, `output`, `parameter`, `window`,
  `patch`, `edge`, `stack`, `framework`, `program-code`, `dense-layer`,
  `vertical`, `jupyter-notebook` (added for Arabic) have no Hindi term yet,
  although `hi/style.md` §5.1 and §9.1 name इनपुट, आउटपुट, पैरामीटर, विंडो,
  पैच, किनारा, फ़्रेमवर्क, कोड and §13 डेंस लेयर.

## Second round (reviewer, 2026-11-27)

- Confirmed: the English once per page; इमेज for "image" (चित्र would also do, but it is the word for "figure"); कन्वोल्यूशन / कन्वोल्यूशनल, not कॉन्वोल्यूशन; all the judgement calls above.
- **weight → वेट** in 8 entries (network weights; practitioners use the English word): `weight` भार → वेट; `weight-decay` भार क्षय → वेट डिके; `weight-initialization` भार इनिशियलाइज़ेशन → वेट इनिशियलाइज़ेशन; `weight-sharing` भार साझाकरण → वेट शेयरिंग; `importance-weight` महत्त्व भार → इम्पॉर्टेंस वेट (old form kept as an alternative); `decoupled-weight-decay` डीकपल्ड भार क्षय → डीकपल्ड वेट डिके; `attention-weight` अटेंशन भार → अटेंशन वेट; `weight-tying` वेट टाइइंग (भार बाँधना) → वेट टाइइंग. This resolves the weight-decay, weight-sharing and weight-tying items below.
- Still open: the other doubtful cases below.

