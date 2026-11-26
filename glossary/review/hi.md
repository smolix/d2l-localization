# Hindi (hi) glossary review

Reviewed 2026-11-26 against `glossary/terms/hi.jsonl` (1,184 entries: 56 `verified_external`, 1 `invariant`, 1,127 `proposed`). All 57 verified/invariant entries and all 1,127 proposed entries were read against the sense in `concepts.jsonl`. Corrections: `glossary/review/hi-corrections.jsonl` (499 lines).

## Verdict

The glossary is not usable as a constraint file yet. Its two tiers follow opposite policies: the 56 verified entries are English loans in Devanagari (ग्रेडिएंट डिसेंट, लॉस फ़ंक्शन, लेयर, ट्रेनिंग सेट, रेगुलराइज़ेशन), the 1,127 proposed entries are Sanskritised (प्रवणता, हानि, स्तर, प्रशिक्षण, नियमितीकरण). A translator following it would write ग्रेडिएंट डिसेंट and यादृच्छिक प्रवणता अवरोहण in the same paragraph. About 25 head-word rulings account for most of the 499 corrections; 92 entries are wrong in the plain sense.

| Tier | Entries | Changed | wrong sense | calque / MT | non-standard | inconsistent | minor |
|---|---:|---:|---:|---:|---:|---:|---:|
| verified_external + invariant | 57 | 25 (44%) | 0 | 3 | 0 | 18 | 4 |
| proposed, frequency ≥ 50 | 253 | 98 (39%) | 17 | 0 | 42 | 32 | 7 |
| proposed, frequency < 50 | 874 | 376 (43%) | 64 | 8 | 137 | 141 | 26 |
| **Total** | 1,184 | 499 (42%) | 81 | 11 | 179 | 191 | 37 |

No verified entry has a wrong meaning. The verified tier's faults are MT paraphrases (प्रतीक्षा अवधि for latency, चुने जाने से जुड़ा पूर्वाग्रह for selection bias, छिपी हुई लेयर) and head words that contradict the rest of the file. The proposed tier has the real errors: most come from building compounds word by word from a component lexicon (`scripts/build_hi.py`).

## Provenance

There is no earlier Hindi D2L edition, so nothing is inherited from a machine-translated book. The verified tier has two sources:

- **51 entries from Google's Hindi ML glossary pages**, which declare machine translation (`lang="hi-x-mtfrom-en"`). They show that Google's MT system produced the string in this sense, not that a Hindi author chose it. They are internally inconsistent (एन्ट्रॉपी but क्रॉस-एंट्रॉपी; `epoch` left in English).
- **5 entries from the CBSE AI manual** (तंत्रिका नेटवर्क, छवि पहचान, परीक्षण डेटा …): human-written school text, reliable but thin.

Recommendation: treat the Google entries as evidence of loan-word usage, not as binding forms.

## Recommended policy: one rule, three bands

Decide each **word** by its band, and render it the same way in every compound.

| Band | Rule | Examples |
|---|---|---|
| 1. School and first-degree mathematics and statistics, and everyday Hindi words | Hindi term in the NCERT / IGNOU form. English in parentheses on first use in a chapter. | आव्यूह, सदिश, अवकलज, प्रायिकता, प्रसरण, माध्य, फलन, वितरण, प्रतिगमन, प्रत्याशा, संभाविता, इष्टतमीकरण, अभिसरण; प्रशिक्षण, भार, नीति, अवस्था, पुरस्कार, शोर, त्रुटि |
| 2. ML and computing terms of art (first met in an ML course, a paper or framework documentation) | English word in Devanagari, with nukta (फ़, ज़). No coinage. | ग्रेडिएंट, लॉस, लेयर, बायस, फ़ीचर, ऐक्टिवेशन, नॉर्मलाइज़ेशन, रेगुलराइज़ेशन, कन्वोल्यूशन, अटेंशन, एंबेडिंग, टोकन, सैंपलिंग, इन्फ़रेंस, लेटेंट, लर्निंग रेट, ओवरफ़िटिंग, बैच, मेमोरी |
| 3. Names: models, algorithms, optimizers, libraries, datasets, acronyms, symbols | Latin script, never translated or transliterated. Expansion in Latin once, then the acronym. | Adam, ReLU, LSTM, PPO, DQN, word2vec, GloVe, BERT, SGD, ELBO, IoU, Q-लर्निंग, n-ग्राम |

Mixed compounds follow from the rule and are normal Hindi: लॉस फलन, फ़ीचर सदिश, ग्रेडिएंट का नॉर्म, रैखिक प्रतिगमन, बैच नॉर्मलाइज़ेशन.

Why this line. Engineering in India is taught in English; Hindi ML teaching (NPTEL/SWAYAM Hindi courses, Google's Hindi pages, Hindi Wikipedia) keeps the English technical nouns and uses Hindi for grammar and for school mathematics. Hindi Wikipedia has डीप लर्निंग (18 hits) against गहन अधिगम (3), मशीन लर्निंग (78) against यंत्र अधिगम (5), and no article text at all with प्रवणता अवरोहण or प्रबलन अधिगम. School mathematics is the opposite case: NCERT and IGNOU vocabulary is stable and known to every Hindi-medium student.

**CSTT forms that readers would not understand.** The CSTT Information Technology glossary (2005) gives आच्छद (mask), पृच्छा (query), कूटलेखक (encoder), द्रुतिका (cache), प्रसुप्ति (latency), प्रवाह क्षमता (throughput), निस्यंदक (filter), गतिक क्रमादेशन (dynamic programming), यंत्राधिगम (machine learning), प्रतिरूप अभिज्ञान (pattern recognition). The glossary already avoids these. It still carries coinages of the same register, several of them not even the CSTT form: अंशशोधन (calibration), परिमाणीकरण (quantization), विविक्तीकरण (discretization), पूर्वअनुबंधन (preconditioning), समकालिकन (CSTT: तुल्यकालन), संक्रियक (CSTT: संकारक), अतिप्रवाह (CSTT: अधिप्रवाह), देहलीकरण, सुगमीकरण, अनीलित, मास्कित, वर्गकारक, खंडीकरण, व्यतिरेकी, पूर्वभराई, पुनर्चालन, अधिलाभ, मॉडलनीयता, सहप्रकटता, न्यूनक/अधिकतमक. All are replaced by loans in the corrections file. CSTT forms worth keeping because they are transparent: इष्टतमीकरण, आरंभीकरण, अभिसरण, अंतर्वेशन, अनुरेख, विमा, पर्यवेक्षित.

## Systematic problems

1. **Head words rendered two or three ways.** Each row needs one ruling; the corrections file applies the recommended one.

   | English | In the glossary now | Recommended |
   |---|---|---|
   | gradient | ग्रेडिएंट (1 verified), प्रवणता (29) | ग्रेडिएंट |
   | loss | लॉस (5 verified), हानि (17) | लॉस |
   | layer | लेयर (3 verified), स्तर (7); स्तर is also "level" | लेयर |
   | function | फ़ंक्शन (3 verified), फलन (20) | फलन |
   | training | ट्रेनिंग (3 verified), प्रशिक्षण (13) | प्रशिक्षण; प्री-ट्रेनिंग stays as a fixed term |
   | regression, linear | रिग्रेशन, लीनियर (verified); प्रतिगमन, रैखिक (proposed) | रैखिक प्रतिगमन (IGNOU) |
   | learning | सीखने की (verified), अधिगम (15) | लर्निंग; पर्यवेक्षित लर्निंग, लर्निंग रेट |
   | regularization, validation, activation, feature, matrix, vector | loan in verified, Hindi in proposed | रेगुलराइज़ेशन, वैलिडेशन, ऐक्टिवेशन, फ़ीचर (band 2); आव्यूह, सदिश (band 1) |
   | posterior | पश्च, पश्चवर्ती | पश्च |
   | latent / hidden | गुप्त, अव्यक्त / छिपी हुई, छिपी, आंतरिक | लेटेंट / हिडन |
   | forward, reverse | अग्र, अग्रगामी / उलटा, उल्टा, पश्चगामी | फ़ॉरवर्ड, रिवर्स |
   | flow, score, schedule, descent | प्रवाह/फ़्लो, स्कोर/लॉग घनत्व की प्रवणता, अनुसूची, अवरोहण/डिसेंट | फ़्लो, स्कोर, शेड्यूल, डिसेंट |

2. **Sense collisions** (one Hindi word for two concepts; 81 wrong-sense entries).

   | Hindi word | Used for | Fix |
   |---|---|---|
   | सामान्यीकरण | generalization (verified) and normalization (9 entries) | normalization → नॉर्मलाइज़ेशन (CSTT: प्रसामान्यकरण) |
   | अनुकूलन | optimization; अनुकूली = adaptive ("अनुकूली अति अनुकूलन" for adaptive overfitting) | इष्टतमीकरण (CSTT) |
   | संभाव्यता | likelihood; in Hindi usage also probability | संभाविता (IGNOU) |
   | मानक | norm; also standard (मानक विचलन) | नॉर्म |
   | कोटि | rank; in NCERT the order m × n of a matrix | रैंक |
   | स्थिर बिंदु | fixed point and stationary point | नियत बिंदु / स्तब्ध बिंदु |
   | संयोजन | connection, concatenation, combination | कनेक्शन, कॉनकैटनेशन, संयोजन |
   | आकार | shape and size | शेप / आकार |
   | अनुमान | prediction (verified), estimate | पूर्वानुमान / आकलन |
   | ऋणात्मक | negative number; negative sampling, false negative | नेगेटिव for class outcomes |
   | एकल | single (precision) and singular (value) | सिंगुलर मान |
   | सीमा | bound, boundary, limit, threshold | बाउंड, थ्रेशोल्ड; सीमा for limit and boundary |
   | सीमांत | marginal; also used for asymptotic | अनंतस्पर्शी |
   | विचलन | deviation and divergence | डाइवर्जेंस |
   | वर्ग | class and square | क्लास for class labels |
   | मानचित्र, हाशिया, प्रसारण, जनन, आलेख | geographic map, page margin, radio broadcast, biological reproduction, plot | मैप, मार्जिन, ब्रॉडकास्टिंग, जनरेशन, ग्राफ़ |

3. **Descriptions in place of terms** (about 40 entries): अवशिष्ट संयोजनों से आगे बढ़ती निरूपण धारा (residual stream, frequency 81), अब से मिलने वाले पुरस्कारों का योग (reward-to-go), प्रति बाइट अंकगणितीय कार्य की मात्रा (arithmetic intensity), प्रतिद्वंद्वी जनन मॉडल का आलोचक (critic), लॉग घनत्व की प्रवणता (score function). A term must fit into a sentence and a compound.
4. **Translated names** (about 30 entries): समीपस्थ नीति अनुकूलन (PPO), सॉफ़्ट कर्ता आलोचक (SAC), गहन क्यू नेटवर्क (DQN), दीर्घ अल्पकालिक स्मृति (LSTM), वर्डटूवेक (word2vec), वैश्विक सदिश (GloVe), रेक्टिफाइड लीनियर यूनिट (ReLU).
5. **Word-order calques**: बैकप्रॉपैगेशन के माध्यम से समय (BPTT: "time by means of backpropagation"), समाकलन से भाग (integration by parts; NCERT: खंडशः समाकलन), प्रतिच्छेदन अनुपात संघ (IoU), पश्च खोज रेखा खोज, भूल गेट, यादृच्छिक बीज.
6. **Half-translated compounds**: स्व अटेंशन, बहु हेड अटेंशन, परिबद्ध बॉक्स, इंस्टेंस खंडीकरण, गतिशील प्रोग्रामिंग, स्किप संयोजन.
7. **Orthography and script**: श्रृंखला for शृंखला; एन्ट्रॉपी/एंट्रॉपी, एम्बेडिंग/एंबेडिंग, उलटा/उल्टा; hyphens (मिनी-बैच/मिनीबैच, क्रॉस-/क्रॉस); person names in both scripts for the same person (ऑयलर मरुयामा विधि but Euler विधि; यूक्लिडीय दूरी but Euclidean मानक; जैकोबियन but Jacobian सारणिक).

Sound as they stand: the school-mathematics vocabulary (आव्यूह, सदिश, अदिश, सारणिक, अवकलज, प्रायिकता, प्रसरण, माध्यिका, अंतःखंड, तत्समक आव्यूह), the RL core (नीति, अवस्था, क्रिया, पुरस्कार, परिवेश, एजेंट) and the existing loans (टोकन, अटेंशन, ड्रॉपआउट, पूलिंग, टेंसर, not प्रदिश).

## Decisions for the author or a native reviewer

1. **The band-2 line itself**, above all ग्रेडिएंट against प्रवणता (frequency 3,176; प्रवणता is good Hindi and is what Hindi Wikipedia titles the article), कन्वोल्यूशन against संवलन, and न्यूरल नेटवर्क against तंत्रिका नेटवर्क (CBSE). The corrections take the loan in all three; reversing one ruling reverses about 30 lines.
2. **Does भार stay for weight** (and भार क्षय for weight decay)? Kept here; वेट is what Google writes.
3. **NCERT forms where the glossary has a transparent alternative**, left unchanged: वितरण (NCERT बंटन; Hindi Wikipedia text has both), उत्तल (Wikipedia अवमुख), सशर्त (NCERT सप्रतिबंध), स्थानीय न्यूनतम (NCERT स्थानीय निम्नतम), व्यवहार्य (NCERT सुसंगत). Changed to the NCERT form: असमिका, चरघातांकी, व्यवरोध, प्रत्याशा, एकसमान, बेज़.
4. **प्रायर/पोस्टीरियर or पूर्व/पश्च.** Kept पूर्व वितरण, पश्च वितरण; पूर्व also reads "former" and "east".
5. **स्टोकेस्टिक against यादृच्छिक** for stochastic (corrections: यादृच्छिक = random, स्टोकेस्टिक = stochastic).
6. **समष्टि** is both space (सदिश समष्टि) and, in statistics, population (समष्टि माध्य). Standard Hindi lives with this; confirm.
7. **ऑफ़-पॉलिसी / ऑन-पॉलिसी** stay loans although policy is नीति (पॉलिसी alone means an insurance policy).
8. **Person names**: Devanagari for names with a settled Hindi spelling (न्यूटन, टेलर, फूरिये, गाउस, मार्कोव, बेज़, लग्रांज, ऑयलर, यूक्लिड), Latin for the rest (Frobenius, Wasserstein, Cholesky, Kronecker, Nesterov). The style guide needs the list.
9. **Spelling rule for loans**: anusvara before a homorganic stop (एंट्रॉपी, एंबेडिंग, टेंसर), except after the prefixes en-, in-, con- (एन्कोडर, इन्फ़रेंस, कॉन्टेक्स्ट).
10. **Coinages I replaced without a Hindi source for the replacement**: स्तब्ध बिंदु, निश्चरता/समचरता, अनभिनत आकलक, प्रसामान्य लांबिक आधार. A statistician or mathematician who writes Hindi should confirm them.

## Sources and limits

Used: CSTT *Glossary of Information Technology* (2005; PDF text extracted), IGNOU eGyanKosh Unit 16 (simple linear regression) and Block 4 (multivariate analysis; legacy-font text decoded), Hindi Wikipedia (interlanguage titles for about 290 English articles, phrase-search counts for about 70 terms), and the Google and CBSE sources already in `sources.jsonl`.

Not reachable during this review: the CSTT AI/ML glossary and the CSTT mathematics glossary (`shabd.education.gov.in`, `cstt.education.gov.in`: connection refused), `ncert.nic.in`, and the NPTEL/SWAYAM Hindi course pages (rendered by script). NCERT forms are therefore cited from memory and marked so in the corrections; only 17 corrections are `verified_external`, the other 482 stay `proposed`. The CSTT AI/ML glossary should be checked against the band-2 list before the glossary is frozen, expecting it to offer coinages of the kind listed above.
