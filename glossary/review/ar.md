# Arabic (ar) glossary review

Reviewed 2026-11-26 against `glossary/concepts.jsonl` senses. All 1,184 rows
were read (131 `verified_d2l`, 73 `verified_external`, 980 `proposed`), not a
sample. Corrections: `glossary/review/ar-corrections.jsonl` (226 rows).

## Verdict

The glossary is usable as a first draft but not as a binding term list.
"Verified" means "found in a source", and both source families are weak for
Arabic: the D2L source is a machine translation, and the external sources are a
machine-translated Google glossary and volunteer cheat sheets that disagree
with each other.

| Status | Rows | Rows I would change | wrong sense | calque / MT | non-standard | inconsistent | minor |
|---|---|---|---|---|---|---|---|
| `verified_d2l` | 131 | 42 (32%) | 4 | 11 | 3 | 18 | 6 |
| `verified_external` | 73 | 11 (15%) | 3 | 0 | 2 | 4 | 2 |
| `proposed` | 980 | 173 (18%) | 26 | 6 | 23 | 82 | 36 |
| **Total** | 1,184 | 226 (19%) | 33 | 17 | 28 | 104 | 44 |

About 80 of the 226 are substantive (wrong sense, calque, non-standard); the
rest are one head word rendered two or three ways. The ordinary mathematical
and statistical vocabulary (matrix, vector, variance, covariance, convexity,
eigenvalue, expectation, regression, classification) is correct.

## Provenance of the 131 `verified_d2l` rows

All 131 come from one source: the three-part PDF "التعمق في التعلم العميق" by
Dr. Alaa Taima (علاء طعيمة, University of Al-Qadisiyah, Iraq;
`taima-ar-part1/2/3`: 54 + 47 + 30 rows). It is an independent, unofficial
translation of D2L 1.0-alpha (PDFs created September 2022, PyTorch/MXNet/
TensorFlow edition, three-part plan). There is no official Arabic D2L edition;
the other Arabic source, `mohamed-ali/d2l-ar`, has about six translated files
and contributes 10 `proposed` rows only.

The translator's preface calls the work human ("يبقى عملاً بشرياً يحتمل
النقص"), but the body is machine translation with light post-editing. I
downloaded the three PDFs and read the text. Evidence:

- Word-for-word idiom: "تخصصًا أكاديميًا في السماء الزرقاء" (blue-sky
  discipline), "البريد الوارد الكبير بشكل قمعي" (oppressively large inboxes),
  "الكتاب المدرسي الممتاز لكريس بيشوب Chris Bishop's excellent textbook".
- Wrong-sense terms of the kind found in the Japanese edition: translation
  invariance → "ثبات الترجمة" (language translation), automatic
  differentiation → "التمايز التلقائي" (differentiation = distinction), models
  → "الطرز" (product models), code → "رمز", loss function → "دالة الخطأ".
- No term discipline: gradient descent appears as التدرج الاشتقاقي, هبوط
  التدرج and الانحدار الاشتقاقي; overfitting as الضبط الزائد and فرط التجهيز;
  bounding box as مربع محيط, الصندوق المحيط and مربع الإحاطة; weight decay as
  اضمحلال الوزن and تناقص الوزن. The glossary picked one variant of each.
- The PDF metadata title is Persian ("یادگیری عمیق با پای‌تورچ", author "Milad
  vazan"): the Word template of an unrelated Persian book was reused.

Weigh these rows as "attested in a machine translation", the same standing as
the Japanese `translator = aws` edition. The glossary's builder already
avoided the worst Taima errors (ثبات الترجمة is not in the glossary), which is
why the wrong-sense count is lower than for Japanese; the damage here is
calques and inconsistency. Reuse rights for the PDF are not established.

The 73 `verified_external` rows: 43 from Google's ML glossary (page declares
`x-mtfrom-en`, machine-translated), 30 from the Amidi CS229 Arabic sheets
(volunteers; the sheets disagree with each other: features are ميزات / خصائص,
parameters are مُدخلات / مُعاملات, random variables are متغيرات / متحولات,
Gaussian is غاوسي / جاوسي; the Arabic CS229 deep-learning sheet is untranslated).
Two Google citations are for a different sense than the book's: `implicit-bias`
(unconscious human bias) and `shrinkage` (gradient-boosting learning rate).

## Systematic problems

1. **Sense collisions** (same Arabic word, two concepts):
   prediction التوقّع = expectation التوقع; perplexity الارتباك = confusion
   (مصفوفة الارتباك), one letter from correlation الارتباط; recall الاسترجاع =
   retrieval; prompt موجّه = router موجّه; stationary point = critical point
   (نقطة حرجة); total variation = total variance (تباين كلي); variational
   (تغايري) = covariance adjective; overfitting الضبط الزائد vs fine-tuning
   الضبط الدقيق; dropout التسرب vs leaky; minimum الحد الأدنى = lower bound;
   "global" عالمي = worldwide; نقل used for transfer, transport, transpose and
   translation; تهيئة for initialization and configuration; دقة for accuracy,
   numeric precision, resolution and fidelity (the last can move to أمانة).
2. **One English head, several Arabic heads**: descent (نزول / انحدار / هبوط /
   omitted), likelihood (أرجحية / احتمالية / احتمال / إمكان), flow (تدفق /
   جريان, even between flow matching and conditional flow matching),
   normalization (تسوية / تطبيع), residual (متبقٍ / باقٍ), encoder–decoder
   (تشفير / ترميز / شفرة), duality (ازدواجية / ثنائية), condition (شرط / تكييف
   / تهيئة), bias (انحياز / تحيز), product (ضرب / جداء), theorem (مبرهنة /
   نظرية), derivative gender (مشتق / مشتقة), neural (عصبية / عصبونية),
   convolutional (التفافية / تلافيفية), feature (ميزة / معالم), shift (تحول /
   انزياح), action (إجراء / فعل), loss (خسارة / خطأ).
3. **Names transliterated into homographs**: Adam → آدم (the person), He
   initialization → "تهيئة هي" ("she-initialization"), Muon → ميون (particle),
   Shampoo → شامبو, AdamW → آدم دبليو, RMSProp → آر إم إس بروب, InfoNCE → إنفو
   إن سي إي, Q → كيو, F1 → إف 1, top-k → أفضل ك. About 40 rows.
4. **Form**: verified rows carry the article (التدرج), proposed rows do not
   (تدرج); Google/Amidi rows carry partial vowelling (معدّل التعلّم,
   المعلَمة, دالّة), others none; `vc-dimension` is a truncated string with an
   unbalanced parenthesis; `normal-equations` holds two alternatives joined by
   a slash.

## Recommended convention

Arabic ML terminology has no authority that writers follow, and usage differs
by region. One consistent convention for the book:

- **Modern Standard Arabic terms for concepts**, taken from the mathematical
  and statistical tradition where one exists (مشتق, تدرج, إمكان, مبرهنة,
  التفاف, جداء), and established ML usage otherwise (انتباه, تضمين, شبكة
  عصبية).
- **Latin script for names**: optimizers, models, architectures' proper names,
  acronyms, function names that are code identifiers (Adam, AdamW, RMSProp,
  ResNet, BERT, LSTM, GPU, ReLU, softmax, word2vec, DQN, PPO), and all
  mathematical symbols (Q, k, f, n). Give the Arabic expansion once at first
  mention. "Transformer" as a common noun is المحوّل; model names stay Latin.
- **Arabicised loanwords only where entrenched**: إنتروبيا, بكسل, لوجستي,
  لوغاريتم, خوارزمية, ديناميكي, بيرسبترون, لوجيت.
- **Eponyms transliterated**, /g/ always as غ: غاوسي, غرام, لاغرانج, غيبس
  (not جاوسي / جرام).
- **No vowelling** except where a bare form is ambiguous (مَعلَمة vs معلّمة,
  مُحسِّن); glossary lemmas without the article; Western digits; math and code
  left-to-right; English term in parentheses at first mention in a chapter.

## Decisions for the author or a native reviewer

Regional or unsettled choices; the corrections file takes the first option.

1. **token** (2,117 uses): رمز (kept; also means symbol/code) vs وحدة نصية vs
   the colloquial توكن. Decide first; many compounds depend on it.
2. **label** (2,671): وسم (proposed; already in all compounds) vs تسمية.
3. **gradient descent**: النزول التدرجي (proposed) vs الانحدار التدرجي (most
   common informally, collides with regression) vs أصل التدرج (Wikipedia).
4. **overfitting**: فرط الملاءمة (Wikipedia) vs فرط التخصيص (Amidi, Google).
5. **likelihood**: الإمكان (statistics texts, Wikipedia) vs الأرجحية (Amidi).
6. **normalization**: تطبيع (dominant; political connotation) vs تسوية vs
   استنظام. **regularization**: التنظيم (kept) vs الضبط (Amidi).
7. **neural**: عصبية (Egypt, Gulf, majority) vs عصبونية (Levant, Wikipedia).
8. **derivative**: مشتق (Levant, Wikipedia; kept) vs مشتقة (Egypt, Gulf).
9. **theorem**: مبرهنة (proposed) vs نظرية (Egyptian and Gulf school usage).
10. **product**: جداء (Levant, Maghreb) vs ضرب / حاصل الضرب (Egypt, Gulf).
11. **parameter**: مَعلَمة (kept) vs وسيط (Levant) vs بارامتر (Egypt).
    **function**: دالة (kept) vs تابع (Syria). **norm**: معيار (kept) vs نظيم.
12. **encoder / decoder**: مشفّر / مفكك التشفير (proposed) vs مرمِّز / مفكك
    الترميز (purist: تشفير is encryption).
13. **feature**: ميزة (kept) vs سمة vs خاصية. **supervised**: الخاضع للإشراف
    (kept) vs الموجَّه (Saudi usage, Amidi) vs المراقَب (Wikipedia).
14. **precision / accuracy**: sources contradict each other (Amidi: accuracy
    الضبط, precision الدقة; Wikipedia: دقة ومراجعة). Kept: accuracy الدقة,
    precision الإحكام, recall الاستدعاء.
15. **prompt vs router** (both موجّه), **perplexity** (الحيرة), **confusion
    matrix** (مصفوفة الالتباس), **dropout** (الحذف العشوائي), **score** of a
    density, **subgradient**, **equivariance**, **duality** (ثنوية),
    **compile** (تصريف vs ترجمة), **optimization** (التحسين kept; الاستمثال /
    الأمثَلة in mathematics; note policy improvement and PPO both become
    تحسين السياسة), **adversarial** (خصومي kept vs تنافسي, Wikipedia's GAN).

The Saudi SDAIA / King Salman Academy "معجم البيانات والذكاء الاصطناعي"
(1,200+ reviewed terms, on the Siwar platform) is the one institutional
Arabic AI term list; I could not retrieve its entries. A native reviewer
should check items 1–15 against it before the list is frozen.

## Sources used

- Taima PDFs, parts 1–3 (author-hosted Google Drive links in
  `glossary/DISCOVERY.md`), full text read locally.
- Arabic Wikipedia article titles via the English articles' language links
  (overfitting فرط الملاءمة, likelihood دالة الإمكان, CLT مبرهنة النهاية
  المركزية, lemma توطئة, dot product جداء نقطي, bandwidth عرض نطاق, automatic
  differentiation الاشتقاق الآلي, stationary point نقطة مستقرة, overflow طفح,
  perceptron بيرسيبترون, Jacobian مصفوفة ياكوبية, odds ratio نسبة الأرجحية,
  graph بيان, compiler مصرف, and others cited per row).
- Amidi CS229 Arabic sheets at the pinned commit (recall الاستدعاء, prediction
  التنبؤ, unbiased غير منحاز).
- University of Jijel course notes, optimisation chapter (النزول التدرجي):
  https://elearning.univ-jijel.dz/pluginfile.php/276655/mod_resource/content/1/chapitre3.pdf
  (seen through a search summary, not read in full).
- Google ML glossary / crash course in Arabic: machine-translated; its own
  crash-course page titles render gradient descent two ways (انحدار التدرج,
  التدرّج المتّجه). Used only as a negative example.
