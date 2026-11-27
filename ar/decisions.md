# Decisions log: العربية (ar)

## chapter_preface/index.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "Preface" → تمهيد (the glossary's مقدمة is the NLI "premise" and is kept for the Introduction chapter); "Preface to the Second/First Edition" → تمهيد الطبعة الثانية / الأولى; edition → طبعة
- Part names: Basics → الأساسيات; Advanced → المواضيع المتقدمة; Mathematics for Deep Learning → الرياضيات للتعلم العميق; Language Models → النماذج اللغوية; Image Models → نماذج الصور; "the X part" → جزء X
- machine learning → تعلم الآلة (SDAIA; no glossary entry); semi-supervised learning → التعلم شبه الخاضع للإشراف (from the glossary's تعلم خاضع للإشراف)
- agentic AI → الذكاء الاصطناعي الوكيلي; GitHub issues → البلاغات; PRs / pull requests → طلبات السحب (no Latin plural "PRs" in prose); bug report → تقرير أخطاء
- JIT compilation → التصريف في أثناء التنفيذ (JIT) at first mention (glossary term plus acronym)
- hardware → العتاد; state space models → نماذج فضاء الحالات; GANs → شبكات GAN; flows (generative) → التدفقات; speech recognition → التعرف على الكلام; natural language processing → معالجة اللغة الطبيعية
- imports (Python) → عبارات الاستيراد; package → حزمة; module → وحدة; helper function → دالة مساعدة; library header → ترويسة المكتبة
- Signature lines (*Name, date*) → *بقلم Name، date*, so that the block starts with an Arabic word (style §3.2); months as نوفمبر etc.
- Companies stay in Latin like people (style §7, author 2026-11-27): Anthropic, OpenAI, Google, NVIDIA; the game Go → لعبة Go

## chapter_notation/index.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "notation" → الرموز (heading الرموز; "notation table" → جدول رموز; "consistent notation" → نظام رموز); not الترميز, which is the glossary's word for encoding.
- Notation list items keep the symbol first, as in English: `* $x$: عدد قياسي`. A symbol list is the one block type that does not start with an Arabic word. Two or more symbols before the colon are joined by و with spaces (`$\sigma$ و $\sigma_X$:`), replacing the English comma.
- The English a/the convention of the notation lists is rendered by Arabic indefinite/definite nouns: "a scalar" → عدد قياسي, "the identity matrix" → مصفوفة الوحدة / المصفوفة ….
- "the $i^\textrm{th}$ element of" → العنصر $i^\textrm{th}$ في ….
- "given $Y$" (conditioning) → بمعلومية $Y$.
- "argument" (of a function) → وسيط; "subscript", "index" → دليل (سفلي).
- "cardinality" → أصلية المجموعة; "set difference" → فرق المجموعتين; "indicator function" → دالة المؤشر; "antiderivative" → الدالة الأصلية; "divergence" (vector calculus and KL) → تباعد; "Laplacian" → لابلاسي; "differential entropy" → الإنتروبيا التفاضلية; "generative adversarial networks" → الشبكات التوليدية الخصومية; "calculus" → التفاضل والتكامل; "operator" → مؤثر; "univariate / multivariate" → أحادي المتغير / متعدد المتغيرات; "discrete / continuous random variable" → متغير عشوائي متقطع / مستمر.
- Names not in a glossary term stay Latin: Moore–Penrose, Pearson, Kullback–Leibler (المعكوس الزائف بمعنى Moore–Penrose، معامل ارتباط Pearson، تباعد Kullback–Leibler).
- Typography words: lowercase/uppercase → الحروف الصغيرة / الكبيرة; italic → المائلة; bold → الغامقة; sans-serif → عديمة الزوائد; calligraphic → المزخرفة.

## chapter_introduction/index.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "machine learning" → تعلم الآلة; "deep learning" → التعلم العميق (style guide); glossed once on this page.
- "wake word" → كلمة التنبيه; "wake-word recognizer / detector" → متعرف على كلمة التنبيه / كاشف كلمة التنبيه.
- "business logic" → منطق الأعمال.
- "model family" → عائلة النماذج; "learning algorithm" → خوارزمية التعلم.
- "knobs" (the parameter metaphor) → مقابض (not مفاتيح, which is the attention "key").
- "mapping" (noun, a function from inputs to outputs) → دالة; "transformation" → تحويل.
- "example" → مثال (أمثلة); "data point" → نقطة بيانات; "instance" → مثيل (مثيلات; حالة is "state"); "covariates" → المتغيرات المصاحبة.
- "dimensionality" → الأبعادية.
- "token ID" → معرف الرمز المميز; "corpus" → مدونة نصية.
- "surrogate loss" → خسارة بديلة; "error rate" → معدل الخطأ; "capacity" (of a model) → السعة.
- "inference" → الاستدلال; "supervised problems" → مسائل التعلم الخاضع للإشراف.
- "frontier model" → نموذج من نماذج الصدارة.
- Spoken wake phrases (“Hey Siri”, “Alexa”, “OK Google”, “Apricot”) stay in Latin inside “…”; an illustrative quoted utterance ("it stinks!") is translated.
- "garbage in, garbage out" → مدخلات رديئة، مخرجات رديئة.

## chapter_installation/index.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- Installation pages address the reader in the masculine singular (imperatives ثبّت، نزّل، شغّل، نفذ; جهازك) where the English gives the reader a command; statements about the book keep نحن.
- "bundle" (the per-framework .zip) → رزمة (رزم); "package" (Python package) → حزمة (حزم); "archive" → أرشيف; "Python wheel" → حزمة wheel.
- "lock file" → ملف القفل; "pinned" → محدد الإصدارات / يحدد إصدارات الحزم (not تثبيت, which is "install"); "dependencies" → الاعتماديات; "virtual environment" → بيئة افتراضية; "synchronize" → يزامن.
- "interpreter" → مفسر; "installer" → المثبِّت; "terminal" → الطرفية; "directory", "folder" → مجلد; "driver" → برنامج التشغيل; "build" (a built release) → النسخة المبنية; "build scripts" → النصوص البرمجية للبناء; "repository" → المستودع; "clone" → يستنسخ; "cache directory" → مجلد التخزين المؤقت; "compiler flags" → رايات المصرِّف.
- "standard" (imports, preamble, NumPy, models) → معياري, to keep it apart from قياسي "scalar"; "preamble cell" → خلية الديباجة (not تمهيدية: تمهيد is the Preface); "boilerplate" → شيفرة نمطية; "alias" → اسم مستعار; "backend" → الواجهة الخلفية; "companion library" → المكتبة المرافقة; Python "module" → وحدة; "array" (NumPy) → مصفوفة; "stateful / stateless" → ذات حالة / عديمة الحالة; "optimizer" → المحسن (المحسنات); "initializer" → مهيئ.
- Part names as in the preface entries (Advanced → المواضيع المتقدمة, "the Advanced part" → جزء المواضيع المتقدمة).
- The site menu "Notebooks" → قائمة **الدفاتر**, with (Notebooks) at its first mention on a page; "hosted notebook" → دفتر مستضاف; "cloud instance" → مثيل سحابي; "hardware" → العتاد; "laptop" → حاسوب محمول; "software stack" → مجموعة برمجيات; "edition" (web, PDF, published) → طبعة.
- NVIDIA stays in Latin (style §7: companies follow the rule for people); GitHub, Hugging Face, OpenAI, Apple Silicon, Windows, macOS, Linux stay in Latin.

## index.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- Part titles: "Basics" → الأساسيات; "Advanced" → المواضيع المتقدمة; "Language Models" → النماذج اللغوية; "Mathematics for Deep Learning" → الرياضيات للتعلم العميق; "Image Models" → نماذج الصور; "Attic" → الأرشيف; "math appendix" → ملحق الرياضيات
- "edition" (of the book) → الطبعة (الطبعة الأولى، الطبعة الثانية); "Version" (build date) → الإصدار; "Preface" → التمهيد; "Introduction" → المقدمة; "Acknowledgments" → الشكر والتقدير
- "slide deck" → عرض شرائح (عروض الشرائح); "forum" → المنتدى; "framework tab" → تبويب; "outline" (of a chapter) → مخطط; "pull request" → طلب سحب; "download" → نزّل
- Landing page and other direct invitations to the visitor ("Begin here", "you will build"): masculine singular second person (ابدأ، تبني), not نحن
- Names of people, universities, institutes and affiliations in bylines and credit lists stay in Latin script; companies stay in Latin too (NVIDIA), style §7
- "GANs" → شبكات GAN; "flow models" → نماذج التدفق; "interpretability" → قابلية التفسير; "semi-supervised learning" → التعلم شبه الخاضع للإشراف; "graph deep learning" → التعلم العميق على البيانيات; "natural language processing" → معالجة اللغة الطبيعية; "machine learning" → تعلم الآلة; "benchmark" → اختبار معياري
- "serving" (of a model) → تقديم النماذج; "post-training" → التدريب اللاحق; "data curation" → انتقاء البيانات

## chapter_notation/index.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "introduced in" (symbols, section openers) → ترد هذه الرموز أول مرة في …; "these symbols belong to" → تخص هذه الرموز ….
- optimizer → محسن (from the glossary's حالة المحسن); "gradient-based" → يعتمد على التدرج.
- "design matrix" → مصفوفة تصميم; "empirical loss" → الخسارة التجريبية; "target" → الهدف, "target label / vector" → الوسم المستهدف / المتجه المستهدف; "per-example" → للمثال الواحد.
- "attention scoring function" → دالة درجات الانتباه (not دالة الدرجة, which is the score function of a density).
- State space models: "state transition matrix" → مصفوفة انتقال الحالة; "readout" → قراءة المخرجات; "feedthrough (skip) term" → حد التمرير المباشر (التخطي); "latent" → كامن; "selective" → انتقائي; "write strength" → قوة الكتابة; "delta update rule" → قاعدة التحديث دلتا; "matrix-valued state" → حالة على هيئة مصفوفة.
- RL: "stochastic / deterministic policy" → سياسة عشوائية / حتمية; "discounted return" → العائد المخصوم; "state-value function" → دالة قيمة الحالة; "under policy $\pi$" → وفق السياسة $\pi$; "taking action $a$" → اتخاذ الإجراء $a$; "$\epsilon$-greedy" → الطريقة الجشعة $\epsilon$ (epsilon-greedy); "parameterized by $\theta$" → تحددها المعلَمات $\theta$ / معلَماتها $\theta$.
- GANs: "real / generated / synthetic sample" → عينة حقيقية / مولَّدة / اصطناعية; "realness logit" → لوجيت الواقعية; "mixture" → خليط; "minimax game" → لعبة أدنى أعلى (minimax); "log-loss" → الخسارة اللوغاريتمية; "Bayes-optimal" → الأمثل بمعنى بايز; "$f$-divergence" → تباعد $f$; "pairing (relativistic) objective" → الهدف الاقتراني (النسبي); "zero-centered" → متمركز حول الصفر; "conditioning variable / signal" → متغير تكييف / إشارة تكييف (from the glossary's تكييف على الفئة).
- Diffusion: "sampler" → خوارزمية أخذ العينات; "clean data" → البيانات النظيفة; "pure noise" → الضوضاء الخالصة; "noisy state" → الحالة المشوبة بالضوضاء; "marginal density" → الكثافة الهامشية; "velocity field" → حقل السرعة; "$\mathbf{v}$-prediction" → تنبؤ $\mathbf{v}$; "classifier-free guided" → الموجَّه بلا مصنف; "few-step distillation" → التقطير بخطوات قليلة; "teacher model" → نموذج معلم; "the bar" (over a symbol) → الشرطة; "papers" → الأوراق البحثية.

## chapter_introduction/index.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "problem formulation" → صياغة المسألة (صياغات المسائل).
- "class" and "category" (classification) → فئة (فئات), as in the glossary's تصنيف متعدد الفئات; "classifier" → مصنف; "binary" → ثنائي; "single-label" → أحادي الوسم.
- "tagging" → التوسيم (الوسم is the label itself); "detection" in headings → الاكتشاف (from the glossary's اكتشاف الكائنات).
- "death cap" → فطر قبعة الموت.
- "multimodal" → متعدد الوسائط; "vision–language models" → نماذج الرؤية واللغة.
- "personalization" → التخصيص; "censored" (feedback) → خاضعة للحجب; "feedback loop" → حلقة تغذية راجعة; "implicit signals" → الإشارات الضمنية.
- "deploy / deployment" (of a model) → ينشر / النشر.
- "speech synthesis" → توليف الكلام; "speech transcription" → تحويل الكلام إلى نص; "text-to-speech" → تحويل النص إلى كلام; "one-to-one" → واحد لواحد.
- "pipeline" → خط معالجة.
- "clustering" → العنقدة, "cluster" → عنقود (عناقيد); التجميع is the glossary's pooling.
- "manifold" → متعدد الشعب (متعددات الشعب).
- "generative modeling" → النمذجة التوليدية; "diffusion" (models) → الانتشار; "synthesize" (samples) → تخليق.
- "base model" → النموذج الأساسي (leaving النموذج التأسيسي for "foundation model"); "backbone" → الهيكل; "general-purpose" → متعدد الأغراض.
- "human-preference alignment" → المواءمة مع تفضيلات البشر (المحاذاة is sequence alignment); "reasoning" → الاستنتاج (الاستدلال is inference).
- "offline" → دون اتصال; "post-training" → التدريب اللاحق.
- "credit assignment" → إسناد الفضل; "partial observability" → قابلية المشاهدة الجزئية; "exploitation" → الاستغلال.
- "verifier" → متحقق (متحققات); "unit tests" → اختبارات الوحدات; "shell" → الصدفة; "multi-turn" → متعدد الأدوار.
- Place names in Arabic letters: مانهاتن، بيتسبرغ، بريمن; Atari and the game Go stay Latin. The acronyms CT and ICU are kept in parentheses after their Arabic expansion.
- Emphasized terms joined by "and": the و goes inside the emphasis markers with its word (*والاستكشاف*), per style §3.3.

## chapter_preface/index.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- Part names (continued): Tools for Deep Learning → أدوات التعلم العميق; Attic → الأرشيف (the part keeps first-edition chapters; العلية is not used); "appendix" for the math part → ملحق الرياضيات للتعلم العميق
- A framework "edition" of the book (MXNet/PyTorch/TensorFlow/JAX edition) → نسخة (طبعة is kept for the first/second edition); tab → تبويب; backend → الواجهة الخلفية (الواجهة الخلفية PyTorch); "pinned" (version, environment) → المحدد في إعدادات البيئة، البيئة محددة الإصدارات (as on the installation page; not مثبَّت, which reads as "installed"); release/version → إصدار; environment configuration → إعدادات البيئة
- architecture → معمارية (معماريات); optimizer → محسن (محسنات), as in the glossary's حالة المحسن; practitioner → ممارس; workflow → مسار عمل; textbook → كتاب دراسي; tutorial → درس (تطبيقي / تفاعلي); resource (learning resource, "a unified resource") → مرجع / مصادر التعلم
- API → واجهة API (واجهات API عالية المستوى; "API reference" → مرجع واجهة API); class (Python) → صنف (أصناف); boilerplate → الشيفرة النمطية; code block → كتلة شيفرة; cell (notebook) → خلية; preamble cell → خلية الديباجة; namespace → فضاء الأسماء; "marked with `#@save`" → الموسومة بالعلامة `#@save`
- runnable code → شيفرة قابلة للتشغيل; executable → قابلة للتنفيذ; working example / working model → مثال عامل / نموذج عامل; self-contained → مكتف بذاته; "from the ground up" → من أساسه; formal (theory, generality) → صوري
- "fit a model to data" → يلائم النموذج مع البيانات (from the glossary's فرط الملاءمة); intuition → حدس (حدوس); time series → السلاسل الزمنية; recurrence → التكرار; recurrent sequence models → نماذج التسلسل المتكررة; tabular (RL problems) → جدولية; "solve exactly" → يحل حلًا مضبوطًا; corruption process (diffusion) → عملية إفساد; generative modeling → النمذجة التوليدية
- alignment of language models (via preferences and rewards) → المواءمة; the glossary's محاذاة is the sequence-alignment sense and is not used for it
- serving (models) → تقديم النماذج; "in production" → في بيئة الإنتاج; distributed training → التدريب الموزع; provisioning → تجهيز
- functional analysis → التحليل الدالي; statistical inference → الاستدلال الإحصائي; statistical estimation → التقدير الإحصائي; generalization bounds → حدود التعميم; differential equations → المعادلات التفاضلية; multivariable derivatives → المشتقات متعددة المتغيرات; vectorize (JAX) → يحول إلى صيغة متجهية; compile → يصرّف
- Amazon stays in Latin (style §7); Roman enumerators (i)…(v) are copied, with a free-standing و before the last one
- English book titles stay in English after the class noun كتاب (كتاب *Linear Analysis*); "X's textbook" → الكتاب الدراسي الذي وضعه X

## chapter_preface/index.md#u03 (claude fable-5.2 (medium), 2026-11-27)

- Credit lists of names and GitHub IDs are copied as one Latin run with their Latin commas; only the final "and" becomes a free-standing و. No و between the items of a long name list (short lists of people in a sentence still take و).
- Company and publisher names stay in Latin script (style §7, author 2026-11-27): Baidu, Amazon Web Services, Cambridge University Press (with the class noun دار النشر).
- "commissioning editor" → محرر التكليف; "the press" → دار النشر; "errata" → أخطاء; "broken links" → روابط معطلة; "GitHub IDs" → معرفاتهم على GitHub; "adapting code into X implementations" → تحويل الشيفرة إلى تنفيذات في X; "July" → يوليو.

## chapter_introduction/index.md#u03 (claude fable-5.2 (medium), 2026-11-27)

- "kernel methods" → طرق النواة; "decision trees" → أشجار القرار; "graphical models" → النماذج البيانية; "gated recurrences" → البنى المتكررة ذات البوابات.
- "compute" (noun) → الحوسبة (حوسبة التدريب) or, as a growing resource, القدرة الحاسوبية; "training run" → جولة تدريب; "pass" (over the data) → مرور (مرورات).
- "nonlinearity" → لاخطية (لاخطيات); "biological circuits" → الدارات البيولوجية; "hardware sweet spot" → النقطة المثلى للعتاد.
- "foundation model" → نموذج تأسيسي (النموذج الأساسي stays "base model"); "preference alignment" → المواءمة مع التفضيلات; "multi-step reasoning agents" → وكلاء استنتاج متعدد الخطوات.
- "top-5 error rate" → معدل خطأ top-5; "accelerator" → مسرع (مسرعات); "imperative" (framework) → أمري; "compilation" → التصريف; "open-source" → مفتوح المصدر.
- "feature extractor" → مستخلص ميزات; "feature engineering" → هندسة الميزات; "hand-engineered / hand-crafted" → مهندَس يدويًا / مصمَّم يدويًا; "shallow classifier" → مصنف ضحل; "differentiable" → قابل للاشتقاق; "descriptor" (SIFT) → واصف.
- "predictable scaling" → التحجيم القابل للتنبؤ; "power law" → قانون قوة; "frontier" (of a loss curve family) → حد; "envelope" → غلاف; "bitter lesson" → الدرس المر; "heuristics" → قواعد اجتهادية.
- "long tail" → الذيل الطويل; "autonomous vehicles" → المركبات ذاتية القيادة; "full autonomy" → الاستقلالية التامة; "reproducible code" → شيفرة قابلة لإعادة الإنتاج; "structured spatial perception" → الإدراك المكاني ذو البنية.
- People: برنولي، غاوس، شانون، فيشر in Arabic letters (their names are in glossary terms); Legendre, Turing, Hebb, Rich Sutton, Canny stay Latin. Carnegie Mellon stays Latin after جامعة; "Nobel Prize" → جائزة نوبل.
- Table cells "Name, $10^k$ items" → "Name من $10^k$ …" so that an Arabic word separates the Latin name from the math span.

## chapter_introduction/index.md#u04 (claude fable-5.2 (medium), 2026-11-27)

- "heuristic" (noun, a design choice or rule) → قاعدة اجتهادية; "heuristic design choices" → خيارات تصميم اجتهادية; "hand-tuned" → مضبوط يدويًا; "threshold" → عتبة ("decision threshold" → عتبة القرار).
- Exercise titles in bold are translated as noun phrases with the full stop inside the bold; sub-items of a task list ("Estimating …", "Drawing …") are verbal nouns (تقدير، رسم، كتابة).
- "edible mushroom" → فطر صالح للأكل; "discard" (the mushroom) → رمي.
- "dense transformer" → المحول الكثيف; "exact kernel methods" → طرق النواة المضبوطة; "on the order of $n^2$" → من مرتبة $n^2$; "datum" → نقطة بيانات; "byte", "terabyte" → بايت، تيرابايت.
- "driver-assistance system" → نظام مساعدة السائق; "perception network" → شبكة إدراك; "motion planner" → مخطِّط حركة (kasra+shadda against مخطط "diagram"); "controller" → متحكم; "modular stages" → مراحل مستقلة; "dashcam" → كاميرا لوحة القيادة.
- "online experiment" (on a deployed system) → تجربة على النظام المنشور (not دون اتصال/متصل wording); "unbiased sample" → عينة غير منحازة; "overstate / understate" → تبالغ في تقديره / تبخسه حقه.
- "Consider a …" in exercises → لنأخذ …; "Suppose" → لنفترض أن.

## front-matter consistency pass (claude fable-5.2 (medium), 2026-11-27)

- One rendering per term across index, preface, installation, notation and introduction; the entries above were edited to agree. Chosen forms: Advanced part → جزء المواضيع المتقدمة; Attic → الأرشيف; natural language processing → معالجة اللغة الطبيعية; post-training → التدريب اللاحق; serving → تقديم النماذج; preamble cell → خلية الديباجة; "standard" → معياري; "pinned" → محدد (الإصدارات); boilerplate → شيفرة نمطية; "standard imports" → عبارات الاستيراد المعيارية.
- "architecture" (of a model) → معمارية (معماريات) on every page; بنية (بنى) is "structure" (بنية الكتاب، البنى ثلاثية الأبعاد للبروتينات) and stays in the fixed البنى المتكررة ذات البوابات.
- Companies in Latin without exception (Amazon, Google, NVIDIA).

## chapter_basics/index.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- A chapter range "(:numref:`a`--:numref:`b`)" → (من :numref:`a` إلى :numref:`b`), also inside bold stage titles (style §3.2: ranges in prose with words).
- `if`--`else` is copied as one left-to-right run (the name of the construct); no و between the two code spans.
- "from scratch" → *من الصفر*; "concisely" → *بإيجاز*; "concise implementation" → التنفيذ الموجز; "building block" → لبنة بناء; "black-box" → تعمل عمل الصندوق الأسود.
- "vanishing gradients" → تلاشي التدرجات; "regularizer" → منظِّم (kasra+shadda against منظم "organized"); "high-capacity" → عالي السعة; "decision boundary" → حد القرار (حدود القرار).
- "fully connected network" → شبكة كاملة الاتصال; "spatial locality" → المحلية المكانية; "symmetries" → التناظرات; "striding" → خطوة الانزلاق; "design space" → فضاء التصميم; "multi-branch" → متعدد الفروع.
- "autoregressive" → ذاتي الانحدار; "decoding" → فك التشفير; "byte-level BPE tokenization" → التجزئة إلى رموز بخوارزمية BPE على مستوى البايتات; "$n$-gram models" → نماذج $n$-gram; "grayscale" → بتدرج الرمادي; "pixel intensities" → شدة البكسلات.
- Overview pages address the reader with نحن ("you will have built" → سنكون قد بنينا); English plural acronyms in a gloss become the singular: (multilayer perceptron, MLP), (convolutional neural network, CNN), (recurrent neural network, RNN).
- Book titles and their authors stay in Latin after class nouns: رواية *The Time Machine* للكاتب H. G. Wells.

## chapter_preliminaries/index.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "Preliminaries" (chapter title) → المبادئ الأولية (not تمهيد, which is the Preface); "prerequisites" → المتطلبات المسبقة; "crash course" → دورة مكثفة.
- "scale" (what deep learning adds; "scale breaks scalar code") → الحجم; the scale of a feature ("scale gaps") → مقياس; "size" of a vector (norms) → مقدار.
- "tensor library" → مكتبة موترات (مكتبات الموترات); "compiled operations" → عمليات مصرَّفة; "tabular data" → البيانات الجدولية; "categorical" → فئوي; "raw files" → الملفات الخام; "missing entries / gaps" → قيم مفقودة.
- "backward pass" → تمرير خلفي (glossary); "backward sweep" → اجتياز خلفي; "vector–Jacobian product" → جداء المتجه بالمصفوفة اليعقوبية; "matrix product" → جداء المصفوفات; "matrix calculus" → التفاضل المصفوفي.
- "DAG" on slides → مخطط DAG (مخطط التنفيذ DAG); "control flow" → تدفق التحكم; "in-place" → في الموضع; "memory aliasing" → تشارك الذاكرة (not the glossary's تراكب الترددات, the signal-processing sense).
- "buffer" → مخزن مؤقت; "contiguous" (memory, buffers, strides) → متجاور on every page (متصل stays "continuous" and a solid line); "unboxed" → غير معلَّب (with the mark); "impute" → يعوض (تعويض القيم المفقودة); "leakage" (data) → تسرب.
- "error bars" → أشرطة الخطأ; "confidence bands" → نطاقات الثقة; "finite-sample" → للعينات المحدودة; "screening" (medical) → الفحص المسحي; "asymptotics" → السلوك التقاربي; "Bayes' rule" → قاعدة بايز.
- Slide enumerators "Pillar 1 — …" → الركيزة 1، … (the em dash becomes a comma); "`a` / `b`" between two code spans → `a` أو `b`.

## chapter_preliminaries/autograd.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "automatic differentiation (*autograd*)" → *الاشتقاق التلقائي* (automatic differentiation, *autograd*) at first mention, then الاشتقاق التلقائي; "autograd" as a common noun in slides and titles → الاشتقاق التلقائي; "AD" → الاشتقاق التلقائي (no Latin acronym)
- "sweep" (backward / reverse / forward) → اجتياز (اجتياز خلفي / عكسي / أمامي); "pass" stays تمرير (glossary); "traverse" → يجتاز
- "trace" (the recorded operations) → الأثر (الأثر المسجل، أثر التنفيذ، الأثر الأمامي); "to trace" → يتتبع; "gradient tracking" → تتبع التدرج; "mark for gradients" → يضع علامة على … لتتبع التدرج
- "buffer" → مخزن مؤقت (مخزن التدرج المؤقت); "gradient tape" → شريط التدرج; "to cache" → يخزن مؤقتًا; "to free" (memory) → يحرر
- "node" → عقدة; "elementary / primitive operation" → عملية أولية; "sensitivities" → الحساسيات; "local rule" → القاعدة المحلية
- "seed" (of a sweep) → بذرة ("seeded with $\mathbf{v}$" → بذرته $\mathbf{v}$); "adjoint" → قرين (القيم القرينة), from the glossary's طريقة المؤثر القرين; "tangent" → مماس
- "vector–Jacobian product" → جداء المتجه بالمصفوفة اليعقوبية (جداء VJP); "matrix–vector product" → جداء مصفوفة بمتجه (products take بـ, as in the glossary's جداء المصفوفة الهيسية بمتجه; the verb stays يضرب … في …); "materialize" (a Jacobian) → يكوّن صراحةً
- "detach" → يفصل (الفصل); "attached / detached" → موصول / مفصول; "view" (of a tensor) → عرض; "ancestors" (in the graph) → الأسلاف; "history" → التاريخ; "subgraph" → مخطط جزئي; "leaf tensor" → موتر ورقي
- "accumulate gradients" → مراكمة التدرجات، تراكم التدرجات (التجميع is pooling); "micro-batch" → دفعة جزئية; "stale gradient" → تدرج قديم; "by default" → افتراضيًا
- "pure function" → دالة نقية; "side effects" → آثار جانبية; "functional" (approach, transform, programming style) → وظيفي ("purely functional" → وظيفي بحت); دالي stays for functional analysis; "mutable" → قابل للتغيير; "stateful / stateless" → ذات حالة / عديمة الحالة; "in place" → في موضعه
- "bootstrap target" → هدف البوتستراب; "deep Q-learning" → تعلم Q العميق; "validation" → التحقق; "line model" → نموذج الخط المستقيم; "residuals" → البواقي; "analytical gradient" → التدرج التحليلي; "running example" → المثال المستمر
- "symbolic algebra" → الجبر الرمزي; "expression swell" → تضخم التعابير; "subtraction cancellation" → الإلغاء عند الطرح; "reverse topological order" → ترتيب طوبولوجي معكوس; "opt in" (to gradient tracking) → تفعيل
- Tab notes that begin with a code span in English begin with a class noun in Arabic (الدالة، الاستدعاء، الكتلة), style §3.2; an imperative addressed to the reader inside a tab note stays a masculine singular imperative (امسح، اجمع)

## chapter_preliminaries/pandas.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "preprocessing" → المعالجة المسبقة; "preprocessing pipeline" → خط المعالجة المسبقة; "tabular data" → البيانات الجدولية; "raw" → خام; "record" → سجل; "field" → حقل; "column header" → ترويسة العمود.
- "imputation" → التعويض (معوَّض; "mean imputation" → التعويض بالمتوسط); "deletion" → الحذف; "missingness indicator" → مؤشر فقدان البيانات; "missing completely at random" → مفقودة عشوائيًا تمامًا; "observed entries" → العناصر المرصودة; "spread" → التشتت.
- "categorical" → فئوي (الميزات الفئوية); "indicator column" → عمود مؤشر; "dummy variable encoding" → ترميز المتغيرات الوهمية; "high-cardinality" (categorical column) → كثير الفئات (كثرة الفئات).
- "scale" (of a feature) → مقياس (مقاييس); "scaling" → التحجيم; "unscaled" → دون تحجيم; "z-score standardization" → التوحيد المعياري بدرجة z.
- DataFrame stays in Latin, with the class noun جدول where grammar needs one (جدول DataFrame); "row index" of a DataFrame → فهرس الصفوف; "sentinel" → قيمة حارسة; "slicing" → الاقتطاع, اقتطاع الشرائح ("positional slicing" → الاقتطاع الموضعي); "select" (columns) → ينتقي; "promoted" (dtype) → رُقّي; "bit pattern" → نمط بتات; "Boolean" → منطقي; "buffer" → مخزن مؤقت; "flag" (0/1 column) → راية.
- "schema" → مخطط البيانات (المخطط); "data leakage" → تسرب البيانات (يسرّب); "training split" → قسم التدريب; "trade-off" → مقايضة; "quartiles" → الربيعيات; "survey respondents" → المشاركون في الاستبيانات; "luxury estates" → العقارات الفاخرة.
- "(Exercise 1)" → (التمرين 1); slide "(cont.)" → (تابع); unit strings "(sq ft)" and "($)" in slides are copied.

## chapter_preliminaries/probability.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "Probability and Statistics" (the subject, headings) → الاحتمالات والإحصاء; "probability theory" → نظرية الاحتمالات; *probability* as a quantity or function → الاحتمال.
- coin toss → رمية العملة; heads → الصورة, tails → الكتابة, each with (heads)/(tails) once on the page because math and code output keep the English words; fair coin → عملة عادلة; biased → منحازة; die roll → رمية نرد.
- "frequency" (relative frequency of an outcome) → التكرار النسبي; "running frequency" → التكرار النسبي التراكمي; "repeat an experiment" → إعادة التجربة (so that تكرار is not used in two senses).
- "estimator" → مقدر; "unbiased" → غير منحاز; "sampling fluctuation" → تقلبات أخذ العينات; "sampling error" → خطأ المعاينة (glossary); "realizations" → تحققات; "draws" → سحبات; "sample size" → حجم العينة; "counts" → الأعداد (المشاهدة); "spread" → التشتت.
- "multinomial" (sampler, distribution) → متعدد الحدود (التوزيع متعدد الحدود); "PRNG key" → مفتاح PRNG; "seeded from" → نأخذ بذرته من.
- "sample space" → فضاء العينة; "event" → حدث; "outcome" → ناتج (نواتج); "subset" → مجموعة جزئية; "axioms" → مسلمات (مسلمات كولموغوروف الثلاث); "nonnegativity" → اللاسالبية; "normalization" (axiom) → التطبيع; "countable additivity" → الجمعية العدودة; "mutually exclusive" → متنافية; "complement rule" → قاعدة المتمم; "inclusion–exclusion formula" → صيغة الاشتمال والاستبعاد (not التضمين, which is embedding).
- "probability mass" → كتلة احتمالية; "marginalization" → التهميش; "summing out" → الجمع على قيم المتغير; "factorizes into the product" → يتحلل إلى جداء; "probability-weighted average" → المتوسط المرجح بالاحتمالات; "indicator" (random variable) → المتغير المؤشر; "linear combination" → تركيب خطي; "pairwise covariances" → التغايرات الزوجية; "dimensionless" → عديم الأبعاد; "co-movement" → التحرك المشترك.
- "Bessel's correction" → تصحيح Bessel (the name is in no glossary term, so Latin).
- "aleatoric uncertainty" → عدم اليقين الصدفي; "epistemic" → المعرفي (glossary); "irreducible / reducible" → غير قابل للاختزال / قابل للاختزال; "conditioning" (probability) → الاشتراط (from احتمال شرطي; التكييف stays for conditioning signals of generative models); "error bars" → أشرطة الخطأ.
- "reasoning under uncertainty" → الاستنتاج في ظل عدم اليقين; "formalize" → يصوغ صياغة صورية; "calibrated" → معايَر; "asymptotic consistency" → الاتساق التقاربي ("asymptotic" → تقاربي; الخط المقارب is the asymptote); "the LLN" → قانون LLN after the first full mention.
- English glosses on this page so far (do not repeat in later units): standard error, sampling, multinomial, frequency, estimator, law of large numbers, sample space, event, random variable, probability density, joint distribution, marginal distribution, marginalization, expectation, variance, standard deviation, covariance, correlation, covariance matrix, unbiased, heads, tails. Conditional probability and Bayes' theorem are named in the opening paragraph without a gloss; they can take it where the page defines them.

## chapter_preliminaries/linear-algebra.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- Tensor "order" → مرتبة (مراتب), as on the notation page (الموترات من المراتب الأعلى; موتر من المرتبة $2^\textrm{nd}$); "rank" as its synonym on slides → رتبة; "axis" → محور (محاور); "leading / trailing axes" → المحاور المتقدمة / المحوران الأخيران; "trailing dimensions" → البُعدان الأخيران.
- "$n$-dimensional vector" → متجه بُعده $n$ (as on the notation page); "dimensionality" → الأبعادية; "component" (of a vector) → مركبة (مركبات); "entry", "element" → عنصر (عناصر); "square (matrix)" → مربعة; "main diagonal" → القطر الرئيسي; "inverse" → المعكوس.
- "reduction" → اختزال (اختزالات), "the reduced axis" → المحور المختزل; "row sums / column sums" → مجاميع الصفوف / مجاميع الأعمدة; "column means" → متوسطات الأعمدة; "normalize a row by its sum" → يطبّع، تطبيع; "centering" → مركزة.
- "product" (dot, Hadamard, matrix) → جداء (جداءات); "matrix--vector product" → جداء مصفوفة بمتجه; "matrix--matrix product" → جداء مصفوفة بمصفوفة; "matrix product" → جداء المصفوفات; the verb "multiply A by B" → يضرب … في ….
- "linear combination" → تركيب خطي; "weighted combination / sum / average" → تركيب / مجموع / متوسط موزون; "standard basis vector" → متجه الأساس المعياري; "spanned by" → الذي يولده; "unit square" → مربع الوحدة; "parallelogram" → متوازي الأضلاع; "rotation" → دوران; "scaling" (geometric, and "rescale") → تحجيم; "shear" → قص.
- Python "operator" (`@`, `*`) → العامل (العامل البيني for "infix operator"); مؤثر stays the mathematical operator and معامل the coefficient. "method" of a tensor (`.sum`) → دالة. "tuple" → مجموعة مرتبة (tuple). "indexing" → الفهرسة ("1-based / 0-based indexing" → الفهرسة تبدأ من 1 / من 0); "string" → سلسلة نصية; "loop" → حلقة.
- "grayscale images" → صور بتدرج الرمادي; "category offsets" ($\mathbf{b}$ in the opening example) → إزاحات الفئات; "category scores" → درجات الفئات; "class templates" → قوالب الفئات; "pre-activation" → قيمة ما قبل التفعيل; "gating" → آليات البوابات.
- "FLOPs" in prose → عمليات FLOP (class noun, no Latin plural); inside math `\text{ FLOPs}` is copied.
- "a stack of matrices" → مجموعة مكدسة من المصفوفات; "factor" of a product (einsum) → طرف في الجداء.
- English glosses given in u01 of this page (do not repeat in later units): minibatch, forward pass, broadcasting, tensors, reductions, norms, eigenvalues, order, scalars, dimensionality, tuple, transpose, Hadamard product, dot product, inner product, linear combination, standard basis vector, identity matrix.

## chapter_preliminaries/calculus.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "calculus" → التفاضل والتكامل (page title, kickers); "differential calculus" → حساب التفاضل; "integral calculus" → حساب التكامل; "matrix calculus" → التفاضل المصفوفي; "differentiation" → الاشتقاق; "analytical" → تحليلي.
- "difference quotient" → نسبة الفرق; "secant line" → الخط القاطع, then القاطع; "tangent line" → المماس; "slope" → ميل (ميول); "limit" → نهاية; "perturbation" → اضطراب, "perturb" → يغيّر.
- "product rule" → قاعدة الجداء; "quotient rule" → قاعدة القسمة; "first-order linear approximation" → التقريب الخطي من المرتبة الأولى; "expand" (an expression) → يفك, "expansion" → فك (Taylor: مفكوك تايلور).
- "steepest ascent / descent" → الصعود الأشد / النزول الأشد (not الانحدار, which is regression); "uphill / downhill" → صعودًا / نزولًا.
- "level set" → مجموعة المستوى; "level curves" → منحنيات المستوى; "contours" → خطوط الكفاف; "zigzag" → يتعرج; "valley" → الوادي.
- "affine" → تآلفي (خط تآلفي، تحويل تآلفي, from the glossary's تحويل تآلفي); "coordinate" → إحداثي; "identity" (equation) → متطابقة; "component" (of a vector) → مركبة; "residual vector" → متجه البواقي.
- "machine epsilon" → إبسيلون الآلة; "catastrophic cancellation" → الإلغاء الكارثي; "digits" (of a floating-point number) → خانات; "64-bit / 32-bit" → ذو 64 بتًا / ذات 32 بتًا; "asymptote" → الخط المقارب; "step grid" → شبكة قيم الخطوة; "unit-test gradient checking" → فحص التدرجات في اختبارات الوحدات; "exact derivative" → المشتق المضبوط.
- "computation DAG" → مخطط DAG حسابي (مخططات DAG الحسابية), from the glossary's مخطط حسابي; "node" → عقدة (عقد); "edge" (of a graph) → ضلع (أضلاع); "directed path" → مسار موجه; "sensitivity" → حساسية; "adjoint" → قرين; "upstream" → الوارد من جهة المخرج; "step function" → الدالة الدرجية.
- Archimedes stays in Latin (style §7: a person outside glossary terms); Cauchy–Schwarz → متباينة Cauchy–Schwarz.
- Two adjacent math or code spans separated in English only by a bracket or colon get a short Arabic word between them (أي، حيث، مع), per style §3.2.

## chapter_preliminaries/ndarray.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "order" (of a tensor: its number of axes) → مرتبة (موتر من المرتبة $k^\textrm{th}$); "rank" where the English writes it as a synonym → الرتبة (the glossary's matrix rank). "axis" → محور (محاور); "size" (element count) → حجم; "element", "entry" → عنصر.
- Array "index" → فهرس (فهارس), "indexing" → الفهرسة, verb يفهرس; the notation page's دليل stays the mathematical subscript. "slicing" → اقتطاع الشرائح (glossary slice → شريحة); "half-open" → نصف مفتوحة.
- "tuple" → مجموعة مرتبة (tuple); "1-tuple" → مجموعة مرتبة ذات عنصر واحد.
- "reshape" → إعادة التشكيل; "singleton axis" → محور مفرد; "row-major order" → ترتيب الصفوف أولًا; "transpose" as an operation on axes → المنقول (glossary); "exchange two axes" → يبادل بين محورين.
- "gather" → الالتقاط (not التجميع, the glossary's pooling); "integer index tensors" → موترات فهارس صحيحة; "Boolean mask" → قناع منطقي; "eager mode" → نمط التنفيذ الفوري.
- "mutable / immutable" → قابل للتغيير / غير قابل للتغيير; "in place" → في موضعه (in-place assignment → الإسناد في الموضع); "overwrite" → يكتب فوق; "assign" → يسند.
- "view" (zero-copy) → عرض (عروض), "zero-copy" → بلا نسخ; "buffer" → مخزن مؤقت (مخازن مؤقتة), "storage buffer" → مخزن البيانات المؤقت; "allocate" → يحجز; "contiguous" → متجاور; "metadata header" → ترويسة بيانات وصفية; memory "stride" → خطوة (خطوات), not the convolution term خطوة الانزلاق.
- "data type" → نوع البيانات; "cast" → يحول النوع; "32-bit" → من 32 بتًا; "byte" → بايت; "unsigned integer" → عدد صحيح بلا إشارة; "defaults to" → يعتمد … افتراضيًا.
- "pseudorandom" → شبه عشوائي; "seed" → بذرة (بذور); "global generator" → المولد العام; "global state" → حالة عامة; "subkey" → مفتاح فرعي; "sequence" of random numbers → متتالية (style §10.1, mathematical sense); "draw" → سحبة.
- "purely functional", "functional update", "pure function" → وظيفي بحت، تحديث وظيفي، دالة نقية (التحليل الدالي stays functional analysis).
- "method" (of a class) → دالة; "expression" → تعبير; "statement" → عبارة; "grayscale" → بتدرج الرمادي; "our running minibatch" → دفعتنا المصغرة، مثالنا المتكرر; "rescale" → يعيد التحجيم.
- Slides: "0D/1D/2D/4D" → صفري / أحادي / ثنائي / رباعي الأبعاد (style §6); "Concept Check" → اختبار الفهم, kicker "Check Your Understanding" → التحقق من الفهم; "cache miss" → إخفاق في الذاكرة المخبئية; "heap" → الكومة; "pointer" → مؤشر; "unboxed" → غير معلَّب; "lookup table" → جدول بحث; "accumulator" → مركم; "literal" → قيمة حرفية; "state machine" → آلة حالة; "debugging" → التنقيح.

## chapter_preliminaries/pandas.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "standardization" → التوحيد المعياري (glossed once); "z-score normalization" → التطبيع بدرجة z; "to standardize" → يوحد معياريًا; "standardized" → موحد معياريًا; "z-scores" → درجات $z$.
- "fit" (preprocessing statistics on a split) → يقدّر (التقدير على مجموعة التدريب); يلائم stays for fitting a model. "held-out" → محجوز; "test score" → درجة الاختبار; "category vocabulary" → مفردات الفئات.
- "Bessel's $n - 1$ denominator" → المقام $n - 1$ المنسوب إلى Bessel (Bessel stays Latin: no glossary term carries the name).
- "downcast" → تحويل نزولي; "tensor constructor" → مُنشئ الموترات; "32-bit / 64-bit floating point" → الفاصلة العائمة ذات 32 / 64 بتًا; "gigabyte" → غيغابايت.
- "relational tables" → جداول علائقية; "join" (tables) → يربط; "corrupted records" → سجلات تالفة; "sentinel codes" → القيم الحارسة; "zero-variance feature" → ميزة ذات تباين صفري; "weight penalties" → عقوبات الأوزان; "numerator / denominator" → بسط / مقام.
- Exercise credit line: *مقتبس بتصرف من دورة [Course Title](url) في Kaggle Learn، التمرين N.* (course title stays English; an Arabic word separates the two Latin names).
- Slides: "Concept Check" → اختبار الفهم; "Check Your Understanding" (kicker) → التحقق من الفهم on every page (a noun phrase, style §8); "Key Takeaways" → أهم الخلاصات; "Snippet A" → المقطع A; quiz prompts and answer fixes are verbal nouns (المطلوب تحديد …، الصحيح حساب …، الحل محاذاة …).

## chapter_preliminaries/probability.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "Estimation Error" → خطأ التقدير; "standard error of a frequency" → الخطأ المعياري للتكرار النسبي; "$k$ standard errors from" → على بُعد $k$ خطأ معياري من.
- "p-value" → القيمة الاحتمالية (glossary), with (p-value) once on the page; "tail probabilities" → احتمالات الذيول; "Gaussian bell curve" → منحنى الجرس الغاوسي; "confidence interval (CI)" → فترة ثقة (no Latin acronym).
- "Chebyshev's inequality" → متباينة Chebyshev (the name is in no glossary term, so Latin); "Bernoulli indicator" → مؤشر برنولي.
- "prior" / "posterior" as terms of Bayes' theorem → التوزيع القبلي / التوزيع البعدي (glossary); the posterior of one event ("posterior probability of infection", "a 13.1% posterior") → الاحتمال البعدي; "evidence" $P(D)$ → الدليل; "prior beliefs" → المعتقدات القبلية.
- "dependence" (statistical) → التبعية; "dependent" → غير مستقلين; "negatively dependent" → تبعية سالبة; "marginal independence" → الاستقلال الهامشي.
- "common cause" → سبب مشترك; "common effect" → نتيجة مشتركة; "explaining away" → الاستبعاد بالتفسير, with (explaining away) once on the page; the verb "explains away" → تكفي لتفسير.
- Diagnostic testing: "sensitivity" → الحساسية; "specificity" → النوعية; "false-positive rate" → معدل الموجبات الكاذبة, on slides معدل FPR; "false alarm" → إنذار كاذب; "prevalence" → معدل الانتشار; "test positive" → يأتي الاختبار موجبًا; "follow-up test" → اختبار متابعة; "HIV" → فيروس HIV; "healthy" → سليم (الأصحاء); "base-rate neglect" → إهمال المعدل الأساسي; "anomaly screening" → فحص الشذوذ.
- "sum rule" → قاعدة الجمع; "product rule" → قاعدة الجداء; "nuisance variables" → المتغيرات الدخيلة; "factorize" (a model, a joint) → يحلل إلى عوامل; "exact factorization" → تحليل مضبوط إلى عوامل.
- "center of mass" → مركز الكتلة; "second central moment" → العزم المركزي الثاني; "linearity of expectation" → خطية التوقع; "shift invariance" (of variance) → الثبات تحت الإزاحة; "quadratic scaling" → التحجيم التربيعي.
- "log–log plot" → رسم لوغاريتمي المحورين; "slope" → ميل; "from first principles" → من المبادئ الأولى; "paired evaluation / comparison" → التقييم المقترن / المقارنات المقترنة; "ensembles" → مجموعات النماذج (التجميع is pooling); "LLMs" → النماذج اللغوية الكبيرة; "MAP estimation" → تقدير MAP; "active data collection" → الجمع النشط للبيانات.
- Slide labels: "Worked Example" → مثال محلول; "Key Takeaways" → أهم الخلاصات; "Check Your Understanding" → التحقق من الفهم; quoted words inside a slide title attribute take ‘…’.
- English glosses added on this page in this unit (do not repeat): central limit theorem, epistemic uncertainty, aleatoric uncertainty, p-value, conditional probability, Bayes' theorem, prior, likelihood, posterior, conditionally independent, Markov chain, explaining away.

## chapter_preliminaries/autograd.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "piecewise linear" → خطية قطعيًا; "doubling" → مضاعفة (مضاعفات); "loop count / loop iterations" → عدد دورات الحلقة; "threshold point" → نقطة عتبة; "discontinuous" → غير متصلة; "conditionals" → الشروط.
- "Jacobian–vector product" → جداء المصفوفة اليعقوبية بالمتجه (JVP), parallel to جداء المتجه بالمصفوفة اليعقوبية (VJP); "Hessian–vector product" → جداء المصفوفة الهيسية بمتجه (glossary); "memory overhead" → عبء ذاكرة إضافي; "counting argument" → حجة العد; "function evaluations" (cost unit) → عدد مرات حساب الدالة.
- "reverse-mode / forward-mode sweep" → اجتياز بالنمط العكسي / بالنمط الأمامي; "AD mode" → نمط الاشتقاق التلقائي.
- To "predict" as a reader (exercises, concept checks) → توقع (توقعاتك), as on the ndarray page; تنبؤ stays the model's prediction. Concept-check lead-in "Predict …" → المطلوب: توقع …; its scenarios are written as verbal nouns (حساب، تنفيذ), style §8.
- "rebinding" (a Python name) → إعادة توجيه الاسم (ربط is concatenation); "enclosing scope" → النطاق المحيط; "scope" (`autograd.record()`) → نطاق; "persistent" (tape) → دائم; "reset" (a gradient buffer) → يصفّر، التصفير.
- "define-by-run" → التعريف بالتنفيذ; "execution path" → مسار تنفيذ; "caveat" (slide label) → تنبيه; "Key Takeaways" (kicker) → أهم الخلاصات; "Summary: …" (slide title) → الملخص: …; table header "API" → واجهة API; "None:" in a table cell → لا شيء:.
- "upstream vector" → متجه وارد من جهة المخرج (as on the calculus page); "backward edge" → الضلع الخلفي; "zero-copy" → بلا نسخ; "frozen" (factor) → مجمد; "physics simulator" → برنامج محاكاة فيزيائية; "mesh" → الشبكة; "production inference" → الاستدلال في بيئة الإنتاج; "RL" in a table cell → التعلم المعزز.
- English glosses on this page: reverse mode, forward mode and Jacobian–vector product were not glossed in u02 (they occur only in bold lead-ins, a heading and slides).

## chapter_preliminaries/calculus.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "forward difference" → الفرق الأمامي (الفروق الأمامية); "central difference" → الفرق المركزي (الفروق المركزية); "one-sided" → أحادي الجانب; "log--log axes" → محوران لوغاريتميان; "truncation branch" (of an error curve) → فرع القطع.
- "composition" (of functions) → الدوال المركبة; "local derivative" → مشتق محلي; "contribution" (of a path) → إسهام; "mechanical" (backward pass) → ميكانيكي (not آلي, which reads as "automatic").
- "power rule" → قاعدة القوة; "exponential rule" → قاعدة الدالة الأسية; "base / exponent" → الأساس / الأس; "naive answer" → إجابة ساذجة; "closed form" → صيغة مغلقة; "origin" → نقطة الأصل.
- "isotropic quadratic bowl" → الوعاء التربيعي المتناحي; "strictly decrease" → ينقص نقصانًا تامًا; "strictly less / greater than" → الأصغر / الأكبر تمامًا من; "absolute error" → الخطأ المطلق; "relative error" → الخطأ النسبي.
- "gold standard" → المرجع الذهبي (not المعيار, which is the norm); "buggy" (formula, implementation) → معيب; "custom-kernel gradients" → تدرجات النوى المخصصة; "non-symmetric" → غير متناظرة.
- Slides: "mantissa bits" → بتات الجزء الكسري; "V-curve" → المنحنى على شكل V; "reverse-mode AD" → الاشتقاق التلقائي بالنمط العكسي; "blowup" (of cost) → تضخم; "Concept Check" → اختبار الفهم; "Check Your Understanding" → التحقق من الفهم; "Key Takeaways" → أهم الخلاصات; "calculus objects" → كائنات التفاضل; "evals" → حسابًا (حسابات).
- A slide answer that begins with a math span in English gets a leading Arabic word (نجد، التدرج), style §3.2.
- English glosses given in u02 of this page: Jacobian matrix, backpropagation, automatic differentiation.

## chapter_preliminaries/linear-algebra.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "penalty" (regularization, weight) → عقوبة (عقوبات تنظيم، عقوبات $\ell_1$ على الأوزان), as in the glossary's عقوبة التدرج; "robust" → أكثر متانة (glossary متانة).
- "$\ell_2$ norm" → المعيار $\ell_2$ (class noun before the symbol; "$\ell_1$ loss" → خسارة $\ell_1$); "Manhattan distance" → مسافة مانهاتن; "absolute homogeneity" → التجانس المطلق; "positive definiteness" (norm axiom) → التعريف الموجب; "magnitudes" of eigenvalues → القيم المطلقة.
- "stretch" (a matrix stretches a vector) → يمدد، التمديد; "stretch factor" → معامل التمديد; "scale" (an eigenvector) stays يحجّم; "rotate" → يدير; "dominant eigenvector" → المتجه الذاتي المهيمن; "growth rate" → معدل النمو.
- "unit eigenvectors" → المتجهات الذاتية ذات الطول الواحد; "eigenbasis" → أساس ذاتي (أساس ذاتي متعامد معياري); "complete" basis → تام; "spectral theorem" → المبرهنة الطيفية; "expanding a vector in a basis" → كتابة المتجه على الصورة … / في الأساس (فك stays for expanding an algebraic expression).
- "law of cosines" → قانون جيب التمام; "bilinear expansion" → فك الجداء ثنائي الخطية; "planar vectors" → متجهان في المستوي; "signed length" → الطول ذو الإشارة; "aligned / opposed" (vectors) → اتجاه واحد / اتجاهان متعاكسان; "ellipse" → قطع ناقص; "unit circle" → دائرة الوحدة; "$\ell_1$ diamond" → معيَّن $\ell_1$; "$\ell_2$ ball" → كرة $\ell_2$.
- "explode / vanish" (activations, gradients, signals) → تنفجر / تتلاشى (انفجار الإشارات وتلاشيها); "thought experiment" → تجربة ذهنية; "linearly separable" → قابلة للفصل خطيًا; "scale invariance" → الثبات أمام التحجيم; "HBM bandwidth" → عرض نطاق ذاكرة HBM; "clipping" (gradients) → قص التدرج (glossary).
- Slide kickers: "Check Your Understanding" → التحقق من الفهم; "Key Takeaways" → أهم الخلاصات; slide title "Concept Check: …" → اختبار الفهم: ….
- English glosses given in u02 of this page (do not repeat in later units): loss functions, regularization penalties, Euclidean norm, cosine similarity, Manhattan distance, Frobenius norm, eigenvector, spectral theorem, power iteration, spectral norm, singular values, rank. Cauchy–Schwarz stays Latin and takes no gloss.

## chapter_preliminaries/ndarray.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- Section headings: "Operations" → العمليات; "Elementwise Operations" → العمليات عنصرًا بعنصر; "Broadcasting" → البث; "Concatenation" → الربط; "Memory and Conversion" → الذاكرة والتحويل; "Allocation and In-Place Updates" → الحجز والتحديث في الموضع.
- "virtual / virtually" (broadcast copies, virtual stride) → ظاهري / ظاهريًا (نسخ ظاهرية، خطوة ظاهرية); not افتراضي, which is "by default" and "virtual environment", and not وهمي (dummy variables). "physical copies" → نسخ فعلية.
- "stretch" (an axis, in broadcasting) → يمدد، تمديد; "align shapes" → محاذاة الأشكال, "aligned dimensions" → بُعدان متحاذيان; "lower-order tensor" → الموتر الأدنى مرتبة; "compatible" → متوافق.
- "rebind" (a Python name to a new buffer) → يعيد توجيه الاسم إلى …, "rebinding" → إعادة توجيه الاسم; ربط is kept for concatenation (glossary). `concatenate` "joins" → تربط; `stack` "joins along a new axis" → تضم، التكديس; slide "Joining Tensors" → ضم الموترات.
- "binary / unary" (operators, functions) → ثنائي / أحادي; "relational operators" → عوامل المقارنة; "type promotion", "promotes to" → ترقية النوع، يرقّي إلى; "augmented assignment" → الإسناد المركب; "slice assignment" → الإسناد إلى شريحة; "preallocated" → محجوز مسبقًا; "stale buffer" → مخزن مؤقت قديم; "reference" (to an object) → مرجع (مراجع).
- "column vector / row vector" → متجه عمودي / متجه صفي; "pairwise sums / differences" → المجاميع الزوجية / الفروق الزوجية; "mean image" → صورة متوسط; "exception" → استثناء; "raises an error" → يثير خطأ; "bug" → خلل ("silent bug" → خلل صامت), "loud error" → خطأ صريح; "pitfalls" → مزالق.
- "read-only" → للقراءة فقط; "pending / queued GPU kernels" → نوى GPU معلقة / مصطفة; "sync barrier" → حاجز مزامنة; "host and device" → المضيف والجهاز; "gigabytes" → غيغابايتات; "trainable" → قابل للتدريب; "temporary" (tensor) → موتر مؤقت; "interoperability" → التشغيل البيني; "ecosystem" → المنظومة.
- "attributes" (of a Python object) → السمات; "signature" → توقيع; "docstring" → سلسلة التوثيق; "Tab completion" → الإكمال بمفتاح Tab; "source" → المصدر.
- Slides: "aggregate" (reductions) → تُجمل (not تجمع: التجميع is pooling); "cropping" → الاقتصاص (القص is shear); "filtering" → التصفية (ترشيح is the glossary's alternative for convolution); "feature fusion" → دمج الميزات; "minibatch assembly" → تكوين الدفعات المصغرة; "target log-probs" → لوغاريتمات احتمالات الأهداف; "Key Takeaways" → أهم الخلاصات; "donates buffers" → يهب المخازن المؤقتة.
- English glosses given in u02 of this page (do not repeat in later units): broadcasting.

## chapter_preliminaries/probability.md#u03 (claude fable-5.2 (medium), 2026-11-27)

- "portfolio" → محفظة; "stock" → سهم (أسهم); financial "return" → عائد (عوائد), "expected return" → العائد المتوقع; "risk" (of a portfolio) → المخاطرة; "equally weighted" → متساوية الأوزان; "capital" → رأس المال.
- "to bound" (a probability) → يحصل على حد أعلى لـ; "the bound" → الحد; "account for the gap" → يفسر الفجوة.
- "labels N images correctly" → يصيب في وسم N صورة; a classifier "is correct" → يصيب; "correctness" → إصابة المصنف.
- "false negative rate" → معدل السوالب الكاذبة (parallel to معدل الموجبات الكاذبة); "baseline probability" → الاحتمال الأساسي; "coupled" (outcomes) → ترتبط النتيجتان.
- A sequence of random variables $X_1, \ldots, X_T$ → متتالية (mathematical sense, style §10.1); "arbitrary" → اعتباطي.
- "posterior probability of guilt" → الاحتمال البعدي للذنب; "jury" → هيئة المحلفين; "acquit" → يبرئ; "history of abuse" → سجل الإساءة.
- Exercise credit line for a university course: *مقتبس بتصرف من مقرر MIT 18.05، [مجموعة المسائل 7](url)، المسألة 4.* ("problem set" → مجموعة المسائل; "Problem N" → المسألة N).
- "(Optional.)" in exercises → (اختياري.)

## chapter_preliminaries/linear-algebra.md#u03 (claude fable-5.2 (medium), 2026-11-27)

- "associativity" / "associative" (matrix multiplication) → الخاصية التجميعية / تجميعي (the school-mathematics term; not related to التجميع "pooling"); "parenthesize" → وضع الأقواس; "intermediate product" → الجداء الوسيط; "arithmetic work" → العمل الحسابي; "by orders of magnitude" → بمراتب مقدار عدة.
- "concatenate" (tensors into one vector) → يربط (glossary ربط); "flatten" → يسطح; "global $\ell_2$ norm" → المعيار $\ell_2$ الكلي; "learnable parameters" → المعلَمات القابلة للتعلم; "vector-norm routines" → دوال معيار المتجهات.
- "mutually obtuse" vectors → متجهات الزوايا بينها منفرجة كلها; "pairwise" → زوجي (الجداءات النقطية الزوجية); "bound" → حد; "unit-length" → طوله واحد.
- "all-pairs" → بين كل الأزواج; "quadratic expansion" → الفك التربيعي; "clamp" → يحصر; "at scale" → عند الأحجام الكبيرة; "$k$-nearest-neighbor" → أقرب $k$ جارًا (glossary أقرب جار).
- "iterate" (noun, the vector of one iteration) → ناتج التكرار (ناتجا تكرارين متعاقبين); "shifted matrix" → المصفوفة المزاحة; "per-step shrinkage ratio" → نسبة التقليص في الخطوة الواحدة; "asymptotic ratio" → النسبة التقاربية; "the true eigenvector" → المتجه الذاتي الفعلي (حقيقي stays "real-valued"); "overall sign" → إشارة كلية.
- "shape error" → خطأ في الأشكال; "silent broadcasting bug" → خلل البث الصامت ("bug" → خلل, "error" → خطأ); "raise an error" → يثير خطأ; "cast to `float32`" → تحويل النوع إلى `float32`.
- Credit lines for university courses: *مقتبس بتصرف من مقرر MIT 18.06، مجموعة المسائل [Problem Set 1](url)، القسم N، المسألة N.*; "Assignment 1" → الواجب [Assignment 1](url); a class noun (مجموعة المسائل، الواجب) separates the Latin course name from the Latin link text; "Problems 29–30" → المسألتان 29 و 30.

## chapter_preliminaries/ndarray.md#u03 (claude fable-5.2 (medium), 2026-11-27)

- Section headings: "Summary" → الملخص; "Exercises" → تمارين (style §7).
- "wrap-around" (integer arithmetic) → دوران القيمة (not الالتفاف, the convolution term); "clamped" → بحصر القيم; "integer overflow" → طفح الأعداد الصحيحة; "intensity levels" → مستويات شدة; "brighten" → يزيد سطوع.
- "significand (mantissa)" → الجزء المعنوي (الجزء الكسري); "represent exactly" → يمثل تمثيلًا مضبوطًا; "counter" → عداد, "freezes" → يتجمد; "training loop" → حلقة تدريب.
- "intermediate tensor / array" → موتر وسيط / مصفوفة وسيطة; "out-of-place expression" → التعبير خارج الموضع; "keyword arguments" → الوسائط المسماة; "container" (`tf.Variable`) → حاوية; "zero-temporary-allocation requirement" → شرط عدم حجز أي موتر مؤقت.
- "compiler fusion" → دمج المصرِّف, "fused kernels" → نوى مدمجة; "lowered (HLO) program" → برنامج HLO الناتج عن التخفيض; "output buffer" → مخزن مؤقت للمخرجات.
- "ground-truth targets" → الأهداف من الحقيقة المرجعية; "true label" → الوسم الحقيقي; "residual" → الباقي; "silently" → بصمت; "batch dimension" → بُعد الدفعة; "target shape" (of a reshape) → الشكل المستهدف.
- "pairwise absolute-difference matrix" → مصفوفة الفروق المطلقة الزوجية; "wall-clock execution time" → زمن التنفيذ الفعلي; "speedup factor" → معامل التسريع; "megabytes" → ميغابايت; "peak memory footprint" → ذروة بصمة الذاكرة; "outer operations" → العمليات الخارجية.
- Exercise credit line for a named author's collection: *مقتبس بتصرف من مجموعة [Title](url) التي أعدها Author، المسألة N.* (an Arabic word between the two Latin names).
