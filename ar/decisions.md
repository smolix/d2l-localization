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

## chapter_linear-regression/index.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "Linear Regression" (chapter and section title) → الانحدار الخطي; "regression" → الانحدار; "workhorse of" → عماد; "proving ground" → ميدان اختبار.
- "minibatch stochastic gradient descent" → النزول التدرجي العشوائي بالدفعات المصغرة; "minibatch SGD" on slides → خوارزمية SGD بالدفعات المصغرة.
- "training pipeline" → خط التدريب (from خط معالجة); "training harness" → أدوات التدريب; "parameterized forward pass" → التمرير الأمامي ذو المعلَمات; "weight-norm regularization" → التنظيم بمعيار الأوزان.
- "affine model" → النموذج التآلفي; "squared error loss" → خسارة الخطأ التربيعي; "in closed form" → بصيغة مغلقة; "iteratively" → تكراريًا; "noise floor" → أرضية الضوضاء.
- "synthetic benchmark" → اختبار معياري اصطناعي; "synthetic testbed" → منصة اختبار اصطناعية; "oracle" (known ground truth to verify against) → مرجع.
- "polynomial" → كثير حدود (كثيرات الحدود); "polynomial curve fitting" → ملاءمة المنحنيات بكثيرات الحدود; "degree" → درجة; "cubic" → دالة تكعيبية; "interpolate" → يستوفي (glossary استيفاء).
- "validation split" → قسم التحقق; "$K$-fold cross-validation" → التحقق المتقاطع بـ $K$ طية ("fold" → طية); "squared bias" → مربع الانحياز; "peek at test data" → اختلاس النظر إلى بيانات الاختبار.
- "squared $\ell_2$ penalty" → عقوبة $\ell_2$ تربيعية; "penalized loss" → الخسارة ذات العقوبة; "ridge" → ريدج (glossary انحدار ريدج; تقليص ريدج); "collinear" → متسامت; "effective parameters" → المعلَمات الفعالة; "held-out risk" → الخطر على البيانات المحجوزة; "scalable" → قابل للتوسع; "misfit" → سوء الملاءمة.
- English glosses given on this page: regression, linear regression, maximum likelihood, minibatch stochastic gradient descent, cross-validation, overfitting, weight decay.

## chapter_linear-regression/weight-decay.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "weight decay" → اضمحلال الأوزان (glossary); "decay" (verb) → يضمحل; "multiplicative decay / shrinkage factor" → الاضمحلال الضربي / عامل التقليص الضربي; "shrink" → يقلص (glossary تقليص).
- "penalized" (loss, objective, optimization, gradient step) → معاقَب (with the fatha), "to penalize" → يعاقب, "unpenalized" → غير معاقَب / لا يعاقَب; "regularization constant" → ثابت التنظيم; "unregularized" → غير منظم; "tax" (metaphor) → ضريبة.
- "polynomial" → كثير الحدود (متعدد الحدود stays "multinomial"); "polynomial features" → ميزات كثير الحدود; "feature selection" → انتقاء الميزات; "collinear" → متسامت (ميزات متسامتة); "high-dimensional" → عالي الأبعاد.
- "thermometer" → ميزان حرارة; "sensor" → مستشعر; "sensor drift" → انجراف المستشعر; "reading" → قراءة; Kelvin → الكلفن, Celsius → الدرجات المئوية.
- "ridge regression" → انحدار ريدج (glossary), short "ridge" → ريدج; "Tikhonov regularization" → تنظيم Tikhonov (name in Latin, no glossary term); "the Lasso" → Lasso after a class noun (طريقة Lasso، انحدار Lasso).
- "factor of $\frac{1}{2}$", "by a factor $(1-\eta\lambda)$" → العامل; "amplified by $500$", "cuts by $10^6$" → بمعامل; "effective capacity" → السعة الفعالة; "isotropic" → متناحي; "unit variance" → تباين مقداره واحد.
- "constraint set" → مجموعة القيود; "subject to" → تحت القيد; "disk" → قرص; "sphere / norm ball" → كرة / كرة معيار; "elliptical contours" → خطوط الكفاف الإهليلجية (ellipse itself stays قطع ناقص); "diamond" → معيَّن; "cross-polytope" → متعدد السطوح المتقاطع; "vertices" → رؤوس; "corner" → زاوية; "boundary" → حدود.
- "posterior density" → الكثافة البعدية; "maximum a posteriori" → تقدير الاحتمال البعدي الأقصى (glossary), then تقدير MAP; "Bayesian statistics" → الإحصاء البايزي; "prior belief" → اعتقاد قبلي; "negative logarithm" → سالب اللوغاريتم.
- "minibatch SGD" → خوارزمية SGD بالدفعات المصغرة; "plain SGD" → خوارزمية SGD العادية; "adaptive optimizers" → المحسنات التكيفية; "decoupling" → فصل.
- Slides: "Lagrangian duality" → الثنوية اللاغرانجية; "Lipschitz sensitivity bound" → حد ليبشيتز للحساسية; "intercept" → المقطع; "damp" → يخمد; "poorly determined directions" → الاتجاهات ضعيفة التحديد; "geometric decay" → اضمحلال هندسي; "eigenspectrum" → الطيف الذاتي; "spectral shrinkage / regularization" → التقليص الطيفي / التنظيم الطيفي. Slide titles that are "Why …" clauses are written as questions with ؟; quoted words inside a title attribute take ‘…’.
- English glosses given in u01 of this page (do not repeat in later units): collinear, ridge regression, weight decay, regularization constant, cross-polytope, sparsity, prior distribution, posterior density, maximum a posteriori.

## chapter_linear-regression/linear-regression.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "regression" → الانحدار; "method of least squares" → طريقة المربعات الصغرى; "least-squares fit" → ملاءمة المربعات الصغرى; "regression line" → خط الانحدار; "fit a line to data" → يلائم خطًا مع البيانات.
- "model class" → فئة النماذج (parallel to the usual فئة الفرضيات; صنف stays the Python class, عائلة النماذج stays "model family").
- "affine combination" → تركيب تآلفي; "affine model" → النموذج التآلفي; "constant translation" → إزاحة ثابتة; "offset" (of the linear model) → الإزاحة; "intercept" → المقطع (glossary).
- "marginal influence / marginal slopes" → التأثير الحدي / الميول الحدية (هامشي stays the probability sense); "weighted sum" → مجموع موزون.
- "baseline" (model) → نموذج مرجعي; "interpretable" → قابل للتفسير; "candidate" (parameters, slopes) → مرشح.
- "corrupted label / point" → وسم تالف / نقطة تالفة, "to corrupt" → يفسد; "clean points" → النقاط النظيفة; "pull" (of a data point on the fit) → شد.
- "overestimate / underestimate" → التقدير الزائد / التقدير الناقص; "factor of $\frac{1}{2}$" → المعامل $\frac{1}{2}$; "drops down" (on differentiation) → ينزل; "the minimum" (location) → القيمة الصغرى.
- "fitted plane" → المستوي الذي نلائمه; "hyperplane" → المستوي الفائق; "origin" → نقطة الأصل; "hat" (over a symbol) → القبعة.
- "per-example losses" → خسائر الأمثلة المفردة; "per-example scale" → مقياس المثال الواحد; "optimization problem" → مسألة التحسين.
- "bounding-box heads" → رؤوس صناديق الإحاطة; "object detectors" → كاشفات الكائنات; "clinical risk scoring" → تقدير درجات الخطر السريري; "demand forecasting" → التنبؤ بالطلب; "vectorized array operations" → عمليات المصفوفات المتجهية; "vectorizing" (slide) → الصياغة المتجهية; "workhorse optimizer" → المحسن الذي يقوم عليه ….
- Slides: "augmented notation" → الصيغة الموسعة; "ones column" → عمود من الواحدات; "absorbing the bias" → استيعاب الانحياز; "convex quadratic bowl" → وعاء تربيعي محدب; "cross term" → الحد التقاطعي; "vanishes identically" → يساوي الصفر تطابقيًا; "uniqueness" → الوحدانية; "singular" (matrix) → منفردة; "collinearity" → التعدد الخطي; "minimum-norm solution" → الحل ذو المعيار الأصغر; "strictly positive definite" → موجبة التعريف تمامًا; "misfit" → سوء الملاءمة; "failure mode" → نمط إخفاق; "pitfall" → مزلق; "minibatch SGD" → خوارزمية SGD بالدفعات المصغرة; "Gaussian MLE" → تقدير MLE الغاوسي.
- People: Legendre stays Latin (Adrien-Marie Legendre); Gauss in Arabic letters with his given names (كارل فريدريش غاوس); "Pythagorean decomposition" → التفكيك وفق مبرهنة Pythagoras (the name is in no glossary term, so Latin). The dwarf planet Ceres → سيريس.
- Synonym lists in a definition ("the *label* (or *target*)") → *الوسم* (label)، ويسمى أيضًا *الهدف* (target), so that the English gloss and the synonym do not form two adjacent parentheses.
- English glosses given in u01 of this page (do not repeat in later units): regression, method of least squares, linear regression, fully connected layer, minibatch stochastic gradient descent, training dataset, example, label, target, features, covariates, model class, weights, bias, offset, intercept, affine, design matrix, loss function, squared error, cost, objective, mean absolute error. Not glossed yet, because the opening only names them and later sections define them: normal equations, maximum-likelihood estimation, computational graph, residual, outlier, orthogonal projection.

## chapter_linear-regression/linear-regression-concise.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "concise implementation" → التنفيذ الموجز; "from scratch" / "scratch" (loss, run, rate, implementation) → من الصفر (خسارتنا المكتوبة من الصفر، تنفيذنا من الصفر، جولة التدريب من الصفر).
- "built-in" (layer, loss, optimizer of a framework) → مدمج في إطار العمل (المحسن المدمج); "out of the box" → جاهز للاستخدام; "hardware-tuned" → مضبوط للعتاد; "loss primitive" → دالة خسارة أولية.
- "subclass" → صنف فرعي; "base class" → صنف أساسي; "inherit" → يرث; "delegate to" → يحيل إلى; "wrap" → يغلف; "contract" (of `d2l.Module`) → عقد; a `d2l.Module` as a common noun → Module in Latin (as on the oo-design page), "data module" → وحدة البيانات, "module tree" → شجرة الوحدات; "registered parameters" → المعلَمات المسجلة.
- "lazy" (shape inference, layer, initializer) → مؤجل (استنتاجًا مؤجلًا، الطبقة المؤجلة، المهيئ المؤجل); "eager" (allocation, layer) → فوري; "dummy call" → استدعاء وهمي; "materialize" (shapes, weights of a lazy layer) → تحديد الأشكال، تنشأ الأوزان.
- "noise floor" → أرضية الضوضاء (الأرضية); "levels off / plateaus at" → يستقر عند; "benchmark results" → النتائج المرجعية; "halve" → ينصّف، تنصيف; "double" → يتضاعف.
- "ground-truth parameters" → معلَمات الحقيقة المرجعية; "generating parameters / weight" → معلَمات التوليد / وزن التوليد; "recovery" (of parameters) → استعادة المعلَمات; "training run" → جولة تدريب; "diverge" (training) → يتباعد; "single-output" → وحيد المخرج; "output width" → عرض المخرجات.
- MSE: متوسط الخطأ التربيعي (mean squared error, MSE) at the first mention, then MSE where the English writes MSE (خسارة MSE، تدرج MSE); "standard mean squared error" → متوسط الخطأ التربيعي في صيغته المعيارية.
- "minibatch SGD" → SGD بالدفعات المصغرة, with a class noun (محسن، خوارزمية); "adaptive optimizer" → محسن تكيفي; "data pipeline" → خط معالجة البيانات; "object-oriented design" → التصميم كائني التوجه.
- Slides: "Replace A with B" → الاستغناء عن A بـ B (keeps the English token order); "Bare Functions" → دوال بسيطة; "OO Scratch" → كائني التوجه من الصفر; "pedagogical progression" → المسار التعليمي (not التدرج, the gradient).
- English glosses given in u01 of this page: fully connected layer, mean squared error (MSE), dense layer.

## chapter_linear-regression/oo-design.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "object-oriented" → كائني التوجه (التصميم كائني التوجه، تجريدات كائنية التوجه), glossed once; "abstraction" → تجريد (تجريدات); "contract" (of a class) → عقد; "separation of concerns" → فصل الاهتمامات; "coupling" → اقتران.
- The classes `Module`, `DataModule`, `Trainer` take the class noun الصنف where a block would start with them; "the trainer" as a common noun → المدرِّب (kasra+shadda against مدرَّب "trained"); "module" as a common noun for a `Module` → Module in Latin (وحدة stays the Python module, e.g. الوحدة `inspect`).
- "training harness" → أدوات التدريب (as on the chapter index); "one reusable harness" → مجموعة أدوات واحدة; "Utilities" (section) → الأدوات المساعدة, "utility", "helper" → أداة مساعدة / دالة مساعدة; "plumbing" → شيفرة توصيل.
- "metric" → مقياس (مقاييس; قاموس مقاييس); "metric averager" → حاسبة متوسطات المقاييس; "log" (metrics) → يسجل; "live plots / curves" → رسوم حية / منحنيات حية.
- "decorator" → مزخرف; "method" → دالة (من دوال الصنف); "constructor" → مُنشئ (مُنشئات); "attribute" → سمة; "argument" → وسيط; "base class" → الصنف الأساسي; "subclass" → صنف فرعي; "inherit" → يرث; "override" → يعيد تعريف; "instance" → مثيل, "instantiate" → ينشئ مثيلًا; "skeleton" (of a class) → هيكل الصنف; "bind an argument as an attribute on `self`" → يحوّل الوسيط إلى سمة في `self` (ربط stays concatenation); "caller's frame" → إطار المستدعي; "stack frame" → إطار المكدس; "hook" → خطاف (خطافات); "delegate to" → يحيل إلى; "wrap" → يغلف; "unpack" → يفك; "dictionary" → قاموس; "updater" → المحدِّث.
- "pass" over a data loader → مرور (مرورات; اختبار المرورين); "iterable" → كائن قابل للتكرار; "iterate over" → المرور على; "iterator" → مكرِّر (with the mark); "shuffle / reshuffle" → يخلط / يعيد الخلط; "index permutation" → تبديل الفهارس, "random permutation" → تبديل عشوائي; "deterministic order" → ترتيب حتمي; "split" (training / validation) → قسم; "held-out validation loss" → خسارة التحقق على البيانات المحجوزة; "remainder" (the partial last minibatch) → الباقي; "exhausts" (a generator) → ينضب; "synthetic regression data" → بيانات الانحدار الاصطناعية; "literate code" → شيفرة بأسلوب البرمجة الأدبية; "device transfer" → النقل بين الأجهزة.
- Bold list lead-ins and kickers that are a bare class name in English get the class noun الصنف / المزخرف in front (style §3.2); a comma between two code spans inside a lead-in becomes و.
- English glosses given in u01 of this page (do not repeat in later units): object-oriented, method, decorator, constructors, hyperparameters, base class, tuple.

## chapter_linear-regression/linear-regression-scratch.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "training pipeline" → خط التدريب; "minibatch iterator" → مكرِّر الدفعات المصغرة; Python "generator" → مولد (المولد `minibatches`), "yields" → يعطي, "exhausts" → ينضب (نضوب), "single-use" → أحادي الاستعمال / يُستعمل مرة واحدة.
- "noise floor" → أرضية الضوضاء ("irreducible noise floor" → أرضية الضوضاء غير القابلة للاختزال; "the floor" → الأرضية; "statistical estimation floor" → أرضية التقدير الإحصائي); "stability ceiling" → سقف الاستقرار; "stability boundary" → حد الاستقرار.
- "excess loss" → الخسارة الزائدة; "parameter error" → خطأ المعلَمات; "expected loss" → الخسارة المتوقعة; "initial loss" → الخسارة الابتدائية; "decay rate" → معدل التناقص, "exponential decay" → تناقص أسي; "contraction" (of the error per step) → الانكماش, "contracts" → ينكمش; "plateau", "flatten out" → يستوي، الاستواء.
- "random permutation" → تبديل عشوائي; "reshuffling" → إعادة الخلط; "remainder batch" → دفعة البقية; "pass" over a generator → مرور.
- "training run" → جولة تدريب (as in the introduction); "closed-form prediction" → تنبؤ الصيغة المغلقة; "measured" → المقيسة; "cross term" → حد تقاطعي; "sample covariance matrix" → مصفوفة التغاير العينية.
- "true parameters" → المعلَمات الحقيقية; "known ground-truth parameters" → نعرف معلَمات حقيقتها المرجعية; "recovered parameters" → المعلَمات المستعادة, "parameter recovery" → استعادة المعلَمات.
- "guard" (slides: implementation guard, broadcasting guard) → احتياط (احتياطات); "defensive reshape" → إعادة التشكيل الدفاعية; "updater" → دالة التحديث; "object-oriented" → كائني التوجه (التصميم كائني التوجه، أصناف كائنية التوجه); "least-squares solver" → خوارزمية حل المربعات الصغرى.
- "Exercise :exref:`…`" → التمرين :exref:`…` (the word is written, as in the English); "crawls" (too small a learning rate) → يزحف, "divergence" (of training) → التباعد; "quadratic bowl" → وعاء تربيعي; "undiscounted momentum" → زخم بلا خصم; "minimum" (of a function) → القيمة الصغرى; "invariant" (kicker) → لامتغير.
- "by a factor of $c$" → بالمعامل $c$ / بمعامل يقارب …; "$7\times$ per epoch" stays as the math span followed by في الحقبة / لكل حقبة.
- English glosses given in u01 of this page (do not repeat in later units): forward pass, hyperparameters, broadcasting, mean squared error, epoch, irreducible noise floor, parameter error, excess loss.

## chapter_linear-regression/synthetic-regression-data.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "noise floor" → أرضية الضوضاء; "loss floor" → أرضية الخسارة ("the floor" → الأرضية); "irreducible" → غير قابل للاختزال; a loss that "plateaus / levels off" → تستقر; "target plateau" → المستوى المستهدف الذي تستقر عنده.
- "scatter plot" → مخطط تبعثر (مخططات التبعثر); "scatter" → التبعثر, "spread" → التشتت; "band" → شريط; "thick clouds" → سحب سميكة; not مخطط انتشار (الانتشار is diffusion).
- "ground-truth model / parameters" → نموذج يمثل الحقيقة المرجعية / معلَمات الحقيقة المرجعية; "true parameters" → المعلَمات الحقيقية; "learned parameters" → المعلَمات المتعلَّمة; "data-generating" → المولد للبيانات، التي ولّدت البيانات; "data-generating process" → عملية توليد البيانات; "data generator" → مولد البيانات.
- "design matrix" → مصفوفة التصميم; "augmented design matrix" → مصفوفة التصميم الموسعة; "standard normal" → يتبع التوزيع الطبيعي المعياري; "additive Gaussian noise" → ضوضاء غاوسية جمعية; "signal" → الإشارة; "realized noise" → الضوضاء المتحققة.
- "least-squares fit" → ملاءمة المربعات الصغرى; "least-squares solution" → حل المربعات الصغرى; "solver" (least squares) → خوارزمية الحل (the glossary's حالّ is the ODE solver); "closed-form normal equations" → المعادلات الناظمية ذات الصيغة المغلقة; "gold-standard fit" → الملاءمة التي تمثل المرجع الذهبي.
- "reference benchmark" → مرجع معياري للمقارنة; "benchmark quantity" → كمية مرجعية; "our synthetic benchmark" → اختبارنا المعياري الاصطناعي; "testbed" → منصة اختبار; "pass/fail" → النجاح أو الإخفاق; "tolerance" → هامش السماح; "oracle" (pass/fail) → مرجع.
- "minibatch SGD" → خوارزمية SGD بالدفعات المصغرة; "iterative optimization / optimizer" → التحسين التكراري / محسن تكراري; "converged" → متقارب; "training run" → جولة تدريب; "a single run" (of an estimator) → تشغيل واحد.
- "statistical estimation error" → خطأ تقدير إحصائي; "numerical error" → خطأ عددي; "machine precision" → دقة الآلة; "Cramér–Rao bound" → حد Cramér–Rao (Latin name, no glossary term); "predictor" → متنبئ; "a limit of the data" → قيد تفرضه البيانات.
- "misspecified" (model) → خاطئ التحديد (تحديد النموذج خاطئ); "a model that does not fit" → نموذج لا يلائم البيانات; "unmodeled term" → الحد غير المنمذج; "corrupted labels" → وسوم تالفة.
- "correlated features" → ميزات مترابطة; "collinearity" → التعدد الخطي (بين الميزات); "isotropic features" → الميزات المتناحية; "decoupling" → فك الاقتران; "concentrate around" → تتركز حول; "cross-products" → الجداءات المتقاطعة; "out-of-sample" → خارج العينة; "SNR" stays Latin (نسبة SNR).
- The theory's "prediction" (a formula's value, $\sigma/\sqrt{n}$) → التنبؤ; what the reader is asked to predict → توقع، التوقع (as on earlier pages).
- "part (a)" of an exercise → الجزء (a); slide enumerators (a)/(b) → في الحالة (a) / (b); slide series "Recovery I/II/III" → الاستعادة I / II / III.
- English glosses given on this page: noise floor, synthetic data, scatter plot, statistical estimation error.

## chapter_linear-regression/generalization.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "fit" (a model, a curve; noun and verb) → يلائم، الملاءمة (ملاءمة المنحنيات بكثيرات الحدود; "the fitted model" → النموذج الملائم; "refit" → إعادة الملاءمة), from the glossary's فرط الملاءمة; "well matched / balanced fit" → ملاءمة مناسبة / متوازنة.
- "polynomial" → كثير الحدود (كثيرات الحدود), "degree-$k$ polynomial" → كثير حدود من الدرجة $k$; "degree" → درجة; "cubic" → كثير حدود تكعيبي / المنحنى التكعيبي; "monomial" → أحادي الحد (أحاديات الحد); "polynomial features" → ميزات كثير الحدود; "parabola" → قطع مكافئ.
- "$K$-fold cross-validation" → التحقق المتقاطع بـ $K$ طية (invariant in case); "fold" → طية (طيات); "held-out fold" → الطية المحجوزة; "leave-one-out" → ترك مثال واحد (التحقق المتقاطع بترك مثال واحد).
- "true risk" → الخطر الحقيقي; "predictor" → متنبئ; "model class" → فئة النماذج (as on the linear-regression page; عائلة stays "family"); "parametric models" → النماذج المعلَمية.
- "noise floor" → أرضية الضوضاء; "irreducible" → غير قابل للاختزال; "interpolate" → يستوفي، الاستيفاء; "monotonic(ally)" → رتيب (تناقصًا رتيبًا); "U-shaped curve" → منحنى على شكل U; "sweet spot" → النقطة المثلى; "trade-off" → مقايضة; "failure mode" → نمط إخفاق.
- "the contrast" (between results) → الفارق, never التباين, which is variance on this page.
- "MSE" where the English writes the acronym → خطأ MSE (خطأ MSE على الاختبار); the full term متوسط الخطأ التربيعي where the English spells it out.
- "Vandermonde" stays Latin (مصفوفة تصميم Vandermonde); "nearly singular" → شبه منفردة; "nearly collinear" → شبه متسامتتين; "split" (train/validation/test) → قسم; "three-way split" → التقسيم الثلاثي; "sample complexity" → تعقيد العينة; "pessimistic / optimistic bias" → انحياز متشائم / متفائل; "learner" → خوارزمية تعلم.
- "(Exercise :exref:`…`)" → (التمرين :exref:`…`).
- English glosses given in u01 of this page (do not repeat in later units): training set, generalization, underfitting, overfitting, bias-variance decomposition, validation sets, K-fold cross-validation, independent and identically distributed (i.i.d.), training error, empirical risk, generalization error, true risk, population risk, empirical risk minimization, test set, generalization gap, polynomial features, mean squared error.

## chapter_linear-regression/linear-regression-scratch.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "primitives" (of a training loop) → عناصر أولية; "overshoot" (the quadratic bowl) → يتخطى; "silent shape mismatch" → عدم تطابق صامت في الأشكال; "bug-free" → خلو الشيفرة من الأخطاء.
- "full-batch gradient descent" → النزول التدرجي بالدفعة الكاملة; "grid" (of learning rates to try) → شبكة قيم; "component of the parameter error along $\tilde{\mathbf{x}}$" → مركبة خطأ المعلَمات على امتداد ….
- "robust losses" → الخسائر المتينة (glossary متانة); "Huber loss" → خسارة هوبر; "corruption" (of a label) → الإفساد, "corrupted data" → البيانات التالفة; "generating bias" → انحياز التوليد; "penalty curves" → منحنيات العقوبة; "against the iteration" → مقابل رقم التكرار.
- "Replace A with B" in exercises → ضع مكان A B (keeps the English token order and avoids the reversed استبدل بـ); "second derivatives" → المشتقات الثانية; "first-order reverse-mode differentiation" → الاشتقاق بالنمط العكسي من المرتبة الأولى; "outer `tf.GradientTape`" → شريط خارجي من الصنف `tf.GradientTape`.
- Exercise title "Why reshuffle." → لماذا نعيد الخلط. (full stop inside the bold, as the English; no ؟).

## chapter_linear-regression/generalization.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "leave-one-out" as a noun for the estimate → ترك مثال واحد (تقدير ترك مثال واحد، بواقي ترك مثال واحد); "repeated cross-validation" → التحقق المتقاطع المكرر; "5-fold cross-validation" → التحقق المتقاطع بـ 5 طيات ("both 5-fold estimates" → التقديران بـ 5 طيات); "CV" on slides → التحقق المتقاطع (no Latin acronym).
- "disjoint" (splits, subsets) → منفصلة; "round" (of cross-validation) → جولة; "coarse split" → تقسيم أخشن; "unstable learner" → خوارزمية تعلم غير مستقرة; "decision tree" → شجرة القرار; "folklore" → الاعتقاد الشائع; "held in a vault" → تُحفظ في خزنة; "hierarchy" (of features) → تراتب.
- "the $K$ held-out losses", "all $n$ residuals" → the noun, then وعددها $K$ / $n$ (no الـ before a symbol).
- "genuine gap" → فجوة فعلية (حقيقي stays "true/real"); slide "Model A / Model B" → النموذج A / النموذج B; table "train / val errors" → خطأ التدريب / خطأ التحقق; "Summary & Key Takeaways" → الملخص وأهم الخلاصات.
- English glosses given in u02 of this page (do not repeat in later units): hyperparameters, model selection, folds, leave-one-out cross-validation.

## chapter_linear-regression/weight-decay.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "benchmarks" for a loss (reference values to compare against) → قيمتان مرجعيتان (slide: القيم المرجعية); "oracle parameters" → المعلَمات المرجعية (oracle → مرجع, as on the chapter page); "trivial constant model / predictor" → نموذج ثابت بديهي / المتنبئ الثابت; "baseline" → خط الأساس.
- "subclass" (verb) → يشتق صنفًا فرعيًا من; "object-oriented" → كائني التوجه (as on the oo-design references); "fitted model" → النموذج الملائَم; "report" (a loss) → الإبلاغ عن.
- "training run", "the unpenalized run" → جولة التدريب (غير المعاقَبة); "sweet spot" (stopping) → نقطة مثلى; "bottom out" → تبلغ قاعها; "textbook display of" → مثال نموذجي على.
- "singular" (matrix) → منفردة (شاذة stays "anomalous"); "invertible" → قابلة للعكس; "strictly positive definite" → موجبة التعريف تمامًا; "positive semidefinite" (feminine) → موجبة شبه معرفة; "orthonormal" → متعامد معياري; "eigendirection" → اتجاه ذاتي; "eigenbasis" → الأساس الذاتي; "matrix form" → الصيغة المصفوفية; "centered" → ممركزة.
- "sample second-moment (covariance) matrix" → مصفوفة العزم الثاني (التغاير) العيّنية; "empirical variance (spread)" → التباين التجريبي (التشتت).
- "shrinkage factor" → عامل التقليص; "well-determined / poorly determined directions" → الاتجاهات جيدة التحديد / ضعيفة التحديد; "noisy" (direction, component) → مشوب بالضوضاء; "damp", "damping" → يخمد، الإخماد; "effective number of parameters" → العدد الفعال للمعلَمات.
- "filtering", "filter" in the spectral sense (directional filtering, spectral filter) → ترشيح، مرشح (glossary مرشح); التصفية stays for filtering data with a mask. "blind scaling" → تحجيم أعمى.
- "orthogonal invariance" → الثبات تحت التحويلات المتعامدة; "project onto" → إسقاط على; "breakdown" (of the error) → تفصيل; "spectral lens / picture" → العدسة الطيفية / الصورة الطيفية; "gap" (to the least-squares solution) → فجوة; "half-power threshold" → عتبة نصف القدرة; "null-space annihilation" → إفناء الفضاء الصفري.
- "exercise :exref:`…`" in running text → التمرين :exref:`…`.
- English glosses given in u02 of this page (do not repeat in later units): early stopping, normal equations, eigendecomposition, shrinkage factor, effective number of parameters.

## chapter_linear-regression/oo-design.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- `self.history` / "the History" as a common noun → السجل (حلقة التدريب والسجل); "progress board" → لوحة التقدم (اللوحة `ProgressBoard`); "setup hooks" → خطافات الإعداد; "forward-looking hooks … dormant" → خطافات تستشرف الفصول اللاحقة وتبقى خاملة.
- "learning-rate scheduler" → مجدول معدل التعلم (مجدولات), beside the glossary's جدول معدل التعلم for "schedule"; "gradient transformation" (Optax) → تحويل التدرج; "views of the model" → عرضان للنموذج; "mode" (training / evaluation) → نمط.
- "epoch means" → متوسطات الحقب; "unweighted average" → المتوسط غير الموزون; "size-weighted" → موزون بالحجم; "the same vote" → الصوت نفسه; "running sum" → المجموع التراكمي; "partial (last) minibatch" → دفعة مصغرة (أخيرة) جزئية; "discard" (a partial minibatch) → يهمل، الإهمال; "skew" (means) → يحرف.
- "sweep" (hyperparameter sweep, silent sweeps) → عمليات مسح (مسح المعلَمات الفائقة); "headless mode" → النمط بلا عرض; "fit_epoch sweeps through the loader" → تمر على (not اجتياز, the autograd sweep).
- "asynchronous backend" → الواجهة الخلفية غير المتزامنة; "bounded asynchrony" → لاتزامن محدود; "overhead" → عبء إضافي; "eagerly" → فوريًا; "fused graph / computation" → مخطط مدمج / حساب مدمج; "compiled step" → خطوة مصرَّفة; "to decorate" → يزخرف (زخرفة الدالة بالمزخرف).
- "memorize" (label noise) → يحفظ، الحفظ; "plunge / drop to the noise floor" → تهبط إلى أرضية الضوضاء; "to within about a thousandth" → بفارق لا يتجاوز نحو جزء من ألف; "plateaus" (validation loss) → تستقر.
- Planck stays Latin (قانون Planck، ثابت Planck: the name is in no glossary term); Boltzmann in Arabic letters (ثابت بولتزمان, style §7); "black body" → جسم أسود; "spectral radiance" → الإشعاعية الطيفية; "wavelength" → طول موجي; "of order one" → من مرتبة الواحد.
- "on-the-fly generation" → التوليد عند الطلب; "data-generating process" → عملية توليد البيانات; "stream of minibatches" → سيل من الدفعات المصغرة; "vision / sequence datasets" → مجموعات البيانات البصرية / التسلسلية.
- Slide title "Summary & Key Takeaways" → الملخص وأهم الخلاصات; table cells that are a bare class name in bold (`d2l.Module`) are copied without a class noun.
- English glosses on this page: none added in u02 (gradient clipping, dropout, batch normalization, early stopping and learning-rate schedule are used only in passing, in tab notes, exercises or slides).

## chapter_linear-regression/linear-regression.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- Section headings: "Minimizing the Loss" → تصغير الخسارة; "The Normal Equations" → المعادلات الناظمية; "Least Squares as Projection" → المربعات الصغرى بوصفها إسقاطًا ("X as Y" in headings → X بوصفه Y); "Vectorization for Speed" → الصياغة المتجهية من أجل السرعة.
- "normal" (perpendicular, the sense behind "normal equations") → ناظم (ناظمًا، الناظمية), with (normal) once where the text explains the name; "perpendicular" and "orthogonal" both → متعامد مع.
- "full-batch gradient descent", "full-batch GD" → النزول التدرجي بالدفعة الكاملة (no Latin "GD"); "full-batch gradient" → تدرج الدفعة الكاملة; "minibatch SGD" in prose → خوارزمية SGD بالدفعات المصغرة.
- "statistical redundancy", "dataset redundancy" → الفائض الإحصائي، فائض مجموعة البيانات (not تكرار, which is repetition/iteration); "redundant features" → ميزات مكررة.
- "computational scaling" (how cost grows) → تزايد الكلفة الحسابية; "cubic parameter scaling" → تزايد تكعيبي مع عدد المعلَمات; "learning-rate scaling", "scale the learning rate" → تحجيم معدل التعلم.
- "invertible" → قابلة للعكس; "matrix inversion" → عكس المصفوفات; "completing the square" → إكمال المربع; "linearly independent / dependent" → مستقلة خطيًا / مرتبطة خطيًا; "full column rank" → رتبة أعمدة كاملة; "unique global minimizer" → نقطة التصغير الشاملة الوحيدة.
- "dedicated least-squares solvers" → الدوال المتخصصة في حل مسائل المربعات الصغرى (slides: دوال الحل المتخصصة); "QR or SVD factorizations" → تحليل QR أو SVD; "numerically stable" → مستقر عدديًا; "up to `float32` precision" → في حدود دقة `float32`; lower-case "flops" → عملية FLOP.
- "sweep out a subspace" → يرسم فضاءً جزئيًا; "flat subspace" → فضاء جزئي مستو; "$d$-dimensional subspace" → فضاء جزئي بُعده $d$; "observation space" → فضاء المشاهدات.
- "sampled uniformly" → تُسحب بانتظام (أخذ العينات المنتظم); "signed prediction error" → خطأ التنبؤ ذو الإشارة; "overpredicts" → يتنبأ بقيمة أكبر من اللازم; "update rule" → قاعدة التحديث; "held-out validation data" → بيانات تحقق محجوزة.
- "overshoot" → يتجاوز القيمة الصغرى; "fluctuate perpetually" → يتذبذب بلا انقطاع; "neighborhood" → جوار; "terminal noise ball" → كرة الضوضاء النهائية; "halving / quartering / quadrupling" → تنصيف / تخفيض إلى الربع / مضاعفة أربع مرات; "decaying $\eta \to 0$" → تخفيض $\eta \to 0$ تدريجيًا; "steady-state" → في الحالة المستقرة; "jitter" → اهتزاز; "population parameters" → معلَمات المجتمع الإحصائي.
- "vectorize" → يصوغ صياغة متجهية, "vectorization" → الصياغة المتجهية, "vectorized call" → استدعاء متجهي; "dispatch" (operations to a library or device) → يرسل, "dispatch overhead" → عبء الإرسال; "interpreter overhead" → عبء المفسر; "warm-up call" → استدعاء إحماء; "asynchronously" → بصورة غير متزامنة; "GPU cores" → أنوية GPU; "GPU streams" → مجاري GPU; "timing helper" → دالة التوقيت المساعدة; "routine" → دالة.
- "homoscedastic" → متجانس التباين; "additive noise" → ضوضاء جمعية; "light / heavy tails" → ذيول خفيفة / ثقيلة; "robust" (loss) → متين; "smooth" → أملس; "splice A onto B" → يصل A بـ B; "rate" (Poisson) → المعدل; "NLL" stays Latin after سالب لوغاريتم الإمكان (خسارة NLL).
- Laplace → لابلاس in Arabic letters (ضوضاء لابلاس، توزيع لابلاس), as the name is in the glossary's لابلاسي; Poisson → بواسون; Huber → هوبر (glossary خسارة هوبر); Pythagorean theorem → مبرهنة Pythagoras.
- "synaptic weights" → الأوزان المشبكية; "input node / output node" → عقدة مدخل / عقدة المخرج; "tunable parameters" → معلَمات قابلة للضبط; "layer depth" → عمق الطبقات; "the MLP chapter" → فصل MLP.
- Slide lead-ins "Why never …:", "Why stack layers later:" → سبب …: (noun phrase); slide table header "Role & Practical Takeaway" → الدور والخلاصة العملية; "Summary & Key Takeaways" → الملخص وأهم الخلاصات; figure colour words "(black)", "(color)" → (بالأسود)، (بالألوان).
- English glosses given in u02 of this page (do not repeat in later units): normal equations, column space, span, orthogonal projection, normal, gradient descent, minibatch SGD, minibatch, learning rate, unbiased estimator, model parameters, hyperparameters, noise ball, vectorize.

## chapter_linear-regression/generalization.md#u03 (claude fable-5.2 (medium), 2026-11-27)

- "IID" (as the English writes it in exercises) → i.i.d., as glossed in u01 (افتراض i.i.d.); "dependent observations" → المشاهدات غير المستقلة; "train/validation split" → التقسيم إلى تدريب وتحقق.
- "rigid models" → النماذج الصلبة (as in u01); "overly flexible" → المفرطة المرونة; "memorization" → الحفظ; "repeated queries contaminate the test set" → الاستعلامات المتكررة تلوث مجموعة الاختبار.
- "exact" (fit, arithmetic, regression) → مضبوط (الحساب المضبوط، ملاءمة مضبوطة); "distinct" (inputs) → متمايزة; "basis" → أساس; "nearly parallel" (columns) → تكاد تكون متوازية; "span the same functions" → تولد الدوال نفسها; "full column rank" → رتبة أعمدة كاملة; "invertible" → قابلة للعكس.
- Chebyshev and Sherman--Morrison stay in Latin (كثيرات حدود Chebyshev، أساس Chebyshev، متطابقة Sherman--Morrison); "optimism" (of the training error) → تفاؤل; "Hint:" → تلميح:; "fresh labels" → وسوم جديدة.
- "Consider least squares with …" in an exercise → لنأخذ المربعات الصغرى مع …; "Let $x$ be …" → ليكن / لتكن.

## chapter_linear-regression/weight-decay.md#u03 (claude fable-5.2 (medium), 2026-11-27)

- "parameter groups" (of an optimizer) → مجموعات المعلَمات; "gradient transformation" (Optax) → تحويل التدرج; "predicate" → شرط; "exempt from decay" → يعفي من الاضمحلال; "contraction factor" → عامل الانكماش.
- "reproducing kernel Hilbert spaces (RKHS)" → فضاءات Hilbert ذات النواة المولِّدة (reproducing kernel Hilbert spaces, RKHS); Hilbert stays Latin (the name is in no glossary term). "frameworks" in the sense of theoretical frameworks → أطر (without عمل).
- "regularizer" → منظِّم (المنظِّمات), with the marks, as on the Basics page; "norm-based story" → تفسيرنا القائم على المعيار; "scale poorly to" → لا تتوسع جيدًا إلى.
- "proximal step" → الخطوة القريبة (from the glossary's مؤثر قريب); "soft-thresholding" → العتبة الناعمة (glossary); "coordinate-wise" → إحداثيًا بإحداثي; "augmented least squares" → مربعات صغرى موسعة; "ordinary least squares" → المربعات الصغرى العادية; "to center" (imperative) → اجعل … ممركزة.
- "sweep" (over values of a hyperparameter) → مسح (مسح قيم $\lambda$); "undetermined directions" → الاتجاهات غير المحددة.
- "decoupled weight decay" → اضمحلال الأوزان المنفصل (plural الأوزان, to match the page's اضمحلال الأوزان; the glossary proposes اضمحلال وزن منفصل); "decoupled / coupled update" → التحديث المنفصل / التحديث المقترن; "to precondition" (a gradient) → يطبق اشتراطًا مسبقًا (glossary اشتراط مسبق); "under-decay" → يضعف اضمحلال.
- "Lipschitz slope" → ميل ليبشيتز; "noise-to-signal ratio" → نسبة الضوضاء إلى الإشارة; "oracle floor" → الأرضية المرجعية; "optimizer-level" → على مستوى المحسن.
- Exercise credit line for a book: *مقتبس بتصرف من كتاب [Title](url) لمؤلفه Author، المسألة N.* (لمؤلفيه with several authors joined by و; "Exercise N" → التمرين N).
- English glosses given in u03 of this page: hyperparameter, reproducing kernel Hilbert spaces (RKHS).

## chapter_linear-regression/linear-regression.md#u03 (claude fable-5.2 (medium), 2026-11-27)

- Section headings: "Squared Loss and Gaussian Noise" → الخسارة التربيعية والضوضاء الغاوسية; "The Normal Distribution" → التوزيع الطبيعي; "Maximum Likelihood" → الإمكان الأعظم; "Choosing a Loss from a Noise Model" → اختيار الخسارة من نموذج الضوضاء; "Linear Regression as a Neural Network" → الانحدار الخطي بوصفه شبكة عصبية.
- "normally distributed" → يتبع التوزيع الطبيعي; "bell-shaped curve" → منحنى على شكل جرس; "well-behaved noise" → ضوضاء حسنة السلوك; "natural default model" → النموذج الافتراضي البديهي (not الطبيعي, which is "normal" here); "observed label" → وسم مرصود; "conditional mean / median" → المتوسط الشرطي / الوسيط الشرطي.
- "conditional probability density" → كثافة الاحتمال الشرطي (keeps both glossary terms كثافة الاحتمال and احتمال شرطي); "the likelihood factors into the product" → يتحلل الإمكان إلى الجداء.
- "statistic" (a function of the data: mean, median) → إحصاءة (إحصاءات); "$q$-quantile" → التجزيء $q$ (glossary تجزيء); "pinball loss" → خسارة الكرة والدبابيس (pinball loss), a description rather than a transliteration; "forecasters" → خبراء التنبؤ (المتنبئ stays "predictor"); "tilt" (a loss) → يميل.
- "expected squared / absolute error" → الخطأ التربيعي المتوقع / الخطأ المطلق المتوقع; "absolute error ($\ell_1$) loss" → خسارة الخطأ المطلق ($\ell_1$); "finite variance" → تباين منته.
- Table cells: "real-valued" → قيم حقيقية; "positive" → قيم موجبة; "multiplicative fluctuations" → تقلبات ضربية; "counts" → أعداد, "count data" → بيانات العد; "arrivals" → حالات الوصول; "Statistic predicted" → الإحصاءة المتنبأ بها; noise-model cells غاوسي، لابلاس، بواسون.
- "wild outliers" → قيم شاذة جامحة; "sharp corner" → ركن حاد; "smooth optima" → قيم مثلى ملساء; "inset" (of a figure) → الرسم المدرج; "(right)" / "(left)" in captions → (يمينًا) / (يسارًا).
- "input layer / output layer" → طبقة الإدخال / طبقة الإخراج (style §10.1); "computing unit" → وحدة حسابية; "artificial neuron" → عصبون اصطناعي; "cortex" → القشرة الدماغية; "synapses" → المشابك; "dendrites" → التغصنات; "integrate" (signals) → يدمج (not يجمع: التجميع is pooling); "count layers" → يحصي الطبقات (نعد would read as "prepare"); McCulloch and Pitts stay Latin.
- English glosses given in u03 of this page (do not repeat in later units): maximum likelihood, negative log-likelihood, maximum-likelihood estimator, homoscedastic, quantile, pinball loss, Huber loss, input layer, output layer, dense layer. Normal distribution, median and absolute error take no gloss (school statistics; mean absolute error was glossed in u01).

## chapter_linear-regression/linear-regression.md#u04 (claude fable-5.2 (medium), 2026-11-27)

- "parameterization" (a way of writing a quantity in terms of the parameters) → تمثيل معلَمي (التمثيلان المعلَميان), parallel to النماذج المعلَمية.
- "characterize the set of …" (exercises) → أعط وصفًا لمجموعة …; "minimizer of smallest norm" → نقطة التصغير ذات المعيار الأصغر; "rectangular diagonal" (matrix) → قطرية مستطيلة; "nonzero entries" → العناصر غير الصفرية.
- "Black--Scholes model of option pricing" → نموذج Black--Scholes لتسعير الخيارات (names in Latin; no glossary term); "Moore--Penrose pseudoinverse" → المعكوس الزائف بمعنى Moore--Penrose, with the hyphens as the English page writes them.
- "scale" (of the Laplace distribution) → مقياس; "on the log scale" → على المقياس اللوغاريتمي; "substitute" (for a missing derivative) → بديل; "fix" (for a non-converging iteration) → علاج; "iterates" → نواتج التكرار (as on the linear-algebra page).
- "assigns probability to" → الاحتمال الذي يسنده النموذج إلى …; "grocery store" → متجر بقالة; "observed count" → العدد المرصود.
- English glosses on this page in u04: none (summary uses terms already glossed; exercises take no gloss, so subgradient and pseudoinverse stay unglossed).

## chapter_linear-regression (chapter check, claude fable-5.2 (medium), 2026-11-27)

One rendering per term across the eight pages of the chapter; the unit entries above were corrected to match.

- "polynomial" → كثير الحدود (كثير حدود، كثيرات الحدود), masculine; "$K$-fold cross-validation" → التحقق المتقاطع بـ $K$ طية.
- "model class" → فئة النماذج (parallel to the glossary's فئة الفرضيات); صنف stays the Python class, عائلة the "family".
- "collinear" (features, directions) → متسامت; "collinearity" (the statistical phenomenon) → التعدد الخطي. Not التداخل الخطي.
- "singular" (matrix) → منفردة, "nearly singular" → شبه منفردة: the chapter keeps شاذة for outliers (قيمة شاذة), which share pages with singular matrices. The glossary proposes مصفوفة شاذة.
- "full column rank" → رتبة أعمدة كاملة; "the minimum" (of a function) → القيمة الصغرى (glossary قيمة صغرى محلية / شاملة).
- "random permutation" → تبديل عشوائي; a generator that "exhausts" → ينضب (نضوب); "iterator" → مكرِّر, always with the mark (مكرر reads "repeated").
- "corrupted" (labels, points, measurements, data) → تالف; "oracle" (known ground truth to check against) → مرجع.
- "least-squares solver" → خوارزمية حل المربعات الصغرى in prose, دوال الحل المتخصصة for the library routines (`lstsq`); حالّ stays the ODE solver.
- Kept apart on purpose: "baseline model" → نموذج مرجعي and a "baseline" value or level → خط الأساس; the weight "decay" → اضمحلال and the "decay" of an error or loss → تناقص; the function `sgd` → دالة التحديث and the class `SGD` → المحدِّث; "decoupling" of weight decay → فصل and of features → فك الاقتران.

## chapter_linear-classification/index.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "Linear Classification" (chapter title) → التصنيف الخطي; "category" and "class" → فئة (فئات), "classifier" → مصنف, as on the introduction page.
- "one-hot vector / label / target" → متجه الترميز الأحادي / وسوم الترميز الأحادي / هدف بالترميز الأحادي, from the glossary's ترميز أحادي (one-hot encoding).
- "logit" → اللوجيت (اللوجيتات); "linear score" → درجة خطية; the softmax function → الدالة softmax, "softmax layer" → طبقة softmax; "multi-logit head" → رأس متعدد اللوجيتات.
- "categorical" (label model, likelihood) → فئوي (نموذج فئوي للوسوم، سالب لوغاريتم الإمكان الفئوي).
- "residual gradient" → تدرج … وهو الباقي; slide "residual gradient form" → صيغة الباقي للتدرج.
- "evaluation" → التقييم; "deployment" → النشر; "deployment data" → بيانات النشر; "deployment shift" → الانزياح عند النشر; data "drift away from" → تنجرف بعيدًا عن.
- "raw accuracy" → الدقة الخام; "positive examples" → الأمثلة الموجبة; "cost-sensitive thresholds" → العتبات الحساسة للكلفة; "calibration diagnostics" → تشخيصات المعايرة; "calibrated uncertainty" → عدم يقين معايَر; "Bayes cost thresholds" → عتبات الكلفة بمعنى بايز.
- "Hoeffding bounds" → حدود Hoeffding, "Hoeffding radius" → نصف قطر … بحد Hoeffding; "Vapnik--Chervonenkis (VC) theory" → نظرية Vapnik--Chervonenkis (VC) (names in Latin; no glossary term carries them; glossary بُعد VC).
- "importance-weighted", "importance weighting" → الموزون بالأهمية، الترجيح بالأهمية (glossary وزن الأهمية); "runaway feedback loops" → حلقات التغذية الراجعة الجامحة.
- "templates" (learned weight images) → قوالب (القوالب المتعلَّمة); "fused log-sum-exp loss" → خسارة مدمجة تقوم على لوغاريتم مجموع الأسيات, on slides خسارة LogSumExp المدمجة; "image pipeline" → خط معالجة للصور.
- Garments: shirt → قميص; T-shirt → قميص قصير الكمين; coat → معطف; sandal → صندل; pullover → كنزة; trousers → بنطال; dress → فستان; boot (ankle boot) → حذاء برقبة قصيرة; "garments" → قطع ملابس.
- "percentage points" → نقطة مئوية; "sampling noise" → ضوضاء أخذ العينات; "validation accuracy" → دقة التحقق; "prevalence" → معدل الانتشار; "screening" → فحص مسحي; "auditing" → التدقيق; "toy" (example) → مثال مبسط.
- Slide labels: "Self-Assessment" → التقييم الذاتي; "Chapter Roadmap" → خارطة طريق الفصل; "Chapter Study Guide" → دليل دراسة الفصل; "Empirical Anchors" → مرتكزات تجريبية; "The Modelling Recipe" → وصفة النمذجة; "autodiff" → الاشتقاق التلقائي; "GEMM" stays Latin after عمليات.
- English glosses given on this page: classification, one-hot vectors, logit, cross-entropy loss, confusion matrices, log-sum-exp, precision, recall, adaptive overfitting, covariate shift, label shift.

## chapter_linear-classification/softmax-regression.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "softmax regression" → انحدار softmax (glossary); "the softmax" / "softmax function" → دالة softmax (class noun wherever a block or clause would start with the Latin word); "softmax operator" → مؤثر softmax; "multinomial logistic regression" → الانحدار اللوجستي متعدد الحدود.
- "logit" → لوجيت, dual لوجيتان, plural لوجيتات (glossary لوجيت; sound feminine plural); "unnormalized scores" → درجات غير مطبّعة; "affine score" → درجة تآلفية; "raw logits" → اللوجيتات الخام.
- "class" and "category" → فئة (as in the introduction); "class label" → وسم الفئة; "tag" (multi-label) → وسم; "true class" → الفئة الحقيقية; "candidate class" → فئة مرشحة.
- "one-hot encoding" → الترميز الأحادي (glossary); "one-hot vector" → متجه الترميز الأحادي, on slides المتجه الأحادي; "hard assignment" → إسناد قاطع; "soft assignment" → إسناد لين.
- "valid" (probability distribution, simplex) → سليم (صحيح stays "integer" and "correct"); "degenerate distribution" → التوزيع المنحل; "categorical probabilities" → الاحتمالات الفئوية.
- "shift invariance" (softmax) → الثبات تحت الإزاحة (as on the probability page); "log-odds" → لوغاريتم الأرجحية (glossary); "log-odds ratio" → لوغاريتم نسبة الأرجحية; "odds" → الأرجحية; "$k$ times as probable as" → يساوي $k$ أمثال احتمال ….
- "logistic sigmoid" → دالة سيغمويد اللوجستية; "reduces to" (softmax to logistic regression) → يُختزل إلى، الاختزال إلى; "exponentiate" → يحسب الدالة الأسية لـ, "the exponentials" → القيم الأسية / الدوال الأسية; "strictly increasing" → متزايدة تمامًا; "partition sum" → مجموع التجزئة.
- "decision regions" → مناطق القرار; "halfspace" → نصف فضاء; "convex polyhedron" → متعدد سطوح محدب; "flat-sided" → مسطح الأوجه; "flat" (hyperplane) → مسطح; "hyperplane" → المستوي الفائق (مستويات فائقة); two logits "tie" → يتساوى اللوجيتان; "simplex" → مبسط (glossary مبسط الاحتمالات), "vertices" → رؤوس.
- "ordinal regression" → الانحدار الترتيبي; "threshold models" → نماذج العتبات; "competing softmax" → دالة softmax تنافسية; "spam filter" → مرشح البريد المزعج; "online retailer" → متجر إلكتروني; "downstream decisions" → القرارات اللاحقة; "at inference time" → عند الاستدلال; "numerical safeguard" → احتياط عددي.
- Fashion-MNIST class names: قميص قصير الكمين، بنطال، كنزة، فستان، معطف، صندل، قميص، حذاء رياضي، حقيبة، حذاء برقبة قصيرة; cat / chicken / dog → قطة / دجاجة / كلب.
- Slides: "Motivation & Failure Mode" → الدافع ونمط الإخفاق; "Concrete trace" → تتبع بالأرقام; "Memory convention" → اصطلاح التخزين; "redundant parameterization" → تمثيل معلَمي فائض; "scalar regression" → الانحدار القياسي; a slide lead-in "Consider a … image" → a noun phrase (صورة …).
- English glosses given in u01 of this page (do not repeat in later units): supervised learning, hard assignment, soft assignments, softmax regression, multinomial logistic regression, cross-entropy loss, one-hot encoding, logits, log-odds ratio, shift invariance, logistic sigmoid, logistic regression, log-odds, logit, multi-label classification, ordinal regression.

## chapter_linear-classification/softmax-regression-scratch.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "Softmax Regression Implementation" → تنفيذ انحدار softmax; "the softmax" (function) → دالة softmax (class noun, Latin name); "rowwise softmax" → دالة softmax صفًا بصف.
- "overflow" → الطفح, "underflow" → الطفح السفلي (glossary); verbs يطفح / يطفح سفليًا إلى; "overflows to `inf`" → يطفح إلى `inf`.
- "log-sum-exp" → لوغاريتم مجموع الأسيات (glossary), with (log-sum-exp) once; the name "LogSumExp" on slides and in "LogSumExp trick" stays Latin (خسارة LogSumExp المدمجة); "fused" (loss, softmax + log) → مدمج; "log space" → الفضاء اللوغاريتمي.
- "clamp", "clamping" (of probabilities) → الحصر (حصر الاحتمالات، الحصر عند $10^{-12}$), as on the earlier pages; "the clamp triggers" → يُفعَّل الحصر; "stopgap" → حل مؤقت; "zeroes out the gradient" → يصفّر التدرج.
- "one-hot" (matrix, product) → الترميز الأحادي (glossary): مصفوفة ترميز أحادي، جداء الترميز الأحادي, with (one-hot) once.
- "confusion matrix" → مصفوفة الالتباس (glossary); "confusions" → حالات الالتباس; "off-diagonal" → خارج القطر; "recognition rate" → معدل التعرف; "heatmap" → خريطة حرارية; "tally" → يبني بالعد / يحصي.
- "templates" (weight columns shown as images) → قوالب (قوالب الأوزان), as on the linear-algebra page; "silhouette" → الصورة الظلية; "outline" → محيط.
- "sanity check" → فحص السلامة (الفحص السريع للسلامة; "let us sanity-check" → لنتحقق سريعًا من سلامة …); "dual-metric" → ثنائي المقياس / بمقياسين; "overconfident" → مفرط الثقة (أفرط في الثقة); "discrete accuracy" → الدقة المتقطعة; "top-scoring class" → الفئة ذات الدرجة العليا.
- "data loader" → محمِّل (محمِّل التحقق، محمِّل التدريب), with the mark; "binds `self.val_dataloader` to" → تسند إلى; "score a model" → يقيس أداء النموذج; "gather" → الالتقاط (as on the ndarray page).
- Fashion-MNIST garment names are translated, with the English class name once in body text because code output and figures keep the English labels: T-shirt → القميص قصير الكمين, pullover → الكنزة, coat → المعطف, shirt → القميص, trousers → البناطيل, dress → الفستان, bag → الحقيبة, sandal → الصندل, sneaker → الحذاء الرياضي, ankle boot → الحذاء برقبة قصيرة, footwear → الأحذية; "upper-body tops" → الملابس العلوية; "garment" → قطعة ملابس.
- Kickers that begin with a number in English are reordered to begin with an Arabic word: "[7,850 Learnable Parameters]" → [المعلَمات القابلة للتعلم: 7,850]; "[10 Epochs of Minibatch SGD]" → [خوارزمية SGD بالدفعات المصغرة مدة 10 حقب].
- English glosses given in u01 of this page (do not repeat in later units): cross-entropy loss, log-sum-exp, overflow, underflow, accuracy, one-hot, confusion matrix, and the garment names T-shirt, pullover, coat, shirt.

## chapter_linear-classification/classification.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "Measuring Classifiers" → قياس المصنفات; "raw accuracy" → الدقة الخام; "majority class" → فئة الأغلبية; "catch / detect" (positives) → يكشف; "missing" (a sick patient) → إغفال.
- Confusion-matrix cells in the plural: الموجبات الصحيحة، الموجبات الكاذبة، السوالب الكاذبة، السوالب الصحيحة (glossary موجب صحيح etc. for one example); table labels "predicted sick / healthy" → التنبؤ: مريض / التنبؤ: سليم; the acronyms TP, FP, FN, TN, TPR, FPR, PPV, ECE, ROC, AUC stay Latin.
- "sensitivity" → الحساسية; "true-positive rate" → معدل الموجبات الصحيحة; "positive predictive value" → القيمة التنبؤية الموجبة; "false-alarm rate" → معدل الإنذارات الكاذبة; "prevalence" → معدل الانتشار; "population" (of people screened) → المجتمع; "screening (test)" → فحص مسحي / اختبار مسحي; "confirmatory test" → اختبار تأكيدي; "definitive" → قاطع.
- "to flag" (a person as positive) → ينذر بمرضه، إصدار إنذار; "positive flags" → الإنذارات الموجبة; "flagged people" → من أنذر الاختبار بمرضهم (راية stays the 0/1 flag column).
- "cost" → كلفة, plural كلف; "asymmetric" → غير متماثل; "cost-sensitive" → حساس للكلفة; "expected cost" → الكلفة المتوقعة; "correct calls" → القرارات الصحيحة; "intrinsic" (property of the test) → ملازم.
- "harmonic mean" → المتوسط التوافقي; "precision--recall curve" → منحنى الإحكام والاستدعاء; "ROC curve" → منحنى ROC; "sweep thresholds / prevalence" → يمسح، مسح.
- "confidence" → الثقة; "overconfident / underconfident" → مفرط الثقة / ناقص الثقة (فرط الثقة، نقص الثقة); "correctness indicator" → مؤشر الصواب (not الإصابة, which reads as "infection" beside the medical example); "top-class" (calibration, probability, confidence) → الفئة الأعلى احتمالًا.
- "bin" (confidence bin, histogram bin) → حاوية (حاويات; حاويات ثقة); خانات stays the digits of a number and فئة the class. "well-populated bins" → الحاويات الممتلئة جيدًا; "equal-width" → متساوية العرض.
- "reliability diagram" → مخطط الموثوقية; "reliability curve / statistics" → منحنى الموثوقية / إحصاءات الموثوقية; "miscalibration" → سوء المعايرة; "recalibration" → إعادة المعايرة; "well calibrated" → معايَر جيدًا; "perfectly calibrated" → معايَر تمامًا.
- "temperature scaling" → تحجيم درجة الحرارة; "post-hoc" → لاحق للتدريب / بعد التدريب (not بعدي, the posterior); "fit $T$" → يلائم $T$, "fitted temperature" → درجة الحرارة الملائَمة; "damp" → يخمد; "percentage points" → نقاط مئوية; "overparameterized" → مفرط المعلَمات; "pigeonhole principle" → مبدأ برج الحمام; "abstain" → يمتنع (الامتناع); "reject option" → خيار الرفض; "specialist" → مختص; "to audit" (calibration) → يدقق، تدقيق.
- Plain-text "argmax" stays Latin (التصنيف وفق argmax); "always-healthy" rule → قاعدة “سليم دائمًا”.
- English glosses given in u01 of this page (do not repeat in later units): accuracy, false negative, false positive, calibrated, precision, recall, cost-sensitive decision thresholds, reliability diagrams, expected calibration error (ECE), temperature scaling, confusion matrix, true positives, true negatives, sensitivity, true-positive rate, positive predictive value, confidence. Class imbalance occurs only in a bold lead-in and a heading and has no gloss.

## chapter_linear-classification/environment-and-distribution-shift.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "distribution shift" → انزياح التوزيع; "covariate shift" → انزياح المتغيرات المصاحبة; "label shift" → انزياح الوسوم (glossary); "concept shift" → انزياح المفهوم; "shift regime" → نمط الانزياح (أنماط); "shifted" (density, cloud) → مزاح; a distribution or definition that "drifts" → ينجرف (glossary انجراف).
- "source / target distribution" → توزيع المصدر / توزيع الهدف; "source examples", "target inputs", "target labels" → أمثلة المصدر، مدخلات الهدف، وسوم الهدف; "target domain" → مجال الهدف; "target risk" → خطر الهدف; "target accuracy" → الدقة على الهدف.
- "deployment" → النشر (بيانات النشر، المصنف المنشور); "passive / active shift" → الانزياح السلبي / النشط, with (passive shift) once so that سلبي is not read as "negative".
- "importance weight" → وزن الأهمية (glossary); "importance weighting" → الترجيح بالأهمية; "reweighting" → إعادة الترجيح; "to weight an example by $\beta_i$" → يرجح المثال بالوزن $\beta_i$; "importance-weighted", "weighted", "unweighted" → الموزون بالأهمية، الموزون، غير الموزون; "downweight" → يخفض وزن.
- "support" (of a density) → الحامل (style §10.1); "support overlap" → تداخل الحاملين; "missing support" → غياب الحامل.
- "clipping weights" → قص الأوزان (from the glossary's قص التدرج); "clipped" → المقصوص (الأوزان المقصوصة); "cap" → سقف; "clipping threshold" → عتبة القص.
- "domain classifier" → مصنف المجال; "domain label" → وسم المجال; "pool" (source with target inputs) → يضم … إلى …, "pooled inputs" → مدخلات مضمومة (not التجميع, the glossary's pooling).
- "class priors" → الاحتمالات القبلية للفئات (the glossary's توزيع قبلي is the distribution); "class proportions" → نسب الفئات; "class-prior ratios" → نسب الاحتمالات القبلية للفئات.
- "Bayes boundary" → حد بايز; "Bayes-optimal boundary" → الحد الأمثل بمعنى بايز; "Bayes ceiling" → سقف بايز; "at chance", "by chance" → عند مستوى الصدفة، بالصدفة; "majority class" → فئة الأغلبية.
- "classifier two-sample test" → اختبار العينتين بالمصنف; "log-ratio" → لوغاريتم النسبة; "log density ratio" → لوغاريتم نسبة الكثافة; "standard normal distribution function" → دالة التوزيع الطبيعي المعياري.
- "parabola" → قطع مكافئ ("parabolic boundary" → حد على شكل قطع مكافئ; "right arm" → الذراع اليمنى); "point cloud" → سحابة نقاط; "marker" (of a plot) → علامة.
- "misspecified" → خاطئ التحديد (as in the regression chapter); "capacity-limited" → محدود السعة; "weighting scheme" → مخطط ترجيح; "learning primitive" → عنصر أولي في التعلم.
- "default rate / default risk" (loans) → معدل التعثر في السداد / خطر التعثر; "loan applicants" → طالبو القروض; Oxfords → أحذية Oxford (Latin); "Goodhart's law" → قانون Goodhart (Latin name, no glossary term); "control samples" → عينات ضابطة; "startup" → شركة ناشئة.
- "column-normalized confusion matrix" → مصفوفة الالتباس المطبَّعة حسب الأعمدة; "law of total probability" → قانون الاحتمال الكلي; "invertible" → قابلة للعكس; "black-box shift estimation" → تقدير الانزياح بالصندوق الأسود (BBSE); "positive flags" → الإنذارات الموجبة; "missed cases" → حالات فائتة; "unidentifiability" → عدم قابلية التحديد; "label flip" → قلب الوسوم; "adversarial" → خصومي.
- Slides: "Weighted ERM" → مبدأ ERM الموزون (a class noun so that the block starts in Arabic); "2D … benchmark" → اختبار معياري ثنائي الأبعاد; "Why …" slide titles are questions with ؟, "Why …:" lead-ins become سبب …:; figure colour words → (بالأخضر)، (بالأحمر)، (بالبرتقالي); "(left) / (middle) / (right)" → (يسارًا) / (في الوسط) / (يمينًا).
- Image credit lists in captions ("illustrations: …") → الصور التوضيحية: followed by the Latin credits copied as one run with their semicolons.
- English glosses given in u01 of this page (do not repeat in later units): distribution shift, source distribution, target distribution, passive shift, importance-weighted covariate shift correction, confusion-matrix label shift correction, concept shift, target risk, importance weight, weighted empirical risk minimization, support overlap, effective sample size, domain classifier, classifier two-sample test.

## chapter_linear-classification/generalization-classification.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "population error" → خطأ المجتمع (parallel to خطر المجتمع); "population distribution" → توزيع المجتمع; "population accuracy" → الدقة على المجتمع; "empirical error" → الخطأ التجريبي.
- "memorizer" → نموذج حافظ (النماذج الحافظة), "lookup-table memorizer" → نموذج حافظ يعتمد على جدول بحث; "certify" → يشهد بـ, "certificates" → شهادات (ضمان stays "guarantee").
- "validation split" → قسم التحقق; "evaluation split" → قسم التقييم; "validation images" → صور تحقق; "garments" → قطع الملابس; "fresh" (test set, data) → جديد; "untouched" → لم تُمس; "held-out test set" → مجموعة اختبار محجوزة.
- "finite-sample" → للعينات المحدودة (عدم اليقين الناتج عن العينة المحدودة); a "finite" set or class → منته (فئة فرضيات منتهية); "infinite" → لانهائي.
- Confidence intervals: "exact" → مضبوط; "asymptotic" → تقاربي; "normal interval / radius / approximation" → الفترة الطبيعية / نصف القطر الطبيعي / التقريب الطبيعي; "radius" → نصف القطر (نصف قطر الثقة); "a $95\%$ interval" → فترة بمستوى $95\%$; "plug-in" (interval, radius, variance: the estimate put in place of the true value) → تقديري; "coverage" → التغطية, "undercover" → تقصر تغطيتها; "conservative" → متحفظ; "percentage points" → نقطة مئوية; "worst case" → أسوأ الحالات; "simultaneous" (confidence radius) → المتزامن، في آن واحد.
- Hoeffding, Vapnik, Chervonenkis, Boole, Radon, Sauer, Shelah stay Latin (no glossary term carries the names): متباينة Hoeffding، فترة Hoeffding، مبرهنة Radon، توطئة Sauer، حد الاتحاد المنسوب إلى Boole. The first mention of the VC dimension is *بُعد VC* (Vapnik--Chervonenkis dimension), so that the glossary term and the full name both appear; "VC bound" → حد VC; "VC generalization bound" → حد VC للتعميم; "VC theory" → نظرية VC.
- "tail bound" → حد الذيل (حد الذيل الأسي); "looser" (bound) → أرخى, "loose bounds" → حدود رخوة; "vacuous" (bound) → فارغ; "concentration" → التركز; "distribution-free" → بلا افتراض على التوزيع.
- "guesser" → مخمن (مخمنون), "coin-flipping guessers", "coin flippers" → مخمنون يرمون عملة / رماة العملة; "coin-flip labels" → وسوم ناتجة عن رمي عملة; "cherry-pick" → ينتقي (انتقاء الأوفر حظًا); "majority vote" → تصويت الأغلبية; "query" → استعلام; "test feedback" → التغذية الراجعة من الاختبار; "report the winner" → يعلن الفائز.
- "hyperparameter tuning", "tune hyperparameters" → ضبط المعلَمات الفائقة (as on the regression pages; the glossary's تحسين المعلَمات الفائقة is kept for "hyperparameter optimization"); "architecture exploration" → استكشاف المعماريات; "one-shot final evaluation" → تقييم نهائي يُجرى مرة واحدة; "standard benchmarks" → الاختبارات المعيارية المعتمدة.
- "shatter" → يحطم, "shattering" → التحطيم (the definition and its gloss come in the next unit); "labeling" (an assignment of labels to points) → توسيم (توسيمات ثنائية); "dichotomies" → تقسيمات ثنائية; "halfspace" → نصف فضاء (أنصاف الفضاءات الخطية); "halfplane" → نصف مستوٍ; "convex quadrilateral" → رباعي أضلاع محدب; "convex hull" → الغلاف المحدب; "non-collinear" → غير متسامتة; "lifted" (points, feature map) → مرفوع; "linearly dependent" → مرتبطة خطيًا.
- "overparameterized" → مفرط المعلَمات (الشبكات مفرطة المعلَمات); "deep nets" → الشبكات العميقة; "post-training" in the plain sense "after training" → بعد التدريب (التدريب اللاحق stays the LLM stage); "binomial" → ثنائي الحدين; "rule of three" → قاعدة الثلاثة.
- Slides: "error bars" → أشرطة الخطأ; "Section Summary" → ملخص القسم; "Benchmark (…)" as a lead-in for a worked number → المثال المرجعي; figure words "(solid)", "(dashed)", "dotted line", "blue", "orange" → (الخط المتصل)، (الخط المتقطع)، الخط المنقط، بالأزرق، بالبرتقالي; "(middle)" → (في الوسط); a bold lead-in that starts with a math span gets a leading noun (الحد $\mathrm{VC} \ge 3$).
- English glosses given in u01 of this page (do not repeat in later units): selection bias, adaptive overfitting, Vapnik--Chervonenkis dimension, uniform convergence, empirical error, population error, confidence interval, coverage, union bound. i.i.d. is written out once (مستقلة ومتماثلة التوزيع (i.i.d.)). Not glossed yet: shatter, hypothesis class, VC bound (defined in the next unit).

## chapter_linear-classification/image-classification-dataset.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "benchmark" (a dataset to compare models on) → اختبار معياري, glossed once on this page; "to benchmark the loader" → قياس أداء المحمل، قياس إنتاجية المحمل; "landmark benchmark" → اختبار معياري فارق; "headroom" → هامش.
- "channel-first / channel-last layout" → تخطيط القنوات أولًا / تخطيط القنوات أخيرًا, "layout" (of a tensor) → تخطيط (تخطيطات); `NCHW` / `NHWC` stay in code.
- "resize" (images) → يغير حجم، تغيير الحجم ("spatial resize" → تغيير الحجم المكاني); "rescale", "scale" (pixel values) → يعيد تحجيم، يحجّم; "footprint" of a dataset → حجم, "memory footprint" → بصمة الذاكرة (glossary).
- Fashion-MNIST class names: قميص قصير الكمين (T-shirt)، بنطال (trouser)، كنزة (pullover)، فستان (dress)، معطف (coat)، صندل (sandal)، قميص (shirt)، حذاء رياضي (sneaker)، حقيبة (bag)، حذاء برقبة قصيرة (ankle boot); the English name is given once on the page because code and outputs keep it.
- "silhouette" → صورة ظلية (صور ظلية); "outline" → محيط (خارجي); "boxy" → مربع (الهيئة); "upper-body garments / tops" → ملابس الجزء العلوي من الجسم / الملابس العلوية; "footwear cluster" → عنقود أحذية; "labeled gallery" → معرض صور موسومة.
- "split" → قسم (قسم التدريب، قسم الاختبار), "train/test split" as an act → تقسيم إلى تدريب واختبار; "ships with" → تأتي بـ; "official" → رسمي; "nonstandard" → غير معياري.
- "parse" (files) → يحلل; "gzipped" → المضغوطة بصيغة gzip; "IDX byte files" → ملفات IDX الثنائية; "header" → ترويسة; "N-byte" → ذات N بايتًا.
- "native data loader" → محمل البيانات الأصلي في إطار العمل; "wrap" → يغلف; "attach a method to a class" → يلحق دالة بالصنف; "helper" (class or function) → صنف مساعد / دالة مساعدة; "dtype" → نوع البيانات; "leading batch axis" → محور دفعة في المقدمة; "singleton channel axis" → محور القناة المفرد.
- "per-batch overhead" → العبء الإضافي لكل دفعة; "worker" (data-loading process) → عامل (عمال); "prefetch" → يجلب مسبقًا; "producer--consumer pipeline" → خط معالجة من منتج ومستهلك; "wall-clock time" → الزمن الفعلي; "starve the GPU" → تُحرم وحدة GPU من البيانات; "bottleneck" → عنق الزجاجة (أعناق الزجاجة).
- "visual debugging" → التنقيح البصري; "silent data bugs" → مواطن الخلل الصامتة في البيانات; "syntax error" → خطأ نحوي; "runtime error" → خطأ في أثناء التشغيل; "misaligned labels" → وسوم لا تحاذي صورها; "expressive power" → قدرة تعبيرية; "linear template classifier" → مصنف القوالب الخطي.
- "upsampling" → زيادة أخذ العينات (glossary); "linear map" → تحويل خطي (glossary); "the previous part" (of an exercise) → الجزء السابق.
- Number lists in exercises ("1, 16, 64, 256, and 1024") are joined by a free-standing و between all items (style §3.2).
- English glosses given on this page: benchmark, data loader, channel-first, channel-last, and the ten class names.

## chapter_linear-classification/softmax-regression.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "zero-one error" → خطأ صفر-واحد (خطأ التصنيف صفر-واحد); "surrogate" (convex) → بديل (as in خسارة بديلة); "upper bound" → حد أعلى, "upper-bounds" → يحد … من الأعلى; "smooth" → أملس; "strictly decreasing" → متناقص تمامًا; "step function" → دالة درجية, "flat step" → الدرجة المسطحة.
- "nats" / "bits" → نات / بت (وحدة النات، وحدة البت; invariant after a number: $0.37$ نات); inside math `\text{ nats}` is copied; "surprisal" → المفاجأة المعلوماتية (glossary); "code" (coding theory) → شفرة, "encode outcomes" → يرمّز النواتج.
- Claude Shannon → كلود شانون (شانون is in Arabic letters by the introduction decision); "Gibbs' inequality" → متباينة Gibbs; "Kullback--Leibler divergence" → تباعد Kullback--Leibler (names in Latin).
- "prediction minus observation" → التنبؤ ناقص المشاهدة (glossary مشاهدة); "observed label / class" → الوسم المرصود / الفئة المرصودة (as in u01); "globally convex" → محدبة على نحو شامل; "categorical draw" → سحبة فئوية; "per-example loss" → خسارة المثال الواحد.
- "(Exercise :exref:`…`)" → (التمرين :exref:`…`); table "w.r.t." → بالنسبة إلى; "Loss (NLL)" → الخسارة (NLL).
- "temperature" → درجة الحرارة (glossary درجة حرارة softmax); "sharpen / flatten" (a distribution) → يزيده حدة / يسطحه; "peaked" → حاد القمة; "scale factor" → معامل تحجيم; "without bound" → بلا حد; "overconfident guess" → تخمين مفرط الثقة; "confident mistake" → خطأ واثق.
- Slides: "confident blunders" → الأخطاء الفادحة الواثقة; "saturation" → تشبع; "restoring force" → قوة إرجاع; "Scope assumption" → افتراض النطاق; "mutually exclusive" → متنافية; "Batched Matrix Formulation" → الصياغة المصفوفية بالدفعات; "a single GEMM" → عملية GEMM واحدة; "Boundary behavior" → السلوك عند الحدود; "Optimization Pitfall" → مزلق في التحسين.
- English glosses given in u02 of this page (do not repeat in later units): negative log-likelihood, nats, log-sum-exp, information theory, surprisal, bits, entropy, cross-entropy, accuracy, error rate, margin, linearly separable, temperature, calibration.

## chapter_linear-classification/classification.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "figure of merit" → مقياس للجودة; "class-weighted loss" → الخسارة الموزونة بالفئات; "class prior" → التوزيع القبلي للفئات; "class-conditional densities", "class-conditionals" → الكثافات الشرطية للفئات; "reweight" → يعيد الترجيح.
- "receiver operating characteristic" → خاصية تشغيل المستقبل (glossary), then منحنى ROC; "AUC" → مساحة AUC (قيمتا AUC); "area under the ROC curve" → المساحة تحت منحنى ROC; "trapezoidal area" → المساحة المحسوبة بقاعدة شبه المنحرف; "threshold-free" → لا يعتمد على أي عتبة; "ranking quality" → جودة الترتيب; "scores tie" → تتساوى درجتان, "break ties at random" → يحسم التعادل عشوائيًا.
- "simulate" (imperative) → حاكِ, "simulated scores" → الدرجات المحاكاة; "to threshold a classifier at $\tau$" → نضع عتبته عند $\tau$; "guess" → تخمين; "misdiagnosis" → التشخيص الخاطئ; "deferring to a specialist" → الإحالة إلى مختص; "unscaled model" → النموذج قبل التحجيم.
- "This problem … of problem :exref:`…`" → هذه المسألة … من المسألة :exref:`…` (the English writes "problem", not "exercise").
- Garment pairs in the exercise follow the softmax-regression-scratch page: sneaker → الحذاء الرياضي, sandal → الصندل, shirt → القميص, T-shirt → القميص قصير الكمين.
- English glosses on this page in u02: none (the summary uses terms glossed in u01; exercises take no gloss, so receiver operating characteristic keeps only its acronym (ROC) as the English writes it).

## chapter_linear-classification/softmax-regression-scratch.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "Concise Implementation" (heading) → التنفيذ الموجز; "Defining the Model" → تعريف النموذج; "A Stable Cross-Entropy Loss" → خسارة إنتروبيا متقاطعة مستقرة; kicker "Concise Framework Implementation" → التنفيذ الموجز بإطار العمل.
- "LogSumExp trick" → حيلة LogSumExp; "fused log-sum-exp cross-entropy" → الإنتروبيا المتقاطعة المدمجة القائمة على لوغاريتم مجموع الأسيات; "shifted exponent" → أس مزاح; "never materializing intermediate probabilities" → ألا نكوّن الاحتمالات الوسيطة صراحةً.
- "install (a loss) on `d2l.Classifier.loss`" → يضع في (not يثبّت, which is installing software); "clamped" (loss, `cross_entropy`) → المحصورة; "trapping the loss above a floor" → تعلق الخسارة فوق أرضية (عالق, not محصور, which is the clamp); "positive floor" → أرضية موجبة; "the `1e-12` floor" → أرضية الحصر.
- "twin hazards" → الخطران التوأمان; "confident blunder" → خطأ فادح ارتكبه بثقة; "classic pitfall" → مزلق شائع; "rock-solid numerical stability" → استقرار عددي راسخ; "intact gradients" → تدرجات سليمة; "logit magnitudes" → مقادير اللوجيتات; "extreme logits" → اللوجيتات المتطرفة.
- "double softmax" → دالة softmax المزدوجة (خلل softmax المزدوجة); "smooth maximum" → قيمة عظمى ملساء; "the *softplus* of $x$" → دالة *softplus* للمتغير $x$; "naive expression" → التعبير الساذج; "relative accuracy" → الدقة النسبية; "double-precision reference" → قيمة مرجعية بالدقة المزدوجة.
- "multiply-adds" → عمليات الضرب والجمع; "class means" → متوسطات الفئات, "mean image of each class" → صورة المتوسط لكل فئة; "by eye" → بالنظر; "tops" → الملابس العلوية.
- "number formats" → صيغ الأعداد (الصيغة FP16); "exponent bits / mantissa bits" → بتات الأس / بتات الجزء الكسري; "running sum" → المجموع التراكمي.
- Slides: "Critical invariant" → لامتغير حاسم; "identical mathematical trajectory" → مسار رياضي مطابق; "optimization dynamics" → ديناميكيات التحسين; "End-to-End Verification" → التحقق من طرف إلى طرف; "hidden size" → الحجم المخفي; "slots" (of a vector) → مواضع; "developer" → مطور; "immune to" → منيع على; table header "Component" → المكوّن; "Diagnostics ($83\%$ val)" → التشخيصات (تحقق $83\%$).
- No new English gloss in u02 of this page: fully connected layer, softplus and LogSumExp are used in passing or stay Latin.

## chapter_linear-classification/generalization-classification.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "Statistical Learning Theory" → نظرية التعلم الإحصائي; "a priori" → مسبقًا; "learner" → خوارزمية التعلم (as on the regression page); "hypotheses" → الفرضيات.
- "continuous parameter families" → العائلات ذات المعلَمات المستمرة; two classifiers "are indistinguishable on $\mathcal{S}$" → لا يمكن التمييز بينهما على $\mathcal{S}$; functions "realize" a labeling → تحقق; a class "induces" labelings → تحدث (إحداث).
- "shatter" → يحطم, glossed (shatter) at its definition; the VC dimension at its definition in this unit → *بُعد VC* (Vapnik--Chervonenkis), since u01 already gave the full English gloss.
- "Sauer's lemma" → توطئة Sauer; "grows polynomially" → ينمو نموًا كثير الحدود; "symmetrized union bound" → حد الاتحاد بصيغته المتناظرة; "absolute constant" → ثابت مطلق; "over the training draw $\mathcal{S}$" → على سحبة مجموعة التدريب $\mathcal{S}$.
- "conceptual puzzle" → اللغز المفاهيمي; "adversarial distributions" → التوزيعات الخصومية; "natural images" → الصور الطبيعية; "one-vs-rest" → “واحد مقابل البقية”; "train--test gap" → الفجوة بين التدريب والاختبار; "capacity" → السعة; "sharp" (measurement) → دقيق.
- Exercises: "a paper reports" → تذكر ورقة بحثية; "meaningful" (difference) → له دلالة; "submission" (of a model to a test-set holder) → عملية تقديم (يقدم نموذجًا); "recover" (labels) → يستعيد; "leaking a test set" → تسريب مجموعة اختبار; "lower-bound" (verb) → يوجد حدًا أدنى لـ; "hold out" → يحجز; "a 1 appended" → ملحقًا به 1; "indices" → الفهارس; "degree at most five" → من الدرجة الخامسة على الأكثر; "monomials" → أحاديات الحد.
- English glosses given in u02 of this page: hypothesis class, shatter, VC bound.

## chapter_linear-classification/softmax-regression.md#u03 (claude fable-5.2 (medium), 2026-11-27)

- "Summary" → الملخص; "Exercises" → تمارين; "Discussions" → النقاشات (style §7).
- "characterize all … coefficients" → أعط وصفًا لكل معاملات …; "reproduce" (a model, probabilities) → يعيد إنتاج; "redundant logit" → لوجيت فائض; "fitted logistic model" → نموذج لوجستي ملائَم.
- "Bradley--Terry model" → نموذج Bradley--Terry (names in Latin); "default option" → خيار افتراضي; "choice probabilities" → احتمالات الاختيار; "observed choices" → الاختيارات المرصودة.
- "smooth minimum" → صيغة ملساء للقيمة الصغرى; "arbitrarily small" → صغيرة بقدر ما نشاء; "the gap to $\max$" → الفجوة إلى; "equality on the right" → المساواة في الطرف الأيمن; "the largest value" → القيمة العظمى.
- "without ties" (classification) → دون تساوٍ; "scaled parameters" → المعلَمات المحجَّمة; "unbounded norm" → معيار غير محدود; a "sequence of parameters" → متتالية (mathematical sense, style §10.1); "has no minimizer" → ليس لها نقطة تصغير; "finite minimizer" → نقطة تصغير منتهية.
- "Gumbel-max trick" → حيلة Gumbel-max; "Gumbel distribution" → توزيع Gumbel; "probit model" → نموذج probit (Latin, like softmax); "CDF" → دالة التوزيع التراكمي (glossary; no Latin acronym); "perturb a logit with noise" → إضافة ضوضاء إلى اللوجيت اضطرابًا له, "perturbed score" → الدرجة المضطربة; "condition on" → اشترط على; "substitution" → التعويض; "up to Monte Carlo error" → في حدود خطأ مونت كارلو; "label frequencies" → التكرارات النسبية للوسوم; "i.i.d. draws" in an exercise → سحبات مستقلة ومتماثلة التوزيع.
- Slides: "template nudging" → زحزحة القوالب; "rival templates" → القوالب المنافسة; "logit Hessian" → المصفوفة الهيسية بالنسبة إلى اللوجيتات; "KL divergence", "KL" → تباعد KL; "proper scoring rule" → قاعدة تسجيل سليمة; "mistake bound" → حد الأخطاء; "runaway weights" → أوزان جامحة; "honest calibration" → المعايرة الصادقة; "constant forecast" → تنبؤ ثابت; "surrogate error bound" → حد الخطأ بالخسارة البديلة; "order-preserving" → يحافظ على الترتيب; "shift-invariant" → ثابت تحت الإزاحة; "Summary: Key Takeaways for X" → الملخص: أهم خلاصات X; "Section Summary" → ملخص القسم.
- A parenthesis that follows a math span and holds only another math span gets a leading أي (style §3.2): (أي $1\text{ bit}$).
- English glosses on this page in u03: none (summary uses terms already glossed; exercises and slides take no gloss).

## chapter_linear-classification/environment-and-distribution-shift.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "log-prior shift" (the term added to the logits) → إزاحة لوغاريتم الاحتمال القبلي (an offset, style §10.1: الإزاحة; الانزياح stays distribution shift); "Bayes logit adjustment" → تعديل اللوجيتات بقاعدة بايز; "confusion-matrix inversion" → عكس مصفوفة الالتباس; "unscramble" (the confusion) → يفك الالتباس.
- "target class proportions" → نسب الفئات في الهدف; "target priors" → الاحتمالات القبلية للهدف; "source posterior" $p_S(y \mid \mathbf{x})$ → التوزيع البعدي في المصدر; "predicted class frequencies" → التكرارات النسبية للفئات المتنبأ بها.
- "target pool" (the index set `pool`) → مخزون للهدف; "skewed target set" → مجموعة هدف مختلة النسب; "balanced" (split, images) → متوازن.
- The Fashion-MNIST class named in quotes ("shirt") stays Latin after الفئة, as the code output prints it; the garments in running text are translated (قمصان، الملابس العلوية).
- "performative prediction" → التنبؤ الأدائي; "performative" → أدائي; "feedback loop" → حلقة تغذية راجعة, "runaway" → جامحة, "decision-driven" → الناشئة عن القرارات; "self-confirming" → تؤكد نفسها.
- "predictive policing" → الشرطة التنبؤية; "patrols" → دوريات; "incidents" → حوادث; "arrest records" → سجلات الاعتقال; "minor offenses" → مخالفات بسيطة; "highway toll camera" → كاميرا تحصيل الرسوم على طريق سريع; "psychiatry manual" → دليل للطب النفسي.
- "randomized exploration" → استكشاف عشوائي; "bandits" on a slide → خوارزميات bandit (Latin after a class noun; no glossary entry); "data censored by the model's decisions" → بيانات حجبتها قرارات النموذج; "holdout data" → بيانات محجوزة (glossary).
- "a trickle of freshly labeled examples" → قدر قليل متدفق من الأمثلة الموسومة حديثًا; "outdated" (labels) → متقادمة; "fresh target labels" → وسوم جديدة من الهدف.
- "socio-technical systems" → الأنظمة الاجتماعية التقنية; "causal domain knowledge" → معرفة سببية بالمجال; "mixture" → خليط; "binary source domain" → مجال مصدر ثنائي.
- Two terms glossed inside an existing English parenthesis: (*التنبؤ الأدائي*، performative prediction، *وحلقات التغذية الراجعة*، feedback loops), after the style guide's (*الحشو*، padding); a gloss followed by a reference shares one parenthesis: (reinforcement learning، :numref:`…`).
- English glosses given in u02 of this page: performative prediction, feedback loops, reinforcement learning.

## chapter_linear-classification (chapter check, claude fable-5.2 (medium), 2026-11-27)

One rendering per term across the seven pages of the chapter; the unit entries above were corrected to match.

- Fashion-MNIST class names, as on the dataset page (:numref:`sec_fashion_mnist`), on every page: قميص قصير الكمين (T-shirt)، بنطال (trouser; plural بناطيل)، كنزة (pullover)، فستان (dress)، معطف (coat)، صندل (sandal)، قميص (shirt)، حذاء رياضي (sneaker)، حقيبة (bag)، حذاء برقبة قصيرة (ankle boot; plural أحذية برقبة قصيرة). Not تيشيرت, قميص قطني, سروال, حذاء كاحل, حذاء عال, الحذاء نصف الطويل.
- "importance weighting" → الترجيح بالأهمية; "importance-weighted" → الموزون بالأهمية; "importance weight" → وزن الأهمية (glossary). Not الترجيح بأوزان الأهمية.
- "positive flags" (of a screening test or classifier) → الإنذارات الموجبة, with "false alarms" → الإنذارات الكاذبة; "to flag" → ينذر بـ.
- Kept apart on purpose: "class prior" as the whole distribution → التوزيع القبلي للفئات (glossary توزيع قبلي) and "class priors" as the numbers $p(y)$ → الاحتمالات القبلية للفئات; "shift" of a distribution → انزياح and an additive offset (softmax shift invariance, log-prior shift) → إزاحة.
