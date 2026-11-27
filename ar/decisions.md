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
