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

## chapter_multilayer-perceptrons/index.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "Multilayer Perceptrons" (chapter title) → البيرسبترون متعدد الطبقات (glossary, definite singular); gloss (multilayer perceptron, MLP) once, then "an MLP", "MLPs" → شبكة MLP / شبكات MLP (class noun, no Latin plural); "single-layer perceptron" → البيرسبترون وحيد الطبقة.
- "exclusive-OR (XOR)" → دالة “أو” الحصرية (XOR) at first mention, then XOR; "bit" → بت (بتان).
- "linear barrier" → الحاجز الخطي; "feature interactions" → تفاعلات الميزات; "fold (input space)" → يطوي; "linear regions" → مناطق خطية; "pieces" → قطع; "linear / algebraic collapse" → الانهيار الخطي / الجبري; "expressive power" → قدرة تعبيرية.
- "256-unit ReLU MLP" → شبكة MLP من 256 وحدة بدالة ReLU; sigmoid in a list of activation names → سيغمويد (glossary دالة سيغمويد).
- "Xavier / He initialization" → تهيئة Xavier / تهيئة He (Latin names after the class noun, repeated: تهيئة *Xavier* وتهيئة *He*); "variance-preserving initialization" → تهيئة حافظة للتباين.
- "vanishing and exploding gradients" → تلاشي التدرجات وانفجارها; "hidden-unit symmetry" → تناظر الوحدات المخفية; "slope matrices" → مصفوفات الميول.
- "random-label memorization" → حفظ الوسوم العشوائية; "heavily overparameterized" → مفرطة المعلَمات إلى حد بعيد; "multiplicative noise" → الضوضاء الضربية; "interpolation" (fitting the training data exactly) → الاستيفاء; "minimum-norm implicit bias" → الانحياز الضمني نحو المعيار الأصغر.
- "contrast detectors / contrast patterns" (image contrast) → كاشفات تضاد / أنماط تضاد; "whole-garment templates" → قوالب قطع الملابس الكاملة; "blurred" → ضبابي; "localized" → موضعي.
- "activation-memory and FLOP budgets" → ميزانيتا التدريب من ذاكرة التفعيلات ومن عمليات FLOP; "GPU RAM" → ذاكرة RAM لوحدة GPU; "GPU epoch time" → زمن الحقبة على وحدة GPU; "floats" → أعداد بالفاصلة العائمة; "the full toolkit" → العدة كلها.
- Slides: "The Central Problem" → المشكلة المركزية; "What Depth Introduces" → ما يجلبه العمق; "Running Benchmark" → الاختبار المعياري المستمر; "Running Example" → المثال المستمر; "Softmax baseline" → النموذج المرجعي softmax; "Check & Synthesis" → التحقق والتوليف; "Concept Check & Study Guide" → اختبار الفهم ودليل دراسة الفصل; "context-blind" → لا ترى السياق; "leakage-free" → بلا تسرب; "tabular ML" → تعلم الآلة على البيانات الجدولية; "batch scaling" → التزايد مع حجم الدفعة.
- English glosses given on this page: perceptron, multilayer perceptron (MLP), hidden layer, activation function, backpropagation, double descent, implicit bias, dropout.

## chapter_multilayer-perceptrons/numerical-stability-and-init.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "multiplicative factor", "multiplier", "per-layer factor / multiplier" → العامل الضربي (العامل الضربي للطبقة الواحدة); "by a factor of $c$" stays بمعامل $c$.
- "mean square" (of gradients, activations) → متوسط المربع (متوسط مربع التدرج، متوسط مربع التفعيلات); "second moment" → العزم الثاني (glossary).
- "slope" ($\phi'$) → ميل (ميول); "weight and slope matrices" → مصفوفات الأوزان والميول; "matrix--slope factor / operator" → عامل / مؤثر من مصفوفة وميل; "unit slope" → ميل مقداره واحد.
- "fan-in" / "fan-out" → التفرع الداخل / التفرع الخارج, each with its English once; "the two fan counts" → عددا التفرع.
- "permutation symmetry" → تناظر التبديل; "break symmetry" → كسر التناظر; "symmetry breaking" → كسر التناظر.
- "hop" (one layer crossed by the gradient) → قفزة (قفزات); "unroll" (a recurrence) → يفرد، الفرد; "backward recurrence" → العلاقة التكرارية الخلفية.
- "upstream gradient" → التدرج الوارد من جهة المخرج (as on the calculus page); "downstream gradient" → التدرج الصادر نحو جهة المدخل.
- "pre-activation" → قيمة ما قبل التفعيل (قيم ما قبل التفعيل), as on the linear-algebra page; "zero-mean" → ذو متوسط صفري; "zero-centered" → متمركز حول الصفر; "saturated tails" → الذيول المشبعة.
- "Xavier / Glorot / He / Kaiming initialization" → تهيئة Xavier / Glorot / He / Kaiming (names in Latin; "or X initialization" → وتسمى أيضًا تهيئة X); "Explicit He Normal" → تهيئة He الطبيعية الصريحة; "Glorot uniform", "LeCun normal" → توزيع Glorot المنتظم، توزيع LeCun الطبيعي; "default initializer" → المهيئ الافتراضي.
- "attenuation" → توهين; "signal decay" → تناقص الإشارة; "crush / collapse" (activations) → يسحق / انهيار; "inflate" → يضخم; "frozen" (loss) → مجمدة; "sluggish" (epoch) → متثاقلة; "astronomically large" → بقيم فلكية.
- "synthetic probe" → مسبار اصطناعي; "setup" → إعداد (إعدادات); "regime" → نمط (أنماط); "trainability" → قابلية التدريب; "untrainable" → غير قابل للتدريب; "run" → جولة تدريب; "validation accuracy" → دقة التحقق.
- "step-size conditioning" → اشتراط حجم الخطوة (from the glossary's اشتراط مسبق); "catch-22" → مأزق; "semilog plot" → رسم نصف لوغاريتمي; "a deep plain ReLU stack" → مجموعة مكدسة عميقة عادية من طبقات ReLU.
- Sigmoid in running text → دالة سيغمويد ("with sigmoid activations" → مع التفعيل بدالة سيغمويد); slide lead-ins "ReLU (…):" → دالة ReLU (…): so that the block starts in Arabic.
- Table cells "**Framework** `layer`" → الطبقة `layer` في **Framework** (an Arabic word between the Latin name and the code span); "per layer" → لكل طبقة.
- English glosses given in u01 of this page (do not repeat in later units): forward pass, permutation symmetry, i.i.d., pre-activations, vanishing gradients, exploding gradients, fan-in, fan-out, normalization layers, residual connections, gradient clipping.

## chapter_multilayer-perceptrons/dropout.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "dropout" → الحذف العشوائي (glossary), glossed (dropout) once; "inverted dropout" → الحذف العشوائي المعكوس; "dropout probability" → احتمال الحذف العشوائي; "dropout layer" → طبقة الحذف العشوائي; "dropped units" → الوحدات المحذوفة; the class name `Dropout` and BatchNorm stay Latin (طبقة BatchNorm).
- "co-adaptation" → التكيف المشترك (co-adaptation); "partner units" → وحدات شريكة (الشركاء); units "co-occur" → تحضر معًا; "conspire" / "conspiracies" → تتواطأ / مؤامرات.
- "masking" → التقنيع; "mask" → قناع (أقنعة; glossary); "unmasked" → غير مقنّع; "rescaling" → إعادة التحجيم; "surviving activations / units", "survivors" → التفعيلات / الوحدات الباقية، القيم الباقية; "zero out" → يصفّر; "thinned" (subnetworks, ReLUs) → مخففة; "subnetwork" → شبكة جزئية.
- "inject noise" → حقن الضوضاء; "multiplicative / additive noise" → ضوضاء ضربية / جمعية; "noisy" (data, labels, masks) → مشوب بالضوضاء; "perturbation-resilient" → متين أمام الاضطرابات.
- "memorize / memorization" → يحفظ / الحفظ; "late-stage memorization" → الحفظ في المراحل المتأخرة; "training quirks" → خصوصيات عارضة في بيانات التدريب; "corrupted labels" → الوسوم التالفة, "wrong labels" → الوسوم الخاطئة, "corrupt" (verb) → يفسد; "reproduce" (wrong labels) → يعيد إنتاج.
- "test time" → وقت الاختبار; "at inference time" → عند الاستدلال; "disable / turn off dropout" → تعطيل الحذف العشوائي; "deterministic(ally)" → حتمي / على نحو حتمي; "training mode" → نمط التدريب; "training-mode flag" → راية نمط التدريب; "mode switching" → تبديل النمط.
- "ensemble" (of subnetworks) → مجموعة (مجموعة الشبكات الجزئية، تقريب مجموعة النماذج), as on the probability page.
- "adaptive ridge penalty" → عقوبة ريدج تكيفية; "feature energy" → طاقة الميزة, "energy-weighted" → موزون بالطاقة; "variance inflation" → تضخم التباين; "train--test variance shift" → انزياح في التباين بين التدريب والاختبار; "running variance / statistics" (BatchNorm) → التباين التراكمي / الإحصاءات التراكمية; "expectation mismatch" → عدم تطابق التوقع.
- "He-initialized" → مهيأ بتهيئة He (He stays Latin: no glossary term carries the name); "uniform tensor" → موتر منتظم التوزيع; "a vector of ones" → متجه عناصره واحدات; "pack hyperparameters into" → يحزم … في.
- "trajectories" (accuracy curves of two runs) → المساران; "early peak" → الذروة المبكرة; "single-pass pretraining" → التدريب المسبق بمرور واحد; "LLMs" on slides → النماذج اللغوية الكبيرة; "Transformers" → المحولات; "stress test" → اختبار إجهاد; "Warm-Up" (slide title) → إحماء; kicker "Check" → التحقق.
- "MLP" in running text takes the class noun شبكة (شبكة MLP غير منظمة، شبكة MLP ذات طبقتين مخفيتين).
- English glosses given in u01 of this page (do not repeat in later units): dropout, co-adaptation, inverted dropout.

## chapter_multilayer-perceptrons/mlp-implementation.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "MLP" → شبكة MLP (شبكتا MLP، شبكات MLP), the acronym after a class noun as the English writes it; the full term البيرسبترون متعدد الطبقات only where the English spells it out (slide titles); "two-layer MLP" → شبكة MLP ذات الطبقتين; "ReLU units" → وحدات ReLU; "hidden features" → الميزات المخفية; "hidden representation" → تمثيل مخفي.
- "Implementation" (section title) → التنفيذ; "Implementation from Scratch" → التنفيذ من الصفر; "Checks before Training" → فحوص قبل التدريب; "pre-training checks / invariants" → فحوص / لامتغيرات ما قبل التدريب; "sanity checks" → فحوص السلامة; "invariants" → لامتغيرات.
- "contrast" (bright next to dark in an image) → تضاد (أنماط التضاد); التباين stays variance (حافظ للتباين on the same page).
- "output layer" → طبقة الإخراج (style §10.1, as in the chapter index); "outgoing weight" → وزن خارج (أوزانها الخارجة); "high-norm unit" → وحدة عالية المعيار; "whole-garment templates" → قوالب لقطع ملابس كاملة; "blurred" → ضبابية.
- "controlled comparison" → مقارنة مضبوطة; "head-to-head" → مباشرةً / مقارنة مباشرة; "linear baseline" → النموذج الخطي المرجعي; "validation accuracy histories" → سجلات دقة التحقق, "trajectory" → مسار; "trajectory-for-trajectory" → مسارًا بمسار.
- "wall-clock time per epoch" → الزمن الفعلي للحقبة; "GPU epoch time" → زمن الحقبة على وحدة GPU; "runtime" → زمن التشغيل; "data loading" → تحميل البيانات; "kernel-launch overhead" → العبء الإضافي الناتج عن إطلاق النوى; "tensor cores" → أنوية الموترات (أنوية for hardware cores); "in milliseconds / microseconds" → في أجزاء من الألف / من المليون من الثانية.
- "container" (`Sequential`) → حاوية; "parameter class" → صنف المعلَمات; "manual parameter arrays" → مصفوفات معلَمات يدوية; "floats" → أعداد بالفاصلة العائمة; "parameter budget" → ميزانية المعلَمات.
- "active ReLU fraction" → نسبة وحدات ReLU النشطة; "dead" (layer, ReLU units) → ميت; "hidden pre-activations" → قيم ما قبل التفعيل في الطبقة المخفية; "slow warmup" → إحماء بطيء; "suppressed by a factor of" → تُكبح بمعامل; "to trail" → يتأخر عن.
- "tops" (upper-body garments) → الملابس العلوية, one of them → قطعة علوية; "natural suspects" → من البديهي أن نشتبه بها; "widen" a layer, "scaling to $h$" → يوسع، التوسيع; "fan-in-scaled" → محجَّمة حسب عدد مدخلات الطبقة; "overriding the default initialization" → استبدال التهيئة الافتراضية.
- Kickers: "The Empirical Question" → السؤال التجريبي; "Initialization Invariants" → لامتغيرات التهيئة; "High-Level API" → واجهة API عالية المستوى; "Check" → التحقق; "Visualizing First-Layer Weights" → تصوير أوزان الطبقة الأولى; "Quantitative Summary" → ملخص كمي.
- English glosses given in u01 of this page: only the garment names T-shirt, pullover, coat, shirt (code and figure titles keep them). No term gloss: the page introduces no new term.

## chapter_multilayer-perceptrons/backprop.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "backward recurrence" → العلاقة التكرارية الخلفية ("recurrence" as a mathematical recursion → علاقة تكرارية; التكرار alone stays iteration/recurrence in sequence models).
- "upstream gradient / sensitivity / errors" → التدرج الوارد من جهة المخرج / الحساسية الواردة من جهة المخرج / الأخطاء الواردة من جهة المخرج (as on the calculus page); "incoming gradient" → التدرج الوارد.
- "to gate" (a gradient by the activation slope) → يمر / تمرير … عبر بوابة (بوابة ميل التفعيل، بوابة ReLU); "to zero out" → يصفّر; "dead ReLU unit" → وحدة ReLU ميتة (وحدات ReLU الميتة); "inactive" (unit) → غير نشطة.
- "broadcasted bias addition" → جمع الانحياز بالبث; "elementwise activation" → التفعيل عنصرًا بعنصر; "local matrix-calculus rules" → قواعد محلية من التفاضل المصفوفي; "error signal" → إشارة الخطأ; "summand" → حد من حدود المجموع; "factor" (of a product at a multiply node) → عامل.
- "to cache" (forward values) → يخزن مؤقتًا (المدخل المخزن مؤقتًا، التفعيلات المخزنة مؤقتًا); "read off" (a gradient) → يستخرج; "two-step module" (per-layer backward step) → إجراء من خطوتين (not وحدة, which is the hidden unit here).
- "worked example" → مثال محلول; "by hand" → يدويًا; "gradient check" → فحص التدرجات; "rounding floor" → أرضية التقريب; "relative discrepancy" → الفرق النسبي; "a few units of machine epsilon" → بضعة أمثال إبسيلون الآلة.
- "the MLP" as a common noun → شبكة MLP (شبكات MLP); "one-hidden-layer MLP" → شبكة MLP ذات طبقة مخفية واحدة; "$K$-layer MLP" → شبكة MLP من $K$ طبقة; "interior layer" → طبقة داخلية.
- "one-hot label matrix" → مصفوفة الوسوم بالترميز الأحادي; "directed acyclic graph" → مخطط موجه لادوري, "DAG" → مخطط DAG; "children" (of a node) → أبناء; "outgoing edges" → الأضلاع الخارجة; "downstream paths" → مسارات لاحقة.
- Slides: "MACs" → عملية MAC (class noun, no Latin plural); "matmul" → جداء مصفوفات; "rule of thumb" → قاعدة تقريبية; "tangents / adjoints" → المماسات / القيم القرينة; "rank-1 matrix" → مصفوفة من الرتبة 1; "floats" → أعداد بالفاصلة العائمة; "contract" (a product into a small result) → يختزل; "Systems Cost" → كلفة الأنظمة; "credit assignment" → إسناد الفضل; kicker "[Check]" → [التحقق]; "None" in a table cell → لا شيء.
- English glosses given in u01 of this page (do not repeat in later units): reverse-mode automatic differentiation, computational graph, forward pass, backward pass, reverse topological order. Backpropagation occurs only in headings and slides so far and has no gloss yet.

## chapter_multilayer-perceptrons/generalization-deep.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "interpolate" → يستوفي (glossary استيفاء), "interpolator" → مستوفٍ (المستوفي، مستوفيات), "interpolating function" → الدالة المستوفية; "minimum-norm interpolator / solution / fit" → المستوفي / الحل / الملاءمة ذو (ذات) المعيار الأصغر; "minimum-norm least squares" → المربعات الصغرى ذات المعيار الأصغر; "zero-loss solutions" → الحلول ذات الخسارة الصفرية.
- "overparameterized" → مفرط المعلَمات, "overparameterization" → فرط المعلَمات, "underparameterized" → ناقص المعلَمات; "regime" (classical, overparameterized) → نطاق (نطاق فرط المعلَمات).
- "memorization" → الحفظ ("random-label memorization" → حفظ الوسوم العشوائية; "memorization capacity" → سعة الحفظ); "capacity bound" → حد السعة (حدود السعة); "distribution-free" → بلا افتراض على التوزيع; "vacuous" → فارغ; "label-blind" → لا يرى الوسوم; "function class" → فئة الدوال.
- "no free lunch theorem" → مبرهنة لا غداء مجاني; "1-nearest-neighbor classifier" → مصنف أقرب جار واحد (glossary أقرب جار); "random chance" (accuracy level) → مستوى الصدفة.
- "double descent" → النزول المزدوج (glossary), "second descent" → النزول الثاني; "spike" (in test error) → قفزة (يقفز); "heal" → يتعافى; "cliff" → جرف; "U-curve" → منحنى U (منحنى على شكل U).
- "random-features model" → نموذج ميزات عشوائية; "tractable" → قابل للتحليل; "controlled experiments" → تجارب محكمة; "exact results" → نتائج مضبوطة; "freeze" (a layer) → يجمد; "conditioning" → الاشتراط, "poorly / well conditioned directions" → الاتجاهات سيئة / جيدة الاشتراط, "singular conditioning" → الاشتراط المنفرد.
- "maximum-margin separator" → الفاصل ذو الهامش الأقصى (glossary هامش أقصى); "hard-margin SVM" → آلة SVM ذات الهامش الصلب; "support vectors" → متجهات الدعم (the established term; الحامل stays the support of a function); "boundary examples" → الأمثلة الحدية; "row space" → فضاء الصفوف; "full row rank" → رتبة صفوف كاملة; "underdetermined" → ناقص التحديد; "unknowns" → مجاهيل; "solution hyperplane" → المستوي الفائق للحلول; "null-space component" → المركبة في الفضاء الصفري.
- "clean-first learning" → تعلم الأنماط النظيفة أولًا; "structured patterns" → الأنماط ذات البنية; "coherent signal" → الإشارة المتسقة; "grokking" → ظاهرة grokking (Latin after a class noun, like softmax; no glossary entry in this unit).
- "Limits of …" (heading, in the sense of shortcomings) → قصور …; headings "X under squared / logistic loss" → X في حالة الخسارة التربيعية / اللوجستية; "sample-wise" (double descent) → بحسب حجم العينة; "row-space invariance" → الثبات في فضاء الصفوف; "Pythagorean optimality" → الأمثلية وفق مبرهنة Pythagoras.
- English glosses given in u01 of this page (do not repeat in later units): interpolate, overparameterized, minimum-norm interpolator, early stopping, vacuous, no free lunch theorem, inductive bias, 1-nearest-neighbor, random features, double descent, interpolation threshold, implicit bias, implicit regularization, maximum-margin separator.

## chapter_multilayer-perceptrons/kaggle-house-price.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "Predicting House Prices on Kaggle" → التنبؤ بأسعار المنازل على Kaggle; "house", "home" → منزل (منازل); "mansion" → قصر; "starter home" → منزل صغير رخيص; "dwelling style" → نمط المسكن; "duplex" → مسكن مزدوج; "pool", "alley" → مسبح، ممر خلفي. The city is written أيمز (ولاية أيوا); the dataset name Ames Housing stays Latin.
- "attribute" (a raw column describing a house) → صفة (صفات); ميزة stays the model's feature and سمة the Python attribute. "numeric category code" → رمز فئة عددي (رموز الفئات العددية); "row identifier" → معرف الصفوف; "nominal categories" → فئات اسمية.
- log-RMSE stays Latin (مقياس log-RMSE، قيمة log-RMSE); RMSE where the English writes the acronym → خطأ RMSE (على التدريب / على التحقق); the full term is the glossary's جذر متوسط مربع الخطأ.
- "leakage-free" → خال من التسرب (المعالجة المسبقة الخالية من التسرب; on a cover slide بلا تسرب); "leak into" → يتسرب إلى، يسرّب; "leaked statistics" → الإحصاءات المتسربة.
- "linear baseline", "baseline model" → النموذج الخطي المرجعي، النموذج المرجعي (chapter decision: baseline model → نموذج مرجعي); "the right baseline" (a level) → خط الأساس.
- "5-fold cross-validation" → التحقق المتقاطع بـ 5 طيات; "shared 5-fold partition / splits" → التقسيم المشترك إلى 5 طيات; "shared folds" → طيات مشتركة; "fold-to-fold" → من طية إلى أخرى; "per-fold" → لكل طية، الطيات المفردة; "fold-sampling noise" → ضوضاء أخذ عينات الطيات; "round" → جولة.
- "target centering" → مركزة الهدف (الأهداف الممركزة; "un-centering" → إلغاء المركزة); "logarithmic target transformation" → التحويل اللوغاريتمي للهدف; "log price" → لوغاريتم السعر; "exponentiate" → يحسب الدالة الأسية.
- "He-initialized", "He init" → مهيأ بتهيئة He، تهيئة He (name in Latin); "64 ReLU units" → 64 وحدة ReLU; "MLP" → شبكة MLP (class noun, as the English writes the acronym).
- "model factory" → مصنع نماذج; "preprocessor" → معالج مسبق; "fit the preprocessing on train" → تُقدَّر على `train` (decision: fit statistics → يقدّر); "dummy columns" → الأعمدة الوهمية; "binary indicator" → مؤشر ثنائي; "informative missingness" → فقدان يحمل معلومات.
- "jitter" → اهتزاز; "damp" → تخميد; "sweep `wd`" → يمسح قيم `wd`; "textbook U-shape" → شكل U نموذجي; "under-/over-regularized" → تنظيم ناقص / تنظيم مفرط; "head to head" → وجهًا لوجه.
- "submission", "submitting predictions" → التقديم، تقديم التنبؤات; "winning weight decay" → اضمحلال الأوزان الفائز; "ensemble the fold models" → يكوّن مجموعة من نماذج الطيات; "gradient-boosted decision trees" → أشجار القرار المعززة بالتدرج.
- Slides: "capstone" → ختامي (القسم الختامي، التحدي الختامي); "winner's curse" → لعنة الفائز; "optimistic bias" → الانحياز المتفائل; "downward selection bias" → انحياز اختيار نحو الأسفل; "price brackets" → شرائح الأسعار; "Type Hygiene" → سلامة الأنواع; "Pitfall Prevented" → المزلق المتجنَّب; "reality check" → اختبار الواقع; kicker "[Check]" → [التحقق]; "Step 1 & 2" → الخطوتان 1 و 2; "A + B" between two code spans in a table cell → A مع B.
- English glosses given in u01 of this page (do not repeat in later units): data leakage, root-mean-squared error (RMSE).

## chapter_multilayer-perceptrons/dropout.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "Monte Carlo dropout" → الحذف العشوائي بطريقة مونت كارلو; "switch dropout / the masks off, on" → يعطّل / يبقيه مفعّلًا، والأقنعة معطلة / مفعّلة; "swap the two probabilities" → تبديل الاحتمالين أحدهما بالآخر.
- "report" (a number, in exercises) → أبلغ عن; "tolerates" → تتحمل; "strength" of weight decay → شدة (شدات); "mechanism" → آلية.
- "additive" (effects) → جمعي; "diminishing returns" → عوائد متناقصة; "cancel each other out" → يلغي أحدهما الآخر.
- "mismatch between training and test" → عدم التطابق بين التدريب والاختبار; "weighted geometric mean" → المتوسط الهندسي الموزون ("normalized" → المطبَّع); "$\boldsymbol{\xi}$ ranges over the masks" → تأخذ $\boldsymbol{\xi}$ قيمها من الأقنعة كلها; "neither equality holds in general" → لا تصح أي من المساواتين عمومًا.
- English glosses on this page in u02: none (exercises take no gloss).

## chapter_multilayer-perceptrons/mlp.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "Multilayer Perceptrons" (page title) → البيرسبترون متعدد الطبقات (as the chapter title); first mention *البيرسبترون متعدد الطبقات* (multilayer perceptron, MLP), afterwards "an MLP" → شبكة MLP (feminine agreement).
- "monotonicity" → الرتابة, "monotonic / non-monotonic" → رتيب / غير رتيب; "feature interactions" → تفاعلات الميزات (التفاعل بين الميزات); "score" (linear) → درجة, so body temperature is written حرارة الجسم without درجة beside it.
- "XOR problem" → مسألة XOR; "exclusive-OR" → دالة *أو الحصرية* (exclusive-OR, XOR); "corners" of the unit square → أركان; "diagonal" → قطر (قطرا مربع الوحدة); "midpoint" → نقطة المنتصف; "flip" (a bit) → قلب.
- "**Proposition.**" → **قضية.**; "**Proof.**" → **برهان.**; "Suppose for contradiction" → لنفترض على سبيل التناقض; "misclassify" → يخطئ في تصنيف; "decision rule" → قاعدة القرار.
- "pre-activations" → قيم ما قبل التفعيل (pre-activations); "activations" → التفعيلات; "width" (of a layer) → عرض; "hidden layer of $2$ units" on slides → طبقة مخفية عرضها $2$ where the math span must stay.
- "collapse" (layers into one affine map) → ينهار، الانهيار (الانهيار الجبري، الانهيار الخطي); points that "collapse onto" one point → تنطبق على.
- "rectified linear unit" → *الوحدة الخطية المصحَّحة* (rectified linear unit, ReLU), then ReLU with a class noun (دالة ReLU، وحدة ReLU، تفعيلات ReLU); "rectifying activations" → التفعيلات المصحِّحة; "squashing activations" → التفعيلات الضاغطة, "squasher" → ضاغط; "dead ReLU" → وحدات ReLU الميتة; "inactive units" → الوحدات غير النشطة; a unit "turns on" → تنشط.
- "fold" → يطوي، طي (طي الفضاء); "band / strip" → شريط (الشريط القطري); "carve out" → يقتطع; "hidden space" → الفضاء المخفي; "input space" → فضاء المدخلات; "compositional hierarchy" → تراتب تركيبي; "linear readout" → القراءة الخطية.
- "hinge" → مفصل (مفاصل); "knot" → نقطة انكسار; "polyline" → خط منكسر; "linear pieces" → قطع خطية; "linear segments" → قطع مستقيمة; "tent function / map" → دالة الخيمة; "continuous" (function) → متصلة; "compact" → متراص; "non-polynomial" → ليست كثير حدود; "closed bounded interval" → فترة مغلقة ومحدودة; "uniform approximation" → التقريب المنتظم; "universal approximation theorem" → مبرهنة التقريب الشامل.
- "attenuate" (gradients) → يوهن، توهين; "saturate / saturation" → يتشبع، التشبع; "tail" (of an activation) → ذيل; "gradient flow" → سريان التدرج; "kink" → انكسار; "dip" → انخفاض; "expressive power" → القدرة التعبيرية; "learnability" → قابلية التعلم; "depth efficiency" → كفاءة العمق.
- Slides: "Constructive Example" → مثال إنشائي; "Head-to-Head" → وجهًا لوجه; kicker "[Check]" → [التحقق]; "LSTMs/GRUs" → شبكات LSTM و GRU; plain "Sigmoid", "Tanh" in titles and table cells → سيغمويد، الظل الزائدي; "Unit 1 / Unit 2" → الوحدة الأولى / الوحدة الثانية.
- English glosses given in u01 of this page (do not repeat in later units): multilayer perceptron (MLP), hidden layer, activation function, exclusive-OR (XOR), pre-activations, activations, nonlinearity, rectified linear unit (ReLU). Not glossed yet (slides only so far): universal approximation, hinge, sigmoid, hyperbolic tangent, GELU, leaky ReLU, vanishing gradients.

## chapter_multilayer-perceptrons/mlp-implementation.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "the identity" (function, in place of an activation) → الدالة المحايدة (no glossary entry; متطابقة stays the identity equation, مصفوفة الوحدة the identity matrix).
- "hidden width" → العرض المخفي (plural العروض المخفية), as on the generalization-deep page; "affine maps" → التحويلات التآلفية (glossary تحويل تآلفي); "interval of $t$" → فترة من قيم $t$; "regions of input space" → مناطق فضاء المدخلات.
- "your answer to (b)" → إجابتك عن الجزء (b); "run" (one training of five) → جولة تدريب; "fraction of validation examples" → نسبة أمثلة التحقق; "the arithmetic" (cost that grows with width) → العمل الحسابي; "time one pass over the training data" → قس زمن مرور واحد على بيانات التدريب.
- English glosses on this page in u02: none (the summary uses terms glossed earlier; exercises take no gloss).

## chapter_multilayer-perceptrons/kaggle-house-price.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "target leakage" → تسرب الهدف; "target encoding" → ترميز الهدف (verb: يرمّز بترميز الهدف); "encode out of fold" → يرمّز خارج الطية; "in-sample encoding" → الترميز داخل العينة.
- "fold ensemble" → مجموعة نماذج الطيات; "ensembles of decision trees" → مجموعات أشجار القرار; "gradient-boosted trees" → الأشجار المعززة بالتدرج; "fit a model to" (exercise imperative) → لائم نموذج … على.
- "leaderboard score" → الدرجة على لوحة المتصدرين; "cross-validated error / RMSE" → خطأ التحقق المتقاطع، خطأ RMSE في التحقق المتقاطع.
- "median imputation" → التعويض بالوسيط; "imputed feature" → ميزة معوَّضة; "unstandardized / standardized" (model, pipeline, columns) → غير موحد معياريًا / موحد معياريًا; "to within a factor of ten" → بدقة لا تتعدى معاملًا مقداره عشرة.
- "geometric averaging", "geometric mean" → المتوسط الهندسي; "price space" → فضاء الأسعار; "price-space average" → المتوسط في فضاء الأسعار; "log-space average" → المتوسط في الفضاء اللوغاريتمي.
- "Adapted from Kaggle Learn, [course], Exercise N" → مقتبس بتصرف من دورة [course] في Kaggle Learn، التمرين N (as on the pandas page).

## chapter_multilayer-perceptrons/numerical-stability-and-init.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "Symmetry among Hidden Units" → التناظر بين الوحدات المخفية; "permutation-symmetric" → متناظرة تحت التبديل; "permutation invariance" → الثبات تحت التبديل; "single effective neuron / effective unit" → عصبون فعال واحد / وحدة فعالة.
- "He-scaled weights" → أوزان بمقياس تهيئة He; "He scale" → مقياس He.
- "submultiplicative" (norm) → تحت ضربي; "submultiplicativity" → الخاصية تحت الضربية; "largest singular value" → القيمة المفردة العظمى; "grow geometrically" → ينمو هندسيًا, "decay geometrically" → يتناقص هندسيًا; "strictly smaller than one" → أصغر تمامًا من واحد.
- "the function of an MLP" → الدالة التي تحسبها شبكة MLP; "leaves … unchanged" → يُبقي … على حاله; "incoming / outgoing weights" → أوزان داخلة / خارجة; "output bias" → انحياز المخرجات; "replacing A by B" → الاستعاضة عن A بـ B (keeps the token order).
- "averaged fans" → متوسط التفرعين; "widths" (a list of layer widths) → قيم العرض; "tempting shortcut" → طريق مختصر مغرٍ; "negative half-line" → نصف المستقيم السالب.
- "layer gain" → كسب الطبقة; "rescaling symmetry" → تناظر إعادة التحجيم; "architectural stabilizers" → عوامل استقرار معمارية (not مثبتات: المثبِّت is the installer); "residual skip connections" → وصلات التخطي ذات البواقي; BatchNorm, LayerNorm, Leaky ReLU, GELU stay Latin.
- Summary table: "Small / Large Fixed $\mathcal{N}(\cdot)$" → توزيع ثابت صغير / كبير; "(on ReLU)" → (مع ReLU); "`inf`/`NaN`" → `inf` أو `NaN`.
- Credit line with several problems: … لمؤلفه Simon Prince، المسائل 3.5 و 4.3 و 7.15.
- English glosses on this page in u02: none (the section uses terms glossed in u01; exercises and slides take no gloss).

## chapter_multilayer-perceptrons/backprop.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "Arithmetic Cost" (heading) → كلفة العمليات الحسابية; "multiply-adds" after a number → N عملية ضرب وجمع.
- "one-sided finite difference" → الفرق المحدود أحادي الجانب; "central difference" → الفرق المركزي (as on the calculus page).
- "forward / backward sweep" → الاجتياز الأمامي / اجتياز خلفي; "sweeps the graph backward / forward" → يجتاز المخطط إلى الخلف / إلى الأمام; "seeds X at" → يضع البذرة X عند.
- "prediction" in the exercises (the English writes "prediction", not "inference") → التنبؤ; "inference" stays الاستدلال.
- "autograd engine" → محرك اشتقاق تلقائي; "miniature" → مصغر; "parents" (of a graph node) → آباء; "references" (object references) → مراجع; "affordable" → في المتناول.
- "combination of the four indices" → توليفة من الفهارس الأربعة; "object" (a mathematical object) → كائن.
- The tag "[extended]" is copied like "[code]".
- "Adapted from Andrej Karpathy, [micrograd](…), `file`" → مقتبس بتصرف من مشروع [micrograd](…) لمؤلفه Andrej Karpathy، الملف `file`.
- English glosses given in u02 of this page: backpropagation (first body-text occurrence, in the Summary), finite difference.

## chapter_multilayer-perceptrons/generalization-deep.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "patience" (early stopping) → الصبر (نافذة صبر، قاعدة الصبر، الإيقاف القائم على الصبر), glossed (patience) once; "tolerance" $\theta$ → هامش السماح; "stopping epoch" → حقبة التوقف; "retained epoch" → الحقبة المحتفظ بها; "consecutive epochs" → حقب متتالية.
- "checkpoint" → نقطة استعادة (from the glossary's حفظ نقاط الاستعادة); "best validation checkpoint" → نقطة الاستعادة ذات أفضل أداء على التحقق; "restore" (a checkpoint) → الرجوع إلى.
- "modular addition" → الجمع المقاسي, glossed (modular addition) once; "modular arithmetic" → الحساب المقاسي (not المعياري "standard" and not النمطي: نمط is "mode"); "algorithmic tasks" → مهام خوارزمية; "training table" → جدول التدريب.
- "grokking" → ظاهرة grokking (as in u01), also in the caption, the exercise title and slide titles.
- "term" of a bound ($B$, "the second term") → المقدار, so that it is not confused with الحد "bound" in the same sentence; "lower bound" → حد أدنى; "complexity measures" → مقاييس التعقيد; "a member of $\mathcal{F}$" → عنصر من.
- "infimum" → الحد الأدنى الأعظم; "grows without bound" → ينمو بلا حد; "converges in direction" → يتقارب في الاتجاه; "rescalings" → إعادات التحجيم; "coordinate along the $i$-th eigenvector" → الإحداثي على امتداد المتجه الذاتي رقم $i$.
- "peak" (of the double-descent curve) → قمة (قفزة stays "spike"); "tuned model" → النموذج المضبوط; "training fraction" → نسبة التدريب; "weight-decay strength" → شدة اضمحلال الأوزان; "delay" → التأخر; "saturates" (accuracy) → تتشبع; "redundant features" → الميزات المكررة.
- Slides: "Temporal / Spatial implicit bias" → الانحياز الضمني الزمني / المكاني; "clean-first window" → نافذة الأنماط النظيفة أولًا; "memorization phase" → مرحلة الحفظ; "coherent gradients" → التدرجات المتسقة; "implicit ridge" → ريدج ضمني; "Corrupted-label arithmetic" → حساب الوسوم التالفة; kicker "[Check]" → [التحقق]; "Answers:" → الإجابات:.
- Credit line for a book in an exercise: *مقتبس بتصرف من كتاب [Title](url) لمؤلفه Author، الفصل N.* / لمؤلفيه A و B و C، القسم N.
- English glosses given in u02 of this page: patience, modular addition.

## chapter_multilayer-perceptrons/mlp.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "joint" (of a ReLU unit, in the exercises) → نقطة انكسار, the same as "knot"; "one hinge per joint" → مفصل لكل نقطة انكسار.
- "$m$-fold composition" → التركيب $m$ مرة; "composing $g$ with itself" → تركيب $g$ مع نفسها; "sawtooth wave" → موجة سن المنشار.
- "ridge function" → دالة ريدج (دوال ريدج), with ريدج as in the glossary's ridge terms; "Stone–Weierstrass theorem" → مبرهنة Stone–Weierstrass (names in Latin: no glossary term carries them).
- "Gaussian error linear unit" → الوحدة الخطية ذات الخطأ الغاوسي, then GELU; "standard normal CDF" → دالة التوزيع التراكمي للتوزيع الطبيعي المعياري; "its cousin" → قريبتها.
- "positive half-line" → نصف المستقيم الموجب; "horizontal asymptotes" → خطوط مقاربة أفقية; "state squashers" → ضاغطات للحالة; "recurrent architectures" → المعماريات المتكررة; "rectifiers" → الدوال المصحِّحة.
- "the plane" → المستوي; "uniform limit" → نهاية منتظمة; "closed under sums, products, and scalar multiples" → مغلقة تحت الجمع والضرب والضرب في عدد قياسي; "finite linear combinations" → التراكيب الخطية المنتهية; "the graph" (of a function) → المنحنى البياني.
- Exercise title "XOR by hand" → مسألة XOR يدويًا (the class noun so that the block starts in Arabic); credit line for a book chapter: … لمؤلفه Author، الفصل N.
- English glosses given in u02 of this page: hinge, universal approximation theorem, ridge functions, dead ReLU, Gaussian error linear unit, logistic sigmoid, hyperbolic tangent, vanishing gradients.

## chapter_multilayer-perceptrons: chapter check (claude fable-5.2 (medium), 2026-11-27)
- One rendering per term across the chapter's eight pages; the entries above were corrected in place.
- "Multilayer Perceptrons" as chapter title, page title and in slide titles, and the full term in running text → البيرسبترون متعدد الطبقات (glossary, definite singular; no plural البيرسبترونات); "an MLP", "MLPs" → شبكة MLP / شبكات MLP on every page (never نموذج MLP).
- Slide kicker "[Check]" → [التحقق]; التحقق من الفهم stays for "Check Your Understanding" and اختبار الفهم for "Concept Check".
- "inactive" (ReLU unit) → غير نشطة (opposite of النشطة; not خاملة); "dead ReLU" → وحدة ReLU ميتة.
- "contrast" (bright next to dark in an image) → تضاد (كاشفات تضاد، أنماط التضاد); التباين is variance only.
- "linear baseline" → النموذج الخطي المرجعي; "floats" → أعداد بالفاصلة العائمة.
- Kept apart on purpose: "regime" → نطاق for a range of model size (النطاق الكلاسيكي، نطاق فرط المعلَمات) and نمط for a kind of behaviour (أنماط التهيئة، نمط الانزياح); "head-to-head" → وجهًا لوجه in slide kickers and comparisons of models, مباشرةً as an adverb in running text.

## chapter_builders-guide/parameters-state-memory.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "buffer" in the framework sense (a registered, persistent, non-trainable tensor: `register_buffer`, BatchNorm running statistics) → مخزن حالة (مخازن الحالة), glossed (buffers) once per page; the earlier مخزن مؤقت stays the memory-buffer sense (a "persistent مخزن مؤقت" would contradict itself).
- "state tree" → شجرة الحالة; "module tree" → شجرة الوحدات, a framework "module" → وحدة; "registration" → التسجيل; "full-tree traversal" → اجتياز للشجرة كاملة; "leaf" (of the tree) → ورقة (أوراق); "dotted path" → المسار المنقط.
- "residual MLP" → شبكة MLP ذات بواقٍ (definite: شبكة MLP ذات البواقي); "residual block" → كتلة البواقي (glossary), indefinite dual كتلتا بواقٍ; "body" (of a block) → جسم.
- "checkpoint" → نقطة استعادة, glossed (checkpoints) once on this page; "checkpoint savers / writers" → أدوات حفظ / كتابة نقاط الاستعادة; "device transfer / device move" → النقل بين الأجهزة; "GPU transfer" → النقل إلى وحدة GPU; "dtype casting" → تحويل نوع البيانات.
- "running averages / means / variances / stats" (Adam, BatchNorm) → المتوسطات / التباينات / الإحصاءات المتحركة (المجموع التراكمي stays "running sum").
- "plain (Python) attribute" → سمة عادية (سمات Python عادية); "local variable" → متغير محلي; "persistent" (tensor) → دائم; "inventory" → قائمة الجرد; "parameter freezing" → تجميد المعلَمات; "language modeling" → نمذجة اللغة; "static floor" → الأرضية الثابتة; "full-precision" → بالدقة الكاملة.
- Slides and prose: "8B model", "8-billion-parameter" → نموذج بـ 8 مليارات معلَمة / ذو 8 مليارات معلَمة (no Latin "8B", which collides with B = bytes).
- Tab notes "In <framework> it is `x`" → في <framework> هذا الاجتياز هو `x`.
- English glosses given in u01 of this page (do not repeat in later units): checkpoints, registration, state tree, buffers, weight tying, parameter freezing, body.

## chapter_builders-guide/numerics.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "Numerical Formats" (page title) → الصيغ العددية; "floating-point formats" → صيغ الفاصلة العائمة (glossary); fp32, bf16, fp16, tf32, NaN stay Latin; in running text a format name takes the class noun الصيغة (في الصيغة fp16) at the first mention in a sentence, and may then stand alone after a preposition (في fp32).
- "32-bit floating point" → الفاصلة العائمة ذات 32 بتًا; "16-bit formats / matrix multiplications" → الصيغ ذات 16 بتًا / عمليات ضرب المصفوفات ذات 16 بتًا; "in 16 bits" → في 16 بتًا; "bit width" → عرض البتات; "bit budget" → ميزانية البتات; "bit layout" → تخطيط البتات.
- "dtype" → نوع البيانات, glossed (dtype); "dtype promotion" → ترقية النوع; "model casting" → تحويل نوع النموذج (cast → يحول النوع, as on the ndarray page).
- "precision" (floating point) → الدقة (style §10.1; the glossary's إحكام is the classification metric); "relative precision" → الدقة النسبية; "coarser / finer" → أخشن / أدق; "half precision" → نصف الدقة; "brain floating point" stays Latin as the name of the format (*brain floating point*).
- "mixed-precision training" → التدريب بالدقة المختلطة; "master weights", "master copy" → الأوزان الرئيسية، نسخة رئيسية; "loss scaling" → تحجيم الخسارة, "dynamic loss scaling" → تحجيم الخسارة الديناميكي.
- "rounding", "rounds to" → التدوير، يُدوَّر إلى (glossary خطأ التدوير); "rounds back to 1" → يعود بالتدوير إلى 1; "round-to-nearest" → التدوير إلى الأقرب; "rounding noise" → ضوضاء تدوير; "quantization noise" → ضوضاء تكميم.
- "sign bit" → بت الإشارة; "exponent" → الأس; "mantissa" → الجزء الكسري (as on the calculus slides), glossed (mantissa); "significant digit / bit" → خانة معنوية / بت معنوي; "decimal digits" → خانات عشرية.
- "normal" (floating-point number) → نظامي, glossed (normal); "subnormal" → دون نظامي (الأعداد دون النظامية), glossed (subnormal). Not طبيعي (the normal distribution) and not مطبَّع (normalized layers).
- "spacing" (between adjacent representable numbers) → المسافة (بين القيم المتجاورة); "grid" (of representable values) → شبكة القيم; "grid spacing" → مسافة الشبكة; "grid point" → نقطة الشبكة; "grid step" → خطوة من خطوات الشبكة. Not تباعد (divergence) and not الفاصل (too close to الفاصلة).
- "overflows to `inf`" → يطفح إلى `inf`; "underflow to zero" → يطفح سفليًا إلى الصفر; "running out of range" → نفاد المدى; "ceiling" (largest value) → سقف; "octave" (slide) → فترة تضاعف.
- "silently stalling learning", "stall" (training, optimization) → يتوقف التعلم بصمت، يوقف تقدم التحسين، يعطّل التدريب; "frozen" (weight) → مجمد; "swamping" (of an update) → ابتلاع; "hazard" → خطر (مخاطر).
- "GPU matrix units" → وحدات المصفوفات في معالجات GPU; "the 8-billion-parameter model" → النموذج ذو 8 مليارات معلَمة; "field" (of `finfo`) → حقل.
- English glosses given in u01 of this page (do not repeat in later units): dtype, dynamic range, precision, mixed-precision training, loss scaling, half precision, normal, mantissa, subnormal.

## chapter_builders-guide/index.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- Chapter title "Computation" → الحوسبة; cover slide "Deep Learning Computation" → الحوسبة في التعلم العميق.
- "Resources and Further Reading" (chapter index heading) → المصادر وقراءات إضافية; its bold groups: "Books" → الكتب; "Courses and video lectures" → المقررات والمحاضرات المصورة; "Tutorials, notes, and interactive" → الدروس والملاحظات والمواد التفاعلية; "Foundational papers" → الأوراق البحثية التأسيسية.
- Resource list items start with a class noun before the link (كتاب، مقرر، سلسلة، مقالة، صفحة، دليل، صيغة، ورقة), style §3.2; the link text (title and authors) stays in English; several references in one parenthesis are separated by ، ; "free online" → متاح مجانًا على الإنترنت; "archived copy" → نسخة مؤرشفة; "video series" → مقاطع مصورة; "survey" (paper) → مسح; "vendor guide" → دليل الشركة المصنعة.
- "module" (a framework building block in this chapter) → وحدة (وحدات); "tree of modules", "module tree" → شجرة من الوحدات، شجرة الوحدات; "root module" → الوحدة الجذر; "leaves" → الأوراق; "module hierarchies" → تراتبات الوحدات; "module registry" → سجل الوحدات; "registration" → التسجيل; "registered children" → الأبناء المسجلون; "state tree" → شجرة الحالة; "addressed by its path" → عنوانه مساره، العنونة بالمسارات.
- "buffer" (non-trainable state) → مخزن حالة (مخازن حالة ثابتة، غير قابلة للتدريب); "trainable" → قابل للتدريب; "share weights" → يشارك الأوزان; "freeze" → يجمد; "memory floor" → أرضية الذاكرة; "allocate storage" → حجز ذاكرة; "caching allocator" → نظام حجز الذاكرة ذو التخزين المؤقت.
- "lazy shape inference" → الاستنتاج المؤجل للأشكال; "lazy shapes" → الأشكال المؤجلة; "config-driven builders" → دوال البناء التي تقودها الإعدادات ("config" → الإعدادات); "list containers" → حاويات القوائم.
- "residual branch" → فرع البواقي (فروع البواقي); "residual skip connections" → وصلات التخطي ذات البواقي; "residual initialization" → تهيئة شبكات البواقي; "residual variance scaling" → تحجيم تباين البواقي; "Residual MLP" → شبكة MLP ذات بواقٍ; "unscaled" → غير محجَّم / دون تحجيم; "deep stacks" → المجموعات المكدسة العميقة; "block" → كتلة.
- "stateless / parameterized / fixed-buffer layers" → طبقات عديمة الحالة / ذات معلَمات / ذات مخازن حالة ثابتة; "custom" (layer, gradient, backward rule) → مخصص; "backward rule" → قاعدة خلفية; "straight-through estimator" → مقدر العبور المباشر (not التمرير المباشر, which is the feedthrough term); its English gloss is left to the custom-layers page.
- "persistence", "persist models" → الحفظ الدائم، تحفظ النماذج; "serialization" → الحفظ بصيغة …; "archive" → أرشيف; "code-free" → خالٍ من الشيفرة; "atomic checkpoint" → نقطة استعادة ذرية; "five-part" → من خمسة أجزاء; "resume bit for bit" → يستأنف بتًا ببت; "missing / unexpected keys" → المفاتيح المفقودة / غير المتوقعة; "pretrained backbones" → الهياكل المدرَّبة مسبقًا; "safe checkpointing" → حفظ نقاط الاستعادة الآمن; "RCE" stays Latin after ثغرة.
- "master weights" → الأوزان الرئيسية; "16-bit compute" → حساب ذو 16 بتًا; "round away" (small updates) → تمحو … عند التدوير; "rounding" → التدوير; fp32, fp16, bf16 stay Latin, joined by و ; "numerics" (slide) → الصيغ العددية; numeric "precision" → الدقة.
- "command queue" → طابور أوامر (طوابير); "to time" → يقيس زمن; "timer" → مؤقت; "synced" → مزامَن; "placement" (on a GPU) → التوضيع على وحدات GPU; "peak RAM" → ذروة ذاكرة RAM; "segment" (of blocks) → مقطع; "$O(\sqrt{K})$ activation checkpointing" → نقاط استعادة التفعيلات من مرتبة $O(\sqrt{K})$.
- "seed control" → التحكم في البذور; "PRNG seeding" → بذور PRNG; "GPU determinism" → حتمية وحدات GPU; "determinism flags" → رايات الحتمية; "hook" → خطاف (forward / backward hooks → خطافات أمامية / خلفية); "non-invasive" → لا تمس الشيفرة; "diagnostics" → التشخيص، تشخيصات.
- "textbook prototypes" → النماذج الأولية التي تعرضها الكتب الدراسية; "production-scale systems" → أنظمة بحجم بيئة الإنتاج; "behind the scenes" → من وراء الستار; "black box" → الصندوق الأسود; "hardware failures" → أعطال العتاد; "fault tolerance" → تحمل الأعطال; "security" → الأمان; "foundation model" → نموذج تأسيسي; "8B" on slides → 8 مليارات (معلَمة); only the model name Llama 3 8B keeps the Latin "8B".
- "data pipelines" → خطوط معالجة البيانات; "training infrastructure" → البنية التحتية للتدريب; "training loop" → حلقة تدريب; "plain Python" → Python الخالصة; "dispatch" → الإرسال; "framework overhead" → العبء الإضافي لإطار العمل; "gotchas" → مزالق; "imperative style" → الأسلوب الأمري; "design rationale" → مبررات التصميم; "I/O" → الإدخال والإخراج; "recursively" → تعاوديًا; "idioms" → أساليب.
- Slide labels: "The Systems Challenge" → تحدي الأنظمة; "Core Mental Model" → النموذج الذهني الأساسي; "Architectural & Systems Roadmap" → خارطة طريق المعمارية والأنظمة; "Quantitative Anchors" → مرتكزات كمية; "Study & Engineering Guide" → دليل الدراسة والهندسة; slide instructions ("Read …", "Pay close attention …", "Count …") are verbal nouns (قراءة، التنبه، عد); a comma between two § references becomes و.
- English glosses given on this page: tree of modules, state.

## chapter_builders-guide/custom-layers.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "custom layer / function" → طبقة مخصصة / دالة مخصصة; "built-in" (layer, library) → مدمج (الطبقات المدمجة، المكتبة المدمجة في إطار العمل); "native" (implementation, equivalent) → أصلي (الطبقة الأصلية، مكافئ أصلي); "first-class" → من الدرجة الأولى; "third-party" → من طرف ثالث; "codebase" → قاعدة شيفرة; "release cycle" → دورة إصدار.
- Section headings of this page are indefinite noun phrases: طبقات بلا معلَمات، طبقات ذات معلَمات، طبقات ذات حالة ثابتة.
- "module" (framework building block) → وحدة; "base module" → الوحدة الأساسية; "module tree" → شجرة الوحدات; "module interface" → واجهة الوحدات; "to subclass the base module" → يشتق صنفًا فرعيًا من الوحدة الأساسية; "register / registration" → يسجل / التسجيل; "sequential container" → حاوية تسلسلية; "composability" → قابلية التركيب; "slot into a container" → تأخذ مكانها في.
- "serialize into checkpoints", "checkpoint serialization" → تُحفظ في نقاط الاستعادة / الحفظ في نقاط الاستعادة (no separate word for "serialize"); "serialization round-trip" → دورة حفظ وتحميل; "device migration / movement" → الانتقال بين الأجهزة / النقل بين الأجهزة; "clone" (of a model) → نسخة مستنسخة; "bit for bit" → بتًا ببت.
- "buffer" (registered non-trainable state) → مخزن حالة (مخازن حالة غير قابلة للتدريب), as on the parameters page; "random features" → ميزات عشوائية; "random Fourier features" → ميزات فورييه العشوائية; "linear head" → رأس خطي; "Gaussian RBF kernel" → نواة RBF الغاوسية.
- "gain" (RMSNorm's learned vector) → كسب (متجه الكسب), glossed once; "its RMS" → قيمة RMS له (the full term is جذر متوسط المربعات); "mean-centering" → مركزة المتوسط; "activation vector" → متجه تفعيلات; "activation scales" → مقاييس التفعيلات; "learned scale parameter" → معلَمة مقياس متعلَّمة; "stretch" (an axis, a circle) → يمط; "ray from the origin" → شعاع منطلق من نقطة الأصل; "semi-axes" → نصفا المحورين; "ellipsoid" → مجسم قطع ناقص; "axis-aligned" → محاوره موازية لمحاور الإحداثيات.
- "quantize / quantized" → يكمّم / مكمَّم (glossary تكميم); "rounding" → التدوير; "4-bit integers" → أعداد صحيحة من 4 بتات; "straight-through estimator" → مقدر العبور المباشر, glossed once; "surrogate backward pass" → تمرير خلفي بديل; "gradients flow through" → تسري التدرجات عبر, and "gradient flow" on slides → سريان التدرج (as in the MLP chapter; the glossary's تدفق التدرج is the continuous-time system).
- "non-trivial weights / gain" → أوزان غير تافهة / كسب غير تافه; "integration checks" → فحوص التكامل; "numerical sanity check" → فحص السلامة العددي; "analytical hand-check" → فحص تحليلي يدوي; "normalization sites" → مواضع التطبيع; "the gain moves off ones" → يبتعد الكسب عن قيم الواحد.
- The heading and slide titles that begin with RMSNorm take the class noun طبقة (طبقة RMSNorm) so that the block starts in Arabic; NNX, Gluon, Keras stay Latin (مكتبة NNX where a class noun is needed).
- English glosses given in u01 of this page (do not repeat in later units): normalization layer, root mean square, quantization, straight-through estimator, gain, layer normalization.

## chapter_builders-guide/reproducibility-inspection.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "reproducibility" → قابلية إعادة الإنتاج (glossary); "repeatable" (runs) → قابلة للتكرار; "run" → جولة (جولة تدريب; "rerun" → الجولة المعادة، يعيد تشغيل); "training script" → برنامج تدريب.
- "to seed" → يضبط بذرة (ضبط البذور; "seeded source" → مصدر مضبوط البذرة); "re-seed" → يعيد ضبط البذرة; "global seed" → البذرة العامة.
- "pseudorandom number generator (PRNG)" → مولد أعداد شبه عشوائية, then مولد PRNG; "RNG" → مولد الأعداد العشوائية (no Latin acronym in body text); "generator object" → كائن المولد (كائنات المولدات); "private" (generator) → خاص.
- "stream" (of random numbers) → سيل (سيول); "key" (JAX) → مفتاح, "split a key" → يقسم المفتاح; "draw" → سحبة.
- "worker processes" → عمليات العمال; "process" (OS) → عملية; "thread" → خيط (خيوط); "fork" as a noun/verb in prose → التفريع، العمال المتفرعون (the call itself stays `fork`); "OS entropy" → إنتروبيا نظام التشغيل.
- "bit for bit" → بتًا ببت; "down to the last bit" → حتى البت الأخير; "non-finite value" → قيمة غير منتهية; "non-associativity" → عدم التجميعية (غير تجميعي); "atomic reductions / atomics" → الاختزالات الذرية / العمليات الذرية.
- "inspection" (of a model) → فحص النماذج; "hook" → خطاف (glossed here); "forward & backward hooks" → الخطافات الأمامية والخلفية; "non-invasive hooks" → خطافات لا تمس الشيفرة; "activation scales" → مقاييس التفعيلات.
- "seed-to-seed variance" → التباين من بذرة إلى أخرى; "spread" → التشتت; "crash" (of a training run) → انهيار; "failure mode" → نمط إخفاق; "functional random APIs" → واجهات API العشوائية الوظيفية.
- Slides: "Coverage caveat" → تنبيه حول التغطية; "(Row 1)/(Row 2)" → (السطر 1) / (السطر 2); "The fix" → الإصلاح; "bug hunting" → اصطياد الأخطاء.
- English glosses given in u01 of this page (do not repeat in later units): hooks, pseudorandom number generator (PRNG), seed, key, generator object, workers.

## chapter_builders-guide/init.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "Initialization" (page title) → التهيئة; "initializer" → مهيئ (مهيئات), "default initializer" → المهيئ الافتراضي, "built-in initializers" → المهيئات المدمجة, "custom initializer" → مهيئ مخصص; slide "framework defaults" → القيم الافتراضية في أطر العمل.
- "gain" $g$ → الكسب (from the earlier كسب الطبقة), "gain-parameterized" → ذو معلَمة كسب / بمعلَمة كسب; "gain schedule" (how $g$ depends on depth) → قاعدة كسب (قواعد الكسب).
- "truncated normal" → التوزيع الطبيعي المبتور (التوزيعات الطبيعية المبتورة); "truncate" → يبتر، البتر; "cutoffs" → حدا البتر; "clipping the tails" → بتر الذيلين (قص stays gradient clipping); "bounding extreme draws" → تقييد السحبات المتطرفة (الحصر stays "clamp").
- "module tree" → شجرة الوحدات (as in the regression chapter); "walk the tree" → السير في شجرة الوحدات; looping over an iterable → نمر على; "submodule" → وحدة فرعية; Gluon "block tree" → شجرة الكتل; "tree-wide" → على الشجرة كلها.
- Residual terms: "residual network" → شبكة البواقي; "residual block" → كتلة البواقي, indefinite or counted كتلة بواقٍ (32 كتلة بواقٍ), plural كتل البواقي; "ResNets" → شبكات ResNet; "body" (of a block) → الجسم; "output projection" → إسقاط المخرجات; "expansion layer" → طبقة التوسيع; "residual branch" → فرع بواقٍ; "residual stack" → مجموعة بواقٍ مكدسة; "depth-dependent residual scaling" → تحجيم البواقي المعتمد على العمق; "scaling down" → تصغير; "zeroing" → تصفير; "architecture's wiring" → توصيلات المعمارية.
- "mean square" → متوسط المربع (متوسط مربع المدخلات / المخرجات), as on the numerical-stability page; "variance multiplier" → العامل الضربي للتباين; "uncorrelated" → غير مرتبط; "compound" (a factor across layers) → يتراكم; "balloons / blows up by" → يتضخم / ينفجر بمعامل.
- "positively homogeneous" → متجانسة إيجابيًا; "identity map" → دالة محايدة (as الدالة المحايدة); "symmetry trap" → فخ التناظر; "wake-up" → استيقاظ; "paradox" → مفارقة.
- "kernel" as Keras/NNX name of a dense layer's weight matrix → النواة (النوى), without الالتفاف; "random key" → مفتاح عشوائي, "split a key" → يشتق مفتاحًا; "unseeded" → لا بذرة له; "trailing underscore" → شرطة سفلية في آخر الاسم; "in place" → في الموضع / في موضعه; "tracked variable" → المتغير المتتبَّع; "recipes" → وصفات; "variance-preserving schemes" → الطرق الحافظة للتباين.
- "the Vision Transformer" → محول الرؤية (Vision Transformer, ViT) at first mention, then ViT as the slides write it.
- Slide lead-ins that begin with a code span get the class noun الطبقة; a parenthesis holding only a math span after another math span gets أي / حيث (style §3.2).
- English glosses given in u01 of this page (do not repeat in later units): initializer, gain, residual network, residual block, output projection, Vision Transformer (ViT), truncated normal, positively homogeneous.

## chapter_builders-guide/saving-loading.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "pretrained weights" → الأوزان المدرَّبة مسبقًا; "pre-training" → التدريب المسبق; "fine-tune" → يضبط ضبطًا دقيقًا (style §5.3)
- "serialization", "serialize" → التسلسيل، يسلسل (not التسلسل, which is the glossary's "sequence"); "serialization format" → صيغة تسلسيل; "file format" → صيغة ملفات
- "state dictionary" → قاموس الحالة; "state tree" → شجرة الحالة; "buffers" (of a module) → مخازن الحالة ("raw tensor buffers" of a file stay المخازن المؤقتة الخام); "keys" (of a state dictionary) → المفاتيح; "missing / unexpected keys" → المفاتيح المفقودة / غير المتوقعة
- "a mapping from names to tensors" → تقابل من الأسماء إلى الموترات; "to map X to Y" → يقرن X بـ Y (not ربط, which stays concatenation, and not دالة)
- "checkpoint" → نقطة استعادة; "checkpointing" → حفظ نقاط الاستعادة (glossary); "training checkpoint" → نقطة استعادة التدريب; "resume", "resumption" → يستأنف، الاستئناف; "resumable" → قابلة للاستئناف; "interruption" → انقطاع; "crash" → انهيار (عطل stays a hardware "failure"); "crash-safe" → آمن من الانهيار
- "atomic" (write, checkpoint) → ذري (كتابة ذرية); "commits atomically" → يثبت ذريًا; "bit-for-bit" → بتًا ببت; "byte by byte" → بايتًا ببايت; "round trip" (save, then load) → الحفظ ثم التحميل / الحفظ والاسترجاع
- pickle stays in Latin: "pickle files", "pickles" → ملفات pickle; "unpickling" → فك pickle; "unpickler" → دالة فك pickle; "loaders" (`np.load`, `torch.load`) → دوال التحميل (محمل stays the data loader); "restricted / hardened loader" → دالة تحميل مقيدة / محصنة
- "allowlist" → قائمة السماح; "attack surface" → سطح الهجوم; "payload" → الحمولة; "malicious" → خبيث; "arbitrary code / objects" → شيفرة اعتباطية / كائنات اعتباطية; "vulnerabilities" → ثغرات; "trust boundary" → حد ثقة; "cryptographic checksum" → مجموع اختباري تشفيري; "integrity" → السلامة
- "bytecode" → الشيفرة البايتية; "callable" → كائن قابل للاستدعاء; "opcode parser" → محلل رموز العمليات; "stack-based virtual machine" → آلة افتراضية مبنية على مكدس; "shell commands" → أوامر shell; "passive data" → بيانات خاملة
- "metadata" → بيانات وصفية; "little-endian" → بترتيب النهاية الصغرى; "unsigned integer" → عدد صحيح بلا إشارة; "byte offsets" → إزاحات البايتات; "byte range / interval" → نطاق بايتات; "segment" → مقطع; "memory-mapping" → تعيين الملف في الذاكرة; "zero-copy" → بلا نسخ; "random access" → الوصول العشوائي; "packed back to back" → مرصوصة واحدًا تلو الآخر; "raw" → خام
- "process" (operating system) → العملية; "script" → البرنامج; "preemptible cloud instance" → مثيل سحابي قابل للانتزاع; "public hub" → مستودع عام, the Hugging Face Hub → منصة Hugging Face Hub; "the web" → الإنترنت (شبكة stays the network)
- "running moments" (Adam) → العزوم المتحركة; "step counter" → عداد الخطوات; "architecture configuration / config" → إعدادات المعمارية; "converged basin" → حوض التقارب; "toy models" → النماذج المصغرة; "standard" (format, library) → معياري (المكتبة المعيارية في Python)
- Headings and titles that begin with a Latin name get a class noun: "Pickle and safetensors" → صيغتا pickle و safetensors; "The safetensors Format" → صيغة safetensors
- English glosses given in u01 of this page (do not repeat in later units): checkpointing, state, serialization, atomic, state dictionary, bytecode, allowlist, attack surface, metadata, little-endian, cryptographic checksum

## chapter_builders-guide/gpus-devices-memory.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "device" → جهاز (أجهزة), glossed once; "device placement", "placement" → التوضيع (توضيع الموترات على الأجهزة), verb يضع / توضع; "soft placement" → التوضيع المرن; "device scope" → نطاق جهاز; "device-aware" → المدرك للأجهزة; "co-located" → مجتمعة على جهاز واحد.
- "the GPU / the CPU" → وحدة GPU / وحدة CPU (plural وحدات); "GPU $i$" → وحدة GPU رقم $i$; "GPU 0" in captions and slides → وحدة GPU 0; "host" → المضيف (ذاكرة RAM للمضيف); "onboard memory" → الذاكرة المدمجة; "card" → بطاقة.
- "operands" → أطراف العملية (الأطراف), glossed (operands) once; معامل stays the coefficient and عامل the operator.
- "fall back to CPU", "fallback" → يرجع إلى وحدة CPU، الرجوع (الاحتياطي); "degrades gracefully" → ترجع بسلاسة.
- "execution queue" → طابور التنفيذ; "command queue" → طابور الأوامر; "enqueue" → يضيف إلى الطابور (الإضافة إلى الطابور); "queued work" → العمل المصطف; "drain the queue" → تفرغ طابورها; "block" (CPU, a call) → يتوقف / يوقف البرنامج حتى …; "idle" → خامل.
- "synchronization point" → نقطة مزامنة; "synchronization bubble" → فجوة مزامنة; "asynchronous dispatch" → الإرسال غير المتزامن; "pipelining" → التتابع على هيئة خط معالجة; "pipeline" (CPU–GPU) → خط المعالجة.
- "to benchmark", "benchmarks" (timing code) → قياس الأداء، قياسات أداء; the `Benchmark` class → الصنف (المساعد) `Benchmark`; "timer" → المؤقت; "wall clock" → الزمن الفعلي; "resolution" (of a timer) → دقة زمنية.
- "milliseconds" written out → ميلي ثانية (ميلي ثوان); the symbols ms and s are copied.
- "static state" (weights plus optimizer state) → الحالة الثابتة; "optimizer moments" → عزوم المحسن; "in bf16 / fp32 / tf32" → بصيغة bf16 / fp32 / tf32; "16-bit precision" → دقة 16 بتًا; "tensor cores" → أنوية الموترات.
- "pinned (page-locked) memory" → ذاكرة مثبَّتة (pinned، أي مقفلة الصفحات), "pre-pinning" → التثبيت المسبق; "DMA engine" → محرك DMA; "no-op" → عملية بلا أثر; "inner loop" → الحلقة الداخلية; "handle" → مقبض (مقابض); "multiply-adds" → عملية ضرب وجمع.
- "matrix multiplication" → ضرب المصفوفات (glossary); "product", "matmul" → جداء (جداء المصفوفات), as on the linear-algebra page.
- Slide "3 memory views" → 3 منظورات للذاكرة (عرض stays the tensor view).
- English glosses given in u01 of this page (do not repeat in later units): device placement, asynchronous execution, activation checkpointing, device, operands, soft placement, kernel launch, synchronization point, pinned.

## chapter_builders-guide/model-construction.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "module" (the framework abstraction of this chapter: a layer, block or model) → وحدة (وحدات), glossed once as *الوحدة* (module); "module tree" → شجرة الوحدات (as on the linear-regression-concise page); "child module" → وحدة ابنة, "children" → الأبناء / الوحدات الأبناء; "parent" → الوحدة الأم; "base module (class)" → الوحدة الأساسية / صنف الوحدة الأساسي. A hidden "unit" stays وحدة too (256 وحدة ReLU); the context separates them.
- "register / registration / registered" → يسجل / التسجيل / مسجل; "registry" → سجل (سجل الوحدة، سجل الأبناء); "partly registered" → مسجل جزئيًا; "unregistered" → غير مسجل.
- "sequential container" → حاوية تسلسلية; "module container" → حاوية وحدات; "to chain" (layers, children) → يصل … في سلسلة; "straight chain" → سلسلة مستقيمة; "linear chains" → السلاسل الخطية; "insertion order" → ترتيب الإدراج.
- "forward method" → دالة التمرير الأمامي; "backward method" → دالة للتمرير الخلفي; "forward computation" → حساب أمامي; "invoke / call a module" → يستدعي; "dispatches to" → ينتقل التنفيذ إلى.
- "recursive" → عودي; "bookkeeping (problem, abstraction)" → الحصر (مشكلة الحصر، تجريد للحصر); "tree traversal" → اجتياز الشجرة; "leaf parameter" → معلَمة ورقية; tree "edge" → ضلع; "root" → الجذر; "path" → مسار.
- "serialize" (weights to disk, into checkpoints) → يكتب … على القرص / في نقاط الاستعادة (سلسلة would read as "chain"); "saving checkpoints" → حفظ نقاط الاستعادة.
- "plain / ordinary Python `list`" → قائمة `list` عادية من قوائم Python (القائمة العادية); "string index" → فهرس على هيئة سلسلة نصية; "integer index" → فهرس صحيح; "weak references" → مراجع ضعيفة; "graph-tracked container" → حاوية يتتبعها المخطط; "to track" (layers, parameters) → يتتبع; "attribute scan" → فحص السمات; "attach a layer" → إلحاق طبقة; "deferred variables" → المتغيرات المؤجلة.
- "residual MLP" → شبكة MLP ذات البواقي; "skip paths" → مسارات التخطي; "identity map / identity baseline" → الدالة المحايدة; "incremental refinements" → تحسينات تراكمية; "feature width" → عرض الميزات; "shape invariant" → لامتغير الشكل.
- "configuration object", "configs" → كائن إعدادات، الإعدادات; "lazy initialization" → التهيئة المؤجلة; "lazy shapes" → الأشكال المؤجلة; "hands-on" (section) → تطبيقي; "syntax error" → خطأ نحوي.
- Slides: "pitfall" → مزلق; "trap" → فخ; "all-or-nothing" → بمنطق الكل أو لا شيء; "lifecycle" → دورة حياة; "modules all the way down" → وحدات حتى القاع; "mode/hook toggling" → تبديل النمط والخطافات; "builders" (Keras deferred build) → دوال البناء; "wiring" → توصيل; "frozen" (weights) → مجمدة; capitalized "Dropout" as a layer kind → طبقات الحذف العشوائي. A slide title that begins with a Latin name gets a class noun (مكتبة Keras، مكتبة Flax NNX); a code name in a title gets الدالة / الصنف.
- Llama 3 8B stays Latin with the class noun نموذج where grammar needs one; Gluon, Keras, Flax NNX stay Latin.
- English glosses given in u01 of this page (do not repeat in later units): module, registered, residual connections, lazy initialization, sequential container, registry.

## chapter_builders-guide/init.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- Section heading "Initialization of Residual Networks" → تهيئة شبكات البواقي; "deep residual stack" → مجموعة البواقي المكدسة العميقة; "unscaled / scaled stack" → المجموعة غير المحجَّمة / المحجَّمة.
- "skip connection" → وصلة التخطي (glossed once here); "skip signal" → إشارة التخطي; "identity path" (figure) → المسار المحايد; "the sum" (node of the block) → نقطة الجمع.
- "well scaled" (activations) → في مقياس سليم; "signal scale" → مقياس الإشارة; "additive / multiplicative growth" → نمو جمعي / ضربي.
- "the all-zeros trap" → فخ الأصفار الكاملة; "zero-initialized branches" → الفروع المهيأة بالأصفار; "zero output projections" → إسقاطات المخرجات الصفرية; "depth engages" → يدخل العمق في العمل.
- "right-skewed distribution" → توزيع ملتوٍ نحو اليمين (الالتواء = skewness); "depth profile" → منحنى العمق; "configurable" → قابل للضبط; "broken initializations" → التهيئات المعطوبة; "forward-pass diagnostic" → تشخيص بالتمرير الأمامي.
- Fixup takes the class noun طريقة (طريقة Fixup); slide kickers "Experimental Setup" → إعداد التجربة, "Theory Meets Measurement" → النظرية تلتقي بالقياس; slide title "Summary: Parameter Initialization" → الملخص: تهيئة المعلَمات.
- English glosses given in u02 of this page: skip connection.

## chapter_builders-guide/saving-loading.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "running averages" (Adam's $\mathbf{m}$, $\mathbf{v}$) → المتوسطان المتحركان (as on the parameters page); "moments" → العزوم (two moments → عزمان); "smoothed gradient" → التدرج الممهَّد (أملس stays "smooth" for functions); "hats" → علامات القبعة; "bias-correction factors" → عوامل تصحيح الانحياز (معامل stays the coefficient)
- "stateful optimization" → التحسين ذو الحالة; "bare weights file" → ملف أوزان مجرد; "compartments" (of a checkpoint) → أقسام; "components" → مكونات; "model configuration", "config" → إعدادات النموذج، الإعدادات
- "crash-safe" → آمن من الانهيار, glossed (crash-safe) here; "job" → المهمة; "temporary file" → ملف مؤقت; "atomically rename over the destination" → يعيد تسميته ذريًا ليحل محل الملف الهدف; "rename" → إعادة تسمية; "power loss" → انقطاع التيار الكهربائي; "directory" → مجلد; "manifest" → ملف بيان, glossed (manifest); "gap" → فجوة (ثغرة stays a vulnerability)
- "dataclass" → صنف بيانات, glossed (dataclass) with المسمى before the code name; "regressor" → نموذج انحدار; "dedicated random generator seeded at 0" → مولد عشوائي خاص بذرته 0; "ground truth" (the reference run) → الحقيقة المرجعية; "uninterrupted" → غير منقطع / دون انقطاع
- "RNG state" → حالة مولد الأعداد العشوائية in body text and captions, حالة مولد RNG on slides; "replays the first minibatches" → تعيد أولى الدفعات المصغرة نفسها; "freshly constructed / fresh Adam" → محسن Adam أُنشئ من جديد / Adam جديد; "loss spike" → قفزة الخسارة; "blows the loss up a thousandfold" → تضخم الخسارة ألف مرة; "three orders of magnitude" → ثلاث مراتب مقدار; "lazily" (variable creation) → إنشاءً مؤجلًا
- English glosses given in u02 of this page (do not repeat in later units): crash-safe, manifest, dataclass

## chapter_builders-guide/parameters-state-memory.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "output head" → رأس المخرجات (then الرأس); "backbone" → الهيكل (as in the introduction); "pretrained" → مدرَّب مسبقًا; "transfer learning" → نقل التعلم.
- Adam "moments" → عزما Adam / عزوم Adam (العزم الأول، العزم الثاني); "running moments" → العزوم المتحركة.
- "allocate" → يحجز (حجز); "counting without allocating" → العد دون حجز; the verb "count" → يحصي where نعد could read as "prepare"; "storage" (of a tensor) → مساحة التخزين; "allocator fragmentation" → تجزؤ الذاكرة لدى نظام الحجز; "kernel workspaces" → مساحات العمل المؤقتة للنوى.
- "tied / untied" (parameters, model, table) → مربوط / غير مربوط; "to alias A and B to one tensor" → يجعلهما اسمين لـ … واحد; "call site" → موضع استدعاء (مواضع الاستدعاء); "deduplicate" → إزالة التكرار; "tree walkers" → أدوات اجتياز الشجرة.
- "freezing" → التجميد, "frozen" → مجمد; "to lock" (parameters, a BatchNorm layer, running statistics) → يقفل، إقفال; "pitfall" → مزلق (مزلقان).
- "training mode / evaluation mode / inference mode" → نمط التدريب / نمط التقييم / نمط الاستدلال; "stateful layers" → الطبقات ذات الحالة.
- "meta tensors", "meta device" → موترات *meta*، جهاز meta (Latin); "materialize" (a meta-device model) → تحويل … إلى نموذج فعلي; "trace abstractly" → يتتبع تتبعًا مجردًا; "master copies" → نسخ رئيسية; "serialize" → يحفظ في ملف.
- "fp32" stays Latin, after بصيغة where grammar needs a noun; "practical baseline" → خط الأساس العملي; "strict lower bound" → حد أدنى صارم; a framework "filter" object (NNX) → مرشح, "filtering the optimizer" → تصفية معلَمات المحسن.
- Slides: "8B backbone", "all 8B parameters" → الهيكل ذو 8 مليارات معلَمة، المعلَمات الـ 8 مليارات كلها; "Llama 3 8B" (a model name) is copied; "OOM" → خطأ OOM; slide answers that begin with a math span get the lead word المطلوب.
- English glosses given in u02 of this page (do not repeat in later units): activations, mixed precision, token, embedding.

## chapter_builders-guide/reproducibility-inspection.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "Deterministic Arithmetic" (heading) → الحساب الحتمي; "determinism mode" → نمط حتمية; "bitwise determinism / reproducibility" → الحتمية على مستوى البتات / قابلية إعادة الإنتاج بتًا ببت; "deterministic kernel" (GPU) → نواة حتمية (نوى حتمية); "switch" (determinism) → مفتاح.
- "atomic additions" → عمليات جمع ذرية, glossed once; "scatter-add" → الجمع المبعثر (جمع مبعثر), glossed once; "bins" (of a scatter-add) → خانات; "partial results / sums" → النتائج / المجاميع الجزئية; "reduction tree" → شجرة اختزال; "fixed-order reduction" → اختزال ثابت الترتيب; "floating-point reassociation" → تغير ترتيب تجميع الفاصلة العائمة.
- "to bisect" (the step where runs diverge) → يحدد … بالبحث التنصيفي; "bug bisection" → البحث التنصيفي عن الخلل; "diverge" (two runs) → تفترق; "software stack" → حزمة برمجيات; "hardware configuration" → تهيئة العتاد.
- "callback" → دالة رد نداء (دوال رد النداء), glossed once; "hook handle" → مقبض الخطاف (مقابض), glossed once in tab notes; "pre-hooks" → الخطافات القبلية; "execution hooks" → خطافات التنفيذ; "to attach a hook to" → يلحق خطافًا بـ; "hooks fire" → تنطلق الخطافات; "wrapper" → غلاف; "observer" → مراقب.
- "residual stack" → مجموعة مكدسة من كتل البواقي ("eight-block residual stack" → مجموعة مكدسة من ثماني كتل بواقٍ; kicker: مجموعة بواقٍ مكدسة); "head" (output layer) → رأس; "leaf layer" → طبقة ورقية; "top-level module" → وحدة في المستوى الأعلى; "submodule" → وحدة فرعية.
- "finiteness hook / check" → خطاف / فحص يتحقق من انتهاء القيم (after "non-finite" → غير منتهية); "NaN finder" → كاشف قيم NaN; "NaN traps" → مصائد NaN; "planted NaN" → قيمة NaN مزروعة; "symptom" → عرَض; "culprit" → الجاني.
- "autograd graph", "backward computation graph" → مخطط الاشتقاق التلقائي، مخطط الحساب الخلفي (as المخطط الحسابي); "leak-free" (hooks) → بلا تسرب / تحفظ من تسرب الذاكرة; "out-of-memory (OOM) crash" → انهيار بسبب نفاد الذاكرة (OOM).
- "activation profile" → منحنى (نمو) التفعيلات; "profiling" in slide titles → قياس; "variance-growth analysis" → تحليل نمو التباين; "scaled-down output projections" → إسقاطات مخرجات مصغَّرة المقياس; "branch gradient / Jacobian" → تدرج الفرع / المصفوفة اليعقوبية للفرع; "typical ranges" → مجالات نموذجية.
- "resumable checkpoints" → نقاط الاستعادة القابلة للاستئناف; "crash-safe checkpoints" → نقاط استعادة آمنة من الانهيار; "engineering toolkit" → العدة الهندسية; "job" (training) → مهمة.
- English glosses given in u02 of this page (do not repeat in later units): atomic additions, scatter-add, callback, handle.

## chapter_builders-guide/gpus-devices-memory.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "caching allocator" → نظام حجز ذو تخزين مؤقت (as on the chapter index), then نظام الحجز; "slab" → لوح (ألواح); "reserved pool" → المخزون المحجوز (التجميع stays pooling); "live tensors / bytes" → الموترات الحية / البايتات الحية; "CUDA driver" → برنامج تشغيل CUDA; "layer of the stack" (software) → طبقة من طبقات البرمجيات.
- "peak memory" → ذروة الذاكرة; "out-of-memory (OOM)" → نفاد الذاكرة, slide "Activation OOMs" → نفاد الذاكرة بسبب التفعيلات; "step memory" → ذاكرة الخطوة; "activation footprint" → بصمة التفعيلات; "lever" → وسيلة; "memory leak" → تسرب; "computation graphs" → مخططات حسابية.
- "activation checkpointing" → نقاط استعادة التفعيلات (also as the subject of a verb, feminine plural agreement in the singular); "gradient checkpointing" → نقاط استعادة التدرج; "rematerialization" → إعادة التكوين, "rematerializes" → يعيد تكوين; "segment" → مقطع (مقاطع), "segment boundaries" → حدود المقاطع, "active segment" → المقطع النشط; "interior activations" → التفعيلات الداخلية; "checkpoint wrapper" → مغلِّف نقاط الاستعادة.
- "deep stack of N blocks" → مجموعة مكدسة من N كتلة; "group blocks into segments" → يضم الكتل في مقاطع; "standard backpropagation" → الانتشار العكسي المعياري; "forward-pass equivalents" → مكافئات للتمرير الأمامي; "back-of-the-envelope bound" → الحد التقريبي السريع; "correctness check" → فحص الصحة.
- "hybridized graphs" (MXNet) → المخططات المهجَّنة; "profilers" → أدوات تحليل الأداء; "temporary workspace" → مساحة العمل المؤقتة; "process-wide flag" → راية على مستوى العملية كلها; "RNG state" → حالة مولد الأعداد العشوائية.
- Slides: "head-only (fine-tuning)" → (الضبط الدقيق) للرأس وحده; "Dissecting memory" → تشريح الذاكرة; "Measured:" → بالقياس:; "physical minibatch size" → الحجم الفعلي للدفعة المصغرة; "effective step" → الخطوة الفعالة; "half-batches" → أنصاف الدفعات; the 8B model on slides → نموذج بـ 8 مليارات معلَمة.
- English glosses given in u02 of this page (do not repeat in later units): caching allocator, slabs, out-of-memory (OOM), gradient checkpointing, rematerialization, step memory.

## chapter_builders-guide/model-construction.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "Forward Computation" (heading) → الحساب الأمامي; "Eager and Compiled Execution" → التنفيذ الفوري والتنفيذ المصرَّف; "compiled execution / graph / step" → التنفيذ المصرَّف / المخطط المصرَّف / الخطوات المصرَّفة; "eagerly" → فوريًا.
- "to trace" (a computation, for compilation) → يتتبع, "tracing" → التتبع, "trace time" → وقت التتبع; "tracer" (JAX) → متتبِّع (متتبِّعات), glossed (tracers) in the JAX tab note; "symbolic tensor" → موتر رمزي; "concrete values" → قيم فعلية; "symbolic control flow" → تدفق تحكم رمزي; "data-dependent" → المعتمد على البيانات; "unroll" (a static loop) → تنفرد.
- "dry run" → تشغيل تجريبي, glossed (dry run) once; "dummy minibatch / data / forward pass" → دفعة مصغرة وهمية / بيانات وهمية / تمرير أمامي وهمي; "materialize" (shapes, weights) → تتحدد الأشكال، تنشأ الأوزان، تحديد الأشكال; "parameter placeholders" → عناصر نائبة للمعلَمات; "cascade shapes" → انتقال الأشكال تتابعًا; "recorded up front" → يُسجل مسبقًا.
- "initializer callback" → دالة رد نداء للتهيئة; "Xavier's uniform initializer" → مهيئ Xavier المنتظم; "variance-scaling initializers" → مهيئات تحجيم التباين.
- "Models from a Config" → النماذج من الإعدادات; "config" → الإعدادات (grammatically a feminine plural used for one object: إعدادات واحدة، الإعدادات الأكبر), "configuration object (config)" → كائن *إعدادات* مخصص (*config*); "config field" → حقل في الإعدادات; "configuration file" → ملف إعدادات; "builder code" → شيفرة البناء; "factory function" → دالة مصنع (دالة المصنع `build`).
- "topology" → الطوبولوجيا; "hardcoded" → مثبت مباشرةً في; "output head" → رأس المخرجات; "input projection" → إسقاط المدخلات; "implementation code" → شيفرة التنفيذ; "version control" → نظام إدارة الإصدارات; "rapid prototyping" → بناء النماذج الأولية بسرعة; "production models / code" → نماذج / شيفرة بيئة الإنتاج; "programmatically" → برمجيًا; "convention" → العرف.
- "serializable" (`@dataclass`) → قابل للحفظ (following this page's rule of no سلسلة-word for "serialize").
- "buffers" on this page → مخازن الحالة (buffers), as in the section it points to (:numref:`subsec_params_buffers`).
- Slides: "Forward Is Just Python" → التمرير الأمامي شيفرة Python فحسب; "Graph Compilation Limits" → حدود تصريف المخطط الحسابي; "Forward Execution Modes" → أنماط تنفيذ التمرير الأمامي; "Deferred Shape Inference" → الاستنتاج المؤجل للأشكال; "Eager Parameter Allocation" → الحجز الفوري للمعلَمات; "Dataclass Configurations" → إعدادات بأصناف dataclass; "Analytical vs. Tree-Walk Counts" → العد التحليلي مقابل العد باجتياز الشجرة; "Parameter Scaling" → تحجيم عدد المعلَمات; table heads "Module Component / Shape Formula / Parameter Count" → مكون الوحدة / صيغة الشكل / عدد المعلَمات, "Concept / Mechanism / Rule / Practical Systems Implication" → المفهوم / الآلية / القاعدة / الأثر العملي في الأنظمة.
- English glosses given in u02 of this page (do not repeat in later units): body, buffers, compiled execution, eagerly / eager execution, tracers, dry run, config.

## chapter_builders-guide/init.md#u03 (claude fable-5.2 (medium), 2026-11-27)

- "callback" → دالة رد النداء (دوال رد النداء); "tree-wide callbacks" → دوال رد النداء على الشجرة كلها; "constructor arguments" → وسائط المُنشئ.
- "elementary closed form" → صيغة مغلقة بدوال أولية; "standard normal density and distribution function" → دالة الكثافة ودالة التوزيع للتوزيع الطبيعي المعياري; "tabulated values" → القيم المجدولة; "plain normal draw" → السحبة الطبيعية غير المبتورة; "the cut discards" → يستبعدها البتر.
- "normalized inputs / unnormalized blocks" → المدخلات المطبَّعة / الكتل غير المطبَّعة; "measured plot / curve" → الرسم المقيس / منحنى مقيس; "scaled normal initializer" → المهيئ الطبيعي المحجَّم; "uniform initializer" → المهيئ المنتظم; "range" (of weight values) → المدى.
- "Kaiming initialization" → تهيئة Kaiming (name stays Latin, like He); "transposed Jacobian" → منقول المصفوفة اليعقوبية; "argument" (a derivation) → حجة, (of a function) → وسيط.
- "data-driven calibration" → المعايرة المعتمدة على البيانات; "calibrate block by block" → يعاير كتلة بكتلة; "unit-variance inputs" → مدخلات تباينها واحد; "layer-sequential unit-variance initialization" → التهيئة المتتابعة على الطبقات بتباين مقداره واحد.
- Credit line: *مقتبس بتصرف من مقرر CMU 10-714، الواجب [Homework 2](url) والواجب [Homework 4](url).* (the class noun الواجب before each Latin link text).
- English glosses in u03 of this page: none (summary and exercises).

## chapter_builders-guide/custom-layers.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "buffer" on this page stays مخزن حالة (as in u01 of the page), glossed (buffers) once at "register them as buffers"; "fixed-state buffers" → مخازن الحالة الثابتة.
- "random features" → *الميزات العشوائية* (random features), glossed here; "random phases" → أطوار عشوائية; "evenly spaced points" → نقاط موزعة على مسافات متساوية; "lifetime of the model" → عمر النموذج.
- "staircase" (the function round) → الدالة السلمية; "step function" stays الدالة الدرجية; "piecewise constant" → ثابتة قطعيًا; "half-integers" → أنصاف الأعداد الصحيحة.
- "surrogate" (derivative, gradient, backward pass) → بديل (مشتق بديل، تدرج بديل، البدائل); "identity surrogate" → بديل محايد; "non-identity surrogates" → البدائل غير المحايدة; "the identity (function / map)" → الدالة المحايدة.
- "stop-gradient operator / operation" → مؤثر / عملية *إيقاف التدرج* (stop-gradient), glossed here; "gradient check" → *فحص التدرجات* (gradient check), glossed here; "finite-difference slope" → الميل المحسوب بالفروق المحدودة.
- "static method" → دالة ساكنة (method → دالة); "closure" → دالة مغلقة; "decorated function" → الدالة المزخرفة; "one-element tuple" → مجموعة مرتبة من عنصر واحد; "the tape" (TensorFlow) → الشريط.
- "quantizer" → مكمِّم (مكمِّمات); "clipped quantizers" → المكمِّمات ذات القص; "clamp" → يحصر، الحصر; "representable range" → المدى القابل للتمثيل; "saturated exterior" → المنطقة الخارجية المشبعة; "4-bit signed range" → مدى الأعداد ذات الإشارة من 4 بتات; "quantization-aware" → المدرك للتكميم.
- "rounded true weights" → الأوزان الحقيقية بعد تدويرها; "reference point" → نقطة مرجعية; "constant zero baseline" → نموذج مرجعي ثابت يتنبأ بالصفر; "explode" (MSE) → ينفجر; "MSE" on slides and after the full term → خطأ MSE.
- "custom GPU kernels" → نوى GPU مخصصة; "tensor primitives" → عمليات أولية على الموترات; "systems discipline" → تخصص في هندسة الأنظمة; "output head" → رأس المخرجات; "the CPU" → وحدة CPU.
- Slide lead-in "Answer these … questions:" → المطلوب: الإجابة عن …; "Summary: …" → الملخص: …; "Key Takeaways" → أهم الخلاصات; "Pitfall:" → مزلق:; "8B" on slides → 8 مليارات (معلَمة); only the model name Llama 3 8B keeps the Latin "8B".
- English glosses given in u02 of this page (do not repeat in later units): random features, buffers, stop-gradient, gradient check.

## chapter_builders-guide/numerics.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "operands" → أطراف (أطراف العملية); not معاملات, which is the coefficient word. "mismatched operands" → أطراف غير متطابقة الأنواع; "promotes to" → يرقّي إلى / يُرقّى إلى (with the shadda; يرقى is "rises"); "stray" (tensor, buffer) → شارد.
- "master weights" → أوزان رئيسية, glossed (master weights) in this unit; "underflow" (verb) → تطفح سفليًا, glossed (underflow) here; "dynamic loss scaling" glossed (dynamic loss scaling) here. Do not repeat these glosses in later units of the page.
- "block tree" (Gluon) → شجرة الكتل; "global policy" (Keras) → السياسة العامة; "eligible ops" → العمليات المؤهلة; "context manager" → مدير سياق, `torch.autocast` / `autocast` take the class noun السياق; "autocast" as a common noun (kicker) → التحويل التلقائي للنوع.
- "scaled up / scaled back down" (loss, gradients) → تُكبَّر بالتحجيم / تعاد إلى مقياسها الأصلي; "unscale" (gradients) → يلغي تحجيم; "loss scaler" → أداة لتحجيم الخسارة; "scale factor" → معامل تحجيم; "streak of clean steps" → سلسلة من الخطوات السليمة; "skip" (the optimizer step) → يتخطى.
- "enable" (a mode, loss scaling) → يشغّل (not يفعّل, to keep it apart from التفعيل "activation"); "tf32 mode" → نمط tf32; "strict fp32" → fp32 الصارمة; "truncate" (inputs) → يقتطع; "native bf16 support" → الدعم الأصلي للصيغة bf16; "server CPUs" → وحدات CPU في الخوادم.
- "update mass / magnitude" → المقدار الكلي للتحديث, with the printout word “mass” kept in Latin; "share" → حصة; "variants" (of a run) → الحالات; "apples-to-apples comparison" → المقارنة المتكافئة; "trained to convergence" → دُرّب حتى التقارب; "lags behind / trails" → تتأخر / تتخلف.
- "caveats" (running text) → تنبيهان; "underflow zone" → منطقة الطفح السفلي; "subnormal or zero regimes" → نطاق الأعداد دون النظامية / نطاق الصفر; "attenuate" (gradients across layers) → تضعف; "linearity of differentiation" → خطية الاشتقاق; "brittle" → هش; "task accuracy" → الدقة في المهمة; "weight footprint" → بصمة الأوزان في الذاكرة.
- Slides: "Pure bf16 Weights Stall" → أوزان bf16 الخالصة تتعثر; "non-negotiable" → لا غنى عن; "Where mixed precision wins" → موضع تفوق الدقة المختلطة; kicker with two Latin names → [الصنفان GradScaler و LossScaleOptimizer].

## chapter_builders-guide/parameters-state-memory.md#u03 (claude fable-5.2 (medium), 2026-11-27)

- "helper" (a small function the reader writes) → دالة مساعدة; "state entries" (scalar elements of the state) → عناصر الحالة, "bytes per entry" → البايتات لكل عنصر; "top-level block" → كتلة من المستوى الأعلى; "share of the total" → الحصة من مجموع …
- "can the head still adapt?" → هل ما زال الرأس قادرًا على التكيف؟; "leave `x` alone" → اترك `x` على حاله; "does the state still fit?" → هل ما زال للحالة متسع؟
- Exercises "Find a loss …" → أوجد خسارة …; "read off the value" → استخرج قيمة; "Suppose you …" in an exercise → افترض أنك ….

## chapter_builders-guide/reproducibility-inspection.md#u03 (claude fable-5.2 (medium), 2026-11-27)

- Exercise titles: "Residual variance budget" → ميزانية التباين في كتل البواقي; "Seeds and data loading" → البذور وتحميل البيانات; "Divergence monitor" → راصد التباعد ("monitor" as a tool → راصد, since مراقب is the hook "observer"; the verb "to monitor" → يراقب); "Gradient norms" → معايير التدرجات.
- "the stream" (the residual stream that each block adds its body's output to) → مجرى الإشارة (سيل stays the stream of random numbers); "the term this equation omits" → الحد الذي تغفله هذه المعادلة.
- "the pre-hook receives the argument tuple, so unpack it" → يتلقى الخطاف القبلي الوسائط في مجموعة مرتبة، ففكها ("unpack" → يفك); "stash" (a value inside `call`) → يحتفظ بـ; "freshly built model" → نموذج مبني من جديد.
- "shuffle order" → ترتيب الخلط; "base seed" → البذرة الأساسية, "base key" → مفتاح أساسي; the worker "initializer" (`worker_init_fn`) → دالة التهيئة (مهيئ stays the weight initializer); "seeding" → ضبط البذور; "standalone script" → برنامج مستقل; "dataset class" → صنف مجموعة بيانات; "dataset's constructor" → مُنشئ مجموعة البيانات.
- JAX: "fold X into a key" (`fold_in`) → يدمج X في المفتاح (دمج); "per-worker keys" → مفاتيح للعمال; "shard" → حصة (حصص); "derivation" (of keys) → اشتقاق; "four-worker input pipeline" → مسار معالجة مدخلات بأربعة عمال (glossary); a `tf.data` "pipeline" → خط معالجة `tf.data`; "element" (of a `tf.data` dataset) → عنصر.
- "families of norms" (output-gradient norms and parameter-gradient norms) → عائلتا المعايير; "plot against depth" → يرسم مقابل العمق; "depth profiles" → منحنيا العمق; "exploding backward signal" → إشارة خلفية منفجرة; "express A through B" → يعبّر عن A بدلالة B.
- "Adapted from the PyTorch documentation [title](url)" → مقتبس بتصرف من وثائق PyTorch، صفحة [title](url), the page title in English.
- "non-finite" keeps غير منتهية; "by depth" → بحسب العمق; "standard normal targets / inputs" → أهداف / مدخلات تتبع التوزيع الطبيعي المعياري.

## chapter_builders-guide/custom-layers.md#u03 (claude fable-5.2 (medium), 2026-11-27)

- "fold a bias into another" (absorb $\mathbf{b}$ into $\mathbf{c}$) → يضم إلى (ضمه إلى); "the folded bias" → الانحياز الناتج عن الضم (not يطوي, which is folding the input space, and not يدمج, which is "fuse").
- "entry" (of a saved file / state dict / weight list) → بند (يزيد بالبند المتوقع); "parameter state" (JAX) → حالة المعلَمات; "state dict" → قاموس الحالة (as on the saving-loading page).
- "strictly inside" (a range) → داخل … تمامًا; "exactly on the boundary" → على حده بالضبط; "convention" (at a boundary) → عرف; "clamp range" → مدى الحصر.
- "uninformative" (derivative) → لا يفيدنا بشيء; "stop-gradient shortcut" → اختصاره بإيقاف التدرج; "fused-kernel speed" → سرعة النوى المدمجة; "specification" → مواصفات.
- "save `X` as a residual" (JAX custom_vjp) → يحفظ الموتر `X` ضمن البواقي; "gradient closure" → الدالة المغلقة التي تحسب التدرج.
- "Adapted from the PyTorch tutorial [title](…) and the JAX notebook [title](…)" → مقتبس بتصرف من دليل PyTorch التعليمي [title](…) ومن دفتر JAX المعنون [title](…), titles kept in English.
- Exercise title "Random features approximate a kernel" → الميزات العشوائية تقرّب دالة نواة (kernel-methods sense: دالة النواة).
- English glosses on this page: none added in u03 (summary and exercises).

## chapter_builders-guide/model-construction.md#u03 (claude fable-5.2 (medium), 2026-11-27)

- "to log" (a config, an architecture) in a sentence that also speaks of registration → يدوّن / تدوين (u02 wrote يمكن تسجيله where no registered module was near); "easy to log, scale, and reproduce" → يجعل تدوين المعماريات وتحجيمها وإعادة إنتاجها أمرًا سهلًا.
- Exercise titles: "Unregistered modules / containers" → الوحدات / الحاويات غير المسجلة; "Widening a residual block" → توسيع كتلة البواقي; "A module from scratch" → وحدة من الصفر; "Lazy input width" → عرض المدخلات المؤجل; "Shared children" → الأبناء المشتركون.
- "Predict, then verify, what …" → توقع ما …، ثم تحقق من توقعك; "Predict, then verify:" before a question → توقع ثم تحقق:; "a fresh `X`" (newly constructed model) → نموذج `X` جديد.
- "silent failure / raises an error" → إخفاق صامت / يثير خطأ; "from the same missing registration" → بغياب التسجيل نفسه; "repairs" (named by an error message) → حلول, as in u01.
- "method" of a class → دالة (as دالة التمرير الأمامي); "is the inverse of" (load/save) → هي معكوس; "base class" → صنف أساسي; "subclass" → صنف فرعي; "instance" → مثيل; "minimal" (class) → في أبسط صورة; "recurses into" → تنزل عوديًا إلى; "wrapped in" → مغلف بـ.
- "site" (where a child module is used) → موضع (مواضع); the module tree turned "graph" → مخطط; "static graph definition" → تعريف مخطط ثابت; "round trip" (`nnx.split` then `nnx.merge`) → دورة فصل ودمج; the operator `is` → العامل `is`.
- "skip path" → مسار التخطي; "identity path" → المسار المحايد; "output coordinates" → إحداثيات المخرج; "a $d'$-wide layer" → طبقة عرضها $d'$; "sharing every layer after the first" → تكون كل طبقاته بعد الأولى مشتركة.
- "Adapted from CMU 10-714, [Homework 2](…), Question 2." → مقتبس بتصرف من مقرر CMU 10-714، الواجب [Homework 2](…)، السؤال 2. (as on the chapter index page).

## chapter_builders-guide/saving-loading.md#u03 (claude fable-5.2 (medium), 2026-11-27)

- "backbone" → الهيكل; classifier / classification "head" → رأس المصنف / رأس التصنيف, then الرأس (glossed (head) once here); "stage" (of a ResNet backbone) → مرحلة
- "mismatch" (file vs model) → عدم تطابق (حالات عدم التطابق); "mismatched checkpoint" → نقطة استعادة غير مطابقة; "audit" (keys, a partial load) → يدقق / تدقيق; "careful loader" → دالة تحميل حريصة; "partial loading / transfer" → التحميل الجزئي / النقل الجزئي; "non-strictly" → تحميلًا غير صارم
- "red flag" → إنذار خطر; "production code" → شيفرة الإنتاج; "assert explicitly" → تحقق تحققًا صريحًا; "raise an error" → يثير خطأ; "unfreeze" → يلغي تجميد
- "length prefix" → بادئة طول; "header" (safetensors) → الترويسة; "file seeks" → تحريك موضع القراءة في الملف; "decode the bytes" → فك ترميز البايتات; "deserialize" → فك التسلسيل; "object graph" → مخطط الكائنات
- "key-set merge" → دمج لمجموعات المفاتيح; "assign" (a variable) → إسناد قيمة إلى; "file entries" → عناصر الملف; "truncation" (of a file) → بتر; "publish" (a checkpoint via its manifest) → ينشر; "directory entries" → عناصر مجلد; "buffered writes" → الكتابات المخزنة مؤقتًا
- "Dtype round trip" (exercise title) → حفظ نوع البيانات واسترجاعه; "Adapted from <author>, [tutorial], "section"" → مقتبس بتصرف من درس [title] الذي أعده <author>، قسم “section”
- Slide "for 8B" → لنموذج بـ 8 مليارات معلَمة (no Latin "8B" next to B = bytes, as on the parameters page); slide answers that begin with a math span get a class noun (الحجم، المعيار)
- English glosses given in u03 of this page: head, transfer learning

## chapter_builders-guide/gpus-devices-memory.md#u03 (claude fable-5.2 (medium), 2026-11-27)

- Heading and kicker "(The) Device-Aware Trainer" → الصنف Trainer المدرك للأجهزة (class noun first, name in Latin, as in the page introduction); slide title "Wiring … Into d2l.Trainer" → وصل … بالصنف d2l.Trainer; "to wire X into Y" → يصل X بـ Y.
- "reality check" → تذكير بالواقع; "speedup", "gain" (of speed) → تسريع (كسب stays the gain of a layer); "underutilized" (GPU) → يبقى معظم قدرتها معطلًا; "GPU utilization" → استغلال وحدة GPU; "workload" → حجم العمل; "pay off" → تؤتي ثمارها.
- "data loading" → تحميل البيانات; "Python dispatch" → الإرسال في Python; "kernel-launch overhead / latency" → العبء الإضافي لإطلاق النواة / زمن استجابة إطلاق النواة; "post-warmup epoch" → حقبة بعد الإحماء.
- "CPU-only machines" → الحواسيب التي ليس فيها إلا وحدة CPU (جهاز stays "device"); "gigabytes to spare" → غيغابايتات فائضة; "memory budgeting" → حساب ميزانية الذاكرة.
- Exercises: "schedule" (of checkpointing) → خطة; "round trip" (CPU to GPU and back) → رحلة الذهاب والإياب; "extrapolate" → يستكمل خارج المدى; "crossover" → نقطة التقاطع; "host-to-device bandwidth" → عرض نطاق النقل من المضيف إلى الجهاز; "scaling" (with the number of GPUs) → تناسب الأداء مع عدد الأجهزة, "linear scaling" → التناسب الخطي; "instrumentation artifact" → أثر جانبي لأدوات القياس; "readout" → القراءة; "guard with" → احمِ … بالشرط.
- Slides: "systems intuition" → الحدس في الأنظمة; "scenarios" → حالات; "allocated-memory counter" → عداد الذاكرة المخصصة للموترات; "live allocated bytes" → البايتات الحية المخصصة; "host synchronization stalls" → توقفات المزامنة مع المضيف; "Fix:" → الإصلاح:.

## chapter_builders-guide/numerics.md#u03 (claude fable-5.2 (medium), 2026-11-27)

- "failure mode" (numerical) → نمط إخفاق (أنماط الإخفاق العددي); "numerical bugs" → مواطن الخلل العددية; "root cause" → السبب الجذري; "indeterminate arithmetic" (`inf - inf`) → العمليات الحسابية غير المعيَّنة; "poisoned" (steps, Adam) → تتسمم / يسمم; "value ranges" → مجالات القيم.
- "accumulator" → مركم (مراكم), glossed (accumulator) in u03; "to accumulate (a sum) in fp32" → يراكم بالصيغة fp32, heading "Accumulation in fp32" → المراكمة بالصيغة fp32; "running total / running sum" → المجموع التراكمي; "summand" → حد مضاف; "stall" (a sum) → يتوقف; "freezes at 2" → يتجمد عند 2.
- "round-to-nearest-even" → التدوير إلى الأقرب الزوجي, glossed (round-to-nearest-even); "midpoint" → نقطة المنتصف; "exact / exactly represented" → مضبوط / ممثَّل تمامًا.
- "shifted log-sum-exp" → لوغاريتم مجموع الأسيات المزاح, glossed (log-sum-exp) once in the summary; "unshifted exponential" → دالة أسية غير مزاحة; "exponentiating" → حساب الدالة الأسية; "nonpositive" → غير موجب; log-softmax stays Latin after دوال.
- "exponent bias" (fp8 exercise) → انحياز الأس (4 بتات للأس بانحياز 7); "all-ones exponent" → الأس الذي كل بتاته آحاد; "IEEE convention" → عرف IEEE; "largest finite value" → أكبر قيمة منتهية; exercise title "The e4m3fn format" → الصيغة e4m3fn.
- "unscaled gradients" (after dividing by $S$) → التدرجات بعد إلغاء التحجيم; "the scale" (loss scaling) → معامل التحجيم; "replace X by zeros" → وضع أصفار مكان X; "persistent training state" → حالة التدريب الدائمة.
- Slides: "Guarding 16-Bit Range" → حماية مدى 16 بتًا; "Stable reductions" → اختزالات مستقرة; "Format trade-offs" → مقايضات الصيغ; "running-sum stagnation" → توقف المجاميع التراكمية.
- English glosses given in u03 of this page: accumulator, round-to-nearest-even, log-sum-exp.

## chapter_builders-guide: chapter check (claude fable-5.2 (medium), 2026-11-27)
- One rendering per term across the chapter's nine pages; the entries above were corrected in place.
- "buffer" in the framework sense (a registered, persistent, non-trainable tensor) → مخزن حالة (مخازن الحالة; dual مخزني حالة) on every page; مخزن مؤقت stays the memory buffer (raw tensor buffers of a file, buffered writes).
- "crash" (of a process, a job, a training run) → انهيار, "crash-safe" → آمن من الانهيار; عطل (أعطال العتاد، تحمل الأعطال) stays "failure" and "fault".
- A parameter count written "8B" in English → 8 مليارات (معلَمة) in body text and on slides (B would read as bytes next to GB); only the model name Llama 3 8B keeps the Latin "8B".
- "running" (averages, moments, statistics of Adam and BatchNorm) → متحرك (العزوم المتحركة); تراكمي stays "running sum".
- "script" → برنامج (برنامج تدريب، برنامج مستقل); "raise an error" → يثير خطأ; "mode" → نمط (نمط التدريب، نمط tf32، نمط حتمية); "enable", "turn on" → يشغّل، تشغيل (not تفعيل, which is "activation").
- "placement" on a device → التوضيع; "allocator" → نظام الحجز ("caching allocator" → نظام حجز ذو تخزين مؤقت); "non-invasive" hooks → خطافات لا تمس الشيفرة.
- Kept apart on purpose: "serialization" as the topic of the saving-loading page → التسلسيل (يسلسل، فك التسلسيل), glossed there; on the other pages "serialize into checkpoints / to disk" is written with يحفظ / يكتب, and "serializable" → قابل للحفظ.

## chapter_convolutional-neural-networks/index.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "Convolutional Neural Networks" (chapter title) → الشبكات العصبية الالتفافية; gloss (convolutional neural network, CNN) once, then شبكة CNN / شبكات CNN; "convnet", "ConvNets" → شبكة التفافية (الشبكات الالتفافية).
- "locality" → المحلية; "translation equivariance" → تساوق الإزاحة (glossary); "spatial geometry" → الهندسة المكانية; "pixel grid" → الشبكة النقطية من البكسلات; "neighborhood" → جوار.
- "bank of filters", "filter bank" → مجموعة مرشحات (مجموعات المرشحات); "filter" → مرشح (مرشحات), "filtered responses" → الاستجابات بعد الترشيح; "sweep / slide a filter across" → يمرر على; the kernel "slides" → تزلق.
- "oriented edges" → حواف موجهة; "bars" → أشرطة; "outline" (of an object) → المحيط الخارجي; "contours" → الحدود; "textures" (of an object) → نسيج السطح, (image features) → أنماط النسيج on every page of the chapter; "pure static" (TV noise) → تشويش محض; "scrambled images" → الصور المبعثرة; "pixel-shuffled images" → صور خُلطت بكسلاتها.
- "early visual cortex" → المراحل الأولى من القشرة البصرية; "visual field" → المجال البصري; "visual pathway" → المسار البصري; "primates" → الرئيسيات; "neuroscience" → علم الأعصاب; "silicon computation / silicon vision" → الحوسبة على السيليكون / رؤية السيليكون.
- "field of view" → مجال الرؤية; "feature capacity" → سعة الميزات; "$1 \times 1$ bottlenecks" → أعناق الزجاجة $1 \times 1$; "local and global pooling" → التجميع المحلي والتجميع الشامل; "stem" (first convolutional layer) → جذع; "reach" (of stacked layers) → مدى; "shift tolerance" → التسامح مع الإزاحة (as on the pooling page); "handwritten ZIP codes" → الرموز البريدية المكتوبة بخط اليد.
- "geometric prior" → الافتراض القبلي الهندسي; "permutation blindness" → العمى عن التبديل; "megapixel" → ميغابكسل; "parameter-free" → بلا معلَمات.
- Resource lists: "(paywalled)" → (الوصول مدفوع); "free HTML" → نسخة HTML مجانية; "free notes" → ملاحظات مجانية; "Chapter 9 …" of a cited book → الفصل 9 منه; "animations" → رسوم متحركة; an interactive web tool → class noun أداة; "parameter sharing" (Goodfellow's term) → مشاركة المعلَمات, "weight sharing" → مشاركة الأوزان.
- Slide labels: "Chapter Architecture" → معمارية الفصل; "Mathematical and Architectural Roadmap of Chapter N" → خارطة الطريق الرياضية والمعمارية للفصل N; "Concrete Benchmarks Across Chapter N" → اختبارات معيارية محددة على امتداد الفصل N; "Chapter Study Strategy" → استراتيجية دراسة الفصل; "resource split" → توزيع الموارد.
- English glosses given on this page: receptive field, convolutional layer, translation equivariance, convolutional neural network (CNN).

## chapter_convolutional-neural-networks/padding-and-strides.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- Page title → الحشو وخطوة الانزلاق والتوسيع; "zero padding" → حشو الأصفار; "padding values" → قيم الحشو; "unpadded" → غير محشو, "padded input / tensor" → مدخلات محشوة / موتر محشو; "shape-preserving padding" → الحشو الحافظ للشكل; "symmetric / asymmetric padding" → حشو متناظر / غير متناظر; "same" padding on slides → الحشو “same”.
- "reflect padding" → حشو الانعكاس; "replicate padding" → حشو التكرار; "boundary condition" → شرط حدي (glossary); "boundary artifacts" → الآثار المصطنعة عند الحدود; "blind spots" → بقع عمياء; "positional cue" → إشارة إلى الموضع; "border darkening" → تعتيم الحدود; "dark frame" → إطار داكن.
- Pixels by position: "border" (of an image) → الحدود، الحد; "boundary pixels" → بكسلات الحدود; "interior pixel" → بكسل داخلي; "edge pixel" (on the border, not an intensity edge) → بكسل طرفي; "corner pixel" → بكسل ركني; "perimeter" → محيط الصورة.
- "strided convolution" → الالتفاف بخطوة انزلاق; "stride 1" in prose → خطوة انزلاق مقدارها 1; "vertical / horizontal stride" → خطوة الانزلاق العمودية / الأفقية; "dilation" → التوسيع (glossary), so "expand" (a grid by padding) is rendered with يكبّر / تصير, never يوسع.
- "field of view" → مجال رؤية; "dense prediction" → التنبؤ الكثيف; "footprint" (of a kernel) → مساحة التغطية; "geometric controls" → ضوابط هندسية; "kernel taps" → أوزان النواة; "fast kernels" (compute kernels) → النوى الحسابية السريعة; "restoration networks" → شبكات ترميم الصور; "grid" (input / output) → شبكة نقطية; "odd / even kernel" → نواة فردية / زوجية.
- "singleton dimensions" → بُعدان مفردان; "channels-last" → تخطيط القنوات أخيرًا (as on the image-classification page); "per-axis tuple" → مجموعة مرتبة لكل محور.
- English glosses given in u01 of this page (do not repeat in later units): receptive field, semantic segmentation, padding, stride, dilation, boundary condition, reflect padding, replicate padding.

## chapter_convolutional-neural-networks/pooling.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- Pooling terms: "pooling" → التجميع; "max-pooling", "maximum pooling" → التجميع الأقصى (one Arabic term for both; the first mention reads (maximum pooling، واختصاره *max-pooling*)); "average pooling" → تجميع المتوسط; "global pooling" → التجميع الشامل; "global average pooling" → تجميع متوسط شامل (glossary form, used indefinite); "global maximum / global average" of a feature map → القيمة العظمى الشاملة / المتوسط الشامل; "pooled response / output" → الاستجابة المجمَّعة / مخرجات التجميع.
- "tolerance" to shifts → التسامح مع الإزاحة, "tolerant" → متسامح مع (هامش السماح stays the pass/fail tolerance, تحمل الأعطال fault tolerance); "invariant to shifts" → ثابت تحت الإزاحة, "strictly invariant" → ثابت تمامًا, "exact invariance" → ثبات مضبوط; "shift-equivariant" → متساوق مع الإزاحة (glossary تساوق الإزاحة).
- "block" (pooling block) → كتلة (كتل); "block boundary" → حد الكتلة; "response" (of a detector) → استجابة; "peak (response)" → ذروة (الاستجابة); "fires" (a detector, a cell) → ينشط; "neighborhood" → جوار; "winning element" → العنصر الفائز; "impulse" → نبضة; "amplitude" → سعة.
- "absorb" (a shift) → يمتص; "collapse" (spatial coordinates, a channel to a number) → يطوي، طي (الانهيار stays the linear collapse of stacked layers; اختزال stays "reduction"); "dilute" → يخفف; "spatial coverage" → التغطية المكانية; "spatial reach" → المدى المكاني; "spatial extent" → الامتداد المكاني.
- "aggregation" (spatial aggregation, evidence aggregation) → الإجمال, as on the ndarray slides, so that التجميع names only pooling.
- "subsampling" → أخذ العينات الفرعية; "naive subsampling" → أخذ العينات الفرعية الساذج; "downsampling" → خفض أخذ العينات, "downsampler" → طبقة تنفذ خفض أخذ العينات; "sampling rate" → معدل أخذ العينات; "smoothing filter" → مرشح تنعيم; "low-pass box filtering" → الترشيح الصندوقي منخفض التمرير; "high-frequency alternating signal" → إشارة متناوبة عالية التردد; "signal processing" → معالجة الإشارات.
- "stride 2", "stride-2 X" → X بخطوة انزلاق 2; "strided layers" → الطبقات ذات خطوة الانزلاق الأكبر من واحد; "product of the strides" → جداء خطوات الانزلاق; "stride-1 convolutional stacks" → الطبقات الالتفافية المكدسة بخطوة انزلاق 1.
- "border pixel" → البكسل الطرفي; "corner pixel" → البكسل الركني (as on the padding page); "interior pixel" → البكسل الداخلي; "padded border" → الحد المحشو; "zero-padded" → محشو بالأصفار; "image borders / boundaries" → حدود الصورة (حافة stays the intensity edge); "salient content" → المحتوى البارز.
- Names: Neocognitron, Fukushima, Waldo, Nyquist, BlurPool, AvgPool, LeNet stay Latin (نموذج Neocognitron الذي وضعه Fukushima); "computational neuroscience" → علم الأعصاب الحاسوبي; "visual cortex" → القشرة البصرية; "conventions" (of an architecture) → اصطلاحات; "dense head" → الرأس الكثيف; "spatial coarsening" → تخشين مكاني.
- English glosses given in u01 of this page (do not repeat in later units): pooling layer, tolerance, global pooling, pooling window, maximum pooling, max-pooling, average pooling, downsampling, subsampling, aliasing, global average pooling. Translation invariance and equivariance take no gloss here (the why-conv page defines them).

## chapter_convolutional-neural-networks/lenet.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- Page title "LeNet" → شبكة LeNet (class noun so that the heading starts in Arabic); in running text LeNet takes شبكة where a sentence or block would otherwise begin with it.
- "convolutional encoder" → المشفر الالتفافي; "dense (classification) head" → رأس التصنيف الكثيف / الرأس الكثيف; "dense block" → الكتلة الكثيفة; "convolutional block" → كتلة التفاف (كتلتا التفاف; slides: كتلة الالتفاف 1 / 2).
- "average-pooling layer" → طبقة لتجميع المتوسط (طبقة تجميع المتوسط); slide "AvgPool" → تجميع المتوسط; slide "conv", "Flatten", "Dense", "Sigmoid" as chain items → التفاف، تسطيح، طبقة كثيفة، سيغمويد; the layer labels Conv1, Conv2 and `Dense1` stay Latin after الطبقة.
- "support vector machines" → آلات متجهات الدعم (from متجهات الدعم and آلة SVM), glossed once; "radial-basis-function units" → وحدات من دوال الأساس الشعاعي, on slides وحدات RBF; "handwritten digits" → الأرقام المكتوبة بخط اليد; "strokes" → ضربات القلم; "ATM" → أجهزة الصراف الآلي.
- "to assemble" (a network from parts) → يركّب، تركيب (not تجميع, which is pooling); "compressed notation" → صيغة رموز مضغوطة; "pixel-shuffled images" → صور خُلطت بكسلاتها, "pixel-shuffle test" → اختبار خلط البكسلات; "channel depth" → عمق القنوات; "layer widths" → عروض الطبقات; "feature detectors" → كاشفات الميزات.
- "to trade A for B" → يقايض A بـ B; "shrink" (spatial size) → يقلص، التقليص المكاني; "sparse wiring" → توصيل متناثر, "all-to-all wiring" → توصيل كامل; "memory–compute asymmetry" → اللاتناظر بين الذاكرة والحساب; "bit-for-bit" → بتًا ببت.
- English glosses given in u01 of this page (do not repeat in later units): convolutional encoder, dense classification head, support vector machines, multiply-adds.

## chapter_convolutional-neural-networks/conv-layer.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "Convolutions for Images" → الالتفاف على الصور; "edge map" → خريطة الحواف; "edge detector" → كاشف الحواف; "hand-crafted" → الذي صممناه يدويًا / التصميم اليدوي; "two-weight detector" → الكاشف ذو الوزنين.
- "sweep / slide a kernel across" → يمرر، تمرير (slides) and نزلق (body, style §9.1); "sliding window" → نافذة منزلقة; "anchoring at the upper-left corner", "top-left anchor" → التثبيت عند الركن العلوي الأيسر; "valid position" → موضع صالح; "the window fits at N positions" → تتسع المدخلات للنافذة في N مواضع.
- "even-sized kernels" → النوى ذات الأبعاد الزوجية; "center pixel" → بكسل مركزي; "flipped kernel" → نواة مقلوبة; "strict mathematical convolution" → الالتفاف الرياضي بمعناه الدقيق; "family of layers" → عائلة الطبقات.
- An "$h \times w$ convolutional layer / kernel / input" → طبقة التفافية $h \times w$، نواة $2 \times 2$، مدخلات $3 \times 3$ (the math span directly after the noun); with a code name as well: `X` ذات الأبعاد $6 \times 8$.
- "input resolution" → دقة المدخلات (style §10.1); "spatial dimensions" → الأبعاد المكانية; "border of zeros" → إطار من الأصفار; "stripe", "band" (of the test image) → شريط; "boundary" (white-to-black) → الحد الفاصل; "brightness" → السطوع; "intensity" → الشدة; "flat / uniform regions" → المناطق المستوية / المتجانسة.
- "horizontal finite difference" → الفرق المحدود الأفقي (singular, as in style §4; the glossary lemma is فروق محدودة); "horizontal-difference kernel" → نواة الفرق الأفقي; "discrete approximation" → تقريب متقطع.
- "diagonals" (edges) → الحواف القطرية; "corners" (image features) → الأركان; "textures" → أنماط النسيج; "orientation" → اتجاه, "orientation selectivity" → الانتقائية للاتجاه; "filter banks" → مجموعات المرشحات.
- "to supervise a layer with `Y`" → يشرف على الطبقة بـ …; "loss signal" → إشارة خسارة; "task loss" → خسارة المهمة; "end to end" → من طرف إلى طرف; "deep CNN" → شبكة CNN عميقة.
- Slides: "Motivation & Core Mental Model" → الدافع والنموذج الذهني الأساسي; "grid" (of pixels) → شبكة نقطية; "sub-window" → نافذة جزئية; "learnable" → قابل للتعلم; "megapixel" → ميغابكسل; "binary test pattern" → نمط اختبار ثنائي القيم; "piecewise-constant" → ثابت قطعيًا; "signed edge map" → خريطة حواف ذات إشارة; "Layer Module" in a title → طبقة … بصيغة Module; "identical learned class" → الفئة المتعلَّمة نفسها.
- English glosses given in u01 of this page (do not repeat in later units): edge map, two-dimensional cross-correlation, receptive field, padding, finite difference, backpropagation. Translation equivariance, locality, stride, dilation, dense layer, filter and feature are used in passing and take no gloss here; feature map is defined in a later unit.

## chapter_convolutional-neural-networks/why-conv.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "locality" → المحلية (as "spatial locality" → المحلية المكانية); "translation equivariant" (adjective) → متساوق مع الإزاحة; "translation invariance / invariant" → ثبات الإزاحة / ثابت تحت الإزاحة; plain "equivariance", "invariance" → التساوق، الثبات.
- "grid" (of pixels, hidden units) → شبكة نقطية (style §10.1), "hidden grid" → شبكة نقطية مخفية; "hidden map" → خريطة مخفية; "response (map)" → (خريطة) الاستجابة; "score map" → خريطة الدرجات.
- "shift operator" → مؤثر الإزاحة; "commuting diagram" → مخطط تبادلي; "$f$ commutes with shifts" → تتبادل مع الإزاحات; "unit impulse" → نبضة وحدة; "probing" → السبر; "template" → قالب; "Toeplitz matrix" → مصفوفة Toeplitz (name in Latin).
- "fourth-order tensor" → موتر من المرتبة الرابعة; "re-index" → يعيد الفهرسة; a spatial "index" (subscript) → دليل; "panel (a)" → اللوحة (a); "prune" (weights) → يشذب; "free weights" → الأوزان الحرة; "shared weights" → الأوزان المشتركة.
- "megapixel" → ميغابكسل (1MP kept on slides); "sample efficiency" → كفاءة العينات; "textures" → أنماط النسيج; "time-delay neural networks" → الشبكات العصبية ذات التأخير الزمني; "Where's Waldo" → “أين Waldo” (the name stays Latin).
- "sufficiency / necessity" (of a proof) → الكفاية / الضرورة; "converse" → العكس; slide kicker "Commutation invariant" → لامتغير التبادل.
- English glosses given in u01 of this page (do not repeat in later units): locality, translation equivariance, convolutional layer, translation invariant, convolutional neural networks (CNN), convolution kernel, filter, inductive bias.

## chapter_convolutional-neural-networks/channels.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "input channel(s)" / "output channel(s)" → قناة إدخال / قناة إخراج (قنوات الإدخال، قنوات الإخراج; glossary); "multi-channel" → متعدد القنوات; "multi-input kernel" → نواة متعددة المدخلات; page title → قنوات الإدخال المتعددة وقنوات الإخراج المتعددة.
- "luminance" → النصوع; "brightness" → السطوع ("equal luminance", "isoluminant" → متساوي النصوع); "boundary" (between colour regions) → حد (حدود); "neutral gray" → الرمادي المحايد; "color opponency" → تضاد الألوان.
- "third-/fourth-order kernel tensor" → موتر نواة من المرتبة الثالثة / الرابعة (مرتبة as on the linear-algebra page); slides "3D / 4D" → ثلاثي / رباعي الأبعاد.
- "scalar products" (the single multiplications inside one output entry) → جداءات قياسية.
- "kernel slice", "filter slice" → شريحة النواة، شريحة المرشح; "filter bank" → مجموعة المرشحات (as on the chapter page).
- "all-to-all channel wiring" → التوصيل الذي يربط كل قناة بكل قناة (slides: توصيل كل قناة بكل قناة).
- "to map N channels to M channels" (a layer) → ينقل N قناة إلى M قناة; "256-to-256 layer" → الطبقة التي تنقل 256 قناة إلى 256 قناة.
- "… contains [display equation] learnable parameters / performs [display equation] multiply-adds": the block after the equation starts with the counted noun in the singular (معلَمة قابلة للتعلم. / عملية ضرب وجمع.), so the content stays in its block.
- "scales with" (cost) → يتزايد مع; "scaling wall" → جدار التزايد; "output area" → مساحة المخرجات.
- "receptive window" → نافذة الاستقبال; "mental arithmetic" → الحساب الذهني.
- "depthwise-separable convolution" → التفاف عمقي قابل للفصل (glossary التفاف عمقي); "grouped convolutions" → التفافات المجموعات.
- "lightweight mobile models" → النماذج الخفيفة للأجهزة المحمولة; "to project channels down / expand back" → إسقاط … على / التوسيع من جديد إلى; "spatial mixing", "channel mixing" → المزج المكاني، مزج القنوات.
- "stem" (the first layers of a CNN) → الجذع; "stage" → مرحلة (مراحل); "high-res" → عالي الدقة; "VRAM" → ذاكرة VRAM.
- "shift pixel-for-pixel" → تنزاح بكسلًا ببكسل; "tolerance to small spatial shifts" → تسامح مع الإزاحات المكانية الصغيرة (as on the pooling page); "translation-invariant" (slides) → ثابت تحت الإزاحة.
- Kicker "Section Synthesis" → خلاصة القسم; "Solution preview" → لمحة عن الحل; "Roles" → الأدوار.
- English glosses given in u01 of this page (do not repeat in later units): cross-correlation, feature map, multiply-add, activation memory, fully connected layer, bottleneck, grouped convolutions, depthwise-separable convolutions.

## chapter_convolutional-neural-networks/channels.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "patch matrix" (im2col) → مصفوفة الرقع (glossary رقعة); "input patches" → رقع المدخلات; exercise title "Multi-channel im2col" → طريقة im2col متعددة القنوات (class noun طريقة before the Latin name).
- "composition" of two convolutions / kernels → تركيب; "intermediate width" → عرض وسيط; "single-channel kernel" → نواة أحادية القناة.
- "cues" (colour cues across channels) → الدلائل, as on this page's slides; "parameter- and compute-efficient" → مقتصد في المعلَمات وفي الحساب.
- "output values that training must keep" → قيم المخرجات التي يجب أن يحتفظ بها التدريب; "per image" → للصورة الواحدة; "reproduces `f(X, K)`" → يطابق نتيجة `f(X, K)`.
- No English glosses in u02 of this page (all key terms were glossed in u01).

## chapter_convolutional-neural-networks/conv-layer.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "Convolution as Matrix Multiplication" → الالتفاف بوصفه ضرب مصفوفات; "patch matrix" → مصفوفة الرقع; "workspace memory" → ذاكرة مساحة العمل; "layout transformation" → تحويل التخطيط; im2col stays Latin, with a class noun (تحويل im2col، التحويل `im2col`) where a block or bold lead-in would start with it.
- "unroll / unfold" (the kernel, the image) → يفرد، فرد; "sparse banded matrix" → مصفوفة شريطية متناثرة; "W multiplying the image" → تُضرب في; "structured dense layer" → طبقة كثيفة ذات بنية; "panel (c)" → اللوحة (c).
- "theoretical / effective receptive field" → مجال الاستقبال النظري / الفعال; the receptive field or an error "decays" → يتناقص (اضمحلال stays weight decay); "Gaussian bell curve" → منحنى جرس غاوسي; "sensitivity" → حساسية.
- "reaches $k - 1$ pixels farther" → تمتد $k - 1$ بكسلًا أبعد; "side length" → طول الضلع; "tower" (of stacked layers in a figure) → برج; "chain of connections" → سلسلة من الوصلات; "edge input" (at the edge of a receptive field) → مدخل طرفي; "count paths" → يحصي المسارات; "bars" (of a figure) → الأشرطة; "channel pair" → زوج قنوات.
- "under-sampling border pixels" → لا يأخذ من بكسلات الحدود إلا عينات أقل مما ينبغي; "hierarchies of filters" → تراتبات من المرشحات.
- Slides: "Dual View" → النظرة المقابلة; "a single dense GEMM" → عملية GEMM كثيفة واحدة; "GPU Tensor Core" → أنوية الموترات في وحدة GPU; "patch duplication" → نسخ مكررة للرقع; "expressivity" → القدرة التعبيرية; "std" → انحراف معياري; "RF" → مجال الاستقبال (no Latin acronym); "Section Synthesis" → خلاصة القسم; a number list in a caption is joined by free-standing و.
- English glosses given in u02 of this page (do not repeat in later units): feature map. Receptive field was glossed in u01; "effective receptive field" takes no gloss (only the adjective is emphasized).

## chapter_convolutional-neural-networks/why-conv.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "discrete convolution" → الالتفاف المتقطع; "mathematical convolution" → الالتفاف الرياضي; "flipped kernel / flipped copy" → النواة المقلوبة / النسخة المقلوبة, "the flip", "flipping" → القلب; "unflipped" → غير المقلوب; "overlap" (between two functions) → التراكب.
- "third-order tensor" → موتر من المرتبة الثالثة (as "fourth-order" in u01); "the leading channel axis" → المحور الأول، وهو محور القنوات; "output-before-input channel convention" → الاصطلاح الذي يقدم قناة الإخراج على قناة الإدخال; "kernel slice" → شريحة من النواة.
- "the entire stack" (of feature maps) → الخرائط المكدسة كلها (no noun كدسة, style §10.1); "to aggregate feature maps", "pooling evidence" (not the pooling layer) → الإجمال, as on the pooling page; "expand how much of the image units can see" → زيادة ما تراه الوحدات (يوسع is kept for dilation).
- "volumetric medical scans" → الصور الطبية الحجمية, slide "volumes" → الأحجام; "video frames" → إطارات الفيديو; "color contrasts" → تباينات لونية; "corners" → أركان; "textures" → أنماط نسيج (as in u01 of this page).
- Exercises: "periodic grid" → شبكة نقطية دورية, "ring" → حلقة, "wraps around" → تتصل نهايتها ببدايتها; "commutative / commutativity" → تبديلي / تبديلية; "spectrogram" → المخطط الطيفي; "hertz" → الهرتز in prose, Hz on slides; "head" (of a classifier) → رأس; "rotate" → يدير (إدارة), "rotation" → دوران; "permutes cyclically" → يبدل تبديلًا دوريًا; "circulant matrix" → مصفوفة دوارة; "the first part" (of an exercise) → الجزء الأول.
- Slides: "tied weights" → الأوزان المشتركة; "blind spot" → البقعة العمياء; "N-order-of-magnitude collapse" → انخفاض بمقدار N مرتبة مقدار; "harmonic ratios" → النسب التوافقية; "modalities" → أنواع البيانات; "Section Synthesis" → خلاصة القسم; note lead-in "Next (§{…}):" → التالي (§{…}): followed by verbal nouns; table cell "Identical" → متطابق; "intractable" → غير عملي.
- English glosses given in u02 of this page: convolution, cross-correlation, output channel, feature maps. "channel" (everyday word) takes no gloss.

## chapter_convolutional-neural-networks/pooling.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "neutral element" (of max, of a sum) → العنصر المحايد; "neutral constant" → ثابت محايد; "beat" (a padded cell beating an activation) → يتغلب على.
- "override the defaults" (of a layer's arguments) → تجاوز القيم الافتراضية (يعيد تعريف stays for overriding a method); "explicit padding" → حشو صريح; windows "tile the input" → تتراصف النوافذ على المدخلات, "tiles" (table) → بلاطات.
- "minimum pooling", "min-pooling" → التجميع الأدنى; "softmax pooling" → تجميع softmax; "softmax-weighted average" → المتوسط الموزون بأوزان softmax.
- "blur-pool" stays Latin after the class noun طريقة (طريقة blur-pool); "two-tap box filter" → مرشح صندوقي ذو وزنين; "averaging filter" → مرشح متوسط; "subsampler" → طبقة أخذ عينات فرعية; "cycles per input sample" → عدد الدورات لكل عينة من المدخلات.
- "shift equivariance in value" → تساوق الإزاحة في القيمة; "strictly negative" → سالبة تمامًا; "genuine input entries" → عناصر المدخلات الفعلية, "real entries / activations" → العناصر / التفعيلات الحقيقية.
- "two-channel tensor" → موتر ذو قناتين; "channel axis" → محور القنوات; "cross-channel mixing" → المزج بين القنوات (مزج as on the channels page; خلط is shuffling); "pairwise comparisons" → المقارنات الثنائية.
- Slides: "Section Synthesis" (kicker) → خلاصة القسم; "Systems Pitfall" → مزلق في الأنظمة; "Modern Evolution" → التطور الحديث; "Head aggregation" → الإجمال في الرأس; "dense flatten head" → الرأس الكثيف المسطح; "Test your understanding of …" → اختبار فهم …; "Next (§{…}):" → التالي (§{…}):.
- CS231n credit: "Assignment 2, ConvolutionalNetworks, …" → الواجب [Assignment 2](url)، دفتر ConvolutionalNetworks، ….

## chapter_convolutional-neural-networks/lenet.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "plateau" (noun: the stretch of a training curve stuck at chance) → الاستواء (الاستواء الابتدائي); "at chance", "near chance" → عند / قرب مستوى الصدفة; "takes off" → تنطلق; "run" (of training) → جولة (جولة نموذجية، من جولة إلى أخرى).
- "shuffled pixels / images / dataset" → البكسلات / الصور / مجموعة البيانات المخلوطة; "ordered images" → الصور المرتبة; "permuted image" → صورة مبدَّلة; "spatial adjacency" → التجاور المكاني; "destroy" (locality, contours) → يتلف، إتلاف; "white noise" → ضوضاء بيضاء.
- "foreground detector" → كاشف للمقدمة; "filled silhouette" → الصورة الظلية المملوءة; "bright / dark" (columns, panels) → فاتح / داكن; "panel" (of a figure) → لوحة; "template image" → صورة قالب; "spatial cell" → خلية مكانية; "unconstrained scenes" → المشاهد غير المقيدة.
- Garments drawn from the data take their English class name once, since the figure titles print it: حذاءً رياضيًا (sneaker، الفئة 7)، قميصًا (shirt، الفئة 6).
- "spatial prior" → الافتراض القبلي المكاني (as "geometric prior" on the chapter page); "symmetry principles" → مبادئ التناظر; "strictly translation-equivariant / -invariant" → متساوق مع الإزاحة تمامًا / ثابت تحت الإزاحة تمامًا; "shift invariance" (slide) → ثبات الإزاحة (glossary).
- Slides: "(cont.)" → (تابع); "Section & Chapter Synthesis" → خلاصة القسم والفصل; "blueprint" → مخطط; "Signal propagation" → انتشار الإشارة; "residual skips" → وصلات التخطي ذات البواقي; "residual stages" → مراحل بواقٍ; "GAP" → تجميع GAP; "(Chapter N Preview)" → (لمحة عن الفصل N); table labels "Fashion-MNIST (Ordered / Shuffled)" → بيانات Fashion-MNIST (مرتبة / مخلوطة).
- English gloss given in u02 of this page: inductive bias.

## chapter_convolutional-neural-networks/padding-and-strides.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "strided stack / dilated stack" → المجموعة المكدسة بخطوة انزلاق / المجموعة المكدسة الموسعة; "tower" (slides) → برج (برج خطوة الانزلاق، برج التوسيع); "top unit" → الوحدة العليا; "top-level units" → وحدات المستوى الأعلى.
- "effective kernel size" → حجم النواة الفعال; "dilation rate" → معدل التوسيع; "dense kernel / dense convolution" (not dilated) → نواة كثيفة / الالتفاف الكثيف; "dilated layer / kernel" → طبقة / نواة موسعة.
- "floor term" → حد التدوير إلى الأسفل; "rounding up" → مع التدوير إلى الأعلى.
- "patchify" → التقطيع إلى رقع; "tiles" → بلاطات; "stem" → جذع (as in the chapter index); "subscripts" → الأدلة السفلية; "zero padding ($p=0$)" in the sense "no padding" → حشو مقداره صفر (not حشو الأصفار, which is padding with zeros).
- "widen / expand the receptive field" → يزيد مجال الاستقبال; "grow geometrically" → ينمو هندسيًا, "grow exponentially" → ينمو أسيًا (the English uses both for the same growth; each is kept).
- "gridding" → التشبيك ("gridding artifacts" → آثار التشبيك المصطنعة); "lattice" → شبيكة (شبيكات); "dilation schedule" → جدول التوسيع ("constant schedule" → جدول ثابت, "doubling schedule" → جدول المضاعفة).
- Slides: "Section Synthesis" → خلاصة القسم; "Next (§…):" → التالي (§…):; "master equation / Master Formula" → معادلة جامعة / الصيغة الجامعة; "regimes" (padding-and-stride settings) → إعدادات; "RF" → مجال الاستقبال (written out); "MACs" → عمليات MAC; "params" → معلَمات; "conv" → التفاف; "panel (a)" → اللوحة (a).
- "every $s$-th row" → صف واحد من كل $s$ صفًا; "every other pixel" → بكسل واحد من كل بكسلين; "every fourth" → واحد من كل أربعة.
- English glosses given in u02 of this page (do not repeat in later units): strided convolution, dilated convolution, effective kernel size, patchify, vision transformer, dense prediction, transposed convolution.

## chapter_convolutional-neural-networks/conv-layer.md#u03 (claude fable-5.2 (medium), 2026-11-27)

- "blur kernel" → نواة تضبيب; "sharpening kernel" → نواة زيادة الحدة; "identity" (kernel) → النواة المحايدة; "photograph" → صورة فوتوغرافية; "downsample" (an image, in an exercise) → يصغّر بخفض أخذ العينات.
- "locally connected layer" → طبقة متصلة محليًا (no glossary entry); "sparsity pattern" → نمط التناثر; "nonzero entries" → العناصر غير الصفرية; "dense array" → مصفوفة كثيفة; "ties the weights" → يربط الأوزان بعضها ببعض; "banded matrix" → مصفوفة شريطية.
- "derivative of order $d$" → مشتق من المرتبة $d$; "directional vector" → متجه اتجاه; "centered finite-difference gradient" → تدرج محسوب بالفروق المحدودة المركزية; "relative error" → خطأ نسبي.
- "a draw of the kernels" → سحبة للنوى; "unit gradient" → تدرج مقداره واحد; "magnitude" (of a gradient) → مقدار; "support" (of the gradient map) → الحامل; "column profile" → منحنى الأعمدة, "normalized profile" → المنحنى المطبَّع; "expected squared input gradient" → مربع تدرج المدخلات المتوقع; "bounds which inputs can influence" → يحصر المدخلات التي يمكن أن تؤثر.
- Credit line with a notebook: *مقتبس بتصرف من مقرر Stanford CS231n، الواجب [Assignment 2](url)، دفتر ConvolutionalNetworks.* (as on the pooling page); two sources are joined by ، ومن.
- No English glosses in u03 of this page (exercises take none).

## chapter_convolutional-neural-networks/padding-and-strides.md#u03 (claude fable-5.2 (medium), 2026-11-27)

- "mirror padding" → حشو المرآة (the exercise equates it with "reflect padding" → حشو الانعكاس, as in u01); "ramp" (the input $x_{ij} = j$) → المنحدر; "reproduce" (a ramp, a stack) → يعيد إنتاج.
- "entries" of a tensor or kernel → عناصر; "border entry" → عنصر الحد; "distinct entries" → عناصر متمايزة; "by hand" → يدويًا; "in the interior" → في الداخل; "ignoring the border" → مع إهمال الحدود.
- "side" of a receptive field → طول الضلع; "sparse lattice" → شبيكة متناثرة; "the condition suffices" → الشرط كافٍ; "mechanisms" (padding, stride, dilation) → الآليات; "effective footprint" → مساحة تغطية فعالة.
- "patchify layer" → طبقة التقطيع إلى رقع; "one-dimensional stacks" → المجموعتان المكدستان أحاديتا البُعد.
- No English glosses in u03 of this page (summary and exercises; all key terms were glossed in u01 and u02).

## chapter_convolutional-neural-networks/lenet.md#u03 (claude fable-5.2 (medium), 2026-11-27)

- "modernized LeNet" (exercises) → LeNet المحدَّثة; "variant" (of a model) → صيغة; "reduced heads" → الرأسان المصغران; "ten-way linear output layer" → طبقة إخراج خطية ذات عشرة مخرجات; "flattened 400-dimensional output" → المخرجات المسطحة ذات 400 بُعد.
- "validation / test error rate" → معدل خطأ التحقق / معدل خطأ الاختبار; "official test set" → مجموعة الاختبار الرسمية; "data class" → صنف بيانات; "the two measures" (difference and ratio) → المقياسان.
- "problem :exref:`…`" → التمرين :exref:`…` (same word as for "Exercise"); "the text drew / argued" → رسم النص / ذهب النص إلى أن.
- "shifted" images / garments / set → مزاحة; "translated copies" → نسخ مزاحة; "vacated columns" → الأعمدة التي خلت; "uniform random noise" → ضوضاء عشوائية منتظمة; "pixel range" → مجال قيم البكسلات; "maximum activation" → التفعيل الأقصى; "maxima" → القيم العظمى; "pullovers and coats" → الكنزات والمعاطف.
- Summary: "encoder-plus-head blueprint" → المخطط العام المؤلف من مشفر ورأس; "heavy dense heads" → الرؤوس الكثيفة الثقيلة; "training plateaus" → فترات استواء; "scale up in depth, width, and resolution" → ازدياد عمق الشبكات وعرضها ودقة مدخلاتها.
- "Adapted from Michael Nielsen, [book](url), Chapter N" → مقتبس بتصرف من كتاب [book](url) لمؤلفه Michael Nielsen، الفصل N (as on the mlp page).
- No English glosses in u03 of this page (max-pooling, strided convolution, global average pooling and normalization layers are mentioned in passing in the summary; exercises take none).

## chapter_convolutional-neural-networks (chapter check, 2026-11-27)

- One rendering per term across the chapter's seven pages (the entries above are corrected to match): slide kicker "Section Synthesis" → خلاصة القسم ("Section & Chapter Synthesis" → خلاصة القسم والفصل; ملخص القسم stays "Section Summary", and "Check & Synthesis" stays التحقق والتوليف as in earlier chapters); "filter bank" → مجموعة مرشحات; "textures" as image features → أنماط النسيج (نسيج السطح for the texture of an object); "tolerance" to shifts → التسامح مع الإزاحة (تحمل is not used for it); "translation / shift invariance" (noun) → ثبات الإزاحة, adjective ثابت تحت الإزاحة; "channel mixing" → مزج القنوات (خلط is shuffling); "corners" (image features) → الأركان; by position: "edge / border pixel" → بكسل طرفي, "corner pixel" → بكسل ركني, "boundary pixels" → بكسلات الحدود; "plateau" of a training curve → الاستواء.


## chapter_convolutional-modern/index.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- Chapter title "Modern Convolutional Neural Networks" → الشبكات العصبية الالتفافية الحديثة; "efficient convnets" → الشبكات الالتفافية الكفؤة (فعال stays "effective").
- "shortcut" (ResNet) → وصلة تخطٍ, as the glossary's identity-shortcut entry: "additive shortcuts" → وصلات التخطي الجمعية, "residual shortcuts / skips" → وصلات التخطي ذات البواقي; "identity mapping" → الدالة المحايدة; "optimization degradation" → تدهور التحسين, "degradation" → التدهور; "plain network / plain stack" → شبكة عادية / المجموعة المكدسة العادية.
- "stem--body--head layout / template" → تخطيط / قالب من جذع وجسم ورأس (the one form for the whole chapter); "modular building blocks" → لبنات بناء قابلة للتركيب; "hand-wired" → موصول يدويًا; "repeatable blocks" → كتل قابلة للتكرار; "toolkit" → العدة ("Architectural Toolkit" → العدة المعمارية).
- "warmup-and-cosine learning-rate schedule" → جدول معدل التعلم القائم على الإحماء وجيب التمام; "cosine warmup" → الإحماء مع جيب التمام; "weight averaging" → حساب متوسط الأوزان; Mixup, CutMix, RandAugment, LAMB, EMA, BCE stay Latin (طريقتا Mixup و CutMix، خسارة BCE); a slash-joined run of names (RandAugment/Mixup/CutMix, AdamW/Mixup/EMA) is copied as one Latin run.
- "depthwise-separable factorization" → التحليل إلى التفاف عمقي قابل للفصل (slides "depthwise factorization" → التحليل العمقي); "inverted bottleneck" → عنق الزجاجة المعكوس; "compound scaling" → التحجيم المركب; "squeeze-and-excitation gates" → بوابات الضغط والإثارة; "structural re-parameterization" → إعادة تحديد المعلَمات البنيوية; "RepVGG fusion" → دمج RepVGG; "intensity" (arithmetic intensity, slides) → الكثافة الحسابية (glossary).
- "Network in Network (NiN)" → شبكة داخل الشبكة (Network in Network, NiN) at first mention (glossary term), then NiN; "hierarchical Vision Transformers" → محولات الرؤية الهرمية; "design space" → فضاء التصميم; "sampled populations" (of networks) → مجتمعات من الشبكات المسحوبة بأخذ العينات; "population error CDFs" → دوال التوزيع التراكمي للخطأ على مجتمعات الشبكات; "landmark model" → نموذج فارق.
- "hardware efficiency" → كفاءة استخدام العتاد; "operation counts" → عدد العمليات; "usable learning rates / LR band" → معدلات التعلم الصالحة للاستخدام / نطاقها; "energy" (cost of a network) → الطاقة; "dense-head bloat" → تضخم الرأس الكثيف; "confounded recipes" → التباس أثر الوصفات.
- Slides: "The Central Problem" → المشكلة المركزية; "Core Mental Models" → النماذج الذهنية الأساسية; "Empirical Comparison" → مقارنة تجريبية; "Design Invariants" → لامتغيرات تصميم; "nested classes" → الفئات المتداخلة; a lead-in that is a section range is moved into parentheses after its Arabic label, as on the earlier chapter pages; "43M-param" → 43 مليون معلَمة; LR → معدل التعلم, RF → مجال الاستقبال (written out), BN, GAP, MAC, FLOP stay Latin with a class noun (تجميع GAP، عمليات MAC، عمليات FLOP).
- English glosses given on this page: vision transformer, dropout, Network in Network (NiN), squeeze-and-excitation, structural re-parameterization. Section topics (normalization layers, residual networks, training recipes) take their gloss on their own pages.

## chapter_convolutional-modern/batch-norm.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "group normalization" → تطبيع المجموعات (no glossary entry; parallel to تطبيع الدفعات and تطبيع الطبقات); on slides LayerNorm, GroupNorm, BatchNorm and BN stay Latin.
- "scale" and "shift" of a normalization layer ($\gamma$, $\beta$) → المقياس والإزاحة ("learned scale and shift" → مقياس وإزاحة متعلَّمان); "scale invariance" → الثبات أمام التحجيم, "invariant to weight rescaling" → ثابت أمام إعادة تحجيم الأوزان (as on the linear-algebra page).
- "scales drift" → تنجرف المقاييس، انجراف المقاييس (glossary انجراف); "fragility" (of a learning rate) → الهشاشة; "usable learning rate" → معدل التعلم الصالح للاستخدام; "peak learning rate of 0.1" → معدل تعلم ذروته 0.1; "to train cleanly" → يتدرب تدريبًا سليمًا.
- "the *decay* $\rho$" of the running statistics → معامل التناقص (اضمحلال stays weight decay); "exponential moving averages" → المتوسطات المتحركة الأسية; "running statistics / estimates" → الإحصاءات / التقديرات المتحركة (as in the builders' guide); "effective window" → النافذة الفعلية.
- "training mode / prediction mode" → نمط التدريب / نمط التنبؤ ("eval" on slides → التقييم).
- "pool" (statistics over a batch and spatial positions; not the pooling layer) → يضم، ضم القيم; "spatial locations" → المواضع المكانية; "output unit" (of a fully connected layer) → وحدة من وحدات المخرجات.
- "to score one image" (a deployed model) → يحسب درجات صورة; "$N=1$ scoring" → حساب الدرجات عند $N=1$.
- "radial ray of $\mathbf{w}$" → الشعاع القطري للمتجه; "strictly increases" → يزيد زيادة تامة; "self-stabilizing" → ذاتي الاستقرار; "mixing weight" → وزن المزج; "minibatch moments" → عزوم الدفعة المصغرة; "representational freedom" → الحرية التمثيلية.
- English glosses given in u01 of this page (do not repeat in later units): batch normalization, layer normalization, group normalization, pre-activation, scale, shift, scale invariance, training mode, exponential moving averages, decay, prediction mode.

## chapter_convolutional-modern/efficient-convnets.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- Page title "Efficient Convnets" → الشبكات الالتفافية الكفؤة; "efficient networks" → الشبكات الكفؤة.
- "factorization" (of a convolution) → التحليل إلى عوامل; "depthwise-separable factorization" → التحليل العمقي القابل للفصل; "factored form" → الصيغة المحللة إلى عوامل; "rank-1 restriction" → قيد الرتبة 1.
- "depthwise convolution" → الالتفاف العمقي; "pointwise convolution" → الالتفاف النقطي; "depthwise-separable convolution / pair / block / layer" → التفاف عمقي قابل للفصل / الزوج القابل للفصل / كتلة عمقية قابلة للفصل / الطبقة القابلة للفصل; "grouped convolution" → التفاف المجموعات, "grouping" → التقسيم إلى مجموعات; "channel mixer" → مازج قنوات.
- "inverted bottleneck" → عنق الزجاجة المعكوس (as the glossary's "inverted dropout" → معكوس), "inverts" → تعكس; "expansion" of channels → طبقة توسعة / يعرّض، زيادة العرض (not التوسيع, which is dilation); "narrowing" a network → التضييق; "thin ends" → الطرفان الرفيعان; "residual highway" → مسار البواقي الرئيسي; "residual shortcut" → وصلة التخطي ذات البواقي.
- "scaling and search" → التحجيم والبحث; "compound scaling" → التحجيم المركب; "architecture search" → البحث عن المعمارية, "latency-aware NAS" → بحث NAS المراعي لزمن الاستجابة; "structural re-parameterization" → إعادة تحديد المعلَمات البنيوية; "branch fusion" → دمج الفروع, "fused convolutions" → التفافات مدمجة; "SE gates" → بوابات SE.
- "arithmetic" (amount of computation) → العمليات الحسابية / الحساب; "at equal / matched arithmetic" → عند تساوي العمليات الحسابية; "parameter-matched / FLOP-matched" → المطابق في المعلَمات / المطابق في عمليات FLOP; "proxy" → بديل; "memory-bandwidth bound" → مقيد بعرض نطاق الذاكرة.
- "edge hardware", "edge" (devices) → العتاد الطرفي، الأجهزة الطرفية; "mobile-grade", "mobile processors" → للأجهزة المحمولة، معالجات الأجهزة المحمولة; "embedded boards" → اللوحات المدمجة (not مضمنة: التضمين is embedding); "smartphones" → الهواتف الذكية; "hard budgets" → ميزانيات صارمة; "ship" (a model) → ينشر; "battery draw" → استهلاك البطارية; "flash storage" → تخزين الفلاش; "activation RAM" → ذاكرة RAM المخصصة للتفعيلات.
- "top-1 accuracy" → دقة top-1 (as "top-5 error rate" → معدل خطأ top-5); "accuracy leaderboards" → لوحات المتصدرين في الدقة; "real time" → الزمن الحقيقي; "frames per second" → إطارًا في الثانية.
- "block-diagonal" → قطري كتلي; "channel-connectivity matrix" → مصفوفة الاتصال بين القنوات; "all-to-all coupling" → الاقتران بين كل قناة وكل قناة; "kernel stack" → مجموعة نوى مكدسة.
- "half-cosine" → نصف دورة من جيب التمام; "cosine scheduler" → مجدول جيب التمام; "training schedule" → جدول التدريب; "SGD with momentum 0.9" → خوارزمية SGD مع زخم 0.9.
- "VGG-style network" → شبكة على نمط VGG; table row labels take the class noun شبكة (شبكة Mini-MobileNet، شبكة على نمط VGG، ضيقة); the code label `VGG-style, narrow` stays as written.
- Counts: "50M multiply-adds", "76K parameters", "~550K parameters" in body text → 50 مليون عملية ضرب وجمع، 76 ألف معلَمة، نحو 550 ألف معلَمة; slides keep the math spans ($542\text{K}$, $50.3\text{M}$) and "MACs" → عمليات MAC; "15B" in a slide title → 15 مليار.
- Headings of model names take the class noun: "MobileNet" → شبكة MobileNet; "A Mini-MobileNet" → شبكة Mini-MobileNet.
- English glosses given in u01 of this page (do not repeat in later units): multiply-adds, factorization, scaling and search, structural re-parameterization, grouped convolution, depthwise convolution, pointwise, depthwise-separable convolution, inverted bottleneck, arithmetic intensity.

## chapter_convolutional-modern/resnet.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "Residual Networks" (page title) → شبكات البواقي, glossed (residual network, ResNet); "residual block" → كتلة البواقي, "residual branch" → فرع البواقي (فرع بواقٍ), "residual mapping" → دالة الباقي, "residual networks" (indefinite) → شبكات بواقٍ; ResNet with a class noun → شبكة ResNet.
- "shortcut" and "skip connection" → one Arabic term, وصلة التخطي (وصلة تخطٍ), as in the glossary's identity-shortcut entry and the earlier وصلة التخطي; the first mention gives both English names. "identity shortcut" → وصلة التخطي المحايدة (the log's الدالة المحايدة / المسار المحايد for "identity"; not the glossary proposal وصلة تخطي هوية); "projection shortcut" → وصلة التخطي بالإسقاط (slide title: وصلة الإسقاط); "additive shortcuts" → وصلات التخطي الجمعية.
- "plain network / block / layer" → شبكة / كتلة / طبقة عادية; "regular block" → الكتلة المعتادة; "degradation (problem)" → (مشكلة) التدهور, "degradation paradox" → مفارقة التدهور, "to degrade" → تتدهور.
- "identity mapping", "the identity" → الدالة المحايدة; "nested function classes" → فئات الدوال المتداخلة; "input–output mappings" → الدوال من المدخلات إلى المخرجات; "target mapping" → الدالة المستهدفة.
- "reparameterization", "reparameterize" → إعادة صياغة المعلَمات، يعيد صياغة معلَمات …; "bias-free" (convolution) → بلا انحياز; "zero out" → يصفّر; "ensemble" (of shallow networks) → مجموعة نماذج; "batch-normalized networks" → الشبكات المزودة بتطبيع الدفعات; "the second batch norm" → تطبيع الدفعات الثاني; slide "BatchNorm" → طبقة BatchNorm (Latin, as the English slide writes it).
- "to widen blocks" → زيادة عرض الكتل (يوسع stays dilation); "at matched cost" → بكلفة مماثلة; "to concatenate" (feature maps) → يربط (glossary ربط), against يجمع for adding; "descendants" (of an architecture) → نماذج انحدرت منها.
- "Euler discretization" → تقطيع بطريقة أويلر (the name is in the glossary term طريقة أويلر, so Arabic letters); "an ODE" → معادلة تفاضلية عادية (ODE); "unit-step" → بخطوة مقدارها واحد.
- "gradients decorrelate" → يضعف الارتباط بين التدرجات; "rough" (loss surface) → وعر, "smooth" → أملس; "large-scale recipes" → وصفات التدريب واسعة النطاق; "gradient highway" → الطريق السريع للتدرج; "eval mode" → نمط التقييم; kicker "Code Implementation" → التنفيذ في الشيفرة.
- English glosses given in u01 of this page (do not repeat in later units): degradation problem, identity mapping, residual network (ResNet), reparameterization, function class, nested, residual, shortcut connection, skip connection.

## chapter_convolutional-modern/alexnet.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- Page title "AlexNet and Learned Image Representations" → شبكة AlexNet وتمثيلات الصور المتعلَّمة (class noun شبكة so that the heading starts in Arabic; in running text AlexNet is feminine, as LeNet).
- "top-5 error" → خطأ top-5 (from the log's معدل خطأ top-5); "top five predictions" → أفضل خمسة تنبؤات; "competition entry" → مشاركتها في المسابقة; "ensemble of seven networks" → مجموعة من سبع شبكات; "watershed" → نقطة التحول; "catalyst" → محفز (محفزات); "breakthrough" → اختراق.
- "bags of visual words" → حقائب الكلمات البصرية; "kernel classifier" → مصنف بدالة نواة; "hand-engineered" → مهندَس يدويًا, "hand-crafted" → مصمَّم يدويًا (as in the log); "human feature engineering" → هندسة الميزات بأيدي البشر.
- "color-opponent blobs" → بقع متضادة الألوان (slide title "color blobs" → البقع اللونية); "frequency detectors" → كاشفات الترددات; "parts" (of objects) → أجزاء; "category templates" → قوالب الفئات; "Gabor-like" → شبيهة بمرشحات Gabor; "wavelets" → المويجات.
- "crops" → مقاطع (اقتصاص مقاطع عشوائية); "horizontal reflections / flips" → انعكاسات أفقية; "views" (augmented versions of an image) → منظورات; "effective sample diversity" → تنوع العينات الفعال; "on-the-fly augmentation" → زيادة البيانات الآنية; "geometric augmentation" → الزيادة الهندسية; "photometric PCA jitter" → الاهتزاز الضوئي بطريقة PCA; "illumination invariance" → الثبات تحت تغير الإضاءة.
- "graphics processing units (GPUs)" → وحدات معالجة الرسوميات (graphics processing units, GPUs); "quad-core CPU" → وحدة CPU رباعية الأنوية; "high-bandwidth memory" → ذاكرة عالية عرض النطاق; "throughput" → الإنتاجية; "memory-bandwidth bound" → مقيد بعرض نطاق الذاكرة; TFLOPS, GB/s, MB stay Latin.
- "thumbnails" (small $28 \times 28$ images) → الصور المصغرة; "overlapping max-pooling" → تجميع أقصى متداخل; "parameterized layers" → طبقات ذات معلَمات; "local response normalization" → تطبيع الاستجابة المحلية; "historical artifacts" (of the 2012 design) → تفاصيل تاريخية; "saturating / non-saturating" → المتشبعة / غير المتشبعة; "active units" → الوحدات النشطة; "by a factor $4^{-K}$" → بعامل (معامل stays "coefficient").
- "ReLU activations" → التفعيل بدالة ReLU; "classifier head" → رأس المصنف, "classification head" → رأس التصنيف; "trunk" (the convolutional part before the head) → هيكل (الهيكل الالتفافي), since جذع is the stem; "body" → جسم; "macro template / architecture" → القالب الكلي / المعمارية الكلية.
- "dummy image / tensor" → صورة وهمية / موتر وهمي (as the log's دفعة مصغرة وهمية); "unseeded initializer" → مهيئ لا بذرة له; "ledger" (parameter and compute ledger) → سجل; "lopsided" → مختل (أشد اختلالًا); "the arithmetic" → العمل الحسابي (as in the log).
- Table rows: "$k\times k$ convolution, a to b, stride s" → التفاف $k\times k$، من a إلى b، بخطوة انزلاق s; "fully connected, a to b" → كاملة الاتصال، من a إلى b; "total" → المجموع (as on the LeNet page). Slide table labels Conv1 … Dense3 stay Latin after الطبقة / الطبقتان.
- Slide kickers: "Motivation & Historical Bottleneck" → الدافع وعنق الزجاجة التاريخي; "Learned Representations" → التمثيلات المتعلَّمة; "Data vs. Capacity" → البيانات مقابل السعة; "Hardware Scaling" → تحجيم العتاد; "Macro Architecture" → المعمارية الكلية; "Structural Upgrades" → ترقيات بنيوية; "Optimization & Regularization" → التحسين والتنظيم; "Model Definition" → تعريف النموذج; "Shape Verification" → التحقق من الأشكال; "Resource Accounting" → محاسبة الموارد; "Systems Bottleneck" → عنق زجاجة في الأنظمة; slide "Catalyst N: …" → المحفز N: …; "cosine warmup" on the cover → جدول جيب التمام مع الإحماء.
- English glosses given in u01 of this page (do not repeat in later units): bags of visual words, data augmentation, multiply-adds, graphics processing units (GPUs), local response normalization, dropout. top-5 error keeps its Latin "top-5" and takes no gloss; warmup is defined in a later unit.

## chapter_convolutional-modern/cnn-design.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "design space" → فضاء التصميم (فضاءات التصميم), glossed (design spaces) once; "the AnyNet design space" → فضاء التصميم AnyNet; "population" (of sampled networks) → مجتمع (مجتمعات من الشبكات), as the statistical population on the generalization pages
- "neural architecture search (NAS)" → البحث عن المعمارية العصبية (neural architecture search, NAS), then NAS
- "bottleneck ratio" → نسبة عنق الزجاجة; "bottleneck width" → عرض عنق الزجاجة; "group width" → عرض المجموعة; "group count" → عدد المجموعات; "inverted bottleneck" → عنق زجاجة معكوس
- "stem / body / head / stage" → الجذع / الجسم / الرأس / المرحلة (earlier pages), glossed once here because the page defines them; "stem--body--head template" → القالب ثلاثي الأجزاء، جذع وجسم ورأس; "macro template / layout" → القالب الكلي / المخطط الكلي
- "residual shortcut" → وصلة تخطٍ ذات بواقٍ; "project the shortcut", "project $c_i$ channels to …" → يسقط
- "exhaustive search" → البحث الاستقصائي; "brute force" → القوة الغاشمة; "point solution" → حل نقطي
- "stochastic dominance" → الهيمنة العشوائية; "error CDF" on slides → دالة CDF للخطأ (دوال CDF); "refinement" (of a design space) → تنقيح
- "half-cosine" → نصف دورة جيب تمام; "Recipe A" → الوصفة A; "training protocol" → بروتوكول التدريب
- "multi-scale pyramid" → هرم متعدد المقاييس; "segmentation heads" → رؤوس التجزئة; "downstream" (tasks) → المهام اللاحقة
- Slide bullets whose bold lead-in is a math transition ($\text{AnyNetX}_A \to \text{AnyNetX}_B$) start with الانتقال so that the block opens with an Arabic word

## chapter_convolutional-modern/convnext.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "roadmap" → خارطة الطريق; "modernization" → التحديث ("The Modernization Roadmap" → خارطة طريق التحديث; "modernizations" → تحديثات); the page title takes the class noun: شبكة ConvNeXt: ….
- "ablation" → دراسة الاستئصال (ablation), short الاستئصال; "controlled ablation" → استئصال مضبوط; "cumulative ablation" → استئصال تراكمي; "controlled experiment" → تجربة محكمة (as logged).
- "top-1 accuracy" → دقة top-1 (parallel to معدل خطأ top-5); table head "Top-1" → دقة top-1.
- "leaderboard" → لوحة المتصدرين; "pre-training set" → مجموعة تدريب مسبق; "obsolete" → عفا عليه الزمن.
- "stage ratio" → نسبة المراحل; "patchify stem" → جذع التقطيع إلى رقع; "patch stem" → جذع رقع; "patch-merging layers" → طبقات دمج الرقع; "token vector" → متجه رمز مميز.
- "inverted bottleneck" → عنق الزجاجة المعكوس (أعناق الزجاجة المعكوسة); "layer scale" → مقياس الطبقة; "separate downsampling (layers)" → خفض أخذ العينات المنفصل / طبقات منفصلة لخفض أخذ العينات.
- "macro design" → التصميم الكلي ("macro structure" → البنية الكلية); "micro design" → التصميم التفصيلي.
- "activations" meaning activation functions → دوال التفعيل; "normalizations", "normalizer" → طبقات التطبيع، طبقة التطبيع.
- "sublayer" → طبقة فرعية; "pointwise MLP" → شبكة MLP نقطية; "hierarchical structure" → البنية التراتبية; "grid of patches" → رقع مرتبة في شبكة نقطية.
- "spatial filtering" (by a convolution filter) → الترشيح المكاني (التصفية stays for filtering a list); "nominal receptive field" → مجال الاستقبال الاسمي; "dense" convolution or layer as opposed to depthwise → كثيف, as logged.
- "design hygiene", "Block Hygiene", "Micro Hygiene" → تهذيب التصميم، تهذيب الكتلة، التهذيب التفصيلي (not نظافة).
- "confounded" → خلط أثر … بأثر …; slide "Confounded variables" → متغيرات متداخلة; "paradigms" → النهجان; "primitives" (building blocks) → لبنات; "thumbnails" → الصور المصغرة.
- "stem--body--head" → جذع وجسم ورأس; "linear head" → الرأس الخطي; "backbone" → الهيكل (as logged).
- Table cells "ReLU $\to$ GELU" and "BN $\to$ LN" are copied as they are (one left-to-right run); "LN" in a table takes طبقة.
- English glosses given in u01 of this page (do not repeat in later units): attention, vision transformer (ViT), pointwise, downsampling, patch stem, ablation, patchify, depthwise, inverted bottleneck, effective receptive field, layer normalization.

## chapter_convolutional-modern/blocks.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "classification head" → رأس التصنيف (glossed (classification head) once); "fully connected head" → الرأس كامل الاتصال; "dense head" → الرأس الكثيف (as in the decisions log)
- "convolutional trunk" → الهيكل الالتفافي (glossed (trunk) once); "body" of the network (stem–body–head) → جسم الشبكة; "stem–body–head layout" → تخطيط من جذع وجسم ورأس
- "multiply-adds" → عمليات الضرب والجمع, glossed (multiply-adds) once on this page; after a number: N عملية ضرب وجمع; slide "MACs" → عمليات MAC
- "cosine schedule" → جدول جيب التمام; "warmup-and-cosine schedule", "cosine warmup schedule" → جدول الإحماء وجيب التمام
- "peak learning rate" → ذروة معدل التعلم ("at a peak of 0.1" → عند ذروة مقدارها 0.1)
- "quarter-width VGG" → VGG بربع العرض; "channel width" → عرض القنوات; "channel counts" → أعداد القنوات
- "VGG block" → كتلة VGG; "repeatable blocks" → كتل قابلة للتكرار; "reusable" → قابل لإعادة الاستخدام; "modular" (code, data) → تركيبي
- "parameterized family" → عائلة معلَمية; "parameterized layers" (layers that have weights) → طبقات ذات معلَمات; "variant" (of a network) → نسخة
- "landmark architectures" → معماريات فارقة; "bloated" (head) → متضخم; "class scores" → درجات الفئات; "in parallel" → على التوازي; "side by side" → جنبًا إلى جنب
- "hand-wired code" → شيفرة موصولة يدويًا; "ad-hoc" (layer lists) → ارتجالي; "systematic" → منهجي
- "He-normal weights" → أوزان من تهيئة He الطبيعية; "unseeded initializer" → مهيئ لا بذرة له
- "stalling at chance" → تعلق عند مستوى الصدفة; "validation trajectory" → مسار دقة التحقق
- "Blocks regularized the body" (made it regular, not regularization) → أضفت الكتل الانتظام على جسم الشبكة
- Headings and slide kickers that begin with a model name get the class noun: "VGG: …" → شبكة VGG: …; slide kickers "Resource Ledger" → سجل الموارد, "Experimental Evidence" → الدليل التجريبي

## chapter_convolutional-modern/resnet.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- Headings of model names take the class noun: "ResNet-18" → شبكة ResNet-18; in running text ResNet-18, ResNet-34 stay bare unless a block or bold lead-in would start with them.
- "plain block" → الكتلة العادية, glossed (plain block); "bypass addition" → الجمع عبر وصلة التخطي; "matched pair" (of networks) → الزوج المتطابق; "paired experiment" → تجربة الأزواج; "controlled test" → اختبار مضبوط.
- "stem / body / head" → الجذع / الجسم / الرأس, "stage" → مرحلة, each glossed once on this page; "stem--body--head architecture / template" → المعمارية المؤلفة من جذع وجسم ورأس / القالب المؤلف من جذع وجسم ورأس; "entry module" (the stem) → الجزء الافتتاحي (وحدة stays the Python module); "projection block" → كتلة إسقاط; "pooled head" → رأس تجميعي; "linear classifier" → مصنف خطي.
- "configurations" (the four depth-and-shortcut settings) → الإعدادات الأربعة; "training histories" → سجلات التدريب; "runs" → جولات; "logarithmic y-axis" → محور عمودي لوغاريتمي; curve colours and styles on slides → (الأحمر المتقطع)، (الأزرق المتقطع)، (الأحمر المتصل).
- "weight layer" (a layer counted because it has weights) → طبقة ذات أوزان; "parameterized layers" → طبقات ذات معلَمات; "trunk" (main path of a residual network) → المسار الرئيسي.
- "bottleneck block" → كتلة عنق الزجاجة, glossed (bottleneck); the elliptic list "ResNet-50, -101, and -152" is written out as ResNet-50 و ResNet-101 و ResNet-152, since a bare "-101" is unreadable in right-to-left text.
- "cardinality" (ResNeXt: the number of groups) → الأصلية (the set-theory word, as the paper borrows it); kicker "ResNeXt & Cardinality" → شبكة ResNeXt والأصلية.
- "quarter-width VGG" → شبكة VGG بربع العرض; "over-interpret a comparison" → يحمّل المقارنة أكثر مما تحتمل; "cross-family comparison" → مقارنة بين عائلتين; "memorization gap" → فجوة حفظ; kicker "Experimental Evidence" → الدليل التجريبي; "Reproducing Degradation" → إعادة إنتاج التدهور; "Stage Template & Shapes" → قالب المرحلة والأشكال.
- MXNet tab: "`hybridize()` compiles it into a single graph" → تصرّفها في مخطط واحد; "one operator at a time" → تنفيذ العمليات واحدةً تلو الأخرى; "the Trainer" → الصنف Trainer.
- English glosses given in u02 of this page (do not repeat in later units): plain block, stem, body, stages, head, bottleneck.

## chapter_convolutional-modern/training-recipes.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- Page title "Modern Training Recipes for Convnets" → وصفات التدريب الحديثة للشبكات الالتفافية; "ingredient" (of a recipe) → مكون (مكونات); "training practices" → ممارسات التدريب; "outdated recipe" → وصفة متقادمة; "recipe menu" → قائمة الوصفة.
- "early 2020s" → مطلع عشرينيات القرن الحادي والعشرين (decades in words, style §6); "the 2015 baseline" (a recipe) → الوصفة المرجعية لعام 2015; "baseline" (a network compared against) → نموذج مرجعي.
- "step-wise learning-rate decay" → اضمحلال معدل التعلم على درجات; "step decay" → اضمحلال درجي; "step schedule" → جدول درجي; "drop" (of the rate) → الخفض (حقب الخفض، عتبات الخفض); table row label "Schedule" → جدول معدل التعلم (so that it is not read as الجدول "table").
- "half-cosine decay" → اضمحلال جيب التمام بنصف دورة (keeps the glossary's اضمحلال جيب التمام; body text: اضمحلال أملس بنصف دورة من جيب التمام); "peak rate" → ذروة المعدل; the EMA "decay $\rho$" → معامل التناقص (as on the batch-norm page).
- "horizontal flips" → الانعكاس الأفقي; "random erasing" → المحو العشوائي; "evaluated weights" → الأوزان المقيَّمة.
- "hard target" → هدف قاطع; "softened / soft target distribution" → توزيع ملين / توزيع مستهدف لين; "smoothed / unsmoothed cross-entropy" → الإنتروبيا المتقاطعة المنعمة / غير المنعمة; "finite logits / margin" → لوجيتات منتهية / هامش منته; "overconfident" → مفرط الثقة (فرط الثقة، الإفراط في الثقة); "poorly calibrated" → سيئ المعايرة.
- Mixup, CutMix and RandAugment take the feminine class noun طريقة; "mixing coefficient" → معامل المزج; "to mix / blend" → يمزج; "superposed mixtures" → نواتج المزج المتراكبة; "shuffled partner" → شريك مخلوط; "ghostly double exposures" → صور شبحية مزدوجة التعريض; "alpha-blending" → مزج ألفا.
- "to drop" (a residual branch, an example) → يحذف, "drop probability" → احتمال الحذف, as الحذف العشوائي; "bypass" → يتجاوز; "per-example masking" → التقنيع لكل مثال; "ensemble of shallower subnetworks" → مجموعة نماذج من شبكات جزئية أقل عمقًا.
- "stochastic weight averaging (SWA)" → حساب متوسط الأوزان العشوائي; "exponential weight averaging" → المتوسط الأسي للأوزان; "shadow copy" → نسخة ظل, "shadow weight" → وزن الظل; "snapshots" → لقطات; "averaging horizon" → أفق حساب المتوسط, "memory horizon" → أفق الذاكرة; "low-loss basin" → حوض منخفض الخسارة; "recursion" → علاقة تكرارية; "tradeoff" → المفاضلة; "to damp" → يخمد; "ramp" → المنحدر.
- Slides: "11.2M ResNet-18" → شبكة ResNet-18 ذات 11.2 مليون معلَمة; "ep" → حقبة; "1-hot CE" → خسارة CE بالترميز الأحادي; "Head-to-Head Comparison" → مقارنة مباشرة; "Optimizer Mechanics" → آلية عمل المحسن; "Iterate Averaging" → حساب متوسط نواتج التكرار.
- English glosses given in u01 of this page (do not repeat in later units): training recipe, data augmentation, one-hot, label smoothing, warmup, decay, stochastic depth, checkpoint, stochastic weight averaging (SWA), exponential moving average (EMA), shadow copy.

## chapter_convolutional-modern/batch-norm.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "instance normalization" → تطبيع المثيلات (no glossary entry; parallel to تطبيع الدفعات / الطبقات / المجموعات); slide acronyms LN, GN, IN stay Latin in parentheses after the Arabic name.
- "internal covariate shift" → انزياح المتغيرات المصاحبة الداخلي (glossary انزياح المتغيرات المصاحبة), glossed once in u02.
- "the Trainer" (capitalized, without backticks) in tab notes → الصنف Trainer; "flag" (training-mode flag) → راية; "training / prediction view of the model" → عرض للنموذج خاص بالتدريب / بالتنبؤ.
- "warmup-and-cosine schedule" (short form) → جدول الإحماء وجيب التمام; "warming up over the first tenth" → بإحماء يمتد على العُشر الأول; "decay the learning rate to zero" → يجعل معدل التعلم يضمحل إلى الصفر; "anneals to zero" → يتناقص تدريجيًا إلى الصفر (slides: الخفض التدريجي).
- "learning-rate sweep" → مسح (نمسح قيم ذروة معدل التعلم); "band" of learning rates → نطاق; "a 300-fold span" → مدى يتسع 300 ضعف; "within a factor of three" → ضمن معامل مقداره ثلاثة.
- "stalls at chance" → تتعثر عند مستوى الصدفة; "erratic" (training) → يضطرب سلوكها، تضطرب الجولات; "clears 70%" → يتخطى 70%; "stale snapshot" → لقطة قديمة; "running averages lag / lock onto" → تتأخر / تثبت على.
- "single-precision roundoff" → خطأ التدوير في الدقة المفردة; "representational capacity" → السعة التمثيلية; "micro-batch" in the sense of a very small batch ($m \le 2$) → الدفعات بالغة الصغر (دفعة جزئية stays the gradient-accumulation micro-batch).
- "normalizer" on slides → طبقة تطبيع ("Single-Example Normalizers" → طبقات التطبيع على المثال الواحد); "the module" meaning the BatchNorm layer itself → الطبقة.
- English glosses given in u02 of this page (do not repeat in later units): buffers, internal covariate shift.

## chapter_convolutional-modern/efficient-convnets.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "compound scaling" → التحجيم المركب, glossed once here; "neural architecture search (NAS)" → البحث عن المعمارية العصبية (neural architecture search, NAS), then البحث عن المعمارية / NAS; "squeeze-and-excitation (SE) gate" → بوابة الضغط والإثارة (squeeze-and-excitation, SE), then بوابة SE; "squeezes" → تضغط, "the excitation" → الإثارة; "SE module / block" → وحدة SE / كتلة SE.
- "reduction ratio" $r$ → نسبة التخفيض; "reweight channels" → يعيد ترجيح القنوات; "channel-wise" → على مستوى القنوات; "global image context" → السياق الشامل للصورة; "predicted gate" → قيمة البوابة المتنبأ بها; "bottleneck MLP" → شبكة MLP على هيئة عنق زجاجة; "gating MLP" → شبكة MLP للبوابات.
- "to fuse / fusion" (branches into one convolution) → يدمج / الدمج, "fused kernel / bias / weights" → النواة المدمجة / الانحياز المدمج / الأوزان المدمجة; "to fold / BN folding" → يضم / ضم BN (ضم تطبيع الدفعات), "folded kernels / biases" → النوى / الانحيازات الناتجة عن الضم (as the log's الانحياز الناتج عن الضم); "collapsing" the branches into one layer → اختزال … في.
- "training topology / inference topology" → طوبولوجيا التدريب / طوبولوجيا الاستدلال; "single-path" → وحيد المسار; "plain sequential stack" → المجموعة المكدسة العادية المتتابعة; "the best of both worlds" → أفضل ما في العالمين; "train-time block / deploy-time fusion" → كتلة التدريب / الدمج عند النشر.
- "running mean / variance / stats" on this page → المتوسط المتحرك / التباين المتحرك / الإحصاءات المتحركة (as on the batch-norm page); "frozen running stats" → الإحصاءات المتحركة المجمدة; "center tap" → الوزن المركزي (log: taps → أوزان النواة); "identity kernel" → النواة المحايدة; "zero-padded kernel" → نواة محشوة بالأصفار.
- "live activation buffer / tensor" → مخزن مؤقت حي للتفعيلات / موتر تفعيلات حي; "systems overhead" → عبء إضافي على مستوى الأنظمة; "hardware utilization" → استغلال العتاد; "inference compilers" → مصرِّفات الاستدلال; "edge accelerators" → المسرعات الطرفية.
- "8-bit integers" → أعداد صحيحة من 8 بتات; "levels" (of a quantizer) → مستويات, "bins" → خانات; "heavy-tailed dynamic range / outliers" → مدى ديناميكي ثقيل الذيل / قيم شاذة ثقيلة الذيل; "post-training quantization" → التكميم بعد التدريب; "naive quantization" → التكميم الساذج; "low-precision quantization" → التكميم منخفض الدقة; "exact (functional) equivalence" → التكافؤ (الدالي) المضبوط; QAT, FP32, INT8, TF32 stay Latin.
- "in lockstep" → بخطى متلازمة; "integer knob" → مقبض قيمته عدد صحيح; "thumbnails" → صور مصغرة; "core primitives" → العناصر الأولية الأساسية; "measured training speed" → سرعة التدريب المقيسة; "search objective" → دالة هدف البحث; "base architecture / network" → المعمارية الأساسية / الشبكة الأساسية.
- Slides: "Work out these …" → المطلوب حل هذه المسائل …; "compare" in a concept-check item → المطلوب مقارنة …; table headers "Technique" → التقنية, "Mathematical / Structural Form" → الصيغة الرياضية / البنيوية, "Efficiency Gain & Practical Caveat" → مكسب الكفاءة والتنبيه العملي.
- English glosses given in u02 of this page (do not repeat in later units): compound scaling, neural architecture search (NAS), squeeze-and-excitation (SE).

## chapter_convolutional-modern/alexnet.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "warmup" → الإحماء (glossary), glossed once here (*إحماء* (warmup) خطي); "linear warmup" → إحماء خطي; "peak learning rate", "peak rate" → ذروة معدل التعلم / ذروة المعدل (as on the blocks page); "half-cosine decay" → اضمحلال على نصف دورة من جيب التمام (slide: اضمحلال جيب التمام بنصف دورة); "cosine schedule" → جدول جيب التمام; "cosine warmup schedule" → جدول الإحماء وجيب التمام; "cosine term" → حد جيب التمام.
- "multiplicative schedule factor" → العامل الضربي للجدول; "hands-free schedule" → جدول لا يحتاج إلى تدخل يدوي; "anneal" (the learning rate) → يخفض تدريجيًا (slide: خفض سلس); "ramp" (of the rate) → يرفع، صعود خطي; "validation error plateaued" → استقر خطأ التحقق; "bouncing around a minimum" → تتأرجح حول نقطة صغرى.
- "to upsample" → يكبّر بزيادة أخذ العينات (upsampling glossed once here); "upsampling 64-fold / eightfold" → زيادة أخذ العينات 64 مرة / ثماني مرات (slides keep $64\times$); "resized dataset" → مجموعة البيانات التي غُيّر حجم صورها; "to survive" (layers, said of small images) → تصمد أمام.
- "fit `AlexNet` on the dataset" → يدرب (the log's يلائم stays for fitting a model or curve to data); "snapshot" (the initial kernels) → يحفظ نسخة من; "compute profile" → توزيع الكلفة الحسابية; "calibrate expectations" → نضبط توقعاتنا; "edges ahead" → تتقدم.
- "weight updates" → تحديثات الأوزان; "weight difference" → فرق في الأوزان; "structured" (updates) → ذات بنية; "brightness gradients / ramps" → تدرجات سطوع / منحدرات سطوع; "oriented bars" → أشرطة موجهة; "low-frequency counterparts" → نظائر منخفضة التردد; "crisp edge filters" → مرشحات الحواف الحادة; "grayscale range" → مجال من تدرج الرمادي; "spatial statistics" → الإحصاءات المكانية; "relative displacement" → إزاحة نسبية; "He-normal" → تهيئة He الطبيعية.
- Slides: kickers "Optimization Schedule" → جدول التحسين, "Optimizer Configuration" → ضبط المحسن, "Experimental Evidence" → الدليل التجريبي, "Inspecting Weight Updates" → فحص تحديثات الأوزان; "diagnostic convenience" → تسهيل تشخيصي; "Summary: Key Takeaways from X" → الملخص: أهم الخلاصات من X; table header "Role & Systems Takeaway" → الدور والخلاصة من منظور الأنظمة; "MACs/byte" → عملية MAC لكل بايت; "float32 parameters" → معلَمة float32; concept-check lead-in "Work out these two questions" → المطلوب حل هذين السؤالين.
- English glosses given in u02 of this page: upsampling, warmup.

## chapter_convolutional-modern/convnext.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "mixin" → صنف مزيج (mixin), glossed once; slide title: الصنف المزيج. No glossary entry.
- "channels-first" → تخطيط القنوات أولًا (parallel to the logged تخطيط القنوات أخيرًا); "permutes to channels-last and back" → يبدل ترتيب المحاور إلى … ثم يعيده.
- "training stabilizers" → عاملا استقرار للتدريب (as the logged عوامل استقرار); "drop probabilities" (stochastic depth) → احتمالات الحذف; slide "drop-path" → حذف المسار; LayerScale and DropPath stay Latin on slides (مقياس LayerScale).
- "the compact ConvNeXt-atto profile" → الصيغة المصغرة ConvNeXt-atto; "compact" (roadmap, ResNet-18) → مصغر; the "compact modern recipe" of the training-recipes page → الوصفة الحديثة المختصرة (متراص stays the mathematical sense).
- "parameter-matched" → المساوي في عدد المعلَمات; "control" (model in an experiment) → نموذج ضابط; "baseline" model → نموذج مرجعي (as logged); "narrowed" → ضُيّقت; "the two families" (of architectures) → العائلتان.
- "mixed training step" (Mixup) → خطوة التدريب الممزوجة; "toggling `norm`" → تبديل قيمة `norm`; "order-independent" → لا تتوقف على الترتيب.
- Parameter counts: body text "3.35M parameters" → 3.35 مليون معلَمة; table cells keep 3.35M and slides keep the math span $3.38\textrm{M}$.
- "neutral" (a step with no effect on accuracy) → محايد الأثر; "transfer" (design choices to another domain) → ينتقل; "low-resolution domains" → المجالات منخفضة دقة الصورة.
- Slides: "Backbone Implementation" → تنفيذ الهيكل; "Fair Ablation" → استئصال منصف; "Controlled Training Protocol" → بروتوكول تدريب مضبوط; "Local Modernization Roadmap" → خارطة طريق التحديث المحلية; "Frontiers" → الآفاق; "Open Questions" → أسئلة مفتوحة; "Design Dimension" → بُعد التصميم; "Impact & Principle" → الأثر والمبدأ.
- "masked pretraining" → التدريب المسبق المقنع (glossary مقنع); "sparse MAE pretraining" → تدريب مسبق متناثر بطريقة MAE; "submanifold sparse convs" → الالتفافات المتناثرة على المتشعبات الجزئية; "Global Response Norm (GRN)" → تطبيع الاستجابة الشامل (GRN), parallel to تطبيع الاستجابة المحلية; "dead/collapsed channels" → القنوات الميتة/المنهارة.
- "step decay" (learning-rate schedule) → اضمحلال درجي; "crops/flips" → مقاطع/انعكاسات; "deterministic multiplier" → عامل ضربي حتمي; "random gate" → بوابة عشوائية; "at eval" → عند التقييم.
- English glosses given in u02 of this page (do not repeat in later units): layer scale, stochastic depth, mixin.

## chapter_convolutional-modern/blocks.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "Inception block" → كتلة Inception (Latin name after the class noun; no gloss); "branch widths" → عروض الفروع; "four-branch" → رباعي الفروع; "multi-branch blocks" → الكتل متعددة الفروع.
- "channel reduction" ($1 \times 1$) → خفض القنوات (slides: خفض $1\times 1$), parallel to خفض الأبعاد; "project 192 channels down to 16" → إسقاط … إلى; "narrow projection" → إسقاط ضيق.
- "pooled head" → رأس تجميعي; "pooling head" → رأس التجميع; "global-pooling head" → رأس التجميع الشامل; "convolutional body" → الجسم الالتفافي.
- "standard convolution" → الالتفاف المعتاد (قياسي is "scalar"); "thresholded linear filter" → مرشح خطي ذو عتبة; "per-pixel channel mixing" → مزج القنوات عند كل بكسل.
- "score maps" → خرائط درجات; "collapses to ten logits" → يطويها إلى عشرة لوجيتات (as logged for "collapse"); a training run that "collapses" → ينهار; "vanish identically" → ينعدم انعدامًا تامًا; "killing a class", "dead class" → موت الفئة، الفئة الميتة; "clamped to 0" → تُقصر على 0.
- "invariant to permutations" → ثابت تحت التبديل; "objects" in images → الأجسام; "up to border and striding effects" → بصرف النظر عن آثار الحدود وخطوة الانزلاق.
- "primitives" → لبنات (as logged); "tuning knobs" → مقابض ضبط; "stem--body--head blueprint" → المخطط … جذع وجسم ورأس, slide "Macro Blueprint" → المخطط الكلي, "macro template" → القالب الكلي; "vision backbones" → هياكل الرؤية; "3-part backbone layout" → تخطيط الهيكل ثلاثي الأجزاء.
- "its score" (a network's accuracy result) → نتيجتها; "override `VGG.head`" → نعيد تعريف; "matched peak learning rate" → ذروة معدل التعلم المتطابقة; "runs" → جولات (as logged).
- The meme "we need to go deeper" → “علينا أن نتعمق أكثر” (we need to go deeper): Arabic in curly quotes with the English original once in parentheses, since the phrase is known in English.
- Slides: "(result)" in a slide title → (النتيجة); "Head Ablation" → استئصال الرأس; "weight layers" → طبقات ذات أوزان; "1024-vector" → متجه من 1024 عنصرًا; "pixelwise channel features" → ميزات القنوات عند كل بكسل; "conv", "ch", "reduction" in list items → التفاف، قناة، خفض; "Same" padding → الحشو “same”; table headers Model / Input / Params / MACs / Peak / Val Acc → النموذج / المدخلات / المعلَمات / عمليات MAC / ذروة / دقة التحقق; "LeNet (30 ep)" → LeNet (30 حقبة); "Quarter-VGG" → VGG بربع العرض; table cells keep 0.06M, 938M as written.
- English glosses given in u02 of this page (do not repeat in later units): stem, body. Network in Network, global average pooling, bottleneck, classification head, multiply-adds and receptive field were glossed in u01; the name Inception takes none.

## chapter_convolutional-modern/alexnet.md#u03 (claude fable-5.2 (medium), 2026-11-27)

- Exercise titles: "Memory and compute ledger" → سجل الذاكرة والحساب (as the slide title سجل المعلَمات والحساب); "parameter memory" → ذاكرة المعلَمات; "parameter count" (the act, "repeat the parameter count") → إحصاء المعلَمات.
- "compute throughput" → إنتاجية الحوسبة; "single-image inference" → الاستدلال على صورة واحدة; "compute time / parameter-read time" → زمن حسابها / زمن قراءة معلَماتها; "takes the maximum of" (two times) → تستغرق أكبر الزمنين.
- "throughput (images/second)" → الإنتاجية (بالصور في الثانية), as on the minibatch page; "throughput plateaus" → تستقر الإنتاجية; "trains without running out of memory" → يجري معه التدريب دون نفاد الذاكرة; "starting from … and doubling" → ابدأ بـ … وضاعفه مرة بعد مرة; "gap against your prediction" → فرق عن تنبؤك.
- "the ImageNet version" (of the network) → صيغة ImageNet; "fraction of the input side … spans" → أي كسر من ضلع المدخلات … يغطيه.

## chapter_convolutional-modern/efficient-convnets.md#u03 (claude fable-5.2 (medium), 2026-11-27)

- "width multiplier" → مضاعف العرض (no glossary entry; مضاعف as in مضاعف لاغرانج); "scales every stage's channel count by $m$" → يضرب عدد قنوات كل مرحلة في $m$.
- "channel permutation" (ShuffleNet) → تبديل القنوات; "intermediate channels" → القنوات الوسيطة.
- "first-class design objectives" → أهداف التصميم الأساسية (ترفع … إلى مصاف); "peak activation memory" → ذروة ذاكرة التفعيلات; "bottlenecked by memory bandwidth" → يخنقها عرض نطاق الذاكرة; "arithmetic budget" → ميزانية العمليات الحسابية.
- "inference engine" → محرك استدلال; "to tile" (a computation channel-by-channel) → يقسم إلى بلاطات; "materialize" (a tensor in RAM) → ينشئ … كاملًا; "expanded tensor" → الموتر الموسع (as u01's الفضاء الموسع); "expansion by factor $t$" → طبقة توسعة بعامل $t$.
- "symmetric per-tensor scaling" → تحجيم متناظر لكل موتر; "deployed networks" → الشبكتان المنشورتان; "accuracy drop" → انخفاض الدقة; "weight-distribution analysis" → تحليل توزيع الأوزان; "inference mode" → نمط الاستدلال (as in u02).
- "population-level architecture design spaces" → فضاءات تصميم المعماريات على مستوى مجتمعات الشبكات; "single-model search" → البحث عن نموذج واحد.
- No English glosses in u03 of this page (summary and exercises; all key terms were glossed in u01 and u02).

## chapter_convolutional-modern/resnet.md#u03 (claude fable-5.2 (medium), 2026-11-27)

- Headings: "ResNeXt and DenseNet" → شبكتا ResNeXt و DenseNet; "ResNeXt", "DenseNet" → شبكة ResNeXt، شبكة DenseNet (class noun, as for ResNet-18); in running text both are feminine and bare.
- "grouped convolution" → التفاف المجموعات (glossary), glossed here; "grouped $3 \times 3$ convolution / layer" → التفاف مجموعات $3 \times 3$ / طبقة المجموعات $3 \times 3$; "dense $1 \times 1$ convolutions" → التفافان كثيفان; "uniform structure" → بنية متجانسة; "topology" → الطوبولوجيا; "multi-branch" → متعدد الفروع (as logged).
- "wall-clock speedup" → التسريع في الزمن الفعلي; "memory traffic" → حركة البيانات في الذاكرة; "trade" (wider at matched cost) → المقايضة; "top-1 error" → خطأ top-1; "successors" (of an architecture) → معماريتان خلفتاها.
- DenseNet: "dense block" → الكتلة الكثيفة (as logged); "growth rate" → معدل النمو; "transition layer" → طبقة انتقال; "channel(-wise) concatenation" → ربط القنوات / الربط على امتداد القنوات; "concatenative skip connections" → وصلات التخطي بالربط; "feature reuse" → إعادة استخدام الميزات; "distinct feature maps" → خرائط الميزات المتمايزة; "quadratically" → تربيعيًا; "overwritten" → يُمحى بالكتابة فوقه.
- "pre-activation ordering" (BN → ReLU → conv; not the value before the activation, which stays قيمة ما قبل التفعيل) → ترتيب التفعيل المسبق; "pre-activation ResNet" → شبكة ResNet ذات التفعيل المسبق; "post-add ReLU" → دالة ReLU التي بعد الجمع.
- "U-Net backbone" → هيكل U-Net; "backbone of nearly every modern architecture" (figurative) → عماد; "universal default" → الخيار الافتراضي العام; "spatial scale" → مقياس مكاني; "upsampled feature map" → خريطة ميزات مكبَّرة.
- Slides: "ResNeXt Code" → شيفرة ResNeXt; "DenseNet & Skip Topologies" → شبكة DenseNet وطوبولوجيات التخطي; "Variants & Extensions" → نسخ وامتدادات; "Memory Pitfall" → مزلق الذاكرة; "naive DenseNet" → تنفيذ DenseNet الساذج; "Summary: Key Takeaways on …" → الملخص: أهم الخلاصات عن …; "Work out these three design calculations" → المطلوب إجراء حسابات التصميم الثلاثة التالية; "AvgPool" stays Latin after طبقة; "parameter-free" → لا معلَمات لها.
- English glosses given in u03 of this page (do not repeat in later units): grouped convolution, dense convolutional network, pre-activation, growth rate, transition layer.

## chapter_convolutional-modern/cnn-design.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "empirical cumulative distribution function (CDF)" → دالة التوزيع التراكمي التجريبية, glossed (empirical cumulative distribution function, CDF) once; afterwards دالة CDF (دالتا CDF، دوال CDF), "empirical CDF" → دالة CDF التجريبية, "CDF curve" → منحنى دالة CDF; slide "EDFs" → دوال EDF
- "working assumptions" → افتراضات عملية; "short-run ranking transfers" → الترتيب بعد تدريب قصير ينتقل; "error threshold" → عتبة خطأ; "lies uniformly above" / "uniform stochastic dominance" → فوق … عند كل عتبة / الهيمنة العشوائية عند كل عتبة
- "nondecreasing widths / depths" → عروض / أعماق غير متناقصة; "growing / shrinking widths" → العروض المتزايدة / المتناقصة; "stage widths" → عروض المراحل; "monotonic stages" → مراحل رتيبة; "mean population error" → متوسط خطأ المجتمع
- "linear ramp" (of block widths) → منحدر خطي; "staircase" (of quantized widths) → درج (درج منتظم، الدرج المكمَّم); "snaps to" → يُقرَّب إلى; "geometric grid" → شبكة نقطية هندسية; "quantized linear widths" → العروض الخطية المكمَّمة (as on the u01 cover)
- "multiply-adds" → عمليات الضرب والجمع, no gloss on this page (glossed on the earlier pages of the chapter); "400 MFLOPs" copied; "112M" in body text → 112 مليون, in slide kickers 112 مليون عملية MAC
- "at matched compute" → عند تساوي العمليات الحسابية; "to match compute budgets" → يوحد الميزانيات الحسابية; "compute-matched" (slide title) → متساوية في كلفة الحساب; "FLOP-matched" → متطابق في عمليات FLOP
- "legends" (of a plot) → مفاتيح الرسم; "upsampled images" → الصور المكبَّرة; "nuances" → اعتبارات دقيقة; "dataset ceiling" → سقف مجموعة البيانات, "plateaus at" → يستوي عند; "memory traffic" → حركة البيانات في الذاكرة; "GPU runtime" → زمن التشغيل على وحدة GPU; "sobering" → تدعو إلى التواضع
- "cardinality" (number of parallel groups, ResNeXt) → الأصلية; "outlier model" → نموذج هو قيمة شاذة; "lucky point sample" → عينة نقطية محظوظة
- Headings and slide labels: "RegNet" → شبكة RegNet; "RegNetX32 on Fashion-MNIST" → شبكة RegNetX32 على Fashion-MNIST; "Experimental Evidence" → الدليل التجريبي; "Matched-Protocol Baseline" → نموذج مرجعي بالبروتوكول نفسه; "FLOP Budget Accounting" → محاسبة ميزانية عمليات FLOP; "Scientific takeaway" → الخلاصة العلمية; in running text `RegNetX32` is feminine when it stands alone (شبكة) and masculine after النموذج
- English glosses given in u02 of this page (do not repeat in later units): empirical cumulative distribution function (CDF)

## chapter_convolutional-modern/batch-norm.md#u03 (claude fable-5.2 (medium), 2026-11-27)

- "normalizer-free networks (NFNets)" → الشبكات الخالية من طبقات التطبيع (normalizer-free networks, NFNets), then شبكات NFNets; "adaptive gradient clipping" → قص التدرج التكيفي (glossary قص التدرج); "weight standardization" → توحيد الأوزان معياريًا.
- "state-of-the-art accuracy" → دقة في مستوى أحدث ما وصل إليه المجال; "top-1 error" → خطأ top-1 (as خطأ top-5).
- "this is :eqref: transposed" → هذا منقول …; "(second / rightmost) panel" of a figure → اللوحة (الثانية / في أقصى اليمين); "variant" (of a normalization layer) → نسخة.
- "interpolates between two extremes" (group normalization between $G=1$ and $G=c$) → يتدرج بين حالتين حديتين (يستوفي stays fitting data); "memory-heavy tasks" → المهام كثيرة استهلاك الذاكرة; "fine-tuning across domains" → ضبط دقيق على مجال آخر; "hard to debug" → صعب التنقيح.
- "train--test mode split", "train/eval split" → الانقسام بين نمطي التدريب والاختبار / بين التدريب والتقييم; "at inference time" → عند الاستدلال; "frozen" (averages, weights) → مجمدة.
- Exercises: "learning-rate ceiling" → سقف معدل التعلم; "horizon" (of the running average) → أفق; "training-mode passes overwrite" → تكتب التمريرات … فوق; "identically" ($= 0$) → تطابقًا; "to machine precision" → حتى دقة الآلة.
- Slides: "Takeaway" → الخلاصة; "Work out these … problems" → المطلوب حل هذه المسائل; "Eval Bugs" → أخطاء التقييم; "Alternative Scale Control" → ضبط بديل للمقياس; "folds into conv weights" → تُدمج في أوزان الالتفاف; BatchNorm, GroupNorm, LayerNorm, InstanceNorm, LN, IN in bold lead-ins take the class noun طبقة.
- English glosses given in u03 of this page: instance normalization, normalizer-free networks (NFNets).

## chapter_convolutional-modern/blocks.md#u03 (claude fable-5.2 (medium), 2026-11-27)

- Table rows of the comparison table take the class noun شبكة (شبكة LeNet، شبكة VGG بربع العرض، برأس تجميعي); cells keep 0.06M, 938M and the en-dash ranges as written; "(30 epochs)" → (30 حقبة); header "peak rate" → ذروة المعدل.
- Body text: "6.0M vs. 46.8M" → 6.0 مليون مقابل 46.8 مليون; "276M multiply-adds" → 276 مليون عملية ضرب وجمع; "~1.5B" → نحو 1.5 مليار; "144M-parameter VGG models" → نماذج VGG ذات 144 مليون معلَمة. A parenthesis joined by a math arrow (43.1M $\to$ 0.6M) is copied as one left-to-right run, as the logged table cells "ReLU $\to$ GELU".
- "raw arithmetic" → حجم العمل الحسابي وحده; "pixel budget" → ميزانية البكسلات; "slash" → يقلص تقليصًا شديدًا; "trades A for B" → يقايض A بـ B; "shared ceiling" → سقف مشترك; "auxiliary heads" → رؤوس مساعدة; "staged pretraining" → تدريب مسبق على مراحل; "scale fragility" → الهشاشة أمام تغير المقياس.
- "as ranges over repeated runs" → في صورة مدى على جولات متكررة; "agree to within half a point" → تتفق في حدود نصف نقطة; "layer counts" (per-layer tallies of parameters and multiply-adds) → إحصاءات الطبقات; "native resolution" → دقة الصورة الأصلية; "epoch time" → زمن الحقبة; "every stride-2 step" → كل عملية خطوة انزلاقها 2; "to tie" → تتعادل.
- Exercise titles: "Where the savings come from." → مصدر التوفير.; "Single-branch Inception ablation." → استئصال: كتلة Inception بفرع واحد.

## chapter_convolutional-modern/convnext.md#u03 (claude fable-5.2 (medium), 2026-11-27)

- "Limits and Open Problems" → الحدود والمسائل المفتوحة; "active frontiers" (of research) → آفاق بحثية نشطة (as the slide's الآفاق); "at frontier scale" → عند أحجام نماذج الصدارة (from the logged نماذج الصدارة).
- "masked autoencoders" → المشفرات التلقائية المقنعة (glossary مشفر تلقائي + مقنع), glossed once; "masked self-supervised pretraining" → التدريب المسبق المقنع ذاتي الإشراف; "masked holes" → الثغرات المقنعة; "bleed across" → يتسرب أثرها عبر.
- "sparse convolutions" → الالتفافات المتناثرة; "deformable convolutions" → الالتفافات القابلة للتشوه (no glossary entry); "ultra-large kernels" → نوى فائقة الكبر; "hybrid conv--attention stacks" → المجموعات المكدسة الهجينة من الالتفاف والانتباه; "sliding convolutions" → الالتفافات المنزلقة.
- "Global Response Normalization" → تطبيع الاستجابة الشامل, glossed once in body text; with the acronym: تطبيع GRN; "relative response" → الاستجابة النسبية; "residual" (the added $\mathbf{X}_c$ term) → وصلة بواقٍ.
- "Swin-inspired upgrades" → ترقيات مستوحاة من Swin; "enlarge a kernel", "kernel enlargement" → يكبر النواة، تكبير النواة; "the depthwise move" → نقل الالتفاف العمقي; enlarging "in place" → في موضعه; "net change" → التغير الصافي.
- "kernel size sweep" → مسح حجم النواة (as the logged مسح for a learning-rate sweep); "ablation square" → مربع الاستئصال; "architecture-by-recipe" → بين المعمارية والوصفة; "retune" → يعيد ضبط; "momentum SGD" → خوارزمية SGD مع الزخم.
- "asymptotic factor" → عامل تقاربي; "parameter-efficient" → كفء في استخدام المعلَمات (كفاءة استخدام المعلَمات); "evaluation-time network" → الشبكة المستخدمة عند التقييم; "(default)" after a value → (القيمة الافتراضية).
- Enumerations "(i) …, (ii) …" inside an exercise keep the Latin markers, joined by a free-standing و.
- English glosses given in u03 of this page: masked autoencoders, Global Response Normalization.

## chapter_convolutional-modern/training-recipes.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "compact modern recipe" → الوصفة الحديثة المختصرة (متراص stays the mathematical "compact"); "a lean subset" → مجموعة جزئية مقتصدة; "compact subset" → مجموعة جزئية مختصرة.
- "coupled weight decay" → اضمحلال أوزان مقترن (as the earlier التحديث المقترن); "10-fold drops" of the learning rate → قسمة المعدل على 10 (as u01).
- "step schedules" (computed per step from `lr` and `num_steps`) → جدول معدل التعلم خطوةً بخطوة; "Schedule" as a recipe ingredient in prose and slide titles → جدول معدل التعلم (not الجدول alone, which reads as "table").
- "paired run" → جولة مقترنة; "repeated runs" → جولات متكررة; "run-to-run standard deviation / noise floor" → الانحراف المعياري / أرضية الضوضاء من جولة إلى أخرى.
- "bundled comparisons" → المقارنات المجمّعة; "one-at-a-time ablation" → دراسة استئصال تغيّر مكونًا واحدًا في كل مرة (exercise title: استئصال مكونات الوصفة واحدًا واحدًا); ablation glossed (ablation) once on this page.
- "to retune" → يعيد ضبط (إعادة ضبط معدل التعلم); "optimizer swap" → تبديل المحسن; "to sweep" a hyperparameter in exercises → امسح.
- "multi-object ambiguity" → التباس تعدد الكائنات; "transfer to detection or segmentation" → النقل إلى الاكتشاف أو التجزئة; "downstream transfer" → النقل إلى المهام اللاحقة; "multi-label floor" → الأرضية متعددة الوسوم; "sub-point deltas" → فروق تقل عن نقطة.
- "drop-in variant" → بديل مباشر; "clipping the rectangle to the image borders" → حصر المستطيل داخل حدود الصورة (قص stays gradient clipping); "one-hot labels" → وسوم الترميز الأحادي.
- "effective horizon" of an EMA → الأفق الفعلي, "effective window" → النافذة الفعلية (as u01 and the batch-norm page; the glossary's أفق فعال is the RL discount horizon).
- "to extend" a class in an exercise → عمّم (يوسع stays dilation); "live weights" → الأوزان الحية; "BN buffers" → مخازن حالة BN; "frozen" → مع تجميد.
- Slides: "Recipe Implementation" → تنفيذ الوصفتين; "Baseline Trajectory" → المسار المرجعي; "Diagnostic Pitfall" → مزلق تشخيصي, "diagnostic trap" → فخ تشخيصي; "Benchmark Methodology" → منهجية الاختبارات المعيارية; table head "Mathematical Formulation / Practical Role & Tuning Rule" → الصياغة الرياضية / الدور العملي وقاعدة الضبط; summary-table row labels take a class noun (المحسن AdamW، طريقتا Mixup و CutMix، متوسط EMA للأوزان).

## chapter_convolutional-modern/cnn-design.md#u03 (claude fable-5.2 (medium), 2026-11-27)

- "parameterized template" → قالب معلَمي (as the logged عائلة معلَمية); "structural primitives" → اللبنات البنيوية; "activation normalization" → تطبيع التفعيلات; "the discipline of separating …" → الانضباط في فصل … عن …
- "at matched budget" (exercise title) → بميزانية متساوية; "at roughly matched parameter counts" → بأعداد معلَمات متقاربة; "time per epoch" → زمن الحقبة; "spatial feature-map resolution" → الدقة المكانية لخرائط الميزات
- "parameterize the block widths via the linear ramp" → يعبّر عن عروض الكتل بمعلَمات المنحدر الخطي; "rounded to the nearest multiple of 8" → مدوَّرة إلى أقرب مضاعف للعدد 8; "staircase approximation" → تقريب الدرج; "divisible by" → يقبل القسمة على; "from (a)" → في الجزء (a)
- "stage configuration" → إعداد المراحل; "measurable wall-clock time" → زمن فعلي يمكن قياسه; "violate a rule" / "violation" → يخرق قاعدة / خرق; "hand-picked network" → شبكة منتقاة يدويًا; "contrast your findings with" → قابل ما تجده بـ
- "expansion ratio" $t$ → نسبة التوسعة; "degrees of freedom" → درجات الحرية; "depthwise separability" → قابلية الفصل العمقية; "principles carry over to" → المبادئ تنتقل إلى; the exercise tags [code] and [extended] are copied
- No English glosses in u03 of this page (summary and exercises).

## chapter_convolutional-modern/resnet.md#u04 (claude fable-5.2 (medium), 2026-11-27)

- Exercises of this page: "identity shortcut" → وصلة تخطٍ محايدة (as u01); "untouched identity highway" → مسار محايد رئيسي لا يمسه شيء; "post-activation block" → كتلة ذات تفعيل لاحق, "pre-activation block" → كتلة ذات تفعيل مسبق / كتلة التفعيل المسبق; "standard" block → معتادة (قياسي stays "scalar").
- "multiply-adds per output pixel" → عمليات الضرب والجمع لكل بكسل من المخرجات; "512 input / output channels" → 512 قناة إدخال / قناة إخراج (glossary); "additive term" → حد جمعي; "layer index" → فهرس الطبقة; "the $i$-th convolution" → الالتفاف رقم $i$.
- "samples one of every four positions" → يأخذ عيناته من موضع واحد من كل أربعة مواضع; "ResNet-D downsampling shortcut" → وصلة التخطي عند خفض أخذ العينات في ResNet-D; "$2 \times 2$ stride-2 average pooling" → تجميع متوسط $2 \times 2$ بخطوة انزلاق 2 (as in u03); "stage depths" → أعماق المراحل; "catch up with" → تلحق بـ; "across several runs" → على امتداد عدة جولات; "time per epoch" → زمن الحقبة.
- "account for its parameters and activation memory" → احسب معلَماتها وذاكرة التفعيلات فيها; "assemble" (a network from parts) → ركّب; "to halve channels" → تنصّف عدد القنوات.
- No English glosses in u04 of this page (exercises take none).

## chapter_convolutional-modern: chapter check (claude fable-5.2 (medium), 2026-11-27)
- One rendering per term across the chapter's nine pages; the entries above were corrected in place.
- "efficient convnets" → الشبكات الالتفافية الكفؤة (the title of the page), also in the chapter introduction; "structural re-parameterization" → إعادة تحديد المعلَمات البنيوية.
- "stem--body--head" (layout, template, blueprint) → جذع وجسم ورأس (تخطيط من جذع وجسم ورأس، القالب ثلاثي الأجزاء، جذع وجسم ورأس); no dashes between the three words.
- "expansion" of channels (inverted bottleneck, ConvNeXt block) → توسعة (طبقة توسعة، نسبة التوسعة); "to widen" a stage or the channels → زيادة العرض; "to enlarge a kernel" → تكبير النواة; توسيع stays dilation.
- "shortcut", "skip connection" → وصلة تخطٍ / وصلة التخطي, written without a shadda; "pooled head" → رأس تجميعي; "quarter-width VGG" → VGG بربع العرض; "hand-wired" → موصول يدويًا; "unseeded initializer" → مهيئ لا بذرة له.
- "horizontal flips" (augmentation) → الانعكاس الأفقي (انعكاسات أفقية); القلب stays the flip of a kernel or a bit.
- "half-cosine decay" as a slide lead-in → اضمحلال جيب التمام بنصف دورة; body text may write اضمحلال على نصف دورة من جيب التمام.
- "compact modern recipe" → الوصفة الحديثة المختصرة on every page; مصغر stays for a compact network or roadmap.
- "the Trainer" (capitalized, in tab notes) → الصنف Trainer; slide kicker "Experimental Evidence" → الدليل التجريبي; concept-check lead-in "Work out these …" → المطلوب حل … / المطلوب إجراء …, without a colon after المطلوب.
- Kept apart on purpose: "parameter-matched" is المطابق في المعلَمات where it is paired with المطابق في عمليات FLOP (efficient-convnets) and المساوي في عدد المعلَمات for the ConvNeXt control model; "primitives" → لبنات for architectural building blocks, عناصر أولية for low-level features or operations.

## chapter_recurrent-neural-networks/index.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- Chapter title "Sequence Models" → نماذج التسلسل; "sequential data" → البيانات التسلسلية; cover "Sequence Models and Recurrent Neural Networks" → نماذج التسلسل والشبكات العصبية المتكررة.
- H. G. Wells' *The Time Machine* → رواية *The Time Machine* للكاتب H. G. Wells (title and author in Latin, class noun رواية; "novella" → رواية قصيرة); its quoted opening sentence stays in English inside “…”, because it is the dataset text and the argument counts its English words.
- "rollout" (feeding a model its own predictions) → التنبؤ المتتابع; "multistep rollout" → التنبؤ المتتابع متعدد الخطوات; "one-step prediction" → التنبؤ بخطوة واحدة; "rollout drift" → انجراف التنبؤ المتتابع.
- "bits per byte" → البتات لكل بايت (unit on slides: بت/بايت; "bytes/token" → بايت/رمز مميز; bpb and PPL stay Latin where the English writes them); "count-based" → القائم على العد; "count tables" → جداول العد.
- "subword" alone → كلمة فرعية (كلمات فرعية); "subword tokens" → الرموز المميزة للكلمات الفرعية; "subword tokenization" → التجزئة إلى رموز فرعية (glossary); "tokenizer" → المجزئ; "pre-tokenized" → بعد تجزئة مسبقة; "token granularity" → مستوى تفصيل الرموز المميزة.
- "truncated sampling", "truncated samplers" → أخذ العينات المبتور، خوارزميات أخذ العينات المبتور; "tail truncation" → بتر الذيل; "truncated BPTT" (slide) → خوارزمية BPTT المقتطعة.
- "structured state space models" → نماذج فضاء الحالات المهيكلة; "vanilla RNN" → شبكة RNN البسيطة; "Kneser--Ney $n$-grams" → نماذج $n$-gram بطريقة Kneser--Ney; "trigram" → ثلاثية (ثلاثيات الكلمات).
- "restricting attention to" (everyday sense) → قصر النظر على, to keep الانتباه for the mechanism; "degenerate into" (text) → يتردى إلى (not ينحدر, the regression root).
- Slides: "feedforward models" → نماذج التغذية الأمامية; "Motivation & Core Problem" → الدافع والمشكلة الأساسية; "Architectural Progression" → المسار المعماري; "Chapter Synthesis" → توليف الفصل; "sequence modeling pipeline" → خط نمذجة التسلسل; "global gradient clipping", "global norm clipping" → قص التدرج الكلي، قص المعيار الكلي; "length-normalized" → المطبَّع بالطول; "open-ended generation" → التوليد المفتوح; "free generation" → التوليد الحر; "repetition loops" → الحلقات المتكررة.
- English glosses given on this page: language model, token, recurrent neural network (RNN), hidden state, perplexity, byte pair encoding, backpropagation through time, beam search.

## chapter_recurrent-neural-networks/rnn.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "recurrent neural network (RNN)" → الشبكة العصبية المتكررة (glossary), then شبكة RNN (شبكات RNN); "RNN language model" → النموذج اللغوي بشبكة RNN; "recurrent layer / cell / connection / weight" → الطبقة المتكررة / الخلية المتكررة / وصلة متكررة / الوزن المتكرر; "recurrent weight matrix" → مصفوفة أوزان متكررة.
- "this recurrence" (the update $\mathbf{h}_t = g(\cdot)$) → العلاقة التكرارية (as logged for a mathematical recursion); "recursive hidden state" → حالة مخفية تكرارية; "hidden state recurrence" (cover slide) → تكرار الحالة المخفية.
- "trigram model" → النموذج الثلاثي (as on the language-model page); "$n$-gram order" → مرتبة نموذج $n$-gram; "count-based language models" → النماذج اللغوية القائمة على العد; "count table" → جدول عد; "contexts" → سياقات.
- "prefix" (the tokens read so far) → البادئة; "history" → التاريخ; "transition function" → دالة الانتقال; "latent autoregressive model" → نموذج انحدار ذاتي كامن.
- "unrolled" (an RNN over time) → مفرودة، الصورة المفرودة, "temporal unrolling" → الفرد الزمني (the log's يفرد); "horizontally concatenated" → المربوطة أفقيًا, "vertically stacked" → المكدسة عموديًا.
- "to batch" (a computation across steps) → يجمع في دفعة واحدة; "batched input projections" (slide title) → إسقاطات المدخلات بالدفعات; "sequential (time) loop" → الحلقة (الزمنية) التسلسلية; "hoist out of the loop" → إخراج … من الحلقة; "known up front" → معروفة سلفًا.
- "parity" → الزوجية; "bit stream" → دفق من البتات; "running count" → العدد التراكمي; "flip a bit" → قلب بت; "threshold RNN" → شبكة RNN بعتبة.
- "lossy" → فاقد للمعلومات ("lossy compression" → الضغط الفاقد للمعلومات); "512 bytes" → 512 بايت, "those 512 bytes" → هذه البايتات الـ 512; "state bandwidth" → عرض نطاق الحالة; "horizon reach" → مدى الأفق; "horizon wall" → جدار الأفق.
- "teacher forcing" → فرض مخرجات المعلم (glossary); "ground-truth token" → الرمز المميز من الحقيقة المرجعية; "rollout" (free-running generation) → التنبؤ المتتابع; "sampled or greedy token" → رمز مميز مسحوب بأخذ العينات أو مختار بالطريقة الجشعة; "multistep sine-wave forecast" → التنبؤ متعدد الخطوات بالموجة الجيبية.
- "token id" → معرف الرمز المميز (as logged); "vocabulary entries" → بنود المفردات; "special tokens" → الرموز الخاصة; "embedding width" → عرض التضمين; "hidden size" → الحجم المخفي; "vocabulary-sized layers" → الطبقات التي يتبع حجمها حجم المفردات, on slides طبقة إخراج بحجم المفردات، طبقتا المفردات; "output layer" → طبقة الإخراج.
- "look up rows in the embedding table" → تستخرج الصفوف من جدول التضمين, slide "Embedding lookup" → البحث في جدول التضمين; "per lag" → لكل موضع تأخر; "hard cutoff" → قطع حاد; "impulse trace" → تتبع نبضة; "decay" of the impulse → تتناقص.
- Table headers: "part" → الجزء, "shape" → الشكل, "parameters" → المعلَمات, "total" → المجموع; "input / output weight" → وزن المدخلات / وزن المخرجات; "hidden bias" → الانحياز المخفي.
- English glosses given in u01 of this page (do not repeat in later units): latent, hidden state, recurrent neural network (RNN), parity, lossy, embedding table, multiply-adds, ground truth, teacher forcing. Not glossed: hidden layer, attention, perplexity (defined on earlier pages).

## chapter_recurrent-neural-networks/decoding.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "decoding" → فك التشفير (as logged); "decoder" as a decoding algorithm → مفكك التشفير (مفككات التشفير); "greedy decoding" → فك التشفير الجشع, "greedy search" → البحث الجشع; "beam search" → البحث الشعاعي, "beam width" → عرض الشعاع, "a beam" (one kept hypothesis) → شعاع (أشعة); "decoding strategies" → استراتيجيات فك التشفير; decoding token ids into text → فك ترميز.
- "prefix" → بادئة (بادئات); "continuation" → تكملة (تكملات); "prompt" is left for the next unit; "next-token distribution / logits" → توزيع / لوجيتات الرمز المميز التالي; "candidate" → مرشح; "hypothesis" (beam search) → فرضية.
- "maximization" (decoding family) → التعظيم; "sampling" (family) → أخذ العينات; "truncated sampling" → أخذ العينات المبتور, "truncation" → البتر; "prune" → يشذب، التشذيب; "expand" a beam or prefix → يمدد، التمديد (التوسيع stays dilation); "unreliable tail" → الذيل غير الموثوق.
- "open-ended generation / text" → التوليد المفتوح / النص المفتوح; "constrained tasks" → المهام المقيدة; "valid" outputs or continuations → صالحة, "acceptable" → مقبولة; "exact search / maximization" → البحث المضبوط / التعظيم المضبوط; "intractable" → غير عملي; "horizon" → أفق; "search tree" → شجرة البحث.
- "myopia" → قصر النظر; "neural text degeneration" → انحلال النص العصبي; "repetitive loops" → حلقات متكررة; "attracting cycle" → دورة جاذبة, "limit cycle" → دورة حدية, "attractors" → جواذب; "rollout" (generated continuation, on slides) → التوليد (مسار التوليد، التوليد الجشع); "recurrence" a value is fed back into → العلاقة التكرارية.
- "model-agnostic interface" → واجهة مستقلة عن النموذج; "invariance" to architecture and framework → الثبات أمام; "1-D NumPy vector" → متجه NumPy أحادي الأبعاد; "prose" (generated) → نص / نثر.
- Slides: top-k, top-p, min-p, distinct-n stay Latin (أخذ العينات بطرق top-k والنواة (top-p) و min-p; مقياس distinct-n); "nucleus sampling" → أخذ عينات النواة; "Maximization Family I / II" → عائلة التعظيم I / II; "774M parameters" → 774 مليون معلَمة; "interpolating exponent" → الأس … للاستيفاء; "OCR" stays Latin.
- English glosses given in u01 of this page (do not repeat in later units): decoding, maximization, sampling, prefix, greedy decoding, myopia, neural text degeneration. Beam search, beam width, length normalization and temperature scaling are only named here and take their gloss where later units define them.

## chapter_recurrent-neural-networks/text-sequence.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "tokenizer" → مجزئ (مجزئات), glossed once as مجزئ (tokenizer); from the glossary's التجزئة إلى رموز. "tokenization" → التجزئة إلى رموز, short التجزئة after the first mention in a sentence; "to tokenize" → يجزئ إلى رموز; "character-level / word-level / byte-level tokenization (tokenizer)" → التجزئة (المجزئ) على مستوى المحارف / الكلمات / البايتات; "subword tokenizer" → مجزئ الكلمات الفرعية; "production tokenizer" → مجزئ بيئة الإنتاج.
- "character" (of text, Unicode) → محرف (محارف); "letter" → حرف (حروف); "accented letters", "Latin diacritics" → الحروف التي تحمل علامات / الحروف اللاتينية ذات العلامات; "diacritics" → علامات التشكيل; "scripts" → أنظمة الكتابة; "emoji" → الرموز التعبيرية; "character encoding" → ترميز محارف; UTF-8, Unicode, ASCII stay Latin after a class noun (الترميز UTF-8، معيار Unicode، محارف ASCII).
- "byte pair encoding (BPE)" → *ترميز أزواج البايت* (byte pair encoding, BPE), afterwards خوارزمية BPE / مجزئ BPE as the English writes BPE; "merge" → دمج (رتبة الدمج، أشجار الدمج); "greedy" → جشع; "pre-tokenization" → التجزئة المسبقة; "regex" → التعابير النمطية; "special tokens" → الرموز الخاصة.
- "token index" → فهرس (فهارس صحيحة); "token types" (distinct tokens) → أنواع الرموز المميزة; "symbols" of an alphabet (27 symbols, 256 symbols) → رمز (رموز), kept apart from الرمز المميز; vocabulary "entry" → بند (بنود); "vocabulary size" → حجم المفردات; "base alphabet" → أبجدية أساسية; "universal" → شامل (الشمولية).
- "out of vocabulary (OOV)" → *خارج المفردات* (out of vocabulary, OOV), then OOV after a class noun where the English writes OOV (حالة OOV); "zero OOV" → انعدام حالات OOV / بلا أي حالة OOV; "unseen" word or token → غير مرئية; "`<unk>` fallback", "falls back to `<unk>`" → الرجوع إلى `<unk>`.
- "running corpus" → المدونة النصية التي نعتمدها (طوال هذا الفصل); "novella" → رواية قصيرة, "our novella" → روايتنا / الرواية; the title *The Time Machine* stays in Latin after رواية.
- "interpolate between characters and words / between the extremes" → يتدرج بين (as the log's group-normalization entry; يستوفي stays fitting data); "granularity" → مستوى التفصيل; "operating point" → نقطة العمل; "spectrum" → الطيف.
- "long-range dependency" → تبعية بعيدة المدى; "subject–verb agreement" → المطابقة بين الفاعل وفعله; "inflection" → صيغة صرفية; "morphological stems" → الجذوع الصرفية; "proper name" → اسم علم; "typographical error", "typo" → خطأ مطبعي; "capitalization" → كتابة الحرف الأول كبيرًا; "case folding" → توحيد حالة الحروف; "whitespace" → الفراغات.
- text "normalization" → التطبيع (تطبيع النص، نص مطبَّع); "aggressive normalization" → تطبيع شديد; "losslessly" → بلا فقد; "token for token" → رمزًا مميزًا برمز مميز; "effective context horizon" → أفق السياق الفعلي; "open-vocabulary robustness" → المتانة مع المفردات المفتوحة; "parameter footprint" → حجم المعلَمات; "sequence model" → نموذج التسلسل.
- "NLP" → مجال NLP (Latin acronym after a class noun).
- English glosses given in u01 of this page (do not repeat in later units): tokens, tokenizer, tokenization, subword, byte pair encoding (BPE), out of vocabulary (OOV), vocabulary.

## chapter_recurrent-neural-networks/language-model.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "$n$-gram language model" → نموذج $n$-gram اللغوي; an "$n$-gram" as an item (a run of $n$ tokens) → تتابع $n$-gram (تتابعات $n$-gram); "unigram / bigram / trigram model" → النموذج الأحادي / الثنائي / الثلاثي; "bigrams / trigrams" as items → الثنائيات / الثلاثيات (ثنائيات الكلمات، ثلاثيات الكلمات), "word pairs / triples" → أزواج الكلمات / ثلاثيات الكلمات.
- "prefix" (the tokens so far) → البادئة; "continuation" (the token that follows a context) → تتمة; "text continuation" → إكمال نص; "to score a sequence" → يحسب درجة التسلسل; "given $x_1$" → بمعلومية $x_1$; "conditions on" → يشترط على / مشروط بـ.
- "count(s)" → تعداد (تعدادات), "count-based model" → نموذج قائم على العد, "count sparsity" → تناثر التعدادات; "relative frequencies" → التواترات النسبية (word frequency is the glossary's تواتر).
- "tokenizer" → المجزئ (المجزئات), from the glossary's تجزئة إلى رموز; "to tokenize into words" → يجزئ إلى كلمات; "tokenized sequences" → التسلسلات المجزأة إلى رموز.
- "smoothing" → التنعيم (تنعيم لابلاس; Kneser–Ney stays Latin: تنعيم Kneser–Ney); "bits per byte" → البتات لكل بايت.
- "Zipf's law" → قانون Zipf (Latin name, not in the glossary); "Zipf exponent" → أس Zipf; "power law" → قانون القوة; "heavy-tailed" → ثقيل الذيل; "skewed distribution" → توزيع شديد الالتواء; "rank" (of a word by frequency) → الرتبة; "log–log axes / space" → محوران لوغاريتميان.
- "stop words" → كلمات التوقف; "function words" → كلمات وظيفية; "singletons" ($n$-grams that occur once) → وحيدة الورود (not مفردات, which is the vocabulary).
- "verbatim memorization" → الحفظ الحرفي; "longest copied run" → أطول مقطع منسوخ; "contiguous span of words" → سلسلة متصلة من الكلمات; "the sampler" → السحب; "nested hash map" → جدول تجزئة متداخل; "effective branching factor" → عامل التفرع الفعال; "Markov truncation" → اقتطاع ماركوف.
- Example phrases and prompts quoted from English text stay in English inside “…”.
- English glosses given in u01 of this page (do not repeat in later units): language model, perplexity, smoothing, bits per byte, unigram, bigram, trigram, sparsity, stop words, power law, heavy-tailed.

## chapter_recurrent-neural-networks/sequence.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- Page title "Working with Sequences" → التعامل مع التسلسلات; "sequential data" → البيانات التسلسلية (as on the chapter page).
- "stationarity" → الاستقرارية, "stationary" → مستقر (ديناميكيات مستقرة); "time-homogeneous" → متجانس زمنيًا; "time-invariant" → ثابت زمنيًا (no glossary entries; the terms of Arabic time-series texts).
- "Markov condition" → شرط ماركوف; "Markov models" → نماذج ماركوف; "$m^\textrm{th}$-order" → من المرتبة $m^\textrm{th}$, "first-order" → من المرتبة الأولى; "is Markov" (adjective) → ماركوفي.
- "autoregression" → الانحدار الذاتي; "autoregressive" as an adjective → ذاتي الانحدار (التنبؤ ذاتي الانحدار); "a linear autoregressive model" → نموذج خطي للانحدار الذاتي; "latent autoregressive model" → نموذج انحدار ذاتي كامن (as on the RNN page).
- "fixed window" → نافذة ثابتة; "window length" → طول النافذة; "latent summary" → ملخص كامن; "running summary" → ملخص تراكمي; "fixed-input model / architecture / predictor" → ثابت المدخلات; "fixed-size" → ثابت الحجم; "hard horizon" → أفق صارم; "memory horizon" → أفق الذاكرة.
- "chain rule of probability" → قاعدة السلسلة للاحتمالات; "conditionals" → التوزيعات الشرطية (الاحتمالات الشرطية for their values); "factor" (verb, a joint or a sequence) → يتحلل إلى جداء / يحلل إلى عوامل; "this identity is exact" → هذه المتطابقة مضبوطة.
- "entries" of a sequence → عناصر; "time series" → سلسلة زمنية, "numerical series", "the series" → سلسلة عددية، السلسلة (تسلسل stays the ML sequence); the mathematical index $t$ → الدليل.
- "sentiment classification" → تصنيف المشاعر; "captioning an image" → كتابة وصف لصورة; "transcribing speech" → تحويل الكلام إلى نص; "two transcriptions for the same audio clip" → نصان حُوّل إليهما المقطع الصوتي نفسه; "waveform samples" → عينات الموجة الصوتية; "spectrogram frames" → إطارات المخطط الطيفي.
- "email client" → برنامج بريد إلكتروني; "weather service" → خدمة أرصاد جوية; "forecaster" → خبير التنبؤ (as logged); "effect / causes" → الأثر / الأسباب.
- Slides: "Taxonomy of …" → أصناف … (not تصنيف, which is classification); "Problem Formulation" → صياغة المسألة; "Probabilistic Foundation" → الأساس الاحتمالي; "Statistical Assumptions" → الافتراضات الإحصائية; "Sequence-to-Label" → تسلسل إلى وسم; "Fixed-to-Sequence" → ثابت إلى تسلسل; "unsupervised density estimation" → تقدير الكثافة دون إشراف; "candidate likelihood ranking" → ترتيب المرشحين حسب الإمكان; "ASR rescoring" → إعادة حساب الدرجات في أنظمة ASR; "ancestral generation" → التوليد السلفي; "log-domain scoring" → حساب الدرجات في المجال اللوغاريتمي; "tractable / tractability" (count tables, finite-context models) → عملي / الجدوى العملية (from the log's "intractable" → غير عملي); "training pairs" → أزواج تدريبية; "Strategy 1 — …" → الاستراتيجية 1، … (a comma for the dash).
- Example phrases quoted from English text (“Thank you for your”, “help”, “time”, “your thank for you”) stay in English inside “…”, as on the language-model page.
- English glosses given in u01 of this page (do not repeat in later units): sequence modeling, ground truth, sequence, time step, autoregressive model, window length, latent autoregressive model, chain rule of probability, underflow, language modeling. Not glossed here: recurrent neural networks, state space models, hidden state (named in passing; glossed on the pages that define them). Markov condition, stationary and time-homogeneous take their gloss in the next unit, which defines them.

## chapter_recurrent-neural-networks/bptt.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "backpropagation through time" → *الانتشار العكسي عبر الزمن* (backpropagation through time, BPTT), glossed in u01; "BPTT" as the English writes it on slides → خوارزمية BPTT (feminine: خوارزمية BPTT المقتطعة / الكاملة / الخطية); "TBPTT" is copied after the full Arabic term.
- "lag" (number of steps back in time) → التأخر (التأخرات; "lag $k$" → التأخر $k$; "lag-$k$ operator" → مؤثر التأخر $k$).
- "horizon" (how far back gradients or learning reach) → أفق (آفاق): أفق التدرج، آفاق الاقتطاع، على أفق أطول.
- "unroll" (an RNN, a recurrence) → يفرد، الفرد، شبكة مفرودة (as on the numerical-stability page); "backward recursion" → العلاقة التكرارية الخلفية; "scalar recurrence" → علاقة تكرارية قياسية; "structured linear recurrences" → العلاقات التكرارية الخطية المهيكلة, glossed in u01.
- "state Jacobian" → المصفوفة اليعقوبية للحالة; "step Jacobian" → المصفوفة اليعقوبية للخطوة; "right-multiplication" → ضرب من اليمين; "transposed weight matrix" → منقول مصفوفة الأوزان.
- "decay" (of the gradient with lag) → تناقص (as "signal decay" on the numerical-stability page); "gradient flow" (how gradients travel backward) → سريان التدرج, as in the MLP chapter (تدفق التدرج stays the continuous-time ODE).
- "vanilla RNN" → شبكة RNN البسيطة (as decided earlier); "vanilla $\tanh$ RNN" → شبكة RNN البسيطة بدالة $\tanh$; "gated RNNs / gated recurrent cells" → شبكات RNN ذات البوابات / الخلايا المتكررة ذات البوابات, "gated" glossed in u01; "cell state" → حالة الخلية.
- "global gradient clipping" → قص التدرج الكلي; "safeguard" → احتياط; "exploding spikes" → قفزات الانفجار; "pathologies" (of gradients) → علل (علتا التدرج).
- "segment" (of a token stream) → مقطع (مقاطع); "stream" → سيل (سيول); "stateful streaming" → السيل ذو الحالة; "segment boundary" → حد المقطع; "truncation" → الاقتطاع; "detach" → الفصل (as on the autograd page).
- "long-range dependencies" → التبعيات بعيدة المدى; "per-position validation loss" → خسارة التحقق لكل موضع; "Kneser--Ney counts" → تعدادات Kneser--Ney; "a few billionths" → بضعة أجزاء من المليار.
- "razor's edge" → حد السكين; "leading eigenspace" → الفضاء الذاتي الرئيس; "non-normal" (matrix growth) → غير ناظمي; "unit disk" → قرص الوحدة; "runtime" → زمن التنفيذ; "update latency" → زمن الاستجابة لكل تحديث.
- Kickers: "Section Takeaways" → خلاصات القسم; "Empirical Probe" → مسبار تجريبي; table headers "Mathematical Expression / Scaling / Asymptotics / Practical Consequence" → التعبير الرياضي / التحجيم / السلوك التقاربي / الأثر العملي.
- English glosses given in u01 of this page (do not repeat in later units): backpropagation through time (BPTT), exploding gradients, vanishing gradients, spectral radius, gated, structured linear recurrences. Truncated backpropagation through time is not glossed yet (u02 defines it).

## chapter_recurrent-neural-networks/language-model.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "absolute discounting" → الخصم المطلق; "discount" $D$ → الخصم (معلَمة الخصم، الكتلة المخصومة); "interpolation", "interpolated Kneser–Ney" → الاستيفاء، تنعيم Kneser–Ney الاستيفائي; "continuation count" → تعداد التتمات (تعدادات التتمات); "pseudocount", "phantom counts" → تعداد وهمي.
- "back off", "fall back" (to a shorter context) → يرجع إلى، الرجوع; "uniform floor" → الأرضية المنتظمة; "frees mass" → يحرر كتلة احتمالية; "predecessors" (of a word) → الكلمات السابقة; "successors" → الكلمات / الرموز التالية; "versatile" / "context-bound" word → المتعددة السياقات / المقيدة بسياقها.
- "add-$\alpha$ smoothing" → تنعيم إضافة $\alpha$; "Laplace-smoothed" → بتنعيم لابلاس; Laplace the person → لابلاس, Shannon → شانون (Arabic letters, as logged); Tolstoy, Saint-Exupéry and their titles stay Latin after رواية … للكاتب.
- "count models" → نماذج العد; "limiting cases" → حالات حدية; "the scale" (of perplexity or bpb) → السلم; "arithmetic coder" → مشفر حسابي; "spelling cost" (of an unknown word) → كلفة التهجئة; "OOV penalty" → جزاء حالات OOV; "Markov order" → مرتبة ماركوف; "syllable" → مقطع لفظي.
- "the discount has a closed-form estimate" → تقدير بصيغة مغلقة; "defined recursively" → معرّف تعاوديًا, "the recursion" → العلاقة التكرارية (both as logged).
- Slides: "Sequence Data Partitioning" → تقسيم بيانات التسلسل; "sliding input–target windows" → نوافذ منزلقة من المدخلات والأهداف; "dense supervision" → إشراف كثيف; "Tokenizer-Invariant Evaluation" → تقييم ثابت أمام المجزئ; "Cross-Granularity Benchmark" → اختبار معياري عبر مستويات التفصيل; "State-of-the-Art Count Smoothing" → أحدث ما وصل إليه تنعيم التعدادات.
- English glosses given in u02 of this page (do not repeat in later units): surprisal, effective branching factor, Laplace smoothing, absolute discounting, interpolation, continuation count, arithmetic coder.

## chapter_recurrent-neural-networks/decoding.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "prompt" → الموجه (glossary موجه), glossed (prompt) once on this page; "the same two prompts" → الموجهين نفسيهما.
- "ancestral sampling" → المعاينة السلفية (glossary), glossed once; the general "sampling" stays أخذ العينات; "pure / raw sampling" → أخذ العينات الخالص / الخام.
- "top-$k$ / top-$p$ / min-$p$ sampling" → أخذ العينات بطريقة top-$k$ / top-$p$ / min-$p$, then طريقة top-$k$ (feminine agreement); Latin names keep the English capitalization (Top-$k$ in slide tables and titles); "nucleus sampling" → أخذ عينات النواة (as logged), glossed (nucleus sampling) here.
- "cutoff" (truncation rule) → حد البتر (حدود البتر); "threshold" → عتبة; "relative floor" → أرضية نسبية; "truncate by rank / by cumulative mass" → يبتر بحسب الرتبة / الكتلة الاحتمالية التراكمية; "renormalize" → يعيد التطبيع; "surviving support" → الحامل الباقي; "survivors / surviving prefix" → الباقيات / بادئة باقية.
- "head" of the distribution → رأس التوزيع, "head tokens" → الرموز المميزة في الرأس, "tail tokens" → الرموز المميزة في الذيل; "leading tokens", "the leader" → الرموز المميزة المتصدرة، المتصدر; "runner-up" → وصيف; "plausible / implausible" token → معقول / غير معقول.
- "widen / expand the beam" → زيادة عرض الشعاع (التوسيع stays dilation, التمديد is expanding a prefix); "active beams" → الأشعة النشطة; "candidate extensions" → التمديدات المرشحة; "completed hypotheses" → الفرضيات المكتملة; "length-normalized score" → الدرجة المطبَّعة بالطول; "length correction" → تصحيح للطول.
- "interpolates between" (the exponent $\beta$) → يتدرج بين (as logged on the text-sequence page); "blunt instrument" → أداة غليظة; "cooling the softmax" → تبريد softmax; "degrades gracefully" (a threshold) → تتدهور على نحو ألطف.
- "sampler" → خوارزمية أخذ العينات (as logged), "unified sampler" → خوارزمية موحدة لأخذ العينات; "wrapper" function → الدالة الغلاف; "strategy keyword arguments" → وسائط مسماة تخص الاستراتيجية; "toy distribution" → توزيع مبسط; "on log–log axes" → في رسم لوغاريتمي المحورين; "rank" (position in the sorted distribution) → الرتبة.
- "rollout error compounding" → تراكم أخطاء التنبؤ المتتابع; "50-token rollouts" (slides) → مسارات التوليد ذات 50 رمزًا مميزًا; "garbled words" → كلمات مشوهة; "tail gibberish" → هذيان الذيل; "fluency" → السلاسة; "non-looping" → لا يدور في حلقات; "within a factor of twenty of the top probability" → لا يقل احتماله عن جزء من عشرين من الاحتمال الأعلى (slides keep $20\times$); "magenta dashed" → الخط الأرجواني المتقطع.
- Slides: "From-Scratch Implementation & Evaluation" → التنفيذ من الصفر والتقييم; "Sampling Control I / II" → التحكم في أخذ العينات I / II; "Three regimes" → ثلاثة أنماط; "composable" → قابلة للتركيب; "Head-to-Head" → وجهًا لوجه (as logged).
- English glosses given in u02 of this page (do not repeat in later units): beam search, beam width, prompt, length normalization, ancestral sampling, unreliable tail, temperature scaling, nucleus sampling. End-of-sequence token, exhaustive search and entropy are not glossed.

## chapter_recurrent-neural-networks/rnn.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "gradient spike", "Jacobian spike" → قفزة في التدرج / قفزات التدرج، القفزات في المصفوفة اليعقوبية (the log's قفزة for "spike"); "the basin" → الحوض; "caps … at a threshold" → يضع سقفًا لـ … عند عتبة; "ball of radius $\theta$" → الكرة التي نصف قطرها $\theta$; "radial ball projection" → إسقاط قطري على الكرة.
- "clipped model / baseline" → النموذج (المرجعي) المدرَّب مع القص; "unclipped" runs or instances → بلا قص; "untrained" → غير المدرب; "plain model" → نموذج بسيط; "wider regularized model" → النموذج الأعرض المنظَّم.
- "validation perplexity" → حيرة التحقق, "training perplexity" → حيرة التدريب, "initial perplexity" → الحيرة الابتدائية; "2.1 bits per byte" → 2.1 بت لكل بايت (the quantity: البتات لكل بايت); "compression rate" → معدل الضغط; bpb stays Latin on slides.
- "Kneser--Ney trigram / bigram" → النموذج الثلاثي / الثنائي بطريقة Kneser--Ney (as on the language-model page); "4-gram", "3-gram" → نموذج 4-gram، نموذج 3-gram; "interpolated Kneser--Ney smoothing" → تنعيم Kneser--Ney الاستيفائي; "Kneser–Ney counts" → تعدادات Kneser–Ney.
- "tie" (two models) → تعادل، يتعادل; "virtual tie" → تعادل شبه تام; "mixture" → خليط, its "members" → مكوّناه; "yardstick" → معيار للمقارنة; "token by token" → رمزًا مميزًا برمز مميز.
- "batch-major / time-major" → ترتيب الدفعة أولًا / ترتيب الزمن أولًا; "leading time axis" → محور الزمن، وهو المحور الأول; "per-step states" → حالات الخطوات; "batched matrix product" → جداء مصفوفات واحد بالدفعات; "to transpose" token ids → يأخذ منقول.
- "decays its past state" → تتناقص الحالة الماضية; "per-coordinate step sizes" → أحجام الخطوات لكل إحداثي; "uniform learning rate", "uniform-step SGD" → معدل تعلم موحد، SGD ذات الخطوة الموحدة (منتظم stays the uniform distribution); "densely updated" → تُحدَّث تحديثًا كثيفًا; "random feature basis" → أساس ميزات عشوائية; "long lags" → فترات تأخر طويلة; "saturate" → تتشبع; "relative Euclidean displacement" → الإزاحة الإقليدية النسبية.
- "raw novella" → الرواية الخام; "capitals, punctuation, line breaks" → الحروف الكبيرة وعلامات الترقيم وفواصل الأسطر; "SGD schedule" (slide) → خطة SGD; "the JAX trainer" → مدرب JAX; "optax transformation" → تحويل optax.
- Slides: "From-Scratch Implementation" → التنفيذ من الصفر; "Optimization Safeguard" → إجراء وقائي في التحسين; "Ablation:" as a title lead-in → استئصال:; "Complementary errors" → أخطاء متكاملة; "lexical specificity" → تحديد معجمي; "223k / 50k" → 223 ألفًا / 50 ألفًا.
- English glosses given in u02 of this page (do not repeat in later units): gradient clipping, truncated.

## chapter_recurrent-neural-networks/text-sequence.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "merge rule" → قاعدة الدمج; "a merge" (one learned rule) → دمج, plural عمليات الدمج (ثلاث عمليات دمج; counted after a number or symbol: 768 دمجًا); "merge tree" → *شجرة دمج* (merge tree); "merge table" → جدول الدمج; "merge rank" → رتبة الدمج (as logged).
- "toy corpus / toy tokenizer" → المدونة النصية المصغرة / المجزئ المصغر; "to trace by hand", "hand trace" → يتتبع يدويًا، التتبع اليدوي; "tie" → تعادل, "break ties" → يحسم التعادل.
- "trailing space" → *الفراغ اللاحق* (trailing space); "leading space" → *الفراغ السابق* (leading space); "spacing" → الفراغ.
- "chunk" (pre-tokenization piece) → قطعة (قطع), "text chunks" → القطع النصية, "chunk boundaries" → حدود القطع (مقطع stays "segment"); "vocabulary slots" → بنود المفردات (the log's بند for a vocabulary entry); "composite" (multi-word token) → مركب (مركبات), "boundary-crossing composite / crossers" → مركب عابر للحدود / رموز عابرة للحدود; "straddle word boundaries" → يتخطى حدود الكلمات.
- "regular expression", "regex" → تعبير نمطي (as logged); "pattern" → النمط; "branches" (of the pattern) → فروع; "run of letters / digits" → سلسلة حروف / سلسلة أرقام; "digits" (characters) → أرقام, "numbers" → أعداد; "English contractions" → صيغ الاختصار الإنجليزية; "Unicode character classes" → فئات محارف Unicode.
- "to encode / encoding" (text → ids) → يرمّز / الترميز; "to decode / decoding" (ids → text) → يفك الترميز / فك الترميز (as logged; فك التشفير stays the generation-time decoding); "byte sequence" → تسلسل بايتات; "byte string" → سلسلة بايتات; "substring" → سلسلة جزئية.
- "fragments" (of words, of the corpus) → أجزاء; "occurrences" → مواضع / مرات الورود; "pair counts" → تعدادات الأزواج; "collapse" (a merge collapses occurrences) → يختزل; "a handful" → حفنة.
- "production BPE trainers" → برامج تدريب BPE في بيئة الإنتاج; "production vocabularies" → مفردات بيئة الإنتاج; "updating incrementally" → يحدّث تحديثًا تزايديًا; "printing helper" → دالة طباعة مساعدة; "wraps long lines" → تطوي الأسطر الطويلة (يغلف stays wrapping a function or object).
- "sweep vocabulary sizes" → يمر على أحجام المفردات (اجتياز stays the autodiff sweep); "the plot is flattered", "flatters" → يبدو أفضل من حقيقته; "extrapolate blindly" → الاستكمال خارج المدى دون تبصر; "general English prose" → النثر الإنجليزي العام; "multilingual" → متعدد اللغات; "source code" → الشيفرة المصدرية.
- Slides: "byte fallback" → الرجوع إلى البايتات; "stem + suffix" → جذع + لاحقة; "inflected text" → نص مصرَّف; "primitive" → عملية أولية (as logged); "non-overlapping semantics" → دلالة … بلا تراكب; "bigrams" → الثنائيات; "domain-specific words" → كلمات خاصة بالمجال; "in-domain" → داخل المجال; "diminishing marginal counts" → تعدادات حدية متناقصة; "content words" → كلمات المحتوى; "Subword Generalization" → التعميم بالكلمات الفرعية; "From-Scratch Implementation" → التنفيذ من الصفر; "Encoding & Decoding Mechanics" → آلية الترميز وفك الترميز; "Empirical Behavior on Prose" → السلوك التجريبي على النثر; a slide lead-in "Notice the flaw:" → العيب الظاهر: (noun phrase, no imperative).
- English glosses given in u02 of this page (do not repeat in later units): merge tree, trailing space, leading space, pre-tokenization.

## chapter_recurrent-neural-networks/bptt.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "truncated backpropagation through time" → *الانتشار العكسي المقتطع عبر الزمن* (truncated backpropagation through time), glossed in u02; "truncated BPTT" where the English writes the acronym → خوارزمية BPTT المقتطعة; "full backpropagation" → الانتشار العكسي الكامل.
- "caps / bounds memory and compute per update" → يضع سقفًا للذاكرة والحوسبة / تحصر الذاكرة والحوسبة لكل تحديث; "slice a stream into segments" → يقطّع السيل إلى مقاطع; "hard learning horizon" → أفق تعلم صارم; "memory horizon" → أفق الذاكرة; "effective memory" → الذاكرة الفعالة.
- "well inside the 32-token window" → أقل كثيرًا من طول النافذة المؤلفة من 32 رمزًا مميزًا; "contains exploding spikes" → يكبح قفزات الانفجار; "architectural innovations" → ابتكارات معمارية; "loss sensitivities" → حساسيات الخسارة.
- "diagonalizable" → قابلة للاستقطار (Arabic Wikipedia's term; no glossary entry); "with probability one" → باحتمال يساوي واحدًا.
- A vector "becomes aligned with $\mathbf{v}_1$" → يصير محاذيًا للمتجه $\mathbf{v}_1$, "alignment" → المحاذاة (the glossary word, here for directions of vectors).
- "necessary but not sufficient" → شرط لازم لكنه غير كافٍ; "sufficient" → يكفي; "largest singular value" → القيمة المفردة العظمى (as on the numerical-stability page).
- "single-step gradients" → تدرجات الخطوة الواحدة; "problem :exref:`…`" → التمرين :exref:`…`; "stretches of validation tokens" → مقاطع من بيانات التحقق (as u01's validation segments); "loss-by-position measurement" → قياس الخسارة لكل موضع; "Argue from …" → استدل بـ ….
- Credit line: *مقتبس بتصرف من مقرر CMU 11-785، الواجب [Homework 3 Part 1](url).* (as on the conv-layer page).
- English glosses given in u02 of this page: truncated backpropagation through time. Exercises take no gloss (power iteration, diagonalizable, singular value stay unglossed).

## chapter_recurrent-neural-networks/sequence.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "lag" → تأخر (تأخرات); "lag features" → ميزات التأخر; "one column per lag" → عمود لكل موضع تأخر (as logged); "lag windows" → نوافذ تأخر; "4-lag model" → نموذج على 4 تأخرات.
- "period" (of a wave) → دورة; "phase" → طور; "phase lag / phase delay" → تأخر الطور; "amplitude" → سعة (as logged); "oscillation" → تذبذب; "oscillator" → مذبذب; "frequency" → تردد.
- "$k$-step-ahead prediction / forecast" → تنبؤ بـ $k$ خطوة إلى الأمام; "one-step predictions" → تنبؤات الخطوة الواحدة; "1-step MSE" → خطأ MSE بخطوة واحدة; "to roll the model forward / out" → يمضي بالنموذج في تنبؤ متتابع; "horizon" → الأفق; "lookahead horizon" → أفق التنبؤ.
- "moving average" → متوسط متحرك; "averaging model" → نموذج أخذ المتوسط; "fixed point" of the rollout → النقطة الثابتة; "asymptote" → الخط المقارب (as logged); "baseline floor" → الأرضية الأساسية; "noise floor" → أرضية الضوضاء (as logged).
- "to damp" → يخمد, "damped" (a forecast) → مخمد, "damped model" (one that damps) → النموذج المخمِد; "runaway accumulation" → تراكم جامح; "errors compound" → تتضاعف / تتراكم الأخطاء; "to diverge" → يتباعد.
- "learned physical simulators" → برامج المحاكاة الفيزيائية المتعلَّمة; "decoding rule" → قاعدة فك التشفير; "mismatch" → عدم التطابق; "imperfect prefixes" → البادئات المعيبة; "low-probability choices" → الاختيارات ضعيفة الاحتمال.
- "successor" (the entry that follows a context) → عنصر تالٍ; "extending the context" → إطالة السياق (التوسيع stays dilation); "second-order recurrence" → العلاقة التكرارية من المرتبة الثانية; "conserved invariant" → لامتغير محفوظ.
- Slides: "Synthetic Benchmark" → اختبار معياري اصطناعي; "data-generating process" → عملية توليد البيانات; "signal-to-noise scale" → مقياس الإشارة إلى الضوضاء; "chronological split" → تقسيم بالترتيب الزمني; "self-conditioned horizon" → أفق الاشتراط الذاتي; "regime shifts" → تبدل النمط; "Section Takeaways" → خلاصات القسم; "magenta dashed curve" → المنحنى الأرجواني المتقطع; "hard blindness" → عمى تام.
- English glosses given in u02 of this page (do not repeat in later units): Markov condition, time-homogeneous, stationary, rollout.

## chapter_recurrent-neural-networks/language-model.md#u03 (claude fable-5.2 (medium), 2026-11-27)

- "Partitioning Sequences" → تقسيم التسلسلات; "subsequences" → تسلسلات جزئية; "input window / paired target" → نافذة المدخلات / هدفها المقترن بها; "input–target windows" → نوافذ من المدخلات والأهداف (as on the slide in u02).
- "starting offset" (of a window in the corpus) → إزاحة بداية; "start position" → موضع بداية; "corpus positions" → مواضع المدونة.
- "the first `num_train` windows", "the final $T$ tokens" → the noun, then وعددها + the code or symbol (no الـ before code or a symbol, as logged for "the $K$ held-out losses").
- "buffer of boundary windows" → منطقة عازلة قوامها … نافذة عند الحد (not مخزن مؤقت, the memory buffer); "clean gap" → فاصل نظيف; "leakage guard" → الحماية من التسرب.
- "integer ids" → معرفات صحيحة; "integer tensors" → موتران من الأعداد الصحيحة; "tokenization modes" → أنماط التجزئة; "decode ids back to text" → فك ترميز المعرفات إلى نص (as logged on the decoding page).
- "Zipfian heavy tails" → ذيول ثقيلة على نمط قانون Zipf; "vocabulary granularity" → مستوى تفصيل المفردات; "tokenization-invariant" → ثابت أمام التجزئة إلى رموز.
- Table entries (of a count table) → عنصر (عناصر); vocabulary entries stay بند (بنود); "word types" → أنواع الكلمات; "word tokens" → رموز مميزة على مستوى الكلمات; "5-gram model", "5-grams" → نموذج 5-gram، تتابعات 5-gram.
- "sweep $\alpha$ over values" → يمسح قيم $\alpha$ على …; "the best $\alpha$" → قيمة $\alpha$ المثلى; "validation score" → درجة التحقق; "skew" → الالتواء (as "skewed distribution" in u01).
- "Adapted from Daniel Jurafsky and James H. Martin, [Speech and Language Processing](…), 3rd ed. draft, Exercise N" → مقتبس بتصرف من كتاب [Speech and Language Processing](…) لمؤلفيه Daniel Jurafsky و James H. Martin، مسودة الطبعة الثالثة، التمرين N.
- Slides: "$n$-Gram MLE" → تقدير MLE لنماذج $n$-gram, "unigram MLE" → تقدير MLE الأحادي; "pseudocount dilution" → تخفيف التعدادات الوهمية; "swamp" → تطغى على; "Core Strength / Fundamental Limitation" → نقطة القوة الأساسية / القيد الأساسي; "Model / Metric" → النموذج / المقياس; "Section Takeaways" → خلاصات القسم (as on the BPTT page).
- No new English glosses in u03 of this page: its terms were glossed in u01 and u02 or on earlier pages (hidden state, end-of-sequence token is only used in an exercise).

## chapter_recurrent-neural-networks/sequence.md#u03 (claude fable-5.2 (medium), 2026-11-27)

- "nonlinear autoregressor" → نموذج غير خطي للانحدار الذاتي (as the slide's نموذج خطي للانحدار الذاتي); "small MLP" → شبكة MLP صغيرة (acronym after a class noun, as the English writes MLP).
- "dilated lags" → التأخرات الموسعة (the log's توسيع for dilation); "averaged lags" → متوسطات التأخرات; "block means" → متوسطات الكتل; "construction" (of features) → بناء.
- "regime switching" (exercise title) → تبدل النمط (as the slide's "regime shifts"); "regime" of a series → نمط (نمطان); "switching series" → السلسلة المتبدلة; "alternates between" → تتناوب بين.
- "noise term" → حد الضوضاء; "restore the noise" → أعد الضوضاء; "short-range features" → الميزات قصيرة المدى; "compounding drift" → الانجراف المتراكم (as on the slide).
- "the curve bends upward", "the bend" → ينعطف المنحنى صعودًا، الانعطاف; "what does a forecast offer over the constant forecast" → ماذا يقدم … زيادةً على التنبؤ الثابت.
- No English glosses in u03 (summary and exercises).

## chapter_recurrent-neural-networks/rnn.md#u03 (claude fable-5.2 (medium), 2026-11-27)

- "warm up the hidden state on the prefix" → يجري إحماءً للحالة المخفية على البادئة; slide "prefix warm-up loop" → حلقة إحماء على البادئة (the log's إحماء).
- "continuation tokens" → رموز مميزة من التكملة; "greedy continuation" → تكملة جشعة; "greedy argmax" → argmax الجشع; "temperature-scaled softmax" → softmax بعد تحجيمها بدرجة الحرارة.
- "fuse … into low-level kernels" → تدمج … في نوى منخفضة المستوى; "fused (recurrent) kernel" → النواة (المتكررة) المدمجة; "built-in" layer stays مدمج (في إطار العمل), as logged, also beside "fused"; "swap in" → يضع … مكان / يُحل … محل.
- "`RNN` module" → الوحدة `RNN`; "scans it over the input sequence" → تمسح بها … تسلسل المدخلات; "input / recurrent term" (of the update) → حد المدخلات / الحد المتكرر; "input / recurrent projection" → إسقاط المدخلات / الإسقاط المتكرر.
- "derail training" → إخراج التدريب عن مساره; "at a fraction of the … time" → في جزء يسير من ….
- Exercises: "exposure bias" → انحياز التعرض and "scheduled sampling" → أخذ العينات المجدول (glossary; no gloss in exercises); "the concatenated / separate form" → الصيغة المربوطة / المنفصلة; "factors" of a matrix product → طرفا الجداء (as logged), "factors" of a ratio → عاملان, "factors as" → تتحلل إلى الجداء; "threshold activation" → دالة تفعيل بعتبة; "state units" → وحدات الحالة; "loss by position" → الخسارة لكل موضع; "carried state" → الحالة المحمولة; "training horizon" → أفق التدريب; "ablation" (title) → دراسة استئصال; "split by fraction" → يقسم بالنسبة; "trend" → الاتجاه; "cross-book" → عبر الكتب; "Project Gutenberg ebook 36" → الكتاب الإلكتروني رقم 36 في Project Gutenberg; "drawn uniformly at random" → مسحوب عشوائيًا بانتظام; "time both" → قس زمن الطريقتين, "running time" → زمن التشغيل.
- Slides: "Inference & Rollout Behavior" → الاستدلال وسلوك التنبؤ المتتابع; "Greedy Loops" → حلقات فك التشفير الجشع; "locks into an infinite periodic cycle" → ينحبس في دورة دورية لا نهائية; "High-Level Framework APIs" → واجهات API عالية المستوى في أطر العمل; "Equivalence & Speedup" → التكافؤ والتسريع; "looping greedy rollout" → التوليد الجشع الدائر في حلقات; "model ensembling" → تكوين مجموعات النماذج; "the better expert" → الخبير الأفضل; "Standalone" → كل نموذج منفردًا; "Summary: … at a Glance" → الملخص: … في لمحة; table heads "Mathematical Form" → الصيغة الرياضية, "Key Systems & Statistical Insight" → أهم ما يُستفاد في الأنظمة والإحصاء; "row gather" → التقاط صفوف.
- English gloss given in u03 of this page: prompt.

## chapter_recurrent-neural-networks/text-sequence.md#u03 (claude fable-5.2 (medium), 2026-11-27)

- "special tokens" → الرموز الخاصة (as logged in u01); "control tokens" → رموز التحكم, glossed (control tokens) once; "control signal / control ID" → إشارة تحكم / معرف تحكم; "special control tokens" → رموز التحكم الخاصة.
- "prompt injection" → *حقن الموجهات* (prompt injection), from the glossary's موجه; "prompt-injection-safe" → آمنة من حقن الموجهات; "untrusted" → غير موثوق; "system instruction" → تعليمة من النظام; "safety invariant" → لامتغير أمان (the log's لامتغير); "guard" → احتياط (as logged).
- "fertility" (tokens per word) → *الخصوبة* (fertility); unit on slides: رمز مميز/كلمة; "glitch token" → *الرمز المميز المختل* (glitch token), plural الرموز المميزة المختلة (خلل stays "bug").
- "rank table" → جدول الرتب (rank table); "lookup table" → جدول بحث (as logged); "special-token registry / map" → سجل / خريطة الرموز الخاصة; "byte permutation" → تبديل البايتات, "permuted order" → ترتيب مبدَّل; "printable ASCII characters" → محارف ASCII القابلة للطباعة; "leading underscore" → الشرطة السفلية في أول الاسم.
- "encoder" of a tokenizer (text → ids) → مرمّز (from the log's يرمّز / الترميز); مشفر stays the neural encoder. "replay merges" → يعيد تطبيق عمليات الدمج; "pieces" (two parents of a merge) → جزءان (قطعة stays the pre-tokenization chunk); "segmentation" → تقطيع (تقطيعات متنافسة).
- "compression ratio" → نسبة الضغط; "context window" → نافذة السياق (glossary), glossed once; "effective context length" → طول السياق الفعلي; "web text" → نصوص الويب (WebText, the dataset name, stays Latin); "40 gigabytes" → 40 غيغابايت, "40 GB" on slides is copied.
- "positional arithmetic" → الحساب الموضعي; "columns / place values" → الخانات / القيم الموضعية للخانات; "ones, tens, hundreds" → آحاد وعشرات ومئات; "carry" → الحمل (عمليات الحمل); "thousands separators" → فواصل الآلاف; "digit cap" → سقف (سقف من ثلاثة محارف); "misaligned" → غير متحاذية.
- "Tokenizers in the Wild" → المجزئات في الواقع العملي; "widely deployed" → الواسعة الانتشار; "quirks" → غرائب; "evasions" → مراوغات; "username" → اسم مستخدم; "training scrape" → النصوص المجموعة للتدريب.
- Slides: "Exact parity" → تطابق تام (الزوجية stays bit parity); "Domain adaptation" → التكيف مع المجال; "fertility tax" → ضريبة الخصوبة; "English-centric" → المتمحور حول الإنجليزية; "50k / 200k / 1k" vocabularies → 50 ألفًا / 200 ألف / الألف (as the log's 223 ألفًا); "B/tok", "bytes/token" → بايت/رمز مميز; "Tokenization Scheme" → طريقة التجزئة; "Primary Architectural Trade-Off" → المقايضة المعمارية الرئيسة; "industry standard" → المعيار الصناعي; "(ours) / (prod)" → (مجزئنا) / (بيئة الإنتاج); "leftmost pair / rule" → الزوج الأقصى يسارًا / قاعدة الأقصى يسارًا; "Section Takeaways" → خلاصات القسم (as on the BPTT page).
- English glosses given in u03 of this page: control tokens, prompt injection, rank table, fertility, context window, glitch token.

## chapter_recurrent-neural-networks/decoding.md#u03 (claude fable-5.2 (medium), 2026-11-27)

- "cell update" (one application of the RNN cell) → تحديث للخلية (تحديثات الخلية; "$O(T^2)$ cell updates" → $O(T^2)$ تحديثًا للخلية).
- "distinct-3" stays Latin after the class noun مقياس at its first mention (مقياس *distinct-3*), then bare distinct-3; "3-grams" → تتابعات 3-gram (as the logged تتابعات $n$-gram); "unique / repeated 3-grams" → الفريدة / المكررة.
- "held-out human text" → نص بشري محجوز; "tautological(ly)" → تحصيل حاصل; "crown … the winner" → يتوّج … فائزًا; "peak likelihood" → ذروة الإمكان; "likelihood-versus-diversity trade-off" → المقايضة بين الإمكان والتنوع.
- "coherence, factuality, style" → الاتساق ومطابقة الوقائع والأسلوب; "calibrated proxy evaluators" → مقيِّمون بدلاء معايَرون; "calibrated judge evaluations" → تقييمات محكِّمين معايَرة; "blind human preference" → تفضيل بشري معمّى; "task-specific benchmarks" → الاختبارات المعيارية الخاصة بالمهمة; "reference outputs / metrics" → مخرجات / مقاييس مرجعية; "surface diagnostics" → تشخيصات سطحية; "lexical diversity" → التنوع المعجمي.
- "stateless" `step_fn` / decoding → عديم الحالة (slides: بلا حالة); "stateful replacement" → بديل ذو حالة; "caching" (past-token computations) → التخزين المؤقت; "quadratic recomputation" → إعادة الحساب التربيعية; "carry the state forward" → حمل الحالة من خطوة إلى خطوة.
- "untempered probabilities / model" → الاحتمالات قبل تحجيم درجة الحرارة / النموذج دون تحجيم درجة الحرارة; "temperature-scaled distribution" → التوزيع المحجَّم بدرجة الحرارة; "grid of values" (hyperparameter sweep) → شبكة من القيم; "bookkeeping" (per-beam state tracking) → التتبع الإضافي.
- "Part *Basics*" → جزء *الأساسيات*; "nuance" → الفروق الدقيقة; "Fast baseline" (table) → خط أساس سريع; "Temperature Sampling" → أخذ العينات بدرجة الحرارة; "MT, ASR" stay Latin (MT و ASR); "K/V vectors" → متجهات K/V; "gates/SSMs" → البوابات أو نماذج SSM; "mass-adaptive / confidence-adaptive" → متكيفة مع الكتلة / مع الثقة.
- No English glosses were added in u03 of this page (perplexity, attention, bottleneck, teacher forcing are defined on earlier pages or only named here).

## chapter_recurrent-neural-networks/text-sequence.md#u04 (claude fable-5.2 (medium), 2026-11-27)

- Exercises of this page: "rank rule" → قاعدة الرتبة; "leftmost rule" → قاعدة الأقصى يسارًا (as on the u03 slide); "to segment a string" → يقطّع (التقطيع); "occurrence" → ورود.
- "Special-token round trip" (encode, then decode) → ترميز الرموز الخاصة ثم فك ترميزها; "document separator" → فاصل المستندات; "re-encode" → يعيد الترميز; "feeds the text back in as input" → يعيد تغذية النص بوصفه مدخلات; "policy" (of a library's behaviour) → سياسة.
- "Digit chunking" → تقسيم الأرقام إلى قطع; "addition prompts" → موجهات الجمع; "the two summands" → العددان المجموعان; "place values" → الخانات (as logged in u03); "label each token with" → اكتب بجانب كل رمز مميز.
- "Pre-tokenization ablation" → استئصال التجزئة المسبقة (the log's استئصال); "held-out text" → النص المحجوز; "fraction" → كسر.
- Credit line: "Adapted from Andrej Karpathy, [minbpe](…), Step 4." → مقتبس بتصرف من مشروع [minbpe](…) لمؤلفه Andrej Karpathy، الخطوة 4. (as the micrograd line).
- English glosses in u04 of this page: none (exercises take no gloss).

## chapter_recurrent-neural-networks: chapter check (claude fable-5.2 (medium), 2026-11-27)

- One term per concept across the seven pages; the unit entries above were corrected to match. "rollout" (a model fed its own predictions, numbers or text) → التنبؤ المتتابع; a generated sample counted as an item ("50-token rollouts") → مسار التوليد; "greedy rollout" → التوليد الجشع.
- "continuation": the single next token after a context (count models, "continuation count" → تعداد التتمات) → تتمة; a generated multi-token text → تكملة.
- "global gradient clipping" → قص التدرج الكلي (as the logged المعيار $\ell_2$ الكلي; الشامل stays global pooling and global minima); "truncated BPTT" → خوارزمية BPTT المقتطعة (المبتور stays truncated sampling).
- "tokenizer" → المجزئ; "special tokens" → الرموز الخاصة; vocabulary "entries" → بنود المفردات; "granularity" (of tokens, of a vocabulary) → مستوى التفصيل.
- "trigram model" → النموذج الثلاثي; "interpolated Kneser–Ney" → تنعيم Kneser–Ney الاستيفائي, after a model name بطريقة Kneser–Ney الاستيفائية.
- "long-range dependency" → تبعية بعيدة المدى (التبعيات); "loss by position", "per-position loss" → الخسارة لكل موضع; "repetitive / repetition loops" → حلقات متكررة.
- "autoregressive model" → نموذج الانحدار الذاتي (glossary; with أي: أي نموذج من نماذج الانحدار الذاتي); the adjective on other nouns stays ذاتي الانحدار (التنبؤ، فك التشفير، توليد النص).

## chapter_optimization/index.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- Chapter-index terms of the optimization chapter: "ill-conditioning (uneven curvature)" → سوء الاشتراط (انحناء غير متساوٍ); "loss surface" → مشهد الخسارة (glossary); "steepest / flattest direction" (of curvature) → أحدّ الاتجاهات / أكثرها انبساطًا (steep → حاد, flat → منبسط; not الانحدار); "sampling variance" → تباين ناتج عن أخذ العينات; "constant-step-size iterates" → نواتج التكرار ذات حجم الخطوة الثابت; iterates "wander" → تهيم; "root-finding" → إيجاد الجذور; "decaying step sizes" → أحجام خطوات متناقصة; "update step" → خطوة التحديث; "progress stalls" → يتعثر التقدم; hyperparameters "scale" → تحجيم المعلَمات الفائقة.
- "step direction" → اتجاه الخطوة; "step-size schedule" → جدول حجم الخطوة; "coordinate-wise adaptive methods" → الطرائق التكيفية إحداثيًا بإحداثي; "orthogonalization", "orthogonalized methods" → المعامدة، طرائق المعامدة ("spectral orthogonalization" → المعامدة الطيفية; التعامد stays "orthogonality"); "preconditioned direction" → اتجاه مشروط مسبقًا; "smoothed" → منعَّم; "linear warmup" → إحماء خطي; "linear decay" → الاضمحلال الخطي; "warmup–stable–decay (WSD) schedules" → جداول الإحماء ثم الثبات ثم الاضمحلال, then جدول WSD; "early traversal" → قطع المسافة في البداية; "travel budget" → ميزانية المسير.
- "critical batch size" → حجم الدفعة الحرج; "Maximal Update Parametrization" → تحديد المعلَمات بالتحديث الأعظم, then $\mu\text{P}$ (from the log's إعادة تحديد المعلَمات); "proxy model" → نموذج بديل; "zero-shot transfer" (of a learning rate) → نقل بلا إعادة ضبط; "edge of stability" → حافة الاستقرار; "critical damping" → الإخماد الحرج; "timescale" → المقياس الزمني; "momentum velocity" → سرعة الزخم; "per-coordinate RMS normalization" → تطبيع RMS لكل إحداثي; "hidden LR" → معدل التعلم المخفي; "weight EMA" → متوسط EMA للأوزان; "count landmarks" → المعالم المرجعية القائمة على العد; "multi-seed thresholds" → عتبات متعددة البذور; "layer gradient spread" → تفاوت تدرجات الطبقات; "analytical quadratics" → الدوال التربيعية التحليلية; "quadratic valley" → وادٍ تربيعي; "NASA airfoil dataset" → مجموعة بيانات الجنيحات من NASA; "byte-level Transformer language model" → نموذج لغوي مبني على المحول على مستوى البايتات; "convolutional classifier" → المصنف الالتفافي; "byte loader" → محمّل البايتات; "practitioner playbooks" → أدلة عملية للممارسين; "annotated references" → مراجع مشروحة; "convex analysis" → التحليل المحدب; Newton–Schulz stays Latin (a compound name, like Cauchy–Schwarz).
- Slides: "Up next (§{…} Title):" → التالي (§{…} العنوان): as logged for "Next"; "Chapter thesis" → أطروحة الفصل; "Architectural Framework" → الإطار المعماري; "Pedagogical Methodology" → المنهجية التعليمية; "Chapter Map" → خريطة الفصل; "Study Guide" → دليل الدراسة; "Core practitioner sequence" → المسار الأساسي للممارس; "Obstacle 1 — X" → العقبة 1، X (no dash); "Foundations" → الأسس; "Operations" (running training) → التشغيل; "Guards" → الاحتياطات. Section titles quoted on this page: "Landscapes and Gradient Descent" → المشاهد والنزول التدرجي; "Scaling Up" → التحجيم إلى الأكبر; chapters "Attention, Transformers, State Space Models, Performance" → الانتباه والمحولات ونماذج فضاء الحالات والأداء الحسابي.
- English glosses given on this page: momentum, preconditioned, warmup, warmup–stable–decay (WSD), critical batch size, Maximal Update Parametrization, gradient clipping, weight averaging. Ill-conditioning, gradient noise, noise ball, condition number and cosine decay take their gloss on the pages that define them (here they stand in bold lead-ins, a caption or a parenthesis).

## chapter_advanced/index.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- Heading "Advanced" (part overview page) → المواضيع المتقدمة, as the logged part name; part names in prose keep the English emphasis: جزء *الأساسيات*، جزءا *النماذج اللغوية* *ونماذج الصور*.
- Bold roadmap titles with two chapter references: "(:numref:`a` and :numref:`b`)" → (:numref:`a` و :numref:`b`); a sentence that opens with a reference in English opens with its verb in Arabic (يكسر / يعود / يسأل / يبني :numref:`…`).
- "frontier foundation models" → نماذج الصدارة التأسيسية; "frontier system" → نظام من أنظمة الصدارة; "classroom prototype" → نموذج أولي من قاعة الدرس; "elementary primitives" → العناصر الأولية; "toy datasets" → مجموعات البيانات المصغرة; "photorealistic" → واقعي كالصور الفوتوغرافية; "competitive mathematics problems" → مسائل مسابقات الرياضيات; "character" → محرف.
- "optimization stability and scaling" → استقرار التحسين والتحجيم; "poorly conditioned loss surfaces" → مشاهد الخسارة سيئة الاشتراط (glossary مشهد الخسارة); "ill-conditioned curvature" → الانحناء سيئ الاشتراط; "critical batch size" → حجم الدفعة الحرج; "maximal update parametrization ($\mu$P)" → صيغة المعلَمات بالتحديث الأعظمي $\mu$P; "warmup and cosine learning-rate schedules" → جداول معدل التعلم القائمة على الإحماء وجيب التمام.
- "fixed-width hidden vector" → متجه مخفي ثابت العرض; "recurrent training" → تدريب الشبكات المتكررة; "off-chip memory / on-chip registers" → الذاكرة خارج الرقاقة / المسجلات داخل الرقاقة; "network links" → روابط الشبكة; "numerical precision" → الدقة العددية; "communication topology" → طوبولوجيا الاتصال; "raw operation count" → عدد العمليات الخام.
- "autonomous agent" → وكيل مستقل; "delayed, sparse rewards" → مكافآت متأخرة ومتناثرة; "objectives" (training) → دوال هدف; "continuous corruption processes" → عمليات إفساد مستمرة.
- Attention: "content-addressable lookup" → بحث بالعنونة بالمحتوى; "kernel regression" → انحدار النواة (Nadaraya--Watson and Bahdanau stay Latin: انحدار النواة بطريقة Nadaraya--Watson، انتباه Bahdanau); "positionwise feed-forward network" → شبكة تغذية أمامية مطبقة على كل موضع (الموضعي stays "positional"); "Transformer paradigms" → أنماط المحول; "bidirectional encoder" → مشفر ثنائي الاتجاه; "causal autoregressive decoder" → مفكك تشفير سببي ذاتي الانحدار; "key--value caching / cache" → التخزين المؤقت للمفاتيح والقيم / ذاكرة مؤقتة للمفاتيح والقيم; "Mixture-of-Experts (MoE) layers" → طبقات خليط الخبراء (Mixture-of-Experts, MoE).
- State space models: "associative scans" → عمليات المسح التجميعية (associative scans); "nonlinear state recurrence" → التكرار اللاخطي للحالة; "exact parallel evaluation" → حساب متواز مضبوط; "HiPPO polynomial projections" → إسقاطات HiPPO على كثيرات الحدود; "error-correcting delta rules" → قواعد دلتا المصححة للخطأ; "SSMs" in a gloss → (state space models, SSM).
- Performance: "roofline model" → نموذج خط السقف (roofline); "limited by arithmetic" → يحدها العمل الحسابي; "bf16 arithmetic" → الحساب بصيغة bf16; "the allreduce" → عملية allreduce (Latin, as the English writes it); "cost model" → نموذج الكلفة; "overlap of communication with the backward pass" → تداخل الاتصال مع التمرير الخلفي; "sharding" → التقسيم إلى حصص (from the logged حصة); "card" (GPU) → بطاقة.
- RL: "decision-making under uncertainty" → اتخاذ القرار في ظل عدم اليقين; "model-free temporal-difference learning" → تعلم الفرق الزمني الخالي من النموذج; "imitation learning" → التعلم بالتقليد; "optimal control" → التحكم الأمثل; "Model Predictive Control" → التحكم التنبؤي بالنموذج; "inverted pendulum" → البندول المقلوب; "variance-reducing baselines" → خطوط الأساس الخافضة للتباين; "deadly triad" → الثالوث القاتل; "Actor--Critic architectures" → معماريات الفاعل–الناقد (en dash as in the glossary's فاعل–ناقد لين); "offline RL" → التعلم المعزز دون اتصال (logged "offline"; the glossary's غير متصل is not used); "model-based RL" → التعلم المعزز القائم على نموذج; "KL-regularized control" → التحكم المنظَّم بتباعد KL; "Deep RL" → التعلم المعزز العميق. Markov and Bellman are in glossary terms: ماركوف، بيلمان.
- GANs and diffusion: "critic that scores samples in pairs" → ناقد يحسب درجات العينات أزواجًا; "far generator" → المولد البعيد; "gradient penalties" → عقوبتا التدرج; "conditional generator" → مولد شرطي; "adversarial terms" → الحدود الخصومية; "vocoder" → مولد الموجات الصوتية (vocoders) (no transliteration; مشفر stays the neural encoder); "one-step generator" → مولد ذو خطوة واحدة; "distilled" → مقطَّر; Jensen--Shannon stays Latin; "flagship models" → أبرز النماذج; "Gaussian probability path" → مسار احتمالي غاوسي; "the score" (of a density) → دالة الدرجة (glossary score function); "integrate" (an SDE/ODE) → يكامل، المكاملة; "reverse-time SDE" → المعادلة التفاضلية العشوائية بزمن عكسي (glossary); "denoising diffusion" → الانتشار بإزالة الضوضاء; "Diffusion Transformer (DiT)" → محول الانتشار (Diffusion Transformer, DiT); "average-velocity distillation" → تقطير السرعة المتوسطة; "generative paradigm" → النهج التوليدي.
- "dependency diagram" → مخطط الاعتماد; "architectural, mathematical, and systems toolkit" → عدة المعمارية والرياضيات والأنظمة; "large-scale" → واسع النطاق.

## chapter_optimization/gd.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- Page title "Landscapes and Gradient Descent" → مشاهد التحسين والنزول التدرجي (glossary مشهد التحسين / مشهد الخسارة; plural مشاهد); "loss surface" → سطح الخسارة.
- "plateau" as a flat region of a loss landscape → هضبة (هضاب), "activation plateau" → هضبة تفعيل, "saturation plateaus" → هضاب التشبع; the logged الاستواء / يستقر stay for a training curve that plateaus. "ridge" of a saddle → خط القمة (ريدج stays ridge regression).
- "steep" (axis, direction, wall, coordinate) → شديد الميل; "flat" → مستوٍ (المحور المستوي، الاتجاه المستوي، الكتلة المستوية، قيمة صغرى مستوية); "valley floor" → قاع الوادي; "steep wall" → الجدار شديد الميل; "narrow valley" → واد ضيق; "elongated" → متطاولة.
- "stability ceiling" → سقف الاستقرار (as logged), "hard" ceiling → الصارم; "stability threshold" → عتبة الاستقرار; "convergence pace" → وتيرة التقارب; "convergence time" → زمن التقارب; "maximum stable learning rate" → أكبر معدل تعلم مستقر; "iteration count" → عدد التكرارات, "step count" → عدد الخطوات; "optimal step size" → حجم الخطوة الأمثل.
- "multiplier" (per-step factor of a coordinate) → العامل الضربي, "contraction factor" → عامل الانكماش (both as logged); "the coordinates decouple" → ينفصل الإحداثيان, "decoupled updates / eigenmodes" → منفصلة; "eigenmode", "mode" (of the Hessian) → نمط ذاتي، نمط (أنماط); "sharpest mode" → النمط الأحد; "flips sign" → تنقلب إشارته; "overshoot" → يتخطى.
- "strict saddle point" → نقطة سرج صارمة; "degenerate critical point" → نقطة حرجة منحلة; "extremum" → قيمة قصوى; "inflection point" → نقطة انعطاف; "saddle surface" → سطح السرج; "stable manifold" → متعدد الشعب المستقر, "has measure zero" → قياسه صفر; "escape" (a saddle) → يفلت، الإفلات (زمن الإفلات); "(does not) trap gradient descent" → لا يعلق النزول التدرجي عندها; "basin" → حوض.
- "non-convex optimization" → التحسين غير المحدب; "expected risk" → الخطر المتوقع; "empirical objective" → دالة الهدف التجريبية; "poor / bad local minima" → قيم صغرى محلية رديئة; "geometric features" → خصائص هندسية (not ميزة, the ML feature); "local geometry" → الهندسة المحلية; "random matrix theory" → نظرية المصفوفات العشوائية; "random-sign heuristic" → قاعدة اجتهادية تفترض إشارات عشوائية.
- "Hessian spectrum" → طيف المصفوفة الهيسية; "bulk" (of eigenvalues) → كتلة; "outlier eigenvalues" → قيم ذاتية شاذة; "sharpness" → الحدة (glossary); "progressive sharpening" → الاحتداد التدريجي; "edge of stability" → حافة الاستقرار; "descent guarantee" → ضمان النزول; "$L$-smoothness" → الملاسة بثابت $L$ ("$L$-smooth" → أملس بثابت $L$).
- "positive-definite quadratic" → دالة تربيعية موجبة التعريف; "General Quadratics" → الدوال التربيعية العامة; "quadratic valley" → الوادي التربيعي; "scalar learning rate" → معدل تعلم قياسي; "intrinsic difficulty" → الصعوبة الأصيلة (not الذاتية, which is "eigen").
- Slides: "GD" → النزول التدرجي (no Latin GD, as logged); "ReLU/GELU activations, LayerNorm/BatchNorm" → دالتا التفعيل ReLU و GELU وطبقتا LayerNorm و BatchNorm (an Arabic word between Latin names); a parenthesis "($x_2$, $\lambda_2 = 4$)" → ($x_2$ حيث $\lambda_2 = 4$); kickers "Motivation" → الدافع, "Step-Size Sweep" → مسح حجم الخطوة, "Eigenbasis Analysis" → التحليل في الأساس الذاتي, "Escaping Saddles" → الإفلات من نقاط السرج; "Curvature filter" → مرشح الانحناء; "sharp cliff" → جرف حاد; "High d" in a title → حين يكبر d.
- English glosses given in u01 of this page (do not repeat in later units): ill-conditioning, saddle points, edge of stability, preconditioning, condition number, critical points, plateaus, strict saddle points, sharpness.

## chapter_optimization/gd.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- Re-applied u01 to match chapter_optimization/index.md#u01, which was logged while this unit was in work; this entry replaces the corresponding items above. Page title "Landscapes and Gradient Descent" → المشاهد والنزول التدرجي (not مشاهد التحسين والنزول التدرجي); "loss surface" → مشهد الخسارة (glossary; not سطح الخسارة; "the surface" alone stays السطح); "steep" (axis, direction, coordinate, wall) → حاد (not شديد الميل); "flat" (axis, direction, coordinate, floor, bulk, region, minimum) → منبسط (المحور المنبسط، الكتلة المنبسطة، مناطق منبسطة، قيمة صغرى منبسطة; "flattest" → الأكثر انبساطًا), so that steep, sharp and الحدة share one root.

## chapter_optimization/momentum.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "momentum" → الزخم (glossary); "heavy-ball momentum" → زخم الكرة الثقيلة; "momentum coefficient" → معامل الزخم; "velocity" → السرعة ("velocity buffer" → مخزن مؤقت للسرعة; "velocity recursion" → العلاقة التكرارية للسرعة, "unrolling the recursion" → فرد العلاقة التكرارية); "SGD with momentum" → خوارزمية SGD مع الزخم (as logged).
- "plain gradient descent", "plain GD" → النزول التدرجي العادي (no Latin "GD", as logged for full-batch GD); "plain SGD" → خوارزمية SGD العادية; "memoryless" → عديم الذاكرة.
- Valley picture: "valley floor", "flat floor" → قاع الوادي، القاع المستوي; "steep walls" → الجدران شديدة الميل; "steep / steepest" (direction, coordinate, curvature, eigenvalue) → شديد الميل / الأشد (الانحناء الأشد، الأشد ميلًا), never الانحدار (regression); "flat / flattest" → مستو / الأكثر استواءً; "trough" → أخدود; "across" (the walls) → عبر, "along" → على امتداد; "transverse" → مستعرض.
- "persistent" (gradients, components) → ثابت الاتجاه; "alternating (in sign)" → متناوب الإشارة, "alternate in sign" → تتناوب إشارتها, "sign-flipping" → تنقلب إشارته; components that "cancel" → يلغي بعضها بعضًا (noun الإلغاء); "reinforce" → تتعاضد; "amplify" → يضخم.
- "effective step size" → حجم الخطوة الفعال, "effective step" → الخطوة الفعالة (as logged); "effective memory window" → نافذة ذاكرة فعلية, "memory horizon" → أفق الذاكرة (both as logged); "timescale" → المقياس الزمني; "half-life" → عمر النصف.
- "stability ceiling / limit / threshold" → سقف / حد / عتبة الاستقرار; "stabilize" (a trajectory, a coordinate) → يجعله مستقرًا، يضمن استقراره (not يثبّت, which is "install"); "diverge" → يتباعد.
- "contraction factor" → عامل الانكماش, "per-step" → في الخطوة الواحدة, "contracts at rate" → ينكمش بالمعدل; "worst-case" → في أسوأ الحالات; "iteration count" → عدد التكرارات; "asymptotic contraction factor" → عامل الانكماش التقاربي; the iterate "overshoots" → يتجاوز; "ring", "ringing" → يتأرجح، التأرجح.
- Damping: "critical damping" → التخميد الحرج; "overdamped / critically damped / underdamped" → مفرط التخميد / حرج التخميد / ناقص التخميد (parallel to the logged مفرط الثقة / ناقص الثقة); "to damp" → يخمد (as logged); "damped step size" → حجم الخطوة المخمد.
- "mode", "eigenmode" (the dynamics along one eigenvector) → نمط، نمط ذاتي (أنماط); "extreme modes" → النمطان الطرفيان. Because نمط is the mode on this page, "regime" (damping regime, complex-root regime, noise-limited regime) → نظام (أنظمة) here, not the logged نمط.
- "acceleration" (of an optimizer) and "speedup" → التسريع ("acceleration law" → قانون التسريع); "noise-limited" (run) → محدود بالضوضاء, parallel "curvature-limited" → محدود بالانحناء; "at matched effective step size" → عند تساوي حجم الخطوة الفعال; "noise perturbation" → اضطراب ضوضائي; "noise kick" → صدمة ضوضاء (not دفعة, the batch); "integrated impulse" → النبضة المتكاملة; "cumulative displacement" → إزاحة تراكمية; "stationary noise ball" → كرة الضوضاء المستقرة; "stationary variance" → التباين المستقر.
- "characteristic polynomial" → كثير الحدود المميز; "characteristic roots" → الجذران المميزان; "discriminant" → مميِّز المعادلة (with the marks and its noun, against the GAN's المميز and الرمز المميز); "Vieta's formulas" → صيغ Vieta (Latin name); "complex-conjugate pair" → زوج مركب مترافق / جذران مركبان مترافقان; "modulus" (of a complex root) → القيمة المطلقة (as logged for eigenvalue magnitudes; مقياس stays "scale"); "second-order linear recurrence" → علاقة تكرارية خطية من المرتبة الثانية; "geometric series / sum" → المتسلسلة الهندسية / المجموع الهندسي; "envelope rate" → معدل الغلاف.
- "inertia" → قصور ذاتي; "viscous friction" → احتكاك لزج; "temporal gradient alignment" → اتساق اتجاه التدرجات عبر الزمن (محاذاة stays sequence alignment); "Nesterov look-ahead momentum" → زخم Nesterov الاستشرافي, "Nesterov look-ahead" → استشراف Nesterov, "Nesterov Accelerated Gradient" → تدرج Nesterov المسرَّع (Nesterov and Polyak stay Latin).
- Slides: "Polyak's … (1964)" → … لدى Polyak (1964); a kicker that starts with a number is reordered: "[40 Steps vs. 139 Steps]" → [الخطوات: 40 مقابل 139]; "our valley" on a slide → الوادي نفسه (no نحن).
- English glosses given in u01 of this page (do not repeat in later units): condition number, velocity, heavy-ball momentum, critical damping, effective step size, half-life, underdamped.

## chapter_optimization/test-models.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "test models" → نموذجا الاختبار (heading: نموذجان للاختبار); "benchmark" (a test model or architecture) → اختبار معياري, glossed (benchmark) once ("the two benchmark architectures" → معماريتا الاختبار المعياري); "run" → جولة (as logged); "run-to-run variation" → التفاوت من جولة إلى أخرى; "optimizer configuration" → إعداد للمحسن; "optimizer rankings" → ترتيب المحسنات; "by a wide margin" → بفارق كبير (هامش stays the margin of a threshold: "$3\times$ range margin" → هامش يساوي $3\times$ المدى).
- "learning-rate sweep" → مسح معدل التعلم, glossed (learning-rate sweep) once; "to sweep $\eta$" → يمسح، مسح $\eta$; "sweep harness" → أدوات مسح معدل التعلم; "training harness" → أدوات التدريب (as logged); "evaluation helpers / utilities" → دوال التقييم المساعدة / أدوات التقييم المساعدة; "step budget" → ميزانية من الخطوات.
- "baseline" (SGD measurements, runs, a value) → خط الأساس (خطا الأساس; "uniform-guess baseline" → خط أساس التخمين المنتظم); "uniform guess" → التخمين المنتظم; "plain SGD" → خوارزمية SGD العادية (as logged); "constant-rate" → ذات المعدل الثابت.
- "significance threshold" → عتبة الدلالة; "statistically meaningful" → ذو دلالة إحصائية; "tie", "statistical tie" → تعادل، تعادل إحصائي; "multi-seed" → على بذور متعددة; "spread" → التشتت; "range" (over seeds) → المدى.
- "landmark" (a reference loss value) → معلم (معالم الخسارة، المعالم المرجعية), glossed (landmarks) once; "unigram / bigram byte table" → جدول أحادي / ثنائي للبايتات, glossed (unigram), (bigram) once; "nats/byte" → نات لكل بايت; "untrained" on slides → بلا تدريب.
- "byte-level" → على مستوى البايتات; "shard" → حصة, glossed (shard) once; "target bytes" → البايتات المستهدفة; "fresh text" → نص جديد, on slides "fresh, unseen" → جديد لم يُر من قبل; "single-pass" → بمرور واحد; "streaming regime" → نمط السيل; "multi-epoch" → بحقب متعددة; "passes over the images" → مرور على الصور.
- "decoder-only Transformer" → محول بمفكك تشفير فقط, glossed (decoder-only) once; "attention heads" → رؤوس الانتباه; "output head" → رأس المخرجات; "readout head" → رأس القراءة; "pre-norm" → بتطبيع مسبق; "embedding width" → عرض التضمين; "scale/shift vectors" → متجهات التحجيم والإزاحة; LayerNorm takes the class noun طبقة in body text.
- "matrix orthogonalization" → تعميد المصفوفات; "orthogonalized" → تُعمَّد; "orthogonalized optimizers" → محسنات التعميد; "structural groups" → مجموعات بنيوية; "parameter census" → تعداد المعلَمات; "parameter geometry" → هندسة المعلَمات.
- "nuisance hyperparameter" → معلَمة فائقة دخيلة, glossed once; "usable learning-rate range" → نطاق معدلات التعلم الصالحة للاستخدام, glossed once; "log-spaced grid" → شبكة قيم موزعة لوغاريتميًا; "optimum" (of the grid) → القيمة المثلى; "well-tuned" → مضبوط جيدًا; "tuned score" → الدرجة بعد الضبط; "arm" (of a comparison) → طرف في المقارنة; "Deep Learning Tuning Playbook" → دليل Deep Learning Tuning Playbook (title in English).
- "stability threshold" → عتبة الاستقرار; "transient spikes" → قفزات عابرة; "sharpness edge" → حد الحدة; "cumulative travel" → المسافة التراكمية; "overshoots into a steeper basin" → تتخطى إلى حوض أشد انحناءً; "variance penalty" → عقوبة تباين; "factory closures" → الدوال المغلقة التي تصنع …; "smoothed loss" → الخسارة الممهَّدة; "block average" → متوسط الكتل; "step losses" → خسائر الخطوات.
- Slides: list items and table cells that begin with `TinyLM`, a layer name or $\eta = …$ get a class noun (النموذج، الطبقة، المعدل); arrows of the form $\to$ **action** keep the math arrow and the action is a verbal noun (رفع $\eta$، خفض $\eta$); "Where the Parameters Live" → مواضع المعلَمات; "Statistical Rigor" → الصرامة الإحصائية; "Experimental Infrastructure" → البنية التحتية للتجارب; "Pedagogical Role" → الدور التعليمي; "Key Numbers to Remember" → أرقام مهمة للتذكر.
- English glosses given in u01 of this page (do not repeat in later units): benchmark, learning-rate sweep, shard, decoder-only, landmarks, unigram, bigram, nuisance hyperparameter, usable learning-rate range.

## chapter_optimization/momentum.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- Re-applied u01 to match chapter_optimization/index.md#u01 and gd.md#u01, which were logged while this unit was in work; this entry replaces the corresponding items of the momentum.md#u01 entry above. "steep / steepest" (wall, direction, coordinate, curvature, eigenvalue) → حاد / الأحد (الجدران الحادة، الانحناء الأحد; not شديد الميل); "flat / flattest" → منبسط / الأكثر انبساطًا (القاع المنبسط، الإحداثي المنبسط; not مستو); "critical damping" → الإخماد الحرج, "overdamped / critically damped / underdamped" → مفرط الإخماد / حرج الإخماد / ناقص الإخماد, "damping regimes" → أنظمة إخماد (not التخميد); the iterate "overshoots" → يتخطى (kicker "Overshoot Hazard" → خطر التخطي), يتجاوز stays "exceeds". All other items of the entry above stand, including "regime" → نظام on this page.

## chapter_optimization/momentum.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- Re-applied u01 once more: Nesterov is written نيستيروف in running text, because the name is in the glossary term زخم نيستيروف (style §7): "Nesterov look-ahead momentum" → زخم نيستيروف الاستشرافي, "Nesterov look-ahead" → استشراف نيستيروف, "Nesterov Accelerated Gradient" → تدرج نيستيروف المسرَّع. This replaces the Latin "Nesterov" of the first momentum.md#u01 entry; Polyak and Vieta stay Latin (no glossary term carries the names), and citations (:citet:) print the Latin form.

## chapter_optimization/test-models.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- "test models" → نموذجا الاختبار (heading: نموذجان للاختبار); "benchmark" (a test model or architecture) → اختبار معياري, glossed (benchmark) once ("the two benchmark architectures" → معماريتا الاختبار المعياري); "run" → جولة (as logged); "run-to-run variation" → التفاوت من جولة إلى أخرى; "optimizer configuration" → إعداد للمحسن; "optimizer rankings" → ترتيب المحسنات; "by a wide margin" → بفارق كبير (هامش stays the margin of a threshold: "$3\times$ range margin" → هامش يساوي $3\times$ المدى).
- "learning-rate sweep" → مسح معدل التعلم, glossed (learning-rate sweep) once; "to sweep $\eta$" → يمسح، مسح $\eta$; "sweep harness" → أدوات مسح معدل التعلم; "training harness" → أدوات التدريب (as logged); "evaluation helpers / utilities" → دوال التقييم المساعدة / أدوات التقييم المساعدة; "step budget" → ميزانية من الخطوات.
- "baseline" (SGD measurements, runs, a value) → خط الأساس (خطا الأساس; "uniform-guess baseline" → خط أساس التخمين المنتظم); "uniform guess" → التخمين المنتظم; "plain SGD" → خوارزمية SGD العادية (as logged); "constant-rate" → ذات المعدل الثابت.
- "significance threshold" → عتبة الدلالة; "statistically meaningful" → ذو دلالة إحصائية; "tie", "statistical tie" → تعادل، تعادل إحصائي; "multi-seed" → متعدد البذور as an adjective (عتبة دلالة متعددة البذور, as on the chapter index), على بذور متعددة in a phrase (التشتت على بذور متعددة); "spread" → التشتت; "range" (over seeds) → المدى.
- "landmark" (a reference loss value) → معلم (معالم الخسارة، المعالم المرجعية), glossed (landmarks) once; "unigram / bigram byte table" → جدول أحادي / ثنائي للبايتات, glossed (unigram), (bigram) once; "nats/byte" → نات لكل بايت; "untrained" on slides → بلا تدريب.
- "byte-level" → على مستوى البايتات; "shard" → حصة, glossed (shard) once; "target bytes" → البايتات المستهدفة; "fresh text" → نص جديد, on slides "fresh, unseen" → جديد لم يُر من قبل; "single-pass" → بمرور واحد; "streaming regime" → نمط السيل; "multi-epoch" → بحقب متعددة; "passes over the images" → مرور على الصور.
- "decoder-only Transformer" → محول بمفكك تشفير فقط, glossed (decoder-only) once; "attention heads" → رؤوس الانتباه; "output head" → رأس المخرجات; "readout head" → رأس القراءة; "pre-norm" → بتطبيع مسبق; "embedding width" → عرض التضمين; "scale/shift vectors" → متجهات التحجيم والإزاحة; LayerNorm takes the class noun طبقة in body text.
- "matrix orthogonalization" → معامدة المصفوفات; "orthogonalized" (matrices) → تخضع للمعامدة; "orthogonalized optimizers" → محسنات المعامدة (المعامدة as on the chapter index; this replaces the تعميد of the first application of this unit); "structural groups" → مجموعات بنيوية; "parameter census" → تعداد المعلَمات; "parameter geometry" → هندسة المعلَمات.
- "nuisance hyperparameter" → معلَمة فائقة دخيلة, glossed once; "usable learning-rate range" → نطاق معدلات التعلم الصالحة للاستخدام, glossed once; "log-spaced grid" → شبكة قيم موزعة لوغاريتميًا; "optimum" (of the grid) → القيمة المثلى; "well-tuned" → مضبوط جيدًا; "tuned score" → الدرجة بعد الضبط; "arm" (of a comparison) → طرف في المقارنة; "Deep Learning Tuning Playbook" → دليل Deep Learning Tuning Playbook (title in English).
- "stability threshold" → عتبة الاستقرار; "transient spikes" → قفزات عابرة; "sharpness edge" → حافة الحدة (as حافة الاستقرار); "under-budgeted travel" → ميزانية مسير غير كافية, "cumulative travel" → المسير التراكمي (the index's ميزانية المسير); "overshoots into a steeper basin" → تتخطى إلى حوض أحد (steep → حاد, as on the gd page); "variance penalty" → عقوبة تباين; "factory closures" → الدوال المغلقة التي تصنع …; "smoothed loss" → الخسارة المنعَّمة (the chapter index's منعَّم); "block average" → متوسط الكتل; "step losses" → خسائر الخطوات.
- Slides: list items and table cells that begin with `TinyLM`, a layer name or $\eta = …$ get a class noun (النموذج، الطبقة، المعدل); arrows of the form $\to$ **action** keep the math arrow and the action is a verbal noun (رفع $\eta$، خفض $\eta$); "Where the Parameters Live" → مواضع المعلَمات; "Statistical Rigor" → الصرامة الإحصائية; "Experimental Infrastructure" → البنية التحتية للتجارب; "Pedagogical Role" → الدور التعليمي; "Key Numbers to Remember" → أرقام مهمة للتذكر.
- English glosses given in u01 of this page (do not repeat in later units): benchmark, learning-rate sweep, shard, decoder-only, landmarks, unigram, bigram, nuisance hyperparameter, usable learning-rate range.

## chapter_optimization/gd.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "$L$-smooth" → ملساء بثابت $L$, glossed (L-smooth) in plain text (no added math span); "$L$-Lipschitz continuous" → مستمر ليبشيتزيًا بثابت $L$ (glossary); "finite step" → خطوة منتهية; "touching parabola" → قطع مكافئ مماس; "gradient step" → خطوة تدرج; "non-critical point" → نقطة غير حرجة; "linearly stable" → مستقرة خطيًا (glossary استقرار خطي); "stability limit" → حد الاستقرار, "stability threshold" → عتبة الاستقرار, "stability ceiling" → سقف الاستقرار (three English words, three Arabic ones).
- "the learning rate filters minima" → يرشح; "selection" (of minima by sharpness) → الانتقاء، ينتقي; "deep / shallow minimum" → قيمة صغرى عميقة / ضحلة; "sharper / flatter minimum" → أحد / أكثر انبساطًا; "sharpest eigenvector" → المتجه الذاتي الأحد; "run" → جولة (as logged); "1D gradient descent runner" → الدالة التي تشغّل النزول التدرجي أحادي الأبعاد; "trajectory plotter" → الدالة التي ترسم المسار.
- "progressive sharpening" → الاحتداد التدريجي; "hovers near" → تحوم قرب; "operating regime" → نمط التشغيل; "normalized sharpness" → الحدة المطبَّعة; "exact Hessian" → المصفوفة الهيسية المضبوطة; "arrest at" → تتوقف عند; (sharpness) "plateaus" → تستقر (as logged); "restoring drift" → انجراف إرجاع (logged قوة إرجاع, glossary انجراف); "self-stabilization" → الاستقرار الذاتي; "learning-rate warmup" → إحماء معدل التعلم; "the warmup argument" → حجة الإحماء; "preconditioned sharpness" → الحدة المشروطة مسبقًا; "gradient scale" → مقياس التدرج; "momentum coefficient" → معامل الزخم; "third-order derivatives" → المشتقات من المرتبة الثالثة (slide: المشتقات الثالثة).
- Slides: "Empirical Demonstration" → البيان التجريبي; "Key principle" → المبدأ الأساسي; "catapulted out of the basin" → يُقذف خارج الحوض; bold sub-bullets that begin with a math span get the lead word عند; "Second-Order Optimization" → التحسين من المرتبة الثانية; "exact Hessian inversion" → العكس المضبوط للمصفوفة الهيسية (logged عكس المصفوفات); "exact Newton steps" → خطوات نيوتن المضبوطة; "Saddle attraction" → الانجذاب إلى نقاط السرج; "Noise amplification" → تضخيم الضوضاء; "explodes (the variance)" → يفجّر.
- English glosses given in u02 of this page: L-smooth, descent lemma, linearly stable, progressive sharpening, learning-rate warmup, preconditioned sharpness. Heavy-ball momentum is glossed on the momentum page.

## chapter_optimization/sgd.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- Page title "Stochastic and Minibatch Gradient Descent" → النزول التدرجي العشوائي وبالدفعات المصغرة; "sampled gradients" → تدرجات العينات; "single-example SGD" → خوارزمية SGD بمثال واحد; "constant-step SGD" → خوارزمية SGD بخطوة ثابتة; "constant-rate iterates" → نواتج التكرار بمعدل ثابت; "exact gradient / Hessian / minimizer / convergence" → المضبوط (as the log's "exact" → مضبوط); "the optimum" → القيمة المثلى; "point minimum" → نقطة صغرى واحدة.
- "stationary" (variance, noise ball, excess loss, error, displacement, phase: the stationary distribution of SGD) → ساكن (التباين الساكن، كرة الضوضاء الساكنة، الخسارة الزائدة الساكنة), from the glossary's توزيع ساكن, so that it stays apart from "stable" → مستقر (حجم خطوة مستقر، سقف الاستقرار) on the optimization pages; "at steady state" → في الحالة المستقرة (as logged); "transient phase" → المرحلة العابرة, "stationary phase" → المرحلة الساكنة (طور stays the phase of a wave).
- Following gd.md of this chapter: "flat" (axis, coordinate, direction) → مستوٍ, "steep" → شديد الميل, "elongated cloud" → سحابة متطاولة, "mode" → نمط (أنماط).
- "with replacement" → مع الإرجاع; "without replacement" → دون إرجاع; "finite population" → مجتمع منته; "finite-population correction factor" → عامل تصحيح المجتمع المنتهي; "random reshuffling" → إعادة الخلط العشوائي; "disjoint minibatches" → دفعات مصغرة غير متقاطعة; "cancel to first order" → يلغي بعضها بعضًا حتى المرتبة الأولى.
- "regime" here → نطاق (the log's range sense): "small-step regime" → نطاق الخطوات الصغيرة, "linear regime" → النطاق الخطي, "linear scaling regime" → نطاق التحجيم الخطي, "interpolating regime" → نطاق الاستيفاء, "noise-limited regime" → النطاق المحدود بالضوضاء, "underparameterized regime" → نطاق نقص المعلَمات; "interpolating models" → النماذج المستوفية.
- "scaling" as a proportionality law ("inverse curvature scaling", "linear step-size scaling", "variance scales as") → التناسب، يتناسب مع; "linear scaling $\eta \propto b$" (of the learning rate with batch size) → التحجيم الخطي (the log's تحجيم معدل التعلم); table header "Scaling Dependence" → علاقة التناسب.
- "strongly convex" → محدب بقوة; "polynomial schedules" → جداول معدل التعلم كثيرة الحدود; "warmup–stable–decay schedules" → جداول الإحماء ثم الثبات ثم الاضمحلال (as on the chapter page, which glosses it); "cumulative step budget" → ميزانية الخطوات التراكمية; "travel budget" → ميزانية المسير (as on the chapter page); "per-step contraction rate" → معدل الانكماش في الخطوة الواحدة; "noise injection", "injected noise" → حقن الضوضاء، الضوضاء المحقونة; "noise amplitude" → سعة الضوضاء; "tail variance" → تباين الذيل.
- "hardware amortization" → توزيع الكلفة الثابتة للعتاد, "amortizes overhead across" → يوزع العبء الإضافي على; "vector processors" → المعالجات المتجهية; "CPUs and GPUs" → وحدات CPU ووحدات GPU; "examples processed" → الأمثلة المعالَجة; "critical batch size" → حجم الدفعة الحرج (as on the chapter page); "curvature ceiling" → سقف الانحناء.
- "NASA airfoil self-noise dataset" → مجموعة بيانات الضوضاء الذاتية للجنيحات من NASA, then مجموعة بيانات الجنيحات (as on the chapter page); "wind-tunnel measurements" → قياسات في نفق هوائي; "sound-pressure target" → هدف هو ضغط الصوت.
- Slides: kickers "Stochastic Gradients" → التدرجات العشوائية, "Valley Simulation" → محاكاة الوادي, "Second-Moment Recursion" → العلاقة التكرارية للعزم الثاني, "Geometric Interpretation" → التفسير الهندسي, "Empirical Verification" → التحقق التجريبي, "Decaying Step Sizes" → أحجام الخطوات المتناقصة, "Where Training Stalls" → موضع تعثر التدريب, "Exact Data Measurements" → قياسات مضبوطة على البيانات, "Theory vs. Experiment" → النظرية مقابل التجربة, "Practical Data Loaders" → محملات البيانات العملية, "Throughput Benchmark" → اختبار معياري للإنتاجية, "Progress per Example" → التقدم لكل مثال; "Robbins–Monro conditions" → شرطا Robbins–Monro (names in Latin); "LR decay" → اضمحلال معدل التعلم; "Noise Trace" → أثر مصفوفة الضوضاء; "anisotropic" → غير متناحٍ; "Limiting checks" → التحقق عند الحالتين الحديتين.
- English glosses given in u01 of this page (do not repeat in later units): full-batch gradient descent, stochastic gradient descent (SGD), minibatch, noise ball, gradient noise, stationary variance, noise floor, strongly convex, with replacement, excess loss, interpolating regime, without replacement, finite-population correction factor, random reshuffling, hardware amortization, sample efficiency, airfoil self-noise.

## chapter_optimization/test-models.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "anchor the loss scale" → ترسي مرتكزات مقياس الخسارة (the logged مرتكزات for "anchors"; not يثبّت, "install"); "hyperparameter and seed sensitivity" → الحساسية تجاه المعلَمات الفائقة والبذور; "optimal loss" → الخسارة المثلى.
- "edge of divergence" → حافة التباعد (as حافة الاستقرار); "survival rate" → معدل النجاة, "surviving loss / runs" → الجولات الناجية; "divergent runs" → الجولات المتباعدة; a run "stalls" → تتعثر (the chapter index's يتعثر التقدم); "fraction of dead ReLU activations" → نسبة تفعيلات ReLU الميتة.
- "sample standard deviation" → الانحراف المعياري للعينة; "normal range estimator" → مقدر المدى للتوزيع الطبيعي; "at two standard errors" → عند خطأين معياريين; "uniform baseline" → خط الأساس المنتظم; "logarithmic step axis" → محور خطوات لوغاريتمي.
- "add-one smoothed bigram" → النموذج الثنائي بتنعيم إضافة الواحد; "count-based $n$-gram landmarks" → معالم $n$-gram القائمة على العد; "held-out training windows" → نوافذ تدريب محجوزة; "final checkpoint" → نقطة الاستعادة الأخيرة; "restrict the corpus to" → اقصر المدونة النصية على.
- No English glosses added in u02 of this page (summary and exercises only).

## chapter_optimization/test-models.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "anchor the loss scale" → ترسي مرتكزات مقياس الخسارة (the logged مرتكزات for "anchors"; not يثبّت, "install"); "hyperparameter and seed sensitivity" → الحساسية تجاه المعلَمات الفائقة والبذور; "optimal loss" → الخسارة المثلى.
- "edge of divergence" → حافة التباعد (as حافة الاستقرار); "survival rate" → معدل النجاة, "surviving loss / runs" → الجولات الناجية; "divergent runs" → الجولات المتباعدة; a run "stalls" → تتعثر (the chapter index's يتعثر التقدم); "fraction of dead ReLU activations" → نسبة تفعيلات ReLU الميتة.
- "sample standard deviation" → الانحراف المعياري للعينة; "normal range estimator" → مقدر المدى للتوزيع الطبيعي; "at two standard errors" → عند خطأين معياريين; "uniform baseline" → خط الأساس المنتظم; "logarithmic step axis" → محور خطوات لوغاريتمي.
- "add-one smoothed bigram" → النموذج الثنائي بتنعيم إضافة الواحد; "count-based $n$-gram landmarks" → معالم $n$-gram القائمة على العد; "held-out training windows" → نوافذ تدريب محجوزة; "final checkpoint" → نقطة الاستعادة الأخيرة; "restrict the corpus to" → اقصر المدونة النصية على.
- No English glosses added in u02 of this page (summary and exercises only).

## chapter_optimization/sgd.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- Re-applied u01 to match chapter_optimization/gd.md#u01 (re-applied) and momentum.md#u01, which were logged while this unit was in work; this entry replaces the corresponding items of the sgd.md#u01 entry above. "stationary" (variance, noise ball, excess loss, error, displacement, phase) → مستقر (التباين المستقر، كرة الضوضاء المستقرة، الخسارة الزائدة المستقرة، المرحلة المستقرة), as on the momentum page and in the log's time-series entry, not ساكن; where "stable" and "stationary" meet in one sentence, "stable step size" → حجم خطوة يحفظ الاستقرار. "flat" (axis, coordinate, direction) → منبسط and "steep" → حاد (not مستوٍ / شديد الميل). All other items of the entry above stand, including "regime" → نطاق on this page.

## chapter_optimization/muon.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- Page title "Muon" → المحسن Muon (class noun so that the heading starts in Arabic; Muon is masculine in running text, and bare after a verb or preposition: يحقق Muon، في Muon); "the hybrid Muon+AdamW optimizer" → المحسن الهجين Muon+AdamW; the optimizer list "SGD, Momentum, and AdamW" → SGD والزخم و AdamW (Momentum as the glossary's زخم, not Latin).
- "steepest descent" → النزول الأشد (as logged on the calculus page), glossed once; "direction of steepest descent", table "steepest direction" → اتجاه النزول الأشد; "spectral steepest descent" → النزول الأشد الطيفي; "sign descent" → نزول الإشارة (sign descent); "norm ball" → كرة معيار (كرات المعايير); "norm geometries" → هندسات المعايير; "Euclidean ball" → الكرة الإقليدية; "coordinate box" → صندوق الإحداثيات; "coordinate-wise $\ell_\infty$ box norm" → معيار الصندوق $\ell_\infty$ إحداثيًا بإحداثي.
- "orthogonalized gradient" → التدرج المعامَد (with the fatha, against متعامد "orthogonal"; from the logged المعامدة), "its orthogonalization" → معامدتها; "singular value / vectors / direction" → قيمة مفردة / المتجهات المفردة / اتجاه مفرد ("active" → نشط); because مفرد is "singular" on this page, "individual matrix entries" → عناصر المصفوفة كل على حدة; "reduced SVD" → تحليل القيم المفردة المختزل.
- "memoryless limit", "zero-memory limit" (of Adam, of Shampoo) → الحالة الحدية عديمة الذاكرة / ذات الذاكرة الصفرية; slide "memoryless Shampoo" → Shampoo بلا ذاكرة.
- "linearized objective" → دالة الهدف المقرَّبة خطيًا; table "linearized decrease" → النقصان في التقريب الخطي; "dual norm" → المعيار الثنوي (glossary ثنوية); "dual nuclear norm" → المعيار النووي الثنوي; "operator norm" given as a synonym → (معيار المؤثر، operator norm، :numref:`…`).
- "odd matrix polynomial" → كثير حدود مصفوفي فردي (كثيرات الحدود المصفوفية الفردية); "Newton–Schulz iteration" → تكرار Newton–Schulz; "quintic" → كثير الحدود الخماسي, "cubic" → كثير الحدود التكعيبي; "tuned quintic" → كثير الحدود الخماسي مضبوط المعاملات (مضبوط alone is "exact" in this book, so the coefficients are named).
- "cyclic trace invariance" → ثبات أثر المصفوفة تحت التبديل الدوري; slide "trace duality" → ثنوية أثر المصفوفة; "cancel identically" → تلغي بعضها بعضًا تمامًا; "perturbation shifts the output" → يزيح الاضطراب المخرجات; "activation shift" → إزاحة التفعيلات; "max directional stretch" → أقصى تمديد اتجاهي.
- "Kronecker-factored matrix preconditioning" → الاشتراط المسبق المصفوفي المحلل إلى عوامل Kronecker (Kronecker and Gram stay Latin: مصفوفتا Gram اليسرى واليمنى); "matrix preconditioners" → المُشرِّطات المسبقة المصفوفية; "full-matrix preconditioning" → اشتراط مسبق بالمصفوفة الكاملة; "historical accumulation" → تراكم تاريخي; "Nesterov momentum buffers" → مخازن زخم نيستيروف (glossary); "first-moment momentum buffer" → مخزن زخم للعزم الأول.
- "hidden 2D weight matrices" → مصفوفات الأوزان المخفية ثنائية الأبعاد; "1D parameters / biases" → المعلَمات / الانحيازات أحادية الأبعاد; "hidden projection matrices" → مصفوفات إسقاط مخفية; "output head" → رأس المخرجات, "(vocabulary) readout head" → رأس القراءة (النهائي على المفردات); "one-hot row lookups" → استخراج الصفوف بالترميز الأحادي; "LayerNorm scales" → مقاييس LayerNorm; "activation spaces" → فضاءات التفعيلات; "parameter partitioning" → تقسيم المعلَمات; "suited parameter group" → مجموعة المعلَمات المناسبة.
- The expansion of the name Muon stays in English: (اختصار *Momentum Orthogonalized by Newton–Schulz*، وقد اقترحه :citet:`…`), so that an Arabic word separates the Latin name from the citation.
- Slides: "permutation blindness of entrywise methods" → عمى الطرائق العاملة عنصرًا بعنصر عن التبديل ("entrywise" → عنصرًا بعنصر); "equalization" → التسوية; "starve a direction" → تحرم الاتجاه; "feature mode" → نمط ميزات; "Unifying Preconditioners" → توحيد المُشرِّطات المسبقة; "two-step GPU recipe" → وصفة من خطوتين على وحدات GPU; "matmul steps" → خطوات من جداء المصفوفات; "amplification" → تضخيم; "band" (of singular values) → نطاق; "converges to $1$ asymptotically" → يتقارب إلى $1$ في النهاية; slide citations (Bernstein & Newhouse, 2024) are copied in Latin.
- English glosses given in u01 of this page (do not repeat in later units): steepest descent, sign descent, spectral norm, singular value, orthogonalized gradient, singular value decomposition (SVD), odd matrix polynomial, momentum buffers, dual norm, operator norm, nuclear norm. Newton–Schulz, Shampoo, Kronecker and Gram are names and take no gloss; Frobenius norm and preconditioning were glossed on earlier pages.

## chapter_optimization/momentum.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- Section headings: "Momentum on Data" → الزخم على البيانات; "Nesterov Momentum" → زخم نيستيروف (glossary; نيستيروف in Arabic letters in running text, style §7; the slide kicker "Heavy-Ball vs. Nesterov" → الكرة الثقيلة مقابل نيستيروف).
- "curvature-limited / noise-limited regime" → النظام المحدود بالانحناء / المحدود بالضوضاء (regime → نظام on this page, as in u01); "stationary noise floor" → أرضية الضوضاء المستقرة; "convergence speed" → سرعة التقارب.
- "look-ahead point" → نقطة الاستشراف, glossed once; "look-ahead correction" → التصحيح الاستشرافي; "anticipatory damping" → إخماد استباقي; "restoring correction" → تصحيح إرجاع (from the logged قوة إرجاع); "suppress" (oscillations, ringing) → يكبت; "brakes the velocity" → يكبح السرعة; "longitudinal overshoot" → التخطي الطولي; "overshoot the minimum" → يتخطى القيمة الصغرى.
- "evaluate the gradient at" → يحسب التدرج عند (style §5.3); slide "curvature evaluation" → تقدير الانحناء.
- "smooth, strongly convex function" → دالة ملساء شديدة التحدب; "general smooth convex objective" → هدف محدب أملس عام (glossary هدف محدب); "increasing schedule" (of $\beta_t$) → الجدول المتزايد; "iteration complexity", "step complexity" → تعقيد عدد التكرارات / تعقيد عدد الخطوات.
- "robust default" → قيمة افتراضية متينة; "standard default" → قيمة افتراضية معيارية; "a single scalar step size" → حجم خطوة قياسي واحد; "gradient scales" → مقاييس التدرجات; "per-coordinate adaptive scaling" → تحجيم تكيفي لكل إحداثي.
- "preload onto the accelerator" → يحمّل مسبقًا إلى المسرع; "in-memory loader" → محمِّل في الذاكرة; "zero-initialized" → مهيأ بالأصفار; "momentum-parameterized SGD constructor `make_sgd(beta)`" → الدالة `make_sgd(beta)` التي تنشئ محسن SGD بمعلَمة الزخم; "sweep `d2l.lr_sweep` across `grids`" → نجري المسح بالدالة `d2l.lr_sweep` على شبكات القيم `grids`; "configuration" → إعداد; "target loss / final loss" → الخسارة المستهدفة / الخسارة النهائية; "single-seed variation" → تفاوت البذرة الواحدة; "factor-of-two grid" → شبكة القيم ذات المعامل اثنين.
- A numeric range between two math spans in prose ("$\eta \approx 0.1$–$0.2$", "$5$–$10$") → بين … و … / من … إلى … (style §3.2; no en dash).
- Exercises: "Tabulate" → ضع في جدول; "empirical optimum" → القيمة المثلى التجريبية; "noise-to-signal ratio" → نسبة الضوضاء إلى الإشارة (as logged); "stationary mean" → متوسط مستقر; "by induction" → بالاستقراء; "library form" → صيغة المكتبات; "strictly inside the unit circle" → داخل دائرة الوحدة تمامًا; "normalized EMA form" → صيغة EMA المطبَّعة; "exponentially averaged momentum form" → صيغة الزخم بالمتوسط الأسي; "noisy gradient" → التدرج المشوب بالضوضاء; "i.i.d." in an exercise → مستقلة ومتماثلة التوزيع (as logged).
- Slides: "Batch size transition" → الانتقال مع حجم الدفعة; "capped by the noise ball" → تضع كرة الضوضاء سقفًا لـ; table headers "Concept / Rule" → المفهوم أو القاعدة, "Mathematical Expression" → التعبير الرياضي, "Key Property" → الخاصية الأساسية, "Practical Guideline" → الإرشاد العملي; "state buffer" → مخزن حالة مؤقت; "(answers)" in a title → (الإجابات).
- English glosses given in u02 of this page: look-ahead point.

## chapter_optimization/sgd.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "sampling variance" → التباين الناتج عن أخذ العينات (as on the chapter page); "amortizing dispatch and memory-bandwidth costs across the minibatch" → توزع على الدفعة المصغرة كلفة الإرسال وكلفة عرض نطاق الذاكرة; "the flattest directions" → الاتجاهات الأكثر انبساطًا; "the model interpolates every training example" → يستوفي النموذج كل مثال تدريب; "transient progress" → التقدم العابر; "deterministic curvature ceiling" → سقف الانحناء الحتمي.
- Exercises: "asymptotic ansatz" → الصيغة التقاربية المفترضة; "iterate the recursion" → كرر العلاقة التكرارية; "log–log axes" → محوران لوغاريتميان (as the log's رسم لوغاريتمي المحورين); "per-epoch reshuffling" → إعادة الخلط في كل حقبة; "square-root scaling" → تحجيم الجذر التربيعي; "configurations" → الإعدادات; "broken down across" → مفصلًا على; "a two-layer ReLU MLP" → شبكة MLP من طبقتين بدالة التفعيل ReLU; a list of code names (`W1`, `b1`, …) is joined by و.
- No English glosses added in u02 of this page (summary and exercises only).

## chapter_optimization/gd.md#u03 (claude fable-5.2 (medium), 2026-11-27)

- "preconditioner" → المُشرِّط المسبق (glossary, with its marks); "preconditioned gradient descent" → النزول التدرجي المشروط مسبقًا; "diagonal preconditioner / preconditioning" → المُشرِّط المسبق القطري / الاشتراط المسبق القطري; "diagonally preconditioned Hessian" → المصفوفة الهيسية المشروطة مسبقًا قطريًا; "curvature preconditioner" → مُشرِّط مسبق للانحناء; "full-matrix preconditioning" → الاشتراط المسبق بمصفوفة كاملة; "effective condition number" → عدد الشرط الفعال.
- "second-order steps / methods" → خطوات / طرائق المرتبة الثانية; "second-order Taylor approximation" → تقريب تايلور من المرتبة الثانية; "Newton update" → تحديث نيوتن; "quadratic convergence" → تقارب تربيعي; "convergence order" → مرتبة التقارب; "regular local minimum" → قيمة صغرى محلية نظامية; "accurate digits" → الأرقام الصحيحة; "to invert" (the Hessian, an estimate) → عكس (as logged for matrix inversion); negative eigenvalues "invert the step direction" → تعكس اتجاه الخطوة; "backpropagation passes" → تمريرات الانتشار العكسي; "absolute-spectrum projections" → إسقاطات الطيف المطلق; "damping" → الإخماد (as logged).
- "axis-aligned" → محاوره موازية لمحاور الإحداثيات (as logged), short forms الموازي للمحاور ("axis-aligned ill-conditioning" → سوء الاشتراط الموازي للمحاور, "non-axis-aligned directions" → الاتجاهات غير الموازية للمحاور); "principal curvature directions" → اتجاهات الانحناء الرئيسة; "online" (estimated during training) → آنيًا; "per-coordinate scales" → المقاييس لكل إحداثي; "affordable" → ميسور الكلفة.
- Exercises: "global minimizer" → نقطة تصغير شاملة (glossary نقطة التصغير); "hyperbola" → القطع الزائد; "scalar linear network" → شبكة خطية قياسية; "incoming weights" → أوزان واردة; "sign-flip symmetry" → تناظر قلب الإشارة; "symmetric distributions" → توزيعات متماثلة (the matrix stays متناظرة); "uniformly chosen" → مختارة بانتظام; "Wigner's semicircle law" → قانون Wigner لنصف الدائرة (name in Latin); "limiting fraction" → النسبة الحدية; "realization" → تحقق (as logged); "coin-flip estimate" → تقدير رمي العملة; "bounded below" → محدودة من الأسفل; "squared gradient norm" → مربع معيار التدرج; "escape time" → زمن الإفلات; "within reach" → التي يمكن بلوغها; "below the edge" → تحت الحافة.
- Slides: "Affordable Curvature Scaling" → تحجيم بالانحناء ميسور الكلفة; "Work out … for each scenario:" → المطلوب: حساب … في كل حالة: (as logged); "pitfall" → مزلق (as logged); table headers "Concept / Phenomenon" → المفهوم / الظاهرة, "Mathematical Formula" → الصيغة الرياضية, "Scaling & Stability Behavior" → سلوك التناسب والاستقرار (how a quantity scales → التناسب; التحجيم stays rescaling), "Practical Implication" → الدلالة العملية; a table cell that begins with $\eta$ gets the class noun المعدل.
- English glosses given in u03 of this page: quadratic convergence, preconditioned gradient descent, preconditioner, diagonal preconditioner. طريقة نيوتن takes no gloss (a name).

## chapter_optimization/adam.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- Optimizer names take the class noun محسن where one is needed (محسن Adam، محسنا Adam و AdamW، محسنا AdaGrad و RMSProp), SGD keeps خوارزمية SGD (as logged); after a preposition or as a later mention the bare Latin name stands. "Adaptive Moment Estimation" → the Latin expansion once, then تقدير العزوم التكيفي (kicker: تقدير العزوم التكيفي).
- "per-coordinate" → لكل إحداثي (معدلات التعلم التكيفية لكل إحداثي، التحجيم لكل إحداثي، مقياس RMS لكل إحداثي); "gradient scale" → مقياس التدرج (مقاييس التدرج), "historical gradient scale" → مقياس تدرجاته السابقة; "gradient magnitude" → مقدار التدرج; "raw gradients" → التدرجات الخام; "step length" → طول الخطوة ("direct step length" → طول خطوة مباشر; حجم الخطوة stays "step size"); "nominal learning rate" → معدل التعلم الاسمي; "nats/byte" → نات لكل بايت.
- "RMS preconditioner" → مُشرِّط مسبق … يقوم على جذر متوسط المربعات, then المُشرِّط المسبق (القطري، التكيفي); "RMS gradient" → جذر متوسط المربعات للتدرج / مقدار RMS للتدرج / قيم RMS للتدرجات; "equilibrium norm" → معيار التوازن; "weight-decay timescale" → المقياس الزمني لاضمحلال الأوزان; a weight's "decay rate" → معدل اضمحلال الوزن; "coupled / decoupled decay" → الاضمحلال المقترن / المنفصل.
- "cumulative sum", "running sum" → المجموع التراكمي; "leaky moving average" → متوسط متحرك متسرب; "ancient gradients" → التدرجات القديمة; "non-vanishing step size" → حجم خطوة لا يتلاشى; "oversized jump" → وثبة مفرطة الحجم (وثبة for "jump"; قفزة stays "spike"); "cold-start" → البداية الباردة; "uncentered second moment" → العزم الثاني غير المركزي; "bias-corrected" → بعد تصحيح الانحياز; "exact" → مضبوط; the two Adam "buffers" $(\mathbf{m}, \mathbf{v})$ → مخزنا الحالة (the log's مخزن حالة for persistent non-trainable state).
- "functional parameter kinds / groups" → أنواع / مجموعات المعلَمات الوظيفية; a "$40\times$ range" → مدى قدره $40\times$; a "$2\times$ band" → نطاق ضيق عرضه $2\times$; "usable learning-rate window" → نافذة معدلات التعلم الصالحة للاستخدام; "target loss" → الخسارة المستهدفة; "final loss" → الخسارة النهائية; "remains finite" → تبقى خسارته منتهية; "one grid step below divergence" → على بعد خطوة واحدة من خطوات شبكة القيم دون التباعد; "bounded by $\eta$" → محدود من الأعلى بالقيمة $\eta$; "block-level Hessian heterogeneity" → عدم تجانس المصفوفة الهيسية على مستوى الكتل; "heavy-tailed token frequencies" → تواترات الرموز المميزة ثقيلة الذيل; "subword vocabularies" → مفردات الكلمات الفرعية.
- "counterexample" → مثال مضاد; "pathology" (of an optimizer) → اعتلال; "running maxima" → قيم عظمى تراكمية; "sign descent" → نزول الإشارة; "Edge Cases" → الحالات الحدية; "equalization" → التسوية; "disparity" → تفاوت; "starving" (embeddings) → محرومة من التحديث; "spike resilience" → الصمود أمام القفزات; "graceful degradation" → تدهور سلس; "LR" on slides → معدل التعلم; slide "X replaces A with B" → يستعيض عن A بـ B.
- English glosses given in u01 of this page (do not repeat in later units): preconditioner, root-mean-square (RMS), exponential moving average (EMA), bias correction.

## chapter_optimization/muon.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "Orthogonalization by Newton–Schulz" → المعامدة بتكرار Newton–Schulz; the verb "orthogonalizes" → يعامد; "odd scalar polynomial" → كثير حدود قياسي فردي; "compositions" → تركيبات متتالية, "composed iterations" → تكرارات مركبة; "narrow band" → النطاق الضيق; "inflation / amplified" → تضخيم، تُضخَّم; "steep initial slope" → ميل ابتدائي حاد (the chapter's حاد); "multi-polynomial Polar Express variant" → صيغة Polar Express متعددة كثيرات الحدود (name in Latin).
- The iterations "collapse the spectrum into" a band → تضغط الطيف داخل، انضغاط الطيف (ينهار stays the collapse to rank 1 of u01); "flattened", "spectrum flattening" → انبساط الطيف (the chapter's منبسط); "trailing spectrum / trailing singular value" → ذيل الطيف; "noise damping" → إخماد الضوضاء (the chapter's الإخماد); "pure noise directions" → اتجاهات الضوضاء البحتة.
- "token batch size" → حجم الدفعة بالرموز المميزة; "$…$ FLOPs" → $…$ من عمليات FLOP; "FLOP overhead" → عبء إضافي من عمليات FLOP; "production batch sizes" → أحجام دفعات بيئة الإنتاج; "Gaussian matrix / gradient" → مصفوفة غاوسية / تدرج غاوسي; "tall matrices" → المصفوفات الطويلة; "Gram matrix" → مصفوفة غرام (glossary).
- "RMS alignment" → محاذاة RMS; "entrywise RMS", "entry RMS" → قيمة RMS للعناصر (قيمة RMS لعناصر التحديث / الخطوة); "RMS matching" → مطابقة RMS; "unit singular values" → قيم مفردة تساوي الواحد; "nominal" → القيمة الاسمية; "Nesterov look-ahead gradient" → تدرج نيستيروف الاستشرافي; $\mathrm{NS}_5$ is introduced with ونرمز لها بالرمز; "multiplier", "factor" (the $0.2\sqrt{\max(m,n)}$ scale) → المعامل.
- "decoupled weight decay" → اضمحلال الأوزان المنفصل (as logged, plural الأوزان), glossed once; "shrinkage" (the $(1-\eta\lambda)$ factor) → التقليص (glossary).
- "the framework ships one" → يوفر إطار العمل تنفيذًا جاهزًا; "the library's" → تنفيذ المكتبة; tab notes that begin with an API name get its class noun (يقبل الصنف `torch.optim.Muon`، تأخذ الدالة `optax.contrib.muon`); "setting" → الإعداد; "runs in bfloat16" → بالنوع bfloat16.
- "routing … to Muon" → توجيه … إلى Muon (slide "Routed to" → الموجَّه إلى); "internal block matrices" → مصفوفات الكتل الداخلية; "the Muon+AdamW hybrid" → الهجين Muon+AdamW; "optimizer-state memory" → ذاكرة حالة المحسن; "buffers/param" → مخزن لكل معلَمة.
- "We benchmark A against B" → نقارن أداء … بأداء …; "baseline sweep" → مسح خط الأساس; "learning-rate grid" → شبكة معدلات التعلم; "usable learning-rate window" → نطاق معدلات التعلم الصالحة للاستخدام (as on the test-models page); "validation lead" → تقدم في خسارة التحقق; "checkpoint" here (an evaluation every $250$ steps, nothing is saved) → نقطة تقييم, not the logged نقطة استعادة; "stalls" → يتعثر; "trainable" → قابل للتدريب.
- "step reductions" → تقليص في عدد الخطوات; "heavily tuned" → مضبوط بعناية شديدة; "compute efficiency" → كفاءة حسابية; "scaling experiments" → تجارب على زيادة الحجم; "at larger scale" → عند الأحجام الأكبر; "trillion-parameter Kimi K2" → النموذج Kimi K2 ذو التريليون معلَمة; "attention-logit clipping" → قص لوجيتات الانتباه, glossed once; "QK-Clip" → القص QK-Clip (Latin name with class noun), "bilinear" → ثنائي الخطية; "softmax saturation" → تشبع softmax; "safeguards" → ضمانتان; slide plain-text "0.1B", "1.2B", "1T" → 0.1 مليار، 1.2 مليار، 1 تريليون (inside math the B is copied).
- Slides: list items that begin with a math span get the lead word عند (**عند $k = 1$:**) or a class noun (للمصفوفة $…$); "Code Implementation" → تنفيذ الشيفرة; "Hybrid Architecture" → المعمارية الهجينة; "Training & Validation Trajectories" → مسارات التدريب والتحقق; "Long-Horizon … Caveats" → محاذير … الطويل الأمد; "reach" (of the iteration) → مدى; table header "Concept / Component" → المفهوم / المكوّن.
- English glosses added in u02 of this page: decoupled weight decay, attention-logit clipping.

## chapter_optimization/muon.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- Re-applied u01 for one term: "Gram matrices" → مصفوفتا غرام (the glossary's مصفوفة غرام), not the Latin "Gram" of the muon.md#u01 entry above; Kronecker stays Latin. All other items of that entry stand.

## chapter_optimization/muon.md#u03 (claude fable-5.2 (medium), 2026-11-27)

- "matrix operator" → مؤثر مصفوفي; "steps equally along" → يخطو خطوات متساوية على امتداد; "scales the update by" → يضرب التحديث في; "to match AdamW's entrywise RMS step size" → ليطابق حجم خطوة AdamW بقيمة RMS للعناصر; "1D vectors" → المتجهات أحادية الأبعاد.
- Exercises: "diagonal-entry bound" → حد العناصر القطرية; "uniqueness", "unique" (minimizer) → وحدانية، وحيدة; "Instrument `Muon` … to track" → أضف إلى `Muon` أدوات قياس تتتبع; "polynomial degree" → درجة كثير الحدود; "wall-clock step time" → الزمن الفعلي للخطوة; "validation perplexity degrades" → حيرة التحقق تسوء; "Muon parameter group" → مجموعة معلَمات Muon; `num_iters` $\in \{…\}$ gets the lead noun القيم.
- No English glosses added in u03 of this page (summary and exercises only).

## chapter_optimization/lr-scheduler.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- Page title "Schedules" → جداول معدل التعلم (الجداول alone reads as "tables", as logged for the table label "Schedule"); "constant / cosine / stable–then–decay schedule" → الجدول الثابت / جدول جيب التمام / جدول الثبات ثم الاضمحلال; "warmup–stable–decay (WSD)" → الإحماء ثم الثبات ثم الاضمحلال, then جدول WSD (as on the chapter page); "stable–then–decay (late linear decay)" → الثبات ثم الاضمحلال (الاضمحلال الخطي المتأخر); "late decay" → الاضمحلال المتأخر; "finite-horizon schedule" → جدول ذو أفق منته; "training horizon" → أفق تدريب; "horizon-free" → بلا أفق محدد; "anytime stopping" → التوقف في أي وقت.
- "cumulative learning-rate budget" → الميزانية التراكمية لمعدل التعلم (then الميزانية التراكمية); "terminal learning rate", "terminal rate" → معدل التعلم النهائي / المعدل النهائي (the log's نهائي for "terminal"); "remaining budget" → الميزانية المتبقية; "total budget" → الميزانية الكلية; "(cumulative) travel budget" → ميزانية المسير (التراكمية), as on the chapter page; "step budget $T$" → ميزانية من الخطوات; "budget given up" → الميزانية التي يتنازل عنها; "triangle deficit" → عجز المثلث.
- "decay" of the learning rate → الاضمحلال (verb يضمحل), on this page also for "exponential decay" → الاضمحلال الأسي (glossary; the regression page's تناقص أسي stays for a decaying error) ; "piecewise-constant (step) decay" → الاضمحلال الثابت قطعيًا (الدرجي); "polynomial decay" → الاضمحلال كثير الحدود; "decay factor" → عامل الاضمحلال; "exponent" → الأس; "decay shapes" → أشكال الاضمحلال; "decay profile" → منحنى الاضمحلال (the log's منحنى for "profile"); "parametric schedule families" → عائلات من الجداول المعلَمية; "half-cosine wave" → نصف دورة من جيب التمام (as logged), slide "symmetric half-wave" → نصف موجة متناظر.
- "ramp" (of the rate) → صعود, "linear ramp" → صعود خطي, "warmup ramp" → صعود إحماء, "a $20$-step linear ramp / warmup" → صعود خطي / إحماء خطي من $20$ خطوة, "opening ramp" → صعود البداية (as logged: "ramp" → يرفع، صعود خطي); "plateau" of a schedule (the constant-rate stretch) → هضبة (هضبة ذات معدل ثابت، هضبة المعدل العالي، نقطة استعادة على الهضبة); a curve or run that "plateaus at" → يستقر عند (as logged); "cooldown" → التبريد (التبريد الخطي، فرع تبريد، "uncooled" → غير مبرَّد); "branching", "to fork a branch" → التفريع، تفريع فرع; "LR re-warming" → إعادة إحماء معدل التعلم; "cold start" → بداية باردة (glossary).
- "normalized optimizers" → المحسنات ذات التحديث المطبَّع; "update magnitude" → مقدار التحديث; "transient steps" → الخطوات العابرة; "the iterate settles" → يستقر ناتج التكرار; "stabilizes the opening steps" → يضمن استقرار الخطوات الأولى; "early step instability" → عدم استقرار الخطوات الأولى; "local relaxation window" → نافذة الاسترخاء المحلية; "instantaneous noise ball" → كرة الضوضاء اللحظية; "tracks" → يتتبع ("cleanly" → تتبع سليم); "mean-squared error decomposition" → تحليل متوسط الخطأ التربيعي (glossary's تحليل الانحياز والتباين for "bias–variance decomposition"); "initial bias" → الانحياز الابتدائي; "frozen noise / variance" → الضوضاء المجمدة / تباين مجمد, "freezes" → يتجمد, "freezes noise" → يجمد الضوضاء; "mean-squared distance" → متوسط المسافة التربيعية; "erases" → يمحو; "geometric tail starvation" → حرمان الذيل الهندسي.
- "momentum average" → متوسط الزخم; "second-moment average" → متوسط العزم الثاني; "first-moment buffer" → مخزن العزم الأول المؤقت; "multi-batch statistics" → إحصاءات من دفعات متعددة; "relative weight change" → التغير النسبي في الأوزان; "2D non-embedding weight matrices" → مصفوفات الأوزان ثنائية الأبعاد من غير مصفوفات التضمين; "sharpness ceiling" → سقف الحدة; "high-frequency oscillation" → التذبذب عالي التردد; "step jitter" → اهتزاز الخطوات; "step noise" → ضوضاء الخطوات; "unaveraged sign jump" → قفزة الإشارة بلا توسيط; "aggressive" (peak rate) → شديد; "rescues training" → ينقذ التدريب; "model constructor" → مُنشئ النموذج; "scheduled optimizer wrapper" → غلاف المحسن المجدوَل; "competitive checkpoints" → نقاط استعادة منافسة; "finished models" → نماذج مكتملة; "token budgets" → ميزانيات من الرموز المميزة.
- River valley: "river-valley loss surface" → مشهد الخسارة على هيئة وادي نهر; "river-valley geometry" → هندسة وادي النهر; "steep transverse walls" → الجدران المستعرضة الحادة; "flat longitudinal river bed" → مجرى النهر الطولي المنبسط; "downstream progress" → التقدم مع التيار.
- Slides: "Opening / Terminal hazard" → خطر البداية / خطر النهاية; "Schedule remedy" → العلاج بالجدول; "travel plateau" → هضبة مسير; "AdamW/Muon" → AdamW و Muon; "$O(\eta_0)$ steps / jitter" → من مرتبة $O(\eta_0)$; a title that starts with a number is reordered ("1,000 Valley Runs" → جولات الوادي الـ 1,000; "A 20-Step Warmup …" → إحماء من 20 خطوة …; kicker "Step 1,500 vs. Step 2,000" → الخطوة 1,500 مقابل الخطوة 2,000); "Design Parameter / Engineering Rule of Thumb / Quantitative Justification" → معلَمة التصميم / القاعدة الهندسية التقريبية / المسوغ الكمي; "Landscape Interpretation" → التفسير بمشهد الخسارة; "The Mechanics of X" → آلية عمل X (as logged); slide citations "(Wen et al., 2025)", "(WSD, Hu et al., 2024)", "(Loshchilov & Hutter, 2016)" are copied in Latin; "Wen et al. (2024) explain …" → تفسير Wen et al. (2024) لـ ….
- English glosses given in u01 of this page (do not repeat in later units): learning-rate schedule, warmup, decay, cumulative learning-rate budget, terminal learning rate, warmup–stable–decay (WSD), cosine decay, stable–then–decay.

## chapter_optimization/practice.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- Page title "Practice" → الممارسة العملية; "production training run" → جولة تدريب في بيئة الإنتاج; "at extreme scale" → عند الأحجام القصوى; "operational pillars" → ركائز تشغيلية ("operational" → تشغيلي: العلاج التشغيلي، التعافي التشغيلي); "TPU chips" → رقاقة TPU; "unguarded" → بلا احتياط; "battle-tested defaults" → قيم افتراضية أثبتت جدارتها في الميدان; "untuned" (model) → غير مضبوط.
- "global-norm gradient clipping" → قص التدرج بالمعيار الكلي, "global-norm clipping" → القص بالمعيار الكلي (beside the logged قص التدرج الكلي for "global gradient clipping"); "elementwise clipping" → القص عنصرًا بعنصر; clipping "fires" / steps "trigger clipping" → يُفعَّل القص / تفعّل القص (as the logged يُفعَّل الحصر); "falls silent" → يسكت; "clipping fraction" → نسبة الخطوات المقصوصة; "unclipped" → غير مقصوص، بلا قص; "pre-clip gradient norm" → معيار التدرج قبل القص; "clipping indicator" → مؤشر القص; "scaling factor" → عامل التحجيم; "concatenated gradient across all parameters" → التدرج الناتج عن ربط تدرجات المعلَمات كلها.
- "guard" → احتياط (as logged), "brake", "constant brake" → كابح، كابح دائم (from the logged يكبح); "transient instabilities" → حالات عدم الاستقرار العابرة; "transient blow-up" → الانفجار العابر; "anomalous" (step, gradient, batch) → شاذ; "spike recovery" → التعافي من القفزات ("recovery time" → زمن التعافي); "checkpoint-and-rewind", "rewind to a checkpoint" → الرجوع إلى نقطة استعادة (the logged الرجوع إلى for "restore"); "skip" (minibatches) → تخطي; "the trigger" (of a spike) → ما أثار القفزة; "data-stream cursor" → مؤشر سيل البيانات; "scheduler step counter" → عداد خطوات المجدول; "optimizer moments" → عزما المحسن; "second-moment accumulator / state" → مركم العزم الثاني / حالة العزم الثاني; "paralyzes learning" → يشل التعلم; "auxiliary logit regularizers" → منظِّمات اللوجيتات المساعدة; "$z$-loss" → خسارة $z$.
- "weight averaging" → حساب متوسط الأوزان (as on the chapter index); "weight EMA" → متوسط EMA للأوزان (as logged); "iterate smoothing" → تنعيم نواتج التكرار; "live weights / parameters / trajectory / checkpoint" → الحية (as logged); EMA "decay coefficient $\alpha$" → معامل التناقص (as on the training-recipes page); "trailing window of $n$ steps" → نافذة تضم آخر $n$ خطوة; "autocorrelation time" → زمن الارتباط الذاتي; "initial lag" → التأخر الأولي; "jitter" → اهتزاز (as logged); "band" (of accuracies) → نطاق; "range of live accuracies" → مدى دقة الأوزان الحية.
- "proxy run" → جولة بديلة; "horizon" (planned length of a run) → الأفق; "peak learning rate" → ذروة معدل التعلم; "warmup-and-decay schedule" → جدول الإحماء والاضمحلال; "constant plateau of a WSD schedule" → مرحلة الثبات من جدول WSD (the chapter index's الإحماء ثم الثبات ثم الاضمحلال); "cooldown", "cooldown branch", "cooldown gain" → التبريد، فرع التبريد، كسب التبريد; "cosine-decayed run" → جولة اضمحلال جيب التمام; "fully decayed run" → جولة مكتملة الاضمحلال.
- Slides: "Hypothesis A / B — …" → الفرضية أ / ب، … (Arabic letters; the dash becomes a comma); "Engineering rule" → قاعدة هندسية; "Healthy guard" → احتياط سليم; "fixed-step normalized descent" → نزول مطبَّع ثابت الخطوة; "marginal parameter state" → حالة حدية للمعلَمات; "restoring gradient" → تدرج إرجاع; "Industry Reference Configurations" → إعدادات مرجعية من الصناعة; "frontier pretraining recipes" → وصفات التدريب المسبق لنماذج الصدارة; "batch ramps" → رفع تدريجي لحجم الدفعة; "Diagnostic Playbook" → دليل التشخيص; "failure signatures" → بصمات الإخفاق; "From Telemetry to Action" → من القياسات إلى الإجراء; "step-1 jump" → وثبة في الخطوة 1; "1D norms" → معلَمات التطبيع أحادية الأبعاد; figure colours → بالأزرق، بالبرتقالي، بالأخضر.
- English glosses given in u01 of this page (do not repeat in later units): gradient clipping, exponential moving average (EMA).

## chapter_optimization/adam.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "coupled $\ell_2$ regularization / penalty" → تنظيم $\ell_2$ المقترن / عقوبة $\ell_2$ المقترنة; the code-named configurations "coupled `Adam + l2`", "decoupled `AdamW`" → الإعداد المقترن `Adam + l2`، الإعداد المنفصل `AdamW`; "no decay" → غياب الاضمحلال / بلا اضمحلال; "penalized gradient" → التدرج المعاقَب; "penalty term" → حد العقوبة; "to decay" a parameter (transitive) → يخضع للاضمحلال (إخضاع … للاضمحلال); "exemption" → إعفاء.
- "decay timescale" → المقياس الزمني للاضمحلال; "decay factor" → عامل الاضمحلال; "exponentially weighted moving average" → متوسط متحرك موزون أسيًا; "per-step shrinkage" → التقليص في الخطوة الواحدة; "equilibrium norm" → معيار التوازن, "equilibrium weight RMS" → قيمة RMS للأوزان عند التوازن, "stochastic norm equilibrium" → التوازن العشوائي للمعيار, "deterministic / mixed equilibrium" → التوازن الحتمي / المختلط; "rotational equilibrium" → التوازن الدوراني; "relative angular update rate / angular step / angular learning" → معدل التحديث الزاوي النسبي / الخطوة الزاوية النسبية / التعلم الزاوي; "rotation angle" → زاوية الدوران; "to hold fixed" → يبقي ثابتًا (not يثبّت).
- "entrywise RMS" → قيمة RMS عنصرًا بعنصر; "2D non-embedding weight matrices" → مصفوفات الأوزان ثنائية الأبعاد من غير التضمينات; "internal projection matrices" → مصفوفات الإسقاط الداخلية; "gain and shift vectors" → متجهات الكسب والإزاحة ("LayerNorm gains" → معاملات الكسب في LayerNorm); "normalization scales" → مقاييس التطبيع; "token and positional embedding tables" → جداول التضمين للرموز المميزة وللمواضع; "unigram byte-frequency entropy" → إنتروبيا التواترات الأحادية للبايتات; "unregularized baseline" → خط الأساس غير المنظم.
- "optimizer state memory" → ذاكرة حالة المحسن; "static memory / buffers" → الذاكرة الثابتة / المخازن الثابتة; "device memory" → ذاكرة الجهاز; "factoring" (the moments) → التحليل إلى عوامل; "linear operator" → مؤثر خطي; "eject" (an iterate) → يُقذف بعيدًا، القذف; "repairs" (of the second moment) → إصلاحات; "parametrization by $\lambda$" (a search grid) → التعبير بدلالة $\lambda$; "the gain" (an improvement) → التحسن (الكسب stays the LayerNorm gain).
- Slides: a kicker that is only a citation takes a class noun: "[Loshchilov & Hutter (2019)]" → [بحث Loshchilov و Hutter (2019)]; "Hyperparameter decoupling" → فصل المعلَمات الفائقة; "VRAM accounting" → حساب ذاكرة VRAM; table cells "$1$ buffer", "$2$ buffers" → $1$ مخزن حالة، $2$ مخزن حالة (digit in math kept, singular noun, style §6); "/param" → لكل معلَمة; table row labels take the class noun outside the bold (المحسن **AdamW**).
- English glosses given in u02 of this page: decoupled weight decay, equilibrium norm, rotational equilibrium.

## chapter_optimization/lr-scheduler.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- Re-applied u01 to match chapter_optimization/adam.md#u01, which was logged while this unit was in work; this entry replaces the corresponding items of the lr-scheduler.md#u01 entry above. Optimizer names take the class noun محسن where one is needed (في محسن Adam، بالمحسن AdamW; not خوارزمية, which stays for SGD); "unaveraged step-1 sign jump" → وثبة الإشارة في الخطوة 1 بلا توسيط, the cold start "jumps by" → تثب بمقدار (وثبة for "jump", قفزة stays "spike"). All other items of that entry stand.

## chapter_optimization/batch-size.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- Page title "Batch Size" → حجم الدفعة; "minibatch size" → حجم الدفعة المصغرة. "regime" on this page → نطاق, as on the sgd page: "small-batch / large-batch regime" → نطاق الدفعات الصغيرة / الكبيرة; "noise-dominated / curvature-dominated regime" → النطاق الذي تهيمن عليه الضوضاء / يهيمن عليه الانحناء (slides: الضوضاء مهيمنة، الانحناء مهيمن); "step-saturated" → يتشبع عدد الخطوات.
- "gradient-noise scale" → مقياس ضوضاء التدرج (parallel to the logged مقياس الإشارة إلى الضوضاء), "noise scale" → مقياس الضوضاء, "simple gradient-noise scale" → مقياس ضوضاء التدرج البسيط, "curvature-weighted noise scale" → مقياس الضوضاء الموزون بالانحناء; "two-batch estimator" → مقدر بدفعتين / مقدر الدفعتين, "two-batch estimate" → تقدير الدفعتين; "signal power / noise power" → قدرة الإشارة / قدرة الضوضاء; "mean squared gradient norm" → متوسط مربع معيار التدرج; "expected squared norm" → التوقع لمربع المعيار; "full pass over the dataset" → مرور كامل على مجموعة البيانات; "inter-sequence variance / disagreement" → التباين / الاختلاف بين التسلسلات; "true gradient" → التدرج الحقيقي.
- "scaling" on this page: the rule that sets $\eta$ from $b$ → تحجيم معدل التعلم ("linear scaling" → التحجيم الخطي, "square-root scaling" → التحجيم بالجذر التربيعي, "scaling rules" → قواعد التحجيم, "square-root rule / linear rule" → قاعدة الجذر التربيعي / القاعدة الخطية, "scaled-up learning rate" → معدل التعلم المكبَّر, "unscaled" → بلا تحجيم); how step counts follow $b$ → التناسب ("perfect linear scaling" → التناسب الخطي المثالي, "perfect-scaling line" → خط التناسب المثالي, "step-to-target scaling" → تناسب عدد الخطوات حتى الهدف, "scale linearly with $b$" (time) → يتزايد خطيًا مع $b$); "scale training across multiple GPUs" → يوسع التدريب ليشمل عدة وحدات GPU; "increase the batch size" → يكبّر حجم الدفعة.
- "trade-off frontier" → حد المقايضة; "efficiency frontier" → حد الكفاءة; "Pareto frontier / curve" → حد Pareto / منحنى Pareto (name in Latin; frontier → حد as logged); "knee" → الركبة, glossed (knee); "universal curve / law / hyperbola" → الشامل; "upper-left / lower-right arm" (of the curve) → الذراع العلوية اليسرى / السفلية اليمنى (طرف stays the arm of a comparison); "collapse onto the curve" → الانطباق على المنحنى، تطبق … على; "eliminating $x$ between" → حذف $x$ بين.
- "step count" → عدد الخطوات, "example count" → عدد الأمثلة, "steps to target" → عدد الخطوات حتى الهدف, "step-to-target sweep" → مسح عدد الخطوات حتى الهدف; "data budget" → ميزانية البيانات; "data bill" → فاتورة البيانات; "data and compute cost" → كلفة البيانات والحوسبة; "compute FLOPs" → عمليات FLOP الحسابية; "asymptotic step floor" → أرضية الخطوات التقاربية; "both asymptotes" → الخطان المقاربان; "weighted least squares" → المربعات الصغرى الموزونة; "weighted linear regression" → الانحدار الخطي الموزون; "fitted" (values, batch sizes) → الملائَم, "the fit" → الملاءمة; "normalize by" (divide by a reference value) → يطبّع بالقسمة على; "theoretical minimum" → القيمة الدنيا النظرية; "small-batch limit" → الحالة الحدية للدفعات الصغيرة.
- "$L$-smooth descent bound" → حد النزول للدوال الملساء بثابت $L$, "noisy descent bound" → حد النزول مع الضوضاء; "scalar smoothness bound" → حد الملاسة القياسي (الملاسة as logged on the gd page, not the glossary's proposed نعومة); "guaranteed decrease" → النقصان المضمون, slide "single-step drop" → الهبوط في الخطوة الواحدة; "Hessian-free proxy / predictor" → بديل / متنبئ لا يحتاج إلى المصفوفة الهيسية; "single-step drift" → انجراف في الخطوة الواحدة; "step noise" → ضوضاء الخطوة; "RMS scale" → مقياس RMS; "deterministic curvature limit" → حد الانحناء الحتمي; "unnormalized optimizers" → المحسنات التي لا تطبّع التدرج; "adaptive optimizers" → المحسنات التكيفية.
- "random minibatch samplers" (functions) → دالتا سحب الدفعات المصغرة العشوائية; "stopping harness" → أداة الإيقاف; "held-out loss target" → هدف لخسارة البيانات المحجوزة; "in chunks of" → في قطع من; "shared initialization" → تهيئة مشتركة; "terminal losses" → الخسائر النهائية; "the sweep below" → المسح الوارد لاحقًا; "coarse unigram adjustment" → التعديل الخشن في التوزيع الأحادي للبايتات; "global byte-frequency shift" → الإزاحة العامة في ترددات البايتات.
- Procedure list (body text): masculine singular imperatives, also in the bold lead-ins (أشبع، قدّر، حدد، حجّم); on slides the same steps are verbal nouns (ضبط، قياس، اختيار، تحجيم). "saturate hardware arithmetic" → يشبع القدرة الحسابية للعتاد; "batch-size warmup ramps" → إحماء حجم الدفعة برفعه تدريجيًا, slide "batch size ramp", "ramp $b$" → الرفع التدريجي لحجم الدفعة، يرفع $b$ تدريجيًا; "wall-clock bound" → مقيد بالزمن الفعلي (glossary مقيد بالحوسبة); "frontier runs" → جولات نماذج الصدارة; "Engineering Playbook" → الدليل الهندسي العملي; "oversized" (batch) → مفرطة الحجم; "under-budgets the run" → يجعل ميزانية المسير غير كافية.
- Slides: an approximate count after a math span → من التسلسلات / من الصور ($b_{\text{noise}} \approx 2$ من التسلسلات); bold sub-bullets that begin with a math span get the lead word عند (as logged); "$k$ small steps to $1$ large step" keeps the digit: $k$ خطوة صغيرة مع $1$ خطوة كبيرة; an answer item that begins with a formula gets a class noun (أثر مصفوفة التغاير $\operatorname{tr}\boldsymbol{\Sigma} = …$); two math spans separated only by a bracket get حيث (… $\eta_0$ (حيث $b \ll b_{\text{crit}}$)); "grid" of learning rates → شبكة القيم, and the network beside it → الشبكة العصبية.
- English glosses given in u01 of this page (do not repeat in later units): perfect linear scaling, critical batch size, gradient-noise scale, two-batch estimator, signal power, noise power, knee.

## chapter_optimization/scaling.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- Page title "Scaling Up" → التحجيم إلى الأكبر (as on the chapter index); "scale" (small, large, target) → الحجم (الحجم المستهدف); a quantity that "scales with $n$" → يتناسب مع / يتغير حجمه مع, "to scale a learning rate / a layer" → يحجّم, "scaling rule" → قاعدة تحجيم, "scaling classes" → فئات تحجيم.
- "standard parametrization (SP)" → تحديد المعلَمات المعياري (standard parametrization, SP), then SP after a free preposition (في SP، مع SP) or after a class noun (مسح SP، انجراف SP); "Maximal Update Parametrization" → تحديد المعلَمات بالتحديث الأعظم, then $\mu\text{P}$ (masculine: يُبقي $\mu\text{P}$); "width parameterization" (slides) → تحديد المعلَمات حسب العرض; "Transformer parameterization" → تحديد معلَمات المحول.
- "proxy model", "proxy" → نموذج بديل (النموذج البديل الضيق), "proxy networks" → الشبكات البديلة; "target" (model, scale) → المستهدف / الهدف; "zero-shot learning-rate transfer" → نقل معدل التعلم بلا إعادة ضبط (as on the chapter index); "retuned" → بعد إعادة الضبط; "tuned on / at" → المضبوط على / عند.
- "fan-in" / "fan-out" → التفرع الداخل / التفرع الخارج (as logged), glossed once on this page; inside math `\text{fan-in}` is copied; "fixed-fan-in" → ذو التفرع الداخل الثابت; "fan-in law" → قانون التفرع الداخل.
- "coordinate check" → فحص الإحداثيات (coordinate check), "one-step coordinate check" → فحص الإحداثيات بخطوة واحدة / ذو الخطوة الواحدة; "broken" (implementation, check) → معطوب; curves "convict" an implementation → تدين.
- "learning-rate drift" → انجراف معدل التعلم (glossary انجراف), "drifts downward" → ينجرف نزولًا; "widen" (a model, a network) → يزداد عرضه, "wider" → أعرض, "narrow" → ضيق, "widening $16\times$" → زيادة العرض $16\times$ (التوسيع stays dilation).
- "output multiplier" → العامل الضربي للمخرجات, "forward multiplier" → العامل الضربي في التمرير الأمامي (slides: العامل الضربي الأمامي), "readout multipliers" → العوامل الضربية للقراءة; "width multiplier" $m$ → مضاعف العرض (as logged), glossed once; "base width" → العرض الأساسي (base width); "divisor" → القاسم.
- "output readout" (the last layer `fc_out`) → طبقة قراءة المخرجات, in table and slide labels قراءة المخرجات, "readout matrix" → مصفوفة القراءة / مصفوفة قراءة المخرجات, "readout `head`" → رأس القراءة (as logged); "input weights" → أوزان المدخلات; "1D biases/norms" → الانحيازات ومعلَمات التطبيع أحادية الأبعاد; "initial variance", "init variance" → التباين الابتدائي; "unscaled initialization variance" → تباين التهيئة قبل التحجيم; "effective output weights" → أوزان المخرجات الفعالة; "entry magnitude" → مقدار العناصر.
- Sums of terms: "align in sign with" → تتوافق في الإشارة مع, "aligned signs / sign-aligned terms" → إشارات متوافقة / حدود متوافقة الإشارة; "add coherently", "coherent sum" → تُجمع جمعًا متسقًا، المجموع المتسق (the log's متسق for "coherent"), "incoherent" → غير متسق; "cancel" → يلغي بعضها بعضًا; "sums in quadrature" → يُجمع جمعًا تربيعيًا; "correlated terms" → حدود مترابطة.
- Optima that "align" across widths → تتطابق, curves that "align vertically" → تصطف عموديًا (kicker: قيم مثلى مصطفة); "flat" (a coordinate-check curve, an activation scale) → منبسط, as the chapter's "flat"; "activation blow-up" → انفجار التفعيلات, "exploding updates" → التحديثات المنفجرة; "over-shrunk learning rate" → معدل التعلم المفرط التقليص; "invariant to width / across widths" → ثابت أمام العرض / ثابت عبر العروض.
- "log–log slope" → الميل على المحورين اللوغاريتميين (from the logged محوران لوغاريتميان); "log-spaced learning rates" → معدلات تعلم موزعة لوغاريتميًا; "a $64\times$ range of widths" → مدى قدره $64\times$ من العروض; "U-shaped loss curves" → منحنيات الخسارة التي على شكل U; "right arm / right wall" of a curve → الذراع اليمنى / الجدار الأيمن; "seed-averaged" → المتوسطة على البذور; "seed noise" → ضوضاء البذور; "capped at $2.5$" → بسقف مقداره $2.5$; "diverged runs" → الجولات المتباعدة.
- "suboptimality" (in nats) → ابتعاد عن القيمة المثلى; "penalty" (loss paid for a transferred rate) → عقوبة (عقوبة النقل), on slides "loses" → الفاقد; "compromise" → حل وسط; "anchor" (shared reference point of two sweeps) → مرتكز (the log's مرتكزات), "Reference Anchor" → المرتكز المرجعي.
- Slides: "Layer Taxonomy" → أصناف الطبقات; "Quantitative Scaling Breakdown" → تفصيل كمي للتحجيم; "Regime" (table header) → النمط; "feature refinement" → تنقيح الميزات; "overshoots" → تخطي; "Diagnostic asymmetry" → لاتناظر تشخيصي; "Universal spectral condition" → الشرط الطيفي الشامل; "Operator-Norm Unification" → التوحيد بمعيار المؤثر; "spectral optimizers" → المحسنات الطيفية, "spectral-norm optimizers" → محسنات المعيار الطيفي; "Spectral-scaled / AdamW-scaled / operator-scaled Muon" → Muon بالتحجيم الطيفي / بتحجيم AdamW / بتحجيم معيار المؤثر; "135K to 141M parameters" → من 135 ألف إلى 141 مليون معلَمة; slide citations take وفق / لدى before the Latin names where a math span precedes them ((وفق Yang et al., 2022)); bold table-row labels that are a bare "$n = …$" get the class noun العرض; a code name or math span in a bold lead-in is joined to its label by أي / وهي / في instead of a bare bracket or comma.
- English glosses given in u01 of this page (do not repeat in later units): proxy model, standard parametrization (SP), Maximal Update Parametrization, fan-in, fan-out, coordinate check, base width, width multiplier. Pre-activations, outer product and logits take no gloss here (defined on earlier pages).

## chapter_optimization/legacy-timer-lib.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- Titles of the build-only `*-lib.md` pages: "Legacy … Library" → مكتبة … القديمة; "(build-only, unlisted)" → (للبناء فقط، غير مدرجة); "Timer" → المؤقت (as logged). No gloss (heading).

## chapter_optimization/batch-size.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- Re-applied u01 to match chapter_optimization/adam.md#u01, sgd.md#u02 and lr-scheduler.md#u01, which were logged while this unit was in work; this entry replaces the corresponding items of the batch-size.md#u01 entry above. Adam and AdamW take the class noun محسن (محسن Adam، محسنا Adam و AdamW، لمحسن AdamW; خوارزمية stays for SGD); "square-root scaling" → تحجيم الجذر التربيعي (not التحجيم بالجذر التربيعي); "unnormalized optimizers" → المحسنات ذات التحديث غير المطبَّع (from the logged المحسنات ذات التحديث المطبَّع); "grid steps" → خطوات شبكة القيم; "coarse unigram adjustment" → التعديل الخشن في التواترات الأحادية للبايتات, "global byte-frequency shift" → الإزاحة العامة في تواترات البايتات (the adam page's تواترات). All other items of that entry stand.

## chapter_optimization/lr-scheduler.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- Heading "Warmup–Stable–Decay (WSD) and Checkpoint Branching" → جدول الإحماء ثم الثبات ثم الاضمحلال (WSD) وتفريع نقاط الاستعادة; "Branching Off the Plateau" → التفريع من الهضبة; "to branch" → يفرّع (نفرّع، with the shadda); "branched cooldown" → تبريد متفرع; "the continuing plateau" → الهضبة المستمرة; "a single continuous plateau run" → جولة هضبة واحدة متصلة; "a dedicated run planned from scratch" → جولة خاصة خُطط لها من الصفر; "total step horizon" → أفق الخطوات الكلي; "target horizon" → الأفق المستهدف; "fixed-horizon runs" → الجولات ثابتة الأفق; "full-horizon linear decay" → الاضمحلال الخطي على الأفق كله; "splicing schedules" → وصل جدول بآخر (the log's يصل for "splice").
- "clones" (weights, optimizer state) → يستنسخ (as logged); "AdamW's moment buffers $(\mathbf{m}_t, \mathbf{v}_t)$" → مخزنا العزمين في AdamW; "optimizer moments" → عزوم المحسن; "freshly initialized" → مهيأ من جديد; "oversized initial Adam jumps" → الوثبات الأولية مفرطة الحجم في Adam (the adam page's وثبة); "sign jumps" → وثبات إشارة; "steers the trajectory below" → يوجه المسار إلى ما دون; "how far the iterate travels" → المسافة التي يقطعها ناتج التكرار; "late-stage noise damping" → إخماد الضوضاء في المرحلة المتأخرة; "compute savings / compute saved" → ما يوفره … من الحوسبة / توفير … من الحوسبة.
- Exercises and slides: "integrate" (a sum, imperative) → كامِل (with the kasra, against كامل "complete"); "asymptotic frozen variance" → التباين المجمد التقاربي; "a $3$-stage schedule" → جدول من $3$ مراحل; "terminal distance" → المسافة النهائية; "minimum stable warmup" → أقصر إحماء مستقر; "Adam's first-moment window" → نافذة العزم الأول في Adam; "maximum trainable / tolerable peak rate" → أعلى ذروة معدل قابلة للتدريب / يمكن تحملها; "Solve each … question quantitatively:" → المطلوب: حل كل سؤال … حلًا كميًا: (the logged المطلوب lead-in); "You want …" on a slide → المطلوب …; "Bias–Noise Decomposition" → تحليل الانحياز والضوضاء; "Schedule Concept" → مفهوم الجدول; "Mathematical Form / Invariant" → الصيغة الرياضية / اللامتغير; "Budget & Terminal Behavior" → الميزانية والسلوك النهائي; "erases start" → تمحو أثر البداية; "Standard for" → الخيار المعياري لـ; "linear/cosine cooldown" → تبريد خطي أو بجيب التمام; "Multi-Horizon Models" → نماذج لآفاق متعددة.
- English glosses given in u02 of this page: river valley, cooldown.

## chapter_optimization/batch-size.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "hyperbolic trade-off" → مقايضة القطع الزائد; "data-efficient" (regime) → الكفء في البيانات; "step-saturated regime" → النطاق الذي يتشبع فيه عدد الخطوات; "under tuned learning rates" → عند ضبط معدلات التعلم; the optimal rate "levels off" → يستقر; "scales linearly / as the square root" (the optimal rate with $b$) → يتناسب خطيًا مع حجم الدفعة / مع جذره التربيعي.
- Exercises: "time-optimal vs. compute-optimal batch size" → حجم الدفعة الأمثل زمنيًا مقابل الأمثل حسابيًا (glossary أمثل حسابيًا); "data overhead" → عبء إضافي في البيانات; "improves by factor" → يتحسن بالمعامل; "never more than doubling" → لا يزيد على الضعف أبدًا; "clamping $\eta$ at its $b = 256$ value" → إبقاء $\eta$ عند قيمته في $b = 256$ (not يثبّت); "scaling past the critical batch size" → التحجيم بعد حجم الدفعة الحرج; "degrades convergence" → يضر بالتقارب.
- No English glosses added in u02 of this page (summary and exercises only).

## chapter_optimization/scaling.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- "Extensions to Transformers and Spectral Optimizers" → التعميم على المحولات والمحسنات الطيفية ("extending" a rule → تعميم … على); "spectral scaling condition" → شرط التحجيم الطيفي, "spectral norm condition" → شرط المعيار الطيفي; "operator-norm principle" → مبدأ لمعيار المؤثر; "unit spectral norm by construction" → معيار طيفي يساوي الواحد بحكم بنائه; "the AdamW-matched factor" → المعامل المطابق لخوارزمية AdamW (المعامل as on the Muon page).
- "residual stream" → تيار البواقي (glossary; replaces the log's مجرى الإشارة on this page); "residual branch" → فرع البواقي (فروع البواقي), "branch outputs" → مخرجات الفروع; "$L$ residual blocks" → $L$ كتلة بواقٍ; "model dimension $d$" → بُعد النموذج; "head dimension" → بُعد الرأس (glossary); "attention logits" → لوجيتات الانتباه; "attention dot-product scores" → درجات الانتباه بالجداء النقطي; "lookup" (of an embedding row) → استخراج من جدول; "sublayer" → طبقة فرعية.
- "scale-up", "full scale-ups" → التحجيم إلى الأكبر / التحجيم الكامل إلى الأكبر (the page title); "finite size" → الحجم المنتهي, "limit of infinite size" → حد الحجم اللانهائي; rules that are "exact" → مضبوطة (as logged); "grows $m$-fold" → يكبر $m$ مرة; "hold $\tau$ fixed" → إبقاء $\tau$ ثابتًا (as logged, not يثبّت); "asymptotic anchor" → مرتكز تقاربي; sweeps "centered" around it → تتمركز حوله, "the center of a narrow sweep" → مركز لمسح ضيق; "two exponents" → أسّان.
- Exercises: "deliberate bugs" → مواطن خلل متعمدة; "over-shrinking updates" → التقليص المفرط للتحديثات; "multi-step loss sweep" → مسح للخسارة متعدد الخطوات; "input-to-hidden learning-rate ratio" → نسبة معدل تعلم طبقة الإدخال إلى معدل تعلم الطبقات المخفية; "initial logit scale" → مقياس اللوجيتات الابتدائي; "accounts for" → تفسر; `d_model` $\in \{…\}$ gets the lead noun قيم.
- No English glosses added in u02 of this page (fan-in, fan-out, coordinate check, base width, width multiplier were glossed in u01).

## chapter_optimization/further-reading.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- Page title "Further Reading" → قراءات إضافية (as the front matter's المصادر وقراءات إضافية); "survey papers", "surveys" → الأوراق المسحية, "X surveys …" → يقدم X مسحًا لـ (the logged مسح); "playbooks" → الأدلة العملية (the index's أدلة عملية للممارسين); "interactive visualizations" → تمثيلات بصرية تفاعلية, heading "Visualizing" → التمثيل البصري; "standard reference" → المرجع المعياري; "comprehensive treatment" → معالجة شاملة.
- Reading-list items: a :citet: followed by its English title in a parenthesis is the subject itself (يُعد :citet:`…` (*Title*) المرجع …); a named author with a linked title takes a class noun and المعنون/المعنونة (مقالة Gabriel Goh المعنونة *[Title](url)*، دليل EleutherAI المعنون *[…](…)*); titles stay in English; "(Distill, 2017)" is copied; "A and :citet:" → … وكذلك :citet:`…`; an acronym followed by a citation in one parenthesis gets وقد اقترحه/اقترحها (**L-BFGS**، وقد اقترحها :citet:`…`), as on the muon page.
- "step-size decay" → اضمحلال حجم الخطوة; "stochastic gradient noise balls" → كرات الضوضاء في التدرج العشوائي; "coordinate checks" ($\mu\text{P}$) → فحوص الإحداثيات (فحص الإحداثيات), no gloss here (the scaling page defines it); "scientific, nuisance, and fixed hyperparameters" → المعلَمات الفائقة العلمية والدخيلة والثابتة.
- "mode connectivity" → ترابط القيم الصغرى, glossed (mode connectivity): the "modes" are minima, and نمط is the Hessian eigenmode in this chapter; "cross-sections" (of a loss surface) → مقاطع; "filter-by-filter" → مرشحًا بمرشح, "filter-normalized slices" → شرائح مطبَّعة على مستوى المرشحات; "positive-homogeneity scale invariances" → حالات الثبات أمام التحجيم الناتجة عن التجانس الإيجابي; "jagged" → وعر; "linear interpolation" (between two networks) → الاستيفاء الخطي (glossary استيفاء); "high-loss barrier" → حاجز عالي الخسارة; "aligning hidden units" → محاذاة الوحدات المخفية; "wide networks" → الشبكات العريضة.
- "Sharpness-Aware Minimization (SAM)" → التصغير المراعي للحدة (Sharpness-Aware Minimization, SAM), then SAM; "flat minima" → القيم الصغرى المنبسطة; "generalize better" → تعمم أفضل; "gradient-ascent perturbation step" → خطوة اضطراب في اتجاه الصعود التدرجي; "descent gradient" → تدرج النزول; "worst-case loss" → الخسارة في أسوأ الحالات; "raw Hessian sharpness" → حدة المصفوفة الهيسية الخام; "not invariant to layerwise reparametrization" → ليست ثابتة أمام إعادة تحديد المعلَمات طبقة بطبقة ("layerwise" → طبقة بطبقة / الخاصة بكل طبقة).
- "second-order methods" → طرائق المرتبة الثانية (as logged); "variance-reduced methods" → طرائق خفض التباين; "quasi-Newton methods" → طرائق شبه نيوتن; "line search" → البحث الخطي; "trust-region methods" → طرائق منطقة الثقة (glossary terms, each glossed once here); "inverse Hessian" → معكوس المصفوفة الهيسية; "secant pairs" → أزواج القاطع; "Limited-memory BFGS" → طريقة BFGS محدودة الذاكرة; "method of choice" → الطريقة المفضلة; "Armijo sufficient-decrease checks" → فحوص النقصان الكافي بشرط Armijo (name in Latin); "deterministic smooth problems" → المسائل الملساء الحتمية.
- "Hessian-free optimization" → التحسين الخالي من المصفوفة الهيسية, glossed; "Saddle-Free Newton" → the Latin name with طريقة in body text, heading: طرائق نيوتن … الخالية من نقاط السرج; "conjugate gradients" → طريقة التدرج المترافق (glossary); Gauss–Newton and Krylov, Polyak–Ruppert stay Latin (compound names, as Newton–Schulz): مصفوفة Gauss–Newton، فضاء Krylov الجزئي; "negative-curvature directions" → اتجاهات الانحناء السالب; "repel / attract iterates" → تنفّر / تجذب نواتج التكرار.
- "minimax-optimal rate" → المعدل الأمثل بمعنى أدنى الأعلى (minimax) (from the logged لعبة أدنى أعلى); "iterate averaging" → حساب متوسط نواتج التكرار; "optimal asymptotic covariance" → التغاير التقاربي الأمثل; "finite datasets" → مجموعات البيانات المنتهية; "full-dataset control variates" → متغيرات ضبط تُحسب على مجموعة البيانات كاملة; "rapidly shifting features" → الميزات سريعة التغير.
- "structural optimizers" → المحسنات البنيوية; "natural gradient" → التدرج الطبيعي (glossary), glossed; "Kronecker products" → جداءات كرونيكر (the glossary term, name in Arabic letters by style §7); K-FAC, SOAP, Scion, Dion take the class noun المحسن, Polar Express and SVRG take طريقة; "layerwise Fisher blocks" → كتل فيشر الخاصة بكل طبقة; "output-distribution space" → فضاء توزيعات المخرجات; "associative-memory projections" → إسقاطات الذاكرة الترابطية; "tail features" → ميزات الذيل; "rare tokens" → الرموز المميزة النادرة; "data-efficient" → كفؤ في استخدام البيانات; "falling back to" → الرجوع إلى (as logged); "inexact $q$-step orthogonalization" → معامدة غير مضبوطة من $q$ خطوة; "low-rank" → منخفض الرتبة; "cross-device communication" → الاتصال بين الأجهزة; "sharded training" → التدريب المقسم إلى حصص.
- "cosine annealing" → اضمحلال جيب التمام (the glossary's cosine decay; the same schedule); "warm restarts" → إعادات تشغيل دافئة, glossed; "snapshot ensembles" → مجموعات نماذج اللقطات, glossed; "model soups" → حساء النماذج, glossed; "Schedule-Free" → الخالي من الجدول (صيغتا SGD و AdamW الخاليتان من الجدول (Schedule-Free)); "predetermined horizon $T$" → الأفق $T$ المحدد سلفًا; "live iterate" → ناتج التكرار الجاري; "running iterate average" → المتوسط المتحرك لنواتج التكرار.
- "batch-size ramps" → الزيادة التدريجية لحجم الدفعة; "scaling rules" → قواعد التحجيم (glossary قانون التحجيم for "scaling laws"); "SDE scaling rules" → قواعد تحجيم مبنية على المعادلات التفاضلية العشوائية (SDE); "tokens processed" → عدد الرموز المميزة المعالَجة; "hyperparameter transfer across scales" → نقل المعلَمات الفائقة عبر الأحجام; "production language models" → النماذج اللغوية المستخدمة في بيئة الإنتاج.
- "apparent gains" → المكاسب الظاهرة; "standardized benchmarks" → الاختبارات المعيارية الموحدة (موحد معياريًا stays statistical standardization); "benchmarking suites" → مجموعات الاختبارات المعيارية; "optimization track" → مسار التحسين; "tuning budgets" → ميزانيات ضبط; "training logbook" → سجل تدريب; "unfiltered day-by-day record" → تدوين يومي غير منقح; "mid-run" → في أثناء الجولة; "checkpoint rewinds" → الرجوع إلى نقاط استعادة سابقة; "hardware recovery" → التعافي من أعطال العتاد.
- English glosses given on this page: mode connectivity, Sharpness-Aware Minimization (SAM), quasi-Newton methods, line search, trust-region methods, Hessian-free optimization, minimax, natural gradient, warm restarts, snapshot ensembles, model soups, Schedule-Free.

## chapter_optimization/practice.md#u02 (claude fable-5.2 (medium), 2026-11-27)

- Heading "A Worked End-to-End Configuration" → إعداد محلول من طرف إلى طرف ("the worked configuration" → الإعداد المحلول, from the logged مثال محلول); "to configure" (a run, a model, an optimizer) → يعدّ، إعداد; "configured schedule / run / recipe" → الجدول الذي أعددناه، الجولة المعدّة، الوصفة المعدّة; "trial run" → جولة تجريبية ("proxy trials" → جولات بديلة تجريبية); "calibrated peak rate" → ذروة المعدل المعايَرة.
- Tables: "Logged symptom" → العرَض المسجل (the logged عرَض; أعراض); "Diagnosis / Primary remedy" → التشخيص / العلاج الأساسي; "diagnostic lookup table" → جدول بحث تشخيصي; header "Peak LR & schedule" → ذروة معدل التعلم وجدوله, "Clip $\theta$" → عتبة القص $\theta$; a model name in the first cell takes the class noun النموذج; plain-text "8k steps", "4M tokens", "67M tokens" → 8 آلاف خطوة، 4 ملايين … رمز مميز، 67 مليون رمز مميز (as the logged 0.1B → 0.1 مليار), "2k steps" → 2 ألف خطوة (digit kept in a table cell); "seqs" → تسلسل; "(no emb)" → (دون التضمينات); "ramp a $\to$ b" → صعود من a $\to$ b; "WSD cosine to" → جدول WSD بجيب التمام إلى.
- "gradient-noise scale", "noise scale" → مقياس ضوضاء التدرج، مقياس الضوضاء; "hardware throughput saturates" → تتشبع إنتاجية العتاد, "hardware-saturated batch sizes" → أحجام دفعات تشبع العتاد; "critical batch size knee" → ركبة المنحنى عند حجم الدفعة الحرج; "clipping rate" (returned by `train`) → معدل القص; "second-moment buffer" → مخزن العزم الثاني; "continuous firing" → تفعيله المستمر; "seed spread", "inter-seed spread" → تشتت البذور، التشتت بين البذور; "short-horizon" → على أفق قصير; "loss plateaus" in a table → تستوي الخسارة; "step-to-step variance" → تباين من خطوة إلى أخرى.
- Exercises: "Clipping regimes" → نمطا القص; "uniform-shift invariance" → الثبات تحت الإزاحة المنتظمة (the logged الثبات تحت الإزاحة); "auxiliary penalty" → العقوبة المساعدة; "effective window … in epochs" → النافذة الفعلية … مقيسةً بالحقب; "Ablating the worked configuration" → استئصال مكونات الإعداد المحلول, "two-seed ablations" → دراسات استئصال ببذرتين; sub-items (a) … (d) keep the Latin letters and are joined by و.
- Slides: "Case Study" → دراسة حالة; "Putting All 10 Sections Together" → جمع الأقسام الـ 10 كلها; "Diagnose and solve each … scenario:" → المطلوب: تشخيص كل حالة … وحلها:; "You must **lower** …" → الواجب هو **خفض** …; "weight EMA shines" → يتألق; "uncooled … WSD-plateau checkpoints" → نقاط استعادة غير مبرَّدة … على هضبة WSD (the scheduler page's هضبة and غير مبرَّد); table headers "Operational Pillar / Mathematical Rule / Mechanism / Healthy Telemetry Target / Production Role" → الركيزة التشغيلية / القاعدة الرياضية / الآلية / الهدف السليم في القياسات / الدور في بيئة الإنتاج; "Modern Default Recipe" → الوصفة الافتراضية الحديثة.
- No English glosses added in u02 of this page.

## chapter_optimization/practice.md#u01 (claude fable-5.2 (medium), 2026-11-27)

- Page title "Practice" → الممارسة العملية; "production training run" → جولة تدريب في بيئة الإنتاج; "at extreme scale" → عند الأحجام القصوى; "operational pillars" → ركائز تشغيلية ("operational" → تشغيلي: العلاج التشغيلي، التعافي التشغيلي); "TPU chips" → رقاقة TPU; "unguarded" → بلا احتياط; "battle-tested defaults" → قيم افتراضية أثبتت جدارتها في الميدان; "untuned" (model) → غير مضبوط.
- "global-norm gradient clipping" → قص التدرج بالمعيار الكلي, "global-norm clipping" → القص بالمعيار الكلي (beside the logged قص التدرج الكلي for "global gradient clipping"); "elementwise clipping" → القص عنصرًا بعنصر; clipping "fires" / steps "trigger clipping" → يُفعَّل القص / تفعّل القص (as the logged يُفعَّل الحصر); "falls silent" → يسكت; "clipping fraction" → نسبة الخطوات المقصوصة; "unclipped" → غير مقصوص، بلا قص; "pre-clip gradient norm" → معيار التدرج قبل القص; "clipping indicator" → مؤشر القص; "scaling factor" → عامل التحجيم; "concatenated gradient across all parameters" → التدرج الناتج عن ربط تدرجات المعلَمات كلها.
- "guard" → احتياط (as logged), "brake", "constant brake" → كابح، كابح دائم (from the logged يكبح); "transient instabilities" → حالات عدم الاستقرار العابرة; "transient blow-up" → الانفجار العابر; "anomalous" (step, gradient, batch) → شاذ; "spike recovery" → التعافي من القفزات ("recovery time" → زمن التعافي); "checkpoint-and-rewind", "rewind to a checkpoint" → الرجوع إلى نقطة استعادة (the logged الرجوع إلى for "restore"); "skip" (minibatches) → تخطي; "the trigger" (of a spike) → ما أثار القفزة; "data-stream cursor" → مؤشر سيل البيانات; "scheduler step counter" → عداد خطوات المجدول; "optimizer moments" → عزما المحسن; "second-moment accumulator / state" → مركم العزم الثاني / حالة العزم الثاني; "paralyzes learning" → يشل التعلم; "auxiliary logit regularizers" → منظِّمات اللوجيتات المساعدة; "$z$-loss" → خسارة $z$.
- "weight averaging" → حساب متوسط الأوزان (as on the chapter index); "weight EMA" → متوسط EMA للأوزان (as logged); "iterate smoothing" → تنعيم نواتج التكرار; "live weights / parameters / trajectory / checkpoint" → الحية (as logged); EMA "decay coefficient $\alpha$" → معامل التناقص (as on the training-recipes page); "trailing window of $n$ steps" → نافذة تضم آخر $n$ خطوة; "autocorrelation time" → زمن الارتباط الذاتي; "initial lag" → التأخر الأولي; "jitter" → اهتزاز (as logged); "band" (of accuracies) → نطاق; "range of live accuracies" → مدى دقة الأوزان الحية.
- "proxy run" → جولة بديلة; "horizon" (planned length of a run) → الأفق; "peak learning rate" → ذروة معدل التعلم; "warmup-and-decay schedule" → جدول الإحماء والاضمحلال; "constant plateau of a WSD schedule" → الهضبة ذات المعدل الثابت من جدول WSD, "along the plateau" → على امتداد الهضبة, "WSD plateau training" → التدريب على هضبة جدول WSD (the scheduler page's هضبة; this replaces مرحلة الثبات of the first application of this unit); "cooldown", "cooldown branch", "cooldown gain" → التبريد، فرع التبريد، كسب التبريد; "cosine-decayed run" → جولة اضمحلال جيب التمام; "fully decayed run" → جولة مكتملة الاضمحلال.
- Slides: "Hypothesis A / B — …" → الفرضية أ / ب، … (Arabic letters; the dash becomes a comma); "Engineering rule" → قاعدة هندسية; "Healthy guard" → احتياط سليم; "fixed-step normalized descent" → نزول مطبَّع ثابت الخطوة; "marginal parameter state" → حالة حدية للمعلَمات; "restoring gradient" → تدرج إرجاع; "Industry Reference Configurations" → إعدادات مرجعية من الصناعة; "frontier pretraining recipes" → وصفات التدريب المسبق لنماذج الصدارة; "batch ramps" → رفع تدريجي لحجم الدفعة; "Diagnostic Playbook" → دليل التشخيص; "failure signatures" → بصمات الإخفاق; "From Telemetry to Action" → من القياسات إلى الإجراء; "step-1 jump" → وثبة في الخطوة 1; "1D norms" → معلَمات التطبيع أحادية الأبعاد; figure colours → بالأزرق، بالبرتقالي، بالأخضر.
- English glosses given in u01 of this page (do not repeat in later units): gradient clipping, exponential moving average (EMA).

## chapter_optimization: chapter check (claude fable-5.2 (medium), 2026-11-27)

- One term per concept across the chapter's pages and chapter_advanced/index.md; where a unit entry above differs from this entry, this entry holds (the re-applied units left superseded first entries for gd.md#u01, momentum.md#u01, test-models.md#u01, sgd.md#u01: read شديد الميل / مستوٍ / سطح الخسارة / التخميد / تعميد / ساكن there as حاد / منبسط / مشهد الخسارة / الإخماد / المعامدة / مستقر).
- "factor", "multiplier" (a multiplicative quantity: "by a factor of", "by what factor", "factor-of-two grid", Muon's scale factor $0.2\sqrt{\max(m,n)}$, "contraction factor") → عامل (بعامل، بأي عامل، ذات العامل $2$، العامل الضربي، عامل الانكماش); معامل is only "coefficient" (معامل الزخم، كثير الحدود مضبوط المعاملات، معامل التناقص of an EMA, as logged). "width multiplier" stays مضاعف العرض.
- "regime": a range of a quantity delimited by a condition (small-step, noise-dominated / noise-limited, curvature-limited, interpolating, large-batch, linear-scaling, damping regimes, complex-root regime) → نطاق (نطاقات); a mode of operation or kind of behaviour (operating regime, streaming regime, learning-rate regimes, clipping regimes, table header "Regime") → نمط, as logged. نظام is not used for "regime"; inside this chapter نمط is also the eigenmode.
- Class noun of optimizers: محسن for Adam, AdamW, AdaGrad, RMSProp, Muon, Shampoo and the matrix optimizers (محسن Adam، المحسن AdamW، يراكم المحسن Shampoo), masculine agreement; خوارزمية stays for SGD (خوارزمية SGD بالدفعات المصغرة); طريقة for AMSGrad, Yogi, SVRG, Polar Express.
- "Maximal Update Parametrization" → تحديد المعلَمات بالتحديث الأعظم ($\mu$P), "standard parametrization" → تحديد المعلَمات المعياري, "reparametrization" → إعادة تحديد المعلَمات; صيغة المعلَمات is not used.
- "sharper / sharpest", "steeper / steepest" (curvature) → أحدّ / الأحدّ, always with the shadda (bare أحد reads as "one"); "destabilize" → يُفقِد … استقراره, with the marks. "steep" → حاد, "flat" → منبسط, "flattest" → الأكثر انبساطًا; "steepest descent" → النزول الأشد.
- "stationary" (noise ball, variance, excess loss) → مستقر, the same word as "stable"; where both meet, "stable step size" → حجم خطوة يحفظ الاستقرار. "steady state" → الحالة المستقرة.
- "buffer" (optimizer state) → مخزن, without مؤقت: مخزن السرعة، مخزن حالة، مخزن الزخم (glossary)، مخزن العزم الأول، مخزنا العزمين; "accumulator" → مركم.
- "usable learning-rate range / window" → نطاق معدلات التعلم الصالحة للاستخدام (one term for both English words); نافذة stays a window of time or memory. "effective" → فعال (حجم الخطوة الفعال، عدد الشرط الفعال، معدل التعلم الفعال), except the logged النافذة الفعلية / نافذة ذاكرة فعلية for an "effective window".
- "global learning rate" → معدل تعلم عام; الشامل stays "global" for a minimizer or the stability ceiling and renders "universal" (حد Pareto الشامل); "global-norm clipping" → القص بالمعيار الكلي, as logged.
- "too small / too large / too slow" → أصغر / أكبر / أبطأ مما ينبغي, not جدًا or شديد.
- "matrix multiplication", "matmul" → ضرب المصفوفات, also on slides; جداء is a product (الجداء النقطي، جداء المصفوفة بمتجه، جداءات كرونيكر). Kronecker is written كرونيكر (glossary), also in "Kronecker-factored" → المحلل إلى عوامل كرونيكر; Newton–Schulz, Gauss–Newton, Polyak–Ruppert, Robbins–Monro, Cauchy–Schwarz stay Latin as hyphenated compound names.
- "scaling": a learning-rate or layer scaling rule → تحجيم (التحجيم الخطي، تحجيم الجذر التربيعي), "Scaling Up" and "Scaling Muon to …" → التحجيم إلى الأكبر / تحجيم Muon إلى …; "scales as / with" → يتناسب مع; "at scale" → عند الأحجام الكبيرة. توسيع is not used for "scaling".
- "data-efficient" → كفؤ في استخدام البيانات (spelling كفؤ, as الشبكات الالتفافية الكفؤة).
- Exercises: "Show that" → بيّن أن, "Prove" → برهن; أثبت only for a cited result in body text (أثبت :citet:`…` أن).
- "harness" → أدوات (أدوات التدريب، أدوات مسح معدل التعلم) for the collective, أداة for one function (أداة التدريب `train`، أداة الإيقاف); "checkpoint" → نقطة استعادة, an evaluation point of a curve → نقطة تقييم; "spike" → قفزة, "jump" → وثبة; a schedule's "plateau" → هضبة, a curve that "plateaus" → يستقر / تستوي; "stalls" → يتعثر.
- Recurring constructs, the same on every page: ## الملخص, ## تمارين, slide title الملخص: …, kicker أهم الخلاصات ("Key Takeaways"), slide title اختبار الفهم: … with (الإجابات), kicker التحقق من الفهم ("Check Your Understanding"), kicker الدافع, **الإجابات:**, exercise references التمرين :exref:`…`; page titles as the index quotes them (المشاهد والنزول التدرجي، التحجيم إلى الأكبر).
