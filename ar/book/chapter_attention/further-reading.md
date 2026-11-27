<!--en:7902ddfd-->
# قراءات إضافية

<!--en:7ca3a29b-->
طورت الأقسام السابقة تصميمًا نموذجيًا واحدًا لكل مكوّن من مكونات الطبقة: البحث بدالة softmax (:numref:`sec_queries-keys-values`)، وحساب الدرجات بالجداء النقطي المقيس والتقنيع (:numref:`sec_attention-scoring-functions`)، والانتباه الذاتي متعدد الرؤوس وانتباه الاستعلامات المجمعة (:numref:`sec_multihead-attention`)، والترميزات الموضعية (:numref:`sec_positional-information`)، وخوارزمية softmax الآنية المقسمة إلى بلاطات (:numref:`sec_attention-at-scale`)، وانتباه النافذة المنزلقة والانتباه الخطي (:numref:`sec_windowed-linear-attention`)، والدوائر التي تفسر الآلية (:numref:`sec_what-attention-computes`). ويقدم هذا القسم مسحًا للصيغ القريبة منها وللتحسينات المعمارية في النماذج الحديثة مفتوحة الأوزان وللمراجع الأصلية.

<!--en:f9243964-->
## البحث وتطبيعه

<!--en:eacf96f0-->
### الانتباه بوصفه ذاكرة ترابطية

<!--en:7ac2ec98-->
يربط :citet:`Ramsauer.Schafl.Lehner.ea.2021` انتباه softmax بالذاكرات الترابطية الكلاسيكية، إذ يبين أن خطوة واحدة من انتباه softmax تقابل تحديثًا واحدًا *لشبكة Hopfield حديثة* (modern Hopfield network) مستمرة، أنماطها المخزنة هي صفوف المفاتيح والقيم. ويحدد مقلوب درجة الحرارة ($1/\sigma^2$ في انحدار النواة و $1/\sqrt{d}$ في انتباه الجداء النقطي) هل النقاط الثابتة أنماط مخزنة مفردة (مقلوب درجة حرارة مرتفع) أم متوسطات شبه مستقرة لعدة أنماط (مقلوب درجة حرارة منخفض). ومن هذا المنظور يلاحظ المؤلفون أن الطبقات الأولى في المشفرات المدرَّبة مسبقًا مثل BERT تعمل في الغالب في نطاق المتوسط العريض، في حين تعمل الطبقات الأعمق في نطاق الاسترجاع الحاد. وتقابل عروض النطاق التكيفية للنواة :cite:`Silverman86` درجات حرارة softmax المعتمدة على الاستعلام أو على الرأس.

<!--en:43012c03-->
### المحاذاة الرتيبة

<!--en:f1a68df5-->
قبل عام من أن يقدم :citet:`Bahdanau.Cho.Bengio.2014` المحاذاة غير المقيدة القائمة على المحتوى في الترجمة الآلية، قدم :citet:`Graves.2013` آلية انتباه *رتيبة* قابلة للاشتقاق وقائمة على الموضع لتوليف الكتابة اليدوية. تحدد معلَمات هذه الآلية خليطًا من النوافذ الغاوسية تتقدم مواضع متوسطاتها تقدمًا رتيبًا مع كل جرّة قلم، وهذا انحياز استقرائي يلائم الكلام والكتابة اليدوية (:cite:`rabiner1993fundamentals`) لكنه أشد تقييدًا مما يحتمله انقلاب ترتيب الكلمات في الترجمة (:numref:`subsec_learned-alignment`).

<!--en:eeca9527-->
### أوزان لا يلزم أن يساوي مجموعها واحدًا

<!--en:2a2284eb-->
لأن مجموع كل صف من مخرجات softmax يجب أن يساوي واحدًا، كثيرًا ما يلقي الاستعلام الذي ليس له مفاتيح ذات صلة كتلته الاحتمالية على الرمز المميز الأول، وهو *مصرف الانتباه* (attention sink) المذكور في :numref:`subsec_cross-attention-alignment` وفي :numref:`sec_what-attention-computes` (انظر أيضًا :numref:`subsec_kv-cache-sinks`). ويربط :citet:`Sun.Chen.Kolter.ea.2024` مصارف الانتباه بقيم شاذة هائلة من التفعيلات في تيار البواقي تعمل عمل حدود انحياز ضمنية، في حين يرى :citet:`Barbero.Arroyo.Gu.ea.2025` أن المصارف تمنع انهيار التمثيلات في المحولات العميقة ذات السياق الطويل لأنها توفر هدفًا هو عملية بلا أثر.

<!--en:ce9a76df-->
ترخي عدة تصاميم حديثة قيد المجموع الواحد كي تستطيع الرؤوس أن تبقى خاملة دون رمز مميز من المدخلات يعمل مصرفًا:

<!--en:09c2859d-->
* **انتباه سيغمويد** (sigmoid attention) :cite:`Ramapuram.Danieli.Dhekane.ea.2025` يستعيض عن دالة softmax المطبقة صفًا بصف بدالة سيغمويد اللوجستية المطبقة عنصرًا بعنصر والمزاحة بمقدار $-\log n$، فتبدأ الأوزان قرب $1/(n + 1)$.

<!--en:749a28e7-->
* **الانتباه ذو البوابة** (gated attention) :cite:`Qiu.Wang.Zheng.ea.2025`، المستخدم في Qwen3-Next :cite:`Qwen.Team.2025`، يضرب مخرجات softmax لكل رأس في بوابة سيغمويد معتمدة على المدخل.

<!--en:20df9920-->
* **لوجيتات المصرف القابلة للتعلم** (learnable sink logits)، المستخدمة في النموذج `gpt-oss` :cite:`OpenAI.2025`، تضيف إلى مقام softmax درجة مفتاح ظاهري قابلة للتدريب لا يقابلها متجه قيمة.

<!--en:8a6f1e90-->
* **الانتباه التفاضلي** (differential attention) :cite:`Ye.Dong.Xia.ea.2025` يطرح في كل رأس توزيعًا من softmax من توزيع آخر، ليلغي الوزن الخلفي المشترك بينهما الواقع على المفاتيح التي لا صلة لها.

<!--en:c599fa90-->
## الانتباه الذاتي والرؤوس

<!--en:56792267-->
### الأصول وفائض الرؤوس

<!--en:98988fa9-->
تطور الانتباه الذاتي من البنى المتكررة ذات *الانتباه الداخلي* (intra-attention) :cite:`Cheng.Dong.Lapata.2016`، ومن الانتباه القابل للتفكيك الخالي من التكرار في استدلال اللغة الطبيعية :cite:`Parikh.Tackstrom.Das.ea.2016`، ومن تضمينات الجمل متعددة الصفوف مع عقوبات التنوع :cite:`Lin.Feng.Santos.ea.2017`، ثم وحّد :citet:`Vaswani.Shazeer.Parmar.ea.2017` ذلك في الانتباه الذاتي متعدد الرؤوس ذي الرؤوس المسقطة ضمن المحول.

<!--en:c8e64bf8-->
تعدد الرؤوس ضروري لتجاوز عنق زجاجة الخليط الواحد (:numref:`subsec_one-mixture-bottleneck`)، لكن النماذج المدرَّبة كثيرًا ما تُظهر وقت الاختبار فائضًا كبيرًا في الرؤوس: بيّن :citet:`Michel.Levy.Neubig.2019` و :citet:`Voita.Talbot.Moiseev.ea.2019` أن نسبة كبيرة من الرؤوس في نماذج BERT ونماذج NMT يمكن تشذيبها بعد التدريب بتدهور ضئيل، فتبقى مجموعة أساسية من الرؤوس الموضعية والنحوية ورؤوس الكلمات النادرة. أما في المعماريات الحديثة فتتحقق الكفاءة في المعلَمات وفي الذاكرة المؤقتة قبل التدريب عبر انتباه الاستعلامات المجمعة (:numref:`subsec_kv-head-sharing` و :numref:`sec_kv-cache-reduction`).

<!--en:d5ac1b26-->
## المواضع

<!--en:1d6cd3d9-->
### المواضع النسبية المتعلَّمة

<!--en:b7c78f35-->
قبل ترميز RoPE كانت الإزاحات النسبية تُحقن عبر تضمينات مفاتيح قابلة للتدريب لكل إزاحة :cite:`shaw2018self`، أو عبر لوجيتات نسبية مفهرسة بالإمالة في النموذج Music Transformer :cite:`huang2018music`، أو عبر انحيازات قياسية لخانات مسافة لوغاريتمية في النموذج T5 :cite:`raffel2020exploring`. ثم استعاض ترميز ALiBi عن الخانات المتعلَّمة بميول خطية ثابتة، وجعل ترميز RoPE الدرجة تعتمد على الإزاحة وحدها اعتمادًا مضبوطًا عبر دورانات للإحداثيات بلا معلَمات.

<!--en:fe39c899-->
### ترميز RoPE في النماذج المدرَّبة

<!--en:ac7d606f-->
رفعت النماذج مفتوحة الأوزان تدريجيًا التردد الأساسي $b$ في ترميز RoPE لدعم سياقات أطول: من $b = 10^4$ في Llama 2 إلى $5 \times 10^5$ في Llama 2 Long :cite:`Xiong.Liu.Molybog.ea.2024` وفي Llama 3 :cite:`Grattafiori.Dubey.Jauhri.ea.2024`، ثم إلى $10^6$ في الطبقات الشاملة من Gemma 3 :cite:`Gemma.Team.2025`. وتنقل زيادة $b$ عددًا أكبر من أزواج الإحداثيات إلى النطاق البطيء الذي يتجاوز طوله الموجي نافذة التدريب (:numref:`subsec_pos-context-extension`). ويبين :citet:`Barbero.Vitvitskyi.Perivolaropoulos.ea.2025` أن النماذج المدرَّبة تستخدم أزواج RoPE عالية التردد في الرؤوس الموضعية (مثل رؤوس الرمز السابق)، وتستخدم الأزواج منخفضة التردد قنوات دلالية شبه ثابتة أمام الموضع. وهذا دافع الصيغة *p-RoPE* التي تصفّر أدنى الترددات كليًا كي لا تنجرف القنوات الدلالية عند المسافات القصوى. وفي تمديد السياق بعد التدريب إلى أبعد مما تبلغه طريقة YaRN :cite:`Peng.Quesnelle.Fan.ea.2024`، تستخدم طريقة *LongRoPE* :cite:`Ding.Zhang.Zhang.ea.2024` بحثًا تطوريًا عن عوامل إعادة تحجيم لكل بُعد لتمديد نوافذ السياق حتى مليوني رمز مميز.

<!--en:7bbb09f7-->
### التعميم على الأطوال واستغلال السياق الطويل

<!--en:0b71a641-->
يحلل :citet:`Chi.Fan.Rudnicky.ea.2023` مجال الاستقبال الفعال لترميز ALiBi، ويؤيد تحليله ما وجدناه في :numref:`subsec_pos-experiment` من أن بقاء الحيرة منبسطة عند الاستكمال خارج المدى يعكس اقتصارًا على نافذة محلية لا استرجاعًا بعيد المدى. وفي النماذج السببية الخالية من الترميزات الموضعية (NoPE, :cite:`Haviv.Ram.Press.ea.2022`) يبين :citet:`Wang.Ji.Wu.ea.2024` أن إنتروبيا الانتباه تزداد مع طول التسلسل وأن تحجيم درجة الحرارة لكل رأس يعاكس هذه الزيادة، وهي آلية تُستخدم عند الاستدلال في طبقات NoPE الشاملة من Llama 4 :cite:`Meta.2025`. وأخيرًا، انخفاض الحيرة الإجمالية على المستندات الطويلة لا يضمن استرجاعًا فعالًا عبر النافذة كلها: يبين :citet:`Liu.Lin.Hewitt.ea.2024` ظاهرة *“الضياع في الوسط”* (lost in the middle)، وفيها تنخفض الدقة انخفاضًا كبيرًا حين تظهر الحقائق ذات الصلة في وسط سياق طويل لا قرب بدايته أو نهايته (انظر :numref:`sec_lm_long_context`).

<!--en:a4651e26-->
## الكلفة

<!--en:b1ef82f1-->
### التقريبات المبكرة ذات التعقيد الخطي

<!--en:8871cfcf-->
قبل طريقة FlashAttention قرّبت معماريات كثيرة مصفوفة الانتباه $n \times n$: استخدم النموذج *Performer* :cite:`Choromanski.Likhosherstov.Dohan.ea.2021` ميزات عشوائية متعامدة موجبة لتقريب النواة الأسية؛ وطبّق النموذج *Linformer* :cite:`Wang.Li.Khabsa.ea.2020` إسقاطات منخفضة الرتبة للتسلسل؛ واستخدم النموذج *Reformer* :cite:`Kitaev.Kaiser.Levskaya.2020` دوال تجزئة حساسة للمحلية (locality-sensitive hashing)؛ وجمع النموذجان *Longformer* :cite:`beltagy2020longformer` و *Sparse Transformer* :cite:`child2019generating` بين النوافذ المحلية ورموز مميزة متباعدة بخطوة انزلاق أو شاملة (ويقدم :citet:`Tay.Dehghani.Bahri.ea.2020` مسحًا لهذه المعماريات).

<!--en:d25e4a5d-->
### النوافذ والتناثر المتكيف مع المحتوى في النماذج المنشورة

<!--en:b8f5bee0-->
طبّق النموذج Mistral 7B :cite:`Jiang.Sablayrolles.Mensch.ea.2023` النوافذ المنزلقة في كل طبقة (ثم أُسقطت لاحقًا في :cite:`Mistral.AI.2023`)، أما المعماريات التالية فتناوب بين النوافذ المحلية وطبقات شاملة دورية: بنسبة 1:1 في Gemma 2 :cite:`Gemma.Team.2024`، وبنسبة 5:1 في Gemma 3 ($w = 1{,}024$, :cite:`Gemma.Team.2025`)، وبنسبة 1:1 مع $w = 128$ في النموذج `gpt-oss` :cite:`OpenAI.2025`، وبطبقات محلية مقسمة إلى قطع تقترن بطبقات NoPE شاملة في Llama 4 :cite:`Meta.2025`.

<!--en:c64b425b-->
وإلى جانب النوافذ الموضعية الثابتة ظهر *الانتباه المتناثر القائم على المحتوى* (content-based sparse attention) الملائم للعتاد، للتدريب وفك التشفير في السياق الطويل (:numref:`sec_lm_sparse_attention`)، ومنه *Native Sparse Attention* (NSA, :cite:`Yuan.Gao.Dai.ea.2025`) و *Mixture of Block Attention* (MoBA, :cite:`Lu.Jiang.Liu.ea.2025`) و *DeepSeek Sparse Attention* :cite:`Liu.Mei.Lin.ea.2025`.

<!--en:97c18b6a-->
### طرائق FlashAttention-2 و FlashAttention-3 و FlexAttention

<!--en:dac281b1-->
بعد طريقة FlashAttention :cite:`Dao.Fu.Ermon.ea.2022` جاءت طريقة *FlashAttention-2* :cite:`Dao.2024` التي توزع الحساب بالتوازي على بُعد تسلسل الاستعلامات وتؤجل التطبيع إلى نهاية حلقة المفاتيح. وعلى وحدات GPU من جيل Hopper تجعل طريقة *FlashAttention-3* :cite:`Shah.Bikshandi.Zhang.ea.2024` عمليات ضرب المصفوفات غير المتزامنة على أنوية الموترات تتداخل مع حساب الدوال الأسية في softmax ضمن مجموعات خيوط (warps) متخصصة، وتدعم الصيغة FP8 عبر دورانات متعامدة مسبقة غير متماسكة. وتصرّف طريقة *FlexAttention* :cite:`Dong.Feng.Guessous.ea.2025` دوال مغلقة اعتباطية لتعديل الدرجات وأقنعة كتلية إلى نوى Triton مدمجة، أما نوى فك التشفير المتوازية بذاكرة KV المؤقتة (:numref:`sec_lm_decode_kernels`) فتقسم محور المفاتيح وتدمج الحالات الجزئية $(m_B, s_B, \mathbf{o}_B)$ بخوارزمية softmax الآنية.

<!--en:9352878e-->
### البنى المتكررة ذات البوابات والبنى الهجينة

<!--en:aa39988b-->
تعالج النماذج الحديثة المنحدرة من الانتباه الخطي سعة الذاكرة بترقيتين يشرحهما :numref:`chap_modern_rnn`: (1) بوابات نسيان معتمدة على البيانات (*Gated Linear Attention*, :cite:`Yang.Wang.Shen.ea.2024`, :numref:`sec_matrix-state`) و (2) تحديثات دلتا المصححة للخطأ (*DeltaNet*, :cite:`Schlag.Irie.Schmidhuber.2021,Yang.Wang.Zhang.ea.2024`، و *Gated DeltaNet*, :cite:`Yang.Kautz.Hatamizadeh.2024`, :numref:`sec_deltanet`). وتجمع المعماريات المستخدمة في بيئة الإنتاج بين هذه الطبقات المتكررة وطبقات انتباه softmax كاملة دورية (طبقة واحدة من كل 4 إلى 8 طبقات عادةً) للحفاظ على الاستدعاء الترابطي المضبوط :cite:`Qwen.Team.2025,Kimi.Team.2025b,MiniMax.2025` (:numref:`sec_hybrids`).

<!--en:8aefd574-->
## الدوائر في النماذج الكبيرة

<!--en:8e9f08c5-->
### ترقيع التفعيلات والتراكب والتفسير

<!--en:38cb8d5a-->
لتجاوز الاستئصال الصفري أو الاستئصال بالمتوسط (:numref:`subsec_head-ablation`) يضع *ترقيع التفعيلات* (activation patching) تفعيلات مأخوذة من تشغيل سليم في تشغيل تالف لعزل المسارات السببية :cite:`Meng.Bau.Andonian.ea.2022,Wang.Variengien.Conmy.ea.2023`، ويُؤتمت ذلك بطريقة ACDC :cite:`Conmy.MavorParker.Lynch.ea.2023` أو *بترقيع العزو* (attribution patching) من المرتبة الأولى :cite:`Nanda.2023,Kramar.Lieberum.Shah.ea.2024,Zhang.Nanda.2024,Heimersheim.Nanda.2024`. وحين يفوق عدد الميزات عدد أبعاد تيار البواقي (*التراكب*، superposition، :cite:`Elhage.Hume.Olsson.ea.2022`) تفك *المشفرات التلقائية المتناثرة* (sparse autoencoders) وبيانيات العزو تشابك الاتجاهات متعددة الدلالات فتحولها إلى ميزات أحادية الدلالة :cite:`Bricken.Templeton.Batson.ea.2023,Cunningham.Ewart.Riggs.ea.2024,Ameisen.Lindsey.Pearce.ea.2025` (:numref:`sec_lm_interpretability`). وتحسم هذه الأدوات التدخلية كذلك الجدل الكلاسيكي بين :citet:`Jain.Wallace.2019` و :citet:`Wiegreffe.Pinter.2019` حول ما إذا كانت أوزان الانتباه تفسيرات أمينة.

<!--en:04fc81e2-->
### التعلم داخل السياق بوصفه خوارزمية

<!--en:6777c59d-->
إلى جانب النسخ برؤوس الاستقراء :cite:`Olsson.Elhage.Nanda.ea.2022`، تستطيع المحولات المدرَّبة على عائلات من مسائل الانحدار داخل السياق أن تنفذ النزول التدرجي وانحدار ريدج وخوارزميات حل المربعات الصغرى مباشرةً في تمريرها الأمامي :cite:`Garg.Tsipras.Liang.ea.2022,Oswald.Niklasson.Randazzo.ea.2023,Akyurek.Schuurmans.Andreas.ea.2023`، وإن كان مدى ما يستحثه التدريب المسبق على النصوص العامة من نزول تدرجي ضمني لا يزال موضع جدل :cite:`Shen.Mishra.Khashabi.2024`.

<!--en:60b0333c-->
## مصادر موصى بها

<!--en:e8dd6d7a-->
* **مقدمات بصرية:** درس 3Blue1Brown المعنون [*Attention in Transformers, Step-by-Step*](https://www.3blue1brown.com/lessons/attention) (2024)؛ ومقالة Jay Alammar المعنونة [*The Illustrated Transformer*](https://jalammar.github.io/illustrated-transformer/) (2018)؛ ومقالة Peter Bloem المعنونة [*Transformers from Scratch*](https://peterbloem.nl/blog/transformers) (2019)؛ ومقالة Lilian Weng المعنونة [*The Transformer Family Version 2.0*](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) (2023).

<!--en:429e8694-->
* **دراسات تأسيسية ودراسات عن الدوائر:** دراسة Elhage et al. المعنونة [*A Mathematical Framework for Transformer Circuits*](https://transformer-circuits.pub/2021/framework/index.html) (2021)؛ ودراسة Olsson et al. المعنونة [*In-Context Learning and Induction Heads*](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) (2022)؛ وورقة Weiss و Goldberg و Yahav المعنونة [*Thinking Like Transformers*](https://arxiv.org/abs/2106.06981) (2021) ومعها الألغاز المرافقة [*Transformer Puzzles*](https://github.com/srush/Transformer-Puzzles) التي أعدها Sasha Rush.

<!--en:8beae255-->
[النقاشات](https://d2l.discourse.group/t/36852)
