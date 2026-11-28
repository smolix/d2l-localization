# Notes on chapter_computational-performance/sharding.md#u02 (claude fable-5.2 (medium), 2026-11-28 01:11 UTC)

- 33e0eb2b, d634531c, c62bb76c: "two GPUs" / "two RTX 4090s" is the dual without a digit, as in the English.
- 982e1a82: "collect" is نأخذ, because نجمع reads as "sum" on this page.
- d13a5945: "a 1D `DeviceMesh`" → شبكة `DeviceMesh` أحادية الأبعاد (1D); the Latin 1D is kept in parentheses so that the digit is present.
- d13a5945, 7d3b57ef: "column-wise" / "row-wise" name the direction of the split (حسب الأعمدة / حسب الصفوف), not the lexicon's صفًا بصف.
- d13a5945, 7d3b57ef, and the title of exercise 3: tensor parallelism is توازي الموترات, as decided on multiple-gpus.md#u01; the glossary's proposed تواز للموترات cannot take the article.
- 4c286512: "1B GPT" → نموذج GPT بمليار معلَمة (style §15.7), so the digit 1 is not in the translation; "DDP + gradient accumulation" → طريقة DDP مع تراكم التدرجات, so that the bold does not start with a Latin word.
- c62bb76c: "two $24\text{ GiB}$ RTX 4090s" → بطاقتين من $24\text{ GiB}$ هما بطاقتا RTX 4090, keeping the English token order.
- 7014f3c6: the slide shorthand "RS" is written out as عملية reduce-scatter.
- interconnect: this page writes وصلة ربط (u01 of this page, and hardware.md's وصلات الربط بين وحدات GPU); multi-gpu-practice.md#u01, applied in parallel, logged الربط البيني. The editor should settle one form.
- fabric: the slide title b31c925b writes شبكات ربط NVLink 5 after hardware.md#u02 (شبكة ربط NVLink); multi-gpu-practice.md#u01 logged نسيج.
- f98d8942: "glossary term not used: tokens" is a false match; the term stands in the accusative after the number (2,048 رمزًا مميزًا).
- 02b7140a: "glossary term not used: training" is a false match; "training a model" is the verb فتدرب.
