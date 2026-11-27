# Notes on chapter_recurrent-neural-networks/bptt.md#u01 (claude fable-5.2 (medium), 2026-11-27 21:44 UTC)

- 4523d2e4: "1D" is written أحادية الأبعاد (style §6), so the digit 1 is not in the translation.
- a1dd3f4d, 5ad30106, a90e6313 and the slides: "gradient flow" is rendered سريان التدرج (backward travel of gradients, as in the MLP chapter), not the glossary's تدفق التدرج, which is the continuous-time gradient-flow ODE; a glossary warning here is a false match.
- 2e2b0097, 566c7afb: the em dash inside the table cell became a comma (style §4), and "SSMs / Linear RNNs" became نماذج SSM أو شبكات RNN الخطية so that an Arabic word separates the Latin names.
- 85faa7e0: the answer lead-in "$31$ Jacobians" takes the lead word الفاصل so that the bold text does not start with a math span.
- ba77ea2b: "lags 2 and 4 … a third and a tenth" is written out per lag (عند التأخر 2 … وعند التأخر 4 …) for clarity; nothing added.
- 599549e0, e035fd62: "vanishing and exploding gradients" is تلاشي التدرجات وانفجارها (the form decided in the MLP chapter; the pronoun stands for التدرجات), so the check does not find انفجار التدرجات: false match.
- d26f64dd: المعايير is the broken plural of معيار (norms): false match.
- 250e4d3a, 29396e09, 957d7954, 83dedc98: the token term stands in the accusative after a number (رمزًا مميزًا), which the check does not match: false match.
