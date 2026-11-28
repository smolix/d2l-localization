# Notes on chapter_transformers/kv-cache.md#u01 (claude fable-5.2 (medium), 2026-11-27 23:50 UTC)

- 1062e509: the gloss (hardware roofline) stands at the first mention of خط السقف on the page, so **نموذج خط السقف** in a6320d44 is not glossed again; attention sinks, GQA and MQA are glossed where the next unit defines them.
- e5763145: the English says "powers of two from 256 to 4,032"; 4,032 is not a power of two (the code uses 4032 so that 32 decode steps fit in 4,096). Translated as written.
- e58995e3: "250× Redundant Work" rendered as "250× of the necessary work" (ينجز 250× من العمل اللازم), which is what the slide's numbers say.
- 7c95c119, c768d884, 0cfd992d: حيث and وفق added between the Latin acronym, the math span and the Latin citation (style §3.2, §15.4).
- 718cb812, 55fda0b8, a5cbd2bb, ec74c8d8, 845101a4, f630b905: warning "glossary term not used: token(s) → رمز مميز" is a false match; the term stands in the accusative with tanwin (رمزًا مميزًا) after a count.
- d03e0ba6: warning "glossary term not used: batch sizes → حجم الدفعة" is a false match; the broken plural أحجام الدفعات is used.
