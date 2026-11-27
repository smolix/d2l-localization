# Notes on chapter_linear-regression/weight-decay.md#u02 (claude fable-5.2 (medium), 2026-11-27 20:07 UTC)

- 15941bf7, b41b9fdf, 97e1e654: "two or three epochs", "two epochs", "a factor of two to six", "some fifty" are spelled out in English and in Arabic (حقبتين، من اثنين إلى ستة، خمسين); any number or glossary warning there is a false match.
- 726e8dea: "after 1 epoch" → بعد حقبة واحدة (style §6: one of a counted noun without the digit); the digit 1 is reported missing.
- c00fdb38: "span" is the verb twice (enough to span → تكفي لتوليد; eigenvalues span two orders of magnitude → تمتد على); a glossary warning for span (فضاء مولد) is a false match. "return"/"returns" in e4a9ef3a, 7c914706, 568dc56d, 867f67b7 is the function sense (تعيد), not the RL return.
- 867f67b7: `data` moved after the math span $d = 200$ (different token classes).
- a04b02fa: the number 180 moved after the second $\mathbf{X}^\top\mathbf{X}$ so that an Arabic word separates the math span from the digits.
- e71d251e and slides: "filtering" rendered ترشيح although an earlier decision has التصفية for data filtering; see DECISIONS.
- "value" (glossary: attention value) occurs here only as an ordinary word (قيمة).
