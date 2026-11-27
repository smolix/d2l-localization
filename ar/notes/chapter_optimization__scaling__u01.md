# Notes on chapter_optimization/scaling.md#u01 (claude fable-5.2 (medium), 2026-11-27 22:27 UTC)

- 668c78db, 18cd5ae7, 9e229c1a, f71d1e81, a870f9d8, e70900c8: "1 Adam step", "1-step" and "after 1 step" are written خطوة واحدة (style §6: one of a counted noun without the digit), so the check may report the number 1 as missing.
- 5b165635: the ranges "$16$–$32\times$" and "$\sim 1$–$2 \times 10^{-4}$" are written with بين … و … (style §3.2, no en dash in prose); the math spans are unchanged.
- cb5655ed, 07670065, 18cd5ae7: the slide citations (Yang et al., 2022) get لدى / وفق in front, so that an Arabic word separates the Latin citation from the math span or bold term before it.
- 33500d1b, d06cb3c5: the bracketed code names of the bold lead-ins ("Rule 3 (`__init__`)", "Layer 1 (`fc_in`, fan-in $784$)") are joined with في / وهي instead of a bare bracket, so that a digit and a code span are not adjacent left-to-right runs.
- c9a3266a: "all $n$ terms $|h_j| > 0$" keeps the English order of the two math spans (الحدود وعددها $n$، أي $|h_j| > 0$).
- 64f9204f: "135K to 141M parameters" is written من 135 ألف إلى 141 مليون معلَمة, as logged for parameter counts on slides.
- 408dded7, 40a09169: the glossary check reports "hyperparameter(s)" as unused; the term is there in the plural المعلَمات الفائقة (false match on the singular form).
- eeb982ac: "norms" here are normalization-layer parameters, not vector norms, so they are معلَمات التطبيع and the glossary's معيار is a false match.
