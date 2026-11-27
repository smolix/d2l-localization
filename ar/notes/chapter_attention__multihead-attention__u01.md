# Notes on chapter_attention/multihead-attention.md#u01 (claude fable-5.2 (medium), 2026-11-27 22:56 UTC)

- ece735db, 50fc153a: the English segmenter split the proof paragraph inside a math span (the line that starts with `+ \alpha_2^2}` was taken for a list item). The translation keeps the split at the same place, so block 50fc153a starts with the rest of the formula and not with an Arabic word; the two blocks read as one paragraph.
- e3e04557: "4D", "5D" are written رباعي الأبعاد / خماسي الأبعاد (style §6), so the check may report the numbers 4 and 5 as missing; the dual بُعدين is not used, $d_h = 2$ is kept.
- 425f27f4: "1D Conv" → التفاف أحادي الأبعاد; the number 1 is reported as missing (style §6).
- 8d797dd7, 913159b7 and the other blocks with the example sentence: the straight quotes of *"Alice gave Bob the book"* and "the" became “…” (style §4), in title attributes ‘…’.
- 8f58c532, a78be9d0, a27ce4fd, cbe2064b: the class noun الكلمة / الكلمتان was added so that the row starts with an Arabic word (style §3.2).
- 71066c75: the slide title says "Theorem" where the body text has a "Proposition"; translated as written (مبرهنة).
- 1a0dcb62, e3e04557: the warning "glossary term not used: 'trace'" is a false match; "trace" is the verb (نتتبع، لنتتبع), not the trace of a matrix.
