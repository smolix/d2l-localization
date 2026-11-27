# Notes on chapter_attention/what-attention-computes.md#u02 (claude fable-5.2 (medium), 2026-11-27 23:07 UTC)

- 8d20f638 / f21f8c94: the English source wraps the formula "$j = i + 1 - L$" so that a line begins with "- L$.", and the unit splits the sentence there into a paragraph and a list item (the English page renders the same way). The translation keeps the split as given: the paragraph ends with the open "$j = i + 1" and the list-item block is copied unchanged ("- L$."). The English source should be re-wrapped so that the formula stays on one line.
- 3b1569b2 / ed1a3e57 and efbdf9fd / ba2e09f9: the sentence runs through a display equation, as in the English (ليكن … مقدار رفع اللوجيت; بدقة إجمالية مقدارها … وبدقة).
- 56a41d84, 198df1a6, d67fa1b7, 80a9ac2a: "1-block" and "0-block" are written ذو الكتلة الواحدة / بلا كتل (style §6), so a number warning for 1 or 0 is a false match.
- b721adcb, ec1d9ea7, 3fbcb68d: glossary warning "norms → معيار" is a false match: the term is there in its broken plural (معايير التيار).
- 52f1826f, d253f70d: glossary warning "token → رمز مميز" is a false match: the term is there in the indefinite accusative (رمزًا مميزًا مكررًا).
- 8d20f638: "between one and three" is spelled out in the English and in the translation.
