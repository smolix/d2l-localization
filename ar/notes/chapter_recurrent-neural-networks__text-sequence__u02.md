# Notes on chapter_recurrent-neural-networks/text-sequence.md#u02 (claude fable-5.2 (medium), 2026-11-27 21:46 UTC)

- 3617e964 / a5f32a6e: the English source wraps a paragraph so that a line begins with "3. ", and the parser split the paragraph into a paragraph that ends "at step" and a list item "3. Following standard convention …". The translation keeps the same split (the first block ends with في الخطوة, the second starts with "3."), so nothing moves between blocks; the English source should be re-wrapped so that the number does not start a line.
- 42f1fc34, 042232d7, 333fda5b and the heading 1e377853: "decode / decoding" here is the mapping of token ids back to text, translated فك الترميز as the decisions log has it, not the glossary's فك التشفير (generation-time decoding); a glossary warning for these blocks is a false match.
- cd91f443, 3b721bca, 6b224def, d9fa855a and several slide blocks: a free-standing و was put between adjacent code spans or quoted Latin words in lists (style §3.2); the order of the code spans is unchanged.
- ab94b41d: "5-branch" keeps the digit (ذو 5 فروع) on the slide.
- 6b224def: "the 4,000th merge" → الدمج رقم 4,000; the warning "numbers not found: 4" comes from this ordinal and is a false match.
- 3b721bca, 29e155ff: "tokens" is translated رموزًا مميزة (indefinite accusative with tanwin), which the glossary check does not match; false match.
- 3b721bca, 83cbadc7, 99666d17, 042232d7: "trace" is the hand trace of the algorithm (التتبع), not the trace of a matrix; false match.
