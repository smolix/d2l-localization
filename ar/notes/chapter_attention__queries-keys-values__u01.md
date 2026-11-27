# Notes on chapter_attention/queries-keys-values.md#u01 (claude fable-5.2 (medium), 2026-11-27 22:52 UTC)

- 4ba4e9fd, 884264f4: the English sentence "take $a(\mathbf{q}, \mathbf{k}) = -\|\mathbf{q} - \mathbf{k}\|^2$ with query …" is one paragraph that the unit splitter cut in two, because its second source line begins with "- " (the minus sign of the formula, read as a list marker). The math span therefore opens in 4ba4e9fd and closes in 884264f4. The translation keeps the cut at the same place, so 884264f4 has to begin with `- \mathbf{k}\|^2$` and cannot start with an Arabic word; the two blocks read as one sentence when joined. The English source should be re-wrapped so that the line does not start with "- ".
- c1071ac6, 71d62326, 1a70b45a: the verb "returns" (يعيد / أعاد) is not the RL term "return"; "prior to querying" (قبل الاستعلام) is not the Bayesian "prior".
- a5e6c63a: "1-nearest-neighbor" → أقرب جار واحد, the digit 1 is written as واحد (style §6).
- 3a19eee9: "2D" → ثنائي الأبعاد (style §6); 37b42a74: "1-Hop" → بقفزة واحدة.
- 8f9a1575, 7cf1aaaf: تراكيب محدبة / تركيبًا محدبًا are the plural and the accusative of the glossary's تركيب محدب; وسوم is the plural of وسم; مجموعة بيانات تدريب is the glossary's مجموعة البيانات in a genitive chain.
- 19722da5, 3cf4ad1d, eff5268f: the slide titles "Why …" and "How …" are written as questions with ؟, as the style guide's "Why Depth Helps" → لماذا يفيد العمق؟.
- c52a8f04, 34483fb8: "tokens" is رمزًا مميزًا, the glossary's رمز مميز in the accusative after a number (style §5.2); the check does not match the tanwin form.
