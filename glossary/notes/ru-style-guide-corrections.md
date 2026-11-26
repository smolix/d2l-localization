# Russian (ru): glossary corrections from the style guide

Applied 2026-11-26 from `ru/style.md` §13 ("Proposed glossary corrections").
All Russian entries were `proposed`, so no correction conflicted with an attested
entry. Status stays `proposed` throughout; each changed record has a note that
starts "Style guide 2026-11-26:". The glossary has no `avoid` status, so the slang
loans of correction 5 were removed from `alternatives` and named in `notes`.
Concepts the glossary lacks were not added; they are listed as "new concept" for
the coordinator.

| Concept | Before | After | Reason |
|---|---|---|---|
| bias | alternatives include «сдвиг» | «сдвиг» removed | applied: «сдвиг» is reserved for offset, shift and translation (style §9.1) |
| intercept | alternatives include «сдвиг» | «сдвиг» removed | applied: same reason |
| feature | alt. «фича» | removed | applied: colloquial loan (style §5.1) |
| label | alt. «таргет» | removed | applied: colloquial loan |
| target-variable | alt. «таргет» | removed (no alternatives left) | applied: colloquial loan |
| dataset | alt. «датасет» | removed | applied: colloquial loan |
| training-set | alt. «трейн» | removed | applied: colloquial loan |
| loss-function | alt. «лосс» | removed | applied: colloquial loan |
| loss | alt. «лосс» | removed | applied: colloquial loan |
| backpropagation | alt. «бэкпроп» | removed | applied: colloquial loan |
| stride | alt. «страйд» | removed | applied: colloquial loan |
| padding | alt. «паддинг» | removed | applied: colloquial loan |
| padding-mask | alt. «маска паддинга» | removed | applied: contains the colloquial «паддинг» |
| fine-tuning | alt. «файнтюнинг» | removed | applied: colloquial loan |
| score | alt. «скор» | removed | applied: colloquial loan |
| attention-score | alt. «скор внимания» | removed | applied: contains the colloquial «скор» |
| max-pooling | макс-пулинг | макс-пулинг (note only) | applied as the guide's second option: keep, note the asymmetry with «усредняющий пулинг»; the symmetric pair «пулинг по максимуму / по среднему» (already alternatives) is for a native reviewer |
| average-pooling | усредняющий пулинг | unchanged (note only) | applied: same note |
| floating-point | плавающая запятая | unchanged (note only) | applied: the term names the format, not the book's decimal-point notation (style §6, pending §12.1) |
| fully-connected-layer | полносвязный слой | unchanged | new alias: English "dense layer" belongs in `concepts.jsonl` `aliases_en` |
| offset (positional) | none | сдвиг | new concept: never «смещение» |
| shift (positional) | none | сдвиг | new concept: never «смещение» |
| inference (running a model) | none | инференс | new concept: the statistical entries use «вывод»; only «стоимость инференса» had the word |
| framework | none | фреймворк | new concept |
| notebook (Jupyter) | none | ноутбук | new concept |
| edge (in an image) | none | граница | new concept |
| patch (of a convolution input) | none | фрагмент | new concept: distinct from `image-patch` (ViT, «патч») |
| unit (of a feature map) | none | элемент | new concept |
| banded matrix | none | ленточная матрица | new concept |
| Toeplitz matrix | none | тёплицева матрица | new concept |
| locally connected layer | none | локально связный слой | new concept: adverb + adjective written apart |
