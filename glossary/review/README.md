# Glossary review (2026-11-26)

One Fable review per language against `concepts.jsonl` senses: `<locale>.md` is the verdict, `<locale>-corrections.jsonl` the proposed changes (`concept_id`, `current`, `proposed`, `new_status`, `severity`, `reason`, `sources`). Nothing is applied to `terms/` yet; each language's open choices need a native speaker first.

| Locale | Corrections | of 1,184 | wrong sense | calque/MT | non-standard | inconsistent | minor |
|---|--:|--:|--:|--:|--:|--:|--:|
| fr | 102 | 9% | 1 | 8 | 43 | 27 | 23 |
| es | 110 | 9% | 4 | 7 | 44 | 41 | 14 |
| zh-Hans | 116 | 10% | 10 | 29 | 16 | 34 | 27 |
| pt-BR | 124 | 10% | 2 | 4 | 32 | 50 | 36 |
| de | 130 | 11% | 11 | 11 | 23 | 51 | 34 |
| tr | 187 | 16% | 16 | 10 | 24 | 94 | 43 |
| ja | 188 | 16% | 14 | 25 | 43 | 76 | 30 |
| vi | 195 | 16% | 8 | 15 | 56 | 72 | 44 |
| ko | 210 | 18% | 9 | 28 | 37 | 89 | 47 |
| ar | 226 | 19% | 33 | 17 | 28 | 104 | 44 |
| hi | 499 | 42% | 81 | 11 | 179 | 191 | 37 |

Finding across languages: every `verified_d2l` tier taken from an earlier edition produced by Amazon Translate (`translator = aws`: ja, ko, tr, de, pt; vi `main`; the ar PDF reads as MT) is about one third wrong. Shared errors: *translation invariance* rendered as language translation, *GPU* expanded into a description, names (Adam, LSTM, ReLU) translated. Google's ML glossary is machine-translated in every language and is weak evidence. Future extraction from old editions must take the edition's body-text majority usage, never headings, and for vi must read the human chapters at d2l-vi commit `5a2e43ca27`, not `main`.
