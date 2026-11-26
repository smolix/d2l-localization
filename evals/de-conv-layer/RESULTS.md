# Results: German, Convolutions for Images

Work files emitted 2026-11-26 at English revision `370f7b3a15d0` (d2l-neu main),
with the German style guide after the author's rulings and the corrected
glossary. Three units, 144 blocks, 4,246 English words.

## 2026-11-26: Claude models

Each agent ran once in a fresh session with the prompt in PROMPT.md, at medium
effort. Counts are from `bin/l10n compare` with the checker of d2l-localization
commit `83260ff`.

| Agent | Errors | Warnings | Words (× English) | Time |
|---|--:|--:|--:|--:|
| `fable` (Claude Fable 5.2) | 0 | 1 | 4,336 (1.02) | 6.7 min |
| `opus` (Claude Opus 5.5) | 0 | 1 | 4,389 (1.03) | 8.2 min |
| `sonnet` (Claude Sonnet 5.5) | 0 | 31 | 4,527 (1.07) | 5.3 min |

- The remaining warning of `fable` and `opus` is the same false match: an
  English book title in a link.
- `sonnet`'s warnings are real deviations from the glossary, mostly plain *Kern*
  for *Faltungskern*.
- `opus` reworded a slide title to silence a false warning. That led to the rule
  in TRANSLATING.md §9: never change a correct translation to silence a warning.
- `opus` found an error in the English: the slide title "Why Initial Loss Is 12
  and Converges to [1, -1]" means the kernel, not the loss.

**Verdict (Alex Smola, native speaker):** Fable 5.2 writes the best German of the
three. It is the translator for German (`translator` in `languages.yml`).

## 2026-11-26: Astra, Sol and Terra

Each model ran once in a fresh session at medium effort with the same prompt and
pinned work files. All three completed 144 blocks. Counts are from
`l10n compare` against the pinned English revision with the checker at
`8f8980e`, before any German book translations were installed.

| Agent | Model | Errors | Warnings | Words (× English) |
|---|---|--:|--:|--:|
| [Astra](answers-astra/) | `gpt-6.2-astra` | 0 | 1 | 4,488 (1.06) |
| [Sol](answers-sol/) | `gpt-6.2-sol` | 0 | 1 | 4,500 (1.06) |
| [Terra](answers-terra/) | `gpt-6.1-terra` | 0 | 9 | 4,196 (0.99) |

- All three have the same false match in the linked English book title in
  `6ab5b67c`. Terra also uses four compounds with *Kern*, writes a slide's
  `4` as *vier*, and reorders math tokens in three blocks. The formulas in
  those blocks are intact.
- Astra and Sol both break up dense explanations effectively. In the *im2col*
  derivation (`5bdffcd7`), Astra makes the two vectors in the dot product
  explicit. Sol spells out that libraries use this algorithm alongside other
  methods, preserving a nuance in the source.
- The checker misses some issues visible in a side-by-side reading. Sol drops
  the colon that introduces the display equation in `ae7e906d`. In the summary
  (`92688a0c`), Terra's wording obscures that the layer **adds** the bias, and
  its last exercise contains the ungrammatical *die automatisches
  Differenzieren* (`6ab5b67c`).

The side-by-side sample suggests reading Astra and Sol against Fable first. It
covered the introduction, math-heavy explanations, captions, a slide and its
list, and exercises. Native-speaker verdict pending.

## 2026-11-26: Gemini 4 (Carbon, Neon and Hydrogen)

Each model ran once in a fresh `agy` CLI session at `medium` effort (`--effort medium`)
with the same prompt in `PROMPT.md` and the same three pinned work files. All
three completed all 144 blocks with zero errors. Counts are from `bin/l10n compare`
against English revision `370f7b3a15d0`.

| Agent | Model | Errors | Warnings | Words (× English) | Time |
|---|---|--:|--:|--:|--:|
| [Carbon](answers-carbon/) | `gemini-4-carbon-medium` | 0 | 4 | 4,218 (0.99) | 16.1 min |
| [Neon](answers-neon/) | `gemini-4-neon-medium` | 0 | 1 | 4,187 (0.99) | 6.4 min |
| [Hydrogen](answers-hydrogen/) | `gemini-4-hydrogen-medium` | 0 | 6 | 4,134 (0.97) | 4.0 min |

- All three share the same false-match warning on the untranslated English book
  title *Neural Networks and Deep Learning* in `6ab5b67c`, use proper German
  quotation marks („…“) and spaced en dashes (` – `, zero em dashes), keep the
  colon introducing display equations (`ae7e906d`), inflect *ohne automatisches
  Differenzieren* (`6ab5b67c`) accurately, and introduce zero duplicate
  `(engl. …)` glosses across the three units.
- **Warning counts vs. compound style (`de/style.md` §1):** `neon` reaches 1
  warning by forcing `Faltungskern` into four-part compounds
  (*Faltungskernelemente*, *Faltungskerngewichte*, *Faltungskernvektor*,
  *Weichzeichner-Faltungskern*), which violates `de/style.md` §1 (*no compound
  of more than three parts*). `carbon` instead follows both `de/style.md` §1 and
  `TRANSLATING.md` §9 (*never change a correct translation only to silence a
  warning*), writing *Kerngewichte* (`55763603`), *Kerneinträge* (`f0e86ee3`),
  and *Kernvektor* (`250d6877`) by analogy with glossary *Kerngröße* and
  documenting all three in `NOTES`.
- `hydrogen` also uses short *Kern-* compounds (5 warnings plus `6ab5b67c`),
  documented in `NOTES`, and is the fastest of all tested models (4.0 min), but
  slips into plain *den Kern* / *Kerne* in two standalone clauses of the summary
  (`92688a0c`), shifts from authorial *wir* to *man* in the *im2col* derivation
  (`5bdffcd7`), and leaves informal *Backprop* and filler *einfach* on slide
  `f0e86ee3`.
- On the ambiguous English slide kicker `7e4ef44b` (*"Why Initial Loss Is 12 and
  Converges to [1, -1]"*), `carbon` recasts it as a clean noun phrase
  (*„Anfangsverlust 12 und Konvergenz gegen [1, -1]“*) so the scalar loss is not
  the grammatical subject of *konvergiert*, whereas `neon` and `hydrogen` carry
  over the subject ellipsis literally (*„Warum der Anfangsverlust 12 beträgt und
  gegen [1, -1] konvergiert“*). `neon` also glosses plural `(engl. kernels)` in
  `852f5868` instead of singular `(engl. kernel)`.

Among the three Gemini models, `carbon` writes the cleanest, most idiomatic
textbook German and adheres most faithfully to `de/style.md` (with `neon` a fast
runner-up aside from its four-part `Faltungskern-` compounds). Native-speaker
verdict pending.

## Decision

**Translator for German (Alex Smola, 2026-11-26):** Codex GPT-6.2 Astra (`translator` in `languages.yml`). Among the Claude models Fable 5.2 was best.
