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
