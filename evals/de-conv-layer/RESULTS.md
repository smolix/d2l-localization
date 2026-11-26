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
