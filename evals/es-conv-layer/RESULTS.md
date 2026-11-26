# Results: Español, Convolutions for Images

Work files emitted 2026-11-26 at English revision `370f7b3a15d0` (d2l-neu main), the same page and revision as the German and Japanese tests. The language had no style guide yet, so translators followed TRANSLATING.md §8 and recorded their conventions in DECISIONS. Three units, 144 blocks.

## 2026-11-26: Claude models

Each agent ran once in a fresh session with the prompt in PROMPT.md, at medium effort, alongside the other languages. Counts are from `bin/l10n compare` with the checker after the fixes these runs prompted (the agents themselves saw an earlier checker, with more false matches).

| Agent | Errors | Warnings | Words (× English) | Time |
|---|--:|--:|--:|--:|
| `fable` (Claude Fable 5.2) | 0 | 0 | 5,141 (1.21) | 6.6 min |
| `opus` (Claude Opus 5.5) | 0 | 1 | 5,038 (1.19) | 6.4 min |
| `sonnet` (Claude Sonnet 5.5) | 0 | 2 | 5,035 (1.19) | 2.4 min |

- Opus found that a line break inside a term («descenso de\ngradiente») hid it from the glossary check (fixed).

**Verdict:** pending a native reader.
