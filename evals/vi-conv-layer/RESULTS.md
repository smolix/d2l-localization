# Results: Tiếng Việt, Convolutions for Images

Work files emitted 2026-11-26 at English revision `370f7b3a15d0` (d2l-neu main), the same page and revision as the German and Japanese tests. The language had no style guide yet, so translators followed TRANSLATING.md §8 and recorded their conventions in DECISIONS. Three units, 144 blocks.

## 2026-11-26: Claude models

Each agent ran once in a fresh session with the prompt in PROMPT.md, at medium effort, alongside the other languages. Counts are from `bin/l10n compare` with the checker after the fixes these runs prompted (the agents themselves saw an earlier checker, with more false matches).

| Agent | Errors | Warnings | Words (× English) | Time |
|---|--:|--:|--:|--:|
| `fable` (Claude Fable 5.2) | 0 | 2 | 5,871 (1.38) | 4.1 min |
| `opus` (Claude Opus 5.5) | 0 | 2 | 5,872 (1.38) | 6.3 min |
| `sonnet` (Claude Sonnet 5.5) | 0 | 8 | 5,850 (1.38) | 2.5 min |

- Opus flagged an English slip: the slides say "SGD … in 9 steps" for plain gradient descent run for 10 steps.
- Sonnet used the adjective "vô hướng" for "scalar" where the glossary has the noun "số vô hướng".

**Verdict:** pending a native reader.
