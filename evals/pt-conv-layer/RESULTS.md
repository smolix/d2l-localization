# Results: Português, Convolutions for Images

Work files emitted 2026-11-26 at English revision `370f7b3a15d0` (d2l-neu main), the same page and revision as the German and Japanese tests. The language had no style guide yet, so translators followed TRANSLATING.md §8 and recorded their conventions in DECISIONS. Three units, 144 blocks.

## 2026-11-26: Claude models

Each agent ran once in a fresh session with the prompt in PROMPT.md, at medium effort, alongside the other languages. Counts are from `bin/l10n compare` with the checker after the fixes these runs prompted (the agents themselves saw an earlier checker, with more false matches).

| Agent | Errors | Warnings | Words (× English) | Time |
|---|--:|--:|--:|--:|
| `fable` (Claude Fable 5.2) | 0 | 0 | 4,914 (1.16) | 6.2 min |
| `opus` (Claude Opus 5.5) | 0 | 0 | 4,871 (1.15) | 4.6 min |
| `sonnet` (Claude Sonnet 5.5) | 0 | 0 | 4,807 (1.13) | 2.2 min |

- Sonnet reworded two sentences to keep math tokens in English order (against TRANSLATING.md §9).

**Verdict:** pending a native reader.
