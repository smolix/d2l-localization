# Results: العربية, Convolutions for Images

Work files emitted 2026-11-26 at English revision `370f7b3a15d0` (d2l-neu main), the same page and revision as the German and Japanese tests. The language had no style guide yet, so translators followed TRANSLATING.md §8 and recorded their conventions in DECISIONS. Three units, 144 blocks.

## 2026-11-26: Claude models

Each agent ran once in a fresh session with the prompt in PROMPT.md, at medium effort, alongside the other languages. Counts are from `bin/l10n compare` with the checker after the fixes these runs prompted (the agents themselves saw an earlier checker, with more false matches).

| Agent | Errors | Warnings | Words (× English) | Time |
|---|--:|--:|--:|--:|
| `fable` (Claude Fable 5.2) | 0 | 3 | 4,044 (0.95) | 7.1 min |
| `opus` (Claude Opus 5.5) | 0 | 1 | 4,117 (0.97) | 7.8 min |
| `sonnet` (Claude Sonnet 5.5) | 0 | 10 | 3,997 (0.94) | 2.7 min |

- Sonnet wrote several numbers as words, which the remaining "numbers not found" warnings report.
- Fable and Opus write the full نواة الالتفاف first, then the short النواة and the plurals أنوية / نوى; the glossary now accepts these forms.

**Verdict:** pending a native reader.
