# Results: Türkçe, Convolutions for Images

Work files emitted 2026-11-26 at English revision `370f7b3a15d0` (d2l-neu main), the same page and revision as the German and Japanese tests. The language had no style guide yet, so translators followed TRANSLATING.md §8 and recorded their conventions in DECISIONS. Three units, 144 blocks.

## 2026-11-26: Claude models

Each agent ran once in a fresh session with the prompt in PROMPT.md, at medium effort, alongside the other languages. Counts are from `bin/l10n compare` with the checker after the fixes these runs prompted (the agents themselves saw an earlier checker, with more false matches).

| Agent | Errors | Warnings | Words (× English) | Time |
|---|--:|--:|--:|--:|
| `fable` (Claude Fable 5.2) | 0 | 1 | 4,085 (0.96) | 7.5 min |
| `opus` (Claude Opus 5.5) | 0 | 1 | 4,017 (0.95) | 6.3 min |
| `sonnet` (Claude Sonnet 5.5) | 0 | 0 | 4,003 (0.94) | 3.2 min |

- Opus fixed a real fault in its own draft: "6. Bölüm" at the start of a line would have become a list item.
- Sonnet's warnings were Turkish vowel drop (şekil → şekli), which the check did not accept (fixed).

**Verdict:** pending a native reader.
