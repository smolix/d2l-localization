# Results: 中文, Convolutions for Images

Work files emitted 2026-11-26 at English revision `370f7b3a15d0` (d2l-neu main), the same page and revision as the German and Japanese tests. The language had no style guide yet, so translators followed TRANSLATING.md §8 and recorded their conventions in DECISIONS. Three units, 144 blocks.

## 2026-11-26: Claude models

Each agent ran once in a fresh session with the prompt in PROMPT.md, at medium effort, alongside the other languages. Counts are from `bin/l10n compare` with the checker after the fixes these runs prompted (the agents themselves saw an earlier checker, with more false matches).

| Agent | Errors | Warnings | Characters (× English) | Time |
|---|--:|--:|--:|--:|
| `fable` (Claude Fable 5.2) | 0 | 2 | 13,826 (0.56) | 6.1 min |
| `opus` (Claude Opus 5.5) | 0 | 2 | 13,741 (0.56) | 4.5 min |
| `sonnet` (Claude Sonnet 5.5) | 0 | 1 | 13,724 (0.56) | 3.2 min |

- Opus reworded sentences to keep math tokens in English order, which TRANSLATING.md §9 forbids when the translation was correct.
- Fable and Opus wrote 标准高斯分布 for "standard normal distribution" because the glossary then treated *normal distribution* as an alias of *Gaussian distribution*; both noted that 标准正态分布 is usual. The glossary has since been fixed.

**Verdict:** pending a native reader.


## 2026-11-26: Astra, Sol and Terra

The recorded translations use the fixed work files and medium effort,
each in an independent session.
All three completed 144 blocks. Counts use the checker at `1fa7ded`.

| Agent | Model | Errors | Warnings | Characters (× English) |
|---|---|--:|--:|--:|
| [astra](answers-astra/) | `gpt-6.2-astra` | 0 | 2 | 14,073 (0.57) |
| [sol](answers-sol/) | `gpt-6.2-sol` | 0 | 4 | 13,783 (0.56) |
| [terra](answers-terra/) | `gpt-6.1-terra` | 0 | 0 | 13,452 (0.54) |

The native-reader verdict is pending.
