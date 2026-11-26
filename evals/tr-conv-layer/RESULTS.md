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


## 2026-11-26: Astra, Sol and Terra

The recorded translations use the fixed work files and medium effort,
each in an independent session.
All three completed 144 blocks. Counts use the checker at `4fafcb0`.

| Agent | Model | Errors | Warnings | Words (× English) |
|---|---|--:|--:|--:|
| [astra](answers-astra/) | `gpt-6.2-astra` | 0 | 4 | 4,328 (1.02) |
| [sol](answers-sol/) | `gpt-6.2-sol` | 0 | 2 | 4,099 (0.97) |
| [terra](answers-terra/) | `gpt-6.1-terra` | 0 | 0 | 3,922 (0.92) |

The native-reader verdict is pending.


## 2026-11-26: Gemini 4 (Carbon, Neon and Hydrogen)

Each model ran once in an independent session at `medium` effort with the prompt
in `PROMPT.md` and the same three pinned work files. All three completed all 144
blocks with zero errors and zero warnings. Counts use the checker at `1fa7ded`.

| Agent | Model | Errors | Warnings | Words (× English) | Time |
|---|---|--:|--:|--:|--:|
| [carbon](answers-carbon/) | `gemini-4-carbon-medium` | 0 | 0 | 4,092 (0.96) | 15.2 min |
| [neon](answers-neon/) | `gemini-4-neon-medium` | 0 | 0 | 4,008 (0.94) | 3.9 min |
| [hydrogen](answers-hydrogen/) | `gemini-4-hydrogen-medium` | 0 | 0 | 3,940 (0.93) | 3.5 min |

- All three avoid line-leading `<n>. Bölüm` ordered-list collisions, preserve
  Turkish case suffixes after inline math/code, and record SOV math/code token
  reorderings in `NOTES`.

The native-reader verdict is pending.
