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


## 2026-11-26: Astra, Sol and Terra

Each model ran once in a fresh session at medium effort on the fixed work files.
All three completed 144 blocks. Counts use the checker at `4fafcb0`.

| Agent | Model | Errors | Warnings | Words (× English) |
|---|---|--:|--:|--:|
| [astra](answers-astra/) | `gpt-6.2-astra` | 0 | 5 | 4,402 (1.04) |
| [sol](answers-sol/) | `gpt-6.2-sol` | 0 | 0 | 4,328 (1.02) |
| [terra](answers-terra/) | `gpt-6.1-terra` | 0 | 0 | 3,912 (0.92) |

The native-reader verdict is pending.


## 2026-11-26: Gemini 4 (Carbon, Neon and Hydrogen)

Each model ran once in an independent session at `medium` effort with the prompt
in `PROMPT.md` and the same three pinned work files. All three completed all 144
blocks with zero errors. Counts use the checker at `1fa7ded`.

| Agent | Model | Errors | Warnings | Words (× English) | Time |
|---|---|--:|--:|--:|--:|
| [carbon](answers-carbon/) | `gemini-4-carbon-medium` | 0 | 0 | 4,243 (1.00) | 15.8 min |
| [neon](answers-neon/) | `gemini-4-neon-medium` | 0 | 4 | 4,092 (0.96) | 4.5 min |
| [hydrogen](answers-hydrogen/) | `gemini-4-hydrogen-medium` | 0 | 4 | 4,023 (0.95) | 2.8 min |

- `carbon` reaches 0 warnings across all 144 blocks while keeping natural Arabic
  post-nominal dimension order (documented in `NOTES`). The 4 warnings of `neon`
  and `hydrogen` are all `"numbers not found"` where Arabic grammar inflects
  small counts (`1`, `2`, `5`, `6`) as words or dual forms rather than digits,
  recorded in `NOTES`.

The native-reader verdict is pending.
