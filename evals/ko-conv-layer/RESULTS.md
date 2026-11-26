# Results: 한국어, Convolutions for Images

Work files emitted 2026-11-26 at English revision `370f7b3a15d0` (d2l-neu main), the same page and revision as the German and Japanese tests. The language had no style guide yet, so translators followed TRANSLATING.md §8 and recorded their conventions in DECISIONS. Three units, 144 blocks.

## 2026-11-26: Claude models

Each agent ran once in a fresh session with the prompt in PROMPT.md, at medium effort, alongside the other languages. Counts are from `bin/l10n compare` with the checker after the fixes these runs prompted (the agents themselves saw an earlier checker, with more false matches).

| Agent | Errors | Warnings | Words (× English) | Time |
|---|--:|--:|--:|--:|
| `fable` (Claude Fable 5.2) | 0 | 1 | 3,743 (0.88) | 8.8 min |
| `opus` (Claude Opus 5.5) | 0 | 0 | 3,706 (0.87) | 7.6 min |
| `sonnet` (Claude Sonnet 5.5) | 0 | 11 | 3,746 (0.88) | 2.6 min |

- The run exposed a checker bug: digits joined to Korean counters ("5개") were reported as missing numbers (fixed).
- Opus spelled out the acronyms and switched to 가우시안 분포 to satisfy warnings that were false matches at the time; Fable kept the acronyms and named the false matches.
- Fable flagged the proposed 순방향 패스 (forward pass) as less usual than 순전파.

**Verdict:** pending a native reader.


## 2026-11-26: Astra, Sol and Terra

The recorded translations use the fixed work files and medium effort,
each in an independent session.
All three completed 144 blocks. Counts use the checker at `4fafcb0`.

| Agent | Model | Errors | Warnings | Words (× English) |
|---|---|--:|--:|--:|
| [astra](answers-astra/) | `gpt-6.2-astra` | 0 | 3 | 3,937 (0.93) |
| [sol](answers-sol/) | `gpt-6.2-sol` | 0 | 4 | 3,880 (0.91) |
| [terra](answers-terra/) | `gpt-6.1-terra` | 0 | 0 | 3,691 (0.87) |

The native-reader verdict is pending.


## 2026-11-26: Gemini 4 (Carbon, Neon and Hydrogen)

Each model ran once in an independent session at `medium` effort with the prompt
in `PROMPT.md` and the same three pinned work files. All three completed all 144
blocks with zero errors and zero warnings. Counts use the checker at `1fa7ded`.

| Agent | Model | Errors | Warnings | Words (× English) | Time |
|---|---|--:|--:|--:|--:|
| [carbon](answers-carbon/) | `gemini-4-carbon-medium` | 0 | 0 | 3,785 (0.89) | 14.9 min |
| [neon](answers-neon/) | `gemini-4-neon-medium` | 0 | 0 | 3,797 (0.89) | 4.4 min |
| [hydrogen](answers-hydrogen/) | `gemini-4-hydrogen-medium` | 0 | 0 | 3,740 (0.88) | 3.1 min |

- All three preserve Latin acronyms (`MLP`, `CNN`, `SGD`), use polite formal
  Korean (`-합니다` / `-입니다`) in prose and concise noun phrases in headings and
  slides, and document Korean SOV math/code token reorderings in `NOTES`.

The native-reader verdict is pending.
