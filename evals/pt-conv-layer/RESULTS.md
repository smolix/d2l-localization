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


## 2026-11-26: Astra, Sol and Terra

The recorded translations use the fixed work files and medium effort,
each in an independent session.
All three completed 144 blocks. Counts use the checker at `4fafcb0`.

| Agent | Model | Errors | Warnings | Words (× English) |
|---|---|--:|--:|--:|
| [astra](answers-astra/) | `gpt-6.2-astra` | 0 | 0 | 5,120 (1.21) |
| [sol](answers-sol/) | `gpt-6.2-sol` | 0 | 0 | 4,976 (1.17) |
| [terra](answers-terra/) | `gpt-6.1-terra` | 0 | 0 | 4,709 (1.11) |

The native-reader verdict is pending.


## 2026-11-26: Gemini 4 (Carbon, Neon and Hydrogen)

Each model ran once in an independent session at `medium` effort with the prompt
in `PROMPT.md` and the same three pinned work files. All three completed all 144
blocks with zero errors and zero warnings. Counts use the checker at `1fa7ded`.

| Agent | Model | Errors | Warnings | Words (× English) | Time |
|---|---|--:|--:|--:|--:|
| [carbon](answers-carbon/) | `gemini-4-carbon-medium` | 0 | 0 | 4,726 (1.11) | 8.2 min |
| [neon](answers-neon/) | `gemini-4-neon-medium` | 0 | 0 | 4,730 (1.11) | 4.3 min |
| [hydrogen](answers-hydrogen/) | `gemini-4-hydrogen-medium` | 0 | 0 | 4,687 (1.10) | 2.4 min |

- All three document Portuguese post-nominal dimension reorderings
  (*matriz de entrada $\mathbf{X}$ de $n_\textrm{h} \times n_\textrm{w}$*) in
  `NOTES` rather than forcing English token order.

The native-reader verdict is pending.
