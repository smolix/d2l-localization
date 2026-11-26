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


## 2026-11-26: Astra, Sol and Terra

The recorded translations use the fixed work files and medium effort,
each in an independent session.
All three completed 144 blocks. Counts use the checker at `4fafcb0`.

| Agent | Model | Errors | Warnings | Words (× English) |
|---|---|--:|--:|--:|
| [astra](answers-astra/) | `gpt-6.2-astra` | 0 | 0 | 5,328 (1.25) |
| [sol](answers-sol/) | `gpt-6.2-sol` | 0 | 1 | 5,179 (1.22) |
| [terra](answers-terra/) | `gpt-6.1-terra` | 0 | 1 | 4,938 (1.16) |

The native-reader verdict is pending.


## 2026-11-26: Gemini 4 (Carbon, Neon and Hydrogen)

Each model ran once in an independent session at `medium` effort with the prompt
in `PROMPT.md` and the same three pinned work files. All three completed all 144
blocks with zero errors. Counts use the checker at `1fa7ded`.

| Agent | Model | Errors | Warnings | Words (× English) | Time |
|---|---|--:|--:|--:|--:|
| [carbon](answers-carbon/) | `gemini-4-carbon-medium` | 0 | 0 | 5,031 (1.18) | 13.4 min |
| [neon](answers-neon/) | `gemini-4-neon-medium` | 0 | 1 | 4,945 (1.16) | 4.2 min |
| [hydrogen](answers-hydrogen/) | `gemini-4-hydrogen-medium` | 0 | 0 | 4,936 (1.16) | 3.5 min |

- `neon`'s single warning (`bef130ba`) is a false match where English uses
  *slices* as a finite verb (*extrae*) rather than the noun *rebanada*, recorded
  in `NOTES`. All three document post-nominal Spanish dimension reorderings
  (`matriz de entrada $\mathbf{X}$ de $n_\textrm{h} \times n_\textrm{w}$`) in
  `NOTES`.

The native-reader verdict is pending.
