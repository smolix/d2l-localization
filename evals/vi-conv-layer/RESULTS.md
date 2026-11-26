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


## 2026-11-26: Astra, Sol and Terra

The recorded translations use the fixed work files and medium effort,
each in an independent session.
All three completed 144 blocks. Counts use the checker at `4fafcb0`.

| Agent | Model | Errors | Warnings | Words (× English) |
|---|---|--:|--:|--:|
| [astra](answers-astra/) | `gpt-6.2-astra` | 0 | 6 | 6,360 (1.50) |
| [sol](answers-sol/) | `gpt-6.2-sol` | 0 | 10 | 6,078 (1.43) |
| [terra](answers-terra/) | `gpt-6.1-terra` | 0 | 1 | 5,711 (1.35) |

The native-reader verdict is pending.


## 2026-11-26: Gemini 4 (Carbon, Neon and Hydrogen)

Each model ran once in an independent session at `medium` effort with the prompt
in `PROMPT.md` and the same three pinned work files. All three completed all 144
blocks with zero errors. Counts use the checker at `1fa7ded`.

| Agent | Model | Errors | Warnings | Words (× English) | Time |
|---|---|--:|--:|--:|--:|
| [carbon](answers-carbon/) | `gemini-4-carbon-medium` | 0 | 1 | 6,045 (1.42) | 12.9 min |
| [neon](answers-neon/) | `gemini-4-neon-medium` | 0 | 3 | 6,079 (1.43) | 4.8 min |
| [hydrogen](answers-hydrogen/) | `gemini-4-hydrogen-medium` | 0 | 7 | 5,944 (1.40) | 2.8 min |

- All three share the false-match warning on `eed4e2a0` (*"the bell shape"* →
  *hình dạng quả chuông*, geometric curve shape rather than tensor shape *kích
  thước*). `neon` (2 blocks) and `hydrogen` (4 blocks) also use adjectival *vô
  hướng* in *scalar bias* / *scalar loss* where the glossary expects the noun
  *số vô hướng*, and `hydrogen` translates geometric *Bounding box* (`33bb377d`)
  as *Hộp giới hạn* rather than *khung chứa*; all are documented in `NOTES`.

The native-reader verdict is pending.
