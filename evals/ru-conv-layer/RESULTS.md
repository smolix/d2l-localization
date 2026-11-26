# Results: Русский, Convolutions for Images

Work files emitted 2026-11-26 at English revision `370f7b3a15d0` (d2l-neu main), the same page and revision as the other language tests. The Russian glossary was new: drafted by Opus 5.5 and reviewed by Fable 5.2 the same day (`glossary/notes/ru-review.md`); every entry is still a proposal. The language had no style guide yet, so translators followed TRANSLATING.md §8 and recorded their conventions in DECISIONS. Three units, 144 blocks.

## 2026-11-26: Claude models

Each agent ran once in a fresh session with the prompt in PROMPT.md, at medium effort. Counts are from `bin/l10n compare` with the current checker.

| Agent | Errors | Warnings | Words (× English) | Time |
|---|--:|--:|--:|--:|
| `fable` (Claude Fable 5.2) | 0 | 0 | 4,187 (0.99) | 6.2 min |
| `opus` (Claude Opus 5.5) | 0 | 0 | 4,102 (0.97) | 6.0 min |
| `sonnet` (Claude Sonnet 5.5) | 0 | 2 | 4,109 (0.97) | 2.2 min |

- Fable moved math tokens into Russian word order in four blocks («матрицы $\mathbf{X}$ размера …») and named each in NOTES. It also flagged two English slide/prose mismatches (loss "$12.0$" vs "near 12"; corner pixels vs edge inputs), which were fixed in English after this revision.
- Opus reworded one sentence to restore the English math order and clear a "math tokens reordered" warning; it judged the warning real, not a false match.
- Sonnet's two warnings are "numbers not found: 1" where it wrote the number as a word («одной»), a correct rendering.
- Opus assumed the build prints cross-references with a nominative prefix ("Раздел 7.1") and so never put a case-governing preposition directly before a bare `:numref:`. A Russian style guide should settle how references are phrased.

**Verdict:** pending a native reader.


## 2026-11-26: Gemini 4 (Carbon, Neon and Hydrogen)

Each model ran once in an independent session at `medium` effort with the prompt
in `PROMPT.md` and the same three pinned work files. All three completed all 144
blocks with zero errors and zero warnings. Counts are from `bin/l10n compare`
with the current checker.

| Agent | Model | Errors | Warnings | Words (× English) | Time |
|---|---|--:|--:|--:|--:|
| [carbon](answers-carbon/) | `gemini-4-carbon-medium` | 0 | 0 | 4,222 (0.99) | 16.8 min |
| [neon](answers-neon/) | `gemini-4-neon-medium` | 0 | 0 | 4,105 (0.97) | 4.8 min |
| [hydrogen](answers-hydrogen/) | `gemini-4-hydrogen-medium` | 0 | 0 | 4,048 (0.95) | 2.7 min |

- Like `fable`, all three place Russian dimension modifiers post-nominally
  («матрицы $\mathbf{X}$ размера $n_\textrm{h} \times n_\textrm{w}$», «выхода
  $\mathbf{Y}$ размера $2 \times 2$», «матрицу $\mathbf{W}$ размера
  $4 \times 16$») and record each math-token reordering in `NOTES` rather than
  rewording to keep English token order (`TRANSLATING.md` §9).
- In `33bb377d`, `carbon` writes out *Теоретическое рецептивное поле* while
  `neon` and `hydrogen` abbreviate *Теоретическое РП*; `carbon` and `neon` render
  geometric *Bounding box* as *ограничивающая рамка*, whereas `hydrogen` uses
  *Область* (which still passes with 0 warnings).

The native-reader verdict is pending.


## 2026-11-26: Astra, Sol and Terra

The recorded translations use the fixed work files and medium effort,
each in an independent session.
All three completed 144 blocks. Counts use the checker at `c95293c`.

| Agent | Model | Errors | Warnings | Words (× English) |
|---|---|--:|--:|--:|
| [astra](answers-astra/) | `gpt-6.2-astra` | 0 | 2 | 4,401 (1.04) |
| [sol](answers-sol/) | `gpt-6.2-sol` | 0 | 1 | 4,313 (1.02) |
| [terra](answers-terra/) | `gpt-6.1-terra` | 0 | 0 | 4,074 (0.96) |

The native-reader verdict is pending.
