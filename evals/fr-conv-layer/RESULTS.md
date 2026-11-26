# Results: Français, Convolutions for Images

Work files emitted 2026-11-26 at English revision `370f7b3a15d0` (d2l-neu main), the same page and revision as the German and Japanese tests. The language had no style guide yet, so translators followed TRANSLATING.md §8 and recorded their conventions in DECISIONS. Three units, 144 blocks.

## 2026-11-26: Claude models

Each agent ran once in a fresh session with the prompt in PROMPT.md, at medium effort, alongside the other languages. Counts are from `bin/l10n compare` with the checker after the fixes these runs prompted (the agents themselves saw an earlier checker, with more false matches).

| Agent | Errors | Warnings | Words (× English) | Time |
|---|--:|--:|--:|--:|
| `fable` (Claude Fable 5.2) | 0 | 0 | 5,018 (1.18) | 6.4 min |
| `opus` (Claude Opus 5.5) | 0 | 0 | 4,943 (1.16) | 5.8 min |
| `sonnet` (Claude Sonnet 5.5) | 0 | 0 | 4,943 (1.16) | 2.6 min |

- Fable noted that French needs no-break spaces before : ; ? ! and inside « »; `fr` has no `typography` setting yet.
- Sonnet's warnings were inflected forms (« champs récepteurs », « matricielle ») the check did not accept (fixed).

**Verdict:** pending a native reader.


## 2026-11-26: Astra, Sol and Terra

The recorded translations use the fixed work files and medium effort,
each in an independent session.
All three completed 144 blocks. Counts use the checker at `4fafcb0`.

| Agent | Model | Errors | Warnings | Words (× English) |
|---|---|--:|--:|--:|
| [astra](answers-astra/) | `gpt-6.2-astra` | 0 | 3 | 5,367 (1.26) |
| [sol](answers-sol/) | `gpt-6.2-sol` | 0 | 5 | 5,136 (1.21) |
| [terra](answers-terra/) | `gpt-6.1-terra` | 0 | 0 | 4,898 (1.15) |

The native-reader verdict is pending.


## 2026-11-26: Gemini 4 (Carbon, Neon and Hydrogen)

Each model ran once in an independent session at `medium` effort with the prompt
in `PROMPT.md` and the same three pinned work files. All three completed all 144
blocks with zero errors. Counts use the checker at `1fa7ded`.

| Agent | Model | Errors | Warnings | Words (× English) | Time |
|---|---|--:|--:|--:|--:|
| [carbon](answers-carbon/) | `gemini-4-carbon-medium` | 0 | 0 | 4,945 (1.16) | 13.3 min |
| [neon](answers-neon/) | `gemini-4-neon-medium` | 0 | 0 | 4,863 (1.15) | 4.5 min |
| [hydrogen](answers-hydrogen/) | `gemini-4-hydrogen-medium` | 0 | 1 | 4,899 (1.15) | 2.9 min |

- `carbon` and `neon` reach 0 warnings without forcing English token order;
  `hydrogen`'s single warning (`e3aa1086`) comes from rendering `1` idiomatically
  as *d'une seule*, documented in `NOTES`.

The native-reader verdict is pending.
