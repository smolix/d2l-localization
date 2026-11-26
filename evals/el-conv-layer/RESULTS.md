# Results: Ελληνικά, Convolutions for Images

Work files emitted 2026-11-26 at English revision `370f7b3a15d0` (d2l-neu main), the same page and revision as the other language tests. The Greek glossary was new: drafted by Opus 5.5 and reviewed by Fable 5.2 the same day (`glossary/notes/el-review.md`); every entry is still a proposal. The language had no style guide yet, so translators followed TRANSLATING.md §8 and recorded their conventions in DECISIONS. Three units, 144 blocks.

## 2026-11-26: Claude models

Each agent ran once in a fresh session with the prompt in PROMPT.md, at medium effort. Counts are from `bin/l10n compare` with the current checker.

| Agent | Errors | Warnings | Words (× English) | Time |
|---|--:|--:|--:|--:|
| `fable` (Claude Fable 5.2) | 0 | 0 | 4,755 (1.12) | 8.5 min |
| `opus` (Claude Opus 5.5) | 0 | 0 | 4,654 (1.10) | 10.5 min |
| `sonnet` (Claude Sonnet 5.5) | 0 | 0 | 4,714 (1.11) | 3.1 min |

- Fable and Opus both put a gendered article before `:numref:` to match the label Quarto prints in Greek (Ενότητα and Εξίσωση feminine, Σχήμα and Κεφάλαιο neuter). If `el/strings.yml` later changes a label (e.g. Σχήμα → Εικόνα), those articles must follow; a style guide should fix the labels first.
- Opus and Sonnet each reworded one sentence to keep the English math order and clear a "math tokens reordered" warning.
- Opus used the glossary's alternative «περιβάλλον πλαίσιο» for "bounding box" where it means the rectangle around pixels, not the object-detection term, and said so in NOTES.
- Translators differ on "patch": Fable «τεμάχιο», Opus «περιοχή», Sonnet «ψηφίδα»; the glossary has no entry.
- Sonnet left the four "Discussions" link labels in English.
- Fable translated the evident meaning of the slide kicker that says the loss "converges to [1, -1]" (the kernel does), noting it; the English was fixed after this revision.

**Verdict:** pending a native reader.


## 2026-11-26: Gemini 4 (Carbon, Neon and Hydrogen)

Each model ran once in an independent session at `medium` effort with the prompt
in `PROMPT.md` and the same three pinned work files. All three completed all 144
blocks with zero errors and zero warnings. Counts are from `bin/l10n compare`
with the current checker.

| Agent | Model | Errors | Warnings | Words (× English) | Time |
|---|---|--:|--:|--:|--:|
| [carbon](answers-carbon/) | `gemini-4-carbon-medium` | 0 | 0 | 4,572 (1.08) | 16.3 min |
| [neon](answers-neon/) | `gemini-4-neon-medium` | 0 | 0 | 4,582 (1.08) | 4.8 min |
| [hydrogen](answers-hydrogen/) | `gemini-4-hydrogen-medium` | 0 | 0 | 4,504 (1.06) | 2.6 min |

- Unlike `opus` and `sonnet`, all three keep natural Greek post-nominal dimension
  order («πίνακα εισόδου $\mathbf{X}$ διαστάσεων
  $n_\textrm{h} \times n_\textrm{w}$») and record the resulting math-token
  reorderings in `NOTES` (`TRANSLATING.md` §9).
- Before `:numref:`, `carbon` and `neon` use a gendered Greek article matching
  the Quarto label («Στην `:numref:`»), whereas `hydrogen` inserts an explicit
  noun («Στην ενότητα `:numref:`», which would duplicate Quarto's printed prefix).

The native-reader verdict is pending.
