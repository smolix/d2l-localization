# Results: Japanese, Convolutions for Images

Work files emitted 2026-11-26 at English revision `370f7b3a15d0` (d2l-neu main),
the same page and revision as the German test, with the Japanese style guide
before the native-speaker review (its open questions in `ja/style.md` were
still unanswered). Three units, 144 blocks.

## 2026-11-26: Claude models

Each agent ran once in a fresh session with the prompt in PROMPT.md, at medium
effort, all three at the same time.

| Agent | Errors | Warnings | Characters (× English) | Time |
|---|--:|--:|--:|--:|
| `fable` (Claude Fable 5.2) | 0 | 9 | 16,673 (0.67) | 6.7 min |
| `opus` (Claude Opus 5.5) | 0 | 8 | 16,558 (0.67) | 9.1 min |
| `sonnet` (Claude Sonnet 5.5) | 0 | 6 | 16,687 (0.68) | 7.2 min |

- Every warning is a false match, and all three agents left them in place
  and named them in NOTES. The acronyms MLP, CNN, SGD stay in English as
  style §3.2 asks, but the glossary check expects the long Japanese form; "Tensor
  Core" is matched as *tensor*; "standard normal distribution" is 標準正規分布
  under style §7.1, not the glossary's ガウス分布. Fable also has "sparsity
  pattern". These point at the checker (acronym handling) and the glossary
  entry for *normal distribution*, not at the translations.
- The agents shared a scratch directory and two of them overwrote each
  other's helper script; the answers were checked afterwards and are
  independent (pairwise character similarity about 0.9 for every pair, with
  the identical long blocks all formula-heavy slide bullets). PROMPT.md now
  asks each agent to keep its own work directory.

**Verdict:** pending a native reader.

## 2026-11-26: Astra, Sol and Terra

Each model ran once in an independent session at medium effort, using the same
prompt and pinned work files. All three completed 144 blocks. Counts are from
`l10n compare` with the checker at `50e89c2`.

| Agent | Model | Errors | Warnings | Characters (× English) |
|---|---|--:|--:|--:|
| [Astra](answers-astra/) | `gpt-6.2-astra` | 0 | 7 | 17,039 (0.69) |
| [Sol](answers-sol/) | `gpt-6.2-sol` | 0 | 6 | 17,053 (0.69) |
| [Terra](answers-terra/) | `gpt-6.1-terra` | 0 | 8 | 15,950 (0.65) |

- The answers document every remaining warning. Most arise from acronyms,
  Tensor Core, or the style guide's convention for 標準正規分布. Sol rendered
  both SGD labels as 勾配降下法 because the example computes its gradient over
  all 42 outputs. Astra's other warning matches *sparsity* within *sparsity
  pattern*.
- In the side-by-side sample, Astra and Sol both make the kernel weights the
  subject of convergence in the initial-loss slide kicker (`7e4ef44b`).
  Terra's version carries through the original ambiguity. Astra and Sol also
  make explicit that each layer adds two pixels to each side length of the
  receptive field (`b302e3ef`).

The native-reader verdict is pending.
