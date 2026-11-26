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
