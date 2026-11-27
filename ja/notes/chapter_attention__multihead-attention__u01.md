# Notes on chapter_attention/multihead-attention.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27 05:15 UTC)

- 1a0dcb62: glossary false match on verb "trace" (step-by-step walkthrough 「たどる」, not matrix trace 「トレース」; see `ja/style.md` §7.1).
- 3f14d969: math tokens `$i$` and `$\mathbf{X} \in \mathbb{R}^{n \times d}$` reordered for natural Japanese modifier order（「$\mathbf{X} \in \mathbb{R}^{n \times d}$ の第 $i$ 行」）.
- e3e04557: glossary false match on verb "trace" (step-by-step walkthrough 「たどる」, not matrix trace 「トレース」; see `ja/style.md` §7.1).
- ece735db: the English source split inline math `$1 - \tfrac{\alpha_1^2}{\alpha_1^2\n+ \alpha_2^2} = \dots$` across blocks `ece735db` and `50fc153a` because `+` fell at column 0; preserved the exact split across `ece735db` and `50fc153a` so the reassembled Markdown renders valid LaTeX.
