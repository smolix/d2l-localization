# Notes on chapter_optimization/practice.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27 05:15 UTC)

- 39b19ec8, b4c5e360: `$540$` and `$3.3$` are protected inline-math tokens, so `-billion-parameter` and `-million-parameter` are rendered as `パラメータ数 $540$B` and `パラメータ数 $3.3$M` (consistent with `$3.3\text{M}$` and `405B`/`671B` elsewhere on the page) to preserve the math tokens verbatim.
- 6ff92b20: false glossary match on `time steps` (`タイムステップ`); after stripping inline math `$\sim 1/(\eta\lambda_{\text{curv}})$`, `"autocorrelation time ... steps"` becomes `"time   steps"`, whereas the source means autocorrelation time measured in optimization steps (`自己相関時間` / `ステップ`).
