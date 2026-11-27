# Notes on chapter_attention/what-attention-computes.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27 05:15 UTC)

- 8d20f638, f21f8c94: In the English source, line-wrapping `$j = i + 1\n- L$.` caused `- L$.` to be segmented as a separate `list_item` block (`f21f8c94`). Kept the inline math split `$j = i + 1` / `- L$` across the two blocks so that `reconstruct()` yields valid inline math `$j = i + 1\n- L$` in a single sentence while satisfying per-block token and prefix checks.
