# Notes on chapter_convolutional-modern/efficient-convnets.md#u02 (gemini gemini-4-carbon (medium), 2026-11-27 04:27 UTC)

- 2359a2f7, eefd35e9: in the English Markdown source, a line wrap before `2)` in `(mean 1, standard deviation\n2)` caused the segmenter to split the parenthetical across a paragraph (`2359a2f7`) and a `list_item` (`eefd35e9`) with required prefix `2)`. Kept half-width `(平均 1、標準偏差` at the end of `2359a2f7` and `2) ` at the start of `eefd35e9` so that `reconstruct()` re-joins them seamlessly.
