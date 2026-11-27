# Notes on chapter_attention/queries-keys-values.md#u01 (gemini gemini-4-carbon (medium), 2026-11-27 05:15 UTC)

- 4ba4e9fd, 884264f4: in the English source, the inline math `$a(\mathbf{q}, \mathbf{k}) = -\|\mathbf{q} - \mathbf{k}\|^2$` wraps across a newline before `- \mathbf{k}\|^2$`, causing the line-based block splitter to split the equation across paragraph `4ba4e9fd` and list_item `884264f4`. Preserved the exact split across both blocks so that per-block checks pass and reassembly produces valid inline math.
