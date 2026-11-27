# Notes on chapter_appendix-tools-for-deep-learning/model-serving.md#u01 (Codex Astra (medium), 2026-11-27 00:07 UTC)

- 99ef6cb8: Das Zielmodell bewertet γ+1 Positionen parallel. Nur γ Token stammen vom Entwurfsmodell, daher „Positionen“.
- 469cf17d, b040a898: „kernels“ bezeichnet hier GPU-Rechenkerne (Kernel), keine Faltungskerne.
- 24d8be3d: „1 single-occupancy tail step“ ist als „Nur im letzten Schritt ist ein einziger Platz belegt“ wiedergegeben.
- e857af46: „positional bias“ bezeichnet hier die bevorzugte Platzierung relevanter Dokumente, keinen additiven Bias und keine Schätzerverzerrung.
