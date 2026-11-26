# Notes on chapter_computational-performance/sharding.md#u02 (Codex Astra (medium), 2026-11-26 19:56 UTC)

- b42d90d7: The 2× ratio of HLO result size to sent bytes at k=2 applies to all-gather. For reduce-scatter the two sizes coincide, as the source's figures show.
- 1148a2a5, a083b8f3, 8f6ea900: „kernels“ bezeichnet ausführbaren GPU-Code für die NCCL-Kommunikation. Der Glossareintrag „Faltungskern“ gehört zur Faltung.
- 4c4420fa: Die Folie vergleicht die Übertragungszeit der 11.5 GB für den ganzen Schritt mit der Rechenzeit eines einzelnen Blocks. Der Vergleich ist wie im Original wiedergegeben.
