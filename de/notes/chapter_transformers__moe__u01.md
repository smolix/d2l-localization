# Notes on chapter_transformers/moe.md#u01 (Codex Astra (medium), 2026-11-26 19:29 UTC)

- df01466a: Die Vorlage gibt Byte für Gewichte direkt als Parameteranzahl an. Für die tatsächliche Bytezahl ist zusätzlich die Anzahl der Byte pro Gewicht maßgeblich.
- 7c05ef14, 269efbe3: Bei gleichmäßiger Last ist der Hilfsverlust 1. Die Aussage über das Minimum setzt einen zusätzlichen Zusammenhang zwischen den Zuweisungsanteilen und den mittleren Wahrscheinlichkeiten voraus, wie in Übung 2 formuliert.
- 83ebf274: „kernels“ bezeichnet hier kompilierte GPU- und TPU-Kernel. Der Glossartreffer „Faltungskern“ betrifft eine andere Bedeutung.
