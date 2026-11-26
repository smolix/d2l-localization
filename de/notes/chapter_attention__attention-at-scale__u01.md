# Notes on chapter_attention/attention-at-scale.md#u01 (Codex Astra (medium), 2026-11-26 19:15 UTC)

- 9cb0e42f: Der Vergleich mit dem „entire transformer block“ meint den übrigen Rechenaufwand des Blocks einschließlich Feedforward-Netz, wie die Schwelle bei etwa 6d zeigt.
- 837d27ce: Laut angegebener Schranke vervierfacht sich der Verkehr pro Kopf. Bei fester Gesamtbreite halbiert sich die Kopfzahl, sodass sich der Gesamtverkehr verdoppelt. „2× per head“ wurde entsprechend berichtigt.
- 38412659: Die lineare Schranke beschreibt den gespeicherten Zustand für den Rückwärtsdurchlauf. Die Neuberechnung selbst hat keinen linearen Rechenaufwand.
- c0082157, 48391ea4, 764ff8bd, d72194f6, 2713da21, 65dda751, 2eb60036, 68692f2e, 4a8689ba, f69776af, fbf0f657, 4bd9411a, 5d907989, b50ce8b4: „kernel“ bezeichnet ausführbaren GPU-Code. Der Glossartreffer „Faltungskern“ hat eine andere Bedeutung.
- c0082157, 48391ea4, 9e4ecd91, 0bf51322, 68620334, f28faad3, aa244c06, b7e59a4f: „online softmax“ bezeichnet den Algorithmus zur Berechnung der Softmax-Funktion. Die zusammengesetzten Bezeichnungen richten sich nach dem jeweiligen Bezug.
- edae7f4e, fad3f39b, bcbffd69, b7e59a4f: „softmax“ bestimmt den Nenner oder die Gewichte näher und steht entsprechend als Erstglied eines Kompositums.
- 6042461a: „overflows“ ist hier ein Verb („läuft … über“).
