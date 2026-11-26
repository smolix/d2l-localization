# Notes on chapter_transformers/vision-transformer.md#u01 (Codex Astra (medium), 2026-11-26 19:25 UTC)

- 6073c5ae: Eine Permutation der Patch-Token permutiert sowohl Zeilen als auch Spalten der Attention-Matrix. In der ersten Zeile werden die Gewichte umgeordnet. Invariant bleibt die Ausgabe für das Klassifikationstoken. Die Übersetzung präzisiert diese Aussage.
- dcd36d6f: Der Genauigkeitsabfall bezieht sich auf den Test mit gelernten Positionseinbettungen, wie im Haupttext beschrieben.
- 04007229: Die geschützte Formel gibt den Vorsprung in Prozent an. Die Ergebnisse im Haupttext (94% gegenüber 89%) ergeben einen Vorsprung von fünf Prozentpunkten.
- ce0809f3, 7086b512: Die Zahlenprüfung zerlegt die englische Schreibweise „6.5M“. Die deutsche Angabe „6.5 Millionen“ bezeichnet dieselbe Parameteranzahl.
- fd210403: „GPU-Kernel“ bezeichnet ein auf der GPU ausgeführtes Programm. Der Glossareintrag „Faltungskern“ trifft hier nicht zu.
