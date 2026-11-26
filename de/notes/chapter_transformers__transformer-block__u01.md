# Notes on chapter_transformers/transformer-block.md#u01 (Codex Astra (medium), 2026-11-26 19:22 UTC)

- 50534d1d: „unit scalar loss“ bezeichnet hier den skalaren Verlust, dessen Rückwärtsdurchlauf mit dem Gradienten eins beginnt. Der Beispielcode setzt den Verlustwert selbst nicht auf eins.
- beb09a8e: Die Gleichungen zeigen, dass die Projektionsmatrizen aus der normalisierten Eingabe Abfragen und Schlüssel erzeugen. Die englische Formulierung suggeriert eine zweite Multiplikation.
- 0c156638: Fest ist die Gewichtsskala 0.02. Laut Folientext wächst mit der Modellbreite der Varianzgewinn je Projektion.
- 3cc4f5fd, 18b06421: Der Glossarhinweis auf „Softmax-Funktion“ betrifft hier die Verteilung, deren Entropie sinkt bzw. die kollabiert. Deshalb steht „Softmax-Verteilung“.
- 091bf24c: Der Zahlenhinweis auf „3“ ist ein Fehlalarm. „3-matrix“ ist im Deutschen als „der drei Matrizen“ wiedergegeben.
