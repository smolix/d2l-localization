# Notes on chapter_attention/attention-scoring.md#u01 (Codex Astra (medium), 2026-11-26 19:11 UTC)

- 765f696d: Vor der Normalisierung werden ungültige Paare ausgeschlossen. Null ist ihr endgültiges Gewicht, nicht ihr eingesetzter Score.
- 21b28dfd: Die Linearität und das Wegfallen des Abfrageterms gelten in der Nähe von null nur näherungsweise.
- 8cae2414: Wie im Haupttext ist für den angegebenen Erwartungswert zusätzlich die Zentrierung vorausgesetzt.
- 79bcc1ee, e084fc70, 2d46eee9: „kernel“ bezeichnet den Gauß-Kern als Ähnlichkeitsfunktion, keinen Faltungskern.
- 8f9827da: „loss“ bezeichnet die eingeschränkte Allgemeinheit, keinen Trainingsverlust.
- 916f53f1, 72db801b: „softmax“ bestimmt hier das Gewicht bzw. die Schärfe der resultierenden Verteilung näher. Die Übersetzungen „Softmax-Gewicht“ und „Schärfe der Softmax-Verteilung“ bezeichnen diese Größen.
