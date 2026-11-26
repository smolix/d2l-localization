# Notes on chapter_linear-classification/softmax-regression-scratch.md#u02 (Codex Astra (medium), 2026-11-26 17:35 UTC)

- 5a8ebda0: Die Intervallgrenzen sind eingeschlossen. Insbesondere erreicht die Summe bei gleichen Logits die obere Grenze.
- 8da8096a: Die Untergrenze gilt für die Wahrscheinlichkeit, nicht für den Verlust.
- 9877c042: Bei fortlaufender Addition von Summanden bis 1 kann die FP16-Summe unter Rundung zur nächsten geraden Mantisse bereits bei 2048 stehen bleiben.
- 13ec7d45: „in einfacher Genauigkeit“ und „in doppelter Genauigkeit“ sind die grammatisch erforderlichen Dativformen der Glossareinträge.
- 9877c042: Die Zahlenprüfung erkennt „2,048“ nicht als „2048“. Vierstellige Zahlen stehen gemäß deutschem Stil ohne Tausendertrennzeichen.
