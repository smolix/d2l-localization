# Notes on chapter_deep-reinforcement-learning/sac.md#u04 (Codex Astra (medium), 2026-11-26 20:35 UTC)

- bba4361e: „primacy bias“ bezeichnet die Übergewichtung früher Erfahrungen, weder einen additiven Bias noch die Verzerrung eines Schätzers.
- 1fe5cca1: Die einzelne Ziehung ist ausgeschrieben als „anhand einer neu gezogenen Aktion“.
- 0ed24a86: Die Formel für den vorhergesagten Aktionswert enthält den Entropiebonus der ersten Aktion nicht. Die Übersetzung unterscheidet diesen Wert vom gemessenen Ertrag einschließlich dieses Bonus.
- 7d15b2bb, e6ec7571: Laut Fließtext kann Exploration bis zu etwa zehn Ertragspunkte kosten und das Minimum den Ertrag gelegentlich überschätzen. Die Folientexte geben diese empirischen Aussagen wieder.
- a69fac6c: Die Subtraktion ergibt null, weil tanh in float32 bereits auf eins rundet.
