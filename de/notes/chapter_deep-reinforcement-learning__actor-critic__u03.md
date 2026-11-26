# Notes on chapter_deep-reinforcement-learning/actor-critic.md#u03 (Codex Astra (medium), 2026-11-26 20:27 UTC)

- b60050df: „weight … decay“ bezeichnet hier das Abklingen des GAE-Gewichts über die Zeit, nicht die Regularisierung durch Gewichtszerfall.
- d30eeb70, 213baf08: Die Frage nennt den TD-Fehler, die englische Lösung beschreibt die Verzerrung des Policy-Gradienten. Ein konstanter Fehler kann den bedingten Erwartungswert des TD-Fehlers verschieben, auch wenn sein Beitrag zum erwarteten Gradienten verschwindet. Die deutsche Frage präzisiert die gemeinte Größe. „biases“ ist ein Verb („verzerrt“).
- 638662b0: Auch bei Lambda gleich eins klingt das Gewicht wegen der Diskontierung weiter ab, wie die Folie mit den Halbwertszeiten zeigt. Die Übersetzung präzisiert „stops decaying“ entsprechend.
