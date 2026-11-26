# Notes on chapter_multilayer-perceptrons/numerical-stability-and-init.md#u02 (Codex Astra (medium), 2026-11-26 17:45 UTC)

- 9e2c6d60: Bei konstanten Gewichten ist die Varianz null. Die Formel in der Spalte für die Gewichtsvarianz gibt hier das zweite Moment an. Die Aussage zur Normerhaltung gilt nicht allgemein.
- cdd4e784: Die Symmetrie bleibt unter SGD erhalten. Die Parameter selbst müssen keine Fixpunkte von SGD sein.
- baee39dd: Bei Fan-in hängt der Gesamtfaktor vom Verhältnis der Endbreiten ab. Eine tiefenunabhängige Schranke für beliebige Breitenfolgen setzt eine entsprechende Beschränkung dieses Verhältnisses voraus.
- afea9464, b0dab5ba, 3e6d2b8c: „ersten Schritt“ gibt die Ziffer 1 in „step 1“ als deutsches Ordinalzahlwort wieder.
- 3e6d2b8c: Die Rechnung zeigt nur, dass im ersten Schritt alle Gradienten der Eingangsgewichte null sind. Verschiedene Bias-Werte können bereits verschiedene Aktivierungen und damit verschiedene Gradienten der Ausgangsgewichte erzeugen.
