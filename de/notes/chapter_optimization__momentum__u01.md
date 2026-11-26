# Notes on chapter_optimization/momentum.md#u01 (Codex Astra (medium), 2026-11-26 18:56 UTC)

- 756f357f: „Kugel um das Minimum“ folgt dem Entscheidungsprotokoll. Im Fließtext gibt die Formel den Beitrag der Störung ohne das Minuszeichen des Abstiegsschritts an, auf der Folie 600f704d mit Minuszeichen. Beide Formeln sind unverändert.
- c4df96af: „within 1% of the origin“ meint einen Abstand von höchstens 1% des jeweiligen Anfangsabstands zum Ursprung, wie im Code für die relativen Toleranzen.
- 17ef1ef4: Die Wurzeln selbst ändern sich bei der Iteration nicht. Es klingen ihre Beiträge zur Lösung ab. Die angegebene positive Doppelwurzel gilt für die flache Richtung. Bei der steilen Richtung kann die Doppelwurzel negativ sein und das Vorzeichen der Iterierten wechseln.
- a0ba2b60: Die Spanne von 63–75% gilt für die betrachteten Werte des Momentum-Koeffizienten, nicht für alle zulässigen Werte.
