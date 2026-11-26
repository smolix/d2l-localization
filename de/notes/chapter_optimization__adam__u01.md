# Notes on chapter_optimization/adam.md#u01 (Codex Astra (medium), 2026-11-26 19:01 UTC)

- 5c2b5f66, 5d1ad436: Die Formel für den ersten Schritt ist nur bei verschwindendem Epsilon exakt. Bei vernachlässigbarem Epsilon gilt sie näherungsweise.
- 7102bfae, 4c50be20: Die angegebene obere Schranke für die Schrittlänge gilt nicht allgemein bei unterschiedlichen Abklingfaktoren für die beiden Momente. Die Aussage bezieht sich hier auf den Versuch.
- 4bbed372, 7d8bb88a: Die als Faktor zwei bezeichneten Bereiche sind gerundet. Formeln und Zahlenbereiche sind wie im Original übernommen.
- 88ac056d, 5d1ad436: Die Zahlenprüfung erkennt die ausgeschriebene Ordnungszahl „ersten“ für Schritt 1 nicht. Alle Schrittangaben sind erhalten.
