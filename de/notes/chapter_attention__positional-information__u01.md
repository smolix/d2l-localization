# Notes on chapter_attention/positional-information.md#u01 (Codex Astra (medium), 2026-11-26 19:14 UTC)

- 7794bbee, 0a33d6a6: Die kausale Maske schließt die aktuelle Position ein. Die Übersetzung zählt sie bei den zugänglichen Positionen mit.
- 5fdb4a90, 220abbf2: Die Zerlegung in reine Drehblöcke gilt für die bei RoPE verwendeten Matrizen. Eine beliebige reelle orthogonale Matrix kann auch eindimensionale Spiegelungsblöcke enthalten.
- 5b455e43: Die Ausgaben stimmen nach entsprechender Umordnung überein, wie es die Proposition und der Code zeigen.
- a3fce575: Der obere Wert ist ein Grenzwert der geometrischen Staffelung. Für den angegebenen endlichen Index liegt die Periode darunter.
- 92dc3930: Eine volle Umdrehung deckt den Winkelbereich ab. Bei ganzzahligen Positionen treten die einzelnen Winkel nur näherungsweise wieder auf, wie die Übung zur Positionsinterpolation präzisiert.
- 71e13ed8: Der Glossartreffer „embeddings“ bezieht sich hier auf den festgelegten Verfahrensnamen „Rotary Position Embeddings“.
- c5d3ea58: „recency bias“ bezeichnet die Bevorzugung jüngerer Positionen. Der Ausdruck meint hier weder einen additiven Modellparameter noch die Verzerrung eines Schätzers.
