# Notes on chapter_recurrent-modern/ssm.md#u02 (Codex Astra (medium), 2026-11-26 19:36 UTC)

- c35c7df7, 5b39bf14, 20ffc36d: Die Quelle verknüpft die relative Maschinengenauigkeit mit dem Unterlauf. Ein Wert um 4 × 10⁻⁴⁴ ist in float32 noch subnormal darstellbar. Ob der Gradient ganz verschwindet, hängt auch von den weiteren Rechenschritten und der Behandlung subnormaler Zahlen ab.
