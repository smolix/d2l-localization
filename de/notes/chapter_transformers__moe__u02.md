# Notes on chapter_transformers/moe.md#u02 (Codex Astra (medium), 2026-11-26 19:29 UTC)

- 7cad27bf: Wie der Code im Kontext zeigt, gilt der Switch-Hilfsverlust nur für das MoE-Modell.
- e3657d89: Die Bedingung für das Verschwinden gilt für den gesamten Gradientenvektor bezüglich aller Router-Logits eines Tokens. Eine einzelne Komponente kann auch bei ungleichen Lastanteilen null sein.
