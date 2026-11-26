# Notes on chapter_recurrent-modern/lstm.md#u02 (Codex Astra (medium), 2026-11-26 19:33 UTC)

- 44b8bb72, b02d74d4: Der kleine Beitrag ist für sich genommen in float32 darstellbar. Erst bei der Addition geht er durch Rundung verloren. „Underflow“ bezeichnet hier keinen Unterlauf des Exponenten.
- bd91aa99: „recurrent kernels“ sind die rekurrenten Gewichtsmatrizen von Flax. Der Glossartreffer „Faltungskern“ passt hier nicht.
