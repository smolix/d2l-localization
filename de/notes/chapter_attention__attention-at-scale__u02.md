# Notes on chapter_attention/attention-at-scale.md#u02 (Codex Astra (medium), 2026-11-26 19:15 UTC)

- 4277568f, bb716442: „online softmax“ bezeichnet das Online-Verfahren, „softmax weights“ dessen Gewichte. Dafür stehen im Text die entsprechenden Softmax-Komposita.
- bb716442, 54636535: „kernel“ bezeichnet GPU-Code und keinen Faltungskern.
- c6f562c4: Die Rundungsaussage wurde präzisiert. Unterhalb der kleinsten positiven float16-Zahl liegt die Rundungsschwelle für null bei der Hälfte dieser Zahl.
