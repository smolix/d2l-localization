# Notes on chapter_attention/queries-keys-values.md#u01 (Codex Astra (medium), 2026-11-26 19:09 UTC)

- 4ba4e9fd, 884264f4: Die Blockgrenze verläuft mitten durch die Formel am Minuszeichen. Beide Teile stehen an denselben Stellen wie im Original.
- f70c40a8: Die Ableitung ist fast überall null. An den Sprungstellen existiert sie nicht. Die fett gedruckte Überschrift folgt dem mathematisch präzisen Fließtext.
- 796dea6f: Bei endlicher Skalierung nähert sich die Ausgabe dem Eckpunkt, ohne ihn zu erreichen.
- 6654a09a: Die letzte Gleichheit im Score gilt nur für den Gauß-Kernel.
- be3c1440, 0de8192f: Der Kehrwert der Varianz ist die inverse Temperatur, wie die Gleichung zeigt.
- 00de58de, f957e553, 265b8c11, d880db0d, bf1acd80, 515d2a21, e296f2df, 8f9a1575, 6654a09a: „kernel“ bezeichnet den Kernel der nichtparametrischen Regression. Der Glossartreffer „Faltungskern“ betrifft eine andere Bedeutung.
- 884264f4, 94d3274c: „softmax weights“ bezeichnet die resultierenden Softmax-Gewichte. Der Glossartreffer „Softmax-Funktion“ bezieht sich auf die Funktion selbst.
- a5e6c63a, 9a99010a: „Interpolation mit dem nächsten Nachbarn“ bezeichnet die Interpolation mit genau einem Nachbarn. Die ausgeschriebene Bezeichnung löst die Zahlenwarnung aus.
- 37b42a74: „in einem Schritt“ gibt „1-Hop“ wieder. Die Zahl ist ausgeschrieben.
