# Notes on chapter_linear-regression/oo-design.md#u02 (Codex Astra (medium), 2026-11-26 17:27 UTC)

- f912df6a: Laut Code ruft die äußere Trainingsschleife nach `fit_epoch` die Validierung auf.
- 4b7f4aae: „subclass Trainer.fit“ meint das Überschreiben der Methode in einer Unterklasse.
- 40fdfbb0: Der Dekorator wird angewandt, nicht geerbt. Die Vererbung bezieht sich auf die Hilfsmethoden von HyperParameters.
- ff8d32c8: „kernels“ bezeichnet kompilierte Rechenkernel, keine Faltungskerne.
- 765b63c7: „der linearen Regression“ steht im Genitiv.
- cc6c6283: „training“ bezeichnet die Tätigkeit, daher „trainieren wir“.
- 592cd636: „den selbst implementierten quadratischen Verlust“ steht im Akkusativ.
- 259780c9: „neuronalen Netze“ ist die flektierte Pluralform.
- e6257a93: Gegenüber der korrekten Gewichtung beträgt der Faktor 1000/(32 × 8), also rund vier. Genau vier beträgt er im Vergleich zu einem Beispiel aus einem vollen Batch.
