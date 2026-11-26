# Notes on chapter_transformers/encoders-decoders.md#u01 (Codex Astra (medium), 2026-11-26 19:25 UTC)

- f37e8b4d: „embedding a document“ ist hier ein Verb und wird als „ein Dokument … einbetten“ wiedergegeben.
- 37481860, 1208c7fa: „1-sided“ und „2-sided“ sind als „einseitig“ und „beidseitig“ wiedergegeben. Die Ziffernwarnungen betreffen diese ausgeschriebenen Adjektive.
- 63347537, 63347537-2, 2c2052fc, 7eb3676e: Die Folien setzen voraus, dass die Schlüssel- und Wertprojektionen zwischengespeichert werden. Der Beispielcode speichert die Encoder-Ausgabe, ruft die Cross-Attention aber bei jedem Dekodierschritt erneut mit dieser Ausgabe auf.
- 37481860: Im Englischen nimmt die Rekonstruktion selbst ab. Gemeint ist der Rekonstruktionsverlust.
