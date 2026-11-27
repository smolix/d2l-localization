# Notes on chapter_preliminaries/ndarray.md#u01 (claude fable-5.2 (medium), 2026-11-27 19:49 UTC)

- Numbers reported as missing are intended (style §6): "1D", "0D", "2D", "4D" are written أحادي / صفري / ثنائي / رباعي الأبعاد (c9c043b0, ed556481, 5937e24b, 0b79e11b, 649036ed, 78966a08, ed229995, be526d20), "1-tuple" as المجموعة المرتبة ذات العنصر الواحد (81342bbc).
- 38c45ac0: "precision" here is floating-point precision → الدقة, not the glossary's metric إحكام (false match).
- 9943c1e2, f73a242e, 3122909d: "sequence" of pseudorandom numbers → متتالية (mathematical sense, style §10.1), not تسلسل.
- 27b4502e, 6b7323fd, 3e65fa0e, d23bfc02: "stride" is the memory stride → خطوة, not the convolution term خطوة الانزلاق. b52a1394, 778c16ef, 5d595ca8: "kernel" is compiled CPU/GPU code → نواة / نوى. Verbs "returns", "initialized", "sampling … from" are rendered by يعيد، تُهيأ، سحب (false noun matches, if reported).
- 649036ed: the protected `$\;\to\;$` arrows are drawn left-to-right inside a right-to-left line, so the chain scalar → vector → matrix reads against the arrow direction in the Arabic slide; the build may need to handle this.
- f73a242e: "(PRNGs)" is written (PRNG), no Latin plural in prose (style §5.1).
- 32babc1e, 83fa9602, a8143282: a connecting أي / كما في was put between two left-to-right items (style §3.2); it adds no content.
