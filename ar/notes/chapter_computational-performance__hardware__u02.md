# Notes on chapter_computational-performance/hardware.md#u02 (claude fable-5.2 (medium), 2026-11-28 01:02 UTC)

- a146d685: "(SMs; 128 on the RTX 4090)" split into the gloss (streaming multiprocessor, SM) and a clause وعددها 128 في بطاقة RTX 4090; SM follows the chapter index (معالج SM).
- 9ac905a3: "(2 FLOPs)" → (عمليتا FLOP), the dual without the digit (style §6); the number 2 is reported as missing.
- 878b1873, 6c670de2, 445aee07: "1B-parameter GPT" / "1B GPT" → مليار معلَمة (style §15.7); the digit 1 is not written.
- 878b1873: "< 5 ms" → أقل من 5 ms in prose.
- 369927ea: "PCIe peer-to-peer (P2P) access" → وصول PCIe المباشر بين الأقران (peer-to-peer, P2P), with the English gloss added at the defining mention.
- 9eaa867a: "spans two regimes" is the verb (يمتد على), not the glossary's span (فضاء مولد); false match if reported.
- 9c20bb79, 2881e5bb: "workload" here means a kind of computation (inference, decode), not an amount of work; the lexicon's حجم العمل does not fit a heading, so حمل العمل / أحمال العمل is used and logged as a separate sense.
- ce62d29d: "1-stream ceiling" → سقف السيل الواحد, no digit.
- a65c09fa: cells rewritten with the class noun وصلة; "via host" → عبر المضيف.
- ebf9d672: the cell "`tf32` $83\text{ TF/s} \to$ `bf16` $165\text{ TF/s}$" is copied as one left-to-right run.
- 9eaa867a, 7695a731, 04c98d60: the glossary term is used in the accusative with tanwin (رمزًا مميزًا), which the check does not match to رمز مميز; false match.
- d62d1ae2, 811a5b39: "**1 stream**" → **لسيل واحد** / **سيل واحد**, no digit (style §6); $B = 1$ keeps the digit in math.
