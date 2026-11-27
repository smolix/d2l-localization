# Notes on chapter_linear-regression/oo-design.md#u02 (claude fable-5.2 (medium), 2026-11-27 20:08 UTC)

- 2d23c93c: "temperature" is the physical temperature of a black body (درجة الحرارة), not the glossary's softmax temperature; a glossary warning here is a false match.
- 31b6a56c, 14782a83, 28577fae, 9fc407f6, 3a961683, 4b7f4aae, 631dc0d5: the English verb "return" / "return-value" is a function's return (يعيد، القيم المعادة), not the RL term; "trajectory" in 577e31cd is the path of a loss curve (المسار); "traces" there is a verb, not the matrix trace. Any glossary warning for these is a false match.
- 7bbf9706, d6deb0fb: جداول معدل التعلم is the broken plural of the glossary's جدول معدل التعلم; مجدولات معدل التعلم renders "schedulers".
- ff8d32c8: "parameter-sharing views" is written with the verb (يتشاركان المعلَمات), not the glossary noun مشاركة الأوزان.
- 9fc407f6: "weights each minibatch metric" is the verb (يزن), not the glossary noun وزن; 7ed3a992: المعلَمات الفائقة is the plural of the glossary's معلَمة فائقة. Both glossary warnings are false matches, as is 'traces' in 577e31cd.
- 58703ff3: "across our four frameworks" is translated as written, although each reader sees one framework tab.
- 40fdfbb0: "Inherited by every `Module`, `DataModule`, and `Trainer`" strictly applies to `HyperParameters` only; translated as written (يرث منها).
