# Notes on chapter_convolutional-modern/training-recipes.md#u01 (claude fable-5.2 (medium), 2026-11-27 21:24 UTC)

- ef4de680, 8981fbca, a9e73cbd: "early(-)2020s" is written in words (مطلع عشرينيات القرن الحادي والعشرين, style §6), so the number 2020 is reported as missing.
- 8df91990: "by a factor of 10" → تقسمه على 10; "two-epoch" / "three-epoch" are duals and words as in the English.
- 5ff67b96: number warning (11, 1) is a false match: "1-hot" → بالترميز الأحادي, so the digit 1 is missing, and "11.2M" → 11.2 مليون معلَمة (chapter decision for "43M-param") keeps the number as 11.2.
- 8df91990: glossary warning for "hyperparameters" is a false match: the text has the indefinite plural معلَمات فائقة أخرى.
- 596e3671: glossary warning for "Sampling" is a false match: "sampling one Bernoulli mask" → سحب قناع برنولي (style §5.3: to sample → يسحب).
- 4faa31ef, 81025784: "identity shortcut" → وصلة التخطي المحايدة (chapter decision on the resnet page, not the glossary proposal وصلة تخطي هوية).
- bbf1305c: "Two observations" → ملاحظتان (the everyday word, not the glossary's مشاهدة for an RL/statistical observation).
- 707dbcc2: the framework sense of "buffers" → مخازن حالة (decisions log); no English gloss there because the sentence is already inside parentheses.
- c48ef72c, c2642ee2: the slides write "half-cosine decay" as اضمحلال جيب التمام بنصف دورة; slides and table cells keep the en dash ranges (300–600, 1–2) while body text writes من … إلى ….
- Broken plurals and marked forms that the glossary check may not match: حقب، جداول، هوامش، وسوم، أمثلة التدريب، العزوم الثانية، البواقي; اضمحلال الأوزان المنفصل follows the decisions log (glossary: اضمحلال وزن منفصل).
