# Notes on chapter_optimization/sgd.md#u01 (claude fable-5.2 (medium), 2026-11-27 22:18 UTC)

- c8f86ee2: "Equation :eqref:`…`" is written without the word المعادلة (style §9.1: the build prints the prefix), with a feminine verb.
- 8f847d8f: warmup–stable–decay is not glossed here; the chapter page (index.md) glosses it. If the chapter check prefers a gloss on this page too, add (warmup–stable–decay) after الاضمحلال.
- 9671d8be: "$1$ standardized sound-pressure target" keeps the protected `$1$` with a singular noun (و $1$ هدف), since the math span cannot be replaced by واحد.
- ff97bc25: "a $6$th weight" → الوزن رقم $6$; b3dd1a58: "the $k$-th eigenvector" → المتجه الذاتي رقم $k$ (as logged); 62baf0b6: "$1/5{,}000$th the cost" → بجزء $1/5{,}000$ من كلفة الخطوة.
- d3382b4a: "Cost the Same as 1" is written with the word واحد (مثال واحد), so the digit 1 is not in the title (style §6).
- 8dfc6501: "1D" → أحادي الأبعاد (style §6), so the digit is not in the text.
- 42cb59b7: the class noun المقدار is added between the two math spans of each sub-bullet (style §3.2).
- af3a9895: the sub-bullet begins with the class noun الحالتان so that it does not start with math.
- 836db8ca: false glossary match: the plural النماذج اللغوية الكبيرة is used; the check knows only the singular نموذج لغوي كبير.
- 48966da6, bb8d9199: false glossary match: "Sampling $b$ examples" is the verb, rendered سحب $b$ مثالًا (style §5.3), not the noun أخذ العينات.
- 70bea9ba: false glossary match: the broken plural أحجام الدفعات is used for "batch sizes"; the check knows only حجم الدفعة.
- Chapter check: "stationary" is مستقر here, as on the momentum page, although "stable" is مستقر too (the glossary's توزيع ساكن would keep the two apart: التباين الساكن); and "regime" is نطاق on this page (نطاق الاستيفاء، النطاق المحدود بالضوضاء), نظام on the momentum page and نمط on the test-models page.
- 446ef162, a96a3133, 445a78a4, 1bbcbfde, 130b7049, fc91a634, cddb18dc, 7feea5a7, b3dd1a58, 6c3053f2, d40c0ffd, b0d9b1a2, bb8d9199, d2043201, 267e1984, d0335cd1, 0f26254b, df1a209b, 48aa7909, 2c6eef42, d8e155e8, 87dbd50d, 7f25f273, 107558b7, 31577dd3, f0898de7, 190a72fa, 4f616e47: re-applied after the first application of this unit, to align "stationary" → مستقر, "flat" → منبسط and "steep" → حاد with gd.md and momentum.md of this chapter (see DECISIONS); no other change.
