# Notes on chapter_convolutional-modern/convnext.md#u02 (claude fable-5.2 (medium), 2026-11-27 21:25 UTC)

- fa3d9c60: "expansion" is توسيع here, as in u01 of this page (the efficient-convnets page logged زيادة العرض / طبقة توسعة); "adds the residual input" is rendered as adding the input of the residual branch to the result.
- fb403840: "3.35M parameters" → 3.35 مليون معلَمة, as logged for body text; the number warning is a false match.
- ae3af592, 79e05660, d29f0747: glossary warnings for "norm" are false matches: "single GELU/norm" and "Activations & norms" mean normalization layers (طبقة تطبيع), "Global Response Norm" is a normalization (تطبيع), and "L2 norms" is معايير, the broken plural of معيار.
- df64c61a: "~2-point drop" → نحو نقطتين (dual, so the digit 2 is not written); "~43--46M" → نحو 43 إلى 46 مليون.
- d29f0747: "1 GELU", "1 per-pixel LayerNorm", "2 nonlinearities" → واحدة / لاخطيتين (singular with واحد and dual, style §6), so the digits 1 and 2 are not written.
- b8c74fb7: "masked zero-bleeding" is read as the zeros of masked regions bleeding into visible ones.
- bb2785fb: the row starts with the Latin run BN $\to$ LN, copied as the log prescribes.
- 9cdffe32: "Downsampling & scaling" is read as downsampling and layer scale (المقياس), which is what the row lists.
