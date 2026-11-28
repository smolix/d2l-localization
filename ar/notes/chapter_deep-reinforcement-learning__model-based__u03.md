# Notes on chapter_deep-reinforcement-learning/model-based.md#u03 (claude fable-5.2 (medium), 2026-11-28 02:36 UTC)

- 6f2edf6c: "collects 1 episode" is written حلقة واحدة, so the check reports the number 1 as missing (style §6); false match.
- b3118d92, df39a631: "UTD" is never introduced on this page, so the slides write it out as نسبة تحديث (style §5.4); the code label `UTD = 10` is copied.
- 889cc4d8: "DAgger aggregates its labels over all rounds" uses مراحل for rounds, the word this page uses for the rounds of search and distillation; imitation.md calls DAgger's rounds جولات, which on this page are the runs.
