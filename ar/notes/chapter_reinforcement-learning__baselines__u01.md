# Notes on chapter_reinforcement-learning/baselines.md#u01 (claude fable-5.2 (medium), 2026-11-28 01:49 UTC)

- 5d2afa11: glossary "bootstrap" (بوتستراب, the resampling method) is a false match; the RL verb is يمهد ذاتيًا, as qlearning.md logs it.
- b04bfb8d: "precision" here is the accuracy of an estimate (الدقة), not the classification metric إحكام; false glossary match if reported.
- 67492c16, 675ba320: "1-Step" is written out as خطوة واحدة / ذات الخطوة الواحدة (style §6), so the digit 1 is not in the translation.
- eb3c70c4: "goal" is الهدف, as the chapter index, value-iter.md and imitation.md write it; mdp.md and qlearning.md#u02 log المقصد. In 5d2afa11 "targets" is الأهداف (regression targets); the editor reconciles goal.
- aac9fe65: "have the same mean" is phrased without a dual noun before the two :eqref: roles, so that no prefix of our own precedes a reference (style §9.1).
- fe1073e8: the reference :eqref:`eq_rtg` comes before $\gamma < 1$ (reorder across classes only).
- 633fae28, 238141df: the verb "weight" (weight the score by …) is يرجح; the glossary noun وزن is a false match.
