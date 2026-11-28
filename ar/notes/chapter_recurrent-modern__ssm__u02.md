# Notes on chapter_recurrent-modern/ssm.md#u02 (claude fable-5.2 (medium), 2026-11-28 00:24 UTC)

- d59e130a: "under 1 step" is written أقل من خطوة واحدة (style §6, no digit), so the number check may report 1 as missing.
- c35c7df7: "100–200 steps" is written in prose as ما بين 100 و 200 خطوة (style §3.2); f90ef5bc is a slide and keeps the en dash.
- 24c722e9: "the 4D state" is written رباعية الأبعاد (style §6), no digit 4.
- 92b9e130, f62ddc7b: math spans that the English source breaks across lines are written on one line; their content is unchanged.
- 52257ad4, bf8440a8, 3370971e: the two formulas of a cell are joined by مع / و instead of a bare comma (style §3.2).
- 45663d15: the title of the next section is given in the plural, as the page title of mamba.md has it (نماذج فضاء الحالات الانتقائية); a glossary warning for the singular "selective state space model" is a false match.
- Terms of this page that differ from other pages of the chapter, for the editor: (1) zero-order hold is الإمساك من المرتبة الصفرية here and in mamba.md, but الماسك من المرتبة الصفرية in index.md#u01; (2) "minimal GRU" is وحدة GRU الدنيا here (u01, as lstm.md has وحدة GRU), but شبكة GRU الدنيا / شبكة GRU in linear-recurrence.md#u01; (3) the slide lead-in "Exact Convex Blend" in u01 (6daaceda) is مزج محدب مضبوط, while lstm.md logs convex blend → مزيج محدب (used in this unit, 92b9e130); (4) "skip `D`" is وصلة التخطي here, mamba.md logs skip term → حد التخطي; (5) "stretch (a half-life)" follows the lexicon يمط, while "dilate (a timescale)" is يمدد in u01.
- e7479483: the warning "glossary term not used: token" is a false match; the term is there in the accusative (رمزًا مميزًا واحدًا).
