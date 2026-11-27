# Style-guide follow-ups (2026-11-26)

Settings, string overrides, checker/build code and glossary corrections proposed by the Fable-written style guides (`<code>/style.md`). None is applied yet. Each guide's final sections hold the details and the questions for a native reviewer.

## ja (ja/style.md; reviewed by a native speaker, PR #2, 2026-11-27)
- languages.yml ja: translator gemini-4-carbon (done). No check/typography settings needed.
- ja/strings.yml (new, `l10n strings ja`): confirm on the first Japanese build that references print 図, 表, 式 (and 第 N 章 / N 節) from Quarto's Japanese defaults; pin them in strings.yml if not.
- New code (decided 2026-11-27): the Japanese build sets `*…*` in bold, in HTML, the PDFs and the Beamer decks (d2l-neu catalog setting `fonts.emphasis: bold`).
- Checker: nothing new; reordered tokens with a NOTE and `N×`/en-dash ranges are already handled (protect.py).
- Glossary: nothing; 訓練, 自己アテンション and the §7.2 corrections are already in ja.jsonl.

## zh (zh/style.md, 606 lines; 17 to-confirm, 11 glossary corrections)
- languages.yml: none.
- zh/strings.yml (new): quarto.crossref-sec-prefix "§" (to confirm; Quarto zh prints 小节/章节); pin fig/tbl/eq to 图/表/式.
- New code: chapter-ref suffix (第 3 章); glossary match ignoring CJK–Latin space (6 binding mixed entries like Xavier初始化); CJK–Latin spacing/line-break check; Kai/bold for *…*; optional east_asian_line_breaks.
- Glossary: 30 mixed-script entries need the space; kernel-launch 核函数启动→内核启动; kv-cache→KV 缓存; activation alternatives (false match); drop 精度/准确性 alternatives.
## pt (pt/style.md, 683 lines; 16 to-confirm, 9 glossary corrections)
- languages.yml pt: check {thousands: space}; typography {nbsp: true}.
- pt/strings.yml: lang "pt-BR", pdf.lang "pt-BR"; quarto.crossref-ch-prefix "Capítulo" (Quarto pt-BR has none → would print "Chapter 7"; i18n.py reads only that file); book.exercises_heading "Exercícios"; book.discussions_link "Discussões".
- New code: article-before-reference agreement check (or show printed noun next to refs in work files); named warning for decimal comma in prose.
- Decisions to flag: decimal point (vs Brazilian comma); article+contraction before :numref: (na Seção) — departs from all 3 test translators; loans not italic; SGD masculine.
- Glossary: strided-convolution → convolução com stride; new: exploitation (explotação), patch, deep/machine learning, built-in/entry/edge.
- NOTE: evals/pt-conv-layer/answers-sol, -astra, -terra appeared (Alex's own runs) — never touch/commit.
## es (es/style.md, 704 lines; 13 to-confirm)
- languages.yml es: check {thousands: space}; typography {nbsp: true}.
- es/strings.yml: pdf.lang "es"; quarto.callout-tip-title "Consejo"; optional lower-case in-text crossref prefixes (after build test).
- New code: nbsp for multi-letter abbreviations (p. ej., EE. UU., n.º); babel-spanish PDF options (decimal point in math, lím/máx, "Cuadro" for tables) — untested; optional ¿ check; optional article-before-ref check.
- Decisions to flag: usted (2 of 3 translators chose tú); decimal point; article before refs (la Sección) — departs from German no-article ruling and all 3 translators; loans roman not italic.
- Glossary: transformador→Transformer; incrustación compounds→embedding; agrupación→pooling; coma flotante→punto flotante; temperatura de softmax→temperatura; drop alt coste; add elemento, parche, aprendizaje profundo, posentrenamiento.
- evals/es-conv-layer also has answers-astra/-sol/-terra (Alex's).
## ko (ko/style.md, 584 lines; 15 to-confirm, 12 glossary rows)
- languages.yml ko: none.
- ko/strings.yml: quarto.crossref-eq-prefix "식" (default 방정식); crossref-sec-prefix "§" interim (default 섹션); chapter prefix: no good prefix-only value (Quarto prints 장 7); book.exercises_heading "연습 문제"; book.discussions_link "토론".
- New code: number-first reference words (7.2절, 7장); ko build step removing space between ref/inline-math and following particle (pandoc: [@fig-x]에서 works; @fig-x를 breaks key); optional build-time particle agreement; optional cjk_adjacent: refs mode; Hangul emphasis styling.
- Decisions: 합니다체 body; slides 한다체/noun ending; only single-form particles after refs (all 3 translators broke this).
- Glossary: 매개변수→파라미터 family; 순방향/역방향 패스→순전파/역전파; regularization 정칙화 consistency; spacing 정규분포 vs 가우시안 분포; 풀링 창; 뉴턴 단계; 네트워크 entries; 임계값; false match "standard normal distribution".
- evals/ko-conv-layer also has answers-astra/-sol/-terra (Alex's).
## fr (fr/style.md, 665 lines; 15 to-confirm, 8 glossary corrections)
- languages.yml fr: check {thousands: space}; typography {nbsp: true}.
- fr/strings.yml: quarto.crossref-tbl-title + crossref-tbl-prefix "Tableau" (Quarto fr prints "Table"); pdf.lang "fr"; book.exercises_heading "Exercices"; book.discussions_link "Discussions".
- New code: French punctuation nbsp (before : ; ? ! », after «; skip table align rows, ::: fences, lines starting ':'); glossary._present doesn't inflect terms with ' or - (jeux d'entraînement false-missing; typed ’ breaks match); nbsp for "p. ex.".
- Decisions to flag: decimal point; "45 %" (vs German no-space); article before refs; (angl. …) gloss; jeton/plongement/réglage fin/transformeur vs loans.
## ar (ar/style.md, 628 lines; 18 to-confirm, 9 glossary corrections)
- languages.yml ar: none (dir rtl set).
- ar/strings.yml: Quarto 1.9.37 has NO _language-ar.yml → refs print English. Set quarto.crossref-ch-prefix "الفصل", sec "القسم", fig prefix+title "الشكل", tbl prefix+title "الجدول", eq "المعادلة", apx "الملحق", section-title-references "المراجع"; book.exercises_heading "تمارين"; book.discussions_link "النقاشات"; fonts.html.sans/google Arabic family; pdf.lang null.
- New code: RTL CSS isolating code/pre/math as LTR ([1, -1] would render reversed); upright em; glossary check stripping marks+tatweel (+ broken plurals); checker error for Eastern digits ٠–٩ ٫ ٬ ٪; checker error for ref role touching an Arabic letter (pandoc: ب@sec-x not a ref; @sec-xو eats و); nbsp after free-standing و and الـ/بـ before a token; RTL PDF/slides/pptx/beamer with Arabic fonts; wrap plain 3×3 outside math.
- Glossary: marks on ~45 entries (incl. verified), معلمة fatha, acronyms in preferred (JIT), maximum likelihood الأعظم/الأقصى, encode تشفير/ترميز, missing input/output/parameter/code entries.
## vi (vi/style.md, 514 lines; 15 to-confirm, 13 glossary corrections)
- languages.yml vi: check {thousands: space}; typography {nbsp: true}.
- vi/strings.yml: lang "vi"; pdf.lang "vi" only if babel vi builds (untested); Quarto has NO _language-vi.yml → quarto.crossref-* Chương, Mục, Hình, Bảng, Công thức + section-title-references "Tài liệu tham khảo" + other Quarto UI keys; book.exercises_heading "Bài tập"; book.discussions_link "Thảo luận". Check PDF font coverage U+1EA0–1EF9.
- New code: reject non-NFC blocks + warn on new-style tone placement; NFC-normalize both sides in glossary._present; smart-quote test for lang vi.
- Decisions: section = Mục (all 3 translators wrote phần, colliding with Parts); old-style tones (hóa); decimal point.
- Glossary: kernel size → kích thước hạt nhân; scalar adj vô hướng; drop parentheses in on/off-policy & SAC entries; new: edge, standard normal, dense layer, patch, implement.
## ru (ru/style.md, 674 lines; 14 to-confirm, 7 glossary corrections)
- languages.yml ru: check {thousands: space}; typography {nbsp: true}.
- ru/strings.yml: quarto.crossref-fig-prefix "рис.", tbl "табл.", sec "разд.", ch "гл." (Quarto ru prints «Глава» for a SECTION + capitalised nominative); section-title-references "Литература"; callout-note-title "Примечание"; book.exercises_heading "Упражнения"; book.discussions_link "Обсуждение"; pdf.lang "ru". Inconsolata lacks Cyrillic (code spans only).
- New code: empty crossref-eq-prefix for bare equation numbers (fix_crossref_numbers.py builds regex from printed word → empty would match any number); **d2l-neu d2l_preprocess.py writes reference ranges with literal English "to" (7 places) → should come from string catalog (affects ALL languages)**; Russian nbsp rules (т. е., и т. д., before em dash, after one-letter words); optional decimal comma; glossary._present can't match inflected short words (слой→слоя).
- Glossary: drop сдвиг alt from bias/intercept; add offset/shift → сдвиг; add inference → инференс; макс-пулинг/усредняющий пулинг asymmetry; mark slang alts avoid; add фреймворк, граница, фрагмент, ленточная матрица, локально связный слой.
## hi (hi/style.md, 591 lines; 14 to-confirm, 9 glossary corrections)
- languages.yml hi: none (don't set cjk_adjacent: all — rejects danda after math/ref).
- hi/strings.yml: Quarto has NO _language-hi.yml → crossref ch अध्याय, sec अनुभाग, fig prefix+title चित्र, tbl prefix+title तालिका, eq समीकरण, apx परिशिष्ट, section-title-references संदर्भ; book.exercises_heading अभ्यास; book.discussions_link चर्चा; Devanagari font in fonts.html.google/sans.
- New code: error for ref role touching Devanagari letter (pandoc reads :numref:`sec_x`में as label sec-xम, dropped silently); Hindi lint (०–९, precomposed nukta/non-NFC, ZWJ/ZWNJ/NBSP, '.' or '|' ending sentence); glossary._present no Devanagari inflection (नमूनों, बैच के आकार); PDF Devanagari font slot (fonts.pdf has CJK only; pdf.lang hi untested); emphasis style (no italic).
- Decisions: dimension विमा (all 3 wrote आयाम = amplitude); section अनुभाग; ML terms as Devanagari loans, no CSTT coinages; no gloss after loans (Opus/Sonnet did).
- Glossary: प्रसामान्य बंटन→प्रसामान्य वितरण; कम्प्यूटेशनल→कंप्यूटेशनल; कंप्यूटर दृष्टि→कंप्यूटर विज़न; आरंभीकरण→इनिशियलाइज़ेशन; +8 head words.
## tr (tr/style.md, 695 lines; 16 to-confirm, 9 glossary rows) — read all 6 answer sets incl. Alex's astra/sol/terra
- languages.yml tr: check {thousands: space}; typography {nbsp: true}.
- tr/strings.yml: no crossref override needed (Quarto tr OK: Şekil/Tablo/Bölüm/Eşitlik); book.exercises_heading "Alıştırmalar"; book.discussions_link "Tartışmalar"; pdf.lang "tr" only after test (babel shorthands = : !).
- New code: Turkish case folding in glossary.py (lower() breaks İ/I: İlkleme, İz, İzdüşüm, İnce Ayar, İstem, Isınma false-missing); uppercase in build needs lang=tr (CSS text-transform, \MakeUppercase); optional build-attached case endings on refs (Şekil 6.2.1'de); optional check for apostrophe/letter after ref role.
- Decisions: no suffix after refs (4/6 wrote ":numref: bölümünde" → "Bölüm 6.1 bölümünde"); decimal point; %92.3 (vs TRANSLATING.md §4 copies 92.3%); title case headings.
- Glossary: transformer→Transformer; "1. tür hata"→birinci tür hata; Langevin dinamikleri→dinamiği; remove parenthetical acronyms from 7 entries.
## el (el/style.md, 555 lines; 15 to-confirm, 13 glossary rows)
- languages.yml el: check {thousands: space}; typography {nbsp: true}.
- el/strings.yml: quarto.crossref-tbl-prefix "Πίνακα" only (accusative in refs; captions keep Πίνακας); book.exercises_heading "Ασκήσεις"; book.discussions_link "Συζητήσεις"; pdf.lang "el" (verify at first PDF build).
- New code: accent-insensitive glossary matching (πίνακας→πινάκων); Greek lint (?, —, U+037E, U+0387, U+00B5, ASCII ; + lowercase, Latin letters in Greek words, genitive/wrong-gender article before ref, tab_ ref without τον/βλ.).
- Decisions: patch → τεμάχιο; article before refs, never genitive (Opus «της :numref:» prints «της Ενότητα»); Greek ? = ASCII ;.
- Glossary: patch family (τεμάχιο, τεμάχιο εικόνας, ενσωμάτωση τεμαχίων; slicing→λήψη τμήματος); τα perceptrons; new entries κυλιόμενο παράθυρο, ενεργό υποδεκτικό πεδίο, ανακλασμένος πυρήνας, αντιστάθμιση, ισοπεδώνω, module, εξάγω, μέρος.
