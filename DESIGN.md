# d2l-localization: design

This repository holds the translations of *Dive into Deep Learning* (the
English source is `smolix/d2l-neu`). It contains text only. A translated book is
an English release with its prose replaced: code, outputs, figures and data come
unchanged from a tagged English release, so no language ever re-executes code.

Decisions recorded on 2026-11-26 (Alex) are marked **[ruled]**.

## 1. English releases

- A language builds against a tag of `d2l-neu` named `en-YYYY.MM.DD`
  **[ruled]**. Tags are cut when English is worth releasing, not daily.
- `make release TAG=en-YYYY.MM.DD` (to be added to d2l-neu) refuses to tag
  unless the tree is clean: no stale notebook outputs (`make audit-outputs`),
  every output captured in `outputs/`, every LFS image present, and the
  no-execution render (`make all-quick`) green.
- A tag is therefore a complete, buildable snapshot: sources, the `outputs/`
  store, images and the `DATA_HUB` pins. Each language pins its own tag in
  `languages.yml` and may lag English.

## 2. Repository layout

```
d2l-localization/
  DESIGN.md            # this file
  TRANSLATING.md       # instructions for any translation agent or person
  languages.yml        # per language: code, locale, names, bucket, hosts, pinned tag,
                       #   fonts, direction, forum rule, old edition; the translation
                       #   scope (section 13)
  glossary/            # concepts.jsonl + terms/<locale>.jsonl (from d2l-neu docs/multilingual)
  tools/l10n/          # the CLI (bin/l10n, or python -m l10n from tools/): status, pin, emit,
                       #   apply, check, bump, pseudo, build, deploy, menu, strings, roundtrip
  bin/l10n             # wrapper: bin/l10n <command> ...
  templates/style.md   # starting point for a language's style guide
  site/languages.json  # the language menu, written by `l10n menu`
  <lang>/
    book/              # translated blocks, one file per English path (chapter_*/..., section 3)
    strings.yml        # UI strings and language settings: the build's D2L_LOCALE_FILE
    style.md           # this language's style guide (register, punctuation, spacing)
    decisions.md       # running log of translation choices, read by every later unit
    notes/             # NOTES returned with each unit, one file per unit
    units.jsonl        # one record per applied unit: time, unit, revision, agent, model, edits
    agents.md          # bake-off results and the chosen agent (section 6)
  work/<lang>/         # work files written by `l10n emit` (gitignored)
  _build/<lang>/       # the build worktree of d2l-neu (gitignored)
```

Language codes are ISO 639-1 (`ja`, not `jp`); the bucket is `d2l-<code>` and
the host `<code>.d2l.ai`. English is also served at `d2l.ai`.

## 3. What a translated file looks like

A block is a heading, paragraph, list item (an exercise with its sub-items and
indented equations is one block), table row, figure or table caption, kicker,
the text of a slide, a slide or callout title, or a text-bearing line of the
landing page's raw HTML. `l10n` segments every English page into blocks; the
blocks partition the file, so concatenating them gives the file back byte for
byte (`l10n roundtrip` checks this over the whole book). Everything else is
structure: code fences, display math that stands alone, HTML comments
(including build directives such as `<!-- execution: locked; ... -->`), Quarto
div fences, `:begin_tab:`/`:end_tab:` lines, `:label:`/`:eqlabel:`/`:width:`
lines, slide placeholders (`@cell-id`, `@fig:name`), table separator rows, the
cover kicker `[Dive into Deep Learning · §N.M]{.kicker}` (rewritten by
`gen_slides.py`), and generated regions (`<!-- @x-begin -->` … `<!-- @x-end -->`).

**The translated file holds only translated blocks**, in English order, each
under an anchor with the hash of the English block it translates:

```
<!--en:3f2a91c0-->
## 損失関数

<!--en:8c11d0e4-->
損失は*確率的勾配降下法*（:numref:`sec_sgd`）で最小化します。

<!--en:a90f13c2 was=1b77a2f9 from=en-2026.11.01-->
A stale translation: it translates English block 1b77a2f9 of en-2026.11.01.
```

- The hash is the first 8 hex digits of SHA-256 over the English block after
  whitespace normalisation; a text that repeats in a file gets `-2`, `-3`, ...
  Together with the aligned revision it identifies the exact English text, so
  the English/translation pairs need no separate database.
- `l10n build` reconstructs each page from the English file at the aligned
  revision: translated blocks replace their English text, untranslated and
  stale blocks stay English, and all structure is copied. **Code, display
  math, comments and structure are never stored or edited in this
  repository**, so there is nothing to keep in sync and no cell IDs to
  reference. Code comments stay English **[ruled]**.
- `was=`/`from=` (and `src=` when the English file was renamed) mark a stale
  translation left by `l10n bump`; an update unit refreshes it.
- Math, labels (`:label:`), references (`:numref:`, `:ref:`, `:cite:`,
  `§{sec_x}`), inline code, image paths, URLs and HTML/Quarto attributes inside
  a block are byte-identical to English. `l10n check` enforces this.
- Figures stay English **[ruled]**. Their captions are in the `.md` files and
  are translated.
- `l10n status <lang>` reports per file the translated, stale and missing
  blocks and the share of English words covered.

## 4. Updating to a new English tag

`l10n bump <lang> --to en-YYYY.MM.DD` aligns each language file with the new
English file:

| English block at the new tag | Action |
|---|---|
| Hash matches an anchor | unchanged: keep the translation |
| Changed: no hash match, but aligned with an old block by position between unchanged neighbours, headings, labels and cell IDs | update packet: old English, new English, old translation; edit only what the English diff changed |
| New: no counterpart | new packet, with neighbouring translated blocks and glossary terms as context |
| Deleted | dropped |
| Code | nothing to do (inserted from the tag) |

Blocks are aligned with `difflib` over the block sequence, keyed by text hash
for prose, cell ID for code and label for `:label:` lines; inside a changed
region, old and new prose blocks are paired by word-level similarity (≥ 0.3,
same kind). Renamed or moved files are followed by git rename detection, then
by their `:label:`s. The result is a small diff per language, as surgical as
the English change, reviewed as one PR. `bump` writes the target to
`pinned_next` in `languages.yml` (all tools then work against it) and emits
the update work files; the pin moves only when `l10n check <lang> --promote`
finds every block translated and clean.

## 5. Work units: whole sections, not isolated blocks

Blocks are the unit of **alignment** (anchors, checks, updates), not the unit of
**translation**. A translator that sees one paragraph at a time cannot write a
cohesive narrative: it cannot see where the argument is going, which term it
chose three paragraphs ago, or that "this" refers to a figure above. So the
translator always works on a whole section and sees more around it.

- **Unit.** One `##` section of a page (or the whole page when it is short),
  typically 1,000-5,000 words, including the section's slides and exercises.
- **Work file.** `l10n emit <lang>` writes one Markdown work file per unit
  (format in `TRANSLATING.md`). It holds, in order:
  1. a brief: the language's `style.md`, the language's running decisions log
     (below), and the glossary entries for every concept in the unit;
  2. context to read, not translate: the chapter outline (English and, where
     it exists, translated headings), the whole English page, and the
     translation of the preceding section;
  3. the unit itself: anchored English blocks, each followed by an empty slot
     (new) or the current translation (update).
- **Answer.** The translator returns the unit as Markdown with the same anchors,
  one translated block per anchor. Markdown in, Markdown out: no JSON escaping
  of long text, and it works for chat models, CLI agents editing files, and
  people alike.
- **Decisions log** (`<lang>/decisions.md`). After each unit the translator
  appends the choices a later unit must follow: a term chosen without glossary
  support, how a recurring phrase is rendered, a convention for a construct.
  Every later work file includes it, so choices stay consistent across
  chapters and across agents.
- **Editorial pass.** After all units of a chapter are translated, an editor
  (normally a different agent from the translator) reads the whole translated
  chapter against the English for flow, transitions between sections, and
  consistent terms and register. It may edit any block, keeping the anchors.
- **Updates.** An update work file contains the whole section: unchanged blocks
  locked with their translation, changed blocks with old English, new English
  and old translation, new blocks empty. The translator edits only the open
  blocks but reads everything. If an English change requires a locked
  neighbour to change (a connective, a now-ambiguous "this"), it may edit that
  block and must say so; `l10n check` lists such edits for review.

Slides go to the unit whose section holds the code cells they show (the
`@cell-id` placeholders); a slide without cells follows the previous one.
Sections are merged until a unit has about 1,000 words (at most 5,000); a page
of at most 2,000 words is one unit.

`l10n apply <lang> <answers> --agent <who>` checks each answer (section 10) and
merges it; a unit with errors is rejected as a whole. Each unit records the
agent and model that produced it in `<lang>/units.jsonl`, so quality can be
compared across agents and languages.

## 6. Agents per language

Claude agents within the subscription are the default for now **[ruled]**, but
no language is tied to one agent: some languages may need Fable, some are fine
with Opus, others may do better with Gemini or Codex **[ruled]**.

- `TRANSLATING.md` is self-contained and written for any capable agent.
- Before a language's full run, a bake-off: the same sample units (three or
  four sections chosen to cover narrative prose, a derivation, slides and
  exercises, plus two update units) go to each candidate agent. Automated checks score preservation and
  terminology; Alex (or a native reviewer) judges accuracy and fluency blind.
  Results go to `<lang>/agents.md`, which also records the chosen agent.

## 7. Building and deploying one language

`l10n build <lang> [--html-only]`:

1. Check out d2l-neu at the pinned tag into `_build/<lang>` (a git worktree),
   with `.venv-*`, `.env` and `data` symlinked to the main checkout's.
2. Overwrite each translated chapter `.md` with the page reconstructed from
   the English file and `<lang>/book/` (section 3).
3. Set `D2L_LANG=<code>` and `D2L_LOCALE_FILE=<lang>/strings.yml` (created and
   updated from the tag's `i18n/en.yml` by `l10n strings <lang>`), which carry
   the language settings from `languages.yml`:
   Quarto `lang:` (localises Quarto's own UI and search), fonts (Noto Sans/Serif
   CJK, Devanagari, Arabic), `dir: rtl` for Arabic, the PDF preamble per script.
4. Run d2l-neu's no-execution pipeline (`make all-quick`; `make html` with
   `--html-only`): HTML, PDFs, slides, notebook zips. The
   outputs come from the English store (numbers and plots do not depend on the
   language); freshness stays clean because code cells are identical.
5. With a translation scope (section 13), step 2 also writes the placeholder
   pages and the English links, and `_book/` is post-processed after make.
6. The tag's `tools/gen_pdf.py` gives the PDF book no `output-file`, so Quarto
   names the `.tex` after the translated title and `build_one_pdf.sh` (which
   looks for `Dive-into-Deep-Learning.tex`) fails; the build adds
   `output-file: Dive-into-Deep-Learning` to the worktree's copy (until
   d2l-neu sets it itself).

`l10n deploy <lang>` uploads to bucket `d2l-<lang>` with the worktree's
`tools/upload_r2.sh --delete` (keys in `~/.config/cloudflare/d2l-buckets.env`;
the upload manifest is kept in `_state/` between builds). The only shared resource is the `d2l-data` bucket, used
through the `DATA_HUB` URLs and for the language menu.

## 8. Language menu and old editions

- The menu is loaded at runtime from `languages.json` in the `d2l-data` bucket
  **[ruled]**, generated from `languages.yml` by `l10n menu` (`--upload` puts it
  at `site/languages.json`): English, every language with a pinned tag, then
  the old editions. Each build carries a static copy
  as fallback. Adding a language does not require rebuilding the others.
- Order: current languages, a divider, then old editions ("English (old)" at
  en-v2.d2l.ai, later "中文 (old)" and others).
- Each language in `languages.json` carries `pages`, the list of HTML pages
  its site serves. Every real upload to a language bucket rewrites that
  language's list from the uploaded build (d2l-neu
  `tools/update_language_pages.py`, called by `upload_r2.sh` for `d2l-<code>`
  buckets, so `l10n deploy` and English deploys both update it); `l10n menu`
  keeps the published lists. A menu entry links to the same page in the other
  language when its list has the page, otherwise to that language's home
  page **[ruled]**. Nothing relies on a 404 page: the zone's Free plan has no
  error rule.

## 9. Discussions

- Every language links to the English forum topics (category 5, English
  Version) **[ruled]**.
- Exception: Chinese **[ruled]**. Chinese pages link to their own topics in the
  Chinese Version category (16); where a page has none, `l10n` creates one,
  following the English convention (one topic per page, first post = the page
  URL).

## 10. Checks

`l10n apply` checks every answer and `l10n check <lang>` every translated file.
Errors (an answer with an error is rejected):

- a missing or unknown marker in an answer; an anchor whose hash is not in the
  English file at the aligned revision; with `--complete` (and `--promote`),
  any block still missing or stale in a file inside the translation scope
  (section 13);
- protected tokens (section 3) of a class whose multiset differs from English;
  a heading level, list marker, `![`, `|` or kicker bracket that changed; a
  table row with another cell count; a slide/callout fence changed outside its
  `title` attribute;
- unbalanced `*`/`**` emphasis or brackets (where the English is balanced);
- a reference role or inline math set directly against a CJK letter
  (``:numref:`x`で``: pandoc does not recognise the role; full-width
  punctuation such as （） is fine);
- bidi control characters, a closing `$` followed by a digit.

Warnings: protected tokens in a different order (Japanese is verb-final; a NOTE
that names the block acknowledges it), numbers of the English missing from the
translation (digits only, so `2×` → 2 倍 and `40–55` → 40〜55 pass), list items
re-wrapped into a different marker count, an edited locked block without a
NOTE, and glossary violations: a `verified_*` or `invariant` term absent where
its English concept appears. `l10n check` also lists every locked block edited
at the current revision, for review. A language can change the switches in
`languages.yml` (`check: {cjk_adjacent: letters|all|off, order:
warning|error|off, numbers: warning|off, thousands: keep|space}`; `space`
accepts 10 000 for the English 10,000, as German writes it).

`typography: {nbsp: true}` in `languages.yml` (German) makes the build put
no-break spaces into abbreviations (z. B.), grouped numbers (10 000) and between
a number and its unit (24 GiB), outside code and math. Translators type ordinary
spaces; the check never sees the no-break ones.

A sampled back-translation check reports meaning drift for human review; it does
not fail the build.

## 11. Work needed in d2l-neu

1. UI strings: move hard-coded English UI strings into one catalog with a
   `D2L_LANG` switch (navbar, `_d2l-tabs.html`, slide index, PPTX and Beamer
   templates, PDF title pages, landing page).
2. Overlay builds: let the pipeline render a source tree that is not the git
   checkout itself.
3. `make release TAG=en-YYYY.MM.DD` with the checks of section 1.
4. The language menu from `languages.json` (runtime, with fallback).
5. Move the glossary (`docs/multilingual/`) to this repository (done: `glossary/`).

## 12. Rollout

1. Prove the pipeline: d2l-neu preparation, then one chapter of one language
   end to end (translate, check, build, deploy to a test bucket path).
2. First language: the whole book (HTML, PDF, slides), then launch.
3. Update cycle: cut a second English tag and run `bump`, to show that only the
   difference is translated.
4. Scale out to the other languages. Arabic (right-to-left PDF) and Hindi
   (Devanagari shaping) come last: they carry the most typesetting risk.
   Greek and Russian have no glossary yet.

## 13. Translation scope

A language need not translate the whole book. For now only the front matter
(preface, installation, notation, introduction, landing page), Basics,
Advanced, *Mathematics for Deep Learning* and Tools are translated
**[ruled]**; Language Models, Image Models and the Attic stay English.

**Configuration.** `scope:` at the top of `languages.yml` is the default for
every language; an entry may set its own (`scope: all`, or another list). It
names part directories (`chapter_basics`) or chapter directories
(`chapter_lm-agents`) of the pinned tag's `_quarto.yml`:

```yaml
scope: [chapter_preface, chapter_basics, chapter_advanced,
        chapter_mathematics-for-deep-learning, chapter_tools-for-deep-learning]
```

A part brings its part page and all its chapters; a chapter brings itself and
its part page. `index.md`, top-level pages and files outside the part
structure are always in scope. Membership is read from the parts of the tag's
`_quarto.yml`, never from chapter numbers, so renumbering does not change it;
a name that the tag does not have is an error. `l10n scope <lang>` prints it.

**Tools.** `emit`, `status`, `pseudo`, the update units of `bump` and
`check --complete`/`--promote` consider only in-scope files, so a release is
complete without the skipped parts (`emit --files` of an out-of-scope file is
an error). `check` without `--complete` still checks every translated file.

**Build** (`l10n build`, tools/l10n/scope.py):

- Each out-of-scope *chapter* renders as one placeholder page, its index page
  at the same path: the English chapter number, the translated title if
  `<lang>/book/` has one (else English), a "not yet translated" note in the
  language, and the chapter's section list with the English numbers and
  titles, linking to the English pages on d2l.ai (the `en` entry's URL). An
  out-of-scope part page is a placeholder that lists its chapters. Both carry
  `<meta name="robots" content="noindex">`.
- The section pages of out-of-scope chapters are not built: they are dropped
  from the worktree's `_quarto.yml`. Their sources stay on disk, so the d2l
  library, the generated notebooks and the freshness gate are those of
  English, and the remaining pages keep their `CHAPTER_NUMBERING` numbers
  (Linear Algebra stays 33, Tools 39).
- `:numref:`/`:ref:`/`:eqref:` in in-scope pages to labels defined in
  out-of-scope pages (chapter, section, subsection, figure, table, equation)
  become links to the English page and anchor, with the English number
  (Quarto's words in the language, numbered by the tag's own tools) and the
  marker, e.g. `[Abschnitt 18.4 (englisch)](https://d2l.ai/chapter_lm-pretrained/scoring.html)`.
  A narrow no-break space joins word and number, so the tag's
  `fix_crossref_numbers.py` does not renumber it as a local reference.
- The sidebar keeps the out-of-scope parts and chapters at their English
  numbers, each marked "(English)"; any other relative link to a page that is
  not built points to d2l.ai.
- PDFs leave the out-of-scope parts and chapters out entirely, through the
  tag's `PDF_EXCLUDED_FILES`; `fix_latex.py` pairs `\chapter`s with
  `PDF_CHAPTER_FILES` and sets each chapter's counter from `CHAPTER_NUMBERING`,
  so the remaining chapters keep their English numbers (d2l-neu
  build-system.md section 4.1). Part PDFs are built for in-scope parts only
  (`PDF_PARTS`) and the navbar drops the others. Slides skip out-of-scope
  chapters (`SLIDES_FILTER`), and the notebook zips are rewritten without them.
- The language menu's `pages` list (section 8) is computed from the uploaded
  `_book/`, so it has the placeholder pages and not the skipped section pages.

The tag's tools have no scope switch, so the build edits the worktree's copies
of `tools/d2l_preprocess.py` (adds the out-of-scope files to
`PDF_EXCLUDED_FILES`), `tools/gen_pdf.py` (its order guard skips the unbuilt
pages) and `tools/check_book_artifacts.py` (expects no deck or part PDF of an
out-of-scope chapter). Each edit replaces an exact anchor; if a new tag changes
one, the build stops and names the file (update `patches()` in scope.py).

**Strings.** The note texts and the marker are localization-side keys
(`l10n.english_marker`, `l10n.chapter_not_translated`,
`l10n.part_not_translated`), not in the tag's `i18n/en.yml`: `scope.py` holds
the English defaults and `l10n strings <lang>` adds them, marked TODO, at the
end of `<lang>/strings.yml`. The build passes the tag's tools a copy without
them (`D2L_LOCALE_FILE`), since the tag's `i18n.py` warns about unknown keys.

## First language

Japanese (`ja`) first **[ruled]**; Chinese next.
