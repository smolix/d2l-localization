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
                       #   fonts, direction, forum rule, old edition
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
- A menu entry links to the same page in the other language when that
  language's pinned tag has it, otherwise to that language's home page.

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
  any block still missing or stale;
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
warning|error|off, numbers: warning|off}`).

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
   Greek has no glossary yet.

## First language

Japanese (`ja`) first **[ruled]**; Chinese next.
