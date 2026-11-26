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
  tools/l10n/          # CLI: status, extract, bump, translate, check, build, deploy
  templates/style.md   # starting point for a language's style guide
  <lang>/
    book/              # translated .md files at the English paths (chapter_*/...)
    strings.yml        # UI strings (menu, buttons, template text)
    style.md           # this language's style guide (register, punctuation, spacing)
    agents.md          # which agent translated what, and review notes (section 6)
```

Language codes are ISO 639-1 (`ja`, not `jp`); the bucket is `d2l-<code>` and
the host `<code>.d2l.ai`. English is also served at `d2l.ai`.

## 3. What a translated file looks like

A translated file mirrors its English file block for block. A block is a
heading, paragraph, list item, table row, figure caption, exercise item, or the
text of a slide. Before each translated block sits an anchor with a hash of the
English block it translates:

```
<!--en:3f2a91c0-->
## 損失関数
```

- The anchor is an HTML comment, stripped at render time.
- The hash is the first 8 hex digits of SHA-256 over the English block after
  whitespace normalisation. Together with the pinned tag it identifies the exact
  English text, so the English/translation pairs need no separate database.
- **Code is not stored.** A Python cell appears only as a reference by its
  English cell ID, e.g. `@cell rcnn-fast-r-cnn-1`; the build inserts the code
  from the tag. Other fences (shell, plain text) are copied by the tools, never
  edited by translators. Code comments stay English **[ruled]**.
- Math, labels (`:label:`), references (`:numref:`, `:ref:`, `:cite:`,
  `§{sec_x}`), slide placeholders (`@id`, `@!id[out=1]`), image paths, URLs and
  HTML/Quarto attributes are byte-identical to English. `l10n check` enforces
  this.
- Figures stay English **[ruled]**. Their captions are in the `.md` files and
  are translated.

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

Renamed or moved files are followed through their section labels. The result
is a small diff per language, as surgical as the English change, reviewed as
one PR. The pin in `languages.yml` moves only after `l10n check` passes.

## 5. Translation packets (agent-independent)

Translation runs on packets so that any agent can do it (Claude Fable or Opus,
Gemini, Codex, a person). `l10n translate --emit` writes JSONL packets;
the agent returns JSONL answers; `l10n translate --apply` checks and merges
them. The exact formats and rules are in `TRANSLATING.md`. Each answer carries
the agent and model that produced it, recorded per block, so translation
quality can be compared across agents and languages.

## 6. Agents per language

Claude agents within the subscription are the default for now **[ruled]**, but
no language is tied to one agent: some languages may need Fable, some are fine
with Opus, others may do better with Gemini or Codex **[ruled]**.

- `TRANSLATING.md` is self-contained and written for any capable agent.
- Before a language's full run, a bake-off: the same sample packets (about 30
  blocks across prose, math-heavy text, slides and exercises, plus 10 update
  packets) go to each candidate agent. Automated checks score preservation and
  terminology; Alex (or a native reviewer) judges accuracy and fluency blind.
  Results go to `<lang>/agents.md`, which also records the chosen agent.

## 7. Building and deploying one language

`l10n build <lang>`:

1. Check out d2l-neu at the pinned tag into a scratch directory (git worktree).
2. Overlay `<lang>/book/`, inserting code by cell ID and dropping anchors.
3. Apply `<lang>/strings.yml` and the language settings from `languages.yml`:
   Quarto `lang:` (localises Quarto's own UI and search), fonts (Noto Sans/Serif
   CJK, Devanagari, Arabic), `dir: rtl` for Arabic, the PDF preamble per script.
4. Run d2l-neu's no-execution pipeline: HTML, PDFs, slides, notebook zips. The
   outputs come from the English store (numbers and plots do not depend on the
   language); freshness stays clean because code cells are identical.

`l10n deploy <lang>` uploads to bucket `d2l-<lang>` with d2l-neu's
`tools/upload_r2.sh` (keys in `~/.config/cloudflare/d2l-buckets.env`, one
manifest per bucket). The only shared resource is the `d2l-data` bucket, used
through the `DATA_HUB` URLs and for the language menu.

## 8. Language menu and old editions

- The menu is loaded at runtime from `languages.json` in the `d2l-data` bucket
  **[ruled]**, generated from `languages.yml`; each build carries a static copy
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

`l10n check <lang>` fails on:

- a block without an anchor, an anchor whose hash is not in the pinned English
  file, or a block count that differs from English;
- any change to protected tokens (section 3);
- unbalanced Markdown emphasis, links or Quarto divs;
- glossary violations: a verified or invariant term not used where its English
  concept appears;
- characters that break the PDF build (bidi control characters, a `$` followed
  by a digit).

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
5. Move the glossary (`docs/multilingual/`) to this repository.

## 12. Rollout

1. Prove the pipeline: d2l-neu preparation, then one chapter of one language
   end to end (translate, check, build, deploy to a test bucket path).
2. First language: the whole book (HTML, PDF, slides), then launch.
3. Update cycle: cut a second English tag and run `bump`, to show that only the
   difference is translated.
4. Scale out to the other languages. Arabic (right-to-left PDF) and Hindi
   (Devanagari shaping) come last: they carry the most typesetting risk.
   Greek has no glossary yet.

## Open

- First language: zh or ja.
