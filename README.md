# d2l-localization

Translations of *Dive into Deep Learning*. The English book and all code live in
`smolix/d2l-neu`; this repository holds text only. Each language builds against
a tagged English release (`en-YYYY.MM.DD`) and is served from its own bucket
(`d2l-<code>`, host `<code>.d2l.ai`).

- [DESIGN.md](DESIGN.md): how translation, updates, builds and deploys work.
- [TRANSLATING.md](TRANSLATING.md): instructions for any translation agent or person.
- [languages.yml](languages.yml): the languages, their buckets, pinned English tags, chosen translators and the translation scope.
- [evals/](evals/README.md): fixed translation tests for choosing a translator per language (`bin/l10n compare`).

- [tools/l10n/](tools/l10n/): the `l10n` command (`bin/l10n`; Python 3.12+, PyYAML).
- [glossary/](glossary/): the terminology catalog (moved from d2l-neu).
- `<code>/style.md`: each language's style guide, read by translators with every unit.

## Native-speaker review of the style guides

German (`de/style.md`) is confirmed. Japanese (`ja/style.md`) was reviewed by a
native speaker (PR #2, 2026-11-27: だ・である body text, 〜せよ exercises, the
other proposals accepted; its §9 records the decisions). Hindi (`hi/style.md`)
had its terminology reviewed by a native speaker (2026-11-27: terms of art in
Devanagari, never coined, with the English at first use; image = इमेज; its §14
records the decisions; the rest is still a draft). Every other style guide is a draft written on
2026-11-26 by Fable 5.2 from TRANSLATING.md, the glossary and the translation
tests in `evals/`; no native speaker has reviewed it yet. Each guide ends with
its settings, a numbered "For the author to confirm" list (each question has a
proposed answer) and proposed glossary corrections. Native speakers are asked to:

1. Answer the numbered questions at the end of their language's guide.
2. Correct any rule that is wrong for technical writing in that language,
   keeping the shape: one decision, one reason, one example.
3. Open a pull request against `<code>/style.md`, saying which variety and
   publishing tradition the corrections follow.

[STYLE_FOLLOWUPS.md](STYLE_FOLLOWUPS.md) lists the build and checker changes the
guides propose; none is applied yet.

## The workflow

```bash
bin/l10n pin ja en-2026.12.01           # first release a language builds against
bin/l10n emit ja --files chapter_linear-regression --units 4   # work files in work/ja/
bin/l10n apply ja answers/*.md --agent "claude fable-5.2"      # check + merge into ja/book/
bin/l10n check ja                       # every translated file; --complete for a release
bin/l10n fold ar lines.md                # merge lexicon lines into ar/decisions.md (chapter check)
bin/l10n status ja                      # coverage per file (inside the translation scope)
bin/l10n scope ja                       # which parts/chapters are translated (languages.yml scope:)
bin/l10n strings ja                     # ja/strings.yml from the tag's i18n/en.yml
bin/l10n build ja --html-only           # d2l-neu worktree in _build/ja, make html
bin/l10n deploy ja                      # upload to bucket d2l-ja
bin/l10n bump ja --to en-2027.01.15     # align with a new English tag, emit update units
bin/l10n check ja --promote             # after the update: move the pin
bin/l10n menu --upload                  # site/languages.json for the language menu
bin/l10n compare de evals/de-conv-layer # side-by-side page of several agents' translations
```

`bin/l10n pseudo xx` fills the test language `xx` with every block wrapped in
⟦…⟧, for testing the pipeline. Tests: `cd tools && python3 -m unittest discover
-s l10n/tests -t .` (`bin/l10n roundtrip` checks that segmenting and rebuilding
every English page is byte-identical).

Status: tools built; no language is translated yet.
