# d2l-localization

Translations of *Dive into Deep Learning*. The English book and all code live in
`smolix/d2l-neu`; this repository holds text only. Each language builds against
a tagged English release (`en-YYYY.MM.DD`) and is served from its own bucket
(`d2l-<code>`, host `<code>.d2l.ai`).

- [DESIGN.md](DESIGN.md): how translation, updates, builds and deploys work.
- [TRANSLATING.md](TRANSLATING.md): instructions for any translation agent or person.
- [languages.yml](languages.yml): the languages, their buckets, pinned English tags and chosen translators.
- [evals/](evals/README.md): fixed translation tests for choosing a translator per language (`bin/l10n compare`).

- [tools/l10n/](tools/l10n/): the `l10n` command (`bin/l10n`; Python 3.12+, PyYAML).
- [glossary/](glossary/): the terminology catalog (moved from d2l-neu).

## The workflow

```bash
bin/l10n pin ja en-2026.12.01           # first release a language builds against
bin/l10n emit ja --files chapter_linear-regression --units 4   # work files in work/ja/
bin/l10n apply ja answers/*.md --agent "claude fable-5.2"      # check + merge into ja/book/
bin/l10n check ja                       # every translated file; --complete for a release
bin/l10n status ja                      # coverage per file
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
