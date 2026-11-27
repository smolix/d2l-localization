# Translation evaluations

Each directory here is a fixed translation test: the same work files go to
several translators (models or people), and `bin/l10n compare` puts their
answers side by side. Use it to choose the translator for a language before a
large run.

| Directory | Language | Page | Result |
|---|---|---|---|
| [zh-conv-layer](zh-conv-layer/) | 中文 (zh) | the same page | Fable 5.2, Opus 5.5, Sonnet 5.5 translated 2026-11-26; awaiting a native reader; see [RESULTS.md](zh-conv-layer/RESULTS.md) |
| [ko-conv-layer](ko-conv-layer/) | 한국어 (ko) | the same page | Fable 5.2, Opus 5.5, Sonnet 5.5 translated 2026-11-26; awaiting a native reader; see [RESULTS.md](ko-conv-layer/RESULTS.md) |
| [es-conv-layer](es-conv-layer/) | Español (es) | the same page | Fable 5.2, Opus 5.5, Sonnet 5.5 translated 2026-11-26; awaiting a native reader; see [RESULTS.md](es-conv-layer/RESULTS.md) |
| [fr-conv-layer](fr-conv-layer/) | Français (fr) | the same page | Fable 5.2, Opus 5.5, Sonnet 5.5 translated 2026-11-26; awaiting a native reader; see [RESULTS.md](fr-conv-layer/RESULTS.md) |
| [pt-conv-layer](pt-conv-layer/) | Português (pt) | the same page | Fable 5.2, Opus 5.5, Sonnet 5.5 translated 2026-11-26; awaiting a native reader; see [RESULTS.md](pt-conv-layer/RESULTS.md) |
| [tr-conv-layer](tr-conv-layer/) | Türkçe (tr) | the same page | Fable 5.2, Opus 5.5, Sonnet 5.5 translated 2026-11-26; awaiting a native reader; see [RESULTS.md](tr-conv-layer/RESULTS.md) |
| [vi-conv-layer](vi-conv-layer/) | Tiếng Việt (vi) | the same page | Fable 5.2, Opus 5.5, Sonnet 5.5 translated 2026-11-26; awaiting a native reader; see [RESULTS.md](vi-conv-layer/RESULTS.md) |
| [hi-conv-layer](hi-conv-layer/) | हिन्दी (hi) | the same page | Fable 5.2, Opus 5.5, Sonnet 5.5 translated 2026-11-26; awaiting a native reader; see [RESULTS.md](hi-conv-layer/RESULTS.md) |
| [ar-conv-layer](ar-conv-layer/) | العربية (ar) | the same page | Fable 5.2, Opus 5.5, Sonnet 5.5 translated 2026-11-26; awaiting a native reader; see [RESULTS.md](ar-conv-layer/RESULTS.md) |
| [ru-conv-layer](ru-conv-layer/) | Русский (ru) | the same page | Fable 5.2, Opus 5.5, Sonnet 5.5 translated 2026-11-26 (new glossary); awaiting a native reader; see [RESULTS.md](ru-conv-layer/RESULTS.md) |
| [el-conv-layer](el-conv-layer/) | Ελληνικά (el) | the same page | Fable 5.2, Opus 5.5, Sonnet 5.5 translated 2026-11-26 (new glossary); awaiting a native reader; see [RESULTS.md](el-conv-layer/RESULTS.md) |
| [ja-conv-layer](ja-conv-layer/) | Japanese (ja) | the same page | Gemini 4 Carbon chosen for the Japanese translation (author, 2026-11-27); style guide reviewed by a native speaker; see [RESULTS.md](ja-conv-layer/RESULTS.md) |
| [de-conv-layer](de-conv-layer/) | German (de) | Convolutions for Images (`chapter_convolutional-neural-networks/conv-layer.md`, 3 units, about 4,200 words) | Codex GPT-6.2 Astra chosen for the German translation; Fable 5.2 was the best of the Claude models (author, 2026-11-26); see [RESULTS.md](de-conv-layer/RESULTS.md) |

## Running a test with another agent

The agent needs this repository. The work files contain everything it
translates from: the English page, the style guide, the glossary entries and
the context. The format check (`l10n apply --fresh`, a dry run against an untranslated page) and `l10n compare` also
read the English page from a checkout of the English book (smolix/d2l-neu, at
`$D2L_NEU`, default `/home/smola/d2l-neu`). An agent without that checkout
skips the check, and the comparison runs it for every agent afterwards.

1. **Pick a name** for the agent, lower case without spaces: `gemini-3-pro`,
   `codex-gpt-6`, `human-nn`. Its answers go to
   `evals/<test>/answers-<name>/`.
2. **Give it the prompt** in `evals/<test>/PROMPT.md`, with `{NAME}` replaced by
   that name. Use the agent's default settings unless the test is about
   settings (the stored answers used medium effort). Start a fresh session, so
   that nothing else is in its context.
3. **Let it finish.** It writes one answer file per work file. The prompt
   allows it to run the format check (`bin/l10n apply ... --fresh`) and fix
   errors; it must not edit anything else.
4. **Compare:**

   ```bash
   bin/l10n compare de evals/de-conv-layer        # writes evals/de-conv-layer/compare.html
   bin/l10n compare all evals                     # every test plus an overview, in evals/_compare/
   ```

   It checks every `answers-*/` directory and writes one HTML page: every
   English block beside each translation, the error and warning counts, the
   length against English, and the DECISIONS and NOTES each translator
   recorded. Open it in a browser (it loads Markdown and math renderers from
   cdnjs). For Japanese and Chinese, length is counted in characters.
5. **Read, do not count.** The check counts measure rule-following (format,
   protected tokens, glossary terms), not quality. Read at least ten blocks
   side by side, including a heading, a dense paragraph with math, a list, an
   exercise and a slide; a native speaker decides.
6. **Record the result** in the test's `RESULTS.md` (date, agents, settings,
   counts, the reader's verdict) and commit the new `answers-<name>/`
   directory, so the next comparison includes it. `compare.html` is
   generated; do not commit it.

## Adding a test

```bash
bin/l10n emit <lang> --rev <English commit or tag> --files <chapter/file.md> --out evals/<lang>-<name>
cp evals/de-conv-layer/PROMPT.md evals/<lang>-<name>/   # then adjust the paths and the language
```

Keep the work files once committed: they pin the style guide, glossary and
English text the translators saw. A test emitted again later gives different
inputs and starts a new test.
