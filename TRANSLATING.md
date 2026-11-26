# How to translate *Dive into Deep Learning*

These instructions are for any translator: a Claude agent (Fable, Opus), Gemini,
Codex, another model, or a person. They are self-contained. Read all of them
before the first packet, and read the language's `style.md` and the glossary
excerpt that comes with each packet.

The book teaches deep learning to students and engineers. Its English is plain,
direct and precise, and the translation must be the same: a reader of the
translation should learn exactly what a reader of the English learns, in natural
prose of their own language.

## 1. What you receive and what you return

You work on **packets**, one JSON object per line (JSONL). You never see or edit
whole files; the tools assemble the book.

### Input packet

```json
{
  "id": "ja/chapter_optimization/sgd.md#14",
  "lang": "ja",
  "mode": "new",
  "kind": "paragraph",
  "anchor": "3f2a91c0",
  "en": "English text of the block, in Markdown.",
  "context_before": [{"en": "...", "tr": "..."}],
  "context_after": [{"en": "...", "tr": "..."}],
  "glossary": [{"en": "stochastic gradient descent", "tr": "確率的勾配降下法", "status": "verified_d2l"}],
  "old_en": null,
  "old_tr": null
}
```

- `mode` is `new` (translate `en`) or `update` (the English changed: edit
  `old_tr` so it matches `en`; see section 5).
- `kind` is one of `heading`, `paragraph`, `list_item`, `table_row`, `caption`,
  `exercise`, `slide_title`, `slide_text`, `kicker`, `tab_note`, `note`.
- `context_before` / `context_after` hold up to three neighbouring blocks, with
  their translation when one exists. Use them for continuity (terms already
  chosen, pronoun reference, the flow of an argument). Never translate them.
- `glossary` lists the glossary entries whose English concept occurs in `en`.

### Output answer

Return exactly one JSON object per input packet, in the same order:

```json
{"id": "ja/chapter_optimization/sgd.md#14", "anchor": "3f2a91c0", "tr": "Translated block.", "notes": ""}
```

- `tr` is the translated block in Markdown, and nothing else: no comments, no
  quotation marks around it, no explanation.
- `notes` is empty unless something needs a human: an ambiguous source, a
  glossary term that does not fit the context, an error in the English, a term
  you had to choose without glossary support. One or two sentences.
- Copy `id` and `anchor` verbatim. Do not add fields. The tool adds the agent and
  model name.

## 2. Never change these (protected tokens)

The tools check every answer against the English and reject it if any of these
differ. Copy them byte for byte, in the same order:

| Construct | Example |
|---|---|
| Inline and display math | `$\mathbf{w}$`, `$$ … $$`, `\(…\)` |
| Inline code | `` `torch.Tensor` ``, `` `num_epochs` `` |
| Cross-references and labels | ``:numref:`sec_sgd` ``, ``:eqref:`eq_loss` ``, ``:label:`fig_x` ``, ``:eqlabel:`…` ``, ``:exlabel:`…` ``, ``:exref:`…` ``, ``:cite:`…` ``, ``:citet:`…` `` |
| Section references | `§{sec_sgd}`, `§{num:sec_sgd}` |
| Slide placeholders | `@sgd-update-1`, `@!sgd-update-1[out=2]` |
| Quarto div fences and attributes | `::: {.cols .vc}`, `::: {.col .fig}`, `::: {.d2l-note .warn}`, `{.kicker}`, `{.unnumbered}` |
| Framework tab markers | ``:begin_tab:`pytorch` ``, `:end_tab:` |
| Image paths, URLs, link targets | `![…](../img/sgd.svg)`, `[…](https://…)` |
| HTML tags and comments | `<span class="…">`, `<!-- … -->` |
| Numbers that are results, units, identifiers | `92.3%`, `24 GiB`, `RTX 4090`, `GPT-2` |

What you **do** translate inside these constructs: the visible text of a link
(`[this text](url)`), image captions (`![this text](path)`), the text of a
kicker (`[this text]{.kicker}`), and a slide title attribute
(`::: {.slide title="This Text"}`), keeping its quotes and the rest of the
attribute list unchanged.

Inline code stays English even when it reads like a word (`` `loss` ``). Variable
names in math stay as they are. Do not convert between number formats (keep
`0.5`, not `0,5`) unless `style.md` says otherwise for that language.

## 3. Faithfulness

- Translate meaning, not words. Restructure sentences as your language needs,
  but add nothing and drop nothing: no extra explanations, no softened claims, no
  omitted caveats, no summaries.
- Keep the block's scope: one English block becomes one translated block. Do not
  merge with or split into neighbouring blocks. Within a block you may split or
  join sentences.
- Keep the logical connectives exactly (because, therefore, however, only if,
  at most). They carry the argument.
- Keep quantities, inequalities, signs and orders of magnitude exact. "About
  2×" stays "about 2×", "at most 15 lines" stays "at most".
- Keep the English emphasis (`*…*`, `**…**`) on the corresponding words.
- If the English looks wrong (a typo, a contradiction, a broken sentence),
  translate what it evidently means and say so in `notes`.

## 4. Terminology

The glossary entries in the packet decide technical terms:

| Status | What you do |
|---|---|
| `verified_d2l`, `verified_external`, `invariant` | Use `tr` exactly (inflect only as grammar requires). |
| `proposed` | Use `tr` unless it is wrong in this context; then choose better and explain in `notes`. |
| `unresolved` or no entry | Choose the established term in your language's ML literature; explain in `notes` on first use in the packet. |

- Use the same term for the same concept throughout; check `context_before`.
- On the first use of a key term in a section, follow `style.md` on whether to
  add the English in parentheses, e.g. 確率的勾配降下法（stochastic gradient
  descent）.
- Names of models, libraries, datasets and people stay as in English (`ResNet`,
  `PyTorch`, `ImageNet`), unless `style.md` lists a transliteration.
- Keep acronyms (SGD, MLP, KV cache) unless `style.md` or the glossary says
  otherwise.

## 5. Update packets

When `mode` is `update`, the English block changed between two releases. You get
the old English (`old_en`), the new English (`en`) and the old translation
(`old_tr`).

1. Find what changed between `old_en` and `en`.
2. Edit `old_tr` so it says what `en` says, changing only the parts that
   correspond to the English change.
3. Leave every sentence whose English did not change byte-identical, even if you
   would phrase it differently. Reviewers read the diff; unrelated edits hide the
   real change.
4. If the English change is formatting only (emphasis, a reference, a number),
   make only that change.
5. If the English was rewritten so much that editing is not sensible, translate
   `en` fresh and write `rewritten` in `notes`.

## 6. Kinds of block

- `heading`: short, noun-phrase style as usual in your language's technical
  books; no full stop. Keep `{.unnumbered}` and other attributes.
- `caption`: a complete caption; keep `(a)`/`(b)` panel markers.
- `exercise`: imperative, as in the English. Keep the numbering and
  `:exlabel:` markers.
- `slide_title`, `slide_text`, `kicker`: slides are read at a distance. Keep
  them as short as the English; a bullet stays one line where possible.
- `tab_note`: a note that appears only for one framework (between
  `:begin_tab:` and `:end_tab:`). Translate the text; keep the markers.
- `note`: the text of a `::: {.d2l-note}` box.
- `table_row`: translate cell text; keep the `|` structure and alignment row.

## 7. Style

Each language has a `style.md` (register, punctuation, spacing, how to write
English terms in parentheses, numbers and units). Follow it. General rules:

- Register: the book addresses the reader as a colleague ("we" do things
  together). Use the register `style.md` names; be consistent.
- Write natural prose a native-speaking engineer would write, not translationese.
- Do not insert Unicode direction marks or other invisible characters; the
  renderer handles right-to-left text and embedded math.
- In CJK languages, follow `style.md` on full-width punctuation and on spacing
  between CJK text and Latin letters, numbers or math.

## 8. Before you return an answer, check

1. Every protected token of `en` appears in `tr`, unchanged, in the same order.
2. Nothing added, nothing dropped; connectives, quantities and negations intact.
3. Glossary terms used as their status requires.
4. Markdown still balanced: emphasis, brackets, links, quotes in attributes.
5. One block in, one block out; no text outside the JSON.
6. For updates: unchanged sentences are byte-identical to `old_tr`.

## 9. Examples (Japanese)

New block:

```json
{"id": "ja/chapter_optimization/sgd.md#14", "lang": "ja", "mode": "new", "kind": "paragraph", "anchor": "3f2a91c0",
 "en": "We minimize the loss with *stochastic gradient descent* (:numref:`sec_sgd`), which updates $\\mathbf{w}$ after every minibatch.",
 "glossary": [{"en": "stochastic gradient descent", "tr": "確率的勾配降下法", "status": "verified_d2l"},
              {"en": "minibatch", "tr": "ミニバッチ", "status": "verified_d2l"}]}
```

```json
{"id": "ja/chapter_optimization/sgd.md#14", "anchor": "3f2a91c0", "tr": "損失は*確率的勾配降下法*（:numref:`sec_sgd`）で最小化します。この方法では、ミニバッチごとに $\\mathbf{w}$ を更新します。", "notes": ""}
```

Update block (the English added the batch size):

```json
{"id": "ja/chapter_optimization/sgd.md#14", "lang": "ja", "mode": "update", "kind": "paragraph", "anchor": "8c11d0e4",
 "old_en": "We minimize the loss with *stochastic gradient descent* (:numref:`sec_sgd`), which updates $\\mathbf{w}$ after every minibatch.",
 "en": "We minimize the loss with *stochastic gradient descent* (:numref:`sec_sgd`), which updates $\\mathbf{w}$ after every minibatch of 256 examples.",
 "old_tr": "損失は*確率的勾配降下法*（:numref:`sec_sgd`）で最小化します。この方法では、ミニバッチごとに $\\mathbf{w}$ を更新します。"}
```

```json
{"id": "ja/chapter_optimization/sgd.md#14", "anchor": "8c11d0e4", "tr": "損失は*確率的勾配降下法*（:numref:`sec_sgd`）で最小化します。この方法では、256 個の例からなるミニバッチごとに $\\mathbf{w}$ を更新します。", "notes": ""}
```

The first sentence is unchanged byte for byte; only the phrase that corresponds
to the English change was edited.
