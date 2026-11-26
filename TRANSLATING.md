# How to translate *Dive into Deep Learning*

These instructions are for any translator: a Claude agent (Fable, Opus), Gemini,
Codex, another model, or a person. They are self-contained. Read all of them
before your first unit.

The book teaches deep learning to students and engineers. Its English is plain,
direct and precise, and its sections build an argument: a question, an idea, a
derivation, an experiment, a conclusion. The translation must do the same. A
reader of the translation should learn exactly what a reader of the English
learns, from prose that reads as if it had been written in their language.

## 1. The work file

You translate one **unit** at a time: a whole section of a page (or a whole
short page), with its slides and exercises. You get it as a Markdown work file
with three parts.

```
=== BRIEF ===
(the language's style.md, the decisions log, the glossary entries for this unit)

=== CONTEXT (read, do not translate) ===
(the chapter outline, the whole English page, the translation of the previous section)

=== UNIT ===
<!--unit:chapter_optimization/sgd.md#u02 lang=ja rev=en-2026.12.01-->

<!--en:3f2a91c0 kind=heading-->
## Stochastic Gradient Descent
<!--tr:3f2a91c0-->

<!--en:8c11d0e4 kind=paragraph-->
We minimize the loss with *stochastic gradient descent* ...
<!--tr:8c11d0e4-->

<!--en:1b77a2f9 kind=caption locked-->
English text of a block that did not change.
<!--tr:1b77a2f9 locked-->
Its existing translation.
```

- The `<!--unit:…-->` line names the page, the unit and the English release.
- Each English block sits under an `<!--en:…-->` marker; its translation goes
  under the matching `<!--tr:…-->` marker. In a new unit the translation slots
  are empty.
- `kind` tells you what the block is (section 6).
- `locked` marks a block whose English did not change (update units only;
  section 5).
- Read the BRIEF and the CONTEXT before you start. They exist so you can write
  one coherent text: where the argument is going, how the previous section
  ended, which terms are already fixed.

## 2. What you return

Return the UNIT part only, with every marker line copied exactly and each
translation slot filled:

```
<!--unit:chapter_optimization/sgd.md#u02 lang=ja rev=en-2026.12.01-->

<!--en:3f2a91c0 kind=heading-->
## Stochastic Gradient Descent
<!--tr:3f2a91c0-->
## 確率的勾配降下法
```

Then, after the unit, two optional sections:

```
=== DECISIONS ===
- "warm-up" → 「ウォームアップ」 (no glossary entry; katakana is standard in ML texts)

=== NOTES ===
- 8c11d0e4: the English says "at least", the formula says "at most"; translated the formula's meaning.
```

- **DECISIONS** lists choices a later unit must follow: a term chosen without
  glossary support, the rendering of a recurring phrase, a convention for a
  construct. The tools append them to the language's decisions log.
- **NOTES** lists anything a human should look at, by block hash: an ambiguous
  or wrong English source, a glossary term that does not fit, an edit to a
  locked block. Leave it out when there is nothing to say.
- Write nothing else: no preface, no summary of what you did.

## 3. Writing a cohesive translation

- Translate the unit as one text. Read the whole English unit first, then
  write. Choose transitions, pronouns and sentence order so the section flows
  in your language, and so its first sentence connects to how the previous
  section ended (CONTEXT).
- **One block in, one block out.** Each English block has exactly one
  translated block. Within a block you may reorder, split or join sentences as
  your language needs. Do not move content between blocks: the alignment is
  what lets later updates touch only what changed.
- Translate meaning, not words. Add nothing and drop nothing: no extra
  explanations, no softened claims, no omitted caveats.
- Keep the logical connectives exact (because, therefore, however, only if,
  at most). They carry the argument.
- Keep quantities, inequalities, signs and orders of magnitude exact. "About
  2×" stays "about 2×".
- Keep the English emphasis (`*…*`, `**…**`) on the corresponding words.
- If the English looks wrong (a typo, a contradiction), translate what it
  evidently means and say so in NOTES.

## 4. Never change these (protected tokens)

The tools check every block against its English and reject the unit if any of
these differ. Copy them byte for byte; keep their order unless your
language's word order requires another (then name the block in NOTES):

| Construct | Example |
|---|---|
| Inline and display math | `$\mathbf{w}$`, `$$ … $$`, `\(…\)` |
| Inline code | `` `torch.Tensor` ``, `` `num_epochs` `` |
| Cross-references and labels | ``:numref:`sec_sgd` ``, ``:eqref:`eq_loss` ``, ``:label:`fig_x` ``, ``:eqlabel:`…` ``, ``:exlabel:`…` ``, ``:exref:`…` ``, ``:cite:`…` ``, ``:citet:`…` `` |
| Section references | `§{sec_sgd}`, `§{num:sec_sgd}` |
| Quarto div fences and attributes | `::: {.cols .vc}`, `::: {.col .fig}`, `::: {.d2l-note .warn}`, `{.kicker}`, `{.unnumbered}` |
| Framework tab markers | ``:begin_tab:`pytorch` ``, `:end_tab:` |
| Image paths, URLs, link targets | `![…](../img/sgd.svg)`, `[…](https://…)` |
| HTML tags and comments | `<span class="…">`, `<!-- … -->` |
| Results, units, identifiers | `92.3%`, `24 GiB`, `RTX 4090`, `GPT-2` |

What you **do** translate inside these constructs: the visible text of a link
(`[this text](url)`), image captions (`![this text](path)`), the text of a
kicker (`[this text]{.kicker}`), and a slide title attribute
(`::: {.slide title="This Text"}`), keeping its quotes and the rest of the
attribute list unchanged.

Inline code stays English even when it reads like a word (`` `loss` ``). Code,
display equations and slide placeholders are never in the unit; read them in
the English page in CONTEXT. Keep `0.5` as written unless `style.md` says
otherwise for prose numbers.

Never set a reference (``:numref:`…` ``, ``:eqref:`…` ``, ``:cite:`…` ``) or
inline math `$…$` directly against a CJK letter: put a space, or full-width
punctuation such as （）, between them (``:numref:`sec_sgd` で``, not
``:numref:`sec_sgd`で``); otherwise the build does not recognise the reference.

## 5. Update units

When the English book moves to a new release, only changed blocks are
translated again. An update unit is still the whole section, so you can see the
context, but most blocks are `locked`.

For an open (changed) block you get three things:

```
<!--en:8c11d0e4 kind=paragraph old-->
Old English text.
<!--en:a90f13c2 kind=paragraph-->
New English text.
<!--tr:a90f13c2 was=8c11d0e4-->
The old translation (of the old English).
```

1. Find what changed between the old and the new English.
2. Edit the old translation so it says what the new English says, changing only
   the parts that correspond to the English change.
3. Leave every sentence whose English did not change byte-identical, even if you
   would phrase it differently now. Reviewers read the diff.
4. If the English change is formatting only (emphasis, a reference, a number),
   make only that change.
5. If the English was rewritten so much that editing makes no sense, translate
   it fresh and write `rewritten` for that block in NOTES.
6. New blocks have an empty slot: translate them so they fit their neighbours.
7. Do not edit `locked` blocks, with one exception: when the change makes a
   locked neighbour wrong (a connective that no longer fits, a "this" that now
   points elsewhere), edit it minimally and list it in NOTES.

## 6. Kinds of block

- `heading`: short, noun-phrase style as usual in your language's technical
  books; no full stop. Keep `{.unnumbered}` and other attributes.
- `paragraph`, `list_item`: body text.
- `caption`: a complete caption; keep `(a)`/`(b)` panel markers.
- `exercise`: imperative, as in the English. Keep `:exlabel:` markers.
- `slide_title`, `slide_text`, `kicker`: slides are read at a distance. Keep
  them as short as the English; a bullet stays one line where possible. A slide
  must still make sense on its own.
- `tab_note`: text shown for one framework only (between `:begin_tab:` and
  `:end_tab:`). Translate the text; keep the markers.
- `note`: the text of a `::: {.d2l-note}` box.
- `table_row`: translate cell text; keep the `|` structure and alignment row.

## 7. Terminology

The glossary entries in the BRIEF decide technical terms:

| Status | What you do |
|---|---|
| `verified_d2l`, `verified_external`, `invariant` | Use the given term (inflect only as grammar requires). |
| `proposed` | Use it unless it is wrong in context; then choose better and record it in DECISIONS. |
| `unresolved` or no entry | Use the established term in your language's ML literature and record it in DECISIONS. |

- The decisions log in the BRIEF is binding like a verified glossary entry.
- Use one term per concept throughout. Senses matter: statistical *bias* and a
  layer's *bias*, a convolution *kernel* and a covariance *kernel*, an RL
  *return* and a function's return value are different concepts.
- On the first use of a key term in a section, follow `style.md` on adding the
  English in parentheses, e.g. 確率的勾配降下法（stochastic gradient descent）.
- Names of models, libraries, datasets and people stay as in English (`ResNet`,
  `PyTorch`, `ImageNet`) unless `style.md` lists a transliteration.

## 8. Style

Each language's `style.md` sets register, punctuation, spacing, English terms in
parentheses, numbers and units. Follow it. In general:

- The book addresses the reader as a colleague ("we" work through things
  together). Keep one register throughout.
- Write natural prose that a native-speaking engineer would write, not
  translationese.
- Do not insert Unicode direction marks or other invisible characters; the
  renderer handles right-to-left text and embedded math.

## 9. Before you return the unit, check

1. Every marker line is present and unchanged; every slot is filled.
2. Every protected token of each English block appears in its translation,
   unchanged (reordered only where grammar requires it, and then named in NOTES).
3. Nothing added or dropped; connectives, quantities and negations intact.
4. Terms follow the glossary and the decisions log; new choices are in
   DECISIONS.
5. Markdown is balanced: emphasis, brackets, links, quotes in attributes.
6. Update units: locked blocks untouched (or listed in NOTES); unchanged
   sentences in open blocks byte-identical.
7. The unit reads as one text, and its opening follows from the previous
   section.

If you can run `bin/l10n apply <lang> <answer files> --dry-run`, run it and fix
every error. A warning can be a false match (a glossary check that takes the
verb "returns" for the reinforcement-learning term *return*). Never change the
translation only to silence a warning: write the correct translation and name
the false match in NOTES.

## 10. Example (Japanese, update)

```
<!--en:8c11d0e4 kind=paragraph old-->
We minimize the loss with *stochastic gradient descent* (:numref:`sec_sgd`), which updates $\mathbf{w}$ after every minibatch.
<!--en:a90f13c2 kind=paragraph-->
We minimize the loss with *stochastic gradient descent* (:numref:`sec_sgd`), which updates $\mathbf{w}$ after every minibatch of 256 examples.
<!--tr:a90f13c2 was=8c11d0e4-->
損失は*確率的勾配降下法*（:numref:`sec_sgd`）で最小化します。この方法では、ミニバッチごとに $\mathbf{w}$ を更新します。
```

Returned (marker lines and the old English copied unchanged; only the
translation under `<!--tr:…-->` is edited):

```
<!--en:8c11d0e4 kind=paragraph old-->
We minimize the loss with *stochastic gradient descent* (:numref:`sec_sgd`), which updates $\mathbf{w}$ after every minibatch.
<!--en:a90f13c2 kind=paragraph-->
We minimize the loss with *stochastic gradient descent* (:numref:`sec_sgd`), which updates $\mathbf{w}$ after every minibatch of 256 examples.
<!--tr:a90f13c2 was=8c11d0e4-->
損失は*確率的勾配降下法*（:numref:`sec_sgd`）で最小化します。この方法では、256 個の例からなるミニバッチごとに $\mathbf{w}$ を更新します。
```

The first sentence is unchanged byte for byte; only the phrase that corresponds
to the English change was edited.
