# Results: Japanese, Convolutions for Images

Work files emitted 2026-11-26 at English revision `370f7b3a15d0` (d2l-neu main),
the same page and revision as the German test, with the Japanese style guide
before the native-speaker review (its open questions in `ja/style.md` were
still unanswered). Three units, 144 blocks.

## 2026-11-26: Claude models

Each agent ran once in a fresh session with the prompt in PROMPT.md, at medium
effort, all three at the same time.

| Agent | Errors | Warnings | Characters (× English) | Time |
|---|--:|--:|--:|--:|
| `fable` (Claude Fable 5.2) | 0 | 9 | 16,673 (0.67) | 6.7 min |
| `opus` (Claude Opus 5.5) | 0 | 8 | 16,558 (0.67) | 9.1 min |
| `sonnet` (Claude Sonnet 5.5) | 0 | 6 | 16,687 (0.68) | 7.2 min |

- Every warning is a false match, and all three agents left them in place
  and named them in NOTES. The acronyms MLP, CNN, SGD stay in English as
  style §3.2 asks, but the glossary check expects the long Japanese form; "Tensor
  Core" is matched as *tensor*; "standard normal distribution" is 標準正規分布
  under style §7.1, not the glossary's ガウス分布. Fable also has "sparsity
  pattern". These point at the checker (acronym handling) and the glossary
  entry for *normal distribution*, not at the translations.
- The agents shared a scratch directory and two of them overwrote each
  other's helper script; the answers were checked afterwards and are
  independent (pairwise character similarity about 0.9 for every pair, with
  the identical long blocks all formula-heavy slide bullets). PROMPT.md now
  asks each agent to keep its own work directory.

**Verdict:** pending a native reader.

## 2026-11-26: Astra, Sol and Terra

Each model ran once in an independent session at medium effort, using the same
prompt and pinned work files. All three completed 144 blocks. Counts are from
`l10n compare` with the checker at `50e89c2`.

| Agent | Model | Errors | Warnings | Characters (× English) |
|---|---|--:|--:|--:|
| [Astra](answers-astra/) | `gpt-6.2-astra` | 0 | 7 | 17,039 (0.69) |
| [Sol](answers-sol/) | `gpt-6.2-sol` | 0 | 6 | 17,053 (0.69) |
| [Terra](answers-terra/) | `gpt-6.1-terra` | 0 | 8 | 15,950 (0.65) |

- The answers document every remaining warning. Most arise from acronyms,
  Tensor Core, or the style guide's convention for 標準正規分布. Sol rendered
  both SGD labels as 勾配降下法 because the example computes its gradient over
  all 42 outputs. Astra's other warning matches *sparsity* within *sparsity
  pattern*.
- In the side-by-side sample, Astra and Sol both make the kernel weights the
  subject of convergence in the initial-loss slide kicker (`7e4ef44b`).
  Terra's version carries through the original ambiguity. Astra and Sol also
  make explicit that each layer adds two pixels to each side length of the
  receptive field (`b302e3ef`).

The native-reader verdict is pending.

## 2026-11-26: Gemini 4 (Carbon, Neon and Hydrogen)

Each model ran once in an independent `agy` CLI session at `medium` effort
(`--effort medium`) with the prompt in `PROMPT.md` and the same three pinned
work files. All three completed all 144 blocks with zero errors. Counts are from
`bin/l10n compare` with the checker at `50e89c2` (under `1fa7ded`, warnings drop to `0`, `0`, and `2`).

| Agent | Model | Errors | Warnings | Characters (× English) | Time |
|---|---|--:|--:|--:|--:|
| [Carbon](answers-carbon/) | `gemini-4-carbon-medium` | 0 | 8 | 16,773 (0.68) | 16.4 min |
| [Neon](answers-neon/) | `gemini-4-neon-medium` | 0 | 8 | 16,434 (0.67) | 9.4 min |
| [Hydrogen](answers-hydrogen/) | `gemini-4-hydrogen-medium` | 0 | 10 | 16,542 (0.67) | 4.5 min |

- All three pass every automated `ja/style.md` check across all 144 blocks: zero
  translationese pronouns (*私たち* / *あなた*), zero `です・ます` leaks in
  headings, captions, or slides, consistent ML katakana long-vowel forms
  (*パラメータ*, *フィルタ*, *層*), half-width spacing around inline math, code,
  Latin runs, and `:numref:` roles, and `〜してください` / `〜ですか。` in
  exercises.
- All 8 warnings of `carbon` and `neon` are the same checker false matches seen
  on `opus` (standalone acronyms `MLP`, `CNN`/`CNNs`, `SGD` kept in Latin script
  per `ja/style.md` §3.2; `GPU Tensor Core` per §5; and `標準正規分布` per §7.1),
  documented in `NOTES`; both also avoid the `sparsity` warning in `ec8621f8` by
  writing *スパース性のパターン*. `hydrogen` adds two further warnings: rendering
  geometric *Bounding box* as *外接矩形* in `33bb377d` (noted in `NOTES`), and
  reversing `$k \times k$` and `$(a, b)$` in `ec8621f8` (*オフセット $(a, b)$ の
  $k \times k$ ウィンドウ*) without a `NOTES` entry (`carbon` and `neon` keep
  token order with *$k \times k$ ウィンドウのオフセット $(a, b)$*).
- **Side-by-side differences missed by the checker:**
  - `hydrogen` glosses katakana transliterations (`*エッジマップ*（edge map）` in
    `55f0a41b`, `*ウィンドウ*（window）` in `ae7e906d`, `パッチ行列（patch matrix）`
    in `5bdffcd7`), contrary to `ja/style.md` §3.1 (*do not gloss katakana that
    transliterates the English*), whereas `carbon` and `neon` gloss only kanji
    and mixed terms.
  - In exercise `6f4ddec6`, `neon` and `hydrogen` (like `fable`) use the standard
    image-processing kanji *鮮鋭化カーネル* for *sharpening kernel*, whereas
    `carbon` writes the homophone *先鋭化カーネル*. Conversely, `neon` and
    `hydrogen` wrap the English book title in `『…』` (`6ab5b67c`, per `ja/style.md`
    §2.1), while `carbon` and `fable` leave it unbracketed.
  - On slide kicker `7e4ef44b` (*"Why Initial Loss Is 12 and Converges to
    [1, -1]"*), all three rely on Japanese subject drop (*初期損失が 12 となり
    [1, -1] に/へ収束する理由*) rather than supplying *カーネル* explicitly as
    `astra` and `sol` do.

Overall, `neon` (9.4 min, 8 false-match warnings) and `carbon` (16.4 min, 8
false-match warnings, except for the *先鋭化* kanji in `6f4ddec6`) write clean,
natural `です・ます` technical Japanese on par with `fable` and `opus`.
Native-reader verdict pending.
