# Results: Français, Convolutions for Images

Work files emitted 2026-11-26 at English revision `370f7b3a15d0` (d2l-neu main), the same page and revision as the German and Japanese tests. The language had no style guide yet, so translators followed TRANSLATING.md §8 and recorded their conventions in DECISIONS. Three units, 144 blocks.

## 2026-11-26: Claude models

Each agent ran once in a fresh session with the prompt in PROMPT.md, at medium effort, alongside the other languages. Counts are from `bin/l10n compare` with the checker after the fixes these runs prompted (the agents themselves saw an earlier checker, with more false matches).

| Agent | Errors | Warnings | Words (× English) | Time |
|---|--:|--:|--:|--:|
| `fable` (Claude Fable 5.2) | 0 | 0 | 5,018 (1.18) | 6.4 min |
| `opus` (Claude Opus 5.5) | 0 | 0 | 4,943 (1.16) | 5.8 min |
| `sonnet` (Claude Sonnet 5.5) | 0 | 0 | 4,943 (1.16) | 2.6 min |

- Fable noted that French needs no-break spaces before : ; ? ! and inside « »; `fr` has no `typography` setting yet.
- Sonnet's warnings were inflected forms (« champs récepteurs », « matricielle ») the check did not accept (fixed).

**Verdict:** pending a native reader.
