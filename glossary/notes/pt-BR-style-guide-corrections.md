# Brazilian Portuguese: style-guide glossary corrections (2026-11-26)

Applied from `pt/style.md` §12 to `glossary/terms/pt-BR.jsonl`. Every edited
entry was `proposed` and stays `proposed`; each carries a "Style guide
2026-11-26" note. No attested entry was touched.

| Concept | Before | After | Reason |
|---|---|---|---|
| `strided-convolution` | convolução com passo | convolução com stride | *stride* stays English, as in `stride`; *passo* means a step (§8.1) |
| `ordinary-differential-equation` | equação diferencial ordinária | same; alternative EDO | translated acronym (§3.1, question 6) |
| `stochastic-differential-equation` | equação diferencial estocástica | same; alternative EDE | translated acronym (§3.1, question 6) |
| `ode-solver` | solucionador de equações diferenciais ordinárias | solucionador de EDOs (long form kept as alternative) | written-out form too long for running text |
| `neural-ode` | equação diferencial ordinária neural | EDO neural (long form kept as alternative) | same |
| `probability-flow-ode` | equação diferencial ordinária de fluxo de probabilidade | EDO de fluxo de probabilidade (long form kept as alternative) | same |
| `reverse-time-sde` | equação diferencial estocástica em tempo reverso | EDE em tempo reverso (long form kept as alternative) | same |
| `image-patch` | patch de imagem | same; alternative patch | bare *patch* (os patches) is the short form; there is no separate generic concept (§8.1) |
| `soft-actor-critic` | ator-crítico suave | SAC (Portuguese kept as alternative, the first-mention gloss) | acronyms stay English (§3.1) |
| `proximal-policy-optimization` | otimização proximal de política | PPO (Portuguese kept as alternative) | same |
| `deep-q-network` | rede Q profunda | DQN (Portuguese kept as alternative) | same |
| `exploitation` | no concept | new concept: explotação | collides with exploração (§8.1) |
| `trade-off` | no concept | new concept: compromisso | §8.1, question 10 |
| `bias-variance-tradeoff` | no concept (`bias-variance-decomposition` is a different sense) | new concept: compromisso entre viés e variância | §8.1 |
| `deep-learning` | no concept | new concept: aprendizado profundo | used on every page |
| `machine-learning` | no concept | new concept: aprendizado de máquina | used on every page |
| `built-in` | no concept | new concept: embutido | §8.1, question 8 |
| `matrix-entry` | no concept | new concept: elemento | §8.1 |
| `image-edge` | no concept | new concept: borda | §8.1 |

New concepts are not added to any file; they await the coordinator.
