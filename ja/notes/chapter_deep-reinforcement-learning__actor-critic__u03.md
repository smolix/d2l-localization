# Notes on chapter_deep-reinforcement-learning/actor-critic.md#u03 (gemini gemini-4-carbon (medium), 2026-11-27 05:15 UTC)

- 8f706c1a, cd5cc470: glossary check warns `'backward pass' -> 逆伝播` (false match; here "one backward pass over the episode" means GAE's reverse-time scan along the trajectory, `1 回の逆方向パス`, not neural-network backpropagation).
- b60050df: glossary check warns `'weight   decay' -> 重み減衰` (false match after stripping `$(\gamma\lambda)^l$` between the noun "weight" and the verb "decay").
