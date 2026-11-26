# zh-Hans: corrections from the style guide (2026-11-26)

Source: `zh/style.md` §11, applied to `glossary/terms/zh-Hans.jsonl` as
proposals. `proposed` entries were changed; source-attested entries
(`verified_d2l`, `verified_external`) keep their preferred term and status,
and at most gain an alternative ("conflict").

| Concept | Before | After | Reason |
|---|---|---|---|
| q-learning | Q学习 | Q 学习 | §2.2 spacing |
| p-value | p值 | p 值 | §2.2 spacing |
| softmax-attention | softmax注意力 | softmax 注意力 | §2.2 spacing |
| n-gram | n元语法 | n 元语法 | §2.2 spacing |
| wasserstein-distance | Wasserstein距离 | Wasserstein 距离 | §2.2 spacing |
| transformer-block | Transformer块 | Transformer 块 | §2.2 spacing |
| vision-transformer | 视觉Transformer | 视觉 Transformer | §2.2 spacing |
| he-initialization | He初始化 | He 初始化 | §2.2 spacing |
| deep-q-network | 深度Q网络 | 深度 Q 网络 | §2.2 spacing |
| jensen-s-inequality | Jensen不等式 | Jensen 不等式 | §2.2 spacing |
| f1-score | F1分数 | F1 分数 | §2.2 spacing |
| huber-loss | Huber损失 | Huber 损失 | §2.2 spacing |
| softmax-temperature | softmax温度 | softmax 温度 | §2.2 spacing |
| transformer-encoder | Transformer编码器 | Transformer 编码器 | §2.2 spacing |
| transformer-decoder | Transformer解码器 | Transformer 解码器 | §2.2 spacing |
| diffusion-transformer | 扩散Transformer | 扩散 Transformer | §2.2 spacing |
| cholesky-decomposition | Cholesky分解 | Cholesky 分解 | §2.2 spacing |
| cholesky-factor | Cholesky因子 | Cholesky 因子 | §2.2 spacing |
| f-divergence | f散度 | f 散度 | §2.2 spacing |
| qk-circuit | QK电路 | QK 电路 | §2.2 spacing |
| ov-circuit | OV电路 | OV 电路 | §2.2 spacing |
| softmax-normalization | softmax归一化 | softmax 归一化 | §2.2 spacing |
| top-k-routing | Top-k路由 | Top-k 路由 | §2.2 spacing |
| frechet-inception-distance | Fréchet Inception距离（FID） | Fréchet Inception 距离 | the acronym is not part of the term; §2.2 spacing |
| kernel-launch | 核函数启动 | 内核启动 | GPU kernel is 内核; 核函数 is kernel methods (§7.1) |
| kv-cache | 键值缓存 | KV 缓存 (alternative 键值缓存) | English writes the acronym; §3.1 keeps acronyms |
| sampling-error | 抽样误差 | unchanged, note added | the statistics term stays as the named exception |
| sample | 样本 | unchanged, note names the 抽样误差 exception | as above |
| normalizing-flow | 标准化流 | 归一化流 | one family with 归一化, 归一化常数 (review decision 8) |
| continuous-normalizing-flow | 连续标准化流 | 连续归一化流 | follows normalizing-flow |
| xavier-initialization | Xavier初始化 | conflict: alternative Xavier 初始化 | verified_d2l; §2.2 spacing |
| vc-dimension | VC维 | conflict: alternative VC 维 | verified_external; §2.2 spacing |
| frobenius-norm | Frobenius范数 | conflict: alternative Frobenius 范数 | verified_d2l; §2.2 spacing |
| hadamard-product | Hadamard积 | conflict: alternative Hadamard 积 | verified_d2l; §2.2 spacing |
| softmax-regression | softmax回归 | conflict: alternative softmax 回归 | verified_d2l; §2.2 spacing |
| nesterov-momentum | Nesterov动量 | conflict: alternative Nesterov 动量 | verified_external; §2.2 spacing |
| long-short-term-memory | 长短期记忆网络 | conflict: alternative 长短期记忆 | verified_d2l; the mechanism, not the network |
| activation | 激活值 | conflict: alternatives 激活, 非线性激活 | verified_d2l; "nonlinear activations" between layers |
| dropout | alternative 暂退法（Dropout） | conflict: unchanged | verified_d2l; the guide asks to remove an attested alternative |
| precision | alternative 精确度 | conflict: unchanged | verified_external; removal of an alternative not applied |
| accuracy | alternatives 精度, 准确性 | conflict: unchanged | verified_d2l, source writes 准确率/精度; removal not applied |
| dense layer | (no concept) | skipped | fits `fully-connected-layer` (全连接层); needs the English alias "dense layer" in concepts.jsonl |
| patch | (no concept) | skipped | fits `image-patch` (图像块); needs the English alias "patch" in concepts.jsonl |
| error, inference, statistical inference, window, element, unit, edge map, trade-off, optimizer, memory, GPU memory, memory (recurrent) | (no concept) | new concept | proposed to the coordinator; nothing added here |

The spacing alternative on the six attested entries matters for the glossary
check (`glossary._present` matches the exact string, §9 item 2): with it, a
translator who follows §2.2 is not reported as missing the term.
