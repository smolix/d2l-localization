# Simplified Chinese (zh-Hans) glossary review

Reviewed 2026-11-26. Scope: all 1,184 entries of `glossary/terms/zh-Hans.jsonl`
were read against the senses in `glossary/concepts.jsonl` (295 `verified_d2l`,
26 `verified_external`, 863 `proposed`; the proposed set was read in full, not
sampled). Corrections are in `zh-Hans-corrections.jsonl` (116 lines).

## Verdict

The Chinese glossary is in much better shape than the Japanese one. Its base is
sound: the core vocabulary (梯度, 张量, 损失函数, 过拟合, 泛化, 正则化, 反向传播,
多层感知机, 注意力, 智能体, 策略, 回报, 词元 …) is correct and matches both the
earlier edition and mainland textbooks. None of the Japanese failure types
(transliterated common nouns, "gradation" for gradient, 一般化 for
generalization) occurs. The problems are narrower and of three kinds:
extraction errors on the verified side, house-style coinages of d2l-zh that
conflict with the proposed entries, and explanatory glosses in place of terms
on the proposed side.

| Status | Entries | Would change | Of which substantive (wrong sense / calque / non-standard) | Inconsistent or minor |
|---|---:|---:|---:|---:|
| verified_d2l | 295 | 28 (9%) | 10 | 18 |
| verified_external | 26 | 6 (23%) | 5 | 1 |
| proposed, frequency ≥ 50 | 136 | 32 (24%) | 18 | 14 |
| proposed, frequency < 50 | 727 | 50 (7%) | 22 | 28 |
| **Total** | **1,184** | **116 (10%)** | **55** | **61** |

By severity: wrong_sense 10, calque_or_mt 29, non_standard 16, inconsistent 34,
minor 27. Of the 116 proposed replacements, 17 are backed by a cited source
using the term in this sense (`verified_external`); the other 99 stay
`proposed`.

## Provenance

- **d2l-zh is a human translation**, edited by the authors' group and
  published in print (《动手学深度学习》), with an editorial term list
  (`TERMINOLOGY.md`, 138 pairs). Its `config.ini` does contain
  `translator = aws`, the same field that flagged the Japanese and Korean
  editions, but here it is template residue: the prose is idiomatic, uses
  consistent coined house terms, and introduces terms bilingually
  (`*术语*（english）`). The field alone is not evidence of machine translation
  for zh; `DISCOVERY.md` is right to call it a strong edited translation.
- **"Verified" still only means "attested".** The verified_d2l errors are not
  translation errors in d2l-zh; they are extraction errors: a phrase next to
  the English term was taken as its translation. Examples: 未规范化的预测 for
  logit (a gloss), 显存 for memory footprint (GPU memory, said loosely in one
  sentence), 平均汇聚层 for average pooling (the layer), 小批量随机梯度下降 for
  minibatch gradient descent (a neighbouring term), 序列转换模型 for sequence
  transduction, 链式规则 (1 occurrence, against 11 × 链式法则 in the same
  edition), 样式迁移 (1 occurrence, against 17 × 风格迁移).
- **The external sources are the weak part.** All 6 verified_external
  changes come from two sources: the crowd-translated Amidi CS229 cheatsheet
  (联盟 for union bound, 拉格朗日 for Lagrangian) and the Google ML Glossary,
  whose pages declare machine translation (模型能力 for model capacity, 正类别 /
  负类别, 平方损失函数). Neither should confer `verified` status for zh.
- Alternatives lists contain raw machine-translation residue that must not be
  offered to translators: 重量 for weight (the physical sense), 功能 for
  feature (the software sense).

Reference sources used in this review: the d2l-zh repository at the pinned
commit (term list and chapters), the terminology file of the Chinese
translation of Goodfellow et al. (花书,
`exacity/deeplearningbook-chinese/terminology.tex`, 862 pairs), Chinese
Wikipedia titles, and Easy RL (蘑菇书, Datawhale) for reinforcement learning.
周志华《机器学习》 and 李航《统计学习方法》 were used from knowledge of the
books, not from a page fetched here; the corresponding entries stay `proposed`.

## Systematic problems

1. **Glosses instead of terms (29 entries, almost all proposed).** Proper names
   were given an explanatory tail: Transformer架构, AdamW优化算法,
   Mamba选择性状态空间模型, Reflow轨迹校正, BERT模型, GELU激活函数,
   InfoNCE对比损失, BLEU翻译评价指标, word2vec词向量方法, Hyperband超参数优化.
   Acronyms and operator names were expanded into Chinese that no reader would
   recognise: 图形处理器 for GPU (1,195 occurrences), 线性偏置注意力 for ALiBi,
   均方根归一化 for RMSNorm, 全局词向量 for GloVe, 全收集 / 全规约 / 规约散射 for
   all-gather / all-reduce / reduce-scatter, 对数和指数 for log-sum-exp. These
   must be the bare Latin name.
2. **House terms of d2l-zh against mainland usage.** d2l-zh v2 deliberately
   coined 汇聚 (pooling), 规范化 (normalization), 暂退法 (dropout), 跳元模型
   (skip-gram) and 词元 (token). The proposed entries were written in
   mainstream vocabulary, so the glossary now contradicts itself: 汇聚 /
   平均汇聚层 / 全局平均汇聚 next to 最大池化; 批量规范化 / 层规范化 next to
   归一化层, 自适应层归一化, 谱归一化. 词元 has since become the accepted
   translation of token and should stay.
3. **Head words not held constant.** sampling: 抽样 (head) vs. 采样 (all
   compounds); score (∇log p): 得分函数 vs. 分数匹配 / 分数网络; latent: 潜 vs.
   潜在; Monte Carlo: 蒙特卡洛 vs. 蒙特卡罗; Euclidean: 欧氏 vs. 欧几里得;
   momentum: 动量法 vs. 动量; style: 样式 vs. 风格; dilation: 膨胀率 vs. 空洞卷积;
   region proposal: 候选区域 vs. 区域提议网络; on/off-policy: 同策略 with 离策略;
   precision/recall: 查准率 with 召回率; softmax capitalised in three compounds.
4. **Sense collisions.** 下采样 serves both subsampling and downsampling;
   预处理 serves preconditioning while meaning preprocessing elsewhere;
   注意力汇聚点 (attention sink) collides with 注意力汇聚 (attention pooling);
   激活参数量 (active parameters) reuses 激活 = activation. Collisions that are
   handled correctly: 偏置 (layer) vs. 偏差 (estimator), 卷积核 vs. 核函数 /
   核矩阵 / 转移核, 回报 (RL return), 值 (attention value) vs. 价值 (RL value
   function), 分数 (attention score). One collision is inherent in Chinese and
   cannot be removed: 特征向量 is both feature vector and eigenvector.
5. **Person names.** Frobenius范数, Hadamard积, Xavier初始化, Huber损失, He初始化
   stay Latin, while 内斯特罗夫, 瓦瑟斯坦, 乔列斯基, 詹森, 弗雷歇 are
   transliterated, and Bayes定理 sits beside 贝叶斯推断. A rule is needed (see
   below).
6. **Reinforcement learning.** The core is right (智能体, 状态, 动作, 奖励, 策略,
   回报, 价值函数, 回合, 轨迹). The actor–critic family is not: 行动者 / 评价器 /
   软行动者评价器 are rare, 贴现系数 is finance vocabulary (RL: 折扣因子), and
   观察 should be 观测.
7. **Coverage gaps** (not errors): there are no concepts for bare loss, cost,
   sample / example (noun), temperature, score as a metric, KL divergence, or
   inference, so those sense distinctions (损失 vs. 代价, 样本 vs. 样例 / 采样)
   are currently not controlled by the glossary at all.

## Decisions for the author or a native reviewer

1. **汇聚 or 池化; 规范化 or 归一化.** Recommendation: 池化 and 归一化, as in
   essentially all mainland textbooks, papers and framework documentation;
   the corrections file is written that way. The alternative is continuity
   with the printed d2l-zh; then flip the proposed entries instead (最大汇聚,
   规范化层, …). Either way one family per head word.
2. **dropout and skip-gram.** 暂退法 and 跳元模型 are d2l-zh coinages with
   little use outside the book; most Chinese texts write dropout and
   skip-gram in Latin (花书: Dropout). No correction filed; needs a ruling.
3. **Actor–critic.** 演员–评论员 (Easy RL, filed here), 演员–评论家, or
   行动器–评判器 (Sutton and Barto translation); and the WGAN critic (评论器 is
   filed as a placeholder).
4. **logit.** Keep in Latin (filed) or adopt 花书's 分对数; 对数几率 stays for
   log-odds.
5. **Person-name rule.** Proposed: transliterate household names only (高斯,
   贝叶斯, 马尔可夫, 拉格朗日, 牛顿, 泰勒, 伯努利, 泊松, 欧拉, 傅里叶, 香农,
   玻尔兹曼, 朗之万, 伊藤); keep all others in Latin (Frobenius, Hadamard,
   Jensen, Lipschitz, Nesterov, Wasserstein, Cholesky, Fisher, Huber). Open
   cases under that rule: 利普希茨, 费希尔 / Fisher, 格拉姆 / Gram, 雅可比 and
   海森 (花书 keeps Jacobian and Hessian in Latin; Wikipedia's title is 黑塞矩阵).
6. **边际 vs. 边缘.** The glossary uses 边际 throughout (分布, 概率, 似然, 速度),
   as d2l-zh and 周志华 do; 李航 and 花书 use 边缘. Consistent as it stands; no
   change filed.
7. **score.** 分数 (diffusion literature, filed) or 得分 (花书) for ∇log p;
   得分 would also separate it from attention and metric scores.
8. **Terms that need a specialist's eye:** union bound (联合界 is filed as
   proposed; not confirmed on a primary textbook page), preconditioning
   (预条件), normalizing flow (标准化流 vs. 归一化流), standard error (标准误),
   and the newly coined Language-Models terms with no established Chinese:
   残差流, 归纳头, OV电路 / QK电路, 注意力汇点, 噪声球, 屋顶线模型, 可建模性,
   首词元延迟.
9. **Status policy.** Stop granting `verified_external` from the Google ML
   Glossary (machine-translated) and from single cheatsheet lines; and treat
   `verified_d2l` entries taken from one prose sentence (`bilingual_introduction`,
   `manual_body`) as weaker than those from `TERMINOLOGY.md`.

## Sources

- d2l-zh, commit `e6b18cce`: <https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/TERMINOLOGY.md>, and `chapter_preliminaries/calculus.md`, `chapter_preliminaries/probability.md`, `chapter_multilayer-perceptrons/backprop.md`, `chapter_multilayer-perceptrons/mlp.md`, `chapter_linear-networks/softmax-regression.md`, `chapter_computational-performance/multiple-gpus.md`, `chapter_optimization/gd.md`, `chapter_computer-vision/neural-style.md`, `config.ini`.
- 花书 Chinese translation term list: <https://github.com/exacity/deeplearningbook-chinese/blob/master/terminology.tex>
- Easy RL (蘑菇书), chapters 1, 5, 9: <https://github.com/datawhalechina/easy-rl/tree/master/docs>
- Chinese Wikipedia: 精确率与召回率, 线性整流函数, 自编码器, 线性无关, 费希尔信息, 布尔不等式, 黑塞矩阵.
