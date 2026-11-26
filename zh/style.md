# Style guide: Simplified Chinese (zh, zh-Hans)

Draft of 2026-11-26 (Fable 5.2) for native-speaker review. Translators read this with every
unit (TRANSLATING.md §7–§8). Each rule has one reason and one example
(English → Chinese, or ✗ wrong / ✓ right). Points that need the author's or a
native reviewer's ruling are marked "(to confirm)" and collected in §10.

Order of authority: the decisions log, then this guide, then the glossary. The
glossary says *what* a term is; this guide says *how to choose, write and
space* terms. Where the two disagree today, §11 lists the entry.

The earlier Chinese edition (*动手学深度学习*, zh.d2l.ai, `d2l-ai/d2l-zh`) is a
human translation published in print. This guide keeps its register, its way of
introducing terms (`*术语*（english）`), its section names (小结, 练习) and most
of its vocabulary, and departs from it in spacing (§2.2) and in a few house
coinages (§3.2).

Write Simplified Chinese in mainland usage (§7.3), as plain modern written
Chinese (现代汉语书面语): no classical particles for effect, no colloquial
filler, no internet slang.

## 1. Register

**Decision: authorial 我们, the reader as 你 where the English really says
"you", never 您.** Reason: 我们 is the register of d2l-zh and of mainland
university textbooks and carries the book's "we work through this together"
tone (TRANSLATING.md §8); 您 is the register of manuals and customer service,
and 笔者, 咱们, 大家 belong to essays and lectures.

| English | Chinese | Example |
|---|---|---|
| we + action in the text | 我们 + verb; drop a repeated 我们 in the next clause | "We define the kernel and compute the output." → 我们定义卷积核，然后计算输出。 |
| "let us", "let's", "we now turn to" | 我们来…, 下面…, 现在…, 先看… | "Let us look at the gradient." → 我们来看梯度。 ✗ 让我们看一看梯度吧。 |
| "we will see", "we will use" | bare verb or 会; 将 only for a real contrast with now | "We will return to this in …" → 我们会在 … 中回到这一点。 |
| "in this section/chapter/book, we …" | 本节, 本章, 本书 (+ 我们) | "In this section, we implement cross-correlation." → 本节中，我们实现互相关。 |
| "you" = the reader (advice, a question, exercises) | 你 | "How would you design a blur kernel?" → 你会如何设计一个模糊卷积核？ |
| "you can", "one can" = anyone | no subject, 可以, or 我们 | "You can verify this numerically." → 这一点可以用数值方法验证。 |
| "we recommend", "we believe" | 我们建议, 我们认为 | "We recommend AdamW." → 我们建议使用 AdamW。 |
| "our model", "our dataset" | 这个, 该, or nothing; 我们的 only for something the text built itself | "our model overfits" → 这个模型过拟合了 |
| "we" = people, the field | 人们, 我们人类, 研究者 | "We humans attend selectively." → 我们人类会有选择地分配注意力。 |

**Sentence shape** (each point is checkable):

- **Verbs, not 进行.** No 进行, 加以, 做出, 予以 + verbal noun where the verb
  alone works. ✗ 对参数进行更新 ✓ 更新参数. ✗ 对输入进行填充 ✓ 填充输入.
- **No 被 where Chinese needs none.** "is called" → 称为; "is defined as" →
  定义为; "is given by" → 由 … 给出; "is computed" → an active clause.
  ✗ 损失被计算，然后梯度被反向传播。 ✓ 先计算损失，再反向传播梯度。
- **At most two 的 in one noun phrase.** Turn the rest into a verb or a clause.
  "the shape of the output of the first layer of the network" ✗
  网络的第一层的输出的形状 ✓ 网络第一层输出的形状.
- **Short modifiers before the noun.** An English relative clause that would
  put more than about twenty characters before 的 becomes a second clause or
  sentence (TRANSLATING.md §3 allows splitting within a block).
  ✗ 我们在上一节中用学习率 0.1 在 Fashion-MNIST 上训练了 10 轮的模型 ✓
  上一节的模型用学习率 0.1 在 Fashion-MNIST 上训练了 10 轮；现在我们评估它。
- **No 一个 for every "a".** Keep 一个 where it counts or introduces a new
  object; drop it in definitions and after 是. ✗ 一个卷积层是一个将输入与一个卷积核做互相关的层。
  ✓ 卷积层将输入与卷积核做互相关。
- **No 和 between clauses or verbs.** 和 joins nouns; clauses take 并, 而, 再
  or a comma. ✗ 我们计算损失和更新参数。 ✓ 我们计算损失并更新参数。
- **No 们 on things, no 它们 chains.** ✗ 模型们, ✗ 它们的它们; repeat the noun.
  "they overfit" (models) → 这些模型会过拟合.
- **Not every "when" is 当 … 时, not every "if" 如果 … 的话, not every "by
  doing" 通过 … 来.** "By minimizing the loss, we obtain …" → 最小化损失，就得到 ….
- **Modal particles:** only 吗 and 呢 in questions. No 吧, 啦, 哦, 嘛, 哈.
- **Classical words:** 也, 所以, 它的, 就 in body text, not 亦, 故, 其, 乃, 便.
  The fixed words of mathematical Chinese stay: 设, 令, 若 … 则 …, 即, 其中,
  记作, 当且仅当.
- **Four-character idioms** only where they are the plain way to say it
  (成千上万, 一目了然); never two in one sentence.
- **Characters:** 的 before nouns, 地 before verbs, 得 after verbs
  (对称地居中, 训练得更快); 其他 (not 其它); 图像 (not 图象); 做卷积, 做互相关
  but 作为, 称作; 账号, 部署, 阈值 (not 阀值).

## 2. Punctuation and spacing

### 2.1 Punctuation

Full-width Chinese punctuation in Chinese text (GB/T 15834—2011); half-width
punctuation only inside math, code, numbers, URLs and Latin-only expressions.

| Item | Rule | Example |
|---|---|---|
| Comma, full stop | ， and 。 (never `,` `.` or ．as Chinese punctuation), also after a sentence that ends in math, code or a Latin word. | "…is the sum of `X[i:i+h, j:j+w] * K`." → … 是 `X[i:i+h, j:j+w] * K` 的和。 ; … $= 19$。 |
| Enumeration comma | 、 between parallel words and short phrases, with 和 or 或 before the last; ， between clauses. No comma before 和. Latin items take 、 too. | "strides, padding, and dilation" → 步幅、填充和空洞率 ; "Adam, RMSProp, and SGD" → Adam、RMSProp 和 SGD |
| Colon, semicolon | ： and ；. A paragraph that ends in `:` before code or an equation ends in ：; one that ends without punctuation before a display equation stays without. | "returning outputs of shapes $2 \times 2$ and $5 \times 7$:" → … 输出形状分别为 $2 \times 2$ 和 $5 \times 7$： |
| Question, exclamation | ？ for every question. ！ only where the English has `!`. | "Why does this work?" → 这为什么可行？ |
| Parentheses | Full-width （） around anything set inside Chinese text: Chinese words, an English gloss, a reference, math, code. Half-width `()` only where they belong to math, code, a panel marker `(a)`, or sit inside a Latin-only phrase. | "black (`0`)" → 黑色（`0`） ; "(see :numref:`sec_padding`)" → （见 :numref:`sec_padding`） ; panel (c) → 子图 (c) |
| Quotation marks | “…”, inside them ‘…’ (U+201C/D, U+2018/9). Never 「」『』 (Traditional and Japanese usage) and never ASCII `"…"`. | "and call cross-correlation "convolution"" → 把互相关称为“卷积” |
| Quotes in attributes | A slide `title="…"` is delimited by ASCII `"`; inside it use “…” only. | `title="“深”是什么意思？"` |
| Titles | 《》 for a work whose title is given in Chinese. A title kept in English stays as the source has it (italic or link text), without 《》. | 《动手学深度学习》 ; [Neural Networks and Deep Learning](http://…) |
| Dash | The English em dash `—` becomes a comma, a colon or （）; ——(two U+2014) at most once per paragraph. Never a single `—`, `--` or a spaced hyphen. | "the bias—an offset—shifts the plane" → 偏置（一个偏移量）会平移这个平面 |
| Connecting dash | An en dash `–` without spaces between paired names, as in the glossary. | 偏差–方差权衡, 欧拉–丸山法 |
| Ellipsis | …… (two U+2026) in prose; `\ldots` and `...` in math and code are untouched. | |
| Interpunct, slash | `·` and `/` are copied from the source. | 二维互相关 · 边缘检测 · 感受野 |
| Width | Latin letters and digits are always half-width. No full-width Ａ, １, no ideographic space (U+3000). | ✗ ＧＰＵ ✗ １０ |

### 2.2 Spacing: one rule

**Put one half-width space between a Chinese character and any half-width run:
a Latin word, a number, inline math `$…$`, inline code, or a reference role
(``:numref:`…` ``, ``:eqref:`…` ``, ``:cite:`…` ``, `§{…}`). Put no space next
to full-width punctuation or brackets (，。、；：？！（）“”《》).** Markup
characters (`*`, `**`, `[`, `]`) do not count: look at the characters they
enclose.

Reasons: (1) the build requires it around references and math: a role or `$…$`
that touches a Chinese character is an error in `l10n check`, because pandoc
does not recognise ``:numref:`sec_sgd`中`` (the form d2l-zh's source has);
(2) with the space typed, every output (HTML,
notebooks, slides, PowerPoint, PDF) shows the same gap, where automatic CJK–
Latin spacing exists only in some renderers; (3) one rule for all half-width
runs can be checked mechanically. All three test translators chose this rule
independently. It departs from d2l-zh and from print practice (GB/T 15834
types no space; the typesetter adds the gap).

| English | Chinese |
|---|---|
| "In :numref:`sec_why-conv`, we derived …" | 在 :numref:`sec_why-conv` 中，我们推导出了 … |
| "a $28 \times 28$ Fashion-MNIST image with a $5 \times 5$ kernel" | $28 \times 28$ 的 Fashion-MNIST 图像配上 $5 \times 5$ 的卷积核 |
| "(5 parameters total)" | （共 5 个参数） |
| "an accuracy of 92.3%" | 92.3% 的准确率 |
| "a 180° flipped kernel $\mathbf{K}'$" | 翻转 180° 的卷积核 $\mathbf{K}'$ |
| "`Conv2D` registers both parameters" | `Conv2D` 注册了这两个参数 |
| "a sliding *window*" | 滑动*窗口* |
| "let gradient descent *learn* $(1, -1)$" | 让梯度下降*学出* $(1, -1)$ |
| "ResNet appeared in 2015." | ResNet 于 2015 年问世。 |
| "the Transformer block", "GPU memory" | Transformer 块, GPU 显存 |

- Measure words, ordinals and units in Chinese follow the same rule: 5 个,
  第 3 章, 10 轮, 2 倍, $d$ 维, 4 列宽.
- Between two half-width runs keep the single space of the English: 卷积核 `K`
  与张量 `X`; 24 GiB.
- No space inside `%`, `°` and the number before them (92.3%, 180°), and none
  between a full-width bracket and what it encloses: （`0`）, （$\mathbf{K}'$）.
- **One block, one line.** Do not wrap a paragraph, caption or list item
  inside a sentence: pandoc turns a line break into a space, which then stands
  between two Chinese characters. Keep only the line breaks that carry
  structure (a sub-list, a continuation line that the English block also has).
  ✗ `*改编自 Stanford CS231n 的⏎[作业 2](…)` (a visible gap after 的) ✓ the same
  on one line.

### 2.3 Emphasis markers and punctuation

**Punctuation that follows an emphasised phrase goes outside the markers:
`**小标题**：正文`, `**题名**。正文`, `*术语*（english）`.** Reason: CommonMark
renderers (Jupyter, Colab, which show the notebooks) do not close `**…：**`
when a Chinese character follows directly; tested with the repository's
pandoc, whose own Markdown reader accepts both forms, so only the outside form
is safe everywhere. Two of the three test translators found this on their own.

| English | Chinese |
|---|---|
| `- **Top-left anchor:** Zero-based indices …` | `- **左上角锚点**：从零开始的下标 …` ✗ `**左上角锚点：**从零…` ✗ `**左上角锚点：** 从零…` (a stray space) |
| `1. [code] **Designing kernels by hand.** Hand-design a few kernels.` | `1. [code] **手工设计卷积核**。手工设计几个卷积核。` |
| `**Proposition.** Let …` / `**Proof.** …` | `**命题**。设 …` / `**证明**。…` (to confirm) |

Keep `*…*` and `**…**` on the corresponding word(s) and add none. Do not
replace emphasis with quotation marks. An opening quotation mark or bracket
never stands directly inside the markers (✗ `*“卷积”*是` ✓ `“*卷积*”是`).

### 2.4 References and citations

The build prints the prefix; translators do not control it. For `zh` Quarto
prints 图 4.1, 表 2.1, 式 3.1, 小节 5.1 and 章节 3 (Quarto's
`_language-zh.yml`; §9 proposes §5.1 and 第 3 章). **Write no noun of your
own before or after a reference role, and no measure word or demonstrative
before it**: the text must read correctly whichever noun is printed, and
before or after the number.

| English | Chinese |
|---|---|
| "In :numref:`fig_correlation`, we slide …" | 在 :numref:`fig_correlation` 中，我们让 … 滑动 |
| "as shown in :numref:`sec_why-conv`" | 如 :numref:`sec_why-conv` 所示 ; 正如 :numref:`sec_why-conv` 所示 |
| "see :numref:`sec_padding`" | 见 :numref:`sec_padding` ✗ 见第 :numref:`sec_padding` 节 ✗ 见图 :numref:`fig_x` ✗ 这张 :numref:`fig_x` |
| "We implement :eqref:`eq_corr2d` directly" | 我们直接按 :eqref:`eq_corr2d` 实现 ✗ 公式 :eqref:`eq_corr2d` |
| ":numref:`sec_a` and :numref:`sec_b`" | :numref:`sec_a` 和 :numref:`sec_b` |
| ":citet:`Luo.Li.Urtasun.ea.2016` showed that …" | :citet:`Luo.Li.Urtasun.ea.2016` 指出，… (no 等, no 的 directly on the role: ✓ :citet:`…` 提出的模型) |
| "least squares :cite:`Legendre.1805`." | 最小二乘法 :cite:`Legendre.1805`。 (at the end of its clause, before the ，or 。) |
| `§{sec_sgd}` (prints "§9.3") | 见 §{sec_sgd} |
| "Chapter §{num:chap_x}" (prints the bare number) | 第 §{num:chap_x} 章 (here, and only here, the noun is yours) |

A range such as `§{sec_a}–§{num:sec_b}` is copied as it is. A citation written
by hand in a slide ("Luo et al., 2016") is copied in the English form, to match
what `:citet:` prints: （Luo et al., 2016）.

### 2.5 Math and code in a sentence

- A sentence may start with math, code or a reference; no filler is needed.
- Give a symbol its noun where the sentence would otherwise be unclear:
  卷积核 $\mathbf{K}$, 学习率 $\eta$, 张量 `X`.
- A dimension before a noun takes 的 in body text and may drop it in slides:
  一个 $2 \times 2$ 的卷积核 ; slide: $2 \times 2$ 卷积核.
- Protected math is copied as it is, even when it contains English: "the
  $i^{\textrm{th}}$ layer" → $i^{\textrm{th}}$ 层 (no 第 added; do not report
  it in NOTES) (to confirm).
- **Token order.** Chinese puts modifiers before the noun, so a faithful
  sentence sometimes needs the protected tokens in another order ("the kernel
  $\mathbf{K}$ of shape $h \times w$" → 形状为 $h \times w$ 的卷积核
  $\mathbf{K}$). Write correct Chinese first. Where an equally natural sentence
  keeps the English order, prefer it; never bend a sentence only to keep the
  order (the test's Opus run did, and RESULTS.md records it as a fault). A
  reorder is a warning: name the block in NOTES.
- End the sentence before a display equation if the Chinese verb would
  otherwise have to follow it; continue after it with 其中 …: "… is given by
  $$…$$ where $n$ is …" → … 由下式给出： $$…$$ 其中 $n$ 是 …。

## 3. Terminology

### 3.1 Chinese term or English

**Use the Chinese term wherever mainland textbooks and papers have one. Leave
in English only names, acronyms and the few words that Chinese practice itself
leaves in Latin. Never mix an English common noun into a Chinese sentence
(中英夹杂), and never coin a Chinese word for a name.** The glossary decides
each case; for a term without an entry, apply this table, check a mainland
textbook or Chinese Wikipedia, and record the choice in DECISIONS.

| Case | Rule | Examples |
|---|---|---|
| Established Chinese term | Use it. | 层 (✗ layer), 损失 (✗ loss), 批量大小 (✗ batch size), 权重, 梯度下降, 过拟合, 正则化, 嵌入, 学习率, 卷积核, 步幅, 填充, 池化, 注意力, 微调, 预训练, 词元, 提示 |
| Names of models, architectures, optimizers, libraries, datasets, metrics, hardware; acronyms; code | Leave in English, in the source's capitalisation; never translate or transliterate a name, in whole or in part. | Transformer (✗ 变换器, ✗ 变压器), ResNet, BERT, Adam, AdamW, PyTorch, ImageNet, Fashion-MNIST, BLEU, ReLU, SGD, MLP, CNN, RNN, LSTM, GPU, KL, Tensor Core (✗ 张量核心), Toeplitz (✗ 托普利茨) |
| Words Chinese ML writing keeps in Latin | As in the glossary, lower case. | softmax, logit, im2col, notebook (Jupyter notebook) |
| Name or acronym + Chinese noun | Name, space, noun (§2.2). | Adam 优化器, ReLU 函数, softmax 回归, Transformer 块, GPU 显存, KV 缓存, Toeplitz 矩阵, Xavier 初始化 |
| Descriptive multi-word names with an established Chinese form | Chinese, as in the glossary. | 多层感知机, 卷积神经网络, 长短期记忆网络, 随机梯度下降, 批量归一化 |

- **Acronyms:** write what the English writes. Where the English has "CNN",
  "SGD" or "MLP", keep the acronym; where it writes the words out, use the
  Chinese term. ✗ expanding "Deep CNNs" on a slide to 深层卷积神经网络 ✓ 深层
  CNN. Reason: slides must stay as short as the English, and the acronym is
  what readers meet in code and papers. No plural -s: CNNs → CNN.
- **Word forms:** Chinese terms serve as noun and verb (正则化, 采样, 填充,
  归一化, 微调, 预训练); do not add 化 or 性 to a glossary term unless the
  glossary has that form, and do not use an English verb stem (✗ fine-tune
  一下, ✗ dropout 掉).
- **One term per concept**, also across near-synonyms the glossary lists as
  alternatives: 准确率 (not 精度, 准确性), 学习率 (not 学习速率), 批量大小 (not
  批次大小), 特征图 (not 特征映射).

### 3.2 What is kept from d2l-zh, and what is not

| Kept | Reason |
|---|---|
| 词元 (token), 小批量, 批量大小, 多层感知机, 隐藏层, 隐状态, 步幅, 填充, 感受野, 特征图, 权重衰减, 束搜索, 智能体, 策略, 回报, 独热编码, 真实值 (ground truth) | Sound terms; 词元 has since become the accepted translation of *token*. |
| 暂退法 (dropout), 跳元模型 (skip-gram), always with the English gloss on first use (§3.3) (to confirm) | d2l-zh coinages with little use outside the book; `verified_d2l` in the glossary, so they bind until the author rules. |
| 轮 for *epoch* (训练 10 轮, 第 3 轮), not 迭代周期 | Short, and it is its own measure word. |
| 小结, 练习, 讨论 for Summary, Exercises, Discussions | Readers of the earlier edition know them. |
| `*术语*（english）` for introducing a term | §3.3. |

| Not kept | Now |
|---|---|
| 汇聚 (pooling), 规范化 (normalization layers) | 池化, 归一化: mainland textbooks, papers and framework documentation; the glossary has been changed (`glossary/review/zh-Hans.md`). |
| No space around Latin, numbers and math (在PyTorch中, $\mathbf{V}$被称为) | §2.2. |
| 精度 for accuracy | 准确率; 精度 is numeric precision (混合精度, 单精度). |
| 未规范化的预测 for logit | logit. |

### 3.3 The English term in parentheses

**Add the English once per page (one numbered section of the book), in
full-width parentheses directly after the Chinese term, lower case as in the
glossary, outside the emphasis markers: `*边缘图*（edge map）`.** Reason: readers
go on to English papers, documentation and code; d2l-zh does the same, and
once per page keeps the text clean. Check the English page in CONTEXT: if an
earlier unit of the page introduced the term, do not repeat it.

Gloss a term at its first occurrence in body text when

1. the English introduces it there with emphasis (`*edge map*`, `*finite
   difference*`): *有限差分*（finite difference）; or
2. it is a term whose English a reader cannot recover from the Chinese: the
   d2l-zh coinages and the short words with several competing translations:
   暂退法（dropout）, 词元（token）, 轮（epoch）, 步幅（stride）, 回报（return）,
   提示（prompt）, 池化（pooling）, 鲁棒性（robustness）.

Do not gloss terms left in English, school vocabulary (函数, 矩阵, 概率, 导数,
均值, 向量), or a term the page only mentions in passing.

- With an acronym: *随机梯度下降*（stochastic gradient descent，SGD）, with a
  full-width comma; afterwards write what the English writes (SGD).
- Where the English term already stands in parentheses, put the gloss inside
  them after a full-width comma: "a border of zeros (*padding*)" →
  补上一圈零（*填充*，padding）.
- Where the English explains an abbreviation by its words, keep the English
  words and add the meaning: "*im2col* (image to columns)" → *im2col*（image to
  columns，图像到列）.
- Never in headings, captions, slides, exercises or table cells.

## 4. Numbers, measure words and units

**Decision: digits where the English has digits, Chinese numerals where the
English spells the number out; the decimal point and the thousands comma as in
the source.** Reason: `l10n check` looks for every English numeral in the
translation, mainland usage (GB/T 15835—2011) writes measured values in Arabic
digits with `.` and allows `,` grouping, and code, outputs and figures show the
same form.

| Item | Rule | Example |
|---|---|---|
| Digits or characters | As the English. Spelled-out counts take 两 before a measure word, 二 in ordinals and set terms. | "two hidden layers" → 两个隐藏层 ; "(5 parameters total)" → （共 5 个参数） ; "four positions" → 四个位置 ; "initialize to zero" → 初始化为零 |
| Set terms | Chinese numerals whatever the English writes. | "2D", "3D", "4D" → 二维, 三维, 四维 ; 二阶导数, 一维, 二元, 三元组, 一方面 |
| Decimal, thousands | `0.5`, `10,000`, as in the source. No space grouping, no 千分空. | "10,000 images" → 10,000 张图像 |
| Number words | 万 and 亿, without a comma: million → 100 万, billion → 10 亿, trillion → 万亿 (not 兆, which also means mega). The digits change, so `l10n check` warns "numbers not found"; name the block in NOTES. | "3 million parameters" → 300 万个参数 ; "1.5 billion tokens" → 15 亿个词元 ; "thousands of kernels" → 成千上万个卷积核 |
| Percent | Half-width `%` attached to the number; "percentage point" → 个百分点. | "an accuracy of 92.3%" → 92.3% 的准确率 ; ✗ 92.3 ％ ✗ 百分之 92.3 |
| Units | Symbols as in the source, with the source's space; never Chinese unit names for symbols (✗ 24 吉字节). Time words written out become 秒, 分钟, 小时, 天, 年. | 24 GiB, 5 ms ; "3 hours" → 3 小时 |
| Ranges | ～ (U+FF5E) without spaces; the unit once, but `%` and 万/亿 on both numbers (GB/T 15835). Math keeps its own notation. (to confirm) | "40–55" → 40～55 ; "45–50%" → 45%～50% ; "2017–2024" → 2017～2024 年 |
| Multipliers | "N× as large/fast" → 是 … 的 N 倍, N 倍于; for speed also 快 N 倍 only with 约. A reduction is never 减少 N 倍: "2× smaller" → 缩小到二分之一, 减半. Keep `×` in slides, tables, captions and math. | "about 2× faster" → 速度约为原来的 2 倍 ; "3× larger" → 是 … 的 3 倍 ; slide: 快 2× |
| Approximation | 约, 大约 before the number (not both 约 and 左右); "an order of magnitude" → 一个数量级. | "roughly 90%" → 约 90% |
| Ordinals | 第 + numeral + measure word. | "the third epoch" → 第三轮 ; "layer 3" → 第 3 层 |
| Years, decades | "in 2017" → 2017 年; "the 2010s" → 2010 年代. | ResNet 于 2015 年问世。 |
| Money | Amount, space, currency name, in the source's currency. | "\$2.50" → 2.50 美元 |

**Measure words.** A count before a noun needs one; pick from this table and
record any other in DECISIONS. With a symbol: $n$ 个样本, $k$ 层.

| Measure word | For |
|---|---|
| 个 | 参数, 样本, 词元, 通道, 卷积核, 单元, 位置, 元素, 批量, 模型, 函数, 类别, GPU (4 个 GPU) |
| 层 | layers as depth: 3 层, 两层的网络; but 两个隐藏层, 一个卷积层 when the layer is an object |
| 张, 幅 | 张 for 图像, 照片, 图, 表, 幻灯片, 显卡; 幅 also for 图像 (not both in one page) |
| 轮, 步, 次 | 轮 epochs; 步 optimization or time steps (1000 步); 次 iterations, runs, calls, multiplications |
| 维, 阶, 倍, 位 | $d$ 维; 二阶; 2 倍; 32 位 (bits), 两位小数 |
| 行, 列, 项, 对, 组, 种, 类, 条, 篇, 段 | rows, columns; terms of a sum; pairs; groups; kinds; classes; 路径, 序列, 规则, 曲线; 论文; 代码, 文本 |

Quantities, signs and orders of magnitude stay exact; do not round or convert
units or currencies.

## 5. Names

- **Models, libraries, datasets, companies, universities, courses, hardware,
  benchmarks:** exactly as in English: Fashion-MNIST, RTX 4090, OpenAI,
  Stanford CS231n (✗ 斯坦福 CS231n), CMU 11-785.
- **People:** Latin script as in the source, no title (no 教授, 博士, 先生).
  Never turn a romanised Chinese, Japanese or Korean name back into characters
  (Kaiming He stays Kaiming He): the guess is often wrong, and citations and
  searches use the Latin form. "Gauss grounded least squares in probability
  theory." → Gauss 把最小二乘法建立在概率论的基础上。 (to confirm)
- **Terms named after people:** as in the glossary. The household names of
  mathematics are transliterated inside terms: 高斯分布, 贝叶斯定理, 马尔可夫链,
  拉格朗日乘子, 牛顿法, 泰勒展开, 伯努利分布, 泊松分布, 欧拉法, 傅里叶变换,
  香农熵, 玻尔兹曼分布, 朗之万动力学, 伊藤积分, 雅可比矩阵, 海森矩阵,
  利普希茨连续, 贝尔曼方程, 蒙特卡罗. All others stay in Latin with a space:
  Frobenius 范数, Hadamard 积, Jensen 不等式, Cholesky 分解, Nesterov 动量,
  Wasserstein 距离, Huber 损失, Xavier 初始化, He 初始化. A name not in the
  glossary: Latin, and record it in DECISIONS.
- **The book:** *Dive into Deep Learning* stays in English; "this book" → 本书.
- **Languages and places:** Chinese (英语, 美国); programming languages stay
  (Python).
- **Link text:** translate descriptive words, keep titles: "[Assignment 2](…)"
  → [作业 2](…), "Homework 2 Part 1" → 作业 2 第 1 部分; "Adapted from" →
  改编自.
- **Structure words:** chapter 章, section 节, subsection 小节, appendix 附录,
  part 部分; Summary → 小结, Exercises → 练习, Discussions → 讨论; Theorem 定理,
  Proposition 命题, Lemma 引理, Corollary 推论, Definition 定义, Proof 证明,
  Remark 注, Example 例.

## 6. Headings, captions, slides, exercises, notes

| Kind | Rule | Example |
|---|---|---|
| `heading` | A noun phrase or a verb–object phrase; no final punctuation, no English gloss, no 的 chain. English "-ing + object" becomes verb + object or object + 的 + noun. A question keeps ？. Not a single character: add the noun that makes it a title. | "Cross-Correlation" → 互相关 ; "Training the Model" → 训练模型 ; "The Layer" → 层的定义 (✗ 层) ; "Why Depth Helps" → 为什么深度有用？ |
| `caption` | Body register without 我们; 。 exactly when the English caption has a full stop. Keep `(a)`, `(b)`. | "Two-dimensional cross-correlation. The shaded portions are …" → 二维互相关。阴影部分是 … |
| `slide_title`, `kicker` | A short phrase, as short as the English, no final punctuation; "A vs. B" → A 与 B; `&` → 与. | `title="The 2D Cross-Correlation Operation"` → `title="二维互相关运算"` ; `[Sliding-Window Inner Products]{.kicker}` → `[滑动窗口内积]{.kicker}` |
| `slide_text` | Short statements; no 我们, no 你; drop 的, 一个, 了 where the phrase still reads; one line where the English is one line; 。 exactly when the English bullet ends in a full stop. A bold lead-in keeps its bold; its colon goes outside (§2.3). | "- **Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent." → - **学习与手工设计**：深层 CNN 通过梯度下降自动发现成千上万个这样的卷积核。 |
| `exercise` | Imperative without subject and without 请 (证明, 推导, 实现, 计算, 说明); direct questions with ？; 你 where the English says "you". A bold title stays bold, its 。 outside. `[code]`, `●` and `:exlabel:` are copied. | "Derive the finite-difference operator for the second derivative. Which structures respond most strongly?" → 推导二阶导数的有限差分算子。图像中的哪些结构对它的响应最强？ |
| `tab_note` | Body register. Describe this framework only; never compare frameworks. API names stay in code. | "PyTorch's `nn.LazyLinear(1)` requires only the output dimension." → PyTorch 的 `nn.LazyLinear(1)` 只需要给出输出维度。 |
| `note` | Register of its surroundings. Labels: **Note:** → **注意**：, **Warning:** → **警告**：, **Tip:** → **技巧**： (not 提示, which is *prompt*), **Answers:** → **答案**：, "Concept Check" → 概念检查. | |
| `table_row` | Noun phrases; numbers and units as in §4. | "Training time" → 训练时间 |

The cover kicker `[Dive into Deep Learning · §2.1]{.kicker}` is rewritten by the
build; copy it unchanged.

## 7. Known pitfalls

### 7.1 Words with more than one sense

| English | Sense → Chinese | Not |
|---|---|---|
| bias | parameter $b$ → 偏置; of an estimator → 偏差 (偏差–方差); inductive bias → 归纳偏置; social → 偏见 | 偏差 for $b$; 偏见 for an estimator |
| kernel | convolution → 卷积核 (卷积核大小); kernel methods, covariance → 核函数, 核矩阵; Markov → 转移核; GPU or OS code → 内核; of a linear map → 核 (null space → 零空间) | 核函数 for a GPU kernel; 内核 for a convolution |
| return | RL → 回报; of a function → 返回, 返回值 | 回报 for "returns outputs" (the checker may warn; a false match, name it in NOTES) |
| recall | metric → 召回率; "recall that" → 回想一下, 前面讲过 | 召回 for the verb |
| score | $\nabla \log p$ → 分数函数 (always the compound); attention → 注意力分数; a metric → 分数 (F1 分数, BLEU 分数) | 得分 and 分数 mixed in one page |
| sample | noun → 样本 (样本均值, 生成的样本); to sample → 采样 | 抽样 for drawing from a distribution; 取样 |
| example | a data point → 样本 (训练样本); "for example" → 例如; a worked example → 例子 | 实例, 样例 |
| feature | → 特征 (特征图, 特征向量); of a library or product → 功能 | 功能 for an input feature |
| eigenvector, feature vector | both 特征向量. On a page that uses both, gloss each on first use and afterwards write 特征值对应的特征向量 or 样本的特征向量 where the sentence alone does not decide. | 本征向量 |
| value | RL → 价值 (价值函数); attention → 值; a number → 值 | 价值 for an attention value |
| state | → 状态 (隐状态, 状态空间); "state of the art" → 最先进的, 当前最好的 | 州 |
| policy, action, reward, agent | 策略, 动作, 奖励, 智能体 | 政策, 行动, 报酬, 代理 |
| loss, cost, objective, error, risk | Five words, five terms: 损失, 代价 (cost function 代价函数), 目标函数, 误差, 风险. Resource cost → 成本 (推理成本); "at the cost of" → 以 … 为代价. | merging them |
| error | 误差 (训练误差, 泛化误差, 舍入误差); error rate → 错误率; a bug or exception → 错误 | 错误 for 误差 |
| accuracy, precision, recall | 准确率, 精确率, 召回率; numeric precision → 精度 (双精度, 混合精度) | 精度 for accuracy |
| generalization | ML → 泛化 (泛化误差); of a theorem or formula → 推广; *generalized* linear model → 广义线性模型 | 泛化 for a theorem |
| normal, norm, normalization | normal distribution → 正态分布 (标准正态分布; "Gaussian" → 高斯分布); norm → 范数; normalization → 归一化 (批量归一化, 归一化常数); standardization → 标准化; regularization → 正则化; normal equations → 正规方程; normal vector → 法向量 | 标准高斯分布 for "standard normal"; 规范化 |
| inference | running a trained model → 推理; statistics → 推断 (贝叶斯推断, 变分推断); "the layer infers its input size" → 自动推断出 | 推理 for Bayesian inference |
| translation | of an image or function → 平移 (平移不变性, 平移等变性); of language → 翻译 | 翻译不变性 |
| activation | values → 激活值; the function → 激活函数; "nonlinear activations between layers" → 非线性激活 | 激活值 for the function (the checker may warn; a false match) |
| memory | host RAM → 内存; GPU → 显存; LSTM, cognition → 记忆 | 记忆体 |
| attention | 注意力 (自注意力, 多头注意力) | 注意 (it means "note") |
| step size, stride, learning rate | 步长; 步幅; 学习率 | 步长 for a convolution stride |
| batch, epoch, iteration | 批量 (一个批量, 小批量); 轮; 迭代. batching → 批处理 | 批次 and 批量 mixed |
| dense | dense layer → 全连接层 (the same term as *fully connected layer*); dense array, matrix → 稠密 | 密集层 |
| patch, window | image patch → 图像块; sliding window → 窗口 | 补丁 |
| mask | noun → 掩码; verb → 掩蔽, 遮住 | 面具, 口罩 |
| map, mapping | → 映射; feature map → 特征图; edge map → 边缘图 | 地图 |
| model | → 模型; "mental model" → 心智模型 (the checker sees 模型; fine) | |
| likelihood, evidence | 似然; Bayesian → 证据 (证据下界); "empirical evidence" → 实验证据, 依据 | 可能性 |
| rank, trace, order | 秩; 迹; second-order → 二阶; ordering → 顺序; order of magnitude → 数量级; ranking → 排序 | 排名 for matrix rank |
| domain, support | of a function → 定义域, 支撑集; of data → 领域 | |
| performance | of a model → 性能, 效果; of code → 性能, 速度 | 表现 for every case |
| number | a count → 数量, 个数; a number → 数; an identifier → 编号 | 号码 |
| downsample | 下采样 (not 降采样; one form per book) | |
| trade-off | 权衡; where the sentence names what is paid → 代价 | 折中 and 权衡 mixed |
| entry, element, unit | of a matrix or tensor → 元素; of a network → 单元 | 条目, 入口 |

### 7.2 Translationese

| Pattern | Write instead | Example |
|---|---|---|
| "Note that", "It is important to note that" | Drop it, or 注意，… ; 需要注意的是 at most once per page | "Notice that the output shrinks slightly along each axis." → 注意，输出在每个轴上都略有缩小。 |
| "Recall that" | 回想一下，… ; 前面讲过，… | "Recall that $b$ is the bias." → 前面讲过，$b$ 是偏置。 |
| Inanimate subject + "tells us", "allows us to" | 由此可见, 这说明, 这样就可以 | "This result tells us that depth helps." ✗ 这个结果告诉我们深度是有帮助的。 ✓ 由此可见，深度是有用的。 |
| "we can" every time as 我们可以 | The verb itself, or 可以 without subject | "we can compute" → 可以算出 ; 我们算出 |
| "there is / there are" as 存在着, 有着 | 有, or a full verb | "There are three reasons." → 原因有三个。 |
| "is able to", "in order to" | 能, 为了 or nothing | ✗ 为了能够去验证 ✓ 为了验证 |
| "based on" as 基于 every time | 根据, 按, 用, 以 … 为基础 | "a model based on attention" → 以注意力为基础的模型 |
| "in terms of" | 就 … 而言, 用 … 表示 | "derive … in terms of $\partial \ell / \partial \mathbf{Y}$" → 用 $\partial \ell / \partial \mathbf{Y}$ 表示 … |
| "respectively" at the end | 分别 before the verb | 输出形状分别为 $2 \times 2$ 和 $5 \times 7$ |
| "one of the most widely used" | 最常用的 … 之一 | 最常用的优化器之一 |
| "for 10 epochs" | 训练 10 轮 | ✗ 为 10 个轮训练 |
| "more and more", "as … as possible" | 越来越, 尽可能 | |
| Stacked 的 and 性: "the effectiveness of the scalability of" | A verb or adjective | ✗ 可扩展性的有效性 ✓ 是否容易扩展 |
| Idioms translated by image | The meaning | "under the hood" → 在内部 ; "from scratch" → 从零开始 ; "out of the box" → 开箱即用 ; "rule of thumb" → 经验法则 ; "sanity check" → 合理性检查 |
| English punctuation habits | Chinese ones | ✗ 然而, ✗ 因此，， ✗ a comma before 和 ✗ `“卷积.”` ✓ `“卷积”。` |

Logical and mathematical phrases are fixed: because → 因为, 由于; therefore →
因此, 所以; however → 然而, 但是; if … then → 如果 … 那么, 若 … 则; only if →
仅当; if and only if → 当且仅当; at most → 至多; at least → 至少; "let $x$ be
…" → 设 $x$ 为 …, 令 $x = …$; such that → 使得; with respect to → 关于
(derivative: 对 $x$ 求导, 关于 $x$ 的导数); without loss of generality →
不失一般性; for all → 对任意, 对所有; there exists → 存在; respectively → 分别;
almost surely → 几乎必然; i.i.d. → 独立同分布; in expectation → 在期望意义下;
converges to → 收敛到; up to (a sign, a constant) → 至多相差一个 …; bounded →
有界; smooth → 光滑; elementwise → 逐元素; end to end → 端到端; denoted →
记作; dot product → 点积.

### 7.3 Mainland usage, not Taiwan usage

行 is a row and 列 a column (Taiwan usage is the reverse). Write 网络, 程序,
软件, 硬件, 信息, 数据, 内存, 默认, 函数, 变量, 对象, 数组, 文件, 字符串, 算法,
优化, 概率, 标量, 采样, 调用, 代码, 线程, 链接; not 网路, 程式, 软体, 硬体, 资讯,
资料, 记忆体, 预设, 函式, 变数, 物件, 阵列, 档案, 字串, 演算法, 最佳化, 机率,
纯量, 取样, 呼叫, 程式码, 执行绪, 连结.

## 8. Sources

Read for this draft, in this repository, on 2026-11-26:

- `glossary/terms/zh-Hans.jsonl`, `glossary/review/zh-Hans.md` (the glossary
  review with its open decisions) and `glossary/notes/east-asian.md`, which
  record what d2l-zh attests (`TERMINOLOGY.md` and chapter prose at commit
  `e6b18cce`), the term list of the Chinese translation of Goodfellow et al.
  (花书) and Easy RL.
- `evals/zh-conv-layer/`: the three answer sets (Fable, Opus, Sonnet) with
  their DECISIONS and NOTES; the examples in §2 and §6 are taken from them.
- Quarto's `share/language/_language-zh.yml` in d2l-neu's `.venv-build` for the
  reference prefixes (§2.4); `tools/l10n/protect.py` for what the check
  enforces.
- Emphasis and reference behaviour (§2.2–§2.4): tested with the pandoc bundled
  in d2l-neu's `.venv-build`, as pandoc Markdown and as CommonMark.

Not checked (stated from the drafter's knowledge, not from a fetched copy):

- GB/T 15834—2011 《标点符号用法》 and GB/T 15835—2011 《出版物上数字用法》: the
  rules on full-width punctuation, “ ” quotation marks, the 浪纹线 ～ for
  ranges and the repetition of % and 万/亿 in ranges.
- The printed 《动手学深度学习》, 周志华《机器学习》, 李航《统计学习方法》: no
  page was read for this draft.
- W3C *Requirements for Chinese Text Layout* (CLREQ) on Chinese–Latin spacing.
- How the Chinese build prints `:citet:` and `:eqref:`; §2.4 phrases around
  both so that either form reads correctly.

## 9. Settings

`languages.yml` (existing keys, `tools/l10n/config.py` and `CheckOptions`): the
`zh` entry needs **no change**. The defaults are the right ones:

- `check.cjk_adjacent: letters` (default): a role or inline math touching a
  Chinese character is an error; full-width punctuation next to it is fine.
- `check.thousands: keep` (default): `10,000` stays.
- `check.numbers: warning`, `check.order: warning` (defaults).
- no `typography: {nbsp: true}`: its rules are German.
- `forum: own`, `forum_category: 16` are already set; fonts (Noto Sans SC,
  Noto Serif/Sans/Mono CJK SC) are already preset for `zh` in
  `tools/l10n/build.py`.

`zh/strings.yml` (existing mechanism, the `quarto.*` overrides):

- `quarto.crossref-sec-prefix: "§"` so that a section reference prints §5.1,
  the same form `§{sec_x}` prints, instead of Quarto's 小节 5.1 (to confirm).
- `quarto.crossref-fig-prefix: "图"`, `quarto.crossref-tbl-prefix: "表"`,
  `quarto.crossref-eq-prefix: "式"`: Quarto's values, stated so that an
  upgrade does not change them.

Would need new code:

1. **第 3 章 for chapter references.** Quarto has a prefix only (now 章节 3);
   Chinese needs a suffix (第 … 章). A `crossref-ch-suffix` handled in
   `fix_crossref_numbers.py` would do it.
2. **Glossary check across the §2.2 space.** `glossary._present` looks for the
   exact term, so a `verified` mixed-script entry written without a space
   (Xavier初始化) is reported missing when the translator writes Xavier 初始化.
   Either respell the entries (§11) or let the check ignore spaces between a
   Chinese character and a Latin letter or digit.
3. **A spacing check.** `l10n check` enforces §2.2 only for roles and math. A
   warning for a Chinese character directly next to a Latin letter, digit or
   backtick (outside protected tokens), and for a line break between two
   Chinese characters, would make the rule mechanical.
4. **Emphasis style.** Chinese fonts have no italic. The Chinese HTML, slide
   and PDF styles should render `*…*` in a Kai face (楷体) or bold instead of a
   slanted Song face (a build setting, not a translator rule).
5. Optional: `east_asian_line_breaks` in the pandoc reader for `zh`, so that a
   stray line break inside a paragraph does not print a space (§2.2).

## 10. For the author to confirm

1. **Spaces between Chinese and Latin, digits, math, code** (§2.2). Proposed:
   yes, one typed space everywhere, as all three test translators did and as
   the build already requires around references and math. The alternative is
   d2l-zh's no-space text with spaces only where the build needs them.
2. **Reader address** (§1). Proposed: 我们 for the authors, 你 for the reader,
   never 您; exercises as bare imperatives without 请.
3. **Quotation marks** (§2.1). Proposed: “ ” and ‘ ’ (mainland standard), never
   「」.
4. **Punctuation outside emphasis** (§2.3): `**小标题**：` and `**题名**。`.
   Proposed as written; for **Proof.** and **Proposition.** the alternative
   is the textbook form `**证明** ` with a space and no 。.
5. **English gloss** (§3.3). Proposed: once per page, for terms the English
   introduces with emphasis and for the short list of coined or contested
   terms; never in headings, captions, slides, exercises.
6. **Acronyms** (§3.1). Proposed: keep CNN, SGD, MLP wherever the English
   writes the acronym, also in body text.
7. **暂退法 and 跳元模型** (§3.2). Proposed: keep both d2l-zh terms with the
   English gloss on first use. The alternative, which most mainland texts
   follow, is dropout and skip-gram in Latin.
8. **People's names in Latin script** (§5), including Gauss, Bayes, Newton in
   running text (高斯, 贝叶斯, 牛顿 only inside terms). Proposed: yes, one rule;
   the alternative is to write the household names in Chinese everywhere.
9. **Person names inside terms** (§5). Proposed: the transliterated list in §5
   and Latin for all others; open cases are 利普希茨 or Lipschitz, 费希尔 or
   Fisher, 海森 or 黑塞 (Wikipedia) or Hessian (花书), 蒙特卡罗 or 蒙特卡洛.
10. **Ranges with ～** (§4): 40～55, 45%～50%. Proposed: yes (GB/T 15835); the
    alternative is to keep the English en dash, as tables and figures do.
11. **Numbers** (§4). Proposed: digits where the English has digits, Chinese
    numerals where it spells them; 万/亿 for million/billion; `10,000` with
    the comma.
12. **Reference prefixes** (§2.4, §9). Proposed: §5.1 for sections now, 第 3 章
    for chapters once the build can print a suffix; translators write no noun
    around `:numref:` either way.
13. **`$i^{\textrm{th}}$`** (§2.5). Proposed: copy it and add no 第, as ruled
    for German and Japanese; $i^{\textrm{th}}$ 层 reads less well in Chinese
    than 第 $i$ 层, which only a change of the English source could give.
14. **Hand-written citations in slides** (§2.4): （Luo et al., 2016） as in
    English, or Luo 等，2016.
15. **图像块 for *patch*** (§7.1, glossary `image-patch`), against the shorter
    图块 that the Fable test run used; and 下采样 against 降采样.
16. **Section names** (§5): 小结, 练习, 讨论 as in d2l-zh.
17. **Notes** (§6): **Tip:** → **技巧**： because 提示 is *prompt*.

## 11. Proposed glossary corrections

1. **Mixed-script entries need the §2.2 space** (30 entries; the six binding
   ones first): Xavier初始化 → Xavier 初始化, Frobenius范数 → Frobenius 范数,
   Hadamard积 → Hadamard 积, softmax回归 → softmax 回归, VC维 → VC 维,
   Nesterov动量 → Nesterov 动量; likewise the 24 proposed entries of the same
   shape (Q学习, p值, n元语法, F1分数, Jensen不等式, Transformer块,
   视觉Transformer, softmax温度, QK电路, …). Or change the check instead (§9,
   item 2).
2. `frechet-inception-distance`: Fréchet Inception距离（FID） → Fréchet
   Inception 距离; the acronym is not part of the term.
3. `kernel-launch`: 核函数启动 → 内核启动. A GPU kernel is 内核; 核函数 is the
   kernel of kernel methods (§7.1).
4. `kv-cache`: 键值缓存 → KV 缓存. The English writes the acronym, and §3.1
   keeps acronyms.
5. `sampling-error`: 抽样误差, while the note on `sample` says the act of
   drawing is "采样, never 抽样". 抽样误差 is the statistics term and may stay;
   then the note on `sample` should name the exception.
6. `normalizing-flow`, `continuous-normalizing-flow`: 标准化流 → 归一化流, to
   keep one family with 归一化, 归一化常数 (open in the glossary review,
   decision 8).
7. `long-short-term-memory`: 长短期记忆网络 is right for the network; the
   English mostly writes LSTM, which stays LSTM (§3.1). Add 长短期记忆 as an
   alternative for "long short-term memory" used as the mechanism.
8. `activation`: 激活值 is `verified_d2l` and is reported for "nonlinear
   activations" (the functions between layers); all three test translators
   noted the false match. Add 激活 and 非线性激活 as alternatives.
9. `dropout`: the alternative 暂退法（Dropout） has a capital and parentheses;
   drop it (the gloss is §3.3's job, lower case).
10. `precision`, `accuracy`: remove the alternatives 精确度, 精度, 准确性; they
    let the check accept the forms §3.1 excludes.
11. Missing concepts the guide's §7.1 decides and the glossary does not
    control: error (误差), inference (推理 / 推断), dense layer (全连接层),
    patch (图像块), window (窗口), entry/element (元素), unit (单元),
    edge map (边缘图), trade-off (权衡), optimizer (优化器), memory
    (内存 / 显存 / 记忆).
