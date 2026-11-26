<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u03 lang=zh rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:2294243a kind=heading-->
## Exercises
<!--tr:2294243a-->
## 练习

<!--en:6f4ddec6 kind=exercise-->
1. [code] **Hand-designed kernels.** Design some kernels manually.
    1. Given a directional vector $\mathbf{v} = (v_1, v_2)$, derive an
       edge-detection kernel that detects edges orthogonal to $\mathbf{v}$,
       i.e., edges in the direction $(v_2, -v_1)$.
    1. Derive a finite difference operator for the second derivative.
       Which structures in images respond most strongly to it?
    1. What is the minimum size of a kernel to obtain a derivative of
       order $d$?
    1. How would you design a blur kernel? Why might you want to use such
       a kernel?
    1. Load a photograph, convert it to one channel, and downsample it to
       about $100 \times 100$ pixels (`corr2d` loops in Python). Apply the
       edge detector, the blur kernel, and the sharpening kernel "twice
       the identity minus the blur", and display each result next to the
       original. What does the edge map of a photograph show that the
       edge map of the black band could not?

    *Adapted from Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    ConvolutionalNetworks notebook.*
<!--tr:6f4ddec6-->
1. [code] **手工设计卷积核。** 手工设计一些卷积核。
    1. 给定方向向量 $\mathbf{v} = (v_1, v_2)$，推导一个边缘检测卷积核，用来检测与 $\mathbf{v}$ 正交的边缘，即沿方向 $(v_2, -v_1)$ 的边缘。
    1. 推导二阶导数的有限差分算子。图像中的哪些结构对它的响应最强？
    1. 要得到 $d$ 阶导数，卷积核的最小尺寸是多少？
    1. 你会如何设计一个模糊卷积核？为什么会想用这样的卷积核？
    1. 载入一张照片，把它转换为单通道，并下采样到大约 $100 \times 100$ 像素（`corr2d` 是在 Python 中循环的）。分别应用边缘检测器、模糊卷积核和锐化卷积核“两倍的恒等核减去模糊核”，并把每个结果显示在原图旁边。照片的边缘图显示出了哪些黑色竖带的边缘图无法显示的东西？

    *改编自 Stanford CS231n [作业 2](https://cs231n.github.io/assignments2024/assignment2/) 的 ConvolutionalNetworks notebook。*

<!--en:ec8621f8 kind=exercise-->
2. [code] **Convolution as a banded matrix.** The patch matrix of
   :numref:`subsec_conv_matmul` unfolds the input. The opposite
   construction unrolls the kernel. Let $\mathbf{X}$ be a $4 \times 4$
   input, $\mathbf{K}$ a $3 \times 3$ kernel, and $\textrm{vec}(\cdot)$
   the row-by-row flattening.
    1. Construct the $4 \times 16$ matrix $\mathbf{W}$ with
       $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$,
       where $\mathbf{Y}$ is the cross-correlation of $\mathbf{X}$ with
       $\mathbf{K}$. How many nonzero entries does each row have, and how
       many distinct values appear in $\mathbf{W}$?
    1. Build $\mathbf{W}$ in code from `K` and confirm that the product
       reproduces `corr2d(X, K)`.
    1. For an $n \times n$ input and a $k \times k$ kernel, count the
       entries of $\mathbf{W}$ stored as a dense array, the nonzero
       entries of $\mathbf{W}$, and the entries of the patch matrix. How
       does each count grow with $n$?
    1. Keep the sparsity pattern of $\mathbf{W}$ but give every row its
       own $k^2$ values. The result is a locally connected layer: the
       fourth-order tensor $\mathsf{V}$ of :numref:`sec_why-conv` in
       $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$,
       restricted to a $k \times k$ window of offsets $(a, b)$ before
       translation equivariance ties the weights. For an $n \times n$
       input, count the parameters of this layer, of a fully connected
       layer with the same input and output shapes, and of the
       convolution.

    *Adapted from CMU 11-785,
    [Homework 2 Part 1](https://deeplearning.cs.cmu.edu/S21/index.html).*
<!--tr:ec8621f8-->
2. [code] **把卷积写成带状矩阵。** :numref:`subsec_conv_matmul` 中的图块矩阵展开的是输入。相反的构造则展开卷积核。设 $\mathbf{X}$ 为 $4 \times 4$ 的输入，$\mathbf{K}$ 为 $3 \times 3$ 的卷积核，$\textrm{vec}(\cdot)$ 表示逐行展平。
    1. 构造 $4 \times 16$ 的矩阵 $\mathbf{W}$，使得 $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$，其中 $\mathbf{Y}$ 是 $\mathbf{X}$ 与 $\mathbf{K}$ 的互相关。每一行有多少个非零元素？$\mathbf{W}$ 中出现了多少个不同的值？
    1. 用代码构建 $\mathbf{W}$（由 `K` 出发），并确认乘积能复现 `corr2d(X, K)`。
    1. 对于 $n \times n$ 的输入和 $k \times k$ 的卷积核，分别数出以稠密数组存储的 $\mathbf{W}$ 的元素个数、$\mathbf{W}$ 的非零元素个数，以及图块矩阵的元素个数。每个数目随 $n$ 如何增长？
    1. 保留 $\mathbf{W}$ 的稀疏模式，但让每一行拥有自己的 $k^2$ 个值。结果是一个局部连接层：四阶张量 $\mathsf{V}$（来自 :numref:`sec_why-conv` 的 $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$），限制在 $k \times k$ 的偏移量 $(a, b)$ 窗口内，且平移等变性尚未把权重绑定起来。对于 $n \times n$ 的输入，分别数出这个层、具有相同输入和输出形状的全连接层，以及卷积的参数个数。

    *改编自 CMU 11-785 [作业 2 第 1 部分](https://deeplearning.cs.cmu.edu/S21/index.html)。*

<!--en:54bcee26 kind=exercise-->
3. [code] **Effective receptive field.** ● The theoretical receptive
   field $r$ of this section bounds which inputs *can* influence an
   output; the effective receptive field measures how much
   they actually do :cite:`Luo.Li.Urtasun.ea.2016`. Build a stack of $L$
   single-channel $3 \times 3$ convolutional layers with no
   padding, no bias, and no nonlinearity on a $32 \times 32$ input, using
   the built-in layer of :numref:`subsec_learning_kernel`. Draw every
   kernel entry independently from a standard normal distribution.
    1. For $L = 3$ and a single draw of the kernels, backpropagate a unit
       gradient from an output unit near the center of the last layer to
       the input and plot the magnitude of the input gradient. Compare its
       support with the $7 \times 7$ theoretical receptive field, and say
       where the magnitude is largest.
    1. Average the squared input gradient over 500 draws of the kernels,
       for $L = 3$ and for $L = 10$. Sum each averaged map over its rows
       and normalize the resulting column profile to sum to one. How does
       the ratio of the profile's standard deviation to the theoretical
       side length $2L + 1$ change with depth?
    1. :numref:`fig_conv_rf_paths` counts the paths for $L = 3$ in one
       dimension. Show that for general $L$ the expected squared input
       gradient, summed over rows, is proportional to the convolution of
       $L$ copies of the vector $(1, 1, 1)$. Derive how the standard
       deviation of the normalized profile grows with $L$.
<!--tr:54bcee26-->
3. [code] **有效感受野。** ● 本节的理论感受野 $r$ 界定了哪些输入*能够*影响某个输出；有效感受野度量的则是它们实际影响了多少 :cite:`Luo.Li.Urtasun.ea.2016`。堆叠 $L$ 个单通道 $3 \times 3$ 卷积层，不带填充、偏置和非线性，输入为 $32 \times 32$，使用 :numref:`subsec_learning_kernel` 中的内置层。卷积核的每个元素都独立地从标准高斯分布中采样。
    1. 取 $L = 3$ 并对卷积核采样一次，从最后一层中心附近的一个输出单元向输入反向传播一个单位梯度，并画出输入梯度的幅值。把它的支撑集与 $7 \times 7$ 的理论感受野作比较，并说明幅值在哪里最大。
    1. 对卷积核采样 500 次，求输入梯度平方的均值，分别取 $L = 3$ 和 $L = 10$。把每张平均后的图沿各行求和，并把得到的列剖面归一化，使其和为一。剖面的标准差与理论边长 $2L + 1$ 之比如何随深度变化？
    1. :numref:`fig_conv_rf_paths` 在一维情形下数出了 $L = 3$ 时的路径数。证明对于一般的 $L$，输入梯度平方的期望在沿各行求和之后，正比于 $L$ 个向量 $(1, 1, 1)$ 的卷积。推导归一化剖面的标准差如何随 $L$ 增长。

<!--en:6ab5b67c kind=exercise-->
4. [code] **Backward pass of a convolution.** ● Let $\mathbf{Y}$ be the
   single-channel cross-correlation of an input $\mathbf{X}$ with a kernel
   $\mathbf{K}$, as computed by `corr2d`, and let $\ell$ be a scalar loss
   that depends on $\mathbf{X}$ and $\mathbf{K}$ only through $\mathbf{Y}$.
    1. Derive $\partial \ell/\partial \mathbf{K}$ and
       $\partial \ell/\partial \mathbf{X}$ in terms of
       $\partial \ell/\partial \mathbf{Y}$, in the style of the
       backpropagation equations of :numref:`sec_backprop`.
    1. Implement your expression for $\partial \ell/\partial \mathbf{K}$
       without automatic differentiation for the loss
       $\ell = \sum_{i,j} G_{ij} Y_{ij}$, where $\mathbf{G}$ is a fixed
       random matrix of the shape of $\mathbf{Y}$. Compare it with a
       centered finite-difference gradient with step size $10^{-2}$ on a
       small random `X` and `K`, and report the largest relative error.
    1. Check your expression for $\partial \ell/\partial \mathbf{K}$
       against the kernel gradient that automatic differentiation
       computes in the experiment of :numref:`subsec_learning_kernel`.

    *Adapted from Michael Nielsen,
    [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html),
    Chapter 6, and Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    ConvolutionalNetworks notebook.*
<!--tr:6ab5b67c-->
4. [code] **卷积的反向传播。** ● 设 $\mathbf{Y}$ 是输入 $\mathbf{X}$ 与卷积核 $\mathbf{K}$ 的单通道互相关，由 `corr2d` 计算，并设 $\ell$ 是一个标量损失，它对 $\mathbf{X}$ 和 $\mathbf{K}$ 的依赖只通过 $\mathbf{Y}$ 发生。
    1. 推导 $\partial \ell/\partial \mathbf{K}$ 和 $\partial \ell/\partial \mathbf{X}$ 关于 $\partial \ell/\partial \mathbf{Y}$ 的表达式，形式上仿照 :numref:`sec_backprop` 中的反向传播方程。
    1. 不借助自动微分，实现你得到的 $\partial \ell/\partial \mathbf{K}$ 表达式，损失取 $\ell = \sum_{i,j} G_{ij} Y_{ij}$，其中 $\mathbf{G}$ 是一个形状与 $\mathbf{Y}$ 相同的固定随机矩阵。把它与步长为 $10^{-2}$ 的中心有限差分梯度作比较（取小的随机 `X` 和 `K`），并报告最大相对误差。
    1. 把你的 $\partial \ell/\partial \mathbf{K}$ 表达式与 :numref:`subsec_learning_kernel` 的实验中由自动微分算出的卷积核梯度相对照。

    *改编自 Michael Nielsen 的 [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html) 第 6 章，以及 Stanford CS231n [作业 2](https://cs231n.github.io/assignments2024/assignment2/) 的 ConvolutionalNetworks notebook。*

<!--en:657a396a kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/65)
<!--tr:657a396a-->
[讨论](https://d2l.discourse.group/t/65)

<!--en:63c44b48 kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/66)
<!--tr:63c44b48-->
[讨论](https://d2l.discourse.group/t/66)

<!--en:00b73d0b kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/271)
<!--tr:00b73d0b-->
[讨论](https://d2l.discourse.group/t/271)

<!--en:b79af65b kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/17996)
<!--tr:b79af65b-->
[讨论](https://d2l.discourse.group/t/17996)

=== DECISIONS ===
- "Exercises" → 练习；"Discussions" → 讨论
- 练习保留 `[code]` 标签和 ● 标记原样；题名写作 `**题名。** `，出处行写作 `*改编自 …。*`
- 课程作业的链接文字译成中文（Assignment 2 → 作业 2，Homework 2 Part 1 → 作业 2 第 1 部分）；书名、课程名、notebook 名保留英文
- "blur kernel" → 模糊卷积核；"sharpening kernel" → 锐化卷积核；"the identity"（卷积核）→ 恒等核
- "locally connected layer" → 局部连接层
- "sparsity pattern" → 稀疏模式；"dense array" → 稠密数组
- "support"（函数的）→ 支撑集；"profile" → 剖面；"magnitude" → 幅值
- "standard normal distribution" → 标准高斯分布（依术语表“高斯分布”）
- "tie the weights" → 把权重绑定起来

=== NOTES ===
- 6f4ddec6: 引号内的 "twice the identity minus the blur" 中，identity 与 blur 简写为“恒等核”“模糊核”，以保持公式般的简洁；正文其余处用“模糊卷积核”。
- 54bcee26: "standard normal distribution" 按术语表写作“标准高斯分布”；中文教材更常见的说法是“标准正态分布”，请确认术语表是否应为该固定搭配留一个条目。
- ec8621f8: "sparsity pattern" 译为通行的“稀疏模式”，不写“稀疏性模式”；检查的警告属误匹配。
- 6ab5b67c: 书名 Neural Networks and Deep Learning 保留英文，检查把其中的 Neural Networks 当作术语“神经网络”，属误匹配。
- 54bcee26: "sum to one" 保留为汉字“一”，未引入原文没有的数字。
