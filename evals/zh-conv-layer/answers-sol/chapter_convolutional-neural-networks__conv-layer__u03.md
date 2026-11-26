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
1. [code] **手工设计卷积核。** 手工设计几个卷积核。
    1. 给定方向向量 $\mathbf{v} = (v_1, v_2)$，推导一个边缘检测卷积核，用来检测与 $\mathbf{v}$ 正交的边缘，也就是沿 $(v_2, -v_1)$ 方向的边缘。
    1. 推导一个用于计算二阶导数的有限差分算子。图像中的哪些结构对它的响应最强？
    1. 要求得 $d$ 阶导数，卷积核的最小尺寸是多少？
    1. 你会怎样设计模糊卷积核？为什么会用到这种卷积核？
    1. 加载一张照片，将其转换为单通道，并下采样至约 $100 \times 100$ 像素（`corr2d` 用 Python 循环计算）。分别应用边缘检测器、模糊卷积核和锐化卷积核（“两倍的恒等卷积核减去模糊卷积核”），将每个结果与原图并排显示。照片的边缘图能显示哪些黑条图像的边缘图无法显示的信息？

    *改编自 Stanford CS231n [作业 2](https://cs231n.github.io/assignments2024/assignment2/) 的 ConvolutionalNetworks 笔记本。*

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
2. [code] **用带状矩阵表示卷积。** :numref:`subsec_conv_matmul` 中的图块矩阵展开的是输入。反过来，也可以展开卷积核。设 $\mathbf{X}$ 为 $4 \times 4$ 的输入，$\mathbf{K}$ 为 $3 \times 3$ 的卷积核，$\textrm{vec}(\cdot)$ 表示按行展平。
    1. 构造 $4 \times 16$ 矩阵 $\mathbf{W}$，使其满足 $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$，其中 $\mathbf{Y}$ 是 $\mathbf{X}$ 与 $\mathbf{K}$ 的互相关结果。每行有多少个非零元素？$\mathbf{W}$ 中有多少个不同的值？
    1. 编写代码构造 $\mathbf{W}$，以 `K` 为依据，并验证矩阵乘法的结果与 `corr2d(X, K)` 一致。
    1. 对于 $n \times n$ 的输入和 $k \times k$ 的卷积核，分别计算以下数量：将 $\mathbf{W}$ 存为稠密数组时的元素数、$\mathbf{W}$ 的非零元素数，以及图块矩阵的元素数。这三个数量各自如何随 $n$ 增长？
    1. 保持 $\mathbf{W}$ 的稀疏模式，但让每行使用各自的 $k^2$ 个值。得到的是一个局部连接层。取四阶张量 $\mathsf{V}$，即 :numref:`sec_why-conv` 中公式 $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$ 中的张量，只保留 $k \times k$ 窗口内的偏移量 $(a, b)$，暂不施加平移等变性所要求的权重共享。对于 $n \times n$ 的输入，分别计算这个局部连接层、输入和输出形状均与之相同的全连接层，以及卷积层的参数量。

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
3. [code] **有效感受野。** ● 本节的理论感受野 $r$ 界定了哪些输入*能够*影响输出，有效感受野则衡量这些输入的实际影响有多大 :cite:`Luo.Li.Urtasun.ea.2016`。堆叠 $L$ 个单通道 $3 \times 3$ 卷积层，不加填充、偏置或非线性变换，输入大小为 $32 \times 32$。使用 :numref:`subsec_learning_kernel` 中的内置卷积层。卷积核的每个元素都独立地从标准高斯分布中采样。
    1. 设 $L = 3$，对各卷积核采样一次。从最后一层靠近中心的一个输出单元，将值为 1 的梯度反向传播至输入，并绘制输入梯度的幅值。将其非零区域与 $7 \times 7$ 的理论感受野比较，并指出幅值在哪里最大。
    1. 分别对 $L = 3$ 和 $L = 10$，将各卷积核重新采样 500 次，对所得输入梯度的平方取均值。对每幅均值图逐列求和，得到各列的分布，再将其归一化，使分布的总和为 1。该分布的标准差与理论感受野边长 $2L + 1$ 的比值如何随深度变化？
    1. :numref:`fig_conv_rf_paths` 统计了一维情形下 $L = 3$ 时的路径数。证明对于一般的 $L$，输入梯度平方的期望逐列求和后，正比于 $L$ 个向量 $(1, 1, 1)$ 的卷积。推导归一化后的分布的标准差如何随 $L$ 增长。

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
4. [code] **卷积的反向传播。** ● 设 $\mathbf{Y}$ 为输入 $\mathbf{X}$ 与卷积核 $\mathbf{K}$ 的单通道互相关结果，由 `corr2d` 计算。设 $\ell$ 为标量损失，它对 $\mathbf{X}$ 和 $\mathbf{K}$ 的依赖仅通过 $\mathbf{Y}$ 体现。
    1. 推导 $\partial \ell/\partial \mathbf{K}$ 和 $\partial \ell/\partial \mathbf{X}$ 关于 $\partial \ell/\partial \mathbf{Y}$ 的表达式，写法参照 :numref:`sec_backprop` 中的反向传播公式。
    1. 不使用自动微分，实现你推导的 $\partial \ell/\partial \mathbf{K}$ 表达式，取损失为 $\ell = \sum_{i,j} G_{ij} Y_{ij}$，其中 $\mathbf{G}$ 是形状与 $\mathbf{Y}$ 相同的固定随机矩阵。用步长为 $10^{-2}$ 的中心有限差分法在小规模的随机 `X` 和 `K` 上计算梯度，与解析结果比较，并报告最大相对误差。
    1. 检验你推导的 $\partial \ell/\partial \mathbf{K}$ 表达式，将其结果与 :numref:`subsec_learning_kernel` 实验中用自动微分算出的卷积核梯度比较。

    *改编自 Michael Nielsen 的[《神经网络与深度学习》](http://neuralnetworksanddeeplearning.com/chap6.html)第 6 章，以及 Stanford CS231n [作业 2](https://cs231n.github.io/assignments2024/assignment2/)的 ConvolutionalNetworks 笔记本。*

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
- “blur kernel” → 「模糊卷积核」，“sharpening kernel” → 「锐化卷积核」，“locally connected layer” → 「局部连接层」。
- “sparsity pattern” → 「稀疏模式」，“support” → 「非零区域」，“centered finite difference” → 「中心有限差分」。

=== NOTES ===
- ec8621f8: 术语检查把 sparsity pattern 中的 sparsity 匹配为“稀疏性”，这里的术语是“稀疏模式”。题目没有指定卷积核的元素值。每行实际非零元素数和矩阵中的不同取值数取决于卷积核里是否含零或重复值。
