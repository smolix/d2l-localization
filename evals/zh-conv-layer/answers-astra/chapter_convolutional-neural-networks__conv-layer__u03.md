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
1. [code] **手工设计卷积核。** 试着手工设计以下卷积核。
    1. 给定方向向量 $\mathbf{v} = (v_1, v_2)$，推导一个边缘检测卷积核，使它能检测到与 $\mathbf{v}$ 正交的边缘，即沿 $(v_2, -v_1)$ 方向的边缘。
    1. 推导计算二阶导数的有限差分算子。图像中哪些结构会使它产生最强的响应？
    1. 要计算 $d$ 阶导数，所需的最小卷积核大小是多少？
    1. 你会怎样设计一个模糊卷积核？什么情况下会用到这样的卷积核？
    1. 载入一张照片，转换为单通道图像，再下采样到约 $100 \times 100$ 像素（因为 `corr2d` 在 Python 中用循环计算）。分别应用边缘检测器、模糊卷积核，以及用“两倍的恒等卷积核减去模糊卷积核”得到的锐化卷积核。将每个结果与原图并排显示。照片的边缘图揭示了哪些在黑色条带的边缘图中看不到的信息？

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
2. [code] **用带状矩阵表示卷积。** :numref:`subsec_conv_matmul` 中的图像块矩阵是通过展开输入得到的。反过来，也可以展开卷积核。设 $\mathbf{X}$ 为 $4 \times 4$ 输入，$\mathbf{K}$ 为 $3 \times 3$ 卷积核，$\textrm{vec}(\cdot)$ 表示逐行展平。
    1. 构造一个 $4 \times 16$ 矩阵 $\mathbf{W}$，使 $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$ 成立，其中 $\mathbf{Y}$ 是 $\mathbf{X}$ 与 $\mathbf{K}$ 的互相关。每行有多少个非零元素？$\mathbf{W}$ 中有多少个不同的值？
    1. 编写代码，构造矩阵 $\mathbf{W}$，其元素取自 `K`，并确认矩阵乘法的结果与 `corr2d(X, K)` 一致。
    1. 对于 $n \times n$ 输入和 $k \times k$ 卷积核，分别计算将 $\mathbf{W}$ 存为稠密数组时的元素总数、$\mathbf{W}$ 的非零元素数，以及图像块矩阵的元素数。随着 $n$ 增大，这三个数量各如何增长？
    1. 保留 $\mathbf{W}$ 的稀疏性结构，但让每一行都有自己的一组 $k^2$ 个值。这样就得到一个局部连接层。它对应的四阶张量 $\mathsf{V}$ 来自 :numref:`sec_why-conv` 中的公式 $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$：只保留 $k \times k$ 窗口内的偏移量 $(a, b)$，此时尚未用平移等变性约束各位置共享权重。对于 $n \times n$ 输入，分别计算这一层、输入和输出形状与之相同的全连接层，以及相应卷积层的参数量。

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
3. [code] **有效感受野。** ● 本节中的理论感受野 $r$ 限定了哪些输入*可能*影响某个输出，有效感受野则衡量这些输入实际影响有多大 :cite:`Luo.Li.Urtasun.ea.2016`。叠加 $L$ 个单通道 $3 \times 3$ 卷积层，不加填充、偏置或非线性变换，输入大小为 $32 \times 32$。使用 :numref:`subsec_learning_kernel` 中的内置卷积层，从标准高斯分布中独立抽取每个卷积核元素的值。
    1. 取 $L = 3$，随机抽取一组卷积核，从最后一层靠近中心的一个输出单元出发，将大小为 1 的梯度反向传播到输入，画出各输入位置的梯度幅值。将梯度的非零区域与 $7 \times 7$ 理论感受野比较，并指出幅值最大的位置。
    1. 随机抽取 500 组卷积核，对各输入位置的梯度平方取均值。对 $L = 3$ 和 $L = 10$ 两种情形分别计算。对所得的每幅均值图，将各行相加，得到每列的总和，再将这组数值归一化，使其总和为 1。归一化后列分布的标准差与理论感受野边长 $2L + 1$ 的比值如何随深度变化？
    1. :numref:`fig_conv_rf_paths` 给出了 $L = 3$ 时一维情形下的路径数。证明：对于一般的 $L$，将输入梯度平方的期望逐行相加，结果与 $L$ 个向量 $(1, 1, 1)$ 的卷积成正比。推导归一化后列分布的标准差如何随 $L$ 增长。

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
4. [code] **卷积的反向传播。** ● 设 $\mathbf{Y}$ 是输入 $\mathbf{X}$ 与卷积核 $\mathbf{K}$ 的单通道互相关，由 `corr2d` 计算。设 $\ell$ 是一个标量损失，它对 $\mathbf{X}$ 和 $\mathbf{K}$ 的依赖仅通过 $\mathbf{Y}$ 产生。
    1. 推导 $\partial \ell/\partial \mathbf{K}$ 和 $\partial \ell/\partial \mathbf{X}$，用 $\partial \ell/\partial \mathbf{Y}$ 表示，形式参照 :numref:`sec_backprop` 中的反向传播方程。
    1. 不使用自动微分，实现你推导的 $\partial \ell/\partial \mathbf{K}$ 表达式。取损失 $\ell = \sum_{i,j} G_{ij} Y_{ij}$，其中 $\mathbf{G}$ 是形状与 $\mathbf{Y}$ 相同的固定随机矩阵。用步长为 $10^{-2}$ 的中心有限差分计算梯度，在小型随机输入 `X` 和卷积核 `K` 上比较两种方法，报告最大相对误差。
    1. 检验你的 $\partial \ell/\partial \mathbf{K}$ 表达式是否与自动微分在 :numref:`subsec_learning_kernel` 的实验中算出的卷积核梯度一致。

    *改编自 Michael Nielsen 的《[神经网络与深度学习](http://neuralnetworksanddeeplearning.com/chap6.html)》第 6 章，以及 Stanford CS231n [作业 2](https://cs231n.github.io/assignments2024/assignment2/) 的 ConvolutionalNetworks 笔记本。*

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
- locally connected layer 译为“局部连接层”，centered finite difference 译为“中心有限差分”，column profile 在此按归一化后的含义译为“列分布”。
