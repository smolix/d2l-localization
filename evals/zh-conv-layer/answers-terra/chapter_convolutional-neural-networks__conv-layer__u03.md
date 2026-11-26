<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u03 lang=zh rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:2294243a kind=heading-->
## Exercises
<!--tr:2294243a-->

## 习题

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

1. [code] **手工设计卷积核。** 手动设计一些卷积核。
    1. 给定方向向量 $\mathbf{v} = (v_1, v_2)$，推导一个检测与 $\mathbf{v}$ 正交的边缘（即方向为 $(v_2, -v_1)$ 的边缘）的卷积核。
    1. 推导二阶导数的有限差分算子。图像中的哪些结构对它的响应最强？
    1. 要得到 $d$ 阶导数，卷积核的最小尺寸是多少？
    1. 你会如何设计模糊卷积核？为什么可能需要使用这样的卷积核？
    1. 加载一张照片，将其转换为单通道，并下采样至约 $100 \times 100$ 像素（`corr2d` 在 Python 中使用循环）。分别应用边缘检测器、模糊卷积核和“二倍恒等映射减去模糊”的锐化卷积核，并将每个结果与原图并排显示。照片的边缘图展示了哪些黑色条带的边缘图无法展示的内容？

    *改编自 Stanford CS231n 的 [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/) 中的 ConvolutionalNetworks notebook。*

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

2. [code] **将卷积表示为带状矩阵。** :numref:`subsec_conv_matmul` 中的图像块矩阵展开输入。相反的构造则展开卷积核。设 $\mathbf{X}$ 是 $4 \times 4$ 输入，$\mathbf{K}$ 是 $3 \times 3$ 卷积核，$\textrm{vec}(\cdot)$ 表示逐行展平。
    1. 构造 $4 \times 16$ 矩阵 $\mathbf{W}$，使得 $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$，其中 $\mathbf{Y}$ 是 $\mathbf{X}$ 与 $\mathbf{K}$ 的互相关。每行有多少个非零元素？$\mathbf{W}$ 中出现了多少个不同的值？
    1. 在代码中从 `K` 构造 $\mathbf{W}$，并确认其乘积重现了 `corr2d(X, K)`。
    1. 对于 $n \times n$ 输入和 $k \times k$ 卷积核，分别计算以稠密数组存储的 $\mathbf{W}$ 的元素数、$\mathbf{W}$ 的非零元素数以及图像块矩阵的元素数。各个数量如何随 $n$ 增长？
    1. 保留 $\mathbf{W}$ 的稀疏性及其模式，但给每行各自的 $k^2$ 个值。结果是一个局部连接层：即 :numref:`sec_why-conv` 中的四阶张量 $\mathsf{V}$，在 $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$ 中，在平移等变性约束权重相同之前，将偏移 $(a, b)$ 限制在 $k \times k$ 窗口内。对于 $n \times n$ 输入，计算该层、具有相同输入和输出形状的全连接层以及卷积各自的参数数量。

    *改编自 CMU 11-785 的 [Homework 2 Part 1](https://deeplearning.cs.cmu.edu/S21/index.html)。*

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

3. [code] **有效感受野。** ● 本节的理论感受野 $r$ 限定了哪些输入*能够*影响输出；有效感受野衡量它们实际影响的程度 :cite:`Luo.Li.Urtasun.ea.2016`。使用 :numref:`subsec_learning_kernel` 中的内置层，在 $32 \times 32$ 输入上构建由 $L$ 个单通道 $3 \times 3$ 卷积层组成的堆叠，不使用填充、偏置或非线性操作。从标准正态分布中独立抽取每个卷积核元素。
    1. 令 $L = 3$，对卷积核进行一次抽样，从最后一层靠近中心的一个输出单元向输入反向传播单位梯度，并绘制输入梯度的幅值。将其支撑集与 $7 \times 7$ 理论感受野比较，并说明幅值在哪里最大。
    1. 分别对 $L = 3$ 和 $L = 10$，在 500 次卷积核抽样上计算输入梯度平方的平均值。将每个平均图沿行求和，并将得到的列分布归一化，使其总和为 1。该分布的标准差与理论边长 $2L + 1$ 的比值如何随深度变化？
    1. :numref:`fig_conv_rf_paths` 统计了 $L = 3$ 时一维情况下的路径。证明对于一般的 $L$，沿行求和的输入梯度平方的期望与 $L$ 个向量 $(1, 1, 1)$ 的卷积成正比。推导归一化分布的标准差如何随 $L$ 增长。

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

4. [code] **卷积的反向传播。** ● 设 $\mathbf{Y}$ 是输入 $\mathbf{X}$ 与卷积核 $\mathbf{K}$ 的单通道互相关，如 `corr2d` 所计算的那样；设 $\ell$ 为仅通过 $\mathbf{Y}$ 依赖于 $\mathbf{X}$ 和 $\mathbf{K}$ 的标量损失。
    1. 仿照 :numref:`sec_backprop` 中反向传播方程的形式，用 $\partial \ell/\partial \mathbf{Y}$ 推导 $\partial \ell/\partial \mathbf{K}$ 和 $\partial \ell/\partial \mathbf{X}$。
    1. 对于损失 $\ell = \sum_{i,j} G_{ij} Y_{ij}$，不使用自动微分，实现你得到的 $\partial \ell/\partial \mathbf{K}$ 表达式，其中 $\mathbf{G}$ 是形状与 $\mathbf{Y}$ 相同的固定随机矩阵。在小型随机 `X` 和 `K` 上，将其与步长为 $10^{-2}$ 的中心有限差分梯度比较，并报告最大相对误差。
    1. 将你的 $\partial \ell/\partial \mathbf{K}$ 表达式与 :numref:`subsec_learning_kernel` 的实验中自动微分计算出的卷积核梯度进行核对。

    *改编自 Michael Nielsen 的 [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html) 第 6 章，以及 Stanford CS231n 的 [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/) 中的 ConvolutionalNetworks notebook。*

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
- locally connected layer → 局部连接层；support → 支撑集；column profile → 列分布。

=== NOTES ===
- ec8621f8、54bcee26、6ab5b67c：为符合中文语序调整了公式出现的顺序。
