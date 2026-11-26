
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
    1. 给定方向向量 $\mathbf{v} = (v_1, v_2)$，推导一个边缘检测卷积核，使其检测与 $\mathbf{v}$ 正交的边缘，即方向为 $(v_2, -v_1)$ 的边缘。
    1. 推导二阶导数的有限差分算子。图像中的哪些结构对它的响应最强？
    1. 要得到 $d$ 阶导数，卷积核的最小尺寸是多少？
    1. 你会如何设计一个模糊卷积核？为什么可能需要使用这样的卷积核？
    1. 载入一张照片，把它转换为单通道，并下采样到约 $100 \times 100$ 像素（`corr2d` 是 Python 循环）。应用边缘检测器、模糊卷积核和“两倍恒等减去模糊”的锐化卷积核，并把每个结果与原图并排显示。照片的边缘图显示出了哪些黑色带的边缘图所不能显示的东西？

    *改编自斯坦福 CS231n，
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/)，
    ConvolutionalNetworks 笔记本。*

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
2. [code] **作为带状矩阵的卷积。** :numref:`subsec_conv_matmul` 中的块矩阵展开的是输入，相反的构造则展开卷积核。设 $\mathbf{X}$ 是 $4 \times 4$ 的输入，$\mathbf{K}$ 是 $3 \times 3$ 的卷积核，$\textrm{vec}(\cdot)$ 表示逐行展平。
    1. 构造 $4 \times 16$ 的矩阵 $\mathbf{W}$，使得 $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$，其中 $\mathbf{Y}$ 是 $\mathbf{X}$ 与 $\mathbf{K}$ 的互相关。每一行有多少个非零元素，$\mathbf{W}$ 中出现了多少个不同的值？
    1. 用代码根据 `K` 构建 $\mathbf{W}$，并确认乘积复现了 `corr2d(X, K)`。
    1. 对于 $n \times n$ 的输入和 $k \times k$ 的卷积核，分别数出：把 $\mathbf{W}$ 存为稠密数组时的元素个数、$\mathbf{W}$ 的非零元素个数，以及块矩阵的元素个数。每个计数如何随 $n$ 增长？
    1. 保留 $\mathbf{W}$ 稀疏性的模式，但让每一行拥有各自的 $k^2$ 个值。结果是一个局部连接层：即 :numref:`sec_why-conv` 中的四阶张量 $\mathsf{V}$，位于 $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$ 之中，它被限制在 $k \times k$ 的窗口内，窗口由偏移量 $(a, b)$ 构成，然后平移等变性才把权重绑定在一起。对于 $n \times n$ 的输入，数出这个层的参数个数、输入和输出形状相同的全连接层的参数个数，以及卷积的参数个数。

    *改编自 CMU 11-785，
    [Homework 2 Part 1](https://deeplearning.cs.cmu.edu/S21/index.html)。*

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
3. [code] **有效感受野。** ● 本节的理论感受野 $r$ 界定了哪些输入*可能*影响输出；有效感受野则度量它们实际上影响多少 :cite:`Luo.Li.Urtasun.ea.2016`。使用 :numref:`subsec_learning_kernel` 中的内置层，构建由 $L$ 个单通道 $3 \times 3$ 卷积层组成的堆叠，不加填充、不加偏置、不加非线性，输入为 $32 \times 32$。每个卷积核元素都独立地从标准高斯分布中抽取。
    1. 取 $L = 3$ 和一次卷积核抽样，从最后一层中心附近的一个输出单元出发，向输入反向传播一个单位梯度，并绘制输入梯度的幅值。将它的支撑集与 $7 \times 7$ 的理论感受野比较，并说明幅值在哪里最大。
    1. 对 500 次卷积核抽样，分别在 $L = 3$ 和 $L = 10$ 时，对输入梯度的平方取平均。把每个平均后的图按行求和，并将得到的列剖面归一化使其和为一。剖面的标准差与理论边长 $2L + 1$ 之比如何随深度变化？
    1. :numref:`fig_conv_rf_paths` 在一维情形下数出了 $L = 3$ 时的路径。证明对一般的 $L$，对行求和后的输入梯度平方的期望，与 $L$ 个向量 $(1, 1, 1)$ 的卷积成正比。推导归一化剖面的标准差如何随 $L$ 增长。

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
4. [code] **卷积的反向传播。** ● 设 $\mathbf{Y}$ 是输入 $\mathbf{X}$ 与卷积核 $\mathbf{K}$ 的单通道互相关（即 `corr2d` 的计算结果），设 $\ell$ 是标量损失，它仅通过 $\mathbf{Y}$ 依赖于 $\mathbf{X}$ 和 $\mathbf{K}$。
    1. 推导 $\partial \ell/\partial \mathbf{K}$ 和 $\partial \ell/\partial \mathbf{X}$，用 $\partial \ell/\partial \mathbf{Y}$ 表示，仿照 :numref:`sec_backprop` 中的反向传播方程。
    1. 不使用自动微分，实现你推导的 $\partial \ell/\partial \mathbf{K}$ 的表达式，损失取 $\ell = \sum_{i,j} G_{ij} Y_{ij}$（其中 $\mathbf{G}$ 是与 $\mathbf{Y}$ 形状相同的固定随机矩阵）。在一个小的随机 `X` 和 `K` 上，将它与步长为 $10^{-2}$ 的中心有限差分梯度比较，并报告最大相对误差。
    1. 将你的 $\partial \ell/\partial \mathbf{K}$ 表达式与自动微分在 :numref:`subsec_learning_kernel` 的实验中计算出的卷积核梯度对照检查。

    *改编自 Michael Nielsen，
    [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html)，
    第 6 章，以及斯坦福 CS231n，
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/)，
    ConvolutionalNetworks 笔记本。*

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
- "Exercises" → 练习; "Summary" → 小结; "Discussions" → 讨论; the [code] tag and ● marker kept as is
- "standard normal" → 标准高斯分布 (glossary 高斯分布); "patch matrix" → 块矩阵; "locally connected layer" → 局部连接层; "support" (of a gradient map) → 支撑集; "profile" → 剖面; "sparsity pattern" → 稀疏模式
- Course names and assignment titles (Stanford CS231n, CMU 11-785, "Assignment 2", book titles) stay in English; "Adapted from" → 改编自

=== NOTES ===
- 6ab5b67c: the glossary warning on 'Neural Networks' is a false match: it is part of the book title "Neural Networks and Deep Learning", kept in English.
- ec8621f8: 'sparsity' warning resolved by 稀疏性的模式.
