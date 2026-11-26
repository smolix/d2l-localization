<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u01 lang=zh rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:33d3b96b kind=heading-->
# Convolutions for Images
<!--tr:33d3b96b-->
# 图像卷积

<!--en:852f5868 kind=paragraph-->
In :numref:`sec_why-conv`, we derived the convolutional layer on paper from two
geometric principles: translation equivariance and locality. Now we see what
that formula actually computes on an image, and whether gradient descent can
discover useful kernels automatically from data.
<!--tr:852f5868-->
在 :numref:`sec_why-conv` 中，我们从平移等变性和局部性这两条几何原则出发，在纸面上推导出了卷积层。现在我们来看看，这个公式作用在图像上究竟算出了什么，以及梯度下降能否自动从数据中发现有用的卷积核。

<!--en:55f0a41b kind=paragraph-->
Consider locating vertical edges where a bright region meets a dark one.
Subtracting each pixel's right-hand neighbor ($[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$)
yields zero inside uniform regions, $+1$ where white transitions to black, and
$-1$ where black returns to white. Sweeping the two-weight detector $(1, -1)$
across every valid image location produces an *edge map*. In this section, we
implement two-dimensional cross-correlation, test our hand-crafted edge
detector, let gradient descent *learn* $(1, -1)$ directly from data, show how
a convolution becomes a single matrix multiplication (*im2col*), and
derive how stacking layers widens a unit's receptive field.
<!--tr:55f0a41b-->
设想我们要找出明亮区域与黑暗区域相接处的垂直边缘。用每个像素减去它右侧的相邻像素（$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$），在均匀区域内部得到零，在由白变黑处得到 $+1$，在由黑变回白处得到 $-1$。让这个只有两个权重的检测器 $(1, -1)$ 扫过图像上每一个有效位置，就得到一张*边缘图*（edge map）。在本节中，我们将实现二维互相关，检验这个手工设计的边缘检测器，让梯度下降直接从数据中*学出* $(1, -1)$，说明卷积如何变成一次矩阵乘法（*im2col*），并推导堆叠多层如何扩大一个单元的感受野。

<!--en:47958ed8 kind=heading-->
## Cross-Correlation
<!--tr:47958ed8-->
## 互相关

<!--en:5e4296fe kind=heading-->
### The Operation
<!--tr:5e4296fe-->
### 运算

<!--en:ae7e906d kind=paragraph-->
Leaving channels aside for the moment, let the input $\mathbf{X}$ be an
$n_\textrm{h} \times n_\textrm{w}$ matrix and the kernel $\mathbf{K}$ be a
$k_\textrm{h} \times k_\textrm{w}$ matrix. The kernel defines a sliding
*window* of shape $k_\textrm{h} \times k_\textrm{w}$. Placing the window's
upper-left corner at $(i, j)$, we multiply the covered input patch elementwise
with $\mathbf{K}$ and sum the products to produce the output entry at $(i, j)$:
<!--tr:ae7e906d-->
暂且不考虑通道，设输入 $\mathbf{X}$ 是一个 $n_\textrm{h} \times n_\textrm{w}$ 的矩阵，卷积核 $\mathbf{K}$ 是一个 $k_\textrm{h} \times k_\textrm{w}$ 的矩阵。卷积核定义了一个形状为 $k_\textrm{h} \times k_\textrm{w}$ 的滑动*窗口*。把窗口的左上角放在 $(i, j)$ 处，将它覆盖的输入图块与 $\mathbf{K}$ 逐元素相乘，再把乘积求和，就得到 $(i, j)$ 处的输出元素：

<!--en:3688b38b kind=paragraph-->
This is the *two-dimensional cross-correlation* of $\mathbf{X}$ with
$\mathbf{K}$, written $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ with
zero-based indices matching code. Whereas :numref:`sec_why-conv` centered
offsets symmetrically from $-\Delta$ to $\Delta$, anchoring at the upper-left
corner differs only by a fixed shift of $\Delta$ while naturally handling
even-sized kernels (such as a $2 \times 2$ kernel or our $1 \times 2$ edge
detector) that have no center pixel.
<!--tr:3688b38b-->
这就是 $\mathbf{X}$ 与 $\mathbf{K}$ 的*二维互相关*（two-dimensional cross-correlation），记作 $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$，其中下标从零开始，与代码一致。:numref:`sec_why-conv` 让偏移量在 $-\Delta$ 到 $\Delta$ 之间对称地居中，而以左上角为锚点只相差一个固定的平移 $\Delta$，同时还能自然地处理没有中心像素的偶数尺寸卷积核（例如 $2 \times 2$ 的卷积核，或者我们那个 $1 \times 2$ 的边缘检测器）。

<!--en:fa1b7da9 kind=caption-->
![Two-dimensional cross-correlation. The shaded portions are the first output element and the input and kernel elements used to compute it: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
<!--tr:fa1b7da9-->
![二维互相关。阴影部分是第一个输出元素，以及计算它所用的输入元素和卷积核元素：$0\times0+1\times1+3\times2+4\times3=19$。](../img/correlation.svg)

<!--en:2f8e5deb kind=paragraph-->
In :numref:`fig_correlation`, we slide a $2 \times 2$ kernel across a
$3 \times 3$ input one pixel at a time, left to right and top to bottom. The
window fits cleanly at four positions, yielding the $2 \times 2$ output:
<!--tr:2f8e5deb-->
在 :numref:`fig_correlation` 中，我们让一个 $2 \times 2$ 的卷积核在 $3 \times 3$ 的输入上滑动，每次移动一个像素，从左到右、从上到下。窗口能完整放下的位置恰好有四个，由此得到 $2 \times 2$ 的输出：

<!--en:4741e012 kind=paragraph-->
Notice that the output shrinks slightly along each axis because the kernel must
stay wholly inside the input. Fitting a width-$k$ window into length $n$ leaves
$n - k + 1$ valid positions, so the output shape is
<!--tr:4741e012-->
注意，输出在每个轴上都略有缩小，因为卷积核必须完全落在输入内部。把宽度为 $k$ 的窗口放进长度 $n$ 中，只有 $n - k + 1$ 个有效位置，因此输出的形状为

<!--en:12d81c14 kind=paragraph-->
Thus a $3 \times 3$ input with a $2 \times 2$ kernel yields $2 \times 2$, a
$6 \times 8$ image with a $1 \times 2$ kernel yields $6 \times 7$, and a
$28 \times 28$ Fashion-MNIST image with a $5 \times 5$ kernel yields
$24 \times 24$. Whenever we want to preserve spatial dimensions, we surround the
input with a border of zeros (*padding*), which we examine alongside strides and
dilation in :numref:`sec_padding`.
<!--tr:12d81c14-->
于是，$3 \times 3$ 的输入配上 $2 \times 2$ 的卷积核得到 $2 \times 2$ 的输出，$6 \times 8$ 的图像配上 $1 \times 2$ 的卷积核得到 $6 \times 7$ 的输出，而 $28 \times 28$ 的 Fashion-MNIST 图像配上 $5 \times 5$ 的卷积核得到 $24 \times 24$ 的输出。每当需要保持空间尺寸不变时，我们就在输入四周补上一圈零（*填充*，padding）；我们将在 :numref:`sec_padding` 中把它与步幅、空洞率一并讨论。

<!--en:869d223f kind=heading-->
### Implementation
<!--tr:869d223f-->
### 实现

<!--en:6a869f7c kind=paragraph-->
We implement :eqref:`eq_corr2d` directly in `corr2d` using a pair of loops over
every output coordinate.
<!--tr:6a869f7c-->
我们直接按 :eqref:`eq_corr2d` 实现 `corr2d`，用一对循环遍历每个输出坐标。

<!--en:81e1f383 kind=paragraph-->
To verify `corr2d` against our hand calculation in :numref:`fig_correlation`,
we construct the $3 \times 3$ input tensor `X` and $2 \times 2$ kernel `K` and
evaluate `corr2d(X, K)`.
<!--tr:81e1f383-->
为了验证 `corr2d` 与 :numref:`fig_correlation` 中的手算结果一致，我们构造 $3 \times 3$ 的输入张量 `X` 和 $2 \times 2$ 的卷积核 `K`，然后计算 `corr2d(X, K)`。

<!--en:f567e1b2 kind=paragraph-->
The output reproduces :numref:`fig_correlation` entry for entry (and as shown in
:numref:`sec_why-conv`, strict mathematical convolution with a flipped kernel
$\mathbf{K}'$ learns the exact same family of layers, so we follow standard
deep learning terminology and call cross-correlation "convolution").
<!--tr:f567e1b2-->
输出与 :numref:`fig_correlation` 逐个元素吻合（而且正如 :numref:`sec_why-conv` 所示，使用翻转后的卷积核 $\mathbf{K}'$ 做严格的数学卷积，学到的是完全相同的一族层，因此我们沿用深度学习的通行术语，把互相关称为“卷积”）。

<!--en:bfe11c62 kind=heading-->
## Convolutional Layers
<!--tr:bfe11c62-->
## 卷积层

<!--en:2cdc020d kind=heading-->
### The Layer
<!--tr:2cdc020d-->
### 层的定义

<!--en:c6371ece kind=paragraph-->
A two-dimensional convolutional layer cross-correlates its input with a learned
kernel and adds a scalar bias. Just like our MLP weights in
:numref:`chap_perceptrons`, we initialize the kernel from
$\mathcal{N}(0, 0.01^2)$ and the bias to zero. Below, `Conv2D` registers both
parameters and invokes `corr2d` in its forward pass; a layer with a kernel of
height $h$ and width $w$ is called an $h \times w$ convolutional layer.
<!--tr:c6371ece-->
二维卷积层将输入与学到的卷积核做互相关，再加上一个标量偏置。与 :numref:`chap_perceptrons` 中多层感知机的权重一样，我们从 $\mathcal{N}(0, 0.01^2)$ 中采样来初始化卷积核，并把偏置初始化为零。下面的 `Conv2D` 注册了这两个参数，并在前向传播中调用 `corr2d`；卷积核高为 $h$、宽为 $w$ 的层称为 $h \times w$ 卷积层。

<!--en:29cd9061 kind=paragraph-->
Unlike a dense layer, a convolutional layer is not locked to a fixed input
resolution because its parameters depend only on the kernel size. The exact same
$2 \times 2$ layer (5 parameters total) processes both our $3 \times 3$ tensor
`X` and a $6 \times 8$ image, returning outputs of shapes $2 \times 2$ and
$5 \times 7$:
<!--tr:29cd9061-->
与全连接层不同，卷积层并不绑定于固定的输入分辨率，因为它的参数只取决于卷积核大小。同一个 $2 \times 2$ 的层（共 5 个参数）既能处理我们的 $3 \times 3$ 张量 `X`，也能处理 $6 \times 8$ 的图像，返回的输出形状分别为 $2 \times 2$ 和 $5 \times 7$：

<!--en:c9e5aa07 kind=heading-->
### Edge Detection
<!--tr:c9e5aa07-->
### 边缘检测

<!--en:3d9a357b kind=paragraph-->
We now test the vertical edge detector from the introduction on a
$6 \times 8$ synthetic image whose middle four columns are black (`0`) and outer
columns are white (`1`).
<!--tr:3d9a357b-->
现在我们在一幅 $6 \times 8$ 的合成图像上检验引言中的垂直边缘检测器：图像中间四列为黑色（`0`），外侧各列为白色（`1`）。

<!--en:68513ff0 kind=paragraph-->
We define the $1 \times 2$ kernel
<!--tr:68513ff0-->
我们定义 $1 \times 2$ 的卷积核

<!--en:eb45803b kind=paragraph-->
At every position $(i, j)$, this computes the horizontal *finite difference*
$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, a discrete approximation (up to sign)
of the horizontal derivative of brightness, which vanishes across flat regions
and spikes wherever intensity jumps.
<!--tr:eb45803b-->
在每个位置 $(i, j)$，它计算水平方向的*有限差分*（finite difference）$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$，即亮度的水平导数的离散近似（至多相差一个符号）：它在平坦区域为零，在亮度发生跳变的地方出现尖峰。

<!--en:e810b3a9 kind=paragraph-->
Cross-correlating the $6 \times 8$ stripe image `X` with the $1 \times 2$
horizontal-difference kernel `K` via `corr2d(X, K)` produces the $6 \times 7$
edge map `Y`:
<!--tr:e810b3a9-->
将 $6 \times 8$ 的条纹图像 `X` 与 $1 \times 2$ 的水平差分卷积核 `K` 通过 `corr2d(X, K)` 做互相关，得到 $6 \times 7$ 的边缘图 `Y`：

<!--en:568358e4 kind=paragraph-->
Sure enough, `Y` fires `+1` along the white-to-black boundary, `-1` along the
black-to-white boundary, and `0` everywhere else. Applying the exact same kernel
to the transposed image (turning the vertical stripe into a horizontal one)
yields all zeros:
<!--tr:568358e4-->
果然，`Y` 在由白到黑的边界上输出 `+1`，在由黑到白的边界上输出 `-1`，其余位置均为 `0`。把同一个卷积核作用于转置后的图像（垂直条纹变成了水平条纹），得到的则全是零：

<!--en:71c591a4 kind=paragraph-->
Clearly `[1, -1]` responds strictly to vertical edges. Detecting horizontal
edges, diagonals, corners, or textures requires different kernels altogether.
<!--tr:71c591a4-->
显然，`[1, -1]` 只对垂直边缘有响应。要检测水平边缘、对角线、角点或纹理，就需要完全不同的卷积核。

<!--en:f0af4887 kind=heading-->
### Learning a Kernel
<!--tr:f0af4887-->
### 学习卷积核

<!--en:61cacfaf kind=paragraph-->
Hand-crafting `[1, -1]` is easy when we know the exact feature we want, yet no
engineer can specify thousands of intermediate filters across a deep network by
hand. Can gradient descent recover the kernel `[1, -1]` from the input–output
pair `(X, Y)` alone? We test this using each framework's built-in 2D
convolutional layer (with bias omitted and weights initialized from
$\mathcal{N}(0, 0.01^2)$), minimizing squared error summed across all 42 output
entries over 10 gradient steps at learning rate $0.03$. Because built-in layers
expect 4D tensors including batch and channel axes (:numref:`sec_channels`), we
reshape `X` and `Y` accordingly.
<!--tr:61cacfaf-->
当我们确切知道想要什么特征时，手工设计 `[1, -1]` 很容易，但没有哪位工程师能够手工指定深层网络中成千上万个中间滤波器。梯度下降能否恢复出卷积核 `[1, -1]`，而所依据的只有输入–输出对 `(X, Y)`？我们用各框架内置的二维卷积层来检验（不使用偏置，权重从 $\mathcal{N}(0, 0.01^2)$ 初始化），最小化全部 42 个输出元素上的平方误差之和，共做 10 步梯度更新，学习率为 $0.03$。由于内置层要求输入是包含批量轴和通道轴的四维张量（:numref:`sec_channels`），我们相应地调整 `X` 和 `Y` 的形状。

<!--en:c29c5d61 kind=paragraph-->
Initially the loss sits near 12 (the near-zero random kernel misses all 12
$\pm 1$ edge pixels by roughly 1), and within just nine steps it drops below
0.01. Inspecting the learned kernel confirms what it found:
<!--tr:c29c5d61-->
起初损失在 12 左右（接近零的随机卷积核在全部 12 个 $\pm 1$ 边缘像素上都差了大约 1），而仅仅九步之后它就降到 0.01 以下。查看学到的卷积核，就能确认它找到了什么：

<!--en:67d25b0b kind=paragraph-->
Both entries land within roughly 0.02 of `[1, -1]`. Nothing in this loop knew
anything about edges: given a loss signal, gradient descent discovered the
filter on its own. Here we supervised the single layer directly with `Y`; in a
deep CNN, backpropagation learns every layer's kernels end to end from the final
task loss alone.
<!--tr:67d25b0b-->
两个元素与 `[1, -1]` 的差距都在 0.02 左右以内。这个循环里没有任何关于边缘的知识：只要给出损失信号，梯度下降就自己发现了这个滤波器。这里我们用 `Y` 直接监督这一个层；而在深层卷积神经网络中，反向传播仅凭最终任务的损失，就能端到端地学出每一层的卷积核。

<!--en:91b0010d kind=slide_text-->
Convolutions for Images<br>**2D cross-correlation · edge detection · learned kernels · im2col · receptive fields**
<!--tr:91b0010d-->
图像卷积<br>**二维互相关 · 边缘检测 · 学习卷积核 · im2col · 感受野**

<!--en:1d94295a kind=slide_title-->
::: {.slide title="Detecting Local Edges Across an Image"}
<!--tr:1d94295a-->
::: {.slide title="检测图像中的局部边缘"}

<!--en:687ed302 kind=kicker-->
[Motivation & Core Mental Model]{.kicker}
<!--tr:687ed302-->
[动机与核心心智模型]{.kicker}

<!--en:05fceed0 kind=slide_text-->
- **Synthetic $6 \times 8$ test image:** White outer columns (`1`) surrounding a 4-column black vertical band (`0`).
<!--tr:05fceed0-->
- **$6 \times 8$ 合成测试图像**：外侧各列为白色（`1`），中间夹着一条 4 列宽的黑色竖带（`0`）。

<!--en:71e700b3 kind=slide_text-->
- **Horizontal finite difference:** Sweeping $\mathbf{K} = [1, -1]$ computes $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, firing `+1` at white-to-black and `-1` at black-to-white edges.
<!--tr:71e700b3-->
- **水平有限差分**：用 $\mathbf{K} = [1, -1]$ 扫过图像，计算 $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$，在由白到黑的边缘输出 `+1`，在由黑到白的边缘输出 `-1`。

<!--en:957dc033 kind=slide_text-->
- **Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent.
<!--tr:957dc033-->
- **学习与手工设计**：深层卷积神经网络通过梯度下降自动发现成千上万个这样的卷积核。

<!--en:cb1045e1 kind=slide_title-->
::: {.slide title="The 2D Cross-Correlation Operation"}
<!--tr:cb1045e1-->
::: {.slide title="二维互相关运算"}

<!--en:17bcece5 kind=kicker-->
[Sliding-Window Inner Products]{.kicker}
<!--tr:17bcece5-->
[滑动窗口内积]{.kicker}

<!--en:8d45aae8 kind=slide_text-->
Given an $n_\textrm{h} \times n_\textrm{w}$ input matrix $\mathbf{X}$ and a $k_\textrm{h} \times k_\textrm{w}$ kernel $\mathbf{K}$, the 2D cross-correlation $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ places the window's top-left corner at $(i, j)$ and sums elementwise products:
<!--tr:8d45aae8-->
给定 $n_\textrm{h} \times n_\textrm{w}$ 的输入矩阵 $\mathbf{X}$ 和 $k_\textrm{h} \times k_\textrm{w}$ 的卷积核 $\mathbf{K}$，二维互相关 $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ 把窗口的左上角放在 $(i, j)$ 处，并对逐元素乘积求和：

<!--en:e484380c kind=slide_text-->
- **Top-left anchor:** Zero-based indices $a, b \ge 0$ match array indexing in code and handle even-sized kernels (like $2 \times 2$ or $1 \times 2$) cleanly.
<!--tr:e484380c-->
- **左上角锚点**：从零开始的下标 $a, b \ge 0$ 与代码中的数组索引一致，也能自然地处理偶数尺寸的卷积核（如 $2 \times 2$ 或 $1 \times 2$）。

<!--en:137c0980 kind=slide_text-->
- **Shaded top-left output $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.
<!--tr:137c0980-->
- **阴影标出的左上角输出 $(0,0)$**：
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$。

<!--en:575c4aea kind=slide_title-->
::: {.slide title="Valid Window Positions and Output Spatial Shape"}
<!--tr:575c4aea-->
::: {.slide title="有效窗口位置与输出的空间形状"}

<!--en:7eda4531 kind=kicker-->
[Why Unpadded Convolutions Shrink the Grid]{.kicker}
<!--tr:7eda4531-->
[为什么无填充的卷积会缩小网格]{.kicker}

<!--en:891243ae kind=slide_text-->
- **Valid window placements:** A $k_\textrm{h} \times k_\textrm{w}$ window fits inside an $n_\textrm{h} \times n_\textrm{w}$ grid at $n - k + 1$ positions per axis.
<!--tr:891243ae-->
- **有效的窗口位置**：$k_\textrm{h} \times k_\textrm{w}$ 的窗口放在 $n_\textrm{h} \times n_\textrm{w}$ 的网格内，每个轴上有 $n - k + 1$ 个位置。

<!--en:55cc8c32 kind=slide_text-->
- **Shape examples:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.
<!--tr:55cc8c32-->
- **形状示例**：$3 \times 3 \star 2 \times 2 \to 2 \times 2$（$19, 25, 37, 43$）；$28 \times 28 \star 5 \times 5 \to 24 \times 24$。

<!--en:1e3463da kind=slide_title-->
::: {.slide title="Implementing 2D Cross-Correlation in Code"}
<!--tr:1e3463da-->
::: {.slide title="用代码实现二维互相关"}

<!--en:07566992 kind=kicker-->
[Direct Nested-Loop Implementation]{.kicker}
<!--tr:07566992-->
[直接的嵌套循环实现]{.kicker}

<!--en:202aa5d3 kind=slide_text-->
The function `corr2d` allocates the $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ output tensor and computes each entry as the sum of `X[i:i+h, j:j+w] * K`:
<!--tr:202aa5d3-->
函数 `corr2d` 分配形状为 $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ 的输出张量，并把每个元素计算为 `X[i:i+h, j:j+w] * K` 的和：

<!--en:bef130ba kind=slide_text-->
Each iteration slices a $k_\textrm{h} \times k_\textrm{w}$ sub-window of `X`, multiplies elementwise with `K`, and reduces via `sum`.
<!--tr:bef130ba-->
每次迭代切片出一个 $k_\textrm{h} \times k_\textrm{w}$ 的 `X` 子窗口，与 `K` 逐元素相乘，再用 `sum` 归约。

<!--en:4ee9f6b1 kind=slide_title-->
::: {.slide title="Verifying corr2d and the Flipped-Kernel Equivalence"}
<!--tr:4ee9f6b1-->
::: {.slide title="验证 corr2d 以及翻转卷积核的等价性"}

<!--en:2b0be423 kind=kicker-->
[Cross-Correlation vs. Mathematical Convolution]{.kicker}
<!--tr:2b0be423-->
[互相关与数学卷积]{.kicker}

<!--en:74b9b745 kind=slide_text-->
- **180° flipped kernel $\mathbf{K}'$:** Cross-correlating with $\mathbf{K}$ equals convolving with $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.
<!--tr:74b9b745-->
- **翻转 180° 的卷积核 $\mathbf{K}'$**：与 $\mathbf{K}$ 做互相关，等于与 $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$ 做卷积。

<!--en:5c8af7cd kind=slide_text-->
- **Identical learned class:** Frameworks compute cross-correlation and call it convolution.
<!--tr:5c8af7cd-->
- **学到的函数类相同**：各框架计算的是互相关，却称之为卷积。

<!--en:a69e29b5 kind=slide_title-->
::: {.slide title="Building a 2D Convolutional Layer Module"}
<!--tr:a69e29b5-->
::: {.slide title="构建二维卷积层模块"}

<!--en:55763603 kind=kicker-->
[Learnable Kernel Weights Plus Scalar Bias]{.kicker}
<!--tr:55763603-->
[可学习的卷积核权重加标量偏置]{.kicker}

<!--en:5b8bad24 kind=slide_text-->
A convolutional layer wraps `corr2d` with a learnable $k_\textrm{h} \times k_\textrm{w}$ weight parameter (initialized from $\mathcal{N}(0, 0.01^2)$) and a scalar bias (initialized to $0$):
<!--tr:5b8bad24-->
卷积层对 `corr2d` 做了一层封装，带有一个可学习的 $k_\textrm{h} \times k_\textrm{w}$ 权重参数（从 $\mathcal{N}(0, 0.01^2)$ 初始化）和一个标量偏置（初始化为 $0$）：

<!--en:c16d97d6 kind=slide_title-->
::: {.slide title="Resolution Independence of Convolutional Layers"}
<!--tr:c16d97d6-->
::: {.slide title="卷积层与分辨率无关"}

<!--en:a7d1e48f kind=kicker-->
[Fixed Parameters Across Arbitrary Input Sizes]{.kicker}
<!--tr:a7d1e48f-->
[参数固定，输入尺寸任意]{.kicker}

<!--en:53a2bce7 kind=slide_text-->
- **Decoupled from $H \times W$:** The exact same 5 parameters ($2 \times 2$ kernel + 1 bias) process $3 \times 3$, $6 \times 8$, or megapixel inputs without reshaping weights.
<!--tr:53a2bce7-->
- **与 $H \times W$ 解耦**：同样的 5 个参数（$2 \times 2$ 卷积核 + 1 个偏置）无需调整权重形状，就能处理 $3 \times 3$、$6 \times 8$ 乃至百万像素的输入。

<!--en:017fec24 kind=slide_title-->
::: {.slide title="Constructing a Synthetic Vertical-Edge Image"}
<!--tr:017fec24-->
::: {.slide title="构造含垂直边缘的合成图像"}

<!--en:896e641c kind=kicker-->
[A 6×8 Binary Test Pattern]{.kicker}
<!--tr:896e641c-->
[一个 6×8 的二值测试图案]{.kicker}

<!--en:0a4b456b kind=slide_text-->
We construct the $6 \times 8$ test image `X` with ones on the left and right borders and zeros across the middle four columns (`X[:, 2:6] = 0`):
<!--tr:0a4b456b-->
我们构造 $6 \times 8$ 的测试图像 `X`，左右两侧为 1，中间四列为 0（`X[:, 2:6] = 0`）：

<!--en:dfc7e3e3 kind=slide_text-->
- **Piecewise-constant columns:** Intensity is constant (`1` or `0`) within columns $0\text{--}1$, $2\text{--}5$, and $6\text{--}7$, jumping only between columns $1 \to 2$ and $5 \to 6$.
<!--tr:dfc7e3e3-->
- **按列分段恒定**：亮度保持恒定（`1` 或 `0`）的列段为 $0\text{--}1$、$2\text{--}5$ 和 $6\text{--}7$，只在第 $1 \to 2$ 列和第 $5 \to 6$ 列之间发生跳变。

<!--en:3ce31cbe kind=slide_title-->
::: {.slide title="Applying the Horizontal Finite-Difference Kernel"}
<!--tr:3ce31cbe-->
::: {.slide title="应用水平有限差分卷积核"}

<!--en:b37a1029 kind=kicker-->
[Detecting Vertical Edges with K = [1, -1]]{.kicker}
<!--tr:b37a1029-->
[用 K = [1, -1] 检测垂直边缘]{.kicker}

<!--en:9be85019 kind=slide_text-->
Cross-correlating `X` with the $1 \times 2$ kernel $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ computes $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ at every horizontal pair:
<!--tr:9be85019-->
将 `X` 与 $1 \times 2$ 的卷积核 $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ 做互相关，就是在每一对水平相邻的像素上计算 $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$：

<!--en:f649cfae kind=slide_text-->
- **Signed edge map:** Column 1 fires `+1` (white-to-black transition), column 5 fires `-1` (black-to-white transition), and all constant interior columns evaluate to `0`.
<!--tr:f649cfae-->
- **带符号的边缘图**：第 1 列输出 `+1`（由白到黑），第 5 列输出 `-1`（由黑到白），内部所有恒定的列都得到 `0`。

<!--en:e6d155be kind=slide_title-->
::: {.slide title="Orientation Selectivity of Edge Kernels"}
<!--tr:e6d155be-->
::: {.slide title="边缘卷积核的方向选择性"}

<!--en:b525cde4 kind=kicker-->
[Why One Kernel Cannot Detect All Orientations]{.kicker}
<!--tr:b525cde4-->
[为什么一个卷积核无法检测所有方向]{.kicker}

<!--en:38159de6 kind=slide_text-->
- **Blind to horizontal edges:** Transposing `X` (`X.T`) makes horizontal neighbors identical, so $[1, -1]$ outputs all zeros.
<!--tr:38159de6-->
- **对水平边缘视而不见**：将 `X` 转置（`X.T`）后，水平相邻的像素完全相同，因此 $[1, -1]$ 的输出全为零。

<!--en:f0f30f76 kind=slide_text-->
- **Filter banks (§{sec_channels}):** Vision layers learn parallel channels for horizontal, vertical, and diagonal orientations.
<!--tr:f0f30f76-->
- **滤波器组（§{sec_channels}）**：视觉网络的层并行地学习多个通道，分别对应水平、垂直和对角方向。

<!--en:b755e31b kind=slide_title-->
::: {.slide title="Learning an Edge Detector via Gradient Descent"}
<!--tr:b755e31b-->
::: {.slide title="通过梯度下降学习边缘检测器"}

<!--en:86bbeea7 kind=kicker-->
[Supervising a Built-In Convolutional Layer on (X, Y)]{.kicker}
<!--tr:86bbeea7-->
[在 (X, Y) 上监督内置卷积层]{.kicker}

<!--en:a569780c kind=slide_text-->
- **4D framework tensors:** Reshape `X` and `Y` to 4D tensors (one example, one channel) and minimize $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ at $\eta = 0.03$.
<!--tr:a569780c-->
- **框架要求四维张量**：将 `X` 和 `Y` 重塑为四维张量（一个样本、一个通道），最小化 $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$，取 $\eta = 0.03$。

<!--en:2489ce6e kind=slide_title-->
::: {.slide title="Inspecting the Learned Kernel Weights"}
<!--tr:2489ce6e-->
::: {.slide title="查看学到的卷积核权重"}

<!--en:7e4ef44b kind=kicker-->
[Why Initial Loss Is 12 and Converges to [1, -1]]{.kicker}
<!--tr:7e4ef44b-->
[为什么初始损失是 12，并收敛到 [1, -1]]{.kicker}

<!--en:9cd1c645 kind=slide_text-->
- **Why loss starts at $12.0$:** Near-zero weights predict $\hat{Y} \approx 0$, missing the twelve $\pm 1$ edge pixels in `Y` ($12 \times 1^2 = 12$).
<!--tr:9cd1c645-->
- **为什么损失从 $12.0$ 开始**：接近零的权重给出预测 $\hat{Y} \approx 0$，错过了 `Y` 中的十二个 $\pm 1$ 边缘像素（$12 \times 1^2 = 12$）。

<!--en:456670c5 kind=slide_text-->
- **Convergence in 9 steps:** SGD recovers $[0.99, -0.99]$ automatically from data.
<!--tr:456670c5-->
- **9 步内收敛**：随机梯度下降自动从数据中恢复出 $[0.99, -0.99]$。

=== DECISIONS ===
- 标点与空格：正文用全角标点；汉字与行内数学、行内代码、引用（:numref: 等）、拉丁字母或数字之间留一个半角空格，紧邻全角标点时不留。每个块写成一行，不在句中折行。
- 语气：用“我们”，书面但平实；标题用名词或动宾短语，不加句号。
- 英文括注：只在正文中首次定义的术语（原文斜体处）后加英文，如 *边缘图*（edge map）；幻灯片和练习不重复括注。
- 粗体小标题后的冒号放在粗体之外（`**小标题**：正文`），避免 `**…：**` 紧接汉字时强调无法闭合；练习题名保留 `**题名。** ` 加空格的形式。
- 2D / 4D → 二维 / 四维；引号用“”。
- "edge map" → 边缘图；"edge detector" → 边缘检测器
- "window" → 窗口；"patch" → 图块；"patch matrix" → 图块矩阵
- "entry / element"（矩阵、张量的）→ 元素
- "unit"（网络中的）→ 单元
- "dense layer" → 全连接层（与 fully connected layer 同一术语）
- "hand-crafted / hand-designed" → 手工设计
- "elementwise" → 逐元素；"flatten" → 展平；"unroll / unfold" → 展开；"stack" → 堆叠
- "anchor"（窗口定位）→ 锚点；"offset" → 偏移量
- "framework" → 框架；"example"（数据）→ 样本；"built-in layer" → 内置层
- "end to end" → 端到端
- 幻灯片中的缩写 SGD 按术语表展开为“随机梯度下降”
- im2col、GEMM、Fashion-MNIST、Python 等名称保留英文

=== NOTES ===
- 2cdc020d: "The Layer" 译为“层的定义”，单字标题“层”在中文里不成题。
- 55f0a41b, 29cd9061: 原文的 "returns to white" / "returning outputs" 是普通动词，不是强化学习的 return（回报）；若检查报告缺少“回报”，属误匹配。
- 687ed302: "Mental Model" 译为“心智模型”，不是术语表中的 model（模型）义项，但恰好包含该词。
