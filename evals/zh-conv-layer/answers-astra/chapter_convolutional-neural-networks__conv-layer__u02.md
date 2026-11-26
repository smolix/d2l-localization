<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u02 lang=zh rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:0b85caf9 kind=heading-->
## Convolution as Matrix Multiplication
<!--tr:0b85caf9-->
## 用矩阵乘法计算卷积

<!--en:5bdffcd7 kind=paragraph-->
While the nested loops in `corr2d` make the math transparent, visiting one
output pixel at a time in Python is far too slow for real workloads. A classic
remedy, and still one of the algorithms that libraries choose from, is to recast
cross-correlation as a single large matrix multiplication, the exact operation
GPUs execute at peak speed. Notice that every
output entry in :numref:`fig_correlation` is simply the dot product of the
flattened $k_\textrm{h} k_\textrm{w}$-element kernel with the corresponding
flattened input window. Extracting all
$(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ windows and
stacking them as rows yields a patch matrix; multiplying it by the flattened
kernel computes the entire cross-correlation in one shot. This standard layout
transformation is called *im2col* ("image to columns", or rows here).
:numref:`fig_conv_im2col` carries it out on the example of
:numref:`fig_correlation`.
<!--tr:5bdffcd7-->
`corr2d` 中的嵌套循环直观地呈现了计算公式，但在 Python 中一次只计算一个输出像素，实际使用时太慢。一个经典做法是把互相关改写成一次大规模矩阵乘法，恰好能让 GPU 以峰值速度运行。如今的计算库仍将它作为备选算法之一。仔细看 :numref:`fig_correlation`，每个输出元素其实都是两个向量的点积：一个由卷积核的 $k_\textrm{h} k_\textrm{w}$ 个元素展平而成，另一个由对应的输入窗口展平而成。提取全部 $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ 个窗口，将每个窗口展平为一行，再叠成一个图像块矩阵。将这个矩阵乘以展平的卷积核，一次就能算出互相关的全部结果。这种标准的数据布局变换叫作 *im2col*（意为“图像转列”，这里用的是行）。:numref:`fig_conv_im2col` 用 :numref:`fig_correlation` 的例子演示了这一过程。

<!--en:107554a5 kind=caption-->
![The cross-correlation of :numref:`fig_correlation` as one matrix product. Each of the four $2 \times 2$ windows of the input becomes a row of the patch matrix, and multiplying by the flattened kernel gives the flattened output. The shaded cells follow the first window to its row and to the output 19. The center pixel 4 lies in all four windows, so it appears in every row.](../img/conv-im2col.svg)
<!--tr:107554a5-->
![用一次矩阵乘法计算 :numref:`fig_correlation` 中的互相关。输入的四个 $2 \times 2$ 窗口各自构成图像块矩阵的一行，该矩阵乘以展平的卷积核即得到展平的输出。阴影依次标出第一个窗口、它对应的行和输出值 19。中心像素 4 位于全部四个窗口中，因此每一行都有它。](../img/conv-im2col.svg)

<!--en:76713479 kind=paragraph-->
The trade-off is extra workspace memory, since each interior pixel is duplicated
into up to $k_\textrm{h} k_\textrm{w}$ patches. Below we verify on
:numref:`fig_correlation` that this single matrix multiply matches `corr2d`
exactly:
<!--tr:76713479-->
这种做法需要额外的工作内存，因为每个内部像素最多会被复制到 $k_\textrm{h} k_\textrm{w}$ 个图像块中。下面用 :numref:`fig_correlation` 的例子验证，一次矩阵乘法就能得到与 `corr2d` 完全相同的结果：

<!--en:3bcd7d8a kind=paragraph-->
The discrepancy is identically zero: one matrix product reproduces `corr2d`
perfectly.
<!--tr:3bcd7d8a-->
差值全为零：一次矩阵乘法就完整复现了 `corr2d` 的结果。

<!--en:c30f2dea kind=paragraph-->
Equivalently, we can unroll the *kernel* instead of the input into a sparse
banded matrix $\mathbf{W}$ multiplying the flattened image—the exact 2D analog
of panel (c) in :numref:`fig_why_conv_weight_sharing`, where every row repeats
the same handful of kernel weights. Seeing a convolution as a structured dense
layer brings us full circle to the derivation of :numref:`sec_why-conv` (and the
exercises build $\mathbf{W}$ explicitly).
<!--tr:c30f2dea-->
还有一种等价的做法：展开*卷积核*，得到稀疏带状矩阵 $\mathbf{W}$，再乘以展平的图像。这正是 :numref:`fig_why_conv_weight_sharing` 中子图 (c) 的二维版本，每一行都重复使用同一组卷积核权重。把卷积看作具有特定结构的全连接层，就回到了 :numref:`sec_why-conv` 的推导起点。我们将在练习中显式构造 $\mathbf{W}$。

<!--en:ad4fc956 kind=heading-->
## Feature Maps and Receptive Fields
<!--tr:ad4fc956-->
## 特征图与感受野

<!--en:e7983df8 kind=paragraph-->
A single convolutional layer sees only a patch the size of its kernel, whereas
recognizing whole objects demands integrating context across much wider areas.
Recall from :numref:`subsec_why-conv-channels` that a convolutional layer's
output is called a *feature map* (like our vertical edge map `Y`). For any
element $x$ in a feature map, its *receptive field* is the region of the input
image (or of an earlier layer) whose values can influence $x$ during forward
propagation—borrowing the neurophysiology term from :numref:`field_visual`.
<!--tr:e7983df8-->
单个卷积层的每个输出单元只能看到与卷积核一样大的输入块，而识别完整物体需要综合更大范围内的上下文信息。回顾 :numref:`subsec_why-conv-channels`，卷积层的输出称为*特征图*，前面的垂直边缘图 `Y` 就是一例。对于特征图中的任意元素 $x$，它的*感受野*是输入图像（或前面某一层）中的一个区域，该区域内的值在前向传播时可以影响 $x$。“感受野”一词借自 :numref:`field_visual` 中介绍的神经生理学概念。

<!--en:b36f221e kind=heading-->
### Stacked Layers
<!--tr:b36f221e-->
### 叠加卷积层

<!--en:24a25ec4 kind=paragraph-->
In :numref:`fig_correlation`, the shaded output `19` has a $2 \times 2$ receptive
field covering the top-left four pixels of $\mathbf{X}$. Now stack a *second*
$2 \times 2$ convolutional layer on top of the $2 \times 2$ output $\mathbf{Y}$
to produce a single scalar $z$: $z$ depends on all four elements of $\mathbf{Y}$,
which together depend on all nine elements of $\mathbf{X}$. Stacking layers
expands the receptive field smoothly with depth.
<!--tr:24a25ec4-->
在 :numref:`fig_correlation` 中，阴影标出的输出 `19` 有一个 $2 \times 2$ 的感受野，覆盖 $\mathbf{X}$ 左上角的四个像素。现在叠加*第二个* $2 \times 2$ 卷积层，让它接收 $2 \times 2$ 输出 $\mathbf{Y}$，得到一个标量 $z$。其中 $z$ 依赖于 $\mathbf{Y}$ 的全部四个元素，而这四个元素合起来又依赖于 $\mathbf{X}$ 的全部九个元素。这样逐层叠加，感受野就随深度逐渐扩大。

<!--en:f95c9394 kind=paragraph-->
How quickly does it grow? Each $k \times k$ convolutional layer reaches $k - 1$
pixels farther than a single element of its input, so stacking $L$ layers with
kernel sizes $k_1, \ldots, k_L$ yields a receptive field of side length
<!--tr:f95c9394-->
感受野扩大得有多快？每经过一个 $k \times k$ 卷积层，感受野的边长都比该层单个输入元素的感受野边长多 $k - 1$ 个像素。因此，叠加 $L$ 层，若各层的卷积核大小分别为 $k_1, \ldots, k_L$，则感受野的边长为

<!--en:b302e3ef kind=paragraph-->
For a stack of $L$ consecutive $3 \times 3$ layers, each layer adds 2 pixels,
covering $(2L + 1) \times (2L + 1)$ inputs: $5 \times 5$ after two layers and
$7 \times 7$ after three (:numref:`fig_conv_receptive_field`). This linear
growth is actually a blessing rather than a reason to reach for huge kernels:
three stacked $3 \times 3$ layers achieve the same $7 \times 7$ receptive field
as a single $7 \times 7$ layer using far fewer parameters ($3 \times 9 = 27$
weights vs. 49 per channel pair) while interleaving three nonlinearities instead
of one.
<!--tr:b302e3ef-->
连续叠加 $L$ 个 $3 \times 3$ 卷积层时，每一层都使感受野的边长增加 2 个像素，最终覆盖 $(2L + 1) \times (2L + 1)$ 个输入位置：两层后为 $5 \times 5$，三层后为 $7 \times 7$（:numref:`fig_conv_receptive_field`）。这种线性增长恰是小卷积核的优势，无需急于换用巨大的卷积核。叠加三个 $3 \times 3$ 卷积层，可以获得 $7 \times 7$ 的感受野，与单个 $7 \times 7$ 卷积层相同，但参数少得多（每对通道只需 $3 \times 9 = 27$ 个权重，单层则需 49 个），而且可以穿插三次非线性变换，而非一次。

<!--en:39917513 kind=caption-->
![Receptive field of one unit after three one-dimensional layers with kernel size 3. Shaded cells are the units that the top unit depends on, and the lines connect each of them to its inputs. Each layer widens the receptive field by 2, to 7 inputs.](../img/conv-receptive-field.svg)
<!--tr:39917513-->
![卷积核大小均为 3 的三个一维卷积层叠加后，一个单元的感受野。阴影方格表示顶层单元所依赖的各个单元，连线将每个单元与其输入相连。每一层都使感受野加宽 2 个位置，最终覆盖 7 个输入位置。](../img/conv-receptive-field.svg)

<!--en:48e4c9cc kind=heading-->
### Effective Receptive Fields
<!--tr:48e4c9cc-->
### 有效感受野

<!--en:77d044db kind=paragraph-->
Keep in mind that $r$ measures the *theoretical* receptive field (every input
pixel with *any* path to an output unit) without saying how strongly each pixel
actually contributes. Measuring input gradients directly,
:citet:`Luo.Li.Urtasun.ea.2016` showed that the *effective* receptive field
decays sharply away from the center in a roughly Gaussian bell curve, in both
untrained and trained networks. Deep units can theoretically see far across the
image, but their sensitivity concentrates heavily near the center.
<!--tr:77d044db-->
注意，$r$ 衡量的是*理论*感受野：只要有*任何一条*路径通向某个输出单元，输入像素就属于该单元的理论感受野。它没有说明各像素的实际贡献有多大。:citet:`Luo.Li.Urtasun.ea.2016` 直接测量了输入梯度，发现在未经训练和训练过的网络中，*有效*感受野内的影响都随到中心的距离增大而迅速衰减，大致呈高斯钟形曲线。深层单元在理论上能看到图像中很远的地方，但最敏感的区域高度集中在感受野中心附近。

<!--en:eed4e2a0 kind=paragraph-->
Counting paths explains the bell shape. An input at the edge of the receptive
field reaches the output unit along a single chain of connections, whereas the
center input can take many routes through the layers in between.
:numref:`fig_conv_rf_paths` counts them for the tower of
:numref:`fig_conv_receptive_field`: the center input has seven paths to the top
unit and each edge input has one. If all weights have a similar magnitude, the
influence of an input grows with its number of paths.
<!--tr:eed4e2a0-->
数一数路径就能解释这种钟形。感受野边缘的输入只有一条连接路径通往输出单元，而中心输入可以沿多条路径穿过中间各层。:numref:`fig_conv_rf_paths` 给出了 :numref:`fig_conv_receptive_field` 中各单元的路径数：中心输入到顶层单元有七条路径，两端的输入各只有一条。如果各权重的幅值相近，路径越多，输入的影响就越大。

<!--en:b92775ee kind=caption-->
![Number of paths from each unit to the top unit of :numref:`fig_conv_receptive_field` (three one-dimensional layers with kernel size 3). Each count is the sum of the counts of the units that the cell feeds. Darker cells have more paths than the other cells of their layer, and the bars repeat the counts of the input row: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)
<!--tr:b92775ee-->
![各单元到 :numref:`fig_conv_receptive_field` 顶层单元的路径数（三个一维卷积层，卷积核大小均为 3）。每个方格的路径数等于所有直接接收其输出的单元的路径数之和。同一层中，颜色越深，路径越多。下方柱形表示输入行的路径数，从左到右依次为 1、3、6、7、6、3、1。](../img/conv-rf-paths.svg)

<!--en:405695e8 kind=paragraph-->
Two practical gaps remain in our basic layer: every unpadded convolution shrinks
spatial dimensions while under-sampling border pixels, and a single kernel
detects only one pattern. We address spatial sizing and borders next in
:numref:`sec_padding`, and multi-channel feature maps in :numref:`sec_channels`.
<!--tr:405695e8-->
这个基本卷积层在实际应用中还有两点局限：每次不加填充的卷积都会缩小空间尺寸，对边界像素的采样也较少；单个卷积核只能检测一种模式。接下来，我们将在 :numref:`sec_padding` 中讨论空间尺寸和边界，在 :numref:`sec_channels` 中讨论多通道特征图。

<!--en:30ac03ff kind=heading-->
## Summary
<!--tr:30ac03ff-->
## 小结

<!--en:92688a0c kind=paragraph-->
A convolutional layer applies two-dimensional cross-correlation with a learned
kernel plus a scalar bias, sliding the kernel across every valid window of the
input. A simple $1 \times 2$ finite-difference kernel `[1, -1]` detects vertical
edges anywhere in an image, and gradient descent recovers those exact weights
from input–output data alone, just as deep CNNs learn entire hierarchies of
filters end to end. Mathematical convolution flips the kernel, making no
difference once kernels are learned. A convolution can be evaluated as a single
matrix multiplication (*im2col*), and stacking $L$ convolutional
layers widens each unit's receptive field linearly by $k_i - 1$ per layer.
<!--tr:92688a0c-->
卷积层将学习得到的卷积核滑过输入的每个有效窗口，计算二维互相关，再加上标量偏置。一个简单的 $1 \times 2$ 有限差分卷积核 `[1, -1]` 就能检测出图像中任意位置的垂直边缘。梯度下降仅凭输入和输出数据就能恢复这组权重，正如深度 CNN 可以端到端地学到各个层次的全部滤波器。数学定义下的卷积需要翻转卷积核，但卷积核由学习得到时，这不影响它能表示的运算。一次矩阵乘法（*im2col*）就能算出卷积的结果。叠加 $L$ 个卷积层时，每个单元的感受野边长逐层线性增长，第 i 层增加 $k_i - 1$ 个像素。

<!--en:3ba929d2 kind=slide_title-->
::: {.slide title="Fast Execution: Convolution via Matrix Multiplication"}
<!--tr:3ba929d2-->
::: {.slide title="用矩阵乘法加速卷积"}

<!--en:1711f812 kind=kicker-->
[The im2col (Image-to-Columns) Transformation]{.kicker}
<!--tr:1711f812-->
[im2col（图像转列）变换]{.kicker}

<!--en:a040f3d7 kind=slide_text-->
- **Single dense GEMM:** Flatten each $k_\textrm{h} \times k_\textrm{w}$ window into a row of `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ and multiply by $\textrm{vec}(\mathbf{K})$.
<!--tr:a040f3d7-->
- **一次稠密矩阵乘法（GEMM）：** 将每个 $k_\textrm{h} \times k_\textrm{w}$ 窗口展平为矩阵 `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ 的一行，再将该矩阵乘以 $\textrm{vec}(\mathbf{K})$。

<!--en:1fb18ec4 kind=slide_text-->
- **Trade-off:** Peak GPU Tensor Core speed at the cost of $k_\textrm{h} k_\textrm{w}\times$ patch workspace duplication.
<!--tr:1fb18ec4-->
- **代价：** GPU Tensor Core 能以峰值速度运行，但图像块占用的工作内存可因复制而增至 $k_\textrm{h} k_\textrm{w}\times$。

<!--en:0d757cf6 kind=slide_title-->
::: {.slide title="Dual View: Unrolling the Kernel into a Banded Matrix"}
<!--tr:0d757cf6-->
::: {.slide title="另一种视角：将卷积核展开成带状矩阵"}

<!--en:7f0defaa kind=kicker-->
[Toeplitz Weight Matrix Multiplication]{.kicker}
<!--tr:7f0defaa-->
[Toeplitz 权重矩阵乘法]{.kicker}

<!--en:5898ba57 kind=slide_text-->
Instead of unfolding the input image into duplicated patches, we can unroll the *kernel* into a sparse banded matrix $\mathbf{W}$ that multiplies the flattened input vector $\textrm{vec}(\mathbf{X})$:
<!--tr:5898ba57-->
除了将输入图像展开为含有重复像素的图像块，还可以将*卷积核*展开为稀疏带状矩阵 $\mathbf{W}$，用它乘以展平的输入向量 $\textrm{vec}(\mathbf{X})$：

<!--en:dae51741 kind=slide_text-->
- **Matrix–vector form:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, where $\mathbf{W}$ has $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ rows and $n_\textrm{h} n_\textrm{w}$ columns.
<!--tr:dae51741-->
- **矩阵与向量相乘：** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$，其中 $\mathbf{W}$ 有 $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ 行、$n_\textrm{h} n_\textrm{w}$ 列。

<!--en:f0e86ee3 kind=slide_text-->
- **Connection to §{sec_why-conv} and backprop:** Every row of $\mathbf{W}$ holds the same $k_\textrm{h} k_\textrm{w}$ kernel entries shifted across columns (panel c), making the backward pass with respect to $\mathbf{X}$ simply multiplication by $\mathbf{W}^\top$ (a transposed convolution).
<!--tr:f0e86ee3-->
- **与 §{sec_why-conv} 及反向传播的联系：** $\mathbf{W}$ 的各行都含有相同的 $k_\textrm{h} k_\textrm{w}$ 个卷积核元素，只是所在的列有所平移（见子图 c）。因此，对 $\mathbf{X}$ 做反向传播时，只需左乘 $\mathbf{W}^\top$（即转置卷积）。

<!--en:af8cf19a kind=slide_title-->
::: {.slide title="Feature Maps and Receptive Fields Across Depth"}
<!--tr:af8cf19a-->
::: {.slide title="不同深度的特征图与感受野"}

<!--en:4eb965f3 kind=kicker-->
[Linear Receptive-Field Growth in Stacked Layers]{.kicker}
<!--tr:4eb965f3-->
[逐层叠加时感受野线性增长]{.kicker}

<!--en:5599a6e9 kind=slide_text-->
A unit's **receptive field** is the region of the original input image whose pixels can influence that unit's activation during forward propagation:
<!--tr:5599a6e9-->
一个单元的**感受野**是原始输入图像中的一个区域，其中的像素在前向传播时可以影响该单元的激活值：

<!--en:eba74f75 kind=slide_text-->
- **Single layer ($L=1$):** A $k_1 \times k_1$ kernel sees a $k_1 \times k_1$ input patch.
<!--tr:eba74f75-->
- **单层（$L=1$）：** 一个 $k_1 \times k_1$ 卷积核可以看到一个 $k_1 \times k_1$ 输入块。

<!--en:01c175bf kind=slide_text-->
- **Stacking $L$ stride-1 layers:** Layer $i$ with kernel size $k_i$ reaches $k_i - 1$ pixels farther along each axis than a single element of layer $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$
<!--tr:01c175bf-->
- **叠加 $L$ 个步幅为 1 的层：** 第 $i$ 层的卷积核大小为 $k_i$，其单元的感受野沿每个轴都多覆盖 $k_i - 1$ 个像素，这里比较的是第 $i-1$ 层单个元素的感受野：
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$

<!--en:2bde5822 kind=slide_text-->
- **$L$ stacked $3 \times 3$ layers:** Each layer adds $3 - 1 = 2$ pixels, giving side length $r = 2L + 1$ ($3 \to 5 \to 7$).
<!--tr:2bde5822-->
- **叠加 $L$ 个 $3 \times 3$ 卷积层：** 每层使感受野边长增加 $3 - 1 = 2$ 个像素，得到边长 $r = 2L + 1$（$3 \to 5 \to 7$）。

<!--en:cbb9e97a kind=slide_title-->
::: {.slide title="Why Deep Stacks of 3×3 Kernels Beat One Large Kernel"}
<!--tr:cbb9e97a-->
::: {.slide title="为什么叠加多个 3×3 卷积核优于单个大卷积核"}

<!--en:95e46e70 kind=kicker-->
[Parameter Efficiency & Nonlinear Expressivity]{.kicker}
<!--tr:95e46e70-->
[参数效率与非线性表达能力]{.kicker}

<!--en:bd1d7cc0 kind=slide_text-->
- **Matched $7 \times 7$ receptive field:** Three stacked $3 \times 3$ layers ($r = 1 + 2 + 2 + 2 = 7$) see the exact same $7 \times 7$ patch as one $7 \times 7$ layer.
<!--tr:bd1d7cc0-->
- **相同的 $7 \times 7$ 感受野：** 叠加三个 $3 \times 3$ 卷积层（$r = 1 + 2 + 2 + 2 = 7$），看到的 $7 \times 7$ 输入块与单个 $7 \times 7$ 卷积层完全相同。

<!--en:755ac11a kind=slide_text-->
- **45% fewer parameters:** $3 \times 9 c^2 = 27 c^2$ weights instead of $49 c^2$ across $c$ channels.
<!--tr:755ac11a-->
- **参数减少 45%：** 叠加三层只需 $3 \times 9 c^2 = 27 c^2$ 个权重，单层则需 $49 c^2$ 个，两者的通道数均为 $c$。

<!--en:e3aa1086 kind=slide_text-->
- **Richer expressivity:** Compounds **3 nonlinear activations** instead of 1 (the core VGG design rule in §{sec_blocks}).
<!--tr:e3aa1086-->
- **表达能力更强：** 由一次激活变为连续 **3 次非线性激活**（这是 §{sec_blocks} 中 VGG 的核心设计原则）。

<!--en:9ec83bf6 kind=slide_title-->
::: {.slide title="Theoretical vs. Effective Receptive Fields"}
<!--tr:9ec83bf6-->
::: {.slide title="理论感受野与有效感受野"}

<!--en:72a52c3b kind=kicker-->
[Gaussian Decay of Sensitivity Away from the Center]{.kicker}
<!--tr:72a52c3b-->
[灵敏度从中心向外呈高斯衰减]{.kicker}

<!--en:33bb377d kind=slide_text-->
- **Theoretical RF ($r = 2L+1$):** Bounding box of all pixels with *at least one* path to the output unit.
<!--tr:33bb377d-->
- **理论感受野（$r = 2L+1$）：** 所有*至少有一条*路径通往输出单元的像素所构成的边界框。

<!--en:293436eb kind=slide_text-->
- **Central Limit Theorem:** Center pixels connect via exponentially many paths, whereas corner pixels connect via only **1 path**.
<!--tr:293436eb-->
- **中心极限定理：** 中心像素通往输出单元的路径数随深度呈指数级增长，而角落像素只有 **1 条路径**。

<!--en:f47d181e kind=slide_text-->
- **Effective RF (Luo et al., 2016):** Input gradient magnitude $\partial y / \partial \mathbf{X}$ decays in a Gaussian bell curve with std $\sigma \propto \sqrt{L}$.
<!--tr:f47d181e-->
- **有效感受野（Luo 等，2016）：** 输入梯度 $\partial y / \partial \mathbf{X}$ 的幅值从中心向外沿高斯钟形曲线衰减，标准差 $\sigma \propto \sqrt{L}$。

<!--en:fac792d0 kind=slide_title-->
::: {.slide title="Concept Check: Cross-Correlation & Receptive Fields"}
<!--tr:fac792d0-->
::: {.slide title="概念自测：互相关与感受野"}

<!--en:26a317d8 kind=kicker-->
[Concept Check]{.kicker}
<!--tr:26a317d8-->
[概念自测]{.kicker}

<!--en:8fe554e5 kind=slide_text-->
Test your understanding of output shapes, `im2col`, and receptive-field arithmetic:
<!--tr:8fe554e5-->
检验一下你对输出形状、`im2col` 和感受野计算的理解：

<!--en:281c7c78 kind=slide_text-->
1. **im2col matrix dimensions:** You apply an unpadded $3 \times 3$ single-channel kernel to a $10 \times 10$ image via `im2col`. What is the output spatial shape, and what are the exact dimensions of the `patches` matrix and flattened kernel vector?
<!--tr:281c7c78-->
1. **im2col 矩阵的维度：** 将不加填充的 $3 \times 3$ 单通道卷积核作用于 $10 \times 10$ 图像，用 `im2col` 计算。输出的空间形状是什么？`patches` 矩阵和展平的卷积核向量的确切形状各是什么？

<!--en:16976d18 kind=slide_text-->
2. **Depth to span Fashion-MNIST:** At stride 1, how many stacked $3 \times 3$ layers are required so a single unit's theoretical receptive field spans an entire $28 \times 28$ image? How many with $5 \times 5$ layers?
<!--tr:16976d18-->
2. **覆盖 Fashion-MNIST 图像所需的深度：** 步幅为 1 时，需要叠加多少个 $3 \times 3$ 卷积层，单个单元的理论感受野才能覆盖一整幅 $28 \times 28$ 图像？换成 $5 \times 5$ 卷积层呢？

<!--en:ab2e0d6a kind=slide_text-->
3. **Edge-loss boundary shift:** In our $6 \times 8$ vertical-edge learning experiment, why did the loss start at $\approx 12$, and what would the initial loss be if the black band extended all the way to the right edge (`X[:, 2:8] = 0`)?
<!--tr:ab2e0d6a-->
3. **边界移动对边缘检测损失的影响：** 在前面的 $6 \times 8$ 垂直边缘学习实验中，为什么初始损失为 $\approx 12$？如果黑色条带一直延伸到右边界（`X[:, 2:8] = 0`），初始损失会是多少？

<!--en:250d6877 kind=slide_text-->
**Answers:**
(1) Output shape is $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ outputs). The `patches` matrix has shape $(64, 9)$ and multiplies the $(9, 1)$ kernel vector.
(2) For $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ layers. For $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ layers.
(3) Each of the 6 rows had two $\pm 1$ edges ($6 \times 2 \times 1^2 = 12$). Extending the black band to the right border eliminates the right-hand black-to-white edge, leaving one edge per row and halving the initial loss to $\approx 6$.
<!--tr:250d6877-->
**答案：**
(1) 输出形状为 $(10-3+1) \times (10-3+1) = 8 \times 8$（共 $64$ 个输出）。`patches` 矩阵的形状为 $(64, 9)$，与之相乘的卷积核向量形状为 $(9, 1)$。
(2) 使用 $3 \times 3$ 卷积核时，$2L + 1 \ge 28 \implies L = 14$ 层；使用 $5 \times 5$ 卷积核时，$4L + 1 \ge 28 \implies L = 7$ 层。
(3) 6 行中每行都有两处取值为 $\pm 1$ 的边缘（$6 \times 2 \times 1^2 = 12$）。将黑色条带延伸到右边界后，右侧由黑变白的边缘消失，每行只剩一处边缘，初始损失减半至 $\approx 6$。

<!--en:ba43a2dd kind=slide_title-->
::: {.slide title="Summary & Key Takeaways"}
<!--tr:ba43a2dd-->
::: {.slide title="小结与要点"}

<!--en:ce12e85f kind=kicker-->
[Section Synthesis]{.kicker}
<!--tr:ce12e85f-->
[本节回顾]{.kicker}

<!--en:00011fb7 kind=slide_text-->
- **2D Cross-Correlation & Edge Learning:** Slides $\mathbf{K}$ across valid windows ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); SGD learns $[1, -1]$ in 9 steps.
<!--tr:00011fb7-->
- **二维互相关与边缘检测器的学习：** 让 $\mathbf{K}$ 滑过所有有效窗口（共 $(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$ 个），SGD 在 9 步内学得 $[1, -1]$。

<!--en:09efe4f0 kind=slide_text-->
- **im2col Matrix Multiply:** Trades $k_\textrm{h} k_\textrm{w}\times$ patch duplication for peak GPU GEMM throughput.
<!--tr:09efe4f0-->
- **im2col 矩阵乘法：** 图像块复制后占用的空间可增至 $k_\textrm{h} k_\textrm{w}\times$，换来 GPU 执行 GEMM 时的峰值吞吐量。

<!--en:3af8acd0 kind=slide_text-->
- **Stacked Receptive Fields:** $r = 1 + \sum (k_i - 1) = 2L+1$ for $3 \times 3$ stacks, with Gaussian center sensitivity.
<!--tr:3af8acd0-->
- **叠加卷积层的感受野：** 感受野边长为 $r = 1 + \sum (k_i - 1) = 2L+1$（各层均使用 $3 \times 3$ 卷积核），灵敏度以中心为峰值，呈高斯分布。

=== DECISIONS ===
- patch matrix 译为“图像块矩阵”，flatten 译为“展平”，banded matrix 译为“带状矩阵”，effective receptive field 译为“有效感受野”。

=== NOTES ===
- e3aa1086: 数字检查未识别“一次”中的“一”。此处 activations 指依次施加的非线性激活，而非神经元的“激活值”。
