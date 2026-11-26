<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u02 lang=zh rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:0b85caf9 kind=heading-->
## Convolution as Matrix Multiplication
<!--tr:0b85caf9-->
## 作为矩阵乘法的卷积

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
虽然 `corr2d` 中的嵌套循环使数学原理一目了然，但在 Python 中每次只访问一个输出像素对于实际工作负载来说实在太慢了。一种经典的改进方法（至今仍是各类计算库选用的算法之一）是将互相关改写为单次大规模矩阵乘法——这正是 GPU 能以峰值速度执行的运算。注意到 :numref:`fig_correlation` 中的每个输出元素，本质上就是展平后的 $k_\textrm{h} k_\textrm{w}$ 元卷积核与对应展平输入窗口的点积。提取所有 $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ 个窗口并将它们按行堆叠，即可得到一个图块矩阵（patch matrix）；将其与展平后的卷积核相乘，就能一次性算出整个互相关结果。这种标准的布局变换称为 *im2col*（“image to columns”，即图像到列，此处按行排列）。:numref:`fig_conv_im2col` 在 :numref:`fig_correlation` 的示例上演示了这一过程。

<!--en:107554a5 kind=caption-->
![The cross-correlation of :numref:`fig_correlation` as one matrix product. Each of the four $2 \times 2$ windows of the input becomes a row of the patch matrix, and multiplying by the flattened kernel gives the flattened output. The shaded cells follow the first window to its row and to the output 19. The center pixel 4 lies in all four windows, so it appears in every row.](../img/conv-im2col.svg)
<!--tr:107554a5-->
![将 :numref:`fig_correlation` 的互相关表示为单次矩阵乘积。输入的四个 $2 \times 2$ 窗口各自成为图块矩阵的一行，与展平后的卷积核相乘即得到展平后的输出。阴影单元格追踪了第一个窗口到对应行及输出 19 的映射。中心像素 4 位于所有四个窗口中，因此出现在每一行里。](../img/conv-im2col.svg)

<!--en:76713479 kind=paragraph-->
The trade-off is extra workspace memory, since each interior pixel is duplicated
into up to $k_\textrm{h} k_\textrm{w}$ patches. Below we verify on
:numref:`fig_correlation` that this single matrix multiply matches `corr2d`
exactly:
<!--tr:76713479-->
其代价是需要额外的工作空间内存，因为每个内部像素最多会被复制到 $k_\textrm{h} k_\textrm{w}$ 个图块中。下面我们在 :numref:`fig_correlation` 上验证，这种单次矩阵乘法与 `corr2d` 的结果完全一致：

<!--en:3bcd7d8a kind=paragraph-->
The discrepancy is identically zero: one matrix product reproduces `corr2d`
perfectly.
<!--tr:3bcd7d8a-->
误差严格为零：一次矩阵乘积完美复现了 `corr2d`。

<!--en:c30f2dea kind=paragraph-->
Equivalently, we can unroll the *kernel* instead of the input into a sparse
banded matrix $\mathbf{W}$ multiplying the flattened image—the exact 2D analog
of panel (c) in :numref:`fig_why_conv_weight_sharing`, where every row repeats
the same handful of kernel weights. Seeing a convolution as a structured dense
layer brings us full circle to the derivation of :numref:`sec_why-conv` (and the
exercises build $\mathbf{W}$ explicitly).
<!--tr:c30f2dea-->
等价地，我们也可以不展开输入，而是将*卷积核*展开为一个稀疏带状矩阵 $\mathbf{W}$，去乘以展平后的图像——这正是 :numref:`fig_why_conv_weight_sharing` 中 (c) 图的二维推广，其中每一行都重复着相同的几个卷积核权重。将卷积视为结构化的全连接层，使我们完整回到了 :numref:`sec_why-conv` 的推导（练习中也会显式构造 $\mathbf{W}$）。

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
单个卷积层只能看到与其卷积核大小相当的局部区域，而识别完整物体则需要在更广阔的区域内整合上下文信息。回想 :numref:`subsec_why-conv-channels`，卷积层的输出称为*特征图*（feature map，如我们的垂直边缘图 `Y`）。对于特征图中的任意元素 $x$，其*感受野*（receptive field）是指在前向传播过程中，其数值能够影响 $x$ 的输入图像（或更早层）区域——这一术语借自 :numref:`field_visual` 中的神经生理学概念。

<!--en:b36f221e kind=heading-->
### Stacked Layers
<!--tr:b36f221e-->
### 堆叠层

<!--en:24a25ec4 kind=paragraph-->
In :numref:`fig_correlation`, the shaded output `19` has a $2 \times 2$ receptive
field covering the top-left four pixels of $\mathbf{X}$. Now stack a *second*
$2 \times 2$ convolutional layer on top of the $2 \times 2$ output $\mathbf{Y}$
to produce a single scalar $z$: $z$ depends on all four elements of $\mathbf{Y}$,
which together depend on all nine elements of $\mathbf{X}$. Stacking layers
expands the receptive field smoothly with depth.
<!--tr:24a25ec4-->
在 :numref:`fig_correlation` 中，阴影输出 `19` 具有一个 $2 \times 2$ 的感受野，覆盖 $\mathbf{X}$ 左上角的四个像素。现在在 $2 \times 2$ 输出 $\mathbf{Y}$ 之上再堆叠*第二个* $2 \times 2$ 卷积层，以生成单个标量 $z$：$z$ 依赖于 $\mathbf{Y}$ 的全部四个元素，而这四个元素共同依赖于 $\mathbf{X}$ 的全部九个元素。堆叠层会使感受野随深度平滑扩大。

<!--en:f95c9394 kind=paragraph-->
How quickly does it grow? Each $k \times k$ convolutional layer reaches $k - 1$
pixels farther than a single element of its input, so stacking $L$ layers with
kernel sizes $k_1, \ldots, k_L$ yields a receptive field of side length
<!--tr:f95c9394-->
它增长得有多快？每个 $k \times k$ 卷积层比其输入的单个元素多覆盖 $k - 1$ 个像素，因此堆叠 $L$ 个卷积核大小分别为 $k_1, \ldots, k_L$ 的层，得到的感受野边长为

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
对于由 $L$ 个连续 $3 \times 3$ 层组成的堆叠，每层增加 2 个像素，共覆盖 $(2L + 1) \times (2L + 1)$ 个输入：两层后为 $5 \times 5$，三层后为 $7 \times 7$（:numref:`fig_conv_receptive_field`）。这种线性增长实际上是一种优势，而非使用超大卷积核的理由：三个堆叠的 $3 \times 3$ 层能以远少于单个 $7 \times 7$ 层的参数量（每对通道 $3 \times 9 = 27$ 个权重，而非 49 个）达到相同的 $7 \times 7$ 感受野，同时还能交错引入三次非线性而非一次。

<!--en:39917513 kind=caption-->
![Receptive field of one unit after three one-dimensional layers with kernel size 3. Shaded cells are the units that the top unit depends on, and the lines connect each of them to its inputs. Each layer widens the receptive field by 2, to 7 inputs.](../img/conv-receptive-field.svg)
<!--tr:39917513-->
![经过三个卷积核大小为 3 的一维层后单个单元的感受野。阴影单元格表示顶层单元所依赖的单元，连线将它们各自连接到对应的输入。每一层将感受野拓宽 2，最终达到 7 个输入。](../img/conv-receptive-field.svg)

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
需要记住，$r$ 度量的是*理论*感受野（存在*任意*路径通向输出单元的所有输入像素），而未说明每个像素实际贡献的强弱。通过直接测量输入梯度，:citet:`Luo.Li.Urtasun.ea.2016` 表明，无论在未训练还是已训练的网络中，*有效*感受野（effective receptive field）都会在偏离中心后急剧衰减，呈近似高斯钟形曲线。深层单元理论上可以观察到图像的很大范围，但其敏感度高度集中在中心附近。

<!--en:eed4e2a0 kind=paragraph-->
Counting paths explains the bell shape. An input at the edge of the receptive
field reaches the output unit along a single chain of connections, whereas the
center input can take many routes through the layers in between.
:numref:`fig_conv_rf_paths` counts them for the tower of
:numref:`fig_conv_receptive_field`: the center input has seven paths to the top
unit and each edge input has one. If all weights have a similar magnitude, the
influence of an input grows with its number of paths.
<!--tr:eed4e2a0-->
统计路径数即可解释这种钟形分布。感受野边缘的输入仅沿单一连接链到达输出单元，而中心输入则可通过中间各层的多条路径到达。:numref:`fig_conv_rf_paths` 统计了 :numref:`fig_conv_receptive_field` 所示层叠结构中的路径数：中心输入有 7 条路径通向顶层单元，而每个边缘输入只有 1 条。如果所有权重的量级相近，输入的影响力就会随其路径数的增加而增大。

<!--en:b92775ee kind=caption-->
![Number of paths from each unit to the top unit of :numref:`fig_conv_receptive_field` (three one-dimensional layers with kernel size 3). Each count is the sum of the counts of the units that the cell feeds. Darker cells have more paths than the other cells of their layer, and the bars repeat the counts of the input row: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)
<!--tr:b92775ee-->
![从每个单元到 :numref:`fig_conv_receptive_field` 顶层单元的路径数（三个卷积核大小为 3 的一维层）。每个计数等于该单元格所馈送单元的计数之和。颜色越深的单元格比同层其他单元格拥有更多路径，下方的柱状图重复显示了输入行的路径计数：1, 3, 6, 7, 6, 3, 1。](../img/conv-rf-paths.svg)

<!--en:405695e8 kind=paragraph-->
Two practical gaps remain in our basic layer: every unpadded convolution shrinks
spatial dimensions while under-sampling border pixels, and a single kernel
detects only one pattern. We address spatial sizing and borders next in
:numref:`sec_padding`, and multi-channel feature maps in :numref:`sec_channels`.
<!--tr:405695e8-->
我们的基础卷积层还存在两个实际缺陷：每次无填充卷积都会缩小空间尺寸并降低边界像素的采样频率，且单个卷积核只能检测一种模式。接下来，我们将在 :numref:`sec_padding` 中解决空间尺寸与边界问题，并在 :numref:`sec_channels` 中探讨多通道特征图。

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
卷积层使用学到的卷积核对输入执行二维互相关并加上标量偏置，将卷积核滑过输入的每一个有效窗口。简单的 $1 \times 2$ 有限差分卷积核 `[1, -1]` 可检测图像中任意位置的垂直边缘，而梯度下降仅凭输入—输出数据就能恢复这些精确权重，正如深度卷积神经网络端到端地学习整个滤波器层次结构一样。数学卷积会翻转卷积核，但在卷积核通过学习获得的情况下并无差异。卷积可以作为单次矩阵乘法（*im2col*）来计算，而堆叠 $L$ 个卷积层会使每个单元的感受野按每层 $k_i - 1$ 线性扩大。

<!--en:3ba929d2 kind=slide_title-->
::: {.slide title="Fast Execution: Convolution via Matrix Multiplication"}
<!--tr:3ba929d2-->
::: {.slide title="高效执行：通过矩阵乘法实现卷积"}

<!--en:1711f812 kind=kicker-->
[The im2col (Image-to-Columns) Transformation]{.kicker}
<!--tr:1711f812-->
[im2col（图像到列）变换]{.kicker}

<!--en:a040f3d7 kind=slide_text-->
- **Single dense GEMM:** Flatten each $k_\textrm{h} \times k_\textrm{w}$ window into a row of `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ and multiply by $\textrm{vec}(\mathbf{K})$.
<!--tr:a040f3d7-->
- **单次稠密矩阵乘法（GEMM）：**将每个 $k_\textrm{h} \times k_\textrm{w}$ 窗口展平为 `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ 的一行，并与 $\textrm{vec}(\mathbf{K})$ 相乘。

<!--en:1fb18ec4 kind=slide_text-->
- **Trade-off:** Peak GPU Tensor Core speed at the cost of $k_\textrm{h} k_\textrm{w}\times$ patch workspace duplication.
<!--tr:1fb18ec4-->
- **权衡：**以 $k_\textrm{h} k_\textrm{w}\times$ 的图块工作空间重复存储为代价，换取 GPU Tensor Core 的峰值速度。

<!--en:0d757cf6 kind=slide_title-->
::: {.slide title="Dual View: Unrolling the Kernel into a Banded Matrix"}
<!--tr:0d757cf6-->
::: {.slide title="对偶视角：将卷积核展开为带状矩阵"}

<!--en:7f0defaa kind=kicker-->
[Toeplitz Weight Matrix Multiplication]{.kicker}
<!--tr:7f0defaa-->
[Toeplitz 权重矩阵乘法]{.kicker}

<!--en:5898ba57 kind=slide_text-->
Instead of unfolding the input image into duplicated patches, we can unroll the *kernel* into a sparse banded matrix $\mathbf{W}$ that multiplies the flattened input vector $\textrm{vec}(\mathbf{X})$:
<!--tr:5898ba57-->
我们也可以不将输入图像展开为重复的图块，而是将*卷积核*展开为稀疏带状矩阵 $\mathbf{W}$，用于乘以展平的输入向量 $\textrm{vec}(\mathbf{X})$：

<!--en:dae51741 kind=slide_text-->
- **Matrix–vector form:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, where $\mathbf{W}$ has $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ rows and $n_\textrm{h} n_\textrm{w}$ columns.
<!--tr:dae51741-->
- **矩阵—向量形式：**$\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$，其中 $\mathbf{W}$ 有 $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ 行和 $n_\textrm{h} n_\textrm{w}$ 列。

<!--en:f0e86ee3 kind=slide_text-->
- **Connection to §{sec_why-conv} and backprop:** Every row of $\mathbf{W}$ holds the same $k_\textrm{h} k_\textrm{w}$ kernel entries shifted across columns (panel c), making the backward pass with respect to $\mathbf{X}$ simply multiplication by $\mathbf{W}^\top$ (a transposed convolution).
<!--tr:f0e86ee3-->
- **与 §{sec_why-conv} 及反向传播的联系：**$\mathbf{W}$ 的每一行包含沿列平移的相同 $k_\textrm{h} k_\textrm{w}$ 个卷积核元素（图 c），使对 $\mathbf{X}$ 的反向传播简化为乘以 $\mathbf{W}^\top$（即转置卷积）。

<!--en:af8cf19a kind=slide_title-->
::: {.slide title="Feature Maps and Receptive Fields Across Depth"}
<!--tr:af8cf19a-->
::: {.slide title="特征图与随深度变化的感受野"}

<!--en:4eb965f3 kind=kicker-->
[Linear Receptive-Field Growth in Stacked Layers]{.kicker}
<!--tr:4eb965f3-->
[堆叠层中感受野的线性增长]{.kicker}

<!--en:5599a6e9 kind=slide_text-->
A unit's **receptive field** is the region of the original input image whose pixels can influence that unit's activation during forward propagation:
<!--tr:5599a6e9-->
一个单元的**感受野**是指原始输入图像中在前向传播期间能够影响该单元激活值的像素区域：

<!--en:eba74f75 kind=slide_text-->
- **Single layer ($L=1$):** A $k_1 \times k_1$ kernel sees a $k_1 \times k_1$ input patch.
<!--tr:eba74f75-->
- **单层（$L=1$）：**$k_1 \times k_1$ 卷积核看到 $k_1 \times k_1$ 的输入区域。

<!--en:01c175bf kind=slide_text-->
- **Stacking $L$ stride-1 layers:** Layer $i$ with kernel size $k_i$ reaches $k_i - 1$ pixels farther along each axis than a single element of layer $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$
<!--tr:01c175bf-->
- **堆叠 $L$ 个步幅为 1 的层：**卷积核大小为 $k_i$ 的第 $i$ 层在每个轴上比第 $i-1$ 层的单个元素多覆盖 $k_i - 1$ 个像素：
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$

<!--en:2bde5822 kind=slide_text-->
- **$L$ stacked $3 \times 3$ layers:** Each layer adds $3 - 1 = 2$ pixels, giving side length $r = 2L + 1$ ($3 \to 5 \to 7$).
<!--tr:2bde5822-->
- **$L$ 个堆叠的 $3 \times 3$ 层：**每层增加 $3 - 1 = 2$ 个像素，边长为 $r = 2L + 1$（$3 \to 5 \to 7$）。

<!--en:cbb9e97a kind=slide_title-->
::: {.slide title="Why Deep Stacks of 3×3 Kernels Beat One Large Kernel"}
<!--tr:cbb9e97a-->
::: {.slide title="为何多层 3×3 卷积核堆叠优于单个大卷积核"}

<!--en:95e46e70 kind=kicker-->
[Parameter Efficiency & Nonlinear Expressivity]{.kicker}
<!--tr:95e46e70-->
[参数效率与非线性表达能力]{.kicker}

<!--en:bd1d7cc0 kind=slide_text-->
- **Matched $7 \times 7$ receptive field:** Three stacked $3 \times 3$ layers ($r = 1 + 2 + 2 + 2 = 7$) see the exact same $7 \times 7$ patch as one $7 \times 7$ layer.
<!--tr:bd1d7cc0-->
- **匹配的 $7 \times 7$ 感受野：**三个堆叠的 $3 \times 3$ 层（$r = 1 + 2 + 2 + 2 = 7$）看到的 $7 \times 7$ 区域与单个 $7 \times 7$ 层完全相同。

<!--en:755ac11a kind=slide_text-->
- **45% fewer parameters:** $3 \times 9 c^2 = 27 c^2$ weights instead of $49 c^2$ across $c$ channels.
<!--tr:755ac11a-->
- **参数量减少 45%：**在 $c$ 个通道间仅需 $3 \times 9 c^2 = 27 c^2$ 个权重，而非 $49 c^2$ 个。

<!--en:e3aa1086 kind=slide_text-->
- **Richer expressivity:** Compounds **3 nonlinear activations** instead of 1 (the core VGG design rule in §{sec_blocks}).
<!--tr:e3aa1086-->
- **更强的表达能力：**复合了 **3 次非线性激活**而非 1 次（§{sec_blocks} 中 VGG 的核心设计原则）。

<!--en:9ec83bf6 kind=slide_title-->
::: {.slide title="Theoretical vs. Effective Receptive Fields"}
<!--tr:9ec83bf6-->
::: {.slide title="理论感受野与有效感受野"}

<!--en:72a52c3b kind=kicker-->
[Gaussian Decay of Sensitivity Away from the Center]{.kicker}
<!--tr:72a52c3b-->
[敏感度从中心向四周的高斯衰减]{.kicker}

<!--en:33bb377d kind=slide_text-->
- **Theoretical RF ($r = 2L+1$):** Bounding box of all pixels with *at least one* path to the output unit.
<!--tr:33bb377d-->
- **理论感受野（$r = 2L+1$）：**所有存在*至少一条*路径通向输出单元的像素边界框。

<!--en:293436eb kind=slide_text-->
- **Central Limit Theorem:** Center pixels connect via exponentially many paths, whereas corner pixels connect via only **1 path**.
<!--tr:293436eb-->
- **中心极限定理：**中心像素通过指数级数量的路径相连，而角点像素仅通过 **1 条路径**相连。

<!--en:f47d181e kind=slide_text-->
- **Effective RF (Luo et al., 2016):** Input gradient magnitude $\partial y / \partial \mathbf{X}$ decays in a Gaussian bell curve with std $\sigma \propto \sqrt{L}$.
<!--tr:f47d181e-->
- **有效感受野（Luo 等，2016）：**输入梯度幅值 $\partial y / \partial \mathbf{X}$ 呈高斯钟形曲线衰减，标准差 $\sigma \propto \sqrt{L}$。

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
检验你对输出形状、`im2col` 和感受野计算的理解：

<!--en:281c7c78 kind=slide_text-->
1. **im2col matrix dimensions:** You apply an unpadded $3 \times 3$ single-channel kernel to a $10 \times 10$ image via `im2col`. What is the output spatial shape, and what are the exact dimensions of the `patches` matrix and flattened kernel vector?
<!--tr:281c7c78-->
1. **im2col 矩阵维度：**通过 `im2col` 将无填充的 $3 \times 3$ 单通道卷积核应用于 $10 \times 10$ 图像。输出空间形状是什么？`patches` 矩阵和展平卷积核向量的精确维度各是多少？

<!--en:16976d18 kind=slide_text-->
2. **Depth to span Fashion-MNIST:** At stride 1, how many stacked $3 \times 3$ layers are required so a single unit's theoretical receptive field spans an entire $28 \times 28$ image? How many with $5 \times 5$ layers?
<!--tr:16976d18-->
2. **覆盖 Fashion-MNIST 所需的深度：**步幅为 1 时，需要堆叠多少个 $3 \times 3$ 层才能使单个单元的理论感受野覆盖整张 $28 \times 28$ 图像？若使用 $5 \times 5$ 层又需要多少个？

<!--en:ab2e0d6a kind=slide_text-->
3. **Edge-loss boundary shift:** In our $6 \times 8$ vertical-edge learning experiment, why did the loss start at $\approx 12$, and what would the initial loss be if the black band extended all the way to the right edge (`X[:, 2:8] = 0`)?
<!--tr:ab2e0d6a-->
3. **边缘损失的边界移动：**在我们的 $6 \times 8$ 垂直边缘学习实验中，为何损失从 $\approx 12$ 开始？如果黑色条带一直延伸到右边界（`X[:, 2:8] = 0`），初始损失会是多少？

<!--en:250d6877 kind=slide_text-->
**Answers:**
(1) Output shape is $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ outputs). The `patches` matrix has shape $(64, 9)$ and multiplies the $(9, 1)$ kernel vector.
(2) For $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ layers. For $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ layers.
(3) Each of the 6 rows had two $\pm 1$ edges ($6 \times 2 \times 1^2 = 12$). Extending the black band to the right border eliminates the right-hand black-to-white edge, leaving one edge per row and halving the initial loss to $\approx 6$.
<!--tr:250d6877-->
**答案：**
(1) 输出形状为 $(10-3+1) \times (10-3+1) = 8 \times 8$（$64$ 个输出）。`patches` 矩阵形状为 $(64, 9)$，与 $(9, 1)$ 卷积核向量相乘。
(2) 对于 $3 \times 3$：$2L + 1 \ge 28 \implies L = 14$ 层。对于 $5 \times 5$：$4L + 1 \ge 28 \implies L = 7$ 层。
(3) 6 行中每行都有两条 $\pm 1$ 边缘（$6 \times 2 \times 1^2 = 12$）。将黑色条带延伸到右边界会消除右侧的黑到白边缘，每行只剩一条边缘，使初始损失减半至 $\approx 6$。

<!--en:ba43a2dd kind=slide_title-->
::: {.slide title="Summary & Key Takeaways"}
<!--tr:ba43a2dd-->
::: {.slide title="总结与核心要点"}

<!--en:ce12e85f kind=kicker-->
[Section Synthesis]{.kicker}
<!--tr:ce12e85f-->
[本节综合]{.kicker}

<!--en:00011fb7 kind=slide_text-->
- **2D Cross-Correlation & Edge Learning:** Slides $\mathbf{K}$ across valid windows ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); SGD learns $[1, -1]$ in 9 steps.
<!--tr:00011fb7-->
- **二维互相关与边缘学习：**将 $\mathbf{K}$ 滑过有效窗口（$(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$）；SGD 在 9 步内学得 $[1, -1]$。

<!--en:09efe4f0 kind=slide_text-->
- **im2col Matrix Multiply:** Trades $k_\textrm{h} k_\textrm{w}\times$ patch duplication for peak GPU GEMM throughput.
<!--tr:09efe4f0-->
- **im2col 矩阵乘法：**以 $k_\textrm{h} k_\textrm{w}\times$ 的图块重复存储换取 GPU GEMM 峰值吞吐量。

<!--en:3af8acd0 kind=slide_text-->
- **Stacked Receptive Fields:** $r = 1 + \sum (k_i - 1) = 2L+1$ for $3 \times 3$ stacks, with Gaussian center sensitivity.
<!--tr:3af8acd0-->
- **堆叠感受野：**$3 \times 3$ 堆叠的感受野为 $r = 1 + \sum (k_i - 1) = 2L+1$，且中心敏感度呈高斯分布。

=== DECISIONS ===
- Spacing: insert a single half-width space between CJK characters and ASCII/math/inline-code/cross-references (`:numref:`, `:eqref:`, `:cite:`, `§{...}`), except when adjacent to full-width Chinese punctuation（，。：；（）？！“”）.
- Punctuation: use full-width Chinese punctuation in prose and slides; keep `—` (em-dash) or Chinese phrasing where natural.
- First-use English annotations: introduce key new terms in full-width parentheses on first occurrence, e.g., 二维互相关（two-dimensional cross-correlation）.
- Register: use "我们" (we) throughout, matching standard Chinese D2L textbook style.
- Terminology: follow BRIEF glossary strictly (`卷积核` for kernel, `互相关` for cross-correlation, `步幅` for stride, `填充` for padding, `感受野` for receptive field, `特征图` for feature map, `有限差分` for finite difference).
- "patch matrix" → 「图块矩阵」 (no glossary entry; matches "patch" = 图块 in `im2col` context).

=== NOTES ===
- 24a25ec4, b302e3ef, 01c175bf, 755ac11a, 3af8acd0: math tokens reordered to match Chinese grammar.
- e3aa1086: glossary false match on "3 nonlinear activations" (means 3 nonlinear activation steps/functions「3 次非线性激活」, not activation values `激活值`).
