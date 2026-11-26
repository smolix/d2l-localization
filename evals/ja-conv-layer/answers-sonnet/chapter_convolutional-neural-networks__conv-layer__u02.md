<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u02 lang=ja rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:0b85caf9 kind=heading-->
## Convolution as Matrix Multiplication
<!--tr:0b85caf9-->
## 行列積としての畳み込み

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
`corr2d` の入れ子ループは計算の意味を明確にしますが、Python で出力ピクセルを 1 つずつ訪れるのは、実際の計算負荷には遅すぎます。古典的な対処法であり、いまもライブラリが選択肢の 1 つとして持つアルゴリズムが、相互相関を 1 つの大きな行列積として書き直すことです。行列積は、GPU がピーク速度で実行できる演算そのものです。:numref:`fig_correlation` の各出力要素は、$k_\textrm{h} k_\textrm{w}$ 個の要素を持つ平坦化したカーネルと、対応する平坦化した入力ウィンドウとのドット積にほかならない、という点に注目してください。そこで、$(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ 個のウィンドウをすべて取り出して行として積み重ねるとパッチ行列ができ、これに平坦化したカーネルを掛ければ、相互相関全体が一度に計算されます。この標準的なレイアウト変換を*im2col*（「image to columns」。ここでは行）と呼びます。:numref:`fig_conv_im2col` は、:numref:`fig_correlation` の例でこの変換を実行したものです。

<!--en:107554a5 kind=caption-->
![The cross-correlation of :numref:`fig_correlation` as one matrix product. Each of the four $2 \times 2$ windows of the input becomes a row of the patch matrix, and multiplying by the flattened kernel gives the flattened output. The shaded cells follow the first window to its row and to the output 19. The center pixel 4 lies in all four windows, so it appears in every row.](../img/conv-im2col.svg)
<!--tr:107554a5-->
![:numref:`fig_correlation` の相互相関を 1 つの行列積として表したもの。入力の 4 つの $2 \times 2$ ウィンドウがそれぞれパッチ行列の 1 行になり、平坦化したカーネルを掛けると平坦化した出力が得られる。網掛けのセルは、最初のウィンドウをたどって、そのパッチ行列の行と出力 19 に至る。中心のピクセル 4 は 4 つのウィンドウすべてに含まれるため、すべての行に現れる。](../img/conv-im2col.svg)

<!--en:76713479 kind=paragraph-->
The trade-off is extra workspace memory, since each interior pixel is duplicated
into up to $k_\textrm{h} k_\textrm{w}$ patches. Below we verify on
:numref:`fig_correlation` that this single matrix multiply matches `corr2d`
exactly:
<!--tr:76713479-->
代償は追加の作業用メモリです。内部の各ピクセルが最大 $k_\textrm{h} k_\textrm{w}$ 個のパッチに複製されるからです。以下では :numref:`fig_correlation` の例で、この 1 回の行列積が `corr2d` と完全に一致することを確かめます。

<!--en:3bcd7d8a kind=paragraph-->
The discrepancy is identically zero: one matrix product reproduces `corr2d`
perfectly.
<!--tr:3bcd7d8a-->
差はまったく 0 であり、1 回の行列積で `corr2d` が完全に再現されます。

<!--en:c30f2dea kind=paragraph-->
Equivalently, we can unroll the *kernel* instead of the input into a sparse
banded matrix $\mathbf{W}$ multiplying the flattened image—the exact 2D analog
of panel (c) in :numref:`fig_why_conv_weight_sharing`, where every row repeats
the same handful of kernel weights. Seeing a convolution as a structured dense
layer brings us full circle to the derivation of :numref:`sec_why-conv` (and the
exercises build $\mathbf{W}$ explicitly).
<!--tr:c30f2dea-->
同様に、入力のかわりに*カーネル*を展開して、平坦化した画像に掛かる疎な帯行列 $\mathbf{W}$ にすることもできます。これは :numref:`fig_why_conv_weight_sharing` のパネル (c) の 2 次元版にそのまま対応し、どの行も同じ少数のカーネルの重みを繰り返します。畳み込みを構造を持つ全結合層とみなすと、:numref:`sec_why-conv` の導出に立ち返ることになります（演習問題では $\mathbf{W}$ を実際に構成します）。

<!--en:ad4fc956 kind=heading-->
## Feature Maps and Receptive Fields
<!--tr:ad4fc956-->
## 特徴マップと受容野

<!--en:e7983df8 kind=paragraph-->
A single convolutional layer sees only a patch the size of its kernel, whereas
recognizing whole objects demands integrating context across much wider areas.
Recall from :numref:`subsec_why-conv-channels` that a convolutional layer's
output is called a *feature map* (like our vertical edge map `Y`). For any
element $x$ in a feature map, its *receptive field* is the region of the input
image (or of an earlier layer) whose values can influence $x$ during forward
propagation—borrowing the neurophysiology term from :numref:`field_visual`.
<!--tr:e7983df8-->
1 つの畳み込み層が見るのは、カーネルと同じ大きさのパッチだけですが、物体全体を認識するには、はるかに広い範囲の文脈を統合する必要があります。:numref:`subsec_why-conv-channels` で述べたように、畳み込み層の出力は*特徴マップ*（feature map）と呼ばれます（先ほどの垂直エッジのマップ `Y` がその例です）。特徴マップの任意の要素 $x$ について、その*受容野*とは、順伝播の際に値が $x$ に影響しうる、入力画像（または前の層）の領域のことです。この用語は :numref:`field_visual` の神経生理学の用語を借りたものです。

<!--en:b36f221e kind=heading-->
### Stacked Layers
<!--tr:b36f221e-->
### 層の積み重ね

<!--en:24a25ec4 kind=paragraph-->
In :numref:`fig_correlation`, the shaded output `19` has a $2 \times 2$ receptive
field covering the top-left four pixels of $\mathbf{X}$. Now stack a *second*
$2 \times 2$ convolutional layer on top of the $2 \times 2$ output $\mathbf{Y}$
to produce a single scalar $z$: $z$ depends on all four elements of $\mathbf{Y}$,
which together depend on all nine elements of $\mathbf{X}$. Stacking layers
expands the receptive field smoothly with depth.
<!--tr:24a25ec4-->
:numref:`fig_correlation` で網掛けされた出力 `19` の受容野は $2 \times 2$ で、$\mathbf{X}$ の左上の 4 ピクセルを覆います。ここで、*2 つ目*の $2 \times 2$ 畳み込み層を、$2 \times 2$ の出力 $\mathbf{Y}$ の上に積み重ねて、1 つのスカラー $z$ を作るとします。$z$ は $\mathbf{Y}$ の 4 要素すべてに依存し、それらは合わせて $\mathbf{X}$ の 9 要素すべてに依存します。層を積み重ねると、受容野は深さとともになめらかに広がります。

<!--en:f95c9394 kind=paragraph-->
How quickly does it grow? Each $k \times k$ convolutional layer reaches $k - 1$
pixels farther than a single element of its input, so stacking $L$ layers with
kernel sizes $k_1, \ldots, k_L$ yields a receptive field of side length
<!--tr:f95c9394-->
どれくらいの速さで広がるのでしょうか。各 $k \times k$ 畳み込み層は、入力の 1 要素より $k - 1$ ピクセルだけ遠くまで届くため、$L$ 層を、カーネルサイズ $k_1, \ldots, k_L$ で積み重ねると、受容野の一辺の長さは次のようになります。

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
連続する $L$ 個の $3 \times 3$ 層を積み重ねると、各層が 2 ピクセルずつ加わり、$(2L + 1) \times (2L + 1)$ の入力を覆います。2 層で $5 \times 5$、3 層で $7 \times 7$ です（:numref:`fig_conv_receptive_field`）。この線形の広がりは、巨大なカーネルに頼る理由にはならず、むしろ好都合です。$3 \times 3$ の層を 3 つ積み重ねれば、$7 \times 7$ の層 1 つと同じ $7 \times 7$ の受容野が、はるかに少ないパラメータ（チャネルの組あたり、重みは 49 個ではなく $3 \times 9 = 27$ 個）で得られ、しかも非線形性が 1 回ではなく 3 回加わります。

<!--en:39917513 kind=caption-->
![Receptive field of one unit after three one-dimensional layers with kernel size 3. Shaded cells are the units that the top unit depends on, and the lines connect each of them to its inputs. Each layer widens the receptive field by 2, to 7 inputs.](../img/conv-receptive-field.svg)
<!--tr:39917513-->
![カーネルサイズ 3 の 1 次元の層を 3 つ重ねたあとの、1 つのユニットの受容野。網掛けのセルは最上位のユニットが依存するユニットで、線は各ユニットをその入力と結ぶ。各層で受容野は 2 ずつ広がり、7 個の入力に達する。](../img/conv-receptive-field.svg)

<!--en:48e4c9cc kind=heading-->
### Effective Receptive Fields
<!--tr:48e4c9cc-->
### 有効受容野

<!--en:77d044db kind=paragraph-->
Keep in mind that $r$ measures the *theoretical* receptive field (every input
pixel with *any* path to an output unit) without saying how strongly each pixel
actually contributes. Measuring input gradients directly,
:citet:`Luo.Li.Urtasun.ea.2016` showed that the *effective* receptive field
decays sharply away from the center in a roughly Gaussian bell curve, in both
untrained and trained networks. Deep units can theoretically see far across the
image, but their sensitivity concentrates heavily near the center.
<!--tr:77d044db-->
なお、$r$ が測るのは*理論上の*受容野（出力ユニットへの経路が*どれか 1 本でも*あるすべての入力ピクセル）であって、各ピクセルが実際にどれだけ強く寄与するかは表しません。入力勾配を直接測定した :citet:`Luo.Li.Urtasun.ea.2016` は、*有効*受容野が、訓練前のネットワークでも訓練後のネットワークでも、中心から離れるにつれて、ほぼガウス分布の釣り鐘型の曲線を描いて急速に減衰することを示しました。深いユニットは理論上は画像の遠くまで見渡せますが、その感度は中心付近に強く集中します。

<!--en:eed4e2a0 kind=paragraph-->
Counting paths explains the bell shape. An input at the edge of the receptive
field reaches the output unit along a single chain of connections, whereas the
center input can take many routes through the layers in between.
:numref:`fig_conv_rf_paths` counts them for the tower of
:numref:`fig_conv_receptive_field`: the center input has seven paths to the top
unit and each edge input has one. If all weights have a similar magnitude, the
influence of an input grows with its number of paths.
<!--tr:eed4e2a0-->
経路を数えると、釣り鐘型になる理由がわかります。受容野の端にある入力は、ただ 1 本の接続の連鎖で出力ユニットに届きますが、中心の入力は、途中の層を通る経路を多数取れます。:numref:`fig_conv_rf_paths` は、:numref:`fig_conv_receptive_field` の積み重ねた層についてその数を数えたもので、中心の入力から最上位のユニットへの経路は 7 本、端の各入力からは 1 本です。すべての重みの大きさが同程度なら、入力の影響は経路の数とともに大きくなります。

<!--en:b92775ee kind=caption-->
![Number of paths from each unit to the top unit of :numref:`fig_conv_receptive_field` (three one-dimensional layers with kernel size 3). Each count is the sum of the counts of the units that the cell feeds. Darker cells have more paths than the other cells of their layer, and the bars repeat the counts of the input row: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)
<!--tr:b92775ee-->
![各ユニットから :numref:`fig_conv_receptive_field`（カーネルサイズ 3 の 1 次元の 3 層）の最上位ユニットへの経路の数。各数は、そのセルが入力を与えるユニットの数の和である。色の濃いセルは、同じ層の他のセルより経路が多い。棒グラフは入力行の数 1、3、6、7、6、3、1 を繰り返したものである。](../img/conv-rf-paths.svg)

<!--en:405695e8 kind=paragraph-->
Two practical gaps remain in our basic layer: every unpadded convolution shrinks
spatial dimensions while under-sampling border pixels, and a single kernel
detects only one pattern. We address spatial sizing and borders next in
:numref:`sec_padding`, and multi-channel feature maps in :numref:`sec_channels`.
<!--tr:405695e8-->
ここまでの基本的な層には、実用上の問題が 2 つ残っています。パディングなしの畳み込みはどれも空間次元を縮め、境界のピクセルをあまり使わないこと、そして 1 つのカーネルが検出できるパターンは 1 つだけであることです。空間サイズと境界は次に :numref:`sec_padding` で、複数チャネルの特徴マップは :numref:`sec_channels` で扱います。

<!--en:30ac03ff kind=heading-->
## Summary
<!--tr:30ac03ff-->
## まとめ

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
畳み込み層は、学習されたカーネルによる 2 次元相互相関にスカラーのバイアスを加えたもので、カーネルを入力のすべての有効なウィンドウにわたってスライドさせます。単純な $1 \times 2$ の有限差分カーネル `[1, -1]` は、画像のどこにあっても垂直エッジを検出でき、勾配降下法は、入出力データだけからまさにその重みを復元します。これは、深い CNN が、フィルタの階層全体をエンドツーエンドで学習するのと同じです。数学的な畳み込みはカーネルを反転しますが、カーネルを学習する場合は違いが生じません。畳み込みは 1 回の行列積（*im2col*）として評価でき、$L$ 個の畳み込み層を積み重ねると、各ユニットの受容野は 1 層あたり $k_i - 1$ ずつ線形に広がります。

<!--en:3ba929d2 kind=slide_title-->
::: {.slide title="Fast Execution: Convolution via Matrix Multiplication"}
<!--tr:3ba929d2-->
::: {.slide title="行列積による畳み込みの高速化"}

<!--en:1711f812 kind=kicker-->
[The im2col (Image-to-Columns) Transformation]{.kicker}
<!--tr:1711f812-->
[im2col（画像を列に並べる）変換]{.kicker}

<!--en:a040f3d7 kind=slide_text-->
- **Single dense GEMM:** Flatten each $k_\textrm{h} \times k_\textrm{w}$ window into a row of `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ and multiply by $\textrm{vec}(\mathbf{K})$.
<!--tr:a040f3d7-->
- **1 回の密な GEMM：**各 $k_\textrm{h} \times k_\textrm{w}$ ウィンドウを、`patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ の 1 行に平坦化し、$\textrm{vec}(\mathbf{K})$ を掛ける

<!--en:1fb18ec4 kind=slide_text-->
- **Trade-off:** Peak GPU Tensor Core speed at the cost of $k_\textrm{h} k_\textrm{w}\times$ patch workspace duplication.
<!--tr:1fb18ec4-->
- **トレードオフ：**GPU の Tensor Core のピーク速度が得られる代わりに、パッチ用ワークスペースが $k_\textrm{h} k_\textrm{w}\times$ に複製される

<!--en:0d757cf6 kind=slide_title-->
::: {.slide title="Dual View: Unrolling the Kernel into a Banded Matrix"}
<!--tr:0d757cf6-->
::: {.slide title="双対の見方：カーネルを帯行列に展開する"}

<!--en:7f0defaa kind=kicker-->
[Toeplitz Weight Matrix Multiplication]{.kicker}
<!--tr:7f0defaa-->
[トプリッツ型の重み行列の積]{.kicker}

<!--en:5898ba57 kind=slide_text-->
Instead of unfolding the input image into duplicated patches, we can unroll the *kernel* into a sparse banded matrix $\mathbf{W}$ that multiplies the flattened input vector $\textrm{vec}(\mathbf{X})$:
<!--tr:5898ba57-->
入力画像を複製したパッチに展開するかわりに、*カーネル*を疎な帯行列 $\mathbf{W}$ に展開して、平坦化した入力ベクトル $\textrm{vec}(\mathbf{X})$ に掛けることもできる。

<!--en:dae51741 kind=slide_text-->
- **Matrix–vector form:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, where $\mathbf{W}$ has $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ rows and $n_\textrm{h} n_\textrm{w}$ columns.
<!--tr:dae51741-->
- **行列とベクトルの形：**$\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$ で、$\mathbf{W}$ は $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ 行、$n_\textrm{h} n_\textrm{w}$ 列である

<!--en:f0e86ee3 kind=slide_text-->
- **Connection to §{sec_why-conv} and backprop:** Every row of $\mathbf{W}$ holds the same $k_\textrm{h} k_\textrm{w}$ kernel entries shifted across columns (panel c), making the backward pass with respect to $\mathbf{X}$ simply multiplication by $\mathbf{W}^\top$ (a transposed convolution).
<!--tr:f0e86ee3-->
- **§{sec_why-conv} と逆伝播との関係：**$\mathbf{W}$ の各行は、同じ $k_\textrm{h} k_\textrm{w}$ 個のカーネル要素を列方向にずらしたもので（パネル c）、$\mathbf{X}$ に関する逆伝播は、$\mathbf{W}^\top$ を掛けるだけになる（転置畳み込み）

<!--en:af8cf19a kind=slide_title-->
::: {.slide title="Feature Maps and Receptive Fields Across Depth"}
<!--tr:af8cf19a-->
::: {.slide title="深さに沿った特徴マップと受容野"}

<!--en:4eb965f3 kind=kicker-->
[Linear Receptive-Field Growth in Stacked Layers]{.kicker}
<!--tr:4eb965f3-->
[層を積み重ねたときの受容野の線形な拡大]{.kicker}

<!--en:5599a6e9 kind=slide_text-->
A unit's **receptive field** is the region of the original input image whose pixels can influence that unit's activation during forward propagation:
<!--tr:5599a6e9-->
ユニットの**受容野**とは、順伝播の間にそのユニットの活性化に影響しうる、元の入力画像のピクセルの領域である。

<!--en:eba74f75 kind=slide_text-->
- **Single layer ($L=1$):** A $k_1 \times k_1$ kernel sees a $k_1 \times k_1$ input patch.
<!--tr:eba74f75-->
- **1 層（$L=1$）：**$k_1 \times k_1$ のカーネルは、$k_1 \times k_1$ の入力パッチを見る

<!--en:01c175bf kind=slide_text-->
- **Stacking $L$ stride-1 layers:** Layer $i$ with kernel size $k_i$ reaches $k_i - 1$ pixels farther along each axis than a single element of layer $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$
<!--tr:01c175bf-->
- **ストライド 1 の $L$ 層の積み重ね：**第 $i$ 層（カーネルサイズ $k_i$）は、第 $i-1$ 層の 1 要素より各軸に沿って $k_i - 1$ ピクセル遠くまで届く
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$

<!--en:2bde5822 kind=slide_text-->
- **$L$ stacked $3 \times 3$ layers:** Each layer adds $3 - 1 = 2$ pixels, giving side length $r = 2L + 1$ ($3 \to 5 \to 7$).
<!--tr:2bde5822-->
- **$L$ 個の $3 \times 3$ 層の積み重ね：**各層が $3 - 1 = 2$ ピクセルを加え、一辺の長さは $r = 2L + 1$（$3 \to 5 \to 7$）になる

<!--en:cbb9e97a kind=slide_title-->
::: {.slide title="Why Deep Stacks of 3×3 Kernels Beat One Large Kernel"}
<!--tr:cbb9e97a-->
::: {.slide title="3×3 カーネルの深い積み重ねが大きな 1 つのカーネルに勝る理由"}

<!--en:95e46e70 kind=kicker-->
[Parameter Efficiency & Nonlinear Expressivity]{.kicker}
<!--tr:95e46e70-->
[パラメータ効率と非線形の表現力]{.kicker}

<!--en:bd1d7cc0 kind=slide_text-->
- **Matched $7 \times 7$ receptive field:** Three stacked $3 \times 3$ layers ($r = 1 + 2 + 2 + 2 = 7$) see the exact same $7 \times 7$ patch as one $7 \times 7$ layer.
<!--tr:bd1d7cc0-->
- **一致する $7 \times 7$ の受容野：**$3 \times 3$ の層を 3 つ積み重ねると（$r = 1 + 2 + 2 + 2 = 7$）、$7 \times 7$ の層 1 つとまったく同じ $7 \times 7$ のパッチが見える

<!--en:755ac11a kind=slide_text-->
- **45% fewer parameters:** $3 \times 9 c^2 = 27 c^2$ weights instead of $49 c^2$ across $c$ channels.
<!--tr:755ac11a-->
- **パラメータが 45% 少ない：**重みは $3 \times 9 c^2 = 27 c^2$ 個で、$49 c^2$ 個より少ない（$c$ チャネルの場合）

<!--en:e3aa1086 kind=slide_text-->
- **Richer expressivity:** Compounds **3 nonlinear activations** instead of 1 (the core VGG design rule in §{sec_blocks}).
<!--tr:e3aa1086-->
- **より豊かな表現力：**非線形活性化を、1 つではなく **3 つ**重ねられる（§{sec_blocks} の VGG の中核となる設計規則）

<!--en:9ec83bf6 kind=slide_title-->
::: {.slide title="Theoretical vs. Effective Receptive Fields"}
<!--tr:9ec83bf6-->
::: {.slide title="理論上の受容野と有効受容野"}

<!--en:72a52c3b kind=kicker-->
[Gaussian Decay of Sensitivity Away from the Center]{.kicker}
<!--tr:72a52c3b-->
[中心から離れるにつれた感度のガウス的な減衰]{.kicker}

<!--en:33bb377d kind=slide_text-->
- **Theoretical RF ($r = 2L+1$):** Bounding box of all pixels with *at least one* path to the output unit.
<!--tr:33bb377d-->
- **理論上の RF（$r = 2L+1$）：**出力ユニットへの経路が*少なくとも 1 本*あるすべてのピクセルのバウンディングボックス

<!--en:293436eb kind=slide_text-->
- **Central Limit Theorem:** Center pixels connect via exponentially many paths, whereas corner pixels connect via only **1 path**.
<!--tr:293436eb-->
- **中心極限定理：**中心のピクセルは指数的に多くの経路でつながるが、隅のピクセルはわずか **1 本の経路**でしかつながらない

<!--en:f47d181e kind=slide_text-->
- **Effective RF (Luo et al., 2016):** Input gradient magnitude $\partial y / \partial \mathbf{X}$ decays in a Gaussian bell curve with std $\sigma \propto \sqrt{L}$.
<!--tr:f47d181e-->
- **有効 RF（Luo et al., 2016）：**入力勾配の大きさ $\partial y / \partial \mathbf{X}$ は、標準偏差 $\sigma \propto \sqrt{L}$ のガウス型の釣り鐘曲線に沿って減衰する

<!--en:fac792d0 kind=slide_title-->
::: {.slide title="Concept Check: Cross-Correlation & Receptive Fields"}
<!--tr:fac792d0-->
::: {.slide title="確認問題：相互相関と受容野"}

<!--en:26a317d8 kind=kicker-->
[Concept Check]{.kicker}
<!--tr:26a317d8-->
[確認問題]{.kicker}

<!--en:8fe554e5 kind=slide_text-->
Test your understanding of output shapes, `im2col`, and receptive-field arithmetic:
<!--tr:8fe554e5-->
出力の形状、`im2col`、受容野の計算について、理解を確かめてみましょう。

<!--en:281c7c78 kind=slide_text-->
1. **im2col matrix dimensions:** You apply an unpadded $3 \times 3$ single-channel kernel to a $10 \times 10$ image via `im2col`. What is the output spatial shape, and what are the exact dimensions of the `patches` matrix and flattened kernel vector?
<!--tr:281c7c78-->
1. **im2col の行列の次元：**パディングなしの $3 \times 3$ の単一チャネルカーネルを、`im2col` を使って $10 \times 10$ の画像に適用します。出力の空間形状はどうなり、`patches` 行列と平坦化したカーネルベクトルの正確な次元はいくつですか。

<!--en:16976d18 kind=slide_text-->
2. **Depth to span Fashion-MNIST:** At stride 1, how many stacked $3 \times 3$ layers are required so a single unit's theoretical receptive field spans an entire $28 \times 28$ image? How many with $5 \times 5$ layers?
<!--tr:16976d18-->
2. **Fashion-MNIST を覆う深さ：**ストライド 1 のとき、$3 \times 3$ の層を何層積み重ねると、1 つのユニットの理論上の受容野が $28 \times 28$ の画像全体に及びますか。$5 \times 5$ の層では何層ですか。

<!--en:ab2e0d6a kind=slide_text-->
3. **Edge-loss boundary shift:** In our $6 \times 8$ vertical-edge learning experiment, why did the loss start at $\approx 12$, and what would the initial loss be if the black band extended all the way to the right edge (`X[:, 2:8] = 0`)?
<!--tr:ab2e0d6a-->
3. **エッジ損失と境界のずれ：**$6 \times 8$ の垂直エッジ学習の実験で、損失が $\approx 12$ から始まったのはなぜですか。また、黒い帯が右端まで広がっていたら（`X[:, 2:8] = 0`）、初期損失はいくらになりますか。

<!--en:250d6877 kind=slide_text-->
**Answers:**
(1) Output shape is $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ outputs). The `patches` matrix has shape $(64, 9)$ and multiplies the $(9, 1)$ kernel vector.
(2) For $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ layers. For $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ layers.
(3) Each of the 6 rows had two $\pm 1$ edges ($6 \times 2 \times 1^2 = 12$). Extending the black band to the right border eliminates the right-hand black-to-white edge, leaving one edge per row and halving the initial loss to $\approx 6$.
<!--tr:250d6877-->
**解答：**
(1) 出力の形状は $(10-3+1) \times (10-3+1) = 8 \times 8$（出力は $64$ 個）。`patches` 行列の形状は $(64, 9)$ で、$(9, 1)$ のカーネルベクトルに掛ける。
(2) $3 \times 3$ の場合は $2L + 1 \ge 28 \implies L = 14$ 層。$5 \times 5$ の場合は $4L + 1 \ge 28 \implies L = 7$ 層。
(3) 6 行のそれぞれに $\pm 1$ のエッジが 2 つあった（$6 \times 2 \times 1^2 = 12$）。黒い帯を右端まで広げると、右側の黒から白へのエッジがなくなり、各行のエッジが 1 つになって、初期損失は半分の $\approx 6$ になる。

<!--en:ba43a2dd kind=slide_title-->
::: {.slide title="Summary & Key Takeaways"}
<!--tr:ba43a2dd-->
::: {.slide title="まとめと重要ポイント"}

<!--en:ce12e85f kind=kicker-->
[Section Synthesis]{.kicker}
<!--tr:ce12e85f-->
[この節のまとめ]{.kicker}

<!--en:00011fb7 kind=slide_text-->
- **2D Cross-Correlation & Edge Learning:** Slides $\mathbf{K}$ across valid windows ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); SGD learns $[1, -1]$ in 9 steps.
<!--tr:00011fb7-->
- **2 次元相互相関とエッジの学習：**$\mathbf{K}$ を有効なすべてのウィンドウにわたってスライドさせ（$(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$）、SGD が 9 ステップで $[1, -1]$ を学習する

<!--en:09efe4f0 kind=slide_text-->
- **im2col Matrix Multiply:** Trades $k_\textrm{h} k_\textrm{w}\times$ patch duplication for peak GPU GEMM throughput.
<!--tr:09efe4f0-->
- **im2col による行列積：**パッチの $k_\textrm{h} k_\textrm{w}\times$ の複製と引き換えに、GPU の GEMM のピークスループットが得られる

<!--en:3af8acd0 kind=slide_text-->
- **Stacked Receptive Fields:** $r = 1 + \sum (k_i - 1) = 2L+1$ for $3 \times 3$ stacks, with Gaussian center sensitivity.
<!--tr:3af8acd0-->
- **積み重ねた受容野：**$r = 1 + \sum (k_i - 1) = 2L+1$ となり（$3 \times 3$ の積み重ねの場合）、中心の感度はガウス型になる

=== DECISIONS ===
- "feature map" → 特徴マップ（glossed once on this page）; "effective receptive field" → 有効受容野; "theoretical receptive field" → 理論上の受容野
- "im2col" stays in English; "image to columns" is glossed as 「image to columns」（ここでは行）
- "flatten" → 平坦化する; "unroll" → 展開する; "patch matrix" → パッチ行列; "banded matrix" → 帯行列; "workspace" → 作業用メモリ / ワークスペース
- "nonlinearity" → 非線形性 (glossary); "nonlinear activation" (slides) → 非線形活性化; "dot product" → ドット積; "bounding box" → バウンディングボックス
- "transposed convolution" → 転置畳み込み; "backward pass" → 逆伝播; "Toeplitz" → トプリッツ
- "RF" (slides) stays RF

=== NOTES ===
- 250d6877: the slide answers are written in plain form (常体), as slide text; sentences end in 。 because the block has several sentences.
- 01c175bf: the colon before the display equation is dropped (no sentence ending in a bullet).
- 33bb377d: the emphasis *少なくとも 1 本* moved inside the sentence so that it does not touch the closing bold marker.
- Warnings on 'CNNs' (92688a0c) and 'SGD' (00011fb7): the acronyms stay in English, as in style.md §3.1/§3.2.
- Warning on 'Tensor' (1fb18ec4): false match, it is part of the name "Tensor Core".
