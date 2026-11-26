<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u02 lang=ja rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:0b85caf9 kind=heading-->
## Convolution as Matrix Multiplication
<!--tr:0b85caf9-->
## 行列積による畳み込み

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
`corr2d` の 二重ループなら計算の仕組みは明快ですが、Python で出力ピクセルを 1 つずつ処理する方法は、実際の処理には遅すぎます。古くから使われ、今もライブラリが選ぶアルゴリズムの候補に入っている解決策は、相互相関を 1 回の大きな行列積に置き換えることです。行列積は、GPU が最高速度で実行できる演算です。:numref:`fig_correlation` の各出力要素は、$k_\textrm{h} k_\textrm{w}$ 個の要素を持つカーネルと、対応する入力ウィンドウをそれぞれ 1 次元に並べたベクトルどうしのドット積にすぎません。$(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ 個すべてのウィンドウを取り出し、ウィンドウごとに 1 行ずつ並べると、パッチ行列が得られます。この行列に 1 次元に並べたカーネルを掛ければ、相互相関の出力を一度に計算できます。この標準的なデータ配置の変換を *im2col*（image to columns、つまり「画像を列に」。ここでは列の代わりに行に並べます）と呼びます。:numref:`fig_conv_im2col` では、:numref:`fig_correlation` の例にこの変換を適用しています。

<!--en:107554a5 kind=caption-->
![The cross-correlation of :numref:`fig_correlation` as one matrix product. Each of the four $2 \times 2$ windows of the input becomes a row of the patch matrix, and multiplying by the flattened kernel gives the flattened output. The shaded cells follow the first window to its row and to the output 19. The center pixel 4 lies in all four windows, so it appears in every row.](../img/conv-im2col.svg)
<!--tr:107554a5-->
![:numref:`fig_correlation` の相互相関を 1 回の行列積で計算。入力から切り出した 4 つの $2 \times 2$ ウィンドウがそれぞれパッチ行列の 1 行になり、1 次元に並べたカーネルを掛けると、出力を 1 次元に並べたものが得られる。網掛けは、最初のウィンドウが行列の 1 行になり、出力 19 に対応することを示す。中央のピクセル 4 は 4 つのウィンドウすべてに含まれるため、どの行にも現れる。](../img/conv-im2col.svg)

<!--en:76713479 kind=paragraph-->
The trade-off is extra workspace memory, since each interior pixel is duplicated
into up to $k_\textrm{h} k_\textrm{w}$ patches. Below we verify on
:numref:`fig_correlation` that this single matrix multiply matches `corr2d`
exactly:
<!--tr:76713479-->
その代わり、入力内部の各ピクセルを最大 $k_\textrm{h} k_\textrm{w}$ 個のパッチに複製するため、作業用メモリが余分に必要です。以下では :numref:`fig_correlation` の例で、この 1 回の行列積の結果が `corr2d` と完全に一致することを確かめます。

<!--en:3bcd7d8a kind=paragraph-->
The discrepancy is identically zero: one matrix product reproduces `corr2d`
perfectly.
<!--tr:3bcd7d8a-->
差は厳密に 0 です。1 回の行列積で `corr2d` の結果を完全に再現できました。

<!--en:c30f2dea kind=paragraph-->
Equivalently, we can unroll the *kernel* instead of the input into a sparse
banded matrix $\mathbf{W}$ multiplying the flattened image—the exact 2D analog
of panel (c) in :numref:`fig_why_conv_weight_sharing`, where every row repeats
the same handful of kernel weights. Seeing a convolution as a structured dense
layer brings us full circle to the derivation of :numref:`sec_why-conv` (and the
exercises build $\mathbf{W}$ explicitly).
<!--tr:c30f2dea-->
同じ計算は、入力ではなく*カーネル*を疎な帯行列（banded matrix）$\mathbf{W}$ に展開し、1 次元に並べた画像を掛ける形でも表せます。これは :numref:`fig_why_conv_weight_sharing` のパネル (c) をそのまま 2 次元に拡張したもので、各行に同じ少数のカーネルの重みが繰り返し現れます。畳み込みを特定の構造を持つ全結合層として捉え直すと、:numref:`sec_why-conv` での導出に立ち戻ります（演習問題では $\mathbf{W}$ を実際に作ります）。

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
畳み込み層が 1 つだけなら、一度に参照できるのはカーネルと同じ大きさのパッチだけです。一方、物体全体を認識するには、はるかに広い範囲の情報を統合する必要があります。:numref:`subsec_why-conv-channels` で見たように、畳み込み層の出力を*特徴マップ*（feature map）と呼びます。先ほどの縦方向のエッジマップ `Y` がその一例です。特徴マップの任意の要素 $x$ の*受容野*とは、入力画像（または前段の層）のうち、順伝播の際に値が $x$ に影響しうる領域です。この名称は :numref:`field_visual` に出てきた神経生理学の用語から借りています。

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
:numref:`fig_correlation` で網掛けされた出力 `19` の受容野は $2 \times 2$ で、$\mathbf{X}$ の左上 4 ピクセルを覆います。ここで*2 層目*となる $2 \times 2$ の畳み込み層を、$2 \times 2$ の出力 $\mathbf{Y}$ の上に重ね、スカラー $z$ を 1 つ得るとしましょう。$z$ は $\mathbf{Y}$ の 4 要素すべてに依存し、その 4 要素は合わせて $\mathbf{X}$ の 9 要素すべてに依存します。このように、層を重ねて深くするにつれて受容野は着実に広がります。

<!--en:f95c9394 kind=paragraph-->
How quickly does it grow? Each $k \times k$ convolutional layer reaches $k - 1$
pixels farther than a single element of its input, so stacking $L$ layers with
kernel sizes $k_1, \ldots, k_L$ yields a receptive field of side length
<!--tr:f95c9394-->
受容野はどのくらいの速さで広がるのでしょうか。$k \times k$ の畳み込み層を通るたび、入力の 1 要素を基準に、受容野の 1 辺は $k - 1$ ピクセル伸びます。したがって、$L$ 層を重ね、カーネルサイズを順に $k_1, \ldots, k_L$ とすると、受容野の 1 辺の長さは次のようになります。

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
$L$ 層の $3 \times 3$ の畳み込み層を連続して重ねると、受容野の 1 辺は 1 層ごとに 2 ピクセル伸び、$(2L + 1) \times (2L + 1)$ 個の入力を覆います。2 層なら $5 \times 5$、3 層なら $7 \times 7$ です（:numref:`fig_conv_receptive_field`）。このように 1 辺の長さが線形に伸びることには利点があり、大きなカーネルに頼る必要はありません。$3 \times 3$ の層を 3 つ重ねると、$7 \times 7$ の受容野が得られます。これは $7 \times 7$ の層を 1 つ使う場合と同じですが、必要なパラメータは大幅に少なくなります（チャネルの組 1 つあたり、重みは $3 \times 9 = 27$ 個に対して 49 個）。さらに、非線形性（nonlinearity）を導入する処理を 1 回ではなく 3 回挟めます。

<!--en:39917513 kind=caption-->
![Receptive field of one unit after three one-dimensional layers with kernel size 3. Shaded cells are the units that the top unit depends on, and the lines connect each of them to its inputs. Each layer widens the receptive field by 2, to 7 inputs.](../img/conv-receptive-field.svg)
<!--tr:39917513-->
![カーネルサイズが 3 の 1 次元の層を 3 つ重ねたときの、1 ユニットの受容野。網掛けは最上段のユニットが依存するユニットを示し、各ユニットとその入力を線で結んでいる。層を重ねるたびに受容野の幅が 2 ずつ広がり、入力 7 個に達する。](../img/conv-receptive-field.svg)

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
ここで、$r$ が表すのは*理論上*の受容野（theoretical receptive field）です。ある出力ユニットに至る経路が*1 本でも*ある入力ピクセルはすべて含みますが、個々のピクセルが実際にどの程度寄与するかはわかりません。入力に関する勾配を直接測定した :citet:`Luo.Li.Urtasun.ea.2016` は、訓練前と訓練後のどちらのネットワークでも、*有効*受容野（effective receptive field）における感度が中心から離れるにつれて急激に低下し、その分布がおおむねガウス分布（Gaussian distribution）の釣鐘形になることを示しました。深い層のユニットは理論上、画像の広い範囲を参照できますが、感度は中心付近に強く集中します。

<!--en:eed4e2a0 kind=paragraph-->
Counting paths explains the bell shape. An input at the edge of the receptive
field reaches the output unit along a single chain of connections, whereas the
center input can take many routes through the layers in between.
:numref:`fig_conv_rf_paths` counts them for the tower of
:numref:`fig_conv_receptive_field`: the center input has seven paths to the top
unit and each edge input has one. If all weights have a similar magnitude, the
influence of an input grows with its number of paths.
<!--tr:eed4e2a0-->
経路を数えると、なぜ釣鐘形になるのかがわかります。受容野の端の入力から出力ユニットに至る経路は 1 本しかありません。一方、中心の入力からは、間にある層をさまざまな経路で通り抜けられます。:numref:`fig_conv_rf_paths` は :numref:`fig_conv_receptive_field` の層の積み重ねについて経路を数えたものです。中心の入力から最上段のユニットまでは 7 本、両端の入力からはそれぞれ 1 本あります。すべての重みの絶対値が同程度なら、経路の多い入力ほど影響が大きくなります。

<!--en:b92775ee kind=caption-->
![Number of paths from each unit to the top unit of :numref:`fig_conv_receptive_field` (three one-dimensional layers with kernel size 3). Each count is the sum of the counts of the units that the cell feeds. Darker cells have more paths than the other cells of their layer, and the bars repeat the counts of the input row: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)
<!--tr:b92775ee-->
![:numref:`fig_conv_receptive_field`（カーネルサイズが 3 の 1 次元の層を 3 つ重ねたもの）で、各ユニットから最上段のユニットに至る経路の本数。各マスの本数は、そのマスから入力を受け取るユニットの本数の和になる。色の濃いマスほど、同じ層の他のマスよりも経路が多い。棒グラフは入力行の本数（1、3、6、7、6、3、1）を表す。](../img/conv-rf-paths.svg)

<!--en:405695e8 kind=paragraph-->
Two practical gaps remain in our basic layer: every unpadded convolution shrinks
spatial dimensions while under-sampling border pixels, and a single kernel
detects only one pattern. We address spatial sizing and borders next in
:numref:`sec_padding`, and multi-channel feature maps in :numref:`sec_channels`.
<!--tr:405695e8-->
ここまでの基本的な層には、実用上の課題が 2 つ残っています。パディングなしの畳み込みでは、縦横のサイズが縮み、境界のピクセルが使われる回数も少なくなります。さらに、1 つのカーネルでは 1 種類のパターンしか検出できません。次の :numref:`sec_padding` では縦横のサイズと境界の扱いを、:numref:`sec_channels` では複数チャネルの特徴マップを取り上げます。

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
畳み込み層では、学習したカーネルを入力内に収まるすべてのウィンドウにわたって動かして 2 次元の相互相関を計算し、スカラーのバイアスを加えます。単純な $1 \times 2$ の有限差分カーネル `[1, -1]` は、画像内のどこにある縦方向のエッジでも検出します。深い CNN が階層全体のフィルタをエンドツーエンドで学習するように、勾配降下法は入出力データだけからこの重みを復元します。数学的な意味での畳み込みではカーネルを反転しますが、学習するカーネルであれば、反転の有無で表現できる層は変わりません。畳み込みは 1 回の行列積（*im2col*）で計算できます。また、$L$ 個の畳み込み層を重ねると、各ユニットの受容野の 1 辺は層ごとに $k_i - 1$ ずつ、線形に広がります。

<!--en:3ba929d2 kind=slide_title-->
::: {.slide title="Fast Execution: Convolution via Matrix Multiplication"}
<!--tr:3ba929d2-->
::: {.slide title="行列積による畳み込みの高速化"}

<!--en:1711f812 kind=kicker-->
[The im2col (Image-to-Columns) Transformation]{.kicker}
<!--tr:1711f812-->
[画像を列に並べる im2col 変換]{.kicker}

<!--en:a040f3d7 kind=slide_text-->
- **Single dense GEMM:** Flatten each $k_\textrm{h} \times k_\textrm{w}$ window into a row of `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ and multiply by $\textrm{vec}(\mathbf{K})$.
<!--tr:a040f3d7-->
- **1 回の密行列積（GEMM）：**各 $k_\textrm{h} \times k_\textrm{w}$ ウィンドウを 1 次元に並べて `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ の 1 行とし、$\textrm{vec}(\mathbf{K})$ を掛ける

<!--en:1fb18ec4 kind=slide_text-->
- **Trade-off:** Peak GPU Tensor Core speed at the cost of $k_\textrm{h} k_\textrm{w}\times$ patch workspace duplication.
<!--tr:1fb18ec4-->
- **速度とメモリのトレードオフ：**GPU の Tensor Core で最高速度が得られる一方、パッチの複製で作業領域は $k_\textrm{h} k_\textrm{w}\times$ に増加

<!--en:0d757cf6 kind=slide_title-->
::: {.slide title="Dual View: Unrolling the Kernel into a Banded Matrix"}
<!--tr:0d757cf6-->
::: {.slide title="もう一つの見方：カーネルを帯行列に展開"}

<!--en:7f0defaa kind=kicker-->
[Toeplitz Weight Matrix Multiplication]{.kicker}
<!--tr:7f0defaa-->
[テプリッツ型の重み行列との積]{.kicker}

<!--en:5898ba57 kind=slide_text-->
Instead of unfolding the input image into duplicated patches, we can unroll the *kernel* into a sparse banded matrix $\mathbf{W}$ that multiplies the flattened input vector $\textrm{vec}(\mathbf{X})$:
<!--tr:5898ba57-->
入力画像をパッチに分けて複製する代わりに、*カーネル*を疎な帯行列 $\mathbf{W}$ に展開し、入力を 1 次元に並べたベクトル $\textrm{vec}(\mathbf{X})$ を掛けてもよい。

<!--en:dae51741 kind=slide_text-->
- **Matrix–vector form:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, where $\mathbf{W}$ has $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ rows and $n_\textrm{h} n_\textrm{w}$ columns.
<!--tr:dae51741-->
- **行列とベクトルの積：**$\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$。$\mathbf{W}$ は $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ 行、$n_\textrm{h} n_\textrm{w}$ 列

<!--en:f0e86ee3 kind=slide_text-->
- **Connection to §{sec_why-conv} and backprop:** Every row of $\mathbf{W}$ holds the same $k_\textrm{h} k_\textrm{w}$ kernel entries shifted across columns (panel c), making the backward pass with respect to $\mathbf{X}$ simply multiplication by $\mathbf{W}^\top$ (a transposed convolution).
<!--tr:f0e86ee3-->
- **§{sec_why-conv} と誤差逆伝播法との関係：**$\mathbf{W}$ の各行には同じ $k_\textrm{h} k_\textrm{w}$ 個のカーネル要素が、列の位置をずらして入る（パネル (c)）。$\mathbf{X}$ に関する逆伝播は $\mathbf{W}^\top$ を掛けるだけ（転置畳み込み）

<!--en:af8cf19a kind=slide_title-->
::: {.slide title="Feature Maps and Receptive Fields Across Depth"}
<!--tr:af8cf19a-->
::: {.slide title="深さと特徴マップ・受容野"}

<!--en:4eb965f3 kind=kicker-->
[Linear Receptive-Field Growth in Stacked Layers]{.kicker}
<!--tr:4eb965f3-->
[層を重ねると線形に広がる受容野]{.kicker}

<!--en:5599a6e9 kind=slide_text-->
A unit's **receptive field** is the region of the original input image whose pixels can influence that unit's activation during forward propagation:
<!--tr:5599a6e9-->
ユニットの**受容野**とは、順伝播でそのユニットの活性化値に影響しうるピクセルを含む、元の入力画像の領域。

<!--en:eba74f75 kind=slide_text-->
- **Single layer ($L=1$):** A $k_1 \times k_1$ kernel sees a $k_1 \times k_1$ input patch.
<!--tr:eba74f75-->
- **単一の層（$L=1$）：**$k_1 \times k_1$ のカーネルが参照する入力パッチは $k_1 \times k_1$

<!--en:01c175bf kind=slide_text-->
- **Stacking $L$ stride-1 layers:** Layer $i$ with kernel size $k_i$ reaches $k_i - 1$ pixels farther along each axis than a single element of layer $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$
<!--tr:01c175bf-->
- **$L$ 個の層をストライド 1 で重ねる場合：**層 $i$ のカーネルサイズが $k_i$ なら、受容野は各軸で $k_i - 1$ ピクセル分、層 $i-1$ の 1 要素の受容野より広がる
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$

<!--en:2bde5822 kind=slide_text-->
- **$L$ stacked $3 \times 3$ layers:** Each layer adds $3 - 1 = 2$ pixels, giving side length $r = 2L + 1$ ($3 \to 5 \to 7$).
<!--tr:2bde5822-->
- **$L$ 個の $3 \times 3$ の層：**1 層につき $3 - 1 = 2$ ピクセルずつ広がり、1 辺の長さは $r = 2L + 1$（$3 \to 5 \to 7$）

<!--en:cbb9e97a kind=slide_title-->
::: {.slide title="Why Deep Stacks of 3×3 Kernels Beat One Large Kernel"}
<!--tr:cbb9e97a-->
::: {.slide title="大きなカーネル 1 つより 3×3 を重ねる利点"}

<!--en:95e46e70 kind=kicker-->
[Parameter Efficiency & Nonlinear Expressivity]{.kicker}
<!--tr:95e46e70-->
[パラメータ効率と非線形な表現力]{.kicker}

<!--en:bd1d7cc0 kind=slide_text-->
- **Matched $7 \times 7$ receptive field:** Three stacked $3 \times 3$ layers ($r = 1 + 2 + 2 + 2 = 7$) see the exact same $7 \times 7$ patch as one $7 \times 7$ layer.
<!--tr:bd1d7cc0-->
- **同じ $7 \times 7$ の受容野：**$3 \times 3$ の層を 3 つ重ねると（$r = 1 + 2 + 2 + 2 = 7$）、同じ $7 \times 7$ のパッチを $7 \times 7$ の層 1 つと同様に参照できる

<!--en:755ac11a kind=slide_text-->
- **45% fewer parameters:** $3 \times 9 c^2 = 27 c^2$ weights instead of $49 c^2$ across $c$ channels.
<!--tr:755ac11a-->
- **パラメータを 45% 削減：**重みは $3 \times 9 c^2 = 27 c^2$ 個（$49 c^2$ 個から削減、チャネル数は $c$）

<!--en:e3aa1086 kind=slide_text-->
- **Richer expressivity:** Compounds **3 nonlinear activations** instead of 1 (the core VGG design rule in §{sec_blocks}).
<!--tr:e3aa1086-->
- **より豊かな表現力：**1 回の代わりに**非線形な活性化を 3 回**重ねる（§{sec_blocks} の VGG の基本設計）

<!--en:9ec83bf6 kind=slide_title-->
::: {.slide title="Theoretical vs. Effective Receptive Fields"}
<!--tr:9ec83bf6-->
::: {.slide title="理論上の受容野と有効受容野"}

<!--en:72a52c3b kind=kicker-->
[Gaussian Decay of Sensitivity Away from the Center]{.kicker}
<!--tr:72a52c3b-->
[中心から離れるとガウス分布に沿って低下する感度]{.kicker}

<!--en:33bb377d kind=slide_text-->
- **Theoretical RF ($r = 2L+1$):** Bounding box of all pixels with *at least one* path to the output unit.
<!--tr:33bb377d-->
- **理論上の受容野（$r = 2L+1$）：**出力ユニットへ至る経路が*少なくとも 1 本*あるピクセルすべてのバウンディングボックス

<!--en:293436eb kind=slide_text-->
- **Central Limit Theorem:** Center pixels connect via exponentially many paths, whereas corner pixels connect via only **1 path**.
<!--tr:293436eb-->
- **中心極限定理：**中心のピクセルからの経路は指数的に増えるが、隅のピクセルからの経路は**1 本**だけ

<!--en:f47d181e kind=slide_text-->
- **Effective RF (Luo et al., 2016):** Input gradient magnitude $\partial y / \partial \mathbf{X}$ decays in a Gaussian bell curve with std $\sigma \propto \sqrt{L}$.
<!--tr:f47d181e-->
- **有効受容野（Luo ら、2016 年）：**入力に関する勾配 $\partial y / \partial \mathbf{X}$ の大きさは、標準偏差 $\sigma \propto \sqrt{L}$ のガウス分布の釣鐘形に沿って減衰

<!--en:fac792d0 kind=slide_title-->
::: {.slide title="Concept Check: Cross-Correlation & Receptive Fields"}
<!--tr:fac792d0-->
::: {.slide title="理解度チェック：相互相関と受容野"}

<!--en:26a317d8 kind=kicker-->
[Concept Check]{.kicker}
<!--tr:26a317d8-->
[理解度チェック]{.kicker}

<!--en:8fe554e5 kind=slide_text-->
Test your understanding of output shapes, `im2col`, and receptive-field arithmetic:
<!--tr:8fe554e5-->
出力の形状、`im2col`、受容野の計算を確認する。

<!--en:281c7c78 kind=slide_text-->
1. **im2col matrix dimensions:** You apply an unpadded $3 \times 3$ single-channel kernel to a $10 \times 10$ image via `im2col`. What is the output spatial shape, and what are the exact dimensions of the `patches` matrix and flattened kernel vector?
<!--tr:281c7c78-->
1. **im2col の行列の形状：**$3 \times 3$ の単一チャネルのカーネルをパディングなしで $10 \times 10$ の画像に `im2col` で適用する。出力の縦横のサイズはどうなるか。`patches` 行列と、1 次元に並べたカーネルベクトルの形状はどうなるか。

<!--en:16976d18 kind=slide_text-->
2. **Depth to span Fashion-MNIST:** At stride 1, how many stacked $3 \times 3$ layers are required so a single unit's theoretical receptive field spans an entire $28 \times 28$ image? How many with $5 \times 5$ layers?
<!--tr:16976d18-->
2. **Fashion-MNIST の画像全体を覆う深さ：**ストライド 1 で $3 \times 3$ の層をいくつ重ねると、1 ユニットの理論上の受容野が $28 \times 28$ の画像全体を覆うか。$5 \times 5$ の層ならいくつか。

<!--en:ab2e0d6a kind=slide_text-->
3. **Edge-loss boundary shift:** In our $6 \times 8$ vertical-edge learning experiment, why did the loss start at $\approx 12$, and what would the initial loss be if the black band extended all the way to the right edge (`X[:, 2:8] = 0`)?
<!--tr:ab2e0d6a-->
3. **黒い帯の境界を動かしたときの損失：**$6 \times 8$ の画像で縦方向のエッジを学習した実験で、初期損失が $\approx 12$ だったのはなぜか。黒い帯が右端まで伸びていたら（`X[:, 2:8] = 0`）、初期損失はいくつか。

<!--en:250d6877 kind=slide_text-->
**Answers:**
(1) Output shape is $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ outputs). The `patches` matrix has shape $(64, 9)$ and multiplies the $(9, 1)$ kernel vector.
(2) For $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ layers. For $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ layers.
(3) Each of the 6 rows had two $\pm 1$ edges ($6 \times 2 \times 1^2 = 12$). Extending the black band to the right border eliminates the right-hand black-to-white edge, leaving one edge per row and halving the initial loss to $\approx 6$.
<!--tr:250d6877-->
**解答：**
(1) 出力の形状は $(10-3+1) \times (10-3+1) = 8 \times 8$（出力は $64$ 個）。`patches` 行列は形状が $(64, 9)$ で、$(9, 1)$ のカーネルベクトルを掛ける。
(2) $3 \times 3$ の場合は $2L + 1 \ge 28 \implies L = 14$ 層。$5 \times 5$ の場合は $4L + 1 \ge 28 \implies L = 7$ 層。
(3) 6 行それぞれに値が $\pm 1$ のエッジが 2 つあった（$6 \times 2 \times 1^2 = 12$）。帯を右端まで伸ばすと、黒から白に変わる右側のエッジがなくなり、各行にエッジが 1 つだけ残る。したがって初期損失は半分の $\approx 6$ になる。

<!--en:ba43a2dd kind=slide_title-->
::: {.slide title="Summary & Key Takeaways"}
<!--tr:ba43a2dd-->
::: {.slide title="まとめと要点"}

<!--en:ce12e85f kind=kicker-->
[Section Synthesis]{.kicker}
<!--tr:ce12e85f-->
[本節の総括]{.kicker}

<!--en:00011fb7 kind=slide_text-->
- **2D Cross-Correlation & Edge Learning:** Slides $\mathbf{K}$ across valid windows ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); SGD learns $[1, -1]$ in 9 steps.
<!--tr:00011fb7-->
- **2 次元の相互相関とエッジの学習：**$\mathbf{K}$ を入力内に収まるウィンドウ（$(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$）にわたって移動。勾配降下法は $[1, -1]$ を 9 ステップで学習

<!--en:09efe4f0 kind=slide_text-->
- **im2col Matrix Multiply:** Trades $k_\textrm{h} k_\textrm{w}\times$ patch duplication for peak GPU GEMM throughput.
<!--tr:09efe4f0-->
- **im2col による行列積：**パッチを $k_\textrm{h} k_\textrm{w}\times$ に複製し、GPU の GEMM のスループットを最大化

<!--en:3af8acd0 kind=slide_text-->
- **Stacked Receptive Fields:** $r = 1 + \sum (k_i - 1) = 2L+1$ for $3 \times 3$ stacks, with Gaussian center sensitivity.
<!--tr:3af8acd0-->
- **層を重ねたときの受容野：**$r = 1 + \sum (k_i - 1) = 2L+1$（$3 \times 3$ の層を積み重ねた場合）。感度は中心を頂点とするガウス分布に従う

=== DECISIONS ===
- "flatten" → 「1 次元に並べる」 and "unroll" (kernel to matrix) → 「展開する」.
- "patch matrix" → 「パッチ行列」 and "banded matrix" → 「帯行列」.
- "theoretical receptive field" → 「理論上の受容野」 and "effective receptive field" → 「有効受容野」.
=== NOTES ===
- 92688a0c: CNN is the acronym prescribed by the style guide after the full term is introduced in 67d25b0b. The glossary warning is a false positive.
- 1fb18ec4: “Tensor” is part of the hardware name Tensor Core, which the style guide preserves in Latin script. The tensor glossary match is a false positive.
- 00011fb7: As in 456670c5, the code uses ordinary gradient descent on the one fixed example with the loss summed over all output elements. Rendered “SGD” as 勾配降下法 in the recap.
