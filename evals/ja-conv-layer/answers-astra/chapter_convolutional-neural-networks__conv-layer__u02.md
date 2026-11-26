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
`corr2d` の二重ループでは数式との対応がよくわかりますが、Python で出力を 1 ピクセルずつ計算していては、実用には遅すぎます。古くから使われ、今でもライブラリが選択肢の 1 つとしているのが、相互相関を 1 回の大きな行列積に置き換える方法です。行列積はまさに GPU が最大の速度を発揮する演算です。:numref:`fig_correlation` の各出力要素は、$k_\textrm{h} k_\textrm{w}$ 個の要素を持つカーネルと、それに対応する入力の窓をそれぞれ 1 次元に並べたベクトルどうしのドット積（dot product）です。$(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ 個の窓をすべて取り出し、それぞれを 1 行として積み重ねると、パッチ行列（patch matrix）ができます。これに 1 次元に並べたカーネルを掛ければ、相互相関全体を一度に計算できます。この標準的なデータの並べ替えを *im2col* と呼びます。名前の由来は「画像を列に変換する」（image to columns）ですが、ここでは行に並べています。:numref:`fig_conv_im2col` は、この操作を :numref:`fig_correlation` の例に適用したものです。

<!--en:107554a5 kind=caption-->
![The cross-correlation of :numref:`fig_correlation` as one matrix product. Each of the four $2 \times 2$ windows of the input becomes a row of the patch matrix, and multiplying by the flattened kernel gives the flattened output. The shaded cells follow the first window to its row and to the output 19. The center pixel 4 lies in all four windows, so it appears in every row.](../img/conv-im2col.svg)
<!--tr:107554a5-->
![:numref:`fig_correlation` の相互相関を 1 回の行列積で計算する。入力から取り出した 4 つの $2 \times 2$ の窓がそれぞれパッチ行列の 1 行になり、これに 1 次元に並べたカーネルを掛けると、1 次元に並んだ出力が得られる。色付きのセルをたどると、最初の窓がどの行に入り、どのように出力 19 につながるかがわかる。中央のピクセル 4 は 4 つの窓すべてに含まれるため、どの行にも現れる。](../img/conv-im2col.svg)

<!--en:76713479 kind=paragraph-->
The trade-off is extra workspace memory, since each interior pixel is duplicated
into up to $k_\textrm{h} k_\textrm{w}$ patches. Below we verify on
:numref:`fig_correlation` that this single matrix multiply matches `corr2d`
exactly:
<!--tr:76713479-->
その代わり、作業用のメモリが余分に必要です。内部の各ピクセルが、最大 $k_\textrm{h} k_\textrm{w}$ 個のパッチにコピーされるからです。以下では :numref:`fig_correlation` の例を使い、この 1 回の行列積で `corr2d` とまったく同じ結果が得られることを確かめます。

<!--en:3bcd7d8a kind=paragraph-->
The discrepancy is identically zero: one matrix product reproduces `corr2d`
perfectly.
<!--tr:3bcd7d8a-->
差はすべての要素で 0 です。1 回の行列積で `corr2d` の結果を正確に再現できました。

<!--en:c30f2dea kind=paragraph-->
Equivalently, we can unroll the *kernel* instead of the input into a sparse
banded matrix $\mathbf{W}$ multiplying the flattened image—the exact 2D analog
of panel (c) in :numref:`fig_why_conv_weight_sharing`, where every row repeats
the same handful of kernel weights. Seeing a convolution as a structured dense
layer brings us full circle to the derivation of :numref:`sec_why-conv` (and the
exercises build $\mathbf{W}$ explicitly).
<!--tr:c30f2dea-->
同じ計算を別の形で表せます。入力ではなく*カーネル*を展開して疎な帯行列（banded matrix）$\mathbf{W}$ を作り、これを 1 次元に並べた画像に掛けるのです。これは :numref:`fig_why_conv_weight_sharing` の (c) をそのまま 2 次元に拡張したもので、どの行にも同じ少数のカーネルの重みが現れます。畳み込みをこのように構造を持つ全結合層と捉えると、出発点だった :numref:`sec_why-conv` の導出に戻ります。演習問題では $\mathbf{W}$ を実際に組み立てます。

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
1 つの畳み込み層が参照できるのはカーネルと同じ大きさのパッチだけですが、物体の全体像を認識するには、もっと広い範囲の情報を統合する必要があります。:numref:`subsec_why-conv-channels` で見たように、畳み込み層の出力を*特徴マップ*（feature map）と呼びます。先ほどの縦のエッジマップ `Y` もその例です。特徴マップのある要素 $x$ に対し、その*受容野*とは、入力画像（またはそれより前の層）のうち、順伝播で値が $x$ に影響しうる領域のことです。これは :numref:`field_visual` で見た神経生理学の用語に由来します。

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
:numref:`fig_correlation` で色を付けた出力 `19` の受容野は $2 \times 2$ で、$\mathbf{X}$ の左上の 4 ピクセルを覆っています。ここで、*2 層目*として $2 \times 2$ の畳み込み層を $2 \times 2$ の出力 $\mathbf{Y}$ に重ね、1 つのスカラー $z$ を得ます。$z$ は $\mathbf{Y}$ の 4 要素すべてに依存し、それらを合わせると $\mathbf{X}$ の 9 要素すべてに依存します。このように、層を重ねると、深さとともに受容野が徐々に広がります。

<!--en:f95c9394 kind=paragraph-->
How quickly does it grow? Each $k \times k$ convolutional layer reaches $k - 1$
pixels farther than a single element of its input, so stacking $L$ layers with
kernel sizes $k_1, \ldots, k_L$ yields a receptive field of side length
<!--tr:f95c9394-->
受容野はどのくらいの速さで広がるのでしょうか。$k \times k$ の畳み込み層では、入力の 1 要素に比べて受容野の各辺が $k - 1$ ピクセル長くなります。したがって、$L$ 個の層を重ね、そのカーネルサイズを順に $k_1, \ldots, k_L$ とすると、受容野の 1 辺の長さは次の式で表せます。

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
$L$ 個の $3 \times 3$ の層を続けて重ねると、各層で 1 辺が 2 ピクセルずつ伸び、受容野は入力の $(2L + 1) \times (2L + 1)$ の領域を覆います。2 層なら $5 \times 5$、3 層なら $7 \times 7$ です（:numref:`fig_conv_receptive_field`）。受容野が深さに比例して広がることには利点があり、大きなカーネルに頼る必要はありません。$3 \times 3$ の層を 3 つ重ねると受容野は $7 \times 7$ になり、$7 \times 7$ の層を 1 つ使う場合と同じです。しかも、必要なパラメータははるかに少なく（入出力チャネルの組ごとの重みは $3 \times 9 = 27$ 個で、1 層の場合は 49 個）、非線形変換を 1 回ではなく 3 回挟めます。

<!--en:39917513 kind=caption-->
![Receptive field of one unit after three one-dimensional layers with kernel size 3. Shaded cells are the units that the top unit depends on, and the lines connect each of them to its inputs. Each layer widens the receptive field by 2, to 7 inputs.](../img/conv-receptive-field.svg)
<!--tr:39917513-->
![カーネルサイズ 3 の 1 次元の層を 3 つ重ねたときの 1 ユニットの受容野。色付きのセルは最上段のユニットが依存するユニットを表し、線は各ユニットとその入力を結ぶ。1 層ごとに受容野が 2 ずつ広がり、最終的に 7 個の入力を覆う。](../img/conv-receptive-field.svg)

<!--en:48e4c9cc kind=heading-->
### Effective Receptive Fields
<!--tr:48e4c9cc-->
### 実効受容野

<!--en:77d044db kind=paragraph-->
Keep in mind that $r$ measures the *theoretical* receptive field (every input
pixel with *any* path to an output unit) without saying how strongly each pixel
actually contributes. Measuring input gradients directly,
:citet:`Luo.Li.Urtasun.ea.2016` showed that the *effective* receptive field
decays sharply away from the center in a roughly Gaussian bell curve, in both
untrained and trained networks. Deep units can theoretically see far across the
image, but their sensitivity concentrates heavily near the center.
<!--tr:77d044db-->
$r$ が表すのは*理論上の*受容野（theoretical receptive field）です。ある出力ユニットへの経路が*どれか 1 つでも*ある入力ピクセルはすべて含まれますが、各ピクセルが実際にどの程度寄与するかはわかりません。入力に関する勾配（gradient）を直接測定した :citet:`Luo.Li.Urtasun.ea.2016` は、訓練前のネットワークでも訓練後のネットワークでも、*実効*受容野（effective receptive field）では中心から離れるほど感度が急速に低下し、その分布がおおよそガウス分布のような釣鐘形になることを示しました。深い層のユニットは理論上は画像の遠くまで参照できますが、感度の高い領域は中心付近に集中しています。

<!--en:eed4e2a0 kind=paragraph-->
Counting paths explains the bell shape. An input at the edge of the receptive
field reaches the output unit along a single chain of connections, whereas the
center input can take many routes through the layers in between.
:numref:`fig_conv_rf_paths` counts them for the tower of
:numref:`fig_conv_receptive_field`: the center input has seven paths to the top
unit and each edge input has one. If all weights have a similar magnitude, the
influence of an input grows with its number of paths.
<!--tr:eed4e2a0-->
釣鐘形になる理由は、経路の数を調べるとわかります。受容野の端の入力から出力ユニットに至る経路は 1 本だけですが、中央の入力からは途中の層を経由してさまざまな経路をたどれます。:numref:`fig_conv_rf_paths` は、:numref:`fig_conv_receptive_field` のように層を重ねた場合の経路数です。中央の入力から最上段のユニットに至る経路は 7 本あり、両端の入力からはそれぞれ 1 本です。すべての重みの大きさが同程度なら、経路が多い入力ほど影響力が大きくなります。

<!--en:b92775ee kind=caption-->
![Number of paths from each unit to the top unit of :numref:`fig_conv_receptive_field` (three one-dimensional layers with kernel size 3). Each count is the sum of the counts of the units that the cell feeds. Darker cells have more paths than the other cells of their layer, and the bars repeat the counts of the input row: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)
<!--tr:b92775ee-->
![各ユニットから最上段のユニットに至る経路の数。構成は :numref:`fig_conv_receptive_field` と同じ（カーネルサイズ 3 の 1 次元の層が 3 つ）。各セルの経路数は、そのセルの出力を受け取る各ユニットの経路数を足したもの。同じ層では経路が多いセルほど色が濃い。棒グラフは入力の行の経路数 1、3、6、7、6、3、1 を再掲したもの。](../img/conv-rf-paths.svg)

<!--en:405695e8 kind=paragraph-->
Two practical gaps remain in our basic layer: every unpadded convolution shrinks
spatial dimensions while under-sampling border pixels, and a single kernel
detects only one pattern. We address spatial sizing and borders next in
:numref:`sec_padding`, and multi-channel feature maps in :numref:`sec_channels`.
<!--tr:405695e8-->
この基本的な層には、実用上 2 つの課題が残ります。パディングなしの畳み込みを重ねるたびに空間方向のサイズが小さくなり、境界付近のピクセルを拾う回数も少なくなります。また、1 つのカーネルで検出できるのは 1 種類のパターンだけです。次の :numref:`sec_padding` では空間方向のサイズと境界の扱いを、:numref:`sec_channels` では複数のチャネルを持つ特徴マップを取り上げます。

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
畳み込み層は、学習したカーネルを入力内に収まるすべての位置に順に当てて 2 次元相互相関を計算し、スカラーのバイアスを加えます。単純な $1 \times 2$ の有限差分カーネル `[1, -1]` でも、画像内のどこにある縦のエッジも検出できます。勾配降下法は入出力データだけからまさにこの重みを復元します。同じように、深い CNN はフィルタの階層全体をエンドツーエンドで学習します。数学的な畳み込みではカーネルを反転しますが、カーネル自体を学習する場合、この違いは問題になりません。畳み込みは 1 回の行列積（*im2col*）で計算できます。また、$L$ 個の畳み込み層を重ねると、各ユニットの受容野は 1 層につき $k_i - 1$ ずつ線形に広がります。

<!--en:3ba929d2 kind=slide_title-->
::: {.slide title="Fast Execution: Convolution via Matrix Multiplication"}
<!--tr:3ba929d2-->
::: {.slide title="行列積による畳み込みの高速化"}

<!--en:1711f812 kind=kicker-->
[The im2col (Image-to-Columns) Transformation]{.kicker}
<!--tr:1711f812-->
[im2col による画像から列への変換]{.kicker}

<!--en:a040f3d7 kind=slide_text-->
- **Single dense GEMM:** Flatten each $k_\textrm{h} \times k_\textrm{w}$ window into a row of `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ and multiply by $\textrm{vec}(\mathbf{K})$.
<!--tr:a040f3d7-->
- **1 回の密行列積（GEMM）：**各 $k_\textrm{h} \times k_\textrm{w}$ の窓を 1 次元に並べて `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ の 1 行とし、この行列に $\textrm{vec}(\mathbf{K})$ を掛ける

<!--en:1fb18ec4 kind=slide_text-->
- **Trade-off:** Peak GPU Tensor Core speed at the cost of $k_\textrm{h} k_\textrm{w}\times$ patch workspace duplication.
<!--tr:1fb18ec4-->
- **トレードオフ：**パッチを作業領域に $k_\textrm{h} k_\textrm{w}\times$ 複製するコストと引き換えに、GPU の Tensor Core の最高速度を引き出す

<!--en:0d757cf6 kind=slide_title-->
::: {.slide title="Dual View: Unrolling the Kernel into a Banded Matrix"}
<!--tr:0d757cf6-->
::: {.slide title="もう 1 つの見方：カーネルの帯行列への展開"}

<!--en:7f0defaa kind=kicker-->
[Toeplitz Weight Matrix Multiplication]{.kicker}
<!--tr:7f0defaa-->
[テプリッツ構造を持つ重み行列との積]{.kicker}

<!--en:5898ba57 kind=slide_text-->
Instead of unfolding the input image into duplicated patches, we can unroll the *kernel* into a sparse banded matrix $\mathbf{W}$ that multiplies the flattened input vector $\textrm{vec}(\mathbf{X})$:
<!--tr:5898ba57-->
入力画像を展開して重複するパッチを作る代わりに、*カーネル*を疎な帯行列 $\mathbf{W}$ に展開し、1 次元に並べた入力ベクトル $\textrm{vec}(\mathbf{X})$ に掛けることもできる。

<!--en:dae51741 kind=slide_text-->
- **Matrix–vector form:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, where $\mathbf{W}$ has $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ rows and $n_\textrm{h} n_\textrm{w}$ columns.
<!--tr:dae51741-->
- **行列とベクトルの積：**$\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$ と書け、$\mathbf{W}$ は $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ 行 $n_\textrm{h} n_\textrm{w}$ 列

<!--en:f0e86ee3 kind=slide_text-->
- **Connection to §{sec_why-conv} and backprop:** Every row of $\mathbf{W}$ holds the same $k_\textrm{h} k_\textrm{w}$ kernel entries shifted across columns (panel c), making the backward pass with respect to $\mathbf{X}$ simply multiplication by $\mathbf{W}^\top$ (a transposed convolution).
<!--tr:f0e86ee3-->
- **§{sec_why-conv} および逆伝播との関係：**$\mathbf{W}$ の各行には、同じ $k_\textrm{h} k_\textrm{w}$ 個のカーネル要素が列をずらして並ぶ（図の (c)）。そのため $\mathbf{X}$ に関する逆伝播は、単に $\mathbf{W}^\top$ を掛ける操作（転置畳み込み）になる

<!--en:af8cf19a kind=slide_title-->
::: {.slide title="Feature Maps and Receptive Fields Across Depth"}
<!--tr:af8cf19a-->
::: {.slide title="深さに伴う特徴マップと受容野の変化"}

<!--en:4eb965f3 kind=kicker-->
[Linear Receptive-Field Growth in Stacked Layers]{.kicker}
<!--tr:4eb965f3-->
[層を重ねると線形に広がる受容野]{.kicker}

<!--en:5599a6e9 kind=slide_text-->
A unit's **receptive field** is the region of the original input image whose pixels can influence that unit's activation during forward propagation:
<!--tr:5599a6e9-->
あるユニットの**受容野**は、元の入力画像のうち、順伝播でそのユニットの活性化値に影響しうるピクセルの領域。

<!--en:eba74f75 kind=slide_text-->
- **Single layer ($L=1$):** A $k_1 \times k_1$ kernel sees a $k_1 \times k_1$ input patch.
<!--tr:eba74f75-->
- **1 層の場合（$L=1$）：**$k_1 \times k_1$ のカーネルが参照する入力パッチは $k_1 \times k_1$

<!--en:01c175bf kind=slide_text-->
- **Stacking $L$ stride-1 layers:** Layer $i$ with kernel size $k_i$ reaches $k_i - 1$ pixels farther along each axis than a single element of layer $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$
<!--tr:01c175bf-->
- **ストライド 1 の層を $L$ 個重ねた場合：**第 $i$ 層のカーネルサイズが $k_i$ なら、受容野は各軸に沿って $k_i - 1$ ピクセルだけ第 $i-1$ 層の 1 要素の受容野より広がる
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$

<!--en:2bde5822 kind=slide_text-->
- **$L$ stacked $3 \times 3$ layers:** Each layer adds $3 - 1 = 2$ pixels, giving side length $r = 2L + 1$ ($3 \to 5 \to 7$).
<!--tr:2bde5822-->
- **$L$ 個の $3 \times 3$ の層を重ねた場合：**各層で $3 - 1 = 2$ ピクセルずつ広がり、1 辺の長さは $r = 2L + 1$（$3 \to 5 \to 7$）

<!--en:cbb9e97a kind=slide_title-->
::: {.slide title="Why Deep Stacks of 3×3 Kernels Beat One Large Kernel"}
<!--tr:cbb9e97a-->
::: {.slide title="3×3 カーネルを重ねると大きなカーネル 1 つより有利な理由"}

<!--en:95e46e70 kind=kicker-->
[Parameter Efficiency & Nonlinear Expressivity]{.kicker}
<!--tr:95e46e70-->
[パラメータ効率と非線形な表現力]{.kicker}

<!--en:bd1d7cc0 kind=slide_text-->
- **Matched $7 \times 7$ receptive field:** Three stacked $3 \times 3$ layers ($r = 1 + 2 + 2 + 2 = 7$) see the exact same $7 \times 7$ patch as one $7 \times 7$ layer.
<!--tr:bd1d7cc0-->
- **同じ $7 \times 7$ の受容野：**$3 \times 3$ の層を 3 つ重ねると（$r = 1 + 2 + 2 + 2 = 7$）、参照する $7 \times 7$ のパッチは $7 \times 7$ の層 1 つの場合とまったく同じ

<!--en:755ac11a kind=slide_text-->
- **45% fewer parameters:** $3 \times 9 c^2 = 27 c^2$ weights instead of $49 c^2$ across $c$ channels.
<!--tr:755ac11a-->
- **パラメータを 45% 削減：**必要な重みは $3 \times 9 c^2 = 27 c^2$ 個。大きなカーネルでは $49 c^2$ 個（いずれも $c$ チャネルの場合）

<!--en:e3aa1086 kind=slide_text-->
- **Richer expressivity:** Compounds **3 nonlinear activations** instead of 1 (the core VGG design rule in §{sec_blocks}).
<!--tr:e3aa1086-->
- **表現力が向上：**非線形活性化を 1 回適用する代わりに、**3 回の非線形活性化**を合成（§{sec_blocks} の VGG の基本設計原則）

<!--en:9ec83bf6 kind=slide_title-->
::: {.slide title="Theoretical vs. Effective Receptive Fields"}
<!--tr:9ec83bf6-->
::: {.slide title="理論上の受容野と実効受容野"}

<!--en:72a52c3b kind=kicker-->
[Gaussian Decay of Sensitivity Away from the Center]{.kicker}
<!--tr:72a52c3b-->
[中心から離れるにつれてガウス分布状に低下する感度]{.kicker}

<!--en:33bb377d kind=slide_text-->
- **Theoretical RF ($r = 2L+1$):** Bounding box of all pixels with *at least one* path to the output unit.
<!--tr:33bb377d-->
- **理論上の受容野（$r = 2L+1$）：**出力ユニットへの経路が*少なくとも 1 本*ある全ピクセルを囲むバウンディングボックス

<!--en:293436eb kind=slide_text-->
- **Central Limit Theorem:** Center pixels connect via exponentially many paths, whereas corner pixels connect via only **1 path**.
<!--tr:293436eb-->
- **中心極限定理：**中央のピクセルからの経路数は深さとともに指数関数的に増えるが、隅のピクセルからの経路は **1 本**だけ

<!--en:f47d181e kind=slide_text-->
- **Effective RF (Luo et al., 2016):** Input gradient magnitude $\partial y / \partial \mathbf{X}$ decays in a Gaussian bell curve with std $\sigma \propto \sqrt{L}$.
<!--tr:f47d181e-->
- **実効受容野（Luo ほか、2016 年）：**入力に関する勾配の大きさ $\partial y / \partial \mathbf{X}$ は、標準偏差 $\sigma \propto \sqrt{L}$ のガウス分布のような釣鐘形を描いて減衰

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
出力の形状、`im2col`、受容野の計算を確認しよう。

<!--en:281c7c78 kind=slide_text-->
1. **im2col matrix dimensions:** You apply an unpadded $3 \times 3$ single-channel kernel to a $10 \times 10$ image via `im2col`. What is the output spatial shape, and what are the exact dimensions of the `patches` matrix and flattened kernel vector?
<!--tr:281c7c78-->
1. **im2col の行列の次元：**パディングなしで $3 \times 3$ の単一チャネルのカーネルを $10 \times 10$ の画像に `im2col` で適用する。出力の空間形状はどうなるか。`patches` 行列と 1 次元に並べたカーネルベクトルの次元はいくつか

<!--en:16976d18 kind=slide_text-->
2. **Depth to span Fashion-MNIST:** At stride 1, how many stacked $3 \times 3$ layers are required so a single unit's theoretical receptive field spans an entire $28 \times 28$ image? How many with $5 \times 5$ layers?
<!--tr:16976d18-->
2. **Fashion-MNIST の画像全体を覆う深さ：**ストライドが 1 のとき、$3 \times 3$ の層をいくつ重ねれば 1 ユニットの理論上の受容野が $28 \times 28$ の画像全体を覆うか。$5 \times 5$ の層ならいくつか

<!--en:ab2e0d6a kind=slide_text-->
3. **Edge-loss boundary shift:** In our $6 \times 8$ vertical-edge learning experiment, why did the loss start at $\approx 12$, and what would the initial loss be if the black band extended all the way to the right edge (`X[:, 2:8] = 0`)?
<!--tr:ab2e0d6a-->
3. **境界の移動とエッジの損失：**$6 \times 8$ の縦エッジの学習実験で初期損失が $\approx 12$ になったのはなぜか。黒い帯を右端まで延ばすと（`X[:, 2:8] = 0`）、初期損失はいくつになるか

<!--en:250d6877 kind=slide_text-->
**Answers:**
(1) Output shape is $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ outputs). The `patches` matrix has shape $(64, 9)$ and multiplies the $(9, 1)$ kernel vector.
(2) For $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ layers. For $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ layers.
(3) Each of the 6 rows had two $\pm 1$ edges ($6 \times 2 \times 1^2 = 12$). Extending the black band to the right border eliminates the right-hand black-to-white edge, leaving one edge per row and halving the initial loss to $\approx 6$.
<!--tr:250d6877-->
**解答：**
(1) 出力の形状は $(10-3+1) \times (10-3+1) = 8 \times 8$（出力は $64$ 個）。`patches` 行列の形状は $(64, 9)$ で、これに $(9, 1)$ のカーネルベクトルを掛ける。
(2) $3 \times 3$ の場合は $2L + 1 \ge 28 \implies L = 14$ 層、$5 \times 5$ の場合は $4L + 1 \ge 28 \implies L = 7$ 層。
(3) 全 6 行のそれぞれに値が $\pm 1$ となるエッジが 2 か所あった（$6 \times 2 \times 1^2 = 12$）。黒い帯を右端まで延ばすと右側の黒から白へのエッジがなくなり、各行に残るエッジは 1 つなので、初期損失は半分の $\approx 6$ になる。

<!--en:ba43a2dd kind=slide_title-->
::: {.slide title="Summary & Key Takeaways"}
<!--tr:ba43a2dd-->
::: {.slide title="まとめと要点"}

<!--en:ce12e85f kind=kicker-->
[Section Synthesis]{.kicker}
<!--tr:ce12e85f-->
[本節のまとめ]{.kicker}

<!--en:00011fb7 kind=slide_text-->
- **2D Cross-Correlation & Edge Learning:** Slides $\mathbf{K}$ across valid windows ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); SGD learns $[1, -1]$ in 9 steps.
<!--tr:00011fb7-->
- **2 次元相互相関とエッジの学習：**$\mathbf{K}$ を入力内に収まる窓の位置（$(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$）に沿って動かす。SGD は 9 ステップで $[1, -1]$ を学習

<!--en:09efe4f0 kind=slide_text-->
- **im2col Matrix Multiply:** Trades $k_\textrm{h} k_\textrm{w}\times$ patch duplication for peak GPU GEMM throughput.
<!--tr:09efe4f0-->
- **im2col による行列積：**パッチが $k_\textrm{h} k_\textrm{w}\times$ 重複する代わりに、GPU の GEMM で最大のスループットを引き出す

<!--en:3af8acd0 kind=slide_text-->
- **Stacked Receptive Fields:** $r = 1 + \sum (k_i - 1) = 2L+1$ for $3 \times 3$ stacks, with Gaussian center sensitivity.
<!--tr:3af8acd0-->
- **層を重ねたときの受容野：**$r = 1 + \sum (k_i - 1) = 2L+1$（$3 \times 3$ の層を重ねた場合）。感度は中心を頂点とするガウス分布状になる

=== DECISIONS ===
- “patch matrix” → 「パッチ行列」, “banded matrix” → 「帯行列」, “effective receptive field” → 「実効受容野」.

=== NOTES ===
- 92688a0c, 00011fb7: the glossary checker looks for the expanded names. CNN and SGD follow the style guide for acronyms.
- 1fb18ec4: Tensor Core is a hardware name, not an instance of the glossary term tensor.
