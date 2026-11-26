<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u01 lang=ja rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:33d3b96b kind=heading-->
# Convolutions for Images
<!--tr:33d3b96b-->
# 画像に対する畳み込み

<!--en:852f5868 kind=paragraph-->
In :numref:`sec_why-conv`, we derived the convolutional layer on paper from two
geometric principles: translation equivariance and locality. Now we see what
that formula actually computes on an image, and whether gradient descent can
discover useful kernels automatically from data.
<!--tr:852f5868-->
:numref:`sec_why-conv` では、平行移動同変性（translation equivariance）と局所性（locality）という 2 つの幾何学的な原理から、畳み込み層を紙の上で導きました。本節では、その式が画像に対して実際に何を計算するのか、そして勾配降下法が有用なカーネルをデータから自動的に見つけられるのかを確かめます。

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
明るい領域と暗い領域が接する垂直エッジの位置を求めることを考えましょう。各ピクセルから右隣のピクセルを引くと（$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$）、一様な領域の内部では 0、白から黒に変わる位置では $+1$、黒から白に戻る位置では $-1$ になります。2 つの重みからなる検出器 $(1, -1)$ を画像上の有効な位置すべてに走らせると、*エッジマップ*が得られます。本節では、2 次元相互相関（two-dimensional cross-correlation）を実装し、手作りのエッジ検出器を試したうえで、勾配降下法に $(1, -1)$ をデータから直接*学習*させます。さらに、畳み込みが 1 回の行列積になること（*im2col*）を示し、層を積み重ねるとユニットの受容野（receptive field）がどのように広がるかを導きます。

<!--en:47958ed8 kind=heading-->
## Cross-Correlation
<!--tr:47958ed8-->
## 相互相関

<!--en:5e4296fe kind=heading-->
### The Operation
<!--tr:5e4296fe-->
### 演算

<!--en:ae7e906d kind=paragraph-->
Leaving channels aside for the moment, let the input $\mathbf{X}$ be an
$n_\textrm{h} \times n_\textrm{w}$ matrix and the kernel $\mathbf{K}$ be a
$k_\textrm{h} \times k_\textrm{w}$ matrix. The kernel defines a sliding
*window* of shape $k_\textrm{h} \times k_\textrm{w}$. Placing the window's
upper-left corner at $(i, j)$, we multiply the covered input patch elementwise
with $\mathbf{K}$ and sum the products to produce the output entry at $(i, j)$:
<!--tr:ae7e906d-->
チャネルのことはひとまず脇に置き、入力 $\mathbf{X}$ を $n_\textrm{h} \times n_\textrm{w}$ の行列、カーネル $\mathbf{K}$ を $k_\textrm{h} \times k_\textrm{w}$ の行列とします。カーネルは、形状 $k_\textrm{h} \times k_\textrm{w}$ のスライドする*ウィンドウ*を定めます。ウィンドウの左上隅を $(i, j)$ に置き、覆われた入力パッチと $\mathbf{K}$ を要素ごとに掛けて、その積の和をとったものが、位置 $(i, j)$ の出力要素です。

<!--en:3688b38b kind=paragraph-->
This is the *two-dimensional cross-correlation* of $\mathbf{X}$ with
$\mathbf{K}$, written $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ with
zero-based indices matching code. Whereas :numref:`sec_why-conv` centered
offsets symmetrically from $-\Delta$ to $\Delta$, anchoring at the upper-left
corner differs only by a fixed shift of $\Delta$ while naturally handling
even-sized kernels (such as a $2 \times 2$ kernel or our $1 \times 2$ edge
detector) that have no center pixel.
<!--tr:3688b38b-->
これが $\mathbf{X}$ と $\mathbf{K}$ の *2 次元相互相関*で、$\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ と書きます。添字は、コードに合わせて 0 始まりです。:numref:`sec_why-conv` ではオフセットを $-\Delta$ から $\Delta$ まで対称にとりましたが、左上隅を基準にしても、違いは $\Delta$ だけの固定のずれにすぎません。しかも、中心ピクセルを持たない偶数サイズのカーネル（$2 \times 2$ のカーネルや、ここでの $1 \times 2$ のエッジ検出器など）も自然に扱えます。

<!--en:fa1b7da9 kind=caption-->
![Two-dimensional cross-correlation. The shaded portions are the first output element and the input and kernel elements used to compute it: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
<!--tr:fa1b7da9-->
![2 次元相互相関。網掛けの部分は、最初の出力要素と、その計算に使われる入力とカーネルの要素である：$0\times0+1\times1+3\times2+4\times3=19$。](../img/correlation.svg)

<!--en:2f8e5deb kind=paragraph-->
In :numref:`fig_correlation`, we slide a $2 \times 2$ kernel across a
$3 \times 3$ input one pixel at a time, left to right and top to bottom. The
window fits cleanly at four positions, yielding the $2 \times 2$ output:
<!--tr:2f8e5deb-->
:numref:`fig_correlation` では、$2 \times 2$ のカーネルを $3 \times 3$ の入力の上で、左から右、上から下へ 1 ピクセルずつずらしていきます。ウィンドウがちょうど収まる位置は 4 つあり、次の $2 \times 2$ の出力が得られます。

<!--en:4741e012 kind=paragraph-->
Notice that the output shrinks slightly along each axis because the kernel must
stay wholly inside the input. Fitting a width-$k$ window into length $n$ leaves
$n - k + 1$ valid positions, so the output shape is
<!--tr:4741e012-->
カーネルは入力の内側に完全に収まっていなければならないため、出力は各軸に沿って少し小さくなります。幅 $k$ のウィンドウを長さ $n$ の中に収めると、有効な位置は $n - k + 1$ 個です。したがって、出力の形状は次のようになります。

<!--en:12d81c14 kind=paragraph-->
Thus a $3 \times 3$ input with a $2 \times 2$ kernel yields $2 \times 2$, a
$6 \times 8$ image with a $1 \times 2$ kernel yields $6 \times 7$, and a
$28 \times 28$ Fashion-MNIST image with a $5 \times 5$ kernel yields
$24 \times 24$. Whenever we want to preserve spatial dimensions, we surround the
input with a border of zeros (*padding*), which we examine alongside strides and
dilation in :numref:`sec_padding`.
<!--tr:12d81c14-->
したがって、$3 \times 3$ の入力と $2 \times 2$ のカーネルからは $2 \times 2$ の出力が、$6 \times 8$ の画像と $1 \times 2$ のカーネルからは $6 \times 7$ の出力が、$28 \times 28$ の Fashion-MNIST 画像と $5 \times 5$ のカーネルからは $24 \times 24$ の出力が得られます。空間方向の大きさを保ちたいときは、入力の周囲を 0 で縁取ります（*パディング*）。パディングは、ストライドやダイレーションとあわせて :numref:`sec_padding` で詳しく見ます。

<!--en:869d223f kind=heading-->
### Implementation
<!--tr:869d223f-->
### 実装

<!--en:6a869f7c kind=paragraph-->
We implement :eqref:`eq_corr2d` directly in `corr2d` using a pair of loops over
every output coordinate.
<!--tr:6a869f7c-->
:eqref:`eq_corr2d` を、`corr2d` としてそのまま実装します。出力のすべての座標を 2 重のループでたどります。

<!--en:81e1f383 kind=paragraph-->
To verify `corr2d` against our hand calculation in :numref:`fig_correlation`,
we construct the $3 \times 3$ input tensor `X` and $2 \times 2$ kernel `K` and
evaluate `corr2d(X, K)`.
<!--tr:81e1f383-->
`corr2d` を :numref:`fig_correlation` の手計算と照らし合わせて検証するために、$3 \times 3$ の入力テンソル `X` と $2 \times 2$ のカーネル `K` を作り、`corr2d(X, K)` を評価します。

<!--en:f567e1b2 kind=paragraph-->
The output reproduces :numref:`fig_correlation` entry for entry (and as shown in
:numref:`sec_why-conv`, strict mathematical convolution with a flipped kernel
$\mathbf{K}'$ learns the exact same family of layers, so we follow standard
deep learning terminology and call cross-correlation "convolution").
<!--tr:f567e1b2-->
出力は :numref:`fig_correlation` と要素ごとに一致します（なお、:numref:`sec_why-conv` で示したように、反転したカーネル $\mathbf{K}'$ を使う厳密な数学的畳み込みでも、学習される層の族はまったく同じです。そこで、深層学習の標準的な用語に従い、相互相関を「畳み込み」と呼びます）。

<!--en:bfe11c62 kind=heading-->
## Convolutional Layers
<!--tr:bfe11c62-->
## 畳み込み層

<!--en:2cdc020d kind=heading-->
### The Layer
<!--tr:2cdc020d-->
### 層の定義

<!--en:c6371ece kind=paragraph-->
A two-dimensional convolutional layer cross-correlates its input with a learned
kernel and adds a scalar bias. Just like our MLP weights in
:numref:`chap_perceptrons`, we initialize the kernel from
$\mathcal{N}(0, 0.01^2)$ and the bias to zero. Below, `Conv2D` registers both
parameters and invokes `corr2d` in its forward pass; a layer with a kernel of
height $h$ and width $w$ is called an $h \times w$ convolutional layer.
<!--tr:c6371ece-->
2 次元の畳み込み層は、入力と学習されたカーネルの相互相関を計算し、スカラーのバイアスを加えます。:numref:`chap_perceptrons` の MLP の重みと同じように、カーネルは $\mathcal{N}(0, 0.01^2)$ で初期化し、バイアスは 0 で初期化します。以下の `Conv2D` は、この 2 つのパラメータを登録し、順伝播で `corr2d` を呼び出します。なお、高さ $h$、幅 $w$ のカーネルを持つ層を、$h \times w$ 畳み込み層と呼びます。

<!--en:29cd9061 kind=paragraph-->
Unlike a dense layer, a convolutional layer is not locked to a fixed input
resolution because its parameters depend only on the kernel size. The exact same
$2 \times 2$ layer (5 parameters total) processes both our $3 \times 3$ tensor
`X` and a $6 \times 8$ image, returning outputs of shapes $2 \times 2$ and
$5 \times 7$:
<!--tr:29cd9061-->
全結合層と違って、畳み込み層は決まった入力解像度に縛られません。パラメータがカーネルサイズだけで決まるからです。まったく同じ $2 \times 2$ の層（パラメータは合計 5 個）が、$3 \times 3$ のテンソル `X` も $6 \times 8$ の画像も処理でき、出力の形状はそれぞれ $2 \times 2$ と $5 \times 7$ になります。

<!--en:c9e5aa07 kind=heading-->
### Edge Detection
<!--tr:c9e5aa07-->
### エッジ検出

<!--en:3d9a357b kind=paragraph-->
We now test the vertical edge detector from the introduction on a
$6 \times 8$ synthetic image whose middle four columns are black (`0`) and outer
columns are white (`1`).
<!--tr:3d9a357b-->
次に、冒頭で述べた垂直エッジ検出器を、$6 \times 8$ の合成画像で試します。この画像は、中央の 4 列が黒（`0`）、外側の列が白（`1`）です。

<!--en:68513ff0 kind=paragraph-->
We define the $1 \times 2$ kernel
<!--tr:68513ff0-->
$1 \times 2$ のカーネルを次のように定義します。

<!--en:eb45803b kind=paragraph-->
At every position $(i, j)$, this computes the horizontal *finite difference*
$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, a discrete approximation (up to sign)
of the horizontal derivative of brightness, which vanishes across flat regions
and spikes wherever intensity jumps.
<!--tr:eb45803b-->
このカーネルは、すべての位置 $(i, j)$ で水平方向の*有限差分*（finite difference）$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ を計算します。これは、明るさの水平方向の導関数を（符号を除いて）離散的に近似したもので、平坦な領域では 0 になり、輝度が急に変わる位置で鋭く立ち上がります。

<!--en:e810b3a9 kind=paragraph-->
Cross-correlating the $6 \times 8$ stripe image `X` with the $1 \times 2$
horizontal-difference kernel `K` via `corr2d(X, K)` produces the $6 \times 7$
edge map `Y`:
<!--tr:e810b3a9-->
$6 \times 8$ の縞模様の画像 `X` と $1 \times 2$ の水平差分カーネル `K` の相互相関を `corr2d(X, K)` で計算すると、$6 \times 7$ のエッジマップ `Y` が得られます。

<!--en:568358e4 kind=paragraph-->
Sure enough, `Y` fires `+1` along the white-to-black boundary, `-1` along the
black-to-white boundary, and `0` everywhere else. Applying the exact same kernel
to the transposed image (turning the vertical stripe into a horizontal one)
yields all zeros:
<!--tr:568358e4-->
確かに、`Y` は白から黒への境界に沿って `+1`、黒から白への境界に沿って `-1`、それ以外では `0` になっています。まったく同じカーネルを転置した画像（縦の帯が横の帯になります）に適用すると、結果はすべて 0 です。

<!--en:71c591a4 kind=paragraph-->
Clearly `[1, -1]` responds strictly to vertical edges. Detecting horizontal
edges, diagonals, corners, or textures requires different kernels altogether.
<!--tr:71c591a4-->
`[1, -1]` が垂直エッジにしか反応しないことは明らかです。水平エッジ、斜めのエッジ、角、テクスチャを検出するには、まったく別のカーネルが要ります。

<!--en:f0af4887 kind=heading-->
### Learning a Kernel
<!--tr:f0af4887-->
### カーネルの学習

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
求める特徴が正確にわかっていれば、`[1, -1]` を人手で設計するのは簡単です。しかし、深いネットワーク全体にわたる何千もの中間フィルタを人手で指定できる技術者はいません。では、カーネル `[1, -1]` は、入出力の組 `(X, Y)` だけから勾配降下法で復元できるでしょうか。これを確かめるために、各フレームワークに組み込みの 2 次元畳み込み層を使い（バイアスは省き、重みは $\mathcal{N}(0, 0.01^2)$ で初期化します）、42 個の出力要素すべてにわたって和をとった二乗誤差を最小化します。勾配ステップは 10 回、学習率は $0.03$ です。組み込みの層は、バッチ軸とチャネル軸を含む 4 次元テンソルを入力として想定しているため（:numref:`sec_channels`）、`X` と `Y` をそれに合わせて変形します。

<!--en:c29c5d61 kind=paragraph-->
Initially the loss sits near 12 (the near-zero random kernel misses all 12
$\pm 1$ edge pixels by roughly 1), and within just nine steps it drops below
0.01. Inspecting the learned kernel confirms what it found:
<!--tr:c29c5d61-->
損失は最初 12 付近にあります（ほぼ 0 のランダムなカーネルでは、12 個ある $\pm 1$ のエッジピクセルすべてで約 1 ずつ外れるためです）。そこからわずか 9 ステップで 0.01 を下回ります。学習されたカーネルを調べると、何が見つかったのかを確認できます。

<!--en:67d25b0b kind=paragraph-->
Both entries land within roughly 0.02 of `[1, -1]`. Nothing in this loop knew
anything about edges: given a loss signal, gradient descent discovered the
filter on its own. Here we supervised the single layer directly with `Y`; in a
deep CNN, backpropagation learns every layer's kernels end to end from the final
task loss alone.
<!--tr:67d25b0b-->
どちらの要素も、約 0.02 以内の誤差で `[1, -1]` に一致しています。このループには、エッジについての知識は何も組み込まれていません。損失という信号を与えただけで、勾配降下法が自力でこのフィルタを見つけたのです。ここでは 1 つの層を `Y` で直接教師あり学習させましたが、深い CNN では、誤差逆伝播法が最終的なタスクの損失だけから、すべての層のカーネルをエンドツーエンドで学習します。

<!--en:91b0010d kind=slide_text-->
Convolutions for Images<br>**2D cross-correlation · edge detection · learned kernels · im2col · receptive fields**
<!--tr:91b0010d-->
画像に対する畳み込み<br>**2 次元相互相関 · エッジ検出 · 学習されるカーネル · im2col · 受容野**

<!--en:1d94295a kind=slide_title-->
::: {.slide title="Detecting Local Edges Across an Image"}
<!--tr:1d94295a-->
::: {.slide title="画像全体での局所的なエッジの検出"}

<!--en:687ed302 kind=kicker-->
[Motivation & Core Mental Model]{.kicker}
<!--tr:687ed302-->
[動機と中核となるメンタルモデル]{.kicker}

<!--en:05fceed0 kind=slide_text-->
- **Synthetic $6 \times 8$ test image:** White outer columns (`1`) surrounding a 4-column black vertical band (`0`).
<!--tr:05fceed0-->
- **合成した $6 \times 8$ のテスト画像：**外側の白い列（`1`）が、4 列分の黒い縦帯（`0`）を挟む

<!--en:71e700b3 kind=slide_text-->
- **Horizontal finite difference:** Sweeping $\mathbf{K} = [1, -1]$ computes $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, firing `+1` at white-to-black and `-1` at black-to-white edges.
<!--tr:71e700b3-->
- **水平方向の有限差分：**$\mathbf{K} = [1, -1]$ を走らせると $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ が計算され、白から黒へのエッジで `+1`、黒から白へのエッジで `-1` を出力する

<!--en:957dc033 kind=slide_text-->
- **Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent.
<!--tr:957dc033-->
- **学習と手作りの対比：**深い CNN は、このようなカーネルを何千も勾配降下法で自動的に見つける

<!--en:cb1045e1 kind=slide_title-->
::: {.slide title="The 2D Cross-Correlation Operation"}
<!--tr:cb1045e1-->
::: {.slide title="2 次元相互相関の演算"}

<!--en:17bcece5 kind=kicker-->
[Sliding-Window Inner Products]{.kicker}
<!--tr:17bcece5-->
[スライディングウィンドウの内積]{.kicker}

<!--en:8d45aae8 kind=slide_text-->
Given an $n_\textrm{h} \times n_\textrm{w}$ input matrix $\mathbf{X}$ and a $k_\textrm{h} \times k_\textrm{w}$ kernel $\mathbf{K}$, the 2D cross-correlation $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ places the window's top-left corner at $(i, j)$ and sums elementwise products:
<!--tr:8d45aae8-->
$n_\textrm{h} \times n_\textrm{w}$ の入力行列 $\mathbf{X}$ と $k_\textrm{h} \times k_\textrm{w}$ のカーネル $\mathbf{K}$ に対して、2 次元相互相関 $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ は、ウィンドウの左上隅を $(i, j)$ に置き、要素ごとの積の和をとる。

<!--en:e484380c kind=slide_text-->
- **Top-left anchor:** Zero-based indices $a, b \ge 0$ match array indexing in code and handle even-sized kernels (like $2 \times 2$ or $1 \times 2$) cleanly.
<!--tr:e484380c-->
- **左上隅を基準：**0 始まりの添字 $a, b \ge 0$ はコードの配列の添字と一致し、偶数サイズのカーネル（$2 \times 2$ や $1 \times 2$ など）も無理なく扱える

<!--en:137c0980 kind=slide_text-->
- **Shaded top-left output $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.
<!--tr:137c0980-->
- **網掛けされた左上の出力 $(0,0)$：**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$

<!--en:575c4aea kind=slide_title-->
::: {.slide title="Valid Window Positions and Output Spatial Shape"}
<!--tr:575c4aea-->
::: {.slide title="有効なウィンドウ位置と出力の空間的な形状"}

<!--en:7eda4531 kind=kicker-->
[Why Unpadded Convolutions Shrink the Grid]{.kicker}
<!--tr:7eda4531-->
[パディングなしの畳み込みでグリッドが縮む理由]{.kicker}

<!--en:891243ae kind=slide_text-->
- **Valid window placements:** A $k_\textrm{h} \times k_\textrm{w}$ window fits inside an $n_\textrm{h} \times n_\textrm{w}$ grid at $n - k + 1$ positions per axis.
<!--tr:891243ae-->
- **有効なウィンドウの配置：**$k_\textrm{h} \times k_\textrm{w}$ のウィンドウは、$n_\textrm{h} \times n_\textrm{w}$ のグリッドの中に、各軸 $n - k + 1$ か所で収まる

<!--en:55cc8c32 kind=slide_text-->
- **Shape examples:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.
<!--tr:55cc8c32-->
- **形状の例：**$3 \times 3 \star 2 \times 2 \to 2 \times 2$（$19, 25, 37, 43$）、$28 \times 28 \star 5 \times 5 \to 24 \times 24$

<!--en:1e3463da kind=slide_title-->
::: {.slide title="Implementing 2D Cross-Correlation in Code"}
<!--tr:1e3463da-->
::: {.slide title="2 次元相互相関のコードによる実装"}

<!--en:07566992 kind=kicker-->
[Direct Nested-Loop Implementation]{.kicker}
<!--tr:07566992-->
[入れ子のループによる直接的な実装]{.kicker}

<!--en:202aa5d3 kind=slide_text-->
The function `corr2d` allocates the $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ output tensor and computes each entry as the sum of `X[i:i+h, j:j+w] * K`:
<!--tr:202aa5d3-->
関数 `corr2d` は、$(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ の出力テンソルを確保し、各要素を `X[i:i+h, j:j+w] * K` の和として計算する。

<!--en:bef130ba kind=slide_text-->
Each iteration slices a $k_\textrm{h} \times k_\textrm{w}$ sub-window of `X`, multiplies elementwise with `K`, and reduces via `sum`.
<!--tr:bef130ba-->
各反復では、$k_\textrm{h} \times k_\textrm{w}$ の部分ウィンドウを `X` からスライスし、`K` と要素ごとに掛け、`sum` で集約する。

<!--en:4ee9f6b1 kind=slide_title-->
::: {.slide title="Verifying corr2d and the Flipped-Kernel Equivalence"}
<!--tr:4ee9f6b1-->
::: {.slide title="corr2d の検証と反転カーネルの等価性"}

<!--en:2b0be423 kind=kicker-->
[Cross-Correlation vs. Mathematical Convolution]{.kicker}
<!--tr:2b0be423-->
[相互相関と数学的な畳み込み]{.kicker}

<!--en:74b9b745 kind=slide_text-->
- **180° flipped kernel $\mathbf{K}'$:** Cross-correlating with $\mathbf{K}$ equals convolving with $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.
<!--tr:74b9b745-->
- **180° 反転したカーネル $\mathbf{K}'$：**$\mathbf{K}$ との相互相関は、$[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$ との畳み込みに等しい

<!--en:5c8af7cd kind=slide_text-->
- **Identical learned class:** Frameworks compute cross-correlation and call it convolution.
<!--tr:5c8af7cd-->
- **学習される関数のクラスは同一：**フレームワークは相互相関を計算し、それを畳み込みと呼ぶ

<!--en:a69e29b5 kind=slide_title-->
::: {.slide title="Building a 2D Convolutional Layer Module"}
<!--tr:a69e29b5-->
::: {.slide title="2 次元畳み込み層モジュールの構築"}

<!--en:55763603 kind=kicker-->
[Learnable Kernel Weights Plus Scalar Bias]{.kicker}
<!--tr:55763603-->
[学習可能なカーネルの重みとスカラーのバイアス]{.kicker}

<!--en:5b8bad24 kind=slide_text-->
A convolutional layer wraps `corr2d` with a learnable $k_\textrm{h} \times k_\textrm{w}$ weight parameter (initialized from $\mathcal{N}(0, 0.01^2)$) and a scalar bias (initialized to $0$):
<!--tr:5b8bad24-->
畳み込み層は `corr2d` を包み、学習可能な $k_\textrm{h} \times k_\textrm{w}$ の重みパラメータ（$\mathcal{N}(0, 0.01^2)$ で初期化）とスカラーのバイアス（$0$ で初期化）を持たせたものである。

<!--en:c16d97d6 kind=slide_title-->
::: {.slide title="Resolution Independence of Convolutional Layers"}
<!--tr:c16d97d6-->
::: {.slide title="畳み込み層の解像度非依存性"}

<!--en:a7d1e48f kind=kicker-->
[Fixed Parameters Across Arbitrary Input Sizes]{.kicker}
<!--tr:a7d1e48f-->
[任意の入力サイズに対して固定のパラメータ]{.kicker}

<!--en:53a2bce7 kind=slide_text-->
- **Decoupled from $H \times W$:** The exact same 5 parameters ($2 \times 2$ kernel + 1 bias) process $3 \times 3$, $6 \times 8$, or megapixel inputs without reshaping weights.
<!--tr:53a2bce7-->
- **$H \times W$ から独立：**まったく同じ 5 個のパラメータ（$2 \times 2$ カーネル + バイアス 1 個）が、重みを変形することなく $3 \times 3$、$6 \times 8$、さらにはメガピクセルの入力を処理する

<!--en:017fec24 kind=slide_title-->
::: {.slide title="Constructing a Synthetic Vertical-Edge Image"}
<!--tr:017fec24-->
::: {.slide title="垂直エッジを持つ合成画像の作成"}

<!--en:896e641c kind=kicker-->
[A 6×8 Binary Test Pattern]{.kicker}
<!--tr:896e641c-->
[6×8 の 2 値テストパターン]{.kicker}

<!--en:0a4b456b kind=slide_text-->
We construct the $6 \times 8$ test image `X` with ones on the left and right borders and zeros across the middle four columns (`X[:, 2:6] = 0`):
<!--tr:0a4b456b-->
$6 \times 8$ のテスト画像 `X` を、左右の縁は 1、中央の 4 列は 0（`X[:, 2:6] = 0`）として作成する。

<!--en:dfc7e3e3 kind=slide_text-->
- **Piecewise-constant columns:** Intensity is constant (`1` or `0`) within columns $0\text{--}1$, $2\text{--}5$, and $6\text{--}7$, jumping only between columns $1 \to 2$ and $5 \to 6$.
<!--tr:dfc7e3e3-->
- **区分的に一定な列：**輝度が一定（`1` または `0`）なのは列 $0\text{--}1$、$2\text{--}5$、$6\text{--}7$ のそれぞれの内部で、変化するのは列 $1 \to 2$ と $5 \to 6$ の間だけ

<!--en:3ce31cbe kind=slide_title-->
::: {.slide title="Applying the Horizontal Finite-Difference Kernel"}
<!--tr:3ce31cbe-->
::: {.slide title="水平方向の有限差分カーネルの適用"}

<!--en:b37a1029 kind=kicker-->
[Detecting Vertical Edges with K = [1, -1]]{.kicker}
<!--tr:b37a1029-->
[K = [1, -1] による垂直エッジの検出]{.kicker}

<!--en:9be85019 kind=slide_text-->
Cross-correlating `X` with the $1 \times 2$ kernel $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ computes $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ at every horizontal pair:
<!--tr:9be85019-->
`X` と $1 \times 2$ カーネル $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ の相互相関は、水平方向に隣り合う組のすべてで $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ を計算する。

<!--en:f649cfae kind=slide_text-->
- **Signed edge map:** Column 1 fires `+1` (white-to-black transition), column 5 fires `-1` (black-to-white transition), and all constant interior columns evaluate to `0`.
<!--tr:f649cfae-->
- **符号付きのエッジマップ：**列 1 は `+1`（白から黒への変化）、列 5 は `-1`（黒から白への変化）を出力し、値が一定の内部の列はすべて `0` になる

<!--en:e6d155be kind=slide_title-->
::: {.slide title="Orientation Selectivity of Edge Kernels"}
<!--tr:e6d155be-->
::: {.slide title="エッジカーネルの方向選択性"}

<!--en:b525cde4 kind=kicker-->
[Why One Kernel Cannot Detect All Orientations]{.kicker}
<!--tr:b525cde4-->
[1 つのカーネルではすべての方向を検出できない理由]{.kicker}

<!--en:38159de6 kind=slide_text-->
- **Blind to horizontal edges:** Transposing `X` (`X.T`) makes horizontal neighbors identical, so $[1, -1]$ outputs all zeros.
<!--tr:38159de6-->
- **水平エッジは見えない：**`X` を転置すると（`X.T`）水平方向の隣どうしが等しくなるため、$[1, -1]$ の出力はすべて 0 になる

<!--en:f0f30f76 kind=slide_text-->
- **Filter banks (§{sec_channels}):** Vision layers learn parallel channels for horizontal, vertical, and diagonal orientations.
<!--tr:f0f30f76-->
- **フィルタバンク（§{sec_channels}）：**画像を扱う層は、水平・垂直・斜めの方向に対応するチャネルを並列に学習する

<!--en:b755e31b kind=slide_title-->
::: {.slide title="Learning an Edge Detector via Gradient Descent"}
<!--tr:b755e31b-->
::: {.slide title="勾配降下法によるエッジ検出器の学習"}

<!--en:86bbeea7 kind=kicker-->
[Supervising a Built-In Convolutional Layer on (X, Y)]{.kicker}
<!--tr:86bbeea7-->
[組み込みの畳み込み層を (X, Y) で教師あり学習]{.kicker}

<!--en:a569780c kind=slide_text-->
- **4D framework tensors:** Reshape `X` and `Y` to 4D tensors (one example, one channel) and minimize $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ at $\eta = 0.03$.
<!--tr:a569780c-->
- **フレームワークの 4 次元テンソル：**`X` と `Y` を 4 次元テンソル（1 例、1 チャネル）に変形し、$\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ を $\eta = 0.03$ で最小化する

<!--en:2489ce6e kind=slide_title-->
::: {.slide title="Inspecting the Learned Kernel Weights"}
<!--tr:2489ce6e-->
::: {.slide title="学習されたカーネルの重みの確認"}

<!--en:7e4ef44b kind=kicker-->
[Why Initial Loss Is 12 and Converges to [1, -1]]{.kicker}
<!--tr:7e4ef44b-->
[初期損失が 12 で、[1, -1] に収束する理由]{.kicker}

<!--en:9cd1c645 kind=slide_text-->
- **Why loss starts at $12.0$:** Near-zero weights predict $\hat{Y} \approx 0$, missing the twelve $\pm 1$ edge pixels in `Y` ($12 \times 1^2 = 12$).
<!--tr:9cd1c645-->
- **損失が $12.0$ から始まる理由：**ほぼ 0 の重みは $\hat{Y} \approx 0$ を予測し、12 個ある $\pm 1$ のエッジピクセル（`Y` 内）をすべて外す（$12 \times 1^2 = 12$）

<!--en:456670c5 kind=slide_text-->
- **Convergence in 9 steps:** SGD recovers $[0.99, -0.99]$ automatically from data.
<!--tr:456670c5-->
- **9 ステップで収束：**SGD はデータから自動的に $[0.99, -0.99]$ を復元する

=== DECISIONS ===
- "cross-correlation" → 相互相関; "two-dimensional" → 2 次元 (half-width digit, style §4); "2D"/"4D" in prose and slides → 2 次元 / 4 次元
- English gloss (style §3.1) given once on this page, in the opening paragraphs, for 平行移動同変性, 局所性, 2 次元相互相関 and 受容野; 有限差分 is glossed at its first body use. Later units do not repeat them
- "edge" → エッジ; "edge map" → エッジマップ; "edge detector" → エッジ検出器; "vertical / horizontal edge" → 垂直エッジ / 水平エッジ
- "window" → ウィンドウ; "patch" → パッチ; "entry / element" (of a matrix or tensor) → 要素; "upper-left / top-left corner" → 左上隅
- "unit" (of a network) → ユニット; "stack (layers)" → 積み重ねる; "shaded" (in figures) → 網掛け
- "dense layer" → 全結合層 (same term as "fully connected layer")
- "built-in layer" → 組み込みの層; "framework" → フレームワーク; "hand-crafted" → 手作りの / 人手で設計した
- "valid position / window" → 有効な位置 / 有効なウィンドウ; "unpadded" → パディングなしの
- "zero-based indices" → 0 始まりの添字; "offset" → オフセット
- "end to end" → エンドツーエンドで
- "intensity" (of a pixel) → 輝度; "brightness" → 明るさ; "stripe / band" → 帯 (縞模様 for "stripe image")
- "filter bank" → フィルタバンク; "mental model" → メンタルモデル
- Slides: a full-sentence paragraph ends with 。; a single-sentence bullet has no 。; a bold lead-in takes ：
- A sentence that introduces a display equation or code output ends with 。 (次のようになります。), never with a colon

=== NOTES ===
- fa1b7da9: the caption's colon before the worked sum is kept as ： because the sum is the content the caption labels; splitting it into a sentence would detach it.
- 137c0980: the trailing full stop after the formula is dropped (single-item slide bullet, style §6).
- 61cacfaf: sentence order follows the English token order, so "10 gradient steps at learning rate $0.03$" became its own sentence.
- c6371ece, 67d25b0b, 957dc033, 456670c5: check warnings "glossary term not used" for MLP, CNN, CNNs, SGD are false matches. The English writes the acronym, and style §3.2 leaves acronyms in English (MLP, CNN, SGD); the long forms 多層パーセプトロン etc. are not called for.
