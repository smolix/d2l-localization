<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u01 lang=ja rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:33d3b96b kind=heading-->
# Convolutions for Images
<!--tr:33d3b96b-->
# 画像のための畳み込み

<!--en:852f5868 kind=paragraph-->
In :numref:`sec_why-conv`, we derived the convolutional layer on paper from two
geometric principles: translation equivariance and locality. Now we see what
that formula actually computes on an image, and whether gradient descent can
discover useful kernels automatically from data.
<!--tr:852f5868-->
:numref:`sec_why-conv` では、平行移動同変性（translation equivariance）と局所性（locality）という 2 つの幾何学的な原理から、畳み込み層（convolutional layer）を紙の上で導きました。本節では、その式が画像に対して実際に何を計算するのかを確認し、勾配降下法（gradient descent）によって有用なカーネルをデータから自動的に見つけられるかどうかを調べます。

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
明るい領域と暗い領域の境目にある垂直方向のエッジを見つけることを考えます。各ピクセルの値から右隣のピクセルの値を引く（$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$）と、一様な領域の内部では 0、白から黒に変わる位置では $+1$、黒から白に戻る位置では $-1$ になります。2 つの重みからなる検出器 $(1, -1)$ を画像上の有効なすべての位置で走査すると、*エッジマップ*が得られます。本節では、2 次元の相互相関（cross-correlation）を実装して手作りのエッジ検出器を試し、勾配降下法に $(1, -1)$ をデータから直接*学習*させ、畳み込みが 1 回の行列積になる仕組み（*im2col*）を示し、層を積み重ねるとユニットの受容野（receptive field）がどのように広がるかを導きます。

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
ここではチャネルを考えないことにして、入力 $\mathbf{X}$ を $n_\textrm{h} \times n_\textrm{w}$ 行列、カーネル $\mathbf{K}$ を $k_\textrm{h} \times k_\textrm{w}$ 行列とします。カーネルは、形状 $k_\textrm{h} \times k_\textrm{w}$ の*ウィンドウ*を定め、これを入力上でスライドさせます。ウィンドウの左上隅を $(i, j)$ に置き、ウィンドウが覆う入力のパッチと $\mathbf{K}$ を要素ごとに掛け合わせて総和を取り、位置 $(i, j)$ の出力要素を求めます。

<!--en:3688b38b kind=paragraph-->
This is the *two-dimensional cross-correlation* of $\mathbf{X}$ with
$\mathbf{K}$, written $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ with
zero-based indices matching code. Whereas :numref:`sec_why-conv` centered
offsets symmetrically from $-\Delta$ to $\Delta$, anchoring at the upper-left
corner differs only by a fixed shift of $\Delta$ while naturally handling
even-sized kernels (such as a $2 \times 2$ kernel or our $1 \times 2$ edge
detector) that have no center pixel.
<!--tr:3688b38b-->
これが $\mathbf{X}$ と $\mathbf{K}$ の*2 次元相互相関*であり、コードの添字に合わせてゼロ始まりの添字を用いて $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ と書きます。:numref:`sec_why-conv` ではオフセットを $-\Delta$ から $\Delta$ まで対称に取っていました。左上隅を基準にしても、ずれは $\Delta$ だけの固定量にすぎません。そのうえ、中心のピクセルを持たない偶数サイズのカーネル（$2 \times 2$ のカーネルや、本節の $1 \times 2$ のエッジ検出器など）も自然に扱えます。

<!--en:fa1b7da9 kind=caption-->
![Two-dimensional cross-correlation. The shaded portions are the first output element and the input and kernel elements used to compute it: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
<!--tr:fa1b7da9-->
![2 次元相互相関。網掛けの部分は最初の出力要素と、その計算に使う入力およびカーネルの要素であり、$0\times0+1\times1+3\times2+4\times3=19$ となる。](../img/correlation.svg)

<!--en:2f8e5deb kind=paragraph-->
In :numref:`fig_correlation`, we slide a $2 \times 2$ kernel across a
$3 \times 3$ input one pixel at a time, left to right and top to bottom. The
window fits cleanly at four positions, yielding the $2 \times 2$ output:
<!--tr:2f8e5deb-->
:numref:`fig_correlation` では、$2 \times 2$ のカーネルを $3 \times 3$ の入力上で 1 ピクセルずつ、左から右、上から下へとスライドさせます。ウィンドウがきちんと収まる位置は 4 つあり、次の $2 \times 2$ の出力が得られます。

<!--en:4741e012 kind=paragraph-->
Notice that the output shrinks slightly along each axis because the kernel must
stay wholly inside the input. Fitting a width-$k$ window into length $n$ leaves
$n - k + 1$ valid positions, so the output shape is
<!--tr:4741e012-->
出力が各軸方向にわずかに縮むのは、カーネルが入力の内側に完全に収まらなければならないためです。幅 $k$ のウィンドウを長さ $n$ の軸に当てはめると、有効な位置は $n - k + 1$ 個なので、出力の形状は次のようになります。

<!--en:12d81c14 kind=paragraph-->
Thus a $3 \times 3$ input with a $2 \times 2$ kernel yields $2 \times 2$, a
$6 \times 8$ image with a $1 \times 2$ kernel yields $6 \times 7$, and a
$28 \times 28$ Fashion-MNIST image with a $5 \times 5$ kernel yields
$24 \times 24$. Whenever we want to preserve spatial dimensions, we surround the
input with a border of zeros (*padding*), which we examine alongside strides and
dilation in :numref:`sec_padding`.
<!--tr:12d81c14-->
したがって、$3 \times 3$ の入力に $2 \times 2$ のカーネルを適用すると $2 \times 2$ の出力、$6 \times 8$ の画像に $1 \times 2$ のカーネルを適用すると $6 \times 7$ の出力、$28 \times 28$ の Fashion-MNIST 画像に $5 \times 5$ のカーネルを適用すると $24 \times 24$ の出力が得られます。空間次元を保ちたいときは、入力の周囲をゼロの枠で囲みます（*パディング*）。これについては、ストライドやダイレーションとあわせて :numref:`sec_padding` で扱います。

<!--en:869d223f kind=heading-->
### Implementation
<!--tr:869d223f-->
### 実装

<!--en:6a869f7c kind=paragraph-->
We implement :eqref:`eq_corr2d` directly in `corr2d` using a pair of loops over
every output coordinate.
<!--tr:6a869f7c-->
:eqref:`eq_corr2d` を、すべての出力座標に対する二重ループを使って `corr2d` に直接実装します。

<!--en:81e1f383 kind=paragraph-->
To verify `corr2d` against our hand calculation in :numref:`fig_correlation`,
we construct the $3 \times 3$ input tensor `X` and $2 \times 2$ kernel `K` and
evaluate `corr2d(X, K)`.
<!--tr:81e1f383-->
:numref:`fig_correlation` の手計算の結果と `corr2d` を照合するため、$3 \times 3$ の入力テンソル `X` と $2 \times 2$ のカーネル `K` を作り、`corr2d(X, K)` を評価します。

<!--en:f567e1b2 kind=paragraph-->
The output reproduces :numref:`fig_correlation` entry for entry (and as shown in
:numref:`sec_why-conv`, strict mathematical convolution with a flipped kernel
$\mathbf{K}'$ learns the exact same family of layers, so we follow standard
deep learning terminology and call cross-correlation "convolution").
<!--tr:f567e1b2-->
出力は :numref:`fig_correlation` と全要素で一致します（また :numref:`sec_why-conv` で示したように、カーネルを反転した $\mathbf{K}'$ を用いる厳密な数学的畳み込みでも、学習できる層のまったく同じ族が得られます。そのため、深層学習の標準的な用語に従い、相互相関を「畳み込み」と呼びます）。

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
2 次元の畳み込み層は、入力を学習されるカーネルと相互相関させ、スカラーのバイアスを加えます。:numref:`chap_perceptrons` の多層パーセプトロン（multilayer perceptron、MLP）の重みと同様に、カーネルは $\mathcal{N}(0, 0.01^2)$ から、バイアスは 0 で初期化します。以下の `Conv2D` は両方のパラメータを登録し、順伝播（forward propagation）で `corr2d` を呼び出します。高さ $h$、幅 $w$ のカーネルを持つ層を、$h \times w$ の畳み込み層と呼びます。

<!--en:29cd9061 kind=paragraph-->
Unlike a dense layer, a convolutional layer is not locked to a fixed input
resolution because its parameters depend only on the kernel size. The exact same
$2 \times 2$ layer (5 parameters total) processes both our $3 \times 3$ tensor
`X` and a $6 \times 8$ image, returning outputs of shapes $2 \times 2$ and
$5 \times 7$:
<!--tr:29cd9061-->
全結合層（dense layer）と違って、畳み込み層は入力の解像度に固定されません。パラメータはカーネルサイズだけで決まるからです。まったく同じ $2 \times 2$ の層（パラメータは合計 5 個）で、$3 \times 3$ のテンソル `X` も $6 \times 8$ の画像も処理でき、出力の形状はそれぞれ $2 \times 2$ と $5 \times 7$ になります。

<!--en:c9e5aa07 kind=heading-->
### Edge Detection
<!--tr:c9e5aa07-->
### エッジ検出

<!--en:3d9a357b kind=paragraph-->
We now test the vertical edge detector from the introduction on a
$6 \times 8$ synthetic image whose middle four columns are black (`0`) and outer
columns are white (`1`).
<!--tr:3d9a357b-->
導入で触れた垂直エッジ検出器を、$6 \times 8$ の合成画像で試します。この画像は、中央の 4 列が黒（`0`）、外側の列が白（`1`）です。

<!--en:68513ff0 kind=paragraph-->
We define the $1 \times 2$ kernel
<!--tr:68513ff0-->
$1 \times 2$ のカーネルを定義します。

<!--en:eb45803b kind=paragraph-->
At every position $(i, j)$, this computes the horizontal *finite difference*
$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, a discrete approximation (up to sign)
of the horizontal derivative of brightness, which vanishes across flat regions
and spikes wherever intensity jumps.
<!--tr:eb45803b-->
このカーネルは各位置 $(i, j)$ で、水平方向の*有限差分*（finite difference）$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ を計算します。これは明るさの水平方向の導関数（derivative）を、符号を除いて離散的に近似したものであり、平坦な領域では 0 になり、強度が跳ぶ位置ではいつでも大きな値を取ります。

<!--en:e810b3a9 kind=paragraph-->
Cross-correlating the $6 \times 8$ stripe image `X` with the $1 \times 2$
horizontal-difference kernel `K` via `corr2d(X, K)` produces the $6 \times 7$
edge map `Y`:
<!--tr:e810b3a9-->
$6 \times 8$ の縞模様の画像 `X` を、$1 \times 2$ の水平差分カーネル `K` と `corr2d(X, K)` で相互相関させると、$6 \times 7$ のエッジマップ `Y` が得られます。

<!--en:568358e4 kind=paragraph-->
Sure enough, `Y` fires `+1` along the white-to-black boundary, `-1` along the
black-to-white boundary, and `0` everywhere else. Applying the exact same kernel
to the transposed image (turning the vertical stripe into a horizontal one)
yields all zeros:
<!--tr:568358e4-->
予想どおり、`Y` は白から黒への境界に沿って `+1`、黒から白への境界に沿って `-1`、それ以外の場所ではすべて `0` を示します。まったく同じカーネルを転置した画像（垂直の縞が水平の縞になります）に適用すると、出力はすべて 0 になります。

<!--en:71c591a4 kind=paragraph-->
Clearly `[1, -1]` responds strictly to vertical edges. Detecting horizontal
edges, diagonals, corners, or textures requires different kernels altogether.
<!--tr:71c591a4-->
明らかに `[1, -1]` は垂直エッジにだけ反応します。水平エッジ、斜めのエッジ、角、テクスチャを検出するには、まったく別のカーネルが必要です。

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
`[1, -1]` を手作りするのは、求める特徴が正確にわかっていれば簡単です。しかし、深いネットワークに現れる何千もの中間フィルタを、人手ですべて指定できる技術者はいません。では、勾配降下法は、カーネル `[1, -1]` を入出力の組 `(X, Y)` だけから復元できるでしょうか。これを、各フレームワークが備える組み込みの 2 次元畳み込み層（バイアスなし、重みは $\mathcal{N}(0, 0.01^2)$ で初期化）を使って試します。42 個の出力要素すべてについて合計した二乗誤差（squared error）を最小化し、学習率（learning rate）$0.03$ で勾配降下を 10 ステップ行います。組み込みの層はバッチ軸とチャネル軸を含む 4 次元テンソルを要求するため（:numref:`sec_channels`）、`X` と `Y` をそれに合わせて整形します。

<!--en:c29c5d61 kind=paragraph-->
Initially the loss sits near 12 (the near-zero random kernel misses all 12
$\pm 1$ edge pixels by roughly 1), and within just nine steps it drops below
0.01. Inspecting the learned kernel confirms what it found:
<!--tr:c29c5d61-->
最初の損失は 12 付近にあります。ほぼ 0 のランダムなカーネルは、$\pm 1$ である 12 個のエッジピクセルのすべてで約 1 ずつ予測を外すからです。わずか 9 ステップで損失は 0.01 を下回ります。学習されたカーネルを調べると、何が見つかったかがわかります。

<!--en:67d25b0b kind=paragraph-->
Both entries land within roughly 0.02 of `[1, -1]`. Nothing in this loop knew
anything about edges: given a loss signal, gradient descent discovered the
filter on its own. Here we supervised the single layer directly with `Y`; in a
deep CNN, backpropagation learns every layer's kernels end to end from the final
task loss alone.
<!--tr:67d25b0b-->
2 つの要素はどちらも `[1, -1]` から約 0.02 以内に収まっています。このループには、エッジに関する知識がまったく含まれていません。損失という信号が与えられただけで、勾配降下法がフィルタを自力で発見したのです。ここでは `Y` を使って 1 つの層を直接教師あり学習させましたが、深い畳み込みニューラルネットワーク（convolutional neural network、CNN）では、誤差逆伝播法（backpropagation）が、最終的なタスクの損失だけから、すべての層のカーネルをエンドツーエンドで学習します。

<!--en:91b0010d kind=slide_text-->
Convolutions for Images<br>**2D cross-correlation · edge detection · learned kernels · im2col · receptive fields**
<!--tr:91b0010d-->
画像のための畳み込み<br>**2 次元相互相関 · エッジ検出 · 学習されたカーネル · im2col · 受容野**

<!--en:1d94295a kind=slide_title-->
::: {.slide title="Detecting Local Edges Across an Image"}
<!--tr:1d94295a-->
::: {.slide title="画像全体にわたる局所エッジの検出"}

<!--en:687ed302 kind=kicker-->
[Motivation & Core Mental Model]{.kicker}
<!--tr:687ed302-->
[動機と基本の考え方]{.kicker}

<!--en:05fceed0 kind=slide_text-->
- **Synthetic $6 \times 8$ test image:** White outer columns (`1`) surrounding a 4-column black vertical band (`0`).
<!--tr:05fceed0-->
- **$6 \times 8$ の合成テスト画像：**白（`1`）の外側の列が、4 列幅の黒（`0`）の縦帯を挟む

<!--en:71e700b3 kind=slide_text-->
- **Horizontal finite difference:** Sweeping $\mathbf{K} = [1, -1]$ computes $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, firing `+1` at white-to-black and `-1` at black-to-white edges.
<!--tr:71e700b3-->
- **水平方向の有限差分：**$\mathbf{K} = [1, -1]$ を走査すると $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ が計算され、白から黒のエッジで `+1`、黒から白のエッジで `-1` が現れる

<!--en:957dc033 kind=slide_text-->
- **Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent.
<!--tr:957dc033-->
- **学習と手作り：**深い CNN は、勾配降下法によって、このようなカーネルを何千個も自動的に見つける

<!--en:cb1045e1 kind=slide_title-->
::: {.slide title="The 2D Cross-Correlation Operation"}
<!--tr:cb1045e1-->
::: {.slide title="2 次元相互相関の演算"}

<!--en:17bcece5 kind=kicker-->
[Sliding-Window Inner Products]{.kicker}
<!--tr:17bcece5-->
[スライディングウィンドウによる内積]{.kicker}

<!--en:8d45aae8 kind=slide_text-->
Given an $n_\textrm{h} \times n_\textrm{w}$ input matrix $\mathbf{X}$ and a $k_\textrm{h} \times k_\textrm{w}$ kernel $\mathbf{K}$, the 2D cross-correlation $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ places the window's top-left corner at $(i, j)$ and sums elementwise products:
<!--tr:8d45aae8-->
$n_\textrm{h} \times n_\textrm{w}$ の入力行列 $\mathbf{X}$ と $k_\textrm{h} \times k_\textrm{w}$ のカーネル $\mathbf{K}$ が与えられたとき、2 次元相互相関 $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ は、ウィンドウの左上隅を $(i, j)$ に置き、要素ごとの積を合計する。

<!--en:e484380c kind=slide_text-->
- **Top-left anchor:** Zero-based indices $a, b \ge 0$ match array indexing in code and handle even-sized kernels (like $2 \times 2$ or $1 \times 2$) cleanly.
<!--tr:e484380c-->
- **左上のアンカー：**ゼロ始まりの添字 $a, b \ge 0$ はコードでの配列の添字と対応し、偶数サイズのカーネル（$2 \times 2$ や $1 \times 2$ など）も無理なく扱える

<!--en:137c0980 kind=slide_text-->
- **Shaded top-left output $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.
<!--tr:137c0980-->
- **網掛けの左上の出力 $(0,0)$：**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$

<!--en:575c4aea kind=slide_title-->
::: {.slide title="Valid Window Positions and Output Spatial Shape"}
<!--tr:575c4aea-->
::: {.slide title="有効なウィンドウ位置と出力の空間形状"}

<!--en:7eda4531 kind=kicker-->
[Why Unpadded Convolutions Shrink the Grid]{.kicker}
<!--tr:7eda4531-->
[パディングなしの畳み込みでグリッドが縮む理由]{.kicker}

<!--en:891243ae kind=slide_text-->
- **Valid window placements:** A $k_\textrm{h} \times k_\textrm{w}$ window fits inside an $n_\textrm{h} \times n_\textrm{w}$ grid at $n - k + 1$ positions per axis.
<!--tr:891243ae-->
- **有効なウィンドウの置き方：**$k_\textrm{h} \times k_\textrm{w}$ のウィンドウは、$n_\textrm{h} \times n_\textrm{w}$ のグリッドの中に、各軸で $n - k + 1$ 通りに収まる

<!--en:55cc8c32 kind=slide_text-->
- **Shape examples:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.
<!--tr:55cc8c32-->
- **形状の例：**$3 \times 3 \star 2 \times 2 \to 2 \times 2$（$19, 25, 37, 43$）、$28 \times 28 \star 5 \times 5 \to 24 \times 24$

<!--en:1e3463da kind=slide_title-->
::: {.slide title="Implementing 2D Cross-Correlation in Code"}
<!--tr:1e3463da-->
::: {.slide title="コードによる 2 次元相互相関の実装"}

<!--en:07566992 kind=kicker-->
[Direct Nested-Loop Implementation]{.kicker}
<!--tr:07566992-->
[入れ子ループによる直接実装]{.kicker}

<!--en:202aa5d3 kind=slide_text-->
The function `corr2d` allocates the $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ output tensor and computes each entry as the sum of `X[i:i+h, j:j+w] * K`:
<!--tr:202aa5d3-->
`corr2d` 関数は $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ の出力テンソルを確保し、各要素を `X[i:i+h, j:j+w] * K` の総和として計算する。

<!--en:bef130ba kind=slide_text-->
Each iteration slices a $k_\textrm{h} \times k_\textrm{w}$ sub-window of `X`, multiplies elementwise with `K`, and reduces via `sum`.
<!--tr:bef130ba-->
各反復で `X` から $k_\textrm{h} \times k_\textrm{w}$ の部分ウィンドウを切り出し、`K` と要素ごとに掛け、`sum` で縮約する。

<!--en:4ee9f6b1 kind=slide_title-->
::: {.slide title="Verifying corr2d and the Flipped-Kernel Equivalence"}
<!--tr:4ee9f6b1-->
::: {.slide title="corr2d の検証と反転カーネルとの等価性"}

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
- **学習できる層のクラスは同一：**フレームワークは相互相関を計算し、それを畳み込みと呼んでいる

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
畳み込み層は `corr2d` に、学習可能な $k_\textrm{h} \times k_\textrm{w}$ の重みパラメータ（$\mathcal{N}(0, 0.01^2)$ で初期化）とスカラーのバイアス（$0$ で初期化）を持たせたものである。

<!--en:c16d97d6 kind=slide_title-->
::: {.slide title="Resolution Independence of Convolutional Layers"}
<!--tr:c16d97d6-->
::: {.slide title="畳み込み層の解像度非依存性"}

<!--en:a7d1e48f kind=kicker-->
[Fixed Parameters Across Arbitrary Input Sizes]{.kicker}
<!--tr:a7d1e48f-->
[任意の入力サイズに対して固定されたパラメータ]{.kicker}

<!--en:53a2bce7 kind=slide_text-->
- **Decoupled from $H \times W$:** The exact same 5 parameters ($2 \times 2$ kernel + 1 bias) process $3 \times 3$, $6 \times 8$, or megapixel inputs without reshaping weights.
<!--tr:53a2bce7-->
- **$H \times W$ から独立：**まったく同じ 5 個のパラメータ（$2 \times 2$ のカーネル + バイアス 1 個）で、$3 \times 3$、$6 \times 8$、さらにはメガピクセル級の入力まで、重みの形を変えずに処理できる

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
$6 \times 8$ のテスト画像 `X` を、左右の端の列を 1、中央の 4 列を 0 として作る（`X[:, 2:6] = 0`）。

<!--en:dfc7e3e3 kind=slide_text-->
- **Piecewise-constant columns:** Intensity is constant (`1` or `0`) within columns $0\text{--}1$, $2\text{--}5$, and $6\text{--}7$, jumping only between columns $1 \to 2$ and $5 \to 6$.
<!--tr:dfc7e3e3-->
- **区分的に一定な列：**強度は一定（`1` または `0`）で、列 $0\text{--}1$、$2\text{--}5$、$6\text{--}7$ の内部では変化せず、列 $1 \to 2$ と $5 \to 6$ の間でだけ変化する

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
`X` と $1 \times 2$ のカーネル $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ を相互相関させると、水平方向に隣り合うすべての組で $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ が計算される。

<!--en:f649cfae kind=slide_text-->
- **Signed edge map:** Column 1 fires `+1` (white-to-black transition), column 5 fires `-1` (black-to-white transition), and all constant interior columns evaluate to `0`.
<!--tr:f649cfae-->
- **符号付きのエッジマップ：**列 1 は `+1`（白から黒への遷移）、列 5 は `-1`（黒から白への遷移）となり、一定の内部の列はすべて `0` になる

<!--en:e6d155be kind=slide_title-->
::: {.slide title="Orientation Selectivity of Edge Kernels"}
<!--tr:e6d155be-->
::: {.slide title="エッジカーネルの方向選択性"}

<!--en:b525cde4 kind=kicker-->
[Why One Kernel Cannot Detect All Orientations]{.kicker}
<!--tr:b525cde4-->
[1 つのカーネルがすべての向きを検出できない理由]{.kicker}

<!--en:38159de6 kind=slide_text-->
- **Blind to horizontal edges:** Transposing `X` (`X.T`) makes horizontal neighbors identical, so $[1, -1]$ outputs all zeros.
<!--tr:38159de6-->
- **水平エッジには反応しない：**`X` を転置する（`X.T`）と水平方向の隣接要素が等しくなるため、$[1, -1]$ の出力はすべて 0 になる

<!--en:f0f30f76 kind=slide_text-->
- **Filter banks (§{sec_channels}):** Vision layers learn parallel channels for horizontal, vertical, and diagonal orientations.
<!--tr:f0f30f76-->
- **フィルタバンク（§{sec_channels}）：**視覚系の層は、水平・垂直・斜めの向きに対応する並列のチャネルを学習する

<!--en:b755e31b kind=slide_title-->
::: {.slide title="Learning an Edge Detector via Gradient Descent"}
<!--tr:b755e31b-->
::: {.slide title="勾配降下法によるエッジ検出器の学習"}

<!--en:86bbeea7 kind=kicker-->
[Supervising a Built-In Convolutional Layer on (X, Y)]{.kicker}
<!--tr:86bbeea7-->
[組み込みの畳み込み層を (X, Y) で教師あり学習する]{.kicker}

<!--en:a569780c kind=slide_text-->
- **4D framework tensors:** Reshape `X` and `Y` to 4D tensors (one example, one channel) and minimize $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ at $\eta = 0.03$.
<!--tr:a569780c-->
- **4 次元のフレームワーク用テンソル：**`X` と `Y` を 4 次元テンソル（1 例、1 チャネル）に整形し、$\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ を $\eta = 0.03$ で最小化する

<!--en:2489ce6e kind=slide_title-->
::: {.slide title="Inspecting the Learned Kernel Weights"}
<!--tr:2489ce6e-->
::: {.slide title="学習されたカーネルの重みの確認"}

<!--en:7e4ef44b kind=kicker-->
[Why Initial Loss Is 12 and Converges to [1, -1]]{.kicker}
<!--tr:7e4ef44b-->
[初期損失が 12 になり、[1, -1] に収束する理由]{.kicker}

<!--en:9cd1c645 kind=slide_text-->
- **Why loss starts at $12.0$:** Near-zero weights predict $\hat{Y} \approx 0$, missing the twelve $\pm 1$ edge pixels in `Y` ($12 \times 1^2 = 12$).
<!--tr:9cd1c645-->
- **損失が $12.0$ から始まる理由：**ほぼ 0 の重みでは $\hat{Y} \approx 0$ と予測するため、`Y` にある 12 個の $\pm 1$ のエッジピクセルをすべて外す（$12 \times 1^2 = 12$）

<!--en:456670c5 kind=slide_text-->
- **Convergence in 9 steps:** SGD recovers $[0.99, -0.99]$ automatically from data.
<!--tr:456670c5-->
- **9 ステップで収束：**SGD はデータから $[0.99, -0.99]$ を自動的に復元する

=== DECISIONS ===
- "edge" → エッジ; "edge map" → エッジマップ; "edge detector" → エッジ検出器
- "window" (the sliding window of a kernel) → ウィンドウ; "patch" → パッチ
- "dense layer" → 全結合層（glossed once on this page）
- "finite difference" → 有限差分; "derivative" → 導関数; "filter bank" → フィルタバンク
- "unit" (a neuron/activation in a layer) → ユニット
- "gradient step" → 勾配降下の 1 ステップ（ステップ）
- English glosses given on this page so far: 畳み込み層, 平行移動同変性, 局所性, 勾配降下法, 相互相関, 受容野, 有限差分, 導関数, 二乗誤差, 学習率, 誤差逆伝播法, 畳み込みニューラルネットワーク, 多層パーセプトロン, 全結合層, 順伝播
- Slide non-bullet sentences that introduce a display equation or code end in 。 (plain form)

=== NOTES ===
- fa1b7da9: the English caption puts the arithmetic after a colon; written as a sentence ending in 「となる。」 so that no colon is needed.
- 4741e012, 2f8e5deb, ae7e906d, 6a869f7c: sentences that ended in a colon before a display equation or code output now end in 。 (style §2.1).
- 957dc033, 456670c5: the warnings for 'CNNs' and 'SGD' are intended: after its first use (67d25b0b glosses the term, and 勾配降下法 is used in the body) the acronym stays in English as in style.md §3.1/§3.2.
