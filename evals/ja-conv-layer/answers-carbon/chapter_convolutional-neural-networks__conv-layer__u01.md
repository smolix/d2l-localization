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
:numref:`sec_why-conv` では、平行移動同変性（translation equivariance）と局所性（locality）という 2 つの幾何学的原理から、畳み込み層（convolutional layer）を数式の上で導出しました。ここからは、その式が画像に対して実際に何を計算するのか、そして勾配降下法（gradient descent）によってデータから有用なカーネルを自動的に見つけ出せるのかを見ていきましょう。

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
明るい領域と暗い領域が接する垂直方向のエッジを見つけることを考えます。各ピクセルから右隣のピクセルを引くと（$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$）、一様な領域の内部では 0 になり、白から黒へ移る箇所では $+1$、黒から白へ戻る箇所では $-1$ になります。2 つの重みを持つ検出器 $(1, -1)$ を画像のすべての有効な位置にわたって走査すると、*エッジマップ*が得られます。本節では、2 次元の相互相関（cross-correlation）を実装して手作りのエッジ検出器を試し、勾配降下法によってデータから直接 $(1, -1)$ を*学習*させます。さらに、畳み込み（convolution）が単一の行列積（*im2col*）になることを示し、層を積み重ねることでユニットの受容野（receptive field）がどのように広がるかを導出します。

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
ひとまずチャネルのことは脇に置き、入力 $\mathbf{X}$ を $n_\textrm{h} \times n_\textrm{w}$ 行列、カーネル $\mathbf{K}$ を $k_\textrm{h} \times k_\textrm{w}$ 行列とします。このカーネルは、形状 $k_\textrm{h} \times k_\textrm{w}$ のスライディング*ウィンドウ*を定めます。ウィンドウの左上隅を $(i, j)$ に置き、重なった入力パッチと $\mathbf{K}$ を要素ごとに掛けてその積を足し合わせると、$(i, j)$ における出力要素が得られます。

<!--en:3688b38b kind=paragraph-->
This is the *two-dimensional cross-correlation* of $\mathbf{X}$ with
$\mathbf{K}$, written $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ with
zero-based indices matching code. Whereas :numref:`sec_why-conv` centered
offsets symmetrically from $-\Delta$ to $\Delta$, anchoring at the upper-left
corner differs only by a fixed shift of $\Delta$ while naturally handling
even-sized kernels (such as a $2 \times 2$ kernel or our $1 \times 2$ edge
detector) that have no center pixel.
<!--tr:3688b38b-->
これが $\mathbf{X}$ と $\mathbf{K}$ の *2 次元相互相関*であり、コードに合わせて 0 始まりのインデックスを用いて $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ と書きます。:numref:`sec_why-conv` ではオフセットを $-\Delta$ から $\Delta$ まで対称に中心化していましたが、左上隅を基準にしても $\Delta$ の一定のシフト分しか違わないうえ、中心ピクセルを持たない偶数サイズのカーネル（$2 \times 2$ カーネルや先ほどの $1 \times 2$ エッジ検出器など）も自然に扱えます。

<!--en:fa1b7da9 kind=caption-->
![Two-dimensional cross-correlation. The shaded portions are the first output element and the input and kernel elements used to compute it: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
<!--tr:fa1b7da9-->
![2 次元相互相関。網掛け部分は、最初の出力要素と、その計算に使われる入力要素およびカーネル要素を表す（$0\times0+1\times1+3\times2+4\times3=19$）。](../img/correlation.svg)

<!--en:2f8e5deb kind=paragraph-->
In :numref:`fig_correlation`, we slide a $2 \times 2$ kernel across a
$3 \times 3$ input one pixel at a time, left to right and top to bottom. The
window fits cleanly at four positions, yielding the $2 \times 2$ output:
<!--tr:2f8e5deb-->
:numref:`fig_correlation` では、$2 \times 2$ カーネルを $3 \times 3$ の入力上で左から右、上から下へと 1 ピクセルずつスライドさせます。ウィンドウは 4 つの位置にぴったり収まり、次の $2 \times 2$ の出力が得られます。

<!--en:4741e012 kind=paragraph-->
Notice that the output shrinks slightly along each axis because the kernel must
stay wholly inside the input. Fitting a width-$k$ window into length $n$ leaves
$n - k + 1$ valid positions, so the output shape is
<!--tr:4741e012-->
カーネルは入力の内側に完全に収まらなければならないため、出力は各軸に沿って少し縮むことに注目してください。幅 $k$ のウィンドウを長さ $n$ に収めると有効な位置は $n - k + 1$ 個残るので、出力の形状は次のようになります。

<!--en:12d81c14 kind=paragraph-->
Thus a $3 \times 3$ input with a $2 \times 2$ kernel yields $2 \times 2$, a
$6 \times 8$ image with a $1 \times 2$ kernel yields $6 \times 7$, and a
$28 \times 28$ Fashion-MNIST image with a $5 \times 5$ kernel yields
$24 \times 24$. Whenever we want to preserve spatial dimensions, we surround the
input with a border of zeros (*padding*), which we examine alongside strides and
dilation in :numref:`sec_padding`.
<!--tr:12d81c14-->
したがって、$3 \times 3$ の入力と $2 \times 2$ カーネルでは $2 \times 2$、$6 \times 8$ の画像と $1 \times 2$ カーネルでは $6 \times 7$、$28 \times 28$ の Fashion-MNIST 画像と $5 \times 5$ カーネルでは $24 \times 24$ になります。空間次元を保ちたいときは、入力の周囲を 0 の枠で囲みます（*パディング*）。これについては、ストライドやダイレーションとあわせて :numref:`sec_padding` で詳しく見ます。

<!--en:869d223f kind=heading-->
### Implementation
<!--tr:869d223f-->
### 実装

<!--en:6a869f7c kind=paragraph-->
We implement :eqref:`eq_corr2d` directly in `corr2d` using a pair of loops over
every output coordinate.
<!--tr:6a869f7c-->
:eqref:`eq_corr2d` を `corr2d` として直接実装し、すべての出力座標にわたる二重ループで計算します。

<!--en:81e1f383 kind=paragraph-->
To verify `corr2d` against our hand calculation in :numref:`fig_correlation`,
we construct the $3 \times 3$ input tensor `X` and $2 \times 2$ kernel `K` and
evaluate `corr2d(X, K)`.
<!--tr:81e1f383-->
`corr2d` を :numref:`fig_correlation` の手計算と照らし合わせて検証するため、$3 \times 3$ の入力テンソル `X` と $2 \times 2$ のカーネル `K` を作成し、`corr2d(X, K)` を評価します。

<!--en:f567e1b2 kind=paragraph-->
The output reproduces :numref:`fig_correlation` entry for entry (and as shown in
:numref:`sec_why-conv`, strict mathematical convolution with a flipped kernel
$\mathbf{K}'$ learns the exact same family of layers, so we follow standard
deep learning terminology and call cross-correlation "convolution").
<!--tr:f567e1b2-->
出力は :numref:`fig_correlation` を要素ごとに再現しています（また、:numref:`sec_why-conv` で示したように、反転したカーネル $\mathbf{K}'$ を用いる数学的に厳密な畳み込みもまったく同じ層の族を学習するため、深層学習の標準的な用語に従い、相互相関を「畳み込み」と呼びます）。

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
2 次元畳み込み層は、学習されるカーネルと入力の相互相関を計算し、スカラーのバイアスを加えます。MLP の重み（:numref:`chap_perceptrons`）と同様に、カーネルは $\mathcal{N}(0, 0.01^2)$ から初期化し、バイアスは 0 に初期化します。以下の `Conv2D` は両方のパラメータを登録し、順伝播（forward pass）で `corr2d` を呼び出します。高さ $h$、幅 $w$ のカーネルを持つ層は、$h \times w$ 畳み込み層と呼ばれます。

<!--en:29cd9061 kind=paragraph-->
Unlike a dense layer, a convolutional layer is not locked to a fixed input
resolution because its parameters depend only on the kernel size. The exact same
$2 \times 2$ layer (5 parameters total) processes both our $3 \times 3$ tensor
`X` and a $6 \times 8$ image, returning outputs of shapes $2 \times 2$ and
$5 \times 7$:
<!--tr:29cd9061-->
密結合層（dense layer）とは異なり、畳み込み層のパラメータはカーネルサイズだけに依存するため、固定された入力解像度に縛られません。まったく同じ $2 \times 2$ の層（パラメータ数は合計 5 個）で、先ほどの $3 \times 3$ テンソル `X` と $6 \times 8$ の画像の両方を処理でき、それぞれ形状 $2 \times 2$ と $5 \times 7$ の出力を返します。

<!--en:c9e5aa07 kind=heading-->
### Edge Detection
<!--tr:c9e5aa07-->
### エッジ検出

<!--en:3d9a357b kind=paragraph-->
We now test the vertical edge detector from the introduction on a
$6 \times 8$ synthetic image whose middle four columns are black (`0`) and outer
columns are white (`1`).
<!--tr:3d9a357b-->
それでは、冒頭で述べた垂直方向のエッジ検出器を $6 \times 8$ の合成画像で試してみましょう。この画像は中央の 4 列が黒（`0`）、外側の列が白（`1`）です。

<!--en:68513ff0 kind=paragraph-->
We define the $1 \times 2$ kernel
<!--tr:68513ff0-->
次の $1 \times 2$ カーネルを定義します。

<!--en:eb45803b kind=paragraph-->
At every position $(i, j)$, this computes the horizontal *finite difference*
$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, a discrete approximation (up to sign)
of the horizontal derivative of brightness, which vanishes across flat regions
and spikes wherever intensity jumps.
<!--tr:eb45803b-->
このカーネルは、すべての位置 $(i, j)$ において水平方向の*有限差分*（finite difference）$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ を計算します。これは輝度の水平方向の導関数を（符号を除いて）離散近似したものであり、平坦な領域ではゼロになり、強度が急変する箇所で鋭く立ち上がります。

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
予想どおり、`Y` は白から黒への境界に沿って `+1`、黒から白への境界に沿って `-1` を出力し、それ以外の場所ではすべて `0` になります。まったく同じカーネルを転置した画像（縦縞を横縞に変えたもの）に適用すると、すべて 0 になります。

<!--en:71c591a4 kind=paragraph-->
Clearly `[1, -1]` responds strictly to vertical edges. Detecting horizontal
edges, diagonals, corners, or textures requires different kernels altogether.
<!--tr:71c591a4-->
明らかに、`[1, -1]` は垂直方向のエッジにのみ反応します。水平方向のエッジ、斜めの線、角、テクスチャを検出するには、まったく別のカーネルが必要です。

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
欲しい特徴が正確にわかっていれば `[1, -1]` を手作りするのは簡単ですが、深いネットワーク全体にまたがる何千もの中間フィルタをエンジニアが手作業で指定することはできません。勾配降下法は、カーネル `[1, -1]` を入出力の組 `(X, Y)` だけから復元できるでしょうか。各フレームワークに組み込まれた 2 次元畳み込み層（バイアスを省き、重みを $\mathcal{N}(0, 0.01^2)$ から初期化したもの）を使ってこれを検証します。42 個すべての出力要素にわたる二乗誤差（squared error）の和を、10 回の勾配ステップ、学習率（learning rate）$0.03$ で最小化します。組み込みの層はバッチ軸とチャネル軸を含む 4 次元テンソルを入力にとるため（:numref:`sec_channels`）、それに合わせて `X` と `Y` の形状を変換します。

<!--en:c29c5d61 kind=paragraph-->
Initially the loss sits near 12 (the near-zero random kernel misses all 12
$\pm 1$ edge pixels by roughly 1), and within just nine steps it drops below
0.01. Inspecting the learned kernel confirms what it found:
<!--tr:c29c5d61-->
最初、損失は 12 付近にあります（ゼロに近いランダムなカーネルは、12 個すべての $\pm 1$ のエッジピクセルを約 1 だけ外すためです）。しかし、わずか 9 ステップのうちに 0.01 未満まで下がります。学習されたカーネルを調べると、何が見つかったかを確認できます。

<!--en:67d25b0b kind=paragraph-->
Both entries land within roughly 0.02 of `[1, -1]`. Nothing in this loop knew
anything about edges: given a loss signal, gradient descent discovered the
filter on its own. Here we supervised the single layer directly with `Y`; in a
deep CNN, backpropagation learns every layer's kernels end to end from the final
task loss alone.
<!--tr:67d25b0b-->
どちらの要素も、誤差約 0.02 の範囲内で `[1, -1]` に到達しています。このループにはエッジに関する知識は何も組み込まれていません。損失の信号を与えるだけで、勾配降下法が自力でフィルタを見つけ出したのです。ここでは単一の層に `Y` を直接教師信号として与えましたが、深い CNN では、誤差逆伝播法（backpropagation）が最終的なタスク損失だけからすべての層のカーネルをエンドツーエンドで学習します。

<!--en:91b0010d kind=slide_text-->
Convolutions for Images<br>**2D cross-correlation · edge detection · learned kernels · im2col · receptive fields**
<!--tr:91b0010d-->
画像に対する畳み込み<br>**2 次元相互相関・エッジ検出・学習されたカーネル・im2col・受容野**

<!--en:1d94295a kind=slide_title-->
::: {.slide title="Detecting Local Edges Across an Image"}
<!--tr:1d94295a-->
::: {.slide title="画像全体での局所エッジの検出"}

<!--en:687ed302 kind=kicker-->
[Motivation & Core Mental Model]{.kicker}
<!--tr:687ed302-->
[動機と基本のメンタルモデル]{.kicker}

<!--en:05fceed0 kind=slide_text-->
- **Synthetic $6 \times 8$ test image:** White outer columns (`1`) surrounding a 4-column black vertical band (`0`).
<!--tr:05fceed0-->
- **合成 $6 \times 8$ テスト画像：**外側の白い列（`1`）が、4 列幅の黒い縦帯（`0`）を囲む

<!--en:71e700b3 kind=slide_text-->
- **Horizontal finite difference:** Sweeping $\mathbf{K} = [1, -1]$ computes $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, firing `+1` at white-to-black and `-1` at black-to-white edges.
<!--tr:71e700b3-->
- **水平方向の有限差分：**$\mathbf{K} = [1, -1]$ を走査すると $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ が計算され、白から黒のエッジで `+1`、黒から白のエッジで `-1` が出力される

<!--en:957dc033 kind=slide_text-->
- **Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent.
<!--tr:957dc033-->
- **学習と手設計の対比：**深い CNN は勾配降下法によってこのような何千ものカーネルを自動的に見つけ出す

<!--en:cb1045e1 kind=slide_title-->
::: {.slide title="The 2D Cross-Correlation Operation"}
<!--tr:cb1045e1-->
::: {.slide title="2 次元相互相関演算"}

<!--en:17bcece5 kind=kicker-->
[Sliding-Window Inner Products]{.kicker}
<!--tr:17bcece5-->
[スライディングウィンドウの内積]{.kicker}

<!--en:8d45aae8 kind=slide_text-->
Given an $n_\textrm{h} \times n_\textrm{w}$ input matrix $\mathbf{X}$ and a $k_\textrm{h} \times k_\textrm{w}$ kernel $\mathbf{K}$, the 2D cross-correlation $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ places the window's top-left corner at $(i, j)$ and sums elementwise products:
<!--tr:8d45aae8-->
$n_\textrm{h} \times n_\textrm{w}$ の入力行列 $\mathbf{X}$ と $k_\textrm{h} \times k_\textrm{w}$ のカーネル $\mathbf{K}$ に対し、2 次元相互相関 $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ はウィンドウの左上隅を $(i, j)$ に置いて要素ごとの積の和をとる。

<!--en:e484380c kind=slide_text-->
- **Top-left anchor:** Zero-based indices $a, b \ge 0$ match array indexing in code and handle even-sized kernels (like $2 \times 2$ or $1 \times 2$) cleanly.
<!--tr:e484380c-->
- **左上隅を基準：**0 始まりのインデックス $a, b \ge 0$ はコードの配列インデックスと一致し、偶数サイズのカーネル（$2 \times 2$ や $1 \times 2$ など）も自然に扱える

<!--en:137c0980 kind=slide_text-->
- **Shaded top-left output $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.
<!--tr:137c0980-->
- **網掛けされた左上の出力 $(0,0)$：**
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
- **有効なウィンドウ配置：**$k_\textrm{h} \times k_\textrm{w}$ のウィンドウは、$n_\textrm{h} \times n_\textrm{w}$ のグリッド内に軸あたり $n - k + 1$ 個の位置で収まる

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
[二重ループによる直接実装]{.kicker}

<!--en:202aa5d3 kind=slide_text-->
The function `corr2d` allocates the $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ output tensor and computes each entry as the sum of `X[i:i+h, j:j+w] * K`:
<!--tr:202aa5d3-->
関数 `corr2d` は $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ の出力テンソルを確保し、各要素を `X[i:i+h, j:j+w] * K` の和として計算する。

<!--en:bef130ba kind=slide_text-->
Each iteration slices a $k_\textrm{h} \times k_\textrm{w}$ sub-window of `X`, multiplies elementwise with `K`, and reduces via `sum`.
<!--tr:bef130ba-->
各反復では、サイズ $k_\textrm{h} \times k_\textrm{w}$ の部分ウィンドウを `X` からスライスし、`K` と要素ごとに掛けて `sum` で縮約する。

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
- **180° 反転カーネル $\mathbf{K}'$：**$\mathbf{K}$ との相互相関は、$[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$ との畳み込みに等しい

<!--en:5c8af7cd kind=slide_text-->
- **Identical learned class:** Frameworks compute cross-correlation and call it convolution.
<!--tr:5c8af7cd-->
- **学習される関数族は同一：**フレームワークは相互相関を計算し、それを畳み込みと呼ぶ

<!--en:a69e29b5 kind=slide_title-->
::: {.slide title="Building a 2D Convolutional Layer Module"}
<!--tr:a69e29b5-->
::: {.slide title="2 次元畳み込み層モジュールの構築"}

<!--en:55763603 kind=kicker-->
[Learnable Kernel Weights Plus Scalar Bias]{.kicker}
<!--tr:55763603-->
[学習可能なカーネル重みとスカラーバイアス]{.kicker}

<!--en:5b8bad24 kind=slide_text-->
A convolutional layer wraps `corr2d` with a learnable $k_\textrm{h} \times k_\textrm{w}$ weight parameter (initialized from $\mathcal{N}(0, 0.01^2)$) and a scalar bias (initialized to $0$):
<!--tr:5b8bad24-->
畳み込み層は `corr2d` をラップし、学習可能な $k_\textrm{h} \times k_\textrm{w}$ の重みパラメータ（$\mathcal{N}(0, 0.01^2)$ から初期化）とスカラーのバイアス（$0$ に初期化）を持たせる。

<!--en:c16d97d6 kind=slide_title-->
::: {.slide title="Resolution Independence of Convolutional Layers"}
<!--tr:c16d97d6-->
::: {.slide title="畳み込み層の解像度非依存性"}

<!--en:a7d1e48f kind=kicker-->
[Fixed Parameters Across Arbitrary Input Sizes]{.kicker}
<!--tr:a7d1e48f-->
[任意の入力サイズに対する固定パラメータ]{.kicker}

<!--en:53a2bce7 kind=slide_text-->
- **Decoupled from $H \times W$:** The exact same 5 parameters ($2 \times 2$ kernel + 1 bias) process $3 \times 3$, $6 \times 8$, or megapixel inputs without reshaping weights.
<!--tr:53a2bce7-->
- **$H \times W$ から独立：**まったく同じ 5 個のパラメータ（$2 \times 2$ カーネルと 1 個のバイアス）で、重みの形状を変えずに $3 \times 3$、$6 \times 8$、さらには数メガピクセルの入力を処理できる

<!--en:017fec24 kind=slide_title-->
::: {.slide title="Constructing a Synthetic Vertical-Edge Image"}
<!--tr:017fec24-->
::: {.slide title="垂直エッジを持つ合成画像の作成"}

<!--en:896e641c kind=kicker-->
[A 6×8 Binary Test Pattern]{.kicker}
<!--tr:896e641c-->
[6×8 の二値テストパターン]{.kicker}

<!--en:0a4b456b kind=slide_text-->
We construct the $6 \times 8$ test image `X` with ones on the left and right borders and zeros across the middle four columns (`X[:, 2:6] = 0`):
<!--tr:0a4b456b-->
$6 \times 8$ のテスト画像 `X` を、左右の端が 1、中央の 4 列が 0 になるように作成する（`X[:, 2:6] = 0`）。

<!--en:dfc7e3e3 kind=slide_text-->
- **Piecewise-constant columns:** Intensity is constant (`1` or `0`) within columns $0\text{--}1$, $2\text{--}5$, and $6\text{--}7$, jumping only between columns $1 \to 2$ and $5 \to 6$.
<!--tr:dfc7e3e3-->
- **区分的に一定な列：**強度（`1` または `0`）は列 $0\text{--}1$、$2\text{--}5$、$6\text{--}7$ の内部で一定であり、列 $1 \to 2$ と $5 \to 6$ の間でのみ変化する

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
`X` と $1 \times 2$ カーネル $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ の相互相関をとると、隣り合うすべての水平ペアに対して $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ が計算される。

<!--en:f649cfae kind=slide_text-->
- **Signed edge map:** Column 1 fires `+1` (white-to-black transition), column 5 fires `-1` (black-to-white transition), and all constant interior columns evaluate to `0`.
<!--tr:f649cfae-->
- **符号付きエッジマップ：**列 1 は `+1`（白から黒への変化）、列 5 は `-1`（黒から白への変化）を出力し、値が一様な内部の列はすべて `0` になる

<!--en:e6d155be kind=slide_title-->
::: {.slide title="Orientation Selectivity of Edge Kernels"}
<!--tr:e6d155be-->
::: {.slide title="エッジ検出カーネルの方向選択性"}

<!--en:b525cde4 kind=kicker-->
[Why One Kernel Cannot Detect All Orientations]{.kicker}
<!--tr:b525cde4-->
[1 つのカーネルですべての方向を検出できない理由]{.kicker}

<!--en:38159de6 kind=slide_text-->
- **Blind to horizontal edges:** Transposing `X` (`X.T`) makes horizontal neighbors identical, so $[1, -1]$ outputs all zeros.
<!--tr:38159de6-->
- **水平エッジには無反応：**`X` を転置（`X.T`）すると水平方向に隣り合う値が等しくなるため、$[1, -1]$ の出力はすべて 0 になる

<!--en:f0f30f76 kind=slide_text-->
- **Filter banks (§{sec_channels}):** Vision layers learn parallel channels for horizontal, vertical, and diagonal orientations.
<!--tr:f0f30f76-->
- **フィルタバンク（§{sec_channels}）：**画像処理の層は、水平・垂直・斜めの各方向に対応する並列チャネルを学習する

<!--en:b755e31b kind=slide_title-->
::: {.slide title="Learning an Edge Detector via Gradient Descent"}
<!--tr:b755e31b-->
::: {.slide title="勾配降下法によるエッジ検出器の学習"}

<!--en:86bbeea7 kind=kicker-->
[Supervising a Built-In Convolutional Layer on (X, Y)]{.kicker}
<!--tr:86bbeea7-->
[(X, Y) を教師とする組み込み畳み込み層の学習]{.kicker}

<!--en:a569780c kind=slide_text-->
- **4D framework tensors:** Reshape `X` and `Y` to 4D tensors (one example, one channel) and minimize $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ at $\eta = 0.03$.
<!--tr:a569780c-->
- **フレームワークの 4 次元テンソル：**`X` と `Y` を 4 次元テンソル（1 例、1 チャネル）に変形し、$\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ を $\eta = 0.03$ で最小化する

<!--en:2489ce6e kind=slide_title-->
::: {.slide title="Inspecting the Learned Kernel Weights"}
<!--tr:2489ce6e-->
::: {.slide title="学習されたカーネル重みの確認"}

<!--en:7e4ef44b kind=kicker-->
[Why Initial Loss Is 12 and Converges to [1, -1]]{.kicker}
<!--tr:7e4ef44b-->
[初期損失が 12 となり [1, -1] へ収束する理由]{.kicker}

<!--en:9cd1c645 kind=slide_text-->
- **Why loss starts at $12.0$:** Near-zero weights predict $\hat{Y} \approx 0$, missing the twelve $\pm 1$ edge pixels in `Y` ($12 \times 1^2 = 12$).
<!--tr:9cd1c645-->
- **損失が $12.0$ から始まる理由：**ゼロに近い重みは $\hat{Y} \approx 0$ と予測し、12 個ある $\pm 1$ のエッジピクセルを `Y` 上で外す（$12 \times 1^2 = 12$）

<!--en:456670c5 kind=slide_text-->
- **Convergence in 9 steps:** SGD recovers $[0.99, -0.99]$ automatically from data.
<!--tr:456670c5-->
- **9 ステップでの収束：**SGD がデータから $[0.99, -0.99]$ を自動的に復元する

=== DECISIONS ===
- "locality" → 「局所性」 (no glossary entry; standard mathematical and ML term)
- "edge detection / edge detector" → 「エッジ検出 / エッジ検出器」 (standard CV terminology)
- "edge map" → 「エッジマップ」 (katakana compound, solid)
- "sliding window" → 「スライディングウィンドウ」 (katakana compound, solid)
- "elementwise" → 「要素ごとに」 (standard array/tensor terminology)
- "dense layer" → 「密結合層」 (kept distinct from "fully connected layer" → 「全結合層」)
- "end to end" → 「エンドツーエンド」 (katakana compound, solid)
- "orientation selectivity" → 「方向選択性」 (standard vision terminology)
- "filter bank" → 「フィルタバンク」 (katakana compound, dropping final ー on フィルタ per style.md §3.3)

=== NOTES ===
- c6371ece: false-match glossary warning on `'MLP' -> 多層パーセプトロン`; the source uses the standalone acronym `MLP`, which stays in Latin script per style.md §3.1–§3.2.
- 67d25b0b: false-match glossary warning on `'CNN' -> 畳み込みニューラルネットワーク`; standalone acronym `CNN` stays in Latin script per style.md §3.2.
- 957dc033: false-match glossary warning on `'CNNs' -> 畳み込みニューラルネットワーク`; standalone acronym `CNN` stays in Latin script per style.md §3.2.
- 456670c5: false-match glossary warning on `'SGD' -> 確率的勾配降下法`; standalone acronym `SGD` stays in Latin script per style.md §3.2.
