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
:numref:`sec_why-conv` では、平行移動同変性（translation equivariance）と局所性（locality）という 2 つの幾何学的原理から、畳み込み層（convolutional layer）を理論的に導出しました。ここでは、その式が画像に対して実際に何を計算しているのか、そして勾配降下法（gradient descent）によって有用なカーネルをデータから自動的に見つけ出せるのかを見ていきます。

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
明るい領域と暗い領域が接する垂直エッジを見つける問題を考えましょう。各ピクセルから右隣のピクセルを引く（$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$）と、一様な領域の内部では 0 になり、白から黒へ移る箇所では $+1$、黒から白へ戻る箇所では $-1$ が得られます。2 つの重みからなる検出器 $(1, -1)$ を画像上のすべての有効な位置にわたって走査すると、*エッジマップ*が得られます。本節では、2 次元の相互相関（cross-correlation）を実装し、手動で設計したエッジ検出器を検証したうえで、勾配降下法によりデータから直接 $(1, -1)$ を*学習*させます。さらに、畳み込みを 1 回の行列積（*im2col*）として表す方法を示し、層を積み重ねることでユニットの受容野（receptive field）がどのように広がるかを導出します。

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
ここではいったんチャネルを脇に置き、入力 $\mathbf{X}$ を $n_\textrm{h} \times n_\textrm{w}$ 行列、カーネル $\mathbf{K}$ を $k_\textrm{h} \times k_\textrm{w}$ 行列とします。カーネルは形状 $k_\textrm{h} \times k_\textrm{w}$ のスライドする*ウィンドウ*を定めます。このウィンドウの左上隅を $(i, j)$ に置き、覆われた入力パッチと $\mathbf{K}$ の要素ごとの積を足し合わせることで、位置 $(i, j)$ の出力要素が得られます。

<!--en:3688b38b kind=paragraph-->
This is the *two-dimensional cross-correlation* of $\mathbf{X}$ with
$\mathbf{K}$, written $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ with
zero-based indices matching code. Whereas :numref:`sec_why-conv` centered
offsets symmetrically from $-\Delta$ to $\Delta$, anchoring at the upper-left
corner differs only by a fixed shift of $\Delta$ while naturally handling
even-sized kernels (such as a $2 \times 2$ kernel or our $1 \times 2$ edge
detector) that have no center pixel.
<!--tr:3688b38b-->
これが $\mathbf{X}$ と $\mathbf{K}$ の *2 次元相互相関*であり、コードに合わせた 0 始まりの添字を用いて $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ と書きます。:numref:`sec_why-conv` ではオフセットを $-\Delta$ から $\Delta$ まで対称に配置しましたが、左上隅を基準にする定義は $\Delta$ だけの一定のずれがあるにすぎず、中心ピクセルを持たない偶数サイズのカーネル（$2 \times 2$ カーネルや $1 \times 2$ エッジ検出器など）も自然に扱えます。

<!--en:fa1b7da9 kind=caption-->
![Two-dimensional cross-correlation. The shaded portions are the first output element and the input and kernel elements used to compute it: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
<!--tr:fa1b7da9-->
![2 次元相互相関。網掛け部分は最初の出力要素と、その計算に使われる入力要素およびカーネル要素を示す（$0\times0+1\times1+3\times2+4\times3=19$）。](../img/correlation.svg)

<!--en:2f8e5deb kind=paragraph-->
In :numref:`fig_correlation`, we slide a $2 \times 2$ kernel across a
$3 \times 3$ input one pixel at a time, left to right and top to bottom. The
window fits cleanly at four positions, yielding the $2 \times 2$ output:
<!--tr:2f8e5deb-->
:numref:`fig_correlation` では、$2 \times 2$ カーネルを $3 \times 3$ の入力上で左から右、上から下へ 1 ピクセルずつスライドさせます。ウィンドウは 4 箇所にぴったり収まり、次の $2 \times 2$ の出力が得られます。

<!--en:4741e012 kind=paragraph-->
Notice that the output shrinks slightly along each axis because the kernel must
stay wholly inside the input. Fitting a width-$k$ window into length $n$ leaves
$n - k + 1$ valid positions, so the output shape is
<!--tr:4741e012-->
カーネルは入力の内部に完全に収まる必要があるため、出力は各軸方向にわずかに小さくなります。幅 $k$ のウィンドウを長さ $n$ の範囲に収めると有効な位置は $n - k + 1$ 箇所となるため、出力形状は次のようになります。

<!--en:12d81c14 kind=paragraph-->
Thus a $3 \times 3$ input with a $2 \times 2$ kernel yields $2 \times 2$, a
$6 \times 8$ image with a $1 \times 2$ kernel yields $6 \times 7$, and a
$28 \times 28$ Fashion-MNIST image with a $5 \times 5$ kernel yields
$24 \times 24$. Whenever we want to preserve spatial dimensions, we surround the
input with a border of zeros (*padding*), which we examine alongside strides and
dilation in :numref:`sec_padding`.
<!--tr:12d81c14-->
したがって、$3 \times 3$ の入力に $2 \times 2$ カーネルを適用すると $2 \times 2$、$6 \times 8$ の画像に $1 \times 2$ カーネルを適用すると $6 \times 7$、$28 \times 28$ の Fashion-MNIST 画像に $5 \times 5$ カーネルを適用すると $24 \times 24$ になります。空間次元を保ちたいときは、入力の周囲を 0 の枠で囲みます（*パディング*）。これについては、ストライドやダイレーションとあわせて :numref:`sec_padding` で詳しく見ていきます。

<!--en:869d223f kind=heading-->
### Implementation
<!--tr:869d223f-->
### 実装

<!--en:6a869f7c kind=paragraph-->
We implement :eqref:`eq_corr2d` directly in `corr2d` using a pair of loops over
every output coordinate.
<!--tr:6a869f7c-->
:eqref:`eq_corr2d` は、すべての出力座標を回る二重ループを用いて関数 `corr2d` としてそのまま実装できます。

<!--en:81e1f383 kind=paragraph-->
To verify `corr2d` against our hand calculation in :numref:`fig_correlation`,
we construct the $3 \times 3$ input tensor `X` and $2 \times 2$ kernel `K` and
evaluate `corr2d(X, K)`.
<!--tr:81e1f383-->
`corr2d` の動作を :numref:`fig_correlation` の手計算と照合するため、$3 \times 3$ の入力テンソル `X` と $2 \times 2$ のカーネル `K` を作成し、`corr2d(X, K)` を評価します。

<!--en:f567e1b2 kind=paragraph-->
The output reproduces :numref:`fig_correlation` entry for entry (and as shown in
:numref:`sec_why-conv`, strict mathematical convolution with a flipped kernel
$\mathbf{K}'$ learns the exact same family of layers, so we follow standard
deep learning terminology and call cross-correlation "convolution").
<!--tr:f567e1b2-->
出力は :numref:`fig_correlation` のすべての要素を正確に再現しています（また :numref:`sec_why-conv` で示したように、反転したカーネル $\mathbf{K}'$ を用いる厳密な数学的畳み込みもまったく同じ層の族を学習するため、深層学習の標準的な用語法に従い、相互相関を「畳み込み」と呼びます）。

<!--en:bfe11c62 kind=heading-->
## Convolutional Layers
<!--tr:bfe11c62-->
## 畳み込み層

<!--en:2cdc020d kind=heading-->
### The Layer
<!--tr:2cdc020d-->
### 層

<!--en:c6371ece kind=paragraph-->
A two-dimensional convolutional layer cross-correlates its input with a learned
kernel and adds a scalar bias. Just like our MLP weights in
:numref:`chap_perceptrons`, we initialize the kernel from
$\mathcal{N}(0, 0.01^2)$ and the bias to zero. Below, `Conv2D` registers both
parameters and invokes `corr2d` in its forward pass; a layer with a kernel of
height $h$ and width $w$ is called an $h \times w$ convolutional layer.
<!--tr:c6371ece-->
2 次元畳み込み層は、入力と学習対象のカーネルとの相互相関を計算し、スカラーのバイアスを加えます。:numref:`chap_perceptrons` の MLP の重みと同様に、カーネルは $\mathcal{N}(0, 0.01^2)$ から初期化し、バイアスは 0 で初期化します。以下の `Conv2D` はこれら 2 つのパラメータを登録し、順伝播（forward pass）で `corr2d` を呼び出します。高さ $h$、幅 $w$ のカーネルを持つ層は、$h \times w$ 畳み込み層と呼ばれます。

<!--en:29cd9061 kind=paragraph-->
Unlike a dense layer, a convolutional layer is not locked to a fixed input
resolution because its parameters depend only on the kernel size. The exact same
$2 \times 2$ layer (5 parameters total) processes both our $3 \times 3$ tensor
`X` and a $6 \times 8$ image, returning outputs of shapes $2 \times 2$ and
$5 \times 7$:
<!--tr:29cd9061-->
全結合層とは異なり、畳み込み層のパラメータはカーネルサイズのみに依存するため、固定された入力解像度に縛られません。まったく同じ $2 \times 2$ 層（パラメータは合計 5 個）で、$3 \times 3$ テンソル `X` と $6 \times 8$ 画像のどちらも処理でき、出力の形状はそれぞれ $2 \times 2$ と $5 \times 7$ になります。

<!--en:c9e5aa07 kind=heading-->
### Edge Detection
<!--tr:c9e5aa07-->
### エッジ検出

<!--en:3d9a357b kind=paragraph-->
We now test the vertical edge detector from the introduction on a
$6 \times 8$ synthetic image whose middle four columns are black (`0`) and outer
columns are white (`1`).
<!--tr:3d9a357b-->
それでは、冒頭で述べた垂直エッジ検出器を $6 \times 8$ の合成画像で試してみましょう。この画像は中央の 4 列が黒（`0`）、外側の列が白（`1`）になっています。

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
各位置 $(i, j)$ で、このカーネルは水平方向の*有限差分*（finite difference）$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ を計算します。これは輝度の水平方向導関数の（符号を除いた）離散近似であり、一様な領域では 0 になり、輝度が急変する箇所で大きな値を示します。

<!--en:e810b3a9 kind=paragraph-->
Cross-correlating the $6 \times 8$ stripe image `X` with the $1 \times 2$
horizontal-difference kernel `K` via `corr2d(X, K)` produces the $6 \times 7$
edge map `Y`:
<!--tr:e810b3a9-->
$6 \times 8$ の縞模様画像 `X` と $1 \times 2$ の水平差分カーネル `K` の相互相関を `corr2d(X, K)` で計算すると、$6 \times 7$ のエッジマップ `Y` が得られます。

<!--en:568358e4 kind=paragraph-->
Sure enough, `Y` fires `+1` along the white-to-black boundary, `-1` along the
black-to-white boundary, and `0` everywhere else. Applying the exact same kernel
to the transposed image (turning the vertical stripe into a horizontal one)
yields all zeros:
<!--tr:568358e4-->
確かに、`Y` は白から黒への境界で `+1`、黒から白への境界で `-1` を出力し、それ以外の箇所はすべて `0` になっています。まったく同じカーネルを転置した画像（垂直の縞を水平の縞に変えたもの）に適用すると、出力はすべて 0 になります。

<!--en:71c591a4 kind=paragraph-->
Clearly `[1, -1]` responds strictly to vertical edges. Detecting horizontal
edges, diagonals, corners, or textures requires different kernels altogether.
<!--tr:71c591a4-->
このように、`[1, -1]` は垂直エッジのみに反応します。水平エッジ、斜めのエッジ、角、テクスチャを検出するには、まったく別のカーネルが必要です。

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
欲しい特徴がはっきりわかっているとき、`[1, -1]` を手で設計するのは簡単ですが、深いネットワーク全体にわたる何千もの中間フィルタを人手で指定することはできません。勾配降下法によって、カーネル `[1, -1]` を入出力ペア `(X, Y)` だけから復元できるでしょうか。ここでは、各フレームワークの組み込み 2 次元畳み込み層（バイアスなし、重みは $\mathcal{N}(0, 0.01^2)$ で初期化）を使い、42 個すべての出力要素にわたる二乗誤差（squared error）の和を、10 ステップの勾配更新と学習率（learning rate）$0.03$ で最小化して検証します。組み込み層はバッチ軸とチャネル軸を含む 4 次元テンソルを要求する（:numref:`sec_channels`）ため、`X` と `Y` をそれに合わせて変形します。

<!--en:c29c5d61 kind=paragraph-->
Initially the loss sits near 12 (the near-zero random kernel misses all 12
$\pm 1$ edge pixels by roughly 1), and within just nine steps it drops below
0.01. Inspecting the learned kernel confirms what it found:
<!--tr:c29c5d61-->
初期の損失は約 12 であり（0 に近いランダムなカーネルは 12 個の $\pm 1$ エッジピクセルすべてを約 1 だけ外します）、わずか 9 ステップで 0.01 未満まで低下します。学習されたカーネルの中身を確認してみましょう。

<!--en:67d25b0b kind=paragraph-->
Both entries land within roughly 0.02 of `[1, -1]`. Nothing in this loop knew
anything about edges: given a loss signal, gradient descent discovered the
filter on its own. Here we supervised the single layer directly with `Y`; in a
deep CNN, backpropagation learns every layer's kernels end to end from the final
task loss alone.
<!--tr:67d25b0b-->
どちらの要素も約 0.02 の誤差で `[1, -1]` に近づいています。このループにはエッジに関する知識は何も組み込まれていませんが、損失の信号だけを手がかりに、勾配降下法が自らフィルタを発見しました。ここでは単一の層を `Y` で直接教師あり学習させましたが、深い CNN では、誤差逆伝播法（backpropagation）が最終的なタスクの損失だけからすべての層のカーネルをエンドツーエンドで学習します。

<!--en:91b0010d kind=slide_text-->
Convolutions for Images<br>**2D cross-correlation · edge detection · learned kernels · im2col · receptive fields**
<!--tr:91b0010d-->
画像に対する畳み込み<br>**2 次元相互相関・エッジ検出・カーネルの学習・im2col・受容野**

<!--en:1d94295a kind=slide_title-->
::: {.slide title="Detecting Local Edges Across an Image"}
<!--tr:1d94295a-->
::: {.slide title="画像全体にわたる局所エッジの検出"}

<!--en:687ed302 kind=kicker-->
[Motivation & Core Mental Model]{.kicker}
<!--tr:687ed302-->
[動機と基本イメージ]{.kicker}

<!--en:05fceed0 kind=slide_text-->
- **Synthetic $6 \times 8$ test image:** White outer columns (`1`) surrounding a 4-column black vertical band (`0`).
<!--tr:05fceed0-->
- **$6 \times 8$ の合成テスト画像：**白の外側列（`1`）が 4 列の黒い垂直帯（`0`）を挟む構成

<!--en:71e700b3 kind=slide_text-->
- **Horizontal finite difference:** Sweeping $\mathbf{K} = [1, -1]$ computes $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, firing `+1` at white-to-black and `-1` at black-to-white edges.
<!--tr:71e700b3-->
- **水平方向の有限差分：**$\mathbf{K} = [1, -1]$ を走査すると $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ を計算し、白から黒の境界で `+1`、黒から白の境界で `-1` を出力

<!--en:957dc033 kind=slide_text-->
- **Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent.
<!--tr:957dc033-->
- **学習と手動設計：**深い CNN は勾配降下法でこうしたカーネルを何千個も自動発見

<!--en:cb1045e1 kind=slide_title-->
::: {.slide title="The 2D Cross-Correlation Operation"}
<!--tr:cb1045e1-->
::: {.slide title="2 次元相互相関演算"}

<!--en:17bcece5 kind=kicker-->
[Sliding-Window Inner Products]{.kicker}
<!--tr:17bcece5-->
[スライドウィンドウによる内積]{.kicker}

<!--en:8d45aae8 kind=slide_text-->
Given an $n_\textrm{h} \times n_\textrm{w}$ input matrix $\mathbf{X}$ and a $k_\textrm{h} \times k_\textrm{w}$ kernel $\mathbf{K}$, the 2D cross-correlation $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ places the window's top-left corner at $(i, j)$ and sums elementwise products:
<!--tr:8d45aae8-->
$n_\textrm{h} \times n_\textrm{w}$ 入力行列 $\mathbf{X}$ と $k_\textrm{h} \times k_\textrm{w}$ カーネル $\mathbf{K}$ に対し、2 次元相互相関 $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ はウィンドウの左上隅を $(i, j)$ に置いて要素ごとの積を足し合わせる。

<!--en:e484380c kind=slide_text-->
- **Top-left anchor:** Zero-based indices $a, b \ge 0$ match array indexing in code and handle even-sized kernels (like $2 \times 2$ or $1 \times 2$) cleanly.
<!--tr:e484380c-->
- **左上基準：**0 始まりの添字 $a, b \ge 0$ はコードの配列添字と一致し、偶数サイズのカーネル（$2 \times 2$ や $1 \times 2$ など）も自然に扱える

<!--en:137c0980 kind=slide_text-->
- **Shaded top-left output $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.
<!--tr:137c0980-->
- **網掛けされた左上出力 $(0,0)$：**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$

<!--en:575c4aea kind=slide_title-->
::: {.slide title="Valid Window Positions and Output Spatial Shape"}
<!--tr:575c4aea-->
::: {.slide title="有効なウィンドウ位置と出力の空間形状"}

<!--en:7eda4531 kind=kicker-->
[Why Unpadded Convolutions Shrink the Grid]{.kicker}
<!--tr:7eda4531-->
[パディングなしの畳み込みで出力が縮小する理由]{.kicker}

<!--en:891243ae kind=slide_text-->
- **Valid window placements:** A $k_\textrm{h} \times k_\textrm{w}$ window fits inside an $n_\textrm{h} \times n_\textrm{w}$ grid at $n - k + 1$ positions per axis.
<!--tr:891243ae-->
- **有効なウィンドウ配置：**$k_\textrm{h} \times k_\textrm{w}$ ウィンドウは $n_\textrm{h} \times n_\textrm{w}$ 格子内に各軸 $n - k + 1$ 箇所収まる

<!--en:55cc8c32 kind=slide_text-->
- **Shape examples:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.
<!--tr:55cc8c32-->
- **形状の例：**$3 \times 3 \star 2 \times 2 \to 2 \times 2$（$19, 25, 37, 43$）、$28 \times 28 \star 5 \times 5 \to 24 \times 24$

<!--en:1e3463da kind=slide_title-->
::: {.slide title="Implementing 2D Cross-Correlation in Code"}
<!--tr:1e3463da-->
::: {.slide title="2 次元相互相関の実装"}

<!--en:07566992 kind=kicker-->
[Direct Nested-Loop Implementation]{.kicker}
<!--tr:07566992-->
[二重ループによる直接実装]{.kicker}

<!--en:202aa5d3 kind=slide_text-->
The function `corr2d` allocates the $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ output tensor and computes each entry as the sum of `X[i:i+h, j:j+w] * K`:
<!--tr:202aa5d3-->
関数 `corr2d` は $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ の出力テンソルを確保し、各要素を `X[i:i+h, j:j+w] * K` の総和として計算する。

<!--en:bef130ba kind=slide_text-->
Each iteration slices a $k_\textrm{h} \times k_\textrm{w}$ sub-window of `X`, multiplies elementwise with `K`, and reduces via `sum`.
<!--tr:bef130ba-->
各反復で $k_\textrm{h} \times k_\textrm{w}$ の部分ウィンドウを `X` からスライスし、`K` と要素ごとに乗算して `sum` で集約する。

<!--en:4ee9f6b1 kind=slide_title-->
::: {.slide title="Verifying corr2d and the Flipped-Kernel Equivalence"}
<!--tr:4ee9f6b1-->
::: {.slide title="corr2d の検証と反転カーネルの等価性"}

<!--en:2b0be423 kind=kicker-->
[Cross-Correlation vs. Mathematical Convolution]{.kicker}
<!--tr:2b0be423-->
[相互相関と数学的畳み込みの比較]{.kicker}

<!--en:74b9b745 kind=slide_text-->
- **180° flipped kernel $\mathbf{K}'$:** Cross-correlating with $\mathbf{K}$ equals convolving with $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.
<!--tr:74b9b745-->
- **180° 反転カーネル $\mathbf{K}'$：**$\mathbf{K}$ との相互相関は $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$ との畳み込みに等しい

<!--en:5c8af7cd kind=slide_text-->
- **Identical learned class:** Frameworks compute cross-correlation and call it convolution.
<!--tr:5c8af7cd-->
- **同一の学習対象：**フレームワークは相互相関を計算し、それを畳み込みと呼ぶ

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
畳み込み層は `corr2d` に、学習可能な $k_\textrm{h} \times k_\textrm{w}$ の重みパラメータ（$\mathcal{N}(0, 0.01^2)$ で初期化）とスカラーのバイアス（$0$ で初期化）を組み合わせたもの。

<!--en:c16d97d6 kind=slide_title-->
::: {.slide title="Resolution Independence of Convolutional Layers"}
<!--tr:c16d97d6-->
::: {.slide title="畳み込み層の解像度独立性"}

<!--en:a7d1e48f kind=kicker-->
[Fixed Parameters Across Arbitrary Input Sizes]{.kicker}
<!--tr:a7d1e48f-->
[任意の入力サイズに共通のパラメータ]{.kicker}

<!--en:53a2bce7 kind=slide_text-->
- **Decoupled from $H \times W$:** The exact same 5 parameters ($2 \times 2$ kernel + 1 bias) process $3 \times 3$, $6 \times 8$, or megapixel inputs without reshaping weights.
<!--tr:53a2bce7-->
- **$H \times W$ からの独立性：**まったく同じ 5 個のパラメータ（$2 \times 2$ カーネル + 1 バイアス）が、重みの変形なしに $3 \times 3$、$6 \times 8$、さらにはメガピクセル級の入力も処理可能

<!--en:017fec24 kind=slide_title-->
::: {.slide title="Constructing a Synthetic Vertical-Edge Image"}
<!--tr:017fec24-->
::: {.slide title="合成垂直エッジ画像の作成"}

<!--en:896e641c kind=kicker-->
[A 6×8 Binary Test Pattern]{.kicker}
<!--tr:896e641c-->
[6×8 の 2 値テストパターン]{.kicker}

<!--en:0a4b456b kind=slide_text-->
We construct the $6 \times 8$ test image `X` with ones on the left and right borders and zeros across the middle four columns (`X[:, 2:6] = 0`):
<!--tr:0a4b456b-->
$6 \times 8$ のテスト画像 `X` を、左右の端が 1、中央の 4 列が 0（`X[:, 2:6] = 0`）となるように作成する。

<!--en:dfc7e3e3 kind=slide_text-->
- **Piecewise-constant columns:** Intensity is constant (`1` or `0`) within columns $0\text{--}1$, $2\text{--}5$, and $6\text{--}7$, jumping only between columns $1 \to 2$ and $5 \to 6$.
<!--tr:dfc7e3e3-->
- **区分定数の列：**輝度は一定値（`1` または `0`）を列 $0\text{--}1$、$2\text{--}5$、$6\text{--}7$ 内で保ち、列 $1 \to 2$ と $5 \to 6$ の間でのみ急変

<!--en:3ce31cbe kind=slide_title-->
::: {.slide title="Applying the Horizontal Finite-Difference Kernel"}
<!--tr:3ce31cbe-->
::: {.slide title="水平有限差分カーネルの適用"}

<!--en:b37a1029 kind=kicker-->
[Detecting Vertical Edges with K = [1, -1]]{.kicker}
<!--tr:b37a1029-->
[K = [1, -1] による垂直エッジ検出]{.kicker}

<!--en:9be85019 kind=slide_text-->
Cross-correlating `X` with the $1 \times 2$ kernel $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ computes $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ at every horizontal pair:
<!--tr:9be85019-->
`X` と $1 \times 2$ カーネル $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ の相互相関により、すべての水平ペアで $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ を計算する。

<!--en:f649cfae kind=slide_text-->
- **Signed edge map:** Column 1 fires `+1` (white-to-black transition), column 5 fires `-1` (black-to-white transition), and all constant interior columns evaluate to `0`.
<!--tr:f649cfae-->
- **符号付きエッジマップ：**列 1 は `+1`（白から黒への変化）、列 5 は `-1`（黒から白への変化）を出力し、一様な内部列はすべて `0`

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
- **水平エッジに無反応：**`X` を転置（`X.T`）すると水平方向の隣接ピクセルが等しくなり、$[1, -1]$ の出力はすべて 0

<!--en:f0f30f76 kind=slide_text-->
- **Filter banks (§{sec_channels}):** Vision layers learn parallel channels for horizontal, vertical, and diagonal orientations.
<!--tr:f0f30f76-->
- **フィルタバンク（§{sec_channels}）：**画像処理の層は、水平・垂直・斜め方向を並列チャネルで学習

<!--en:b755e31b kind=slide_title-->
::: {.slide title="Learning an Edge Detector via Gradient Descent"}
<!--tr:b755e31b-->
::: {.slide title="勾配降下法によるエッジ検出器の学習"}

<!--en:86bbeea7 kind=kicker-->
[Supervising a Built-In Convolutional Layer on (X, Y)]{.kicker}
<!--tr:86bbeea7-->
[(X, Y) による組み込み畳み込み層の教師あり学習]{.kicker}

<!--en:a569780c kind=slide_text-->
- **4D framework tensors:** Reshape `X` and `Y` to 4D tensors (one example, one channel) and minimize $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ at $\eta = 0.03$.
<!--tr:a569780c-->
- **4 次元のフレームワークテンソル：**`X` と `Y` を 4 次元テンソル（1 例、1 チャネル）に変形し、$\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ を $\eta = 0.03$ で最小化

<!--en:2489ce6e kind=slide_title-->
::: {.slide title="Inspecting the Learned Kernel Weights"}
<!--tr:2489ce6e-->
::: {.slide title="学習されたカーネル重みの確認"}

<!--en:7e4ef44b kind=kicker-->
[Why Initial Loss Is 12 and Converges to [1, -1]]{.kicker}
<!--tr:7e4ef44b-->
[初期損失が 12 となり [1, -1] に収束する理由]{.kicker}

<!--en:9cd1c645 kind=slide_text-->
- **Why loss starts at $12.0$:** Near-zero weights predict $\hat{Y} \approx 0$, missing the twelve $\pm 1$ edge pixels in `Y` ($12 \times 1^2 = 12$).
<!--tr:9cd1c645-->
- **初期損失が $12.0$ である理由：**0 に近い重みは $\hat{Y} \approx 0$ を予測し、12 個の $\pm 1$ エッジピクセル（`Y` 内）を外すため（$12 \times 1^2 = 12$）

<!--en:456670c5 kind=slide_text-->
- **Convergence in 9 steps:** SGD recovers $[0.99, -0.99]$ automatically from data.
<!--tr:456670c5-->
- **9 ステップで収束：**SGD がデータから $[0.99, -0.99]$ を自動復元

=== DECISIONS ===
- "locality" → 「局所性」 (no glossary entry; standard in Japanese CNN literature)
- "edge detector" → 「エッジ検出器」 (no glossary entry; standard)
- "edge map" → 「エッジマップ」 (no glossary entry; standard)
- "window" → 「ウィンドウ」 (no glossary entry; standard)
- "patch" → 「パッチ」 (no glossary entry; standard)
- "dense layer" → 「全結合層」 (synonym of fully connected layer)
- "end to end" → 「エンドツーエンド」 (no glossary entry; standard)
- "filter bank" → 「フィルタバンク」 (no glossary entry; standard)

=== NOTES ===
- c6371ece: glossary warning for 'MLP' is a false match; acronym kept in English per style.md §3.2.
- 67d25b0b: glossary warning for 'CNN' is a false match; acronym kept in English per style.md §3.2.
- 957dc033: glossary warning for 'CNNs' is a false match; acronym kept in English per style.md §3.2.
- 456670c5: glossary warning for 'SGD' is a false match; acronym kept in English per style.md §3.2.
