<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u03 lang=ja rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:2294243a kind=heading-->
## Exercises
<!--tr:2294243a-->
## 演習問題

<!--en:6f4ddec6 kind=exercise-->
1. [code] **Hand-designed kernels.** Design some kernels manually.
    1. Given a directional vector $\mathbf{v} = (v_1, v_2)$, derive an
       edge-detection kernel that detects edges orthogonal to $\mathbf{v}$,
       i.e., edges in the direction $(v_2, -v_1)$.
    1. Derive a finite difference operator for the second derivative.
       Which structures in images respond most strongly to it?
    1. What is the minimum size of a kernel to obtain a derivative of
       order $d$?
    1. How would you design a blur kernel? Why might you want to use such
       a kernel?
    1. Load a photograph, convert it to one channel, and downsample it to
       about $100 \times 100$ pixels (`corr2d` loops in Python). Apply the
       edge detector, the blur kernel, and the sharpening kernel "twice
       the identity minus the blur", and display each result next to the
       original. What does the edge map of a photograph show that the
       edge map of the black band could not?

    *Adapted from Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    ConvolutionalNetworks notebook.*
<!--tr:6f4ddec6-->
1. [code] **手動で設計するカーネル。**いくつかのカーネルを手動で設計してください。
    1. 方向ベクトル $\mathbf{v} = (v_1, v_2)$ が与えられたとき、$\mathbf{v}$ に直交するエッジ（すなわち方向 $(v_2, -v_1)$ のエッジ）を検出するエッジ検出カーネルを導出してください。
    1. 二階導関数に対する有限差分演算子を導出してください。画像中のどのような構造がそれに最も強く反応しますか。
    1. $d$ 階導関数を得るために必要なカーネルの最小サイズはいくつですか。
    1. ぼかしカーネルはどのように設計しますか。また、そのようなカーネルを使いたくなるのはなぜですか。
    1. 写真を読み込み、1 チャネルに変換して、約 $100 \times 100$ ピクセルにダウンサンプリングしてください（`corr2d` は Python のループで動くためです）。エッジ検出器、ぼかしカーネル、および「恒等変換の 2 倍からぼかしを引いた」鮮鋭化カーネルをそれぞれ適用し、各結果を元の画像と並べて表示してください。写真のエッジマップからは、黒い帯のエッジマップではわからなかったどのようなことがわかりますか。

    *Stanford CS231n、[Assignment 2](https://cs231n.github.io/assignments2024/assignment2/)、ConvolutionalNetworks ノートブックより改変。*

<!--en:ec8621f8 kind=exercise-->
2. [code] **Convolution as a banded matrix.** The patch matrix of
   :numref:`subsec_conv_matmul` unfolds the input. The opposite
   construction unrolls the kernel. Let $\mathbf{X}$ be a $4 \times 4$
   input, $\mathbf{K}$ a $3 \times 3$ kernel, and $\textrm{vec}(\cdot)$
   the row-by-row flattening.
    1. Construct the $4 \times 16$ matrix $\mathbf{W}$ with
       $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$,
       where $\mathbf{Y}$ is the cross-correlation of $\mathbf{X}$ with
       $\mathbf{K}$. How many nonzero entries does each row have, and how
       many distinct values appear in $\mathbf{W}$?
    1. Build $\mathbf{W}$ in code from `K` and confirm that the product
       reproduces `corr2d(X, K)`.
    1. For an $n \times n$ input and a $k \times k$ kernel, count the
       entries of $\mathbf{W}$ stored as a dense array, the nonzero
       entries of $\mathbf{W}$, and the entries of the patch matrix. How
       does each count grow with $n$?
    1. Keep the sparsity pattern of $\mathbf{W}$ but give every row its
       own $k^2$ values. The result is a locally connected layer: the
       fourth-order tensor $\mathsf{V}$ of :numref:`sec_why-conv` in
       $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$,
       restricted to a $k \times k$ window of offsets $(a, b)$ before
       translation equivariance ties the weights. For an $n \times n$
       input, count the parameters of this layer, of a fully connected
       layer with the same input and output shapes, and of the
       convolution.

    *Adapted from CMU 11-785,
    [Homework 2 Part 1](https://deeplearning.cs.cmu.edu/S21/index.html).*
<!--tr:ec8621f8-->
2. [code] **帯行列としての畳み込み。**:numref:`subsec_conv_matmul` のパッチ行列は入力を展開するものでした。逆の構成として、カーネルを展開することもできます。$\mathbf{X}$ を $4 \times 4$ の入力、$\mathbf{K}$ を $3 \times 3$ のカーネルとし、$\textrm{vec}(\cdot)$ を行ごとの平坦化とします。
    1. $4 \times 16$ 行列 $\mathbf{W}$ を $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$ を満たすように構築してください。ここで、$\mathbf{Y}$ は $\mathbf{X}$ と $\mathbf{K}$ の相互相関です。各行には非零要素がいくつあり、$\mathbf{W}$ 全体には異なる値がいくつ現れますか。
    1. $\mathbf{W}$ をコード上で `K` から構築し、その積が `corr2d(X, K)` を再現することを確認してください。
    1. $n \times n$ の入力と $k \times k$ のカーネルに対して、$\mathbf{W}$ を密配列として格納した場合の要素数、$\mathbf{W}$ の非零要素数、およびパッチ行列の要素数を数えてください。それぞれの個数は $n$ に対してどのように増加しますか。
    1. $\mathbf{W}$ のスパース性のパターンは保ったまま、各行に独自の $k^2$ 個の値を与えます。得られるのは局所結合層であり、4 階テンソル $\mathsf{V}$（:numref:`sec_why-conv` の $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$ に現れるもの）を、平行移動同変性によって重みを共有する前の段階で $k \times k$ ウィンドウのオフセット $(a, b)$ に制限したものにあたります。$n \times n$ の入力に対して、この層、同じ入出力形状を持つ全結合層、および畳み込みのパラメータ数をそれぞれ数えてください。

    *CMU 11-785、[Homework 2 Part 1](https://deeplearning.cs.cmu.edu/S21/index.html) より改変。*

<!--en:54bcee26 kind=exercise-->
3. [code] **Effective receptive field.** ● The theoretical receptive
   field $r$ of this section bounds which inputs *can* influence an
   output; the effective receptive field measures how much
   they actually do :cite:`Luo.Li.Urtasun.ea.2016`. Build a stack of $L$
   single-channel $3 \times 3$ convolutional layers with no
   padding, no bias, and no nonlinearity on a $32 \times 32$ input, using
   the built-in layer of :numref:`subsec_learning_kernel`. Draw every
   kernel entry independently from a standard normal distribution.
    1. For $L = 3$ and a single draw of the kernels, backpropagate a unit
       gradient from an output unit near the center of the last layer to
       the input and plot the magnitude of the input gradient. Compare its
       support with the $7 \times 7$ theoretical receptive field, and say
       where the magnitude is largest.
    1. Average the squared input gradient over 500 draws of the kernels,
       for $L = 3$ and for $L = 10$. Sum each averaged map over its rows
       and normalize the resulting column profile to sum to one. How does
       the ratio of the profile's standard deviation to the theoretical
       side length $2L + 1$ change with depth?
    1. :numref:`fig_conv_rf_paths` counts the paths for $L = 3$ in one
       dimension. Show that for general $L$ the expected squared input
       gradient, summed over rows, is proportional to the convolution of
       $L$ copies of the vector $(1, 1, 1)$. Derive how the standard
       deviation of the normalized profile grows with $L$.
<!--tr:54bcee26-->
3. [code] **有効受容野。** ● 本節の理論的受容野 $r$ は出力に*影響を与えうる*入力の範囲を定めるのに対し、有効受容野は各入力が実際にどれだけ影響を与えるかを測ります :cite:`Luo.Li.Urtasun.ea.2016`。$L$ 層の単一チャネル $3 \times 3$ 畳み込み層（パディングなし、バイアスなし、非線形性なし）を $32 \times 32$ の入力に対して :numref:`subsec_learning_kernel` の組み込み層で構築し、カーネルの各要素を標準正規分布から独立にサンプリングしてください。
    1. $L = 3$ としてカーネルを 1 回サンプリングし、最終層の中央付近にある 1 つの出力ユニットから入力へ単位勾配を逆伝播して、入力勾配の大きさをプロットしてください。その台（非零となる範囲）を $7 \times 7$ の理論的受容野と比較し、大きさが最大となる位置を述べてください。
    1. カーネルのサンプリングを 500 回繰り返して入力勾配の二乗を平均し、$L = 3$ と $L = 10$ のそれぞれについて求めてください。各平均マップを行方向に足し合わせ、得られた列プロファイルを総和が 1 になるよう正規化してください。プロファイルの標準偏差と理論上の 1 辺の長さ $2L + 1$ の比は、深さとともにどのように変化しますか。
    1. :numref:`fig_conv_rf_paths` は 1 次元で $L = 3$ の場合の経路数を数えています。一般の $L$ に対して、行方向に足し合わせた入力勾配の二乗の期待値が、$L$ 個のベクトル $(1, 1, 1)$ の畳み込みに比例することを示してください。また、正規化されたプロファイルの標準偏差が $L$ とともにどのように増加するかを導出してください。

<!--en:6ab5b67c kind=exercise-->
4. [code] **Backward pass of a convolution.** ● Let $\mathbf{Y}$ be the
   single-channel cross-correlation of an input $\mathbf{X}$ with a kernel
   $\mathbf{K}$, as computed by `corr2d`, and let $\ell$ be a scalar loss
   that depends on $\mathbf{X}$ and $\mathbf{K}$ only through $\mathbf{Y}$.
    1. Derive $\partial \ell/\partial \mathbf{K}$ and
       $\partial \ell/\partial \mathbf{X}$ in terms of
       $\partial \ell/\partial \mathbf{Y}$, in the style of the
       backpropagation equations of :numref:`sec_backprop`.
    1. Implement your expression for $\partial \ell/\partial \mathbf{K}$
       without automatic differentiation for the loss
       $\ell = \sum_{i,j} G_{ij} Y_{ij}$, where $\mathbf{G}$ is a fixed
       random matrix of the shape of $\mathbf{Y}$. Compare it with a
       centered finite-difference gradient with step size $10^{-2}$ on a
       small random `X` and `K`, and report the largest relative error.
    1. Check your expression for $\partial \ell/\partial \mathbf{K}$
       against the kernel gradient that automatic differentiation
       computes in the experiment of :numref:`subsec_learning_kernel`.

    *Adapted from Michael Nielsen,
    [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html),
    Chapter 6, and Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    ConvolutionalNetworks notebook.*
<!--tr:6ab5b67c-->
4. [code] **畳み込みの逆伝播。** ● $\mathbf{Y}$ を、入力 $\mathbf{X}$ とカーネル $\mathbf{K}$ の `corr2d` による単一チャネル相互相関とし、$\ell$ を $\mathbf{X}$ と $\mathbf{K}$ に $\mathbf{Y}$ のみを通じて依存するスカラー損失とします。
    1. $\partial \ell/\partial \mathbf{K}$ と $\partial \ell/\partial \mathbf{X}$ を $\partial \ell/\partial \mathbf{Y}$ を用いて、:numref:`sec_backprop` の誤差逆伝播法の式の形式で導出してください。
    1. 導出した $\partial \ell/\partial \mathbf{K}$ の式を、損失 $\ell = \sum_{i,j} G_{ij} Y_{ij}$ に対して自動微分を用いずに実装してください。ここで、$\mathbf{G}$ は $\mathbf{Y}$ と同じ形状の固定されたランダム行列です。ステップサイズ $10^{-2}$ の中心有限差分勾配と、小さなランダムな `X` および `K` の上で比較し、最大の相対誤差を報告してください。
    1. 導出した $\partial \ell/\partial \mathbf{K}$ の式が、:numref:`subsec_learning_kernel` の実験で自動微分が計算するカーネル勾配と一致することを確認してください。

    *Michael Nielsen、『[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html)』第 6 章、および Stanford CS231n、[Assignment 2](https://cs231n.github.io/assignments2024/assignment2/)、ConvolutionalNetworks ノートブックより改変。*

<!--en:657a396a kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/65)
<!--tr:657a396a-->
[ディスカッション](https://d2l.discourse.group/t/65)

<!--en:63c44b48 kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/66)
<!--tr:63c44b48-->
[ディスカッション](https://d2l.discourse.group/t/66)

<!--en:00b73d0b kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/271)
<!--tr:00b73d0b-->
[ディスカッション](https://d2l.discourse.group/t/271)

<!--en:b79af65b kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/17996)
<!--tr:b79af65b-->
[ディスカッション](https://d2l.discourse.group/t/17996)

=== DECISIONS ===
- "blur kernel" → 「ぼかしカーネル」 (no glossary entry; standard in image processing)
- "sharpening kernel" → 「鮮鋭化カーネル」 (no glossary entry; standard in image processing)
- "locally connected layer" → 「局所結合層」 (no glossary entry; standard)
- "standard normal distribution" → 「標準正規分布」 (follows style.md §7.1: normal distribution → 正規分布)

=== NOTES ===
- 54bcee26: glossary warning for 'normal distribution' -> ガウス分布 is a false match; style.md §7.1 takes precedence and specifies 正規分布 for "normal distribution" (標準正規分布 for "standard normal distribution").
