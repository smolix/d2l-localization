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
1. [code] **手作りのカーネル。**カーネルをいくつか手作業で設計してください。
    1. 方向ベクトル $\mathbf{v} = (v_1, v_2)$ が与えられたとき、$\mathbf{v}$ に直交するエッジ、すなわち方向 $(v_2, -v_1)$ のエッジを検出するエッジ検出カーネルを導いてください。
    1. 2 階微分の有限差分演算子を導いてください。画像中のどのような構造が、それに最も強く反応しますか。
    1. $d$ 階の導関数を得るために必要なカーネルの最小サイズはいくつですか。
    1. ぼかしカーネルはどのように設計しますか。このようなカーネルを使いたいのはなぜでしょうか。
    1. 写真を読み込み、1 チャネルに変換し、約 $100 \times 100$ ピクセルにダウンサンプリングしてください（`corr2d` が Python のループだからです）。エッジ検出器、ぼかしカーネル、そして「恒等カーネルの 2 倍からぼかしを引いたもの」であるシャープ化カーネルを適用し、それぞれの結果を元の画像の隣に表示してください。写真のエッジマップには、黒い帯のエッジマップでは示せなかった何が表れますか。

    *Stanford CS231n の [課題 2](https://cs231n.github.io/assignments2024/assignment2/) のうち ConvolutionalNetworks ノートブックから改作。*

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
2. [code] **帯行列としての畳み込み。**:numref:`subsec_conv_matmul` のパッチ行列は入力を展開したものでした。その逆の構成では、カーネルを展開します。$\mathbf{X}$ を $4 \times 4$ の入力、$\mathbf{K}$ を $3 \times 3$ のカーネル、$\textrm{vec}(\cdot)$ を行ごとの平坦化とします。
    1. $4 \times 16$ の行列 $\mathbf{W}$ を、$\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$ を満たすように構成してください。ここで、$\mathbf{Y}$ は $\mathbf{X}$ と $\mathbf{K}$ の相互相関です。各行のゼロでない要素はいくつで、$\mathbf{W}$ には異なる値が何種類現れますか。
    1. `K` から $\mathbf{W}$ をコードで構築し、その積が `corr2d(X, K)` を再現することを確認してください。
    1. $n \times n$ の入力と $k \times k$ のカーネルについて、密な配列として保存した $\mathbf{W}$ の要素数、$\mathbf{W}$ のゼロでない要素の数、およびパッチ行列の要素数を数えてください。それぞれの数は $n$ に応じてどのように増えますか。
    1. $\mathbf{W}$ のスパース性のパターンはそのままにして、各行に独自の $k^2$ 個の値を持たせてください。得られるのは局所結合層です。つまり、4 階のテンソル $\mathsf{V}$（:numref:`sec_why-conv`）を、$[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$ において、平行移動同変性が重みを結びつける前に、$k \times k$ の窓のオフセット $(a, b)$ に制限したものです。$n \times n$ の入力について、この層のパラメータ数、同じ入力形状と出力形状を持つ全結合層のパラメータ数、および畳み込みのパラメータ数を数えてください。

    *CMU 11-785 の [課題 2 の第 1 部](https://deeplearning.cs.cmu.edu/S21/index.html)から改作。*

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
3. [code] **有効受容野。** ● 本節の理論上の受容野 $r$ は、どの入力が出力に影響を*与えうる*かを限定します。有効受容野は、それらの入力が実際にどれだけ影響しているかを測ります :cite:`Luo.Li.Urtasun.ea.2016`。$L$ 層の積み重ね（各層は 1 チャネルの $3 \times 3$ 畳み込み層で、パディング、バイアス、非線形性はなし）を、$32 \times 32$ の入力に対して、:numref:`subsec_learning_kernel` の組み込み層を使って構築してください。カーネルの各要素は、標準正規分布から独立に引いてください。
    1. $L = 3$ で、カーネルを 1 回だけサンプリングしたとき、最後の層の中心付近の出力ユニットから入力まで単位勾配を逆伝播し、入力勾配の大きさをプロットしてください。その非ゼロの範囲を $7 \times 7$ の理論上の受容野と比較し、大きさが最大になる場所を答えてください。
    1. カーネルを 500 回サンプリングして、$L = 3$ と $L = 10$ について、入力勾配の二乗を平均してください。平均した各マップを行方向に合計し、得られた列方向のプロファイルを、合計が 1 になるように正規化してください。プロファイルの標準偏差と、理論上の一辺の長さ $2L + 1$ との比は、深さとともにどう変化しますか。
    1. :numref:`fig_conv_rf_paths` は、1 次元で $L = 3$ の経路を数えました。一般の $L$ について、行方向に合計した入力勾配の二乗の期待値が、$L$ 個のベクトル $(1, 1, 1)$ の畳み込みに比例することを示してください。正規化したプロファイルの標準偏差が $L$ とともにどのように増えるかを導いてください。

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
4. [code] **畳み込みの逆伝播。** ● $\mathbf{Y}$ を、入力 $\mathbf{X}$ とカーネル $\mathbf{K}$ の 1 チャネルの相互相関（`corr2d` で計算されるもの）とし、$\ell$ を、$\mathbf{X}$ と $\mathbf{K}$ に $\mathbf{Y}$ を通じてのみ依存するスカラーの損失とします。
    1. $\partial \ell/\partial \mathbf{K}$ と $\partial \ell/\partial \mathbf{X}$ を、$\partial \ell/\partial \mathbf{Y}$ を用いて、:numref:`sec_backprop` の誤差逆伝播法の式と同じ形で導いてください。
    1. $\partial \ell/\partial \mathbf{K}$ の式を、自動微分を使わずに、損失 $\ell = \sum_{i,j} G_{ij} Y_{ij}$（$\mathbf{G}$ は $\mathbf{Y}$ と同じ形状の固定されたランダムな行列）について実装してください。これを、ステップ幅 $10^{-2}$ の中心有限差分による勾配と、小さなランダムな `X` と `K` の上で比較し、最大の相対誤差を報告してください。
    1. $\partial \ell/\partial \mathbf{K}$ の式を、:numref:`subsec_learning_kernel` の実験で自動微分が計算するカーネルの勾配と照らして確認してください。

    *Michael Nielsen の [『Neural Networks and Deep Learning』](http://neuralnetworksanddeeplearning.com/chap6.html) の第 6 章、および Stanford CS231n の [課題 2](https://cs231n.github.io/assignments2024/assignment2/) のうち ConvolutionalNetworks ノートブックから改作。*

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
- "locally connected layer" → 局所結合層
- "support" (of a gradient map) → 非ゼロの範囲; "draw/sample the kernels" → サンプリングする
- Adaptation credits: 「…から改作」; link texts translated (課題 2, 課題 2 の第 1 部, 『Neural Networks and Deep Learning』)
- Exercise bold titles keep their closing 。 inside the bold markers: **手作りのカーネル。**

=== NOTES ===
- 54bcee26 (warning "normal distribution → ガウス分布"): 標準正規分布 follows style.md §7.1 (normal distribution → 正規分布; ガウス分布 for "Gaussian"); the usual Japanese term for the standard normal.
- 6f4ddec6: the English "(`corr2d` loops in Python)" is read as the reason for downsampling; written （`corr2d` が Python のループだからです）.
- ec8621f8: the clause order of the locally-connected sentence was rearranged so the protected tokens keep the English order (V, numref, H-equation, k x k, (a, b)); meaning is unchanged.
