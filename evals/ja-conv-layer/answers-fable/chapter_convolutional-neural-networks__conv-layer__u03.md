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
1. [code] **人手で設計するカーネル。**何種類かのカーネルを人手で設計してください。
    1. 方向ベクトル $\mathbf{v} = (v_1, v_2)$ が与えられたとき、$\mathbf{v}$ に直交するエッジ、すなわち $(v_2, -v_1)$ 方向のエッジを検出するエッジ検出カーネルを導いてください。
    1. 二階導関数に対する有限差分作用素を導いてください。画像中のどのような構造が、これに最も強く反応しますか。
    1. $d$ 階の導関数を得るには、カーネルの大きさは最小でいくつ必要ですか。
    1. ぼかしカーネルはどのように設計しますか。そのようなカーネルを使いたくなるのは、なぜですか。
    1. 写真を読み込み、1 チャネルに変換して、約 $100 \times 100$ ピクセルに縮小してください（`corr2d` は Python でループするためです）。エッジ検出器、ぼかしカーネル、および「恒等カーネルの 2 倍からぼかしを引いたもの」である鮮鋭化カーネルを適用し、それぞれの結果を元の画像と並べて表示してください。写真のエッジマップには、黒い帯のエッジマップでは見られなかった何が現れますか。

    *Stanford CS231n の
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/)、
    ConvolutionalNetworks ノートブックより改変。*

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
2. [code] **帯行列としての畳み込み。**パッチ行列（:numref:`subsec_conv_matmul`）は入力を展開します。これとは逆に、カーネルのほうを展開する構成もあります。$\mathbf{X}$ を $4 \times 4$ の入力、$\mathbf{K}$ を $3 \times 3$ のカーネル、$\textrm{vec}(\cdot)$ を行ごとの平坦化とします。
    1. $4 \times 16$ の行列 $\mathbf{W}$ を、$\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$ を満たすように構成してください。ここで、$\mathbf{Y}$ は $\mathbf{X}$ と $\mathbf{K}$ の相互相関です。各行には非ゼロ要素がいくつありますか。また、$\mathbf{W}$ には異なる値が何種類現れますか。
    1. $\mathbf{W}$ を `K` からコードで組み立て、その積が `corr2d(X, K)` を再現することを確認してください。
    1. $n \times n$ の入力と $k \times k$ のカーネルについて、密な配列として格納した $\mathbf{W}$ の要素数、$\mathbf{W}$ の非ゼロ要素数、パッチ行列の要素数を数えてください。それぞれの数は $n$ とともにどのように増えますか。
    1. $\mathbf{W}$ のスパースパターンはそのままにして、各行に独自の $k^2$ 個の値を持たせます。こうして得られるのが局所結合層です。これは、4 階テンソル $\mathsf{V}$（:numref:`sec_why-conv` の $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$ に現れるもの）を、$k \times k$ のオフセット $(a, b)$ のウィンドウに制限したものにあたり、平行移動同変性によって重みが共有される前の段階です。$n \times n$ の入力に対して、この層、入力と出力の形状が同じ全結合層、そして畳み込みのパラメータ数をそれぞれ数えてください。

    *CMU 11-785 の
    [Homework 2 Part 1](https://deeplearning.cs.cmu.edu/S21/index.html) より改変。*

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
3. [code] **有効受容野。** ● 本節の理論的受容野 $r$ は、どの入力が出力に影響*しうる*かの範囲を定めます。一方、有効受容野は、入力が実際にどれだけ影響するかを測ります :cite:`Luo.Li.Urtasun.ea.2016`。$L$ 個の単一チャネル $3 \times 3$ 畳み込み層を、パディングなし、バイアスなし、非線形性なしで積み重ね、$32 \times 32$ の入力に適用してください。層には :numref:`subsec_learning_kernel` の組み込みの層を使います。カーネルの各要素は、標準正規分布から独立にサンプリングします。
    1. $L = 3$ とし、カーネルを 1 回サンプリングした場合について、最終層の中心付近にある出力ユニットから入力まで単位勾配を逆伝播させ、入力に関する勾配の大きさをプロットしてください。そのサポートを $7 \times 7$ の理論的受容野と比較し、大きさが最大になる場所を述べてください。
    1. 入力に関する勾配の二乗を、カーネルの 500 回のサンプリングにわたって平均してください。これを $L = 3$ と $L = 10$ について行います。平均したマップをそれぞれ行方向に足し合わせ、得られた列方向のプロファイルを、和が 1 になるように正規化してください。プロファイルの標準偏差と理論的な 1 辺の長さ $2L + 1$ との比は、深さとともにどう変化しますか。
    1. :numref:`fig_conv_rf_paths` は、$L = 3$ の場合の経路を 1 次元で数えています。一般の $L$ について、入力に関する勾配の二乗の期待値を行方向に足し合わせたものが、$L$ 個のベクトル $(1, 1, 1)$ を畳み込んだものに比例することを示してください。正規化したプロファイルの標準偏差が $L$ とともにどのように大きくなるかを導いてください。

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
4. [code] **畳み込みの逆伝播。** ● $\mathbf{Y}$ を、入力 $\mathbf{X}$ とカーネル $\mathbf{K}$ の単一チャネルの相互相関（`corr2d` が計算するもの）とします。また、$\ell$ を、$\mathbf{X}$ と $\mathbf{K}$ には $\mathbf{Y}$ を通じてのみ依存するスカラーの損失とします。
    1. $\partial \ell/\partial \mathbf{K}$ と $\partial \ell/\partial \mathbf{X}$ を、$\partial \ell/\partial \mathbf{Y}$ を使って、:numref:`sec_backprop` の誤差逆伝播法の式にならった形で導いてください。
    1. 自分で導いた $\partial \ell/\partial \mathbf{K}$ の式を、自動微分を使わずに、損失 $\ell = \sum_{i,j} G_{ij} Y_{ij}$ について実装してください。ここで、$\mathbf{G}$ は $\mathbf{Y}$ と同じ形状の固定されたランダム行列です。これを、ステップサイズ $10^{-2}$ の中心有限差分で求めた勾配と、小さなランダムな `X` と `K` の上で比較し、最大の相対誤差を報告してください。
    1. 自分で導いた $\partial \ell/\partial \mathbf{K}$ の式を、:numref:`subsec_learning_kernel` の実験で自動微分が計算するカーネルの勾配と照合してください。

    *Michael Nielsen の
    [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html)
    第 6 章、および Stanford CS231n の
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/)、
    ConvolutionalNetworks ノートブックより改変。*

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
- Exercise titles: the bold title keeps its full stop inside the bold (**…。**); the `[code]` tag and the ● marker are copied unchanged
- "Adapted from X, [title](url), …" → *X の [title](url)、…より改変。*; course names, assignment titles and notebook names stay in English
- "Discussions" (link text) → ディスカッション
- "locally connected layer" → 局所結合層; "sparsity pattern" → スパースパターン; "fourth-order tensor" → 4 階テンソル
- "derivative of order $d$" → $d$ 階の導関数; "finite difference operator" → 有限差分作用素; "centered finite-difference" → 中心有限差分
- "blur kernel" → ぼかしカーネル; "sharpening kernel" → 鮮鋭化カーネル; "downsample" (an image) → 縮小する
- "draw" (from a distribution) → サンプリングする / サンプリング; "support" (of a function) → サポート; "profile" → プロファイル
- "automatic differentiation" → 自動微分; "relative error" → 相対誤差; "expected (value)" → 期待値

=== NOTES ===
- ec8621f8: "The patch matrix of :numref:`subsec_conv_matmul`" is written パッチ行列（:numref:`subsec_conv_matmul`） so that the reference does not stand directly after the bold title's closing `**`.
- ec8621f8 (last sub-item): the clause order is changed ("restricted to a $k \times k$ window … before translation equivariance ties the weights") so that the protected tokens keep the English order; the meaning is unchanged.
- 6f4ddec6: "(`corr2d` loops in Python)" gives the reason for downsampling and is translated as a reason (…ためです).
- 54bcee26: the label :numref:`subsec_learning_kernel` and, in ec8621f8, :numref:`subsec_conv_matmul` are copied as in the source; I did not verify that these labels exist on the English page.
- ec8621f8: check warning 'sparsity' → スパース性 is a false match; the English is "sparsity pattern", translated スパースパターン.
- 54bcee26: check warning 'normal distribution' → ガウス分布: the English is "standard normal distribution", translated 標準正規分布 as style §7.1 rules (normal distribution → 正規分布; ガウス分布 only for "Gaussian").
