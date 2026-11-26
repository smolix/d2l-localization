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
1. [code] **カーネルの手設計。** いくつかのカーネルを手作業で設計してください。
    1. 方向ベクトル $\mathbf{v} = (v_1, v_2)$ が与えられたとき、$\mathbf{v}$ に直交するエッジ、すなわち $(v_2, -v_1)$ 方向のエッジを検出するカーネルを導いてください。
    1. 二階導関数を近似する有限差分演算子を導いてください。画像中のどのような構造で、この演算子の応答が最も強くなりますか。
    1. $d$ 階の導関数を求めるために必要な最小のカーネルサイズはいくつですか。
    1. ぼかしカーネルはどのように設計すればよいですか。どのような理由でこのカーネルを使うのでしょうか。
    1. 写真を読み込み、1 チャネルに変換し、約 $100 \times 100$ ピクセルまでダウンサンプリングしてください（`corr2d` は Python でループを回すためです）。エッジ検出器、ぼかしカーネル、そして「恒等カーネルの 2 倍からぼかしカーネルを引いた」鮮鋭化カーネルを適用し、それぞれの結果を元画像と並べて表示してください。写真のエッジマップを見ると、黒い帯のエッジマップでは見えなかった何が見えますか。

    *Stanford CS231n の[課題 2](https://cs231n.github.io/assignments2024/assignment2/) にある ConvolutionalNetworks ノートブックをもとに改題。*

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
2. [code] **帯行列としての畳み込み。** :numref:`subsec_conv_matmul` のパッチ行列は入力を展開したものです。逆の構成では、カーネルを展開します。$\mathbf{X}$ を $4 \times 4$ の入力、$\mathbf{K}$ を $3 \times 3$ のカーネル、$\textrm{vec}(\cdot)$ を行ごとに 1 次元に並べる操作とします。
    1. $4 \times 16$ の行列 $\mathbf{W}$ で、$\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$ を満たすものを作ってください。ここで、$\mathbf{Y}$ は $\mathbf{X}$ と $\mathbf{K}$ の相互相関です。各行には非ゼロ要素がいくつあり、$\mathbf{W}$ 全体には異なる値が何種類現れますか。
    1. $\mathbf{W}$ を `K` から作るコードを書き、その行列積で `corr2d(X, K)` の結果を再現できることを確かめてください。
    1. $n \times n$ の入力と $k \times k$ のカーネルについて、$\mathbf{W}$ を密な配列として保存した場合の全要素数、$\mathbf{W}$ の非ゼロ要素数、およびパッチ行列の全要素数を求めてください。それぞれの要素数は $n$ に応じてどう増えますか。
    1. $\mathbf{W}$ のスパース性を決める非ゼロ要素の配置を保ったまま、各行に固有の値を $k^2$ 個ずつ持たせてください。これが局所結合層です。これは、4 階テンソル $\mathsf{V}$ を使った :numref:`sec_why-conv` の式 $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$ で、オフセットを $k \times k$ のウィンドウ内の $(a, b)$ に制限し、平行移動同変性による重みの共有をまだ課していないものです。$n \times n$ の入力に対し、この層、入出力の形状が同じ全結合層、および畳み込みのパラメータ数をそれぞれ求めてください。

    *CMU 11-785 の[宿題 2 第 1 部](https://deeplearning.cs.cmu.edu/S21/index.html) をもとに改題。*

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
3. [code] **有効受容野。** ● 本節の理論上の受容野 $r$ は出力に影響を*及ぼしうる*入力の範囲を定めます。有効受容野は、その入力が実際にどの程度影響するかを表します :cite:`Luo.Li.Urtasun.ea.2016`。$L$ 個の単一チャネルの $3 \times 3$ の畳み込み層を、パディング、バイアス、非線形性を導入せず、$32 \times 32$ の入力の上に重ねてください。層は :numref:`subsec_learning_kernel` の組み込みのものを使い、カーネルの各要素を標準正規分布から独立にサンプリングしてください。
    1. $L = 3$ とし、カーネルを 1 組サンプリングしてください。最終層の中央付近にある出力ユニット 1 つに値が 1 の勾配を与え、入力まで逆伝播させて、入力に関する勾配の大きさをプロットしてください。勾配が非ゼロとなる範囲を $7 \times 7$ の理論上の受容野と比べ、勾配の大きさがどこで最大になるかを答えてください。
    1. カーネルを 500 回サンプリングし、各回で求めた入力に関する勾配の二乗を平均してください。$L = 3$ と $L = 10$ のそれぞれについて行ってください。平均したマップを列ごとに全行にわたって足し合わせ、得られた列ごとの値の総和が 1 になるように正規化してください。この列方向の分布の標準偏差と理論上の受容野の辺の長さ $2L + 1$ との比は、層が深くなるとどう変わりますか。
    1. :numref:`fig_conv_rf_paths` は $L = 3$ の 1 次元の場合について経路を数えたものです。一般の $L$ について、入力に関する勾配の二乗の期待値を列ごとに全行にわたって足し合わせたものが、$L$ 個のベクトル $(1, 1, 1)$ を畳み込んだ結果に比例することを示してください。正規化した列方向の分布の標準偏差が $L$ に応じてどのように増えるかを導いてください。

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
4. [code] **畳み込みの逆伝播。** ● $\mathbf{Y}$ を、入力 $\mathbf{X}$ とカーネル $\mathbf{K}$ の単一チャネルの相互相関（`corr2d` で計算）とします。また、$\ell$ を、$\mathbf{X}$ と $\mathbf{K}$ に $\mathbf{Y}$ を介してのみ依存するスカラーの損失とします。
    1. $\partial \ell/\partial \mathbf{K}$ と $\partial \ell/\partial \mathbf{X}$ を、$\partial \ell/\partial \mathbf{Y}$ を使って表してください。:numref:`sec_backprop` の誤差逆伝播法の式と同様に導いてください。
    1. $\partial \ell/\partial \mathbf{K}$ の式を、自動微分を使わずに実装してください。損失は $\ell = \sum_{i,j} G_{ij} Y_{ij}$ とします。ここで、$\mathbf{G}$ は $\mathbf{Y}$ と同じ形状の固定した乱数行列です。ステップサイズ $10^{-2}$ の中心有限差分で求めた勾配と比較してください。比較には小さなランダム配列 `X` と `K` を使い、最大の相対誤差を報告してください。
    1. $\partial \ell/\partial \mathbf{K}$ の式を、:numref:`subsec_learning_kernel` の実験で自動微分が計算するカーネルの勾配と照合してください。

    *Michael Nielsen の[『Neural Networks and Deep Learning』](http://neuralnetworksanddeeplearning.com/chap6.html) 第 6 章と、Stanford CS231n の[課題 2](https://cs231n.github.io/assignments2024/assignment2/) にある ConvolutionalNetworks ノートブックをもとに改題。*

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
- "locally connected layer" → 「局所結合層」.
- "blur kernel" → 「ぼかしカーネル」 and "sharpening kernel" → 「鮮鋭化カーネル」.
- "centered finite difference" → 「中心有限差分」.
=== NOTES ===
- 54bcee26: The style guide §7.1 calls for 正規分布 for “normal distribution” and ガウス分布 for “Gaussian”. “Standard normal distribution” is therefore 標準正規分布. The glossary warning is a false positive.
