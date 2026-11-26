<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u03 lang=ko rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:2294243a kind=heading-->
## Exercises
<!--tr:2294243a-->
## 연습문제

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
1. [code] **직접 설계한 커널.** 몇 가지 커널을 수동으로 설계해 보세요.
    1. 방향 벡터 $\mathbf{v} = (v_1, v_2)$ 가 주어졌을 때, $\mathbf{v}$ 에 직교하는 경계선, 즉 $(v_2, -v_1)$ 방향의 경계선을 감지하는 경계선 감지 커널을 유도하세요.
    1. 이계 도함수를 위한 유한 차분 연산자를 유도하세요. 이미지의 어떤 구조가 이에 가장 강하게 반응할까요?
    1. $d$ 차 도함수를 얻기 위한 커널의 최소 크기는 얼마인가요?
    1. 흐림(blur) 커널은 어떻게 설계할 수 있나요? 이러한 커널을 사용하는 이유는 무엇일까요?
    1. 사진 한 장을 불러와 단일 채널로 변환한 뒤 약 $100 \times 100$ 픽셀로 다운샘플링하세요(Python의 `corr2d` 반복문 사용). 경계선 감지기, 흐림 커널, 그리고 "항등 커널의 두 배에서 흐림 커널을 뺀" 선명화(sharpening) 커널을 각각 적용하고 결과를 원본 옆에 나란히 표시하세요. 사진의 경계선 맵은 검은 띠 이미지의 경계선 맵이 보여 주지 못했던 어떤 점을 보여 주나요?

    *Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    ConvolutionalNetworks 노트북에서 각색.*

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
2. [code] **띠 행렬로서의 합성곱.** :numref:`subsec_conv_matmul` 의 패치 행렬은 입력을 펼칩니다. 반대의 구성은 커널을 펼치는 것입니다. $\mathbf{X}$ 를 $4 \times 4$ 입력, $\mathbf{K}$ 를 $3 \times 3$ 커널, $\textrm{vec}(\cdot)$ 을 행 우선 평탄화라고 합시다.
    1. $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$ 를 만족하는 $4 \times 16$ 행렬 $\mathbf{W}$ 를 구성하세요. 여기서 $\mathbf{Y}$ 는 $\mathbf{X}$ 와 $\mathbf{K}$ 의 교차 상관입니다. 각 행에는 0이 아닌 원소가 몇 개 있으며, $\mathbf{W}$ 전체에는 서로 다른 값이 몇 개 나타나나요?
    1. 코드로 `K` 로부터 $\mathbf{W}$ 를 만들고 그 행렬 곱이 `corr2d(X, K)` 를 재현함을 확인하세요.
    1. $n \times n$ 입력과 $k \times k$ 커널에 대해, 밀집 배열로 저장한 $\mathbf{W}$ 의 원소 수, $\mathbf{W}$ 의 0이 아닌 원소 수, 그리고 패치 행렬의 원소 수를 각각 구하세요. 각 개수는 $n$ 에 따라 어떻게 증가하나요?
    1. $\mathbf{W}$ 의 희소성 패턴은 유지하되 각 행마다 고유한 $k^2$ 개의 값을 부여해 보세요. 그 결과는 지역 연결층(locally connected layer)이 됩니다. 이는 평행이동 등변성이 가중치를 묶기 전, 오프셋 $(a, b)$ 의 $k \times k$ 윈도우로 제한된 :numref:`sec_why-conv` 의 4차 텐서 $\mathsf{V}$ ( $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$ )에 해당합니다. $n \times n$ 입력에 대해 이 층의 매개변수 수, 입출력 모양이 같은 완전 연결층의 매개변수 수, 그리고 합성곱의 매개변수 수를 각각 구하세요.

    *CMU 11-785,
    [Homework 2 Part 1](https://deeplearning.cs.cmu.edu/S21/index.html) 에서 각색.*

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
3. [code] **유효 수용 영역.** ● 이 절의 이론적 수용 영역 $r$ 은 어떤 입력이 출력에 영향을 *줄 수 있는지*를 한정하고, 유효 수용 영역은 실제로 얼마나 영향을 미치는지를 측정합니다 :cite:`Luo.Li.Urtasun.ea.2016`. :numref:`subsec_learning_kernel` 의 내장 층을 사용해 $32 \times 32$ 입력 위에 패딩, 편향, 비선형성이 모두 없는 단일 채널 $3 \times 3$ 합성곱층 $L$ 개를 쌓으세요. 모든 커널 원소는 표준 정규 분포에서 독립적으로 추출합니다.
    1. $L = 3$ 인 경우 커널을 한 번 추출한 뒤, 마지막 층의 중심 부근에 있는 출력 유닛 하나에서 입력까지 단위 기울기를 역전파하고 입력 기울기의 크기를 그래프로 그리세요. 그 지지 영역(support)을 $7 \times 7$ 이론적 수용 영역과 비교하고, 크기가 가장 큰 위치가 어디인지 설명하세요.
    1. $L = 3$ 과 $L = 10$ 에 대해 커널을 500번 추출하여 입력 기울기의 제곱을 평균하세요. 각 평균 맵을 행 방향으로 합산하고 결과로 나온 열 분포(column profile)의 합이 1이 되도록 정규화하세요. 깊이가 깊어질수록 이 분포의 표준 편차 대 이론적 한 변 길이 $2L + 1$ 의 비율은 어떻게 변하나요?
    1. :numref:`fig_conv_rf_paths` 는 일차원에서 $L = 3$ 일 때의 경로 수를 셉니다. 일반적인 $L$ 에 대해 행 방향으로 합산한 입력 기울기 제곱의 기댓값이 벡터 $(1, 1, 1)$ 을 $L$ 번 합성곱한 결과에 비례함을 보이세요. 또한 정규화된 분포의 표준 편차가 $L$ 에 따라 어떻게 커지는지 유도하세요.

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
4. [code] **합성곱의 역방향 패스.** ● $\mathbf{Y}$ 를 `corr2d` 로 계산한 입력 $\mathbf{X}$ 와 커널 $\mathbf{K}$ 의 단일 채널 교차 상관이라 하고, $\ell$ 을 $\mathbf{Y}$ 를 통해서만 $\mathbf{X}$ 와 $\mathbf{K}$ 에 의존하는 스칼라 손실이라고 합시다.
    1. :numref:`sec_backprop` 의 역전파 방정식 형식으로 $\partial \ell/\partial \mathbf{K}$ 와 $\partial \ell/\partial \mathbf{X}$ 를 $\partial \ell/\partial \mathbf{Y}$ 로 나타내어 유도하세요.
    1. $\mathbf{Y}$ 와 같은 모양의 고정된 무작위 행렬 $\mathbf{G}$ 에 대해 손실 $\ell = \sum_{i,j} G_{ij} Y_{ij}$ 를 두고, 자동 미분 없이 $\partial \ell/\partial \mathbf{K}$ 의 수식을 구현하세요. 작은 무작위 `X` 와 `K` 에 대해 스텝 크기 $10^{-2}$ 의 중심 유한 차분 기울기와 비교하고 최대 상대 오차를 보고하세요.
    1. :numref:`subsec_learning_kernel` 의 실험에서 자동 미분이 계산하는 커널 기울기와 여러분이 유도한 $\partial \ell/\partial \mathbf{K}$ 의 수식을 대조하여 확인하세요.

    *Michael Nielsen,
    [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html),
    6장 및 Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    ConvolutionalNetworks 노트북에서 각색.*

<!--en:657a396a kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/65)
<!--tr:657a396a-->
[토론](https://d2l.discourse.group/t/65)

<!--en:63c44b48 kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/66)
<!--tr:63c44b48-->
[토론](https://d2l.discourse.group/t/66)

<!--en:00b73d0b kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/271)
<!--tr:00b73d0b-->
[토론](https://d2l.discourse.group/t/271)

<!--en:b79af65b kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/17996)
<!--tr:b79af65b-->
[토론](https://d2l.discourse.group/t/17996)

=== DECISIONS ===
- Register: use polite `-ㅂ니다/습니다` style (`~하세요`, `~인가요`) in exercises, consistent with `-ㅂ니다/습니다` in body prose.
- CJK spacing: place a space between `:numref:`/`:cite:`/`$...$` and adjacent Hangul particles/letters per TRANSLATING.md §4.
- "edge" → 「경계선」 (consistent with u01, u02).
- "banded matrix" → 「띠 행렬」 (consistent with u02).
- "locally connected layer" → 「지역 연결층」 (analogous to `완전 연결층`).
- "Discussions" tab link → 「토론」.

=== NOTES ===
- ec8621f8, 54bcee26, 6ab5b67c: math tokens reordered to match Korean SOV word order.
