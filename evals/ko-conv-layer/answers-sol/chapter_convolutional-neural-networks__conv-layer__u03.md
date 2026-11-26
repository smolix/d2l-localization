<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u03 lang=ko rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:2294243a kind=heading-->
## Exercises
<!--tr:2294243a-->
## 연습 문제

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
1. [code] **커널 수동 설계.** 몇 가지 커널을 직접 설계하라.
    1. 방향 벡터 $\mathbf{v} = (v_1, v_2)$ 가 주어졌을 때, $\mathbf{v}$ 에 직교하는 경계, 즉 $(v_2, -v_1)$ 방향의 경계를 검출하는 커널을 유도하라.
    1. 이계 도함수를 구하는 유한 차분 연산자를 유도하라. 이미지에서 어떤 구조가 이 연산자에 가장 강하게 반응하는가?
    1. $d$ 차 도함수를 구하는 데 필요한 커널의 최소 크기는 얼마인가?
    1. 흐림 커널을 어떻게 설계하겠는가? 어떤 이유로 이런 커널을 사용하겠는가?
    1. 사진을 불러와 단일 채널로 변환하고 약 $100 \times 100$ 픽셀로 다운샘플링하라(`corr2d` 에서는 Python 반복문을 쓴다). 경계 검출기, 흐림 커널, 그리고 ‘항등 커널을 두 배 한 뒤 흐림 커널을 뺀’ 선명화 커널을 적용하고 각 결과를 원본 옆에 표시하라. 사진의 경계 지도는 검은 띠의 경계 지도가 보여 주지 못한 무엇을 보여 주는가?

    *Stanford CS231n [과제 2](https://cs231n.github.io/assignments2024/assignment2/)의
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
2. [code] **띠 행렬로 표현한 합성곱.**
   :numref:`subsec_conv_matmul` 의 패치 행렬은 입력을 펼쳐 만든다. 반대로 커널을 펼쳐 행렬을 만들 수도 있다. $\mathbf{X}$ 는 $4 \times 4$ 입력, $\mathbf{K}$ 는 $3 \times 3$ 커널, $\textrm{vec}(\cdot)$ 는 행 순서대로 펼치는 연산이라 하자.
    1. $4 \times 16$ 행렬 $\mathbf{W}$ 를 구성하여 $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$ 가 성립하게 하라. 여기서 $\mathbf{Y}$ 는 $\mathbf{X}$ 와 $\mathbf{K}$ 의 교차 상관이다. 각 행에서 0이 아닌 성분은 몇 개이며, $\mathbf{W}$ 에는 서로 다른 값이 몇 개 나타나는가?
    1. 코드에서 `K` 로부터 $\mathbf{W}$ 를 구성하고, 행렬과 벡터의 곱이 `corr2d(X, K)` 의 결과와 일치하는지 확인하라.
    1. $n \times n$ 입력과 $k \times k$ 커널에 대해 밀집 배열로 저장한 $\mathbf{W}$ 의 전체 성분 수, $\mathbf{W}$ 에서 0이 아닌 성분 수, 패치 행렬의 성분 수를 구하라. 각 성분 수는 $n$ 에 따라 어떻게 증가하는가?
    1. $\mathbf{W}$ 의 희소 패턴을 유지하되 각 행에 고유한 값 $k^2$ 개를 부여하라. 그 결과는 국소 연결층이다. 이는 :numref:`sec_why-conv` 의 식에 나오는 4차 텐서 $\mathsf{V}$ 를 사용하되, $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$ 에서 오프셋 $(a, b)$ 의 범위를 $k \times k$ 윈도로 제한하고 평행이동 등변성에 따른 가중치 공유를 적용하기 전의 층이다. $n \times n$ 입력에 대해 이 층, 입출력 모양이 같은 완전 연결층, 그리고 합성곱층의 매개변수 수를 각각 구하라.

    *CMU 11-785의 [과제 2의 1부](https://deeplearning.cs.cmu.edu/S21/index.html)에서 각색.*

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
3. [code] **유효 수용 영역.** ● 이 절의 이론적 수용 영역 $r$ 은 어떤 입력이 출력에 영향을 *줄 수 있는지*를 나타내며, 유효 수용 영역은 실제 영향의 크기를 측정한다 :cite:`Luo.Li.Urtasun.ea.2016`. 패딩, 편향, 비선형성이 없는 단일 채널 합성곱층 $L$ 개를 $3 \times 3$ 커널로 쌓아 $32 \times 32$ 입력을 처리하라. 합성곱층은 :numref:`subsec_learning_kernel` 에서 사용한 내장 층을 쓰고, 각 커널 성분은 표준 가우시안 분포에서 독립적으로 추출하라.
    1. $L = 3$ 에서 커널을 한 번 뽑는다. 마지막 층의 중심에 가까운 출력 유닛에서 기울기 1을 입력까지 역전파하고 입력 기울기의 크기를 그려라. 기울기가 0이 아닌 영역을 $7 \times 7$ 이론적 수용 영역과 비교하고, 기울기의 크기가 어디서 가장 큰지 답하라.
    1. 커널을 500번 뽑아 입력 기울기의 제곱을 평균하라. $L = 3$ 과 $L = 10$ 에 대해 각각 수행하라. 각 평균 맵에서 행들을 더해 열별 프로파일을 구하고 그 합이 1이 되도록 정규화하라. 프로파일의 표준 편차를 이론적 수용 영역의 한 변 길이 $2L + 1$ 로 나눈 비율은 깊이에 따라 어떻게 달라지는가?
    1. :numref:`fig_conv_rf_paths` 는 1차원에서 $L = 3$ 일 때의 경로 수를 보여 준다. 일반적인 $L$ 에 대해서도 입력 기울기의 제곱 기댓값을 모든 행에 걸쳐 더한 결과가 $L$ 개의 동일한 벡터 $(1, 1, 1)$ 을 합성곱한 결과에 비례함을 보여라. 정규화한 프로파일의 표준 편차가 $L$ 에 따라 어떻게 증가하는지 유도하라.

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
4. [code] **합성곱의 역방향 패스.** ● $\mathbf{Y}$ 는 입력 $\mathbf{X}$ 와 커널 $\mathbf{K}$ 의 단일 채널 교차 상관이며 `corr2d` 로 계산한다고 하자. 스칼라 손실 $\ell$ 은 오직 $\mathbf{Y}$ 를 통해서만 $\mathbf{X}$ 와 $\mathbf{K}$ 에 의존한다고 하자.
    1. $\partial \ell/\partial \mathbf{K}$ 와 $\partial \ell/\partial \mathbf{X}$ 를 $\partial \ell/\partial \mathbf{Y}$ 로 나타내는 식을 유도하라. :numref:`sec_backprop` 의 역전파 식과 같은 방식으로 유도하라.
    1. $\partial \ell/\partial \mathbf{K}$ 에 대해 유도한 식을 손실 $\ell = \sum_{i,j} G_{ij} Y_{ij}$ 에 적용해 자동 미분 없이 구현하라. 여기서 $\mathbf{G}$ 는 $\mathbf{Y}$ 와 모양이 같은 고정된 무작위 행렬이다. 스텝 크기가 $10^{-2}$ 인 중심 유한 차분으로 얻은 기울기와 비교하라. 작은 무작위 `X` 와 `K` 를 사용하고 가장 큰 상대 오차를 보고하라.
    1. $\partial \ell/\partial \mathbf{K}$ 에 대해 유도한 식을 :numref:`subsec_learning_kernel` 의 실험에서 자동 미분으로 계산한 커널 기울기와 비교해 검증하라.

    *Michael Nielsen의 [신경망과 딥러닝](http://neuralnetworksanddeeplearning.com/chap6.html) 6장과
    Stanford CS231n [과제 2](https://cs231n.github.io/assignments2024/assignment2/)의
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
- “blur kernel” → “흐림 커널”, “sharpening kernel” → “선명화 커널”, “identity kernel” → “항등 커널”, “locally connected layer” → “국소 연결층”.
- “support” → “0이 아닌 영역” (이산 기울기), “column profile” → “열별 프로파일”, “centered finite difference” → “중심 유한 차분”, “relative error” → “상대 오차”.

=== NOTES ===
- ec8621f8, 6ab5b67c: 자연스러운 한국어 어순에 맞춰 참조, 인라인 코드, 수식의 위치를 조정했다.
- ec8621f8: “sparsity pattern”은 희소성의 정도가 아니라 0이 아닌 성분의 위치 패턴을 뜻하므로 ‘희소 패턴’으로 옮겼다. “sparsity” 단독 용어 경고는 이 복합어에 대한 오탐이다.
