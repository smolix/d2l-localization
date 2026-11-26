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
1. [code] **수작업으로 설계한 커널.** 몇 가지 커널을 직접 설계해 보십시오.
    1. 방향 벡터 $\mathbf{v} = (v_1, v_2)$ 가 주어졌을 때, $\mathbf{v}$ 에 직교하는 에지, 즉 $(v_2, -v_1)$ 방향의 에지를 검출하는 에지 검출 커널을 유도하십시오.
    1. 이계 도함수에 대한 유한 차분 연산자를 유도하십시오. 이미지의 어떤 구조가 이에 가장 강하게 반응합니까?
    1. $d$ 계 도함수를 얻기 위한 커널의 최소 크기는 얼마입니까?
    1. 블러(blur) 커널을 어떻게 설계하겠습니까? 이러한 커널을 사용하는 이유는 무엇입니까?
    1. 사진 한 장을 불러와 단일 채널로 변환하고 약 $100 \times 100$ 픽셀로 다운샘플링하십시오(`corr2d`는 Python에서 루프를 돕니다). 에지 검출기, 블러 커널, 그리고 "항등 커널의 두 배 빼기 블러 커널"인 샤프닝(sharpening) 커널을 적용하고, 각 결과를 원본 옆에 나란히 표시하십시오. 검은색 띠의 에지 맵에서는 볼 수 없었던 어떤 모습이 사진의 에지 맵에 나타납니까?

    *Stanford CS231n,
    [과제 2](https://cs231n.github.io/assignments2024/assignment2/),
    ConvolutionalNetworks 노트북에서 발췌·변형함.*

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
2. [code] **띠 행렬로서의 합성곱.** :numref:`subsec_conv_matmul` 의 패치 행렬은 입력을 펼칩니다. 그 반대의 구성은 커널을 펼칩니다. $\mathbf{X}$ 를 $4 \times 4$ 입력, $\mathbf{K}$ 를 $3 \times 3$ 커널, $\textrm{vec}(\cdot)$ 을 행 단위로 평탄화하는 연산이라 합시다.
    1. $4 \times 16$ 행렬 $\mathbf{W}$ 를 $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$ 가 성립하도록 구성하십시오(여기서 $\mathbf{Y}$ 는 $\mathbf{X}$ 와 $\mathbf{K}$ 의 교차 상관입니다). 각 행에는 0이 아닌 원소가 몇 개 있으며, $\mathbf{W}$ 에는 서로 다른 값이 몇 개 나타납니까?
    1. 코드로 $\mathbf{W}$ 를 `K`로부터 구성하고 그 행렬 곱이 `corr2d(X, K)`를 재현하는지 확인하십시오.
    1. $n \times n$ 입력과 $k \times k$ 커널에 대해 밀집 배열로 저장된 $\mathbf{W}$ 의 원소 수, $\mathbf{W}$ 의 0이 아닌 원소 수, 그리고 패치 행렬의 원소 수를 세어 보십시오. 각 개수는 $n$ 에 따라 어떻게 증가합니까?
    1. $\mathbf{W}$ 의 희소성 패턴은 유지하되 각 행에 고유한 $k^2$ 개의 값을 부여해 보십시오. 그 결과는 지역 연결층(locally connected layer)이 됩니다. 이는 4차 텐서 $\mathsf{V}$(:numref:`sec_why-conv` 의 $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$ 에 등장함)를 평행이동 등변성으로 가중치를 묶기 전에 $k \times k$ 크기의 오프셋 $(a, b)$ 윈도우로 제한한 것입니다. $n \times n$ 입력에 대해 이 층, 입출력 모양이 동일한 완전 연결층, 그리고 합성곱의 매개변수 수를 각각 구하십시오.

    *CMU 11-785,
    [과제 2 파트 1](https://deeplearning.cs.cmu.edu/S21/index.html)에서 발췌·변형함.*

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
3. [code] **유효 수용 영역.** ● 이 절에서 다룬 이론적 수용 영역 $r$ 은 어떤 입력이 출력에 영향을 줄 *수 있는지*의 범위를 제한하는 반면, 유효 수용 영역은 실제로 얼마나 영향을 주는지를 측정합니다 :cite:`Luo.Li.Urtasun.ea.2016`. 패딩, 편향, 비선형성이 없는 $L$ 개의 단일 채널 $3 \times 3$ 합성곱층을 쌓아 $32 \times 32$ 입력에 적용하는 모델을 :numref:`subsec_learning_kernel` 의 내장 층을 사용하여 구축하십시오. 모든 커널 원소는 표준 가우시안 분포에서 독립적으로 추출하십시오.
    1. $L = 3$ 이고 커널을 한 번 추출했을 때, 마지막 층의 중심 부근에 있는 출력 유닛에서 입력까지 단위 기울기를 역전파하고 입력 기울기의 크기를 시각화하십시오. 그 지지집합(support)을 $7 \times 7$ 이론적 수용 영역과 비교하고, 크기가 어디에서 가장 큰지 답하십시오.
    1. 커널을 500번 추출하여 제곱 입력 기울기의 평균을 $L = 3$ 과 $L = 10$ 에 대해 각각 구하십시오. 평균 낸 각 맵을 행 방향으로 합산하고, 그 결과로 얻은 열 프로파일의 합이 1이 되도록 정규화하십시오. 프로파일의 표준 편차와 이론적 한 변의 길이 $2L + 1$ 의 비율이 깊이에 따라 어떻게 변합니까?
    1. :numref:`fig_conv_rf_paths` 는 $L = 3$ 일 때 1차원에서의 경로 수를 세어 보여 줍니다. 일반적인 $L$ 에 대해 행 방향으로 합산한 제곱 입력 기울기의 기댓값이 $L$ 개의 벡터 $(1, 1, 1)$ 을 합성곱한 결과에 비례함을 보이십시오. 정규화된 프로파일의 표준 편차가 $L$ 에 따라 어떻게 증가하는지 유도하십시오.

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
4. [code] **합성곱의 역방향 패스.** ● $\mathbf{Y}$ 를 입력 $\mathbf{X}$ 와 커널 $\mathbf{K}$ 의 단일 채널 교차 상관(`corr2d`로 계산된 결과)이라 하고, $\ell$ 을 $\mathbf{X}$ 와 $\mathbf{K}$ 에 오직 $\mathbf{Y}$ 를 통해서만 의존하는 스칼라 손실이라 합시다.
    1. $\partial \ell/\partial \mathbf{K}$ 와 $\partial \ell/\partial \mathbf{X}$ 를 $\partial \ell/\partial \mathbf{Y}$ 에 대한 식으로 :numref:`sec_backprop` 의 역전파 방정식 형태로 유도하십시오.
    1. 유도한 $\partial \ell/\partial \mathbf{K}$ 식을 손실 $\ell = \sum_{i,j} G_{ij} Y_{ij}$ 에 대해 자동 미분 없이 구현하십시오(여기서 $\mathbf{G}$ 는 $\mathbf{Y}$ 와 같은 모양의 고정된 무작위 행렬입니다). 이를 스텝 크기가 $10^{-2}$ 인 중심 유한 차분 기울기와 작은 무작위 `X` 및 `K`에서 비교하고 최대 상대 오차를 보고하십시오.
    1. 유도한 $\partial \ell/\partial \mathbf{K}$ 식을 자동 미분이 :numref:`subsec_learning_kernel` 의 실험에서 계산하는 커널 기울기와 대조하여 확인하십시오.

    *Michael Nielsen,
    [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html)
    6장 및 Stanford CS231n,
    [과제 2](https://cs231n.github.io/assignments2024/assignment2/),
    ConvolutionalNetworks 노트북에서 발췌·변형함.*

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
- Register (문체): Unified polite formal style (합쇼체: `-합니다`, `-입니다`, `-하십시오`, `-입니까?`) across body prose, captions, slides, and exercises; concise noun-phrase style (명사형) without trailing periods for headings, slide titles, and kickers.
- Parenthetical English & punctuation: Add English in ASCII parentheses without preceding space (`한국어(English)`) on first introduction of key terms; keep parentheses outside `*...*` emphasis so CommonMark delimiter flanking is always valid before Korean particles.
- Reference & inline-math spacing: Leave a space (or ASCII punctuation) between any role (`:numref:`, `:eqref:`, `:cite:`, `:citet:`) or inline math `$…$` and adjacent Hangul syllables per TRANSLATING.md §4.
- "locality" → "지역성"
- "edge" / "edge detection" / "edge detector" / "edge map" → "에지" / "에지 검출" / "에지 검출기" / "에지 맵"
- "sliding window" / "window" → "슬라이딩 윈도우" / "윈도우"
- "patch" / "patch matrix" → "패치" / "패치 행렬"
- "elementwise" → "원소별"
- "parameter" → "매개변수"
- "dense layer" → "밀집층"
- "end-to-end" → "종단간(end-to-end)"
- "flatten" / "flattened" → "평탄화하다" / "평탄화된"
- "unfold" / "unroll" → "펼치다"
- "sparse banded matrix" / "banded matrix" → "희소 띠 행렬" / "띠 행렬"
- "Toeplitz" → "퇴플리츠(Toeplitz)"
- "stacked layers" → "적층된 층"
- "theoretical receptive field" / "effective receptive field" → "이론적 수용 영역" / "유효 수용 영역"
- "blur kernel" / "sharpening kernel" → "블러(blur) 커널" / "샤프닝(sharpening) 커널"
- "downsample" → "다운샘플링하다"
- "locally connected layer" → "지역 연결층"
- "support" (of a function/gradient) → "지지집합(support)"
- "Discussions" → "토론"

=== NOTES ===
- 6ab5b67c: Glossary entry `neural network` → `신경망` appears in the English only inside the cited English book title *Neural Networks and Deep Learning* by Michael Nielsen, which is kept in its original title per TRANSLATING.md §7.
