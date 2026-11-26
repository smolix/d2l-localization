<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u01 lang=ko rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:33d3b96b kind=heading-->
# Convolutions for Images
<!--tr:33d3b96b-->
# 이미지를 위한 합성곱

<!--en:852f5868 kind=paragraph-->
In :numref:`sec_why-conv`, we derived the convolutional layer on paper from two
geometric principles: translation equivariance and locality. Now we see what
that formula actually computes on an image, and whether gradient descent can
discover useful kernels automatically from data.
<!--tr:852f5868-->
:numref:`sec_why-conv` 에서는 평행이동 등변성(translation equivariance)과 국소성이라는 두 기하학적 원리로부터 합성곱층(convolutional layer)을 종이 위에서 유도했습니다. 이제 그 공식이 실제 이미지에서 무엇을 계산하는지, 그리고 경사 하강법(gradient descent)이 데이터로부터 유용한 커널(kernel)을 자동으로 찾아낼 수 있는지 살펴봅니다.

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
밝은 영역과 어두운 영역이 만나는 수직 가장자리를 찾는 문제를 생각해 봅시다. 각 픽셀(pixel)에서 오른쪽 이웃 픽셀을 빼면($[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$) 균일한 영역 안에서는 0이 되고, 흰색에서 검은색으로 바뀌는 곳에서는 $+1$, 검은색에서 다시 흰색으로 돌아오는 곳에서는 $-1$ 이 됩니다. 가중치가 두 개인 검출기 $(1, -1)$ 을 이미지의 유효한 모든 위치에 걸쳐 훑으면 *가장자리 맵*(edge map)이 만들어집니다. 이 절에서는 2차원 교차 상관(cross-correlation)을 구현하고, 직접 설계한 가장자리 검출기를 시험하며, 경사 하강법이 $(1, -1)$ 을 데이터로부터 직접 *학습*하게 해 봅니다. 또한 합성곱이 하나의 행렬 곱셈(*im2col*)이 되는 방식을 보이고, 층을 쌓을수록 한 유닛의 수용 영역(receptive field)이 어떻게 넓어지는지 유도합니다.

<!--en:47958ed8 kind=heading-->
## Cross-Correlation
<!--tr:47958ed8-->
## 교차 상관

<!--en:5e4296fe kind=heading-->
### The Operation
<!--tr:5e4296fe-->
### 연산

<!--en:ae7e906d kind=paragraph-->
Leaving channels aside for the moment, let the input $\mathbf{X}$ be an
$n_\textrm{h} \times n_\textrm{w}$ matrix and the kernel $\mathbf{K}$ be a
$k_\textrm{h} \times k_\textrm{w}$ matrix. The kernel defines a sliding
*window* of shape $k_\textrm{h} \times k_\textrm{w}$. Placing the window's
upper-left corner at $(i, j)$, we multiply the covered input patch elementwise
with $\mathbf{K}$ and sum the products to produce the output entry at $(i, j)$:
<!--tr:ae7e906d-->
잠시 채널(channel)은 제쳐 두고, 입력 $\mathbf{X}$ 를 $n_\textrm{h} \times n_\textrm{w}$ 행렬로, 커널 $\mathbf{K}$ 를 $k_\textrm{h} \times k_\textrm{w}$ 행렬로 둡시다. 커널은 모양이 $k_\textrm{h} \times k_\textrm{w}$ 인 슬라이딩 *윈도*를 정의합니다. 윈도의 왼쪽 위 모서리를 $(i, j)$ 에 놓고, 윈도가 덮는 입력 패치를 $\mathbf{K}$ 와 원소별로 곱한 뒤 그 곱을 모두 더하면 $(i, j)$ 위치의 출력 원소가 나옵니다.

<!--en:3688b38b kind=paragraph-->
This is the *two-dimensional cross-correlation* of $\mathbf{X}$ with
$\mathbf{K}$, written $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ with
zero-based indices matching code. Whereas :numref:`sec_why-conv` centered
offsets symmetrically from $-\Delta$ to $\Delta$, anchoring at the upper-left
corner differs only by a fixed shift of $\Delta$ while naturally handling
even-sized kernels (such as a $2 \times 2$ kernel or our $1 \times 2$ edge
detector) that have no center pixel.
<!--tr:3688b38b-->
이것이 $\mathbf{X}$ 와 $\mathbf{K}$ 의 *2차원 교차 상관*이며, 코드와 맞도록 0부터 시작하는 인덱스를 써서 $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ 로 씁니다. :numref:`sec_why-conv` 에서는 오프셋을 $-\Delta$ 에서 $\Delta$ 까지 대칭으로 잡았지만, 왼쪽 위 모서리를 기준으로 삼으면 $\Delta$ 만큼의 고정된 이동만 달라질 뿐이고, 중심 픽셀이 없는 짝수 크기의 커널(예를 들어 $2 \times 2$ 커널이나 $1 \times 2$ 가장자리 검출기)도 자연스럽게 다룰 수 있습니다.

<!--en:fa1b7da9 kind=caption-->
![Two-dimensional cross-correlation. The shaded portions are the first output element and the input and kernel elements used to compute it: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
<!--tr:fa1b7da9-->
![2차원 교차 상관. 음영 부분은 첫 번째 출력 원소와, 그것을 계산하는 데 쓰인 입력 및 커널 원소입니다: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)

<!--en:2f8e5deb kind=paragraph-->
In :numref:`fig_correlation`, we slide a $2 \times 2$ kernel across a
$3 \times 3$ input one pixel at a time, left to right and top to bottom. The
window fits cleanly at four positions, yielding the $2 \times 2$ output:
<!--tr:2f8e5deb-->
:numref:`fig_correlation` 에서는 $2 \times 2$ 커널을 $3 \times 3$ 입력 위에서 왼쪽에서 오른쪽, 위에서 아래로 한 픽셀씩 밀면서 적용합니다. 윈도가 온전히 들어맞는 위치는 네 곳이며, 그 결과 $2 \times 2$ 출력이 나옵니다.

<!--en:4741e012 kind=paragraph-->
Notice that the output shrinks slightly along each axis because the kernel must
stay wholly inside the input. Fitting a width-$k$ window into length $n$ leaves
$n - k + 1$ valid positions, so the output shape is
<!--tr:4741e012-->
커널이 입력 안에 온전히 들어 있어야 하므로 출력이 각 축을 따라 조금 줄어든다는 점에 주목하세요. 너비 $k$ 인 윈도를 길이 $n$ 인 입력에 넣을 수 있는 유효한 위치는 $n - k + 1$ 곳이므로, 출력의 모양(shape)은 다음과 같습니다.

<!--en:12d81c14 kind=paragraph-->
Thus a $3 \times 3$ input with a $2 \times 2$ kernel yields $2 \times 2$, a
$6 \times 8$ image with a $1 \times 2$ kernel yields $6 \times 7$, and a
$28 \times 28$ Fashion-MNIST image with a $5 \times 5$ kernel yields
$24 \times 24$. Whenever we want to preserve spatial dimensions, we surround the
input with a border of zeros (*padding*), which we examine alongside strides and
dilation in :numref:`sec_padding`.
<!--tr:12d81c14-->
따라서 $3 \times 3$ 입력에 $2 \times 2$ 커널을 쓰면 $2 \times 2$, $6 \times 8$ 이미지에 $1 \times 2$ 커널을 쓰면 $6 \times 7$, $28 \times 28$ Fashion-MNIST 이미지에 $5 \times 5$ 커널을 쓰면 $24 \times 24$ 가 됩니다. 공간 차원을 유지하고 싶을 때는 입력 둘레를 0으로 된 테두리로 감싸는데(*패딩*(padding)), 이는 :numref:`sec_padding` 에서 스트라이드(stride), 팽창(dilation)과 함께 다룹니다.

<!--en:869d223f kind=heading-->
### Implementation
<!--tr:869d223f-->
### 구현

<!--en:6a869f7c kind=paragraph-->
We implement :eqref:`eq_corr2d` directly in `corr2d` using a pair of loops over
every output coordinate.
<!--tr:6a869f7c-->
:eqref:`eq_corr2d` 를 `corr2d` 에서 직접 구현합니다. 모든 출력 좌표에 대해 반복문 두 개를 사용합니다.

<!--en:81e1f383 kind=paragraph-->
To verify `corr2d` against our hand calculation in :numref:`fig_correlation`,
we construct the $3 \times 3$ input tensor `X` and $2 \times 2$ kernel `K` and
evaluate `corr2d(X, K)`.
<!--tr:81e1f383-->
`corr2d` 를 :numref:`fig_correlation` 의 손계산과 대조해 검증하기 위해, $3 \times 3$ 입력 텐서 `X` 와 $2 \times 2$ 커널 `K` 를 만들고 `corr2d(X, K)` 를 계산합니다.

<!--en:f567e1b2 kind=paragraph-->
The output reproduces :numref:`fig_correlation` entry for entry (and as shown in
:numref:`sec_why-conv`, strict mathematical convolution with a flipped kernel
$\mathbf{K}'$ learns the exact same family of layers, so we follow standard
deep learning terminology and call cross-correlation "convolution").
<!--tr:f567e1b2-->
출력은 :numref:`fig_correlation` 과 원소 하나하나까지 일치합니다. (:numref:`sec_why-conv` 에서 보았듯이, 뒤집은 커널 $\mathbf{K}'$ 를 쓰는 엄밀한 수학적 합성곱도 정확히 같은 계열의 층을 학습하므로, 우리는 딥러닝의 표준 용어를 따라 교차 상관을 "합성곱"이라고 부릅니다.)

<!--en:bfe11c62 kind=heading-->
## Convolutional Layers
<!--tr:bfe11c62-->
## 합성곱층

<!--en:2cdc020d kind=heading-->
### The Layer
<!--tr:2cdc020d-->
### 층

<!--en:c6371ece kind=paragraph-->
A two-dimensional convolutional layer cross-correlates its input with a learned
kernel and adds a scalar bias. Just like our MLP weights in
:numref:`chap_perceptrons`, we initialize the kernel from
$\mathcal{N}(0, 0.01^2)$ and the bias to zero. Below, `Conv2D` registers both
parameters and invokes `corr2d` in its forward pass; a layer with a kernel of
height $h$ and width $w$ is called an $h \times w$ convolutional layer.
<!--tr:c6371ece-->
2차원 합성곱층은 입력을 학습된 커널과 교차 상관한 다음 스칼라 편향(bias)을 더합니다. :numref:`chap_perceptrons` 의 다층 퍼셉트론(MLP)의 가중치(weight)와 마찬가지로, 커널은 $\mathcal{N}(0, 0.01^2)$ 에서, 편향은 0으로 초기화합니다. 아래의 `Conv2D` 는 두 매개변수를 모두 등록하고 순방향 패스(forward pass)에서 `corr2d` 를 호출합니다. 커널의 높이가 $h$, 너비가 $w$ 인 층을 $h \times w$ 합성곱층이라고 부릅니다.

<!--en:29cd9061 kind=paragraph-->
Unlike a dense layer, a convolutional layer is not locked to a fixed input
resolution because its parameters depend only on the kernel size. The exact same
$2 \times 2$ layer (5 parameters total) processes both our $3 \times 3$ tensor
`X` and a $6 \times 8$ image, returning outputs of shapes $2 \times 2$ and
$5 \times 7$:
<!--tr:29cd9061-->
완전 연결층과 달리 합성곱층은 매개변수가 커널 크기(kernel size)에만 의존하므로 고정된 입력 해상도에 묶이지 않습니다. 똑같은 $2 \times 2$ 층(매개변수는 모두 5개)이 $3 \times 3$ 텐서 `X` 와 $6 \times 8$ 이미지를 모두 처리하며, 출력의 모양은 각각 $2 \times 2$ 와 $5 \times 7$ 입니다.

<!--en:c9e5aa07 kind=heading-->
### Edge Detection
<!--tr:c9e5aa07-->
### 가장자리 검출

<!--en:3d9a357b kind=paragraph-->
We now test the vertical edge detector from the introduction on a
$6 \times 8$ synthetic image whose middle four columns are black (`0`) and outer
columns are white (`1`).
<!--tr:3d9a357b-->
이제 도입부의 수직 가장자리 검출기를 $6 \times 8$ 합성 이미지에 시험합니다. 이 이미지는 가운데 네 열이 검은색(`0`)이고 바깥쪽 열은 흰색(`1`)입니다.

<!--en:68513ff0 kind=paragraph-->
We define the $1 \times 2$ kernel
<!--tr:68513ff0-->
$1 \times 2$ 커널을 다음과 같이 정의합니다.

<!--en:eb45803b kind=paragraph-->
At every position $(i, j)$, this computes the horizontal *finite difference*
$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, a discrete approximation (up to sign)
of the horizontal derivative of brightness, which vanishes across flat regions
and spikes wherever intensity jumps.
<!--tr:eb45803b-->
이 커널은 모든 위치 $(i, j)$ 에서 수평 *유한 차분*(finite difference) $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ 을 계산합니다. 이는 (부호를 제외하면) 밝기의 수평 도함수(derivative)에 대한 이산 근사이며, 평탄한 영역에서는 0이 되고 밝기가 급변하는 곳에서 값이 솟구칩니다.

<!--en:e810b3a9 kind=paragraph-->
Cross-correlating the $6 \times 8$ stripe image `X` with the $1 \times 2$
horizontal-difference kernel `K` via `corr2d(X, K)` produces the $6 \times 7$
edge map `Y`:
<!--tr:e810b3a9-->
$6 \times 8$ 줄무늬 이미지 `X` 와 $1 \times 2$ 수평 차분 커널 `K` 를 `corr2d(X, K)` 로 교차 상관하면 $6 \times 7$ 가장자리 맵 `Y` 가 나옵니다.

<!--en:568358e4 kind=paragraph-->
Sure enough, `Y` fires `+1` along the white-to-black boundary, `-1` along the
black-to-white boundary, and `0` everywhere else. Applying the exact same kernel
to the transposed image (turning the vertical stripe into a horizontal one)
yields all zeros:
<!--tr:568358e4-->
예상대로 `Y` 는 흰색에서 검은색으로 바뀌는 경계를 따라 `+1`, 검은색에서 흰색으로 바뀌는 경계를 따라 `-1`, 그 외의 모든 곳에서 `0` 을 냅니다. 똑같은 커널을 전치한 이미지(수직 줄무늬를 수평 줄무늬로 바꾼 것)에 적용하면 모두 0이 됩니다.

<!--en:71c591a4 kind=paragraph-->
Clearly `[1, -1]` responds strictly to vertical edges. Detecting horizontal
edges, diagonals, corners, or textures requires different kernels altogether.
<!--tr:71c591a4-->
`[1, -1]` 은 수직 가장자리에만 반응함이 분명합니다. 수평 가장자리, 대각선, 모서리, 질감을 검출하려면 전혀 다른 커널이 필요합니다.

<!--en:f0af4887 kind=heading-->
### Learning a Kernel
<!--tr:f0af4887-->
### 커널 학습

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
`[1, -1]` 을 직접 설계하는 일은 원하는 특징(feature)을 정확히 알 때는 쉽지만, 깊은 신경망 전체에 걸친 수천 개의 중간 필터(filter)를 사람이 일일이 지정할 수 있는 엔지니어는 없습니다. 경사 하강법은 입출력 쌍 `(X, Y)` 만으로 커널 `[1, -1]` 을 복원할 수 있을까요? 각 프레임워크에 내장된 2D 합성곱층(편향은 생략하고 가중치는 $\mathcal{N}(0, 0.01^2)$ 에서 초기화)을 사용해, 42개 출력 원소 전체에 걸친 제곱 오차(squared error)의 합을 학습률(learning rate) $0.03$ 으로 10번의 기울기 갱신 동안 최소화하며 이를 시험합니다. 내장 층은 배치(batch) 축과 채널 축을 포함한 4차원 텐서를 요구하므로(:numref:`sec_channels`), `X` 와 `Y` 의 모양을 그에 맞게 바꿉니다.

<!--en:c29c5d61 kind=paragraph-->
Initially the loss sits near 12 (the near-zero random kernel misses all 12
$\pm 1$ edge pixels by roughly 1), and within just nine steps it drops below
0.01. Inspecting the learned kernel confirms what it found:
<!--tr:c29c5d61-->
처음에 손실(loss)은 12 근처입니다(0에 가까운 무작위 커널은 $\pm 1$ 인 12개의 가장자리 픽셀을 모두 대략 1씩 빗나갑니다). 그리고 단 아홉 단계 만에 0.01 아래로 떨어집니다. 학습된 커널을 살펴보면 무엇을 찾았는지 확인할 수 있습니다.

<!--en:67d25b0b kind=paragraph-->
Both entries land within roughly 0.02 of `[1, -1]`. Nothing in this loop knew
anything about edges: given a loss signal, gradient descent discovered the
filter on its own. Here we supervised the single layer directly with `Y`; in a
deep CNN, backpropagation learns every layer's kernels end to end from the final
task loss alone.
<!--tr:67d25b0b-->
두 값 모두 `[1, -1]` 에서 대략 0.02 이내에 있습니다. 이 반복문은 가장자리에 대해 아무것도 몰랐지만, 손실 신호만 주어지자 경사 하강법이 스스로 그 필터를 찾아냈습니다. 여기서는 `Y` 로 단일 층을 직접 지도했지만, 깊은 합성곱 신경망(CNN)에서는 역전파(backpropagation)가 최종 과제의 손실만으로 모든 층의 커널을 종단간으로 학습합니다.

<!--en:91b0010d kind=slide_text-->
Convolutions for Images<br>**2D cross-correlation · edge detection · learned kernels · im2col · receptive fields**
<!--tr:91b0010d-->
이미지를 위한 합성곱<br>**2D 교차 상관 · 가장자리 검출 · 학습된 커널 · im2col · 수용 영역**

<!--en:1d94295a kind=slide_title-->
::: {.slide title="Detecting Local Edges Across an Image"}
<!--tr:1d94295a-->
::: {.slide title="이미지 전반의 국소 가장자리 검출"}

<!--en:687ed302 kind=kicker-->
[Motivation & Core Mental Model]{.kicker}
<!--tr:687ed302-->
[동기와 핵심 사고 모형]{.kicker}

<!--en:05fceed0 kind=slide_text-->
- **Synthetic $6 \times 8$ test image:** White outer columns (`1`) surrounding a 4-column black vertical band (`0`).
<!--tr:05fceed0-->
- **합성 $6 \times 8$ 시험 이미지:** 흰색 바깥 열(`1`)이 4열 폭의 검은 수직 띠(`0`)를 둘러싸고 있습니다.

<!--en:71e700b3 kind=slide_text-->
- **Horizontal finite difference:** Sweeping $\mathbf{K} = [1, -1]$ computes $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, firing `+1` at white-to-black and `-1` at black-to-white edges.
<!--tr:71e700b3-->
- **수평 유한 차분:** $\mathbf{K} = [1, -1]$ 로 훑으면 $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ 이 계산되어, 흰색에서 검은색으로의 가장자리에서 `+1`, 검은색에서 흰색으로의 가장자리에서 `-1` 이 나옵니다.

<!--en:957dc033 kind=slide_text-->
- **Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent.
<!--tr:957dc033-->
- **학습 대 직접 설계:** 깊은 합성곱 신경망은 이런 커널 수천 개를 경사 하강법으로 자동으로 찾아냅니다.

<!--en:cb1045e1 kind=slide_title-->
::: {.slide title="The 2D Cross-Correlation Operation"}
<!--tr:cb1045e1-->
::: {.slide title="2D 교차 상관 연산"}

<!--en:17bcece5 kind=kicker-->
[Sliding-Window Inner Products]{.kicker}
<!--tr:17bcece5-->
[슬라이딩 윈도 내적]{.kicker}

<!--en:8d45aae8 kind=slide_text-->
Given an $n_\textrm{h} \times n_\textrm{w}$ input matrix $\mathbf{X}$ and a $k_\textrm{h} \times k_\textrm{w}$ kernel $\mathbf{K}$, the 2D cross-correlation $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ places the window's top-left corner at $(i, j)$ and sums elementwise products:
<!--tr:8d45aae8-->
$n_\textrm{h} \times n_\textrm{w}$ 입력 행렬 $\mathbf{X}$ 와 $k_\textrm{h} \times k_\textrm{w}$ 커널 $\mathbf{K}$ 가 주어지면, 2D 교차 상관 $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ 는 윈도의 왼쪽 위 모서리를 $(i, j)$ 에 놓고 원소별 곱을 더합니다.

<!--en:e484380c kind=slide_text-->
- **Top-left anchor:** Zero-based indices $a, b \ge 0$ match array indexing in code and handle even-sized kernels (like $2 \times 2$ or $1 \times 2$) cleanly.
<!--tr:e484380c-->
- **왼쪽 위 기준:** 0부터 시작하는 인덱스 $a, b \ge 0$ 은 코드의 배열 인덱싱과 일치하며, 짝수 크기 커널($2 \times 2$ 또는 $1 \times 2$ 등)도 깔끔하게 처리합니다.

<!--en:137c0980 kind=slide_text-->
- **Shaded top-left output $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.
<!--tr:137c0980-->
- **음영 처리된 왼쪽 위 출력 $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.

<!--en:575c4aea kind=slide_title-->
::: {.slide title="Valid Window Positions and Output Spatial Shape"}
<!--tr:575c4aea-->
::: {.slide title="유효한 윈도 위치와 출력의 공간 모양"}

<!--en:7eda4531 kind=kicker-->
[Why Unpadded Convolutions Shrink the Grid]{.kicker}
<!--tr:7eda4531-->
[패딩 없는 합성곱이 격자를 줄이는 이유]{.kicker}

<!--en:891243ae kind=slide_text-->
- **Valid window placements:** A $k_\textrm{h} \times k_\textrm{w}$ window fits inside an $n_\textrm{h} \times n_\textrm{w}$ grid at $n - k + 1$ positions per axis.
<!--tr:891243ae-->
- **유효한 윈도 배치:** $k_\textrm{h} \times k_\textrm{w}$ 윈도는 $n_\textrm{h} \times n_\textrm{w}$ 격자 안에 축마다 $n - k + 1$ 곳에 들어맞습니다.

<!--en:55cc8c32 kind=slide_text-->
- **Shape examples:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.
<!--tr:55cc8c32-->
- **모양 예시:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$), $28 \times 28 \star 5 \times 5 \to 24 \times 24$.

<!--en:1e3463da kind=slide_title-->
::: {.slide title="Implementing 2D Cross-Correlation in Code"}
<!--tr:1e3463da-->
::: {.slide title="코드로 구현하는 2D 교차 상관"}

<!--en:07566992 kind=kicker-->
[Direct Nested-Loop Implementation]{.kicker}
<!--tr:07566992-->
[직접 중첩 반복문으로 구현]{.kicker}

<!--en:202aa5d3 kind=slide_text-->
The function `corr2d` allocates the $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ output tensor and computes each entry as the sum of `X[i:i+h, j:j+w] * K`:
<!--tr:202aa5d3-->
함수 `corr2d` 는 $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ 출력 텐서를 할당하고, 각 원소를 `X[i:i+h, j:j+w] * K` 의 합으로 계산합니다.

<!--en:bef130ba kind=slide_text-->
Each iteration slices a $k_\textrm{h} \times k_\textrm{w}$ sub-window of `X`, multiplies elementwise with `K`, and reduces via `sum`.
<!--tr:bef130ba-->
매 반복마다 `X` 의 $k_\textrm{h} \times k_\textrm{w}$ 부분 윈도를 슬라이스(slice)해 `K` 와 원소별로 곱하고, `sum` 으로 줄입니다.

<!--en:4ee9f6b1 kind=slide_title-->
::: {.slide title="Verifying corr2d and the Flipped-Kernel Equivalence"}
<!--tr:4ee9f6b1-->
::: {.slide title="corr2d 검증과 뒤집은 커널과의 동치성"}

<!--en:2b0be423 kind=kicker-->
[Cross-Correlation vs. Mathematical Convolution]{.kicker}
<!--tr:2b0be423-->
[교차 상관 대 수학적 합성곱]{.kicker}

<!--en:74b9b745 kind=slide_text-->
- **180° flipped kernel $\mathbf{K}'$:** Cross-correlating with $\mathbf{K}$ equals convolving with $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.
<!--tr:74b9b745-->
- **180° 뒤집은 커널 $\mathbf{K}'$:** $\mathbf{K}$ 로 교차 상관하는 것은 $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$ 로 합성곱하는 것과 같습니다.

<!--en:5c8af7cd kind=slide_text-->
- **Identical learned class:** Frameworks compute cross-correlation and call it convolution.
<!--tr:5c8af7cd-->
- **같은 학습 계열:** 프레임워크는 교차 상관을 계산하면서 이를 합성곱이라고 부릅니다.

<!--en:a69e29b5 kind=slide_title-->
::: {.slide title="Building a 2D Convolutional Layer Module"}
<!--tr:a69e29b5-->
::: {.slide title="2D 합성곱층 모듈 만들기"}

<!--en:55763603 kind=kicker-->
[Learnable Kernel Weights Plus Scalar Bias]{.kicker}
<!--tr:55763603-->
[학습 가능한 커널 가중치와 스칼라 편향]{.kicker}

<!--en:5b8bad24 kind=slide_text-->
A convolutional layer wraps `corr2d` with a learnable $k_\textrm{h} \times k_\textrm{w}$ weight parameter (initialized from $\mathcal{N}(0, 0.01^2)$) and a scalar bias (initialized to $0$):
<!--tr:5b8bad24-->
합성곱층은 `corr2d` 를 감싸면서, 학습 가능한 $k_\textrm{h} \times k_\textrm{w}$ 가중치 매개변수($\mathcal{N}(0, 0.01^2)$ 에서 초기화)와 스칼라 편향($0$ 으로 초기화)을 둡니다.

<!--en:c16d97d6 kind=slide_title-->
::: {.slide title="Resolution Independence of Convolutional Layers"}
<!--tr:c16d97d6-->
::: {.slide title="합성곱층의 해상도 독립성"}

<!--en:a7d1e48f kind=kicker-->
[Fixed Parameters Across Arbitrary Input Sizes]{.kicker}
<!--tr:a7d1e48f-->
[임의의 입력 크기에 걸친 고정된 매개변수]{.kicker}

<!--en:53a2bce7 kind=slide_text-->
- **Decoupled from $H \times W$:** The exact same 5 parameters ($2 \times 2$ kernel + 1 bias) process $3 \times 3$, $6 \times 8$, or megapixel inputs without reshaping weights.
<!--tr:53a2bce7-->
- **$H \times W$ 와 무관:** 똑같은 5개 매개변수($2 \times 2$ 커널 + 편향 1개)가 가중치 모양을 바꾸지 않고도 $3 \times 3$, $6 \times 8$, 메가픽셀 입력을 처리합니다.

<!--en:017fec24 kind=slide_title-->
::: {.slide title="Constructing a Synthetic Vertical-Edge Image"}
<!--tr:017fec24-->
::: {.slide title="합성 수직 가장자리 이미지 만들기"}

<!--en:896e641c kind=kicker-->
[A 6×8 Binary Test Pattern]{.kicker}
<!--tr:896e641c-->
[6×8 이진 시험 패턴]{.kicker}

<!--en:0a4b456b kind=slide_text-->
We construct the $6 \times 8$ test image `X` with ones on the left and right borders and zeros across the middle four columns (`X[:, 2:6] = 0`):
<!--tr:0a4b456b-->
$6 \times 8$ 시험 이미지 `X` 를 만듭니다. 왼쪽과 오른쪽 테두리는 1, 가운데 네 열은 0입니다(`X[:, 2:6] = 0`).

<!--en:dfc7e3e3 kind=slide_text-->
- **Piecewise-constant columns:** Intensity is constant (`1` or `0`) within columns $0\text{--}1$, $2\text{--}5$, and $6\text{--}7$, jumping only between columns $1 \to 2$ and $5 \to 6$.
<!--tr:dfc7e3e3-->
- **구간별 상수 열:** 밝기는 열 $0\text{--}1$, $2\text{--}5$, $6\text{--}7$ 안에서 일정하며(`1` 또는 `0`), 열 $1 \to 2$ 와 $5 \to 6$ 사이에서만 바뀝니다.

<!--en:3ce31cbe kind=slide_title-->
::: {.slide title="Applying the Horizontal Finite-Difference Kernel"}
<!--tr:3ce31cbe-->
::: {.slide title="수평 유한 차분 커널 적용하기"}

<!--en:b37a1029 kind=kicker-->
[Detecting Vertical Edges with K = [1, -1]]{.kicker}
<!--tr:b37a1029-->
[K = [1, -1] 로 수직 가장자리 검출]{.kicker}

<!--en:9be85019 kind=slide_text-->
Cross-correlating `X` with the $1 \times 2$ kernel $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ computes $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ at every horizontal pair:
<!--tr:9be85019-->
`X` 를 $1 \times 2$ 커널 $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ 과 교차 상관하면 모든 수평 쌍에서 $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ 이 계산됩니다.

<!--en:f649cfae kind=slide_text-->
- **Signed edge map:** Column 1 fires `+1` (white-to-black transition), column 5 fires `-1` (black-to-white transition), and all constant interior columns evaluate to `0`.
<!--tr:f649cfae-->
- **부호 있는 가장자리 맵:** 열 1은 `+1`(흰색에서 검은색으로의 전이), 열 5는 `-1`(검은색에서 흰색으로의 전이)을 내고, 값이 일정한 나머지 열은 모두 `0` 이 됩니다.

<!--en:e6d155be kind=slide_title-->
::: {.slide title="Orientation Selectivity of Edge Kernels"}
<!--tr:e6d155be-->
::: {.slide title="가장자리 커널의 방향 선택성"}

<!--en:b525cde4 kind=kicker-->
[Why One Kernel Cannot Detect All Orientations]{.kicker}
<!--tr:b525cde4-->
[커널 하나로 모든 방향을 검출할 수 없는 이유]{.kicker}

<!--en:38159de6 kind=slide_text-->
- **Blind to horizontal edges:** Transposing `X` (`X.T`) makes horizontal neighbors identical, so $[1, -1]$ outputs all zeros.
<!--tr:38159de6-->
- **수평 가장자리는 보지 못함:** `X` 를 전치(`X.T`)하면 수평으로 이웃한 값이 같아지므로 $[1, -1]$ 의 출력은 모두 0입니다.

<!--en:f0f30f76 kind=slide_text-->
- **Filter banks (§{sec_channels}):** Vision layers learn parallel channels for horizontal, vertical, and diagonal orientations.
<!--tr:f0f30f76-->
- **필터 뱅크(§{sec_channels}):** 비전 층은 수평, 수직, 대각선 방향을 위한 채널을 병렬로 학습합니다.

<!--en:b755e31b kind=slide_title-->
::: {.slide title="Learning an Edge Detector via Gradient Descent"}
<!--tr:b755e31b-->
::: {.slide title="경사 하강법으로 가장자리 검출기 학습하기"}

<!--en:86bbeea7 kind=kicker-->
[Supervising a Built-In Convolutional Layer on (X, Y)]{.kicker}
<!--tr:86bbeea7-->
[내장 합성곱층을 (X, Y) 로 지도하기]{.kicker}

<!--en:a569780c kind=slide_text-->
- **4D framework tensors:** Reshape `X` and `Y` to 4D tensors (one example, one channel) and minimize $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ at $\eta = 0.03$.
<!--tr:a569780c-->
- **4차원 프레임워크 텐서:** `X` 와 `Y` 를 4차원 텐서(예제 하나, 채널 하나)로 바꾸고, $\eta = 0.03$ 으로 $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ 를 최소화합니다.

<!--en:2489ce6e kind=slide_title-->
::: {.slide title="Inspecting the Learned Kernel Weights"}
<!--tr:2489ce6e-->
::: {.slide title="학습된 커널 가중치 살펴보기"}

<!--en:7e4ef44b kind=kicker-->
[Why Initial Loss Is 12 and Converges to [1, -1]]{.kicker}
<!--tr:7e4ef44b-->
[초기 손실이 12인 이유와 [1, -1] 로의 수렴]{.kicker}

<!--en:9cd1c645 kind=slide_text-->
- **Why loss starts at $12.0$:** Near-zero weights predict $\hat{Y} \approx 0$, missing the twelve $\pm 1$ edge pixels in `Y` ($12 \times 1^2 = 12$).
<!--tr:9cd1c645-->
- **손실이 $12.0$ 에서 시작하는 이유:** 0에 가까운 가중치는 $\hat{Y} \approx 0$ 을 예측해 `Y` 의 $\pm 1$ 가장자리 픽셀 열두 개를 놓칩니다($12 \times 1^2 = 12$).

<!--en:456670c5 kind=slide_text-->
- **Convergence in 9 steps:** SGD recovers $[0.99, -0.99]$ automatically from data.
<!--tr:456670c5-->
- **9단계 만의 수렴(convergence):** 확률적 경사 하강법(SGD)은 $[0.99, -0.99]$ 를 데이터로부터 자동으로 복원합니다.

=== DECISIONS ===
- Register: 합니다체 (습니다/입니다), "우리"를 생략한 담담한 서술; 명령은 ~세요/~봅시다.
- First use of a key term in a page: Korean(English) with no space before the parenthesis, e.g. 합성곱층(convolutional layer); later uses Korean only. Emphasized terms keep their `*…*`.
- Particles after `$math$`/code/`:numref:`: a space is inserted (`$\mathbf{X}$ 는`), per TRANSLATING.md section 4.
- "window" → 윈도; "patch" → 패치; "edge map" → 가장자리 맵; "edge detector" → 가장자리 검출기; "finite difference" → 유한 차분; "border" → 테두리; "feature map" → 특징 맵 (follows glossary 특징) (to be used in later units).
- "im2col", "Conv2D", "corr2d" stay in English. "cross-correlation" → 교차 상관; "convolution" → 합성곱 (including slide titles); "(MLP) weights" → 가중치.
- Slide kickers/titles: noun-phrase style, no final period; slide bullets keep their bold lead-in with a trailing colon.
=== NOTES ===
- Check warnings "numbers not found" are false matches: numbers kept as digits but attached to Korean counters or particles (5개, 12개, 9단계 ...). "math tokens reordered" and "code tokens reordered" (61cacfaf) come from Korean SOV order.
