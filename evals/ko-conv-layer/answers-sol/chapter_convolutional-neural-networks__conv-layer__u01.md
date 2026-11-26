<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u01 lang=ko rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:33d3b96b kind=heading-->
# Convolutions for Images
<!--tr:33d3b96b-->
# 이미지에 적용하는 합성곱

<!--en:852f5868 kind=paragraph-->
In :numref:`sec_why-conv`, we derived the convolutional layer on paper from two
geometric principles: translation equivariance and locality. Now we see what
that formula actually computes on an image, and whether gradient descent can
discover useful kernels automatically from data.
<!--tr:852f5868-->
:numref:`sec_why-conv` 에서는 평행이동 등변성과 국소성이라는 두 가지 기하학적 원리로부터 합성곱층의 식을 유도했다. 이제 그 식이 이미지에서 실제로 무엇을 계산하는지, 그리고 경사 하강법이 데이터에서 유용한 커널을 스스로 찾아낼 수 있는지 살펴보자.

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
밝은 영역과 어두운 영역이 만나는 수직 경계를 찾는다고 하자. 각 픽셀 값에서 바로 오른쪽 픽셀 값을 빼면 ($[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$) 밝기가 일정한 영역에서는 0, 흰색에서 검은색으로 바뀌는 곳에서는 $+1$, 검은색에서 다시 흰색으로 바뀌는 곳에서는 $-1$ 이 나온다. 가중치가 두 개인 검출기 $(1, -1)$ 을 이미지의 모든 유효 위치에 적용하면 *경계 지도*를 얻는다. 이 절에서는 2차원 교차 상관을 구현하고, 직접 설계한 경계 검출기를 시험하고, 경사 하강법으로 데이터에서 직접 $(1, -1)$ 을 *학습*한다. 이어서 합성곱을 한 번의 행렬 곱셈으로 바꾸는 방법(*im2col*)을 살펴보고, 층을 쌓을 때 유닛의 수용 영역이 어떻게 넓어지는지 유도한다.

<!--en:47958ed8 kind=heading-->
## Cross-Correlation
<!--tr:47958ed8-->
## 교차 상관

<!--en:5e4296fe kind=heading-->
### The Operation
<!--tr:5e4296fe-->
### 연산 방식

<!--en:ae7e906d kind=paragraph-->
Leaving channels aside for the moment, let the input $\mathbf{X}$ be an
$n_\textrm{h} \times n_\textrm{w}$ matrix and the kernel $\mathbf{K}$ be a
$k_\textrm{h} \times k_\textrm{w}$ matrix. The kernel defines a sliding
*window* of shape $k_\textrm{h} \times k_\textrm{w}$. Placing the window's
upper-left corner at $(i, j)$, we multiply the covered input patch elementwise
with $\mathbf{K}$ and sum the products to produce the output entry at $(i, j)$:
<!--tr:ae7e906d-->
채널은 일단 제쳐 두고, 입력 $\mathbf{X}$ 는 $n_\textrm{h} \times n_\textrm{w}$ 행렬, 커널 $\mathbf{K}$ 는 $k_\textrm{h} \times k_\textrm{w}$ 행렬이라 하자. 커널은 모양이 $k_\textrm{h} \times k_\textrm{w}$ 인 이동 *윈도*를 정한다. 윈도의 왼쪽 위 꼭짓점을 $(i, j)$ 에 놓고, 윈도가 덮는 입력 패치와 $\mathbf{K}$ 의 대응 성분을 곱한 뒤 그 곱을 모두 더해 $(i, j)$ 위치의 출력 성분을 구한다.

<!--en:3688b38b kind=paragraph-->
This is the *two-dimensional cross-correlation* of $\mathbf{X}$ with
$\mathbf{K}$, written $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ with
zero-based indices matching code. Whereas :numref:`sec_why-conv` centered
offsets symmetrically from $-\Delta$ to $\Delta$, anchoring at the upper-left
corner differs only by a fixed shift of $\Delta$ while naturally handling
even-sized kernels (such as a $2 \times 2$ kernel or our $1 \times 2$ edge
detector) that have no center pixel.
<!--tr:3688b38b-->
이 연산은 $\mathbf{X}$ 와 $\mathbf{K}$ 의 *2차원 교차 상관*이다. 코드와 마찬가지로 인덱스를 0부터 매기며 $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ 로 쓴다. :numref:`sec_why-conv` 에서는 오프셋을 $-\Delta$ 부터 $\Delta$ 까지 중심을 기준으로 대칭 배치했다. 왼쪽 위 꼭짓점을 기준으로 삼으면 위치가 $\Delta$ 만큼 일정하게 이동할 뿐이며, $2 \times 2$ 커널이나 앞서 본 $1 \times 2$ 경계 검출기처럼 중심 픽셀이 없는 짝수 크기의 커널도 자연스럽게 다룰 수 있다.

<!--en:fa1b7da9 kind=caption-->
![Two-dimensional cross-correlation. The shaded portions are the first output element and the input and kernel elements used to compute it: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
<!--tr:fa1b7da9-->
![2차원 교차 상관. 음영으로 표시한 부분은 첫 번째 출력 성분과 그 값을 계산하는 데 쓰인 입력 및 커널 성분이다. $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)

<!--en:2f8e5deb kind=paragraph-->
In :numref:`fig_correlation`, we slide a $2 \times 2$ kernel across a
$3 \times 3$ input one pixel at a time, left to right and top to bottom. The
window fits cleanly at four positions, yielding the $2 \times 2$ output:
<!--tr:2f8e5deb-->
:numref:`fig_correlation` 에서는 $2 \times 2$ 커널을 $3 \times 3$ 입력 위에서 왼쪽에서 오른쪽으로, 위에서 아래로 한 픽셀씩 옮긴다. 윈도가 입력 안에 온전히 들어가는 위치는 네 곳이므로 $2 \times 2$ 출력을 얻는다.

<!--en:4741e012 kind=paragraph-->
Notice that the output shrinks slightly along each axis because the kernel must
stay wholly inside the input. Fitting a width-$k$ window into length $n$ leaves
$n - k + 1$ valid positions, so the output shape is
<!--tr:4741e012-->
커널이 입력 안에 완전히 들어가야 하므로 출력은 각 축을 따라 조금씩 줄어든다. 너비가 $k$ 인 윈도를 길이가 $n$ 인 구간에 놓을 수 있는 위치는 $n - k + 1$ 곳이므로 출력의 모양은 다음과 같다.

<!--en:12d81c14 kind=paragraph-->
Thus a $3 \times 3$ input with a $2 \times 2$ kernel yields $2 \times 2$, a
$6 \times 8$ image with a $1 \times 2$ kernel yields $6 \times 7$, and a
$28 \times 28$ Fashion-MNIST image with a $5 \times 5$ kernel yields
$24 \times 24$. Whenever we want to preserve spatial dimensions, we surround the
input with a border of zeros (*padding*), which we examine alongside strides and
dilation in :numref:`sec_padding`.
<!--tr:12d81c14-->
따라서 $3 \times 3$ 입력에 $2 \times 2$ 커널을 적용하면 $2 \times 2$, $6 \times 8$ 이미지에 $1 \times 2$ 커널을 적용하면 $6 \times 7$, $28 \times 28$ Fashion-MNIST 이미지에 $5 \times 5$ 커널을 적용하면 $24 \times 24$ 출력이 나온다. 공간 크기를 유지하려면 입력 둘레를 0으로 채운다(*패딩*). :numref:`sec_padding` 에서 패딩을 스트라이드 및 팽창과 함께 살펴본다.

<!--en:869d223f kind=heading-->
### Implementation
<!--tr:869d223f-->
### 구현

<!--en:6a869f7c kind=paragraph-->
We implement :eqref:`eq_corr2d` directly in `corr2d` using a pair of loops over
every output coordinate.
<!--tr:6a869f7c-->
:eqref:`eq_corr2d` 를 `corr2d` 에서 직접 구현한다. 두 개의 중첩 반복문으로 모든 출력 좌표를 순회한다.

<!--en:81e1f383 kind=paragraph-->
To verify `corr2d` against our hand calculation in :numref:`fig_correlation`,
we construct the $3 \times 3$ input tensor `X` and $2 \times 2$ kernel `K` and
evaluate `corr2d(X, K)`.
<!--tr:81e1f383-->
`corr2d` 의 결과를 :numref:`fig_correlation` 에서 손으로 계산한 값과 비교하기 위해 $3 \times 3$ 입력 텐서 `X` 와 $2 \times 2$ 커널 `K` 를 만들고 `corr2d(X, K)` 를 실행한다.

<!--en:f567e1b2 kind=paragraph-->
The output reproduces :numref:`fig_correlation` entry for entry (and as shown in
:numref:`sec_why-conv`, strict mathematical convolution with a flipped kernel
$\mathbf{K}'$ learns the exact same family of layers, so we follow standard
deep learning terminology and call cross-correlation "convolution").
<!--tr:f567e1b2-->
출력의 각 성분은 :numref:`fig_correlation` 과 정확히 일치한다. (:numref:`sec_why-conv` 에서 보았듯이 뒤집은 커널 $\mathbf{K}'$ 을 사용하는 엄밀한 수학적 합성곱으로도 정확히 같은 층의 집합을 학습할 수 있다. 따라서 딥러닝의 표준 용어를 따라 교차 상관을 ‘합성곱’이라 부른다.)

<!--en:bfe11c62 kind=heading-->
## Convolutional Layers
<!--tr:bfe11c62-->
## 합성곱층

<!--en:2cdc020d kind=heading-->
### The Layer
<!--tr:2cdc020d-->
### 합성곱층의 구성

<!--en:c6371ece kind=paragraph-->
A two-dimensional convolutional layer cross-correlates its input with a learned
kernel and adds a scalar bias. Just like our MLP weights in
:numref:`chap_perceptrons`, we initialize the kernel from
$\mathcal{N}(0, 0.01^2)$ and the bias to zero. Below, `Conv2D` registers both
parameters and invokes `corr2d` in its forward pass; a layer with a kernel of
height $h$ and width $w$ is called an $h \times w$ convolutional layer.
<!--tr:c6371ece-->
2차원 합성곱층은 입력과 학습한 커널의 교차 상관을 계산한 뒤 스칼라 편향을 더한다. :numref:`chap_perceptrons` 에서 다층 퍼셉트론(MLP)의 가중치를 초기화했을 때처럼, 커널의 초깃값은 $\mathcal{N}(0, 0.01^2)$ 에서 뽑고 편향은 0으로 초기화한다. 아래의 `Conv2D` 는 두 매개변수를 모두 등록하고 순방향 패스에서 `corr2d` 를 호출한다. 높이가 $h$, 너비가 $w$ 인 커널을 쓰는 층을 $h \times w$ 합성곱층이라 부른다.

<!--en:29cd9061 kind=paragraph-->
Unlike a dense layer, a convolutional layer is not locked to a fixed input
resolution because its parameters depend only on the kernel size. The exact same
$2 \times 2$ layer (5 parameters total) processes both our $3 \times 3$ tensor
`X` and a $6 \times 8$ image, returning outputs of shapes $2 \times 2$ and
$5 \times 7$:
<!--tr:29cd9061-->
완전 연결층과 달리 합성곱층의 매개변수는 커널 크기에만 의존하므로 입력 해상도가 고정되어 있지 않다. 같은 $2 \times 2$ 층(매개변수 총 5개)으로 앞의 $3 \times 3$ 텐서 `X` 와 $6 \times 8$ 이미지를 모두 처리할 수 있으며, 출력 모양은 각각 $2 \times 2$ 와 $5 \times 7$ 이다.

<!--en:c9e5aa07 kind=heading-->
### Edge Detection
<!--tr:c9e5aa07-->
### 경계 검출

<!--en:3d9a357b kind=paragraph-->
We now test the vertical edge detector from the introduction on a
$6 \times 8$ synthetic image whose middle four columns are black (`0`) and outer
columns are white (`1`).
<!--tr:3d9a357b-->
이제 도입부의 수직 경계 검출기를 $6 \times 8$ 크기의 합성 이미지에서 시험해 보자. 가운데 네 열은 검은색(`0`)이고 양쪽 바깥 열은 흰색(`1`)이다.

<!--en:68513ff0 kind=paragraph-->
We define the $1 \times 2$ kernel
<!--tr:68513ff0-->
$1 \times 2$ 커널을 다음과 같이 정의한다.

<!--en:eb45803b kind=paragraph-->
At every position $(i, j)$, this computes the horizontal *finite difference*
$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, a discrete approximation (up to sign)
of the horizontal derivative of brightness, which vanishes across flat regions
and spikes wherever intensity jumps.
<!--tr:eb45803b-->
각 위치 $(i, j)$ 에서 이 커널은 가로 방향의 *유한 차분* $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ 을 계산한다. 이 값은 밝기의 가로 방향 도함수를 부호 차이까지 허용하여 이산적으로 근사한 것으로, 밝기가 일정한 영역에서는 0이고 밝기가 급변하는 곳에서는 크기가 급격히 커진다.

<!--en:e810b3a9 kind=paragraph-->
Cross-correlating the $6 \times 8$ stripe image `X` with the $1 \times 2$
horizontal-difference kernel `K` via `corr2d(X, K)` produces the $6 \times 7$
edge map `Y`:
<!--tr:e810b3a9-->
$6 \times 8$ 줄무늬 이미지 `X` 와 $1 \times 2$ 가로 차분 커널 `K` 의 교차 상관을 `corr2d(X, K)` 로 계산하면 $6 \times 7$ 경계 지도 `Y` 를 얻는다.

<!--en:568358e4 kind=paragraph-->
Sure enough, `Y` fires `+1` along the white-to-black boundary, `-1` along the
black-to-white boundary, and `0` everywhere else. Applying the exact same kernel
to the transposed image (turning the vertical stripe into a horizontal one)
yields all zeros:
<!--tr:568358e4-->
예상대로 `Y` 는 흰색에서 검은색으로 바뀌는 경계에서 `+1`, 검은색에서 흰색으로 바뀌는 경계에서 `-1`, 나머지 모든 곳에서 `0` 이다. 이미지를 전치해 세로 줄무늬를 가로 줄무늬로 바꾼 뒤 같은 커널을 적용하면 모든 값이 0이 된다.

<!--en:71c591a4 kind=paragraph-->
Clearly `[1, -1]` responds strictly to vertical edges. Detecting horizontal
edges, diagonals, corners, or textures requires different kernels altogether.
<!--tr:71c591a4-->
이처럼 `[1, -1]` 은 수직 경계에만 반응한다. 수평 경계, 대각선, 모서리, 질감을 검출하려면 각각 다른 커널이 필요하다.

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
원하는 특징이 무엇인지 정확히 알면 `[1, -1]` 을 직접 설계하기는 쉽다. 하지만 심층 신경망의 중간 층에 있는 수천 개의 필터를 사람이 일일이 지정할 수는 없다. 경사 하강법으로 커널 `[1, -1]` 을 입력과 출력의 쌍 `(X, Y)` 만으로 복원할 수 있을까? 각 프레임워크에 내장된 2차원 합성곱층을 사용해 확인해 보자. 편향은 생략하고 가중치의 초깃값은 $\mathcal{N}(0, 0.01^2)$ 에서 뽑는다. 출력 성분 42개 전체의 제곱 오차를 합한 값을 최소화하도록 학습률 $0.03$ 으로 10단계에 걸쳐 경사 하강을 수행한다. 내장 합성곱층은 배치 축과 채널 축을 포함하는 4차원 텐서를 입력으로 받으므로 (:numref:`sec_channels`), `X` 와 `Y` 의 모양을 이에 맞게 바꾼다.

<!--en:c29c5d61 kind=paragraph-->
Initially the loss sits near 12 (the near-zero random kernel misses all 12
$\pm 1$ edge pixels by roughly 1), and within just nine steps it drops below
0.01. Inspecting the learned kernel confirms what it found:
<!--tr:c29c5d61-->
처음에 손실은 12에 가깝다. 무작위로 초기화한 커널 값이 0에 가까워 $\pm 1$ 값을 갖는 경계 픽셀 12개 모두에서 약 1씩 빗나가기 때문이다. 손실은 단 9단계 만에 0.01 미만으로 떨어진다. 학습한 커널을 살펴보면 어떤 값을 찾았는지 확인할 수 있다.

<!--en:67d25b0b kind=paragraph-->
Both entries land within roughly 0.02 of `[1, -1]`. Nothing in this loop knew
anything about edges: given a loss signal, gradient descent discovered the
filter on its own. Here we supervised the single layer directly with `Y`; in a
deep CNN, backpropagation learns every layer's kernels end to end from the final
task loss alone.
<!--tr:67d25b0b-->
두 성분 모두 `[1, -1]` 의 대응 성분과 약 0.02 이내의 차이를 보인다. 이 반복문에는 경계에 관한 정보가 들어 있지 않았다. 경사 하강법은 손실 신호를 받아 스스로 필터를 찾아냈다. 여기서는 단일 층에 `Y` 를 정답으로 직접 제공했다. 심층 합성곱 신경망(CNN)에서는 최종 과제의 손실만으로 역전파를 통해 모든 층의 커널을 종단 간 학습한다.

<!--en:91b0010d kind=slide_text-->
Convolutions for Images<br>**2D cross-correlation · edge detection · learned kernels · im2col · receptive fields**
<!--tr:91b0010d-->
이미지에 적용하는 합성곱<br>**2차원 교차 상관 · 경계 검출 · 커널 학습 결과 · im2col · 수용 영역**

<!--en:1d94295a kind=slide_title-->
::: {.slide title="Detecting Local Edges Across an Image"}
<!--tr:1d94295a-->
::: {.slide title="이미지 전역에서 국소 경계 검출하기"}

<!--en:687ed302 kind=kicker-->
[Motivation & Core Mental Model]{.kicker}
<!--tr:687ed302-->
[경계 검출의 동기와 기본 원리]{.kicker}

<!--en:05fceed0 kind=slide_text-->
- **Synthetic $6 \times 8$ test image:** White outer columns (`1`) surrounding a 4-column black vertical band (`0`).
<!--tr:05fceed0-->
- **$6 \times 8$ 합성 테스트 이미지:** 흰색 바깥 열(`1`) 사이에 검은색 세로 띠 4열(`0`).

<!--en:71e700b3 kind=slide_text-->
- **Horizontal finite difference:** Sweeping $\mathbf{K} = [1, -1]$ computes $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, firing `+1` at white-to-black and `-1` at black-to-white edges.
<!--tr:71e700b3-->
- **가로 방향 유한 차분:** $\mathbf{K} = [1, -1]$ 을 옮겨 가며 $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ 을 계산한다. 흰색에서 검은색으로 바뀌는 경계는 `+1`, 검은색에서 흰색으로 바뀌는 경계는 `-1` 이다.

<!--en:957dc033 kind=slide_text-->
- **Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent.
<!--tr:957dc033-->
- **학습과 수동 설계:** 심층 CNN은 경사 하강법으로 이런 커널 수천 개를 자동으로 찾는다.

<!--en:cb1045e1 kind=slide_title-->
::: {.slide title="The 2D Cross-Correlation Operation"}
<!--tr:cb1045e1-->
::: {.slide title="2차원 교차 상관 연산"}

<!--en:17bcece5 kind=kicker-->
[Sliding-Window Inner Products]{.kicker}
<!--tr:17bcece5-->
[윈도를 옮기며 계산하는 내적]{.kicker}

<!--en:8d45aae8 kind=slide_text-->
Given an $n_\textrm{h} \times n_\textrm{w}$ input matrix $\mathbf{X}$ and a $k_\textrm{h} \times k_\textrm{w}$ kernel $\mathbf{K}$, the 2D cross-correlation $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ places the window's top-left corner at $(i, j)$ and sums elementwise products:
<!--tr:8d45aae8-->
$n_\textrm{h} \times n_\textrm{w}$ 입력 행렬 $\mathbf{X}$ 와 $k_\textrm{h} \times k_\textrm{w}$ 커널 $\mathbf{K}$ 의 2차원 교차 상관 $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ 는 윈도의 왼쪽 위 꼭짓점을 $(i, j)$ 에 놓고 대응 성분의 곱을 합산한다.

<!--en:e484380c kind=slide_text-->
- **Top-left anchor:** Zero-based indices $a, b \ge 0$ match array indexing in code and handle even-sized kernels (like $2 \times 2$ or $1 \times 2$) cleanly.
<!--tr:e484380c-->
- **왼쪽 위를 기준으로:** $a, b \ge 0$ 인 0 기반 인덱스는 코드의 배열 인덱스와 같으며, $2 \times 2$ 나 $1 \times 2$ 같은 짝수 크기의 커널도 쉽게 다룬다.

<!--en:137c0980 kind=slide_text-->
- **Shaded top-left output $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.
<!--tr:137c0980-->
- **음영으로 표시한 왼쪽 위 출력 $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.

<!--en:575c4aea kind=slide_title-->
::: {.slide title="Valid Window Positions and Output Spatial Shape"}
<!--tr:575c4aea-->
::: {.slide title="윈도의 유효 위치와 출력의 공간 모양"}

<!--en:7eda4531 kind=kicker-->
[Why Unpadded Convolutions Shrink the Grid]{.kicker}
<!--tr:7eda4531-->
[패딩 없는 합성곱에서 격자가 작아지는 이유]{.kicker}

<!--en:891243ae kind=slide_text-->
- **Valid window placements:** A $k_\textrm{h} \times k_\textrm{w}$ window fits inside an $n_\textrm{h} \times n_\textrm{w}$ grid at $n - k + 1$ positions per axis.
<!--tr:891243ae-->
- **윈도의 유효 위치:** $k_\textrm{h} \times k_\textrm{w}$ 윈도가 $n_\textrm{h} \times n_\textrm{w}$ 격자 안에 온전히 들어가는 위치는 축마다 $n - k + 1$ 곳이다.

<!--en:55cc8c32 kind=slide_text-->
- **Shape examples:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.
<!--tr:55cc8c32-->
- **출력 모양의 예:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$), $28 \times 28 \star 5 \times 5 \to 24 \times 24$.

<!--en:1e3463da kind=slide_title-->
::: {.slide title="Implementing 2D Cross-Correlation in Code"}
<!--tr:1e3463da-->
::: {.slide title="2차원 교차 상관의 코드 구현"}

<!--en:07566992 kind=kicker-->
[Direct Nested-Loop Implementation]{.kicker}
<!--tr:07566992-->
[중첩 반복문으로 직접 구현]{.kicker}

<!--en:202aa5d3 kind=slide_text-->
The function `corr2d` allocates the $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ output tensor and computes each entry as the sum of `X[i:i+h, j:j+w] * K`:
<!--tr:202aa5d3-->
함수 `corr2d` 는 $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ 출력 텐서를 할당하고 `X[i:i+h, j:j+w] * K` 의 합으로 각 성분을 구한다.

<!--en:bef130ba kind=slide_text-->
Each iteration slices a $k_\textrm{h} \times k_\textrm{w}$ sub-window of `X`, multiplies elementwise with `K`, and reduces via `sum`.
<!--tr:bef130ba-->
반복할 때마다 $k_\textrm{h} \times k_\textrm{w}$ 크기의 윈도를 `X` 에서 잘라 내고 `K` 와 성분별로 곱한 뒤 `sum` 으로 합산한다.

<!--en:4ee9f6b1 kind=slide_title-->
::: {.slide title="Verifying corr2d and the Flipped-Kernel Equivalence"}
<!--tr:4ee9f6b1-->
::: {.slide title="corr2d 검증과 커널을 뒤집었을 때의 동등성"}

<!--en:2b0be423 kind=kicker-->
[Cross-Correlation vs. Mathematical Convolution]{.kicker}
<!--tr:2b0be423-->
[교차 상관과 수학적 합성곱 비교]{.kicker}

<!--en:74b9b745 kind=slide_text-->
- **180° flipped kernel $\mathbf{K}'$:** Cross-correlating with $\mathbf{K}$ equals convolving with $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.
<!--tr:74b9b745-->
- **180° 뒤집은 커널 $\mathbf{K}'$:** $\mathbf{K}$ 와의 교차 상관은 $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$ 로 정의한 커널과의 합성곱과 같다.

<!--en:5c8af7cd kind=slide_text-->
- **Identical learned class:** Frameworks compute cross-correlation and call it convolution.
<!--tr:5c8af7cd-->
- **학습할 수 있는 층의 집합은 동일:** 프레임워크는 교차 상관을 계산하고 이를 합성곱이라 부른다.

<!--en:a69e29b5 kind=slide_title-->
::: {.slide title="Building a 2D Convolutional Layer Module"}
<!--tr:a69e29b5-->
::: {.slide title="2차원 합성곱층 모듈 구현"}

<!--en:55763603 kind=kicker-->
[Learnable Kernel Weights Plus Scalar Bias]{.kicker}
<!--tr:55763603-->
[학습 가능한 커널 가중치와 스칼라 편향]{.kicker}

<!--en:5b8bad24 kind=slide_text-->
A convolutional layer wraps `corr2d` with a learnable $k_\textrm{h} \times k_\textrm{w}$ weight parameter (initialized from $\mathcal{N}(0, 0.01^2)$) and a scalar bias (initialized to $0$):
<!--tr:5b8bad24-->
합성곱층은 `corr2d` 에 학습 가능한 $k_\textrm{h} \times k_\textrm{w}$ 가중치 매개변수(초깃값은 $\mathcal{N}(0, 0.01^2)$ 에서 추출)와 스칼라 편향(초깃값은 $0$)을 결합한다.

<!--en:c16d97d6 kind=slide_title-->
::: {.slide title="Resolution Independence of Convolutional Layers"}
<!--tr:c16d97d6-->
::: {.slide title="입력 해상도에 구애받지 않는 합성곱층"}

<!--en:a7d1e48f kind=kicker-->
[Fixed Parameters Across Arbitrary Input Sizes]{.kicker}
<!--tr:a7d1e48f-->
[입력 크기와 무관한 매개변수]{.kicker}

<!--en:53a2bce7 kind=slide_text-->
- **Decoupled from $H \times W$:** The exact same 5 parameters ($2 \times 2$ kernel + 1 bias) process $3 \times 3$, $6 \times 8$, or megapixel inputs without reshaping weights.
<!--tr:53a2bce7-->
- **$H \times W$ 와 무관:** 같은 매개변수 5개($2 \times 2$ 커널과 편향 1개)로 가중치의 모양을 바꾸지 않고 $3 \times 3$, $6 \times 8$, 수백만 픽셀 규모의 입력을 처리한다.

<!--en:017fec24 kind=slide_title-->
::: {.slide title="Constructing a Synthetic Vertical-Edge Image"}
<!--tr:017fec24-->
::: {.slide title="수직 경계가 있는 합성 이미지 만들기"}

<!--en:896e641c kind=kicker-->
[A 6×8 Binary Test Pattern]{.kicker}
<!--tr:896e641c-->
[6×8 이진 테스트 패턴]{.kicker}

<!--en:0a4b456b kind=slide_text-->
We construct the $6 \times 8$ test image `X` with ones on the left and right borders and zeros across the middle four columns (`X[:, 2:6] = 0`):
<!--tr:0a4b456b-->
$6 \times 8$ 테스트 이미지 `X` 의 좌우 바깥쪽 열은 1로, 가운데 네 열은 0으로 채운다(`X[:, 2:6] = 0`).

<!--en:dfc7e3e3 kind=slide_text-->
- **Piecewise-constant columns:** Intensity is constant (`1` or `0`) within columns $0\text{--}1$, $2\text{--}5$, and $6\text{--}7$, jumping only between columns $1 \to 2$ and $5 \to 6$.
<!--tr:dfc7e3e3-->
- **구간별로 일정한 열의 밝기:** 밝기는 각 열 구간 $0\text{--}1$, $2\text{--}5$, $6\text{--}7$ 안에서 일정하며(`1` 또는 `0`), $1 \to 2$ 와 $5 \to 6$ 에서만 바뀐다.

<!--en:3ce31cbe kind=slide_title-->
::: {.slide title="Applying the Horizontal Finite-Difference Kernel"}
<!--tr:3ce31cbe-->
::: {.slide title="가로 방향 유한 차분 커널 적용"}

<!--en:b37a1029 kind=kicker-->
[Detecting Vertical Edges with K = [1, -1]]{.kicker}
<!--tr:b37a1029-->
[K = [1, -1]로 수직 경계 검출]{.kicker}

<!--en:9be85019 kind=slide_text-->
Cross-correlating `X` with the $1 \times 2$ kernel $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ computes $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ at every horizontal pair:
<!--tr:9be85019-->
`X` 와 $1 \times 2$ 커널 $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ 의 교차 상관은 가로로 이웃한 모든 픽셀 쌍에 대해 $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ 을 계산한다.

<!--en:f649cfae kind=slide_text-->
- **Signed edge map:** Column 1 fires `+1` (white-to-black transition), column 5 fires `-1` (black-to-white transition), and all constant interior columns evaluate to `0`.
<!--tr:f649cfae-->
- **부호가 있는 경계 지도:** 1번 열은 `+1`(흰색에서 검은색), 5번 열은 `-1`(검은색에서 흰색), 밝기가 일정한 구간 내부의 열은 모두 `0` 이다.

<!--en:e6d155be kind=slide_title-->
::: {.slide title="Orientation Selectivity of Edge Kernels"}
<!--tr:e6d155be-->
::: {.slide title="경계 검출 커널의 방향 선택성"}

<!--en:b525cde4 kind=kicker-->
[Why One Kernel Cannot Detect All Orientations]{.kicker}
<!--tr:b525cde4-->
[커널 하나로 모든 방향을 검출할 수 없는 이유]{.kicker}

<!--en:38159de6 kind=slide_text-->
- **Blind to horizontal edges:** Transposing `X` (`X.T`) makes horizontal neighbors identical, so $[1, -1]$ outputs all zeros.
<!--tr:38159de6-->
- **수평 경계에는 반응하지 않음:** `X` 를 전치하면(`X.T`) 가로로 이웃한 픽셀 값이 같아져 $[1, -1]$ 의 출력은 모두 0이다.

<!--en:f0f30f76 kind=slide_text-->
- **Filter banks (§{sec_channels}):** Vision layers learn parallel channels for horizontal, vertical, and diagonal orientations.
<!--tr:f0f30f76-->
- **필터 뱅크 (§{sec_channels}):** 영상 처리 층은 수평, 수직, 대각선 방향에 대응하는 채널을 병렬로 학습한다.

<!--en:b755e31b kind=slide_title-->
::: {.slide title="Learning an Edge Detector via Gradient Descent"}
<!--tr:b755e31b-->
::: {.slide title="경사 하강법으로 경계 검출기 학습"}

<!--en:86bbeea7 kind=kicker-->
[Supervising a Built-In Convolutional Layer on (X, Y)]{.kicker}
<!--tr:86bbeea7-->
[(X, Y)로 내장 합성곱층 지도 학습]{.kicker}

<!--en:a569780c kind=slide_text-->
- **4D framework tensors:** Reshape `X` and `Y` to 4D tensors (one example, one channel) and minimize $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ at $\eta = 0.03$.
<!--tr:a569780c-->
- **프레임워크의 4차원 텐서:** `X` 와 `Y` 를 4차원 텐서(예제 1개, 채널 1개)로 바꾸고 $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ 를 $\eta = 0.03$ 으로 최소화한다.

<!--en:2489ce6e kind=slide_title-->
::: {.slide title="Inspecting the Learned Kernel Weights"}
<!--tr:2489ce6e-->
::: {.slide title="학습한 커널 가중치 확인"}

<!--en:7e4ef44b kind=kicker-->
[Why Initial Loss Is 12 and Converges to [1, -1]]{.kicker}
<!--tr:7e4ef44b-->
[초기 손실이 12인 이유와 [1, -1]로 수렴하는 가중치]{.kicker}

<!--en:9cd1c645 kind=slide_text-->
- **Why loss starts at $12.0$:** Near-zero weights predict $\hat{Y} \approx 0$, missing the twelve $\pm 1$ edge pixels in `Y` ($12 \times 1^2 = 12$).
<!--tr:9cd1c645-->
- **초기 손실이 $12.0$ 인 이유:** 0에 가까운 가중치는 $\hat{Y} \approx 0$ 으로 예측해 $\pm 1$ 값을 갖는 `Y` 의 경계 픽셀 12개를 모두 놓친다($12 \times 1^2 = 12$).

<!--en:456670c5 kind=slide_text-->
- **Convergence in 9 steps:** SGD recovers $[0.99, -0.99]$ automatically from data.
<!--tr:456670c5-->
- **9단계 만에 수렴:** 경사 하강법이 데이터에서 $[0.99, -0.99]$ 를 자동으로 복원한다.

=== DECISIONS ===
- 본문은 ‘-다’체로 쓰고 함께 진행하는 대목에서는 ‘-해 보자’를 쓴다. 연습 문제의 지시에는 ‘-하라’를 쓰고 질문은 의문문으로 유지한다. 제목은 명사구, 슬라이드는 짧은 구나 문장으로 쓴다.
- 한국어 용어를 기본으로 쓰고 영문 용어를 반복 병기하지 않는다. 원문에 있는 모델 약어는 본문의 첫 용례에서 한글 명칭과 병기하고, 이후에는 약어를 쓸 수 있다. 모형·라이브러리·데이터셋 이름과 im2col 표기는 유지한다. 인라인 수식과 참조는 한글과 띄어 쓴다. 소수점 표기는 원문을 따르고 숫자와 단위성 명사는 ‘10단계’, ‘42개’처럼 붙여 쓴다.
- “locality” → “국소성”, “offset” → “오프셋”, “window” → “윈도”, “patch” → “패치”, “entry / element” → “성분” (픽셀에는 “픽셀 값”), “parameter” → “매개변수”.
- “edge / edge detection / edge detector / edge map” → “경계 / 경계 검출 / 경계 검출기 / 경계 지도”. 이미지 윤곽의 경계와 배열 바깥쪽을 뜻하는 “border”는 문맥상 구별한다.
- “dense layer / fully connected layer” → “완전 연결층”, “unit” → “유닛”, “reshape” → “모양을 바꾸다”, “forward propagation” → “순방향 패스”, “end to end” → “종단 간”.
- “synthetic image” → “합성 이미지”, “resolution” → “해상도”, “filter bank” → “필터 뱅크”, “orientation selectivity” → “방향 선택성”.

=== NOTES ===
- 456670c5: 원문은 SGD라고 하지만 실험 코드는 전체 출력의 제곱 오차 합으로 기울기를 계산하므로 ‘경사 하강법’으로 옮겼다.
- 7e4ef44b: 원문에서 [1, -1]로 수렴하는 것은 손실이 아니라 커널 가중치이므로 주체를 명시했다.
- 74b9b745: 음수 인덱스 식은 원점 기준의 수학적 커널 반전을 나타낸다. 0부터 시작하는 배열의 인덱스 식으로 읽지 않도록 문맥에 유의해야 한다.
- dfc7e3e3: 자연스러운 한국어 어순에 맞춰 인라인 코드의 위치를 조정했다.
- 61cacfaf: 용어 경고의 “gradient”는 “gradient steps”의 일부이며 경사 하강법의 갱신 단계를 뜻한다. 이 문맥에서는 ‘10단계에 걸쳐 경사 하강을 수행한다’로 옮겼다.
