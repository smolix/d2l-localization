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

:numref:`sec_why-conv` 에서는 평행이동 등변성과 국소성이라는 두 기하학적 원리에서 합성곱층을 수식으로 유도했다. 이제 그 수식이 이미지에서 실제로 무엇을 계산하는지, 경사 하강법이 데이터에서 유용한 커널을 자동으로 찾아낼 수 있는지 살펴보자.

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

밝은 영역과 어두운 영역이 만나는 수직 에지를 찾는 상황을 생각해 보자. 각 픽셀에서 바로 오른쪽 픽셀을 빼면 ($[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$) 균일한 영역에서는 0, 흰색에서 검은색으로 바뀌는 곳에서는 $+1$, 검은색에서 다시 흰색으로 바뀌는 곳에서는 $-1$ 이 나온다. 두 가중치로 이루어진 검출기 $(1, -1)$ 를 이미지의 유효한 모든 위치에 훑어 적용하면 *에지 맵*이 만들어진다. 이 절에서는 2차원 교차 상관을 구현하고, 직접 만든 에지 검출기를 시험하며, 경사 하강법으로 데이터에서 $(1, -1)$ 을 직접 *학습*한다. 또한 합성곱을 하나의 행렬 곱셈으로 바꾸는 방법 (*im2col*)을 보이고, 층을 쌓으면 유닛의 수용 영역이 어떻게 넓어지는지 유도한다.

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

잠시 채널은 제쳐 두고, 입력 $\mathbf{X}$ 는 $n_\textrm{h} \times n_\textrm{w}$ 행렬이고 커널 $\mathbf{K}$ 는 $k_\textrm{h} \times k_\textrm{w}$ 행렬이라고 하자. 커널은 $k_\textrm{h} \times k_\textrm{w}$ 모양의 이동하는 *윈도*를 정의한다. 윈도의 왼쪽 위 모서리를 $(i, j)$ 에 놓고, 그 안의 입력 패치와 $\mathbf{K}$ 를 원소별로 곱한 뒤 그 곱을 합하여 $(i, j)$ 위치의 출력 원소를 구한다:

<!--en:3688b38b kind=paragraph-->
This is the *two-dimensional cross-correlation* of $\mathbf{X}$ with
$\mathbf{K}$, written $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ with
zero-based indices matching code. Whereas :numref:`sec_why-conv` centered
offsets symmetrically from $-\Delta$ to $\Delta$, anchoring at the upper-left
corner differs only by a fixed shift of $\Delta$ while naturally handling
even-sized kernels (such as a $2 \times 2$ kernel or our $1 \times 2$ edge
detector) that have no center pixel.
<!--tr:3688b38b-->

이것이 $\mathbf{X}$ 와 $\mathbf{K}$ 의 *2차원 교차 상관*이며, $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ 로 쓴다. 인덱스는 코드와 같이 0부터 시작한다. :numref:`sec_why-conv` 에서는 오프셋을 $-\Delta$ 부터 $\Delta$ 까지 중심을 기준으로 대칭으로 잡았다. 왼쪽 위 모서리를 기준으로 삼으면 고정된 이동량 $\Delta$ 만큼만 차이가 나면서도 중심 픽셀이 없는 짝수 크기 커널 ($2 \times 2$ 커널이나 앞의 $1 \times 2$ 에지 검출기 등)을 자연스럽게 처리할 수 있다.

<!--en:fa1b7da9 kind=caption-->
![Two-dimensional cross-correlation. The shaded portions are the first output element and the input and kernel elements used to compute it: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
<!--tr:fa1b7da9-->

![2차원 교차 상관. 음영 부분은 첫 번째 출력 원소와 이를 계산하는 데 사용한 입력 및 커널 원소이다: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)

<!--en:2f8e5deb kind=paragraph-->
In :numref:`fig_correlation`, we slide a $2 \times 2$ kernel across a
$3 \times 3$ input one pixel at a time, left to right and top to bottom. The
window fits cleanly at four positions, yielding the $2 \times 2$ output:
<!--tr:2f8e5deb-->

:numref:`fig_correlation` 에서는 $2 \times 2$ 커널을 $3 \times 3$ 입력 위에서 왼쪽에서 오른쪽으로, 위에서 아래로 한 번에 한 픽셀씩 이동한다. 윈도는 네 위치에 온전히 들어가므로 $2 \times 2$ 출력을 얻는다:

<!--en:4741e012 kind=paragraph-->
Notice that the output shrinks slightly along each axis because the kernel must
stay wholly inside the input. Fitting a width-$k$ window into length $n$ leaves
$n - k + 1$ valid positions, so the output shape is
<!--tr:4741e012-->

커널이 입력 안에 완전히 들어가야 하므로 출력은 각 축을 따라 조금씩 줄어든다. 길이 $n$ 안에 너비 $k$ 인 윈도를 놓을 수 있는 유효한 위치는 $n - k + 1$ 개이므로 출력 모양은 다음과 같다.

<!--en:12d81c14 kind=paragraph-->
Thus a $3 \times 3$ input with a $2 \times 2$ kernel yields $2 \times 2$, a
$6 \times 8$ image with a $1 \times 2$ kernel yields $6 \times 7$, and a
$28 \times 28$ Fashion-MNIST image with a $5 \times 5$ kernel yields
$24 \times 24$. Whenever we want to preserve spatial dimensions, we surround the
input with a border of zeros (*padding*), which we examine alongside strides and
dilation in :numref:`sec_padding`.
<!--tr:12d81c14-->

따라서 $3 \times 3$ 입력에 $2 \times 2$ 커널을 적용하면 $2 \times 2$, $6 \times 8$ 이미지에 $1 \times 2$ 커널을 적용하면 $6 \times 7$, $28 \times 28$ Fashion-MNIST 이미지에 $5 \times 5$ 커널을 적용하면 $24 \times 24$ 가 된다. 공간 차원을 유지하려면 입력을 0으로 이루어진 테두리로 둘러싼다 (*패딩*). 이는 스트라이드 및 팽창과 함께 :numref:`sec_padding` 에서 살펴본다.

<!--en:869d223f kind=heading-->
### Implementation
<!--tr:869d223f-->

### 구현

<!--en:6a869f7c kind=paragraph-->
We implement :eqref:`eq_corr2d` directly in `corr2d` using a pair of loops over
every output coordinate.
<!--tr:6a869f7c-->

모든 출력 좌표를 순회하는 두 개의 반복문을 사용해 :eqref:`eq_corr2d` 를 `corr2d` 에 직접 구현한다.

<!--en:81e1f383 kind=paragraph-->
To verify `corr2d` against our hand calculation in :numref:`fig_correlation`,
we construct the $3 \times 3$ input tensor `X` and $2 \times 2$ kernel `K` and
evaluate `corr2d(X, K)`.
<!--tr:81e1f383-->

:numref:`fig_correlation` 의 수작업 계산과 대조해 `corr2d` 를 검증하기 위해 $3 \times 3$ 입력 텐서 `X` 와 $2 \times 2$ 커널 `K` 를 만들고 `corr2d(X, K)` 를 계산한다.

<!--en:f567e1b2 kind=paragraph-->
The output reproduces :numref:`fig_correlation` entry for entry (and as shown in
:numref:`sec_why-conv`, strict mathematical convolution with a flipped kernel
$\mathbf{K}'$ learns the exact same family of layers, so we follow standard
deep learning terminology and call cross-correlation "convolution").
<!--tr:f567e1b2-->

출력은 :numref:`fig_correlation` 의 원소를 하나하나 재현한다 (또한 :numref:`sec_why-conv` 에서 보았듯 뒤집힌 커널 $\mathbf{K}'$ 을 사용하는 엄밀한 수학적 합성곱도 정확히 같은 층의 집합을 학습하므로, 표준 딥러닝 용어를 따라 교차 상관을 "합성곱"이라고 부른다).

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

2차원 합성곱층은 입력과 학습된 커널의 교차 상관을 계산하고 스칼라 편향을 더한다. :numref:`chap_perceptrons` 의 다층 퍼셉트론 가중치와 마찬가지로 커널은 $\mathcal{N}(0, 0.01^2)$ 에서 초기화하고 편향은 0으로 초기화한다. 아래에서 `Conv2D` 는 두 파라미터를 모두 등록하고 순방향 패스에서 `corr2d` 를 호출한다. 높이 $h$, 너비 $w$ 인 커널을 가진 층을 $h \times w$ 합성곱층이라고 한다.

<!--en:29cd9061 kind=paragraph-->
Unlike a dense layer, a convolutional layer is not locked to a fixed input
resolution because its parameters depend only on the kernel size. The exact same
$2 \times 2$ layer (5 parameters total) processes both our $3 \times 3$ tensor
`X` and a $6 \times 8$ image, returning outputs of shapes $2 \times 2$ and
$5 \times 7$:
<!--tr:29cd9061-->

밀집층과 달리 합성곱층의 파라미터는 커널 크기에만 의존하므로 입력 해상도가 고정되지 않는다. 정확히 같은 $2 \times 2$ 층 (총 파라미터 5개)이 $3 \times 3$ 텐서 `X` 와 $6 \times 8$ 이미지를 모두 처리하여 각각 $2 \times 2$ 와 $5 \times 7$ 모양의 출력을 반환한다:

<!--en:c9e5aa07 kind=heading-->
### Edge Detection
<!--tr:c9e5aa07-->

### 에지 검출

<!--en:3d9a357b kind=paragraph-->
We now test the vertical edge detector from the introduction on a
$6 \times 8$ synthetic image whose middle four columns are black (`0`) and outer
columns are white (`1`).
<!--tr:3d9a357b-->

이제 도입부의 수직 에지 검출기를 가운데 네 열은 검은색 (`0`), 바깥쪽 열은 흰색 (`1`)인 $6 \times 8$ 합성 이미지에서 시험해 보자.

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

이는 각 위치 $(i, j)$ 에서 수평 *유한 차분* $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ 을 계산한다. 이는 밝기의 수평 도함수를 이산적으로 근사한 값 (부호 차이까지 고려)으로, 평탄한 영역에서는 0이고 밝기가 급변하는 곳에서는 치솟는다.

<!--en:e810b3a9 kind=paragraph-->
Cross-correlating the $6 \times 8$ stripe image `X` with the $1 \times 2$
horizontal-difference kernel `K` via `corr2d(X, K)` produces the $6 \times 7$
edge map `Y`:
<!--tr:e810b3a9-->

`corr2d(X, K)` 를 통해 $6 \times 8$ 줄무늬 이미지 `X` 와 $1 \times 2$ 수평 차분 커널 `K` 의 교차 상관을 계산하면 $6 \times 7$ 에지 맵 `Y` 가 나온다:

<!--en:568358e4 kind=paragraph-->
Sure enough, `Y` fires `+1` along the white-to-black boundary, `-1` along the
black-to-white boundary, and `0` everywhere else. Applying the exact same kernel
to the transposed image (turning the vertical stripe into a horizontal one)
yields all zeros:
<!--tr:568358e4-->

예상대로 `Y` 는 흰색에서 검은색으로 바뀌는 경계에서 `+1`, 검은색에서 흰색으로 바뀌는 경계에서 `-1`, 나머지 모든 곳에서 `0` 을 출력한다. 정확히 같은 커널을 전치한 이미지 (수직 줄무늬가 수평 줄무늬로 바뀐 이미지)에 적용하면 모두 0이 나온다:

<!--en:71c591a4 kind=paragraph-->
Clearly `[1, -1]` responds strictly to vertical edges. Detecting horizontal
edges, diagonals, corners, or textures requires different kernels altogether.
<!--tr:71c591a4-->

분명히 `[1, -1]` 은 수직 에지에만 반응한다. 수평 에지, 대각선, 모서리 또는 질감을 검출하려면 완전히 다른 커널이 필요하다.

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

원하는 특징을 정확히 알 때는 `[1, -1]` 을 직접 만들기 쉽지만, 깊은 신경망의 수천 개 중간 필터를 사람이 일일이 지정할 수는 없다. 경사 하강법은 입출력 쌍 `(X, Y)` 만으로 커널 `[1, -1]` 을 복원할 수 있을까? 각 프레임워크에 내장된 2차원 합성곱층 (편향은 생략하고 가중치는 $\mathcal{N}(0, 0.01^2)$ 에서 초기화)을 사용하여 이를 시험한다. 학습률 $0.03$ 으로 기울기 갱신을 10회 수행하면서 42개 출력 원소 전체에 걸쳐 합한 제곱 오차를 최소화한다. 내장 층은 배치와 채널 축을 포함한 4차원 텐서를 입력받으므로 (:numref:`sec_channels`) `X` 와 `Y` 의 모양을 그에 맞게 바꾼다.

<!--en:c29c5d61 kind=paragraph-->
Initially the loss sits near 12 (the near-zero random kernel misses all 12
$\pm 1$ edge pixels by roughly 1), and within just nine steps it drops below
0.01. Inspecting the learned kernel confirms what it found:
<!--tr:c29c5d61-->

처음에 손실은 12에 가깝고 (거의 0인 무작위 커널이 12개의 $\pm 1$ 에지 픽셀을 모두 약 1만큼 빗나간다), 불과 아홉 단계 안에 0.01 아래로 떨어진다. 학습된 커널을 살펴보면 무엇을 찾아냈는지 확인할 수 있다:

<!--en:67d25b0b kind=paragraph-->
Both entries land within roughly 0.02 of `[1, -1]`. Nothing in this loop knew
anything about edges: given a loss signal, gradient descent discovered the
filter on its own. Here we supervised the single layer directly with `Y`; in a
deep CNN, backpropagation learns every layer's kernels end to end from the final
task loss alone.
<!--tr:67d25b0b-->

두 원소 모두 `[1, -1]` 에서 약 0.02 이내에 있다. 이 반복문에는 에지에 관한 지식이 전혀 없었다. 손실 신호가 주어지자 경사 하강법이 스스로 필터를 찾아낸 것이다. 여기서는 단일 층을 `Y` 로 직접 지도했지만, 깊은 CNN에서는 최종 과제의 손실만으로 역전파가 모든 층의 커널을 종단 간 학습한다.

<!--en:91b0010d kind=slide_text-->
Convolutions for Images<br>**2D cross-correlation · edge detection · learned kernels · im2col · receptive fields**
<!--tr:91b0010d-->

이미지를 위한 합성곱<br>**2차원 교차 상관 · 에지 검출 · 학습된 커널 · im2col · 수용 영역**

<!--en:1d94295a kind=slide_title-->
::: {.slide title="Detecting Local Edges Across an Image"}
<!--tr:1d94295a-->

::: {.slide title="이미지 전체에서 국소 에지 검출하기"}

<!--en:687ed302 kind=kicker-->
[Motivation & Core Mental Model]{.kicker}
<!--tr:687ed302-->

[동기와 핵심 개념]{.kicker}

<!--en:05fceed0 kind=slide_text-->
- **Synthetic $6 \times 8$ test image:** White outer columns (`1`) surrounding a 4-column black vertical band (`0`).
<!--tr:05fceed0-->

- **합성 $6 \times 8$ 테스트 이미지:** 흰색 바깥 열 (`1`)이 4개 열로 된 검은색 수직 띠 (`0`)를 둘러싼다.

<!--en:71e700b3 kind=slide_text-->
- **Horizontal finite difference:** Sweeping $\mathbf{K} = [1, -1]$ computes $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, firing `+1` at white-to-black and `-1` at black-to-white edges.
<!--tr:71e700b3-->

- **수평 유한 차분:** $\mathbf{K} = [1, -1]$ 을 훑어 적용하면 $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ 을 계산하여 흰색에서 검은색으로 가는 에지에서 `+1`, 반대 에지에서 `-1` 을 출력한다.

<!--en:957dc033 kind=slide_text-->
- **Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent.
<!--tr:957dc033-->

- **학습과 수작업:** 깊은 CNN은 경사 하강법으로 수천 개의 이런 커널을 자동으로 찾아낸다.

<!--en:cb1045e1 kind=slide_title-->
::: {.slide title="The 2D Cross-Correlation Operation"}
<!--tr:cb1045e1-->

::: {.slide title="2차원 교차 상관 연산"}

<!--en:17bcece5 kind=kicker-->
[Sliding-Window Inner Products]{.kicker}
<!--tr:17bcece5-->

[이동 윈도의 내적]{.kicker}

<!--en:8d45aae8 kind=slide_text-->
Given an $n_\textrm{h} \times n_\textrm{w}$ input matrix $\mathbf{X}$ and a $k_\textrm{h} \times k_\textrm{w}$ kernel $\mathbf{K}$, the 2D cross-correlation $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ places the window's top-left corner at $(i, j)$ and sums elementwise products:
<!--tr:8d45aae8-->

$n_\textrm{h} \times n_\textrm{w}$ 입력 행렬 $\mathbf{X}$ 와 $k_\textrm{h} \times k_\textrm{w}$ 커널 $\mathbf{K}$ 가 주어지면, 2차원 교차 상관 $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ 은 윈도의 왼쪽 위 모서리를 $(i, j)$ 에 놓고 원소별 곱을 합한다:

<!--en:e484380c kind=slide_text-->
- **Top-left anchor:** Zero-based indices $a, b \ge 0$ match array indexing in code and handle even-sized kernels (like $2 \times 2$ or $1 \times 2$) cleanly.
<!--tr:e484380c-->

- **왼쪽 위 기준점:** 0부터 시작하는 인덱스 $a, b \ge 0$ 는 코드의 배열 인덱싱과 일치하며 짝수 크기 커널 ($2 \times 2$ 또는 $1 \times 2$ 등)을 깔끔하게 처리한다.

<!--en:137c0980 kind=slide_text-->
- **Shaded top-left output $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.
<!--tr:137c0980-->

- **음영으로 표시된 왼쪽 위 출력 $(0,0)$:**
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

- **유효한 윈도 배치:** $k_\textrm{h} \times k_\textrm{w}$ 윈도는 $n_\textrm{h} \times n_\textrm{w}$ 격자 안에 각 축당 $n - k + 1$ 개의 위치로 들어간다.

<!--en:55cc8c32 kind=slide_text-->
- **Shape examples:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.
<!--tr:55cc8c32-->

- **모양 예시:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.

<!--en:1e3463da kind=slide_title-->
::: {.slide title="Implementing 2D Cross-Correlation in Code"}
<!--tr:1e3463da-->

::: {.slide title="코드로 2차원 교차 상관 구현하기"}

<!--en:07566992 kind=kicker-->
[Direct Nested-Loop Implementation]{.kicker}
<!--tr:07566992-->

[직접 중첩 반복문으로 구현]{.kicker}

<!--en:202aa5d3 kind=slide_text-->
The function `corr2d` allocates the $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ output tensor and computes each entry as the sum of `X[i:i+h, j:j+w] * K`:
<!--tr:202aa5d3-->

함수 `corr2d` 는 $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ 출력 텐서를 할당하고 각 원소를 `X[i:i+h, j:j+w] * K` 의 합으로 계산한다:

<!--en:bef130ba kind=slide_text-->
Each iteration slices a $k_\textrm{h} \times k_\textrm{w}$ sub-window of `X`, multiplies elementwise with `K`, and reduces via `sum`.
<!--tr:bef130ba-->

각 반복에서 `X` 의 $k_\textrm{h} \times k_\textrm{w}$ 하위 윈도를 슬라이스하고 `K` 와 원소별로 곱한 다음 `sum` 으로 합산한다.

<!--en:4ee9f6b1 kind=slide_title-->
::: {.slide title="Verifying corr2d and the Flipped-Kernel Equivalence"}
<!--tr:4ee9f6b1-->

::: {.slide title="corr2d와 뒤집힌 커널의 동등성 검증하기"}

<!--en:2b0be423 kind=kicker-->
[Cross-Correlation vs. Mathematical Convolution]{.kicker}
<!--tr:2b0be423-->

[교차 상관과 수학적 합성곱]{.kicker}

<!--en:74b9b745 kind=slide_text-->
- **180° flipped kernel $\mathbf{K}'$:** Cross-correlating with $\mathbf{K}$ equals convolving with $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.
<!--tr:74b9b745-->

- **180° 뒤집힌 커널 $\mathbf{K}'$:** $\mathbf{K}$ 와의 교차 상관은 $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$ 와의 합성곱과 같다.

<!--en:5c8af7cd kind=slide_text-->
- **Identical learned class:** Frameworks compute cross-correlation and call it convolution.
<!--tr:5c8af7cd-->

- **동일한 학습 가능 클래스:** 프레임워크는 교차 상관을 계산하고 이를 합성곱이라고 부른다.

<!--en:a69e29b5 kind=slide_title-->
::: {.slide title="Building a 2D Convolutional Layer Module"}
<!--tr:a69e29b5-->

::: {.slide title="2차원 합성곱층 모듈 만들기"}

<!--en:55763603 kind=kicker-->
[Learnable Kernel Weights Plus Scalar Bias]{.kicker}
<!--tr:55763603-->

[학습 가능한 커널 가중치와 스칼라 편향]{.kicker}

<!--en:5b8bad24 kind=slide_text-->
A convolutional layer wraps `corr2d` with a learnable $k_\textrm{h} \times k_\textrm{w}$ weight parameter (initialized from $\mathcal{N}(0, 0.01^2)$) and a scalar bias (initialized to $0$):
<!--tr:5b8bad24-->

합성곱층은 `corr2d` 에 학습 가능한 $k_\textrm{h} \times k_\textrm{w}$ 가중치 파라미터 ($\mathcal{N}(0, 0.01^2)$ 에서 초기화)와 스칼라 편향 ($0$ 으로 초기화)을 결합한다:

<!--en:c16d97d6 kind=slide_title-->
::: {.slide title="Resolution Independence of Convolutional Layers"}
<!--tr:c16d97d6-->

::: {.slide title="합성곱층의 해상도 독립성"}

<!--en:a7d1e48f kind=kicker-->
[Fixed Parameters Across Arbitrary Input Sizes]{.kicker}
<!--tr:a7d1e48f-->

[임의의 입력 크기에서 고정된 파라미터]{.kicker}

<!--en:53a2bce7 kind=slide_text-->
- **Decoupled from $H \times W$:** The exact same 5 parameters ($2 \times 2$ kernel + 1 bias) process $3 \times 3$, $6 \times 8$, or megapixel inputs without reshaping weights.
<!--tr:53a2bce7-->

- **$H \times W$ 와 독립:** 정확히 같은 5개 파라미터 ($2 \times 2$ 커널 + 편향 1개)로 가중치의 모양을 바꾸지 않고 $3 \times 3$, $6 \times 8$ 또는 메가픽셀 입력을 처리한다.

<!--en:017fec24 kind=slide_title-->
::: {.slide title="Constructing a Synthetic Vertical-Edge Image"}
<!--tr:017fec24-->

::: {.slide title="합성 수직 에지 이미지 만들기"}

<!--en:896e641c kind=kicker-->
[A 6×8 Binary Test Pattern]{.kicker}
<!--tr:896e641c-->

[6×8 이진 테스트 패턴]{.kicker}

<!--en:0a4b456b kind=slide_text-->
We construct the $6 \times 8$ test image `X` with ones on the left and right borders and zeros across the middle four columns (`X[:, 2:6] = 0`):
<!--tr:0a4b456b-->

왼쪽과 오른쪽 테두리는 1이고 가운데 네 열은 0인 $6 \times 8$ 테스트 이미지 `X` 를 만든다 (`X[:, 2:6] = 0`):

<!--en:dfc7e3e3 kind=slide_text-->
- **Piecewise-constant columns:** Intensity is constant (`1` or `0`) within columns $0\text{--}1$, $2\text{--}5$, and $6\text{--}7$, jumping only between columns $1 \to 2$ and $5 \to 6$.
<!--tr:dfc7e3e3-->

- **구간별로 일정한 열:** 열 $0\text{--}1$, $2\text{--}5$, $6\text{--}7$ 각각에서 밝기는 일정하며 (`1` 또는 `0`), 열 $1 \to 2$ 와 $5 \to 6$ 사이에서만 급변한다.

<!--en:3ce31cbe kind=slide_title-->
::: {.slide title="Applying the Horizontal Finite-Difference Kernel"}
<!--tr:3ce31cbe-->

::: {.slide title="수평 유한 차분 커널 적용하기"}

<!--en:b37a1029 kind=kicker-->
[Detecting Vertical Edges with K = [1, -1]]{.kicker}
<!--tr:b37a1029-->

[K = [1, -1]로 수직 에지 검출하기]{.kicker}

<!--en:9be85019 kind=slide_text-->
Cross-correlating `X` with the $1 \times 2$ kernel $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ computes $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ at every horizontal pair:
<!--tr:9be85019-->

`X` 와 $1 \times 2$ 커널 $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ 의 교차 상관은 모든 수평 쌍에서 $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ 을 계산한다:

<!--en:f649cfae kind=slide_text-->
- **Signed edge map:** Column 1 fires `+1` (white-to-black transition), column 5 fires `-1` (black-to-white transition), and all constant interior columns evaluate to `0`.
<!--tr:f649cfae-->

- **부호가 있는 에지 맵:** 1열은 `+1` (흰색에서 검은색으로 전환), 5열은 `-1` (검은색에서 흰색으로 전환)을 출력하며, 일정한 내부 열은 모두 `0` 이 된다.

<!--en:e6d155be kind=slide_title-->
::: {.slide title="Orientation Selectivity of Edge Kernels"}
<!--tr:e6d155be-->

::: {.slide title="에지 커널의 방향 선택성"}

<!--en:b525cde4 kind=kicker-->
[Why One Kernel Cannot Detect All Orientations]{.kicker}
<!--tr:b525cde4-->

[하나의 커널이 모든 방향을 검출할 수 없는 이유]{.kicker}

<!--en:38159de6 kind=slide_text-->
- **Blind to horizontal edges:** Transposing `X` (`X.T`) makes horizontal neighbors identical, so $[1, -1]$ outputs all zeros.
<!--tr:38159de6-->

- **수평 에지에는 반응하지 않음:** `X` 를 전치하면 (`X.T`) 수평 이웃이 같아지므로 $[1, -1]$ 은 모두 0을 출력한다.

<!--en:f0f30f76 kind=slide_text-->
- **Filter banks (§{sec_channels}):** Vision layers learn parallel channels for horizontal, vertical, and diagonal orientations.
<!--tr:f0f30f76-->

- **필터 뱅크 (§{sec_channels}):** 비전 층은 수평, 수직, 대각선 방향에 대한 병렬 채널을 학습한다.

<!--en:b755e31b kind=slide_title-->
::: {.slide title="Learning an Edge Detector via Gradient Descent"}
<!--tr:b755e31b-->

::: {.slide title="경사 하강법으로 에지 검출기 학습하기"}

<!--en:86bbeea7 kind=kicker-->
[Supervising a Built-In Convolutional Layer on (X, Y)]{.kicker}
<!--tr:86bbeea7-->

[(X, Y)로 내장 합성곱층 지도하기]{.kicker}

<!--en:a569780c kind=slide_text-->
- **4D framework tensors:** Reshape `X` and `Y` to 4D tensors (one example, one channel) and minimize $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ at $\eta = 0.03$.
<!--tr:a569780c-->

- **프레임워크의 4차원 텐서:** `X` 와 `Y` 를 4차원 텐서 (예제 하나, 채널 하나)로 바꾸고 $\eta = 0.03$ 에서 $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ 을 최소화한다.

<!--en:2489ce6e kind=slide_title-->
::: {.slide title="Inspecting the Learned Kernel Weights"}
<!--tr:2489ce6e-->

::: {.slide title="학습된 커널 가중치 살펴보기"}

<!--en:7e4ef44b kind=kicker-->
[Why Initial Loss Is 12 and Converges to [1, -1]]{.kicker}
<!--tr:7e4ef44b-->

[초기 손실이 12이고 [1, -1]로 수렴하는 이유]{.kicker}

<!--en:9cd1c645 kind=slide_text-->
- **Why loss starts at $12.0$:** Near-zero weights predict $\hat{Y} \approx 0$, missing the twelve $\pm 1$ edge pixels in `Y` ($12 \times 1^2 = 12$).
<!--tr:9cd1c645-->

- **손실이 $12.0$ 에서 시작하는 이유:** 거의 0인 가중치는 $\hat{Y} \approx 0$ 으로 예측하여 `Y` 의 열두 $\pm 1$ 에지 픽셀을 빗나간다 ($12 \times 1^2 = 12$).

<!--en:456670c5 kind=slide_text-->
- **Convergence in 9 steps:** SGD recovers $[0.99, -0.99]$ automatically from data.
<!--tr:456670c5-->

- **9단계 만에 수렴:** SGD가 데이터에서 $[0.99, -0.99]$ 을 자동으로 복원한다.

=== DECISIONS ===
- 문체: 해라체를 사용하고 수식과 참조 앞뒤에 공백을 둔다.
- edge, edge map → 에지, 에지 맵; window, patch → 윈도, 패치; locality → 국소성.
- dense layer → 밀집층; parameter → 파라미터; forward propagation → 순전파; unit → 유닛.

=== NOTES ===
- 4741e012, e810b3a9, 61cacfaf, a569780c: 한국어 어순에 맞게 수식 또는 코드 토큰의 순서를 바꾸었다.
