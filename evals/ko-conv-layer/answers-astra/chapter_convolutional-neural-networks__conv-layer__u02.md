<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u02 lang=ko rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:0b85caf9 kind=heading-->
## Convolution as Matrix Multiplication
<!--tr:0b85caf9-->
## 행렬 곱셈으로 계산하는 합성곱

<!--en:5bdffcd7 kind=paragraph-->
While the nested loops in `corr2d` make the math transparent, visiting one
output pixel at a time in Python is far too slow for real workloads. A classic
remedy, and still one of the algorithms that libraries choose from, is to recast
cross-correlation as a single large matrix multiplication, the exact operation
GPUs execute at peak speed. Notice that every
output entry in :numref:`fig_correlation` is simply the dot product of the
flattened $k_\textrm{h} k_\textrm{w}$-element kernel with the corresponding
flattened input window. Extracting all
$(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ windows and
stacking them as rows yields a patch matrix; multiplying it by the flattened
kernel computes the entire cross-correlation in one shot. This standard layout
transformation is called *im2col* ("image to columns", or rows here).
:numref:`fig_conv_im2col` carries it out on the example of
:numref:`fig_correlation`.
<!--tr:5bdffcd7-->
`corr2d`의 이중 반복문은 수식을 그대로 보여 주지만, Python에서 출력 픽셀을 하나씩 계산하는 방식은 실제 작업에 쓰기에는 너무 느리다. 오래전부터 쓰였고 지금도 라이브러리에서 선택하는 알고리즘 중 하나는 교차 상관을 큰 행렬 곱셈 한 번으로 바꾸는 것이다. 바로 GPU가 최고의 속도를 내는 연산이다. :numref:`fig_correlation` 의 각 출력 원소는 원소가 $k_\textrm{h} k_\textrm{w}$ 개인 커널을 한 줄로 펼친 벡터와 대응하는 입력 윈도를 한 줄로 펼친 벡터의 내적이다. 윈도 $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ 개를 모두 추출해 각각을 한 행으로 쌓으면 패치 행렬을 얻는다. 이 행렬에 한 줄로 펼친 커널을 곱하면 전체 교차 상관을 한 번에 계산할 수 있다. 이 표준적인 데이터 재배열을 *im2col*이라고 한다. ‘이미지를 열로 바꾼다’는 뜻으로, 여기서는 행으로 바꾼다. :numref:`fig_conv_im2col` 은 이 변환을 :numref:`fig_correlation` 의 예에 적용한 결과다.

<!--en:107554a5 kind=caption-->
![The cross-correlation of :numref:`fig_correlation` as one matrix product. Each of the four $2 \times 2$ windows of the input becomes a row of the patch matrix, and multiplying by the flattened kernel gives the flattened output. The shaded cells follow the first window to its row and to the output 19. The center pixel 4 lies in all four windows, so it appears in every row.](../img/conv-im2col.svg)
<!--tr:107554a5-->
![:numref:`fig_correlation` 의 교차 상관을 행렬 곱셈 한 번으로 계산하는 과정. 입력의 $2 \times 2$ 윈도 네 개가 각각 패치 행렬의 한 행이 된다. 이 행렬에 한 줄로 펼친 커널을 곱하면 한 줄로 펼친 출력을 얻는다. 음영을 따라가면 첫 번째 윈도가 어느 행이 되고 어떻게 출력값 19를 만드는지 볼 수 있다. 중앙의 픽셀 값 4는 네 윈도에 모두 속하므로 모든 행에 나타난다.](../img/conv-im2col.svg)

<!--en:76713479 kind=paragraph-->
The trade-off is extra workspace memory, since each interior pixel is duplicated
into up to $k_\textrm{h} k_\textrm{w}$ patches. Below we verify on
:numref:`fig_correlation` that this single matrix multiply matches `corr2d`
exactly:
<!--tr:76713479-->
이 방법은 내부 픽셀 하나를 최대 $k_\textrm{h} k_\textrm{w}$ 개의 패치에 복사하므로 작업용 메모리가 더 필요하다. 아래에서는 :numref:`fig_correlation` 의 예를 이용해 행렬 곱셈 한 번으로 계산한 값이 `corr2d`의 결과와 정확히 일치하는지 확인한다.

<!--en:3bcd7d8a kind=paragraph-->
The discrepancy is identically zero: one matrix product reproduces `corr2d`
perfectly.
<!--tr:3bcd7d8a-->
차이는 모든 위치에서 0이다. 행렬 곱셈 한 번으로 `corr2d`의 결과를 정확히 재현했다.

<!--en:c30f2dea kind=paragraph-->
Equivalently, we can unroll the *kernel* instead of the input into a sparse
banded matrix $\mathbf{W}$ multiplying the flattened image—the exact 2D analog
of panel (c) in :numref:`fig_why_conv_weight_sharing`, where every row repeats
the same handful of kernel weights. Seeing a convolution as a structured dense
layer brings us full circle to the derivation of :numref:`sec_why-conv` (and the
exercises build $\mathbf{W}$ explicitly).
<!--tr:c30f2dea-->
입력 대신 *커널*을 펼쳐 희소 띠 행렬 $\mathbf{W}$ 를 만들고 여기에 한 줄로 펼친 이미지를 곱해도 같은 결과를 얻는다. 이는 :numref:`fig_why_conv_weight_sharing` 의 패널 (c)를 그대로 2차원으로 옮긴 것으로, 행마다 동일한 소수의 커널 가중치가 반복된다. 합성곱을 이렇게 특정한 구조를 지닌 완전 연결층으로 보면 :numref:`sec_why-conv` 에서 합성곱을 유도한 출발점으로 돌아오게 된다. 연습문제에서는 $\mathbf{W}$ 를 직접 구성해 본다.

<!--en:ad4fc956 kind=heading-->
## Feature Maps and Receptive Fields
<!--tr:ad4fc956-->
## 특징 맵과 수용 영역

<!--en:e7983df8 kind=paragraph-->
A single convolutional layer sees only a patch the size of its kernel, whereas
recognizing whole objects demands integrating context across much wider areas.
Recall from :numref:`subsec_why-conv-channels` that a convolutional layer's
output is called a *feature map* (like our vertical edge map `Y`). For any
element $x$ in a feature map, its *receptive field* is the region of the input
image (or of an earlier layer) whose values can influence $x$ during forward
propagation—borrowing the neurophysiology term from :numref:`field_visual`.
<!--tr:e7983df8-->
합성곱층 하나는 커널 크기의 패치만 볼 수 있지만, 물체 전체를 인식하려면 훨씬 넓은 영역의 정보를 종합해야 한다. :numref:`subsec_why-conv-channels` 에서 보았듯이 합성곱층의 출력을 *특징 맵*이라고 한다. 앞서 구한 수직 에지 맵 `Y`가 한 예다. 특징 맵의 한 원소 $x$ 의 *수용 영역*은 순방향 패스에서 그 안의 값들이 $x$ 에 영향을 줄 수 있는 입력 이미지(또는 이전 층)의 영역이다. 이는 :numref:`field_visual` 에서 본 신경생리학 용어를 빌려온 것이다.

<!--en:b36f221e kind=heading-->
### Stacked Layers
<!--tr:b36f221e-->
### 여러 층을 쌓을 때의 수용 영역

<!--en:24a25ec4 kind=paragraph-->
In :numref:`fig_correlation`, the shaded output `19` has a $2 \times 2$ receptive
field covering the top-left four pixels of $\mathbf{X}$. Now stack a *second*
$2 \times 2$ convolutional layer on top of the $2 \times 2$ output $\mathbf{Y}$
to produce a single scalar $z$: $z$ depends on all four elements of $\mathbf{Y}$,
which together depend on all nine elements of $\mathbf{X}$. Stacking layers
expands the receptive field smoothly with depth.
<!--tr:24a25ec4-->
:numref:`fig_correlation` 에서 음영으로 표시한 출력 `19`의 수용 영역은 $2 \times 2$ 로, $\mathbf{X}$ 의 왼쪽 위 픽셀 네 개를 포함한다. 이제 *두 번째* $2 \times 2$ 합성곱층을 $2 \times 2$ 출력 $\mathbf{Y}$ 위에 쌓아 스칼라 $z$ 하나를 얻어 보자. $z$ 는 $\mathbf{Y}$ 의 네 원소 모두에 의존하고, 이 네 원소는 합쳐서 $\mathbf{X}$ 의 아홉 원소 모두에 의존한다. 이처럼 층을 쌓으면 깊이가 늘어날수록 수용 영역이 점차 넓어진다.

<!--en:f95c9394 kind=paragraph-->
How quickly does it grow? Each $k \times k$ convolutional layer reaches $k - 1$
pixels farther than a single element of its input, so stacking $L$ layers with
kernel sizes $k_1, \ldots, k_L$ yields a receptive field of side length
<!--tr:f95c9394-->
얼마나 빨리 넓어질까? $k \times k$ 합성곱층을 하나 거칠 때마다 입력 원소 하나가 보는 영역보다 한 변의 길이가 $k - 1$ 픽셀씩 늘어난다. 따라서 $L$ 개 층의 커널 크기가 각각 $k_1, \ldots, k_L$ 이면 수용 영역의 한 변의 길이는 다음과 같다.

<!--en:b302e3ef kind=paragraph-->
For a stack of $L$ consecutive $3 \times 3$ layers, each layer adds 2 pixels,
covering $(2L + 1) \times (2L + 1)$ inputs: $5 \times 5$ after two layers and
$7 \times 7$ after three (:numref:`fig_conv_receptive_field`). This linear
growth is actually a blessing rather than a reason to reach for huge kernels:
three stacked $3 \times 3$ layers achieve the same $7 \times 7$ receptive field
as a single $7 \times 7$ layer using far fewer parameters ($3 \times 9 = 27$
weights vs. 49 per channel pair) while interleaving three nonlinearities instead
of one.
<!--tr:b302e3ef-->
$L$ 개의 $3 \times 3$ 층을 연달아 쌓으면 층마다 한 변이 2픽셀씩 늘어나 입력에서 $(2L + 1) \times (2L + 1)$ 영역을 덮는다. 두 층을 쌓으면 $5 \times 5$, 세 층을 쌓으면 $7 \times 7$ 이다(:numref:`fig_conv_receptive_field`). 이렇게 선형으로 커진다는 것은 큰 커널을 써야 할 이유가 아니라 오히려 이점이다. $3 \times 3$ 층 세 개를 쌓으면 수용 영역이 $7 \times 7$ 로, $7 \times 7$ 층 하나와 같다. 그러면서도 매개변수가 훨씬 적고(채널 쌍마다 가중치가 $3 \times 9 = 27$ 개로, 큰 커널의 49개보다 적다), 비선형 변환도 한 번이 아니라 층마다 한 번씩 총 세 번 적용할 수 있다.

<!--en:39917513 kind=caption-->
![Receptive field of one unit after three one-dimensional layers with kernel size 3. Shaded cells are the units that the top unit depends on, and the lines connect each of them to its inputs. Each layer widens the receptive field by 2, to 7 inputs.](../img/conv-receptive-field.svg)
<!--tr:39917513-->
![커널 크기가 3인 1차원 층을 세 개 쌓았을 때 한 유닛의 수용 영역. 음영을 칠한 칸은 맨 위 유닛이 의존하는 유닛이고, 선은 각 유닛과 그 입력을 연결한다. 층을 하나 쌓을 때마다 수용 영역이 2씩 넓어져 입력 7개를 덮는다.](../img/conv-receptive-field.svg)

<!--en:48e4c9cc kind=heading-->
### Effective Receptive Fields
<!--tr:48e4c9cc-->
### 유효 수용 영역

<!--en:77d044db kind=paragraph-->
Keep in mind that $r$ measures the *theoretical* receptive field (every input
pixel with *any* path to an output unit) without saying how strongly each pixel
actually contributes. Measuring input gradients directly,
:citet:`Luo.Li.Urtasun.ea.2016` showed that the *effective* receptive field
decays sharply away from the center in a roughly Gaussian bell curve, in both
untrained and trained networks. Deep units can theoretically see far across the
image, but their sensitivity concentrates heavily near the center.
<!--tr:77d044db-->
$r$ 은 *이론적* 수용 영역, 즉 출력 유닛까지 가는 경로가 *하나라도* 있는 모든 입력 픽셀의 범위를 나타낸다. 각 픽셀이 실제로 얼마나 크게 기여하는지는 알려 주지 않는다. 입력에 대한 기울기를 직접 측정한 :citet:`Luo.Li.Urtasun.ea.2016` 은 학습 전후의 신경망 모두에서 *유효* 수용 영역의 민감도가 중심에서 멀어질수록 급격히 떨어지며 대략 가우시안 종 모양 곡선을 그린다는 것을 보였다. 깊은 층의 유닛은 이론상 이미지의 먼 곳까지 볼 수 있지만, 실제로 민감하게 반응하는 곳은 중심 부근에 집중된다.

<!--en:eed4e2a0 kind=paragraph-->
Counting paths explains the bell shape. An input at the edge of the receptive
field reaches the output unit along a single chain of connections, whereas the
center input can take many routes through the layers in between.
:numref:`fig_conv_rf_paths` counts them for the tower of
:numref:`fig_conv_receptive_field`: the center input has seven paths to the top
unit and each edge input has one. If all weights have a similar magnitude, the
influence of an input grows with its number of paths.
<!--tr:eed4e2a0-->
경로 수를 세어 보면 종 모양이 나타나는 이유를 알 수 있다. 수용 영역의 가장자리에 있는 입력이 출력 유닛에 이르는 경로는 하나뿐이지만, 중심에 있는 입력은 중간층들을 거쳐 여러 경로로 출력에 이를 수 있다. :numref:`fig_conv_rf_paths` 은 :numref:`fig_conv_receptive_field` 에 쌓아 올린 층에서 이 경로들을 센 결과다. 중심 입력에서 맨 위 유닛까지는 경로가 일곱 개이고 양 끝의 입력에서는 각각 하나다. 모든 가중치의 크기가 비슷하다면 입력의 영향력은 경로 수와 함께 커진다.

<!--en:b92775ee kind=caption-->
![Number of paths from each unit to the top unit of :numref:`fig_conv_receptive_field` (three one-dimensional layers with kernel size 3). Each count is the sum of the counts of the units that the cell feeds. Darker cells have more paths than the other cells of their layer, and the bars repeat the counts of the input row: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)
<!--tr:b92775ee-->
![각 유닛에서 :numref:`fig_conv_receptive_field` 의 맨 위 유닛까지 이르는 경로 수(커널 크기 3인 1차원 층 세 개). 각 칸의 경로 수는 그 칸에서 입력을 받는 유닛들의 경로 수를 더한 값이다. 같은 층에서는 색이 진한 칸일수록 경로가 많다. 막대는 입력 행의 경로 수인 1, 3, 6, 7, 6, 3, 1을 다시 나타낸 것이다.](../img/conv-rf-paths.svg)

<!--en:405695e8 kind=paragraph-->
Two practical gaps remain in our basic layer: every unpadded convolution shrinks
spatial dimensions while under-sampling border pixels, and a single kernel
detects only one pattern. We address spatial sizing and borders next in
:numref:`sec_padding`, and multi-channel feature maps in :numref:`sec_channels`.
<!--tr:405695e8-->
지금까지 살펴본 기본 층에는 실제로 사용할 때 두 가지 한계가 있다. 패딩 없는 합성곱은 적용할 때마다 공간 차원의 크기가 줄고 가장자리 픽셀을 덜 자주 사용한다. 또 커널 하나는 패턴 하나만 검출한다. 다음으로 :numref:`sec_padding` 에서 공간 크기 조절과 가장자리 처리를 살펴보고, :numref:`sec_channels` 에서 다중 채널 특징 맵을 다룬다.

<!--en:30ac03ff kind=heading-->
## Summary
<!--tr:30ac03ff-->
## 요약

<!--en:92688a0c kind=paragraph-->
A convolutional layer applies two-dimensional cross-correlation with a learned
kernel plus a scalar bias, sliding the kernel across every valid window of the
input. A simple $1 \times 2$ finite-difference kernel `[1, -1]` detects vertical
edges anywhere in an image, and gradient descent recovers those exact weights
from input–output data alone, just as deep CNNs learn entire hierarchies of
filters end to end. Mathematical convolution flips the kernel, making no
difference once kernels are learned. A convolution can be evaluated as a single
matrix multiplication (*im2col*), and stacking $L$ convolutional
layers widens each unit's receptive field linearly by $k_i - 1$ per layer.
<!--tr:92688a0c-->
합성곱층은 학습된 커널을 입력의 모든 유효한 윈도 위치로 옮겨 가며 2차원 교차 상관을 계산하고 스칼라 편향을 더한다. 간단한 $1 \times 2$ 유한 차분 커널 `[1, -1]`은 이미지 어디에서든 수직 에지를 검출한다. 깊은 합성곱 신경망이 모든 층의 필터를 처음부터 끝까지 함께 학습하듯이, 경사 하강법은 입력과 출력 데이터만으로 그 가중치를 찾아낸다. 수학적 합성곱에서는 커널을 뒤집지만 커널을 학습하면 이 차이는 사라진다. 합성곱은 행렬 곱셈 한 번으로 계산할 수 있다(*im2col*). 합성곱층을 $L$ 개 쌓으면 각 유닛의 수용 영역은 층마다 $k_i - 1$ 씩 선형으로 넓어진다.

<!--en:3ba929d2 kind=slide_title-->
::: {.slide title="Fast Execution: Convolution via Matrix Multiplication"}
<!--tr:3ba929d2-->
::: {.slide title="행렬 곱셈을 이용한 빠른 합성곱 계산"}

<!--en:1711f812 kind=kicker-->
[The im2col (Image-to-Columns) Transformation]{.kicker}
<!--tr:1711f812-->
[이미지를 열로 바꾸는 im2col 변환]{.kicker}

<!--en:a040f3d7 kind=slide_text-->
- **Single dense GEMM:** Flatten each $k_\textrm{h} \times k_\textrm{w}$ window into a row of `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ and multiply by $\textrm{vec}(\mathbf{K})$.
<!--tr:a040f3d7-->
- **밀집 행렬 곱셈(GEMM) 한 번** 각 $k_\textrm{h} \times k_\textrm{w}$ 윈도를 한 줄로 펼쳐 `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ 의 한 행으로 놓고 이 행렬에 $\textrm{vec}(\mathbf{K})$ 를 곱한다.

<!--en:1fb18ec4 kind=slide_text-->
- **Trade-off:** Peak GPU Tensor Core speed at the cost of $k_\textrm{h} k_\textrm{w}\times$ patch workspace duplication.
<!--tr:1fb18ec4-->
- **속도와 메모리의 절충** 패치 작업 공간에 데이터를 $k_\textrm{h} k_\textrm{w}\times$ 까지 복제하는 비용으로 GPU Tensor Core의 최고 속도를 활용한다.

<!--en:0d757cf6 kind=slide_title-->
::: {.slide title="Dual View: Unrolling the Kernel into a Banded Matrix"}
<!--tr:0d757cf6-->
::: {.slide title="합성곱을 보는 또 다른 방법 — 커널을 띠 행렬로 펼치기"}

<!--en:7f0defaa kind=kicker-->
[Toeplitz Weight Matrix Multiplication]{.kicker}
<!--tr:7f0defaa-->
[퇴플리츠 가중치 행렬 곱셈]{.kicker}

<!--en:5898ba57 kind=slide_text-->
Instead of unfolding the input image into duplicated patches, we can unroll the *kernel* into a sparse banded matrix $\mathbf{W}$ that multiplies the flattened input vector $\textrm{vec}(\mathbf{X})$:
<!--tr:5898ba57-->
입력 이미지를 중복되는 패치로 펼치는 대신 *커널*을 희소 띠 행렬 $\mathbf{W}$ 로 펼친 뒤 여기에 한 줄로 펼친 입력 벡터 $\textrm{vec}(\mathbf{X})$ 를 곱할 수도 있다.

<!--en:dae51741 kind=slide_text-->
- **Matrix–vector form:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, where $\mathbf{W}$ has $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ rows and $n_\textrm{h} n_\textrm{w}$ columns.
<!--tr:dae51741-->
- **행렬과 벡터의 곱** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$ 이며, $\mathbf{W}$ 의 행 수는 $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$, 열 수는 $n_\textrm{h} n_\textrm{w}$ 이다.

<!--en:f0e86ee3 kind=slide_text-->
- **Connection to §{sec_why-conv} and backprop:** Every row of $\mathbf{W}$ holds the same $k_\textrm{h} k_\textrm{w}$ kernel entries shifted across columns (panel c), making the backward pass with respect to $\mathbf{X}$ simply multiplication by $\mathbf{W}^\top$ (a transposed convolution).
<!--tr:f0e86ee3-->
- **§{sec_why-conv} 및 역전파와의 관계** $\mathbf{W}$ 의 각 행에는 동일한 커널 원소 $k_\textrm{h} k_\textrm{w}$ 개가 열 방향으로 위치만 옮겨 들어간다(패널 c). 따라서 $\mathbf{X}$ 에 대한 역방향 패스는 $\mathbf{W}^\top$ 를 곱하는 연산, 즉 전치 합성곱이다.

<!--en:af8cf19a kind=slide_title-->
::: {.slide title="Feature Maps and Receptive Fields Across Depth"}
<!--tr:af8cf19a-->
::: {.slide title="깊이에 따른 특징 맵과 수용 영역"}

<!--en:4eb965f3 kind=kicker-->
[Linear Receptive-Field Growth in Stacked Layers]{.kicker}
<!--tr:4eb965f3-->
[층을 쌓을 때 선형으로 넓어지는 수용 영역]{.kicker}

<!--en:5599a6e9 kind=slide_text-->
A unit's **receptive field** is the region of the original input image whose pixels can influence that unit's activation during forward propagation:
<!--tr:5599a6e9-->
한 유닛의 **수용 영역**은 순방향 패스에서 그 유닛의 활성화에 영향을 줄 수 있는 픽셀들이 차지하는 원본 입력 이미지의 영역이다.

<!--en:eba74f75 kind=slide_text-->
- **Single layer ($L=1$):** A $k_1 \times k_1$ kernel sees a $k_1 \times k_1$ input patch.
<!--tr:eba74f75-->
- **단일 층($L=1$)** $k_1 \times k_1$ 커널은 $k_1 \times k_1$ 입력 패치를 본다.

<!--en:01c175bf kind=slide_text-->
- **Stacking $L$ stride-1 layers:** Layer $i$ with kernel size $k_i$ reaches $k_i - 1$ pixels farther along each axis than a single element of layer $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$
<!--tr:01c175bf-->
- **$L$ 개 층을 스트라이드 1로 쌓는 경우** $i$ 번째 층의 커널 크기가 $k_i$ 이면 수용 영역은 각 축을 따라 $k_i - 1$ 픽셀씩 넓어진다. 기준은 $i-1$ 번째 층의 원소 하나가 보는 영역이다.
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$

<!--en:2bde5822 kind=slide_text-->
- **$L$ stacked $3 \times 3$ layers:** Each layer adds $3 - 1 = 2$ pixels, giving side length $r = 2L + 1$ ($3 \to 5 \to 7$).
<!--tr:2bde5822-->
- **$L$ 개의 $3 \times 3$ 층을 쌓는 경우** 층마다 $3 - 1 = 2$ 픽셀씩 늘어나 한 변의 길이는 $r = 2L + 1$ 이 된다($3 \to 5 \to 7$).

<!--en:cbb9e97a kind=slide_title-->
::: {.slide title="Why Deep Stacks of 3×3 Kernels Beat One Large Kernel"}
<!--tr:cbb9e97a-->
::: {.slide title="3×3 커널을 여러 층 쌓으면 큰 커널 하나보다 유리한 이유"}

<!--en:95e46e70 kind=kicker-->
[Parameter Efficiency & Nonlinear Expressivity]{.kicker}
<!--tr:95e46e70-->
[매개변수 효율과 비선형 표현력]{.kicker}

<!--en:bd1d7cc0 kind=slide_text-->
- **Matched $7 \times 7$ receptive field:** Three stacked $3 \times 3$ layers ($r = 1 + 2 + 2 + 2 = 7$) see the exact same $7 \times 7$ patch as one $7 \times 7$ layer.
<!--tr:bd1d7cc0-->
- **동일한 $7 \times 7$ 수용 영역** $3 \times 3$ 층 세 개를 쌓으면($r = 1 + 2 + 2 + 2 = 7$) $7 \times 7$ 패치를 보게 되며, 이는 $7 \times 7$ 층 하나가 보는 영역과 정확히 같다.

<!--en:755ac11a kind=slide_text-->
- **45% fewer parameters:** $3 \times 9 c^2 = 27 c^2$ weights instead of $49 c^2$ across $c$ channels.
<!--tr:755ac11a-->
- **매개변수 45% 감소** 가중치가 $3 \times 9 c^2 = 27 c^2$ 개로 줄어든다. 큰 커널을 쓰면 $49 c^2$ 개가 필요하다($c$ 개 채널 기준).

<!--en:e3aa1086 kind=slide_text-->
- **Richer expressivity:** Compounds **3 nonlinear activations** instead of 1 (the core VGG design rule in §{sec_blocks}).
<!--tr:e3aa1086-->
- **더 풍부한 표현력** 활성화 함수 하나 대신 **비선형 활성화 함수 세 개**를 연이어 적용한다(§{sec_blocks} 에서 다루는 VGG의 핵심 설계 원칙).

<!--en:9ec83bf6 kind=slide_title-->
::: {.slide title="Theoretical vs. Effective Receptive Fields"}
<!--tr:9ec83bf6-->
::: {.slide title="이론적 수용 영역과 유효 수용 영역"}

<!--en:72a52c3b kind=kicker-->
[Gaussian Decay of Sensitivity Away from the Center]{.kicker}
<!--tr:72a52c3b-->
[중심에서 멀어질수록 가우시안 곡선을 따라 감소하는 민감도]{.kicker}

<!--en:33bb377d kind=slide_text-->
- **Theoretical RF ($r = 2L+1$):** Bounding box of all pixels with *at least one* path to the output unit.
<!--tr:33bb377d-->
- **이론적 수용 영역($r = 2L+1$)** 출력 유닛까지 경로가 *적어도 하나* 있는 모든 픽셀을 둘러싼 경계 상자다.

<!--en:293436eb kind=slide_text-->
- **Central Limit Theorem:** Center pixels connect via exponentially many paths, whereas corner pixels connect via only **1 path**.
<!--tr:293436eb-->
- **중심 극한 정리** 중심 픽셀에서 오는 경로 수는 깊이에 따라 지수적으로 늘어나지만, 모서리 픽셀은 **경로 하나**로만 연결된다.

<!--en:f47d181e kind=slide_text-->
- **Effective RF (Luo et al., 2016):** Input gradient magnitude $\partial y / \partial \mathbf{X}$ decays in a Gaussian bell curve with std $\sigma \propto \sqrt{L}$.
<!--tr:f47d181e-->
- **유효 수용 영역(Luo 외, 2016)** 입력 기울기 $\partial y / \partial \mathbf{X}$ 의 크기는 표준 편차가 $\sigma \propto \sqrt{L}$ 인 가우시안 종 모양 곡선을 따라 감소한다.

<!--en:fac792d0 kind=slide_title-->
::: {.slide title="Concept Check: Cross-Correlation & Receptive Fields"}
<!--tr:fac792d0-->
::: {.slide title="개념 확인 — 교차 상관과 수용 영역"}

<!--en:26a317d8 kind=kicker-->
[Concept Check]{.kicker}
<!--tr:26a317d8-->
[개념 확인]{.kicker}

<!--en:8fe554e5 kind=slide_text-->
Test your understanding of output shapes, `im2col`, and receptive-field arithmetic:
<!--tr:8fe554e5-->
출력의 모양, `im2col`, 수용 영역 계산을 얼마나 이해했는지 확인해 보자.

<!--en:281c7c78 kind=slide_text-->
1. **im2col matrix dimensions:** You apply an unpadded $3 \times 3$ single-channel kernel to a $10 \times 10$ image via `im2col`. What is the output spatial shape, and what are the exact dimensions of the `patches` matrix and flattened kernel vector?
<!--tr:281c7c78-->
1. **im2col 행렬의 크기** 패딩 없는 $3 \times 3$ 단일 채널 커널을 $10 \times 10$ 이미지에 `im2col`로 적용한다. 출력의 공간적 모양은 무엇인가? `patches` 행렬과 한 줄로 펼친 커널 벡터의 크기는 각각 정확히 얼마인가?

<!--en:16976d18 kind=slide_text-->
2. **Depth to span Fashion-MNIST:** At stride 1, how many stacked $3 \times 3$ layers are required so a single unit's theoretical receptive field spans an entire $28 \times 28$ image? How many with $5 \times 5$ layers?
<!--tr:16976d18-->
2. **Fashion-MNIST 이미지 전체를 보는 데 필요한 깊이** 스트라이드가 1일 때, $3 \times 3$ 층을 몇 개 쌓아야 유닛 하나의 이론적 수용 영역이 $28 \times 28$ 이미지 전체를 덮는가? $5 \times 5$ 층이라면 몇 개가 필요한가?

<!--en:ab2e0d6a kind=slide_text-->
3. **Edge-loss boundary shift:** In our $6 \times 8$ vertical-edge learning experiment, why did the loss start at $\approx 12$, and what would the initial loss be if the black band extended all the way to the right edge (`X[:, 2:8] = 0`)?
<!--tr:ab2e0d6a-->
3. **경계 위치에 따른 에지 검출 손실** $6 \times 8$ 수직 에지 학습 실험에서 손실이 $\approx 12$ 로 시작한 이유는 무엇인가? 검은 띠를 오른쪽 끝까지 늘리면(`X[:, 2:8] = 0`) 초기 손실은 얼마가 될까?

<!--en:250d6877 kind=slide_text-->
**Answers:**
(1) Output shape is $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ outputs). The `patches` matrix has shape $(64, 9)$ and multiplies the $(9, 1)$ kernel vector.
(2) For $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ layers. For $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ layers.
(3) Each of the 6 rows had two $\pm 1$ edges ($6 \times 2 \times 1^2 = 12$). Extending the black band to the right border eliminates the right-hand black-to-white edge, leaving one edge per row and halving the initial loss to $\approx 6$.
<!--tr:250d6877-->
**정답**
(1) 출력 모양은 $(10-3+1) \times (10-3+1) = 8 \times 8$ 이다(출력 $64$ 개). `patches` 행렬의 모양은 $(64, 9)$ 이며 여기에 $(9, 1)$ 커널 벡터를 곱한다.
(2) $3 \times 3$ 층은 $2L + 1 \ge 28 \implies L = 14$ 개, $5 \times 5$ 층은 $4L + 1 \ge 28 \implies L = 7$ 개가 필요하다.
(3) 6개 행마다 값이 $\pm 1$ 인 에지가 두 개씩 있었다($6 \times 2 \times 1^2 = 12$). 검은 띠를 오른쪽 끝까지 늘리면 검은색에서 흰색으로 바뀌는 오른쪽 에지가 사라진다. 그러면 행마다 에지가 하나씩만 남아 초기 손실이 절반인 $\approx 6$ 이 된다.

<!--en:ba43a2dd kind=slide_title-->
::: {.slide title="Summary & Key Takeaways"}
<!--tr:ba43a2dd-->
::: {.slide title="요약과 핵심 내용"}

<!--en:ce12e85f kind=kicker-->
[Section Synthesis]{.kicker}
<!--tr:ce12e85f-->
[이 절의 내용 종합]{.kicker}

<!--en:00011fb7 kind=slide_text-->
- **2D Cross-Correlation & Edge Learning:** Slides $\mathbf{K}$ across valid windows ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); SGD learns $[1, -1]$ in 9 steps.
<!--tr:00011fb7-->
- **2차원 교차 상관과 에지 검출 학습** $\mathbf{K}$ 를 유효한 윈도 위치($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$)로 옮겨 가며 계산한다. 확률적 경사 하강법은 9단계 만에 $[1, -1]$ 을 학습한다.

<!--en:09efe4f0 kind=slide_text-->
- **im2col Matrix Multiply:** Trades $k_\textrm{h} k_\textrm{w}\times$ patch duplication for peak GPU GEMM throughput.
<!--tr:09efe4f0-->
- **im2col 행렬 곱셈** 패치 데이터를 $k_\textrm{h} k_\textrm{w}\times$ 까지 복제하는 대신 GPU 행렬 곱셈(GEMM)의 최대 처리량을 활용한다.

<!--en:3af8acd0 kind=slide_text-->
- **Stacked Receptive Fields:** $r = 1 + \sum (k_i - 1) = 2L+1$ for $3 \times 3$ stacks, with Gaussian center sensitivity.
<!--tr:3af8acd0-->
- **층을 쌓을 때의 수용 영역** $r = 1 + \sum (k_i - 1) = 2L+1$ 이다($3 \times 3$ 층을 쌓는 경우). 민감도는 중심에서 가장 큰 가우시안 곡선을 그린다.

=== DECISIONS ===
- patch matrix → 패치 행렬, banded matrix → 띠 행렬, sparse banded matrix → 희소 띠 행렬, effective receptive field → 유효 수용 영역, theoretical receptive field → 이론적 수용 영역.
- flatten은 배열의 원소를 ‘한 줄로 펼치다’로 옮긴다. im2col은 알고리즘 이름을 유지한다.

=== NOTES ===
- e3aa1086: 숫자 3과 1은 각각 ‘세 개’와 ‘하나’로 옮겼다.
- 293436eb: 경로 수 1은 ‘경로 하나’로 옮겼다.
