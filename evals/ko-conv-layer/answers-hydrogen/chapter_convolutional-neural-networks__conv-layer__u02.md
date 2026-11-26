<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u02 lang=ko rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:0b85caf9 kind=heading-->
## Convolution as Matrix Multiplication
<!--tr:0b85caf9-->
## 행렬 곱셈으로서의 합성곱

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
`corr2d` 의 중첩 반복문은 수학적 의미를 투명하게 보여 주지만, Python에서 출력 픽셀을 하나씩 방문하는 방식은 실제 작업에 쓰기에는 너무 느립니다. 고전적인 해결책이자 지금도 라이브러리가 선택하는 알고리즘 중 하나는 교차 상관을 하나의 큰 행렬 곱셈으로 바꾸는 것입니다. 이는 바로 GPU가 최고 속도로 실행하는 연산입니다. :numref:`fig_correlation` 의 모든 출력 원소는 $k_\textrm{h} k_\textrm{w}$ 개의 원소를 가진 평탄화된 커널과 그에 대응하는 평탄화된 입력 윈도우의 내적에 불과합니다. $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ 개의 윈도우를 모두 추출해 행으로 쌓으면 패치 행렬이 되며, 여기에 평탄화된 커널을 곱하면 전체 교차 상관을 단번에 계산할 수 있습니다. 이러한 표준 배치 변환을 *im2col*("image to columns", 여기서는 행으로 쌓음)이라고 합니다. :numref:`fig_conv_im2col` 은 :numref:`fig_correlation` 의 예에 이를 적용한 모습을 보여 줍니다.

<!--en:107554a5 kind=caption-->
![The cross-correlation of :numref:`fig_correlation` as one matrix product. Each of the four $2 \times 2$ windows of the input becomes a row of the patch matrix, and multiplying by the flattened kernel gives the flattened output. The shaded cells follow the first window to its row and to the output 19. The center pixel 4 lies in all four windows, so it appears in every row.](../img/conv-im2col.svg)
<!--tr:107554a5-->
![하나의 행렬 곱으로 나타낸 :numref:`fig_correlation` 의 교차 상관. 입력의 네 $2 \times 2$ 윈도우가 각각 패치 행렬의 한 행이 되며, 여기에 평탄화된 커널을 곱하면 평탄화된 출력이 됩니다. 음영 처리된 칸은 첫 번째 윈도우가 해당 행과 출력 19로 이어지는 과정을 나타냅니다. 가운데 픽셀 4는 네 윈도우 모두에 속하므로 모든 행에 나타납니다.](../img/conv-im2col.svg)

<!--en:76713479 kind=paragraph-->
The trade-off is extra workspace memory, since each interior pixel is duplicated
into up to $k_\textrm{h} k_\textrm{w}$ patches. Below we verify on
:numref:`fig_correlation` that this single matrix multiply matches `corr2d`
exactly:
<!--tr:76713479-->
그 대가는 추가 작업 공간 메모리입니다. 내부 픽셀 하나가 최대 $k_\textrm{h} k_\textrm{w}$ 개의 패치에 중복 복사되기 때문입니다. 아래에서는 :numref:`fig_correlation` 에 대해 이 단일 행렬 곱셈이 `corr2d` 와 정확히 일치함을 검증합니다.

<!--en:3bcd7d8a kind=paragraph-->
The discrepancy is identically zero: one matrix product reproduces `corr2d`
perfectly.
<!--tr:3bcd7d8a-->
차이는 정확히 0입니다. 단 한 번의 행렬 곱셈이 `corr2d` 를 완벽하게 재현합니다.

<!--en:c30f2dea kind=paragraph-->
Equivalently, we can unroll the *kernel* instead of the input into a sparse
banded matrix $\mathbf{W}$ multiplying the flattened image—the exact 2D analog
of panel (c) in :numref:`fig_why_conv_weight_sharing`, where every row repeats
the same handful of kernel weights. Seeing a convolution as a structured dense
layer brings us full circle to the derivation of :numref:`sec_why-conv` (and the
exercises build $\mathbf{W}$ explicitly).
<!--tr:c30f2dea-->
이와 동등하게, 입력 대신 *커널*을 펼처서 평탄화된 이미지에 곱해지는 희소 띠 행렬 $\mathbf{W}$ 로 만들 수도 있습니다. 이는 :numref:`fig_why_conv_weight_sharing` 의 (c)에서 모든 행이 동일한 소수의 커널 가중치를 반복하는 구조의 정확한 2D 대응물입니다. 합성곱을 구조화된 밀집층으로 바라보면 :numref:`sec_why-conv` 의 유도로 다시 돌아오게 됩니다(연습문제에서 $\mathbf{W}$ 를 직접 구성해 봅니다).

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
단일 합성곱층은 커널 크기만 한 패치만 볼 수 있는 반면, 전체 객체를 인식하려면 훨씬 넓은 영역의 문맥을 통합해야 합니다. :numref:`subsec_why-conv-channels` 에서 살펴보았듯이 합성곱층의 출력을 *특징 맵*(feature map)이라고 부릅니다(앞서 본 수직 경계선 맵 `Y` 처럼요). 특징 맵의 임의의 원소 $x$ 에 대해, *수용 영역*(receptive field)은 순방향 패스 동안 $x$ 에 영향을 줄 수 있는 입력 이미지(또는 앞선 층)의 영역을 뜻하며, 이는 :numref:`field_visual` 에서 다룬 신경생리학 용어를 빌려온 것입니다.

<!--en:b36f221e kind=heading-->
### Stacked Layers
<!--tr:b36f221e-->
### 층 쌓기

<!--en:24a25ec4 kind=paragraph-->
In :numref:`fig_correlation`, the shaded output `19` has a $2 \times 2$ receptive
field covering the top-left four pixels of $\mathbf{X}$. Now stack a *second*
$2 \times 2$ convolutional layer on top of the $2 \times 2$ output $\mathbf{Y}$
to produce a single scalar $z$: $z$ depends on all four elements of $\mathbf{Y}$,
which together depend on all nine elements of $\mathbf{X}$. Stacking layers
expands the receptive field smoothly with depth.
<!--tr:24a25ec4-->
:numref:`fig_correlation` 에서 음영 처리된 출력 `19` 는 $\mathbf{X}$ 의 왼쪽 위 네 픽셀을 덮는 $2 \times 2$ 수용 영역을 갖습니다. 이제 $2 \times 2$ 출력 $\mathbf{Y}$ 위에 *두 번째* $2 \times 2$ 합성곱층을 쌓아 단일 스칼라 $z$ 를 만든다고 해 봅시다. $z$ 는 $\mathbf{Y}$ 의 네 원소 모두에 의존하고, 이 네 원소는 합쳐서 $\mathbf{X}$ 의 아홉 원소 모두에 의존합니다. 층을 쌓으면 깊이에 따라 수용 영역이 매끄럽게 넓어집니다.

<!--en:f95c9394 kind=paragraph-->
How quickly does it grow? Each $k \times k$ convolutional layer reaches $k - 1$
pixels farther than a single element of its input, so stacking $L$ layers with
kernel sizes $k_1, \ldots, k_L$ yields a receptive field of side length
<!--tr:f95c9394-->
수용 영역은 얼마나 빠르게 커질까요? 각 $k \times k$ 합성곱층은 입력의 단일 원소보다 $k - 1$ 픽셀 더 멀리까지 닿으므로, 커널 크기가 $k_1, \ldots, k_L$ 인 $L$ 개의 층을 쌓으면 한 변의 길이가 다음과 같은 수용 영역을 얻습니다.

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
연속된 $L$ 개의 $3 \times 3$ 층을 쌓으면 각 층이 2픽셀씩 더해져 $(2L + 1) \times (2L + 1)$ 입력을 덮습니다. 두 층 뒤에는 $5 \times 5$, 세 층 뒤에는 $7 \times 7$ 이 됩니다 (:numref:`fig_conv_receptive_field`). 이러한 선형 성장은 거대한 커널을 써야 할 이유가 아니라 오히려 장점입니다. 세 개의 $3 \times 3$ 층을 쌓으면 단일 $7 \times 7$ 층과 똑같은 $7 \times 7$ 수용 영역을 달성하면서도 매개변수는 훨씬 적게 쓰고(채널 쌍당 49개 대신 $3 \times 9 = 27$ 개의 가중치), 비선형 변환도 한 번 대신 세 번 끼워 넣을 수 있습니다.

<!--en:39917513 kind=caption-->
![Receptive field of one unit after three one-dimensional layers with kernel size 3. Shaded cells are the units that the top unit depends on, and the lines connect each of them to its inputs. Each layer widens the receptive field by 2, to 7 inputs.](../img/conv-receptive-field.svg)
<!--tr:39917513-->
![커널 크기가 3인 일차원 층 세 개를 거친 후 한 유닛의 수용 영역. 음영 처리된 칸은 맨 위 유닛이 의존하는 유닛들이며, 선은 각 유닛을 그 입력에 연결합니다. 각 층마다 수용 영역이 2씩 넓어져 총 7개의 입력에 이릅니다.](../img/conv-receptive-field.svg)

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
$r$ 은 각 픽셀이 실제로 얼마나 강하게 기여하는지는 말하지 않고 *이론적* 수용 영역(출력 유닛으로 이어지는 경로가 *하나라도* 있는 모든 입력 픽셀)만을 측정한다는 점을 유념해야 합니다. :citet:`Luo.Li.Urtasun.ea.2016` 은 입력 기울기를 직접 측정하여, 학습되지 않은 신경망과 학습된 신경망 모두에서 *유효* 수용 영역이 중심에서 멀어질수록 대략 가우시안 종 모양 곡선을 그리며 급격히 감소함을 보였습니다. 깊은 층의 유닛은 이론적으로 이미지의 먼 곳까지 볼 수 있지만, 실제 민감도는 중심 부근에 크게 집중됩니다.

<!--en:eed4e2a0 kind=paragraph-->
Counting paths explains the bell shape. An input at the edge of the receptive
field reaches the output unit along a single chain of connections, whereas the
center input can take many routes through the layers in between.
:numref:`fig_conv_rf_paths` counts them for the tower of
:numref:`fig_conv_receptive_field`: the center input has seven paths to the top
unit and each edge input has one. If all weights have a similar magnitude, the
influence of an input grows with its number of paths.
<!--tr:eed4e2a0-->
경로의 개수를 세어 보면 종 모양이 설명됩니다. 수용 영역 가장자리에 있는 입력은 단 하나의 연결 사슬을 따라 출력 유닛에 도달하는 반면, 중심 입력은 중간 층들을 거쳐 여러 경로로 도달할 수 있습니다. :numref:`fig_conv_rf_paths` 는 :numref:`fig_conv_receptive_field` 의 구조에서 경로 수를 센 결과입니다. 중심 입력은 맨 위 유닛까지 7개의 경로를 가지며 양 끝 입력은 각각 1개의 경로만 가집니다. 모든 가중치의 크기가 비슷하다면 입력의 영향력은 경로의 수에 비례해 커집니다.

<!--en:b92775ee kind=caption-->
![Number of paths from each unit to the top unit of :numref:`fig_conv_receptive_field` (three one-dimensional layers with kernel size 3). Each count is the sum of the counts of the units that the cell feeds. Darker cells have more paths than the other cells of their layer, and the bars repeat the counts of the input row: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)
<!--tr:b92775ee-->
![:numref:`fig_conv_receptive_field` 의 맨 위 유닛으로 이어지는 각 유닛의 경로 수(커널 크기가 3인 일차원 층 세 개). 각 숫자는 해당 칸이 연결되는 상위 유닛들의 경로 수 합입니다. 어두운 칸일수록 같은 층의 다른 칸보다 경로가 많으며, 아래 막대는 입력 행의 경로 수인 1, 3, 6, 7, 6, 3, 1을 그대로 나타냅니다.](../img/conv-rf-paths.svg)

<!--en:405695e8 kind=paragraph-->
Two practical gaps remain in our basic layer: every unpadded convolution shrinks
spatial dimensions while under-sampling border pixels, and a single kernel
detects only one pattern. We address spatial sizing and borders next in
:numref:`sec_padding`, and multi-channel feature maps in :numref:`sec_channels`.
<!--tr:405695e8-->
이 기본 층에는 두 가지 실용적인 한계가 남아 있습니다. 패딩 없는 합성곱은 매번 공간 차원을 줄이면서 경계 픽셀을 덜 샘플링하며, 단일 커널은 하나의 패턴만 감지합니다. 공간 크기 조절과 경계 처리는 다음 :numref:`sec_padding` 에서, 다중 채널 특징 맵은 :numref:`sec_channels` 에서 다룹니다.

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
합성곱층은 입력의 모든 유효한 윈도우 위로 학습된 커널을 밀어 이차원 교차 상관을 수행한 뒤 스칼라 편향을 더합니다. 단순한 $1 \times 2$ 유한 차분 커널 `[1, -1]` 은 이미지 어디서나 수직 경계선을 감지하며, 깊은 합성곱 신경망이 전체 필터 계층을 종단간 학습하듯이 경사 하강법도 입출력 데이터만으로 그 정확한 가중치를 복원합니다. 수학적 합성곱은 커널을 뒤집지만 커널을 학습하고 나면 차이가 없습니다. 합성곱은 하나의 행렬 곱셈(*im2col*)으로 계산할 수 있으며, $L$ 개의 합성곱층을 쌓으면 각 유닛의 수용 영역이 층마다 $k_i - 1$ 만큼 선형으로 넓어집니다.

<!--en:3ba929d2 kind=slide_title-->
::: {.slide title="Fast Execution: Convolution via Matrix Multiplication"}
<!--tr:3ba929d2-->
::: {.slide title="고속 실행: 행렬 곱셈을 통한 합성곱"}

<!--en:1711f812 kind=kicker-->
[The im2col (Image-to-Columns) Transformation]{.kicker}
<!--tr:1711f812-->
[im2col (Image-to-Columns) 변환]{.kicker}

<!--en:a040f3d7 kind=slide_text-->
- **Single dense GEMM:** Flatten each $k_\textrm{h} \times k_\textrm{w}$ window into a row of `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ and multiply by $\textrm{vec}(\mathbf{K})$.
<!--tr:a040f3d7-->
- **단일 밀집 GEMM:** 각 $k_\textrm{h} \times k_\textrm{w}$ 윈도우를 `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ 의 행으로 펼쳐 $\textrm{vec}(\mathbf{K})$ 와 곱함.

<!--en:1fb18ec4 kind=slide_text-->
- **Trade-off:** Peak GPU Tensor Core speed at the cost of $k_\textrm{h} k_\textrm{w}\times$ patch workspace duplication.
<!--tr:1fb18ec4-->
- **트레이드오프:** $k_\textrm{h} k_\textrm{w}\times$ 패치 작업 공간 중복 비용을 내고 GPU 텐서 코어 최고 속도 확보.

<!--en:0d757cf6 kind=slide_title-->
::: {.slide title="Dual View: Unrolling the Kernel into a Banded Matrix"}
<!--tr:0d757cf6-->
::: {.slide title="쌍대 관점: 커널을 띠 행렬로 펼치기"}

<!--en:7f0defaa kind=kicker-->
[Toeplitz Weight Matrix Multiplication]{.kicker}
<!--tr:7f0defaa-->
[퇴플리츠(Toeplitz) 가중치 행렬 곱셈]{.kicker}

<!--en:5898ba57 kind=slide_text-->
Instead of unfolding the input image into duplicated patches, we can unroll the *kernel* into a sparse banded matrix $\mathbf{W}$ that multiplies the flattened input vector $\textrm{vec}(\mathbf{X})$:
<!--tr:5898ba57-->
입력 이미지를 중복 패치로 펼치는 대신, *커널*을 희소 띠 행렬 $\mathbf{W}$ 로 펼쳐서 평탄화된 입력 벡터 $\textrm{vec}(\mathbf{X})$ 에 곱할 수 있음:

<!--en:dae51741 kind=slide_text-->
- **Matrix–vector form:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, where $\mathbf{W}$ has $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ rows and $n_\textrm{h} n_\textrm{w}$ columns.
<!--tr:dae51741-->
- **행렬–벡터 형태:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, 여기서 $\mathbf{W}$ 는 $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ 행과 $n_\textrm{h} n_\textrm{w}$ 열을 가짐.

<!--en:f0e86ee3 kind=slide_text-->
- **Connection to §{sec_why-conv} and backprop:** Every row of $\mathbf{W}$ holds the same $k_\textrm{h} k_\textrm{w}$ kernel entries shifted across columns (panel c), making the backward pass with respect to $\mathbf{X}$ simply multiplication by $\mathbf{W}^\top$ (a transposed convolution).
<!--tr:f0e86ee3-->
- **§{sec_why-conv} 및 역전파와의 연결:** $\mathbf{W}$ 의 각 행은 동일한 $k_\textrm{h} k_\textrm{w}$ 개 커널 원소를 열 방향으로 이동해 배치하므로(패널 c), $\mathbf{X}$ 에 대한 역방향 패스는 단순히 $\mathbf{W}^\top$ 곱셈(전치 합성곱)이 됨.

<!--en:af8cf19a kind=slide_title-->
::: {.slide title="Feature Maps and Receptive Fields Across Depth"}
<!--tr:af8cf19a-->
::: {.slide title="깊이에 따른 특징 맵과 수용 영역"}

<!--en:4eb965f3 kind=kicker-->
[Linear Receptive-Field Growth in Stacked Layers]{.kicker}
<!--tr:4eb965f3-->
[층을 쌓을 때의 선형적 수용 영역 성장]{.kicker}

<!--en:5599a6e9 kind=slide_text-->
A unit's **receptive field** is the region of the original input image whose pixels can influence that unit's activation during forward propagation:
<!--tr:5599a6e9-->
유닛의 **수용 영역**은 순방향 패스 중 해당 유닛의 활성화에 영향을 줄 수 있는 원본 입력 이미지의 영역임:

<!--en:eba74f75 kind=slide_text-->
- **Single layer ($L=1$):** A $k_1 \times k_1$ kernel sees a $k_1 \times k_1$ input patch.
<!--tr:eba74f75-->
- **단일 층($L=1$):** $k_1 \times k_1$ 커널은 $k_1 \times k_1$ 입력 패치를 봄.

<!--en:01c175bf kind=slide_text-->
- **Stacking $L$ stride-1 layers:** Layer $i$ with kernel size $k_i$ reaches $k_i - 1$ pixels farther along each axis than a single element of layer $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$
<!--tr:01c175bf-->
- **스트라이드 1인 $L$ 개 층 쌓기:** 커널 크기 $k_i$ 인 $i$ 번째 층은 $i-1$ 번째 층의 단일 원소보다 각 축으로 $k_i - 1$ 픽셀 더 멀리 도달:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$

<!--en:2bde5822 kind=slide_text-->
- **$L$ stacked $3 \times 3$ layers:** Each layer adds $3 - 1 = 2$ pixels, giving side length $r = 2L + 1$ ($3 \to 5 \to 7$).
<!--tr:2bde5822-->
- **쌓인 $L$ 개의 $3 \times 3$ 층:** 각 층이 $3 - 1 = 2$ 픽셀을 더해 한 변 길이가 $r = 2L + 1$ 이 됨 ($3 \to 5 \to 7$).

<!--en:cbb9e97a kind=slide_title-->
::: {.slide title="Why Deep Stacks of 3×3 Kernels Beat One Large Kernel"}
<!--tr:cbb9e97a-->
::: {.slide title="3×3 커널을 깊게 쌓는 것이 큰 커널 하나보다 나은 이유"}

<!--en:95e46e70 kind=kicker-->
[Parameter Efficiency & Nonlinear Expressivity]{.kicker}
<!--tr:95e46e70-->
[매개변수 효율성 및 비선형 표현력]{.kicker}

<!--en:bd1d7cc0 kind=slide_text-->
- **Matched $7 \times 7$ receptive field:** Three stacked $3 \times 3$ layers ($r = 1 + 2 + 2 + 2 = 7$) see the exact same $7 \times 7$ patch as one $7 \times 7$ layer.
<!--tr:bd1d7cc0-->
- **동일한 $7 \times 7$ 수용 영역:** 세 개의 $3 \times 3$ 층($r = 1 + 2 + 2 + 2 = 7$)은 하나의 $7 \times 7$ 층과 정확히 같은 $7 \times 7$ 패치를 봄.

<!--en:755ac11a kind=slide_text-->
- **45% fewer parameters:** $3 \times 9 c^2 = 27 c^2$ weights instead of $49 c^2$ across $c$ channels.
<!--tr:755ac11a-->
- **매개변수 45% 절감:** $c$ 개 채널에 대해 $49 c^2$ 대신 $3 \times 9 c^2 = 27 c^2$ 개 가중치 사용.

<!--en:e3aa1086 kind=slide_text-->
- **Richer expressivity:** Compounds **3 nonlinear activations** instead of 1 (the core VGG design rule in §{sec_blocks}).
<!--tr:e3aa1086-->
- **더 풍부한 표현력:** 1개 대신 **3개의 비선형 활성화**를 중첩(§{sec_blocks} 의 핵심 VGG 설계 규칙).

<!--en:9ec83bf6 kind=slide_title-->
::: {.slide title="Theoretical vs. Effective Receptive Fields"}
<!--tr:9ec83bf6-->
::: {.slide title="이론적 수용 영역 vs. 유효 수용 영역"}

<!--en:72a52c3b kind=kicker-->
[Gaussian Decay of Sensitivity Away from the Center]{.kicker}
<!--tr:72a52c3b-->
[중심에서 멀어질수록 가우시안 형태로 감쇠하는 민감도]{.kicker}

<!--en:33bb377d kind=slide_text-->
- **Theoretical RF ($r = 2L+1$):** Bounding box of all pixels with *at least one* path to the output unit.
<!--tr:33bb377d-->
- **이론적 수용 영역($r = 2L+1$):** 출력 유닛으로 이어지는 경로가 *최소 하나* 있는 모든 픽셀의 경계 상자.

<!--en:293436eb kind=slide_text-->
- **Central Limit Theorem:** Center pixels connect via exponentially many paths, whereas corner pixels connect via only **1 path**.
<!--tr:293436eb-->
- **중심 극한 정리:** 중심 픽셀은 지수적으로 많은 경로로 연결되는 반면 모서리 픽셀은 단 **1개 경로**로만 연결.

<!--en:f47d181e kind=slide_text-->
- **Effective RF (Luo et al., 2016):** Input gradient magnitude $\partial y / \partial \mathbf{X}$ decays in a Gaussian bell curve with std $\sigma \propto \sqrt{L}$.
<!--tr:f47d181e-->
- **유효 수용 영역(Luo et al., 2016):** 입력 기울기 크기 $\partial y / \partial \mathbf{X}$ 는 표준 편차 $\sigma \propto \sqrt{L}$ 인 가우시안 종 모양 곡선으로 감쇠.

<!--en:fac792d0 kind=slide_title-->
::: {.slide title="Concept Check: Cross-Correlation & Receptive Fields"}
<!--tr:fac792d0-->
::: {.slide title="개념 확인: 교차 상관과 수용 영역"}

<!--en:26a317d8 kind=kicker-->
[Concept Check]{.kicker}
<!--tr:26a317d8-->
[개념 확인]{.kicker}

<!--en:8fe554e5 kind=slide_text-->
Test your understanding of output shapes, `im2col`, and receptive-field arithmetic:
<!--tr:8fe554e5-->
출력 모양, `im2col`, 수용 영역 계산에 대한 이해를 점검해 보세요:

<!--en:281c7c78 kind=slide_text-->
1. **im2col matrix dimensions:** You apply an unpadded $3 \times 3$ single-channel kernel to a $10 \times 10$ image via `im2col`. What is the output spatial shape, and what are the exact dimensions of the `patches` matrix and flattened kernel vector?
<!--tr:281c7c78-->
1. **im2col 행렬 차원:** 패딩 없는 $3 \times 3$ 단일 채널 커널을 $10 \times 10$ 이미지에 `im2col` 로 적용할 때, 출력 공간 모양과 `patches` 행렬 및 평탄화된 커널 벡터의 정확한 차원은?

<!--en:16976d18 kind=slide_text-->
2. **Depth to span Fashion-MNIST:** At stride 1, how many stacked $3 \times 3$ layers are required so a single unit's theoretical receptive field spans an entire $28 \times 28$ image? How many with $5 \times 5$ layers?
<!--tr:16976d18-->
2. **Fashion-MNIST 전체를 덮는 깊이:** 스트라이드 1에서 단일 유닛의 이론적 수용 영역이 $28 \times 28$ 이미지 전체를 포괄하려면 $3 \times 3$ 층을 몇 개 쌓아야 하는가? $5 \times 5$ 층이라면 몇 개인가?

<!--en:ab2e0d6a kind=slide_text-->
3. **Edge-loss boundary shift:** In our $6 \times 8$ vertical-edge learning experiment, why did the loss start at $\approx 12$, and what would the initial loss be if the black band extended all the way to the right edge (`X[:, 2:8] = 0`)?
<!--tr:ab2e0d6a-->
3. **경계선 손실 이동:** $6 \times 8$ 수직 경계선 학습 실험에서 초기 손실이 $\approx 12$ 인 이유는 무엇이며, 검은 띠가 오른쪽 끝까지 이어진다면(`X[:, 2:8] = 0`) 초기 손실은 얼마가 될까?

<!--en:250d6877 kind=slide_text-->
**Answers:**
(1) Output shape is $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ outputs). The `patches` matrix has shape $(64, 9)$ and multiplies the $(9, 1)$ kernel vector.
(2) For $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ layers. For $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ layers.
(3) Each of the 6 rows had two $\pm 1$ edges ($6 \times 2 \times 1^2 = 12$). Extending the black band to the right border eliminates the right-hand black-to-white edge, leaving one edge per row and halving the initial loss to $\approx 6$.
<!--tr:250d6877-->
**정답:**
(1) 출력 모양은 $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ 개 출력). `patches` 행렬은 $(64, 9)$ 모양이며 $(9, 1)$ 커널 벡터와 곱해짐.
(2) $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ 개 층. $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ 개 층.
(3) 6개 행마다 두 개의 $\pm 1$ 경계선이 있었음($6 \times 2 \times 1^2 = 12$). 검은 띠를 오른쪽 끝까지 늘리면 오른쪽 검은색→흰색 경계선이 사라져 행당 경계선이 1개만 남고 초기 손실이 $\approx 6$ 으로 절반이 됨.

<!--en:ba43a2dd kind=slide_title-->
::: {.slide title="Summary & Key Takeaways"}
<!--tr:ba43a2dd-->
::: {.slide title="요약 및 핵심 정리"}

<!--en:ce12e85f kind=kicker-->
[Section Synthesis]{.kicker}
<!--tr:ce12e85f-->
[절 요약]{.kicker}

<!--en:00011fb7 kind=slide_text-->
- **2D Cross-Correlation & Edge Learning:** Slides $\mathbf{K}$ across valid windows ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); SGD learns $[1, -1]$ in 9 steps.
<!--tr:00011fb7-->
- **2D 교차 상관과 경계선 학습:** 유효 윈도우($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$) 위로 $\mathbf{K}$ 를 이동; SGD가 9단계 만에 $[1, -1]$ 을 학습.

<!--en:09efe4f0 kind=slide_text-->
- **im2col Matrix Multiply:** Trades $k_\textrm{h} k_\textrm{w}\times$ patch duplication for peak GPU GEMM throughput.
<!--tr:09efe4f0-->
- **im2col 행렬 곱셈:** $k_\textrm{h} k_\textrm{w}\times$ 패치 중복을 감수하고 GPU GEMM 최고 처리량 달성.

<!--en:3af8acd0 kind=slide_text-->
- **Stacked Receptive Fields:** $r = 1 + \sum (k_i - 1) = 2L+1$ for $3 \times 3$ stacks, with Gaussian center sensitivity.
<!--tr:3af8acd0-->
- **층을 쌓은 수용 영역:** $3 \times 3$ 적층 시 $r = 1 + \sum (k_i - 1) = 2L+1$ 이며 중심에 집중된 가우시안 민감도를 보임.

=== DECISIONS ===
- Register: use polite `-ㅂ니다/습니다` style (`~합니다`, `~입니다`) in body prose, noun-phrase style without trailing periods in headings, and concise nominal/clause style (`~함`, `~임`) on slides.
- CJK spacing: place a space between `:numref:`/`:cite:`/`$...$` and adjacent Hangul particles/letters per TRANSLATING.md §4.
- First-use English parentheticals: include English in parentheses on first introduction of key concepts (e.g., `특징 맵(feature map)`, `수용 영역(receptive field)`).
- "edge" → 「경계선」 (consistent with u01).
- "banded matrix" → 「띠 행렬」 (standard Korean linear algebra term).
- "trade-off" → 「대가」 in body prose where natural, 「트레이드오프」 on slides for conciseness.

=== NOTES ===
- 24a25ec4, f95c9394, 01c175bf, 755ac11a, 16976d18, 00011fb7, 3af8acd0: math tokens reordered to match Korean SOV word order.
