<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u01 lang=vi rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:33d3b96b kind=heading-->
# Convolutions for Images
<!--tr:33d3b96b-->
# Tích chập cho ảnh

<!--en:852f5868 kind=paragraph-->
In :numref:`sec_why-conv`, we derived the convolutional layer on paper from two
geometric principles: translation equivariance and locality. Now we see what
that formula actually computes on an image, and whether gradient descent can
discover useful kernels automatically from data.
<!--tr:852f5868-->
Trong :numref:`sec_why-conv`, chúng ta đã suy ra tầng tích chập (convolutional layer) trên giấy từ hai nguyên lý hình học: tính đẳng biến tịnh tiến (translation equivariance) và tính cục bộ. Giờ đây chúng ta sẽ xem công thức đó thực sự tính ra điều gì trên một bức ảnh, và liệu hạ gradient có thể tự động tìm ra các hạt nhân hữu ích từ dữ liệu hay không.

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
Hãy xét bài toán xác định các cạnh dọc, nơi một vùng sáng gặp một vùng tối. Lấy điểm ảnh (pixel) trừ đi điểm ảnh lân cận bên phải của nó ($[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$) cho kết quả bằng không bên trong các vùng đồng nhất, bằng $+1$ ở chỗ màu trắng chuyển sang đen, và bằng $-1$ ở chỗ màu đen quay lại trắng. Quét bộ dò hai trọng số $(1, -1)$ qua mọi vị trí hợp lệ của ảnh sẽ tạo ra một *bản đồ cạnh* (edge map). Trong phần này, chúng ta cài đặt tương quan chéo (cross-correlation) hai chiều, kiểm tra bộ dò cạnh tự thiết kế, để hạ gradient *học* $(1, -1)$ trực tiếp từ dữ liệu, chỉ ra cách một phép tích chập trở thành một phép nhân ma trận duy nhất (*im2col*), và suy ra việc xếp chồng các tầng làm rộng vùng tiếp nhận (receptive field) của một đơn vị như thế nào.

<!--en:47958ed8 kind=heading-->
## Cross-Correlation
<!--tr:47958ed8-->
## Tương quan chéo

<!--en:5e4296fe kind=heading-->
### The Operation
<!--tr:5e4296fe-->
### Phép toán

<!--en:ae7e906d kind=paragraph-->
Leaving channels aside for the moment, let the input $\mathbf{X}$ be an
$n_\textrm{h} \times n_\textrm{w}$ matrix and the kernel $\mathbf{K}$ be a
$k_\textrm{h} \times k_\textrm{w}$ matrix. The kernel defines a sliding
*window* of shape $k_\textrm{h} \times k_\textrm{w}$. Placing the window's
upper-left corner at $(i, j)$, we multiply the covered input patch elementwise
with $\mathbf{K}$ and sum the products to produce the output entry at $(i, j)$:
<!--tr:ae7e906d-->
Tạm thời bỏ qua các kênh, giả sử đầu vào $\mathbf{X}$ là một ma trận $n_\textrm{h} \times n_\textrm{w}$ và hạt nhân $\mathbf{K}$ là một ma trận $k_\textrm{h} \times k_\textrm{w}$. Hạt nhân xác định một *cửa sổ* (window) trượt có kích thước $k_\textrm{h} \times k_\textrm{w}$. Đặt góc trên bên trái của cửa sổ tại $(i, j)$, chúng ta nhân từng phần tử của mảnh đầu vào được cửa sổ phủ lên với $\mathbf{K}$, rồi cộng các tích để được phần tử đầu ra tại $(i, j)$:

<!--en:3688b38b kind=paragraph-->
This is the *two-dimensional cross-correlation* of $\mathbf{X}$ with
$\mathbf{K}$, written $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ with
zero-based indices matching code. Whereas :numref:`sec_why-conv` centered
offsets symmetrically from $-\Delta$ to $\Delta$, anchoring at the upper-left
corner differs only by a fixed shift of $\Delta$ while naturally handling
even-sized kernels (such as a $2 \times 2$ kernel or our $1 \times 2$ edge
detector) that have no center pixel.
<!--tr:3688b38b-->
Đây là *tương quan chéo hai chiều* của $\mathbf{X}$ với $\mathbf{K}$, ký hiệu $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$, với chỉ số bắt đầu từ không để khớp với mã nguồn. Trong khi :numref:`sec_why-conv` lấy các độ lệch đối xứng từ $-\Delta$ đến $\Delta$, việc neo ở góc trên bên trái chỉ khác nhau một độ dịch cố định $\Delta$, đồng thời xử lý tự nhiên các hạt nhân có kích thước chẵn (như hạt nhân $2 \times 2$ hay bộ dò cạnh $1 \times 2$ của chúng ta) vốn không có điểm ảnh ở tâm.

<!--en:fa1b7da9 kind=caption-->
![Two-dimensional cross-correlation. The shaded portions are the first output element and the input and kernel elements used to compute it: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
<!--tr:fa1b7da9-->
![Tương quan chéo hai chiều. Các phần được tô bóng là phần tử đầu ra đầu tiên cùng các phần tử của đầu vào và hạt nhân dùng để tính nó: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)

<!--en:2f8e5deb kind=paragraph-->
In :numref:`fig_correlation`, we slide a $2 \times 2$ kernel across a
$3 \times 3$ input one pixel at a time, left to right and top to bottom. The
window fits cleanly at four positions, yielding the $2 \times 2$ output:
<!--tr:2f8e5deb-->
Trong :numref:`fig_correlation`, chúng ta trượt một hạt nhân $2 \times 2$ qua một đầu vào $3 \times 3$ từng điểm ảnh một, từ trái sang phải và từ trên xuống dưới. Cửa sổ nằm gọn trong đầu vào ở bốn vị trí, cho ra đầu ra $2 \times 2$:

<!--en:4741e012 kind=paragraph-->
Notice that the output shrinks slightly along each axis because the kernel must
stay wholly inside the input. Fitting a width-$k$ window into length $n$ leaves
$n - k + 1$ valid positions, so the output shape is
<!--tr:4741e012-->
Lưu ý rằng đầu ra co lại đôi chút dọc theo mỗi trục vì hạt nhân phải nằm hoàn toàn bên trong đầu vào. Đặt một cửa sổ rộng $k$ vào một đoạn dài $n$ thì có $n - k + 1$ vị trí hợp lệ, do đó kích thước của đầu ra là

<!--en:12d81c14 kind=paragraph-->
Thus a $3 \times 3$ input with a $2 \times 2$ kernel yields $2 \times 2$, a
$6 \times 8$ image with a $1 \times 2$ kernel yields $6 \times 7$, and a
$28 \times 28$ Fashion-MNIST image with a $5 \times 5$ kernel yields
$24 \times 24$. Whenever we want to preserve spatial dimensions, we surround the
input with a border of zeros (*padding*), which we examine alongside strides and
dilation in :numref:`sec_padding`.
<!--tr:12d81c14-->
Như vậy, đầu vào $3 \times 3$ với hạt nhân $2 \times 2$ cho đầu ra $2 \times 2$, ảnh $6 \times 8$ với hạt nhân $1 \times 2$ cho đầu ra $6 \times 7$, và ảnh Fashion-MNIST $28 \times 28$ với hạt nhân $5 \times 5$ cho đầu ra $24 \times 24$. Mỗi khi muốn giữ nguyên các chiều không gian, chúng ta bao quanh đầu vào bằng một viền các số không (*đệm*, padding); chúng ta sẽ xem xét kỹ điều này cùng với sải bước (stride) và độ giãn (dilation) trong :numref:`sec_padding`.

<!--en:869d223f kind=heading-->
### Implementation
<!--tr:869d223f-->
### Cài đặt

<!--en:6a869f7c kind=paragraph-->
We implement :eqref:`eq_corr2d` directly in `corr2d` using a pair of loops over
every output coordinate.
<!--tr:6a869f7c-->
Chúng ta cài đặt trực tiếp :eqref:`eq_corr2d` trong hàm `corr2d` bằng một cặp vòng lặp duyệt qua mọi tọa độ đầu ra.

<!--en:81e1f383 kind=paragraph-->
To verify `corr2d` against our hand calculation in :numref:`fig_correlation`,
we construct the $3 \times 3$ input tensor `X` and $2 \times 2$ kernel `K` and
evaluate `corr2d(X, K)`.
<!--tr:81e1f383-->
Để kiểm tra `corr2d` so với phép tính tay của chúng ta trong :numref:`fig_correlation`, chúng ta tạo tensor đầu vào `X` kích thước $3 \times 3$ và hạt nhân `K` kích thước $2 \times 2$, rồi tính `corr2d(X, K)`.

<!--en:f567e1b2 kind=paragraph-->
The output reproduces :numref:`fig_correlation` entry for entry (and as shown in
:numref:`sec_why-conv`, strict mathematical convolution with a flipped kernel
$\mathbf{K}'$ learns the exact same family of layers, so we follow standard
deep learning terminology and call cross-correlation "convolution").
<!--tr:f567e1b2-->
Đầu ra tái tạo :numref:`fig_correlation` từng phần tử một (và như đã chỉ ra trong :numref:`sec_why-conv`, phép tích chập toán học thực sự với hạt nhân đã lật $\mathbf{K}'$ học đúng cùng một họ các tầng, nên chúng ta theo thuật ngữ chuẩn của học sâu và gọi tương quan chéo là "tích chập").

<!--en:bfe11c62 kind=heading-->
## Convolutional Layers
<!--tr:bfe11c62-->
## Tầng tích chập

<!--en:2cdc020d kind=heading-->
### The Layer
<!--tr:2cdc020d-->
### Tầng tích chập

<!--en:c6371ece kind=paragraph-->
A two-dimensional convolutional layer cross-correlates its input with a learned
kernel and adds a scalar bias. Just like our MLP weights in
:numref:`chap_perceptrons`, we initialize the kernel from
$\mathcal{N}(0, 0.01^2)$ and the bias to zero. Below, `Conv2D` registers both
parameters and invokes `corr2d` in its forward pass; a layer with a kernel of
height $h$ and width $w$ is called an $h \times w$ convolutional layer.
<!--tr:c6371ece-->
Một tầng tích chập hai chiều thực hiện tương quan chéo giữa đầu vào với một hạt nhân đã học, rồi cộng thêm một hệ số điều chỉnh (bias) vô hướng. Giống như các trọng số của perceptron đa tầng (MLP) trong :numref:`chap_perceptrons`, chúng ta khởi tạo hạt nhân từ $\mathcal{N}(0, 0.01^2)$ và hệ số điều chỉnh bằng không. Dưới đây, `Conv2D` đăng ký cả hai tham số và gọi `corr2d` trong lượt truyền xuôi của nó; một tầng có hạt nhân cao $h$ và rộng $w$ được gọi là tầng tích chập $h \times w$.

<!--en:29cd9061 kind=paragraph-->
Unlike a dense layer, a convolutional layer is not locked to a fixed input
resolution because its parameters depend only on the kernel size. The exact same
$2 \times 2$ layer (5 parameters total) processes both our $3 \times 3$ tensor
`X` and a $6 \times 8$ image, returning outputs of shapes $2 \times 2$ and
$5 \times 7$:
<!--tr:29cd9061-->
Khác với tầng dày đặc, tầng tích chập không bị khóa vào một độ phân giải đầu vào cố định vì các tham số của nó chỉ phụ thuộc vào kích thước hạt nhân. Chính tầng $2 \times 2$ đó (tổng cộng 5 tham số) xử lý được cả tensor $3 \times 3$ `X` lẫn một ảnh $6 \times 8$, trả về các đầu ra có kích thước $2 \times 2$ và $5 \times 7$:

<!--en:c9e5aa07 kind=heading-->
### Edge Detection
<!--tr:c9e5aa07-->
### Dò cạnh

<!--en:3d9a357b kind=paragraph-->
We now test the vertical edge detector from the introduction on a
$6 \times 8$ synthetic image whose middle four columns are black (`0`) and outer
columns are white (`1`).
<!--tr:3d9a357b-->
Bây giờ chúng ta thử bộ dò cạnh dọc từ phần mở đầu trên một ảnh tổng hợp $6 \times 8$, trong đó bốn cột giữa màu đen (`0`) và các cột ngoài cùng màu trắng (`1`).

<!--en:68513ff0 kind=paragraph-->
We define the $1 \times 2$ kernel
<!--tr:68513ff0-->
Chúng ta định nghĩa hạt nhân $1 \times 2$

<!--en:eb45803b kind=paragraph-->
At every position $(i, j)$, this computes the horizontal *finite difference*
$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, a discrete approximation (up to sign)
of the horizontal derivative of brightness, which vanishes across flat regions
and spikes wherever intensity jumps.
<!--tr:eb45803b-->
Tại mỗi vị trí $(i, j)$, hạt nhân này tính *sai phân hữu hạn* (finite difference) theo phương ngang $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, một phép xấp xỉ rời rạc (sai khác một dấu) của đạo hàm theo phương ngang của độ sáng; giá trị này triệt tiêu trên các vùng phẳng và vọt lên ở bất cứ nơi nào cường độ nhảy bậc.

<!--en:e810b3a9 kind=paragraph-->
Cross-correlating the $6 \times 8$ stripe image `X` with the $1 \times 2$
horizontal-difference kernel `K` via `corr2d(X, K)` produces the $6 \times 7$
edge map `Y`:
<!--tr:e810b3a9-->
Thực hiện tương quan chéo giữa ảnh sọc `X` kích thước $6 \times 8$ và hạt nhân sai phân ngang `K` kích thước $1 \times 2$ qua `corr2d(X, K)` cho ra bản đồ cạnh `Y` kích thước $6 \times 7$:

<!--en:568358e4 kind=paragraph-->
Sure enough, `Y` fires `+1` along the white-to-black boundary, `-1` along the
black-to-white boundary, and `0` everywhere else. Applying the exact same kernel
to the transposed image (turning the vertical stripe into a horizontal one)
yields all zeros:
<!--tr:568358e4-->
Đúng như mong đợi, `Y` nhận giá trị `+1` dọc theo ranh giới trắng sang đen, `-1` dọc theo ranh giới đen sang trắng, và `0` ở mọi chỗ còn lại. Áp dụng chính hạt nhân đó lên ảnh chuyển vị (biến sọc dọc thành sọc ngang) cho kết quả toàn số không:

<!--en:71c591a4 kind=paragraph-->
Clearly `[1, -1]` responds strictly to vertical edges. Detecting horizontal
edges, diagonals, corners, or textures requires different kernels altogether.
<!--tr:71c591a4-->
Rõ ràng `[1, -1]` chỉ đáp ứng với các cạnh dọc. Để phát hiện cạnh ngang, đường chéo, góc hay kết cấu bề mặt thì cần những hạt nhân hoàn toàn khác.

<!--en:f0af4887 kind=heading-->
### Learning a Kernel
<!--tr:f0af4887-->
### Học một hạt nhân

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
Tự thiết kế `[1, -1]` thì dễ khi chúng ta biết chính xác đặc trưng mình muốn, nhưng không kỹ sư nào có thể tự tay chỉ định hàng nghìn bộ lọc trung gian trong một mạng sâu. Liệu hạ gradient có thể khôi phục hạt nhân `[1, -1]` chỉ từ cặp đầu vào–đầu ra `(X, Y)` hay không? Chúng ta thử bằng tầng tích chập 2D có sẵn của mỗi framework (bỏ hệ số điều chỉnh và khởi tạo trọng số từ $\mathcal{N}(0, 0.01^2)$), cực tiểu hóa sai số bình phương cộng trên cả 42 phần tử đầu ra qua 10 bước gradient với tốc độ học $0.03$. Vì các tầng có sẵn đòi hỏi tensor 4D gồm cả trục batch và trục kênh (:numref:`sec_channels`), chúng ta định hình lại `X` và `Y` cho phù hợp.

<!--en:c29c5d61 kind=paragraph-->
Initially the loss sits near 12 (the near-zero random kernel misses all 12
$\pm 1$ edge pixels by roughly 1), and within just nine steps it drops below
0.01. Inspecting the learned kernel confirms what it found:
<!--tr:c29c5d61-->
Ban đầu mất mát ở mức gần 12 (hạt nhân ngẫu nhiên gần bằng không bỏ sót cả 12 điểm ảnh cạnh $\pm 1$, mỗi điểm lệch khoảng 1), và chỉ sau chín bước nó đã giảm xuống dưới 0.01. Kiểm tra hạt nhân đã học sẽ xác nhận điều nó tìm được:

<!--en:67d25b0b kind=paragraph-->
Both entries land within roughly 0.02 of `[1, -1]`. Nothing in this loop knew
anything about edges: given a loss signal, gradient descent discovered the
filter on its own. Here we supervised the single layer directly with `Y`; in a
deep CNN, backpropagation learns every layer's kernels end to end from the final
task loss alone.
<!--tr:67d25b0b-->
Cả hai phần tử đều nằm trong khoảng 0.02 quanh `[1, -1]`. Không có gì trong vòng lặp này biết gì về cạnh: chỉ với một tín hiệu mất mát, hạ gradient đã tự khám phá ra bộ lọc. Ở đây chúng ta giám sát trực tiếp tầng đơn bằng `Y`; trong một mạng nơ-ron tích chập (CNN) sâu, lan truyền ngược học hạt nhân của mọi tầng từ đầu đến cuối, chỉ từ mất mát của tác vụ cuối cùng.

<!--en:91b0010d kind=slide_text-->
Convolutions for Images<br>**2D cross-correlation · edge detection · learned kernels · im2col · receptive fields**
<!--tr:91b0010d-->
Tích chập cho ảnh<br>**tương quan chéo 2D · dò cạnh · hạt nhân được học · im2col · vùng tiếp nhận**

<!--en:1d94295a kind=slide_title-->
::: {.slide title="Detecting Local Edges Across an Image"}
<!--tr:1d94295a-->
::: {.slide title="Dò các cạnh cục bộ trên ảnh"}

<!--en:687ed302 kind=kicker-->
[Motivation & Core Mental Model]{.kicker}
<!--tr:687ed302-->
[Động lực & mô hình tư duy cốt lõi]{.kicker}

<!--en:05fceed0 kind=slide_text-->
- **Synthetic $6 \times 8$ test image:** White outer columns (`1`) surrounding a 4-column black vertical band (`0`).
<!--tr:05fceed0-->
- **Ảnh kiểm tra tổng hợp $6 \times 8$:** Các cột ngoài màu trắng (`1`) bao quanh một dải dọc màu đen rộng 4 cột (`0`).

<!--en:71e700b3 kind=slide_text-->
- **Horizontal finite difference:** Sweeping $\mathbf{K} = [1, -1]$ computes $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, firing `+1` at white-to-black and `-1` at black-to-white edges.
<!--tr:71e700b3-->
- **Sai phân hữu hạn theo phương ngang:** Quét $\mathbf{K} = [1, -1]$ tính $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, cho `+1` ở cạnh trắng sang đen và `-1` ở cạnh đen sang trắng.

<!--en:957dc033 kind=slide_text-->
- **Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent.
<!--tr:957dc033-->
- **Học so với tự thiết kế:** Các mạng nơ-ron tích chập (CNN) sâu tự động khám phá hàng nghìn hạt nhân như vậy bằng hạ gradient.

<!--en:cb1045e1 kind=slide_title-->
::: {.slide title="The 2D Cross-Correlation Operation"}
<!--tr:cb1045e1-->
::: {.slide title="Phép tương quan chéo 2D"}

<!--en:17bcece5 kind=kicker-->
[Sliding-Window Inner Products]{.kicker}
<!--tr:17bcece5-->
[Tích trong của cửa sổ trượt]{.kicker}

<!--en:8d45aae8 kind=slide_text-->
Given an $n_\textrm{h} \times n_\textrm{w}$ input matrix $\mathbf{X}$ and a $k_\textrm{h} \times k_\textrm{w}$ kernel $\mathbf{K}$, the 2D cross-correlation $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ places the window's top-left corner at $(i, j)$ and sums elementwise products:
<!--tr:8d45aae8-->
Cho ma trận đầu vào $\mathbf{X}$ kích thước $n_\textrm{h} \times n_\textrm{w}$ và hạt nhân $\mathbf{K}$ kích thước $k_\textrm{h} \times k_\textrm{w}$, tương quan chéo 2D $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ đặt góc trên bên trái của cửa sổ tại $(i, j)$ và cộng các tích từng phần tử:

<!--en:e484380c kind=slide_text-->
- **Top-left anchor:** Zero-based indices $a, b \ge 0$ match array indexing in code and handle even-sized kernels (like $2 \times 2$ or $1 \times 2$) cleanly.
<!--tr:e484380c-->
- **Neo ở góc trên bên trái:** Chỉ số bắt đầu từ không $a, b \ge 0$ khớp với cách đánh chỉ số mảng trong mã nguồn và xử lý gọn các hạt nhân kích thước chẵn (như $2 \times 2$ hay $1 \times 2$).

<!--en:137c0980 kind=slide_text-->
- **Shaded top-left output $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.
<!--tr:137c0980-->
- **Đầu ra $(0,0)$ được tô bóng:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.

<!--en:575c4aea kind=slide_title-->
::: {.slide title="Valid Window Positions and Output Spatial Shape"}
<!--tr:575c4aea-->
::: {.slide title="Vị trí cửa sổ hợp lệ và kích thước đầu ra"}

<!--en:7eda4531 kind=kicker-->
[Why Unpadded Convolutions Shrink the Grid]{.kicker}
<!--tr:7eda4531-->
[Vì sao tích chập không đệm làm co lưới]{.kicker}

<!--en:891243ae kind=slide_text-->
- **Valid window placements:** A $k_\textrm{h} \times k_\textrm{w}$ window fits inside an $n_\textrm{h} \times n_\textrm{w}$ grid at $n - k + 1$ positions per axis.
<!--tr:891243ae-->
- **Các vị trí đặt cửa sổ hợp lệ:** Một cửa sổ $k_\textrm{h} \times k_\textrm{w}$ nằm gọn trong lưới $n_\textrm{h} \times n_\textrm{w}$ tại $n - k + 1$ vị trí trên mỗi trục.

<!--en:55cc8c32 kind=slide_text-->
- **Shape examples:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.
<!--tr:55cc8c32-->
- **Ví dụ về kích thước:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.

<!--en:1e3463da kind=slide_title-->
::: {.slide title="Implementing 2D Cross-Correlation in Code"}
<!--tr:1e3463da-->
::: {.slide title="Cài đặt tương quan chéo 2D bằng mã"}

<!--en:07566992 kind=kicker-->
[Direct Nested-Loop Implementation]{.kicker}
<!--tr:07566992-->
[Cài đặt trực tiếp bằng vòng lặp lồng nhau]{.kicker}

<!--en:202aa5d3 kind=slide_text-->
The function `corr2d` allocates the $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ output tensor and computes each entry as the sum of `X[i:i+h, j:j+w] * K`:
<!--tr:202aa5d3-->
Hàm `corr2d` cấp phát tensor đầu ra kích thước $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ và tính mỗi phần tử bằng tổng của `X[i:i+h, j:j+w] * K`:

<!--en:bef130ba kind=slide_text-->
Each iteration slices a $k_\textrm{h} \times k_\textrm{w}$ sub-window of `X`, multiplies elementwise with `K`, and reduces via `sum`.
<!--tr:bef130ba-->
Mỗi lần lặp cắt một cửa sổ con $k_\textrm{h} \times k_\textrm{w}$ của `X`, nhân từng phần tử với `K`, rồi rút gọn bằng `sum`.

<!--en:4ee9f6b1 kind=slide_title-->
::: {.slide title="Verifying corr2d and the Flipped-Kernel Equivalence"}
<!--tr:4ee9f6b1-->
::: {.slide title="Kiểm tra corr2d và sự tương đương với hạt nhân lật"}

<!--en:2b0be423 kind=kicker-->
[Cross-Correlation vs. Mathematical Convolution]{.kicker}
<!--tr:2b0be423-->
[Tương quan chéo so với tích chập toán học]{.kicker}

<!--en:74b9b745 kind=slide_text-->
- **180° flipped kernel $\mathbf{K}'$:** Cross-correlating with $\mathbf{K}$ equals convolving with $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.
<!--tr:74b9b745-->
- **Hạt nhân $\mathbf{K}'$ lật 180°:** Tương quan chéo với $\mathbf{K}$ bằng tích chập với $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.

<!--en:5c8af7cd kind=slide_text-->
- **Identical learned class:** Frameworks compute cross-correlation and call it convolution.
<!--tr:5c8af7cd-->
- **Cùng một lớp hàm học được:** Các framework tính tương quan chéo và gọi nó là tích chập.

<!--en:a69e29b5 kind=slide_title-->
::: {.slide title="Building a 2D Convolutional Layer Module"}
<!--tr:a69e29b5-->
::: {.slide title="Xây dựng một mô-đun tầng tích chập 2D"}

<!--en:55763603 kind=kicker-->
[Learnable Kernel Weights Plus Scalar Bias]{.kicker}
<!--tr:55763603-->
[Trọng số hạt nhân học được cộng hệ số điều chỉnh vô hướng]{.kicker}

<!--en:5b8bad24 kind=slide_text-->
A convolutional layer wraps `corr2d` with a learnable $k_\textrm{h} \times k_\textrm{w}$ weight parameter (initialized from $\mathcal{N}(0, 0.01^2)$) and a scalar bias (initialized to $0$):
<!--tr:5b8bad24-->
Một tầng tích chập bọc `corr2d` với một tham số trọng số học được kích thước $k_\textrm{h} \times k_\textrm{w}$ (khởi tạo từ $\mathcal{N}(0, 0.01^2)$) và một hệ số điều chỉnh vô hướng (khởi tạo bằng $0$):

<!--en:c16d97d6 kind=slide_title-->
::: {.slide title="Resolution Independence of Convolutional Layers"}
<!--tr:c16d97d6-->
::: {.slide title="Tầng tích chập không phụ thuộc độ phân giải"}

<!--en:a7d1e48f kind=kicker-->
[Fixed Parameters Across Arbitrary Input Sizes]{.kicker}
<!--tr:a7d1e48f-->
[Số tham số cố định với mọi kích thước đầu vào]{.kicker}

<!--en:53a2bce7 kind=slide_text-->
- **Decoupled from $H \times W$:** The exact same 5 parameters ($2 \times 2$ kernel + 1 bias) process $3 \times 3$, $6 \times 8$, or megapixel inputs without reshaping weights.
<!--tr:53a2bce7-->
- **Tách khỏi $H \times W$:** Chính 5 tham số đó ($2 \times 2$ hạt nhân + 1 hệ số điều chỉnh) xử lý đầu vào $3 \times 3$, $6 \times 8$ hay cỡ megapixel mà không cần định hình lại trọng số.

<!--en:017fec24 kind=slide_title-->
::: {.slide title="Constructing a Synthetic Vertical-Edge Image"}
<!--tr:017fec24-->
::: {.slide title="Tạo ảnh tổng hợp có cạnh dọc"}

<!--en:896e641c kind=kicker-->
[A 6×8 Binary Test Pattern]{.kicker}
<!--tr:896e641c-->
[Mẫu kiểm tra nhị phân 6×8]{.kicker}

<!--en:0a4b456b kind=slide_text-->
We construct the $6 \times 8$ test image `X` with ones on the left and right borders and zeros across the middle four columns (`X[:, 2:6] = 0`):
<!--tr:0a4b456b-->
Chúng ta tạo ảnh kiểm tra `X` kích thước $6 \times 8$ với các số một ở hai cột biên trái và phải và các số không ở bốn cột giữa (`X[:, 2:6] = 0`):

<!--en:dfc7e3e3 kind=slide_text-->
- **Piecewise-constant columns:** Intensity is constant (`1` or `0`) within columns $0\text{--}1$, $2\text{--}5$, and $6\text{--}7$, jumping only between columns $1 \to 2$ and $5 \to 6$.
<!--tr:dfc7e3e3-->
- **Các cột hằng từng đoạn:** Cường độ không đổi (`1` hoặc `0`) trong các cột $0\text{--}1$, $2\text{--}5$ và $6\text{--}7$, chỉ nhảy bậc giữa các cột $1 \to 2$ và $5 \to 6$.

<!--en:3ce31cbe kind=slide_title-->
::: {.slide title="Applying the Horizontal Finite-Difference Kernel"}
<!--tr:3ce31cbe-->
::: {.slide title="Áp dụng hạt nhân sai phân hữu hạn theo phương ngang"}

<!--en:b37a1029 kind=kicker-->
[Detecting Vertical Edges with K = [1, -1]]{.kicker}
<!--tr:b37a1029-->
[Dò cạnh dọc bằng K = [1, -1]]{.kicker}

<!--en:9be85019 kind=slide_text-->
Cross-correlating `X` with the $1 \times 2$ kernel $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ computes $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ at every horizontal pair:
<!--tr:9be85019-->
Tương quan chéo `X` với hạt nhân $1 \times 2$ $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ tính $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ tại mọi cặp điểm ảnh liền kề theo chiều ngang:

<!--en:f649cfae kind=slide_text-->
- **Signed edge map:** Column 1 fires `+1` (white-to-black transition), column 5 fires `-1` (black-to-white transition), and all constant interior columns evaluate to `0`.
<!--tr:f649cfae-->
- **Bản đồ cạnh có dấu:** Cột 1 cho `+1` (chuyển từ trắng sang đen), cột 5 cho `-1` (chuyển từ đen sang trắng), và mọi cột bên trong có giá trị hằng đều cho `0`.

<!--en:e6d155be kind=slide_title-->
::: {.slide title="Orientation Selectivity of Edge Kernels"}
<!--tr:e6d155be-->
::: {.slide title="Tính chọn lọc hướng của hạt nhân dò cạnh"}

<!--en:b525cde4 kind=kicker-->
[Why One Kernel Cannot Detect All Orientations]{.kicker}
<!--tr:b525cde4-->
[Vì sao một hạt nhân không thể dò mọi hướng]{.kicker}

<!--en:38159de6 kind=slide_text-->
- **Blind to horizontal edges:** Transposing `X` (`X.T`) makes horizontal neighbors identical, so $[1, -1]$ outputs all zeros.
<!--tr:38159de6-->
- **Mù với cạnh ngang:** Chuyển vị `X` (`X.T`) làm các điểm ảnh lân cận theo chiều ngang giống hệt nhau, nên $[1, -1]$ cho đầu ra toàn số không.

<!--en:f0f30f76 kind=slide_text-->
- **Filter banks (§{sec_channels}):** Vision layers learn parallel channels for horizontal, vertical, and diagonal orientations.
<!--tr:f0f30f76-->
- **Bộ lọc nhiều kênh (§{sec_channels}):** Các tầng thị giác học các kênh song song cho hướng ngang, dọc và chéo.

<!--en:b755e31b kind=slide_title-->
::: {.slide title="Learning an Edge Detector via Gradient Descent"}
<!--tr:b755e31b-->
::: {.slide title="Học bộ dò cạnh bằng hạ gradient"}

<!--en:86bbeea7 kind=kicker-->
[Supervising a Built-In Convolutional Layer on (X, Y)]{.kicker}
<!--tr:86bbeea7-->
[Giám sát một tầng tích chập có sẵn trên (X, Y)]{.kicker}

<!--en:a569780c kind=slide_text-->
- **4D framework tensors:** Reshape `X` and `Y` to 4D tensors (one example, one channel) and minimize $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ at $\eta = 0.03$.
<!--tr:a569780c-->
- **Tensor 4D của framework:** Định hình lại `X` và `Y` thành tensor 4D (một mẫu, một kênh) và cực tiểu hóa $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ với $\eta = 0.03$.

<!--en:2489ce6e kind=slide_title-->
::: {.slide title="Inspecting the Learned Kernel Weights"}
<!--tr:2489ce6e-->
::: {.slide title="Kiểm tra trọng số hạt nhân đã học"}

<!--en:7e4ef44b kind=kicker-->
[Why Initial Loss Is 12 and Converges to [1, -1]]{.kicker}
<!--tr:7e4ef44b-->
[Vì sao mất mát ban đầu là 12 và hội tụ về [1, -1]]{.kicker}

<!--en:9cd1c645 kind=slide_text-->
- **Why loss starts at $12.0$:** Near-zero weights predict $\hat{Y} \approx 0$, missing the twelve $\pm 1$ edge pixels in `Y` ($12 \times 1^2 = 12$).
<!--tr:9cd1c645-->
- **Vì sao mất mát bắt đầu từ $12.0$:** Trọng số gần bằng không dự đoán $\hat{Y} \approx 0$, bỏ sót mười hai điểm ảnh cạnh $\pm 1$ trong `Y` ($12 \times 1^2 = 12$).

<!--en:456670c5 kind=slide_text-->
- **Convergence in 9 steps:** SGD recovers $[0.99, -0.99]$ automatically from data.
<!--tr:456670c5-->
- **Hội tụ sau 9 bước:** hạ gradient ngẫu nhiên (SGD) tự động khôi phục $[0.99, -0.99]$ từ dữ liệu.

=== DECISIONS ===
- Register: "chúng ta" for the authorial "we"; neutral, no "bạn" except in imperative exercises.
- On first use in a unit the English term follows in parentheses, e.g. tầng tích chập (convolutional layer); later uses are Vietnamese only.
- "cross-correlation" → tương quan chéo; "convolution" → tích chập; "edge map" → bản đồ cạnh; "window" (sliding) → cửa sổ; "patch" → mảnh (patch matrix → ma trận mảnh); "im2col" stays English in italics.
- "dense layer" → tầng dày đặc; "unit" (of a network) → đơn vị; "synthetic image" → ảnh tổng hợp.
- "framework" stays "framework"; "batch" axis stays "batch" (glossary proposed term); "mini-batch" → minibatch.
- Glossary "shape" → kích thước is used for tensor shapes; "size" along an axis → cỡ / chiều.
- Numbers keep a decimal point as in the English (0.01, 0.03), not a comma.
- Slide kicker/title: sentence case, no trailing period.

=== NOTES ===
- 61cacfaf: "built-in 2D convolutional layer" is rendered as "tầng tích chập 2D có sẵn".
- 5e4296fe: heading "The Operation" rendered as "Phép toán".
- Remaining warnings about "scalar" (số vô hướng) are false matches: "scalar bias" is rendered "hệ số điều chỉnh vô hướng", where vô hướng is the adjective.
- 8d45aae8 (u01), 01c175bf (u02), ec8621f8 (u03): "math tokens reordered" reflects Vietnamese word order only; no math token was changed.
