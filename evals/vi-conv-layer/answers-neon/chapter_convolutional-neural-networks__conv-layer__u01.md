<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u01 lang=vi rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:33d3b96b kind=heading-->
# Convolutions for Images
<!--tr:33d3b96b-->
# Phép tích chập cho ảnh

<!--en:852f5868 kind=paragraph-->
In :numref:`sec_why-conv`, we derived the convolutional layer on paper from two
geometric principles: translation equivariance and locality. Now we see what
that formula actually computes on an image, and whether gradient descent can
discover useful kernels automatically from data.
<!--tr:852f5868-->
Trong :numref:`sec_why-conv`, chúng ta đã suy ra tầng tích chập trên giấy từ hai
nguyên lý hình học: tính đẳng biến tịnh tiến và tính cục bộ. Bây giờ chúng ta sẽ
xem công thức đó thực sự tính toán điều gì trên một bức ảnh, và liệu hạ gradient
có thể tự động phát hiện ra các hạt nhân hữu ích từ dữ liệu hay không.

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
Hãy xét bài toán xác định các biên dọc nơi vùng sáng tiếp giáp với vùng tối.
Việc lấy mỗi điểm ảnh trừ đi điểm ảnh lân cận bên phải của nó
($[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$) cho kết quả bằng không bên trong
các vùng đồng nhất, $+1$ tại nơi màu trắng chuyển sang màu đen, và $-1$ tại nơi
màu đen chuyển lại sang màu trắng. Quét bộ phát hiện gồm hai trọng số $(1, -1)$
qua mọi vị trí hợp lệ của ảnh sẽ tạo ra một *bản đồ biên* (*edge map*). Trong
phần này, chúng ta sẽ lập trình phép tương quan chéo hai chiều, kiểm thử bộ phát
hiện biên thiết kế thủ công, để hạ gradient tự *học* $(1, -1)$ trực tiếp từ dữ
liệu, chỉ ra cách biến một phép tích chập thành một phép nhân ma trận duy nhất
(*im2col*), và suy ra cách việc xếp chồng nhiều tầng giúp mở rộng vùng tiếp nhận
của một đơn vị.

<!--en:47958ed8 kind=heading-->
## Cross-Correlation
<!--tr:47958ed8-->
## Phép tương quan chéo

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
Tạm gác các kênh sang một bên, giả sử đầu vào $\mathbf{X}$ là một ma trận kích
thước $n_\textrm{h} \times n_\textrm{w}$ và hạt nhân $\mathbf{K}$ là một ma trận
kích thước $k_\textrm{h} \times k_\textrm{w}$. Hạt nhân này xác định một *cửa
sổ* trượt có kích thước $k_\textrm{h} \times k_\textrm{w}$. Đặt góc trên bên
trái của cửa sổ tại $(i, j)$, ta nhân theo từng phần tử mảng con đầu vào bị phủ
với $\mathbf{K}$ rồi cộng các tích lại để thu được phần tử đầu ra tại $(i, j)$:

<!--en:3688b38b kind=paragraph-->
This is the *two-dimensional cross-correlation* of $\mathbf{X}$ with
$\mathbf{K}$, written $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ with
zero-based indices matching code. Whereas :numref:`sec_why-conv` centered
offsets symmetrically from $-\Delta$ to $\Delta$, anchoring at the upper-left
corner differs only by a fixed shift of $\Delta$ while naturally handling
even-sized kernels (such as a $2 \times 2$ kernel or our $1 \times 2$ edge
detector) that have no center pixel.
<!--tr:3688b38b-->
Đây là phép *tương quan chéo hai chiều* (*two-dimensional cross-correlation*)
của $\mathbf{X}$ với $\mathbf{K}$, ký hiệu là $\mathbf{Y} = \mathbf{X} \star
\mathbf{K}$ với chỉ số bắt đầu từ 0 khớp với mã nguồn. Trong khi
:numref:`sec_why-conv` đặt các độ lệch đối xứng quanh tâm từ $-\Delta$ đến
$\Delta$, việc neo tại góc trên bên trái chỉ khác đi một độ dịch cố định bằng
$\Delta$, đồng thời xử lý tự nhiên các hạt nhân có kích thước chẵn (chẳng hạn
hạt nhân $2 \times 2$ hoặc bộ phát hiện biên $1 \times 2$ của chúng ta) vốn
không có điểm ảnh trung tâm.

<!--en:fa1b7da9 kind=caption-->
![Two-dimensional cross-correlation. The shaded portions are the first output element and the input and kernel elements used to compute it: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
<!--tr:fa1b7da9-->
![Phép tương quan chéo hai chiều. Các phần tô đậm biểu thị phần tử đầu ra đầu tiên cùng với các phần tử đầu vào và hạt nhân được dùng để tính nó: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)

<!--en:2f8e5deb kind=paragraph-->
In :numref:`fig_correlation`, we slide a $2 \times 2$ kernel across a
$3 \times 3$ input one pixel at a time, left to right and top to bottom. The
window fits cleanly at four positions, yielding the $2 \times 2$ output:
<!--tr:2f8e5deb-->
Trong :numref:`fig_correlation`, chúng ta trượt một hạt nhân $2 \times 2$ qua
đầu vào $3 \times 3$ từng điểm ảnh một, từ trái sang phải và từ trên xuống dưới.
Cửa sổ khớp vừa vặn tại bốn vị trí, cho ra đầu ra có kích thước $2 \times 2$:

<!--en:4741e012 kind=paragraph-->
Notice that the output shrinks slightly along each axis because the kernel must
stay wholly inside the input. Fitting a width-$k$ window into length $n$ leaves
$n - k + 1$ valid positions, so the output shape is
<!--tr:4741e012-->
Lưu ý rằng đầu ra bị thu nhỏ đi một chút theo mỗi trục vì hạt nhân phải nằm hoàn
toàn bên trong đầu vào. Việc đặt một cửa sổ có bề rộng $k$ vào bên trong chiều
dài $n$ cho $n - k + 1$ vị trí hợp lệ, do đó kích thước đầu ra là

<!--en:12d81c14 kind=paragraph-->
Thus a $3 \times 3$ input with a $2 \times 2$ kernel yields $2 \times 2$, a
$6 \times 8$ image with a $1 \times 2$ kernel yields $6 \times 7$, and a
$28 \times 28$ Fashion-MNIST image with a $5 \times 5$ kernel yields
$24 \times 24$. Whenever we want to preserve spatial dimensions, we surround the
input with a border of zeros (*padding*), which we examine alongside strides and
dilation in :numref:`sec_padding`.
<!--tr:12d81c14-->
Do đó, đầu vào $3 \times 3$ với hạt nhân $2 \times 2$ cho kết quả $2 \times 2$,
ảnh $6 \times 8$ với hạt nhân $1 \times 2$ cho kết quả $6 \times 7$, và ảnh
Fashion-MNIST $28 \times 28$ với hạt nhân $5 \times 5$ cho kết quả
$24 \times 24$. Bất cứ khi nào muốn giữ nguyên các chiều không gian, chúng ta
bao quanh đầu vào bằng một viền gồm các số không (*đệm* — *padding*), kỹ thuật
mà chúng ta sẽ tìm hiểu cùng với sải bước và độ giãn trong :numref:`sec_padding`.

<!--en:869d223f kind=heading-->
### Implementation
<!--tr:869d223f-->
### Cài đặt

<!--en:6a869f7c kind=paragraph-->
We implement :eqref:`eq_corr2d` directly in `corr2d` using a pair of loops over
every output coordinate.
<!--tr:6a869f7c-->
Chúng ta cài đặt trực tiếp :eqref:`eq_corr2d` trong hàm `corr2d` bằng một cặp
vòng lặp duyệt qua mọi tọa độ đầu ra.

<!--en:81e1f383 kind=paragraph-->
To verify `corr2d` against our hand calculation in :numref:`fig_correlation`,
we construct the $3 \times 3$ input tensor `X` and $2 \times 2$ kernel `K` and
evaluate `corr2d(X, K)`.
<!--tr:81e1f383-->
Để kiểm chứng `corr2d` so với phép tính tay trong :numref:`fig_correlation`,
chúng ta tạo tensor đầu vào `X` kích thước $3 \times 3$ và hạt nhân `K` kích
thước $2 \times 2$, rồi tính `corr2d(X, K)`.

<!--en:f567e1b2 kind=paragraph-->
The output reproduces :numref:`fig_correlation` entry for entry (and as shown in
:numref:`sec_why-conv`, strict mathematical convolution with a flipped kernel
$\mathbf{K}'$ learns the exact same family of layers, so we follow standard
deep learning terminology and call cross-correlation "convolution").
<!--tr:f567e1b2-->
Đầu ra khớp từng phần tử với :numref:`fig_correlation` (và như đã chỉ ra trong
:numref:`sec_why-conv`, phép tích chập toán học chặt chẽ với hạt nhân bị lật
$\mathbf{K}'$ học chính xác cùng một họ tầng, nên chúng ta tuân theo thuật ngữ
học sâu tiêu chuẩn và gọi phép tương quan chéo là "tích chập").

<!--en:bfe11c62 kind=heading-->
## Convolutional Layers
<!--tr:bfe11c62-->
## Tầng tích chập

<!--en:2cdc020d kind=heading-->
### The Layer
<!--tr:2cdc020d-->
### Định nghĩa tầng

<!--en:c6371ece kind=paragraph-->
A two-dimensional convolutional layer cross-correlates its input with a learned
kernel and adds a scalar bias. Just like our MLP weights in
:numref:`chap_perceptrons`, we initialize the kernel from
$\mathcal{N}(0, 0.01^2)$ and the bias to zero. Below, `Conv2D` registers both
parameters and invokes `corr2d` in its forward pass; a layer with a kernel of
height $h$ and width $w$ is called an $h \times w$ convolutional layer.
<!--tr:c6371ece-->
Một tầng tích chập hai chiều thực hiện phép tương quan chéo giữa đầu vào với một
hạt nhân được học rồi cộng thêm một hệ số điều chỉnh dạng số vô hướng. Giống như
các trọng số của MLP trong :numref:`chap_perceptrons`, chúng ta khởi tạo hạt
nhân từ $\mathcal{N}(0, 0.01^2)$ và hệ số điều chỉnh bằng không. Dưới đây,
`Conv2D` khai báo cả hai tham số này và gọi `corr2d` trong lượt truyền xuôi; một
tầng có hạt nhân chiều cao $h$ và chiều rộng $w$ được gọi là tầng tích chập
$h \times w$.

<!--en:29cd9061 kind=paragraph-->
Unlike a dense layer, a convolutional layer is not locked to a fixed input
resolution because its parameters depend only on the kernel size. The exact same
$2 \times 2$ layer (5 parameters total) processes both our $3 \times 3$ tensor
`X` and a $6 \times 8$ image, returning outputs of shapes $2 \times 2$ and
$5 \times 7$:
<!--tr:29cd9061-->
Khác với tầng kết nối đầy đủ, tầng tích chập không bị ràng buộc vào một độ phân
giải đầu vào cố định vì các tham số của nó chỉ phụ thuộc vào kích thước hạt
nhân. Chính cùng một tầng $2 \times 2$ đó (tổng cộng 5 tham số) có thể xử lý cả
tensor `X` kích thước $3 \times 3$ lẫn một ảnh kích thước $6 \times 8$, cho ra
các đầu ra có kích thước lần lượt là $2 \times 2$ và $5 \times 7$:

<!--en:c9e5aa07 kind=heading-->
### Edge Detection
<!--tr:c9e5aa07-->
### Phát hiện biên

<!--en:3d9a357b kind=paragraph-->
We now test the vertical edge detector from the introduction on a
$6 \times 8$ synthetic image whose middle four columns are black (`0`) and outer
columns are white (`1`).
<!--tr:3d9a357b-->
Bây giờ chúng ta kiểm thử bộ phát hiện biên dọc từ phần mở đầu trên một ảnh tổng
hợp kích thước $6 \times 8$, trong đó bốn cột ở giữa có màu đen (`0`) và các cột
ngoài cùng có màu trắng (`1`).

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
Tại mỗi vị trí $(i, j)$, hạt nhân này tính *sai phân hữu hạn* (*finite
difference*) theo phương ngang $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ — một
xấp xỉ rời rạc (sai khác không quá một dấu) của đạo hàm độ sáng theo phương
ngang, triệt tiêu trên các vùng phẳng và tăng vọt tại bất cứ nơi nào cường độ
sáng thay đổi đột ngột.

<!--en:e810b3a9 kind=paragraph-->
Cross-correlating the $6 \times 8$ stripe image `X` with the $1 \times 2$
horizontal-difference kernel `K` via `corr2d(X, K)` produces the $6 \times 7$
edge map `Y`:
<!--tr:e810b3a9-->
Thực hiện tương quan chéo ảnh sọc `X` kích thước $6 \times 8$ với hạt nhân sai
phân ngang `K` kích thước $1 \times 2$ thông qua `corr2d(X, K)` sẽ tạo ra bản đồ
biên `Y` kích thước $6 \times 7$:

<!--en:568358e4 kind=paragraph-->
Sure enough, `Y` fires `+1` along the white-to-black boundary, `-1` along the
black-to-white boundary, and `0` everywhere else. Applying the exact same kernel
to the transposed image (turning the vertical stripe into a horizontal one)
yields all zeros:
<!--tr:568358e4-->
Quả nhiên, `Y` cho giá trị `+1` dọc theo ranh giới trắng sang đen, `-1` dọc theo
ranh giới đen sang trắng, và `0` ở mọi nơi khác. Khi áp dụng chính hạt nhân đó
lên ảnh chuyển vị (biến dải sọc dọc thành dải sọc ngang), kết quả thu được toàn
là số không:

<!--en:71c591a4 kind=paragraph-->
Clearly `[1, -1]` responds strictly to vertical edges. Detecting horizontal
edges, diagonals, corners, or textures requires different kernels altogether.
<!--tr:71c591a4-->
Rõ ràng `[1, -1]` chỉ phản hồi với các biên dọc. Để phát hiện biên ngang, đường
chéo, góc hay kết cấu bề mặt, ta cần những hạt nhân hoàn toàn khác.

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
Thiết kế thủ công `[1, -1]` rất dễ khi chúng ta biết chính xác đặc trưng mình
muốn, nhưng không kỹ sư nào có thể tự tay xác định hàng nghìn bộ lọc trung gian
xuyên suốt một mạng sâu. Liệu hạ gradient có thể khôi phục hạt nhân `[1, -1]`
chỉ từ cặp đầu vào–đầu ra `(X, Y)` hay không? Chúng ta kiểm nghiệm điều này bằng
tầng tích chập 2D tích hợp sẵn của từng framework (bỏ qua hệ số điều chỉnh và
khởi tạo trọng số từ $\mathcal{N}(0, 0.01^2)$), cực tiểu hóa tổng sai số bình
phương trên toàn bộ 42 phần tử đầu ra trong 10 bước cập nhật gradient với tốc độ
học $0.03$. Vì các tầng tích hợp sẵn yêu cầu tensor 4D bao gồm cả trục batch và
trục kênh (:numref:`sec_channels`), chúng ta biến đổi kích thước của `X` và `Y`
cho phù hợp.

<!--en:c29c5d61 kind=paragraph-->
Initially the loss sits near 12 (the near-zero random kernel misses all 12
$\pm 1$ edge pixels by roughly 1), and within just nine steps it drops below
0.01. Inspecting the learned kernel confirms what it found:
<!--tr:c29c5d61-->
Ban đầu, giá trị mất mát nằm gần 12 (hạt nhân ngẫu nhiên gần bằng không sai lệch
khoảng 1 trên toàn bộ 12 điểm ảnh biên $\pm 1$), và chỉ sau chín bước nó đã giảm
xuống dưới 0.01. Kiểm tra hạt nhân đã học sẽ xác nhận kết quả mà mô hình tìm
được:

<!--en:67d25b0b kind=paragraph-->
Both entries land within roughly 0.02 of `[1, -1]`. Nothing in this loop knew
anything about edges: given a loss signal, gradient descent discovered the
filter on its own. Here we supervised the single layer directly with `Y`; in a
deep CNN, backpropagation learns every layer's kernels end to end from the final
task loss alone.
<!--tr:67d25b0b-->
Cả hai phần tử đều nằm trong phạm vi khoảng 0.02 so với `[1, -1]`. Không có gì
trong vòng lặp này biết trước về biên ảnh: chỉ từ tín hiệu mất mát, hạ gradient
đã tự mình phát hiện ra bộ lọc. Ở đây chúng ta giám sát trực tiếp một tầng đơn
bằng `Y`; trong một mạng CNN sâu, lan truyền ngược học hạt nhân của mọi tầng từ
đầu đến cuối chỉ dựa trên hàm mất mát của tác vụ cuối cùng.

<!--en:91b0010d kind=slide_text-->
Convolutions for Images<br>**2D cross-correlation · edge detection · learned kernels · im2col · receptive fields**
<!--tr:91b0010d-->
Phép tích chập cho ảnh<br>**tương quan chéo 2D · phát hiện biên · học hạt nhân · im2col · vùng tiếp nhận**

<!--en:1d94295a kind=slide_title-->
::: {.slide title="Detecting Local Edges Across an Image"}
<!--tr:1d94295a-->
::: {.slide title="Phát hiện biên cục bộ trên ảnh"}

<!--en:687ed302 kind=kicker-->
[Motivation & Core Mental Model]{.kicker}
<!--tr:687ed302-->
[Động lực & Tư duy cốt lõi]{.kicker}

<!--en:05fceed0 kind=slide_text-->
- **Synthetic $6 \times 8$ test image:** White outer columns (`1`) surrounding a 4-column black vertical band (`0`).
<!--tr:05fceed0-->
- **Ảnh kiểm thử tổng hợp $6 \times 8$:** Các cột ngoài màu trắng (`1`) bao quanh dải dọc màu đen gồm 4 cột (`0`).

<!--en:71e700b3 kind=slide_text-->
- **Horizontal finite difference:** Sweeping $\mathbf{K} = [1, -1]$ computes $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, firing `+1` at white-to-black and `-1` at black-to-white edges.
<!--tr:71e700b3-->
- **Sai phân hữu hạn theo phương ngang:** Quét $\mathbf{K} = [1, -1]$ tính $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, cho `+1` tại biên trắng–đen và `-1` tại biên đen–trắng.

<!--en:957dc033 kind=slide_text-->
- **Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent.
<!--tr:957dc033-->
- **Học so với thiết kế thủ công:** Các mạng CNN sâu tự động phát hiện hàng nghìn hạt nhân như vậy nhờ hạ gradient.

<!--en:cb1045e1 kind=slide_title-->
::: {.slide title="The 2D Cross-Correlation Operation"}
<!--tr:cb1045e1-->
::: {.slide title="Phép toán tương quan chéo 2D"}

<!--en:17bcece5 kind=kicker-->
[Sliding-Window Inner Products]{.kicker}
<!--tr:17bcece5-->
[Tích trong trên cửa sổ trượt]{.kicker}

<!--en:8d45aae8 kind=slide_text-->
Given an $n_\textrm{h} \times n_\textrm{w}$ input matrix $\mathbf{X}$ and a $k_\textrm{h} \times k_\textrm{w}$ kernel $\mathbf{K}$, the 2D cross-correlation $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ places the window's top-left corner at $(i, j)$ and sums elementwise products:
<!--tr:8d45aae8-->
Với ma trận đầu vào $\mathbf{X}$ kích thước $n_\textrm{h} \times n_\textrm{w}$ và hạt nhân $\mathbf{K}$ kích thước $k_\textrm{h} \times k_\textrm{w}$, phép tương quan chéo 2D $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ đặt góc trên bên trái của cửa sổ tại $(i, j)$ và cộng các tích theo từng phần tử:

<!--en:e484380c kind=slide_text-->
- **Top-left anchor:** Zero-based indices $a, b \ge 0$ match array indexing in code and handle even-sized kernels (like $2 \times 2$ or $1 \times 2$) cleanly.
<!--tr:e484380c-->
- **Neo góc trên bên trái:** Chỉ số từ 0 ($a, b \ge 0$) khớp với chỉ mục mảng trong mã nguồn và xử lý gọn các hạt nhân kích thước chẵn (như $2 \times 2$ hay $1 \times 2$).

<!--en:137c0980 kind=slide_text-->
- **Shaded top-left output $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.
<!--tr:137c0980-->
- **Đầu ra góc trên bên trái được tô đậm $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.

<!--en:575c4aea kind=slide_title-->
::: {.slide title="Valid Window Positions and Output Spatial Shape"}
<!--tr:575c4aea-->
::: {.slide title="Vị trí cửa sổ hợp lệ và kích thước không gian đầu ra"}

<!--en:7eda4531 kind=kicker-->
[Why Unpadded Convolutions Shrink the Grid]{.kicker}
<!--tr:7eda4531-->
[Vì sao tích chập không đệm làm thu nhỏ lưới ảnh]{.kicker}

<!--en:891243ae kind=slide_text-->
- **Valid window placements:** A $k_\textrm{h} \times k_\textrm{w}$ window fits inside an $n_\textrm{h} \times n_\textrm{w}$ grid at $n - k + 1$ positions per axis.
<!--tr:891243ae-->
- **Các vị trí cửa sổ hợp lệ:** Một cửa sổ $k_\textrm{h} \times k_\textrm{w}$ vừa khít trong lưới $n_\textrm{h} \times n_\textrm{w}$ tại $n - k + 1$ vị trí theo mỗi trục.

<!--en:55cc8c32 kind=slide_text-->
- **Shape examples:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.
<!--tr:55cc8c32-->
- **Ví dụ kích thước:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.

<!--en:1e3463da kind=slide_title-->
::: {.slide title="Implementing 2D Cross-Correlation in Code"}
<!--tr:1e3463da-->
::: {.slide title="Cài đặt phép tương quan chéo 2D bằng mã nguồn"}

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
Mỗi bước lặp lấy lát cắt một cửa sổ con $k_\textrm{h} \times k_\textrm{w}$ của `X`, nhân theo từng phần tử với `K`, rồi rút gọn bằng `sum`.

<!--en:4ee9f6b1 kind=slide_title-->
::: {.slide title="Verifying corr2d and the Flipped-Kernel Equivalence"}
<!--tr:4ee9f6b1-->
::: {.slide title="Kiểm chứng corr2d và tính tương đương khi lật hạt nhân"}

<!--en:2b0be423 kind=kicker-->
[Cross-Correlation vs. Mathematical Convolution]{.kicker}
<!--tr:2b0be423-->
[Tương quan chéo so với tích chập toán học]{.kicker}

<!--en:74b9b745 kind=slide_text-->
- **180° flipped kernel $\mathbf{K}'$:** Cross-correlating with $\mathbf{K}$ equals convolving with $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.
<!--tr:74b9b745-->
- **Hạt nhân lật 180° $\mathbf{K}'$:** Tương quan chéo với $\mathbf{K}$ tương đương tích chập với $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.

<!--en:5c8af7cd kind=slide_text-->
- **Identical learned class:** Frameworks compute cross-correlation and call it convolution.
<!--tr:5c8af7cd-->
- **Cùng lớp hàm được học:** Các framework tính phép tương quan chéo và gọi đó là tích chập.

<!--en:a69e29b5 kind=slide_title-->
::: {.slide title="Building a 2D Convolutional Layer Module"}
<!--tr:a69e29b5-->
::: {.slide title="Xây dựng mô-đun tầng tích chập 2D"}

<!--en:55763603 kind=kicker-->
[Learnable Kernel Weights Plus Scalar Bias]{.kicker}
<!--tr:55763603-->
[Trọng số hạt nhân học được cộng hệ số điều chỉnh vô hướng]{.kicker}

<!--en:5b8bad24 kind=slide_text-->
A convolutional layer wraps `corr2d` with a learnable $k_\textrm{h} \times k_\textrm{w}$ weight parameter (initialized from $\mathcal{N}(0, 0.01^2)$) and a scalar bias (initialized to $0$):
<!--tr:5b8bad24-->
Một tầng tích chập bao bọc `corr2d` với tham số trọng số $k_\textrm{h} \times k_\textrm{w}$ học được (khởi tạo từ $\mathcal{N}(0, 0.01^2)$) và một hệ số điều chỉnh dạng số vô hướng (khởi tạo bằng $0$):

<!--en:c16d97d6 kind=slide_title-->
::: {.slide title="Resolution Independence of Convolutional Layers"}
<!--tr:c16d97d6-->
::: {.slide title="Tính độc lập độ phân giải của tầng tích chập"}

<!--en:a7d1e48f kind=kicker-->
[Fixed Parameters Across Arbitrary Input Sizes]{.kicker}
<!--tr:a7d1e48f-->
[Tham số cố định trên mọi kích thước đầu vào]{.kicker}

<!--en:53a2bce7 kind=slide_text-->
- **Decoupled from $H \times W$:** The exact same 5 parameters ($2 \times 2$ kernel + 1 bias) process $3 \times 3$, $6 \times 8$, or megapixel inputs without reshaping weights.
<!--tr:53a2bce7-->
- **Độc lập với $H \times W$:** Chính cùng 5 tham số (hạt nhân $2 \times 2$ + 1 hệ số điều chỉnh) xử lý được đầu vào $3 \times 3$, $6 \times 8$ hay ảnh hàng megapixel mà không cần đổi kích thước trọng số.

<!--en:017fec24 kind=slide_title-->
::: {.slide title="Constructing a Synthetic Vertical-Edge Image"}
<!--tr:017fec24-->
::: {.slide title="Tạo ảnh tổng hợp chứa biên dọc"}

<!--en:896e641c kind=kicker-->
[A 6×8 Binary Test Pattern]{.kicker}
<!--tr:896e641c-->
[Mẫu kiểm thử nhị phân 6×8]{.kicker}

<!--en:0a4b456b kind=slide_text-->
We construct the $6 \times 8$ test image `X` with ones on the left and right borders and zeros across the middle four columns (`X[:, 2:6] = 0`):
<!--tr:0a4b456b-->
Chúng ta tạo ảnh kiểm thử `X` kích thước $6 \times 8$ với giá trị 1 ở hai viền trái, phải và giá trị 0 trên bốn cột giữa (`X[:, 2:6] = 0`):

<!--en:dfc7e3e3 kind=slide_text-->
- **Piecewise-constant columns:** Intensity is constant (`1` or `0`) within columns $0\text{--}1$, $2\text{--}5$, and $6\text{--}7$, jumping only between columns $1 \to 2$ and $5 \to 6$.
<!--tr:dfc7e3e3-->
- **Các cột hằng từng khúc:** Cường độ sáng không đổi (`1` hoặc `0`) trong các cột $0\text{--}1$, $2\text{--}5$ và $6\text{--}7$, chỉ nhảy bậc giữa cột $1 \to 2$ và $5 \to 6$.

<!--en:3ce31cbe kind=slide_title-->
::: {.slide title="Applying the Horizontal Finite-Difference Kernel"}
<!--tr:3ce31cbe-->
::: {.slide title="Áp dụng hạt nhân sai phân hữu hạn theo phương ngang"}

<!--en:b37a1029 kind=kicker-->
[Detecting Vertical Edges with K = [1, -1]]{.kicker}
<!--tr:b37a1029-->
[Phát hiện biên dọc với K = [1, -1]]{.kicker}

<!--en:9be85019 kind=slide_text-->
Cross-correlating `X` with the $1 \times 2$ kernel $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ computes $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ at every horizontal pair:
<!--tr:9be85019-->
Tương quan chéo `X` với hạt nhân $1 \times 2$ $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ tính $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ tại mọi cặp điểm ảnh kề nhau theo phương ngang:

<!--en:f649cfae kind=slide_text-->
- **Signed edge map:** Column 1 fires `+1` (white-to-black transition), column 5 fires `-1` (black-to-white transition), and all constant interior columns evaluate to `0`.
<!--tr:f649cfae-->
- **Bản đồ biên có dấu:** Cột 1 cho `+1` (chuyển trắng sang đen), cột 5 cho `-1` (chuyển đen sang trắng), và mọi cột hằng bên trong đều cho `0`.

<!--en:e6d155be kind=slide_title-->
::: {.slide title="Orientation Selectivity of Edge Kernels"}
<!--tr:e6d155be-->
::: {.slide title="Tính chọn lọc hướng của hạt nhân phát hiện biên"}

<!--en:b525cde4 kind=kicker-->
[Why One Kernel Cannot Detect All Orientations]{.kicker}
<!--tr:b525cde4-->
[Vì sao một hạt nhân không thể phát hiện mọi hướng biên]{.kicker}

<!--en:38159de6 kind=slide_text-->
- **Blind to horizontal edges:** Transposing `X` (`X.T`) makes horizontal neighbors identical, so $[1, -1]$ outputs all zeros.
<!--tr:38159de6-->
- **Không thấy biên ngang:** Chuyển vị `X` (`X.T`) làm các điểm ảnh kề nhau theo phương ngang giống hệt nhau, nên $[1, -1]$ cho toàn số không.

<!--en:f0f30f76 kind=slide_text-->
- **Filter banks (§{sec_channels}):** Vision layers learn parallel channels for horizontal, vertical, and diagonal orientations.
<!--tr:f0f30f76-->
- **Ngân hàng bộ lọc (§{sec_channels}):** Các tầng thị giác học song song nhiều kênh cho hướng ngang, dọc và chéo.

<!--en:b755e31b kind=slide_title-->
::: {.slide title="Learning an Edge Detector via Gradient Descent"}
<!--tr:b755e31b-->
::: {.slide title="Học bộ phát hiện biên bằng hạ gradient"}

<!--en:86bbeea7 kind=kicker-->
[Supervising a Built-In Convolutional Layer on (X, Y)]{.kicker}
<!--tr:86bbeea7-->
[Giám sát tầng tích chập tích hợp sẵn trên (X, Y)]{.kicker}

<!--en:a569780c kind=slide_text-->
- **4D framework tensors:** Reshape `X` and `Y` to 4D tensors (one example, one channel) and minimize $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ at $\eta = 0.03$.
<!--tr:a569780c-->
- **Tensor 4D của framework:** Đổi kích thước `X` và `Y` sang tensor 4D (một mẫu, một kênh) và cực tiểu hóa $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ với $\eta = 0.03$.

<!--en:2489ce6e kind=slide_title-->
::: {.slide title="Inspecting the Learned Kernel Weights"}
<!--tr:2489ce6e-->
::: {.slide title="Kiểm tra trọng số hạt nhân đã học"}

<!--en:7e4ef44b kind=kicker-->
[Why Initial Loss Is 12 and Converges to [1, -1]]{.kicker}
<!--tr:7e4ef44b-->
[Vì sao mất mát ban đầu bằng 12 và hội tụ về [1, -1]]{.kicker}

<!--en:9cd1c645 kind=slide_text-->
- **Why loss starts at $12.0$:** Near-zero weights predict $\hat{Y} \approx 0$, missing the twelve $\pm 1$ edge pixels in `Y` ($12 \times 1^2 = 12$).
<!--tr:9cd1c645-->
- **Vì sao mất mát khởi đầu ở $12.0$:** Trọng số gần không dự đoán $\hat{Y} \approx 0$, bỏ sót 12 điểm ảnh biên $\pm 1$ trong `Y` ($12 \times 1^2 = 12$).

<!--en:456670c5 kind=slide_text-->
- **Convergence in 9 steps:** SGD recovers $[0.99, -0.99]$ automatically from data.
<!--tr:456670c5-->
- **Hội tụ sau 9 bước:** SGD tự động khôi phục $[0.99, -0.99]$ từ dữ liệu.

=== DECISIONS ===
- Style & register (no `vi/style.md` yet): use "chúng ta" for inclusive "we" and "bạn" for direct reader address; keep decimal points as in English (`0.01`, `0.03`, `12.0`); keep framework/model/dataset/algorithm names and acronyms (`CNN`, `MLP`, `SGD`, `GPU`, `GEMM`, `Fashion-MNIST`, `im2col`, `Tensor Core`) in English; add English in parentheses on first introduction of key concepts where helpful.
- "kernel size" → "kích thước hạt nhân" (consistent with verified_d2l `kernel` → `hạt nhân`, instead of proposed `kích thước nhân tích chập`).
- "locality" → "tính cục bộ"
- "edge map" → "bản đồ biên"
- "edge detector" / "edge detection" → "bộ phát hiện biên" / "phát hiện biên"
- "window" / "sliding window" → "cửa sổ" / "cửa sổ trượt"
- "patch" → "mảng con"
- "elementwise" → "theo từng phần tử"
- "dense layer" → "tầng kết nối đầy đủ"
- "filter bank" → "ngân hàng bộ lọc"

=== NOTES ===
- 55f0a41b, 29cd9061: "return" / "returning" are ordinary verbs ("chuyển lại", "cho ra"), not the RL glossary term `return` (`tổng phần thưởng tích lũy`).
- 8d45aae8: math tokens reordered (`$\mathbf{X}$` before `$n_\textrm{h} \times n_\textrm{w}$`, `$\mathbf{K}$` before `$k_\textrm{h} \times k_\textrm{w}$`) to follow Vietnamese noun-before-modifier word order.
- 55763603: "Scalar" is used adjectivally in "Scalar Bias" (`hệ số điều chỉnh vô hướng`), so "vô hướng" is used rather than the noun form `số vô hướng`.
