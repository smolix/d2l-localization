<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u01 lang=vi rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:33d3b96b kind=heading-->
# Convolutions for Images
<!--tr:33d3b96b-->
# Tích chập trên ảnh

<!--en:852f5868 kind=paragraph-->
In :numref:`sec_why-conv`, we derived the convolutional layer on paper from two
geometric principles: translation equivariance and locality. Now we see what
that formula actually computes on an image, and whether gradient descent can
discover useful kernels automatically from data.
<!--tr:852f5868-->
Trong :numref:`sec_why-conv`, chúng ta đã suy ra công thức của tầng tích chập từ hai nguyên lý hình học là tính đẳng biến tịnh tiến và tính cục bộ. Bây giờ, chúng ta sẽ xem công thức ấy thực sự tính gì trên ảnh và liệu phương pháp hạ gradient có thể tự tìm ra các hạt nhân hữu ích từ dữ liệu hay không.

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
Hãy xét bài toán tìm các cạnh dọc ở nơi vùng sáng tiếp giáp với vùng tối. Lấy giá trị của mỗi điểm ảnh trừ đi giá trị của điểm ảnh ngay bên phải ($[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$), ta được 0 trong các vùng đồng nhất, $+1$ ở chỗ chuyển từ trắng sang đen và $-1$ ở chỗ chuyển từ đen sang trắng. Quét bộ phát hiện có hai trọng số $(1, -1)$ qua mọi vị trí hợp lệ trên ảnh cho ta một *bản đồ cạnh*. Trong mục này, chúng ta sẽ cài đặt phép tương quan chéo hai chiều, thử bộ phát hiện cạnh tự thiết kế, dùng phương pháp hạ gradient để *học* $(1, -1)$ trực tiếp từ dữ liệu, biểu diễn phép tích chập bằng một phép nhân ma trận duy nhất (*im2col*) và suy ra vùng tiếp nhận của một đơn vị mở rộng ra sao khi xếp chồng các tầng.

<!--en:47958ed8 kind=heading-->
## Cross-Correlation
<!--tr:47958ed8-->
## Tương quan chéo

<!--en:5e4296fe kind=heading-->
### The Operation
<!--tr:5e4296fe-->
### Phép tương quan chéo hai chiều

<!--en:ae7e906d kind=paragraph-->
Leaving channels aside for the moment, let the input $\mathbf{X}$ be an
$n_\textrm{h} \times n_\textrm{w}$ matrix and the kernel $\mathbf{K}$ be a
$k_\textrm{h} \times k_\textrm{w}$ matrix. The kernel defines a sliding
*window* of shape $k_\textrm{h} \times k_\textrm{w}$. Placing the window's
upper-left corner at $(i, j)$, we multiply the covered input patch elementwise
with $\mathbf{K}$ and sum the products to produce the output entry at $(i, j)$:
<!--tr:ae7e906d-->
Tạm thời bỏ qua các kênh, hãy xét đầu vào $\mathbf{X}$ là một ma trận kích thước $n_\textrm{h} \times n_\textrm{w}$ và hạt nhân $\mathbf{K}$ là một ma trận kích thước $k_\textrm{h} \times k_\textrm{w}$. Hạt nhân xác định một *cửa sổ* trượt có kích thước $k_\textrm{h} \times k_\textrm{w}$. Đặt góc trên bên trái của cửa sổ tại $(i, j)$, ta nhân từng phần tử trong vùng đầu vào mà cửa sổ phủ lên với phần tử tương ứng của $\mathbf{K}$ rồi cộng các tích để thu được phần tử đầu ra tại $(i, j)$:

<!--en:3688b38b kind=paragraph-->
This is the *two-dimensional cross-correlation* of $\mathbf{X}$ with
$\mathbf{K}$, written $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ with
zero-based indices matching code. Whereas :numref:`sec_why-conv` centered
offsets symmetrically from $-\Delta$ to $\Delta$, anchoring at the upper-left
corner differs only by a fixed shift of $\Delta$ while naturally handling
even-sized kernels (such as a $2 \times 2$ kernel or our $1 \times 2$ edge
detector) that have no center pixel.
<!--tr:3688b38b-->
Đây là phép *tương quan chéo hai chiều* của $\mathbf{X}$ với $\mathbf{K}$, ký hiệu là $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$, với các chỉ số bắt đầu từ 0 như trong mã. Trong :numref:`sec_why-conv`, các độ lệch được đặt đối xứng quanh tâm, từ $-\Delta$ đến $\Delta$. Lấy góc trên bên trái làm mốc chỉ tạo ra một độ dịch cố định bằng $\Delta$, đồng thời xử lý được cả các hạt nhân có kích thước chẵn (như hạt nhân $2 \times 2$ hoặc bộ phát hiện cạnh $1 \times 2$ của chúng ta) vốn không có điểm ảnh ở chính giữa.

<!--en:fa1b7da9 kind=caption-->
![Two-dimensional cross-correlation. The shaded portions are the first output element and the input and kernel elements used to compute it: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
<!--tr:fa1b7da9-->
![Tương quan chéo hai chiều. Các ô tô màu gồm phần tử đầu ra đầu tiên và những phần tử của đầu vào và hạt nhân dùng để tính nó: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)

<!--en:2f8e5deb kind=paragraph-->
In :numref:`fig_correlation`, we slide a $2 \times 2$ kernel across a
$3 \times 3$ input one pixel at a time, left to right and top to bottom. The
window fits cleanly at four positions, yielding the $2 \times 2$ output:
<!--tr:2f8e5deb-->
Trong :numref:`fig_correlation`, chúng ta trượt hạt nhân $2 \times 2$ qua đầu vào $3 \times 3$, mỗi lần một điểm ảnh, từ trái sang phải và từ trên xuống dưới. Cửa sổ nằm trọn trong đầu vào tại bốn vị trí, cho đầu ra $2 \times 2$:

<!--en:4741e012 kind=paragraph-->
Notice that the output shrinks slightly along each axis because the kernel must
stay wholly inside the input. Fitting a width-$k$ window into length $n$ leaves
$n - k + 1$ valid positions, so the output shape is
<!--tr:4741e012-->
Lưu ý rằng đầu ra nhỏ đi một chút dọc theo mỗi trục vì hạt nhân phải nằm hoàn toàn bên trong đầu vào. Một cửa sổ rộng $k$ có thể nằm trọn trong một đoạn dài $n$ tại $n - k + 1$ vị trí hợp lệ, nên kích thước đầu ra là

<!--en:12d81c14 kind=paragraph-->
Thus a $3 \times 3$ input with a $2 \times 2$ kernel yields $2 \times 2$, a
$6 \times 8$ image with a $1 \times 2$ kernel yields $6 \times 7$, and a
$28 \times 28$ Fashion-MNIST image with a $5 \times 5$ kernel yields
$24 \times 24$. Whenever we want to preserve spatial dimensions, we surround the
input with a border of zeros (*padding*), which we examine alongside strides and
dilation in :numref:`sec_padding`.
<!--tr:12d81c14-->
Vậy đầu vào $3 \times 3$ với hạt nhân $2 \times 2$ cho đầu ra $2 \times 2$, ảnh $6 \times 8$ với hạt nhân $1 \times 2$ cho đầu ra $6 \times 7$, và ảnh Fashion-MNIST $28 \times 28$ với hạt nhân $5 \times 5$ cho đầu ra $24 \times 24$. Khi muốn giữ nguyên kích thước không gian, chúng ta thêm một viền toàn số 0 quanh đầu vào (phép *đệm*). Chúng ta sẽ xét phép đệm cùng với sải bước và độ giãn trong :numref:`sec_padding`.

<!--en:869d223f kind=heading-->
### Implementation
<!--tr:869d223f-->
### Cài đặt

<!--en:6a869f7c kind=paragraph-->
We implement :eqref:`eq_corr2d` directly in `corr2d` using a pair of loops over
every output coordinate.
<!--tr:6a869f7c-->
Chúng ta cài đặt trực tiếp :eqref:`eq_corr2d` trong `corr2d` bằng hai vòng lặp lồng nhau, duyệt qua mọi tọa độ đầu ra.

<!--en:81e1f383 kind=paragraph-->
To verify `corr2d` against our hand calculation in :numref:`fig_correlation`,
we construct the $3 \times 3$ input tensor `X` and $2 \times 2$ kernel `K` and
evaluate `corr2d(X, K)`.
<!--tr:81e1f383-->
Để đối chiếu `corr2d` với kết quả tính tay trong :numref:`fig_correlation`, chúng ta tạo tensor đầu vào $3 \times 3$ `X` và hạt nhân $2 \times 2$ `K`, rồi tính `corr2d(X, K)`.

<!--en:f567e1b2 kind=paragraph-->
The output reproduces :numref:`fig_correlation` entry for entry (and as shown in
:numref:`sec_why-conv`, strict mathematical convolution with a flipped kernel
$\mathbf{K}'$ learns the exact same family of layers, so we follow standard
deep learning terminology and call cross-correlation "convolution").
<!--tr:f567e1b2-->
Từng phần tử của đầu ra đều khớp với :numref:`fig_correlation`. Như đã thấy trong :numref:`sec_why-conv`, phép tích chập theo đúng nghĩa toán học với hạt nhân lật $\mathbf{K}'$ cũng học được đúng họ tầng đó, nên theo cách gọi thông dụng trong học sâu, chúng ta gọi phép tương quan chéo là “tích chập”.

<!--en:bfe11c62 kind=heading-->
## Convolutional Layers
<!--tr:bfe11c62-->
## Tầng tích chập

<!--en:2cdc020d kind=heading-->
### The Layer
<!--tr:2cdc020d-->
### Cấu tạo của tầng

<!--en:c6371ece kind=paragraph-->
A two-dimensional convolutional layer cross-correlates its input with a learned
kernel and adds a scalar bias. Just like our MLP weights in
:numref:`chap_perceptrons`, we initialize the kernel from
$\mathcal{N}(0, 0.01^2)$ and the bias to zero. Below, `Conv2D` registers both
parameters and invokes `corr2d` in its forward pass; a layer with a kernel of
height $h$ and width $w$ is called an $h \times w$ convolutional layer.
<!--tr:c6371ece-->
Một tầng tích chập hai chiều tính tương quan chéo giữa đầu vào và một hạt nhân được học, rồi cộng thêm một hệ số điều chỉnh vô hướng. Tương tự cách khởi tạo trọng số của perceptron đa tầng (MLP) trong :numref:`chap_perceptrons`, chúng ta lấy mẫu các trọng số của hạt nhân từ $\mathcal{N}(0, 0.01^2)$ và đặt hệ số điều chỉnh ban đầu bằng 0. Dưới đây, `Conv2D` đăng ký cả hai tham số và gọi `corr2d` trong lượt truyền xuôi. Tầng có hạt nhân cao $h$ và rộng $w$ được gọi là tầng tích chập $h \times w$.

<!--en:29cd9061 kind=paragraph-->
Unlike a dense layer, a convolutional layer is not locked to a fixed input
resolution because its parameters depend only on the kernel size. The exact same
$2 \times 2$ layer (5 parameters total) processes both our $3 \times 3$ tensor
`X` and a $6 \times 8$ image, returning outputs of shapes $2 \times 2$ and
$5 \times 7$:
<!--tr:29cd9061-->
Khác với tầng kết nối đầy đủ, tầng tích chập không bị giới hạn ở một độ phân giải đầu vào cố định vì các tham số chỉ phụ thuộc vào kích thước nhân tích chập. Cùng một tầng $2 \times 2$ (tổng cộng 5 tham số) xử lý được cả tensor $3 \times 3$ `X` lẫn ảnh $6 \times 8$, cho đầu ra có kích thước lần lượt là $2 \times 2$ và $5 \times 7$:

<!--en:c9e5aa07 kind=heading-->
### Edge Detection
<!--tr:c9e5aa07-->
### Phát hiện cạnh

<!--en:3d9a357b kind=paragraph-->
We now test the vertical edge detector from the introduction on a
$6 \times 8$ synthetic image whose middle four columns are black (`0`) and outer
columns are white (`1`).
<!--tr:3d9a357b-->
Bây giờ, chúng ta thử bộ phát hiện cạnh dọc ở phần mở đầu trên một ảnh tổng hợp $6 \times 8$ có bốn cột giữa màu đen (`0`) và các cột ngoài màu trắng (`1`).

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
Tại mỗi vị trí $(i, j)$, hạt nhân này tính *sai phân hữu hạn* theo chiều ngang $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, một xấp xỉ rời rạc (không xét dấu) của đạo hàm độ sáng theo chiều ngang. Giá trị này bằng 0 ở những vùng có độ sáng đồng đều và có độ lớn tăng vọt ở nơi cường độ sáng thay đổi đột ngột.

<!--en:e810b3a9 kind=paragraph-->
Cross-correlating the $6 \times 8$ stripe image `X` with the $1 \times 2$
horizontal-difference kernel `K` via `corr2d(X, K)` produces the $6 \times 7$
edge map `Y`:
<!--tr:e810b3a9-->
Tính tương quan chéo giữa ảnh sọc $6 \times 8$ `X` và hạt nhân sai phân ngang $1 \times 2$ `K` bằng `corr2d(X, K)` cho bản đồ cạnh $6 \times 7$ `Y`:

<!--en:568358e4 kind=paragraph-->
Sure enough, `Y` fires `+1` along the white-to-black boundary, `-1` along the
black-to-white boundary, and `0` everywhere else. Applying the exact same kernel
to the transposed image (turning the vertical stripe into a horizontal one)
yields all zeros:
<!--tr:568358e4-->
Đúng như dự đoán, `Y` cho giá trị `+1` dọc theo ranh giới chuyển từ trắng sang đen, `-1` dọc theo ranh giới chuyển từ đen sang trắng và `0` ở mọi vị trí khác. Áp dụng chính hạt nhân đó lên ảnh đã chuyển vị (biến sọc dọc thành sọc ngang) cho đầu ra toàn số 0:

<!--en:71c591a4 kind=paragraph-->
Clearly `[1, -1]` responds strictly to vertical edges. Detecting horizontal
edges, diagonals, corners, or textures requires different kernels altogether.
<!--tr:71c591a4-->
Rõ ràng `[1, -1]` chỉ đáp ứng với các cạnh dọc. Để phát hiện cạnh ngang, cạnh chéo, góc hoặc kết cấu, chúng ta cần những hạt nhân khác hẳn.

<!--en:f0af4887 kind=heading-->
### Learning a Kernel
<!--tr:f0af4887-->
### Học hạt nhân

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
Khi biết chính xác đặc trưng cần tìm, ta dễ dàng tự thiết kế `[1, -1]`. Tuy vậy, không kỹ sư nào có thể tự tay thiết kế hàng nghìn bộ lọc trung gian trong cả một mạng sâu. Liệu phương pháp hạ gradient có thể tìm lại hạt nhân `[1, -1]` chỉ từ cặp đầu vào–đầu ra `(X, Y)` hay không? Chúng ta kiểm tra điều này bằng tầng tích chập 2D có sẵn trong mỗi thư viện (bỏ hệ số điều chỉnh và khởi tạo trọng số từ $\mathcal{N}(0, 0.01^2)$), rồi tối thiểu hóa tổng sai số bình phương trên cả 42 phần tử đầu ra qua 10 bước cập nhật theo gradient với tốc độ học $0.03$. Vì các tầng có sẵn nhận tensor 4D gồm cả trục batch và trục kênh (:numref:`sec_channels`), chúng ta đổi kích thước của `X` và `Y` cho phù hợp.

<!--en:c29c5d61 kind=paragraph-->
Initially the loss sits near 12 (the near-zero random kernel misses all 12
$\pm 1$ edge pixels by roughly 1), and within just nine steps it drops below
0.01. Inspecting the learned kernel confirms what it found:
<!--tr:c29c5d61-->
Ban đầu, giá trị mất mát xấp xỉ 12 (hạt nhân ngẫu nhiên có trọng số gần 0 dự đoán sai khoảng 1 ở cả 12 điểm ảnh trên các cạnh có giá trị $\pm 1$). Chỉ trong chín bước, giá trị này giảm xuống dưới 0.01. Hạt nhân học được cho thấy kết quả:

<!--en:67d25b0b kind=paragraph-->
Both entries land within roughly 0.02 of `[1, -1]`. Nothing in this loop knew
anything about edges: given a loss signal, gradient descent discovered the
filter on its own. Here we supervised the single layer directly with `Y`; in a
deep CNN, backpropagation learns every layer's kernels end to end from the final
task loss alone.
<!--tr:67d25b0b-->
Cả hai phần tử đều cách giá trị tương ứng trong `[1, -1]` không quá khoảng 0.02. Vòng lặp không hề có thông tin về cạnh. Chỉ từ tín hiệu mất mát, phương pháp hạ gradient đã tự tìm ra bộ lọc. Ở đây, chúng ta dùng `Y` để giám sát trực tiếp một tầng duy nhất. Trong mạng nơ-ron tích chập (CNN) sâu, lan truyền ngược cho phép học hạt nhân của mọi tầng từ đầu đến cuối chỉ dựa vào mất mát của tác vụ cuối cùng.

<!--en:91b0010d kind=slide_text-->
Convolutions for Images<br>**2D cross-correlation · edge detection · learned kernels · im2col · receptive fields**
<!--tr:91b0010d-->
Tích chập trên ảnh<br>**Tương quan chéo 2D · phát hiện cạnh · học hạt nhân · im2col · vùng tiếp nhận**

<!--en:1d94295a kind=slide_title-->
::: {.slide title="Detecting Local Edges Across an Image"}
<!--tr:1d94295a-->
::: {.slide title="Phát hiện cạnh cục bộ trên ảnh"}

<!--en:687ed302 kind=kicker-->
[Motivation & Core Mental Model]{.kicker}
<!--tr:687ed302-->
[Động cơ và trực giác chính]{.kicker}

<!--en:05fceed0 kind=slide_text-->
- **Synthetic $6 \times 8$ test image:** White outer columns (`1`) surrounding a 4-column black vertical band (`0`).
<!--tr:05fceed0-->
- **Ảnh thử tổng hợp $6 \times 8$:** Các cột trắng (`1`) hai bên bao quanh dải đen dọc rộng 4 cột (`0`).

<!--en:71e700b3 kind=slide_text-->
- **Horizontal finite difference:** Sweeping $\mathbf{K} = [1, -1]$ computes $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, firing `+1` at white-to-black and `-1` at black-to-white edges.
<!--tr:71e700b3-->
- **Sai phân hữu hạn theo chiều ngang:** Quét $\mathbf{K} = [1, -1]$ để tính $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, cho `+1` ở cạnh từ trắng sang đen và `-1` ở cạnh từ đen sang trắng.

<!--en:957dc033 kind=slide_text-->
- **Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent.
<!--tr:957dc033-->
- **Học và thiết kế thủ công:** CNN sâu tự tìm ra hàng nghìn hạt nhân như vậy bằng phương pháp hạ gradient.

<!--en:cb1045e1 kind=slide_title-->
::: {.slide title="The 2D Cross-Correlation Operation"}
<!--tr:cb1045e1-->
::: {.slide title="Phép tương quan chéo 2D"}

<!--en:17bcece5 kind=kicker-->
[Sliding-Window Inner Products]{.kicker}
<!--tr:17bcece5-->
[Tích trong trên cửa sổ trượt]{.kicker}

<!--en:8d45aae8 kind=slide_text-->
Given an $n_\textrm{h} \times n_\textrm{w}$ input matrix $\mathbf{X}$ and a $k_\textrm{h} \times k_\textrm{w}$ kernel $\mathbf{K}$, the 2D cross-correlation $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ places the window's top-left corner at $(i, j)$ and sums elementwise products:
<!--tr:8d45aae8-->
Cho ma trận đầu vào kích thước $n_\textrm{h} \times n_\textrm{w}$ là $\mathbf{X}$ và hạt nhân kích thước $k_\textrm{h} \times k_\textrm{w}$ là $\mathbf{K}$. Phép tương quan chéo 2D $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ đặt góc trên bên trái của cửa sổ tại $(i, j)$ rồi cộng các tích theo từng phần tử:

<!--en:e484380c kind=slide_text-->
- **Top-left anchor:** Zero-based indices $a, b \ge 0$ match array indexing in code and handle even-sized kernels (like $2 \times 2$ or $1 \times 2$) cleanly.
<!--tr:e484380c-->
- **Mốc ở góc trên bên trái:** Chỉ số $a, b \ge 0$ bắt đầu từ 0 như chỉ số mảng trong mã và dùng được với hạt nhân kích thước chẵn (như $2 \times 2$ hay $1 \times 2$).

<!--en:137c0980 kind=slide_text-->
- **Shaded top-left output $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.
<!--tr:137c0980-->
- **Ô đầu ra tô màu ở góc trên bên trái $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.

<!--en:575c4aea kind=slide_title-->
::: {.slide title="Valid Window Positions and Output Spatial Shape"}
<!--tr:575c4aea-->
::: {.slide title="Vị trí cửa sổ hợp lệ và kích thước không gian của đầu ra"}

<!--en:7eda4531 kind=kicker-->
[Why Unpadded Convolutions Shrink the Grid]{.kicker}
<!--tr:7eda4531-->
[Vì sao tích chập không đệm làm lưới nhỏ đi]{.kicker}

<!--en:891243ae kind=slide_text-->
- **Valid window placements:** A $k_\textrm{h} \times k_\textrm{w}$ window fits inside an $n_\textrm{h} \times n_\textrm{w}$ grid at $n - k + 1$ positions per axis.
<!--tr:891243ae-->
- **Vị trí hợp lệ:** Cửa sổ $k_\textrm{h} \times k_\textrm{w}$ nằm trọn trong lưới $n_\textrm{h} \times n_\textrm{w}$ tại $n - k + 1$ vị trí trên mỗi trục.

<!--en:55cc8c32 kind=slide_text-->
- **Shape examples:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.
<!--tr:55cc8c32-->
- **Ví dụ về kích thước:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.

<!--en:1e3463da kind=slide_title-->
::: {.slide title="Implementing 2D Cross-Correlation in Code"}
<!--tr:1e3463da-->
::: {.slide title="Cài đặt phép tương quan chéo 2D"}

<!--en:07566992 kind=kicker-->
[Direct Nested-Loop Implementation]{.kicker}
<!--tr:07566992-->
[Cài đặt trực tiếp bằng vòng lặp lồng nhau]{.kicker}

<!--en:202aa5d3 kind=slide_text-->
The function `corr2d` allocates the $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ output tensor and computes each entry as the sum of `X[i:i+h, j:j+w] * K`:
<!--tr:202aa5d3-->
Hàm `corr2d` cấp phát tensor đầu ra có kích thước $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ rồi tính từng phần tử bằng cách lấy tổng của `X[i:i+h, j:j+w] * K`:

<!--en:bef130ba kind=slide_text-->
Each iteration slices a $k_\textrm{h} \times k_\textrm{w}$ sub-window of `X`, multiplies elementwise with `K`, and reduces via `sum`.
<!--tr:bef130ba-->
Mỗi lượt lặp lấy một lát cắt $k_\textrm{h} \times k_\textrm{w}$ của `X`, nhân từng phần tử với phần tử tương ứng của `K` rồi cộng lại bằng `sum`.

<!--en:4ee9f6b1 kind=slide_title-->
::: {.slide title="Verifying corr2d and the Flipped-Kernel Equivalence"}
<!--tr:4ee9f6b1-->
::: {.slide title="Kiểm chứng corr2d và tính tương đương khi lật hạt nhân"}

<!--en:2b0be423 kind=kicker-->
[Cross-Correlation vs. Mathematical Convolution]{.kicker}
<!--tr:2b0be423-->
[Tương quan chéo và tích chập theo nghĩa toán học]{.kicker}

<!--en:74b9b745 kind=slide_text-->
- **180° flipped kernel $\mathbf{K}'$:** Cross-correlating with $\mathbf{K}$ equals convolving with $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.
<!--tr:74b9b745-->
- **Lật 180° để được hạt nhân $\mathbf{K}'$:** Tương quan chéo với $\mathbf{K}$ tương đương với tích chập với $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.

<!--en:5c8af7cd kind=slide_text-->
- **Identical learned class:** Frameworks compute cross-correlation and call it convolution.
<!--tr:5c8af7cd-->
- **Cùng họ tầng học được:** Các thư viện tính tương quan chéo và gọi đó là tích chập.

<!--en:a69e29b5 kind=slide_title-->
::: {.slide title="Building a 2D Convolutional Layer Module"}
<!--tr:a69e29b5-->
::: {.slide title="Xây dựng mô-đun tầng tích chập 2D"}

<!--en:55763603 kind=kicker-->
[Learnable Kernel Weights Plus Scalar Bias]{.kicker}
<!--tr:55763603-->
[Trọng số hạt nhân học được và hệ số điều chỉnh vô hướng]{.kicker}

<!--en:5b8bad24 kind=slide_text-->
A convolutional layer wraps `corr2d` with a learnable $k_\textrm{h} \times k_\textrm{w}$ weight parameter (initialized from $\mathcal{N}(0, 0.01^2)$) and a scalar bias (initialized to $0$):
<!--tr:5b8bad24-->
Tầng tích chập bao bọc `corr2d` với tham số trọng số học được có kích thước $k_\textrm{h} \times k_\textrm{w}$ (khởi tạo từ $\mathcal{N}(0, 0.01^2)$) và hệ số điều chỉnh vô hướng (khởi tạo bằng $0$):

<!--en:c16d97d6 kind=slide_title-->
::: {.slide title="Resolution Independence of Convolutional Layers"}
<!--tr:c16d97d6-->
::: {.slide title="Tầng tích chập độc lập với độ phân giải"}

<!--en:a7d1e48f kind=kicker-->
[Fixed Parameters Across Arbitrary Input Sizes]{.kicker}
<!--tr:a7d1e48f-->
[Cùng bộ tham số cho mọi kích thước đầu vào]{.kicker}

<!--en:53a2bce7 kind=slide_text-->
- **Decoupled from $H \times W$:** The exact same 5 parameters ($2 \times 2$ kernel + 1 bias) process $3 \times 3$, $6 \times 8$, or megapixel inputs without reshaping weights.
<!--tr:53a2bce7-->
- **Độc lập với $H \times W$:** Cùng 5 tham số (hạt nhân $2 \times 2$ và 1 hệ số điều chỉnh) xử lý đầu vào $3 \times 3$, $6 \times 8$ hoặc cỡ megapixel mà không cần đổi kích thước trọng số.

<!--en:017fec24 kind=slide_title-->
::: {.slide title="Constructing a Synthetic Vertical-Edge Image"}
<!--tr:017fec24-->
::: {.slide title="Tạo ảnh tổng hợp có cạnh dọc"}

<!--en:896e641c kind=kicker-->
[A 6×8 Binary Test Pattern]{.kicker}
<!--tr:896e641c-->
[Mẫu thử nhị phân 6×8]{.kicker}

<!--en:0a4b456b kind=slide_text-->
We construct the $6 \times 8$ test image `X` with ones on the left and right borders and zeros across the middle four columns (`X[:, 2:6] = 0`):
<!--tr:0a4b456b-->
Chúng ta tạo ảnh thử $6 \times 8$ `X` với giá trị 1 ở hai biên trái và phải, và giá trị 0 ở bốn cột giữa (`X[:, 2:6] = 0`):

<!--en:dfc7e3e3 kind=slide_text-->
- **Piecewise-constant columns:** Intensity is constant (`1` or `0`) within columns $0\text{--}1$, $2\text{--}5$, and $6\text{--}7$, jumping only between columns $1 \to 2$ and $5 \to 6$.
<!--tr:dfc7e3e3-->
- **Độ sáng không đổi theo từng nhóm cột:** Cường độ không đổi (`1` hoặc `0`) trong từng nhóm cột $0\text{--}1$, $2\text{--}5$ và $6\text{--}7$, chỉ thay đổi khi chuyển từ cột $1 \to 2$ và $5 \to 6$.

<!--en:3ce31cbe kind=slide_title-->
::: {.slide title="Applying the Horizontal Finite-Difference Kernel"}
<!--tr:3ce31cbe-->
::: {.slide title="Áp dụng hạt nhân sai phân hữu hạn theo chiều ngang"}

<!--en:b37a1029 kind=kicker-->
[Detecting Vertical Edges with K = [1, -1]]{.kicker}
<!--tr:b37a1029-->
[Phát hiện cạnh dọc với K = [1, -1]]{.kicker}

<!--en:9be85019 kind=slide_text-->
Cross-correlating `X` with the $1 \times 2$ kernel $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ computes $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ at every horizontal pair:
<!--tr:9be85019-->
Tính tương quan chéo của `X` với hạt nhân $1 \times 2$ là $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ cho giá trị $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ ở mỗi cặp điểm ảnh kề nhau theo chiều ngang:

<!--en:f649cfae kind=slide_text-->
- **Signed edge map:** Column 1 fires `+1` (white-to-black transition), column 5 fires `-1` (black-to-white transition), and all constant interior columns evaluate to `0`.
<!--tr:f649cfae-->
- **Bản đồ cạnh có dấu:** Cột 1 cho `+1` (từ trắng sang đen), cột 5 cho `-1` (từ đen sang trắng), còn các cột nằm trong vùng đồng nhất đều cho `0`.

<!--en:e6d155be kind=slide_title-->
::: {.slide title="Orientation Selectivity of Edge Kernels"}
<!--tr:e6d155be-->
::: {.slide title="Khả năng chọn hướng của hạt nhân phát hiện cạnh"}

<!--en:b525cde4 kind=kicker-->
[Why One Kernel Cannot Detect All Orientations]{.kicker}
<!--tr:b525cde4-->
[Vì sao một hạt nhân không phát hiện được cạnh theo mọi hướng]{.kicker}

<!--en:38159de6 kind=slide_text-->
- **Blind to horizontal edges:** Transposing `X` (`X.T`) makes horizontal neighbors identical, so $[1, -1]$ outputs all zeros.
<!--tr:38159de6-->
- **Cạnh ngang không được phát hiện:** Chuyển vị `X` (`X.T`) làm các điểm ảnh kề nhau theo chiều ngang có cùng giá trị, nên $[1, -1]$ cho đầu ra toàn số 0.

<!--en:f0f30f76 kind=slide_text-->
- **Filter banks (§{sec_channels}):** Vision layers learn parallel channels for horizontal, vertical, and diagonal orientations.
<!--tr:f0f30f76-->
- **Tập bộ lọc (§{sec_channels}):** Các tầng xử lý ảnh học song song các kênh theo hướng ngang, dọc và chéo.

<!--en:b755e31b kind=slide_title-->
::: {.slide title="Learning an Edge Detector via Gradient Descent"}
<!--tr:b755e31b-->
::: {.slide title="Học bộ phát hiện cạnh bằng phương pháp hạ gradient"}

<!--en:86bbeea7 kind=kicker-->
[Supervising a Built-In Convolutional Layer on (X, Y)]{.kicker}
<!--tr:86bbeea7-->
[Dùng (X, Y) để giám sát tầng tích chập có sẵn]{.kicker}

<!--en:a569780c kind=slide_text-->
- **4D framework tensors:** Reshape `X` and `Y` to 4D tensors (one example, one channel) and minimize $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ at $\eta = 0.03$.
<!--tr:a569780c-->
- **Tensor 4D của thư viện:** Đổi kích thước `X` và `Y` thành tensor 4D (một mẫu, một kênh) rồi tối thiểu hóa $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ với $\eta = 0.03$.

<!--en:2489ce6e kind=slide_title-->
::: {.slide title="Inspecting the Learned Kernel Weights"}
<!--tr:2489ce6e-->
::: {.slide title="Kiểm tra trọng số hạt nhân học được"}

<!--en:7e4ef44b kind=kicker-->
[Why Initial Loss Is 12 and Converges to [1, -1]]{.kicker}
<!--tr:7e4ef44b-->
[Vì sao mất mát ban đầu xấp xỉ 12 và trọng số hội tụ về [1, -1]]{.kicker}

<!--en:9cd1c645 kind=slide_text-->
- **Why loss starts at $12.0$:** Near-zero weights predict $\hat{Y} \approx 0$, missing the twelve $\pm 1$ edge pixels in `Y` ($12 \times 1^2 = 12$).
<!--tr:9cd1c645-->
- **Vì sao mất mát ban đầu xấp xỉ $12.0$:** Trọng số gần 0 cho dự đoán $\hat{Y} \approx 0$, bỏ sót 12 điểm ảnh trên các cạnh có giá trị $\pm 1$ trong `Y` ($12 \times 1^2 = 12$).

<!--en:456670c5 kind=slide_text-->
- **Convergence in 9 steps:** SGD recovers $[0.99, -0.99]$ automatically from data.
<!--tr:456670c5-->
- **Hội tụ sau 9 bước:** Phương pháp hạ gradient tự tìm lại $[0.99, -0.99]$ từ dữ liệu.

=== DECISIONS ===
- Giữ giọng đồng nghiệp và dùng “chúng ta” khi cùng lập luận hoặc thực hành. Dùng dấu chấm thập phân, giữ nguyên ký hiệu và tên riêng. Dùng thuật ngữ tiếng Việt trong văn xuôi, giữ tên kỹ thuật như im2col và giới thiệu chữ viết tắt thông dụng như MLP, CNN khi khai triển lần đầu. Dùng 2D và 4D cho dạng viết gọn.
- “locality” → “tính cục bộ”, “sliding window” → “cửa sổ trượt”, “input patch” → “vùng đầu vào” mà cửa sổ phủ lên, “valid position” → “vị trí hợp lệ”.
- “edge detection” → “phát hiện cạnh”, “edge detector” → “bộ phát hiện cạnh”, “vertical/horizontal edge” → “cạnh dọc/ngang”, “edge map” → “bản đồ cạnh”, “texture” → “kết cấu”.
- “unit” (trong một tầng) → “đơn vị”, “dense layer” và “fully connected layer” → “tầng kết nối đầy đủ”, “framework” → “thư viện” trong ngữ cảnh mã học sâu, “example” (một phần tử của batch) → “mẫu”.
- “elementwise multiplication” → “nhân theo từng phần tử”, “spatial dimensions/shape” → “kích thước không gian”, “reshape” → “đổi kích thước”, “end to end” → “từ đầu đến cuối”.
=== NOTES ===
- 7e4ef44b: Câu gốc gán việc hội tụ về [1, -1] cho mất mát. Trọng số mới là đại lượng tiến tới cặp đó; mất mát ban đầu chỉ xấp xỉ 12, như đoạn c29c5d61.
- 9cd1c645: Giá trị ban đầu chỉ xấp xỉ $12.0$ vì trọng số ngẫu nhiên ban đầu gần 0.
- 456670c5: Câu gốc gọi phép cập nhật là SGD, nhưng vòng lặp dùng gradient của mất mát trên toàn bộ cặp đầu vào–đầu ra ở mỗi bước. Vì vậy bản dịch dùng “hạ gradient” và giữ lại cảnh báo “SGD”.
- c6371ece, 55763603, 5b8bad24: Cảnh báo “scalar” là khớp giả. Từ này bổ nghĩa cho “bias”, nên dùng “hệ số điều chỉnh vô hướng”. “Số vô hướng” là dạng danh từ.
