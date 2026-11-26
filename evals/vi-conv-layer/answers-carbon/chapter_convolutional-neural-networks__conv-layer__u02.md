<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u02 lang=vi rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:0b85caf9 kind=heading-->
## Convolution as Matrix Multiplication
<!--tr:0b85caf9-->
## Phép tích chập dưới dạng phép nhân ma trận

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
Mặc dù các vòng lặp lồng nhau trong `corr2d` giúp biểu thức toán học trở nên minh bạch, việc duyệt từng
điểm ảnh đầu ra một trong Python là quá chậm cho các tác vụ thực tế. Một giải pháp
kinh điển, và hiện vẫn là một trong những thuật toán được các thư viện lựa chọn, là chuyển đổi
phép tương quan chéo thành một phép nhân ma trận lớn duy nhất, đúng phép toán mà
GPU thực thi ở tốc độ đỉnh. Lưu ý rằng mỗi
phần tử đầu ra trong :numref:`fig_correlation` đơn giản là tích vô hướng giữa
hạt nhân trải phẳng gồm $k_\textrm{h} k_\textrm{w}$ phần tử với cửa sổ đầu vào
trải phẳng tương ứng. Trích xuất toàn bộ
$(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ cửa sổ rồi
xếp chồng chúng thành các hàng sẽ tạo ra một ma trận vùng con; nhân ma trận này với hạt nhân
trải phẳng sẽ tính được toàn bộ phép tương quan chéo chỉ trong một lần. Phép biến đổi bố cục
tiêu chuẩn này được gọi là *im2col* ("image to columns" — chuyển ảnh thành cột, hay ở đây là thành các hàng).
:numref:`fig_conv_im2col` minh họa phép biến đổi này trên ví dụ của
:numref:`fig_correlation`.

<!--en:107554a5 kind=caption-->
![The cross-correlation of :numref:`fig_correlation` as one matrix product. Each of the four $2 \times 2$ windows of the input becomes a row of the patch matrix, and multiplying by the flattened kernel gives the flattened output. The shaded cells follow the first window to its row and to the output 19. The center pixel 4 lies in all four windows, so it appears in every row.](../img/conv-im2col.svg)
<!--tr:107554a5-->
![Phép tương quan chéo ở :numref:`fig_correlation` được biểu diễn dưới dạng một tích ma trận. Mỗi cửa sổ trong bốn cửa sổ $2 \times 2$ của đầu vào trở thành một hàng của ma trận vùng con, và nhân với hạt nhân trải phẳng sẽ cho đầu ra trải phẳng. Các ô tô đậm dõi theo cửa sổ đầu tiên đến hàng tương ứng của nó và đến phần tử đầu ra 19. Điểm ảnh trung tâm 4 nằm trong cả bốn cửa sổ, nên xuất hiện ở mọi hàng.](../img/conv-im2col.svg)

<!--en:76713479 kind=paragraph-->
The trade-off is extra workspace memory, since each interior pixel is duplicated
into up to $k_\textrm{h} k_\textrm{w}$ patches. Below we verify on
:numref:`fig_correlation` that this single matrix multiply matches `corr2d`
exactly:
<!--tr:76713479-->
Sự đánh đổi ở đây là tốn thêm bộ nhớ đệm làm việc, vì mỗi điểm ảnh bên trong bị nhân bản
vào tối đa $k_\textrm{h} k_\textrm{w}$ vùng con. Dưới đây, ta kiểm chứng trên ví dụ ở
:numref:`fig_correlation` rằng phép nhân ma trận đơn lẻ này khớp chính xác với `corr2d`:

<!--en:3bcd7d8a kind=paragraph-->
The discrepancy is identically zero: one matrix product reproduces `corr2d`
perfectly.
<!--tr:3bcd7d8a-->
Độ sai lệch bằng không tuyệt đối: một tích ma trận duy nhất tái hiện `corr2d`
một cách hoàn hảo.

<!--en:c30f2dea kind=paragraph-->
Equivalently, we can unroll the *kernel* instead of the input into a sparse
banded matrix $\mathbf{W}$ multiplying the flattened image—the exact 2D analog
of panel (c) in :numref:`fig_why_conv_weight_sharing`, where every row repeats
the same handful of kernel weights. Seeing a convolution as a structured dense
layer brings us full circle to the derivation of :numref:`sec_why-conv` (and the
exercises build $\mathbf{W}$ explicitly).
<!--tr:c30f2dea-->
Một cách tương đương, ta có thể trải *hạt nhân* thay vì đầu vào thành một ma trận
dải thưa $\mathbf{W}$ nhân với ảnh trải phẳng—phiên bản tương tự 2D chính xác
của hình (c) trong :numref:`fig_why_conv_weight_sharing`, nơi mỗi hàng lặp lại
cùng một nhóm nhỏ các trọng số hạt nhân. Nhìn nhận phép tích chập như một tầng dày đặc
có cấu trúc đưa chúng ta khép kín vòng tròn quay lại phép suy dẫn ở :numref:`sec_why-conv` (và phần
bài tập sẽ xây dựng tường minh $\mathbf{W}$).

<!--en:ad4fc956 kind=heading-->
## Feature Maps and Receptive Fields
<!--tr:ad4fc956-->
## Bản đồ đặc trưng và vùng tiếp nhận

<!--en:e7983df8 kind=paragraph-->
A single convolutional layer sees only a patch the size of its kernel, whereas
recognizing whole objects demands integrating context across much wider areas.
Recall from :numref:`subsec_why-conv-channels` that a convolutional layer's
output is called a *feature map* (like our vertical edge map `Y`). For any
element $x$ in a feature map, its *receptive field* is the region of the input
image (or of an earlier layer) whose values can influence $x$ during forward
propagation—borrowing the neurophysiology term from :numref:`field_visual`.
<!--tr:e7983df8-->
Một tầng tích chập đơn lẻ chỉ nhìn thấy một vùng con có kích thước bằng hạt nhân của nó, trong khi
việc nhận dạng toàn bộ vật thể đòi hỏi phải tích hợp ngữ cảnh trên những vùng rộng hơn nhiều.
Nhắc lại từ :numref:`subsec_why-conv-channels` rằng đầu ra của một tầng tích chập
được gọi là *bản đồ đặc trưng* (như bản đồ cạnh dọc `Y` của chúng ta). Với bất kỳ
phần tử $x$ nào trong bản đồ đặc trưng, *vùng tiếp nhận* của nó là vùng trên ảnh
đầu vào (hoặc trên một tầng trước đó) mà các giá trị tại đó có thể ảnh hưởng đến $x$ trong lượt
truyền xuôi—mượn thuật ngữ sinh lý học thần kinh từ :numref:`field_visual`.

<!--en:b36f221e kind=heading-->
### Stacked Layers
<!--tr:b36f221e-->
### Các tầng xếp chồng

<!--en:24a25ec4 kind=paragraph-->
In :numref:`fig_correlation`, the shaded output `19` has a $2 \times 2$ receptive
field covering the top-left four pixels of $\mathbf{X}$. Now stack a *second*
$2 \times 2$ convolutional layer on top of the $2 \times 2$ output $\mathbf{Y}$
to produce a single scalar $z$: $z$ depends on all four elements of $\mathbf{Y}$,
which together depend on all nine elements of $\mathbf{X}$. Stacking layers
expands the receptive field smoothly with depth.
<!--tr:24a25ec4-->
Trong :numref:`fig_correlation`, phần tử đầu ra tô đậm `19` có vùng tiếp nhận $2 \times 2$
bao phủ bốn điểm ảnh góc trên bên trái của $\mathbf{X}$. Bây giờ, hãy xếp chồng một tầng tích chập *thứ hai*
kích thước $2 \times 2$ lên trên đầu ra $2 \times 2$ là $\mathbf{Y}$
để tạo ra một số vô hướng duy nhất $z$: $z$ phụ thuộc vào cả bốn phần tử của $\mathbf{Y}$,
mà các phần tử này lại phụ thuộc vào cả chín phần tử của $\mathbf{X}$. Việc xếp chồng các tầng
mở rộng vùng tiếp nhận một cách đều đặn theo độ sâu.

<!--en:f95c9394 kind=paragraph-->
How quickly does it grow? Each $k \times k$ convolutional layer reaches $k - 1$
pixels farther than a single element of its input, so stacking $L$ layers with
kernel sizes $k_1, \ldots, k_L$ yields a receptive field of side length
<!--tr:f95c9394-->
Nó mở rộng nhanh đến mức nào? Mỗi tầng tích chập $k \times k$ vươn xa hơn $k - 1$
điểm ảnh so với một phần tử đơn lẻ của đầu vào, do đó việc xếp chồng $L$ tầng có
kích thước nhân tích chập $k_1, \ldots, k_L$ tạo ra vùng tiếp nhận có độ dài cạnh là

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
Với một khối gồm $L$ tầng $3 \times 3$ liên tiếp, mỗi tầng cộng thêm 2 điểm ảnh,
bao phủ $(2L + 1) \times (2L + 1)$ đầu vào: $5 \times 5$ sau hai tầng và
$7 \times 7$ sau ba tầng (:numref:`fig_conv_receptive_field`). Mức tăng trưởng
tuyến tính này thực chất là một lợi thế chứ không phải lý do để tìm đến các hạt nhân khổng lồ:
ba tầng $3 \times 3$ xếp chồng đạt được cùng vùng tiếp nhận $7 \times 7$
như một tầng $7 \times 7$ duy nhất nhưng dùng ít tham số hơn hẳn ($3 \times 9 = 27$
trọng số so với 49 trên mỗi cặp kênh), đồng thời đan xen ba phép biến đổi phi tuyến thay
vì một.

<!--en:39917513 kind=caption-->
![Receptive field of one unit after three one-dimensional layers with kernel size 3. Shaded cells are the units that the top unit depends on, and the lines connect each of them to its inputs. Each layer widens the receptive field by 2, to 7 inputs.](../img/conv-receptive-field.svg)
<!--tr:39917513-->
![Vùng tiếp nhận của một đơn vị sau ba tầng một chiều với kích thước nhân tích chập bằng 3. Các ô tô đậm là những đơn vị mà đơn vị trên cùng phụ thuộc vào, và các đường kẻ nối mỗi đơn vị đó với các đầu vào của nó. Mỗi tầng mở rộng vùng tiếp nhận thêm 2, lên thành 7 đầu vào.](../img/conv-receptive-field.svg)

<!--en:48e4c9cc kind=heading-->
### Effective Receptive Fields
<!--tr:48e4c9cc-->
### Vùng tiếp nhận hiệu dụng

<!--en:77d044db kind=paragraph-->
Keep in mind that $r$ measures the *theoretical* receptive field (every input
pixel with *any* path to an output unit) without saying how strongly each pixel
actually contributes. Measuring input gradients directly,
:citet:`Luo.Li.Urtasun.ea.2016` showed that the *effective* receptive field
decays sharply away from the center in a roughly Gaussian bell curve, in both
untrained and trained networks. Deep units can theoretically see far across the
image, but their sensitivity concentrates heavily near the center.
<!--tr:77d044db-->
Cần lưu ý rằng $r$ đo vùng tiếp nhận *lý thuyết* (mọi điểm ảnh đầu vào
có *bất kỳ* đường truyền nào tới một đơn vị đầu ra) mà không cho biết mức độ đóng góp
thực sự của từng điểm ảnh mạnh đến đâu. Đo trực tiếp gradient theo đầu vào,
:citet:`Luo.Li.Urtasun.ea.2016` đã chỉ ra rằng vùng tiếp nhận *hiệu dụng*
suy giảm mạnh khi ra xa tâm theo đường cong hình chuông xấp xỉ dạng Gauss, ở cả
mạng chưa huấn luyện lẫn mạng đã huấn luyện. Về mặt lý thuyết, các đơn vị ở tầng sâu có thể nhìn bao quát rất xa trên
ảnh, nhưng độ nhạy của chúng lại tập trung dày đặc quanh vùng tâm.

<!--en:eed4e2a0 kind=paragraph-->
Counting paths explains the bell shape. An input at the edge of the receptive
field reaches the output unit along a single chain of connections, whereas the
center input can take many routes through the layers in between.
:numref:`fig_conv_rf_paths` counts them for the tower of
:numref:`fig_conv_receptive_field`: the center input has seven paths to the top
unit and each edge input has one. If all weights have a similar magnitude, the
influence of an input grows with its number of paths.
<!--tr:eed4e2a0-->
Việc đếm số đường truyền giải thích cho hình dạng quả chuông này. Một đầu vào ở rìa vùng tiếp nhận
chỉ đi tới đơn vị đầu ra dọc theo một chuỗi kết nối duy nhất, trong khi đầu vào ở
chính giữa có thể đi theo nhiều lộ trình qua các tầng trung gian.
:numref:`fig_conv_rf_paths` đếm số đường truyền này cho tháp mạng ở
:numref:`fig_conv_receptive_field`: đầu vào ở tâm có bảy đường truyền tới đơn vị trên cùng,
còn mỗi đầu vào ở rìa chỉ có một. Nếu mọi trọng số có độ lớn tương đương nhau,
mức độ ảnh hưởng của một đầu vào sẽ tăng theo số lượng đường truyền của nó.

<!--en:b92775ee kind=caption-->
![Number of paths from each unit to the top unit of :numref:`fig_conv_receptive_field` (three one-dimensional layers with kernel size 3). Each count is the sum of the counts of the units that the cell feeds. Darker cells have more paths than the other cells of their layer, and the bars repeat the counts of the input row: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)
<!--tr:b92775ee-->
![Số đường truyền từ mỗi đơn vị tới đơn vị trên cùng của :numref:`fig_conv_receptive_field` (ba tầng một chiều với kích thước nhân tích chập bằng 3). Mỗi số đếm bằng tổng các số đếm của những đơn vị mà ô đó truyền tới. Các ô sẫm màu hơn có nhiều đường truyền hơn các ô khác trong cùng tầng, và biểu đồ cột lặp lại các số đếm của hàng đầu vào: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)

<!--en:405695e8 kind=paragraph-->
Two practical gaps remain in our basic layer: every unpadded convolution shrinks
spatial dimensions while under-sampling border pixels, and a single kernel
detects only one pattern. We address spatial sizing and borders next in
:numref:`sec_padding`, and multi-channel feature maps in :numref:`sec_channels`.
<!--tr:405695e8-->
Còn lại hai hạn chế thực tế trong tầng cơ bản của chúng ta: mỗi phép tích chập không đệm đều làm co
kích thước không gian đồng thời lấy mẫu thiếu các điểm ảnh ở viền, và một hạt nhân đơn lẻ
chỉ phát hiện được một mẫu duy nhất. Chúng ta sẽ giải quyết vấn đề kích thước không gian và viền ảnh tiếp theo trong
:numref:`sec_padding`, cũng như bản đồ đặc trưng đa kênh trong :numref:`sec_channels`.

<!--en:30ac03ff kind=heading-->
## Summary
<!--tr:30ac03ff-->
## Tóm tắt

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
Một tầng tích chập áp dụng phép tương quan chéo hai chiều với một hạt nhân học được
cộng thêm một hệ số điều chỉnh là số vô hướng, trượt hạt nhân qua mọi cửa sổ hợp lệ của
đầu vào. Một hạt nhân sai phân hữu hạn $1 \times 2$ đơn giản `[1, -1]` phát hiện được các cạnh dọc
ở bất kỳ đâu trong ảnh, và hạ gradient khôi phục chính xác các trọng số đó
chỉ từ dữ liệu đầu vào–đầu ra, giống như cách các mạng nơ-ron tích chập (CNN) sâu học toàn bộ các tầng phân cấp
bộ lọc từ đầu đến cuối. Phép tích chập toán học lật ngược hạt nhân, nên không tạo ra
khác biệt nào một khi các hạt nhân được học từ dữ liệu. Phép tích chập có thể được tính dưới dạng một
phép nhân ma trận duy nhất (*im2col*), và việc xếp chồng $L$ tầng tích chập
mở rộng tuyến tính vùng tiếp nhận của mỗi đơn vị thêm $k_i - 1$ sau mỗi tầng.

<!--en:3ba929d2 kind=slide_title-->
::: {.slide title="Fast Execution: Convolution via Matrix Multiplication"}
<!--tr:3ba929d2-->
::: {.slide title="Thực thi nhanh: Tích chập qua phép nhân ma trận"}

<!--en:1711f812 kind=kicker-->
[The im2col (Image-to-Columns) Transformation]{.kicker}
<!--tr:1711f812-->
[Phép biến đổi im2col (Image-to-Columns)]{.kicker}

<!--en:a040f3d7 kind=slide_text-->
- **Single dense GEMM:** Flatten each $k_\textrm{h} \times k_\textrm{w}$ window into a row of `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ and multiply by $\textrm{vec}(\mathbf{K})$.
<!--tr:a040f3d7-->
- **Một phép GEMM dày đặc duy nhất:** Trải phẳng mỗi cửa sổ $k_\textrm{h} \times k_\textrm{w}$ thành một hàng của `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ rồi nhân với $\textrm{vec}(\mathbf{K})$.

<!--en:1fb18ec4 kind=slide_text-->
- **Trade-off:** Peak GPU Tensor Core speed at the cost of $k_\textrm{h} k_\textrm{w}\times$ patch workspace duplication.
<!--tr:1fb18ec4-->
- **Sự đánh đổi:** Đạt tốc độ đỉnh trên Tensor Core của GPU đổi lại chi phí nhân bản bộ nhớ đệm vùng con gấp $k_\textrm{h} k_\textrm{w}\times$.

<!--en:0d757cf6 kind=slide_title-->
::: {.slide title="Dual View: Unrolling the Kernel into a Banded Matrix"}
<!--tr:0d757cf6-->
::: {.slide title="Góc nhìn đối ngẫu: Trải hạt nhân thành ma trận dải"}

<!--en:7f0defaa kind=kicker-->
[Toeplitz Weight Matrix Multiplication]{.kicker}
<!--tr:7f0defaa-->
[Phép nhân ma trận trọng số Toeplitz]{.kicker}

<!--en:5898ba57 kind=slide_text-->
Instead of unfolding the input image into duplicated patches, we can unroll the *kernel* into a sparse banded matrix $\mathbf{W}$ that multiplies the flattened input vector $\textrm{vec}(\mathbf{X})$:
<!--tr:5898ba57-->
Thay vì bung ảnh đầu vào thành các vùng con nhân bản, ta có thể trải *hạt nhân* thành một ma trận dải thưa $\mathbf{W}$ nhân với vector đầu vào trải phẳng $\textrm{vec}(\mathbf{X})$:

<!--en:dae51741 kind=slide_text-->
- **Matrix–vector form:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, where $\mathbf{W}$ has $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ rows and $n_\textrm{h} n_\textrm{w}$ columns.
<!--tr:dae51741-->
- **Dạng ma trận–vector:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, trong đó $\mathbf{W}$ có $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ hàng và $n_\textrm{h} n_\textrm{w}$ cột.

<!--en:f0e86ee3 kind=slide_text-->
- **Connection to §{sec_why-conv} and backprop:** Every row of $\mathbf{W}$ holds the same $k_\textrm{h} k_\textrm{w}$ kernel entries shifted across columns (panel c), making the backward pass with respect to $\mathbf{X}$ simply multiplication by $\mathbf{W}^\top$ (a transposed convolution).
<!--tr:f0e86ee3-->
- **Liên hệ với §{sec_why-conv} và lan truyền ngược:** Mỗi hàng của $\mathbf{W}$ chứa cùng $k_\textrm{h} k_\textrm{w}$ phần tử hạt nhân dịch chuyển qua các cột (hình c), biến lượt truyền ngược theo $\mathbf{X}$ đơn giản thành phép nhân với $\mathbf{W}^\top$ (một phép tích chập chuyển vị).

<!--en:af8cf19a kind=slide_title-->
::: {.slide title="Feature Maps and Receptive Fields Across Depth"}
<!--tr:af8cf19a-->
::: {.slide title="Bản đồ đặc trưng và vùng tiếp nhận theo độ sâu"}

<!--en:4eb965f3 kind=kicker-->
[Linear Receptive-Field Growth in Stacked Layers]{.kicker}
<!--tr:4eb965f3-->
[Tăng trưởng tuyến tính vùng tiếp nhận ở các tầng xếp chồng]{.kicker}

<!--en:5599a6e9 kind=slide_text-->
A unit's **receptive field** is the region of the original input image whose pixels can influence that unit's activation during forward propagation:
<!--tr:5599a6e9-->
**Vùng tiếp nhận** của một đơn vị là vùng trên ảnh đầu vào gốc mà các điểm ảnh tại đó có thể ảnh hưởng đến giá trị kích hoạt của đơn vị đó trong lượt truyền xuôi:

<!--en:eba74f75 kind=slide_text-->
- **Single layer ($L=1$):** A $k_1 \times k_1$ kernel sees a $k_1 \times k_1$ input patch.
<!--tr:eba74f75-->
- **Một tầng đơn ($L=1$):** Hạt nhân $k_1 \times k_1$ nhìn thấy một vùng con đầu vào $k_1 \times k_1$.

<!--en:01c175bf kind=slide_text-->
- **Stacking $L$ stride-1 layers:** Layer $i$ with kernel size $k_i$ reaches $k_i - 1$ pixels farther along each axis than a single element of layer $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$
<!--tr:01c175bf-->
- **Xếp chồng $L$ tầng có sải bước bằng 1:** Tầng $i$ có kích thước nhân tích chập $k_i$ vươn xa hơn $k_i - 1$ điểm ảnh dọc theo mỗi trục so với một phần tử đơn lẻ của tầng $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$

<!--en:2bde5822 kind=slide_text-->
- **$L$ stacked $3 \times 3$ layers:** Each layer adds $3 - 1 = 2$ pixels, giving side length $r = 2L + 1$ ($3 \to 5 \to 7$).
<!--tr:2bde5822-->
- **$L$ tầng $3 \times 3$ xếp chồng:** Mỗi tầng cộng thêm $3 - 1 = 2$ điểm ảnh, cho độ dài cạnh $r = 2L + 1$ ($3 \to 5 \to 7$).

<!--en:cbb9e97a kind=slide_title-->
::: {.slide title="Why Deep Stacks of 3×3 Kernels Beat One Large Kernel"}
<!--tr:cbb9e97a-->
::: {.slide title="Vì sao xếp chồng hạt nhân 3×3 vượt trội một hạt nhân lớn"}

<!--en:95e46e70 kind=kicker-->
[Parameter Efficiency & Nonlinear Expressivity]{.kicker}
<!--tr:95e46e70-->
[Hiệu quả tham số & Khả năng biểu diễn phi tuyến]{.kicker}

<!--en:bd1d7cc0 kind=slide_text-->
- **Matched $7 \times 7$ receptive field:** Three stacked $3 \times 3$ layers ($r = 1 + 2 + 2 + 2 = 7$) see the exact same $7 \times 7$ patch as one $7 \times 7$ layer.
<!--tr:bd1d7cc0-->
- **Cùng vùng tiếp nhận $7 \times 7$:** Ba tầng $3 \times 3$ xếp chồng ($r = 1 + 2 + 2 + 2 = 7$) nhìn thấy đúng cùng vùng con $7 \times 7$ như một tầng $7 \times 7$.

<!--en:755ac11a kind=slide_text-->
- **45% fewer parameters:** $3 \times 9 c^2 = 27 c^2$ weights instead of $49 c^2$ across $c$ channels.
<!--tr:755ac11a-->
- **Ít hơn 45% tham số:** $3 \times 9 c^2 = 27 c^2$ trọng số thay vì $49 c^2$ trên $c$ kênh.

<!--en:e3aa1086 kind=slide_text-->
- **Richer expressivity:** Compounds **3 nonlinear activations** instead of 1 (the core VGG design rule in §{sec_blocks}).
<!--tr:e3aa1086-->
- **Biểu diễn phong phú hơn:** Kết hợp **3 hàm kích hoạt phi tuyến** thay vì 1 (quy tắc thiết kế cốt lõi của VGG trong §{sec_blocks}).

<!--en:9ec83bf6 kind=slide_title-->
::: {.slide title="Theoretical vs. Effective Receptive Fields"}
<!--tr:9ec83bf6-->
::: {.slide title="Vùng tiếp nhận lý thuyết so với hiệu dụng"}

<!--en:72a52c3b kind=kicker-->
[Gaussian Decay of Sensitivity Away from the Center]{.kicker}
<!--tr:72a52c3b-->
[Độ nhạy suy giảm dạng Gauss khi ra xa tâm]{.kicker}

<!--en:33bb377d kind=slide_text-->
- **Theoretical RF ($r = 2L+1$):** Bounding box of all pixels with *at least one* path to the output unit.
<!--tr:33bb377d-->
- **Vùng tiếp nhận lý thuyết ($r = 2L+1$):** Khung chứa mọi điểm ảnh có *ít nhất một* đường truyền tới đơn vị đầu ra.

<!--en:293436eb kind=slide_text-->
- **Central Limit Theorem:** Center pixels connect via exponentially many paths, whereas corner pixels connect via only **1 path**.
<!--tr:293436eb-->
- **Định lý giới hạn trung tâm:** Các điểm ảnh ở tâm kết nối qua số lượng đường truyền tăng theo hàm mũ, trong khi điểm ảnh ở góc chỉ kết nối qua **1 đường truyền**.

<!--en:f47d181e kind=slide_text-->
- **Effective RF (Luo et al., 2016):** Input gradient magnitude $\partial y / \partial \mathbf{X}$ decays in a Gaussian bell curve with std $\sigma \propto \sqrt{L}$.
<!--tr:f47d181e-->
- **Vùng tiếp nhận hiệu dụng (Luo et al., 2016):** Độ lớn gradient đầu vào $\partial y / \partial \mathbf{X}$ suy giảm theo đường cong hình chuông Gauss với độ lệch chuẩn $\sigma \propto \sqrt{L}$.

<!--en:fac792d0 kind=slide_title-->
::: {.slide title="Concept Check: Cross-Correlation & Receptive Fields"}
<!--tr:fac792d0-->
::: {.slide title="Kiểm tra khái niệm: Tương quan chéo & Vùng tiếp nhận"}

<!--en:26a317d8 kind=kicker-->
[Concept Check]{.kicker}
<!--tr:26a317d8-->
[Kiểm tra khái niệm]{.kicker}

<!--en:8fe554e5 kind=slide_text-->
Test your understanding of output shapes, `im2col`, and receptive-field arithmetic:
<!--tr:8fe554e5-->
Kiểm tra mức độ hiểu của bạn về kích thước đầu ra, `im2col` và phép tính vùng tiếp nhận:

<!--en:281c7c78 kind=slide_text-->
1. **im2col matrix dimensions:** You apply an unpadded $3 \times 3$ single-channel kernel to a $10 \times 10$ image via `im2col`. What is the output spatial shape, and what are the exact dimensions of the `patches` matrix and flattened kernel vector?
<!--tr:281c7c78-->
1. **Kích thước ma trận im2col:** Bạn áp dụng một hạt nhân đơn kênh $3 \times 3$ không đệm lên ảnh $10 \times 10$ bằng `im2col`. Kích thước không gian của đầu ra là bao nhiêu, và kích thước chính xác của ma trận `patches` cùng vector hạt nhân trải phẳng là gì?

<!--en:16976d18 kind=slide_text-->
2. **Depth to span Fashion-MNIST:** At stride 1, how many stacked $3 \times 3$ layers are required so a single unit's theoretical receptive field spans an entire $28 \times 28$ image? How many with $5 \times 5$ layers?
<!--tr:16976d18-->
2. **Độ sâu để bao phủ Fashion-MNIST:** Với sải bước bằng 1, cần bao nhiêu tầng $3 \times 3$ xếp chồng để vùng tiếp nhận lý thuyết của một đơn vị bao phủ toàn bộ ảnh $28 \times 28$? Cần bao nhiêu nếu dùng tầng $5 \times 5$?

<!--en:ab2e0d6a kind=slide_text-->
3. **Edge-loss boundary shift:** In our $6 \times 8$ vertical-edge learning experiment, why did the loss start at $\approx 12$, and what would the initial loss be if the black band extended all the way to the right edge (`X[:, 2:8] = 0`)?
<!--tr:ab2e0d6a-->
3. **Thay đổi mất mát tại biên cạnh:** Trong thí nghiệm học cạnh dọc $6 \times 8$, vì sao mất mát bắt đầu ở $\approx 12$, và mất mát ban đầu sẽ bằng bao nhiêu nếu dải đen kéo dài hết tới tận mép phải (`X[:, 2:8] = 0`)?

<!--en:250d6877 kind=slide_text-->
**Answers:**
(1) Output shape is $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ outputs). The `patches` matrix has shape $(64, 9)$ and multiplies the $(9, 1)$ kernel vector.
(2) For $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ layers. For $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ layers.
(3) Each of the 6 rows had two $\pm 1$ edges ($6 \times 2 \times 1^2 = 12$). Extending the black band to the right border eliminates the right-hand black-to-white edge, leaving one edge per row and halving the initial loss to $\approx 6$.
<!--tr:250d6877-->
**Đáp án:**
(1) Kích thước đầu ra là $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ phần tử đầu ra). Ma trận `patches` có kích thước $(64, 9)$ và nhân với vector hạt nhân $(9, 1)$.
(2) Với $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ tầng. Với $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ tầng.
(3) Mỗi hàng trong 6 hàng có hai cạnh $\pm 1$ ($6 \times 2 \times 1^2 = 12$). Kéo dài dải đen tới sát viền phải sẽ loại bỏ cạnh đen-sang-trắng bên phải, chỉ còn một cạnh trên mỗi hàng nên giảm một nửa mất mát ban đầu xuống $\approx 6$.

<!--en:ba43a2dd kind=slide_title-->
::: {.slide title="Summary & Key Takeaways"}
<!--tr:ba43a2dd-->
::: {.slide title="Tóm tắt & Điểm cốt lõi"}

<!--en:ce12e85f kind=kicker-->
[Section Synthesis]{.kicker}
<!--tr:ce12e85f-->
[Tổng hợp mục]{.kicker}

<!--en:00011fb7 kind=slide_text-->
- **2D Cross-Correlation & Edge Learning:** Slides $\mathbf{K}$ across valid windows ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); SGD learns $[1, -1]$ in 9 steps.
<!--tr:00011fb7-->
- **Tương quan chéo 2D & Học cạnh:** Trượt $\mathbf{K}$ qua các cửa sổ hợp lệ ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); hạ gradient ngẫu nhiên (SGD) học được $[1, -1]$ sau 9 bước.

<!--en:09efe4f0 kind=slide_text-->
- **im2col Matrix Multiply:** Trades $k_\textrm{h} k_\textrm{w}\times$ patch duplication for peak GPU GEMM throughput.
<!--tr:09efe4f0-->
- **Nhân ma trận im2col:** Đánh đổi việc nhân bản vùng con gấp $k_\textrm{h} k_\textrm{w}\times$ để đạt thông lượng GEMM đỉnh trên GPU.

<!--en:3af8acd0 kind=slide_text-->
- **Stacked Receptive Fields:** $r = 1 + \sum (k_i - 1) = 2L+1$ for $3 \times 3$ stacks, with Gaussian center sensitivity.
<!--tr:3af8acd0-->
- **Vùng tiếp nhận xếp chồng:** $r = 1 + \sum (k_i - 1) = 2L+1$ với các khối $3 \times 3$, cùng độ nhạy tập trung ở tâm dạng Gauss.

=== DECISIONS ===
- Register & pronouns: Use academic, natural Vietnamese prose addressing the reader collegially with "chúng ta" / "ta" for inclusive "we" and "bạn" (or imperative "Hãy ...") for "you".
- Numbers & punctuation: Keep decimal points ("0.01", "0.03", "12.0") as in English per TRANSLATING.md §4; no trailing periods on headings.
- "locality" → "tính cục bộ"
- "edge map" → "bản đồ cạnh"
- "edge detection" / "edge detector" → "phát hiện cạnh" / "bộ phát hiện cạnh"
- "sliding window" / "window" → "cửa sổ trượt" / "cửa sổ"
- "patch" / "patch matrix" → "vùng con" / "ma trận vùng con"
- "dense layer" → "tầng dày đặc"
- "elementwise" → "theo từng phần tử"
- "reshape" → "định hình lại"
- "filter bank" → "ngân hàng bộ lọc"
- "framework" → "framework"
- "flatten" / "flattened" → "trải phẳng"
- "unfold (input)" / "unroll (kernel)" → "bung (đầu vào)" / "trải (hạt nhân)"
- "banded matrix" / "sparse banded matrix" → "ma trận dải" / "ma trận dải thưa"
- "theoretical receptive field" / "effective receptive field" → "vùng tiếp nhận lý thuyết" / "vùng tiếp nhận hiệu dụng"
- "nonlinear activations" (as activation functions/stages, vs. unit output "giá trị kích hoạt") → "hàm kích hoạt phi tuyến"

=== NOTES ===
- eed4e2a0: false glossary match on "shape" ("the bell shape" refers to geometric curve form → "hình dạng quả chuông", not array/tensor axis shape "kích thước").
