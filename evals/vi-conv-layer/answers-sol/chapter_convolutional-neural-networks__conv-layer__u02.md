<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u02 lang=vi rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:0b85caf9 kind=heading-->
## Convolution as Matrix Multiplication
<!--tr:0b85caf9-->
## Tích chập dưới dạng phép nhân ma trận

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
Hai vòng lặp lồng nhau trong `corr2d` giúp ta thấy rõ phép tính, nhưng duyệt từng điểm ảnh đầu ra bằng Python thì quá chậm cho các bài toán thực tế. Một cách kinh điển mà các thư viện vẫn dùng là chuyển phép tương quan chéo thành một phép nhân ma trận lớn duy nhất. Đây chính là phép toán mà GPU thực hiện với tốc độ cao nhất. Hãy để ý rằng mỗi phần tử đầu ra trong :numref:`fig_correlation` chỉ là tích vô hướng giữa hai vector: hạt nhân gồm $k_\textrm{h} k_\textrm{w}$ phần tử được trải phẳng và cửa sổ đầu vào tương ứng cũng được trải phẳng. Trích tất cả $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ cửa sổ rồi xếp mỗi cửa sổ thành một hàng, ta thu được ma trận các vùng ảnh. Nhân ma trận đó với hạt nhân đã trải phẳng sẽ tính được toàn bộ phép tương quan chéo trong một lần. Cách sắp xếp lại dữ liệu này thường được gọi là *im2col* (“chuyển ảnh thành các cột”, dù ở đây ta xếp thành các hàng). :numref:`fig_conv_im2col` minh họa cách làm này trên ví dụ ở :numref:`fig_correlation`.

<!--en:107554a5 kind=caption-->
![The cross-correlation of :numref:`fig_correlation` as one matrix product. Each of the four $2 \times 2$ windows of the input becomes a row of the patch matrix, and multiplying by the flattened kernel gives the flattened output. The shaded cells follow the first window to its row and to the output 19. The center pixel 4 lies in all four windows, so it appears in every row.](../img/conv-im2col.svg)
<!--tr:107554a5-->
![Phép tương quan chéo trong :numref:`fig_correlation` dưới dạng một tích ma trận. Mỗi cửa sổ $2 \times 2$ trong bốn cửa sổ đầu vào trở thành một hàng của ma trận các vùng ảnh. Nhân ma trận này với hạt nhân đã trải phẳng cho ta đầu ra cũng ở dạng trải phẳng. Các ô tô màu cho thấy cửa sổ thứ nhất tương ứng với hàng nào và cho ra giá trị 19 như thế nào. Điểm ảnh ở giữa có giá trị 4 nằm trong cả bốn cửa sổ, nên xuất hiện ở mọi hàng.](../img/conv-im2col.svg)

<!--en:76713479 kind=paragraph-->
The trade-off is extra workspace memory, since each interior pixel is duplicated
into up to $k_\textrm{h} k_\textrm{w}$ patches. Below we verify on
:numref:`fig_correlation` that this single matrix multiply matches `corr2d`
exactly:
<!--tr:76713479-->
Đổi lại, ta cần thêm bộ nhớ làm việc vì mỗi điểm ảnh bên trong có thể được chép vào tới $k_\textrm{h} k_\textrm{w}$ vùng ảnh. Dưới đây, chúng ta dùng ví dụ ở :numref:`fig_correlation` để kiểm tra xem phép nhân ma trận duy nhất này có cho kết quả giống hệt `corr2d` hay không:

<!--en:3bcd7d8a kind=paragraph-->
The discrepancy is identically zero: one matrix product reproduces `corr2d`
perfectly.
<!--tr:3bcd7d8a-->
Sai khác bằng đúng 0. Một tích ma trận duy nhất cho kết quả giống hệt `corr2d`.

<!--en:c30f2dea kind=paragraph-->
Equivalently, we can unroll the *kernel* instead of the input into a sparse
banded matrix $\mathbf{W}$ multiplying the flattened image—the exact 2D analog
of panel (c) in :numref:`fig_why_conv_weight_sharing`, where every row repeats
the same handful of kernel weights. Seeing a convolution as a structured dense
layer brings us full circle to the derivation of :numref:`sec_why-conv` (and the
exercises build $\mathbf{W}$ explicitly).
<!--tr:c30f2dea-->
Một cách tương đương là khai triển *hạt nhân* thay cho đầu vào thành ma trận dải thưa $\mathbf{W}$ để nhân với ảnh đã trải phẳng. Đây chính là phiên bản hai chiều của ô (c) trong :numref:`fig_why_conv_weight_sharing`, với mỗi hàng lặp lại cùng một nhóm nhỏ trọng số hạt nhân. Nhìn phép tích chập như một tầng kết nối đầy đủ có cấu trúc đưa chúng ta trở lại phép suy dẫn trong :numref:`sec_why-conv` (ở phần bài tập, chúng ta sẽ xây dựng tường minh $\mathbf{W}$).

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
Một tầng tích chập chỉ nhìn thấy một vùng ảnh có kích thước bằng hạt nhân của nó, trong khi nhận dạng trọn vẹn một vật thể đòi hỏi kết hợp thông tin trên vùng rộng hơn nhiều. Như đã thấy trong :numref:`subsec_why-conv-channels`, đầu ra của tầng tích chập được gọi là *bản đồ đặc trưng* (chẳng hạn bản đồ cạnh dọc `Y`). Với một phần tử $x$ bất kỳ trong bản đồ đặc trưng, *vùng tiếp nhận* của nó là vùng trên ảnh đầu vào (hoặc ở một tầng trước đó) có các giá trị có thể ảnh hưởng đến $x$ trong lượt truyền xuôi. Tên gọi này mượn từ thuật ngữ sinh lý học thần kinh trong :numref:`field_visual`.

<!--en:b36f221e kind=heading-->
### Stacked Layers
<!--tr:b36f221e-->
### Xếp chồng các tầng

<!--en:24a25ec4 kind=paragraph-->
In :numref:`fig_correlation`, the shaded output `19` has a $2 \times 2$ receptive
field covering the top-left four pixels of $\mathbf{X}$. Now stack a *second*
$2 \times 2$ convolutional layer on top of the $2 \times 2$ output $\mathbf{Y}$
to produce a single scalar $z$: $z$ depends on all four elements of $\mathbf{Y}$,
which together depend on all nine elements of $\mathbf{X}$. Stacking layers
expands the receptive field smoothly with depth.
<!--tr:24a25ec4-->
Trong :numref:`fig_correlation`, ô đầu ra tô màu `19` có vùng tiếp nhận $2 \times 2$ gồm bốn điểm ảnh ở góc trên bên trái của $\mathbf{X}$. Bây giờ, đặt thêm tầng tích chập $2 \times 2$ *thứ hai* lên đầu ra $2 \times 2$ là $\mathbf{Y}$ để thu được một số vô hướng duy nhất $z$. Giá trị $z$ phụ thuộc vào cả bốn phần tử của $\mathbf{Y}$, và bốn phần tử này, xét chung, phụ thuộc vào cả chín phần tử của $\mathbf{X}$. Khi xếp chồng các tầng, vùng tiếp nhận mở rộng dần theo độ sâu.

<!--en:f95c9394 kind=paragraph-->
How quickly does it grow? Each $k \times k$ convolutional layer reaches $k - 1$
pixels farther than a single element of its input, so stacking $L$ layers with
kernel sizes $k_1, \ldots, k_L$ yields a receptive field of side length
<!--tr:f95c9394-->
Vùng tiếp nhận mở rộng nhanh đến mức nào? Mỗi tầng tích chập $k \times k$ mở rộng vùng tiếp nhận thêm $k - 1$ điểm ảnh dọc theo mỗi trục so với vùng tiếp nhận của một phần tử ở đầu vào của tầng đó. Vì vậy, khi xếp chồng $L$ tầng với kích thước nhân tích chập lần lượt là $k_1, \ldots, k_L$, ta được vùng tiếp nhận có độ dài cạnh là

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
Khi xếp chồng $L$ tầng $3 \times 3$ liên tiếp, mỗi tầng làm độ dài cạnh tăng thêm 2 điểm ảnh. Vùng tiếp nhận bao phủ $(2L + 1) \times (2L + 1)$ phần tử đầu vào, tức $5 \times 5$ sau hai tầng và $7 \times 7$ sau ba tầng (:numref:`fig_conv_receptive_field`). Nhờ mức tăng tuyến tính này, chúng ta có thể mở rộng vùng tiếp nhận mà vẫn dùng hạt nhân nhỏ. Ba tầng $3 \times 3$ xếp chồng đạt cùng vùng tiếp nhận $7 \times 7$ như một tầng $7 \times 7$ duy nhất nhưng dùng ít tham số hơn nhiều ($3 \times 9 = 27$ so với 49 trọng số trên mỗi cặp kênh), đồng thời xen ba phép biến đổi phi tuyến thay vì một.

<!--en:39917513 kind=caption-->
![Receptive field of one unit after three one-dimensional layers with kernel size 3. Shaded cells are the units that the top unit depends on, and the lines connect each of them to its inputs. Each layer widens the receptive field by 2, to 7 inputs.](../img/conv-receptive-field.svg)
<!--tr:39917513-->
![Vùng tiếp nhận của một đơn vị sau ba tầng một chiều có kích thước nhân tích chập bằng 3. Các ô tô màu là những đơn vị mà đơn vị trên cùng phụ thuộc vào. Các đường nối mỗi đơn vị đó với đầu vào của nó. Mỗi tầng mở rộng vùng tiếp nhận thêm 2 phần tử, đến tổng cộng 7 phần tử đầu vào.](../img/conv-receptive-field.svg)

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
Hãy nhớ rằng $r$ đo kích thước vùng tiếp nhận *lý thuyết* (gồm mọi điểm ảnh đầu vào có *ít nhất một* đường dẫn đến đơn vị đầu ra) và không cho biết từng điểm ảnh thực sự đóng góp mạnh đến đâu. Bằng cách đo trực tiếp gradient theo đầu vào, :citet:`Luo.Li.Urtasun.ea.2016` cho thấy độ nhạy trong vùng tiếp nhận *hiệu dụng* giảm mạnh khi ra xa tâm theo một đường cong gần giống hình chuông Gauss, ở cả mạng chưa huấn luyện lẫn mạng đã huấn luyện. Về lý thuyết, các đơn vị ở tầng sâu có thể nhìn thấy một vùng rộng trên ảnh, nhưng độ nhạy của chúng tập trung chủ yếu gần tâm.

<!--en:eed4e2a0 kind=paragraph-->
Counting paths explains the bell shape. An input at the edge of the receptive
field reaches the output unit along a single chain of connections, whereas the
center input can take many routes through the layers in between.
:numref:`fig_conv_rf_paths` counts them for the tower of
:numref:`fig_conv_receptive_field`: the center input has seven paths to the top
unit and each edge input has one. If all weights have a similar magnitude, the
influence of an input grows with its number of paths.
<!--tr:eed4e2a0-->
Đếm số đường dẫn giúp giải thích dạng hình chuông. Một phần tử đầu vào ở rìa vùng tiếp nhận chỉ nối với đơn vị đầu ra qua một chuỗi kết nối duy nhất, còn phần tử ở giữa có thể đi theo nhiều đường dẫn qua các tầng trung gian. :numref:`fig_conv_rf_paths` đếm số đường dẫn trong chồng tầng ở :numref:`fig_conv_receptive_field`. Phần tử đầu vào ở giữa có bảy đường dẫn đến đơn vị trên cùng, còn mỗi phần tử ở rìa chỉ có một. Nếu mọi trọng số có độ lớn tương đương nhau, phần tử đầu vào nào có nhiều đường dẫn hơn sẽ có ảnh hưởng lớn hơn.

<!--en:b92775ee kind=caption-->
![Number of paths from each unit to the top unit of :numref:`fig_conv_receptive_field` (three one-dimensional layers with kernel size 3). Each count is the sum of the counts of the units that the cell feeds. Darker cells have more paths than the other cells of their layer, and the bars repeat the counts of the input row: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)
<!--tr:b92775ee-->
![Số đường dẫn từ mỗi đơn vị đến đơn vị trên cùng trong :numref:`fig_conv_receptive_field` (ba tầng một chiều có kích thước nhân tích chập bằng 3). Số đường dẫn của mỗi ô bằng tổng số đường dẫn của các đơn vị nhận đầu vào từ ô đó. Các ô đậm hơn có nhiều đường dẫn hơn các ô còn lại trong cùng tầng. Các thanh biểu diễn lại số đường dẫn của hàng đầu vào: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)

<!--en:405695e8 kind=paragraph-->
Two practical gaps remain in our basic layer: every unpadded convolution shrinks
spatial dimensions while under-sampling border pixels, and a single kernel
detects only one pattern. We address spatial sizing and borders next in
:numref:`sec_padding`, and multi-channel feature maps in :numref:`sec_channels`.
<!--tr:405695e8-->
Tầng cơ bản này còn hai hạn chế trong thực tế. Mỗi phép tích chập không đệm làm giảm kích thước không gian và lấy mẫu các điểm ảnh ở biên ít hơn, còn một hạt nhân chỉ phát hiện được một mẫu hình. Chúng ta sẽ bàn về kích thước không gian và cách xử lý biên ở :numref:`sec_padding`, rồi về bản đồ đặc trưng nhiều kênh ở :numref:`sec_channels`.

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
Tầng tích chập tính tương quan chéo hai chiều bằng cách trượt hạt nhân đã học qua mọi cửa sổ hợp lệ của đầu vào, rồi cộng thêm một hệ số điều chỉnh vô hướng. Hạt nhân sai phân hữu hạn $1 \times 2$ đơn giản `[1, -1]` phát hiện cạnh dọc ở bất kỳ đâu trên ảnh. Phương pháp hạ gradient tìm lại đúng các trọng số đó chỉ từ dữ liệu đầu vào–đầu ra, cũng như CNN sâu học cả một hệ thống các tầng bộ lọc từ đầu đến cuối. Phép tích chập theo nghĩa toán học lật hạt nhân, nhưng việc lật này không làm thay đổi kết quả có thể học được. Một phép tích chập có thể được tính bằng một phép nhân ma trận duy nhất (*im2col*). Khi xếp chồng $L$ tầng tích chập, độ dài cạnh của vùng tiếp nhận của mỗi đơn vị tăng tuyến tính, thêm $k_i - 1$ ở mỗi tầng.

<!--en:3ba929d2 kind=slide_title-->
::: {.slide title="Fast Execution: Convolution via Matrix Multiplication"}
<!--tr:3ba929d2-->
::: {.slide title="Tính tích chập nhanh bằng phép nhân ma trận"}

<!--en:1711f812 kind=kicker-->
[The im2col (Image-to-Columns) Transformation]{.kicker}
<!--tr:1711f812-->
[Phép biến đổi im2col (ảnh thành các cột)]{.kicker}

<!--en:a040f3d7 kind=slide_text-->
- **Single dense GEMM:** Flatten each $k_\textrm{h} \times k_\textrm{w}$ window into a row of `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ and multiply by $\textrm{vec}(\mathbf{K})$.
<!--tr:a040f3d7-->
- **Một phép nhân ma trận đặc (GEMM):** Trải phẳng mỗi cửa sổ $k_\textrm{h} \times k_\textrm{w}$ thành một hàng của `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ rồi nhân với $\textrm{vec}(\mathbf{K})$.

<!--en:1fb18ec4 kind=slide_text-->
- **Trade-off:** Peak GPU Tensor Core speed at the cost of $k_\textrm{h} k_\textrm{w}\times$ patch workspace duplication.
<!--tr:1fb18ec4-->
- **Đánh đổi:** Đạt tốc độ tối đa của Tensor Core trên GPU, đổi lại bộ nhớ làm việc để lưu các vùng ảnh có thể tăng tới $k_\textrm{h} k_\textrm{w}\times$.

<!--en:0d757cf6 kind=slide_title-->
::: {.slide title="Dual View: Unrolling the Kernel into a Banded Matrix"}
<!--tr:0d757cf6-->
::: {.slide title="Cách nhìn đối ngẫu qua ma trận dải của hạt nhân"}

<!--en:7f0defaa kind=kicker-->
[Toeplitz Weight Matrix Multiplication]{.kicker}
<!--tr:7f0defaa-->
[Phép nhân với ma trận trọng số Toeplitz]{.kicker}

<!--en:5898ba57 kind=slide_text-->
Instead of unfolding the input image into duplicated patches, we can unroll the *kernel* into a sparse banded matrix $\mathbf{W}$ that multiplies the flattened input vector $\textrm{vec}(\mathbf{X})$:
<!--tr:5898ba57-->
Thay vì trải ảnh đầu vào thành các vùng ảnh có dữ liệu lặp lại, ta có thể khai triển *hạt nhân* thành ma trận dải thưa $\mathbf{W}$ để nhân với vector đầu vào đã trải phẳng $\textrm{vec}(\mathbf{X})$:

<!--en:dae51741 kind=slide_text-->
- **Matrix–vector form:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, where $\mathbf{W}$ has $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ rows and $n_\textrm{h} n_\textrm{w}$ columns.
<!--tr:dae51741-->
- **Dạng ma trận–vector:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, với $\mathbf{W}$ có $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ hàng và $n_\textrm{h} n_\textrm{w}$ cột.

<!--en:f0e86ee3 kind=slide_text-->
- **Connection to §{sec_why-conv} and backprop:** Every row of $\mathbf{W}$ holds the same $k_\textrm{h} k_\textrm{w}$ kernel entries shifted across columns (panel c), making the backward pass with respect to $\mathbf{X}$ simply multiplication by $\mathbf{W}^\top$ (a transposed convolution).
<!--tr:f0e86ee3-->
- **Liên hệ với §{sec_why-conv} và lan truyền ngược:** Mỗi hàng của $\mathbf{W}$ chứa cùng $k_\textrm{h} k_\textrm{w}$ phần tử hạt nhân, dịch sang các cột khác nhau (ô c). Vì vậy, lượt truyền ngược theo $\mathbf{X}$ chỉ cần nhân với $\mathbf{W}^\top$ (tích chập chuyển vị).

<!--en:af8cf19a kind=slide_title-->
::: {.slide title="Feature Maps and Receptive Fields Across Depth"}
<!--tr:af8cf19a-->
::: {.slide title="Bản đồ đặc trưng và vùng tiếp nhận theo độ sâu"}

<!--en:4eb965f3 kind=kicker-->
[Linear Receptive-Field Growth in Stacked Layers]{.kicker}
<!--tr:4eb965f3-->
[Vùng tiếp nhận tăng tuyến tính khi xếp chồng các tầng]{.kicker}

<!--en:5599a6e9 kind=slide_text-->
A unit's **receptive field** is the region of the original input image whose pixels can influence that unit's activation during forward propagation:
<!--tr:5599a6e9-->
**Vùng tiếp nhận** của một đơn vị là vùng trên ảnh đầu vào ban đầu gồm các điểm ảnh có thể ảnh hưởng đến giá trị kích hoạt của đơn vị đó trong lượt truyền xuôi:

<!--en:eba74f75 kind=slide_text-->
- **Single layer ($L=1$):** A $k_1 \times k_1$ kernel sees a $k_1 \times k_1$ input patch.
<!--tr:eba74f75-->
- **Một tầng ($L=1$):** Hạt nhân $k_1 \times k_1$ nhìn thấy một vùng đầu vào $k_1 \times k_1$.

<!--en:01c175bf kind=slide_text-->
- **Stacking $L$ stride-1 layers:** Layer $i$ with kernel size $k_i$ reaches $k_i - 1$ pixels farther along each axis than a single element of layer $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$
<!--tr:01c175bf-->
- **Xếp chồng $L$ tầng có sải bước 1:** Tầng $i$ có kích thước nhân tích chập $k_i$, nên vùng tiếp nhận rộng thêm $k_i - 1$ điểm ảnh theo mỗi trục so với vùng tiếp nhận của một phần tử ở tầng $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$

<!--en:2bde5822 kind=slide_text-->
- **$L$ stacked $3 \times 3$ layers:** Each layer adds $3 - 1 = 2$ pixels, giving side length $r = 2L + 1$ ($3 \to 5 \to 7$).
<!--tr:2bde5822-->
- **Xếp chồng $L$ tầng $3 \times 3$:** Mỗi tầng thêm $3 - 1 = 2$ điểm ảnh, cho độ dài cạnh $r = 2L + 1$ ($3 \to 5 \to 7$).

<!--en:cbb9e97a kind=slide_title-->
::: {.slide title="Why Deep Stacks of 3×3 Kernels Beat One Large Kernel"}
<!--tr:cbb9e97a-->
::: {.slide title="Vì sao xếp chồng nhiều hạt nhân 3×3 hiệu quả hơn dùng một hạt nhân lớn"}

<!--en:95e46e70 kind=kicker-->
[Parameter Efficiency & Nonlinear Expressivity]{.kicker}
<!--tr:95e46e70-->
[Tiết kiệm tham số và tăng khả năng biểu đạt phi tuyến]{.kicker}

<!--en:bd1d7cc0 kind=slide_text-->
- **Matched $7 \times 7$ receptive field:** Three stacked $3 \times 3$ layers ($r = 1 + 2 + 2 + 2 = 7$) see the exact same $7 \times 7$ patch as one $7 \times 7$ layer.
<!--tr:bd1d7cc0-->
- **Cùng vùng tiếp nhận $7 \times 7$:** Ba tầng $3 \times 3$ xếp chồng ($r = 1 + 2 + 2 + 2 = 7$) nhìn thấy đúng vùng ảnh $7 \times 7$ mà một tầng $7 \times 7$ nhìn thấy.

<!--en:755ac11a kind=slide_text-->
- **45% fewer parameters:** $3 \times 9 c^2 = 27 c^2$ weights instead of $49 c^2$ across $c$ channels.
<!--tr:755ac11a-->
- **Ít hơn 45% tham số:** $3 \times 9 c^2 = 27 c^2$ trọng số thay vì $49 c^2$ với $c$ kênh.

<!--en:e3aa1086 kind=slide_text-->
- **Richer expressivity:** Compounds **3 nonlinear activations** instead of 1 (the core VGG design rule in §{sec_blocks}).
<!--tr:e3aa1086-->
- **Khả năng biểu đạt cao hơn:** Kết hợp **3 phép kích hoạt phi tuyến** thay vì 1 (nguyên tắc thiết kế cốt lõi của VGG trong §{sec_blocks}).

<!--en:9ec83bf6 kind=slide_title-->
::: {.slide title="Theoretical vs. Effective Receptive Fields"}
<!--tr:9ec83bf6-->
::: {.slide title="Vùng tiếp nhận lý thuyết và hiệu dụng"}

<!--en:72a52c3b kind=kicker-->
[Gaussian Decay of Sensitivity Away from the Center]{.kicker}
<!--tr:72a52c3b-->
[Độ nhạy giảm theo đường cong Gauss khi ra xa tâm]{.kicker}

<!--en:33bb377d kind=slide_text-->
- **Theoretical RF ($r = 2L+1$):** Bounding box of all pixels with *at least one* path to the output unit.
<!--tr:33bb377d-->
- **Vùng tiếp nhận lý thuyết ($r = 2L+1$):** Khung chứa mọi điểm ảnh có *ít nhất một* đường dẫn đến đơn vị đầu ra.

<!--en:293436eb kind=slide_text-->
- **Central Limit Theorem:** Center pixels connect via exponentially many paths, whereas corner pixels connect via only **1 path**.
<!--tr:293436eb-->
- **Định lý giới hạn trung tâm:** Số đường dẫn từ các điểm ảnh ở tâm tăng theo cấp số nhân, còn mỗi điểm ảnh ở góc chỉ có **1 đường dẫn**.

<!--en:f47d181e kind=slide_text-->
- **Effective RF (Luo et al., 2016):** Input gradient magnitude $\partial y / \partial \mathbf{X}$ decays in a Gaussian bell curve with std $\sigma \propto \sqrt{L}$.
<!--tr:f47d181e-->
- **Vùng tiếp nhận hiệu dụng (Luo và cộng sự, 2016):** Độ lớn của gradient theo đầu vào $\partial y / \partial \mathbf{X}$ giảm theo đường cong hình chuông Gauss, với độ lệch chuẩn $\sigma \propto \sqrt{L}$.

<!--en:fac792d0 kind=slide_title-->
::: {.slide title="Concept Check: Cross-Correlation & Receptive Fields"}
<!--tr:fac792d0-->
::: {.slide title="Kiểm tra kiến thức về tương quan chéo và vùng tiếp nhận"}

<!--en:26a317d8 kind=kicker-->
[Concept Check]{.kicker}
<!--tr:26a317d8-->
[Kiểm tra kiến thức]{.kicker}

<!--en:8fe554e5 kind=slide_text-->
Test your understanding of output shapes, `im2col`, and receptive-field arithmetic:
<!--tr:8fe554e5-->
Hãy kiểm tra xem bạn đã nắm được cách tính kích thước đầu ra, cách dùng `im2col` và cách tính vùng tiếp nhận chưa:

<!--en:281c7c78 kind=slide_text-->
1. **im2col matrix dimensions:** You apply an unpadded $3 \times 3$ single-channel kernel to a $10 \times 10$ image via `im2col`. What is the output spatial shape, and what are the exact dimensions of the `patches` matrix and flattened kernel vector?
<!--tr:281c7c78-->
1. **Kích thước ma trận im2col:** Dùng `im2col` để áp dụng hạt nhân một kênh $3 \times 3$ lên ảnh $10 \times 10$ mà không đệm. Kích thước không gian của đầu ra là bao nhiêu? Ma trận `patches` và vector hạt nhân đã trải phẳng có kích thước chính xác là bao nhiêu?

<!--en:16976d18 kind=slide_text-->
2. **Depth to span Fashion-MNIST:** At stride 1, how many stacked $3 \times 3$ layers are required so a single unit's theoretical receptive field spans an entire $28 \times 28$ image? How many with $5 \times 5$ layers?
<!--tr:16976d18-->
2. **Độ sâu để bao phủ ảnh Fashion-MNIST:** Với sải bước 1, cần xếp chồng bao nhiêu tầng $3 \times 3$ để vùng tiếp nhận lý thuyết của một đơn vị bao phủ toàn bộ ảnh $28 \times 28$? Nếu dùng tầng $5 \times 5$ thì cần bao nhiêu tầng?

<!--en:ab2e0d6a kind=slide_text-->
3. **Edge-loss boundary shift:** In our $6 \times 8$ vertical-edge learning experiment, why did the loss start at $\approx 12$, and what would the initial loss be if the black band extended all the way to the right edge (`X[:, 2:8] = 0`)?
<!--tr:ab2e0d6a-->
3. **Mất mát khi dịch ranh giới:** Trong thí nghiệm học cạnh dọc trên ảnh $6 \times 8$, vì sao mất mát ban đầu là $\approx 12$? Mất mát ban đầu sẽ là bao nhiêu nếu dải đen kéo dài đến tận biên phải (`X[:, 2:8] = 0`)?

<!--en:250d6877 kind=slide_text-->
**Answers:**
(1) Output shape is $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ outputs). The `patches` matrix has shape $(64, 9)$ and multiplies the $(9, 1)$ kernel vector.
(2) For $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ layers. For $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ layers.
(3) Each of the 6 rows had two $\pm 1$ edges ($6 \times 2 \times 1^2 = 12$). Extending the black band to the right border eliminates the right-hand black-to-white edge, leaving one edge per row and halving the initial loss to $\approx 6$.
<!--tr:250d6877-->
**Đáp án:**
(1) Kích thước đầu ra là $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ phần tử). Ma trận `patches` có kích thước $(64, 9)$ và được nhân với vector hạt nhân kích thước $(9, 1)$.
(2) Với $3 \times 3$, số tầng tối thiểu được tính theo $2L + 1 \ge 28 \implies L = 14$. Với $5 \times 5$, số tầng tối thiểu được tính theo $4L + 1 \ge 28 \implies L = 7$.
(3) Mỗi hàng trong 6 hàng có hai cạnh với giá trị $\pm 1$ ($6 \times 2 \times 1^2 = 12$). Kéo dải đen đến biên phải sẽ loại bỏ cạnh chuyển từ đen sang trắng ở bên phải. Mỗi hàng chỉ còn một cạnh, nên mất mát ban đầu giảm một nửa, còn $\approx 6$.

<!--en:ba43a2dd kind=slide_title-->
::: {.slide title="Summary & Key Takeaways"}
<!--tr:ba43a2dd-->
::: {.slide title="Tóm tắt các ý chính"}

<!--en:ce12e85f kind=kicker-->
[Section Synthesis]{.kicker}
<!--tr:ce12e85f-->
[Tổng kết nội dung]{.kicker}

<!--en:00011fb7 kind=slide_text-->
- **2D Cross-Correlation & Edge Learning:** Slides $\mathbf{K}$ across valid windows ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); SGD learns $[1, -1]$ in 9 steps.
<!--tr:00011fb7-->
- **Tương quan chéo 2D và học phát hiện cạnh:** Trượt $\mathbf{K}$ qua các cửa sổ hợp lệ ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$). Phương pháp hạ gradient học được trọng số xấp xỉ $[1, -1]$ sau 9 bước.

<!--en:09efe4f0 kind=slide_text-->
- **im2col Matrix Multiply:** Trades $k_\textrm{h} k_\textrm{w}\times$ patch duplication for peak GPU GEMM throughput.
<!--tr:09efe4f0-->
- **Phép nhân ma trận im2col:** Chấp nhận sao chép dữ liệu vùng ảnh tới $k_\textrm{h} k_\textrm{w}\times$ để đạt thông lượng GEMM tối đa trên GPU.

<!--en:3af8acd0 kind=slide_text-->
- **Stacked Receptive Fields:** $r = 1 + \sum (k_i - 1) = 2L+1$ for $3 \times 3$ stacks, with Gaussian center sensitivity.
<!--tr:3af8acd0-->
- **Vùng tiếp nhận của các tầng xếp chồng:** $r = 1 + \sum (k_i - 1) = 2L+1$ với các tầng $3 \times 3$. Độ nhạy tập trung quanh tâm theo đường cong Gauss.

=== DECISIONS ===
- “flatten” → “trải phẳng”, “patch matrix” → “ma trận các vùng ảnh”, “unroll the kernel” → “khai triển hạt nhân”, “banded matrix” → “ma trận dải”.
- “stacked layers” → “các tầng xếp chồng”, “theoretical/effective receptive field” → “vùng tiếp nhận lý thuyết/hiệu dụng”, “path” (qua các tầng) → “đường dẫn”, “Gaussian bell curve” → “đường cong hình chuông Gauss”.
- “nonlinearity” chỉ một phép toán → “phép biến đổi phi tuyến”, còn tính chất của phép toán → “tính phi tuyến”.
=== NOTES ===
- f95c9394: Công thức xét mức tăng độ dài cạnh vùng tiếp nhận qua từng tầng có sải bước 1 và không giãn, như trong ví dụ và slide 01c175bf. Câu gốc nói tắt về phần tử đầu vào; bản dịch nêu rõ đại lượng đem so sánh là vùng tiếp nhận của phần tử đó.
- 00011fb7: Bản gốc gọi phép cập nhật là SGD và diễn đạt như thể thu lại chính xác trọng số đích sau 9 bước. Mã dùng toàn bộ cặp đầu vào–đầu ra ở mỗi bước và chỉ thu được trọng số xấp xỉ, như ở 67d25b0b và 456670c5. Cảnh báo “SGD” được giữ lại vì lý do này.
- eed4e2a0: Cảnh báo “shape” là khớp giả. Trong “bell shape”, từ này chỉ dạng hình chuông, không chỉ kích thước của mảng hay tensor.
- 92688a0c: Cảnh báo “scalar” là khớp giả. Bản dịch dùng tính từ “vô hướng” trong “hệ số điều chỉnh vô hướng”.
