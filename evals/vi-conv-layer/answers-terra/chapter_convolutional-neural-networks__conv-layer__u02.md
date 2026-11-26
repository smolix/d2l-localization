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

Dù các vòng lặp lồng nhau trong `corr2d` làm rõ phép toán, việc duyệt từng điểm ảnh đầu ra trong Python quá chậm đối với các tác vụ thực tế. Một giải pháp kinh điển, vẫn nằm trong số những thuật toán mà các thư viện lựa chọn, là chuyển tương quan chéo thành một phép nhân ma trận lớn duy nhất, đúng là phép toán mà GPU thực hiện với tốc độ cao nhất. Lưu ý rằng mỗi phần tử đầu ra trong :numref:`fig_correlation` đơn giản là tích vô hướng của hạt nhân $k_\textrm{h} k_\textrm{w}$ phần tử đã được làm phẳng với cửa sổ đầu vào tương ứng đã được làm phẳng. Trích xuất tất cả $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ cửa sổ và xếp chúng thành các hàng tạo ra một ma trận mảnh; nhân ma trận này với hạt nhân đã làm phẳng sẽ tính toàn bộ phép tương quan chéo trong một lần. Phép biến đổi cách bố trí tiêu chuẩn này được gọi là *im2col* ("ảnh thành các cột", hoặc thành các hàng ở đây). :numref:`fig_conv_im2col` thực hiện phép biến đổi này trên ví dụ của :numref:`fig_correlation`.

<!--en:107554a5 kind=caption-->
![The cross-correlation of :numref:`fig_correlation` as one matrix product. Each of the four $2 \times 2$ windows of the input becomes a row of the patch matrix, and multiplying by the flattened kernel gives the flattened output. The shaded cells follow the first window to its row and to the output 19. The center pixel 4 lies in all four windows, so it appears in every row.](../img/conv-im2col.svg)
<!--tr:107554a5-->

![Phép tương quan chéo trong :numref:`fig_correlation` dưới dạng một phép nhân ma trận. Mỗi cửa sổ $2 \times 2$ trong bốn cửa sổ của đầu vào trở thành một hàng của ma trận mảnh, và phép nhân với hạt nhân đã làm phẳng cho đầu ra đã làm phẳng. Các ô được tô theo dấu cửa sổ đầu tiên đến hàng tương ứng và đến đầu ra 19. Điểm ảnh trung tâm 4 nằm trong cả bốn cửa sổ nên xuất hiện ở mọi hàng.](../img/conv-im2col.svg)

<!--en:76713479 kind=paragraph-->
The trade-off is extra workspace memory, since each interior pixel is duplicated
into up to $k_\textrm{h} k_\textrm{w}$ patches. Below we verify on
:numref:`fig_correlation` that this single matrix multiply matches `corr2d`
exactly:
<!--tr:76713479-->

Đổi lại, ta cần thêm bộ nhớ làm việc vì mỗi điểm ảnh bên trong được sao chép vào tối đa $k_\textrm{h} k_\textrm{w}$ mảnh. Dưới đây chúng ta kiểm tra trên :numref:`fig_correlation` rằng phép nhân ma trận duy nhất này khớp chính xác với `corr2d`:

<!--en:3bcd7d8a kind=paragraph-->
The discrepancy is identically zero: one matrix product reproduces `corr2d`
perfectly.
<!--tr:3bcd7d8a-->

Sai khác bằng đúng không: một phép nhân ma trận tái tạo `corr2d` hoàn toàn chính xác.

<!--en:c30f2dea kind=paragraph-->
Equivalently, we can unroll the *kernel* instead of the input into a sparse
banded matrix $\mathbf{W}$ multiplying the flattened image—the exact 2D analog
of panel (c) in :numref:`fig_why_conv_weight_sharing`, where every row repeats
the same handful of kernel weights. Seeing a convolution as a structured dense
layer brings us full circle to the derivation of :numref:`sec_why-conv` (and the
exercises build $\mathbf{W}$ explicitly).
<!--tr:c30f2dea-->

Tương đương, thay vì trải đầu vào, ta có thể trải *hạt nhân* thành một ma trận thưa dạng dải $\mathbf{W}$ nhân với ảnh đã làm phẳng—đây chính là trường hợp tương tự hai chiều của hình (c) trong :numref:`fig_why_conv_weight_sharing`, nơi mỗi hàng lặp lại cùng một nhóm nhỏ trọng số hạt nhân. Nhìn phép tích chập như một tầng kết nối đầy đủ có cấu trúc đưa chúng ta trở lại phép suy dẫn trong :numref:`sec_why-conv` (và các bài tập xây dựng tường minh $\mathbf{W}$).

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

Một tầng tích chập đơn chỉ nhìn thấy một mảnh có kích thước bằng hạt nhân của nó, trong khi nhận dạng toàn bộ vật thể đòi hỏi tích hợp ngữ cảnh trên những vùng rộng hơn nhiều. Nhớ lại từ :numref:`subsec_why-conv-channels` rằng đầu ra của tầng tích chập được gọi là *bản đồ đặc trưng* (như bản đồ cạnh đứng `Y` của chúng ta). Với phần tử bất kỳ $x$ trong một bản đồ đặc trưng, *vùng tiếp nhận* của nó là vùng của ảnh đầu vào (hoặc của một tầng trước đó) mà các giá trị có thể ảnh hưởng đến $x$ trong lượt truyền xuôi—thuật ngữ này được mượn từ sinh lý học thần kinh trong :numref:`field_visual`.

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

Trong :numref:`fig_correlation`, đầu ra được tô `19` có vùng tiếp nhận $2 \times 2$ bao phủ bốn điểm ảnh trên cùng bên trái của $\mathbf{X}$. Bây giờ hãy xếp chồng một tầng tích chập $2 \times 2$ *thứ hai* lên đầu ra $2 \times 2$ $\mathbf{Y}$ để tạo ra một số vô hướng $z$: $z$ phụ thuộc vào cả bốn phần tử của $\mathbf{Y}$, mà chúng cùng phụ thuộc vào cả chín phần tử của $\mathbf{X}$. Việc xếp chồng các tầng mở rộng vùng tiếp nhận đều đặn theo độ sâu.

<!--en:f95c9394 kind=paragraph-->
How quickly does it grow? Each $k \times k$ convolutional layer reaches $k - 1$
pixels farther than a single element of its input, so stacking $L$ layers with
kernel sizes $k_1, \ldots, k_L$ yields a receptive field of side length
<!--tr:f95c9394-->

Vùng này tăng nhanh đến mức nào? Mỗi tầng tích chập $k \times k$ vươn xa hơn một phần tử đầu vào của nó $k - 1$ điểm ảnh, vì vậy xếp chồng $L$ tầng với kích thước nhân tích chập $k_1, \ldots, k_L$ cho vùng tiếp nhận có độ dài cạnh

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

Với một chồng $L$ tầng $3 \times 3$ liên tiếp, mỗi tầng thêm 2 điểm ảnh, bao phủ $(2L + 1) \times (2L + 1)$ đầu vào: $5 \times 5$ sau hai tầng và $7 \times 7$ sau ba tầng (:numref:`fig_conv_receptive_field`). Sự tăng trưởng tuyến tính này thực ra là một lợi thế chứ không phải lý do để dùng hạt nhân khổng lồ: ba tầng $3 \times 3$ xếp chồng đạt cùng vùng tiếp nhận $7 \times 7$ như một tầng $7 \times 7$ duy nhất với ít tham số hơn nhiều ($3 \times 9 = 27$ trọng số so với 49 cho mỗi cặp kênh), đồng thời xen kẽ ba phép phi tuyến thay vì một.

<!--en:39917513 kind=caption-->
![Receptive field of one unit after three one-dimensional layers with kernel size 3. Shaded cells are the units that the top unit depends on, and the lines connect each of them to its inputs. Each layer widens the receptive field by 2, to 7 inputs.](../img/conv-receptive-field.svg)
<!--tr:39917513-->

![Vùng tiếp nhận của một đơn vị sau ba tầng một chiều có kích thước nhân tích chập 3. Các ô được tô là những đơn vị mà đơn vị trên cùng phụ thuộc vào, và các đường nối mỗi đơn vị đó với đầu vào của nó. Mỗi tầng mở rộng vùng tiếp nhận thêm 2, đến 7 đầu vào.](../img/conv-receptive-field.svg)

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

Hãy nhớ rằng $r$ đo vùng tiếp nhận *lý thuyết* (mọi điểm ảnh đầu vào có *bất kỳ* đường đi nào đến một đơn vị đầu ra) mà không cho biết mỗi điểm ảnh thực sự đóng góp mạnh đến mức nào. Bằng cách đo trực tiếp gradient đầu vào, :citet:`Luo.Li.Urtasun.ea.2016` cho thấy vùng tiếp nhận *hiệu dụng* suy giảm mạnh khi xa tâm theo một đường cong hình chuông gần Gaussian, ở cả mạng chưa huấn luyện và đã huấn luyện. Về lý thuyết, các đơn vị sâu có thể nhìn xa trên ảnh, nhưng độ nhạy của chúng tập trung mạnh gần tâm.

<!--en:eed4e2a0 kind=paragraph-->
Counting paths explains the bell shape. An input at the edge of the receptive
field reaches the output unit along a single chain of connections, whereas the
center input can take many routes through the layers in between.
:numref:`fig_conv_rf_paths` counts them for the tower of
:numref:`fig_conv_receptive_field`: the center input has seven paths to the top
unit and each edge input has one. If all weights have a similar magnitude, the
influence of an input grows with its number of paths.
<!--tr:eed4e2a0-->

Việc đếm đường đi giải thích hình chuông này. Một đầu vào ở rìa vùng tiếp nhận đến được đơn vị đầu ra theo một chuỗi kết nối duy nhất, trong khi đầu vào ở tâm có thể đi theo nhiều tuyến qua các tầng trung gian. :numref:`fig_conv_rf_paths` đếm các đường đi đó cho chồng tầng trong :numref:`fig_conv_receptive_field`: đầu vào ở tâm có bảy đường đến đơn vị trên cùng và mỗi đầu vào ở rìa có một đường. Nếu mọi trọng số có độ lớn tương tự nhau, ảnh hưởng của một đầu vào tăng theo số đường đi của nó.

<!--en:b92775ee kind=caption-->
![Number of paths from each unit to the top unit of :numref:`fig_conv_receptive_field` (three one-dimensional layers with kernel size 3). Each count is the sum of the counts of the units that the cell feeds. Darker cells have more paths than the other cells of their layer, and the bars repeat the counts of the input row: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)
<!--tr:b92775ee-->

![Số đường đi từ mỗi đơn vị đến đơn vị trên cùng của :numref:`fig_conv_receptive_field` (ba tầng một chiều có kích thước nhân tích chập 3). Mỗi số đếm là tổng số đếm của các đơn vị nhận đầu vào từ ô đó. Các ô sẫm màu hơn có nhiều đường đi hơn những ô khác trong cùng tầng, và các thanh lặp lại số đếm của hàng đầu vào: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)

<!--en:405695e8 kind=paragraph-->
Two practical gaps remain in our basic layer: every unpadded convolution shrinks
spatial dimensions while under-sampling border pixels, and a single kernel
detects only one pattern. We address spatial sizing and borders next in
:numref:`sec_padding`, and multi-channel feature maps in :numref:`sec_channels`.
<!--tr:405695e8-->

Tầng cơ bản của chúng ta còn hai hạn chế thực tế: mỗi phép tích chập không đệm làm giảm kích thước không gian trong khi lấy mẫu các điểm ảnh biên không đủ, và một hạt nhân chỉ phát hiện được một mẫu. Tiếp theo, chúng ta xử lý kích thước không gian và biên trong :numref:`sec_padding`, rồi các bản đồ đặc trưng đa kênh trong :numref:`sec_channels`.

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

Một tầng tích chập áp dụng phép tương quan chéo hai chiều với một hạt nhân được học rồi cộng một hệ số điều chỉnh là số vô hướng, bằng cách trượt hạt nhân qua mọi cửa sổ hợp lệ của đầu vào. Hạt nhân sai phân hữu hạn $1 \times 2$ đơn giản `[1, -1]` phát hiện cạnh đứng ở bất kỳ đâu trong ảnh, và hạ gradient khôi phục chính các trọng số đó chỉ từ dữ liệu đầu vào–đầu ra, cũng như CNN sâu học toàn bộ hệ phân cấp bộ lọc từ đầu đến cuối. Tích chập toán học lật hạt nhân, điều này không tạo ra khác biệt một khi các hạt nhân được học. Một phép tích chập có thể được tính bằng một phép nhân ma trận duy nhất (*im2col*), và xếp chồng $L$ tầng tích chập mở rộng vùng tiếp nhận của mỗi đơn vị tuyến tính thêm $k_i - 1$ cho mỗi tầng.

<!--en:3ba929d2 kind=slide_title-->
::: {.slide title="Fast Execution: Convolution via Matrix Multiplication"}
<!--tr:3ba929d2-->

::: {.slide title="Thực thi nhanh: Tích chập bằng phép nhân ma trận"}

<!--en:1711f812 kind=kicker-->
[The im2col (Image-to-Columns) Transformation]{.kicker}
<!--tr:1711f812-->

[Phép biến đổi im2col (ảnh thành cột)]{.kicker}

<!--en:a040f3d7 kind=slide_text-->
- **Single dense GEMM:** Flatten each $k_\textrm{h} \times k_\textrm{w}$ window into a row of `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ and multiply by $\textrm{vec}(\mathbf{K})$.
<!--tr:a040f3d7-->

- **Một phép GEMM đặc:** Làm phẳng mỗi cửa sổ $k_\textrm{h} \times k_\textrm{w}$ thành một hàng của `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ và nhân với $\textrm{vec}(\mathbf{K})$.

<!--en:1fb18ec4 kind=slide_text-->
- **Trade-off:** Peak GPU Tensor Core speed at the cost of $k_\textrm{h} k_\textrm{w}\times$ patch workspace duplication.
<!--tr:1fb18ec4-->

- **Đánh đổi:** Đạt tốc độ tối đa của GPU Tensor Core với chi phí sao chép $k_\textrm{h} k_\textrm{w}\times$ vùng nhớ làm việc cho các mảnh.

<!--en:0d757cf6 kind=slide_title-->
::: {.slide title="Dual View: Unrolling the Kernel into a Banded Matrix"}
<!--tr:0d757cf6-->

::: {.slide title="Cách nhìn đối ngẫu: Trải hạt nhân thành ma trận dạng dải"}

<!--en:7f0defaa kind=kicker-->
[Toeplitz Weight Matrix Multiplication]{.kicker}
<!--tr:7f0defaa-->

[Phép nhân ma trận trọng số Toeplitz]{.kicker}

<!--en:5898ba57 kind=slide_text-->
Instead of unfolding the input image into duplicated patches, we can unroll the *kernel* into a sparse banded matrix $\mathbf{W}$ that multiplies the flattened input vector $\textrm{vec}(\mathbf{X})$:
<!--tr:5898ba57-->

Thay vì trải ảnh đầu vào thành các mảnh trùng lặp, ta có thể trải *hạt nhân* thành ma trận thưa dạng dải $\mathbf{W}$ nhân với vector đầu vào đã làm phẳng $\textrm{vec}(\mathbf{X})$:

<!--en:dae51741 kind=slide_text-->
- **Matrix–vector form:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, where $\mathbf{W}$ has $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ rows and $n_\textrm{h} n_\textrm{w}$ columns.
<!--tr:dae51741-->

- **Dạng ma trận–vector:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, trong đó $\mathbf{W}$ có $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ hàng và $n_\textrm{h} n_\textrm{w}$ cột.

<!--en:f0e86ee3 kind=slide_text-->
- **Connection to §{sec_why-conv} and backprop:** Every row of $\mathbf{W}$ holds the same $k_\textrm{h} k_\textrm{w}$ kernel entries shifted across columns (panel c), making the backward pass with respect to $\mathbf{X}$ simply multiplication by $\mathbf{W}^\top$ (a transposed convolution).
<!--tr:f0e86ee3-->

- **Liên hệ với §{sec_why-conv} và lan truyền ngược:** Mỗi hàng của $\mathbf{W}$ chứa cùng $k_\textrm{h} k_\textrm{w}$ phần tử hạt nhân được dịch qua các cột (hình c), khiến lượt truyền ngược theo $\mathbf{X}$ đơn giản là phép nhân với $\mathbf{W}^\top$ (một phép tích chập chuyển vị).

<!--en:af8cf19a kind=slide_title-->
::: {.slide title="Feature Maps and Receptive Fields Across Depth"}
<!--tr:af8cf19a-->

::: {.slide title="Bản đồ đặc trưng và vùng tiếp nhận theo độ sâu"}

<!--en:4eb965f3 kind=kicker-->
[Linear Receptive-Field Growth in Stacked Layers]{.kicker}
<!--tr:4eb965f3-->

[Vùng tiếp nhận tăng tuyến tính trong các tầng xếp chồng]{.kicker}

<!--en:5599a6e9 kind=slide_text-->
A unit's **receptive field** is the region of the original input image whose pixels can influence that unit's activation during forward propagation:
<!--tr:5599a6e9-->

**Vùng tiếp nhận** của một đơn vị là vùng của ảnh đầu vào ban đầu mà các điểm ảnh có thể ảnh hưởng đến giá trị kích hoạt của đơn vị đó trong lượt truyền xuôi:

<!--en:eba74f75 kind=slide_text-->
- **Single layer ($L=1$):** A $k_1 \times k_1$ kernel sees a $k_1 \times k_1$ input patch.
<!--tr:eba74f75-->

- **Một tầng ($L=1$):** Hạt nhân $k_1 \times k_1$ nhìn thấy một mảnh đầu vào $k_1 \times k_1$.

<!--en:01c175bf kind=slide_text-->
- **Stacking $L$ stride-1 layers:** Layer $i$ with kernel size $k_i$ reaches $k_i - 1$ pixels farther along each axis than a single element of layer $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$
<!--tr:01c175bf-->

- **Xếp chồng $L$ tầng có sải bước 1:** Tầng $i$ với kích thước nhân tích chập $k_i$ vươn xa hơn một phần tử của tầng $i-1$ thêm $k_i - 1$ điểm ảnh theo mỗi trục:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$

<!--en:2bde5822 kind=slide_text-->
- **$L$ stacked $3 \times 3$ layers:** Each layer adds $3 - 1 = 2$ pixels, giving side length $r = 2L + 1$ ($3 \to 5 \to 7$).
<!--tr:2bde5822-->

- **$L$ tầng $3 \times 3$ xếp chồng:** Mỗi tầng thêm $3 - 1 = 2$ điểm ảnh, cho độ dài cạnh $r = 2L + 1$ ($3 \to 5 \to 7$).

<!--en:cbb9e97a kind=slide_title-->
::: {.slide title="Why Deep Stacks of 3×3 Kernels Beat One Large Kernel"}
<!--tr:cbb9e97a-->

::: {.slide title="Vì sao chồng sâu các hạt nhân 3×3 tốt hơn một hạt nhân lớn"}

<!--en:95e46e70 kind=kicker-->
[Parameter Efficiency & Nonlinear Expressivity]{.kicker}
<!--tr:95e46e70-->

[Hiệu quả tham số và khả năng biểu đạt phi tuyến]{.kicker}

<!--en:bd1d7cc0 kind=slide_text-->
- **Matched $7 \times 7$ receptive field:** Three stacked $3 \times 3$ layers ($r = 1 + 2 + 2 + 2 = 7$) see the exact same $7 \times 7$ patch as one $7 \times 7$ layer.
<!--tr:bd1d7cc0-->

- **Cùng vùng tiếp nhận $7 \times 7$:** Ba tầng $3 \times 3$ xếp chồng ($r = 1 + 2 + 2 + 2 = 7$) nhìn thấy đúng cùng mảnh $7 \times 7$ như một tầng $7 \times 7$.

<!--en:755ac11a kind=slide_text-->
- **45% fewer parameters:** $3 \times 9 c^2 = 27 c^2$ weights instead of $49 c^2$ across $c$ channels.
<!--tr:755ac11a-->

- **Ít hơn 45% tham số:** $3 \times 9 c^2 = 27 c^2$ trọng số thay vì $49 c^2$ trên $c$ kênh.

<!--en:e3aa1086 kind=slide_text-->
- **Richer expressivity:** Compounds **3 nonlinear activations** instead of 1 (the core VGG design rule in §{sec_blocks}).
<!--tr:e3aa1086-->

- **Khả năng biểu đạt phong phú hơn:** Kết hợp **3 phép kích hoạt phi tuyến** thay vì 1 (quy tắc thiết kế cốt lõi của VGG trong §{sec_blocks}).

<!--en:9ec83bf6 kind=slide_title-->
::: {.slide title="Theoretical vs. Effective Receptive Fields"}
<!--tr:9ec83bf6-->

::: {.slide title="Vùng tiếp nhận lý thuyết và hiệu dụng"}

<!--en:72a52c3b kind=kicker-->
[Gaussian Decay of Sensitivity Away from the Center]{.kicker}
<!--tr:72a52c3b-->

[Độ nhạy suy giảm theo Gaussian khi xa tâm]{.kicker}

<!--en:33bb377d kind=slide_text-->
- **Theoretical RF ($r = 2L+1$):** Bounding box of all pixels with *at least one* path to the output unit.
<!--tr:33bb377d-->

- **Vùng tiếp nhận lý thuyết ($r = 2L+1$):** Khung chứa mọi điểm ảnh có *ít nhất một* đường đi đến đơn vị đầu ra.

<!--en:293436eb kind=slide_text-->
- **Central Limit Theorem:** Center pixels connect via exponentially many paths, whereas corner pixels connect via only **1 path**.
<!--tr:293436eb-->

- **Định lý giới hạn trung tâm:** Các điểm ảnh ở tâm kết nối qua số đường đi tăng theo hàm mũ, trong khi các điểm ảnh ở góc chỉ kết nối qua **1 đường**.

<!--en:f47d181e kind=slide_text-->
- **Effective RF (Luo et al., 2016):** Input gradient magnitude $\partial y / \partial \mathbf{X}$ decays in a Gaussian bell curve with std $\sigma \propto \sqrt{L}$.
<!--tr:f47d181e-->

- **Vùng tiếp nhận hiệu dụng (Luo et al., 2016):** Độ lớn gradient đầu vào $\partial y / \partial \mathbf{X}$ suy giảm theo đường cong hình chuông Gaussian với độ lệch chuẩn $\sigma \propto \sqrt{L}$.

<!--en:fac792d0 kind=slide_title-->
::: {.slide title="Concept Check: Cross-Correlation & Receptive Fields"}
<!--tr:fac792d0-->

::: {.slide title="Kiểm tra khái niệm: Tương quan chéo và vùng tiếp nhận"}

<!--en:26a317d8 kind=kicker-->
[Concept Check]{.kicker}
<!--tr:26a317d8-->

[Kiểm tra khái niệm]{.kicker}

<!--en:8fe554e5 kind=slide_text-->
Test your understanding of output shapes, `im2col`, and receptive-field arithmetic:
<!--tr:8fe554e5-->

Kiểm tra hiểu biết của bạn về kích thước đầu ra, `im2col` và cách tính vùng tiếp nhận:

<!--en:281c7c78 kind=slide_text-->
1. **im2col matrix dimensions:** You apply an unpadded $3 \times 3$ single-channel kernel to a $10 \times 10$ image via `im2col`. What is the output spatial shape, and what are the exact dimensions of the `patches` matrix and flattened kernel vector?
<!--tr:281c7c78-->

1. **Kích thước ma trận im2col:** Bạn áp dụng hạt nhân một kênh $3 \times 3$ không đệm cho ảnh $10 \times 10$ bằng `im2col`. Kích thước không gian đầu ra là bao nhiêu, và kích thước chính xác của ma trận `patches` cùng vector hạt nhân đã làm phẳng là bao nhiêu?

<!--en:16976d18 kind=slide_text-->
2. **Depth to span Fashion-MNIST:** At stride 1, how many stacked $3 \times 3$ layers are required so a single unit's theoretical receptive field spans an entire $28 \times 28$ image? How many with $5 \times 5$ layers?
<!--tr:16976d18-->

2. **Độ sâu để bao phủ Fashion-MNIST:** Với sải bước 1, cần bao nhiêu tầng $3 \times 3$ xếp chồng để vùng tiếp nhận lý thuyết của một đơn vị bao phủ toàn bộ ảnh $28 \times 28$? Cần bao nhiêu tầng $5 \times 5$?

<!--en:ab2e0d6a kind=slide_text-->
3. **Edge-loss boundary shift:** In our $6 \times 8$ vertical-edge learning experiment, why did the loss start at $\approx 12$, and what would the initial loss be if the black band extended all the way to the right edge (`X[:, 2:8] = 0`)?
<!--tr:ab2e0d6a-->

3. **Dịch biên và mất mát cạnh:** Trong thí nghiệm học cạnh đứng $6 \times 8$, vì sao mất mát bắt đầu ở $\approx 12$, và mất mát ban đầu sẽ là bao nhiêu nếu dải đen kéo dài đến tận biên phải (`X[:, 2:8] = 0`)?

<!--en:250d6877 kind=slide_text-->
**Answers:**
(1) Output shape is $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ outputs). The `patches` matrix has shape $(64, 9)$ and multiplies the $(9, 1)$ kernel vector.
(2) For $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ layers. For $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ layers.
(3) Each of the 6 rows had two $\pm 1$ edges ($6 \times 2 \times 1^2 = 12$). Extending the black band to the right border eliminates the right-hand black-to-white edge, leaving one edge per row and halving the initial loss to $\approx 6$.
<!--tr:250d6877-->

**Đáp án:**
(1) Kích thước đầu ra là $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ đầu ra). Ma trận `patches` có kích thước $(64, 9)$ và nhân với vector hạt nhân $(9, 1)$.
(2) Với $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ tầng. Với $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ tầng.
(3) Mỗi hàng trong 6 hàng có hai cạnh $\pm 1$ ($6 \times 2 \times 1^2 = 12$). Kéo dài dải đen đến biên phải loại bỏ cạnh đen sang trắng bên phải, còn lại một cạnh mỗi hàng và giảm một nửa mất mát ban đầu xuống $\approx 6$.

<!--en:ba43a2dd kind=slide_title-->
::: {.slide title="Summary & Key Takeaways"}
<!--tr:ba43a2dd-->

::: {.slide title="Tóm tắt và các điểm chính"}

<!--en:ce12e85f kind=kicker-->
[Section Synthesis]{.kicker}
<!--tr:ce12e85f-->

[Tổng hợp mục]{.kicker}

<!--en:00011fb7 kind=slide_text-->
- **2D Cross-Correlation & Edge Learning:** Slides $\mathbf{K}$ across valid windows ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); SGD learns $[1, -1]$ in 9 steps.
<!--tr:00011fb7-->

- **Tương quan chéo 2D và học cạnh:** Trượt $\mathbf{K}$ qua các cửa sổ hợp lệ ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); SGD học $[1, -1]$ trong 9 bước.

<!--en:09efe4f0 kind=slide_text-->
- **im2col Matrix Multiply:** Trades $k_\textrm{h} k_\textrm{w}\times$ patch duplication for peak GPU GEMM throughput.
<!--tr:09efe4f0-->

- **Phép nhân ma trận im2col:** Đánh đổi việc sao chép mảnh $k_\textrm{h} k_\textrm{w}\times$ để đạt thông lượng GEMM tối đa của GPU.

<!--en:3af8acd0 kind=slide_text-->
- **Stacked Receptive Fields:** $r = 1 + \sum (k_i - 1) = 2L+1$ for $3 \times 3$ stacks, with Gaussian center sensitivity.
<!--tr:3af8acd0-->

- **Vùng tiếp nhận xếp chồng:** $r = 1 + \sum (k_i - 1) = 2L+1$ với các chồng tầng $3 \times 3$, cùng độ nhạy ở tâm theo Gaussian.

=== DECISIONS ===
- “feature map” → “bản đồ đặc trưng”; “effective receptive field” → “vùng tiếp nhận hiệu dụng”; “banded matrix” → “ma trận dạng dải”; “flatten” → “làm phẳng”.

=== NOTES ===
- eed4e2a0: “shape” chỉ hình dạng đường cong, không phải kích thước tensor.
- 01c175bf: Các biểu thức toán được đổi thứ tự để câu tiếng Việt tự nhiên.
