<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u03 lang=vi rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:2294243a kind=heading-->
## Exercises
<!--tr:2294243a-->
## Bài tập

<!--en:6f4ddec6 kind=exercise-->
1. [code] **Hand-designed kernels.** Design some kernels manually.
    1. Given a directional vector $\mathbf{v} = (v_1, v_2)$, derive an
       edge-detection kernel that detects edges orthogonal to $\mathbf{v}$,
       i.e., edges in the direction $(v_2, -v_1)$.
    1. Derive a finite difference operator for the second derivative.
       Which structures in images respond most strongly to it?
    1. What is the minimum size of a kernel to obtain a derivative of
       order $d$?
    1. How would you design a blur kernel? Why might you want to use such
       a kernel?
    1. Load a photograph, convert it to one channel, and downsample it to
       about $100 \times 100$ pixels (`corr2d` loops in Python). Apply the
       edge detector, the blur kernel, and the sharpening kernel "twice
       the identity minus the blur", and display each result next to the
       original. What does the edge map of a photograph show that the
       edge map of the black band could not?

    *Adapted from Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    ConvolutionalNetworks notebook.*
<!--tr:6f4ddec6-->
1. [code] **Hạt nhân thiết kế thủ công.** Hãy tự thiết kế thủ công một số hạt nhân.
    1. Cho vector chỉ phương $\mathbf{v} = (v_1, v_2)$, hãy suy ra một
       hạt nhân phát hiện cạnh có khả năng phát hiện các cạnh trực giao với $\mathbf{v}$,
       tức là các cạnh theo hướng $(v_2, -v_1)$.
    1. Hãy suy ra một toán tử sai phân hữu hạn cho đạo hàm bậc hai.
       Những cấu trúc nào trong ảnh phản ứng mạnh nhất với toán tử này?
    1. Kích thước tối thiểu của một hạt nhân để thu được đạo hàm bậc
       $d$ là bao nhiêu?
    1. Bạn sẽ thiết kế một hạt nhân làm mờ như thế nào? Vì sao ta lại muốn sử dụng một
       hạt nhân như vậy?
    1. Hãy tải một bức ảnh chụp, chuyển sang một kênh duy nhất và giảm lấy mẫu xuống
       khoảng $100 \times 100$ điểm ảnh (`corr2d` chạy vòng lặp trong Python). Áp dụng bộ
       phát hiện cạnh, hạt nhân làm mờ và hạt nhân làm sắc nét "hai lần
       đồng nhất trừ đi làm mờ", rồi hiển thị từng kết quả bên cạnh ảnh
       gốc. Bản đồ cạnh của một bức ảnh chụp cho thấy điều gì mà bản đồ
       cạnh của dải màu đen không thể hiện được?

    *Phỏng theo Stanford CS231n,
    [Bài tập 2](https://cs231n.github.io/assignments2024/assignment2/),
    notebook ConvolutionalNetworks.*

<!--en:ec8621f8 kind=exercise-->
2. [code] **Convolution as a banded matrix.** The patch matrix of
   :numref:`subsec_conv_matmul` unfolds the input. The opposite
   construction unrolls the kernel. Let $\mathbf{X}$ be a $4 \times 4$
   input, $\mathbf{K}$ a $3 \times 3$ kernel, and $\textrm{vec}(\cdot)$
   the row-by-row flattening.
    1. Construct the $4 \times 16$ matrix $\mathbf{W}$ with
       $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$,
       where $\mathbf{Y}$ is the cross-correlation of $\mathbf{X}$ with
       $\mathbf{K}$. How many nonzero entries does each row have, and how
       many distinct values appear in $\mathbf{W}$?
    1. Build $\mathbf{W}$ in code from `K` and confirm that the product
       reproduces `corr2d(X, K)`.
    1. For an $n \times n$ input and a $k \times k$ kernel, count the
       entries of $\mathbf{W}$ stored as a dense array, the nonzero
       entries of $\mathbf{W}$, and the entries of the patch matrix. How
       does each count grow with $n$?
    1. Keep the sparsity pattern of $\mathbf{W}$ but give every row its
       own $k^2$ values. The result is a locally connected layer: the
       fourth-order tensor $\mathsf{V}$ of :numref:`sec_why-conv` in
       $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$,
       restricted to a $k \times k$ window of offsets $(a, b)$ before
       translation equivariance ties the weights. For an $n \times n$
       input, count the parameters of this layer, of a fully connected
       layer with the same input and output shapes, and of the
       convolution.

    *Adapted from CMU 11-785,
    [Homework 2 Part 1](https://deeplearning.cs.cmu.edu/S21/index.html).*
<!--tr:ec8621f8-->
2. [code] **Phép tích chập dưới dạng ma trận dải.** Ma trận vùng con trong
   :numref:`subsec_conv_matmul` bung đầu vào ra. Cách xây dựng ngược lại
   trải hạt nhân ra. Giả sử $\mathbf{X}$ là đầu vào $4 \times 4$,
   $\mathbf{K}$ là hạt nhân $3 \times 3$, và $\textrm{vec}(\cdot)$
   là phép trải phẳng theo từng hàng.
    1. Hãy xây dựng ma trận kích thước $4 \times 16$ ký hiệu $\mathbf{W}$ sao cho
       $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$,
       trong đó $\mathbf{Y}$ là kết quả tương quan chéo của $\mathbf{X}$ với
       $\mathbf{K}$. Mỗi hàng có bao nhiêu phần tử khác không, và có bao
       nhiêu giá trị phân biệt xuất hiện trong $\mathbf{W}$?
    1. Hãy lập trình tạo $\mathbf{W}$ từ `K` và xác nhận rằng tích thu được
       tái hiện đúng `corr2d(X, K)`.
    1. Với đầu vào $n \times n$ và hạt nhân $k \times k$, hãy đếm số
       phần tử của $\mathbf{W}$ khi lưu dưới dạng mảng dày đặc, số phần tử
       khác không của $\mathbf{W}$, và số phần tử của ma trận vùng con. Mỗi
       số đếm này tăng theo $n$ như thế nào?
    1. Giữ nguyên cấu trúc độ thưa của $\mathbf{W}$ nhưng gán cho mỗi hàng
       $k^2$ giá trị riêng biệt. Kết quả thu được là một tầng kết nối cục bộ:
       tensor bậc bốn $\mathsf{V}$ của :numref:`sec_why-conv` trong công thức
       $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$,
       giới hạn trong một cửa sổ $k \times k$ của các độ lệch $(a, b)$ trước khi
       tính đẳng biến tịnh tiến ràng buộc các trọng số lại với nhau. Với đầu vào $n \times n$,
       hãy đếm số tham số của tầng này, của một tầng kết nối đầy đủ
       có cùng kích thước đầu vào và đầu ra, và của phép
       tích chập.

    *Phỏng theo CMU 11-785,
    [Bài tập về nhà 2 Phần 1](https://deeplearning.cs.cmu.edu/S21/index.html).*

<!--en:54bcee26 kind=exercise-->
3. [code] **Effective receptive field.** ● The theoretical receptive
   field $r$ of this section bounds which inputs *can* influence an
   output; the effective receptive field measures how much
   they actually do :cite:`Luo.Li.Urtasun.ea.2016`. Build a stack of $L$
   single-channel $3 \times 3$ convolutional layers with no
   padding, no bias, and no nonlinearity on a $32 \times 32$ input, using
   the built-in layer of :numref:`subsec_learning_kernel`. Draw every
   kernel entry independently from a standard normal distribution.
    1. For $L = 3$ and a single draw of the kernels, backpropagate a unit
       gradient from an output unit near the center of the last layer to
       the input and plot the magnitude of the input gradient. Compare its
       support with the $7 \times 7$ theoretical receptive field, and say
       where the magnitude is largest.
    1. Average the squared input gradient over 500 draws of the kernels,
       for $L = 3$ and for $L = 10$. Sum each averaged map over its rows
       and normalize the resulting column profile to sum to one. How does
       the ratio of the profile's standard deviation to the theoretical
       side length $2L + 1$ change with depth?
    1. :numref:`fig_conv_rf_paths` counts the paths for $L = 3$ in one
       dimension. Show that for general $L$ the expected squared input
       gradient, summed over rows, is proportional to the convolution of
       $L$ copies of the vector $(1, 1, 1)$. Derive how the standard
       deviation of the normalized profile grows with $L$.
<!--tr:54bcee26-->
3. [code] **Vùng tiếp nhận hiệu dụng.** ● Vùng tiếp nhận lý thuyết
   $r$ trong mục này giới hạn những đầu vào nào *có thể* ảnh hưởng đến một
   đầu ra; vùng tiếp nhận hiệu dụng đo lường mức độ ảnh hưởng
   thực tế của chúng :cite:`Luo.Li.Urtasun.ea.2016`. Hãy xây dựng một khối gồm $L$
   tầng tích chập đơn kênh $3 \times 3$ không đệm,
   không hệ số điều chỉnh và không có tính phi tuyến trên đầu vào $32 \times 32$, sử dụng
   tầng dựng sẵn ở :numref:`subsec_learning_kernel`. Lấy mẫu độc lập từng
   phần tử hạt nhân từ phân phối Gauss chuẩn.
    1. Với $L = 3$ và một lần lấy mẫu các hạt nhân, hãy lan truyền ngược một
       gradient đơn vị từ một đơn vị đầu ra gần tâm của tầng cuối cùng về
       đầu vào và vẽ biểu đồ độ lớn của gradient đầu vào. Hãy so sánh
       miền giá của nó với vùng tiếp nhận lý thuyết $7 \times 7$, và cho biết
       độ lớn đạt cực đại ở đâu.
    1. Lấy trung bình bình phương gradient đầu vào qua 500 lần lấy mẫu hạt nhân,
       với $L = 3$ và với $L = 10$. Cộng từng bản đồ trung bình theo các hàng
       và chuẩn hóa biên dạng cột thu được sao cho tổng bằng một. Tỷ số
       giữa độ lệch chuẩn của biên dạng này và độ dài cạnh lý thuyết
       $2L + 1$ thay đổi như thế nào theo độ sâu?
    1. :numref:`fig_conv_rf_paths` đếm số đường truyền ứng với $L = 3$ trong không gian một
       chiều. Hãy chứng minh rằng với $L$ tổng quát, kỳ vọng của bình phương gradient đầu vào
       sau khi cộng theo các hàng tỷ lệ thuận với tích chập của
       $L$ bản sao vector $(1, 1, 1)$. Hãy suy ra quy luật tăng của độ lệch
       chuẩn của biên dạng chuẩn hóa theo $L$.

<!--en:6ab5b67c kind=exercise-->
4. [code] **Backward pass of a convolution.** ● Let $\mathbf{Y}$ be the
   single-channel cross-correlation of an input $\mathbf{X}$ with a kernel
   $\mathbf{K}$, as computed by `corr2d`, and let $\ell$ be a scalar loss
   that depends on $\mathbf{X}$ and $\mathbf{K}$ only through $\mathbf{Y}$.
    1. Derive $\partial \ell/\partial \mathbf{K}$ and
       $\partial \ell/\partial \mathbf{X}$ in terms of
       $\partial \ell/\partial \mathbf{Y}$, in the style of the
       backpropagation equations of :numref:`sec_backprop`.
    1. Implement your expression for $\partial \ell/\partial \mathbf{K}$
       without automatic differentiation for the loss
       $\ell = \sum_{i,j} G_{ij} Y_{ij}$, where $\mathbf{G}$ is a fixed
       random matrix of the shape of $\mathbf{Y}$. Compare it with a
       centered finite-difference gradient with step size $10^{-2}$ on a
       small random `X` and `K`, and report the largest relative error.
    1. Check your expression for $\partial \ell/\partial \mathbf{K}$
       against the kernel gradient that automatic differentiation
       computes in the experiment of :numref:`subsec_learning_kernel`.

    *Adapted from Michael Nielsen,
    [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html),
    Chapter 6, and Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    ConvolutionalNetworks notebook.*
<!--tr:6ab5b67c-->
4. [code] **Lượt truyền ngược của phép tích chập.** ● Giả sử $\mathbf{Y}$ là
   kết quả tương quan chéo đơn kênh của đầu vào $\mathbf{X}$ với hạt nhân
   $\mathbf{K}$ như được tính bởi `corr2d`, và $\ell$ là mất mát dạng số vô hướng
   chỉ phụ thuộc vào $\mathbf{X}$ và $\mathbf{K}$ thông qua $\mathbf{Y}$.
    1. Hãy suy ra $\partial \ell/\partial \mathbf{K}$ và
       $\partial \ell/\partial \mathbf{X}$ theo
       $\partial \ell/\partial \mathbf{Y}$, theo phong cách các phương trình
       lan truyền ngược trong :numref:`sec_backprop`.
    1. Hãy cài đặt biểu thức $\partial \ell/\partial \mathbf{K}$ của bạn
       mà không dùng tính vi phân tự động cho hàm mất mát
       $\ell = \sum_{i,j} G_{ij} Y_{ij}$, trong đó $\mathbf{G}$ là một ma trận
       ngẫu nhiên cố định có cùng kích thước với $\mathbf{Y}$. Hãy so sánh kết quả với
       gradient sai phân hữu hạn trung tâm có độ dài bước $10^{-2}$ trên
       `X` và `K` ngẫu nhiên cỡ nhỏ, rồi cho biết sai số tương đối lớn nhất.
    1. Hãy kiểm tra biểu thức $\partial \ell/\partial \mathbf{K}$ của bạn
       so với gradient hạt nhân do tính vi phân tự động
       tính được trong thí nghiệm ở :numref:`subsec_learning_kernel`.

    *Phỏng theo Michael Nielsen,
    [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html),
    Chương 6, và Stanford CS231n,
    [Bài tập 2](https://cs231n.github.io/assignments2024/assignment2/),
    notebook ConvolutionalNetworks.*

<!--en:657a396a kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/65)
<!--tr:657a396a-->
[Thảo luận](https://d2l.discourse.group/t/65)

<!--en:63c44b48 kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/66)
<!--tr:63c44b48-->
[Thảo luận](https://d2l.discourse.group/t/66)

<!--en:00b73d0b kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/271)
<!--tr:00b73d0b-->
[Thảo luận](https://d2l.discourse.group/t/271)

<!--en:b79af65b kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/17996)
<!--tr:b79af65b-->
[Thảo luận](https://d2l.discourse.group/t/17996)

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
- "blur kernel" / "sharpening kernel" → "hạt nhân làm mờ" / "hạt nhân làm sắc nét"
- "downsample" → "giảm lấy mẫu"
- "locally connected layer" → "tầng kết nối cục bộ"
- "support (of a function)" → "miền giá"
- "Discussions" → "Thảo luận"
