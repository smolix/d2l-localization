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
1. [code] **Hạt nhân tự thiết kế.** Hãy tự thiết kế một số hạt nhân.
    1. Cho một vector chỉ hướng $\mathbf{v} = (v_1, v_2)$, hãy suy ra một hạt nhân dò cạnh phát hiện các cạnh trực giao với $\mathbf{v}$, tức là các cạnh theo hướng $(v_2, -v_1)$.
    1. Hãy suy ra một toán tử sai phân hữu hạn cho đạo hàm bậc hai. Những cấu trúc nào trong ảnh đáp ứng mạnh nhất với nó?
    1. Kích thước tối thiểu của một hạt nhân để thu được đạo hàm bậc $d$ là bao nhiêu?
    1. Bạn sẽ thiết kế một hạt nhân làm mờ như thế nào? Vì sao có thể bạn muốn dùng một hạt nhân như vậy?
    1. Hãy tải một bức ảnh chụp, chuyển thành một kênh, và giảm mẫu xuống còn khoảng $100 \times 100$ điểm ảnh (các vòng lặp `corr2d` trong Python). Áp dụng bộ dò cạnh, hạt nhân làm mờ, và hạt nhân làm nét "hai lần ánh xạ đồng nhất trừ đi phép làm mờ", rồi hiển thị từng kết quả cạnh ảnh gốc. Bản đồ cạnh của một bức ảnh cho thấy điều gì mà bản đồ cạnh của dải đen không thể cho thấy?

    *Phỏng theo Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    sổ tay ConvolutionalNetworks.*

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
2. [code] **Tích chập như một ma trận dải.** Ma trận mảnh của :numref:`subsec_conv_matmul` trải đầu vào ra. Cách xây dựng ngược lại là trải hạt nhân. Giả sử $\mathbf{X}$ là đầu vào $4 \times 4$, $\mathbf{K}$ là hạt nhân $3 \times 3$, và $\textrm{vec}(\cdot)$ là phép trải phẳng theo từng hàng.
    1. Xây dựng ma trận $\mathbf{W}$ kích thước $4 \times 16$ thỏa mãn $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$, trong đó $\mathbf{Y}$ là tương quan chéo của $\mathbf{X}$ với $\mathbf{K}$. Mỗi hàng có bao nhiêu phần tử khác không, và có bao nhiêu giá trị phân biệt xuất hiện trong $\mathbf{W}$?
    1. Hãy dựng $\mathbf{W}$ bằng mã từ `K` và xác nhận rằng tích tái tạo `corr2d(X, K)`.
    1. Với đầu vào $n \times n$ và hạt nhân $k \times k$, hãy đếm số phần tử của $\mathbf{W}$ được lưu dưới dạng mảng dày đặc, số phần tử khác không của $\mathbf{W}$, và số phần tử của ma trận mảnh. Mỗi số đếm tăng theo $n$ như thế nào?
    1. Giữ nguyên mẫu thưa (độ thưa) của $\mathbf{W}$ nhưng cho mỗi hàng $k^2$ giá trị riêng của nó. Kết quả là một tầng kết nối cục bộ: tensor bậc bốn $\mathsf{V}$ của :numref:`sec_why-conv` trong $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$, bị giới hạn trong một cửa sổ $k \times k$ gồm các độ lệch $(a, b)$ trước khi tính đẳng biến tịnh tiến ràng buộc các trọng số. Với đầu vào $n \times n$, hãy đếm số tham số của tầng này, của một tầng kết nối đầy đủ có cùng kích thước đầu vào và đầu ra, và của phép tích chập.

    *Phỏng theo CMU 11-785,
    [Homework 2 Part 1](https://deeplearning.cs.cmu.edu/S21/index.html).*

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
3. [code] **Vùng tiếp nhận hiệu dụng.** ● Vùng tiếp nhận lý thuyết $r$ của phần này chặn trên những đầu vào *có thể* ảnh hưởng đến một đầu ra; vùng tiếp nhận hiệu dụng đo mức độ chúng thực sự ảnh hưởng :cite:`Luo.Li.Urtasun.ea.2016`. Hãy dựng một chồng $L$ tầng tích chập $3 \times 3$ một kênh, không đệm, không hệ số điều chỉnh và không có tính phi tuyến, trên đầu vào $32 \times 32$, dùng tầng có sẵn của :numref:`subsec_learning_kernel`. Lấy mỗi phần tử hạt nhân độc lập từ phân phối chuẩn tắc.
    1. Với $L = 3$ và một lần lấy mẫu các hạt nhân, hãy lan truyền ngược một gradient đơn vị từ một đơn vị đầu ra gần tâm của tầng cuối về đầu vào và vẽ độ lớn của gradient theo đầu vào. So sánh giá của nó với vùng tiếp nhận lý thuyết $7 \times 7$, và cho biết độ lớn lớn nhất ở đâu.
    1. Lấy trung bình bình phương gradient theo đầu vào trên 500 lần lấy mẫu các hạt nhân, với $L = 3$ và $L = 10$. Cộng mỗi bản đồ đã lấy trung bình theo các hàng và chuẩn hóa profile cột thu được sao cho có tổng bằng một. Tỉ số giữa độ lệch chuẩn của profile và độ dài cạnh lý thuyết $2L + 1$ thay đổi thế nào theo độ sâu?
    1. :numref:`fig_conv_rf_paths` đếm các đường đi cho $L = 3$ trong một chiều. Hãy chỉ ra rằng với $L$ tổng quát, kỳ vọng của bình phương gradient theo đầu vào, cộng theo các hàng, tỉ lệ với tích chập của $L$ bản sao của vector $(1, 1, 1)$. Hãy suy ra độ lệch chuẩn của profile đã chuẩn hóa tăng theo $L$ như thế nào.

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
4. [code] **Lượt truyền ngược của một phép tích chập.** ● Giả sử $\mathbf{Y}$ là tương quan chéo một kênh của đầu vào $\mathbf{X}$ với hạt nhân $\mathbf{K}$, như `corr2d` tính, và $\ell$ là một mất mát vô hướng chỉ phụ thuộc vào $\mathbf{X}$ và $\mathbf{K}$ thông qua $\mathbf{Y}$.
    1. Hãy suy ra $\partial \ell/\partial \mathbf{K}$ và $\partial \ell/\partial \mathbf{X}$ theo $\partial \ell/\partial \mathbf{Y}$, theo phong cách của các phương trình lan truyền ngược trong :numref:`sec_backprop`.
    1. Hãy cài đặt biểu thức của bạn cho $\partial \ell/\partial \mathbf{K}$ mà không dùng tính vi phân tự động, cho mất mát $\ell = \sum_{i,j} G_{ij} Y_{ij}$, trong đó $\mathbf{G}$ là một ma trận ngẫu nhiên cố định có cùng kích thước với $\mathbf{Y}$. So sánh nó với gradient sai phân hữu hạn trung tâm với độ dài bước $10^{-2}$ trên `X` và `K` ngẫu nhiên nhỏ, và báo cáo sai số tương đối lớn nhất.
    1. Hãy kiểm tra biểu thức của bạn cho $\partial \ell/\partial \mathbf{K}$ so với gradient hạt nhân mà tính vi phân tự động tính được trong thí nghiệm của :numref:`subsec_learning_kernel`.

    *Phỏng theo Michael Nielsen,
    [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html),
    Chương 6, và Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    sổ tay ConvolutionalNetworks.*

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
- "Exercises" heading → "Bài tập"; "Adapted from" → "Phỏng theo"; "Discussions" link text → "Thảo luận".
- "locally connected layer" → tầng kết nối cục bộ; "fully connected layer" → tầng kết nối đầy đủ (glossary).
- "profile" (column profile) → "profile" kept in English, no established Vietnamese term.
- "support" (of a gradient map) → "giá" (giá của hàm), "banded matrix" → ma trận dải; "unit gradient" → gradient đơn vị.
- "standard normal distribution" → phân phối chuẩn tắc; "downsample" → giảm mẫu; "sharpening kernel" → hạt nhân làm nét; "blur" → làm mờ.

=== NOTES ===
- 6f4ddec6: "twice the identity minus the blur" rendered as "hai lần ánh xạ đồng nhất trừ đi phép làm mờ"; the quoted name is kept as a phrase.
- 54bcee26: glossary has "Gaussian distribution" → phân phối Gauss but the source says "standard normal distribution"; used phân phối chuẩn tắc, the standard Vietnamese term.
- 6f4ddec6: "notebook" in "ConvolutionalNetworks notebook" rendered "sổ tay".
