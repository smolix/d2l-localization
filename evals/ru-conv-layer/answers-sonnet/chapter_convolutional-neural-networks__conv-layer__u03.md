
<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u03 lang=ru rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:2294243a kind=heading-->
## Exercises
<!--tr:2294243a-->
## Упражнения

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
1. [code] **Ядра, заданные вручную.** Спроектируйте несколько ядер вручную.
    1. Для заданного вектора направления $\mathbf{v} = (v_1, v_2)$ выведите ядро обнаружения границ, которое обнаруживает границы, ортогональные $\mathbf{v}$, то есть границы в направлении $(v_2, -v_1)$.
    1. Выведите оператор конечной разности для второй производной. Какие структуры на изображениях отвечают на него сильнее всего?
    1. Каков минимальный размер ядра, позволяющего получить производную порядка $d$?
    1. Как бы вы спроектировали ядро размытия? Зачем может понадобиться такое ядро?
    1. Загрузите фотографию, переведите её в один канал и уменьшите примерно до $100 \times 100$ пикселей (циклы `corr2d` на Python). Примените детектор границ, ядро размытия и ядро повышения резкости «удвоенное тождественное ядро минус размытие» и выведите каждый результат рядом с оригиналом. Что показывает карта границ фотографии, чего не могла показать карта границ чёрной полосы?

    *Заимствовано из Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    блокнот ConvolutionalNetworks.*

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
2. [code] **Свёртка как ленточная матрица.** Матрица фрагментов из раздела :numref:`subsec_conv_matmul` разворачивает вход. Обратная конструкция разворачивает ядро. Пусть $\mathbf{X}$ — вход $4 \times 4$, $\mathbf{K}$ — ядро $3 \times 3$, а $\textrm{vec}(\cdot)$ — построчное выпрямление.
    1. Постройте матрицу $\mathbf{W}$ размера $4 \times 16$, для которой $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$, где $\mathbf{Y}$ — взаимная корреляция $\mathbf{X}$ с $\mathbf{K}$. Сколько ненулевых элементов в каждой строке и сколько различных значений встречается в $\mathbf{W}$?
    1. Постройте $\mathbf{W}$ в коде по `K` и убедитесь, что произведение воспроизводит `corr2d(X, K)`.
    1. Для входа $n \times n$ и ядра $k \times k$ подсчитайте число элементов $\mathbf{W}$, хранимой как плотный массив, число ненулевых элементов $\mathbf{W}$ и число элементов матрицы фрагментов. Как каждое из этих чисел растёт с $n$?
    1. Сохраните структуру разреженности $\mathbf{W}$, но задайте каждой строке собственные $k^2$ значений. Получится локально связный слой: тензор четвёртого порядка $\mathsf{V}$ из раздела :numref:`sec_why-conv` в формуле $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$, ограниченной окном смещений $(a, b)$ размера $k \times k$ до того, как эквивариантность относительно сдвига свяжет веса. Для входа $n \times n$ подсчитайте параметры этого слоя, полносвязного слоя с теми же формами входа и выхода и свёртки.

    *Заимствовано из CMU 11-785,
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
3. [code] **Эффективное рецептивное поле.** ● Теоретическое рецептивное поле $r$ из этого раздела ограничивает, какие входы *могут* влиять на выход; эффективное рецептивное поле измеряет, насколько сильно они влияют на самом деле :cite:`Luo.Li.Urtasun.ea.2016`. Постройте стопку из $L$ одноканальных свёрточных слоёв $3 \times 3$ без дополнения, без смещения и без нелинейности на входе $32 \times 32$, используя встроенный слой из раздела :numref:`subsec_learning_kernel`. Каждый элемент ядра берите независимо из стандартного нормального распределения.
    1. При $L = 3$ и одной выборке ядер распространите назад единичный градиент от выходного элемента вблизи центра последнего слоя до входа и постройте график модуля градиента по входу. Сравните его носитель с теоретическим рецептивным полем $7 \times 7$ и скажите, где модуль наибольший.
    1. Усредните квадрат градиента по входу по 500 выборкам ядер для $L = 3$ и для $L = 10$. Просуммируйте каждую усреднённую карту по строкам и нормируйте получившийся профиль по столбцам так, чтобы его сумма равнялась единице. Как с глубиной меняется отношение стандартного отклонения профиля к теоретической длине стороны $2L + 1$?
    1. На рис. :numref:`fig_conv_rf_paths` подсчитаны пути для $L = 3$ в одном измерении. Покажите, что для произвольного $L$ математическое ожидание квадрата градиента по входу, просуммированное по строкам, пропорционально свёртке $L$ копий вектора $(1, 1, 1)$. Выведите, как стандартное отклонение нормированного профиля растёт с $L$.

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
4. [code] **Обратный проход свёртки.** ● Пусть $\mathbf{Y}$ — одноканальная взаимная корреляция входа $\mathbf{X}$ с ядром $\mathbf{K}$, вычисляемая `corr2d`, а $\ell$ — скалярные потери, зависящие от $\mathbf{X}$ и $\mathbf{K}$ только через $\mathbf{Y}$.
    1. Выведите $\partial \ell/\partial \mathbf{K}$ и $\partial \ell/\partial \mathbf{X}$ через $\partial \ell/\partial \mathbf{Y}$ в стиле уравнений обратного распространения ошибки из раздела :numref:`sec_backprop`.
    1. Реализуйте своё выражение для $\partial \ell/\partial \mathbf{K}$ без автоматического дифференцирования для потерь $\ell = \sum_{i,j} G_{ij} Y_{ij}$, где $\mathbf{G}$ — фиксированная случайная матрица формы $\mathbf{Y}$. Сравните его с градиентом, найденным центральной конечной разностью с размером шага $10^{-2}$, на небольших случайных `X` и `K` и укажите наибольшую относительную ошибку.
    1. Сверьте своё выражение для $\partial \ell/\partial \mathbf{K}$ с градиентом по ядру, который вычисляет автоматическое дифференцирование в эксперименте из раздела :numref:`subsec_learning_kernel`.

    *Заимствовано из книги Michael Nielsen,
    [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html),
    глава 6, и из Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    блокнот ConvolutionalNetworks.*

<!--en:657a396a kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/65)
<!--tr:657a396a-->
[Обсуждения](https://d2l.discourse.group/t/65)

<!--en:63c44b48 kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/66)
<!--tr:63c44b48-->
[Обсуждения](https://d2l.discourse.group/t/66)

<!--en:00b73d0b kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/271)
<!--tr:00b73d0b-->
[Обсуждения](https://d2l.discourse.group/t/271)

<!--en:b79af65b kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/17996)
<!--tr:b79af65b-->
[Обсуждения](https://d2l.discourse.group/t/17996)

=== DECISIONS ===
- Exercises use the imperative plural polite form ("постройте", "выведите"); "Adapted from" → «Заимствовано из».
- "locally connected layer" → локально связный слой; "support" (of a function) → носитель; "dense array" → плотный массив; "banded matrix" → ленточная матрица; "sparsity pattern" → структура разреженности.
- "Discussions" link text → «Обсуждения».
- "downsample" → уменьшить; "sharpening kernel" → ядро повышения резкости; "blur kernel" → ядро размытия.

=== NOTES ===
- 6f4ddec6: "photograph ... (`corr2d` loops in Python)" translated as a parenthetical, meaning kept.
- 54bcee26: the "●" difficulty marker kept as in the English.
- ec8621f8: math tokens reordered for Russian word order (sub-item 4 sentence structure); content identical.
