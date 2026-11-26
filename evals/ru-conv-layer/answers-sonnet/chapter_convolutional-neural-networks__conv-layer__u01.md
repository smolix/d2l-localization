
<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u01 lang=ru rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:33d3b96b kind=heading-->
# Convolutions for Images
<!--tr:33d3b96b-->
# Свёртки для изображений

<!--en:852f5868 kind=paragraph-->
In :numref:`sec_why-conv`, we derived the convolutional layer on paper from two
geometric principles: translation equivariance and locality. Now we see what
that formula actually computes on an image, and whether gradient descent can
discover useful kernels automatically from data.
<!--tr:852f5868-->
В разделе :numref:`sec_why-conv` мы вывели свёрточный слой на бумаге из двух геометрических принципов: эквивариантности относительно сдвига и локальности. Теперь посмотрим, что эта формула на самом деле вычисляет на изображении и может ли градиентный спуск самостоятельно находить полезные ядра по данным.

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
Допустим, нам нужно найти вертикальные границы, где светлая область переходит в тёмную. Вычитание из каждого пикселя его правого соседа ($[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$) даёт нуль внутри однородных областей, $+1$ там, где белое переходит в чёрное, и $-1$ там, где чёрное снова становится белым. Если провести детектор с двумя весами $(1, -1)$ по всем допустимым положениям на изображении, получится *карта границ* (edge map). В этом разделе мы реализуем двумерную взаимную корреляцию, проверим наш детектор границ, заданный вручную, дадим градиентному спуску *выучить* $(1, -1)$ прямо из данных, покажем, как свёртка превращается в одно умножение матриц (*im2col*), и выведем, как наложение слоёв расширяет рецептивное поле элемента.

<!--en:47958ed8 kind=heading-->
## Cross-Correlation
<!--tr:47958ed8-->
## Взаимная корреляция

<!--en:5e4296fe kind=heading-->
### The Operation
<!--tr:5e4296fe-->
### Операция

<!--en:ae7e906d kind=paragraph-->
Leaving channels aside for the moment, let the input $\mathbf{X}$ be an
$n_\textrm{h} \times n_\textrm{w}$ matrix and the kernel $\mathbf{K}$ be a
$k_\textrm{h} \times k_\textrm{w}$ matrix. The kernel defines a sliding
*window* of shape $k_\textrm{h} \times k_\textrm{w}$. Placing the window's
upper-left corner at $(i, j)$, we multiply the covered input patch elementwise
with $\mathbf{K}$ and sum the products to produce the output entry at $(i, j)$:
<!--tr:ae7e906d-->
Пока оставим каналы в стороне: пусть вход $\mathbf{X}$ — матрица размера $n_\textrm{h} \times n_\textrm{w}$, а ядро $\mathbf{K}$ — матрица размера $k_\textrm{h} \times k_\textrm{w}$. Ядро задаёт скользящее *окно* формы $k_\textrm{h} \times k_\textrm{w}$. Поместив верхний левый угол окна в точку $(i, j)$, мы поэлементно умножаем покрытый фрагмент входа на $\mathbf{K}$ и суммируем произведения, получая элемент выхода в точке $(i, j)$:

<!--en:3688b38b kind=paragraph-->
This is the *two-dimensional cross-correlation* of $\mathbf{X}$ with
$\mathbf{K}$, written $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ with
zero-based indices matching code. Whereas :numref:`sec_why-conv` centered
offsets symmetrically from $-\Delta$ to $\Delta$, anchoring at the upper-left
corner differs only by a fixed shift of $\Delta$ while naturally handling
even-sized kernels (such as a $2 \times 2$ kernel or our $1 \times 2$ edge
detector) that have no center pixel.
<!--tr:3688b38b-->
Это *двумерная взаимная корреляция* (two-dimensional cross-correlation) $\mathbf{X}$ с $\mathbf{K}$, которую мы записываем как $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ с индексами, отсчитываемыми от нуля, как в коде. В разделе :numref:`sec_why-conv` смещения отсчитывались от центра симметрично, от $-\Delta$ до $\Delta$; привязка к верхнему левому углу отличается лишь постоянным сдвигом на $\Delta$, зато естественно работает с ядрами чётного размера (например, с ядром $2 \times 2$ или нашим детектором границ $1 \times 2$), у которых нет центрального пикселя.

<!--en:fa1b7da9 kind=caption-->
![Two-dimensional cross-correlation. The shaded portions are the first output element and the input and kernel elements used to compute it: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
<!--tr:fa1b7da9-->
![Двумерная взаимная корреляция. Закрашены первый элемент выхода, а также элементы входа и ядра, использованные для его вычисления: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)

<!--en:2f8e5deb kind=paragraph-->
In :numref:`fig_correlation`, we slide a $2 \times 2$ kernel across a
$3 \times 3$ input one pixel at a time, left to right and top to bottom. The
window fits cleanly at four positions, yielding the $2 \times 2$ output:
<!--tr:2f8e5deb-->
На рис. :numref:`fig_correlation` мы сдвигаем ядро $2 \times 2$ по входу $3 \times 3$ на один пиксель слева направо и сверху вниз. Окно целиком помещается в четырёх положениях, что даёт выход $2 \times 2$:

<!--en:4741e012 kind=paragraph-->
Notice that the output shrinks slightly along each axis because the kernel must
stay wholly inside the input. Fitting a width-$k$ window into length $n$ leaves
$n - k + 1$ valid positions, so the output shape is
<!--tr:4741e012-->
Заметим, что выход немного уменьшается вдоль каждой оси, потому что ядро должно целиком лежать внутри входа. Окно ширины $k$ укладывается в длину $n$ ровно $n - k + 1$ способами, поэтому форма выхода равна

<!--en:12d81c14 kind=paragraph-->
Thus a $3 \times 3$ input with a $2 \times 2$ kernel yields $2 \times 2$, a
$6 \times 8$ image with a $1 \times 2$ kernel yields $6 \times 7$, and a
$28 \times 28$ Fashion-MNIST image with a $5 \times 5$ kernel yields
$24 \times 24$. Whenever we want to preserve spatial dimensions, we surround the
input with a border of zeros (*padding*), which we examine alongside strides and
dilation in :numref:`sec_padding`.
<!--tr:12d81c14-->
Таким образом, вход $3 \times 3$ с ядром $2 \times 2$ даёт выход $2 \times 2$, изображение $6 \times 8$ с ядром $1 \times 2$ даёт $6 \times 7$, а изображение Fashion-MNIST $28 \times 28$ с ядром $5 \times 5$ даёт $24 \times 24$. Если мы хотим сохранить пространственные размеры, мы окружаем вход рамкой из нулей (*дополнение*, padding); его мы рассмотрим вместе с шагом и расширением в разделе :numref:`sec_padding`.

<!--en:869d223f kind=heading-->
### Implementation
<!--tr:869d223f-->
### Реализация

<!--en:6a869f7c kind=paragraph-->
We implement :eqref:`eq_corr2d` directly in `corr2d` using a pair of loops over
every output coordinate.
<!--tr:6a869f7c-->
Мы реализуем :eqref:`eq_corr2d` напрямую в функции `corr2d` с помощью пары циклов по всем координатам выхода.

<!--en:81e1f383 kind=paragraph-->
To verify `corr2d` against our hand calculation in :numref:`fig_correlation`,
we construct the $3 \times 3$ input tensor `X` and $2 \times 2$ kernel `K` and
evaluate `corr2d(X, K)`.
<!--tr:81e1f383-->
Чтобы сверить `corr2d` с нашим ручным расчётом на рис. :numref:`fig_correlation`, построим входной тензор `X` размера $3 \times 3$ и ядро `K` размера $2 \times 2$ и вычислим `corr2d(X, K)`.

<!--en:f567e1b2 kind=paragraph-->
The output reproduces :numref:`fig_correlation` entry for entry (and as shown in
:numref:`sec_why-conv`, strict mathematical convolution with a flipped kernel
$\mathbf{K}'$ learns the exact same family of layers, so we follow standard
deep learning terminology and call cross-correlation "convolution").
<!--tr:f567e1b2-->
Выход воспроизводит рис. :numref:`fig_correlation` элемент в элемент (а как показано в разделе :numref:`sec_why-conv`, строгая математическая свёртка с перевёрнутым ядром $\mathbf{K}'$ обучает в точности то же семейство слоёв, поэтому мы следуем принятой в глубоком обучении терминологии и называем взаимную корреляцию «свёрткой»).

<!--en:bfe11c62 kind=heading-->
## Convolutional Layers
<!--tr:bfe11c62-->
## Свёрточные слои

<!--en:2cdc020d kind=heading-->
### The Layer
<!--tr:2cdc020d-->
### Слой

<!--en:c6371ece kind=paragraph-->
A two-dimensional convolutional layer cross-correlates its input with a learned
kernel and adds a scalar bias. Just like our MLP weights in
:numref:`chap_perceptrons`, we initialize the kernel from
$\mathcal{N}(0, 0.01^2)$ and the bias to zero. Below, `Conv2D` registers both
parameters and invokes `corr2d` in its forward pass; a layer with a kernel of
height $h$ and width $w$ is called an $h \times w$ convolutional layer.
<!--tr:c6371ece-->
Двумерный свёрточный слой вычисляет взаимную корреляцию входа с обучаемым ядром и прибавляет скалярное смещение. Как и веса нашего многослойного перцептрона в главе :numref:`chap_perceptrons`, ядро мы инициализируем из $\mathcal{N}(0, 0.01^2)$, а смещение — нулём. Ниже `Conv2D` регистрирует оба параметра и вызывает `corr2d` в прямом проходе; слой с ядром высоты $h$ и ширины $w$ называют свёрточным слоем $h \times w$.

<!--en:29cd9061 kind=paragraph-->
Unlike a dense layer, a convolutional layer is not locked to a fixed input
resolution because its parameters depend only on the kernel size. The exact same
$2 \times 2$ layer (5 parameters total) processes both our $3 \times 3$ tensor
`X` and a $6 \times 8$ image, returning outputs of shapes $2 \times 2$ and
$5 \times 7$:
<!--tr:29cd9061-->
В отличие от полносвязного слоя, свёрточный слой не привязан к фиксированному разрешению входа, потому что его параметры зависят только от размера ядра. Один и тот же слой $2 \times 2$ (всего 5 параметров) обрабатывает и наш тензор `X` размера $3 \times 3$, и изображение $6 \times 8$, возвращая выходы форм $2 \times 2$ и $5 \times 7$:

<!--en:c9e5aa07 kind=heading-->
### Edge Detection
<!--tr:c9e5aa07-->
### Обнаружение границ

<!--en:3d9a357b kind=paragraph-->
We now test the vertical edge detector from the introduction on a
$6 \times 8$ synthetic image whose middle four columns are black (`0`) and outer
columns are white (`1`).
<!--tr:3d9a357b-->
Теперь проверим вертикальный детектор границ из введения на синтетическом изображении $6 \times 8$, у которого четыре средних столбца чёрные (`0`), а крайние белые (`1`).

<!--en:68513ff0 kind=paragraph-->
We define the $1 \times 2$ kernel
<!--tr:68513ff0-->
Зададим ядро $1 \times 2$

<!--en:eb45803b kind=paragraph-->
At every position $(i, j)$, this computes the horizontal *finite difference*
$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, a discrete approximation (up to sign)
of the horizontal derivative of brightness, which vanishes across flat regions
and spikes wherever intensity jumps.
<!--tr:eb45803b-->
В каждой позиции $(i, j)$ оно вычисляет горизонтальную *конечную разность* (finite difference) $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ — дискретное приближение (с точностью до знака) горизонтальной производной яркости, которое обращается в нуль на ровных участках и резко возрастает там, где интенсивность скачком меняется.

<!--en:e810b3a9 kind=paragraph-->
Cross-correlating the $6 \times 8$ stripe image `X` with the $1 \times 2$
horizontal-difference kernel `K` via `corr2d(X, K)` produces the $6 \times 7$
edge map `Y`:
<!--tr:e810b3a9-->
Взаимная корреляция изображения-полосы `X` размера $6 \times 8$ с ядром горизонтальной разности `K` размера $1 \times 2$, то есть `corr2d(X, K)`, даёт карту границ `Y` размера $6 \times 7$:

<!--en:568358e4 kind=paragraph-->
Sure enough, `Y` fires `+1` along the white-to-black boundary, `-1` along the
black-to-white boundary, and `0` everywhere else. Applying the exact same kernel
to the transposed image (turning the vertical stripe into a horizontal one)
yields all zeros:
<!--tr:568358e4-->
Действительно, `Y` принимает значение `+1` вдоль границы «белое — чёрное», `-1` вдоль границы «чёрное — белое» и `0` везде в остальных местах. Если применить то же самое ядро к транспонированному изображению (вертикальная полоса превращается в горизонтальную), получатся одни нули:

<!--en:71c591a4 kind=paragraph-->
Clearly `[1, -1]` responds strictly to vertical edges. Detecting horizontal
edges, diagonals, corners, or textures requires different kernels altogether.
<!--tr:71c591a4-->
Очевидно, `[1, -1]` реагирует строго на вертикальные границы. Для обнаружения горизонтальных границ, диагоналей, углов или текстур нужны совсем другие ядра.

<!--en:f0af4887 kind=heading-->
### Learning a Kernel
<!--tr:f0af4887-->
### Обучение ядра

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
Вручную задать `[1, -1]` легко, когда мы точно знаем нужный признак, но ни один инженер не сможет вручную задать тысячи промежуточных фильтров глубокой сети. Сможет ли градиентный спуск восстановить ядро `[1, -1]` только по паре «вход — выход» `(X, Y)`? Проверим это с помощью встроенного двумерного свёрточного слоя каждого фреймворка (без смещения, с весами, инициализированными из $\mathcal{N}(0, 0.01^2)$): минимизируем квадратичную ошибку, просуммированную по всем 42 элементам выхода, за 10 шагов градиентного спуска со скоростью обучения $0.03$. Поскольку встроенные слои ожидают 4-мерные тензоры с осями батча и канала (:numref:`sec_channels`), мы соответственно изменяем форму `X` и `Y`.

<!--en:c29c5d61 kind=paragraph-->
Initially the loss sits near 12 (the near-zero random kernel misses all 12
$\pm 1$ edge pixels by roughly 1), and within just nine steps it drops below
0.01. Inspecting the learned kernel confirms what it found:
<!--tr:c29c5d61-->
Вначале потери близки к 12 (почти нулевое случайное ядро ошибается примерно на 1 на всех 12 пикселях границы со значением $\pm 1$), а всего за девять шагов они падают ниже 0.01. Выученное ядро подтверждает, что оно нашло:

<!--en:67d25b0b kind=paragraph-->
Both entries land within roughly 0.02 of `[1, -1]`. Nothing in this loop knew
anything about edges: given a loss signal, gradient descent discovered the
filter on its own. Here we supervised the single layer directly with `Y`; in a
deep CNN, backpropagation learns every layer's kernels end to end from the final
task loss alone.
<!--tr:67d25b0b-->
Оба элемента отличаются от `[1, -1]` примерно на 0.02. Ничто в этом цикле не знало о границах: получив сигнал потерь, градиентный спуск сам открыл фильтр. Здесь мы обучали один слой напрямую по `Y`; в глубокой свёрточной сети обратное распространение ошибки (backpropagation) обучает ядра всех слоёв сквозным образом, опираясь только на итоговые потери конечной задачи.

<!--en:91b0010d kind=slide_text-->
Convolutions for Images<br>**2D cross-correlation · edge detection · learned kernels · im2col · receptive fields**
<!--tr:91b0010d-->
Свёртки для изображений<br>**двумерная взаимная корреляция · обнаружение границ · обучаемые ядра · im2col · рецептивные поля**

<!--en:1d94295a kind=slide_title-->
::: {.slide title="Detecting Local Edges Across an Image"}
<!--tr:1d94295a-->
::: {.slide title="Обнаружение локальных границ на изображении"}

<!--en:687ed302 kind=kicker-->
[Motivation & Core Mental Model]{.kicker}
<!--tr:687ed302-->
[Мотивация и основная ментальная модель]{.kicker}

<!--en:05fceed0 kind=slide_text-->
- **Synthetic $6 \times 8$ test image:** White outer columns (`1`) surrounding a 4-column black vertical band (`0`).
<!--tr:05fceed0-->
- **Синтетическое изображение $6 \times 8$:** белые крайние столбцы (`1`) обрамляют вертикальную чёрную полосу (`0`) шириной 4 столбца.

<!--en:71e700b3 kind=slide_text-->
- **Horizontal finite difference:** Sweeping $\mathbf{K} = [1, -1]$ computes $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, firing `+1` at white-to-black and `-1` at black-to-white edges.
<!--tr:71e700b3-->
- **Горизонтальная конечная разность:** проход ядром $\mathbf{K} = [1, -1]$ вычисляет $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ и даёт `+1` на границах «белое — чёрное» и `-1` на границах «чёрное — белое».

<!--en:957dc033 kind=slide_text-->
- **Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent.
<!--tr:957dc033-->
- **Обучение вместо ручной настройки:** глубокие свёрточные сети автоматически находят тысячи таких ядер градиентным спуском.

<!--en:cb1045e1 kind=slide_title-->
::: {.slide title="The 2D Cross-Correlation Operation"}
<!--tr:cb1045e1-->
::: {.slide title="Операция двумерной взаимной корреляции"}

<!--en:17bcece5 kind=kicker-->
[Sliding-Window Inner Products]{.kicker}
<!--tr:17bcece5-->
[Скалярные произведения в скользящем окне]{.kicker}

<!--en:8d45aae8 kind=slide_text-->
Given an $n_\textrm{h} \times n_\textrm{w}$ input matrix $\mathbf{X}$ and a $k_\textrm{h} \times k_\textrm{w}$ kernel $\mathbf{K}$, the 2D cross-correlation $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ places the window's top-left corner at $(i, j)$ and sums elementwise products:
<!--tr:8d45aae8-->
Для входной матрицы $\mathbf{X}$ размера $n_\textrm{h} \times n_\textrm{w}$ и ядра $\mathbf{K}$ размера $k_\textrm{h} \times k_\textrm{w}$ двумерная взаимная корреляция $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ помещает верхний левый угол окна в точку $(i, j)$ и суммирует поэлементные произведения:

<!--en:e484380c kind=slide_text-->
- **Top-left anchor:** Zero-based indices $a, b \ge 0$ match array indexing in code and handle even-sized kernels (like $2 \times 2$ or $1 \times 2$) cleanly.
<!--tr:e484380c-->
- **Привязка к верхнему левому углу:** индексы с нуля, $a, b \ge 0$, соответствуют индексации массивов в коде и позволяют без затруднений работать с ядрами чётного размера (например, $2 \times 2$ или $1 \times 2$).

<!--en:137c0980 kind=slide_text-->
- **Shaded top-left output $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.
<!--tr:137c0980-->
- **Закрашенный выход $(0,0)$ в верхнем левом углу:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.

<!--en:575c4aea kind=slide_title-->
::: {.slide title="Valid Window Positions and Output Spatial Shape"}
<!--tr:575c4aea-->
::: {.slide title="Допустимые положения окна и пространственная форма выхода"}

<!--en:7eda4531 kind=kicker-->
[Why Unpadded Convolutions Shrink the Grid]{.kicker}
<!--tr:7eda4531-->
[Почему свёртки без дополнения сжимают сетку]{.kicker}

<!--en:891243ae kind=slide_text-->
- **Valid window placements:** A $k_\textrm{h} \times k_\textrm{w}$ window fits inside an $n_\textrm{h} \times n_\textrm{w}$ grid at $n - k + 1$ positions per axis.
<!--tr:891243ae-->
- **Допустимые положения окна:** окно $k_\textrm{h} \times k_\textrm{w}$ умещается в сетке $n_\textrm{h} \times n_\textrm{w}$ в $n - k + 1$ положениях вдоль каждой оси.

<!--en:55cc8c32 kind=slide_text-->
- **Shape examples:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.
<!--tr:55cc8c32-->
- **Примеры форм:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.

<!--en:1e3463da kind=slide_title-->
::: {.slide title="Implementing 2D Cross-Correlation in Code"}
<!--tr:1e3463da-->
::: {.slide title="Реализация двумерной взаимной корреляции в коде"}

<!--en:07566992 kind=kicker-->
[Direct Nested-Loop Implementation]{.kicker}
<!--tr:07566992-->
[Прямая реализация вложенными циклами]{.kicker}

<!--en:202aa5d3 kind=slide_text-->
The function `corr2d` allocates the $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ output tensor and computes each entry as the sum of `X[i:i+h, j:j+w] * K`:
<!--tr:202aa5d3-->
Функция `corr2d` выделяет выходной тензор размера $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ и вычисляет каждый его элемент как сумму `X[i:i+h, j:j+w] * K`:

<!--en:bef130ba kind=slide_text-->
Each iteration slices a $k_\textrm{h} \times k_\textrm{w}$ sub-window of `X`, multiplies elementwise with `K`, and reduces via `sum`.
<!--tr:bef130ba-->
На каждой итерации берётся срез `X` размера $k_\textrm{h} \times k_\textrm{w}$, поэлементно умножается на `K` и свёртывается функцией `sum`.

<!--en:4ee9f6b1 kind=slide_title-->
::: {.slide title="Verifying corr2d and the Flipped-Kernel Equivalence"}
<!--tr:4ee9f6b1-->
::: {.slide title="Проверка corr2d и эквивалентность перевёрнутому ядру"}

<!--en:2b0be423 kind=kicker-->
[Cross-Correlation vs. Mathematical Convolution]{.kicker}
<!--tr:2b0be423-->
[Взаимная корреляция и математическая свёртка]{.kicker}

<!--en:74b9b745 kind=slide_text-->
- **180° flipped kernel $\mathbf{K}'$:** Cross-correlating with $\mathbf{K}$ equals convolving with $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.
<!--tr:74b9b745-->
- **Ядро $\mathbf{K}'$, повёрнутое на 180°:** взаимная корреляция с $\mathbf{K}$ равна свёртке с $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.

<!--en:5c8af7cd kind=slide_text-->
- **Identical learned class:** Frameworks compute cross-correlation and call it convolution.
<!--tr:5c8af7cd-->
- **Тот же обучаемый класс:** фреймворки вычисляют взаимную корреляцию и называют её свёрткой.

<!--en:a69e29b5 kind=slide_title-->
::: {.slide title="Building a 2D Convolutional Layer Module"}
<!--tr:a69e29b5-->
::: {.slide title="Построение модуля двумерного свёрточного слоя"}

<!--en:55763603 kind=kicker-->
[Learnable Kernel Weights Plus Scalar Bias]{.kicker}
<!--tr:55763603-->
[Обучаемые веса ядра и скалярное смещение]{.kicker}

<!--en:5b8bad24 kind=slide_text-->
A convolutional layer wraps `corr2d` with a learnable $k_\textrm{h} \times k_\textrm{w}$ weight parameter (initialized from $\mathcal{N}(0, 0.01^2)$) and a scalar bias (initialized to $0$):
<!--tr:5b8bad24-->
Свёрточный слой оборачивает `corr2d` обучаемым параметром весов размера $k_\textrm{h} \times k_\textrm{w}$ (инициализируется из $\mathcal{N}(0, 0.01^2)$) и скалярным смещением (инициализируется нулём $0$):

<!--en:c16d97d6 kind=slide_title-->
::: {.slide title="Resolution Independence of Convolutional Layers"}
<!--tr:c16d97d6-->
::: {.slide title="Независимость свёрточных слоёв от разрешения"}

<!--en:a7d1e48f kind=kicker-->
[Fixed Parameters Across Arbitrary Input Sizes]{.kicker}
<!--tr:a7d1e48f-->
[Фиксированные параметры при входах любого размера]{.kicker}

<!--en:53a2bce7 kind=slide_text-->
- **Decoupled from $H \times W$:** The exact same 5 parameters ($2 \times 2$ kernel + 1 bias) process $3 \times 3$, $6 \times 8$, or megapixel inputs without reshaping weights.
<!--tr:53a2bce7-->
- **Не зависит от $H \times W$:** одни и те же 5 параметров (ядро $2 \times 2$ + 1 смещение) обрабатывают входы $3 \times 3$, $6 \times 8$ и даже мегапиксельные без изменения весов.

<!--en:017fec24 kind=slide_title-->
::: {.slide title="Constructing a Synthetic Vertical-Edge Image"}
<!--tr:017fec24-->
::: {.slide title="Построение синтетического изображения с вертикальной границей"}

<!--en:896e641c kind=kicker-->
[A 6×8 Binary Test Pattern]{.kicker}
<!--tr:896e641c-->
[Бинарный тестовый узор 6×8]{.kicker}

<!--en:0a4b456b kind=slide_text-->
We construct the $6 \times 8$ test image `X` with ones on the left and right borders and zeros across the middle four columns (`X[:, 2:6] = 0`):
<!--tr:0a4b456b-->
Построим тестовое изображение `X` размера $6 \times 8$ с единицами по левому и правому краям и нулями в четырёх средних столбцах (`X[:, 2:6] = 0`):

<!--en:dfc7e3e3 kind=slide_text-->
- **Piecewise-constant columns:** Intensity is constant (`1` or `0`) within columns $0\text{--}1$, $2\text{--}5$, and $6\text{--}7$, jumping only between columns $1 \to 2$ and $5 \to 6$.
<!--tr:dfc7e3e3-->
- **Кусочно-постоянные столбцы:** интенсивность постоянна (`1` или `0`) в столбцах $0\text{--}1$, $2\text{--}5$ и $6\text{--}7$ и скачком меняется только между столбцами $1 \to 2$ и $5 \to 6$.

<!--en:3ce31cbe kind=slide_title-->
::: {.slide title="Applying the Horizontal Finite-Difference Kernel"}
<!--tr:3ce31cbe-->
::: {.slide title="Применение ядра горизонтальной конечной разности"}

<!--en:b37a1029 kind=kicker-->
[Detecting Vertical Edges with K = [1, -1]]{.kicker}
<!--tr:b37a1029-->
[Обнаружение вертикальных границ ядром K = [1, -1]]{.kicker}

<!--en:9be85019 kind=slide_text-->
Cross-correlating `X` with the $1 \times 2$ kernel $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ computes $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ at every horizontal pair:
<!--tr:9be85019-->
Взаимная корреляция `X` с ядром $1 \times 2$ $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ вычисляет $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ для каждой горизонтальной пары:

<!--en:f649cfae kind=slide_text-->
- **Signed edge map:** Column 1 fires `+1` (white-to-black transition), column 5 fires `-1` (black-to-white transition), and all constant interior columns evaluate to `0`.
<!--tr:f649cfae-->
- **Знаковая карта границ:** в столбце 1 получаем `+1` (переход «белое — чёрное»), в столбце 5 получаем `-1` (переход «чёрное — белое»), а во всех внутренних столбцах с постоянным значением — `0`.

<!--en:e6d155be kind=slide_title-->
::: {.slide title="Orientation Selectivity of Edge Kernels"}
<!--tr:e6d155be-->
::: {.slide title="Избирательность ядер границ к ориентации"}

<!--en:b525cde4 kind=kicker-->
[Why One Kernel Cannot Detect All Orientations]{.kicker}
<!--tr:b525cde4-->
[Почему одно ядро не обнаружит все ориентации]{.kicker}

<!--en:38159de6 kind=slide_text-->
- **Blind to horizontal edges:** Transposing `X` (`X.T`) makes horizontal neighbors identical, so $[1, -1]$ outputs all zeros.
<!--tr:38159de6-->
- **Слепота к горизонтальным границам:** после транспонирования `X` (`X.T`) горизонтальные соседи совпадают, поэтому $[1, -1]$ даёт одни нули.

<!--en:f0f30f76 kind=slide_text-->
- **Filter banks (§{sec_channels}):** Vision layers learn parallel channels for horizontal, vertical, and diagonal orientations.
<!--tr:f0f30f76-->
- **Банки фильтров (§{sec_channels}):** слои зрения обучают параллельные каналы для горизонтальной, вертикальной и диагональной ориентаций.

<!--en:b755e31b kind=slide_title-->
::: {.slide title="Learning an Edge Detector via Gradient Descent"}
<!--tr:b755e31b-->
::: {.slide title="Обучение детектора границ градиентным спуском"}

<!--en:86bbeea7 kind=kicker-->
[Supervising a Built-In Convolutional Layer on (X, Y)]{.kicker}
<!--tr:86bbeea7-->
[Обучение встроенного свёрточного слоя по паре (X, Y)]{.kicker}

<!--en:a569780c kind=slide_text-->
- **4D framework tensors:** Reshape `X` and `Y` to 4D tensors (one example, one channel) and minimize $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ at $\eta = 0.03$.
<!--tr:a569780c-->
- **4-мерные тензоры фреймворков:** преобразуем `X` и `Y` в 4-мерные тензоры (один пример, один канал) и минимизируем $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ при $\eta = 0.03$.

<!--en:2489ce6e kind=slide_title-->
::: {.slide title="Inspecting the Learned Kernel Weights"}
<!--tr:2489ce6e-->
::: {.slide title="Проверка выученных весов ядра"}

<!--en:7e4ef44b kind=kicker-->
[Why Initial Loss Is 12 and Converges to [1, -1]]{.kicker}
<!--tr:7e4ef44b-->
[Почему начальные потери равны 12 и сходятся к [1, -1]]{.kicker}

<!--en:9cd1c645 kind=slide_text-->
- **Why loss starts at $12.0$:** Near-zero weights predict $\hat{Y} \approx 0$, missing the twelve $\pm 1$ edge pixels in `Y` ($12 \times 1^2 = 12$).
<!--tr:9cd1c645-->
- **Почему потери начинаются с $12.0$:** почти нулевые веса дают $\hat{Y} \approx 0$ и пропускают двенадцать пикселей границы $\pm 1$ в `Y` ($12 \times 1^2 = 12$).

<!--en:456670c5 kind=slide_text-->
- **Convergence in 9 steps:** SGD recovers $[0.99, -0.99]$ automatically from data.
<!--tr:456670c5-->
- **Сходимость за 9 шагов:** SGD автоматически восстанавливает $[0.99, -0.99]$ по данным.

=== DECISIONS ===
- Register: "мы" (first person plural, as in the English), neutral technical prose; no "вы".
- First use of key terms in a section gets the English in parentheses (взаимная корреляция (cross-correlation) etc.); glossary terms otherwise used as given.
- "cross-correlation" → взаимная корреляция; "convolution" → свёртка; "kernel" → ядро (short for ядро свёртки); "padding" → дополнение; "bias" (layer) → смещение; "loss" → потери; "learning rate" → скорость обучения.
- "edge map" → карта границ; "edge detection" → обнаружение границ; "edge" (image) → граница.
- "window" → окно; "patch" → фрагмент; "feature map" → карта признаков (planned); "squared error" → квадратичная ошибка.
- Figure references are written «рис. :numref:`fig_x`», sections «разделе :numref:`sec_x`», chapters «главе :numref:`chap_x`».
- Slide kickers/titles: sentence case, no trailing period.
- "framework" → фреймворк; "tensor shape" → форма.

=== NOTES ===
- 852f5868 etc.: numref text is placed after "рис."/"разделе" because the build expands :numref: to the number only.
- f0f30f76: "§{sec_channels}" kept as-is; "Filter banks" rendered as «банки фильтров».
- 575c4aea/891243ae: English "n - k + 1" formula notation kept as written even though h/w subscripts omitted in the source.
- 8d45aae8: math tokens reordered for Russian word order ("для входной матрицы X размера n_h x n_w"); content identical.
