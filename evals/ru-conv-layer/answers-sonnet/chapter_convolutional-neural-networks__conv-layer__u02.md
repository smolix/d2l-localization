
<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u02 lang=ru rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:0b85caf9 kind=heading-->
## Convolution as Matrix Multiplication
<!--tr:0b85caf9-->
## Свёртка как умножение матриц

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
Вложенные циклы в `corr2d` делают математику прозрачной, но перебирать выходные пиксели по одному на Python слишком медленно для реальных задач. Классическое средство, до сих пор входящее в число алгоритмов, из которых выбирают библиотеки, — переписать взаимную корреляцию как одно большое умножение матриц, то есть именно ту операцию, которую GPU выполняют на пиковой скорости. Заметим, что каждый элемент выхода на рис. :numref:`fig_correlation` — это просто скалярное произведение (dot product) выпрямленного в вектор ядра из $k_\textrm{h} k_\textrm{w}$ элементов и соответствующего выпрямленного окна входа. Если извлечь все $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ окон и сложить их в строки, получится матрица фрагментов; умножив её на выпрямленное ядро, мы вычислим всю взаимную корреляцию за один раз. Это стандартное преобразование раскладки называется *im2col* («изображение в столбцы», а здесь скорее в строки). На рис. :numref:`fig_conv_im2col` оно выполнено на примере рис. :numref:`fig_correlation`.

<!--en:107554a5 kind=caption-->
![The cross-correlation of :numref:`fig_correlation` as one matrix product. Each of the four $2 \times 2$ windows of the input becomes a row of the patch matrix, and multiplying by the flattened kernel gives the flattened output. The shaded cells follow the first window to its row and to the output 19. The center pixel 4 lies in all four windows, so it appears in every row.](../img/conv-im2col.svg)
<!--tr:107554a5-->
![Взаимная корреляция с рис. :numref:`fig_correlation` как одно матричное произведение. Каждое из четырёх окон $2 \times 2$ входа становится строкой матрицы фрагментов, а умножение на выпрямленное ядро даёт выпрямленный выход. Закрашенные ячейки прослеживают первое окно до его строки и до выхода 19. Центральный пиксель 4 лежит во всех четырёх окнах, поэтому он присутствует в каждой строке.](../img/conv-im2col.svg)

<!--en:76713479 kind=paragraph-->
The trade-off is extra workspace memory, since each interior pixel is duplicated
into up to $k_\textrm{h} k_\textrm{w}$ patches. Below we verify on
:numref:`fig_correlation` that this single matrix multiply matches `corr2d`
exactly:
<!--tr:76713479-->
Цена этого подхода — дополнительная рабочая память, поскольку каждый внутренний пиксель дублируется не более чем в $k_\textrm{h} k_\textrm{w}$ фрагментах. Ниже мы проверяем на примере рис. :numref:`fig_correlation`, что это единственное матричное умножение в точности совпадает с `corr2d`:

<!--en:3bcd7d8a kind=paragraph-->
The discrepancy is identically zero: one matrix product reproduces `corr2d`
perfectly.
<!--tr:3bcd7d8a-->
Расхождение тождественно равно нулю: одно матричное произведение в точности воспроизводит `corr2d`.

<!--en:c30f2dea kind=paragraph-->
Equivalently, we can unroll the *kernel* instead of the input into a sparse
banded matrix $\mathbf{W}$ multiplying the flattened image—the exact 2D analog
of panel (c) in :numref:`fig_why_conv_weight_sharing`, where every row repeats
the same handful of kernel weights. Seeing a convolution as a structured dense
layer brings us full circle to the derivation of :numref:`sec_why-conv` (and the
exercises build $\mathbf{W}$ explicitly).
<!--tr:c30f2dea-->
Эквивалентно можно развернуть не вход, а *ядро* — в разреженную ленточную матрицу $\mathbf{W}$, умножаемую на выпрямленное изображение. Это в точности двумерный аналог панели (c) на рис. :numref:`fig_why_conv_weight_sharing`, где каждая строка повторяет одни и те же несколько весов ядра. Взгляд на свёртку как на структурированный полносвязный слой возвращает нас к выводу из раздела :numref:`sec_why-conv` (а в упражнениях матрица $\mathbf{W}$ строится явно).

<!--en:ad4fc956 kind=heading-->
## Feature Maps and Receptive Fields
<!--tr:ad4fc956-->
## Карты признаков и рецептивные поля

<!--en:e7983df8 kind=paragraph-->
A single convolutional layer sees only a patch the size of its kernel, whereas
recognizing whole objects demands integrating context across much wider areas.
Recall from :numref:`subsec_why-conv-channels` that a convolutional layer's
output is called a *feature map* (like our vertical edge map `Y`). For any
element $x$ in a feature map, its *receptive field* is the region of the input
image (or of an earlier layer) whose values can influence $x$ during forward
propagation—borrowing the neurophysiology term from :numref:`field_visual`.
<!--tr:e7983df8-->
Один свёрточный слой видит лишь фрагмент размером с его ядро, тогда как для распознавания целых объектов нужно объединять контекст из гораздо более широких областей. Вспомним из раздела :numref:`subsec_why-conv-channels`, что выход свёрточного слоя называется *картой признаков* (feature map) (как наша карта вертикальных границ `Y`). Для любого элемента $x$ карты признаков его *рецептивное поле* (receptive field) — это область входного изображения (или более раннего слоя), значения которой могут влиять на $x$ при прямом проходе; мы заимствуем этот термин из нейрофизиологии, как в разделе :numref:`field_visual`.

<!--en:b36f221e kind=heading-->
### Stacked Layers
<!--tr:b36f221e-->
### Наложенные слои

<!--en:24a25ec4 kind=paragraph-->
In :numref:`fig_correlation`, the shaded output `19` has a $2 \times 2$ receptive
field covering the top-left four pixels of $\mathbf{X}$. Now stack a *second*
$2 \times 2$ convolutional layer on top of the $2 \times 2$ output $\mathbf{Y}$
to produce a single scalar $z$: $z$ depends on all four elements of $\mathbf{Y}$,
which together depend on all nine elements of $\mathbf{X}$. Stacking layers
expands the receptive field smoothly with depth.
<!--tr:24a25ec4-->
На рис. :numref:`fig_correlation` закрашенный выходной элемент `19` имеет рецептивное поле $2 \times 2$, покрывающее четыре верхних левых пикселя $\mathbf{X}$. Теперь наложим *второй* свёрточный слой $2 \times 2$ на выход $\mathbf{Y}$ размера $2 \times 2$, чтобы получить один скаляр $z$: $z$ зависит от всех четырёх элементов $\mathbf{Y}$, которые вместе зависят от всех девяти элементов $\mathbf{X}$. Наложение слоёв плавно расширяет рецептивное поле с глубиной.

<!--en:f95c9394 kind=paragraph-->
How quickly does it grow? Each $k \times k$ convolutional layer reaches $k - 1$
pixels farther than a single element of its input, so stacking $L$ layers with
kernel sizes $k_1, \ldots, k_L$ yields a receptive field of side length
<!--tr:f95c9394-->
Как быстро оно растёт? Каждый свёрточный слой $k \times k$ достаёт на $k - 1$ пиксель дальше, чем один элемент его входа, поэтому наложение $L$ слоёв с размерами ядер $k_1, \ldots, k_L$ даёт рецептивное поле со стороной

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
Для стопки из $L$ последовательных слоёв $3 \times 3$ каждый слой добавляет 2 пикселя, так что покрывается $(2L + 1) \times (2L + 1)$ элементов входа: $5 \times 5$ после двух слоёв и $7 \times 7$ после трёх (:numref:`fig_conv_receptive_field`). Этот линейный рост на самом деле благо, а не повод браться за огромные ядра: три наложенных слоя $3 \times 3$ дают то же рецептивное поле $7 \times 7$, что и один слой $7 \times 7$, но с гораздо меньшим числом параметров ($3 \times 9 = 27$ весов против 49 на пару каналов), да ещё и перемежаются тремя нелинейностями вместо одной.

<!--en:39917513 kind=caption-->
![Receptive field of one unit after three one-dimensional layers with kernel size 3. Shaded cells are the units that the top unit depends on, and the lines connect each of them to its inputs. Each layer widens the receptive field by 2, to 7 inputs.](../img/conv-receptive-field.svg)
<!--tr:39917513-->
![Рецептивное поле одного элемента после трёх одномерных слоёв с размером ядра 3. Закрашены элементы, от которых зависит верхний элемент, а линии соединяют каждый из них с его входами. Каждый слой расширяет рецептивное поле на 2, до 7 элементов входа.](../img/conv-receptive-field.svg)

<!--en:48e4c9cc kind=heading-->
### Effective Receptive Fields
<!--tr:48e4c9cc-->
### Эффективные рецептивные поля

<!--en:77d044db kind=paragraph-->
Keep in mind that $r$ measures the *theoretical* receptive field (every input
pixel with *any* path to an output unit) without saying how strongly each pixel
actually contributes. Measuring input gradients directly,
:citet:`Luo.Li.Urtasun.ea.2016` showed that the *effective* receptive field
decays sharply away from the center in a roughly Gaussian bell curve, in both
untrained and trained networks. Deep units can theoretically see far across the
image, but their sensitivity concentrates heavily near the center.
<!--tr:77d044db-->
Следует помнить, что $r$ измеряет *теоретическое* рецептивное поле (каждый входной пиксель, от которого к выходному элементу идёт *хотя бы один* путь), но не говорит, насколько сильно каждый пиксель на самом деле вносит вклад. Измеряя градиенты по входу напрямую, :citet:`Luo.Li.Urtasun.ea.2016` показали, что *эффективное* рецептивное поле резко спадает от центра по примерно гауссовой колоколообразной кривой как в необученных, так и в обученных сетях. Глубокие элементы теоретически могут видеть далеко по изображению, но их чувствительность сильно сосредоточена около центра.

<!--en:eed4e2a0 kind=paragraph-->
Counting paths explains the bell shape. An input at the edge of the receptive
field reaches the output unit along a single chain of connections, whereas the
center input can take many routes through the layers in between.
:numref:`fig_conv_rf_paths` counts them for the tower of
:numref:`fig_conv_receptive_field`: the center input has seven paths to the top
unit and each edge input has one. If all weights have a similar magnitude, the
influence of an input grows with its number of paths.
<!--tr:eed4e2a0-->
Подсчёт путей объясняет колоколообразную форму. Вход на краю рецептивного поля достигает выходного элемента по единственной цепочке связей, тогда как центральный вход может пройти по многим маршрутам через промежуточные слои. На рис. :numref:`fig_conv_rf_paths` они подсчитаны для башни из рис. :numref:`fig_conv_receptive_field`: у центрального входа семь путей к верхнему элементу, а у каждого крайнего входа — по одному. Если все веса имеют сходную величину, влияние входа растёт с числом его путей.

<!--en:b92775ee kind=caption-->
![Number of paths from each unit to the top unit of :numref:`fig_conv_receptive_field` (three one-dimensional layers with kernel size 3). Each count is the sum of the counts of the units that the cell feeds. Darker cells have more paths than the other cells of their layer, and the bars repeat the counts of the input row: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)
<!--tr:b92775ee-->
![Число путей от каждого элемента до верхнего элемента на рис. :numref:`fig_conv_receptive_field` (три одномерных слоя с размером ядра 3). Каждое число равно сумме чисел для тех элементов, которые питает данная ячейка. Более тёмные ячейки имеют больше путей, чем остальные ячейки своего слоя, а столбики повторяют числа входной строки: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)

<!--en:405695e8 kind=paragraph-->
Two practical gaps remain in our basic layer: every unpadded convolution shrinks
spatial dimensions while under-sampling border pixels, and a single kernel
detects only one pattern. We address spatial sizing and borders next in
:numref:`sec_padding`, and multi-channel feature maps in :numref:`sec_channels`.
<!--tr:405695e8-->
В нашем базовом слое остаются две практические проблемы: каждая свёртка без дополнения уменьшает пространственные размеры и при этом недостаточно учитывает граничные пиксели, а одно ядро обнаруживает только один шаблон. Пространственные размеры и границы мы рассмотрим далее в разделе :numref:`sec_padding`, а многоканальные карты признаков — в разделе :numref:`sec_channels`.

<!--en:30ac03ff kind=heading-->
## Summary
<!--tr:30ac03ff-->
## Резюме

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
Свёрточный слой применяет двумерную взаимную корреляцию с обучаемым ядром и добавляет скалярное смещение, скользя ядром по всем допустимым окнам входа. Простое ядро конечной разности `[1, -1]` размера $1 \times 2$ обнаруживает вертикальные границы в любом месте изображения, а градиентный спуск восстанавливает именно эти веса только по данным «вход — выход», точно так же как глубокие свёрточные сети сквозным образом обучают целые иерархии фильтров. Математическая свёртка переворачивает ядро, что не играет роли, раз ядра обучаются. Свёртку можно вычислить как единое умножение матриц (*im2col*), а наложение $L$ свёрточных слоёв линейно расширяет рецептивное поле каждого элемента на $k_i - 1$ за слой.

<!--en:3ba929d2 kind=slide_title-->
::: {.slide title="Fast Execution: Convolution via Matrix Multiplication"}
<!--tr:3ba929d2-->
::: {.slide title="Быстрое выполнение: свёртка через умножение матриц"}

<!--en:1711f812 kind=kicker-->
[The im2col (Image-to-Columns) Transformation]{.kicker}
<!--tr:1711f812-->
[Преобразование im2col (изображение в столбцы)]{.kicker}

<!--en:a040f3d7 kind=slide_text-->
- **Single dense GEMM:** Flatten each $k_\textrm{h} \times k_\textrm{w}$ window into a row of `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ and multiply by $\textrm{vec}(\mathbf{K})$.
<!--tr:a040f3d7-->
- **Одно плотное GEMM:** выпрямите каждое окно $k_\textrm{h} \times k_\textrm{w}$ в строку матрицы `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ и умножьте на $\textrm{vec}(\mathbf{K})$.

<!--en:1fb18ec4 kind=slide_text-->
- **Trade-off:** Peak GPU Tensor Core speed at the cost of $k_\textrm{h} k_\textrm{w}\times$ patch workspace duplication.
<!--tr:1fb18ec4-->
- **Компромисс:** пиковая скорость тензорных ядер GPU ценой дублирования фрагментов в рабочей памяти в $k_\textrm{h} k_\textrm{w}\times$ раз.

<!--en:0d757cf6 kind=slide_title-->
::: {.slide title="Dual View: Unrolling the Kernel into a Banded Matrix"}
<!--tr:0d757cf6-->
::: {.slide title="Двойственный взгляд: развёртка ядра в ленточную матрицу"}

<!--en:7f0defaa kind=kicker-->
[Toeplitz Weight Matrix Multiplication]{.kicker}
<!--tr:7f0defaa-->
[Умножение на тёплицеву матрицу весов]{.kicker}

<!--en:5898ba57 kind=slide_text-->
Instead of unfolding the input image into duplicated patches, we can unroll the *kernel* into a sparse banded matrix $\mathbf{W}$ that multiplies the flattened input vector $\textrm{vec}(\mathbf{X})$:
<!--tr:5898ba57-->
Вместо раскладки входного изображения в дублирующиеся фрагменты можно развернуть *ядро* в разреженную ленточную матрицу $\mathbf{W}$, которая умножается на выпрямленный входной вектор $\textrm{vec}(\mathbf{X})$:

<!--en:dae51741 kind=slide_text-->
- **Matrix–vector form:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, where $\mathbf{W}$ has $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ rows and $n_\textrm{h} n_\textrm{w}$ columns.
<!--tr:dae51741-->
- **Матрично-векторная форма:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, где в $\mathbf{W}$ строк $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ и столбцов $n_\textrm{h} n_\textrm{w}$.

<!--en:f0e86ee3 kind=slide_text-->
- **Connection to §{sec_why-conv} and backprop:** Every row of $\mathbf{W}$ holds the same $k_\textrm{h} k_\textrm{w}$ kernel entries shifted across columns (panel c), making the backward pass with respect to $\mathbf{X}$ simply multiplication by $\mathbf{W}^\top$ (a transposed convolution).
<!--tr:f0e86ee3-->
- **Связь с §{sec_why-conv} и обратным распространением:** каждая строка $\mathbf{W}$ содержит одни и те же $k_\textrm{h} k_\textrm{w}$ элементов ядра, сдвинутых по столбцам (панель c), поэтому обратный проход по $\mathbf{X}$ — это просто умножение на $\mathbf{W}^\top$ (транспонированная свёртка).

<!--en:af8cf19a kind=slide_title-->
::: {.slide title="Feature Maps and Receptive Fields Across Depth"}
<!--tr:af8cf19a-->
::: {.slide title="Карты признаков и рецептивные поля по глубине"}

<!--en:4eb965f3 kind=kicker-->
[Linear Receptive-Field Growth in Stacked Layers]{.kicker}
<!--tr:4eb965f3-->
[Линейный рост рецептивного поля в наложенных слоях]{.kicker}

<!--en:5599a6e9 kind=slide_text-->
A unit's **receptive field** is the region of the original input image whose pixels can influence that unit's activation during forward propagation:
<!--tr:5599a6e9-->
**Рецептивное поле** элемента — это область исходного входного изображения, пиксели которой могут влиять на активацию этого элемента при прямом проходе:

<!--en:eba74f75 kind=slide_text-->
- **Single layer ($L=1$):** A $k_1 \times k_1$ kernel sees a $k_1 \times k_1$ input patch.
<!--tr:eba74f75-->
- **Один слой ($L=1$):** ядро $k_1 \times k_1$ видит фрагмент входа $k_1 \times k_1$.

<!--en:01c175bf kind=slide_text-->
- **Stacking $L$ stride-1 layers:** Layer $i$ with kernel size $k_i$ reaches $k_i - 1$ pixels farther along each axis than a single element of layer $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$
<!--tr:01c175bf-->
- **Наложение $L$ слоёв с шагом 1:** слой $i$ с размером ядра $k_i$ достаёт вдоль каждой оси на $k_i - 1$ пиксель дальше, чем один элемент слоя $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$

<!--en:2bde5822 kind=slide_text-->
- **$L$ stacked $3 \times 3$ layers:** Each layer adds $3 - 1 = 2$ pixels, giving side length $r = 2L + 1$ ($3 \to 5 \to 7$).
<!--tr:2bde5822-->
- **$L$ наложенных слоёв $3 \times 3$:** каждый слой добавляет $3 - 1 = 2$ пикселя, давая сторону $r = 2L + 1$ ($3 \to 5 \to 7$).

<!--en:cbb9e97a kind=slide_title-->
::: {.slide title="Why Deep Stacks of 3×3 Kernels Beat One Large Kernel"}
<!--tr:cbb9e97a-->
::: {.slide title="Почему глубокие стопки ядер 3×3 лучше одного большого ядра"}

<!--en:95e46e70 kind=kicker-->
[Parameter Efficiency & Nonlinear Expressivity]{.kicker}
<!--tr:95e46e70-->
[Экономия параметров и нелинейная выразительность]{.kicker}

<!--en:bd1d7cc0 kind=slide_text-->
- **Matched $7 \times 7$ receptive field:** Three stacked $3 \times 3$ layers ($r = 1 + 2 + 2 + 2 = 7$) see the exact same $7 \times 7$ patch as one $7 \times 7$ layer.
<!--tr:bd1d7cc0-->
- **Совпадающее рецептивное поле $7 \times 7$:** три наложенных слоя $3 \times 3$ ($r = 1 + 2 + 2 + 2 = 7$) видят в точности тот же фрагмент $7 \times 7$, что и один слой $7 \times 7$.

<!--en:755ac11a kind=slide_text-->
- **45% fewer parameters:** $3 \times 9 c^2 = 27 c^2$ weights instead of $49 c^2$ across $c$ channels.
<!--tr:755ac11a-->
- **На 45% меньше параметров:** $3 \times 9 c^2 = 27 c^2$ весов вместо $49 c^2$ при $c$ каналах.

<!--en:e3aa1086 kind=slide_text-->
- **Richer expressivity:** Compounds **3 nonlinear activations** instead of 1 (the core VGG design rule in §{sec_blocks}).
<!--tr:e3aa1086-->
- **Большая выразительность:** **3 нелинейные активации** вместо одной (основное правило проектирования VGG, §{sec_blocks}).

<!--en:9ec83bf6 kind=slide_title-->
::: {.slide title="Theoretical vs. Effective Receptive Fields"}
<!--tr:9ec83bf6-->
::: {.slide title="Теоретическое и эффективное рецептивные поля"}

<!--en:72a52c3b kind=kicker-->
[Gaussian Decay of Sensitivity Away from the Center]{.kicker}
<!--tr:72a52c3b-->
[Гауссов спад чувствительности вдали от центра]{.kicker}

<!--en:33bb377d kind=slide_text-->
- **Theoretical RF ($r = 2L+1$):** Bounding box of all pixels with *at least one* path to the output unit.
<!--tr:33bb377d-->
- **Теоретическое РП ($r = 2L+1$):** ограничивающая рамка всех пикселей, у которых *есть хотя бы один* путь к выходному элементу.

<!--en:293436eb kind=slide_text-->
- **Central Limit Theorem:** Center pixels connect via exponentially many paths, whereas corner pixels connect via only **1 path**.
<!--tr:293436eb-->
- **Центральная предельная теорема:** центральные пиксели связаны с выходом экспоненциально многими путями, а угловые — только **одним путём**.

<!--en:f47d181e kind=slide_text-->
- **Effective RF (Luo et al., 2016):** Input gradient magnitude $\partial y / \partial \mathbf{X}$ decays in a Gaussian bell curve with std $\sigma \propto \sqrt{L}$.
<!--tr:f47d181e-->
- **Эффективное РП (Luo et al., 2016):** модуль градиента по входу $\partial y / \partial \mathbf{X}$ спадает по гауссовой колоколообразной кривой со стандартным отклонением $\sigma \propto \sqrt{L}$.

<!--en:fac792d0 kind=slide_title-->
::: {.slide title="Concept Check: Cross-Correlation & Receptive Fields"}
<!--tr:fac792d0-->
::: {.slide title="Проверка понимания: взаимная корреляция и рецептивные поля"}

<!--en:26a317d8 kind=kicker-->
[Concept Check]{.kicker}
<!--tr:26a317d8-->
[Проверка понимания]{.kicker}

<!--en:8fe554e5 kind=slide_text-->
Test your understanding of output shapes, `im2col`, and receptive-field arithmetic:
<!--tr:8fe554e5-->
Проверьте, как вы понимаете формы выходов, `im2col` и арифметику рецептивных полей:

<!--en:281c7c78 kind=slide_text-->
1. **im2col matrix dimensions:** You apply an unpadded $3 \times 3$ single-channel kernel to a $10 \times 10$ image via `im2col`. What is the output spatial shape, and what are the exact dimensions of the `patches` matrix and flattened kernel vector?
<!--tr:281c7c78-->
1. **Размеры матрицы im2col:** вы применяете ядро $3 \times 3$ без дополнения и с одним каналом к изображению $10 \times 10$ через `im2col`. Какова пространственная форма выхода и каковы точные размеры матрицы `patches` и выпрямленного вектора ядра?

<!--en:16976d18 kind=slide_text-->
2. **Depth to span Fashion-MNIST:** At stride 1, how many stacked $3 \times 3$ layers are required so a single unit's theoretical receptive field spans an entire $28 \times 28$ image? How many with $5 \times 5$ layers?
<!--tr:16976d18-->
2. **Глубина для охвата Fashion-MNIST:** при шаге 1 сколько наложенных слоёв $3 \times 3$ нужно, чтобы теоретическое рецептивное поле одного элемента охватило всё изображение $28 \times 28$? А сколько слоёв $5 \times 5$?

<!--en:ab2e0d6a kind=slide_text-->
3. **Edge-loss boundary shift:** In our $6 \times 8$ vertical-edge learning experiment, why did the loss start at $\approx 12$, and what would the initial loss be if the black band extended all the way to the right edge (`X[:, 2:8] = 0`)?
<!--tr:ab2e0d6a-->
3. **Сдвиг границы и потери на границах:** почему в нашем эксперименте по обучению ядра на изображении $6 \times 8$ с вертикальной границей потери начинались с $\approx 12$ и каковы были бы начальные потери, если бы чёрная полоса доходила до правого края (`X[:, 2:8] = 0`)?

<!--en:250d6877 kind=slide_text-->
**Answers:**
(1) Output shape is $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ outputs). The `patches` matrix has shape $(64, 9)$ and multiplies the $(9, 1)$ kernel vector.
(2) For $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ layers. For $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ layers.
(3) Each of the 6 rows had two $\pm 1$ edges ($6 \times 2 \times 1^2 = 12$). Extending the black band to the right border eliminates the right-hand black-to-white edge, leaving one edge per row and halving the initial loss to $\approx 6$.
<!--tr:250d6877-->
**Ответы:**
(1) Форма выхода $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ выхода). Матрица `patches` имеет форму $(64, 9)$ и умножается на вектор ядра $(9, 1)$.
(2) Для $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ слоёв. Для $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ слоёв.
(3) В каждой из 6 строк было по две границы $\pm 1$ ($6 \times 2 \times 1^2 = 12$). Если продлить чёрную полосу до правого края, граница «чёрное — белое» справа исчезает, остаётся по одной границе на строку, и начальные потери уменьшаются вдвое до $\approx 6$.

<!--en:ba43a2dd kind=slide_title-->
::: {.slide title="Summary & Key Takeaways"}
<!--tr:ba43a2dd-->
::: {.slide title="Резюме и главные выводы"}

<!--en:ce12e85f kind=kicker-->
[Section Synthesis]{.kicker}
<!--tr:ce12e85f-->
[Итог раздела]{.kicker}

<!--en:00011fb7 kind=slide_text-->
- **2D Cross-Correlation & Edge Learning:** Slides $\mathbf{K}$ across valid windows ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); SGD learns $[1, -1]$ in 9 steps.
<!--tr:00011fb7-->
- **Двумерная взаимная корреляция и обучение границ:** скользит $\mathbf{K}$ по допустимым окнам ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); SGD выучивает $[1, -1]$ за 9 шагов.

<!--en:09efe4f0 kind=slide_text-->
- **im2col Matrix Multiply:** Trades $k_\textrm{h} k_\textrm{w}\times$ patch duplication for peak GPU GEMM throughput.
<!--tr:09efe4f0-->
- **Матричное умножение im2col:** меняет дублирование фрагментов в $k_\textrm{h} k_\textrm{w}\times$ раз на пиковую пропускную способность GEMM на GPU.

<!--en:3af8acd0 kind=slide_text-->
- **Stacked Receptive Fields:** $r = 1 + \sum (k_i - 1) = 2L+1$ for $3 \times 3$ stacks, with Gaussian center sensitivity.
<!--tr:3af8acd0-->
- **Рецептивные поля в стопках:** $r = 1 + \sum (k_i - 1) = 2L+1$ для стопок $3 \times 3$ с гауссовой чувствительностью в центре.

=== DECISIONS ===
- "receptive field" → рецептивное поле; abbreviation in one slide: РП (introduced as "рецептивное поле" in the slide title just before).
- "feature map" → карта признаков; "unit" (of a layer) → элемент; "patch" → фрагмент; "patch matrix" → матрица фрагментов; "window" → окно.
- "im2col" stays Latin; "GEMM" stays Latin; "workspace memory" → рабочая память.
- "stack(ed) layers" → наложение/наложенные слои; "stack" of layers → стопка.
- "theoretical / effective receptive field" → теоретическое / эффективное рецептивное поле.
- Slides address the reader in imperative plural ("вы", "выпрямите") only in the concept-check/instruction lines; narrative stays "мы".
- "nonlinearity" → нелинейность; "activation" → активация.

=== NOTES ===
- 5bdffcd7, 24a25ec4 etc.: English "Notice that" → «Заметим, что».
- 5599a6e9: "forward propagation" translated «прямой проход» per glossary.
- 293436eb: English says "exponentially many paths", kept as stated.
- f47d181e: "Luo et al., 2016" kept in Latin as a citation.
- 755ac11a: "45% fewer" kept as in English (27/49 is about 45% fewer).
- e3aa1086 and 293436eb: "numbers not found: 1" is a false match; "1" and "single" are written as the words «одной» and «одним путём» (same meaning).
