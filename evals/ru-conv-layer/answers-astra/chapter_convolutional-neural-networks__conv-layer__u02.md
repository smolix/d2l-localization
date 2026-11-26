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
По вложенным циклам в `corr2d` легко проследить вычисления, но для практических задач попиксельный перебор выхода в Python слишком медленен. Классический способ ускорить вычисления, который по-прежнему входит в арсенал библиотек, — свести взаимную корреляцию к одному большому умножению матриц. Именно на этой операции GPU достигают пиковой производительности. Заметим, что каждый элемент выхода на :numref:`fig_correlation` — это скалярное произведение двух векторов: ядра из $k_\textrm{h} k_\textrm{w}$ элементов, развёрнутого в вектор, и соответствующего окна входа, развёрнутого в том же порядке. Извлечём все $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ окон и запишем каждое в отдельную строку. Получим матрицу фрагментов. Умножив её на ядро, развёрнутое в вектор, вычислим всю взаимную корреляцию за одну операцию. Такое стандартное преобразование расположения данных называют *im2col* («изображение в столбцы», а в нашем случае — в строки). На :numref:`fig_conv_im2col` показано это преобразование для примера с :numref:`fig_correlation`.

<!--en:107554a5 kind=caption-->
![The cross-correlation of :numref:`fig_correlation` as one matrix product. Each of the four $2 \times 2$ windows of the input becomes a row of the patch matrix, and multiplying by the flattened kernel gives the flattened output. The shaded cells follow the first window to its row and to the output 19. The center pixel 4 lies in all four windows, so it appears in every row.](../img/conv-im2col.svg)
<!--tr:107554a5-->
![Взаимная корреляция с :numref:`fig_correlation` в виде одного матричного произведения. Каждое из четырёх окон входа размером $2 \times 2$ становится строкой матрицы фрагментов. Умножив эту матрицу на ядро, развёрнутое в вектор, получаем выход в виде вектора. Цветом выделены первое окно, соответствующая ему строка и результат 19. Центральный пиксель со значением 4 входит во все четыре окна, поэтому встречается в каждой строке.](../img/conv-im2col.svg)

<!--en:76713479 kind=paragraph-->
The trade-off is extra workspace memory, since each interior pixel is duplicated
into up to $k_\textrm{h} k_\textrm{w}$ patches. Below we verify on
:numref:`fig_correlation` that this single matrix multiply matches `corr2d`
exactly:
<!--tr:76713479-->
За это приходится платить дополнительной рабочей памятью: каждый внутренний пиксель копируется в несколько фрагментов, число которых достигает $k_\textrm{h} k_\textrm{w}$. Проверим на примере с :numref:`fig_correlation`, что одно умножение матриц даёт в точности тот же результат, что и `corr2d`:

<!--en:3bcd7d8a kind=paragraph-->
The discrepancy is identically zero: one matrix product reproduces `corr2d`
perfectly.
<!--tr:3bcd7d8a-->
Разность тождественно равна нулю: одно матричное произведение точно воспроизводит результат `corr2d`.

<!--en:c30f2dea kind=paragraph-->
Equivalently, we can unroll the *kernel* instead of the input into a sparse
banded matrix $\mathbf{W}$ multiplying the flattened image—the exact 2D analog
of panel (c) in :numref:`fig_why_conv_weight_sharing`, where every row repeats
the same handful of kernel weights. Seeing a convolution as a structured dense
layer brings us full circle to the derivation of :numref:`sec_why-conv` (and the
exercises build $\mathbf{W}$ explicitly).
<!--tr:c30f2dea-->
Есть и равносильный способ: развернуть *ядро* вместо входа в разреженную ленточную матрицу $\mathbf{W}$ и умножить её на вектор, составленный из элементов изображения. Это точный двумерный аналог схемы (c) на :numref:`fig_why_conv_weight_sharing`, где в каждой строке повторяется один и тот же небольшой набор весов ядра. Такое представление свёртки в виде полносвязного слоя с заданной структурой возвращает нас к выводу из :numref:`sec_why-conv`. В упражнениях мы построим $\mathbf{W}$ в явном виде.

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
Один свёрточный слой видит лишь фрагмент размером с ядро, тогда как для распознавания целых объектов нужно объединять информацию из гораздо более широких областей. Вспомним из :numref:`subsec_why-conv-channels`, что выход свёрточного слоя называют *картой признаков*. Пример такой карты — наша карта вертикальных границ `Y`. Для любого элемента $x$ карты признаков его *рецептивное поле* — это область входного изображения (или одного из предыдущих слоёв), значения которой могут влиять на $x$ при прямом проходе. Этот термин мы заимствовали из нейрофизиологии (см. :numref:`field_visual`).

<!--en:b36f221e kind=heading-->
### Stacked Layers
<!--tr:b36f221e-->
### Последовательное соединение слоёв

<!--en:24a25ec4 kind=paragraph-->
In :numref:`fig_correlation`, the shaded output `19` has a $2 \times 2$ receptive
field covering the top-left four pixels of $\mathbf{X}$. Now stack a *second*
$2 \times 2$ convolutional layer on top of the $2 \times 2$ output $\mathbf{Y}$
to produce a single scalar $z$: $z$ depends on all four elements of $\mathbf{Y}$,
which together depend on all nine elements of $\mathbf{X}$. Stacking layers
expands the receptive field smoothly with depth.
<!--tr:24a25ec4-->
На :numref:`fig_correlation` рецептивное поле выделенного элемента выхода `19` имеет размер $2 \times 2$ и охватывает четыре пикселя в левом верхнем углу $\mathbf{X}$. Теперь добавим *второй* свёрточный слой $2 \times 2$. Подадим на него выход размера $2 \times 2$ первого слоя, то есть $\mathbf{Y}$, и получим один скаляр $z$. Значение $z$ зависит от всех четырёх элементов $\mathbf{Y}$, а они вместе зависят от всех девяти элементов $\mathbf{X}$. Так с добавлением слоёв рецептивное поле постепенно расширяется.

<!--en:f95c9394 kind=paragraph-->
How quickly does it grow? Each $k \times k$ convolutional layer reaches $k - 1$
pixels farther than a single element of its input, so stacking $L$ layers with
kernel sizes $k_1, \ldots, k_L$ yields a receptive field of side length
<!--tr:f95c9394-->
Как быстро оно растёт? У свёрточного слоя $k \times k$ сторона рецептивного поля на $k - 1$ пикселей длиннее, чем у одного элемента на его входе. Поэтому для последовательности из $L$ слоёв с размерами ядер $k_1, \ldots, k_L$ длина стороны рецептивного поля равна

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
В последовательности из $L$ слоёв $3 \times 3$ каждый слой добавляет по 2 пикселя к длине стороны, и рецептивное поле охватывает область входа размером $(2L + 1) \times (2L + 1)$: после двух слоёв — $5 \times 5$, а после трёх — $7 \times 7$ (:numref:`fig_conv_receptive_field`). Линейный рост даёт преимущество и позволяет обойтись без огромных ядер: три последовательных слоя $3 \times 3$ обеспечивают такое же рецептивное поле $7 \times 7$, как один слой $7 \times 7$, но требуют гораздо меньше параметров ($3 \times 9 = 27$ весов вместо 49 на пару каналов) и позволяют применить три нелинейности вместо одной.

<!--en:39917513 kind=caption-->
![Receptive field of one unit after three one-dimensional layers with kernel size 3. Shaded cells are the units that the top unit depends on, and the lines connect each of them to its inputs. Each layer widens the receptive field by 2, to 7 inputs.](../img/conv-receptive-field.svg)
<!--tr:39917513-->
![Рецептивное поле одного нейрона после трёх одномерных слоёв с размером ядра 3. Цветом выделены нейроны, от которых зависит верхний нейрон. Линии соединяют каждый из них с его входами. Каждый слой расширяет рецептивное поле на 2 элемента, так что в итоге оно охватывает 7 элементов входа.](../img/conv-receptive-field.svg)

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
Величина $r$ задаёт размер *теоретического* рецептивного поля. В него входит каждый пиксель, от которого к выходному нейрону ведёт *хотя бы один* путь, независимо от того, насколько сильно этот пиксель влияет на результат. Измерив градиенты по входу, :citet:`Luo.Li.Urtasun.ea.2016` показали, что в *эффективном* рецептивном поле чувствительность быстро убывает с удалением от центра примерно по гауссовой колоколообразной кривой. Это справедливо как до, так и после обучения. Теоретически нейроны глубоких слоёв видят обширную область изображения, но наиболее чувствительны к пикселям вблизи центра.

<!--en:eed4e2a0 kind=paragraph-->
Counting paths explains the bell shape. An input at the edge of the receptive
field reaches the output unit along a single chain of connections, whereas the
center input can take many routes through the layers in between.
:numref:`fig_conv_rf_paths` counts them for the tower of
:numref:`fig_conv_receptive_field`: the center input has seven paths to the top
unit and each edge input has one. If all weights have a similar magnitude, the
influence of an input grows with its number of paths.
<!--tr:eed4e2a0-->
Форму колокола можно объяснить, подсчитав пути. От элемента входа на краю рецептивного поля к выходному нейрону ведёт единственная цепочка связей, а от центрального элемента через промежуточные слои проходит множество путей. На :numref:`fig_conv_rf_paths` показано число путей для последовательности слоёв с :numref:`fig_conv_receptive_field`: от центрального элемента входа к верхнему нейрону ведут семь путей, а от каждого крайнего — один. Если все веса близки по модулю, влияние элемента входа растёт с числом путей от него.

<!--en:b92775ee kind=caption-->
![Number of paths from each unit to the top unit of :numref:`fig_conv_receptive_field` (three one-dimensional layers with kernel size 3). Each count is the sum of the counts of the units that the cell feeds. Darker cells have more paths than the other cells of their layer, and the bars repeat the counts of the input row: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)
<!--tr:b92775ee-->
![Число путей от каждого нейрона к верхнему нейрону на :numref:`fig_conv_receptive_field` (три одномерных слоя с размером ядра 3). Для каждой ячейки это число равно сумме чисел в тех ячейках, которым она передаёт своё значение. Чем темнее ячейка, тем больше у неё путей по сравнению с другими ячейками того же слоя. Столбцы диаграммы показывают числа путей для входного ряда: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)

<!--en:405695e8 kind=paragraph-->
Two practical gaps remain in our basic layer: every unpadded convolution shrinks
spatial dimensions while under-sampling border pixels, and a single kernel
detects only one pattern. We address spatial sizing and borders next in
:numref:`sec_padding`, and multi-channel feature maps in :numref:`sec_channels`.
<!--tr:405695e8-->
У нашего простого слоя остаются два практических ограничения. Каждая свёртка без дополнения уменьшает пространственные размеры и реже учитывает граничные пиксели, а одно ядро выявляет только один тип признаков. Далее в :numref:`sec_padding` мы рассмотрим выбор пространственных размеров и обработку границ, а в :numref:`sec_channels` — многоканальные карты признаков.

<!--en:30ac03ff kind=heading-->
## Summary
<!--tr:30ac03ff-->
## Итоги

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
Свёрточный слой перемещает обучаемое ядро по всем допустимым положениям окна на входе, вычисляет двумерную взаимную корреляцию и прибавляет скалярное смещение. Простое ядро конечной разности размера $1 \times 2$ с весами `[1, -1]` обнаруживает вертикальные границы в любой части изображения. Градиентный спуск восстанавливает эти самые веса только по входным и выходным данным, подобно тому как глубокие свёрточные нейронные сети при сквозном обучении находят целые иерархии фильтров. В математической свёртке ядро разворачивают, но при обучаемых ядрах это различие несущественно. Свёртку можно свести к одному умножению матриц (*im2col*). При последовательном соединении $L$ свёрточных слоёв рецептивное поле каждого нейрона растёт линейно, расширяясь на $k_i - 1$ с каждым слоем.

<!--en:3ba929d2 kind=slide_title-->
::: {.slide title="Fast Execution: Convolution via Matrix Multiplication"}
<!--tr:3ba929d2-->
::: {.slide title="Быстрая свёртка через умножение матриц"}

<!--en:1711f812 kind=kicker-->
[The im2col (Image-to-Columns) Transformation]{.kicker}
<!--tr:1711f812-->
[Преобразование im2col («изображение в столбцы»)]{.kicker}

<!--en:a040f3d7 kind=slide_text-->
- **Single dense GEMM:** Flatten each $k_\textrm{h} \times k_\textrm{w}$ window into a row of `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ and multiply by $\textrm{vec}(\mathbf{K})$.
<!--tr:a040f3d7-->
- **Одно умножение плотных матриц (GEMM):** развернём каждое окно $k_\textrm{h} \times k_\textrm{w}$ в строку матрицы `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ и умножим эту матрицу на $\textrm{vec}(\mathbf{K})$.

<!--en:1fb18ec4 kind=slide_text-->
- **Trade-off:** Peak GPU Tensor Core speed at the cost of $k_\textrm{h} k_\textrm{w}\times$ patch workspace duplication.
<!--tr:1fb18ec4-->
- **Цена ускорения:** достигаем пиковой скорости тензорных ядер GPU, но из-за копирования пикселей во фрагменты расход рабочей памяти возрастает (коэффициент $k_\textrm{h} k_\textrm{w}\times$).

<!--en:0d757cf6 kind=slide_title-->
::: {.slide title="Dual View: Unrolling the Kernel into a Banded Matrix"}
<!--tr:0d757cf6-->
::: {.slide title="Другой взгляд на свёртку: развёртка ядра в ленточную матрицу"}

<!--en:7f0defaa kind=kicker-->
[Toeplitz Weight Matrix Multiplication]{.kicker}
<!--tr:7f0defaa-->
[Умножение на теплицеву матрицу весов]{.kicker}

<!--en:5898ba57 kind=slide_text-->
Instead of unfolding the input image into duplicated patches, we can unroll the *kernel* into a sparse banded matrix $\mathbf{W}$ that multiplies the flattened input vector $\textrm{vec}(\mathbf{X})$:
<!--tr:5898ba57-->
Вместо того чтобы копировать перекрывающиеся фрагменты изображения, можно развернуть *ядро* в разреженную ленточную матрицу $\mathbf{W}$ и умножить её на вектор из элементов входа $\textrm{vec}(\mathbf{X})$:

<!--en:dae51741 kind=slide_text-->
- **Matrix–vector form:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, where $\mathbf{W}$ has $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ rows and $n_\textrm{h} n_\textrm{w}$ columns.
<!--tr:dae51741-->
- **Произведение матрицы на вектор:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, где у $\mathbf{W}$ число строк равно $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$, а число столбцов — $n_\textrm{h} n_\textrm{w}$.

<!--en:f0e86ee3 kind=slide_text-->
- **Connection to §{sec_why-conv} and backprop:** Every row of $\mathbf{W}$ holds the same $k_\textrm{h} k_\textrm{w}$ kernel entries shifted across columns (panel c), making the backward pass with respect to $\mathbf{X}$ simply multiplication by $\mathbf{W}^\top$ (a transposed convolution).
<!--tr:f0e86ee3-->
- **Связь с §{sec_why-conv} и обратным распространением ошибки:** в каждой строке $\mathbf{W}$ повторяются одни и те же $k_\textrm{h} k_\textrm{w}$ элементов ядра со сдвигом по столбцам (схема c). Поэтому обратный проход по входу $\mathbf{X}$ сводится к умножению на $\mathbf{W}^\top$ (транспонированная свёртка).

<!--en:af8cf19a kind=slide_title-->
::: {.slide title="Feature Maps and Receptive Fields Across Depth"}
<!--tr:af8cf19a-->
::: {.slide title="Карты признаков и рецептивные поля на разной глубине"}

<!--en:4eb965f3 kind=kicker-->
[Linear Receptive-Field Growth in Stacked Layers]{.kicker}
<!--tr:4eb965f3-->
[Линейный рост рецептивного поля в последовательности слоёв]{.kicker}

<!--en:5599a6e9 kind=slide_text-->
A unit's **receptive field** is the region of the original input image whose pixels can influence that unit's activation during forward propagation:
<!--tr:5599a6e9-->
**Рецептивное поле** нейрона — область исходного изображения, пиксели которой могут влиять на активацию этого нейрона при прямом проходе:

<!--en:eba74f75 kind=slide_text-->
- **Single layer ($L=1$):** A $k_1 \times k_1$ kernel sees a $k_1 \times k_1$ input patch.
<!--tr:eba74f75-->
- **Один слой ($L=1$):** ядро $k_1 \times k_1$ видит фрагмент входа размера $k_1 \times k_1$.

<!--en:01c175bf kind=slide_text-->
- **Stacking $L$ stride-1 layers:** Layer $i$ with kernel size $k_i$ reaches $k_i - 1$ pixels farther along each axis than a single element of layer $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$
<!--tr:01c175bf-->
- **Последовательность из $L$ слоёв с шагом 1:** у слоя $i$ с размером ядра $k_i$ рецептивное поле по каждой оси на $k_i - 1$ пикселей шире, чем у отдельного элемента слоя $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$

<!--en:2bde5822 kind=slide_text-->
- **$L$ stacked $3 \times 3$ layers:** Each layer adds $3 - 1 = 2$ pixels, giving side length $r = 2L + 1$ ($3 \to 5 \to 7$).
<!--tr:2bde5822-->
- **Последовательность из $L$ слоёв $3 \times 3$:** каждый слой добавляет $3 - 1 = 2$ пикселя к длине стороны, так что она равна $r = 2L + 1$ ($3 \to 5 \to 7$).

<!--en:cbb9e97a kind=slide_title-->
::: {.slide title="Why Deep Stacks of 3×3 Kernels Beat One Large Kernel"}
<!--tr:cbb9e97a-->
::: {.slide title="Почему несколько слоёв с ядрами 3×3 выигрывают у одного слоя с большим ядром"}

<!--en:95e46e70 kind=kicker-->
[Parameter Efficiency & Nonlinear Expressivity]{.kicker}
<!--tr:95e46e70-->
[Экономия параметров и выразительность нелинейных преобразований]{.kicker}

<!--en:bd1d7cc0 kind=slide_text-->
- **Matched $7 \times 7$ receptive field:** Three stacked $3 \times 3$ layers ($r = 1 + 2 + 2 + 2 = 7$) see the exact same $7 \times 7$ patch as one $7 \times 7$ layer.
<!--tr:bd1d7cc0-->
- **Одинаковое рецептивное поле $7 \times 7$:** три последовательных слоя $3 \times 3$ ($r = 1 + 2 + 2 + 2 = 7$) видят тот же фрагмент размера $7 \times 7$, что и один слой $7 \times 7$.

<!--en:755ac11a kind=slide_text-->
- **45% fewer parameters:** $3 \times 9 c^2 = 27 c^2$ weights instead of $49 c^2$ across $c$ channels.
<!--tr:755ac11a-->
- **На 45% меньше параметров:** $3 \times 9 c^2 = 27 c^2$ весов вместо $49 c^2$ при $c$ каналах.

<!--en:e3aa1086 kind=slide_text-->
- **Richer expressivity:** Compounds **3 nonlinear activations** instead of 1 (the core VGG design rule in §{sec_blocks}).
<!--tr:e3aa1086-->
- **Больше выразительных возможностей:** последовательно применяются **3 нелинейные функции активации** вместо одной (основной принцип построения VGG в §{sec_blocks}).

<!--en:9ec83bf6 kind=slide_title-->
::: {.slide title="Theoretical vs. Effective Receptive Fields"}
<!--tr:9ec83bf6-->
::: {.slide title="Теоретическое и эффективное рецептивные поля"}

<!--en:72a52c3b kind=kicker-->
[Gaussian Decay of Sensitivity Away from the Center]{.kicker}
<!--tr:72a52c3b-->
[Гауссово убывание чувствительности при удалении от центра]{.kicker}

<!--en:33bb377d kind=slide_text-->
- **Theoretical RF ($r = 2L+1$):** Bounding box of all pixels with *at least one* path to the output unit.
<!--tr:33bb377d-->
- **Теоретическое рецептивное поле ($r = 2L+1$):** ограничивающая рамка для всех пикселей, от которых к выходному нейрону ведёт *хотя бы один* путь.

<!--en:293436eb kind=slide_text-->
- **Central Limit Theorem:** Center pixels connect via exponentially many paths, whereas corner pixels connect via only **1 path**.
<!--tr:293436eb-->
- **Центральная предельная теорема:** для центральных пикселей число путей растёт экспоненциально, тогда как от каждого углового пикселя ведёт всего **1 путь**.

<!--en:f47d181e kind=slide_text-->
- **Effective RF (Luo et al., 2016):** Input gradient magnitude $\partial y / \partial \mathbf{X}$ decays in a Gaussian bell curve with std $\sigma \propto \sqrt{L}$.
<!--tr:f47d181e-->
- **Эффективное рецептивное поле (Luo и др., 2016):** модуль градиента по входу $\partial y / \partial \mathbf{X}$ убывает от центра по гауссовой кривой со стандартным отклонением $\sigma \propto \sqrt{L}$.

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
Проверьте, как вы определяете форму выхода, применяете `im2col` и вычисляете размеры рецептивного поля:

<!--en:281c7c78 kind=slide_text-->
1. **im2col matrix dimensions:** You apply an unpadded $3 \times 3$ single-channel kernel to a $10 \times 10$ image via `im2col`. What is the output spatial shape, and what are the exact dimensions of the `patches` matrix and flattened kernel vector?
<!--tr:281c7c78-->
1. **Размеры матриц в im2col:** к изображению применяется одноканальное ядро $3 \times 3$ без дополнения. Изображение имеет размер $10 \times 10$, а свёртка выполняется через `im2col`. Каковы пространственные размеры выхода? Каковы точные размеры матрицы `patches` и вектора, в который развёрнуто ядро?

<!--en:16976d18 kind=slide_text-->
2. **Depth to span Fashion-MNIST:** At stride 1, how many stacked $3 \times 3$ layers are required so a single unit's theoretical receptive field spans an entire $28 \times 28$ image? How many with $5 \times 5$ layers?
<!--tr:16976d18-->
2. **Глубина сети для Fashion-MNIST:** сколько последовательных слоёв $3 \times 3$ с шагом 1 нужно, чтобы теоретическое рецептивное поле одного нейрона охватило всё изображение $28 \times 28$? А если использовать слои $5 \times 5$?

<!--en:ab2e0d6a kind=slide_text-->
3. **Edge-loss boundary shift:** In our $6 \times 8$ vertical-edge learning experiment, why did the loss start at $\approx 12$, and what would the initial loss be if the black band extended all the way to the right edge (`X[:, 2:8] = 0`)?
<!--tr:ab2e0d6a-->
3. **Потери при смещении границы:** почему в опыте по обучению детектора вертикальных границ на изображении $6 \times 8$ начальные потери составили $\approx 12$? Какими были бы начальные потери, если бы чёрная полоса доходила до правого края (`X[:, 2:8] = 0`)?

<!--en:250d6877 kind=slide_text-->
**Answers:**
(1) Output shape is $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ outputs). The `patches` matrix has shape $(64, 9)$ and multiplies the $(9, 1)$ kernel vector.
(2) For $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ layers. For $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ layers.
(3) Each of the 6 rows had two $\pm 1$ edges ($6 \times 2 \times 1^2 = 12$). Extending the black band to the right border eliminates the right-hand black-to-white edge, leaving one edge per row and halving the initial loss to $\approx 6$.
<!--tr:250d6877-->
**Ответы:**
(1) Форма выхода равна $(10-3+1) \times (10-3+1) = 8 \times 8$ (всего $64$ элемента). Матрицу `patches` формы $(64, 9)$ умножаем на вектор весов ядра формы $(9, 1)$.
(2) Для ядра $3 \times 3$ получаем $2L + 1 \ge 28 \implies L = 14$ слоёв. Для ядра $5 \times 5$ получаем $4L + 1 \ge 28 \implies L = 7$ слоёв.
(3) В каждой из 6 строк было по две границы со значениями $\pm 1$ ($6 \times 2 \times 1^2 = 12$). Если продлить чёрную полосу до правого края, справа исчезнет переход от чёрного к белому. В каждой строке останется одна граница, и начальные потери уменьшатся вдвое, до $\approx 6$.

<!--en:ba43a2dd kind=slide_title-->
::: {.slide title="Summary & Key Takeaways"}
<!--tr:ba43a2dd-->
::: {.slide title="Итоги и основные выводы"}

<!--en:ce12e85f kind=kicker-->
[Section Synthesis]{.kicker}
<!--tr:ce12e85f-->
[Обобщение материала раздела]{.kicker}

<!--en:00011fb7 kind=slide_text-->
- **2D Cross-Correlation & Edge Learning:** Slides $\mathbf{K}$ across valid windows ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); SGD learns $[1, -1]$ in 9 steps.
<!--tr:00011fb7-->
- **Двумерная взаимная корреляция и обучение детектора границ:** ядро $\mathbf{K}$ перемещается по допустимым положениям окна ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$). Стохастический градиентный спуск находит веса $[1, -1]$ за 9 шагов.

<!--en:09efe4f0 kind=slide_text-->
- **im2col Matrix Multiply:** Trades $k_\textrm{h} k_\textrm{w}\times$ patch duplication for peak GPU GEMM throughput.
<!--tr:09efe4f0-->
- **Умножение матриц с im2col:** копируя пиксели во фрагменты (до $k_\textrm{h} k_\textrm{w}\times$), достигаем пиковой пропускной способности GPU при умножении матриц (GEMM).

<!--en:3af8acd0 kind=slide_text-->
- **Stacked Receptive Fields:** $r = 1 + \sum (k_i - 1) = 2L+1$ for $3 \times 3$ stacks, with Gaussian center sensitivity.
<!--tr:3af8acd0-->
- **Рецептивное поле последовательных слоёв:** $r = 1 + \sum (k_i - 1) = 2L+1$ для слоёв $3 \times 3$. Чувствительность убывает по гауссовой кривой с максимумом в центре.

=== DECISIONS ===
- «patch» → «фрагмент», «patch matrix» → «матрица фрагментов». «Flatten» → «развернуть в вектор» с указанием порядка, когда он существенен.
- «banded matrix» → «ленточная матрица», «stacked layers» → «последовательные слои» или «последовательность слоёв» в зависимости от синтаксиса.
- «effective receptive field» → «эффективное рецептивное поле».

=== NOTES ===
- e3aa1086: проверка не находит цифру 1, поскольку в выражении «вместо одной» число записано словом.
