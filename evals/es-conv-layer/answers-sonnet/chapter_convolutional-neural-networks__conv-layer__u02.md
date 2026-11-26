
<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u02 lang=es rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:0b85caf9 kind=heading-->
## Convolution as Matrix Multiplication
<!--tr:0b85caf9-->
## La convolución como multiplicación de matrices

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
Aunque los bucles anidados de `corr2d` hacen transparente la matemática, visitar un píxel de salida cada vez en Python es demasiado lento para cargas de trabajo reales. Un remedio clásico, que sigue siendo uno de los algoritmos entre los que eligen las bibliotecas, consiste en reformular la correlación cruzada como una única multiplicación de matrices grande, justo la operación que las GPU ejecutan a máxima velocidad. Observemos que cada entrada de salida de :numref:`fig_correlation` es simplemente el producto escalar del núcleo aplanado, de $k_\textrm{h} k_\textrm{w}$ elementos, con la ventana de entrada correspondiente, también aplanada. Si extraemos todas las $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ ventanas y las apilamos como filas, obtenemos una matriz de fragmentos; multiplicarla por el núcleo aplanado calcula toda la correlación cruzada de una sola vez. Esta transformación estándar de la disposición de los datos se llama *im2col* ("imagen a columnas", o aquí a filas). :numref:`fig_conv_im2col` la lleva a cabo sobre el ejemplo de :numref:`fig_correlation`.

<!--en:107554a5 kind=caption-->
![The cross-correlation of :numref:`fig_correlation` as one matrix product. Each of the four $2 \times 2$ windows of the input becomes a row of the patch matrix, and multiplying by the flattened kernel gives the flattened output. The shaded cells follow the first window to its row and to the output 19. The center pixel 4 lies in all four windows, so it appears in every row.](../img/conv-im2col.svg)
<!--tr:107554a5-->
![La correlación cruzada de :numref:`fig_correlation` como un único producto de matrices. Cada una de las cuatro ventanas de $2 \times 2$ de la entrada se convierte en una fila de la matriz de fragmentos, y al multiplicar por el núcleo aplanado se obtiene la salida aplanada. Las celdas sombreadas siguen la primera ventana hasta su fila y hasta la salida 19. El píxel central 4 está en las cuatro ventanas, por lo que aparece en todas las filas.](../img/conv-im2col.svg)

<!--en:76713479 kind=paragraph-->
The trade-off is extra workspace memory, since each interior pixel is duplicated
into up to $k_\textrm{h} k_\textrm{w}$ patches. Below we verify on
:numref:`fig_correlation` that this single matrix multiply matches `corr2d`
exactly:
<!--tr:76713479-->
La contrapartida es memoria de trabajo adicional, ya que cada píxel interior se duplica en hasta $k_\textrm{h} k_\textrm{w}$ fragmentos. A continuación verificamos sobre :numref:`fig_correlation` que esta única multiplicación de matrices coincide exactamente con `corr2d`:

<!--en:3bcd7d8a kind=paragraph-->
The discrepancy is identically zero: one matrix product reproduces `corr2d`
perfectly.
<!--tr:3bcd7d8a-->
La discrepancia es idénticamente cero: un único producto de matrices reproduce `corr2d` a la perfección.

<!--en:c30f2dea kind=paragraph-->
Equivalently, we can unroll the *kernel* instead of the input into a sparse
banded matrix $\mathbf{W}$ multiplying the flattened image—the exact 2D analog
of panel (c) in :numref:`fig_why_conv_weight_sharing`, where every row repeats
the same handful of kernel weights. Seeing a convolution as a structured dense
layer brings us full circle to the derivation of :numref:`sec_why-conv` (and the
exercises build $\mathbf{W}$ explicitly).
<!--tr:c30f2dea-->
De forma equivalente, podemos desenrollar el *núcleo* en lugar de la entrada, y obtener una matriz dispersa y de bandas $\mathbf{W}$ que multiplica a la imagen aplanada; es el análogo 2D exacto del panel (c) de :numref:`fig_why_conv_weight_sharing`, donde cada fila repite los mismos pocos pesos del núcleo. Ver una convolución como una capa densa estructurada nos devuelve al punto de partida, a la derivación de :numref:`sec_why-conv` (y en los ejercicios se construye $\mathbf{W}$ de forma explícita).

<!--en:ad4fc956 kind=heading-->
## Feature Maps and Receptive Fields
<!--tr:ad4fc956-->
## Mapas de características y campos receptivos

<!--en:e7983df8 kind=paragraph-->
A single convolutional layer sees only a patch the size of its kernel, whereas
recognizing whole objects demands integrating context across much wider areas.
Recall from :numref:`subsec_why-conv-channels` that a convolutional layer's
output is called a *feature map* (like our vertical edge map `Y`). For any
element $x$ in a feature map, its *receptive field* is the region of the input
image (or of an earlier layer) whose values can influence $x$ during forward
propagation—borrowing the neurophysiology term from :numref:`field_visual`.
<!--tr:e7983df8-->
Una sola capa convolucional ve únicamente un fragmento del tamaño de su núcleo, mientras que reconocer objetos enteros exige integrar el contexto en áreas mucho más amplias. Recordemos de :numref:`subsec_why-conv-channels` que la salida de una capa convolucional se llama *mapa de características* (*feature map*), como nuestro mapa de bordes verticales `Y`. Para cualquier elemento $x$ de un mapa de características, su *campo receptivo* (*receptive field*) es la región de la imagen de entrada (o de una capa anterior) cuyos valores pueden influir en $x$ durante la propagación hacia adelante; el término procede de la neurofisiología, como vimos en :numref:`field_visual`.

<!--en:b36f221e kind=heading-->
### Stacked Layers
<!--tr:b36f221e-->
### Capas apiladas

<!--en:24a25ec4 kind=paragraph-->
In :numref:`fig_correlation`, the shaded output `19` has a $2 \times 2$ receptive
field covering the top-left four pixels of $\mathbf{X}$. Now stack a *second*
$2 \times 2$ convolutional layer on top of the $2 \times 2$ output $\mathbf{Y}$
to produce a single scalar $z$: $z$ depends on all four elements of $\mathbf{Y}$,
which together depend on all nine elements of $\mathbf{X}$. Stacking layers
expands the receptive field smoothly with depth.
<!--tr:24a25ec4-->
En :numref:`fig_correlation`, la salida sombreada `19` tiene un campo receptivo de $2 \times 2$ que cubre los cuatro píxeles superiores izquierdos de $\mathbf{X}$. Apilemos ahora una *segunda* capa convolucional de $2 \times 2$ sobre la salida de $2 \times 2$, $\mathbf{Y}$, para producir un único escalar $z$: $z$ depende de los cuatro elementos de $\mathbf{Y}$, que juntos dependen de los nueve elementos de $\mathbf{X}$. Apilar capas amplía el campo receptivo de forma gradual con la profundidad.

<!--en:f95c9394 kind=paragraph-->
How quickly does it grow? Each $k \times k$ convolutional layer reaches $k - 1$
pixels farther than a single element of its input, so stacking $L$ layers with
kernel sizes $k_1, \ldots, k_L$ yields a receptive field of side length
<!--tr:f95c9394-->
¿Con qué rapidez crece? Cada capa convolucional de $k \times k$ alcanza $k - 1$ píxeles más lejos que un único elemento de su entrada, de modo que apilar $L$ capas con tamaños de núcleo $k_1, \ldots, k_L$ da un campo receptivo de lado

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
En una pila de $L$ capas consecutivas de $3 \times 3$, cada capa añade 2 píxeles y cubre entradas de $(2L + 1) \times (2L + 1)$: $5 \times 5$ tras dos capas y $7 \times 7$ tras tres (:numref:`fig_conv_receptive_field`). Este crecimiento lineal es en realidad una bendición y no una razón para recurrir a núcleos enormes: tres capas de $3 \times 3$ apiladas logran el mismo campo receptivo de $7 \times 7$ que una sola capa de $7 \times 7$ con muchos menos parámetros ($3 \times 9 = 27$ pesos frente a 49 por par de canales), y además intercalan tres no linealidades en lugar de una.

<!--en:39917513 kind=caption-->
![Receptive field of one unit after three one-dimensional layers with kernel size 3. Shaded cells are the units that the top unit depends on, and the lines connect each of them to its inputs. Each layer widens the receptive field by 2, to 7 inputs.](../img/conv-receptive-field.svg)
<!--tr:39917513-->
![Campo receptivo de una unidad tras tres capas unidimensionales con tamaño de núcleo 3. Las celdas sombreadas son las unidades de las que depende la unidad superior, y las líneas conectan cada una con sus entradas. Cada capa amplía el campo receptivo en 2, hasta 7 entradas.](../img/conv-receptive-field.svg)

<!--en:48e4c9cc kind=heading-->
### Effective Receptive Fields
<!--tr:48e4c9cc-->
### Campos receptivos efectivos

<!--en:77d044db kind=paragraph-->
Keep in mind that $r$ measures the *theoretical* receptive field (every input
pixel with *any* path to an output unit) without saying how strongly each pixel
actually contributes. Measuring input gradients directly,
:citet:`Luo.Li.Urtasun.ea.2016` showed that the *effective* receptive field
decays sharply away from the center in a roughly Gaussian bell curve, in both
untrained and trained networks. Deep units can theoretically see far across the
image, but their sensitivity concentrates heavily near the center.
<!--tr:77d044db-->
Tengamos presente que $r$ mide el campo receptivo *teórico* (todo píxel de entrada con *algún* camino hasta una unidad de salida), sin decir con qué intensidad contribuye realmente cada píxel. Midiendo directamente los gradientes respecto de la entrada, :citet:`Luo.Li.Urtasun.ea.2016` mostraron que el campo receptivo *efectivo* decae bruscamente al alejarse del centro, con una forma aproximada de campana gaussiana, tanto en redes sin entrenar como entrenadas. Las unidades profundas pueden ver, en teoría, muy lejos en la imagen, pero su sensibilidad se concentra con fuerza cerca del centro.

<!--en:eed4e2a0 kind=paragraph-->
Counting paths explains the bell shape. An input at the edge of the receptive
field reaches the output unit along a single chain of connections, whereas the
center input can take many routes through the layers in between.
:numref:`fig_conv_rf_paths` counts them for the tower of
:numref:`fig_conv_receptive_field`: the center input has seven paths to the top
unit and each edge input has one. If all weights have a similar magnitude, the
influence of an input grows with its number of paths.
<!--tr:eed4e2a0-->
Contar caminos explica la forma de campana. Una entrada situada en el borde del campo receptivo llega a la unidad de salida por una única cadena de conexiones, mientras que la entrada central puede seguir muchas rutas a través de las capas intermedias. :numref:`fig_conv_rf_paths` los cuenta para la torre de :numref:`fig_conv_receptive_field`: la entrada central tiene siete caminos hasta la unidad superior y cada entrada del borde tiene uno. Si todos los pesos tienen una magnitud similar, la influencia de una entrada crece con su número de caminos.

<!--en:b92775ee kind=caption-->
![Number of paths from each unit to the top unit of :numref:`fig_conv_receptive_field` (three one-dimensional layers with kernel size 3). Each count is the sum of the counts of the units that the cell feeds. Darker cells have more paths than the other cells of their layer, and the bars repeat the counts of the input row: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)
<!--tr:b92775ee-->
![Número de caminos desde cada unidad hasta la unidad superior de :numref:`fig_conv_receptive_field` (tres capas unidimensionales con tamaño de núcleo 3). Cada cuenta es la suma de las cuentas de las unidades a las que alimenta la celda. Las celdas más oscuras tienen más caminos que las demás celdas de su capa, y las barras repiten las cuentas de la fila de entrada: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)

<!--en:405695e8 kind=paragraph-->
Two practical gaps remain in our basic layer: every unpadded convolution shrinks
spatial dimensions while under-sampling border pixels, and a single kernel
detects only one pattern. We address spatial sizing and borders next in
:numref:`sec_padding`, and multi-channel feature maps in :numref:`sec_channels`.
<!--tr:405695e8-->
En nuestra capa básica quedan dos carencias prácticas: toda convolución sin relleno reduce las dimensiones espaciales y submuestrea los píxeles del borde, y un único núcleo detecta un solo patrón. Abordamos a continuación el tamaño espacial y los bordes en :numref:`sec_padding`, y los mapas de características multicanal en :numref:`sec_channels`.

<!--en:30ac03ff kind=heading-->
## Summary
<!--tr:30ac03ff-->
## Resumen

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
Una capa convolucional aplica la correlación cruzada bidimensional con un núcleo aprendido más un sesgo escalar, deslizando el núcleo por todas las ventanas válidas de la entrada. Un sencillo núcleo de diferencia finita `[1, -1]` de $1 \times 2$ detecta bordes verticales en cualquier lugar de una imagen, y el descenso de gradiente recupera esos pesos exactos solo a partir de datos de entrada y salida, igual que las CNN profundas aprenden jerarquías enteras de filtros de extremo a extremo. La convolución matemática invierte el núcleo, lo que no supone ninguna diferencia una vez que los núcleos se aprenden. Una convolución puede evaluarse como una única multiplicación de matrices (*im2col*), y apilar $L$ capas convolucionales amplía linealmente el campo receptivo de cada unidad en $k_i - 1$ por capa.

<!--en:3ba929d2 kind=slide_title-->
::: {.slide title="Fast Execution: Convolution via Matrix Multiplication"}
<!--tr:3ba929d2-->
::: {.slide title="Ejecución rápida: la convolución como multiplicación de matrices"}

<!--en:1711f812 kind=kicker-->
[The im2col (Image-to-Columns) Transformation]{.kicker}
<!--tr:1711f812-->
[La transformación im2col (de imagen a columnas)]{.kicker}

<!--en:a040f3d7 kind=slide_text-->
- **Single dense GEMM:** Flatten each $k_\textrm{h} \times k_\textrm{w}$ window into a row of `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ and multiply by $\textrm{vec}(\mathbf{K})$.
<!--tr:a040f3d7-->
- **Un único GEMM denso:** se aplana cada ventana de $k_\textrm{h} \times k_\textrm{w}$ en una fila de `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ y se multiplica por $\textrm{vec}(\mathbf{K})$.

<!--en:1fb18ec4 kind=slide_text-->
- **Trade-off:** Peak GPU Tensor Core speed at the cost of $k_\textrm{h} k_\textrm{w}\times$ patch workspace duplication.
<!--tr:1fb18ec4-->
- **Contrapartida:** velocidad máxima de los Tensor Cores de la GPU, a costa de duplicar $k_\textrm{h} k_\textrm{w}\times$ el espacio de trabajo de fragmentos.

<!--en:0d757cf6 kind=slide_title-->
::: {.slide title="Dual View: Unrolling the Kernel into a Banded Matrix"}
<!--tr:0d757cf6-->
::: {.slide title="Visión dual: desenrollar el núcleo en una matriz de bandas"}

<!--en:7f0defaa kind=kicker-->
[Toeplitz Weight Matrix Multiplication]{.kicker}
<!--tr:7f0defaa-->
[Multiplicación por la matriz de pesos de Toeplitz]{.kicker}

<!--en:5898ba57 kind=slide_text-->
Instead of unfolding the input image into duplicated patches, we can unroll the *kernel* into a sparse banded matrix $\mathbf{W}$ that multiplies the flattened input vector $\textrm{vec}(\mathbf{X})$:
<!--tr:5898ba57-->
En lugar de desplegar la imagen de entrada en fragmentos duplicados, podemos desenrollar el *núcleo* en una matriz dispersa y de bandas $\mathbf{W}$ que multiplica al vector de entrada aplanado $\textrm{vec}(\mathbf{X})$:

<!--en:dae51741 kind=slide_text-->
- **Matrix–vector form:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, where $\mathbf{W}$ has $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ rows and $n_\textrm{h} n_\textrm{w}$ columns.
<!--tr:dae51741-->
- **Forma matriz-vector:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, donde $\mathbf{W}$ tiene $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ filas y $n_\textrm{h} n_\textrm{w}$ columnas.

<!--en:f0e86ee3 kind=slide_text-->
- **Connection to §{sec_why-conv} and backprop:** Every row of $\mathbf{W}$ holds the same $k_\textrm{h} k_\textrm{w}$ kernel entries shifted across columns (panel c), making the backward pass with respect to $\mathbf{X}$ simply multiplication by $\mathbf{W}^\top$ (a transposed convolution).
<!--tr:f0e86ee3-->
- **Relación con §{sec_why-conv} y la retropropagación:** cada fila de $\mathbf{W}$ contiene las mismas $k_\textrm{h} k_\textrm{w}$ entradas del núcleo desplazadas entre las columnas (panel c), de modo que la pasada hacia atrás respecto de $\mathbf{X}$ es simplemente la multiplicación por $\mathbf{W}^\top$ (una convolución transpuesta).

<!--en:af8cf19a kind=slide_title-->
::: {.slide title="Feature Maps and Receptive Fields Across Depth"}
<!--tr:af8cf19a-->
::: {.slide title="Mapas de características y campos receptivos según la profundidad"}

<!--en:4eb965f3 kind=kicker-->
[Linear Receptive-Field Growth in Stacked Layers]{.kicker}
<!--tr:4eb965f3-->
[Crecimiento lineal del campo receptivo en capas apiladas]{.kicker}

<!--en:5599a6e9 kind=slide_text-->
A unit's **receptive field** is the region of the original input image whose pixels can influence that unit's activation during forward propagation:
<!--tr:5599a6e9-->
El **campo receptivo** de una unidad es la región de la imagen de entrada original cuyos píxeles pueden influir en la activación de esa unidad durante la propagación hacia adelante:

<!--en:eba74f75 kind=slide_text-->
- **Single layer ($L=1$):** A $k_1 \times k_1$ kernel sees a $k_1 \times k_1$ input patch.
<!--tr:eba74f75-->
- **Una sola capa ($L=1$):** un núcleo de $k_1 \times k_1$ ve un fragmento de entrada de $k_1 \times k_1$.

<!--en:01c175bf kind=slide_text-->
- **Stacking $L$ stride-1 layers:** Layer $i$ with kernel size $k_i$ reaches $k_i - 1$ pixels farther along each axis than a single element of layer $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$
<!--tr:01c175bf-->
- **Apilar $L$ capas de stride 1:** la capa $i$ con tamaño de núcleo $k_i$ alcanza $k_i - 1$ píxeles más lejos a lo largo de cada eje que un único elemento de la capa $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$

<!--en:2bde5822 kind=slide_text-->
- **$L$ stacked $3 \times 3$ layers:** Each layer adds $3 - 1 = 2$ pixels, giving side length $r = 2L + 1$ ($3 \to 5 \to 7$).
<!--tr:2bde5822-->
- **$L$ capas de $3 \times 3$ apiladas:** cada capa añade $3 - 1 = 2$ píxeles, lo que da un lado $r = 2L + 1$ ($3 \to 5 \to 7$).

<!--en:cbb9e97a kind=slide_title-->
::: {.slide title="Why Deep Stacks of 3×3 Kernels Beat One Large Kernel"}
<!--tr:cbb9e97a-->
::: {.slide title="Por qué las pilas profundas de núcleos de 3×3 superan a un núcleo grande"}

<!--en:95e46e70 kind=kicker-->
[Parameter Efficiency & Nonlinear Expressivity]{.kicker}
<!--tr:95e46e70-->
[Eficiencia en parámetros y expresividad no lineal]{.kicker}

<!--en:bd1d7cc0 kind=slide_text-->
- **Matched $7 \times 7$ receptive field:** Three stacked $3 \times 3$ layers ($r = 1 + 2 + 2 + 2 = 7$) see the exact same $7 \times 7$ patch as one $7 \times 7$ layer.
<!--tr:bd1d7cc0-->
- **Mismo campo receptivo de $7 \times 7$:** tres capas de $3 \times 3$ apiladas ($r = 1 + 2 + 2 + 2 = 7$) ven exactamente el mismo fragmento de $7 \times 7$ que una capa de $7 \times 7$.

<!--en:755ac11a kind=slide_text-->
- **45% fewer parameters:** $3 \times 9 c^2 = 27 c^2$ weights instead of $49 c^2$ across $c$ channels.
<!--tr:755ac11a-->
- **45% menos de parámetros:** $3 \times 9 c^2 = 27 c^2$ pesos en lugar de $49 c^2$ con $c$ canales.

<!--en:e3aa1086 kind=slide_text-->
- **Richer expressivity:** Compounds **3 nonlinear activations** instead of 1 (the core VGG design rule in §{sec_blocks}).
<!--tr:e3aa1086-->
- **Mayor expresividad:** se componen **3 activaciones no lineales** en lugar de 1 (la regla de diseño central de VGG en §{sec_blocks}).

<!--en:9ec83bf6 kind=slide_title-->
::: {.slide title="Theoretical vs. Effective Receptive Fields"}
<!--tr:9ec83bf6-->
::: {.slide title="Campos receptivos teóricos y efectivos"}

<!--en:72a52c3b kind=kicker-->
[Gaussian Decay of Sensitivity Away from the Center]{.kicker}
<!--tr:72a52c3b-->
[Decaimiento gaussiano de la sensibilidad al alejarse del centro]{.kicker}

<!--en:33bb377d kind=slide_text-->
- **Theoretical RF ($r = 2L+1$):** Bounding box of all pixels with *at least one* path to the output unit.
<!--tr:33bb377d-->
- **CR teórico ($r = 2L+1$):** caja delimitadora de todos los píxeles con *al menos un* camino hasta la unidad de salida.

<!--en:293436eb kind=slide_text-->
- **Central Limit Theorem:** Center pixels connect via exponentially many paths, whereas corner pixels connect via only **1 path**.
<!--tr:293436eb-->
- **Teorema del límite central:** los píxeles centrales se conectan por un número exponencial de caminos, mientras que los píxeles de las esquinas se conectan por **un solo camino**.

<!--en:f47d181e kind=slide_text-->
- **Effective RF (Luo et al., 2016):** Input gradient magnitude $\partial y / \partial \mathbf{X}$ decays in a Gaussian bell curve with std $\sigma \propto \sqrt{L}$.
<!--tr:f47d181e-->
- **CR efectivo (Luo et al., 2016):** la magnitud del gradiente respecto de la entrada $\partial y / \partial \mathbf{X}$ decae como una campana gaussiana con desviación estándar $\sigma \propto \sqrt{L}$.

<!--en:fac792d0 kind=slide_title-->
::: {.slide title="Concept Check: Cross-Correlation & Receptive Fields"}
<!--tr:fac792d0-->
::: {.slide title="Comprobación de conceptos: correlación cruzada y campos receptivos"}

<!--en:26a317d8 kind=kicker-->
[Concept Check]{.kicker}
<!--tr:26a317d8-->
[Comprobación de conceptos]{.kicker}

<!--en:8fe554e5 kind=slide_text-->
Test your understanding of output shapes, `im2col`, and receptive-field arithmetic:
<!--tr:8fe554e5-->
Pon a prueba tu comprensión de las formas de salida, `im2col` y la aritmética de los campos receptivos:

<!--en:281c7c78 kind=slide_text-->
1. **im2col matrix dimensions:** You apply an unpadded $3 \times 3$ single-channel kernel to a $10 \times 10$ image via `im2col`. What is the output spatial shape, and what are the exact dimensions of the `patches` matrix and flattened kernel vector?
<!--tr:281c7c78-->
1. **Dimensiones de la matriz im2col:** aplicas un núcleo de $3 \times 3$ de un solo canal, sin relleno, a una imagen de $10 \times 10$ mediante `im2col`. ¿Cuál es la forma espacial de la salida y cuáles son las dimensiones exactas de la matriz `patches` y del vector del núcleo aplanado?

<!--en:16976d18 kind=slide_text-->
2. **Depth to span Fashion-MNIST:** At stride 1, how many stacked $3 \times 3$ layers are required so a single unit's theoretical receptive field spans an entire $28 \times 28$ image? How many with $5 \times 5$ layers?
<!--tr:16976d18-->
2. **Profundidad para abarcar Fashion-MNIST:** con stride 1, ¿cuántas capas de $3 \times 3$ apiladas hacen falta para que el campo receptivo teórico de una sola unidad abarque una imagen entera de $28 \times 28$? ¿Cuántas con capas de $5 \times 5$?

<!--en:ab2e0d6a kind=slide_text-->
3. **Edge-loss boundary shift:** In our $6 \times 8$ vertical-edge learning experiment, why did the loss start at $\approx 12$, and what would the initial loss be if the black band extended all the way to the right edge (`X[:, 2:8] = 0`)?
<!--tr:ab2e0d6a-->
3. **Desplazamiento de la frontera y pérdida por bordes:** en nuestro experimento de aprendizaje de bordes verticales con una imagen de $6 \times 8$, ¿por qué la pérdida empezó en $\approx 12$ y cuál sería la pérdida inicial si la banda negra se extendiera hasta el borde derecho (`X[:, 2:8] = 0`)?

<!--en:250d6877 kind=slide_text-->
**Answers:**
(1) Output shape is $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ outputs). The `patches` matrix has shape $(64, 9)$ and multiplies the $(9, 1)$ kernel vector.
(2) For $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ layers. For $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ layers.
(3) Each of the 6 rows had two $\pm 1$ edges ($6 \times 2 \times 1^2 = 12$). Extending the black band to the right border eliminates the right-hand black-to-white edge, leaving one edge per row and halving the initial loss to $\approx 6$.
<!--tr:250d6877-->
**Respuestas:**
(1) La forma de la salida es $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ salidas). La matriz `patches` tiene forma $(64, 9)$ y se multiplica por el vector del núcleo de forma $(9, 1)$.
(2) Para $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ capas. Para $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ capas.
(3) Cada una de las 6 filas tenía dos bordes de $\pm 1$ ($6 \times 2 \times 1^2 = 12$). Al extender la banda negra hasta el borde derecho desaparece el borde derecho de negro a blanco, con lo que queda un borde por fila y la pérdida inicial se reduce a la mitad, hasta $\approx 6$.

<!--en:ba43a2dd kind=slide_title-->
::: {.slide title="Summary & Key Takeaways"}
<!--tr:ba43a2dd-->
::: {.slide title="Resumen y conclusiones clave"}

<!--en:ce12e85f kind=kicker-->
[Section Synthesis]{.kicker}
<!--tr:ce12e85f-->
[Síntesis de la sección]{.kicker}

<!--en:00011fb7 kind=slide_text-->
- **2D Cross-Correlation & Edge Learning:** Slides $\mathbf{K}$ across valid windows ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); SGD learns $[1, -1]$ in 9 steps.
<!--tr:00011fb7-->
- **Correlación cruzada 2D y aprendizaje de bordes:** desliza $\mathbf{K}$ por las ventanas válidas ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); SGD aprende $[1, -1]$ en 9 pasos.

<!--en:09efe4f0 kind=slide_text-->
- **im2col Matrix Multiply:** Trades $k_\textrm{h} k_\textrm{w}\times$ patch duplication for peak GPU GEMM throughput.
<!--tr:09efe4f0-->
- **Multiplicación de matrices im2col:** cambia la duplicación de $k_\textrm{h} k_\textrm{w}\times$ fragmentos por el máximo rendimiento de GEMM en la GPU.

<!--en:3af8acd0 kind=slide_text-->
- **Stacked Receptive Fields:** $r = 1 + \sum (k_i - 1) = 2L+1$ for $3 \times 3$ stacks, with Gaussian center sensitivity.
<!--tr:3af8acd0-->
- **Campos receptivos apilados:** $r = 1 + \sum (k_i - 1) = 2L+1$ para pilas de $3 \times 3$, con sensibilidad gaussiana en el centro.

=== DECISIONS ===
- "patch" (im2col) → "fragmento", "patch matrix" → "matriz de fragmentos"; the code identifier `patches` stays as is.
- "feature map" → "mapa de características"; "receptive field" → "campo receptivo" (abbreviated "CR" on slides only, after the full term on the same card).
- "forward propagation" → "propagación hacia adelante"; "backward pass" → "pasada hacia atrás"; "transposed convolution" → "convolución transpuesta".
- "unit" (a neuron/activation in a layer) → "unidad"; "tower" (figure of stacked layers) → "torre"; "dot product" → "producto escalar".
- Slide bullets that address the reader (concept check) use informal "tú" imperative/second person singular ("Pon a prueba", "aplicas").
- "stride" stays English; "framework", "GEMM", "Tensor Cores", "VGG" unchanged.
- "std" → "desviación estándar" (glossary term).

=== NOTES ===
- 3ba929d2/1fb18ec4: "peak GPU Tensor Core speed" translated as "velocidad máxima de los Tensor Cores de la GPU".
- 755ac11a: English says "45% fewer parameters" (27 vs 49 is 45% fewer); kept as written.
- 293436eb: the English slide attributes the bell shape to the "Central Limit Theorem"; translated as written.
- 76713479, 3bcd7d8a, 09efe4f0: 'matrix' warnings are false matches (the plural "matrices" / "matriz" forms are present in the translation as "producto/multiplicación de matrices").
- 293436eb: 'numbers not found: 1' is a false match: "1 path" is written "un solo camino"; the 'Central Limit Theorem' warning: the slide title-cased term is translated "Teorema del límite central", the glossary term, so that warning no longer appears.
