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
Aunque los bucles anidados de `corr2d` dejan las matemáticas a la vista, recorrer en Python los píxeles de la salida de uno en uno es demasiado lento para cargas de trabajo reales. Un remedio clásico, que sigue siendo uno de los algoritmos entre los que eligen las bibliotecas, consiste en reformular la correlación cruzada como una única multiplicación de matrices grande, justo la operación que las GPU ejecutan a máxima velocidad. Observa que cada entrada de la salida en :numref:`fig_correlation` no es más que el producto escalar del núcleo aplanado, de $k_\textrm{h} k_\textrm{w}$ elementos, por la ventana de entrada correspondiente, también aplanada. Si extraemos las $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ ventanas y las apilamos como filas, obtenemos una matriz de parches; al multiplicarla por el núcleo aplanado se calcula toda la correlación cruzada de una sola vez. Esta transformación estándar de la disposición de los datos se llama *im2col* («image to columns», imagen a columnas; aquí, a filas). :numref:`fig_conv_im2col` la lleva a cabo con el ejemplo de :numref:`fig_correlation`.

<!--en:107554a5 kind=caption-->
![The cross-correlation of :numref:`fig_correlation` as one matrix product. Each of the four $2 \times 2$ windows of the input becomes a row of the patch matrix, and multiplying by the flattened kernel gives the flattened output. The shaded cells follow the first window to its row and to the output 19. The center pixel 4 lies in all four windows, so it appears in every row.](../img/conv-im2col.svg)
<!--tr:107554a5-->
![La correlación cruzada de :numref:`fig_correlation` como un solo producto de matrices. Cada una de las cuatro ventanas de $2 \times 2$ de la entrada se convierte en una fila de la matriz de parches, y al multiplicar por el núcleo aplanado se obtiene la salida aplanada. Las celdas sombreadas siguen a la primera ventana hasta su fila y hasta la salida 19. El píxel central 4 está en las cuatro ventanas, así que aparece en todas las filas.](../img/conv-im2col.svg)

<!--en:76713479 kind=paragraph-->
The trade-off is extra workspace memory, since each interior pixel is duplicated
into up to $k_\textrm{h} k_\textrm{w}$ patches. Below we verify on
:numref:`fig_correlation` that this single matrix multiply matches `corr2d`
exactly:
<!--tr:76713479-->
La contrapartida es que hace falta más memoria de trabajo, ya que cada píxel interior se duplica en hasta $k_\textrm{h} k_\textrm{w}$ parches. A continuación comprobamos con el ejemplo de :numref:`fig_correlation` que esta única multiplicación de matrices coincide exactamente con `corr2d`:

<!--en:3bcd7d8a kind=paragraph-->
The discrepancy is identically zero: one matrix product reproduces `corr2d`
perfectly.
<!--tr:3bcd7d8a-->
La discrepancia es idénticamente cero: un solo producto de matrices reproduce `corr2d` a la perfección.

<!--en:c30f2dea kind=paragraph-->
Equivalently, we can unroll the *kernel* instead of the input into a sparse
banded matrix $\mathbf{W}$ multiplying the flattened image—the exact 2D analog
of panel (c) in :numref:`fig_why_conv_weight_sharing`, where every row repeats
the same handful of kernel weights. Seeing a convolution as a structured dense
layer brings us full circle to the derivation of :numref:`sec_why-conv` (and the
exercises build $\mathbf{W}$ explicitly).
<!--tr:c30f2dea-->
De forma equivalente, en lugar de la entrada podemos desenrollar el *núcleo* en una matriz banda dispersa $\mathbf{W}$ que multiplica a la imagen aplanada: es el análogo 2D exacto del panel (c) de :numref:`fig_why_conv_weight_sharing`, donde cada fila repite el mismo puñado de pesos del núcleo. Ver una convolución como una capa densa con estructura nos devuelve al punto de partida, la deducción de :numref:`sec_why-conv` (y en los ejercicios se construye $\mathbf{W}$ explícitamente).

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
Una sola capa convolucional ve únicamente un parche del tamaño de su núcleo, mientras que reconocer objetos enteros exige integrar el contexto de zonas mucho más amplias. Recuerda de :numref:`subsec_why-conv-channels` que la salida de una capa convolucional se llama *mapa de características* (como nuestro mapa de bordes verticales `Y`). Para cualquier elemento $x$ de un mapa de características, su *campo receptivo* es la región de la imagen de entrada (o de una capa anterior) cuyos valores pueden influir en $x$ durante la pasada hacia adelante; el término está tomado de la neurofisiología, como en :numref:`field_visual`.

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
En :numref:`fig_correlation`, la salida sombreada `19` tiene un campo receptivo de $2 \times 2$ que cubre los cuatro píxeles de la esquina superior izquierda de $\mathbf{X}$. Apila ahora una *segunda* capa convolucional de $2 \times 2$ sobre la salida $\mathbf{Y}$, de $2 \times 2$, para producir un único escalar $z$: $z$ depende de los cuatro elementos de $\mathbf{Y}$, que en conjunto dependen de los nueve elementos de $\mathbf{X}$. Al apilar capas, el campo receptivo se amplía de forma gradual con la profundidad.

<!--en:f95c9394 kind=paragraph-->
How quickly does it grow? Each $k \times k$ convolutional layer reaches $k - 1$
pixels farther than a single element of its input, so stacking $L$ layers with
kernel sizes $k_1, \ldots, k_L$ yields a receptive field of side length
<!--tr:f95c9394-->
¿Con qué rapidez crece? Cada capa convolucional de $k \times k$ alcanza $k - 1$ píxeles más allá que un solo elemento de su entrada, de modo que al apilar $L$ capas con tamaños del núcleo $k_1, \ldots, k_L$ se obtiene un campo receptivo de lado

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
En una pila de $L$ capas consecutivas de $3 \times 3$, cada capa añade 2 píxeles, con lo que se cubren $(2L + 1) \times (2L + 1)$ entradas: $5 \times 5$ tras dos capas y $7 \times 7$ tras tres (:numref:`fig_conv_receptive_field`). En realidad, este crecimiento lineal es una ventaja y no un motivo para recurrir a núcleos enormes: tres capas apiladas de $3 \times 3$ logran el mismo campo receptivo de $7 \times 7$ que una sola capa de $7 \times 7$ con muchos menos parámetros ($3 \times 9 = 27$ pesos frente a 49 por cada par de canales) y, además, intercalan tres no linealidades en lugar de una.

<!--en:39917513 kind=caption-->
![Receptive field of one unit after three one-dimensional layers with kernel size 3. Shaded cells are the units that the top unit depends on, and the lines connect each of them to its inputs. Each layer widens the receptive field by 2, to 7 inputs.](../img/conv-receptive-field.svg)
<!--tr:39917513-->
![Campo receptivo de una unidad tras tres capas unidimensionales con tamaño del núcleo 3. Las celdas sombreadas son las unidades de las que depende la unidad superior, y las líneas conectan cada una de ellas con sus entradas. Cada capa ensancha el campo receptivo en 2, hasta llegar a 7 entradas.](../img/conv-receptive-field.svg)

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
Ten en cuenta que $r$ mide el campo receptivo *teórico* (todo píxel de entrada con *algún* camino hasta una unidad de salida), sin decir con qué fuerza contribuye realmente cada píxel. Midiendo directamente los gradientes respecto a la entrada, :citet:`Luo.Li.Urtasun.ea.2016` mostraron que el campo receptivo *efectivo* decae con rapidez al alejarse del centro, con una forma de campana aproximadamente gaussiana, tanto en redes sin entrenar como en redes entrenadas. En teoría, las unidades profundas pueden ver muy lejos en la imagen, pero su sensibilidad se concentra en gran medida cerca del centro.

<!--en:eed4e2a0 kind=paragraph-->
Counting paths explains the bell shape. An input at the edge of the receptive
field reaches the output unit along a single chain of connections, whereas the
center input can take many routes through the layers in between.
:numref:`fig_conv_rf_paths` counts them for the tower of
:numref:`fig_conv_receptive_field`: the center input has seven paths to the top
unit and each edge input has one. If all weights have a similar magnitude, the
influence of an input grows with its number of paths.
<!--tr:eed4e2a0-->
Contar caminos explica la forma de campana. Una entrada situada en el extremo del campo receptivo llega a la unidad de salida por una única cadena de conexiones, mientras que la entrada central puede tomar muchas rutas a través de las capas intermedias. :numref:`fig_conv_rf_paths` las cuenta para la torre de :numref:`fig_conv_receptive_field`: la entrada central tiene siete caminos hasta la unidad superior y cada entrada de los extremos tiene uno. Si todos los pesos tienen una magnitud similar, la influencia de una entrada crece con su número de caminos.

<!--en:b92775ee kind=caption-->
![Number of paths from each unit to the top unit of :numref:`fig_conv_receptive_field` (three one-dimensional layers with kernel size 3). Each count is the sum of the counts of the units that the cell feeds. Darker cells have more paths than the other cells of their layer, and the bars repeat the counts of the input row: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)
<!--tr:b92775ee-->
![Número de caminos desde cada unidad hasta la unidad superior de :numref:`fig_conv_receptive_field` (tres capas unidimensionales con tamaño del núcleo 3). Cada recuento es la suma de los recuentos de las unidades a las que alimenta la celda. Las celdas más oscuras tienen más caminos que las demás celdas de su capa, y las barras repiten los recuentos de la fila de entrada: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)

<!--en:405695e8 kind=paragraph-->
Two practical gaps remain in our basic layer: every unpadded convolution shrinks
spatial dimensions while under-sampling border pixels, and a single kernel
detects only one pattern. We address spatial sizing and borders next in
:numref:`sec_padding`, and multi-channel feature maps in :numref:`sec_channels`.
<!--tr:405695e8-->
A nuestra capa básica le quedan dos carencias prácticas: toda convolución sin relleno encoge las dimensiones espaciales y, a la vez, submuestrea los píxeles del borde de la imagen, y un solo núcleo detecta un único patrón. Abordamos el tamaño espacial y los bordes de la imagen a continuación, en :numref:`sec_padding`, y los mapas de características con varios canales en :numref:`sec_channels`.

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
Una capa convolucional aplica la correlación cruzada bidimensional con un núcleo aprendido, más un sesgo escalar, deslizando el núcleo por todas las ventanas válidas de la entrada. Un sencillo núcleo de diferencia finita de $1 \times 2$, `[1, -1]`, detecta bordes verticales en cualquier lugar de una imagen, y el descenso de gradiente recupera exactamente esos pesos solo a partir de datos de entrada y salida, igual que las CNN profundas aprenden jerarquías enteras de filtros de extremo a extremo. La convolución matemática voltea el núcleo, lo que no supone ninguna diferencia cuando los núcleos se aprenden. Una convolución puede evaluarse como una única multiplicación de matrices (*im2col*), y al apilar $L$ capas convolucionales el campo receptivo de cada unidad se ensancha linealmente, en $k_i - 1$ por capa.

<!--en:3ba929d2 kind=slide_title-->
::: {.slide title="Fast Execution: Convolution via Matrix Multiplication"}
<!--tr:3ba929d2-->
::: {.slide title="Ejecución rápida: la convolución como multiplicación de matrices"}

<!--en:1711f812 kind=kicker-->
[The im2col (Image-to-Columns) Transformation]{.kicker}
<!--tr:1711f812-->
[La transformación im2col (imagen a columnas)]{.kicker}

<!--en:a040f3d7 kind=slide_text-->
- **Single dense GEMM:** Flatten each $k_\textrm{h} \times k_\textrm{w}$ window into a row of `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ and multiply by $\textrm{vec}(\mathbf{K})$.
<!--tr:a040f3d7-->
- **Una sola GEMM densa:** cada ventana de $k_\textrm{h} \times k_\textrm{w}$ se aplana en una fila de `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ y se multiplica por $\textrm{vec}(\mathbf{K})$.

<!--en:1fb18ec4 kind=slide_text-->
- **Trade-off:** Peak GPU Tensor Core speed at the cost of $k_\textrm{h} k_\textrm{w}\times$ patch workspace duplication.
<!--tr:1fb18ec4-->
- **Contrapartida:** velocidad máxima de los Tensor Core de la GPU a cambio de duplicar $k_\textrm{h} k_\textrm{w}\times$ los parches en la memoria de trabajo.

<!--en:0d757cf6 kind=slide_title-->
::: {.slide title="Dual View: Unrolling the Kernel into a Banded Matrix"}
<!--tr:0d757cf6-->
::: {.slide title="Vista dual: el núcleo desenrollado en una matriz banda"}

<!--en:7f0defaa kind=kicker-->
[Toeplitz Weight Matrix Multiplication]{.kicker}
<!--tr:7f0defaa-->
[Multiplicación por una matriz de pesos de Toeplitz]{.kicker}

<!--en:5898ba57 kind=slide_text-->
Instead of unfolding the input image into duplicated patches, we can unroll the *kernel* into a sparse banded matrix $\mathbf{W}$ that multiplies the flattened input vector $\textrm{vec}(\mathbf{X})$:
<!--tr:5898ba57-->
En lugar de desplegar la imagen de entrada en parches duplicados, podemos desenrollar el *núcleo* en una matriz banda dispersa $\mathbf{W}$ que multiplica al vector de entrada aplanado $\textrm{vec}(\mathbf{X})$:

<!--en:dae51741 kind=slide_text-->
- **Matrix–vector form:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, where $\mathbf{W}$ has $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ rows and $n_\textrm{h} n_\textrm{w}$ columns.
<!--tr:dae51741-->
- **Forma matriz-vector:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, donde $\mathbf{W}$ tiene $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ filas y $n_\textrm{h} n_\textrm{w}$ columnas.

<!--en:f0e86ee3 kind=slide_text-->
- **Connection to §{sec_why-conv} and backprop:** Every row of $\mathbf{W}$ holds the same $k_\textrm{h} k_\textrm{w}$ kernel entries shifted across columns (panel c), making the backward pass with respect to $\mathbf{X}$ simply multiplication by $\mathbf{W}^\top$ (a transposed convolution).
<!--tr:f0e86ee3-->
- **Relación con §{sec_why-conv} y con la retropropagación:** cada fila de $\mathbf{W}$ contiene las mismas $k_\textrm{h} k_\textrm{w}$ entradas del núcleo desplazadas entre columnas (panel c), de modo que la pasada hacia atrás respecto a $\mathbf{X}$ es simplemente la multiplicación por $\mathbf{W}^\top$ (una convolución transpuesta).

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
El **campo receptivo** de una unidad es la región de la imagen de entrada original cuyos píxeles pueden influir en la activación de esa unidad durante la pasada hacia adelante:

<!--en:eba74f75 kind=slide_text-->
- **Single layer ($L=1$):** A $k_1 \times k_1$ kernel sees a $k_1 \times k_1$ input patch.
<!--tr:eba74f75-->
- **Una sola capa ($L=1$):** un núcleo de $k_1 \times k_1$ ve un parche de entrada de $k_1 \times k_1$.

<!--en:01c175bf kind=slide_text-->
- **Stacking $L$ stride-1 layers:** Layer $i$ with kernel size $k_i$ reaches $k_i - 1$ pixels farther along each axis than a single element of layer $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$
<!--tr:01c175bf-->
- **Apilamiento de $L$ capas con stride 1:** la capa $i$, con tamaño del núcleo $k_i$, alcanza en cada eje $k_i - 1$ píxeles más allá que un solo elemento de la capa $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$

<!--en:2bde5822 kind=slide_text-->
- **$L$ stacked $3 \times 3$ layers:** Each layer adds $3 - 1 = 2$ pixels, giving side length $r = 2L + 1$ ($3 \to 5 \to 7$).
<!--tr:2bde5822-->
- **$L$ capas apiladas de $3 \times 3$:** cada capa añade $3 - 1 = 2$ píxeles, lo que da un lado $r = 2L + 1$ ($3 \to 5 \to 7$).

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
- **Mismo campo receptivo de $7 \times 7$:** tres capas apiladas de $3 \times 3$ ($r = 1 + 2 + 2 + 2 = 7$) ven exactamente el mismo parche de $7 \times 7$ que una capa de $7 \times 7$.

<!--en:755ac11a kind=slide_text-->
- **45% fewer parameters:** $3 \times 9 c^2 = 27 c^2$ weights instead of $49 c^2$ across $c$ channels.
<!--tr:755ac11a-->
- **Un 45% menos de parámetros:** $3 \times 9 c^2 = 27 c^2$ pesos en lugar de $49 c^2$ con $c$ canales.

<!--en:e3aa1086 kind=slide_text-->
- **Richer expressivity:** Compounds **3 nonlinear activations** instead of 1 (the core VGG design rule in §{sec_blocks}).
<!--tr:e3aa1086-->
- **Mayor expresividad:** compone **3 activaciones no lineales** en lugar de 1 (la regla de diseño central de VGG en §{sec_blocks}).

<!--en:9ec83bf6 kind=slide_title-->
::: {.slide title="Theoretical vs. Effective Receptive Fields"}
<!--tr:9ec83bf6-->
::: {.slide title="Campos receptivos teóricos y efectivos"}

<!--en:72a52c3b kind=kicker-->
[Gaussian Decay of Sensitivity Away from the Center]{.kicker}
<!--tr:72a52c3b-->
[Decaimiento gaussiano de la sensibilidad lejos del centro]{.kicker}

<!--en:33bb377d kind=slide_text-->
- **Theoretical RF ($r = 2L+1$):** Bounding box of all pixels with *at least one* path to the output unit.
<!--tr:33bb377d-->
- **Campo receptivo teórico ($r = 2L+1$):** cuadro delimitador de todos los píxeles con *al menos un* camino hasta la unidad de salida.

<!--en:293436eb kind=slide_text-->
- **Central Limit Theorem:** Center pixels connect via exponentially many paths, whereas corner pixels connect via only **1 path**.
<!--tr:293436eb-->
- **Teorema del límite central:** los píxeles centrales se conectan por un número exponencial de caminos, mientras que los de las esquinas lo hacen por **1 solo camino**.

<!--en:f47d181e kind=slide_text-->
- **Effective RF (Luo et al., 2016):** Input gradient magnitude $\partial y / \partial \mathbf{X}$ decays in a Gaussian bell curve with std $\sigma \propto \sqrt{L}$.
<!--tr:f47d181e-->
- **Campo receptivo efectivo (Luo et al., 2016):** la magnitud del gradiente respecto a la entrada, $\partial y / \partial \mathbf{X}$, decae como una campana gaussiana con desviación estándar $\sigma \propto \sqrt{L}$.

<!--en:fac792d0 kind=slide_title-->
::: {.slide title="Concept Check: Cross-Correlation & Receptive Fields"}
<!--tr:fac792d0-->
::: {.slide title="Repaso de conceptos: correlación cruzada y campos receptivos"}

<!--en:26a317d8 kind=kicker-->
[Concept Check]{.kicker}
<!--tr:26a317d8-->
[Repaso de conceptos]{.kicker}

<!--en:8fe554e5 kind=slide_text-->
Test your understanding of output shapes, `im2col`, and receptive-field arithmetic:
<!--tr:8fe554e5-->
Pon a prueba lo que has entendido sobre las formas de salida, `im2col` y la aritmética de los campos receptivos:

<!--en:281c7c78 kind=slide_text-->
1. **im2col matrix dimensions:** You apply an unpadded $3 \times 3$ single-channel kernel to a $10 \times 10$ image via `im2col`. What is the output spatial shape, and what are the exact dimensions of the `patches` matrix and flattened kernel vector?
<!--tr:281c7c78-->
1. **Dimensiones de las matrices de im2col:** aplicas un núcleo de $3 \times 3$, de un solo canal y sin relleno, a una imagen de $10 \times 10$ mediante `im2col`. ¿Cuál es la forma espacial de la salida, y cuáles son las dimensiones exactas de la matriz `patches` y del vector del núcleo aplanado?

<!--en:16976d18 kind=slide_text-->
2. **Depth to span Fashion-MNIST:** At stride 1, how many stacked $3 \times 3$ layers are required so a single unit's theoretical receptive field spans an entire $28 \times 28$ image? How many with $5 \times 5$ layers?
<!--tr:16976d18-->
2. **Profundidad para abarcar Fashion-MNIST:** con stride 1, ¿cuántas capas apiladas de $3 \times 3$ hacen falta para que el campo receptivo teórico de una sola unidad abarque una imagen entera de $28 \times 28$? ¿Y cuántas con capas de $5 \times 5$?

<!--en:ab2e0d6a kind=slide_text-->
3. **Edge-loss boundary shift:** In our $6 \times 8$ vertical-edge learning experiment, why did the loss start at $\approx 12$, and what would the initial loss be if the black band extended all the way to the right edge (`X[:, 2:8] = 0`)?
<!--tr:ab2e0d6a-->
3. **La pérdida si la banda llega al borde de la imagen:** en nuestro experimento de aprendizaje de bordes verticales con la imagen de $6 \times 8$, ¿por qué empezó la pérdida en $\approx 12$, y cuál sería la pérdida inicial si la banda negra se extendiera hasta el borde derecho de la imagen (`X[:, 2:8] = 0`)?

<!--en:250d6877 kind=slide_text-->
**Answers:**
(1) Output shape is $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ outputs). The `patches` matrix has shape $(64, 9)$ and multiplies the $(9, 1)$ kernel vector.
(2) For $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ layers. For $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ layers.
(3) Each of the 6 rows had two $\pm 1$ edges ($6 \times 2 \times 1^2 = 12$). Extending the black band to the right border eliminates the right-hand black-to-white edge, leaving one edge per row and halving the initial loss to $\approx 6$.
<!--tr:250d6877-->
**Respuestas:**
(1) La forma de la salida es $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ salidas). La matriz `patches` tiene forma $(64, 9)$ y multiplica al vector del núcleo, de forma $(9, 1)$.
(2) Con $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ capas. Con $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ capas.
(3) Cada una de las 6 filas tenía dos bordes $\pm 1$ ($6 \times 2 \times 1^2 = 12$). Al extender la banda negra hasta el borde derecho de la imagen desaparece el borde de negro a blanco de la derecha, queda un borde por fila y la pérdida inicial se reduce a la mitad, $\approx 6$.

<!--en:ba43a2dd kind=slide_title-->
::: {.slide title="Summary & Key Takeaways"}
<!--tr:ba43a2dd-->
::: {.slide title="Resumen e ideas clave"}

<!--en:ce12e85f kind=kicker-->
[Section Synthesis]{.kicker}
<!--tr:ce12e85f-->
[Síntesis de la sección]{.kicker}

<!--en:00011fb7 kind=slide_text-->
- **2D Cross-Correlation & Edge Learning:** Slides $\mathbf{K}$ across valid windows ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); SGD learns $[1, -1]$ in 9 steps.
<!--tr:00011fb7-->
- **Correlación cruzada 2D y aprendizaje de bordes:** desliza $\mathbf{K}$ por las ventanas válidas ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); el SGD aprende $[1, -1]$ en 9 pasos.

<!--en:09efe4f0 kind=slide_text-->
- **im2col Matrix Multiply:** Trades $k_\textrm{h} k_\textrm{w}\times$ patch duplication for peak GPU GEMM throughput.
<!--tr:09efe4f0-->
- **Multiplicación de matrices con im2col:** duplica los parches $k_\textrm{h} k_\textrm{w}\times$ a cambio del máximo throughput de GEMM en la GPU.

<!--en:3af8acd0 kind=slide_text-->
- **Stacked Receptive Fields:** $r = 1 + \sum (k_i - 1) = 2L+1$ for $3 \times 3$ stacks, with Gaussian center sensitivity.
<!--tr:3af8acd0-->
- **Campos receptivos apilados:** $r = 1 + \sum (k_i - 1) = 2L+1$ para pilas de $3 \times 3$, con una sensibilidad gaussiana concentrada en el centro.

=== DECISIONS ===
- "flattened" → «aplanado»; "to flatten" → «aplanar»; "flattening" → «aplanamiento».
- "patch matrix" → «matriz de parches»; "workspace memory" → «memoria de trabajo».
- "to unroll (the kernel)" → «desenrollar»; "to unfold (the input)" → «desplegar».
- "sparse banded matrix" → «matriz banda dispersa»; "banded matrix" → «matriz banda».
- "trade-off" → «contrapartida».
- "matrix product / matrix multiply" → «producto de matrices» / «multiplicación de matrices».
- "forward propagation" → «pasada hacia adelante» (same glossary entry as "forward pass"); "backprop" → «retropropagación».
- "path" (through the layers) → «camino»; "route" → «ruta»; "count" (noun) → «recuento».
- "side length" (of a receptive field) → «lado».
- "stack" (noun, of layers) → «pila»; "tower" → «torre».
- "unit" (of a network) → «unidad»; "top unit" → «unidad superior».
- "input gradient" → «gradiente respecto a la entrada».
- "theoretical / effective receptive field" → «campo receptivo teórico / efectivo»; the slide abbreviation "RF" is written out («campo receptivo»), and "std" as «desviación estándar».
- "Gaussian bell curve" → «campana gaussiana» / «forma de campana … gaussiana».
- "unpadded" → «sin relleno»; "single-channel" → «de un solo canal»; "stride-1" → «con stride 1».
- "untrained / trained networks" → «redes sin entrenar / entrenadas».
- "Summary" → «Resumen»; "Concept Check" → «Repaso de conceptos»; "Answers" → «Respuestas»; "Summary & Key Takeaways" → «Resumen e ideas clave».
- GEMM, Tensor Core, VGG, Toeplitz, im2col stay as in English; «(Luo et al., 2016)» keeps "et al.".
- The English em dash is not copied: use a colon, a semicolon or parentheses.

=== NOTES ===
- 5bdffcd7: «("image to columns", or rows here)» is rendered «(«image to columns», imagen a columnas; aquí, a filas)»: the English expansion of the name is kept and its Spanish meaning added in three words, because the parenthesis explains the name *im2col* and would say nothing to a reader without English. Remove «imagen a columnas» if no gloss is wanted.
- e7983df8: "Recall from" is the verb (recuerda), not the metric *recall* (exhaustividad); a glossary warning here would be a false match. «forward propagation» follows the glossary entry "forward pass (forward propagation)" → «pasada hacia adelante». The dash clause "borrowing the neurophysiology term from :numref:`field_visual`" is rendered as a clause after a semicolon.
- 16976d18: "span" is the verb (abarcar), not the linear-algebra *span* (espacio generado); false match if warned.
- 1fb18ec4: "at the cost of" is rendered «a cambio de»; it is not the glossary term *cost* (costo).
- 09efe4f0: "throughput" kept as «throughput» per the (proposed) glossary; «rendimiento» would be the Spanish alternative.
- ab2e0d6a: the slide label "Edge-loss boundary shift" is a compressed noun stack; translated by its meaning («La pérdida si la banda llega al borde de la imagen»).
- 33bb377d, f47d181e: "RF" and "std" are written out; Spanish has no established abbreviations for them.
- 24a25ec4: math tokens reordered on purpose: «sobre la salida $\mathbf{Y}$, de $2 \times 2$» (English: "the $2 \times 2$ output $\mathbf{Y}$"); Spanish puts the size after the name. All tokens are present and unchanged.
- 76713479, 3bcd7d8a, 09efe4f0: the check reports "glossary term not used: 'matrix' -> matriz"; false match on inflection. The blocks use the plural «matrices» («multiplicación de matrices», the glossary term for "matrix multiplication", and «producto de matrices»).
