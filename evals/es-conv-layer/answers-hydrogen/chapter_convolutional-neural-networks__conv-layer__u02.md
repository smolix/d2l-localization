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
Aunque los bucles anidados en `corr2d` hacen transparentes las matemáticas, visitar un
píxel de salida a la vez en Python resulta demasiado lento para cargas de trabajo reales. Un remedio
clásico, y todavía uno de los algoritmos entre los que eligen las bibliotecas, consiste en reformular
la correlación cruzada como una única multiplicación de matrices de gran tamaño, exactamente la operación
que las GPU ejecutan a máxima velocidad. Nótese que cada
entrada de salida en :numref:`fig_correlation` es simplemente el producto escalar del
núcleo aplanado de $k_\textrm{h} k_\textrm{w}$ elementos con la ventana
de entrada aplanada correspondiente. Extraer todas las
$(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ ventanas y
apilarlas como filas produce una matriz de parches; multiplicarla por el núcleo
aplanado calcula toda la correlación cruzada de una sola vez. Esta transformación estándar de disposición en memoria
se denomina *im2col* (*image to columns*, «imagen a columnas», o a filas en este caso).
:numref:`fig_conv_im2col` la lleva a cabo sobre el ejemplo de
:numref:`fig_correlation`.

<!--en:107554a5 kind=caption-->
![The cross-correlation of :numref:`fig_correlation` as one matrix product. Each of the four $2 \times 2$ windows of the input becomes a row of the patch matrix, and multiplying by the flattened kernel gives the flattened output. The shaded cells follow the first window to its row and to the output 19. The center pixel 4 lies in all four windows, so it appears in every row.](../img/conv-im2col.svg)
<!--tr:107554a5-->
![La correlación cruzada de :numref:`fig_correlation` como un único producto matricial. Cada una de las cuatro ventanas de $2 \times 2$ de la entrada se convierte en una fila de la matriz de parches, y multiplicar por el núcleo aplanado da la salida aplanada. Las celdas sombreadas siguen la primera ventana hasta su fila y hasta la salida 19. El píxel central 4 pertenece a las cuatro ventanas, por lo que aparece en cada fila.](../img/conv-im2col.svg)

<!--en:76713479 kind=paragraph-->
The trade-off is extra workspace memory, since each interior pixel is duplicated
into up to $k_\textrm{h} k_\textrm{w}$ patches. Below we verify on
:numref:`fig_correlation` that this single matrix multiply matches `corr2d`
exactly:
<!--tr:76713479-->
La contrapartida es memoria adicional de trabajo, ya que cada píxel interior se duplica
en hasta $k_\textrm{h} k_\textrm{w}$ parches. A continuación verificamos en
:numref:`fig_correlation` que esta única multiplicación de matrices coincide exactamente con `corr2d`:

<!--en:3bcd7d8a kind=paragraph-->
The discrepancy is identically zero: one matrix product reproduces `corr2d`
perfectly.
<!--tr:3bcd7d8a-->
La discrepancia es idénticamente cero: un solo producto matricial reproduce `corr2d`
a la perfección.

<!--en:c30f2dea kind=paragraph-->
Equivalently, we can unroll the *kernel* instead of the input into a sparse
banded matrix $\mathbf{W}$ multiplying the flattened image—the exact 2D analog
of panel (c) in :numref:`fig_why_conv_weight_sharing`, where every row repeats
the same handful of kernel weights. Seeing a convolution as a structured dense
layer brings us full circle to the derivation of :numref:`sec_why-conv` (and the
exercises build $\mathbf{W}$ explicitly).
<!--tr:c30f2dea-->
De forma equivalente, podemos desenrollar el *núcleo* en lugar de la entrada en una matriz
banda dispersa $\mathbf{W}$ que multiplica a la imagen aplanada: el análogo 2D exacto
del panel (c) en :numref:`fig_why_conv_weight_sharing`, donde cada fila repite
el mismo puñado de pesos del núcleo. Ver una convolución como una capa densa
estructurada nos devuelve al punto de partida de la derivación de :numref:`sec_why-conv` (y los
ejercicios construyen $\mathbf{W}$ explícitamente).

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
Una sola capa convolucional ve únicamente un parche del tamaño de su núcleo, mientras que
reconocer objetos completos exige integrar contexto sobre áreas mucho más amplias.
Recordemos de :numref:`subsec_why-conv-channels` que la salida de una capa convolucional
se denomina *mapa de características* (feature map, como nuestro mapa de bordes verticales `Y`). Para cualquier
elemento $x$ de un mapa de características, su *campo receptivo* (receptive field) es la región de la imagen
de entrada (o de una capa anterior) cuyos valores pueden influir en $x$ durante la pasada hacia
adelante, tomando prestado el término de neurofisiología de :numref:`field_visual`.

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
En :numref:`fig_correlation`, la salida sombreada `19` tiene un campo receptivo de $2 \times 2$
que cubre los cuatro píxeles superiores izquierdos de $\mathbf{X}$. Apilemos ahora una *segunda*
capa convolucional de $2 \times 2$ encima de la salida $\mathbf{Y}$ de $2 \times 2$
para producir un único escalar $z$: $z$ depende de los cuatro elementos de $\mathbf{Y}$,
que en conjunto dependen de los nueve elementos de $\mathbf{X}$. Apilar capas
expande el campo receptivo de forma suave con la profundidad.

<!--en:f95c9394 kind=paragraph-->
How quickly does it grow? Each $k \times k$ convolutional layer reaches $k - 1$
pixels farther than a single element of its input, so stacking $L$ layers with
kernel sizes $k_1, \ldots, k_L$ yields a receptive field of side length
<!--tr:f95c9394-->
¿Qué tan rápido crece? Cada capa convolucional de $k \times k$ alcanza $k - 1$
píxeles más lejos que un único elemento de su entrada, por lo que apilar $L$ capas con
tamaños de núcleo $k_1, \ldots, k_L$ produce un campo receptivo de longitud de lado

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
Para una pila de $L$ capas consecutivas de $3 \times 3$, cada capa añade 2 píxeles,
cubriendo $(2L + 1) \times (2L + 1)$ entradas: $5 \times 5$ tras dos capas y
$7 \times 7$ tras tres (:numref:`fig_conv_receptive_field`). Este crecimiento lineal
es en realidad una ventaja y no un motivo para recurrir a núcleos enormes:
tres capas apiladas de $3 \times 3$ alcanzan el mismo campo receptivo de $7 \times 7$
que una única capa de $7 \times 7$ usando muchos menos parámetros ($3 \times 9 = 27$
pesos frente a 49 por par de canales), a la vez que intercalan tres no linealidades en lugar
de una.

<!--en:39917513 kind=caption-->
![Receptive field of one unit after three one-dimensional layers with kernel size 3. Shaded cells are the units that the top unit depends on, and the lines connect each of them to its inputs. Each layer widens the receptive field by 2, to 7 inputs.](../img/conv-receptive-field.svg)
<!--tr:39917513-->
![Campo receptivo de una unidad tras tres capas unidimensionales con tamaño de núcleo 3. Las celdas sombreadas son las unidades de las que depende la unidad superior, y las líneas conectan cada una de ellas con sus entradas. Cada capa amplía el campo receptivo en 2, hasta 7 entradas.](../img/conv-receptive-field.svg)

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
Tengamos en cuenta que $r$ mide el campo receptivo *teórico* (cada píxel de entrada con
*algún* camino hacia una unidad de salida) sin indicar con qué fuerza contribuye realmente cada píxel.
Al medir directamente los gradientes de entrada,
:citet:`Luo.Li.Urtasun.ea.2016` mostraron que el campo receptivo *efectivo*
decae bruscamente al alejarse del centro siguiendo una curva de campana aproximadamente gaussiana, tanto en
redes sin entrenar como entrenadas. Las unidades profundas pueden ver en teoría muy lejos a través de la
imagen, pero su sensibilidad se concentra fuertemente cerca del centro.

<!--en:eed4e2a0 kind=paragraph-->
Counting paths explains the bell shape. An input at the edge of the receptive
field reaches the output unit along a single chain of connections, whereas the
center input can take many routes through the layers in between.
:numref:`fig_conv_rf_paths` counts them for the tower of
:numref:`fig_conv_receptive_field`: the center input has seven paths to the top
unit and each edge input has one. If all weights have a similar magnitude, the
influence of an input grows with its number of paths.
<!--tr:eed4e2a0-->
Contar los caminos explica la forma de campana. Una entrada en el borde del campo receptivo
llega a la unidad de salida a lo largo de una única cadena de conexiones, mientras que la
entrada central puede tomar muchas rutas a través de las capas intermedias.
:numref:`fig_conv_rf_paths` los cuenta para la torre de
:numref:`fig_conv_receptive_field`: la entrada central tiene siete caminos hacia la unidad superior
y cada entrada del borde tiene uno. Si todos los pesos tienen una magnitud similar, la
influencia de una entrada crece con su número de caminos.

<!--en:b92775ee kind=caption-->
![Number of paths from each unit to the top unit of :numref:`fig_conv_receptive_field` (three one-dimensional layers with kernel size 3). Each count is the sum of the counts of the units that the cell feeds. Darker cells have more paths than the other cells of their layer, and the bars repeat the counts of the input row: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)
<!--tr:b92775ee-->
![Número de caminos desde cada unidad hasta la unidad superior de :numref:`fig_conv_receptive_field` (tres capas unidimensionales con tamaño de núcleo 3). Cada recuento es la suma de los recuentos de las unidades a las que alimenta la celda. Las celdas más oscuras tienen más caminos que las demás celdas de su capa, y las barras repiten los recuentos de la fila de entrada: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)

<!--en:405695e8 kind=paragraph-->
Two practical gaps remain in our basic layer: every unpadded convolution shrinks
spatial dimensions while under-sampling border pixels, and a single kernel
detects only one pattern. We address spatial sizing and borders next in
:numref:`sec_padding`, and multi-channel feature maps in :numref:`sec_channels`.
<!--tr:405695e8-->
Quedan dos limitaciones prácticas en nuestra capa básica: cada convolución sin relleno reduce
las dimensiones espaciales y submuestrea los píxeles del borde, y un único núcleo
detecta solo un patrón. Abordaremos el tamaño espacial y los bordes a continuación en
:numref:`sec_padding`, y los mapas de características con múltiples canales en :numref:`sec_channels`.

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
Una capa convolucional aplica una correlación cruzada bidimensional con un núcleo
aprendido más un sesgo escalar, deslizando el núcleo por cada ventana válida de la
entrada. Un sencillo núcleo de diferencia finita de $1 \times 2$ `[1, -1]` detecta bordes
verticales en cualquier lugar de una imagen, y el descenso de gradiente recupera esos pesos exactos
únicamente a partir de los datos de entrada y salida, tal como las CNN profundas aprenden jerarquías enteras de
filtros de extremo a extremo. La convolución matemática invierte el núcleo, lo que no supone ninguna
diferencia una vez que los núcleos se aprenden. Una convolución puede evaluarse como una única
multiplicación de matrices (*im2col*), y apilar $L$ capas convolucionales
amplía el campo receptivo de cada unidad linealmente en $k_i - 1$ por capa.

<!--en:3ba929d2 kind=slide_title-->
::: {.slide title="Fast Execution: Convolution via Matrix Multiplication"}
<!--tr:3ba929d2-->
::: {.slide title="Ejecución rápida: convolución mediante multiplicación de matrices"}

<!--en:1711f812 kind=kicker-->
[The im2col (Image-to-Columns) Transformation]{.kicker}
<!--tr:1711f812-->
[La transformación im2col (imagen a columnas)]{.kicker}

<!--en:a040f3d7 kind=slide_text-->
- **Single dense GEMM:** Flatten each $k_\textrm{h} \times k_\textrm{w}$ window into a row of `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ and multiply by $\textrm{vec}(\mathbf{K})$.
<!--tr:a040f3d7-->
- **Única GEMM densa:** Aplana cada ventana $k_\textrm{h} \times k_\textrm{w}$ en una fila de `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ y multiplica por $\textrm{vec}(\mathbf{K})$.

<!--en:1fb18ec4 kind=slide_text-->
- **Trade-off:** Peak GPU Tensor Core speed at the cost of $k_\textrm{h} k_\textrm{w}\times$ patch workspace duplication.
<!--tr:1fb18ec4-->
- **Contrapartida:** Máxima velocidad en Tensor Cores de GPU a costa de duplicar $k_\textrm{h} k_\textrm{w}\times$ el espacio de trabajo en parches.

<!--en:0d757cf6 kind=slide_title-->
::: {.slide title="Dual View: Unrolling the Kernel into a Banded Matrix"}
<!--tr:0d757cf6-->
::: {.slide title="Visión dual: desenrollar el núcleo en una matriz banda"}

<!--en:7f0defaa kind=kicker-->
[Toeplitz Weight Matrix Multiplication]{.kicker}
<!--tr:7f0defaa-->
[Multiplicación con matriz de pesos de Toeplitz]{.kicker}

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
- **Conexión con §{sec_why-conv} y retropropagación:** Cada fila de $\mathbf{W}$ contiene las mismas $k_\textrm{h} k_\textrm{w}$ entradas del núcleo desplazadas por columnas (panel c), haciendo que la pasada hacia atrás respecto a $\mathbf{X}$ sea simplemente multiplicar por $\mathbf{W}^\top$ (una convolución transpuesta).

<!--en:af8cf19a kind=slide_title-->
::: {.slide title="Feature Maps and Receptive Fields Across Depth"}
<!--tr:af8cf19a-->
::: {.slide title="Mapas de características y campos receptivos en profundidad"}

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
- **Capa única ($L=1$):** Un núcleo de $k_1 \times k_1$ ve un parche de entrada de $k_1 \times k_1$.

<!--en:01c175bf kind=slide_text-->
- **Stacking $L$ stride-1 layers:** Layer $i$ with kernel size $k_i$ reaches $k_i - 1$ pixels farther along each axis than a single element of layer $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$
<!--tr:01c175bf-->
- **Apilar $L$ capas con stride 1:** La capa $i$ con tamaño de núcleo $k_i$ alcanza $k_i - 1$ píxeles más lejos a lo largo de cada eje que un único elemento de la capa $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$

<!--en:2bde5822 kind=slide_text-->
- **$L$ stacked $3 \times 3$ layers:** Each layer adds $3 - 1 = 2$ pixels, giving side length $r = 2L + 1$ ($3 \to 5 \to 7$).
<!--tr:2bde5822-->
- **$L$ capas apiladas de $3 \times 3$:** Cada capa añade $3 - 1 = 2$ píxeles, dando un lado de longitud $r = 2L + 1$ ($3 \to 5 \to 7$).

<!--en:cbb9e97a kind=slide_title-->
::: {.slide title="Why Deep Stacks of 3×3 Kernels Beat One Large Kernel"}
<!--tr:cbb9e97a-->
::: {.slide title="Por qué pilas de núcleos 3×3 superan a un núcleo grande"}

<!--en:95e46e70 kind=kicker-->
[Parameter Efficiency & Nonlinear Expressivity]{.kicker}
<!--tr:95e46e70-->
[Eficiencia de parámetros y expresividad no lineal]{.kicker}

<!--en:bd1d7cc0 kind=slide_text-->
- **Matched $7 \times 7$ receptive field:** Three stacked $3 \times 3$ layers ($r = 1 + 2 + 2 + 2 = 7$) see the exact same $7 \times 7$ patch as one $7 \times 7$ layer.
<!--tr:bd1d7cc0-->
- **Campo receptivo equivalente de $7 \times 7$:** Tres capas apiladas de $3 \times 3$ ($r = 1 + 2 + 2 + 2 = 7$) ven exactamente el mismo parche de $7 \times 7$ que una sola capa de $7 \times 7$.

<!--en:755ac11a kind=slide_text-->
- **45% fewer parameters:** $3 \times 9 c^2 = 27 c^2$ weights instead of $49 c^2$ across $c$ channels.
<!--tr:755ac11a-->
- **45% menos parámetros:** $3 \times 9 c^2 = 27 c^2$ pesos en lugar de $49 c^2$ a través de $c$ canales.

<!--en:e3aa1086 kind=slide_text-->
- **Richer expressivity:** Compounds **3 nonlinear activations** instead of 1 (the core VGG design rule in §{sec_blocks}).
<!--tr:e3aa1086-->
- **Mayor expresividad:** Compone **3 activaciones no lineales** en lugar de 1 (la regla de diseño central de VGG en §{sec_blocks}).

<!--en:9ec83bf6 kind=slide_title-->
::: {.slide title="Theoretical vs. Effective Receptive Fields"}
<!--tr:9ec83bf6-->
::: {.slide title="Campos receptivos teóricos frente a efectivos"}

<!--en:72a52c3b kind=kicker-->
[Gaussian Decay of Sensitivity Away from the Center]{.kicker}
<!--tr:72a52c3b-->
[Decaimiento gaussiano de la sensibilidad lejos del centro]{.kicker}

<!--en:33bb377d kind=slide_text-->
- **Theoretical RF ($r = 2L+1$):** Bounding box of all pixels with *at least one* path to the output unit.
<!--tr:33bb377d-->
- **Campo receptivo teórico ($r = 2L+1$):** Cuadro delimitador de todos los píxeles con *al menos un* camino hasta la unidad de salida.

<!--en:293436eb kind=slide_text-->
- **Central Limit Theorem:** Center pixels connect via exponentially many paths, whereas corner pixels connect via only **1 path**.
<!--tr:293436eb-->
- **Teorema del límite central:** Los píxeles centrales se conectan por exponencialmente muchos caminos, mientras que los de las esquinas por solo **1 camino**.

<!--en:f47d181e kind=slide_text-->
- **Effective RF (Luo et al., 2016):** Input gradient magnitude $\partial y / \partial \mathbf{X}$ decays in a Gaussian bell curve with std $\sigma \propto \sqrt{L}$.
<!--tr:f47d181e-->
- **Campo receptivo efectivo (Luo et al., 2016):** La magnitud del gradiente de entrada $\partial y / \partial \mathbf{X}$ decae en campana gaussiana con $\sigma \propto \sqrt{L}$.

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
Pon a prueba tu comprensión de las formas de salida, `im2col` y la aritmética de campos receptivos:

<!--en:281c7c78 kind=slide_text-->
1. **im2col matrix dimensions:** You apply an unpadded $3 \times 3$ single-channel kernel to a $10 \times 10$ image via `im2col`. What is the output spatial shape, and what are the exact dimensions of the `patches` matrix and flattened kernel vector?
<!--tr:281c7c78-->
1. **Dimensiones de matriz im2col:** Aplicas un núcleo monocanal de $3 \times 3$ sin relleno a una imagen de $10 \times 10$ mediante `im2col`. ¿Cuál es la forma espacial de salida y cuáles son las dimensiones exactas de la matriz `patches` y del vector del núcleo aplanado?

<!--en:16976d18 kind=slide_text-->
2. **Depth to span Fashion-MNIST:** At stride 1, how many stacked $3 \times 3$ layers are required so a single unit's theoretical receptive field spans an entire $28 \times 28$ image? How many with $5 \times 5$ layers?
<!--tr:16976d18-->
2. **Profundidad para abarcar Fashion-MNIST:** Con stride 1, ¿cuántas capas apiladas de $3 \times 3$ hacen falta para que el campo receptivo teórico de una unidad abarque toda una imagen de $28 \times 28$? ¿Y cuántas con capas de $5 \times 5$?

<!--en:ab2e0d6a kind=slide_text-->
3. **Edge-loss boundary shift:** In our $6 \times 8$ vertical-edge learning experiment, why did the loss start at $\approx 12$, and what would the initial loss be if the black band extended all the way to the right edge (`X[:, 2:8] = 0`)?
<!--tr:ab2e0d6a-->
3. **Desplazamiento del borde en la pérdida:** En nuestro experimento de aprendizaje de bordes verticales de $6 \times 8$, ¿por qué empezó la pérdida en $\approx 12$ y cuál sería la pérdida inicial si la franja negra se extendiera hasta el borde derecho (`X[:, 2:8] = 0`)?

<!--en:250d6877 kind=slide_text-->
**Answers:**
(1) Output shape is $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ outputs). The `patches` matrix has shape $(64, 9)$ and multiplies the $(9, 1)$ kernel vector.
(2) For $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ layers. For $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ layers.
(3) Each of the 6 rows had two $\pm 1$ edges ($6 \times 2 \times 1^2 = 12$). Extending the black band to the right border eliminates the right-hand black-to-white edge, leaving one edge per row and halving the initial loss to $\approx 6$.
<!--tr:250d6877-->
**Respuestas:**
(1) La forma de salida es $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ salidas). La matriz `patches` tiene forma $(64, 9)$ y multiplica al vector del núcleo de $(9, 1)$.
(2) Para $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ capas. Para $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ capas.
(3) Cada una de las 6 filas tenía dos bordes $\pm 1$ ($6 \times 2 \times 1^2 = 12$). Extender la franja negra hasta el borde derecho elimina el borde derecho de negro a blanco, dejando un borde por fila y reduciendo a la mitad la pérdida inicial a $\approx 6$.

<!--en:ba43a2dd kind=slide_title-->
::: {.slide title="Summary & Key Takeaways"}
<!--tr:ba43a2dd-->
::: {.slide title="Resumen y puntos clave"}

<!--en:ce12e85f kind=kicker-->
[Section Synthesis]{.kicker}
<!--tr:ce12e85f-->
[Síntesis de la sección]{.kicker}

<!--en:00011fb7 kind=slide_text-->
- **2D Cross-Correlation & Edge Learning:** Slides $\mathbf{K}$ across valid windows ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); SGD learns $[1, -1]$ in 9 steps.
<!--tr:00011fb7-->
- **Correlación cruzada 2D y aprendizaje de bordes:** Desliza $\mathbf{K}$ por ventanas válidas ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); SGD aprende $[1, -1]$ en 9 pasos.

<!--en:09efe4f0 kind=slide_text-->
- **im2col Matrix Multiply:** Trades $k_\textrm{h} k_\textrm{w}\times$ patch duplication for peak GPU GEMM throughput.
<!--tr:09efe4f0-->
- **Multiplicación matricial im2col:** Intercambia duplicación de parches de $k_\textrm{h} k_\textrm{w}\times$ por máximo throughput GEMM en GPU.

<!--en:3af8acd0 kind=slide_text-->
- **Stacked Receptive Fields:** $r = 1 + \sum (k_i - 1) = 2L+1$ for $3 \times 3$ stacks, with Gaussian center sensitivity.
<!--tr:3af8acd0-->
- **Campos receptivos apilados:** $r = 1 + \sum (k_i - 1) = 2L+1$ para pilas $3 \times 3$, con sensibilidad gaussiana centrada.

=== DECISIONS ===
- Register: first-person plural ("nosotros") for narrative prose, matching the English "we".
- Headings and slide titles: sentence case in Spanish, noun-phrase style without trailing full stops.
- First use of key technical terms: include the English term in parentheses on first introduction where helpful (e.g., *correlación cruzada bidimensional* (two-dimensional cross-correlation), *mapa de bordes* (edge map), *diferencia finita* (finite difference), *mapa de características* (feature map), *campo receptivo* (receptive field)).
- Decimal separator: keep `.` in code/math/prose numbers (e.g., `0.01`, `0.03`).
- "stride" → `stride` (kept as in glossary, plural `strides`).
- "framework" → `framework` (standard in Spanish ML literature).
- "patch" → `parche` (in prose/math).
- "feature map" → `mapa de características`.
- "throughput" → `throughput` (kept as in glossary).

=== NOTES ===
- 24a25ec4: math tokens reordered due to Spanish noun-before-modifier word order (`salida $\mathbf{Y}$ de $2 \times 2$`).
