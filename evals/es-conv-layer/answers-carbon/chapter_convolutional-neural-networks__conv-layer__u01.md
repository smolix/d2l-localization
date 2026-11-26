<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u01 lang=es rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:33d3b96b kind=heading-->
# Convolutions for Images
<!--tr:33d3b96b-->
# Convoluciones para imágenes

<!--en:852f5868 kind=paragraph-->
In :numref:`sec_why-conv`, we derived the convolutional layer on paper from two
geometric principles: translation equivariance and locality. Now we see what
that formula actually computes on an image, and whether gradient descent can
discover useful kernels automatically from data.
<!--tr:852f5868-->
En la :numref:`sec_why-conv`, dedujimos sobre el papel la capa convolucional a partir de dos principios geométricos: la equivarianza a traslaciones y la localidad. Ahora veremos qué calcula realmente esa fórmula sobre una imagen y si el descenso de gradiente puede descubrir núcleos útiles de forma automática a partir de los datos.

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
Consideremos la localización de bordes verticales allí donde una región clara se encuentra con una oscura. Restar a cada píxel su vecino de la derecha ($[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$) produce cero dentro de las regiones uniformes, $+1$ donde el blanco pasa al negro y $-1$ donde el negro vuelve al blanco. Al barrer el detector de dos pesos $(1, -1)$ por todas las posiciones válidas de la imagen se obtiene un *mapa de bordes*. En esta sección implementamos la correlación cruzada bidimensional, ponemos a prueba nuestro detector de bordes diseñado a mano, dejamos que el descenso de gradiente *aprenda* $(1, -1)$ directamente a partir de los datos, mostramos cómo una convolución se convierte en una única multiplicación de matrices (*im2col*) y deducimos cómo al apilar capas se amplía el campo receptivo de una unidad.

<!--en:47958ed8 kind=heading-->
## Cross-Correlation
<!--tr:47958ed8-->
## Correlación cruzada

<!--en:5e4296fe kind=heading-->
### The Operation
<!--tr:5e4296fe-->
### La operación

<!--en:ae7e906d kind=paragraph-->
Leaving channels aside for the moment, let the input $\mathbf{X}$ be an
$n_\textrm{h} \times n_\textrm{w}$ matrix and the kernel $\mathbf{K}$ be a
$k_\textrm{h} \times k_\textrm{w}$ matrix. The kernel defines a sliding
*window* of shape $k_\textrm{h} \times k_\textrm{w}$. Placing the window's
upper-left corner at $(i, j)$, we multiply the covered input patch elementwise
with $\mathbf{K}$ and sum the products to produce the output entry at $(i, j)$:
<!--tr:ae7e906d-->
Dejando los canales de lado por el momento, sean la entrada $\mathbf{X}$ una matriz de $n_\textrm{h} \times n_\textrm{w}$ y el núcleo $\mathbf{K}$ una matriz de $k_\textrm{h} \times k_\textrm{w}$. El núcleo define una *ventana* deslizante de forma $k_\textrm{h} \times k_\textrm{w}$. Al situar la esquina superior izquierda de la ventana en $(i, j)$, multiplicamos el parche de entrada cubierto con $\mathbf{K}$ elemento a elemento y sumamos los productos para obtener el elemento de salida en $(i, j)$:

<!--en:3688b38b kind=paragraph-->
This is the *two-dimensional cross-correlation* of $\mathbf{X}$ with
$\mathbf{K}$, written $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ with
zero-based indices matching code. Whereas :numref:`sec_why-conv` centered
offsets symmetrically from $-\Delta$ to $\Delta$, anchoring at the upper-left
corner differs only by a fixed shift of $\Delta$ while naturally handling
even-sized kernels (such as a $2 \times 2$ kernel or our $1 \times 2$ edge
detector) that have no center pixel.
<!--tr:3688b38b-->
Esta operación es la *correlación cruzada bidimensional* de $\mathbf{X}$ con $\mathbf{K}$, que escribimos $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ usando índices que empiezan en cero, tal como en el código. Mientras que en la :numref:`sec_why-conv` centramos los desplazamientos simétricamente de $-\Delta$ a $\Delta$, anclar la ventana en la esquina superior izquierda solo difiere en un desplazamiento fijo de $\Delta$ y permite tratar de forma natural núcleos de tamaño par (como un núcleo de $2 \times 2$ o nuestro detector de bordes de $1 \times 2$), que carecen de píxel central.

<!--en:fa1b7da9 kind=caption-->
![Two-dimensional cross-correlation. The shaded portions are the first output element and the input and kernel elements used to compute it: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
<!--tr:fa1b7da9-->
![Correlación cruzada bidimensional. Las celdas sombreadas corresponden al primer elemento de la salida y a los elementos de la entrada y del núcleo empleados para calcularlo: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)

<!--en:2f8e5deb kind=paragraph-->
In :numref:`fig_correlation`, we slide a $2 \times 2$ kernel across a
$3 \times 3$ input one pixel at a time, left to right and top to bottom. The
window fits cleanly at four positions, yielding the $2 \times 2$ output:
<!--tr:2f8e5deb-->
En la :numref:`fig_correlation`, deslizamos un núcleo de $2 \times 2$ sobre una entrada de $3 \times 3$ píxel a píxel, de izquierda a derecha y de arriba abajo. La ventana encaja exactamente en cuatro posiciones, lo que da lugar a la salida de $2 \times 2$:

<!--en:4741e012 kind=paragraph-->
Notice that the output shrinks slightly along each axis because the kernel must
stay wholly inside the input. Fitting a width-$k$ window into length $n$ leaves
$n - k + 1$ valid positions, so the output shape is
<!--tr:4741e012-->
Obsérvese que la salida se reduce ligeramente a lo largo de cada eje porque el núcleo debe permanecer por completo dentro de la entrada. Encajar una ventana de anchura $k$ en una longitud $n$ deja $n - k + 1$ posiciones válidas, de modo que la forma de la salida es

<!--en:12d81c14 kind=paragraph-->
Thus a $3 \times 3$ input with a $2 \times 2$ kernel yields $2 \times 2$, a
$6 \times 8$ image with a $1 \times 2$ kernel yields $6 \times 7$, and a
$28 \times 28$ Fashion-MNIST image with a $5 \times 5$ kernel yields
$24 \times 24$. Whenever we want to preserve spatial dimensions, we surround the
input with a border of zeros (*padding*), which we examine alongside strides and
dilation in :numref:`sec_padding`.
<!--tr:12d81c14-->
Así, una entrada de $3 \times 3$ con un núcleo de $2 \times 2$ produce $2 \times 2$, una imagen de $6 \times 8$ con un núcleo de $1 \times 2$ produce $6 \times 7$ y una imagen de Fashion-MNIST de $28 \times 28$ con un núcleo de $5 \times 5$ produce $24 \times 24$. Siempre que queramos conservar las dimensiones espaciales, rodearemos la entrada con un borde de ceros (*relleno*), técnica que examinaremos junto con los strides y la dilatación en la :numref:`sec_padding`.

<!--en:869d223f kind=heading-->
### Implementation
<!--tr:869d223f-->
### Implementación

<!--en:6a869f7c kind=paragraph-->
We implement :eqref:`eq_corr2d` directly in `corr2d` using a pair of loops over
every output coordinate.
<!--tr:6a869f7c-->
Implementamos :eqref:`eq_corr2d` directamente en `corr2d` mediante un par de bucles que recorren cada coordenada de la salida.

<!--en:81e1f383 kind=paragraph-->
To verify `corr2d` against our hand calculation in :numref:`fig_correlation`,
we construct the $3 \times 3$ input tensor `X` and $2 \times 2$ kernel `K` and
evaluate `corr2d(X, K)`.
<!--tr:81e1f383-->
Para verificar `corr2d` frente a nuestro cálculo manual de la :numref:`fig_correlation`, construimos el tensor de entrada `X` de $3 \times 3$ y el núcleo `K` de $2 \times 2$, y evaluamos `corr2d(X, K)`.

<!--en:f567e1b2 kind=paragraph-->
The output reproduces :numref:`fig_correlation` entry for entry (and as shown in
:numref:`sec_why-conv`, strict mathematical convolution with a flipped kernel
$\mathbf{K}'$ learns the exact same family of layers, so we follow standard
deep learning terminology and call cross-correlation "convolution").
<!--tr:f567e1b2-->
La salida reproduce la :numref:`fig_correlation` elemento por elemento (y, como se mostró en la :numref:`sec_why-conv`, la convolución matemática estricta con un núcleo invertido $\mathbf{K}'$ aprende exactamente la misma familia de capas, por lo que seguimos la terminología estándar del aprendizaje profundo y llamamos "convolución" a la correlación cruzada).

<!--en:bfe11c62 kind=heading-->
## Convolutional Layers
<!--tr:bfe11c62-->
## Capas convolucionales

<!--en:2cdc020d kind=heading-->
### The Layer
<!--tr:2cdc020d-->
### La capa

<!--en:c6371ece kind=paragraph-->
A two-dimensional convolutional layer cross-correlates its input with a learned
kernel and adds a scalar bias. Just like our MLP weights in
:numref:`chap_perceptrons`, we initialize the kernel from
$\mathcal{N}(0, 0.01^2)$ and the bias to zero. Below, `Conv2D` registers both
parameters and invokes `corr2d` in its forward pass; a layer with a kernel of
height $h$ and width $w$ is called an $h \times w$ convolutional layer.
<!--tr:c6371ece-->
Una capa convolucional bidimensional calcula la correlación cruzada de su entrada con un núcleo aprendido y le suma un sesgo escalar. Al igual que con los pesos del perceptrón multicapa en el :numref:`chap_perceptrons`, inicializamos el núcleo a partir de $\mathcal{N}(0, 0.01^2)$ y el sesgo a cero. A continuación, `Conv2D` registra ambos parámetros e invoca `corr2d` en su pasada hacia adelante; una capa cuyo núcleo tiene altura $h$ y anchura $w$ se denomina capa convolucional de $h \times w$.

<!--en:29cd9061 kind=paragraph-->
Unlike a dense layer, a convolutional layer is not locked to a fixed input
resolution because its parameters depend only on the kernel size. The exact same
$2 \times 2$ layer (5 parameters total) processes both our $3 \times 3$ tensor
`X` and a $6 \times 8$ image, returning outputs of shapes $2 \times 2$ and
$5 \times 7$:
<!--tr:29cd9061-->
A diferencia de una capa densa, una capa convolucional no está sujeta a una resolución de entrada fija, ya que sus parámetros solo dependen del tamaño del núcleo. Exactamente la misma capa de $2 \times 2$ (5 parámetros en total) procesa tanto nuestro tensor `X` de $3 \times 3$ como una imagen de $6 \times 8$, devolviendo salidas de formas $2 \times 2$ y $5 \times 7$:

<!--en:c9e5aa07 kind=heading-->
### Edge Detection
<!--tr:c9e5aa07-->
### Detección de bordes

<!--en:3d9a357b kind=paragraph-->
We now test the vertical edge detector from the introduction on a
$6 \times 8$ synthetic image whose middle four columns are black (`0`) and outer
columns are white (`1`).
<!--tr:3d9a357b-->
Probemos ahora el detector de bordes verticales de la introducción sobre una imagen sintética de $6 \times 8$ cuyas cuatro columnas centrales son negras (`0`) y cuyas columnas exteriores son blancas (`1`).

<!--en:68513ff0 kind=paragraph-->
We define the $1 \times 2$ kernel
<!--tr:68513ff0-->
Definimos el núcleo de $1 \times 2$

<!--en:eb45803b kind=paragraph-->
At every position $(i, j)$, this computes the horizontal *finite difference*
$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, a discrete approximation (up to sign)
of the horizontal derivative of brightness, which vanishes across flat regions
and spikes wherever intensity jumps.
<!--tr:eb45803b-->
En cada posición $(i, j)$, esto calcula la *diferencia finita* horizontal $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, una aproximación discreta (salvo signo) de la derivada horizontal del brillo, que se anula en las regiones uniformes y presenta picos allí donde la intensidad salta.

<!--en:e810b3a9 kind=paragraph-->
Cross-correlating the $6 \times 8$ stripe image `X` with the $1 \times 2$
horizontal-difference kernel `K` via `corr2d(X, K)` produces the $6 \times 7$
edge map `Y`:
<!--tr:e810b3a9-->
Aplicar la correlación cruzada a la imagen de franja `X` de $6 \times 8$ con el núcleo de diferencias horizontales `K` de $1 \times 2$ mediante `corr2d(X, K)` produce el mapa de bordes `Y` de $6 \times 7$:

<!--en:568358e4 kind=paragraph-->
Sure enough, `Y` fires `+1` along the white-to-black boundary, `-1` along the
black-to-white boundary, and `0` everywhere else. Applying the exact same kernel
to the transposed image (turning the vertical stripe into a horizontal one)
yields all zeros:
<!--tr:568358e4-->
En efecto, `Y` se activa con `+1` a lo largo de la frontera de blanco a negro, con `-1` a lo largo de la frontera de negro a blanco y vale `0` en el resto. Al aplicar exactamente el mismo núcleo a la imagen transpuesta (convirtiendo la franja vertical en una horizontal) se obtienen únicamente ceros:

<!--en:71c591a4 kind=paragraph-->
Clearly `[1, -1]` responds strictly to vertical edges. Detecting horizontal
edges, diagonals, corners, or textures requires different kernels altogether.
<!--tr:71c591a4-->
Queda claro que `[1, -1]` responde estrictamente a los bordes verticales. Detectar bordes horizontales, diagonales, esquinas o texturas requiere núcleos completamente distintos.

<!--en:f0af4887 kind=heading-->
### Learning a Kernel
<!--tr:f0af4887-->
### Aprendizaje de un núcleo

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
Diseñar `[1, -1]` a mano es sencillo cuando conocemos la característica exacta que buscamos, pero ningún ingeniero puede especificar manualmente miles de filtros intermedios a lo largo de una red profunda. ¿Puede el descenso de gradiente recuperar el núcleo `[1, -1]` únicamente a partir del par entrada-salida `(X, Y)`? Lo comprobamos empleando la capa convolucional 2D integrada de cada framework (sin sesgo y con los pesos inicializados a partir de $\mathcal{N}(0, 0.01^2)$), minimizando el error cuadrático sumado sobre las 42 entradas de la salida durante 10 pasos de gradiente con una tasa de aprendizaje de $0.03$. Dado que las capas integradas esperan tensores 4D que incluyan los ejes de lote y de canal (:numref:`sec_channels`), cambiamos la forma de `X` y `Y` en consecuencia.

<!--en:c29c5d61 kind=paragraph-->
Initially the loss sits near 12 (the near-zero random kernel misses all 12
$\pm 1$ edge pixels by roughly 1), and within just nine steps it drops below
0.01. Inspecting the learned kernel confirms what it found:
<!--tr:c29c5d61-->
Inicialmente la pérdida se sitúa cerca de 12 (el núcleo aleatorio casi nulo falla en los 12 píxeles de borde $\pm 1$ por aproximadamente 1) y, en apenas nueve pasos, cae por debajo de 0.01. Al inspeccionar el núcleo aprendido confirmamos lo que ha encontrado:

<!--en:67d25b0b kind=paragraph-->
Both entries land within roughly 0.02 of `[1, -1]`. Nothing in this loop knew
anything about edges: given a loss signal, gradient descent discovered the
filter on its own. Here we supervised the single layer directly with `Y`; in a
deep CNN, backpropagation learns every layer's kernels end to end from the final
task loss alone.
<!--tr:67d25b0b-->
Ambas entradas quedan a una distancia aproximada de 0.02 respecto de `[1, -1]`. Nada en este bucle sabía qué es un borde: dada una señal de pérdida, el descenso de gradiente descubrió el filtro por sí solo. Aquí supervisamos directamente una única capa con `Y`; en una red neuronal convolucional profunda, la retropropagación aprende de extremo a extremo los núcleos de todas las capas únicamente a partir de la pérdida final de la tarea.

<!--en:91b0010d kind=slide_text-->
Convolutions for Images<br>**2D cross-correlation · edge detection · learned kernels · im2col · receptive fields**
<!--tr:91b0010d-->
Convoluciones para imágenes<br>**Correlación cruzada 2D · detección de bordes · núcleos aprendidos · im2col · campos receptivos**

<!--en:1d94295a kind=slide_title-->
::: {.slide title="Detecting Local Edges Across an Image"}
<!--tr:1d94295a-->
::: {.slide title="Detección de bordes locales en una imagen"}

<!--en:687ed302 kind=kicker-->
[Motivation & Core Mental Model]{.kicker}
<!--tr:687ed302-->
[Motivación y modelo mental central]{.kicker}

<!--en:05fceed0 kind=slide_text-->
- **Synthetic $6 \times 8$ test image:** White outer columns (`1`) surrounding a 4-column black vertical band (`0`).
<!--tr:05fceed0-->
- **Imagen sintética de prueba de $6 \times 8$:** Columnas exteriores blancas (`1`) que rodean una franja vertical negra de 4 columnas (`0`).

<!--en:71e700b3 kind=slide_text-->
- **Horizontal finite difference:** Sweeping $\mathbf{K} = [1, -1]$ computes $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, firing `+1` at white-to-black and `-1` at black-to-white edges.
<!--tr:71e700b3-->
- **Diferencia finita horizontal:** Barrer $\mathbf{K} = [1, -1]$ calcula $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, activándose con `+1` en bordes de blanco a negro y con `-1` de negro a blanco.

<!--en:957dc033 kind=slide_text-->
- **Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent.
<!--tr:957dc033-->
- **Aprendizaje frente a diseño manual:** Las redes neuronales convolucionales profundas descubren miles de estos núcleos automáticamente mediante descenso de gradiente.

<!--en:cb1045e1 kind=slide_title-->
::: {.slide title="The 2D Cross-Correlation Operation"}
<!--tr:cb1045e1-->
::: {.slide title="La operación de correlación cruzada 2D"}

<!--en:17bcece5 kind=kicker-->
[Sliding-Window Inner Products]{.kicker}
<!--tr:17bcece5-->
[Productos internos con ventana deslizante]{.kicker}

<!--en:8d45aae8 kind=slide_text-->
Given an $n_\textrm{h} \times n_\textrm{w}$ input matrix $\mathbf{X}$ and a $k_\textrm{h} \times k_\textrm{w}$ kernel $\mathbf{K}$, the 2D cross-correlation $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ places the window's top-left corner at $(i, j)$ and sums elementwise products:
<!--tr:8d45aae8-->
Dadas una matriz de entrada $\mathbf{X}$ de $n_\textrm{h} \times n_\textrm{w}$ y un núcleo $\mathbf{K}$ de $k_\textrm{h} \times k_\textrm{w}$, la correlación cruzada 2D $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ sitúa la esquina superior izquierda de la ventana en $(i, j)$ y suma los productos elemento a elemento:

<!--en:e484380c kind=slide_text-->
- **Top-left anchor:** Zero-based indices $a, b \ge 0$ match array indexing in code and handle even-sized kernels (like $2 \times 2$ or $1 \times 2$) cleanly.
<!--tr:e484380c-->
- **Anclaje superior izquierdo:** Los índices desde cero $a, b \ge 0$ coinciden con la indexación en código y tratan limpiamente núcleos de tamaño par (como $2 \times 2$ o $1 \times 2$).

<!--en:137c0980 kind=slide_text-->
- **Shaded top-left output $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.
<!--tr:137c0980-->
- **Salida superior izquierda sombreada $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.

<!--en:575c4aea kind=slide_title-->
::: {.slide title="Valid Window Positions and Output Spatial Shape"}
<!--tr:575c4aea-->
::: {.slide title="Posiciones válidas de ventana y forma de salida"}

<!--en:7eda4531 kind=kicker-->
[Why Unpadded Convolutions Shrink the Grid]{.kicker}
<!--tr:7eda4531-->
[Por qué las convoluciones sin relleno reducen la cuadrícula]{.kicker}

<!--en:891243ae kind=slide_text-->
- **Valid window placements:** A $k_\textrm{h} \times k_\textrm{w}$ window fits inside an $n_\textrm{h} \times n_\textrm{w}$ grid at $n - k + 1$ positions per axis.
<!--tr:891243ae-->
- **Posiciones válidas de la ventana:** Una ventana de $k_\textrm{h} \times k_\textrm{w}$ cabe en una cuadrícula de $n_\textrm{h} \times n_\textrm{w}$ en $n - k + 1$ posiciones por eje.

<!--en:55cc8c32 kind=slide_text-->
- **Shape examples:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.
<!--tr:55cc8c32-->
- **Ejemplos de formas:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.

<!--en:1e3463da kind=slide_title-->
::: {.slide title="Implementing 2D Cross-Correlation in Code"}
<!--tr:1e3463da-->
::: {.slide title="Implementación en código de la correlación cruzada 2D"}

<!--en:07566992 kind=kicker-->
[Direct Nested-Loop Implementation]{.kicker}
<!--tr:07566992-->
[Implementación directa con bucles anidados]{.kicker}

<!--en:202aa5d3 kind=slide_text-->
The function `corr2d` allocates the $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ output tensor and computes each entry as the sum of `X[i:i+h, j:j+w] * K`:
<!--tr:202aa5d3-->
La función `corr2d` reserva el tensor de salida de $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ y calcula cada entrada como la suma de `X[i:i+h, j:j+w] * K`:

<!--en:bef130ba kind=slide_text-->
Each iteration slices a $k_\textrm{h} \times k_\textrm{w}$ sub-window of `X`, multiplies elementwise with `K`, and reduces via `sum`.
<!--tr:bef130ba-->
Cada iteración toma como rebanada una subventana de $k_\textrm{h} \times k_\textrm{w}$ de `X`, la multiplica elemento a elemento con `K` y reduce el resultado mediante `sum`.

<!--en:4ee9f6b1 kind=slide_title-->
::: {.slide title="Verifying corr2d and the Flipped-Kernel Equivalence"}
<!--tr:4ee9f6b1-->
::: {.slide title="Verificación de corr2d y núcleo invertido equivalente"}

<!--en:2b0be423 kind=kicker-->
[Cross-Correlation vs. Mathematical Convolution]{.kicker}
<!--tr:2b0be423-->
[Correlación cruzada frente a convolución matemática]{.kicker}

<!--en:74b9b745 kind=slide_text-->
- **180° flipped kernel $\mathbf{K}'$:** Cross-correlating with $\mathbf{K}$ equals convolving with $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.
<!--tr:74b9b745-->
- **Núcleo invertido 180° $\mathbf{K}'$:** La correlación cruzada con $\mathbf{K}$ equivale a la convolución con $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.

<!--en:5c8af7cd kind=slide_text-->
- **Identical learned class:** Frameworks compute cross-correlation and call it convolution.
<!--tr:5c8af7cd-->
- **Clase aprendida idéntica:** Los frameworks calculan la correlación cruzada y la llaman convolución.

<!--en:a69e29b5 kind=slide_title-->
::: {.slide title="Building a 2D Convolutional Layer Module"}
<!--tr:a69e29b5-->
::: {.slide title="Construcción de un módulo de capa convolucional 2D"}

<!--en:55763603 kind=kicker-->
[Learnable Kernel Weights Plus Scalar Bias]{.kicker}
<!--tr:55763603-->
[Pesos aprendibles del núcleo más sesgo escalar]{.kicker}

<!--en:5b8bad24 kind=slide_text-->
A convolutional layer wraps `corr2d` with a learnable $k_\textrm{h} \times k_\textrm{w}$ weight parameter (initialized from $\mathcal{N}(0, 0.01^2)$) and a scalar bias (initialized to $0$):
<!--tr:5b8bad24-->
Una capa convolucional envuelve `corr2d` con un parámetro de pesos aprendible de $k_\textrm{h} \times k_\textrm{w}$ (inicializado con $\mathcal{N}(0, 0.01^2)$) y un sesgo escalar (inicializado a $0$):

<!--en:c16d97d6 kind=slide_title-->
::: {.slide title="Resolution Independence of Convolutional Layers"}
<!--tr:c16d97d6-->
::: {.slide title="Independencia de la resolución en capas convolucionales"}

<!--en:a7d1e48f kind=kicker-->
[Fixed Parameters Across Arbitrary Input Sizes]{.kicker}
<!--tr:a7d1e48f-->
[Parámetros fijos para tamaños de entrada arbitrarios]{.kicker}

<!--en:53a2bce7 kind=slide_text-->
- **Decoupled from $H \times W$:** The exact same 5 parameters ($2 \times 2$ kernel + 1 bias) process $3 \times 3$, $6 \times 8$, or megapixel inputs without reshaping weights.
<!--tr:53a2bce7-->
- **Desacoplada de $H \times W$:** Exactamente los mismos 5 parámetros (núcleo de $2 \times 2$ + 1 sesgo) procesan entradas de $3 \times 3$, $6 \times 8$ o megapíxeles sin cambiar la forma de los pesos.

<!--en:017fec24 kind=slide_title-->
::: {.slide title="Constructing a Synthetic Vertical-Edge Image"}
<!--tr:017fec24-->
::: {.slide title="Construcción de una imagen sintética de bordes verticales"}

<!--en:896e641c kind=kicker-->
[A 6×8 Binary Test Pattern]{.kicker}
<!--tr:896e641c-->
[Un patrón binario de prueba de 6×8]{.kicker}

<!--en:0a4b456b kind=slide_text-->
We construct the $6 \times 8$ test image `X` with ones on the left and right borders and zeros across the middle four columns (`X[:, 2:6] = 0`):
<!--tr:0a4b456b-->
Construimos la imagen de prueba `X` de $6 \times 8$ con unos en los bordes izquierdo y derecho y ceros en las cuatro columnas centrales (`X[:, 2:6] = 0`):

<!--en:dfc7e3e3 kind=slide_text-->
- **Piecewise-constant columns:** Intensity is constant (`1` or `0`) within columns $0\text{--}1$, $2\text{--}5$, and $6\text{--}7$, jumping only between columns $1 \to 2$ and $5 \to 6$.
<!--tr:dfc7e3e3-->
- **Columnas constantes a trozos:** La intensidad es constante (`1` o `0`) en las columnas $0\text{--}1$, $2\text{--}5$ y $6\text{--}7$, y solo salta entre las columnas $1 \to 2$ y $5 \to 6$.

<!--en:3ce31cbe kind=slide_title-->
::: {.slide title="Applying the Horizontal Finite-Difference Kernel"}
<!--tr:3ce31cbe-->
::: {.slide title="Aplicación del núcleo de diferencia finita horizontal"}

<!--en:b37a1029 kind=kicker-->
[Detecting Vertical Edges with K = [1, -1]]{.kicker}
<!--tr:b37a1029-->
[Detección de bordes verticales con K = [1, -1]]{.kicker}

<!--en:9be85019 kind=slide_text-->
Cross-correlating `X` with the $1 \times 2$ kernel $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ computes $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ at every horizontal pair:
<!--tr:9be85019-->
La correlación cruzada de `X` con el núcleo de $1 \times 2$, $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$, calcula $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ en cada par horizontal:

<!--en:f649cfae kind=slide_text-->
- **Signed edge map:** Column 1 fires `+1` (white-to-black transition), column 5 fires `-1` (black-to-white transition), and all constant interior columns evaluate to `0`.
<!--tr:f649cfae-->
- **Mapa de bordes con signo:** La columna 1 se activa con `+1` (transición de blanco a negro), la columna 5 con `-1` (transición de negro a blanco) y las columnas interiores constantes valen `0`.

<!--en:e6d155be kind=slide_title-->
::: {.slide title="Orientation Selectivity of Edge Kernels"}
<!--tr:e6d155be-->
::: {.slide title="Selectividad de orientación de los núcleos de bordes"}

<!--en:b525cde4 kind=kicker-->
[Why One Kernel Cannot Detect All Orientations]{.kicker}
<!--tr:b525cde4-->
[Por qué un solo núcleo no detecta todas las orientaciones]{.kicker}

<!--en:38159de6 kind=slide_text-->
- **Blind to horizontal edges:** Transposing `X` (`X.T`) makes horizontal neighbors identical, so $[1, -1]$ outputs all zeros.
<!--tr:38159de6-->
- **Ciego a los bordes horizontales:** Transponer `X` (`X.T`) hace idénticos a los vecinos horizontales, por lo que $[1, -1]$ produce solo ceros.

<!--en:f0f30f76 kind=slide_text-->
- **Filter banks (§{sec_channels}):** Vision layers learn parallel channels for horizontal, vertical, and diagonal orientations.
<!--tr:f0f30f76-->
- **Bancos de filtros (§{sec_channels}):** Las capas de visión aprenden canales paralelos para orientaciones horizontales, verticales y diagonales.

<!--en:b755e31b kind=slide_title-->
::: {.slide title="Learning an Edge Detector via Gradient Descent"}
<!--tr:b755e31b-->
::: {.slide title="Aprender un detector de bordes por descenso de gradiente"}

<!--en:86bbeea7 kind=kicker-->
[Supervising a Built-In Convolutional Layer on (X, Y)]{.kicker}
<!--tr:86bbeea7-->
[Supervisión de una capa convolucional integrada con (X, Y)]{.kicker}

<!--en:a569780c kind=slide_text-->
- **4D framework tensors:** Reshape `X` and `Y` to 4D tensors (one example, one channel) and minimize $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ at $\eta = 0.03$.
<!--tr:a569780c-->
- **Tensores 4D del framework:** Cambiamos la forma de `X` y `Y` a tensores 4D (un ejemplo, un canal) y minimizamos $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ con $\eta = 0.03$.

<!--en:2489ce6e kind=slide_title-->
::: {.slide title="Inspecting the Learned Kernel Weights"}
<!--tr:2489ce6e-->
::: {.slide title="Inspección de los pesos del núcleo aprendido"}

<!--en:7e4ef44b kind=kicker-->
[Why Initial Loss Is 12 and Converges to [1, -1]]{.kicker}
<!--tr:7e4ef44b-->
[Por qué la pérdida inicial es 12 y converge a [1, -1]]{.kicker}

<!--en:9cd1c645 kind=slide_text-->
- **Why loss starts at $12.0$:** Near-zero weights predict $\hat{Y} \approx 0$, missing the twelve $\pm 1$ edge pixels in `Y` ($12 \times 1^2 = 12$).
<!--tr:9cd1c645-->
- **Por qué la pérdida empieza en $12.0$:** Pesos casi nulos predicen $\hat{Y} \approx 0$, fallando en los doce píxeles de borde $\pm 1$ de `Y` ($12 \times 1^2 = 12$).

<!--en:456670c5 kind=slide_text-->
- **Convergence in 9 steps:** SGD recovers $[0.99, -0.99]$ automatically from data.
<!--tr:456670c5-->
- **Convergencia en 9 pasos:** El descenso de gradiente estocástico recupera $[0.99, -0.99]$ automáticamente a partir de los datos.

=== DECISIONS ===
- Style & register: first-person plural inclusive ("dedujimos", "implementamos") in exposition; formal pan-Hispanic academic second-person ("usted": "diseñe", "ponga a prueba") in exercises and concept checks.
- Capitalization: Spanish sentence case (only initial word and proper nouns capitalized) in headings, slide titles, and kickers, with no trailing period.
- Numbers & quotes: decimal point preserved as `.` (`0.01`, `0.03`, `12.0`); straight double quotes `"..."` preserved in prose.
- Cross-references: Spanish definite articles precede `:numref:` according to referent gender (`la :numref:` for sections/subsections/figures, `el :numref:` for chapters).
- "edge" / "edge detection" / "edge detector" / "edge map" → "borde" / "detección de bordes" / "detector de bordes" / "mapa de bordes"
- "patch" → "parche"
- "elementwise" → "elemento a elemento"
- "dense layer" → "capa densa"
- "built-in" (layer) → "integrada"
- "hand-crafted" / "hand-crafting" → "diseñado a mano" / "diseñar ... a mano"
- "flipped kernel" → "núcleo invertido"
- "end to end" → "de extremo a extremo"
- "framework" → "framework" (plural "frameworks", kept distinct from RL "entorno")
- "stride" → "stride" (plural "strides", per proposed glossary entry)

=== NOTES ===
- 8d45aae8: math tokens reordered ("matriz de entrada $\mathbf{X}$ de $n_\textrm{h} \times n_\textrm{w}$" and "núcleo $\mathbf{K}$ de $k_\textrm{h} \times k_\textrm{w}$") as required by Spanish post-nominal modifier order.
