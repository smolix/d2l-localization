<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u03 lang=es rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:2294243a kind=heading-->
## Exercises
<!--tr:2294243a-->
## Ejercicios

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
1. [code] **Núcleos diseñados a mano.** Diseñar algunos núcleos manualmente.
    1. Dado un vector de dirección $\mathbf{v} = (v_1, v_2)$, deducir un
       núcleo de detección de bordes que detecte los bordes ortogonales a
       $\mathbf{v}$, es decir, los bordes en la dirección $(v_2, -v_1)$.
    1. Deducir un operador de diferencias finitas para la segunda derivada.
       ¿Qué estructuras de las imágenes responden con más fuerza a él?
    1. ¿Cuál es el tamaño mínimo de un núcleo para obtener una derivada de
       orden $d$?
    1. ¿Cómo se diseñaría un núcleo de desenfoque? ¿Por qué podría
       interesar usar un núcleo así?
    1. Cargar una fotografía, convertirla a un solo canal y reducir su
       resolución a unos $100 \times 100$ píxeles (`corr2d` usa bucles de
       Python). Aplicar el detector de bordes, el núcleo de desenfoque y el
       núcleo de realce «el doble de la identidad menos el desenfoque», y
       mostrar cada resultado junto al original. ¿Qué muestra el mapa de
       bordes de una fotografía que no podía mostrar el mapa de bordes de la
       banda negra?

    *Adaptado de Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    cuaderno ConvolutionalNetworks.*

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
2. [code] **La convolución como matriz en banda.** La matriz de parches de
   :numref:`subsec_conv_matmul` despliega la entrada. La construcción
   opuesta despliega el núcleo. Sean $\mathbf{X}$ una entrada de
   $4 \times 4$, $\mathbf{K}$ un núcleo de $3 \times 3$ y
   $\textrm{vec}(\cdot)$ el aplanamiento fila a fila.
    1. Construir la matriz de $4 \times 16$, $\mathbf{W}$, tal que
       $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$,
       donde $\mathbf{Y}$ es la correlación cruzada de $\mathbf{X}$ con
       $\mathbf{K}$. ¿Cuántos elementos no nulos tiene cada fila y cuántos
       valores distintos aparecen en $\mathbf{W}$?
    1. Construir $\mathbf{W}$ en código a partir de `K` y confirmar que el
       producto reproduce `corr2d(X, K)`.
    1. Para una entrada de $n \times n$ y un núcleo de $k \times k$, contar
       los elementos de $\mathbf{W}$ almacenada como array denso, los
       elementos no nulos de $\mathbf{W}$ y los elementos de la matriz de
       parches. ¿Cómo crece cada recuento con $n$?
    1. Mantener el patrón de dispersidad de $\mathbf{W}$, pero dar a cada
       fila sus propios $k^2$ valores. El resultado es una capa conectada
       localmente: el tensor de cuarto orden $\mathsf{V}$ de
       :numref:`sec_why-conv` en
       $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$,
       restringido a una ventana de $k \times k$ de desplazamientos
       $(a, b)$ antes de que la equivarianza a traslaciones vincule los
       pesos. Para una entrada de $n \times n$, contar los parámetros de
       esta capa, los de una capa totalmente conectada con las mismas
       formas de entrada y salida, y los de la convolución.

    *Adaptado de CMU 11-785,
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
3. [code] **Campo receptivo efectivo.** ● El campo receptivo teórico $r$
   de esta sección acota qué entradas *pueden* influir en una salida; el
   campo receptivo efectivo mide cuánto influyen en realidad
   :cite:`Luo.Li.Urtasun.ea.2016`. Construir una pila de $L$ capas
   convolucionales de $3 \times 3$ de un solo canal, sin relleno, sin sesgo
   y sin no linealidad, sobre una entrada de $32 \times 32$, usando la capa
   integrada de :numref:`subsec_learning_kernel`. Muestrear cada elemento de
   los núcleos de forma independiente a partir de una distribución normal
   estándar.
    1. Para $L = 3$ y una sola muestra de los núcleos, retropropagar un
       gradiente unitario desde una unidad de salida cercana al centro de la
       última capa hasta la entrada y representar la magnitud del gradiente
       respecto a la entrada. Comparar su soporte con el campo receptivo
       teórico de $7 \times 7$ e indicar dónde es mayor la magnitud.
    1. Promediar el cuadrado del gradiente respecto a la entrada sobre 500
       muestras de los núcleos, para $L = 3$ y para $L = 10$. Sumar cada
       mapa promediado a lo largo de sus filas y normalizar el perfil de
       columnas resultante para que sume uno. ¿Cómo cambia con la
       profundidad el cociente entre la desviación estándar del perfil y el
       lado teórico $2L + 1$?
    1. :numref:`fig_conv_rf_paths` cuenta los caminos para $L = 3$ en una
       dimensión. Demostrar que, para $L$ general, el valor esperado del
       cuadrado del gradiente respecto a la entrada, sumado sobre las filas,
       es proporcional a la convolución de $L$ copias del vector
       $(1, 1, 1)$. Deducir cómo crece con $L$ la desviación estándar del
       perfil normalizado.

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
4. [code] **Pasada hacia atrás de una convolución.** ● Sean $\mathbf{Y}$ la
   correlación cruzada de un solo canal de una entrada $\mathbf{X}$ con un
   núcleo $\mathbf{K}$, tal como la calcula `corr2d`, y $\ell$ una pérdida
   escalar que depende de $\mathbf{X}$ y $\mathbf{K}$ solo a través de
   $\mathbf{Y}$.
    1. Deducir $\partial \ell/\partial \mathbf{K}$ y
       $\partial \ell/\partial \mathbf{X}$ en función de
       $\partial \ell/\partial \mathbf{Y}$, al estilo de las ecuaciones de
       retropropagación de :numref:`sec_backprop`.
    1. Implementar la expresión obtenida para
       $\partial \ell/\partial \mathbf{K}$ sin diferenciación automática
       para la pérdida $\ell = \sum_{i,j} G_{ij} Y_{ij}$, donde $\mathbf{G}$
       es una matriz aleatoria fija con la forma de $\mathbf{Y}$. Compararla
       con un gradiente por diferencias finitas centradas con tamaño de paso
       $10^{-2}$ sobre unos `X` y `K` aleatorios pequeños, e indicar el mayor
       error relativo.
    1. Comprobar la expresión obtenida para $\partial \ell/\partial \mathbf{K}$
       frente al gradiente del núcleo que calcula la diferenciación
       automática en el experimento de :numref:`subsec_learning_kernel`.

    *Adaptado de Michael Nielsen,
    [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html),
    capítulo 6, y de Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    cuaderno ConvolutionalNetworks.*

<!--en:657a396a kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/65)
<!--tr:657a396a-->
[Discusiones](https://d2l.discourse.group/t/65)

<!--en:63c44b48 kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/66)
<!--tr:63c44b48-->
[Discusiones](https://d2l.discourse.group/t/66)

<!--en:00b73d0b kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/271)
<!--tr:00b73d0b-->
[Discusiones](https://d2l.discourse.group/t/271)

<!--en:b79af65b kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/17996)
<!--tr:b79af65b-->
[Discusiones](https://d2l.discourse.group/t/17996)

=== DECISIONS ===
- Exercise instructions use the infinitive as imperative («Diseñar…», «Deducir…», «Comparar…»), the usual form in Spanish textbooks; questions addressed to the reader become impersonal («¿Cómo se diseñaría…?»).
- "derive" → «deducir»; "show that" → «demostrar que»; "draw" (a random sample) → «muestrear»; "a draw" → «una muestra».
- "standard normal distribution" → «distribución normal estándar» (the fixed statistical term; «distribución gaussiana» for "Gaussian distribution" otherwise).
- "blur kernel" → «núcleo de desenfoque»; "sharpening kernel" → «núcleo de realce»; "downsample" (an image) → «reducir su resolución».
- "locally connected layer" → «capa conectada localmente»; "sparsity pattern" → «patrón de dispersidad» (glossary).
- Attribution lines: «Adaptado de …»; titles of external courses, assignments and books stay in English ("Assignment 2", "Homework 2 Part 1", "Neural Networks and Deep Learning"); "notebook" → «cuaderno»; "Chapter 6" → «capítulo 6».
- "[code]" exercise tags and the ● marker are copied unchanged.
- Forum links "Discussions" → «Discusiones».

=== NOTES ===
- 6ab5b67c: the glossary warning for "Neural Networks" → «red neuronal» is a false match: it is part of the book title "Neural Networks and Deep Learning" (Nielsen), kept in English as the link text.
