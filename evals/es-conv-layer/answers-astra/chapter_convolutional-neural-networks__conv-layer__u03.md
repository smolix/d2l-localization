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
1. [code] **Núcleos diseñados a mano.** Diseña algunos núcleos a mano.
    1. Dado un vector de dirección $\mathbf{v} = (v_1, v_2)$, deduce un núcleo que detecte los bordes ortogonales a $\mathbf{v}$, es decir, los bordes en la dirección $(v_2, -v_1)$.
    1. Deduce un operador de diferencias finitas para la segunda derivada. ¿Qué estructuras de una imagen producen la respuesta más intensa al aplicarlo?
    1. ¿Cuál es el tamaño mínimo de un núcleo para obtener una derivada de orden $d$?
    1. ¿Cómo diseñarías un núcleo de desenfoque? ¿Para qué querrías usarlo?
    1. Carga una fotografía, conviértela a un solo canal y reduce su resolución a unos $100 \times 100$ píxeles (`corr2d` usa bucles en Python). Aplica el detector de bordes, el núcleo de desenfoque y el núcleo de realce de nitidez «el doble de la identidad menos el desenfoque». Muestra cada resultado junto al original. ¿Qué revela el mapa de bordes de una fotografía que no podía mostrar el de la banda negra?

    *Adaptado del curso CS231n de Stanford, [tarea 2](https://cs231n.github.io/assignments2024/assignment2/), cuaderno ConvolutionalNetworks.*

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
2. [code] **La convolución como matriz en banda.** La matriz de parches de la :numref:`subsec_conv_matmul` despliega la entrada. La construcción opuesta despliega el núcleo. Sean $\mathbf{X}$ una entrada de $4 \times 4$, $\mathbf{K}$ un núcleo de $3 \times 3$ y $\textrm{vec}(\cdot)$ la operación de aplanar por filas.
    1. Construye una matriz de $4 \times 16$, denotada por $\mathbf{W}$, que satisfaga $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$, donde $\mathbf{Y}$ es la correlación cruzada de $\mathbf{X}$ con $\mathbf{K}$. ¿Cuántos elementos distintos de cero tiene cada fila? ¿Cuántos valores distintos aparecen en $\mathbf{W}$?
    1. Construye $\mathbf{W}$ en código a partir de `K` y comprueba que el producto reproduce `corr2d(X, K)`.
    1. Para una entrada de $n \times n$ y un núcleo de $k \times k$, cuenta los elementos de $\mathbf{W}$ si se almacena como un arreglo denso, los elementos distintos de cero de $\mathbf{W}$ y los elementos de la matriz de parches. ¿Cómo crece cada cantidad con $n$?
    1. Conserva el patrón de dispersidad de $\mathbf{W}$, pero asigna a cada fila sus propios $k^2$ valores. El resultado es una capa con conexiones locales, descrita por el tensor de cuarto orden $\mathsf{V}$ de la :numref:`sec_why-conv` en la expresión $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$, restringido a una ventana de tamaño $k \times k$ para los desplazamientos $(a, b)$ antes de imponer que los pesos se compartan por equivarianza a traslaciones. Para una entrada de $n \times n$, cuenta los parámetros de esta capa, los de una capa totalmente conectada con las mismas formas de entrada y salida, y los de la convolución.

    *Adaptado del curso 11-785 de CMU, [tarea 2, parte 1](https://deeplearning.cs.cmu.edu/S21/index.html).*

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
3. [code] **Campo receptivo efectivo.** ● El campo receptivo teórico $r$ que estudiamos en esta sección delimita qué entradas *pueden* influir en una salida. El campo receptivo efectivo mide cuánto influyen realmente en ella :cite:`Luo.Li.Urtasun.ea.2016`. Apila $L$ capas convolucionales de un solo canal y tamaño $3 \times 3$, sin relleno, sin sesgo y sin no linealidad, sobre una entrada de $32 \times 32$. Usa la capa que ofrece la biblioteca en la :numref:`subsec_learning_kernel`. Para cada elemento de cada núcleo, toma una muestra independiente de una distribución gaussiana estándar.
    1. Para $L = 3$ y una sola muestra del conjunto de núcleos, propaga hacia atrás un gradiente unitario desde una unidad de salida cercana al centro de la última capa hasta la entrada. Representa gráficamente la magnitud del gradiente respecto de la entrada. Compara su soporte con el campo receptivo teórico de $7 \times 7$ e indica dónde es mayor la magnitud.
    1. Calcula el gradiente respecto de la entrada, eleva sus elementos al cuadrado y toma la media sobre 500 muestras del conjunto de núcleos, tanto para $L = 3$ como para $L = 10$. En cada mapa promedio, suma las filas entre sí y normaliza el perfil por columnas resultante para que sus valores sumen uno. ¿Cómo cambia con la profundidad el cociente entre la desviación estándar del perfil y la longitud teórica del lado $2L + 1$?
    1. En la :numref:`fig_conv_rf_paths` se cuentan los caminos para $L = 3$ en una dimensión. Demuestra que, para cualquier $L$, el valor esperado del gradiente al cuadrado respecto de la entrada, sumado sobre las filas, es proporcional a la convolución de $L$ copias del vector $(1, 1, 1)$. Deduce cómo crece la desviación estándar del perfil normalizado en función de $L$.

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
4. [code] **Pasada hacia atrás de una convolución.** ● Sea $\mathbf{Y}$ la correlación cruzada de un solo canal de una entrada $\mathbf{X}$ con un núcleo $\mathbf{K}$, tal como la calcula `corr2d`. Sea $\ell$ una pérdida escalar que depende de $\mathbf{X}$ y de $\mathbf{K}$ solo a través de $\mathbf{Y}$.
    1. Deduce $\partial \ell/\partial \mathbf{K}$ y $\partial \ell/\partial \mathbf{X}$ en función de $\partial \ell/\partial \mathbf{Y}$ siguiendo el modelo de las ecuaciones de retropropagación de la :numref:`sec_backprop`.
    1. Implementa tu expresión de $\partial \ell/\partial \mathbf{K}$ sin diferenciación automática para la pérdida $\ell = \sum_{i,j} G_{ij} Y_{ij}$, donde $\mathbf{G}$ es una matriz aleatoria que se mantiene fija y tiene la misma forma que $\mathbf{Y}$. Compara el resultado con el gradiente calculado mediante diferencias finitas centradas con un tamaño de paso de $10^{-2}$. Usa una entrada `X` y un núcleo `K` pequeños y aleatorios, e indica el mayor error relativo.
    1. Contrasta tu expresión de $\partial \ell/\partial \mathbf{K}$ con el gradiente respecto del núcleo que se obtiene por diferenciación automática en el experimento de la :numref:`subsec_learning_kernel`.

    *Adaptado de Michael Nielsen, [Redes neuronales y aprendizaje profundo](http://neuralnetworksanddeeplearning.com/chap6.html), capítulo 6, y del curso CS231n de Stanford, [tarea 2](https://cs231n.github.io/assignments2024/assignment2/), cuaderno ConvolutionalNetworks.*

<!--en:657a396a kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/65)
<!--tr:657a396a-->
[Foro de discusión](https://d2l.discourse.group/t/65)

<!--en:63c44b48 kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/66)
<!--tr:63c44b48-->
[Foro de discusión](https://d2l.discourse.group/t/66)

<!--en:00b73d0b kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/271)
<!--tr:00b73d0b-->
[Foro de discusión](https://d2l.discourse.group/t/271)

<!--en:b79af65b kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/17996)
<!--tr:b79af65b-->
[Foro de discusión](https://d2l.discourse.group/t/17996)

=== DECISIONS ===
- «blur kernel» → «núcleo de desenfoque», «sharpening kernel» → «núcleo de realce de nitidez», «locally connected layer» → «capa con conexiones locales».
- «input gradient» → «gradiente respecto de la entrada», «support» → «soporte» en el sentido matemático.
