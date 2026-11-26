<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u03 lang=pt rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:2294243a kind=heading-->
## Exercises
<!--tr:2294243a-->
## Exercícios

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
1. [code] **Kernels projetados à mão.** Projete alguns kernels à mão.
    1. Dado um vetor de direção $\mathbf{v} = (v_1, v_2)$, deduza um kernel que detecte bordas ortogonais a $\mathbf{v}$, ou seja, bordas na direção $(v_2, -v_1)$.
    1. Deduza um operador de diferenças finitas para a segunda derivada. Quais estruturas de uma imagem produzem a resposta mais intensa a esse operador?
    1. Qual é o tamanho mínimo de um kernel para obter uma derivada de ordem $d$?
    1. Como você projetaria um kernel de desfoque? Por que seria interessante usá-lo?
    1. Carregue uma fotografia, converta-a para um único canal e reduza sua resolução por subamostragem para cerca de $100 \times 100$ pixels (`corr2d` usa laços em Python). Aplique o detector de bordas, o kernel de desfoque e o kernel de realce de nitidez “duas vezes a identidade menos o desfoque”. Exiba cada resultado ao lado da imagem original. O que o mapa de bordas de uma fotografia revela que o mapa de bordas da faixa preta não conseguia mostrar?

    *Adaptado da disciplina CS231n de Stanford, [atividade 2](https://cs231n.github.io/assignments2024/assignment2/), notebook ConvolutionalNetworks.*

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
2. [code] **Convolução como matriz de banda.** A matriz de trechos de :numref:`subsec_conv_matmul` desdobra a entrada. A construção inversa desdobra o kernel. Seja $\mathbf{X}$ uma entrada de forma $4 \times 4$, seja $\mathbf{K}$ um kernel de forma $3 \times 3$ e seja $\textrm{vec}(\cdot)$ a operação de converter uma matriz em vetor, linha por linha.
    1. Construa uma matriz de forma $4 \times 16$, denotada por $\mathbf{W}$, tal que $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$, em que $\mathbf{Y}$ é a correlação cruzada de $\mathbf{X}$ com $\mathbf{K}$. Quantos elementos não nulos há em cada linha e quantos valores distintos aparecem em $\mathbf{W}$?
    1. Construa $\mathbf{W}$ em código a partir de `K` e confirme que o produto reproduz `corr2d(X, K)`.
    1. Para uma entrada $n \times n$ e um kernel $k \times k$, conte os elementos de $\mathbf{W}$ quando armazenada como um array denso, os elementos não nulos de $\mathbf{W}$ e os elementos da matriz de trechos. Como cada contagem cresce em função de $n$?
    1. Mantenha o padrão de esparsidade de $\mathbf{W}$, mas atribua a cada linha seus próprios $k^2$ valores. O resultado é uma camada localmente conectada. Ela corresponde ao tensor de quarta ordem $\mathsf{V}$ de :numref:`sec_why-conv` na expressão $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$, restrito a uma janela $k \times k$ de deslocamentos $(a, b)$ antes que a equivariância à translação imponha o compartilhamento de pesos. Para uma entrada $n \times n$, conte os parâmetros dessa camada, de uma camada totalmente conectada com as mesmas formas de entrada e de saída e da convolução.

    *Adaptado da disciplina 11-785 da CMU, [lista de exercícios 2, parte 1](https://deeplearning.cs.cmu.edu/S21/index.html).*

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
3. [code] **Campo receptivo efetivo.** ● O campo receptivo teórico $r$ desta seção delimita quais entradas *podem* influenciar uma saída. O campo receptivo efetivo mede a influência que elas de fato exercem :cite:`Luo.Li.Urtasun.ea.2016`. Empilhe $L$ camadas convolucionais de um único canal com kernels $3 \times 3$, sem preenchimento, sem viés e sem não linearidade, sobre uma entrada $32 \times 32$. Use a camada do framework apresentada em :numref:`subsec_learning_kernel`. Sorteie cada elemento de cada kernel de forma independente a partir de uma distribuição gaussiana padrão.
    1. Para $L = 3$, sorteie os kernels uma única vez. Retropropague um gradiente unitário de uma unidade de saída próxima ao centro da última camada até a entrada e faça um gráfico da magnitude do gradiente em relação à entrada. Compare seu suporte com o campo receptivo teórico $7 \times 7$ e indique onde a magnitude é maior.
    1. Calcule a média do quadrado do gradiente em relação à entrada para 500 sorteios dos kernels, tanto para $L = 3$ quanto para $L = 10$. Em cada mapa médio, some as linhas e normalize o perfil resultante por coluna de modo que a soma de seus valores seja um. Como a razão entre o desvio padrão do perfil e o comprimento teórico do lado do campo receptivo, $2L + 1$, varia com a profundidade?
    1. A :numref:`fig_conv_rf_paths` mostra a contagem dos caminhos para $L = 3$ em uma dimensão. Mostre que, para $L$ qualquer, o valor esperado do quadrado do gradiente em relação à entrada, somado ao longo das linhas, é proporcional à convolução de $L$ cópias do vetor $(1, 1, 1)$. Deduza como o desvio padrão do perfil normalizado cresce com $L$.

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
4. [code] **Passagem reversa de uma convolução.** ● Seja $\mathbf{Y}$ a correlação cruzada de um único canal entre uma entrada $\mathbf{X}$ e um kernel $\mathbf{K}$, calculada por `corr2d`. Seja $\ell$ uma perda escalar que depende de $\mathbf{X}$ e de $\mathbf{K}$ apenas por meio de $\mathbf{Y}$.
    1. Deduza $\partial \ell/\partial \mathbf{K}$ e $\partial \ell/\partial \mathbf{X}$ em função de $\partial \ell/\partial \mathbf{Y}$, seguindo o modelo das equações de retropropagação de :numref:`sec_backprop`.
    1. Implemente sua expressão para $\partial \ell/\partial \mathbf{K}$ sem usar diferenciação automática, considerando a perda $\ell = \sum_{i,j} G_{ij} Y_{ij}$, em que $\mathbf{G}$ é uma matriz aleatória fixa com a mesma forma de $\mathbf{Y}$. Compare o resultado com o gradiente calculado por diferenças finitas centradas com tamanho do passo $10^{-2}$, usando uma entrada `X` e um kernel `K` pequenos e aleatórios. Informe o maior erro relativo.
    1. Verifique sua expressão para $\partial \ell/\partial \mathbf{K}$ comparando-a com o gradiente em relação ao kernel calculado por diferenciação automática no experimento de :numref:`subsec_learning_kernel`.

    *Adaptado de Michael Nielsen, [Redes neurais e aprendizado profundo](http://neuralnetworksanddeeplearning.com/chap6.html), capítulo 6, e da disciplina CS231n de Stanford, [atividade 2](https://cs231n.github.io/assignments2024/assignment2/), notebook ConvolutionalNetworks.*

<!--en:657a396a kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/65)
<!--tr:657a396a-->
[Discussões](https://d2l.discourse.group/t/65)

<!--en:63c44b48 kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/66)
<!--tr:63c44b48-->
[Discussões](https://d2l.discourse.group/t/66)

<!--en:00b73d0b kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/271)
<!--tr:00b73d0b-->
[Discussões](https://d2l.discourse.group/t/271)

<!--en:b79af65b kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/17996)
<!--tr:b79af65b-->
[Discussões](https://d2l.discourse.group/t/17996)

=== DECISIONS ===
- “blur kernel” → “kernel de desfoque”, “sharpening kernel” → “kernel de realce de nitidez”, “locally connected layer” → “camada localmente conectada”.
