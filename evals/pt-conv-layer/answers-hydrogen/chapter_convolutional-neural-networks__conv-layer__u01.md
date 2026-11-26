<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u01 lang=pt rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:33d3b96b kind=heading-->
# Convolutions for Images
<!--tr:33d3b96b-->
# Convoluções para imagens

<!--en:852f5868 kind=paragraph-->
In :numref:`sec_why-conv`, we derived the convolutional layer on paper from two
geometric principles: translation equivariance and locality. Now we see what
that formula actually computes on an image, and whether gradient descent can
discover useful kernels automatically from data.
<!--tr:852f5868-->
Em :numref:`sec_why-conv`, derivamos a camada convolucional no papel a partir de dois
princípios geométricos: equivariância à translação e localidade. Agora veremos o que
essa fórmula realmente calcula em uma imagem e se a descida de gradiente consegue
descobrir kernels úteis automaticamente a partir dos dados.

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
Considere a tarefa de localizar bordas verticais onde uma região clara encontra uma escura.
Subtrair o vizinho à direita de cada pixel ($[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$)
produz zero dentro de regiões uniformes, $+1$ onde o branco passa para o preto e
$-1$ onde o preto retorna ao branco. Varrer o detector de dois pesos $(1, -1)$
por todas as posições válidas da imagem gera um *mapa de bordas*. Nesta seção,
implementamos a correlação cruzada bidimensional, testamos nosso detector de bordas
feito à mão, deixamos a descida de gradiente *aprender* $(1, -1)$ diretamente dos dados,
mostramos como uma convolução se torna uma única multiplicação de matrizes (*im2col*) e
derivamos como o empilhamento de camadas amplia o campo receptivo de uma unidade.

<!--en:47958ed8 kind=heading-->
## Cross-Correlation
<!--tr:47958ed8-->
## Correlação cruzada

<!--en:5e4296fe kind=heading-->
### The Operation
<!--tr:5e4296fe-->
### A operação

<!--en:ae7e906d kind=paragraph-->
Leaving channels aside for the moment, let the input $\mathbf{X}$ be an
$n_\textrm{h} \times n_\textrm{w}$ matrix and the kernel $\mathbf{K}$ be a
$k_\textrm{h} \times k_\textrm{w}$ matrix. The kernel defines a sliding
*window* of shape $k_\textrm{h} \times k_\textrm{w}$. Placing the window's
upper-left corner at $(i, j)$, we multiply the covered input patch elementwise
with $\mathbf{K}$ and sum the products to produce the output entry at $(i, j)$:
<!--tr:ae7e906d-->
Deixando os canais de lado por enquanto, seja a entrada $\mathbf{X}$ uma
matriz $n_\textrm{h} \times n_\textrm{w}$ e o kernel $\mathbf{K}$ uma
matriz $k_\textrm{h} \times k_\textrm{w}$. O kernel define uma *janela* deslizante
de forma $k_\textrm{h} \times k_\textrm{w}$. Posicionando o canto superior esquerdo da janela
em $(i, j)$, multiplicamos a região coberta da entrada elemento a elemento
por $\mathbf{K}$ e somamos os produtos para produzir a entrada de saída em $(i, j)$:

<!--en:3688b38b kind=paragraph-->
This is the *two-dimensional cross-correlation* of $\mathbf{X}$ with
$\mathbf{K}$, written $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ with
zero-based indices matching code. Whereas :numref:`sec_why-conv` centered
offsets symmetrically from $-\Delta$ to $\Delta$, anchoring at the upper-left
corner differs only by a fixed shift of $\Delta$ while naturally handling
even-sized kernels (such as a $2 \times 2$ kernel or our $1 \times 2$ edge
detector) that have no center pixel.
<!--tr:3688b38b-->
Essa é a *correlação cruzada bidimensional* de $\mathbf{X}$ com
$\mathbf{K}$, escrita como $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ com
índices baseados em zero que correspondem ao código. Enquanto :numref:`sec_why-conv` centralizou
os deslocamentos simetricamente de $-\Delta$ a $\Delta$, ancorar no canto superior esquerdo
difere apenas por um deslocamento fixo de $\Delta$, lidando naturalmente com
kernels de tamanho par (como um kernel $2 \times 2$ ou nosso detector de bordas $1 \times 2$)
que não possuem pixel central.

<!--en:fa1b7da9 kind=caption-->
![Two-dimensional cross-correlation. The shaded portions are the first output element and the input and kernel elements used to compute it: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
<!--tr:fa1b7da9-->
![Correlação cruzada bidimensional. As partes sombreadas representam o primeiro elemento de saída e os elementos da entrada e do kernel usados para calculá-lo: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)

<!--en:2f8e5deb kind=paragraph-->
In :numref:`fig_correlation`, we slide a $2 \times 2$ kernel across a
$3 \times 3$ input one pixel at a time, left to right and top to bottom. The
window fits cleanly at four positions, yielding the $2 \times 2$ output:
<!--tr:2f8e5deb-->
Em :numref:`fig_correlation`, deslizamos um kernel $2 \times 2$ sobre uma
entrada $3 \times 3$ um pixel por vez, da esquerda para a direita e de cima para baixo. A
janela se encaixa perfeitamente em quatro posições, resultando na saída $2 \times 2$:

<!--en:4741e012 kind=paragraph-->
Notice that the output shrinks slightly along each axis because the kernel must
stay wholly inside the input. Fitting a width-$k$ window into length $n$ leaves
$n - k + 1$ valid positions, so the output shape is
<!--tr:4741e012-->
Observe que a saída encolhe ligeiramente ao longo de cada eixo porque o kernel deve
permanecer inteiramente dentro da entrada. Encaixar uma janela de largura $k$ em um comprimento $n$ deixa
$n - k + 1$ posições válidas, portanto a forma da saída é

<!--en:12d81c14 kind=paragraph-->
Thus a $3 \times 3$ input with a $2 \times 2$ kernel yields $2 \times 2$, a
$6 \times 8$ image with a $1 \times 2$ kernel yields $6 \times 7$, and a
$28 \times 28$ Fashion-MNIST image with a $5 \times 5$ kernel yields
$24 \times 24$. Whenever we want to preserve spatial dimensions, we surround the
input with a border of zeros (*padding*), which we examine alongside strides and
dilation in :numref:`sec_padding`.
<!--tr:12d81c14-->
Assim, uma entrada $3 \times 3$ com um kernel $2 \times 2$ resulta em $2 \times 2$, uma
imagem $6 \times 8$ com um kernel $1 \times 2$ resulta em $6 \times 7$, e uma
imagem Fashion-MNIST $28 \times 28$ com um kernel $5 \times 5$ resulta em
$24 \times 24$. Sempre que queremos preservar as dimensões espaciais, cercamos a
entrada com uma borda de zeros (*preenchimento*, ou *padding*), que examinaremos junto com strides e
dilatação em :numref:`sec_padding`.

<!--en:869d223f kind=heading-->
### Implementation
<!--tr:869d223f-->
### Implementação

<!--en:6a869f7c kind=paragraph-->
We implement :eqref:`eq_corr2d` directly in `corr2d` using a pair of loops over
every output coordinate.
<!--tr:6a869f7c-->
Implementamos :eqref:`eq_corr2d` diretamente em `corr2d` usando um par de laços sobre
cada coordenada de saída.

<!--en:81e1f383 kind=paragraph-->
To verify `corr2d` against our hand calculation in :numref:`fig_correlation`,
we construct the $3 \times 3$ input tensor `X` and $2 \times 2$ kernel `K` and
evaluate `corr2d(X, K)`.
<!--tr:81e1f383-->
Para verificar `corr2d` em comparação com nosso cálculo manual em :numref:`fig_correlation`,
construímos o tensor de entrada $3 \times 3$ `X` e o kernel $2 \times 2$ `K` e
avaliamos `corr2d(X, K)`.

<!--en:f567e1b2 kind=paragraph-->
The output reproduces :numref:`fig_correlation` entry for entry (and as shown in
:numref:`sec_why-conv`, strict mathematical convolution with a flipped kernel
$\mathbf{K}'$ learns the exact same family of layers, so we follow standard
deep learning terminology and call cross-correlation "convolution").
<!--tr:f567e1b2-->
A saída reproduz :numref:`fig_correlation` entrada por entrada (e, como mostrado em
:numref:`sec_why-conv`, a convolução matemática estrita com um kernel invertido
$\mathbf{K}'$ aprende exatamente a mesma família de camadas, portanto seguimos a
terminologia padrão de aprendizado profundo e chamamos a correlação cruzada de "convolução").

<!--en:bfe11c62 kind=heading-->
## Convolutional Layers
<!--tr:bfe11c62-->
## Camadas convolucionais

<!--en:2cdc020d kind=heading-->
### The Layer
<!--tr:2cdc020d-->
### A camada

<!--en:c6371ece kind=paragraph-->
A two-dimensional convolutional layer cross-correlates its input with a learned
kernel and adds a scalar bias. Just like our MLP weights in
:numref:`chap_perceptrons`, we initialize the kernel from
$\mathcal{N}(0, 0.01^2)$ and the bias to zero. Below, `Conv2D` registers both
parameters and invokes `corr2d` in its forward pass; a layer with a kernel of
height $h$ and width $w$ is called an $h \times w$ convolutional layer.
<!--tr:c6371ece-->
Uma camada convolucional bidimensional realiza a correlação cruzada da sua entrada com um kernel
aprendido e adiciona um viés escalar. Assim como nossos pesos de MLP em
:numref:`chap_perceptrons`, inicializamos o kernel a partir de
$\mathcal{N}(0, 0.01^2)$ e o viés em zero. Abaixo, `Conv2D` registra ambos os
parâmetros e invoca `corr2d` em sua passagem direta; uma camada com um kernel de
altura $h$ e largura $w$ é chamada de camada convolucional $h \times w$.

<!--en:29cd9061 kind=paragraph-->
Unlike a dense layer, a convolutional layer is not locked to a fixed input
resolution because its parameters depend only on the kernel size. The exact same
$2 \times 2$ layer (5 parameters total) processes both our $3 \times 3$ tensor
`X` and a $6 \times 8$ image, returning outputs of shapes $2 \times 2$ and
$5 \times 7$:
<!--tr:29cd9061-->
Diferentemente de uma camada densa, uma camada convolucional não fica presa a uma resolução de entrada
fixa porque seus parâmetros dependem apenas do tamanho do kernel. Exatamente a mesma
camada $2 \times 2$ (5 parâmetros no total) processa tanto nosso tensor $3 \times 3$
`X` quanto uma imagem $6 \times 8$, retornando saídas de formas $2 \times 2$ e
$5 \times 7$:

<!--en:c9e5aa07 kind=heading-->
### Edge Detection
<!--tr:c9e5aa07-->
### Detecção de bordas

<!--en:3d9a357b kind=paragraph-->
We now test the vertical edge detector from the introduction on a
$6 \times 8$ synthetic image whose middle four columns are black (`0`) and outer
columns are white (`1`).
<!--tr:3d9a357b-->
Agora testamos o detector de bordas verticais da introdução em uma
imagem sintética $6 \times 8$ cujas quatro colunas centrais são pretas (`0`) e as colunas
externas são brancas (`1`).

<!--en:68513ff0 kind=paragraph-->
We define the $1 \times 2$ kernel
<!--tr:68513ff0-->
Definimos o kernel $1 \times 2$

<!--en:eb45803b kind=paragraph-->
At every position $(i, j)$, this computes the horizontal *finite difference*
$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, a discrete approximation (up to sign)
of the horizontal derivative of brightness, which vanishes across flat regions
and spikes wherever intensity jumps.
<!--tr:eb45803b-->
Em cada posição $(i, j)$, isso calcula a *diferença finita* horizontal
$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, uma aproximação discreta (a menos do sinal)
da derivada horizontal do brilho, que se anula em regiões uniformes
e apresenta picos onde quer que a intensidade salte.

<!--en:e810b3a9 kind=paragraph-->
Cross-correlating the $6 \times 8$ stripe image `X` with the $1 \times 2$
horizontal-difference kernel `K` via `corr2d(X, K)` produces the $6 \times 7$
edge map `Y`:
<!--tr:e810b3a9-->
A correlação cruzada da imagem da faixa $6 \times 8$ `X` com o kernel de diferença
horizontal $1 \times 2$ `K` via `corr2d(X, K)` produz o mapa de bordas $6 \times 7$
`Y`:

<!--en:568358e4 kind=paragraph-->
Sure enough, `Y` fires `+1` along the white-to-black boundary, `-1` along the
black-to-white boundary, and `0` everywhere else. Applying the exact same kernel
to the transposed image (turning the vertical stripe into a horizontal one)
yields all zeros:
<!--tr:568358e4-->
Como esperado, `Y` ativa `+1` ao longo da fronteira do branco para o preto, `-1` ao longo da
fronteira do preto para o branco e `0` em todo o restante. Aplicar exatamente o mesmo kernel
à imagem transposta (transformando a faixa vertical em uma faixa horizontal)
resulta apenas em zeros:

<!--en:71c591a4 kind=paragraph-->
Clearly `[1, -1]` responds strictly to vertical edges. Detecting horizontal
edges, diagonals, corners, or textures requires different kernels altogether.
<!--tr:71c591a4-->
Claramente `[1, -1]` responde estritamente a bordas verticais. Detectar bordas
horizontais, diagonais, cantos ou texturas requer kernels completamente diferentes.

<!--en:f0af4887 kind=heading-->
### Learning a Kernel
<!--tr:f0af4887-->
### Aprendendo um kernel

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
Criar `[1, -1]` manualmente é fácil quando sabemos exatamente a característica que queremos, mas nenhum
engenheiro consegue especificar à mão milhares de filtros intermediários em uma rede profunda.
A descida de gradiente consegue recuperar o kernel `[1, -1]` apenas a partir do par entrada-saída
`(X, Y)`? Testamos isso usando a camada convolucional 2D nativa de cada framework
(com o viés omitido e pesos inicializados a partir de
$\mathcal{N}(0, 0.01^2)$), minimizando o erro quadrático somado em todas as 42 entradas de saída
ao longo de 10 passos de gradiente com taxa de aprendizado $0.03$. Como as camadas nativas
esperam tensores 4D incluindo eixos de lote e canal (:numref:`sec_channels`),
remodelamos `X` e `Y` de acordo.

<!--en:c29c5d61 kind=paragraph-->
Initially the loss sits near 12 (the near-zero random kernel misses all 12
$\pm 1$ edge pixels by roughly 1), and within just nine steps it drops below
0.01. Inspecting the learned kernel confirms what it found:
<!--tr:c29c5d61-->
Inicialmente, a perda fica próxima de 12 (o kernel aleatório próximo de zero erra todos os 12
pixels de borda $\pm 1$ por cerca de 1) e, em apenas nove passos, cai abaixo de
0.01. Inspecionar o kernel aprendido confirma o que foi encontrado:

<!--en:67d25b0b kind=paragraph-->
Both entries land within roughly 0.02 of `[1, -1]`. Nothing in this loop knew
anything about edges: given a loss signal, gradient descent discovered the
filter on its own. Here we supervised the single layer directly with `Y`; in a
deep CNN, backpropagation learns every layer's kernels end to end from the final
task loss alone.
<!--tr:67d25b0b-->
Ambas as entradas ficam a cerca de 0.02 de `[1, -1]`. Nada nesse laço sabia
coisa alguma sobre bordas: dado um sinal de perda, a descida de gradiente descobriu o
filtro por conta própria. Aqui supervisionamos a única camada diretamente com `Y`; em uma
rede neural convolucional profunda, a retropropagação aprende os kernels de todas as camadas de ponta a ponta apenas a partir da
perda final da tarefa.

<!--en:91b0010d kind=slide_text-->
Convolutions for Images<br>**2D cross-correlation · edge detection · learned kernels · im2col · receptive fields**
<!--tr:91b0010d-->
Convoluções para imagens<br>**correlação cruzada 2D · detecção de bordas · kernels aprendidos · im2col · campos receptivos**

<!--en:1d94295a kind=slide_title-->
::: {.slide title="Detecting Local Edges Across an Image"}
<!--tr:1d94295a-->
::: {.slide title="Detectando bordas locais em uma imagem"}

<!--en:687ed302 kind=kicker-->
[Motivation & Core Mental Model]{.kicker}
<!--tr:687ed302-->
[Motivação e modelo mental central]{.kicker}

<!--en:05fceed0 kind=slide_text-->
- **Synthetic $6 \times 8$ test image:** White outer columns (`1`) surrounding a 4-column black vertical band (`0`).
<!--tr:05fceed0-->
- **Imagem sintética de teste $6 \times 8$:** Colunas externas brancas (`1`) cercando uma faixa vertical preta de 4 colunas (`0`).

<!--en:71e700b3 kind=slide_text-->
- **Horizontal finite difference:** Sweeping $\mathbf{K} = [1, -1]$ computes $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, firing `+1` at white-to-black and `-1` at black-to-white edges.
<!--tr:71e700b3-->
- **Diferença finita horizontal:** Varrer $\mathbf{K} = [1, -1]$ calcula $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, ativando `+1` nas bordas branco-para-preto e `-1` nas bordas preto-para-branco.

<!--en:957dc033 kind=slide_text-->
- **Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent.
<!--tr:957dc033-->
- **Aprender vs. projetar à mão:** Redes neurais convolucionais profundas descobrem milhares desses kernels automaticamente via descida de gradiente.

<!--en:cb1045e1 kind=slide_title-->
::: {.slide title="The 2D Cross-Correlation Operation"}
<!--tr:cb1045e1-->
::: {.slide title="A operação de correlação cruzada 2D"}

<!--en:17bcece5 kind=kicker-->
[Sliding-Window Inner Products]{.kicker}
<!--tr:17bcece5-->
[Produtos internos em janela deslizante]{.kicker}

<!--en:8d45aae8 kind=slide_text-->
Given an $n_\textrm{h} \times n_\textrm{w}$ input matrix $\mathbf{X}$ and a $k_\textrm{h} \times k_\textrm{w}$ kernel $\mathbf{K}$, the 2D cross-correlation $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ places the window's top-left corner at $(i, j)$ and sums elementwise products:
<!--tr:8d45aae8-->
Dada uma matriz de entrada $\mathbf{X}$ de dimensões $n_\textrm{h} \times n_\textrm{w}$ e um kernel $\mathbf{K}$ de dimensões $k_\textrm{h} \times k_\textrm{w}$, a correlação cruzada 2D $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ posiciona o canto superior esquerdo da janela em $(i, j)$ e soma os produtos elemento a elemento:

<!--en:e484380c kind=slide_text-->
- **Top-left anchor:** Zero-based indices $a, b \ge 0$ match array indexing in code and handle even-sized kernels (like $2 \times 2$ or $1 \times 2$) cleanly.
<!--tr:e484380c-->
- **Âncora superior esquerda:** Índices baseados em zero $a, b \ge 0$ correspondem à indexação em código e lidam naturalmente com kernels de tamanho par (como $2 \times 2$ ou $1 \times 2$).

<!--en:137c0980 kind=slide_text-->
- **Shaded top-left output $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.
<!--tr:137c0980-->
- **Saída superior esquerda sombreada $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.

<!--en:575c4aea kind=slide_title-->
::: {.slide title="Valid Window Positions and Output Spatial Shape"}
<!--tr:575c4aea-->
::: {.slide title="Posições válidas da janela e forma espacial da saída"}

<!--en:7eda4531 kind=kicker-->
[Why Unpadded Convolutions Shrink the Grid]{.kicker}
<!--tr:7eda4531-->
[Por que convoluções sem preenchimento reduzem a grade]{.kicker}

<!--en:891243ae kind=slide_text-->
- **Valid window placements:** A $k_\textrm{h} \times k_\textrm{w}$ window fits inside an $n_\textrm{h} \times n_\textrm{w}$ grid at $n - k + 1$ positions per axis.
<!--tr:891243ae-->
- **Posições válidas da janela:** Uma janela $k_\textrm{h} \times k_\textrm{w}$ cabe em uma grade $n_\textrm{h} \times n_\textrm{w}$ em $n - k + 1$ posições por eixo.

<!--en:55cc8c32 kind=slide_text-->
- **Shape examples:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.
<!--tr:55cc8c32-->
- **Exemplos de forma:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.

<!--en:1e3463da kind=slide_title-->
::: {.slide title="Implementing 2D Cross-Correlation in Code"}
<!--tr:1e3463da-->
::: {.slide title="Implementando correlação cruzada 2D em código"}

<!--en:07566992 kind=kicker-->
[Direct Nested-Loop Implementation]{.kicker}
<!--tr:07566992-->
[Implementação direta com laços aninhados]{.kicker}

<!--en:202aa5d3 kind=slide_text-->
The function `corr2d` allocates the $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ output tensor and computes each entry as the sum of `X[i:i+h, j:j+w] * K`:
<!--tr:202aa5d3-->
A função `corr2d` aloca o tensor de saída $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ e calcula cada entrada como a soma de `X[i:i+h, j:j+w] * K`:

<!--en:bef130ba kind=slide_text-->
Each iteration slices a $k_\textrm{h} \times k_\textrm{w}$ sub-window of `X`, multiplies elementwise with `K`, and reduces via `sum`.
<!--tr:bef130ba-->
Cada iteração fatia uma subjanela $k_\textrm{h} \times k_\textrm{w}$ de `X`, multiplica elemento a elemento por `K` e reduz via `sum`.

<!--en:4ee9f6b1 kind=slide_title-->
::: {.slide title="Verifying corr2d and the Flipped-Kernel Equivalence"}
<!--tr:4ee9f6b1-->
::: {.slide title="Verificando corr2d e a equivalência com kernel invertido"}

<!--en:2b0be423 kind=kicker-->
[Cross-Correlation vs. Mathematical Convolution]{.kicker}
<!--tr:2b0be423-->
[Correlação cruzada vs. convolução matemática]{.kicker}

<!--en:74b9b745 kind=slide_text-->
- **180° flipped kernel $\mathbf{K}'$:** Cross-correlating with $\mathbf{K}$ equals convolving with $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.
<!--tr:74b9b745-->
- **Kernel invertido em 180° $\mathbf{K}'$:** Correlacionar de forma cruzada com $\mathbf{K}$ equivale a convoluir com $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.

<!--en:5c8af7cd kind=slide_text-->
- **Identical learned class:** Frameworks compute cross-correlation and call it convolution.
<!--tr:5c8af7cd-->
- **Mesma classe aprendida:** Frameworks calculam a correlação cruzada e a chamam de convolução.

<!--en:a69e29b5 kind=slide_title-->
::: {.slide title="Building a 2D Convolutional Layer Module"}
<!--tr:a69e29b5-->
::: {.slide title="Construindo um módulo de camada convolucional 2D"}

<!--en:55763603 kind=kicker-->
[Learnable Kernel Weights Plus Scalar Bias]{.kicker}
<!--tr:55763603-->
[Pesos de kernel aprendíveis mais viés escalar]{.kicker}

<!--en:5b8bad24 kind=slide_text-->
A convolutional layer wraps `corr2d` with a learnable $k_\textrm{h} \times k_\textrm{w}$ weight parameter (initialized from $\mathcal{N}(0, 0.01^2)$) and a scalar bias (initialized to $0$):
<!--tr:5b8bad24-->
Uma camada convolucional encapsula `corr2d` com um parâmetro de peso aprendível $k_\textrm{h} \times k_\textrm{w}$ (inicializado de $\mathcal{N}(0, 0.01^2)$) e um viés escalar (inicializado em $0$):

<!--en:c16d97d6 kind=slide_title-->
::: {.slide title="Resolution Independence of Convolutional Layers"}
<!--tr:c16d97d6-->
::: {.slide title="Independência de resolução em camadas convolucionais"}

<!--en:a7d1e48f kind=kicker-->
[Fixed Parameters Across Arbitrary Input Sizes]{.kicker}
<!--tr:a7d1e48f-->
[Parâmetros fixos para tamanhos de entrada arbitrários]{.kicker}

<!--en:53a2bce7 kind=slide_text-->
- **Decoupled from $H \times W$:** The exact same 5 parameters ($2 \times 2$ kernel + 1 bias) process $3 \times 3$, $6 \times 8$, or megapixel inputs without reshaping weights.
<!--tr:53a2bce7-->
- **Desacoplado de $H \times W$:** Exatamente os mesmos 5 parâmetros (kernel $2 \times 2$ + 1 viés) processam entradas $3 \times 3$, $6 \times 8$ ou de megapixels sem remodelar pesos.

<!--en:017fec24 kind=slide_title-->
::: {.slide title="Constructing a Synthetic Vertical-Edge Image"}
<!--tr:017fec24-->
::: {.slide title="Construindo uma imagem sintética de borda vertical"}

<!--en:896e641c kind=kicker-->
[A 6×8 Binary Test Pattern]{.kicker}
<!--tr:896e641c-->
[Um padrão de teste binário 6×8]{.kicker}

<!--en:0a4b456b kind=slide_text-->
We construct the $6 \times 8$ test image `X` with ones on the left and right borders and zeros across the middle four columns (`X[:, 2:6] = 0`):
<!--tr:0a4b456b-->
Construímos a imagem de teste $6 \times 8$ `X` com uns nas bordas esquerda e direita e zeros nas quatro colunas centrais (`X[:, 2:6] = 0`):

<!--en:dfc7e3e3 kind=slide_text-->
- **Piecewise-constant columns:** Intensity is constant (`1` or `0`) within columns $0\text{--}1$, $2\text{--}5$, and $6\text{--}7$, jumping only between columns $1 \to 2$ and $5 \to 6$.
<!--tr:dfc7e3e3-->
- **Colunas constantes por partes:** A intensidade é constante (`1` ou `0`) nas colunas $0\text{--}1$, $2\text{--}5$ e $6\text{--}7$, saltando apenas entre as colunas $1 \to 2$ e $5 \to 6$.

<!--en:3ce31cbe kind=slide_title-->
::: {.slide title="Applying the Horizontal Finite-Difference Kernel"}
<!--tr:3ce31cbe-->
::: {.slide title="Aplicando o kernel de diferença finita horizontal"}

<!--en:b37a1029 kind=kicker-->
[Detecting Vertical Edges with K = [1, -1]]{.kicker}
<!--tr:b37a1029-->
[Detectando bordas verticais com K = [1, -1]]{.kicker}

<!--en:9be85019 kind=slide_text-->
Cross-correlating `X` with the $1 \times 2$ kernel $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ computes $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ at every horizontal pair:
<!--tr:9be85019-->
A correlação cruzada de `X` com o kernel $1 \times 2$ $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ calcula $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ em cada par horizontal:

<!--en:f649cfae kind=slide_text-->
- **Signed edge map:** Column 1 fires `+1` (white-to-black transition), column 5 fires `-1` (black-to-white transition), and all constant interior columns evaluate to `0`.
<!--tr:f649cfae-->
- **Mapa de bordas com sinal:** A coluna 1 ativa `+1` (transição branco-para-preto), a coluna 5 ativa `-1` (transição preto-para-branco) e todas as colunas internas constantes resultam em `0`.

<!--en:e6d155be kind=slide_title-->
::: {.slide title="Orientation Selectivity of Edge Kernels"}
<!--tr:e6d155be-->
::: {.slide title="Seletividade de orientação dos kernels de borda"}

<!--en:b525cde4 kind=kicker-->
[Why One Kernel Cannot Detect All Orientations]{.kicker}
<!--tr:b525cde4-->
[Por que um único kernel não detecta todas as orientações]{.kicker}

<!--en:38159de6 kind=slide_text-->
- **Blind to horizontal edges:** Transposing `X` (`X.T`) makes horizontal neighbors identical, so $[1, -1]$ outputs all zeros.
<!--tr:38159de6-->
- **Cego a bordas horizontais:** Transpor `X` (`X.T`) torna vizinhos horizontais idênticos, fazendo $[1, -1]$ retornar apenas zeros.

<!--en:f0f30f76 kind=slide_text-->
- **Filter banks (§{sec_channels}):** Vision layers learn parallel channels for horizontal, vertical, and diagonal orientations.
<!--tr:f0f30f76-->
- **Bancos de filtros (§{sec_channels}):** Camadas de visão aprendem canais paralelos para orientações horizontais, verticais e diagonais.

<!--en:b755e31b kind=slide_title-->
::: {.slide title="Learning an Edge Detector via Gradient Descent"}
<!--tr:b755e31b-->
::: {.slide title="Aprendendo um detector de bordas via descida de gradiente"}

<!--en:86bbeea7 kind=kicker-->
[Supervising a Built-In Convolutional Layer on (X, Y)]{.kicker}
<!--tr:86bbeea7-->
[Supervisionando uma camada convolucional nativa em (X, Y)]{.kicker}

<!--en:a569780c kind=slide_text-->
- **4D framework tensors:** Reshape `X` and `Y` to 4D tensors (one example, one channel) and minimize $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ at $\eta = 0.03$.
<!--tr:a569780c-->
- **Tensores 4D do framework:** Remodele `X` e `Y` para tensores 4D (um exemplo, um canal) e minimize $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ com $\eta = 0.03$.

<!--en:2489ce6e kind=slide_title-->
::: {.slide title="Inspecting the Learned Kernel Weights"}
<!--tr:2489ce6e-->
::: {.slide title="Inspecionando os pesos do kernel aprendido"}

<!--en:7e4ef44b kind=kicker-->
[Why Initial Loss Is 12 and Converges to [1, -1]]{.kicker}
<!--tr:7e4ef44b-->
[Por que a perda inicial é 12 e converge para [1, -1]]{.kicker}

<!--en:9cd1c645 kind=slide_text-->
- **Why loss starts at $12.0$:** Near-zero weights predict $\hat{Y} \approx 0$, missing the twelve $\pm 1$ edge pixels in `Y` ($12 \times 1^2 = 12$).
<!--tr:9cd1c645-->
- **Por que a perda começa em $12.0$:** Pesos próximos de zero preveem $\hat{Y} \approx 0$, errando os doze pixels de borda $\pm 1$ em `Y` ($12 \times 1^2 = 12$).

<!--en:456670c5 kind=slide_text-->
- **Convergence in 9 steps:** SGD recovers $[0.99, -0.99]$ automatically from data.
<!--tr:456670c5-->
- **Convergência em 9 passos:** A SGD recupera $[0.99, -0.99]$ automaticamente a partir dos dados.

=== DECISIONS ===
- Register: Brazilian Portuguese (`pt-BR`), first-person plural ("nós": derivamos, implementamos, testamos) matching the book's collegial tone.
- "feature map" -> "mapa de características" (consistent with glossary `feature` = `característica`).
- "edge map" -> "mapa de bordas".
- "input–output pair" -> "par entrada-saída".
- "SGD" -> kept as "SGD" in slide bullets where the English uses the acronym, with feminine agreement ("a SGD") when appropriate (descida de gradiente estocástica).
- Headings and slide titles: sentence case in pt-BR, noun-phrase style, no trailing period.
- Prose decimal numbers: kept with dot (`0.5`, `0.01`, `0.03`, `0.02`, `12.0`) per TRANSLATING.md §4.

=== NOTES ===
- 8d45aae8: math tokens reordered because Portuguese grammar places the dimensions after the matrix/kernel symbol ("matriz de entrada $\mathbf{X}$ de dimensões $n_\textrm{h} \times n_\textrm{w}$", "kernel $\mathbf{K}$ de dimensões $k_\textrm{h} \times k_\textrm{w}$").

