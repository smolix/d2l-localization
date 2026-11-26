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
Em :numref:`sec_why-conv`, deduzimos no papel a camada convolucional a partir de
dois princípios geométricos, a equivariância à translação e a localidade.
Agora veremos o que essa fórmula calcula ao ser aplicada a uma imagem e se a
descida de gradiente consegue descobrir kernels úteis automaticamente a partir
dos dados.

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
Suponha que queiramos localizar bordas verticais onde uma região clara encontra
uma escura. Subtrair de cada pixel o valor do vizinho à direita
($[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$) dá zero no interior de regiões
uniformes, $+1$ na passagem do branco para o preto e $-1$ na passagem do preto
para o branco. Deslizar o detector de dois pesos $(1, -1)$ por todas as posições
válidas da imagem produz um *mapa de bordas*. Nesta seção, vamos implementar a
correlação cruzada bidimensional, testar o detector de bordas que projetamos
manualmente, deixar a descida de gradiente *aprender* $(1, -1)$ diretamente dos
dados, mostrar como uma convolução se reduz a uma única multiplicação de
matrizes (*im2col*) e deduzir como o empilhamento de camadas amplia o campo
receptivo de uma unidade.

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
Deixando os canais de lado por enquanto, seja a entrada $\mathbf{X}$ uma matriz
$n_\textrm{h} \times n_\textrm{w}$ e seja o kernel $\mathbf{K}$ uma matriz
$k_\textrm{h} \times k_\textrm{w}$. O kernel define uma *janela* deslizante de
forma $k_\textrm{h} \times k_\textrm{w}$. Posicionamos o canto superior esquerdo
da janela em $(i, j)$, multiplicamos elemento a elemento o recorte da entrada
coberto pela janela por $\mathbf{K}$ e somamos os produtos para obter o elemento
da saída em $(i, j)$.

<!--en:3688b38b kind=paragraph-->
This is the *two-dimensional cross-correlation* of $\mathbf{X}$ with
$\mathbf{K}$, written $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ with
zero-based indices matching code. Whereas :numref:`sec_why-conv` centered
offsets symmetrically from $-\Delta$ to $\Delta$, anchoring at the upper-left
corner differs only by a fixed shift of $\Delta$ while naturally handling
even-sized kernels (such as a $2 \times 2$ kernel or our $1 \times 2$ edge
detector) that have no center pixel.
<!--tr:3688b38b-->
Essa é a *correlação cruzada bidimensional* de $\mathbf{X}$ com $\mathbf{K}$,
escrita como $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$, com índices que começam
em zero, como no código. Em :numref:`sec_why-conv`, os deslocamentos iam
simetricamente de $-\Delta$ a $\Delta$. Tomar o canto superior esquerdo como
origem muda a referência apenas por um deslocamento fixo de $\Delta$ e acomoda
naturalmente kernels de dimensões pares, como um kernel $2 \times 2$ ou nosso
detector de bordas $1 \times 2$, que não têm um pixel central.

<!--en:fa1b7da9 kind=caption-->
![Two-dimensional cross-correlation. The shaded portions are the first output element and the input and kernel elements used to compute it: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
<!--tr:fa1b7da9-->
![Correlação cruzada bidimensional. O sombreamento indica o primeiro elemento da saída e os elementos da entrada e do kernel usados para calculá-lo ($0\times0+1\times1+3\times2+4\times3=19$).](../img/correlation.svg)

<!--en:2f8e5deb kind=paragraph-->
In :numref:`fig_correlation`, we slide a $2 \times 2$ kernel across a
$3 \times 3$ input one pixel at a time, left to right and top to bottom. The
window fits cleanly at four positions, yielding the $2 \times 2$ output:
<!--tr:2f8e5deb-->
Na :numref:`fig_correlation`, deslizamos um kernel $2 \times 2$ sobre uma entrada
$3 \times 3$, um pixel de cada vez, da esquerda para a direita e de cima para
baixo. A janela cabe inteira em quatro posições, o que produz a saída
$2 \times 2$ a seguir.

<!--en:4741e012 kind=paragraph-->
Notice that the output shrinks slightly along each axis because the kernel must
stay wholly inside the input. Fitting a width-$k$ window into length $n$ leaves
$n - k + 1$ valid positions, so the output shape is
<!--tr:4741e012-->
Observe que a saída é um pouco menor em cada eixo, pois o kernel precisa caber
inteiramente na entrada. Ao encaixar uma janela de largura $k$ em uma extensão
de comprimento $n$, temos $n - k + 1$ posições válidas. Assim, a forma da saída é

<!--en:12d81c14 kind=paragraph-->
Thus a $3 \times 3$ input with a $2 \times 2$ kernel yields $2 \times 2$, a
$6 \times 8$ image with a $1 \times 2$ kernel yields $6 \times 7$, and a
$28 \times 28$ Fashion-MNIST image with a $5 \times 5$ kernel yields
$24 \times 24$. Whenever we want to preserve spatial dimensions, we surround the
input with a border of zeros (*padding*), which we examine alongside strides and
dilation in :numref:`sec_padding`.
<!--tr:12d81c14-->
Assim, uma entrada $3 \times 3$ com um kernel $2 \times 2$ produz uma saída
$2 \times 2$, uma imagem $6 \times 8$ com um kernel $1 \times 2$ produz uma
saída $6 \times 7$, e uma imagem $28 \times 28$ do Fashion-MNIST com um kernel
$5 \times 5$ produz uma saída $24 \times 24$. Quando queremos preservar as
dimensões espaciais, acrescentamos uma borda de zeros ao redor da entrada. Esse
*preenchimento* é examinado junto com stride e dilatação na
:numref:`sec_padding`.

<!--en:869d223f kind=heading-->
### Implementation
<!--tr:869d223f-->
### Implementação

<!--en:6a869f7c kind=paragraph-->
We implement :eqref:`eq_corr2d` directly in `corr2d` using a pair of loops over
every output coordinate.
<!--tr:6a869f7c-->
Implementamos :eqref:`eq_corr2d` diretamente em `corr2d`, com dois laços que
percorrem todas as coordenadas da saída.

<!--en:81e1f383 kind=paragraph-->
To verify `corr2d` against our hand calculation in :numref:`fig_correlation`,
we construct the $3 \times 3$ input tensor `X` and $2 \times 2$ kernel `K` and
evaluate `corr2d(X, K)`.
<!--tr:81e1f383-->
Para conferir se `corr2d` reproduz o cálculo manual da
:numref:`fig_correlation`, construímos o tensor de entrada $3 \times 3$ `X` e
o kernel $2 \times 2$ `K` e executamos `corr2d(X, K)`.

<!--en:f567e1b2 kind=paragraph-->
The output reproduces :numref:`fig_correlation` entry for entry (and as shown in
:numref:`sec_why-conv`, strict mathematical convolution with a flipped kernel
$\mathbf{K}'$ learns the exact same family of layers, so we follow standard
deep learning terminology and call cross-correlation "convolution").
<!--tr:f567e1b2-->
A saída reproduz, elemento por elemento, o resultado da
:numref:`fig_correlation`. Como vimos em :numref:`sec_why-conv`, a convolução
no sentido matemático estrito, com o kernel invertido $\mathbf{K}'$, aprende
exatamente a mesma família de camadas. Por isso, seguimos a terminologia
usual do aprendizado profundo e chamamos a correlação cruzada de “convolução”.

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
Uma camada convolucional bidimensional calcula a correlação cruzada da entrada
com um kernel aprendido e soma um viés escalar. Como fizemos com os pesos do
MLP em :numref:`chap_perceptrons`, inicializamos o kernel com amostras de
$\mathcal{N}(0, 0.01^2)$ e atribuímos zero ao viés. Abaixo, `Conv2D` registra
os dois parâmetros e chama `corr2d` durante a passagem direta. Se o kernel tem
altura $h$ e largura $w$, chamamos a camada de camada convolucional
$h \times w$.

<!--en:29cd9061 kind=paragraph-->
Unlike a dense layer, a convolutional layer is not locked to a fixed input
resolution because its parameters depend only on the kernel size. The exact same
$2 \times 2$ layer (5 parameters total) processes both our $3 \times 3$ tensor
`X` and a $6 \times 8$ image, returning outputs of shapes $2 \times 2$ and
$5 \times 7$:
<!--tr:29cd9061-->
Ao contrário de uma camada densa, uma camada convolucional aceita entradas de
diferentes resoluções, pois seus parâmetros dependem apenas do tamanho do
kernel. A mesma camada $2 \times 2$ (5 parâmetros ao todo) processa tanto nosso
tensor $3 \times 3$ `X` quanto uma imagem $6 \times 8$, produzindo saídas de
formas $2 \times 2$ e $5 \times 7$, respectivamente.

<!--en:c9e5aa07 kind=heading-->
### Edge Detection
<!--tr:c9e5aa07-->
### Detecção de bordas

<!--en:3d9a357b kind=paragraph-->
We now test the vertical edge detector from the introduction on a
$6 \times 8$ synthetic image whose middle four columns are black (`0`) and outer
columns are white (`1`).
<!--tr:3d9a357b-->
Agora testamos o detector de bordas verticais da introdução em uma imagem
sintética $6 \times 8$ cujas quatro colunas centrais são pretas (`0`) e as
colunas externas são brancas (`1`).

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
Em cada posição $(i, j)$, o kernel calcula a *diferença finita* horizontal
$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, uma aproximação discreta (a menos
do sinal) da derivada horizontal do brilho. A diferença se anula nas regiões
uniformes e apresenta picos onde a intensidade muda bruscamente.

<!--en:e810b3a9 kind=paragraph-->
Cross-correlating the $6 \times 8$ stripe image `X` with the $1 \times 2$
horizontal-difference kernel `K` via `corr2d(X, K)` produces the $6 \times 7$
edge map `Y`:
<!--tr:e810b3a9-->
A correlação cruzada entre a imagem $6 \times 8$ com a faixa, `X`, e o kernel
$1 \times 2$ de diferença horizontal, `K`, produz com `corr2d(X, K)` o mapa de
bordas $6 \times 7$ `Y`.

<!--en:568358e4 kind=paragraph-->
Sure enough, `Y` fires `+1` along the white-to-black boundary, `-1` along the
black-to-white boundary, and `0` everywhere else. Applying the exact same kernel
to the transposed image (turning the vertical stripe into a horizontal one)
yields all zeros:
<!--tr:568358e4-->
Como esperado, `Y` vale `+1` ao longo da transição do branco para o preto,
`-1` na transição do preto para o branco e `0` em todas as outras posições.
Quando aplicamos o mesmo kernel à imagem transposta (que passa a ter uma faixa
horizontal), obtemos apenas zeros.

<!--en:71c591a4 kind=paragraph-->
Clearly `[1, -1]` responds strictly to vertical edges. Detecting horizontal
edges, diagonals, corners, or textures requires different kernels altogether.
<!--tr:71c591a4-->
Nesse teste, fica claro que `[1, -1]` responde apenas às bordas verticais.
Detectar bordas horizontais, diagonais, cantos ou texturas exige outros kernels.

<!--en:f0af4887 kind=heading-->
### Learning a Kernel
<!--tr:f0af4887-->
### Aprendizado de um kernel

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
É fácil projetar `[1, -1]` à mão quando sabemos exatamente qual característica
queremos detectar, mas nenhum engenheiro consegue especificar manualmente
milhares de filtros intermediários de uma rede profunda. Será que a descida
de gradiente consegue recuperar o kernel `[1, -1]` apenas a partir do par de
entrada e saída `(X, Y)`? Para testar isso, usamos a camada convolucional 2D
nativa de cada framework (sem viés e com pesos inicializados a partir de
$\mathcal{N}(0, 0.01^2)$). Minimizamos a soma dos erros quadráticos nos 42
elementos da saída ao longo de 10 passos de gradiente, com taxa de aprendizado
$0.03$. Como as camadas nativas esperam tensores 4D com eixos para lote e canal
(:numref:`sec_channels`), ajustamos a forma de `X` e `Y`.

<!--en:c29c5d61 kind=paragraph-->
Initially the loss sits near 12 (the near-zero random kernel misses all 12
$\pm 1$ edge pixels by roughly 1), and within just nine steps it drops below
0.01. Inspecting the learned kernel confirms what it found:
<!--tr:c29c5d61-->
A perda começa perto de 12 (com o kernel aleatório próximo de zero, o erro em
cada um dos 12 pixels de borda de valor $\pm 1$ é de cerca de 1) e cai para
menos de 0.01 em apenas nove passos. Podemos conferir o que foi aprendido
examinando o kernel resultante.

<!--en:67d25b0b kind=paragraph-->
Both entries land within roughly 0.02 of `[1, -1]`. Nothing in this loop knew
anything about edges: given a loss signal, gradient descent discovered the
filter on its own. Here we supervised the single layer directly with `Y`; in a
deep CNN, backpropagation learns every layer's kernels end to end from the final
task loss alone.
<!--tr:67d25b0b-->
Os dois valores ficam a cerca de 0.02 ou menos dos respectivos valores em
`[1, -1]`. O laço não recebeu nenhuma informação específica sobre bordas.
A descida de gradiente descobriu o filtro por conta própria a partir do sinal
de perda. Aqui supervisionamos uma única camada diretamente com `Y`. Em uma
rede neural convolucional profunda, a retropropagação permite aprender de
ponta a ponta os kernels de todas as camadas usando apenas a perda da tarefa
final.

<!--en:91b0010d kind=slide_text-->
Convolutions for Images<br>**2D cross-correlation · edge detection · learned kernels · im2col · receptive fields**
<!--tr:91b0010d-->
Convoluções para imagens<br>**Correlação cruzada 2D · detecção de bordas · kernels aprendidos · im2col · campos receptivos**

<!--en:1d94295a kind=slide_title-->
::: {.slide title="Detecting Local Edges Across an Image"}
<!--tr:1d94295a-->
::: {.slide title="Detecção de bordas locais em uma imagem"}

<!--en:687ed302 kind=kicker-->
[Motivation & Core Mental Model]{.kicker}
<!--tr:687ed302-->
[Motivação e ideia central]{.kicker}

<!--en:05fceed0 kind=slide_text-->
- **Synthetic $6 \times 8$ test image:** White outer columns (`1`) surrounding a 4-column black vertical band (`0`).
<!--tr:05fceed0-->
- **Imagem sintética de teste $6 \times 8$** com colunas externas brancas (`1`) ao redor de uma faixa vertical preta de 4 colunas (`0`).

<!--en:71e700b3 kind=slide_text-->
- **Horizontal finite difference:** Sweeping $\mathbf{K} = [1, -1]$ computes $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, firing `+1` at white-to-black and `-1` at black-to-white edges.
<!--tr:71e700b3-->
- **Diferença finita horizontal.** Deslizar $\mathbf{K} = [1, -1]$ calcula $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, com resposta `+1` do branco para o preto e `-1` do preto para o branco.

<!--en:957dc033 kind=slide_text-->
- **Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent.
<!--tr:957dc033-->
- **Aprendizado e projeto manual.** Redes neurais convolucionais profundas descobrem milhares desses kernels automaticamente por descida de gradiente.

<!--en:cb1045e1 kind=slide_title-->
::: {.slide title="The 2D Cross-Correlation Operation"}
<!--tr:cb1045e1-->
::: {.slide title="A operação de correlação cruzada 2D"}

<!--en:17bcece5 kind=kicker-->
[Sliding-Window Inner Products]{.kicker}
<!--tr:17bcece5-->
[Produtos internos em janelas deslizantes]{.kicker}

<!--en:8d45aae8 kind=slide_text-->
Given an $n_\textrm{h} \times n_\textrm{w}$ input matrix $\mathbf{X}$ and a $k_\textrm{h} \times k_\textrm{w}$ kernel $\mathbf{K}$, the 2D cross-correlation $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ places the window's top-left corner at $(i, j)$ and sums elementwise products:
<!--tr:8d45aae8-->
Dada uma matriz de entrada $n_\textrm{h} \times n_\textrm{w}$, $\mathbf{X}$, e um kernel $k_\textrm{h} \times k_\textrm{w}$, $\mathbf{K}$, a correlação cruzada 2D $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ posiciona o canto superior esquerdo da janela em $(i, j)$ e soma os produtos elemento a elemento.

<!--en:e484380c kind=slide_text-->
- **Top-left anchor:** Zero-based indices $a, b \ge 0$ match array indexing in code and handle even-sized kernels (like $2 \times 2$ or $1 \times 2$) cleanly.
<!--tr:e484380c-->
- **Origem no canto superior esquerdo.** Os índices $a, b \ge 0$ começam em zero, como no código, e acomodam kernels de dimensões pares, como $2 \times 2$ ou $1 \times 2$.

<!--en:137c0980 kind=slide_text-->
- **Shaded top-left output $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.
<!--tr:137c0980-->
- **Saída sombreada no canto superior esquerdo $(0,0)$**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.

<!--en:575c4aea kind=slide_title-->
::: {.slide title="Valid Window Positions and Output Spatial Shape"}
<!--tr:575c4aea-->
::: {.slide title="Posições válidas da janela e forma espacial da saída"}

<!--en:7eda4531 kind=kicker-->
[Why Unpadded Convolutions Shrink the Grid]{.kicker}
<!--tr:7eda4531-->
[Por que a convolução sem preenchimento reduz a grade]{.kicker}

<!--en:891243ae kind=slide_text-->
- **Valid window placements:** A $k_\textrm{h} \times k_\textrm{w}$ window fits inside an $n_\textrm{h} \times n_\textrm{w}$ grid at $n - k + 1$ positions per axis.
<!--tr:891243ae-->
- **Posições válidas da janela.** Uma janela $k_\textrm{h} \times k_\textrm{w}$ cabe em uma grade $n_\textrm{h} \times n_\textrm{w}$ em $n - k + 1$ posições por eixo.

<!--en:55cc8c32 kind=slide_text-->
- **Shape examples:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.
<!--tr:55cc8c32-->
- **Exemplos de formas.** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$) e $28 \times 28 \star 5 \times 5 \to 24 \times 24$.

<!--en:1e3463da kind=slide_title-->
::: {.slide title="Implementing 2D Cross-Correlation in Code"}
<!--tr:1e3463da-->
::: {.slide title="Implementação da correlação cruzada 2D"}

<!--en:07566992 kind=kicker-->
[Direct Nested-Loop Implementation]{.kicker}
<!--tr:07566992-->
[Implementação direta com laços aninhados]{.kicker}

<!--en:202aa5d3 kind=slide_text-->
The function `corr2d` allocates the $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ output tensor and computes each entry as the sum of `X[i:i+h, j:j+w] * K`:
<!--tr:202aa5d3-->
A função `corr2d` aloca o tensor de saída de forma $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ e calcula cada elemento somando os produtos em `X[i:i+h, j:j+w] * K`.

<!--en:bef130ba kind=slide_text-->
Each iteration slices a $k_\textrm{h} \times k_\textrm{w}$ sub-window of `X`, multiplies elementwise with `K`, and reduces via `sum`.
<!--tr:bef130ba-->
A cada iteração, extraímos uma subjanela $k_\textrm{h} \times k_\textrm{w}$ de `X`, multiplicamos elemento a elemento por `K` e somamos com `sum`.

<!--en:4ee9f6b1 kind=slide_title-->
::: {.slide title="Verifying corr2d and the Flipped-Kernel Equivalence"}
<!--tr:4ee9f6b1-->
::: {.slide title="Verificação de corr2d e equivalência com o kernel invertido"}

<!--en:2b0be423 kind=kicker-->
[Cross-Correlation vs. Mathematical Convolution]{.kicker}
<!--tr:2b0be423-->
[Comparação entre correlação cruzada e convolução matemática]{.kicker}

<!--en:74b9b745 kind=slide_text-->
- **180° flipped kernel $\mathbf{K}'$:** Cross-correlating with $\mathbf{K}$ equals convolving with $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.
<!--tr:74b9b745-->
- **Kernel $\mathbf{K}'$ girado 180°.** A correlação cruzada com $\mathbf{K}$ equivale à convolução com $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.

<!--en:5c8af7cd kind=slide_text-->
- **Identical learned class:** Frameworks compute cross-correlation and call it convolution.
<!--tr:5c8af7cd-->
- **Mesma família de camadas aprendidas.** Os frameworks calculam a correlação cruzada e a chamam de convolução.

<!--en:a69e29b5 kind=slide_title-->
::: {.slide title="Building a 2D Convolutional Layer Module"}
<!--tr:a69e29b5-->
::: {.slide title="Implementação de um módulo de camada convolucional 2D"}

<!--en:55763603 kind=kicker-->
[Learnable Kernel Weights Plus Scalar Bias]{.kicker}
<!--tr:55763603-->
[Pesos ajustáveis do kernel e viés escalar]{.kicker}

<!--en:5b8bad24 kind=slide_text-->
A convolutional layer wraps `corr2d` with a learnable $k_\textrm{h} \times k_\textrm{w}$ weight parameter (initialized from $\mathcal{N}(0, 0.01^2)$) and a scalar bias (initialized to $0$):
<!--tr:5b8bad24-->
Uma camada convolucional encapsula `corr2d` com um parâmetro de pesos ajustáveis de forma $k_\textrm{h} \times k_\textrm{w}$ (inicializado com amostras de $\mathcal{N}(0, 0.01^2)$) e um viés escalar (inicializado em $0$).

<!--en:c16d97d6 kind=slide_title-->
::: {.slide title="Resolution Independence of Convolutional Layers"}
<!--tr:c16d97d6-->
::: {.slide title="Camadas convolucionais independem da resolução"}

<!--en:a7d1e48f kind=kicker-->
[Fixed Parameters Across Arbitrary Input Sizes]{.kicker}
<!--tr:a7d1e48f-->
[Mesmos parâmetros para qualquer tamanho de entrada]{.kicker}

<!--en:53a2bce7 kind=slide_text-->
- **Decoupled from $H \times W$:** The exact same 5 parameters ($2 \times 2$ kernel + 1 bias) process $3 \times 3$, $6 \times 8$, or megapixel inputs without reshaping weights.
<!--tr:53a2bce7-->
- **Independência de $H \times W$.** Os mesmos 5 parâmetros (kernel $2 \times 2$ + 1 viés) processam entradas $3 \times 3$, $6 \times 8$ ou com megapixels sem mudar a forma dos pesos.

<!--en:017fec24 kind=slide_title-->
::: {.slide title="Constructing a Synthetic Vertical-Edge Image"}
<!--tr:017fec24-->
::: {.slide title="Construção de uma imagem sintética com bordas verticais"}

<!--en:896e641c kind=kicker-->
[A 6×8 Binary Test Pattern]{.kicker}
<!--tr:896e641c-->
[Padrão binário de teste 6×8]{.kicker}

<!--en:0a4b456b kind=slide_text-->
We construct the $6 \times 8$ test image `X` with ones on the left and right borders and zeros across the middle four columns (`X[:, 2:6] = 0`):
<!--tr:0a4b456b-->
Montamos a imagem de teste $6 \times 8$ `X` com valores um nas bordas esquerda e direita e zero nas quatro colunas centrais (`X[:, 2:6] = 0`).

<!--en:dfc7e3e3 kind=slide_text-->
- **Piecewise-constant columns:** Intensity is constant (`1` or `0`) within columns $0\text{--}1$, $2\text{--}5$, and $6\text{--}7$, jumping only between columns $1 \to 2$ and $5 \to 6$.
<!--tr:dfc7e3e3-->
- **Colunas constantes por faixa.** A intensidade é constante (`1` ou `0`) nas faixas $0\text{--}1$, $2\text{--}5$ e $6\text{--}7$, e só muda nas transições $1 \to 2$ e $5 \to 6$.

<!--en:3ce31cbe kind=slide_title-->
::: {.slide title="Applying the Horizontal Finite-Difference Kernel"}
<!--tr:3ce31cbe-->
::: {.slide title="Aplicação do kernel de diferença finita horizontal"}

<!--en:b37a1029 kind=kicker-->
[Detecting Vertical Edges with K = [1, -1]]{.kicker}
<!--tr:b37a1029-->
[Detecção de bordas verticais com K = [1, -1]]{.kicker}

<!--en:9be85019 kind=slide_text-->
Cross-correlating `X` with the $1 \times 2$ kernel $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ computes $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ at every horizontal pair:
<!--tr:9be85019-->
A correlação cruzada de `X` com o kernel $1 \times 2$ $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ calcula $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ para cada par de pixels vizinhos na horizontal.

<!--en:f649cfae kind=slide_text-->
- **Signed edge map:** Column 1 fires `+1` (white-to-black transition), column 5 fires `-1` (black-to-white transition), and all constant interior columns evaluate to `0`.
<!--tr:f649cfae-->
- **Mapa de bordas com sinal.** A coluna 1 dá `+1` (do branco para o preto), a coluna 5 dá `-1` (do preto para o branco) e todas as colunas internas de intensidade constante dão `0`.

<!--en:e6d155be kind=slide_title-->
::: {.slide title="Orientation Selectivity of Edge Kernels"}
<!--tr:e6d155be-->
::: {.slide title="Seletividade dos kernels de borda à orientação"}

<!--en:b525cde4 kind=kicker-->
[Why One Kernel Cannot Detect All Orientations]{.kicker}
<!--tr:b525cde4-->
[Por que um só kernel não detecta todas as orientações]{.kicker}

<!--en:38159de6 kind=slide_text-->
- **Blind to horizontal edges:** Transposing `X` (`X.T`) makes horizontal neighbors identical, so $[1, -1]$ outputs all zeros.
<!--tr:38159de6-->
- **Sem resposta a bordas horizontais.** Ao transpor `X` (`X.T`), os vizinhos na horizontal ficam idênticos, e a saída de $[1, -1]$ contém apenas zeros.

<!--en:f0f30f76 kind=slide_text-->
- **Filter banks (§{sec_channels}):** Vision layers learn parallel channels for horizontal, vertical, and diagonal orientations.
<!--tr:f0f30f76-->
- **Bancos de filtros (§{sec_channels}).** Camadas de modelos de visão aprendem canais em paralelo para orientações horizontais, verticais e diagonais.

<!--en:b755e31b kind=slide_title-->
::: {.slide title="Learning an Edge Detector via Gradient Descent"}
<!--tr:b755e31b-->
::: {.slide title="Aprendizado de um detector de bordas por descida de gradiente"}

<!--en:86bbeea7 kind=kicker-->
[Supervising a Built-In Convolutional Layer on (X, Y)]{.kicker}
<!--tr:86bbeea7-->
[Supervisão de uma camada convolucional nativa com (X, Y)]{.kicker}

<!--en:a569780c kind=slide_text-->
- **4D framework tensors:** Reshape `X` and `Y` to 4D tensors (one example, one channel) and minimize $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ at $\eta = 0.03$.
<!--tr:a569780c-->
- **Tensores 4D no framework.** Ajuste a forma de `X` e `Y` para obter tensores 4D (um exemplo, um canal) e minimize $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ com $\eta = 0.03$.

<!--en:2489ce6e kind=slide_title-->
::: {.slide title="Inspecting the Learned Kernel Weights"}
<!--tr:2489ce6e-->
::: {.slide title="Inspeção dos pesos aprendidos do kernel"}

<!--en:7e4ef44b kind=kicker-->
[Why Initial Loss Is 12 and Converges to [1, -1]]{.kicker}
<!--tr:7e4ef44b-->
[Por que a perda começa em 12 e os pesos convergem para [1, -1]]{.kicker}

<!--en:9cd1c645 kind=slide_text-->
- **Why loss starts at $12.0$:** Near-zero weights predict $\hat{Y} \approx 0$, missing the twelve $\pm 1$ edge pixels in `Y` ($12 \times 1^2 = 12$).
<!--tr:9cd1c645-->
- **Por que a perda começa em $12.0$.** Pesos próximos de zero predizem $\hat{Y} \approx 0$ e erram nos doze pixels de borda de valor $\pm 1$ em `Y` ($12 \times 1^2 = 12$).

<!--en:456670c5 kind=slide_text-->
- **Convergence in 9 steps:** SGD recovers $[0.99, -0.99]$ automatically from data.
<!--tr:456670c5-->
- **Convergência em 9 passos.** A descida de gradiente recupera $[0.99, -0.99]$ automaticamente a partir dos dados.

=== DECISIONS ===
- Usar português brasileiro com “nós” implícito nas explicações, imperativo nos exercícios e maiúscula inicial nos títulos. Manter as casas decimais e os pontos decimais do original.
- Usar os termos do glossário sem duplicá-los em inglês. Manter siglas consagradas em inglês quando usadas, além de nomes de ferramentas e identificadores. Usar “aprendizado profundo” para “deep learning” e “framework” para a biblioteca de aprendizado profundo.
- “input” / “output” → “entrada” / “saída”; “entry” (de tensor ou matriz) → “elemento”; “patch” → “recorte”; “sliding window” → “janela deslizante”.
- “edge detection” / “edge detector” / “edge map” → “detecção de bordas” / “detector de bordas” / “mapa de bordas”.
- “locality” → “localidade”; “dense layer” → “camada densa”; “end to end” → “de ponta a ponta”; “flipped kernel” → “kernel invertido”; “filter bank” → “banco de filtros”.
- “loop” → “laço”; “built-in layer” → “camada nativa”; “reshape” → “mudar/ajustar a forma” (sem alterar os elementos).
- “brightness” → “brilho”; “intensity” → “intensidade”; “stripe” / “band” → “faixa”.

=== NOTES ===
- 71c591a4: a seletividade estrita vale no teste com as duas orientações mostradas. O kernel também responde a variações horizontais de intensidade em bordas oblíquas. A primeira frase explicita o escopo do teste.
- 7e4ef44b: o original atribui a convergência para [1, -1] à perda. O par representa os pesos do kernel.
- 456670c5: o original diz “SGD”, mas o laço atualiza os pesos com o gradiente do erro somado sobre toda a saída, sem amostragem. A tradução usa “descida de gradiente”.

