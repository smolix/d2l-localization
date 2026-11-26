<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u02 lang=pt rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:0b85caf9 kind=heading-->
## Convolution as Matrix Multiplication
<!--tr:0b85caf9-->
## Convolução como multiplicação de matrizes

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
Embora os laços aninhados em `corr2d` tornem a matemática transparente, percorrer um pixel de saída por vez em Python é lento demais para cargas de trabalho reais. Uma solução clássica, ainda hoje entre os algoritmos escolhidos pelas bibliotecas, é reformular a correlação cruzada como uma única grande multiplicação de matrizes, exatamente a operação que as GPUs executam em velocidade máxima. Observe que cada elemento de saída na :numref:`fig_correlation` é simplesmente o produto escalar do kernel achatado de $k_\textrm{h} k_\textrm{w}$ elementos pela janela de entrada achatada correspondente. Extrair todas as $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ janelas e empilhá-las como linhas produz uma matriz de trechos; multiplicá-la pelo kernel achatado calcula toda a correlação cruzada de uma só vez. Essa transformação padrão é chamada de *im2col* ("imagem para colunas", ou linhas, neste caso). A :numref:`fig_conv_im2col` realiza essa transformação no exemplo da :numref:`fig_correlation`.

<!--en:107554a5 kind=caption-->
![The cross-correlation of :numref:`fig_correlation` as one matrix product. Each of the four $2 \times 2$ windows of the input becomes a row of the patch matrix, and multiplying by the flattened kernel gives the flattened output. The shaded cells follow the first window to its row and to the output 19. The center pixel 4 lies in all four windows, so it appears in every row.](../img/conv-im2col.svg)
<!--tr:107554a5-->
![A correlação cruzada da :numref:`fig_correlation` como um único produto de matrizes. Cada uma das quatro janelas $2 \times 2$ da entrada torna-se uma linha da matriz de trechos, e a multiplicação pelo kernel achatado fornece a saída achatada. As células sombreadas acompanham a primeira janela até sua linha e até a saída 19. O pixel central 4 está nas quatro janelas, portanto aparece em todas as linhas.](../img/conv-im2col.svg)

<!--en:76713479 kind=paragraph-->
The trade-off is extra workspace memory, since each interior pixel is duplicated
into up to $k_\textrm{h} k_\textrm{w}$ patches. Below we verify on
:numref:`fig_correlation` that this single matrix multiply matches `corr2d`
exactly:
<!--tr:76713479-->
O custo é memória adicional de trabalho, pois cada pixel interno é duplicado em até $k_\textrm{h} k_\textrm{w}$ trechos. A seguir, verificamos na :numref:`fig_correlation` que essa única multiplicação de matrizes corresponde exatamente a `corr2d`:

<!--en:3bcd7d8a kind=paragraph-->
The discrepancy is identically zero: one matrix product reproduces `corr2d`
perfectly.
<!--tr:3bcd7d8a-->
A discrepância é identicamente zero: um produto de matrizes reproduz `corr2d` perfeitamente.

<!--en:c30f2dea kind=paragraph-->
Equivalently, we can unroll the *kernel* instead of the input into a sparse
banded matrix $\mathbf{W}$ multiplying the flattened image—the exact 2D analog
of panel (c) in :numref:`fig_why_conv_weight_sharing`, where every row repeats
the same handful of kernel weights. Seeing a convolution as a structured dense
layer brings us full circle to the derivation of :numref:`sec_why-conv` (and the
exercises build $\mathbf{W}$ explicitly).
<!--tr:c30f2dea-->
De modo equivalente, podemos desdobrar o *kernel*, em vez da entrada, em uma matriz esparsa em bandas $\mathbf{W}$ que multiplica a imagem achatada — o análogo 2D exato do painel (c) da :numref:`fig_why_conv_weight_sharing`, em que cada linha repete o mesmo pequeno conjunto de pesos do kernel. Enxergar uma convolução como uma camada densa estruturada nos traz de volta à dedução de :numref:`sec_why-conv` (e os exercícios constroem $\mathbf{W}$ explicitamente).

<!--en:ad4fc956 kind=heading-->
## Feature Maps and Receptive Fields
<!--tr:ad4fc956-->
## Mapas de características e campos receptivos

<!--en:e7983df8 kind=paragraph-->
A single convolutional layer sees only a patch the size of its kernel, whereas
recognizing whole objects demands integrating context across much wider areas.
Recall from :numref:`subsec_why-conv-channels` that a convolutional layer's
output is called a *feature map* (like our vertical edge map `Y`). For any
element $x$ in a feature map, its *receptive field* is the region of the input
image (or of an earlier layer) whose values can influence $x$ during forward
propagation—borrowing the neurophysiology term from :numref:`field_visual`.
<!--tr:e7983df8-->
Uma única camada convolucional vê apenas um trecho do tamanho de seu kernel, enquanto reconhecer objetos inteiros exige integrar contexto em áreas muito maiores. Lembre-se de :numref:`subsec_why-conv-channels` de que a saída de uma camada convolucional é chamada de *mapa de características* (como nosso mapa de bordas verticais `Y`). Para qualquer elemento $x$ em um mapa de características, seu *campo receptivo* é a região da imagem de entrada (ou de uma camada anterior) cujos valores podem influenciar $x$ durante a passagem direta — tomando emprestado o termo da neurofisiologia de :numref:`field_visual`.

<!--en:b36f221e kind=heading-->
### Stacked Layers
<!--tr:b36f221e-->
### Camadas empilhadas

<!--en:24a25ec4 kind=paragraph-->
In :numref:`fig_correlation`, the shaded output `19` has a $2 \times 2$ receptive
field covering the top-left four pixels of $\mathbf{X}$. Now stack a *second*
$2 \times 2$ convolutional layer on top of the $2 \times 2$ output $\mathbf{Y}$
to produce a single scalar $z$: $z$ depends on all four elements of $\mathbf{Y}$,
which together depend on all nine elements of $\mathbf{X}$. Stacking layers
expands the receptive field smoothly with depth.
<!--tr:24a25ec4-->
Na :numref:`fig_correlation`, a saída sombreada `19` tem um campo receptivo $2 \times 2$ que cobre os quatro pixels superiores esquerdos de $\mathbf{X}$. Agora empilhe uma *segunda* camada convolucional $2 \times 2$ sobre a saída $2 \times 2$ $\mathbf{Y}$ para produzir um único escalar $z$: $z$ depende dos quatro elementos de $\mathbf{Y}$, que juntos dependem dos nove elementos de $\mathbf{X}$. Empilhar camadas expande gradualmente o campo receptivo com a profundidade.

<!--en:f95c9394 kind=paragraph-->
How quickly does it grow? Each $k \times k$ convolutional layer reaches $k - 1$
pixels farther than a single element of its input, so stacking $L$ layers with
kernel sizes $k_1, \ldots, k_L$ yields a receptive field of side length
<!--tr:f95c9394-->
Com que rapidez ele cresce? Cada camada convolucional $k \times k$ alcança $k - 1$ pixels além de um único elemento de sua entrada, portanto empilhar $L$ camadas com tamanhos de kernel $k_1, \ldots, k_L$ produz um campo receptivo com comprimento de lado

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
Para uma pilha de $L$ camadas $3 \times 3$ consecutivas, cada camada adiciona 2 pixels, cobrindo $(2L + 1) \times (2L + 1)$ entradas: $5 \times 5$ após duas camadas e $7 \times 7$ após três (:numref:`fig_conv_receptive_field`). Esse crescimento linear é, na verdade, uma vantagem, e não um motivo para recorrer a kernels enormes: três camadas $3 \times 3$ empilhadas alcançam o mesmo campo receptivo $7 \times 7$ que uma única camada $7 \times 7$ usando muito menos parâmetros ($3 \times 9 = 27$ pesos contra 49 por par de canais), ao mesmo tempo que intercalam três não linearidades em vez de uma.

<!--en:39917513 kind=caption-->
![Receptive field of one unit after three one-dimensional layers with kernel size 3. Shaded cells are the units that the top unit depends on, and the lines connect each of them to its inputs. Each layer widens the receptive field by 2, to 7 inputs.](../img/conv-receptive-field.svg)
<!--tr:39917513-->
![Campo receptivo de uma unidade após três camadas unidimensionais com kernel de tamanho 3. As células sombreadas são as unidades das quais a unidade superior depende, e as linhas conectam cada uma delas às suas entradas. Cada camada amplia o campo receptivo em 2, até chegar a 7 entradas.](../img/conv-receptive-field.svg)

<!--en:48e4c9cc kind=heading-->
### Effective Receptive Fields
<!--tr:48e4c9cc-->
### Campos receptivos efetivos

<!--en:77d044db kind=paragraph-->
Keep in mind that $r$ measures the *theoretical* receptive field (every input
pixel with *any* path to an output unit) without saying how strongly each pixel
actually contributes. Measuring input gradients directly,
:citet:`Luo.Li.Urtasun.ea.2016` showed that the *effective* receptive field
decays sharply away from the center in a roughly Gaussian bell curve, in both
untrained and trained networks. Deep units can theoretically see far across the
image, but their sensitivity concentrates heavily near the center.
<!--tr:77d044db-->
Tenha em mente que $r$ mede o campo receptivo *teórico* (cada pixel de entrada com *algum* caminho até uma unidade de saída), sem indicar a intensidade da contribuição de cada pixel. Medindo diretamente os gradientes de entrada, :citet:`Luo.Li.Urtasun.ea.2016` mostraram que o campo receptivo *efetivo* decai acentuadamente à medida que se afasta do centro, seguindo uma curva em sino aproximadamente gaussiana, tanto em redes não treinadas quanto em redes treinadas. Unidades profundas podem teoricamente enxergar longe na imagem, mas sua sensibilidade se concentra fortemente perto do centro.

<!--en:eed4e2a0 kind=paragraph-->
Counting paths explains the bell shape. An input at the edge of the receptive
field reaches the output unit along a single chain of connections, whereas the
center input can take many routes through the layers in between.
:numref:`fig_conv_rf_paths` counts them for the tower of
:numref:`fig_conv_receptive_field`: the center input has seven paths to the top
unit and each edge input has one. If all weights have a similar magnitude, the
influence of an input grows with its number of paths.
<!--tr:eed4e2a0-->
A contagem de caminhos explica a forma de sino. Uma entrada na borda do campo receptivo alcança a unidade de saída por uma única cadeia de conexões, enquanto a entrada central pode seguir muitos caminhos pelas camadas intermediárias. A :numref:`fig_conv_rf_paths` os conta para a pilha da :numref:`fig_conv_receptive_field`: a entrada central tem sete caminhos até a unidade superior e cada entrada da borda tem um. Se todos os pesos tiverem magnitude semelhante, a influência de uma entrada cresce com seu número de caminhos.

<!--en:b92775ee kind=caption-->
![Number of paths from each unit to the top unit of :numref:`fig_conv_receptive_field` (three one-dimensional layers with kernel size 3). Each count is the sum of the counts of the units that the cell feeds. Darker cells have more paths than the other cells of their layer, and the bars repeat the counts of the input row: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)
<!--tr:b92775ee-->
![Número de caminhos de cada unidade até a unidade superior da :numref:`fig_conv_receptive_field` (três camadas unidimensionais com kernel de tamanho 3). Cada contagem é a soma das contagens das unidades que recebem a saída da célula. Células mais escuras têm mais caminhos que as demais células de sua camada, e as barras repetem as contagens da linha de entrada: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)

<!--en:405695e8 kind=paragraph-->
Two practical gaps remain in our basic layer: every unpadded convolution shrinks
spatial dimensions while under-sampling border pixels, and a single kernel
detects only one pattern. We address spatial sizing and borders next in
:numref:`sec_padding`, and multi-channel feature maps in :numref:`sec_channels`.
<!--tr:405695e8-->
Restam duas limitações práticas em nossa camada básica: cada convolução sem preenchimento reduz as dimensões espaciais e subamostra os pixels da borda, e um único kernel detecta apenas um padrão. Trataremos a seguir do tamanho espacial e das bordas em :numref:`sec_padding`, e dos mapas de características multicanal em :numref:`sec_channels`.

<!--en:30ac03ff kind=heading-->
## Summary
<!--tr:30ac03ff-->
## Resumo

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
Uma camada convolucional aplica correlação cruzada bidimensional com um kernel aprendido e um viés escalar, deslizando o kernel por todas as janelas válidas da entrada. Um simples kernel de diferença finita $1 \times 2$ `[1, -1]` detecta bordas verticais em qualquer lugar de uma imagem, e a descida de gradiente recupera esses mesmos pesos apenas dos dados de entrada–saída, assim como CNNs profundas aprendem hierarquias inteiras de filtros de ponta a ponta. A convolução matemática inverte o kernel, o que não faz diferença quando os kernels são aprendidos. Uma convolução pode ser calculada como uma única multiplicação de matrizes (*im2col*), e empilhar $L$ camadas convolucionais amplia linearmente o campo receptivo de cada unidade em $k_i - 1$ por camada.

<!--en:3ba929d2 kind=slide_title-->
::: {.slide title="Fast Execution: Convolution via Matrix Multiplication"}
<!--tr:3ba929d2-->
::: {.slide title="Execução rápida: convolução por multiplicação de matrizes"}

<!--en:1711f812 kind=kicker-->
[The im2col (Image-to-Columns) Transformation]{.kicker}
<!--tr:1711f812-->
[A transformação im2col (imagem para colunas)]{.kicker}

<!--en:a040f3d7 kind=slide_text-->
- **Single dense GEMM:** Flatten each $k_\textrm{h} \times k_\textrm{w}$ window into a row of `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ and multiply by $\textrm{vec}(\mathbf{K})$.
<!--tr:a040f3d7-->
- **Uma única GEMM densa:** Achate cada janela $k_\textrm{h} \times k_\textrm{w}$ em uma linha de `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ e multiplique por $\textrm{vec}(\mathbf{K})$.

<!--en:1fb18ec4 kind=slide_text-->
- **Trade-off:** Peak GPU Tensor Core speed at the cost of $k_\textrm{h} k_\textrm{w}\times$ patch workspace duplication.
<!--tr:1fb18ec4-->
- **Compensação:** Velocidade máxima dos Tensor Cores da GPU ao custo de duplicar $k_\textrm{h} k_\textrm{w}\times$ o espaço de trabalho para trechos.

<!--en:0d757cf6 kind=slide_title-->
::: {.slide title="Dual View: Unrolling the Kernel into a Banded Matrix"}
<!--tr:0d757cf6-->
::: {.slide title="Visão dual: desdobrando o kernel em uma matriz em bandas"}

<!--en:7f0defaa kind=kicker-->
[Toeplitz Weight Matrix Multiplication]{.kicker}
<!--tr:7f0defaa-->
[Multiplicação pela matriz de pesos de Toeplitz]{.kicker}

<!--en:5898ba57 kind=slide_text-->
Instead of unfolding the input image into duplicated patches, we can unroll the *kernel* into a sparse banded matrix $\mathbf{W}$ that multiplies the flattened input vector $\textrm{vec}(\mathbf{X})$:
<!--tr:5898ba57-->
Em vez de desdobrar a imagem de entrada em trechos duplicados, podemos desdobrar o *kernel* em uma matriz esparsa em bandas $\mathbf{W}$ que multiplica o vetor de entrada achatado $\textrm{vec}(\mathbf{X})$:

<!--en:dae51741 kind=slide_text-->
- **Matrix–vector form:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, where $\mathbf{W}$ has $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ rows and $n_\textrm{h} n_\textrm{w}$ columns.
<!--tr:dae51741-->
- **Forma matriz–vetor:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, em que $\mathbf{W}$ tem $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ linhas e $n_\textrm{h} n_\textrm{w}$ colunas.

<!--en:f0e86ee3 kind=slide_text-->
- **Connection to §{sec_why-conv} and backprop:** Every row of $\mathbf{W}$ holds the same $k_\textrm{h} k_\textrm{w}$ kernel entries shifted across columns (panel c), making the backward pass with respect to $\mathbf{X}$ simply multiplication by $\mathbf{W}^\top$ (a transposed convolution).
<!--tr:f0e86ee3-->
- **Conexão com §{sec_why-conv} e retropropagação:** Cada linha de $\mathbf{W}$ contém os mesmos $k_\textrm{h} k_\textrm{w}$ elementos do kernel deslocados pelas colunas (painel c), fazendo da passagem para trás em relação a $\mathbf{X}$ uma simples multiplicação por $\mathbf{W}^\top$ (uma convolução transposta).

<!--en:af8cf19a kind=slide_title-->
::: {.slide title="Feature Maps and Receptive Fields Across Depth"}
<!--tr:af8cf19a-->
::: {.slide title="Mapas de características e campos receptivos ao longo da profundidade"}

<!--en:4eb965f3 kind=kicker-->
[Linear Receptive-Field Growth in Stacked Layers]{.kicker}
<!--tr:4eb965f3-->
[Crescimento linear do campo receptivo em camadas empilhadas]{.kicker}

<!--en:5599a6e9 kind=slide_text-->
A unit's **receptive field** is the region of the original input image whose pixels can influence that unit's activation during forward propagation:
<!--tr:5599a6e9-->
O **campo receptivo** de uma unidade é a região da imagem de entrada original cujos pixels podem influenciar a ativação dessa unidade durante a passagem direta:

<!--en:eba74f75 kind=slide_text-->
- **Single layer ($L=1$):** A $k_1 \times k_1$ kernel sees a $k_1 \times k_1$ input patch.
<!--tr:eba74f75-->
- **Uma camada ($L=1$):** Um kernel $k_1 \times k_1$ vê um trecho de entrada $k_1 \times k_1$.

<!--en:01c175bf kind=slide_text-->
- **Stacking $L$ stride-1 layers:** Layer $i$ with kernel size $k_i$ reaches $k_i - 1$ pixels farther along each axis than a single element of layer $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$
<!--tr:01c175bf-->
- **Empilhando $L$ camadas com stride 1:** A camada $i$ com tamanho de kernel $k_i$ alcança $k_i - 1$ pixels além de um único elemento da camada $i-1$ em cada eixo:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$

<!--en:2bde5822 kind=slide_text-->
- **$L$ stacked $3 \times 3$ layers:** Each layer adds $3 - 1 = 2$ pixels, giving side length $r = 2L + 1$ ($3 \to 5 \to 7$).
<!--tr:2bde5822-->
- **$L$ camadas $3 \times 3$ empilhadas:** Cada camada adiciona $3 - 1 = 2$ pixels, resultando em comprimento de lado $r = 2L + 1$ ($3 \to 5 \to 7$).

<!--en:cbb9e97a kind=slide_title-->
::: {.slide title="Why Deep Stacks of 3×3 Kernels Beat One Large Kernel"}
<!--tr:cbb9e97a-->
::: {.slide title="Por que pilhas profundas de kernels 3×3 superam um kernel grande"}

<!--en:95e46e70 kind=kicker-->
[Parameter Efficiency & Nonlinear Expressivity]{.kicker}
<!--tr:95e46e70-->
[Eficiência de parâmetros e expressividade não linear]{.kicker}

<!--en:bd1d7cc0 kind=slide_text-->
- **Matched $7 \times 7$ receptive field:** Three stacked $3 \times 3$ layers ($r = 1 + 2 + 2 + 2 = 7$) see the exact same $7 \times 7$ patch as one $7 \times 7$ layer.
<!--tr:bd1d7cc0-->
- **Mesmo campo receptivo $7 \times 7$:** Três camadas $3 \times 3$ empilhadas ($r = 1 + 2 + 2 + 2 = 7$) veem exatamente o mesmo trecho $7 \times 7$ que uma camada $7 \times 7$.

<!--en:755ac11a kind=slide_text-->
- **45% fewer parameters:** $3 \times 9 c^2 = 27 c^2$ weights instead of $49 c^2$ across $c$ channels.
<!--tr:755ac11a-->
- **45% menos parâmetros:** $3 \times 9 c^2 = 27 c^2$ pesos em vez de $49 c^2$ para $c$ canais.

<!--en:e3aa1086 kind=slide_text-->
- **Richer expressivity:** Compounds **3 nonlinear activations** instead of 1 (the core VGG design rule in §{sec_blocks}).
<!--tr:e3aa1086-->
- **Maior expressividade:** Compõe **3 ativações não lineares** em vez de 1 (a regra central de projeto da VGG em §{sec_blocks}).

<!--en:9ec83bf6 kind=slide_title-->
::: {.slide title="Theoretical vs. Effective Receptive Fields"}
<!--tr:9ec83bf6-->
::: {.slide title="Campos receptivos teóricos versus efetivos"}

<!--en:72a52c3b kind=kicker-->
[Gaussian Decay of Sensitivity Away from the Center]{.kicker}
<!--tr:72a52c3b-->
[Decaimento gaussiano da sensibilidade longe do centro]{.kicker}

<!--en:33bb377d kind=slide_text-->
- **Theoretical RF ($r = 2L+1$):** Bounding box of all pixels with *at least one* path to the output unit.
<!--tr:33bb377d-->
- **Campo receptivo teórico ($r = 2L+1$):** Retângulo delimitador de todos os pixels com *pelo menos um* caminho até a unidade de saída.

<!--en:293436eb kind=slide_text-->
- **Central Limit Theorem:** Center pixels connect via exponentially many paths, whereas corner pixels connect via only **1 path**.
<!--tr:293436eb-->
- **Teorema central do limite:** Pixels centrais se conectam por um número exponencial de caminhos, enquanto pixels dos cantos se conectam por apenas **1 caminho**.

<!--en:f47d181e kind=slide_text-->
- **Effective RF (Luo et al., 2016):** Input gradient magnitude $\partial y / \partial \mathbf{X}$ decays in a Gaussian bell curve with std $\sigma \propto \sqrt{L}$.
<!--tr:f47d181e-->
- **Campo receptivo efetivo (Luo et al., 2016):** A magnitude do gradiente de entrada $\partial y / \partial \mathbf{X}$ decai em uma curva gaussiana com desvio padrão $\sigma \propto \sqrt{L}$.

<!--en:fac792d0 kind=slide_title-->
::: {.slide title="Concept Check: Cross-Correlation & Receptive Fields"}
<!--tr:fac792d0-->
::: {.slide title="Verificação de conceitos: correlação cruzada e campos receptivos"}

<!--en:26a317d8 kind=kicker-->
[Concept Check]{.kicker}
<!--tr:26a317d8-->
[Verificação de conceitos]{.kicker}

<!--en:8fe554e5 kind=slide_text-->
Test your understanding of output shapes, `im2col`, and receptive-field arithmetic:
<!--tr:8fe554e5-->
Teste sua compreensão das formas de saída, de `im2col` e dos cálculos de campos receptivos:

<!--en:281c7c78 kind=slide_text-->
1. **im2col matrix dimensions:** You apply an unpadded $3 \times 3$ single-channel kernel to a $10 \times 10$ image via `im2col`. What is the output spatial shape, and what are the exact dimensions of the `patches` matrix and flattened kernel vector?
<!--tr:281c7c78-->
1. **Dimensões da matriz im2col:** Você aplica um kernel $3 \times 3$ de um canal, sem preenchimento, a uma imagem $10 \times 10$ usando `im2col`. Qual é a forma espacial da saída e quais são as dimensões exatas da matriz `patches` e do vetor do kernel achatado?

<!--en:16976d18 kind=slide_text-->
2. **Depth to span Fashion-MNIST:** At stride 1, how many stacked $3 \times 3$ layers are required so a single unit's theoretical receptive field spans an entire $28 \times 28$ image? How many with $5 \times 5$ layers?
<!--tr:16976d18-->
2. **Profundidade para cobrir Fashion-MNIST:** Com stride 1, quantas camadas $3 \times 3$ empilhadas são necessárias para que o campo receptivo teórico de uma unidade cubra uma imagem $28 \times 28$ inteira? Quantas com camadas $5 \times 5$?

<!--en:ab2e0d6a kind=slide_text-->
3. **Edge-loss boundary shift:** In our $6 \times 8$ vertical-edge learning experiment, why did the loss start at $\approx 12$, and what would the initial loss be if the black band extended all the way to the right edge (`X[:, 2:8] = 0`)?
<!--tr:ab2e0d6a-->
3. **Deslocamento da fronteira e perda de bordas:** Em nosso experimento de aprendizado de bordas verticais $6 \times 8$, por que a perda começou em $\approx 12$ e qual seria a perda inicial se a faixa preta se estendesse até a borda direita (`X[:, 2:8] = 0`)?

<!--en:250d6877 kind=slide_text-->
**Answers:**
(1) Output shape is $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ outputs). The `patches` matrix has shape $(64, 9)$ and multiplies the $(9, 1)$ kernel vector.
(2) For $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ layers. For $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ layers.
(3) Each of the 6 rows had two $\pm 1$ edges ($6 \times 2 \times 1^2 = 12$). Extending the black band to the right border eliminates the right-hand black-to-white edge, leaving one edge per row and halving the initial loss to $\approx 6$.
<!--tr:250d6877-->
**Respostas:**
(1) A forma de saída é $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ saídas). A matriz `patches` tem forma $(64, 9)$ e multiplica o vetor do kernel $(9, 1)$.
(2) Para $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ camadas. Para $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ camadas.
(3) Cada uma das 6 linhas tinha duas bordas $\pm 1$ ($6 \times 2 \times 1^2 = 12$). Estender a faixa preta até a borda direita elimina a borda de preto para branco à direita, deixando uma borda por linha e reduzindo a perda inicial pela metade, para $\approx 6$.

<!--en:ba43a2dd kind=slide_title-->
::: {.slide title="Summary & Key Takeaways"}
<!--tr:ba43a2dd-->
::: {.slide title="Resumo e pontos principais"}

<!--en:ce12e85f kind=kicker-->
[Section Synthesis]{.kicker}
<!--tr:ce12e85f-->
[Síntese da seção]{.kicker}

<!--en:00011fb7 kind=slide_text-->
- **2D Cross-Correlation & Edge Learning:** Slides $\mathbf{K}$ across valid windows ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); SGD learns $[1, -1]$ in 9 steps.
<!--tr:00011fb7-->
- **Correlação cruzada 2D e aprendizado de bordas:** Desliza $\mathbf{K}$ pelas janelas válidas ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); SGD aprende $[1, -1]$ em 9 passos.

<!--en:09efe4f0 kind=slide_text-->
- **im2col Matrix Multiply:** Trades $k_\textrm{h} k_\textrm{w}\times$ patch duplication for peak GPU GEMM throughput.
<!--tr:09efe4f0-->
- **Multiplicação de matrizes im2col:** Troca a duplicação de trechos em $k_\textrm{h} k_\textrm{w}\times$ pela vazão máxima de GEMM na GPU.

<!--en:3af8acd0 kind=slide_text-->
- **Stacked Receptive Fields:** $r = 1 + \sum (k_i - 1) = 2L+1$ for $3 \times 3$ stacks, with Gaussian center sensitivity.
<!--tr:3af8acd0-->
- **Campos receptivos empilhados:** $r = 1 + \sum (k_i - 1) = 2L+1$ para pilhas $3 \times 3$, com sensibilidade gaussiana no centro.
