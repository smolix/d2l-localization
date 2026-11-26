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
Os laços aninhados em `corr2d` deixam claro como o cálculo funciona, mas percorrer um pixel da saída por vez em Python é lento demais para aplicações reais. Uma solução clássica, que as bibliotecas ainda incluem entre os algoritmos disponíveis, é reescrever a correlação cruzada como uma única multiplicação de matrizes grandes, justamente a operação em que as GPUs atingem sua velocidade máxima. Observe que cada elemento da saída em :numref:`fig_correlation` é simplesmente o produto escalar entre o kernel de $k_\textrm{h} k_\textrm{w}$ elementos e a janela correspondente da entrada, ambos convertidos em vetores. Se extrairmos todas as $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ janelas e as empilharmos como linhas, obteremos uma matriz de trechos. Multiplicá-la pelo vetor do kernel calcula toda a correlação cruzada de uma só vez. Essa reorganização dos dados é conhecida como *im2col* (“de imagem para colunas”, ou, no nosso caso, para linhas). A :numref:`fig_conv_im2col` mostra essa transformação para o exemplo da :numref:`fig_correlation`.

<!--en:107554a5 kind=caption-->
![The cross-correlation of :numref:`fig_correlation` as one matrix product. Each of the four $2 \times 2$ windows of the input becomes a row of the patch matrix, and multiplying by the flattened kernel gives the flattened output. The shaded cells follow the first window to its row and to the output 19. The center pixel 4 lies in all four windows, so it appears in every row.](../img/conv-im2col.svg)
<!--tr:107554a5-->
![A correlação cruzada da :numref:`fig_correlation` expressa por um único produto de matrizes. Cada uma das quatro janelas $2 \times 2$ da entrada se torna uma linha da matriz de trechos. A multiplicação pelo kernel em forma de vetor dá a saída também em forma de vetor. As células sombreadas mostram a primeira janela, a linha correspondente e o valor de saída 19. O pixel central de valor 4 pertence às quatro janelas e, por isso, aparece em todas as linhas.](../img/conv-im2col.svg)

<!--en:76713479 kind=paragraph-->
The trade-off is extra workspace memory, since each interior pixel is duplicated
into up to $k_\textrm{h} k_\textrm{w}$ patches. Below we verify on
:numref:`fig_correlation` that this single matrix multiply matches `corr2d`
exactly:
<!--tr:76713479-->
O custo é a memória de trabalho adicional, pois cada pixel do interior da imagem aparece em até $k_\textrm{h} k_\textrm{w}$ trechos. Vamos verificar, com o exemplo da :numref:`fig_correlation`, que essa única multiplicação de matrizes produz exatamente o mesmo resultado que `corr2d`:

<!--en:3bcd7d8a kind=paragraph-->
The discrepancy is identically zero: one matrix product reproduces `corr2d`
perfectly.
<!--tr:3bcd7d8a-->
A diferença é exatamente zero em todas as posições. Um único produto de matrizes reproduz perfeitamente `corr2d`.

<!--en:c30f2dea kind=paragraph-->
Equivalently, we can unroll the *kernel* instead of the input into a sparse
banded matrix $\mathbf{W}$ multiplying the flattened image—the exact 2D analog
of panel (c) in :numref:`fig_why_conv_weight_sharing`, where every row repeats
the same handful of kernel weights. Seeing a convolution as a structured dense
layer brings us full circle to the derivation of :numref:`sec_why-conv` (and the
exercises build $\mathbf{W}$ explicitly).
<!--tr:c30f2dea-->
De forma equivalente, podemos desdobrar o *kernel*, em vez da entrada, em uma matriz de banda esparsa $\mathbf{W}$ que multiplica a imagem em forma de vetor. É o análogo bidimensional exato do painel (c) da :numref:`fig_why_conv_weight_sharing`, em que cada linha repete o mesmo pequeno conjunto de pesos do kernel. Ao interpretar a convolução como uma camada densa estruturada, voltamos à dedução de :numref:`sec_why-conv`. Nos exercícios, construiremos $\mathbf{W}$ explicitamente.

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
Uma única camada convolucional enxerga apenas um trecho do tamanho de seu kernel, mas reconhecer objetos inteiros exige reunir informações de uma área muito maior. Como vimos em :numref:`subsec_why-conv-channels`, a saída de uma camada convolucional é chamada de *mapa de características* (como o nosso mapa de bordas verticais `Y`). Para qualquer elemento $x$ de um mapa de características, seu *campo receptivo* é a região da imagem de entrada (ou de uma camada anterior) cujos valores podem influenciar $x$ durante a passagem direta. O termo vem da neurofisiologia, como vimos em :numref:`field_visual`.

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
Na :numref:`fig_correlation`, o elemento sombreado da saída, de valor `19`, tem um campo receptivo $2 \times 2$ que abrange os quatro pixels do canto superior esquerdo de $\mathbf{X}$. Vamos empilhar uma *segunda* camada convolucional $2 \times 2$ sobre a saída de forma $2 \times 2$ ($\mathbf{Y}$) para produzir um único escalar $z$. Esse escalar $z$ depende dos quatro elementos de $\mathbf{Y}$, que, em conjunto, dependem dos nove elementos de $\mathbf{X}$. Ao empilhar camadas, ampliamos o campo receptivo gradualmente com a profundidade.

<!--en:f95c9394 kind=paragraph-->
How quickly does it grow? Each $k \times k$ convolutional layer reaches $k - 1$
pixels farther than a single element of its input, so stacking $L$ layers with
kernel sizes $k_1, \ldots, k_L$ yields a receptive field of side length
<!--tr:f95c9394-->
Com que rapidez ele cresce? Cada camada convolucional $k \times k$ alcança $k - 1$ pixels a mais em cada eixo do que um único elemento de sua entrada. Assim, ao empilhar $L$ camadas com kernels de tamanhos $k_1, \ldots, k_L$, obtemos um campo receptivo cujo lado mede

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
Em uma pilha de $L$ camadas $3 \times 3$ consecutivas, cada camada acrescenta 2 pixels ao lado do campo receptivo, que passa a abranger $(2L + 1) \times (2L + 1)$ posições da entrada: $5 \times 5$ após duas camadas e $7 \times 7$ após três (:numref:`fig_conv_receptive_field`). Esse crescimento linear é uma vantagem, e não um motivo para recorrer a kernels enormes. Três camadas $3 \times 3$ empilhadas atingem o mesmo campo receptivo $7 \times 7$ que uma única camada $7 \times 7$ com muito menos parâmetros ($3 \times 9 = 27$ pesos em vez de 49 por par de canais) e permitem intercalar três não linearidades em vez de uma.

<!--en:39917513 kind=caption-->
![Receptive field of one unit after three one-dimensional layers with kernel size 3. Shaded cells are the units that the top unit depends on, and the lines connect each of them to its inputs. Each layer widens the receptive field by 2, to 7 inputs.](../img/conv-receptive-field.svg)
<!--tr:39917513-->
![Campo receptivo de uma unidade após três camadas unidimensionais com tamanho do kernel igual a 3. As células sombreadas são as unidades das quais a unidade do topo depende, e as linhas ligam cada uma delas às suas entradas. Cada camada amplia o campo receptivo em 2, até abranger 7 entradas.](../img/conv-receptive-field.svg)

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
Lembre-se de que $r$ mede o campo receptivo *teórico*, que inclui todo pixel da entrada para o qual exista *algum* caminho até uma unidade de saída. Essa medida não indica quanto cada pixel de fato contribui. Medindo diretamente os gradientes em relação à entrada, :citet:`Luo.Li.Urtasun.ea.2016` mostraram que, no campo receptivo *efetivo*, a influência diminui rapidamente à medida que nos afastamos do centro, seguindo uma curva aproximadamente gaussiana, em forma de sino, tanto em redes treinadas quanto em redes sem treinamento. Em teoria, as unidades das camadas profundas enxergam regiões distantes da imagem, mas sua sensibilidade se concentra fortemente perto do centro.

<!--en:eed4e2a0 kind=paragraph-->
Counting paths explains the bell shape. An input at the edge of the receptive
field reaches the output unit along a single chain of connections, whereas the
center input can take many routes through the layers in between.
:numref:`fig_conv_rf_paths` counts them for the tower of
:numref:`fig_conv_receptive_field`: the center input has seven paths to the top
unit and each edge input has one. If all weights have a similar magnitude, the
influence of an input grows with its number of paths.
<!--tr:eed4e2a0-->
A contagem dos caminhos explica a forma de sino. Uma entrada na borda do campo receptivo chega à unidade de saída por uma única cadeia de conexões, enquanto a entrada central pode seguir muitos caminhos pelas camadas intermediárias. A :numref:`fig_conv_rf_paths` mostra essa contagem para a pilha de camadas da :numref:`fig_conv_receptive_field`. Da entrada central até a unidade do topo há sete caminhos, e de cada entrada da borda há apenas um. Se todos os pesos tiverem magnitudes semelhantes, a influência de uma entrada cresce com o número de caminhos que partem dela.

<!--en:b92775ee kind=caption-->
![Number of paths from each unit to the top unit of :numref:`fig_conv_receptive_field` (three one-dimensional layers with kernel size 3). Each count is the sum of the counts of the units that the cell feeds. Darker cells have more paths than the other cells of their layer, and the bars repeat the counts of the input row: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)
<!--tr:b92775ee-->
![Número de caminhos de cada unidade até a unidade do topo da :numref:`fig_conv_receptive_field` (três camadas unidimensionais com tamanho do kernel igual a 3). A contagem de cada célula é a soma das contagens das unidades que recebem sua saída. As células mais escuras têm mais caminhos que as demais células de sua camada. As barras reproduzem as contagens da linha de entrada: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)

<!--en:405695e8 kind=paragraph-->
Two practical gaps remain in our basic layer: every unpadded convolution shrinks
spatial dimensions while under-sampling border pixels, and a single kernel
detects only one pattern. We address spatial sizing and borders next in
:numref:`sec_padding`, and multi-channel feature maps in :numref:`sec_channels`.
<!--tr:405695e8-->
Nossa camada básica ainda tem duas limitações práticas. Toda convolução sem preenchimento reduz as dimensões espaciais e amostra os pixels das bordas com menor frequência. Além disso, um único kernel detecta apenas um padrão. A seguir, estudaremos as dimensões espaciais e as bordas em :numref:`sec_padding` e os mapas de características com múltiplos canais em :numref:`sec_channels`.

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
Uma camada convolucional desliza um kernel aprendido por todas as janelas válidas da entrada, calcula a correlação cruzada bidimensional e soma um viés escalar ao resultado. Um simples kernel $1 \times 2$ de diferença finita, `[1, -1]`, detecta bordas verticais em qualquer região da imagem. A descida de gradiente recupera esses mesmos pesos usando apenas dados de entrada e saída, assim como as CNNs profundas aprendem hierarquias inteiras de filtros de ponta a ponta. A convolução matemática inverte o kernel, o que não faz diferença quando os kernels são aprendidos. Podemos calcular uma convolução com uma única multiplicação de matrizes (*im2col*). Ao empilhar $L$ camadas convolucionais, o campo receptivo de cada unidade cresce linearmente, $k_i - 1$ por camada.

<!--en:3ba929d2 kind=slide_title-->
::: {.slide title="Fast Execution: Convolution via Matrix Multiplication"}
<!--tr:3ba929d2-->
::: {.slide title="Execução rápida da convolução por multiplicação de matrizes"}

<!--en:1711f812 kind=kicker-->
[The im2col (Image-to-Columns) Transformation]{.kicker}
<!--tr:1711f812-->
[A transformação im2col (de imagem para colunas)]{.kicker}

<!--en:a040f3d7 kind=slide_text-->
- **Single dense GEMM:** Flatten each $k_\textrm{h} \times k_\textrm{w}$ window into a row of `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ and multiply by $\textrm{vec}(\mathbf{K})$.
<!--tr:a040f3d7-->
- **Uma única GEMM densa:** converta cada janela $k_\textrm{h} \times k_\textrm{w}$ em uma linha de `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ e multiplique a matriz por $\textrm{vec}(\mathbf{K})$.

<!--en:1fb18ec4 kind=slide_text-->
- **Trade-off:** Peak GPU Tensor Core speed at the cost of $k_\textrm{h} k_\textrm{w}\times$ patch workspace duplication.
<!--tr:1fb18ec4-->
- **Custo em memória:** os Tensor Cores da GPU atingem a velocidade máxima ao custo de replicar os dados $k_\textrm{h} k_\textrm{w}\times$ na memória de trabalho dos trechos.

<!--en:0d757cf6 kind=slide_title-->
::: {.slide title="Dual View: Unrolling the Kernel into a Banded Matrix"}
<!--tr:0d757cf6-->
::: {.slide title="A construção dual do kernel como matriz de banda"}

<!--en:7f0defaa kind=kicker-->
[Toeplitz Weight Matrix Multiplication]{.kicker}
<!--tr:7f0defaa-->
[Multiplicação pela matriz de pesos de Toeplitz]{.kicker}

<!--en:5898ba57 kind=slide_text-->
Instead of unfolding the input image into duplicated patches, we can unroll the *kernel* into a sparse banded matrix $\mathbf{W}$ that multiplies the flattened input vector $\textrm{vec}(\mathbf{X})$:
<!--tr:5898ba57-->
Em vez de desdobrar a imagem de entrada em trechos com dados repetidos, podemos desdobrar o *kernel* em uma matriz de banda esparsa $\mathbf{W}$ que multiplica o vetor da entrada $\textrm{vec}(\mathbf{X})$:

<!--en:dae51741 kind=slide_text-->
- **Matrix–vector form:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, where $\mathbf{W}$ has $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ rows and $n_\textrm{h} n_\textrm{w}$ columns.
<!--tr:dae51741-->
- **Produto matriz-vetor:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, em que $\mathbf{W}$ tem $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ linhas e $n_\textrm{h} n_\textrm{w}$ colunas.

<!--en:f0e86ee3 kind=slide_text-->
- **Connection to §{sec_why-conv} and backprop:** Every row of $\mathbf{W}$ holds the same $k_\textrm{h} k_\textrm{w}$ kernel entries shifted across columns (panel c), making the backward pass with respect to $\mathbf{X}$ simply multiplication by $\mathbf{W}^\top$ (a transposed convolution).
<!--tr:f0e86ee3-->
- **Relação com §{sec_why-conv} e com a retropropagação:** cada linha de $\mathbf{W}$ contém os mesmos $k_\textrm{h} k_\textrm{w}$ elementos do kernel, deslocados ao longo das colunas (painel c). Assim, a passagem reversa em relação a $\mathbf{X}$ se resume à multiplicação por $\mathbf{W}^\top$ (uma convolução transposta).

<!--en:af8cf19a kind=slide_title-->
::: {.slide title="Feature Maps and Receptive Fields Across Depth"}
<!--tr:af8cf19a-->
::: {.slide title="Mapas de características e campos receptivos ao longo das camadas"}

<!--en:4eb965f3 kind=kicker-->
[Linear Receptive-Field Growth in Stacked Layers]{.kicker}
<!--tr:4eb965f3-->
[Crescimento linear do campo receptivo em camadas empilhadas]{.kicker}

<!--en:5599a6e9 kind=slide_text-->
A unit's **receptive field** is the region of the original input image whose pixels can influence that unit's activation during forward propagation:
<!--tr:5599a6e9-->
O **campo receptivo** de uma unidade é a região da imagem original de entrada cujos pixels podem influenciar a ativação dessa unidade durante a passagem direta:

<!--en:eba74f75 kind=slide_text-->
- **Single layer ($L=1$):** A $k_1 \times k_1$ kernel sees a $k_1 \times k_1$ input patch.
<!--tr:eba74f75-->
- **Uma camada ($L=1$):** um kernel $k_1 \times k_1$ enxerga um trecho $k_1 \times k_1$ da entrada.

<!--en:01c175bf kind=slide_text-->
- **Stacking $L$ stride-1 layers:** Layer $i$ with kernel size $k_i$ reaches $k_i - 1$ pixels farther along each axis than a single element of layer $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$
<!--tr:01c175bf-->
- **Pilha de $L$ camadas com stride 1:** a camada $i$, com tamanho do kernel $k_i$, alcança $k_i - 1$ pixels a mais em cada eixo do que um único elemento da camada $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$

<!--en:2bde5822 kind=slide_text-->
- **$L$ stacked $3 \times 3$ layers:** Each layer adds $3 - 1 = 2$ pixels, giving side length $r = 2L + 1$ ($3 \to 5 \to 7$).
<!--tr:2bde5822-->
- **$L$ camadas $3 \times 3$ empilhadas:** cada camada acrescenta $3 - 1 = 2$ pixels, e o lado do campo receptivo mede $r = 2L + 1$ ($3 \to 5 \to 7$).

<!--en:cbb9e97a kind=slide_title-->
::: {.slide title="Why Deep Stacks of 3×3 Kernels Beat One Large Kernel"}
<!--tr:cbb9e97a-->
::: {.slide title="Por que empilhar kernels 3×3 é mais vantajoso que usar um único kernel grande"}

<!--en:95e46e70 kind=kicker-->
[Parameter Efficiency & Nonlinear Expressivity]{.kicker}
<!--tr:95e46e70-->
[Economia de parâmetros e expressividade das não linearidades]{.kicker}

<!--en:bd1d7cc0 kind=slide_text-->
- **Matched $7 \times 7$ receptive field:** Three stacked $3 \times 3$ layers ($r = 1 + 2 + 2 + 2 = 7$) see the exact same $7 \times 7$ patch as one $7 \times 7$ layer.
<!--tr:bd1d7cc0-->
- **Mesmo campo receptivo $7 \times 7$:** três camadas $3 \times 3$ empilhadas ($r = 1 + 2 + 2 + 2 = 7$) enxergam exatamente o mesmo trecho $7 \times 7$ que uma única camada $7 \times 7$.

<!--en:755ac11a kind=slide_text-->
- **45% fewer parameters:** $3 \times 9 c^2 = 27 c^2$ weights instead of $49 c^2$ across $c$ channels.
<!--tr:755ac11a-->
- **45% menos parâmetros:** $3 \times 9 c^2 = 27 c^2$ pesos em vez de $49 c^2$ para $c$ canais.

<!--en:e3aa1086 kind=slide_text-->
- **Richer expressivity:** Compounds **3 nonlinear activations** instead of 1 (the core VGG design rule in §{sec_blocks}).
<!--tr:e3aa1086-->
- **Maior expressividade:** a pilha compõe **3 ativações não lineares** em vez de 1 (princípio central da arquitetura VGG em §{sec_blocks}).

<!--en:9ec83bf6 kind=slide_title-->
::: {.slide title="Theoretical vs. Effective Receptive Fields"}
<!--tr:9ec83bf6-->
::: {.slide title="Campos receptivos teóricos e efetivos"}

<!--en:72a52c3b kind=kicker-->
[Gaussian Decay of Sensitivity Away from the Center]{.kicker}
<!--tr:72a52c3b-->
[Decaimento gaussiano da sensibilidade com a distância ao centro]{.kicker}

<!--en:33bb377d kind=slide_text-->
- **Theoretical RF ($r = 2L+1$):** Bounding box of all pixels with *at least one* path to the output unit.
<!--tr:33bb377d-->
- **Campo receptivo teórico ($r = 2L+1$):** caixa delimitadora de todos os pixels com *pelo menos um* caminho até a unidade de saída.

<!--en:293436eb kind=slide_text-->
- **Central Limit Theorem:** Center pixels connect via exponentially many paths, whereas corner pixels connect via only **1 path**.
<!--tr:293436eb-->
- **Teorema central do limite:** para os pixels centrais, o número de caminhos cresce exponencialmente. Para cada pixel dos cantos, há apenas **1 caminho**.

<!--en:f47d181e kind=slide_text-->
- **Effective RF (Luo et al., 2016):** Input gradient magnitude $\partial y / \partial \mathbf{X}$ decays in a Gaussian bell curve with std $\sigma \propto \sqrt{L}$.
<!--tr:f47d181e-->
- **Campo receptivo efetivo (Luo et al., 2016):** a magnitude do gradiente em relação à entrada $\partial y / \partial \mathbf{X}$ decai segundo uma curva gaussiana em forma de sino, com desvio padrão $\sigma \propto \sqrt{L}$.

<!--en:fac792d0 kind=slide_title-->
::: {.slide title="Concept Check: Cross-Correlation & Receptive Fields"}
<!--tr:fac792d0-->
::: {.slide title="Revisão de conceitos sobre correlação cruzada e campos receptivos"}

<!--en:26a317d8 kind=kicker-->
[Concept Check]{.kicker}
<!--tr:26a317d8-->
[Revisão de conceitos]{.kicker}

<!--en:8fe554e5 kind=slide_text-->
Test your understanding of output shapes, `im2col`, and receptive-field arithmetic:
<!--tr:8fe554e5-->
Teste sua compreensão das formas de saída, de `im2col` e do cálculo do campo receptivo:

<!--en:281c7c78 kind=slide_text-->
1. **im2col matrix dimensions:** You apply an unpadded $3 \times 3$ single-channel kernel to a $10 \times 10$ image via `im2col`. What is the output spatial shape, and what are the exact dimensions of the `patches` matrix and flattened kernel vector?
<!--tr:281c7c78-->
1. **Dimensões da matriz de im2col:** você aplica um kernel $3 \times 3$ de um único canal a uma imagem $10 \times 10$, sem preenchimento, usando `im2col`. Qual é a forma espacial da saída? Quais são as dimensões exatas da matriz `patches` e do vetor que representa o kernel?

<!--en:16976d18 kind=slide_text-->
2. **Depth to span Fashion-MNIST:** At stride 1, how many stacked $3 \times 3$ layers are required so a single unit's theoretical receptive field spans an entire $28 \times 28$ image? How many with $5 \times 5$ layers?
<!--tr:16976d18-->
2. **Profundidade para cobrir uma imagem do Fashion-MNIST:** com stride 1, quantas camadas $3 \times 3$ precisamos empilhar para que o campo receptivo teórico de uma unidade cubra uma imagem $28 \times 28$ inteira? E se as camadas forem $5 \times 5$?

<!--en:ab2e0d6a kind=slide_text-->
3. **Edge-loss boundary shift:** In our $6 \times 8$ vertical-edge learning experiment, why did the loss start at $\approx 12$, and what would the initial loss be if the black band extended all the way to the right edge (`X[:, 2:8] = 0`)?
<!--tr:ab2e0d6a-->
3. **Efeito da posição da borda sobre a perda:** no experimento em que aprendemos a detectar bordas verticais em uma imagem $6 \times 8$, por que a perda começou em $\approx 12$? Qual seria a perda inicial se a faixa preta se estendesse até a borda direita (`X[:, 2:8] = 0`)?

<!--en:250d6877 kind=slide_text-->
**Answers:**
(1) Output shape is $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ outputs). The `patches` matrix has shape $(64, 9)$ and multiplies the $(9, 1)$ kernel vector.
(2) For $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ layers. For $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ layers.
(3) Each of the 6 rows had two $\pm 1$ edges ($6 \times 2 \times 1^2 = 12$). Extending the black band to the right border eliminates the right-hand black-to-white edge, leaving one edge per row and halving the initial loss to $\approx 6$.
<!--tr:250d6877-->
**Respostas:**
(1) A forma da saída é $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ elementos). A matriz `patches` tem forma $(64, 9)$ e multiplica o vetor de forma $(9, 1)$ que representa o kernel.
(2) Com kernels $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ camadas. Com kernels $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ camadas.
(3) Cada uma das 6 linhas tinha duas bordas de valores $\pm 1$ ($6 \times 2 \times 1^2 = 12$). Estender a faixa preta até o limite direito da imagem elimina a transição do preto para o branco à direita. Resta uma borda por linha e a perda inicial cai pela metade, para $\approx 6$.

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
- **Correlação cruzada 2D e aprendizado de bordas:** deslizamos $\mathbf{K}$ pelas janelas válidas ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$). A SGD aprende $[1, -1]$ em 9 passos.

<!--en:09efe4f0 kind=slide_text-->
- **im2col Matrix Multiply:** Trades $k_\textrm{h} k_\textrm{w}\times$ patch duplication for peak GPU GEMM throughput.
<!--tr:09efe4f0-->
- **Multiplicação de matrizes com im2col:** replicar os dados dos trechos $k_\textrm{h} k_\textrm{w}\times$ permite atingir a vazão máxima da GEMM na GPU.

<!--en:3af8acd0 kind=slide_text-->
- **Stacked Receptive Fields:** $r = 1 + \sum (k_i - 1) = 2L+1$ for $3 \times 3$ stacks, with Gaussian center sensitivity.
<!--tr:3af8acd0-->
- **Campos receptivos em camadas empilhadas:** $r = 1 + \sum (k_i - 1) = 2L+1$ para pilhas de camadas $3 \times 3$, com um perfil gaussiano de sensibilidade centrado no campo receptivo.

=== DECISIONS ===
- “patch matrix” → “matriz de trechos”, com cada trecho da entrada disposto em uma linha. Para “flatten”, usar “converter em vetor” ou indicar a forma vetorial resultante.
- “banded matrix” → “matriz de banda”, “transposed convolution” → “convolução transposta”, “effective receptive field” → “campo receptivo efetivo”.
