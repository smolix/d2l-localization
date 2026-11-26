<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u02 lang=fr rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:0b85caf9 kind=heading-->
## Convolution as Matrix Multiplication
<!--tr:0b85caf9-->
## Convolution par multiplication matricielle

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
Les boucles imbriquées de `corr2d` rendent le calcul facile à suivre, mais parcourir les pixels de sortie un à un en Python est bien trop lent en pratique. Une solution classique, encore utilisée par les bibliothèques parmi plusieurs algorithmes, consiste à ramener la corrélation croisée à une seule multiplication de grandes matrices, une opération que les GPU exécutent à leur vitesse maximale. Dans :numref:`fig_correlation`, chaque élément de sortie est simplement le produit scalaire du noyau de $k_\textrm{h} k_\textrm{w}$ éléments mis à plat et de la fenêtre d’entrée correspondante, elle aussi mise à plat. En extrayant les $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ fenêtres et en plaçant chacune sur une ligne, nous obtenons une matrice de fenêtres. Il suffit de la multiplier par le noyau mis à plat pour calculer toute la corrélation croisée en une seule opération. Cette réorganisation classique des données s’appelle *im2col* (« de l’image aux colonnes », ou aux lignes dans notre cas). La :numref:`fig_conv_im2col` l’illustre sur l’exemple de :numref:`fig_correlation`.

<!--en:107554a5 kind=caption-->
![The cross-correlation of :numref:`fig_correlation` as one matrix product. Each of the four $2 \times 2$ windows of the input becomes a row of the patch matrix, and multiplying by the flattened kernel gives the flattened output. The shaded cells follow the first window to its row and to the output 19. The center pixel 4 lies in all four windows, so it appears in every row.](../img/conv-im2col.svg)
<!--tr:107554a5-->
![La corrélation croisée de :numref:`fig_correlation` sous la forme d’un seul produit matriciel. Chacune des quatre fenêtres $2 \times 2$ de l’entrée devient une ligne de la matrice de fenêtres. En multipliant cette matrice par le noyau mis à plat, on obtient la sortie mise à plat. Les cases ombrées permettent de suivre la première fenêtre jusqu’à la ligne correspondante et à la valeur de sortie 19. Le pixel central, de valeur 4, appartient aux quatre fenêtres et figure donc sur chaque ligne.](../img/conv-im2col.svg)

<!--en:76713479 kind=paragraph-->
The trade-off is extra workspace memory, since each interior pixel is duplicated
into up to $k_\textrm{h} k_\textrm{w}$ patches. Below we verify on
:numref:`fig_correlation` that this single matrix multiply matches `corr2d`
exactly:
<!--tr:76713479-->
Cette méthode demande davantage de mémoire de travail, puisque chaque pixel intérieur est recopié dans un maximum de $k_\textrm{h} k_\textrm{w}$ fenêtres. Vérifions sur l’exemple de :numref:`fig_correlation` que cette seule multiplication matricielle donne exactement le même résultat que `corr2d` :

<!--en:3bcd7d8a kind=paragraph-->
The discrepancy is identically zero: one matrix product reproduces `corr2d`
perfectly.
<!--tr:3bcd7d8a-->
L’écart est identiquement nul. Un seul produit matriciel reproduit exactement le résultat de `corr2d`.

<!--en:c30f2dea kind=paragraph-->
Equivalently, we can unroll the *kernel* instead of the input into a sparse
banded matrix $\mathbf{W}$ multiplying the flattened image—the exact 2D analog
of panel (c) in :numref:`fig_why_conv_weight_sharing`, where every row repeats
the same handful of kernel weights. Seeing a convolution as a structured dense
layer brings us full circle to the derivation of :numref:`sec_why-conv` (and the
exercises build $\mathbf{W}$ explicitly).
<!--tr:c30f2dea-->
De façon équivalente, nous pouvons déplier le *noyau* plutôt que l’entrée pour former une matrice creuse en bandes $\mathbf{W}$ qui multiplie l’image mise à plat. C’est l’analogue exact en 2D du panneau (c) de :numref:`fig_why_conv_weight_sharing`, où chaque ligne reprend les mêmes quelques poids du noyau. Voir la convolution comme une couche dense structurée nous ramène à la construction de :numref:`sec_why-conv`. Nous construirons explicitement $\mathbf{W}$ dans les exercices.

<!--en:ad4fc956 kind=heading-->
## Feature Maps and Receptive Fields
<!--tr:ad4fc956-->
## Cartes de caractéristiques et champs récepteurs

<!--en:e7983df8 kind=paragraph-->
A single convolutional layer sees only a patch the size of its kernel, whereas
recognizing whole objects demands integrating context across much wider areas.
Recall from :numref:`subsec_why-conv-channels` that a convolutional layer's
output is called a *feature map* (like our vertical edge map `Y`). For any
element $x$ in a feature map, its *receptive field* is the region of the input
image (or of an earlier layer) whose values can influence $x$ during forward
propagation—borrowing the neurophysiology term from :numref:`field_visual`.
<!--tr:e7983df8-->
Une seule couche convolutive ne voit qu’une région de la taille de son noyau, alors que la reconnaissance d’objets entiers exige de tenir compte du contexte sur des régions bien plus vastes. Rappelons que, dans :numref:`subsec_why-conv-channels`, la sortie d’une couche convolutive est appelée *carte de caractéristiques*, comme notre carte de contours verticaux `Y`. Pour tout élément $x$ d’une carte de caractéristiques, son *champ récepteur* est la région de l’image d’entrée (ou d’une couche antérieure) dont les valeurs peuvent influer sur $x$ pendant la passe avant. Le terme est emprunté à la neurophysiologie, comme dans :numref:`field_visual`.

<!--en:b36f221e kind=heading-->
### Stacked Layers
<!--tr:b36f221e-->
### Empilement de couches

<!--en:24a25ec4 kind=paragraph-->
In :numref:`fig_correlation`, the shaded output `19` has a $2 \times 2$ receptive
field covering the top-left four pixels of $\mathbf{X}$. Now stack a *second*
$2 \times 2$ convolutional layer on top of the $2 \times 2$ output $\mathbf{Y}$
to produce a single scalar $z$: $z$ depends on all four elements of $\mathbf{Y}$,
which together depend on all nine elements of $\mathbf{X}$. Stacking layers
expands the receptive field smoothly with depth.
<!--tr:24a25ec4-->
Dans :numref:`fig_correlation`, la valeur de sortie ombrée `19` a un champ récepteur $2 \times 2$ qui couvre les quatre pixels en haut à gauche de $\mathbf{X}$. Appliquons maintenant une *deuxième* couche convolutive $2 \times 2$ à la sortie $2 \times 2$ $\mathbf{Y}$ pour obtenir un unique scalaire $z$. La valeur de $z$ dépend des quatre éléments de $\mathbf{Y}$, qui dépendent ensemble des neuf éléments de $\mathbf{X}$. L’empilement de couches élargit progressivement le champ récepteur avec la profondeur.

<!--en:f95c9394 kind=paragraph-->
How quickly does it grow? Each $k \times k$ convolutional layer reaches $k - 1$
pixels farther than a single element of its input, so stacking $L$ layers with
kernel sizes $k_1, \ldots, k_L$ yields a receptive field of side length
<!--tr:f95c9394-->
À quel rythme s’élargit-il ? Chaque couche convolutive $k \times k$ étend le champ de $k - 1$ pixels par rapport à celui d’un seul élément de son entrée. En empilant $L$ couches dont les noyaux ont pour tailles $k_1, \ldots, k_L$, nous obtenons donc un champ récepteur dont la longueur du côté vaut

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
Dans un empilement de $L$ couches $3 \times 3$ consécutives, chaque couche ajoute 2 pixels par axe. Le champ récepteur couvre ainsi $(2L + 1) \times (2L + 1)$ éléments d’entrée, soit $5 \times 5$ après deux couches et $7 \times 7$ après trois (:numref:`fig_conv_receptive_field`). Cette croissance linéaire nous évite de recourir à de très grands noyaux. Trois couches $3 \times 3$ empilées donnent le même champ récepteur $7 \times 7$ qu’une seule couche $7 \times 7$, avec bien moins de paramètres ($3 \times 9 = 27$ poids contre 49 par paire de canaux), tout en intercalant trois non-linéarités au lieu d’une.

<!--en:39917513 kind=caption-->
![Receptive field of one unit after three one-dimensional layers with kernel size 3. Shaded cells are the units that the top unit depends on, and the lines connect each of them to its inputs. Each layer widens the receptive field by 2, to 7 inputs.](../img/conv-receptive-field.svg)
<!--tr:39917513-->
![Champ récepteur d’une unité après trois couches unidimensionnelles de taille de noyau 3. Les cases ombrées représentent les unités dont dépend l’unité du haut et les traits relient chacune d’elles à ses entrées. Chaque couche élargit le champ récepteur de 2, jusqu’à couvrir 7 entrées.](../img/conv-receptive-field.svg)

<!--en:48e4c9cc kind=heading-->
### Effective Receptive Fields
<!--tr:48e4c9cc-->
### Champs récepteurs effectifs

<!--en:77d044db kind=paragraph-->
Keep in mind that $r$ measures the *theoretical* receptive field (every input
pixel with *any* path to an output unit) without saying how strongly each pixel
actually contributes. Measuring input gradients directly,
:citet:`Luo.Li.Urtasun.ea.2016` showed that the *effective* receptive field
decays sharply away from the center in a roughly Gaussian bell curve, in both
untrained and trained networks. Deep units can theoretically see far across the
image, but their sensitivity concentrates heavily near the center.
<!--tr:77d044db-->
Gardons à l’esprit que $r$ mesure le champ récepteur *théorique*, qui comprend tout pixel d’entrée relié par *au moins un chemin* à une unité de sortie, quelle que soit sa contribution réelle. En mesurant directement les gradients par rapport à l’entrée, :citet:`Luo.Li.Urtasun.ea.2016` ont montré que, dans le champ récepteur *effectif*, la sensibilité décroît rapidement à mesure qu’on s’éloigne du centre, suivant une courbe en cloche approximativement gaussienne, dans les réseaux entraînés comme dans les réseaux non entraînés. Les unités des couches profondes peuvent en théorie voir loin dans l’image, mais leur sensibilité se concentre fortement près du centre.

<!--en:eed4e2a0 kind=paragraph-->
Counting paths explains the bell shape. An input at the edge of the receptive
field reaches the output unit along a single chain of connections, whereas the
center input can take many routes through the layers in between.
:numref:`fig_conv_rf_paths` counts them for the tower of
:numref:`fig_conv_receptive_field`: the center input has seven paths to the top
unit and each edge input has one. If all weights have a similar magnitude, the
influence of an input grows with its number of paths.
<!--tr:eed4e2a0-->
Le nombre de chemins explique la forme en cloche. Une entrée au bord du champ récepteur atteint l’unité de sortie par une seule chaîne de connexions, tandis que l’entrée centrale peut emprunter de nombreux chemins à travers les couches intermédiaires. La :numref:`fig_conv_rf_paths` les dénombre pour l’empilement de :numref:`fig_conv_receptive_field`. L’entrée centrale dispose de sept chemins jusqu’à l’unité du haut et chaque entrée au bord n’en a qu’un. Si tous les poids sont du même ordre de grandeur en valeur absolue, l’influence d’une entrée croît avec le nombre de chemins qui en partent.

<!--en:b92775ee kind=caption-->
![Number of paths from each unit to the top unit of :numref:`fig_conv_receptive_field` (three one-dimensional layers with kernel size 3). Each count is the sum of the counts of the units that the cell feeds. Darker cells have more paths than the other cells of their layer, and the bars repeat the counts of the input row: 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)
<!--tr:b92775ee-->
![Nombre de chemins de chaque unité à l’unité du haut de :numref:`fig_conv_receptive_field` (trois couches unidimensionnelles avec des noyaux de taille 3). Le nombre indiqué dans chaque case est la somme des nombres associés aux unités qui en reçoivent la sortie. Les cases les plus foncées ont davantage de chemins que les autres cases de leur couche, et les barres reprennent les nombres de la ligne d’entrée : 1, 3, 6, 7, 6, 3, 1.](../img/conv-rf-paths.svg)

<!--en:405695e8 kind=paragraph-->
Two practical gaps remain in our basic layer: every unpadded convolution shrinks
spatial dimensions while under-sampling border pixels, and a single kernel
detects only one pattern. We address spatial sizing and borders next in
:numref:`sec_padding`, and multi-channel feature maps in :numref:`sec_channels`.
<!--tr:405695e8-->
Notre couche de base présente encore deux limites en pratique. Chaque convolution sans remplissage réduit les dimensions spatiales et échantillonne moins souvent les pixels du bord. De plus, un seul noyau ne détecte qu’un seul motif. Nous traiterons les dimensions spatiales et les bords dans :numref:`sec_padding`, puis les cartes de caractéristiques à plusieurs canaux dans :numref:`sec_channels`.

<!--en:30ac03ff kind=heading-->
## Summary
<!--tr:30ac03ff-->
## Résumé

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
Une couche convolutive fait glisser un noyau appris sur toutes les fenêtres valides de l’entrée pour calculer une corrélation croisée bidimensionnelle, puis ajoute un biais scalaire. Un simple noyau de différence finie $1 \times 2$, `[1, -1]`, détecte les contours verticaux partout dans une image. La descente de gradient retrouve approximativement ces poids à partir des seules données d’entrée et de sortie, tout comme les CNN profonds apprennent de bout en bout des hiérarchies entières de filtres. La convolution mathématique retourne le noyau. Ce retournement est sans effet sur la famille de couches que l’on peut apprendre. Une convolution se calcule en une seule multiplication matricielle (*im2col*). Quand on empile $L$ couches convolutives, le champ récepteur de chaque unité s’élargit linéairement, de $k_i - 1$ par couche.

<!--en:3ba929d2 kind=slide_title-->
::: {.slide title="Fast Execution: Convolution via Matrix Multiplication"}
<!--tr:3ba929d2-->
::: {.slide title="Calcul rapide de la convolution par multiplication matricielle"}

<!--en:1711f812 kind=kicker-->
[The im2col (Image-to-Columns) Transformation]{.kicker}
<!--tr:1711f812-->
[Transformation im2col, de l’image aux colonnes]{.kicker}

<!--en:a040f3d7 kind=slide_text-->
- **Single dense GEMM:** Flatten each $k_\textrm{h} \times k_\textrm{w}$ window into a row of `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$ and multiply by $\textrm{vec}(\mathbf{K})$.
<!--tr:a040f3d7-->
- **Une seule multiplication dense (GEMM) :** mettre chaque fenêtre $k_\textrm{h} \times k_\textrm{w}$ à plat sur une ligne de `patches` $\in \mathbb{R}^{(h_\textrm{o} w_\textrm{o}) \times (k_\textrm{h} k_\textrm{w})}$, puis multiplier cette matrice par $\textrm{vec}(\mathbf{K})$.

<!--en:1fb18ec4 kind=slide_text-->
- **Trade-off:** Peak GPU Tensor Core speed at the cost of $k_\textrm{h} k_\textrm{w}\times$ patch workspace duplication.
<!--tr:1fb18ec4-->
- **Compromis :** vitesse maximale des Tensor Cores du GPU, au prix d’une mémoire de travail multipliée par $k_\textrm{h} k_\textrm{w}\times$ pour stocker les fenêtres.

<!--en:0d757cf6 kind=slide_title-->
::: {.slide title="Dual View: Unrolling the Kernel into a Banded Matrix"}
<!--tr:0d757cf6-->
::: {.slide title="Autre formulation : le noyau déplié en matrice en bandes"}

<!--en:7f0defaa kind=kicker-->
[Toeplitz Weight Matrix Multiplication]{.kicker}
<!--tr:7f0defaa-->
[Multiplication par une matrice de poids de Toeplitz]{.kicker}

<!--en:5898ba57 kind=slide_text-->
Instead of unfolding the input image into duplicated patches, we can unroll the *kernel* into a sparse banded matrix $\mathbf{W}$ that multiplies the flattened input vector $\textrm{vec}(\mathbf{X})$:
<!--tr:5898ba57-->
Au lieu de déplier l’image d’entrée en recopiant ses fenêtres, on peut déplier le *noyau* en une matrice creuse en bandes $\mathbf{W}$ qui multiplie le vecteur d’entrée mis à plat $\textrm{vec}(\mathbf{X})$ :

<!--en:dae51741 kind=slide_text-->
- **Matrix–vector form:** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, where $\mathbf{W}$ has $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ rows and $n_\textrm{h} n_\textrm{w}$ columns.
<!--tr:dae51741-->
- **Forme matrice-vecteur :** $\textrm{vec}(\mathbf{Y}) = \mathbf{W}\,\textrm{vec}(\mathbf{X})$, où $\mathbf{W}$ a $(n_\textrm{h}-k_\textrm{h}+1)(n_\textrm{w}-k_\textrm{w}+1)$ lignes et $n_\textrm{h} n_\textrm{w}$ colonnes.

<!--en:f0e86ee3 kind=slide_text-->
- **Connection to §{sec_why-conv} and backprop:** Every row of $\mathbf{W}$ holds the same $k_\textrm{h} k_\textrm{w}$ kernel entries shifted across columns (panel c), making the backward pass with respect to $\mathbf{X}$ simply multiplication by $\mathbf{W}^\top$ (a transposed convolution).
<!--tr:f0e86ee3-->
- **Lien avec §{sec_why-conv} et la rétropropagation :** chaque ligne de $\mathbf{W}$ contient les mêmes $k_\textrm{h} k_\textrm{w}$ éléments du noyau, décalés d’une ligne à l’autre (panneau c). La passe arrière par rapport à $\mathbf{X}$ se réduit donc à une multiplication par $\mathbf{W}^\top$ (une convolution transposée).

<!--en:af8cf19a kind=slide_title-->
::: {.slide title="Feature Maps and Receptive Fields Across Depth"}
<!--tr:af8cf19a-->
::: {.slide title="Cartes de caractéristiques et champs récepteurs selon la profondeur"}

<!--en:4eb965f3 kind=kicker-->
[Linear Receptive-Field Growth in Stacked Layers]{.kicker}
<!--tr:4eb965f3-->
[Croissance linéaire du champ récepteur avec l’empilement]{.kicker}

<!--en:5599a6e9 kind=slide_text-->
A unit's **receptive field** is the region of the original input image whose pixels can influence that unit's activation during forward propagation:
<!--tr:5599a6e9-->
Le **champ récepteur** d’une unité est la région de l’image d’entrée d’origine dont les pixels peuvent influer sur son activation pendant la passe avant :

<!--en:eba74f75 kind=slide_text-->
- **Single layer ($L=1$):** A $k_1 \times k_1$ kernel sees a $k_1 \times k_1$ input patch.
<!--tr:eba74f75-->
- **Une seule couche ($L=1$) :** un noyau $k_1 \times k_1$ voit une région $k_1 \times k_1$ de l’entrée.

<!--en:01c175bf kind=slide_text-->
- **Stacking $L$ stride-1 layers:** Layer $i$ with kernel size $k_i$ reaches $k_i - 1$ pixels farther along each axis than a single element of layer $i-1$:
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$
<!--tr:01c175bf-->
- **Empilement de $L$ couches de pas 1 :** la couche $i$, dont le noyau est de taille $k_i$, étend le champ de $k_i - 1$ pixels sur chaque axe par rapport à un seul élément de la couche $i-1$ :
  $$r = 1 + \sum_{i=1}^{L} (k_i - 1)$$

<!--en:2bde5822 kind=slide_text-->
- **$L$ stacked $3 \times 3$ layers:** Each layer adds $3 - 1 = 2$ pixels, giving side length $r = 2L + 1$ ($3 \to 5 \to 7$).
<!--tr:2bde5822-->
- **$L$ couches $3 \times 3$ empilées :** chaque couche ajoute $3 - 1 = 2$ pixels. Le côté mesure donc $r = 2L + 1$ ($3 \to 5 \to 7$).

<!--en:cbb9e97a kind=slide_title-->
::: {.slide title="Why Deep Stacks of 3×3 Kernels Beat One Large Kernel"}
<!--tr:cbb9e97a-->
::: {.slide title="Pourquoi empiler des noyaux 3×3 plutôt qu’utiliser un grand noyau"}

<!--en:95e46e70 kind=kicker-->
[Parameter Efficiency & Nonlinear Expressivity]{.kicker}
<!--tr:95e46e70-->
[Moins de paramètres et meilleure expressivité non linéaire]{.kicker}

<!--en:bd1d7cc0 kind=slide_text-->
- **Matched $7 \times 7$ receptive field:** Three stacked $3 \times 3$ layers ($r = 1 + 2 + 2 + 2 = 7$) see the exact same $7 \times 7$ patch as one $7 \times 7$ layer.
<!--tr:bd1d7cc0-->
- **Même champ récepteur $7 \times 7$ :** trois couches $3 \times 3$ empilées ($r = 1 + 2 + 2 + 2 = 7$) voient exactement la même région $7 \times 7$ qu’une couche $7 \times 7$.

<!--en:755ac11a kind=slide_text-->
- **45% fewer parameters:** $3 \times 9 c^2 = 27 c^2$ weights instead of $49 c^2$ across $c$ channels.
<!--tr:755ac11a-->
- **45% de paramètres en moins :** $3 \times 9 c^2 = 27 c^2$ poids au lieu de $49 c^2$ pour $c$ canaux.

<!--en:e3aa1086 kind=slide_text-->
- **Richer expressivity:** Compounds **3 nonlinear activations** instead of 1 (the core VGG design rule in §{sec_blocks}).
<!--tr:e3aa1086-->
- **Plus grande expressivité :** l’empilement compose **3 activations non linéaires** au lieu d’une seule (le principe de conception de VGG dans §{sec_blocks}).

<!--en:9ec83bf6 kind=slide_title-->
::: {.slide title="Theoretical vs. Effective Receptive Fields"}
<!--tr:9ec83bf6-->
::: {.slide title="Champs récepteurs théorique et effectif"}

<!--en:72a52c3b kind=kicker-->
[Gaussian Decay of Sensitivity Away from the Center]{.kicker}
<!--tr:72a52c3b-->
[Décroissance gaussienne de la sensibilité autour du centre]{.kicker}

<!--en:33bb377d kind=slide_text-->
- **Theoretical RF ($r = 2L+1$):** Bounding box of all pixels with *at least one* path to the output unit.
<!--tr:33bb377d-->
- **Champ récepteur théorique ($r = 2L+1$) :** boîte englobante des pixels ayant *au moins un* chemin vers l’unité de sortie.

<!--en:293436eb kind=slide_text-->
- **Central Limit Theorem:** Center pixels connect via exponentially many paths, whereas corner pixels connect via only **1 path**.
<!--tr:293436eb-->
- **Théorème central limite :** le nombre de chemins depuis les pixels centraux croît exponentiellement. Depuis chaque pixel de coin, il n’y a qu’**un seul chemin**.

<!--en:f47d181e kind=slide_text-->
- **Effective RF (Luo et al., 2016):** Input gradient magnitude $\partial y / \partial \mathbf{X}$ decays in a Gaussian bell curve with std $\sigma \propto \sqrt{L}$.
<!--tr:f47d181e-->
- **Champ récepteur effectif (Luo et al., 2016) :** l’amplitude du gradient par rapport à l’entrée $\partial y / \partial \mathbf{X}$ décroît selon une courbe en cloche gaussienne d’écart-type $\sigma \propto \sqrt{L}$.

<!--en:fac792d0 kind=slide_title-->
::: {.slide title="Concept Check: Cross-Correlation & Receptive Fields"}
<!--tr:fac792d0-->
::: {.slide title="Corrélation croisée et champs récepteurs : autoévaluation"}

<!--en:26a317d8 kind=kicker-->
[Concept Check]{.kicker}
<!--tr:26a317d8-->
[Questions de compréhension]{.kicker}

<!--en:8fe554e5 kind=slide_text-->
Test your understanding of output shapes, `im2col`, and receptive-field arithmetic:
<!--tr:8fe554e5-->
Vérifiez votre compréhension des formes de sortie, de `im2col` et du calcul des champs récepteurs :

<!--en:281c7c78 kind=slide_text-->
1. **im2col matrix dimensions:** You apply an unpadded $3 \times 3$ single-channel kernel to a $10 \times 10$ image via `im2col`. What is the output spatial shape, and what are the exact dimensions of the `patches` matrix and flattened kernel vector?
<!--tr:281c7c78-->
1. **Dimensions de la matrice im2col :** vous appliquez un noyau $3 \times 3$ à un seul canal à une image $10 \times 10$, sans remplissage, via `im2col`. Quelles sont les dimensions spatiales de la sortie ? Quelles sont les dimensions exactes de la matrice `patches` et du vecteur obtenu en mettant le noyau à plat ?

<!--en:16976d18 kind=slide_text-->
2. **Depth to span Fashion-MNIST:** At stride 1, how many stacked $3 \times 3$ layers are required so a single unit's theoretical receptive field spans an entire $28 \times 28$ image? How many with $5 \times 5$ layers?
<!--tr:16976d18-->
2. **Profondeur nécessaire pour couvrir Fashion-MNIST :** avec un pas de 1, combien de couches $3 \times 3$ faut-il empiler pour que le champ récepteur théorique d’une unité couvre toute une image $28 \times 28$ ? Et avec des couches $5 \times 5$ ?

<!--en:ab2e0d6a kind=slide_text-->
3. **Edge-loss boundary shift:** In our $6 \times 8$ vertical-edge learning experiment, why did the loss start at $\approx 12$, and what would the initial loss be if the black band extended all the way to the right edge (`X[:, 2:8] = 0`)?
<!--tr:ab2e0d6a-->
3. **Effet du déplacement d’un contour sur la perte :** dans notre expérience d’apprentissage des contours verticaux sur une image $6 \times 8$, pourquoi la perte initiale valait-elle $\approx 12$ ? Quelle serait-elle si la bande noire s’étendait jusqu’au bord droit (`X[:, 2:8] = 0`) ?

<!--en:250d6877 kind=slide_text-->
**Answers:**
(1) Output shape is $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ outputs). The `patches` matrix has shape $(64, 9)$ and multiplies the $(9, 1)$ kernel vector.
(2) For $3 \times 3$: $2L + 1 \ge 28 \implies L = 14$ layers. For $5 \times 5$: $4L + 1 \ge 28 \implies L = 7$ layers.
(3) Each of the 6 rows had two $\pm 1$ edges ($6 \times 2 \times 1^2 = 12$). Extending the black band to the right border eliminates the right-hand black-to-white edge, leaving one edge per row and halving the initial loss to $\approx 6$.
<!--tr:250d6877-->
**Réponses :**
(1) La forme de sortie est $(10-3+1) \times (10-3+1) = 8 \times 8$ ($64$ valeurs de sortie). La matrice `patches` a pour forme $(64, 9)$ et multiplie le vecteur de noyau de forme $(9, 1)$.
(2) Pour $3 \times 3$, $2L + 1 \ge 28 \implies L = 14$ couches. Pour $5 \times 5$, $4L + 1 \ge 28 \implies L = 7$ couches.
(3) Chacune des 6 lignes avait deux contours de valeur $\pm 1$ ($6 \times 2 \times 1^2 = 12$). Prolonger la bande noire jusqu’au bord droit supprime le contour de droite, du noir au blanc. Il reste un contour par ligne et la perte initiale est divisée par deux, à $\approx 6$.

<!--en:ba43a2dd kind=slide_title-->
::: {.slide title="Summary & Key Takeaways"}
<!--tr:ba43a2dd-->
::: {.slide title="Résumé et points à retenir"}

<!--en:ce12e85f kind=kicker-->
[Section Synthesis]{.kicker}
<!--tr:ce12e85f-->
[Synthèse de la section]{.kicker}

<!--en:00011fb7 kind=slide_text-->
- **2D Cross-Correlation & Edge Learning:** Slides $\mathbf{K}$ across valid windows ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$); SGD learns $[1, -1]$ in 9 steps.
<!--tr:00011fb7-->
- **Corrélation croisée 2D et apprentissage des contours :** on fait glisser $\mathbf{K}$ sur les fenêtres valides ($(n_\textrm{h}-k_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+1)$). La descente de gradient apprend un noyau proche de $[1, -1]$ en 9 pas.

<!--en:09efe4f0 kind=slide_text-->
- **im2col Matrix Multiply:** Trades $k_\textrm{h} k_\textrm{w}\times$ patch duplication for peak GPU GEMM throughput.
<!--tr:09efe4f0-->
- **Multiplication matricielle im2col :** la duplication des pixels dans les fenêtres (facteur $k_\textrm{h} k_\textrm{w}\times$) permet d’atteindre le débit maximal des multiplications GEMM sur GPU.

<!--en:3af8acd0 kind=slide_text-->
- **Stacked Receptive Fields:** $r = 1 + \sum (k_i - 1) = 2L+1$ for $3 \times 3$ stacks, with Gaussian center sensitivity.
<!--tr:3af8acd0-->
- **Champ récepteur des couches empilées :** $r = 1 + \sum (k_i - 1) = 2L+1$ pour des couches $3 \times 3$, avec une sensibilité gaussienne centrée.

=== DECISIONS ===
- « flatten » → « mettre à plat », « unfold/unroll » → « déplier », « patch matrix » → « matrice de fenêtres ».
- « banded matrix » → « matrice en bandes », « sparse matrix » → « matrice creuse », « stacked layers » → « couches empilées ».
- « nonlinearity » → « non-linéarité », « standard deviation » → « écart-type ». « GEMM » et « Tensor Cores » conservent leur graphie technique.
- « theoretical/effective receptive field » → « champ récepteur théorique/effectif ».

=== NOTES ===
- 92688a0c et 00011fb7 : les poids sont retrouvés à l’approximation indiquée dans l’expérience, même si le résumé anglais les dit exacts ou donne la cible exacte. L’avertissement sur « SGD » en 00011fb7 vient de la même correction qu’en 456670c5 : la boucle met en œuvre une descente de gradient.
- e3aa1086 et 293436eb : faux positifs du contrôle des nombres. Le chiffre « 1 » est rendu en toutes lettres dans « une seule » et « un seul chemin », respectivement.
