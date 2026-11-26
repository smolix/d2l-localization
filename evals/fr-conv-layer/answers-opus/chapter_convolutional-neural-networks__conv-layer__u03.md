<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u03 lang=fr rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:2294243a kind=heading-->
## Exercises
<!--tr:2294243a-->
## Exercices

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
1. [code] **Noyaux conçus à la main.** Concevez quelques noyaux à la main.
    1. Étant donné un vecteur directeur $\mathbf{v} = (v_1, v_2)$, dérivez
       un noyau de détection de contours qui détecte les contours
       orthogonaux à $\mathbf{v}$, c'est-à-dire les contours de direction
       $(v_2, -v_1)$.
    1. Dérivez un opérateur de différence finie pour la dérivée seconde.
       Quelles structures des images y réagissent le plus fortement ?
    1. Quelle est la taille minimale d'un noyau permettant d'obtenir une
       dérivée d'ordre $d$ ?
    1. Comment concevriez-vous un noyau de flou ? Pourquoi pourriez-vous
       vouloir utiliser un tel noyau ?
    1. Chargez une photographie, convertissez-la en un seul canal et
       sous-échantillonnez-la à environ $100 \times 100$ pixels (`corr2d`
       boucle en Python). Appliquez le détecteur de contours, le noyau de
       flou et le noyau d'accentuation « deux fois l'identité moins le
       flou », et affichez chaque résultat à côté de l'original. Que
       montre la carte des contours d'une photographie que la carte des
       contours de la bande noire ne pouvait pas montrer ?

    *Adapté de Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    notebook ConvolutionalNetworks.*

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
2. [code] **La convolution comme matrice bande.** La matrice de patchs de
   :numref:`subsec_conv_matmul` déplie l'entrée. La construction inverse
   déplie le noyau. Soit $\mathbf{X}$ une entrée $4 \times 4$,
   $\mathbf{K}$ un noyau $3 \times 3$ et $\textrm{vec}(\cdot)$
   l'aplatissement ligne par ligne.
    1. Construisez la matrice $4 \times 16$ $\mathbf{W}$ telle que
       $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$,
       où $\mathbf{Y}$ est la corrélation croisée de $\mathbf{X}$ avec
       $\mathbf{K}$. Combien chaque ligne a-t-elle de coefficients non
       nuls, et combien de valeurs distinctes apparaissent dans
       $\mathbf{W}$ ?
    1. Construisez $\mathbf{W}$ dans le code à partir de `K` et vérifiez
       que le produit reproduit `corr2d(X, K)`.
    1. Pour une entrée $n \times n$ et un noyau $k \times k$, comptez les
       coefficients de $\mathbf{W}$ stockée comme un tableau dense, les
       coefficients non nuls de $\mathbf{W}$ et les coefficients de la
       matrice de patchs. Comment chacun de ces nombres croît-il avec $n$ ?
    1. Conservez la structure de parcimonie de $\mathbf{W}$, mais donnez à
       chaque ligne ses propres $k^2$ valeurs. Le résultat est une couche
       localement connectée : le tenseur d'ordre quatre $\mathsf{V}$ de
       :numref:`sec_why-conv` dans
       $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$,
       restreint à une fenêtre $k \times k$ de décalages $(a, b)$, avant
       que l'équivariance par translation ne lie les poids. Pour une entrée
       $n \times n$, comptez les paramètres de cette couche, ceux d'une
       couche entièrement connectée ayant les mêmes formes d'entrée et de
       sortie, et ceux de la convolution.

    *Adapté de CMU 11-785,
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
3. [code] **Champ récepteur effectif.** ● Le champ récepteur théorique
   $r$ de cette section délimite les entrées qui *peuvent* influencer une
   sortie ; le champ récepteur effectif mesure à quel point elles
   l'influencent réellement :cite:`Luo.Li.Urtasun.ea.2016`. Construisez
   un empilement de $L$ couches convolutives $3 \times 3$ à un seul canal,
   sans remplissage, sans biais et sans non-linéarité, sur une entrée
   $32 \times 32$, à l'aide de la couche intégrée de
   :numref:`subsec_learning_kernel`. Tirez chaque coefficient des noyaux
   indépendamment selon une loi normale centrée réduite.
    1. Pour $L = 3$ et un seul tirage des noyaux, rétropropagez un
       gradient unitaire depuis une unité de sortie proche du centre de la
       dernière couche jusqu'à l'entrée, et tracez l'amplitude du gradient
       par rapport à l'entrée. Comparez son support au champ récepteur
       théorique $7 \times 7$, et indiquez où l'amplitude est la plus
       grande.
    1. Moyennez le carré du gradient par rapport à l'entrée sur 500 tirages
       des noyaux, pour $L = 3$ et pour $L = 10$. Sommez chaque carte
       moyennée sur ses lignes et normalisez le profil de colonnes obtenu
       pour que sa somme vaille un. Comment le rapport entre l'écart-type
       du profil et la longueur de côté théorique $2L + 1$ évolue-t-il avec
       la profondeur ?
    1. :numref:`fig_conv_rf_paths` compte les chemins pour $L = 3$ en une
       dimension. Montrez que, pour $L$ quelconque, l'espérance du carré du
       gradient par rapport à l'entrée, sommée sur les lignes, est
       proportionnelle à la convolution de $L$ copies du vecteur
       $(1, 1, 1)$. Déduisez-en comment l'écart-type du profil normalisé
       croît avec $L$.

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
4. [code] **Passe arrière d'une convolution.** ● Soit $\mathbf{Y}$ la
   corrélation croisée à un seul canal d'une entrée $\mathbf{X}$ avec un
   noyau $\mathbf{K}$, telle que la calcule `corr2d`, et soit $\ell$ une
   perte scalaire qui ne dépend de $\mathbf{X}$ et de $\mathbf{K}$ qu'à
   travers $\mathbf{Y}$.
    1. Exprimez $\partial \ell/\partial \mathbf{K}$ et
       $\partial \ell/\partial \mathbf{X}$ en fonction de
       $\partial \ell/\partial \mathbf{Y}$, dans le style des équations de
       rétropropagation de :numref:`sec_backprop`.
    1. Implémentez votre expression de $\partial \ell/\partial \mathbf{K}$
       sans différentiation automatique pour la perte
       $\ell = \sum_{i,j} G_{ij} Y_{ij}$, où $\mathbf{G}$ est une matrice
       aléatoire fixée de même forme que $\mathbf{Y}$. Comparez-la à un
       gradient par différence finie centrée de taille du pas $10^{-2}$,
       sur de petits `X` et `K` aléatoires, et indiquez la plus grande
       erreur relative.
    1. Vérifiez votre expression de $\partial \ell/\partial \mathbf{K}$
       en la comparant au gradient du noyau que calcule la différentiation
       automatique dans l'expérience de :numref:`subsec_learning_kernel`.

    *Adapté de Michael Nielsen,
    [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html),
    chapitre 6, et de Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    notebook ConvolutionalNetworks.*

<!--en:657a396a kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/65)
<!--tr:657a396a-->
[Discussions](https://d2l.discourse.group/t/65)

<!--en:63c44b48 kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/66)
<!--tr:63c44b48-->
[Discussions](https://d2l.discourse.group/t/66)

<!--en:00b73d0b kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/271)
<!--tr:00b73d0b-->
[Discussions](https://d2l.discourse.group/t/271)

<!--en:b79af65b kind=tab_note-->
[Discussions](https://d2l.discourse.group/t/17996)
<!--tr:b79af65b-->
[Discussions](https://d2l.discourse.group/t/17996)

=== DECISIONS ===
- Exercises use the imperative with « vous » (« Concevez », « Dérivez »), while body prose keeps « nous ».
- Exercise markers `[code]` and `●`, list numbering and indentation are kept as in the English.
- Source credits: "*Adapted from …*" → « *Adapté de …* »; titles of external courses, assignments and books (Assignment 2, Homework 2 Part 1, Neural Networks and Deep Learning) and notebook names stay in English; "notebook" stays « notebook ».
- "standard normal distribution" → « loi normale centrée réduite ».
- "blur kernel" → « noyau de flou »; "sharpening kernel" → « noyau d'accentuation ».
- "locally connected layer" → « couche localement connectée ».
- "Discussions" (forum link) → « Discussions ».

=== NOTES ===
- 6ab5b67c: "Derive … in terms of" is rendered « Exprimez … en fonction de » (to express a derivative in terms of another); « Dérivez » would read as "differentiate".
- 6ab5b67c: the check warns that 'neural networks' -> « réseau de neurones » is not used; a false match, *Neural Networks and Deep Learning* is the title of Michael Nielsen's book and stays in English.
