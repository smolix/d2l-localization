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
1. [code] **Noyaux conçus à la main.** Concevez quelques noyaux manuellement.
    1. Étant donné un vecteur directeur $\mathbf{v} = (v_1, v_2)$, déterminez
       un noyau de détection de contours qui détecte les contours orthogonaux
       à $\mathbf{v}$, c'est-à-dire orientés dans la direction $(v_2, -v_1)$.
    1. Déterminez un opérateur de différence finie pour la dérivée seconde.
       Quelles structures d'une image y répondent le plus fortement ?
    1. Quelle est la taille minimale d'un noyau permettant d'obtenir une
       dérivée d'ordre $d$ ?
    1. Comment concevriez-vous un noyau de flou ? Pourquoi voudrait-on
       utiliser un tel noyau ?
    1. Chargez une photographie, convertissez-la en un seul canal et
       sous-échantillonnez-la à environ $100 \times 100$ pixels (`corr2d`
       boucle en Python). Appliquez le détecteur de contours, le noyau de
       flou et le noyau d'accentuation « deux fois l'identité moins le flou »,
       puis affichez chaque résultat à côté de l'original. Que révèle la carte
       de contours d'une photographie que celle de la bande noire ne pouvait
       pas montrer ?

    *Adapté du cours CS231n de Stanford,
    [Devoir 2](https://cs231n.github.io/assignments2024/assignment2/),
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
2. [code] **Convolution sous forme de matrice bande.** La matrice de patchs de
   la :numref:`subsec_conv_matmul` déplie l'entrée. La construction opposée
   déroule le noyau. Soient $\mathbf{X}$ une entrée $4 \times 4$, $\mathbf{K}$
   un noyau $3 \times 3$, et $\textrm{vec}(\cdot)$ l'aplatissement ligne par
   ligne.
    1. Construisez la matrice $4 \times 16$ $\mathbf{W}$ telle que
       $\mathbf{W}\,\textrm{vec}(\mathbf{X}) = \textrm{vec}(\mathbf{Y})$,
       où $\mathbf{Y}$ est la corrélation croisée de $\mathbf{X}$ avec
       $\mathbf{K}$. Combien de coefficients non nuls compte chaque ligne, et
       combien de valeurs distinctes apparaissent dans $\mathbf{W}$ ?
    1. Construisez $\mathbf{W}$ en code à partir de `K` et vérifiez que le
       produit reproduit `corr2d(X, K)`.
    1. Pour une entrée $n \times n$ et un noyau $k \times k$, comptez les
       coefficients de $\mathbf{W}$ stockée sous forme de tableau dense, les
       coefficients non nuls de $\mathbf{W}$, ainsi que les coefficients de la
       matrice de patchs. Comment chacun de ces nombres croît-il avec $n$ ?
    1. Conservez le motif de parcimonie de $\mathbf{W}$ mais attribuez à
       chaque ligne ses propres $k^2$ valeurs. On obtient alors une couche
       localement connectée : le tenseur d'ordre quatre $\mathsf{V}$ de la
       :numref:`sec_why-conv` dans
       $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$,
       restreint à une fenêtre $k \times k$ de décalages $(a, b)$ avant que
       l'équivariance par translation ne lie les poids. Pour une entrée
       $n \times n$, comptez les paramètres de cette couche, d'une couche
       entièrement connectée ayant les mêmes formes d'entrée et de sortie, et
       de la convolution.

    *Adapté du cours CMU 11-785,
    [Devoir 2, partie 1](https://deeplearning.cs.cmu.edu/S21/index.html).*

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
3. [code] **Champ récepteur effectif.** ● Le champ récepteur théorique $r$ de
   cette section délimite quelles entrées *peuvent* influencer une sortie ; le
   champ récepteur effectif mesure à quel point elles le font réellement
   :cite:`Luo.Li.Urtasun.ea.2016`. Construisez un empilement de $L$ couches
   convolutives à un seul canal $3 \times 3$ sans remplissage, sans biais et
   sans non-linéarité sur une entrée $32 \times 32$, en utilisant la couche
   intégrée de la :numref:`subsec_learning_kernel`. Tirez chaque coefficient
   des noyaux indépendamment selon une loi gaussienne centrée réduite.
    1. Pour $L = 3$ et un seul tirage des noyaux, rétropropagez un gradient
       unitaire depuis une unité de sortie proche du centre de la dernière
       couche jusqu'à l'entrée et tracez l'amplitude du gradient d'entrée.
       Comparez son support avec le champ récepteur théorique $7 \times 7$, et
       indiquez où l'amplitude est maximale.
    1. Calculez la moyenne du gradient d'entrée au carré sur 500 tirages des
       noyaux, pour $L = 3$ et pour $L = 10$. Sommez chaque carte moyennée sur
       ses lignes et normalisez le profil par colonne résultant pour que sa
       somme vaille un. Comment le rapport entre l'écart-type du profil et la
       longueur de côté théorique $2L + 1$ évolue-t-il avec la profondeur ?
    1. La :numref:`fig_conv_rf_paths` compte les chemins pour $L = 3$ en une
       dimension. Montrez que, pour $L$ quelconque, l'espérance du gradient
       d'entrée au carré, sommée sur les lignes, est proportionnelle à la
       convolution de $L$ copies du vecteur $(1, 1, 1)$. Déduisez-en comment
       l'écart-type du profil normalisé croît avec $L$.

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
4. [code] **Passe arrière d'une convolution.** ● Soient $\mathbf{Y}$ la
   corrélation croisée à un seul canal d'une entrée $\mathbf{X}$ avec un noyau
   $\mathbf{K}$, telle que calculée par `corr2d`, et $\ell$ une perte scalaire
   qui ne dépend de $\mathbf{X}$ et $\mathbf{K}$ qu'à travers $\mathbf{Y}$.
    1. Exprimez $\partial \ell/\partial \mathbf{K}$ et
       $\partial \ell/\partial \mathbf{X}$ en fonction de
       $\partial \ell/\partial \mathbf{Y}$, dans le style des équations de
       rétropropagation de la :numref:`sec_backprop`.
    1. Implémentez votre expression de $\partial \ell/\partial \mathbf{K}$ sans
       différentiation automatique pour la perte
       $\ell = \sum_{i,j} G_{ij} Y_{ij}$, où $\mathbf{G}$ est une matrice
       aléatoire fixée de même forme que $\mathbf{Y}$. Comparez-la à un
       gradient par différence finie centrée avec une taille du pas de
       $10^{-2}$ sur de petits `X` et `K` aléatoires, et indiquez la plus
       grande erreur relative.
    1. Vérifiez votre expression de $\partial \ell/\partial \mathbf{K}$ par
       rapport au gradient du noyau calculé par différentiation automatique
       dans l'expérience de la :numref:`subsec_learning_kernel`.

    *Adapté de Michael Nielsen,
    [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html),
    chapitre 6, et du cours CS231n de Stanford,
    [Devoir 2](https://cs231n.github.io/assignments2024/assignment2/),
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
- Style (registre) : première personne du pluriel (« nous ») dans le corps du cours ; vouvoiement (« vous », impératif à la 2e personne du pluriel) dans les exercices et questions d'auto-évaluation.
- Style (titres et typographie) : casse de phrase dans les titres et sous-titres ; espace ASCII avant `:`, `;`, `?`, `!` ; guillemets français `« … »` dans la prose et guillemets droits `"` conservés dans les attributs Quarto ; séparateur décimal `.` conservé dans les littéraux numériques (`0.01`, `0.02`, `45%`) conformément à TRANSLATING.md §4.
- Style (termes anglais) : mention ponctuelle du terme anglais entre parenthèses en romain lors de la première occurrence d'une notion non transparente (p. ex. `*carte de contours* (edge map)`, `*remplissage*, ou padding`).
- "edge" (photométrique) → "contour" ; "edge detection" → "détection de contours" ; "edge detector" → "détecteur de contours" ; "edge map" → "carte de contours" (distinct de "bord" / "bordure" pour la frontière géométrique d'une grille).
- "patch" → "patch" (pl. "patchs")
- "sliding window" → "fenêtre glissante"
- "elementwise" → "élément par élément"
- "entry" (d'une matrice ou d'un tenseur) → "coefficient" (ou "élément")
- "dense layer" → "couche dense"
- "built-in layer" → "couche intégrée"
- "framework" → "framework"
- "reshape" → "remodeler"
- "filter bank" → "banc de filtres"
- "patch matrix" → "matrice de patchs"
- "unfold" (l'entrée) / "unroll" (le noyau) → "déplier" / "dérouler"
- "flatten" / "flattened" → "aplatir" / "aplati(e)"
- "banded matrix" / "sparse banded matrix" → "matrice bande" / "matrice bande creuse"
- "theoretical receptive field" / "effective receptive field" → "champ récepteur théorique" / "champ récepteur effectif"
- "Concept Check" → "Auto-évaluation"
- "blur kernel" → "noyau de flou" ; "sharpening kernel" → "noyau d'accentuation"
- "downsample" → "sous-échantillonner"
- "sparsity pattern" → "motif de parcimonie"
- "locally connected layer" → "couche localement connectée"
- "standard normal distribution" → "loi gaussienne centrée réduite"
- "Discussions" (lien de forum en fin de section) → "Discussions"

=== NOTES ===
- 6ab5b67c: « Neural Networks and Deep Learning » est conservé en anglais en tant que titre de l'ouvrage de Michael Nielsen.
