<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u01 lang=fr rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:33d3b96b kind=heading-->
# Convolutions for Images
<!--tr:33d3b96b-->
# Convolutions pour les images

<!--en:852f5868 kind=paragraph-->
In :numref:`sec_why-conv`, we derived the convolutional layer on paper from two
geometric principles: translation equivariance and locality. Now we see what
that formula actually computes on an image, and whether gradient descent can
discover useful kernels automatically from data.
<!--tr:852f5868-->
Dans la :numref:`sec_why-conv`, nous avons établi sur papier la couche convolutive à partir de deux principes géométriques : l'équivariance par translation et la localité. Nous allons maintenant voir ce que cette formule calcule réellement sur une image, et si la descente de gradient peut découvrir automatiquement des noyaux utiles à partir des données.

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
Considérons la détection de contours verticaux là où une région claire rencontre une région sombre. Soustraire le voisin de droite de chaque pixel ($[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$) donne zéro à l'intérieur des régions uniformes, $+1$ lorsque le blanc passe au noir, et $-1$ lorsque le noir revient au blanc. Balayer le détecteur à deux poids $(1, -1)$ sur chaque position valide de l'image produit une *carte de contours* (*edge map*). Dans cette section, nous implémentons la corrélation croisée bidimensionnelle, testons notre détecteur de contours conçu à la main, laissons la descente de gradient *apprendre* $(1, -1)$ directement à partir des données, montrons comment une convolution devient une unique multiplication matricielle (*im2col*), et établissons comment l'empilement de couches élargit le champ récepteur d'une unité.

<!--en:47958ed8 kind=heading-->
## Cross-Correlation
<!--tr:47958ed8-->
## Corrélation croisée

<!--en:5e4296fe kind=heading-->
### The Operation
<!--tr:5e4296fe-->
### L'opération

<!--en:ae7e906d kind=paragraph-->
Leaving channels aside for the moment, let the input $\mathbf{X}$ be an
$n_\textrm{h} \times n_\textrm{w}$ matrix and the kernel $\mathbf{K}$ be a
$k_\textrm{h} \times k_\textrm{w}$ matrix. The kernel defines a sliding
*window* of shape $k_\textrm{h} \times k_\textrm{w}$. Placing the window's
upper-left corner at $(i, j)$, we multiply the covered input patch elementwise
with $\mathbf{K}$ and sum the products to produce the output entry at $(i, j)$:
<!--tr:ae7e906d-->
En laissant les canaux de côté pour l'instant, soit l'entrée $\mathbf{X}$ une matrice $n_\textrm{h} \times n_\textrm{w}$ et le noyau $\mathbf{K}$ une matrice $k_\textrm{h} \times k_\textrm{w}$. Le noyau définit une *fenêtre* glissante de forme $k_\textrm{h} \times k_\textrm{w}$. En plaçant le coin supérieur gauche de la fenêtre en $(i, j)$, nous multiplions élément par élément le patch d'entrée couvert par $\mathbf{K}$ et sommons les produits pour obtenir l'élément de sortie en $(i, j)$ :

<!--en:3688b38b kind=paragraph-->
This is the *two-dimensional cross-correlation* of $\mathbf{X}$ with
$\mathbf{K}$, written $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ with
zero-based indices matching code. Whereas :numref:`sec_why-conv` centered
offsets symmetrically from $-\Delta$ to $\Delta$, anchoring at the upper-left
corner differs only by a fixed shift of $\Delta$ while naturally handling
even-sized kernels (such as a $2 \times 2$ kernel or our $1 \times 2$ edge
detector) that have no center pixel.
<!--tr:3688b38b-->
Il s'agit de la *corrélation croisée bidimensionnelle* (*two-dimensional cross-correlation*) de $\mathbf{X}$ avec $\mathbf{K}$, notée $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ avec des indices commençant à zéro correspondant au code. Alors que la :numref:`sec_why-conv` centrait les décalages symétriquement de $-\Delta$ à $\Delta$, l'ancrage au coin supérieur gauche ne diffère que par un décalage fixe de $\Delta$ tout en gérant naturellement les noyaux de taille paire (tels qu'un noyau $2 \times 2$ ou notre détecteur de contours $1 \times 2$) qui n'ont pas de pixel central.

<!--en:fa1b7da9 kind=caption-->
![Two-dimensional cross-correlation. The shaded portions are the first output element and the input and kernel elements used to compute it: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
<!--tr:fa1b7da9-->
![Corrélation croisée bidimensionnelle. Les parties grisées représentent le premier élément de sortie ainsi que les éléments d'entrée et de noyau utilisés pour le calculer : $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)

<!--en:2f8e5deb kind=paragraph-->
In :numref:`fig_correlation`, we slide a $2 \times 2$ kernel across a
$3 \times 3$ input one pixel at a time, left to right and top to bottom. The
window fits cleanly at four positions, yielding the $2 \times 2$ output:
<!--tr:2f8e5deb-->
Dans la :numref:`fig_correlation`, nous faisons glisser un noyau $2 \times 2$ sur une entrée $3 \times 3$ un pixel à la fois, de gauche à droite et de haut en bas. La fenêtre tient exactement à quatre positions, produisant la sortie $2 \times 2$ :

<!--en:4741e012 kind=paragraph-->
Notice that the output shrinks slightly along each axis because the kernel must
stay wholly inside the input. Fitting a width-$k$ window into length $n$ leaves
$n - k + 1$ valid positions, so the output shape is
<!--tr:4741e012-->
Remarquez que la sortie rétrécit légèrement le long de chaque axe car le noyau doit rester entièrement à l'intérieur de l'entrée. Placer une fenêtre de largeur $k$ dans une longueur $n$ laisse $n - k + 1$ positions valides, si bien que la forme de sortie est

<!--en:12d81c14 kind=paragraph-->
Thus a $3 \times 3$ input with a $2 \times 2$ kernel yields $2 \times 2$, a
$6 \times 8$ image with a $1 \times 2$ kernel yields $6 \times 7$, and a
$28 \times 28$ Fashion-MNIST image with a $5 \times 5$ kernel yields
$24 \times 24$. Whenever we want to preserve spatial dimensions, we surround the
input with a border of zeros (*padding*), which we examine alongside strides and
dilation in :numref:`sec_padding`.
<!--tr:12d81c14-->
Ainsi, une entrée $3 \times 3$ avec un noyau $2 \times 2$ donne $2 \times 2$, une image $6 \times 8$ avec un noyau $1 \times 2$ donne $6 \times 7$, et une image Fashion-MNIST $28 \times 28$ avec un noyau $5 \times 5$ donne $24 \times 24$. Chaque fois que nous voulons préserver les dimensions spatiales, nous entourons l'entrée d'une bordure de zéros (*remplissage*, ou *padding*), que nous examinons avec le pas et la dilatation dans la :numref:`sec_padding`.

<!--en:869d223f kind=heading-->
### Implementation
<!--tr:869d223f-->
### Implémentation

<!--en:6a869f7c kind=paragraph-->
We implement :eqref:`eq_corr2d` directly in `corr2d` using a pair of loops over
every output coordinate.
<!--tr:6a869f7c-->
Nous implémentons directement :eqref:`eq_corr2d` dans `corr2d` à l'aide d'une paire de boucles sur chaque coordonnée de sortie.

<!--en:81e1f383 kind=paragraph-->
To verify `corr2d` against our hand calculation in :numref:`fig_correlation`,
we construct the $3 \times 3$ input tensor `X` and $2 \times 2$ kernel `K` and
evaluate `corr2d(X, K)`.
<!--tr:81e1f383-->
Pour vérifier `corr2d` par rapport à notre calcul à la main de la :numref:`fig_correlation`, nous construisons le tenseur d'entrée $3 \times 3$ `X` et le noyau $2 \times 2$ `K`, puis évaluons `corr2d(X, K)`.

<!--en:f567e1b2 kind=paragraph-->
The output reproduces :numref:`fig_correlation` entry for entry (and as shown in
:numref:`sec_why-conv`, strict mathematical convolution with a flipped kernel
$\mathbf{K}'$ learns the exact same family of layers, so we follow standard
deep learning terminology and call cross-correlation "convolution").
<!--tr:f567e1b2-->
La sortie reproduit la :numref:`fig_correlation` élément par élément (et comme montré dans la :numref:`sec_why-conv`, la convolution mathématique stricte avec un noyau retourné $\mathbf{K}'$ apprend exactement la même famille de couches ; nous suivons donc la terminologie standard de l'apprentissage profond et appelons la corrélation croisée « convolution »).

<!--en:bfe11c62 kind=heading-->
## Convolutional Layers
<!--tr:bfe11c62-->
## Couches convolutives

<!--en:2cdc020d kind=heading-->
### The Layer
<!--tr:2cdc020d-->
### La couche

<!--en:c6371ece kind=paragraph-->
A two-dimensional convolutional layer cross-correlates its input with a learned
kernel and adds a scalar bias. Just like our MLP weights in
:numref:`chap_perceptrons`, we initialize the kernel from
$\mathcal{N}(0, 0.01^2)$ and the bias to zero. Below, `Conv2D` registers both
parameters and invokes `corr2d` in its forward pass; a layer with a kernel of
height $h$ and width $w$ is called an $h \times w$ convolutional layer.
<!--tr:c6371ece-->
Une couche convolutive bidimensionnelle effectue la corrélation croisée de son entrée avec un noyau appris et ajoute un biais scalaire. Tout comme les poids de notre perceptron multicouche dans le :numref:`chap_perceptrons`, nous initialisons le noyau selon $\mathcal{N}(0, 0.01^2)$ et le biais à zéro. Ci-dessous, `Conv2D` enregistre les deux paramètres et appelle `corr2d` lors de sa passe avant ; une couche avec un noyau de hauteur $h$ et de largeur $w$ est appelée une couche convolutive $h \times w$.

<!--en:29cd9061 kind=paragraph-->
Unlike a dense layer, a convolutional layer is not locked to a fixed input
resolution because its parameters depend only on the kernel size. The exact same
$2 \times 2$ layer (5 parameters total) processes both our $3 \times 3$ tensor
`X` and a $6 \times 8$ image, returning outputs of shapes $2 \times 2$ and
$5 \times 7$:
<!--tr:29cd9061-->
Contrairement à une couche dense, une couche convolutive n'est pas liée à une résolution d'entrée fixe car ses paramètres ne dépendent que de la taille du noyau. Exactement la même couche $2 \times 2$ (5 paramètres au total) traite à la fois notre tenseur $3 \times 3$ `X` et une image $6 \times 8$, en renvoyant des sorties de formes $2 \times 2$ et $5 \times 7$ :

<!--en:c9e5aa07 kind=heading-->
### Edge Detection
<!--tr:c9e5aa07-->
### Détection de contours

<!--en:3d9a357b kind=paragraph-->
We now test the vertical edge detector from the introduction on a
$6 \times 8$ synthetic image whose middle four columns are black (`0`) and outer
columns are white (`1`).
<!--tr:3d9a357b-->
Nous testons maintenant le détecteur de contours verticaux de l'introduction sur une image synthétique $6 \times 8$ dont les quatre colonnes centrales sont noires (`0`) et les colonnes extérieures sont blanches (`1`).

<!--en:68513ff0 kind=paragraph-->
We define the $1 \times 2$ kernel
<!--tr:68513ff0-->
Nous définissons le noyau $1 \times 2$

<!--en:eb45803b kind=paragraph-->
At every position $(i, j)$, this computes the horizontal *finite difference*
$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, a discrete approximation (up to sign)
of the horizontal derivative of brightness, which vanishes across flat regions
and spikes wherever intensity jumps.
<!--tr:eb45803b-->
En chaque position $(i, j)$, cela calcule la *différence finie* horizontale $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, une approximation discrète (au signe près) de la dérivée horizontale de la luminosité, qui s'annule dans les régions uniformes et présente un pic partout où l'intensité varie brusquement.

<!--en:e810b3a9 kind=paragraph-->
Cross-correlating the $6 \times 8$ stripe image `X` with the $1 \times 2$
horizontal-difference kernel `K` via `corr2d(X, K)` produces the $6 \times 7$
edge map `Y`:
<!--tr:e810b3a9-->
La corrélation croisée de l'image à bande $6 \times 8$ `X` avec le noyau de différence horizontale $1 \times 2$ `K` via `corr2d(X, K)` produit la carte de contours $6 \times 7$ `Y` :

<!--en:568358e4 kind=paragraph-->
Sure enough, `Y` fires `+1` along the white-to-black boundary, `-1` along the
black-to-white boundary, and `0` everywhere else. Applying the exact same kernel
to the transposed image (turning the vertical stripe into a horizontal one)
yields all zeros:
<!--tr:568358e4-->
Comme prévu, `Y` produit `+1` le long de la frontière du blanc vers le noir, `-1` le long de la frontière du noir vers le blanc, et `0` partout ailleurs. Appliquer exactement le même noyau à l'image transposée (transformant la bande verticale en bande horizontale) ne donne que des zéros :

<!--en:71c591a4 kind=paragraph-->
Clearly `[1, -1]` responds strictly to vertical edges. Detecting horizontal
edges, diagonals, corners, or textures requires different kernels altogether.
<!--tr:71c591a4-->
Manifestement, `[1, -1]` répond strictement aux contours verticaux. Détecter des contours horizontaux, des diagonales, des coins ou des textures exige des noyaux tout à fait différents.

<!--en:f0af4887 kind=heading-->
### Learning a Kernel
<!--tr:f0af4887-->
### Apprentissage d'un noyau

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
Concevoir `[1, -1]` à la main est facile lorsque nous connaissons la caractéristique exacte souhaitée, mais aucun ingénieur ne peut spécifier à la main des milliers de filtres intermédiaires dans un réseau profond. La descente de gradient peut-elle retrouver le noyau `[1, -1]` à partir de la seule paire entrée-sortie `(X, Y)` ? Nous testons cela à l'aide de la couche convolutive 2D intégrée de chaque framework (sans biais et avec des poids initialisés selon $\mathcal{N}(0, 0.01^2)$), en minimisant l'erreur quadratique sommée sur les 42 éléments de sortie pendant 10 étapes de gradient avec un taux d'apprentissage de $0.03$. Comme les couches intégrées attendent des tenseurs 4D incluant les axes de lot et de canal (:numref:`sec_channels`), nous redimensionnons `X` et `Y` en conséquence.

<!--en:c29c5d61 kind=paragraph-->
Initially the loss sits near 12 (the near-zero random kernel misses all 12
$\pm 1$ edge pixels by roughly 1), and within just nine steps it drops below
0.01. Inspecting the learned kernel confirms what it found:
<!--tr:c29c5d61-->
Au départ, la perte se situe près de 12 (le noyau aléatoire proche de zéro manque chacun des 12 pixels de contour $\pm 1$ d'environ 1), et en seulement neuf étapes elle descend sous 0.01. L'inspection du noyau appris confirme ce qu'il a trouvé :

<!--en:67d25b0b kind=paragraph-->
Both entries land within roughly 0.02 of `[1, -1]`. Nothing in this loop knew
anything about edges: given a loss signal, gradient descent discovered the
filter on its own. Here we supervised the single layer directly with `Y`; in a
deep CNN, backpropagation learns every layer's kernels end to end from the final
task loss alone.
<!--tr:67d25b0b-->
Les deux entrées arrivent à environ 0.02 de `[1, -1]`. Rien dans cette boucle ne savait quoi que ce soit sur les contours : à partir d'un signal de perte, la descente de gradient a découvert le filtre d'elle-même. Ici, nous avons supervisé l'unique couche directement avec `Y` ; dans un réseau de neurones convolutif profond, la rétropropagation apprend de bout en bout les noyaux de chaque couche à partir de la seule perte finale de la tâche.

<!--en:91b0010d kind=slide_text-->
Convolutions for Images<br>**2D cross-correlation · edge detection · learned kernels · im2col · receptive fields**
<!--tr:91b0010d-->
Convolutions pour les images<br>**Corrélation croisée 2D · détection de contours · noyaux appris · im2col · champs récepteurs**

<!--en:1d94295a kind=slide_title-->
::: {.slide title="Detecting Local Edges Across an Image"}
<!--tr:1d94295a-->
::: {.slide title="Détection de contours locaux dans une image"}

<!--en:687ed302 kind=kicker-->
[Motivation & Core Mental Model]{.kicker}
<!--tr:687ed302-->
[Motivation et modèle mental de base]{.kicker}

<!--en:05fceed0 kind=slide_text-->
- **Synthetic $6 \times 8$ test image:** White outer columns (`1`) surrounding a 4-column black vertical band (`0`).
<!--tr:05fceed0-->
- **Image de test synthétique $6 \times 8$ :** Colonnes extérieures blanches (`1`) entourant une bande verticale noire de 4 colonnes (`0`).

<!--en:71e700b3 kind=slide_text-->
- **Horizontal finite difference:** Sweeping $\mathbf{K} = [1, -1]$ computes $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, firing `+1` at white-to-black and `-1` at black-to-white edges.
<!--tr:71e700b3-->
- **Différence finie horizontale :** Le balayage de $\mathbf{K} = [1, -1]$ calcule $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, donnant `+1` aux transitions blanc-noir et `-1` aux transitions noir-blanc.

<!--en:957dc033 kind=slide_text-->
- **Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent.
<!--tr:957dc033-->
- **Apprentissage vs conception manuelle :** Les réseaux de neurones convolutifs profonds découvrent automatiquement des milliers de tels noyaux par descente de gradient.

<!--en:cb1045e1 kind=slide_title-->
::: {.slide title="The 2D Cross-Correlation Operation"}
<!--tr:cb1045e1-->
::: {.slide title="L'opération de corrélation croisée 2D"}

<!--en:17bcece5 kind=kicker-->
[Sliding-Window Inner Products]{.kicker}
<!--tr:17bcece5-->
[Produits scalaires sur fenêtre glissante]{.kicker}

<!--en:8d45aae8 kind=slide_text-->
Given an $n_\textrm{h} \times n_\textrm{w}$ input matrix $\mathbf{X}$ and a $k_\textrm{h} \times k_\textrm{w}$ kernel $\mathbf{K}$, the 2D cross-correlation $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ places the window's top-left corner at $(i, j)$ and sums elementwise products:
<!--tr:8d45aae8-->
Étant donné une matrice d'entrée $\mathbf{X}$ de taille $n_\textrm{h} \times n_\textrm{w}$ et un noyau $\mathbf{K}$ de taille $k_\textrm{h} \times k_\textrm{w}$, la corrélation croisée 2D $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ place le coin supérieur gauche de la fenêtre en $(i, j)$ et somme les produits élément par élément :

<!--en:e484380c kind=slide_text-->
- **Top-left anchor:** Zero-based indices $a, b \ge 0$ match array indexing in code and handle even-sized kernels (like $2 \times 2$ or $1 \times 2$) cleanly.
<!--tr:e484380c-->
- **Ancrage en haut à gauche :** Les indices $a, b \ge 0$ commençant à zéro correspondent à l'indexation dans le code et gèrent proprement les noyaux de taille paire (comme $2 \times 2$ ou $1 \times 2$).

<!--en:137c0980 kind=slide_text-->
- **Shaded top-left output $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.
<!--tr:137c0980-->
- **Sortie grisée en haut à gauche $(0,0)$ :**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.

<!--en:575c4aea kind=slide_title-->
::: {.slide title="Valid Window Positions and Output Spatial Shape"}
<!--tr:575c4aea-->
::: {.slide title="Positions valides et forme spatiale de sortie"}

<!--en:7eda4531 kind=kicker-->
[Why Unpadded Convolutions Shrink the Grid]{.kicker}
<!--tr:7eda4531-->
[Pourquoi les convolutions sans remplissage rétrécissent la grille]{.kicker}

<!--en:891243ae kind=slide_text-->
- **Valid window placements:** A $k_\textrm{h} \times k_\textrm{w}$ window fits inside an $n_\textrm{h} \times n_\textrm{w}$ grid at $n - k + 1$ positions per axis.
<!--tr:891243ae-->
- **Placements valides :** Une fenêtre $k_\textrm{h} \times k_\textrm{w}$ tient dans une grille $n_\textrm{h} \times n_\textrm{w}$ en $n - k + 1$ positions par axe.

<!--en:55cc8c32 kind=slide_text-->
- **Shape examples:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.
<!--tr:55cc8c32-->
- **Exemples de formes :** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$) ; $28 \times 28 \star 5 \times 5 \to 24 \times 24$.

<!--en:1e3463da kind=slide_title-->
::: {.slide title="Implementing 2D Cross-Correlation in Code"}
<!--tr:1e3463da-->
::: {.slide title="Implémentation de la corrélation croisée 2D en code"}

<!--en:07566992 kind=kicker-->
[Direct Nested-Loop Implementation]{.kicker}
<!--tr:07566992-->
[Implémentation directe par boucles imbriquées]{.kicker}

<!--en:202aa5d3 kind=slide_text-->
The function `corr2d` allocates the $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ output tensor and computes each entry as the sum of `X[i:i+h, j:j+w] * K`:
<!--tr:202aa5d3-->
La fonction `corr2d` alloue le tenseur de sortie $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ et calcule chaque entrée comme la somme de `X[i:i+h, j:j+w] * K` :

<!--en:bef130ba kind=slide_text-->
Each iteration slices a $k_\textrm{h} \times k_\textrm{w}$ sub-window of `X`, multiplies elementwise with `K`, and reduces via `sum`.
<!--tr:bef130ba-->
Chaque itération extrait une sous-fenêtre $k_\textrm{h} \times k_\textrm{w}$ de `X`, la multiplie élément par élément avec `K`, et réduit via `sum`.

<!--en:4ee9f6b1 kind=slide_title-->
::: {.slide title="Verifying corr2d and the Flipped-Kernel Equivalence"}
<!--tr:4ee9f6b1-->
::: {.slide title="Vérification de corr2d et équivalence du noyau retourné"}

<!--en:2b0be423 kind=kicker-->
[Cross-Correlation vs. Mathematical Convolution]{.kicker}
<!--tr:2b0be423-->
[Corrélation croisée vs convolution mathématique]{.kicker}

<!--en:74b9b745 kind=slide_text-->
- **180° flipped kernel $\mathbf{K}'$:** Cross-correlating with $\mathbf{K}$ equals convolving with $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.
<!--tr:74b9b745-->
- **Noyau retourné à 180° $\mathbf{K}'$ :** La corrélation croisée avec $\mathbf{K}$ équivaut à la convolution avec $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.

<!--en:5c8af7cd kind=slide_text-->
- **Identical learned class:** Frameworks compute cross-correlation and call it convolution.
<!--tr:5c8af7cd-->
- **Même classe apprise :** Les frameworks calculent la corrélation croisée et l'appellent convolution.

<!--en:a69e29b5 kind=slide_title-->
::: {.slide title="Building a 2D Convolutional Layer Module"}
<!--tr:a69e29b5-->
::: {.slide title="Construction d'un module de couche convolutive 2D"}

<!--en:55763603 kind=kicker-->
[Learnable Kernel Weights Plus Scalar Bias]{.kicker}
<!--tr:55763603-->
[Poids de noyau apprenables et biais scalaire]{.kicker}

<!--en:5b8bad24 kind=slide_text-->
A convolutional layer wraps `corr2d` with a learnable $k_\textrm{h} \times k_\textrm{w}$ weight parameter (initialized from $\mathcal{N}(0, 0.01^2)$) and a scalar bias (initialized to $0$):
<!--tr:5b8bad24-->
Une couche convolutive encapsule `corr2d` avec un paramètre de poids apprenable $k_\textrm{h} \times k_\textrm{w}$ (initialisé selon $\mathcal{N}(0, 0.01^2)$) et un biais scalaire (initialisé à $0$) :

<!--en:c16d97d6 kind=slide_title-->
::: {.slide title="Resolution Independence of Convolutional Layers"}
<!--tr:c16d97d6-->
::: {.slide title="Indépendance à la résolution des couches convolutives"}

<!--en:a7d1e48f kind=kicker-->
[Fixed Parameters Across Arbitrary Input Sizes]{.kicker}
<!--tr:a7d1e48f-->
[Paramètres fixes pour toute taille d'entrée]{.kicker}

<!--en:53a2bce7 kind=slide_text-->
- **Decoupled from $H \times W$:** The exact same 5 parameters ($2 \times 2$ kernel + 1 bias) process $3 \times 3$, $6 \times 8$, or megapixel inputs without reshaping weights.
<!--tr:53a2bce7-->
- **Découplée de $H \times W$ :** Les 5 mêmes paramètres (noyau $2 \times 2$ + 1 biais) traitent des entrées $3 \times 3$, $6 \times 8$ ou de plusieurs mégapixels sans redimensionner les poids.

<!--en:017fec24 kind=slide_title-->
::: {.slide title="Constructing a Synthetic Vertical-Edge Image"}
<!--tr:017fec24-->
::: {.slide title="Construction d'une image synthétique à contours verticaux"}

<!--en:896e641c kind=kicker-->
[A 6×8 Binary Test Pattern]{.kicker}
<!--tr:896e641c-->
[Un motif de test binaire 6×8]{.kicker}

<!--en:0a4b456b kind=slide_text-->
We construct the $6 \times 8$ test image `X` with ones on the left and right borders and zeros across the middle four columns (`X[:, 2:6] = 0`):
<!--tr:0a4b456b-->
Nous construisons l'image de test $6 \times 8$ `X` avec des uns sur les bordures gauche et droite et des zéros sur les quatre colonnes centrales (`X[:, 2:6] = 0`) :

<!--en:dfc7e3e3 kind=slide_text-->
- **Piecewise-constant columns:** Intensity is constant (`1` or `0`) within columns $0\text{--}1$, $2\text{--}5$, and $6\text{--}7$, jumping only between columns $1 \to 2$ and $5 \to 6$.
<!--tr:dfc7e3e3-->
- **Colonnes constantes par morceaux :** L'intensité est constante (`1` ou `0`) dans les colonnes $0\text{--}1$, $2\text{--}5$ et $6\text{--}7$, ne variant qu'entre les colonnes $1 \to 2$ et $5 \to 6$.

<!--en:3ce31cbe kind=slide_title-->
::: {.slide title="Applying the Horizontal Finite-Difference Kernel"}
<!--tr:3ce31cbe-->
::: {.slide title="Application du noyau de différence finie horizontale"}

<!--en:b37a1029 kind=kicker-->
[Detecting Vertical Edges with K = [1, -1]]{.kicker}
<!--tr:b37a1029-->
[Détection des contours verticaux avec K = [1, -1]]{.kicker}

<!--en:9be85019 kind=slide_text-->
Cross-correlating `X` with the $1 \times 2$ kernel $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ computes $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ at every horizontal pair:
<!--tr:9be85019-->
La corrélation croisée de `X` avec le noyau $1 \times 2$ $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ calcule $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ sur chaque paire horizontale :

<!--en:f649cfae kind=slide_text-->
- **Signed edge map:** Column 1 fires `+1` (white-to-black transition), column 5 fires `-1` (black-to-white transition), and all constant interior columns evaluate to `0`.
<!--tr:f649cfae-->
- **Carte de contours signée :** La colonne 1 donne `+1` (transition blanc-noir), la colonne 5 donne `-1` (transition noir-blanc), et toutes les colonnes intérieures constantes valent `0`.

<!--en:e6d155be kind=slide_title-->
::: {.slide title="Orientation Selectivity of Edge Kernels"}
<!--tr:e6d155be-->
::: {.slide title="Sélectivité d'orientation des noyaux de contour"}

<!--en:b525cde4 kind=kicker-->
[Why One Kernel Cannot Detect All Orientations]{.kicker}
<!--tr:b525cde4-->
[Pourquoi un seul noyau ne peut détecter toutes les orientations]{.kicker}

<!--en:38159de6 kind=slide_text-->
- **Blind to horizontal edges:** Transposing `X` (`X.T`) makes horizontal neighbors identical, so $[1, -1]$ outputs all zeros.
<!--tr:38159de6-->
- **Aveugle aux contours horizontaux :** Transposer `X` (`X.T`) rend les voisins horizontaux identiques, donc $[1, -1]$ ne renvoie que des zéros.

<!--en:f0f30f76 kind=slide_text-->
- **Filter banks (§{sec_channels}):** Vision layers learn parallel channels for horizontal, vertical, and diagonal orientations.
<!--tr:f0f30f76-->
- **Bancs de filtres (§{sec_channels}) :** Les couches de vision apprennent des canaux parallèles pour les orientations horizontales, verticales et diagonales.

<!--en:b755e31b kind=slide_title-->
::: {.slide title="Learning an Edge Detector via Gradient Descent"}
<!--tr:b755e31b-->
::: {.slide title="Apprentissage d'un détecteur de contours par descente de gradient"}

<!--en:86bbeea7 kind=kicker-->
[Supervising a Built-In Convolutional Layer on (X, Y)]{.kicker}
<!--tr:86bbeea7-->
[Supervision d'une couche convolutive intégrée sur (X, Y)]{.kicker}

<!--en:a569780c kind=slide_text-->
- **4D framework tensors:** Reshape `X` and `Y` to 4D tensors (one example, one channel) and minimize $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ at $\eta = 0.03$.
<!--tr:a569780c-->
- **Tenseurs 4D du framework :** Redimensionner `X` et `Y` en tenseurs 4D (un exemple, un canal) et minimiser $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ à $\eta = 0.03$.

<!--en:2489ce6e kind=slide_title-->
::: {.slide title="Inspecting the Learned Kernel Weights"}
<!--tr:2489ce6e-->
::: {.slide title="Inspection des poids du noyau appris"}

<!--en:7e4ef44b kind=kicker-->
[Why Initial Loss Is 12 and Converges to [1, -1]]{.kicker}
<!--tr:7e4ef44b-->
[Pourquoi la perte initiale vaut 12 et converge vers [1, -1]]{.kicker}

<!--en:9cd1c645 kind=slide_text-->
- **Why loss starts at $12.0$:** Near-zero weights predict $\hat{Y} \approx 0$, missing the twelve $\pm 1$ edge pixels in `Y` ($12 \times 1^2 = 12$).
<!--tr:9cd1c645-->
- **Pourquoi la perte démarre à $12.0$ :** Des poids quasi nuls prédisent $\hat{Y} \approx 0$, manquant les douze pixels de contour $\pm 1$ dans `Y` ($12 \times 1^2 = 12$).

<!--en:456670c5 kind=slide_text-->
- **Convergence in 9 steps:** SGD recovers $[0.99, -0.99]$ automatically from data.
<!--tr:456670c5-->
- **Convergence en 9 étapes :** La descente de gradient stochastique retrouve $[0.99, -0.99]$ automatiquement à partir des données.

=== DECISIONS ===
- Register: address the reader with "nous" throughout French prose, matching standard French academic/engineering textbooks.
- Punctuation and quotes: use ASCII spaces before `:`, `;`, `?`, `!` (no invisible Unicode characters) and French guillemets `« ... »` for quoted terms in prose.
- Headings and slide titles: use French sentence case (capitalize only the first word and proper nouns).
- Prose numbers: keep decimal point `.` as in English (e.g., `0.01`, `12.0`) since there is no `fr/style.md` yet.
- "edge" (in images) → « contour » ("edge detection" → « détection de contours », "edge map" → « carte de contours », "edge detector" → « détecteur de contours »).
- "patch" (image window) → « patch » (standard in French computer vision literature, avoiding collision with "bloc").
- "feature map" → « carte de caractéristiques ».
- "dense layer" → « couche dense ».
- "sliding window" → « fenêtre glissante ».
- "built-in" (layer) → « intégrée » (« couche convolutive 2D intégrée »).
- "end to end" → « de bout en bout ».

=== NOTES ===
- 8d45aae8: math tokens reordered for natural French word order (« une matrice d'entrée $\mathbf{X}$ de taille $n_\textrm{h} \times n_\textrm{w}$ »).
