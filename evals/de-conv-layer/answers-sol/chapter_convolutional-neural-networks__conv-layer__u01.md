<!--unit:chapter_convolutional-neural-networks/conv-layer.md#u01 lang=de rev=370f7b3a15d0557b44a27c52e06ab7f5b30584e1-->

<!--en:33d3b96b kind=heading-->
# Convolutions for Images
<!--tr:33d3b96b-->
# Faltungen für Bilder

<!--en:852f5868 kind=paragraph-->
In :numref:`sec_why-conv`, we derived the convolutional layer on paper from two
geometric principles: translation equivariance and locality. Now we see what
that formula actually computes on an image, and whether gradient descent can
discover useful kernels automatically from data.
<!--tr:852f5868-->
In :numref:`sec_why-conv` haben wir die Faltungsschicht (engl. convolutional layer) auf dem Papier aus den beiden geometrischen Prinzipien der Translationsäquivarianz (engl. translation equivariance) und der Lokalität hergeleitet. Jetzt sehen wir, was diese Formel für ein Bild tatsächlich berechnet. Außerdem prüfen wir, ob der Gradientenabstieg (engl. gradient descent) selbstständig geeignete Faltungskerne (engl. convolutional kernels) aus Daten lernen kann.

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
Suchen wir zum Beispiel nach vertikalen Kanten, an denen ein heller Bereich auf einen dunklen trifft. Ziehen wir vom Wert jedes Pixels den Wert seines rechten Nachbarn ab ($[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$), erhalten wir in gleichförmigen Bereichen null. Beim Übergang von Weiß zu Schwarz ergibt sich $+1$, beim Übergang von Schwarz zu Weiß $-1$. Schieben wir den Detektor mit den beiden Gewichten $(1, -1)$ über alle gültigen Positionen im Bild, entsteht eine *Kantenkarte* (engl. edge map). In diesem Abschnitt implementieren wir die zweidimensionale Kreuzkorrelation (engl. cross-correlation) und testen unseren von Hand entworfenen Kantendetektor. Dann lassen wir den Gradientenabstieg $(1, -1)$ direkt aus Daten *lernen*. Wir zeigen, wie sich eine Faltung (engl. convolution) als einzige Matrizenmultiplikation (*im2col*) schreiben lässt, und leiten her, wie übereinanderliegende Schichten das rezeptive Feld (engl. receptive field) einer Einheit vergrößern.

<!--en:47958ed8 kind=heading-->
## Cross-Correlation
<!--tr:47958ed8-->
## Kreuzkorrelation

<!--en:5e4296fe kind=heading-->
### The Operation
<!--tr:5e4296fe-->
### Die Operation

<!--en:ae7e906d kind=paragraph-->
Leaving channels aside for the moment, let the input $\mathbf{X}$ be an
$n_\textrm{h} \times n_\textrm{w}$ matrix and the kernel $\mathbf{K}$ be a
$k_\textrm{h} \times k_\textrm{w}$ matrix. The kernel defines a sliding
*window* of shape $k_\textrm{h} \times k_\textrm{w}$. Placing the window's
upper-left corner at $(i, j)$, we multiply the covered input patch elementwise
with $\mathbf{K}$ and sum the products to produce the output entry at $(i, j)$:
<!--tr:ae7e906d-->
Lassen wir die Kanäle vorerst außer Acht. Die Eingabe $\mathbf{X}$ sei eine $n_\textrm{h} \times n_\textrm{w}$-Matrix und der Faltungskern $\mathbf{K}$ eine $k_\textrm{h} \times k_\textrm{w}$-Matrix. Der Faltungskern bestimmt ein verschiebbares *Fenster* der Form $k_\textrm{h} \times k_\textrm{w}$. Liegt die linke obere Ecke des Fensters bei $(i, j)$, multiplizieren wir den überdeckten Eingabeausschnitt elementweise mit $\mathbf{K}$. Die Summe der Produkte ergibt den Ausgabeeintrag bei $(i, j)$.

<!--en:3688b38b kind=paragraph-->
This is the *two-dimensional cross-correlation* of $\mathbf{X}$ with
$\mathbf{K}$, written $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ with
zero-based indices matching code. Whereas :numref:`sec_why-conv` centered
offsets symmetrically from $-\Delta$ to $\Delta$, anchoring at the upper-left
corner differs only by a fixed shift of $\Delta$ while naturally handling
even-sized kernels (such as a $2 \times 2$ kernel or our $1 \times 2$ edge
detector) that have no center pixel.
<!--tr:3688b38b-->
Das ist die *zweidimensionale Kreuzkorrelation* von $\mathbf{X}$ mit $\mathbf{K}$. Wir schreiben $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ und beginnen wie im Code bei Index null. In :numref:`sec_why-conv` haben wir die Versätze symmetrisch von $-\Delta$ bis $\Delta$ um die Mitte angeordnet. Nehmen wir die linke obere Ecke als Bezugspunkt, ändern sich die Versätze nur um die feste Verschiebung $\Delta$. So lassen sich auch Faltungskerne mit geraden Seitenlängen ohne mittleres Pixel direkt behandeln, etwa ein $2 \times 2$-Faltungskern oder unser $1 \times 2$-Kantendetektor.

<!--en:fa1b7da9 kind=caption-->
![Two-dimensional cross-correlation. The shaded portions are the first output element and the input and kernel elements used to compute it: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)
<!--tr:fa1b7da9-->
![Zweidimensionale Kreuzkorrelation. Schattiert sind der erste Ausgabeeintrag sowie die Elemente der Eingabe und des Faltungskerns, aus denen er sich berechnet: $0\times0+1\times1+3\times2+4\times3=19$.](../img/correlation.svg)

<!--en:2f8e5deb kind=paragraph-->
In :numref:`fig_correlation`, we slide a $2 \times 2$ kernel across a
$3 \times 3$ input one pixel at a time, left to right and top to bottom. The
window fits cleanly at four positions, yielding the $2 \times 2$ output:
<!--tr:2f8e5deb-->
In :numref:`fig_correlation` verschieben wir einen $2 \times 2$-Faltungskern pixelweise über eine $3 \times 3$-Eingabe, von links nach rechts und von oben nach unten. An vier Positionen liegt das Fenster vollständig in der Eingabe. Das ergibt die folgende $2 \times 2$-Ausgabe.

<!--en:4741e012 kind=paragraph-->
Notice that the output shrinks slightly along each axis because the kernel must
stay wholly inside the input. Fitting a width-$k$ window into length $n$ leaves
$n - k + 1$ valid positions, so the output shape is
<!--tr:4741e012-->
Die Ausgabe schrumpft entlang jeder Achse etwas, weil der Faltungskern ganz innerhalb der Eingabe bleiben muss. Ein Fenster der Breite $k$ passt auf einer Strecke der Länge $n$ an $n - k + 1$ gültige Positionen. Daraus ergibt sich die Form der Ausgabe.

<!--en:12d81c14 kind=paragraph-->
Thus a $3 \times 3$ input with a $2 \times 2$ kernel yields $2 \times 2$, a
$6 \times 8$ image with a $1 \times 2$ kernel yields $6 \times 7$, and a
$28 \times 28$ Fashion-MNIST image with a $5 \times 5$ kernel yields
$24 \times 24$. Whenever we want to preserve spatial dimensions, we surround the
input with a border of zeros (*padding*), which we examine alongside strides and
dilation in :numref:`sec_padding`.
<!--tr:12d81c14-->
Eine $3 \times 3$-Eingabe mit einem $2 \times 2$-Faltungskern ergibt also $2 \times 2$, ein $6 \times 8$-Bild mit einem $1 \times 2$-Faltungskern ergibt $6 \times 7$ und ein $28 \times 28$-Bild aus Fashion-MNIST mit einem $5 \times 5$-Faltungskern ergibt $24 \times 24$. Um die räumlichen Abmessungen beizubehalten, umgeben wir die Eingabe mit einem Rand aus Nullen (*Padding*). Dieses Verfahren betrachten wir zusammen mit Stride und Dilatation in :numref:`sec_padding`.

<!--en:869d223f kind=heading-->
### Implementation
<!--tr:869d223f-->
### Implementierung

<!--en:6a869f7c kind=paragraph-->
We implement :eqref:`eq_corr2d` directly in `corr2d` using a pair of loops over
every output coordinate.
<!--tr:6a869f7c-->
Wir setzen :eqref:`eq_corr2d` in `corr2d` direkt mit zwei verschachtelten Schleifen über alle Ausgabekoordinaten um.

<!--en:81e1f383 kind=paragraph-->
To verify `corr2d` against our hand calculation in :numref:`fig_correlation`,
we construct the $3 \times 3$ input tensor `X` and $2 \times 2$ kernel `K` and
evaluate `corr2d(X, K)`.
<!--tr:81e1f383-->
Wir vergleichen `corr2d` mit unserer Rechnung von Hand in :numref:`fig_correlation`. Dazu erzeugen wir den $3 \times 3$-Eingabetensor `X` und den $2 \times 2$-Faltungskern `K` und werten `corr2d(X, K)` aus.

<!--en:f567e1b2 kind=paragraph-->
The output reproduces :numref:`fig_correlation` entry for entry (and as shown in
:numref:`sec_why-conv`, strict mathematical convolution with a flipped kernel
$\mathbf{K}'$ learns the exact same family of layers, so we follow standard
deep learning terminology and call cross-correlation "convolution").
<!--tr:f567e1b2-->
Die Ausgabe stimmt Eintrag für Eintrag mit :numref:`fig_correlation` überein. Wie in :numref:`sec_why-conv` gezeigt, lässt sich mit der Faltung im streng mathematischen Sinn und einem umgedrehten Faltungskern $\mathbf{K}'$ genau dieselbe Familie von Schichten lernen. Daher folgen wir dem üblichen Sprachgebrauch im Deep Learning und bezeichnen auch die Kreuzkorrelation als „Faltung“.

<!--en:bfe11c62 kind=heading-->
## Convolutional Layers
<!--tr:bfe11c62-->
## Faltungsschichten

<!--en:2cdc020d kind=heading-->
### The Layer
<!--tr:2cdc020d-->
### Aufbau der Schicht

<!--en:c6371ece kind=paragraph-->
A two-dimensional convolutional layer cross-correlates its input with a learned
kernel and adds a scalar bias. Just like our MLP weights in
:numref:`chap_perceptrons`, we initialize the kernel from
$\mathcal{N}(0, 0.01^2)$ and the bias to zero. Below, `Conv2D` registers both
parameters and invokes `corr2d` in its forward pass; a layer with a kernel of
height $h$ and width $w$ is called an $h \times w$ convolutional layer.
<!--tr:c6371ece-->
Eine zweidimensionale Faltungsschicht berechnet die Kreuzkorrelation ihrer Eingabe mit einem gelernten Faltungskern und addiert einen skalaren Bias. Wie bei den Gewichten des mehrschichtigen Perzeptrons (engl. multilayer perceptron, MLP) in :numref:`chap_perceptrons` ziehen wir die Anfangswerte des Faltungskerns aus $\mathcal{N}(0, 0.01^2)$. Den Bias setzen wir anfangs auf null. Im folgenden Code registriert `Conv2D` beide Parameter und ruft im Vorwärtsdurchlauf (engl. forward pass) `corr2d` auf. Eine Schicht mit einem Faltungskern der Höhe $h$ und der Breite $w$ heißt $h \times w$-Faltungsschicht.

<!--en:29cd9061 kind=paragraph-->
Unlike a dense layer, a convolutional layer is not locked to a fixed input
resolution because its parameters depend only on the kernel size. The exact same
$2 \times 2$ layer (5 parameters total) processes both our $3 \times 3$ tensor
`X` and a $6 \times 8$ image, returning outputs of shapes $2 \times 2$ and
$5 \times 7$:
<!--tr:29cd9061-->
Eine Faltungsschicht kann im Gegensatz zu einer vollständig verbundenen Schicht (engl. dense layer) Eingaben unterschiedlicher Auflösung verarbeiten, weil ihre Parameter nur von der Kerngröße abhängen. Dieselbe $2 \times 2$-Schicht mit insgesamt 5 Parametern verarbeitet sowohl unseren $3 \times 3$-Tensor `X` als auch ein $6 \times 8$-Bild. Die Ausgaben haben die Formen $2 \times 2$ beziehungsweise $5 \times 7$.

<!--en:c9e5aa07 kind=heading-->
### Edge Detection
<!--tr:c9e5aa07-->
### Kantenerkennung

<!--en:3d9a357b kind=paragraph-->
We now test the vertical edge detector from the introduction on a
$6 \times 8$ synthetic image whose middle four columns are black (`0`) and outer
columns are white (`1`).
<!--tr:3d9a357b-->
Nun testen wir den Detektor für vertikale Kanten aus der Einleitung an einem künstlichen $6 \times 8$-Bild. Seine mittleren vier Spalten sind schwarz (`0`), die äußeren weiß (`1`).

<!--en:68513ff0 kind=paragraph-->
We define the $1 \times 2$ kernel
<!--tr:68513ff0-->
Wir definieren den folgenden $1 \times 2$-Faltungskern.

<!--en:eb45803b kind=paragraph-->
At every position $(i, j)$, this computes the horizontal *finite difference*
$[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, a discrete approximation (up to sign)
of the horizontal derivative of brightness, which vanishes across flat regions
and spikes wherever intensity jumps.
<!--tr:eb45803b-->
Der Faltungskern berechnet an jeder Position $(i, j)$ die horizontale *endliche Differenz* (engl. finite difference) $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$. Sie nähert bis auf das Vorzeichen die horizontale Ableitung der Helligkeit diskret an. Die Differenz ist in gleichförmigen Bereichen null und schlägt dort aus, wo die Helligkeit sprunghaft wechselt.

<!--en:e810b3a9 kind=paragraph-->
Cross-correlating the $6 \times 8$ stripe image `X` with the $1 \times 2$
horizontal-difference kernel `K` via `corr2d(X, K)` produces the $6 \times 7$
edge map `Y`:
<!--tr:e810b3a9-->
Wir berechnen die Kreuzkorrelation des $6 \times 8$-Streifenbilds `X` mit dem $1 \times 2$-Faltungskern `K` für horizontale Differenzen. Der Aufruf `corr2d(X, K)` ergibt die $6 \times 7$-Kantenkarte `Y`.

<!--en:568358e4 kind=paragraph-->
Sure enough, `Y` fires `+1` along the white-to-black boundary, `-1` along the
black-to-white boundary, and `0` everywhere else. Applying the exact same kernel
to the transposed image (turning the vertical stripe into a horizontal one)
yields all zeros:
<!--tr:568358e4-->
Wie erwartet liefert `Y` am Übergang von Weiß zu Schwarz `+1`, am Übergang von Schwarz zu Weiß `-1` und sonst überall `0`. Transponieren wir das Bild, wird der vertikale Streifen horizontal. Mit demselben Faltungskern erhalten wir dann ausschließlich Nullen.

<!--en:71c591a4 kind=paragraph-->
Clearly `[1, -1]` responds strictly to vertical edges. Detecting horizontal
edges, diagonals, corners, or textures requires different kernels altogether.
<!--tr:71c591a4-->
Der Faltungskern `[1, -1]` reagiert also ausschließlich auf vertikale Kanten. Für horizontale Kanten, Diagonalen, Ecken oder Texturen brauchen wir ganz andere Faltungskerne.

<!--en:f0af4887 kind=heading-->
### Learning a Kernel
<!--tr:f0af4887-->
### Lernen eines Faltungskerns

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
Wenn wir genau wissen, welches Merkmal wir suchen, lässt sich `[1, -1]` leicht von Hand angeben. In einem tiefen Netz lassen sich jedoch unmöglich Tausende Filter für die Zwischenschichten von Hand festlegen. Kann der Gradientenabstieg den Faltungskern `[1, -1]` allein aus dem Eingabe-Ausgabe-Paar `(X, Y)` ermitteln? Das prüfen wir mit der eingebauten 2D-Faltungsschicht des jeweiligen Frameworks. Wir lassen den Bias weg und ziehen die Anfangsgewichte aus $\mathcal{N}(0, 0.01^2)$. Dann minimieren wir die Summe der quadratischen Fehler (engl. squared errors) über alle 42 Ausgabeeinträge in 10 Gradientenschritten mit der Lernrate (engl. learning rate) $0.03$. Eingebaute Schichten erwarten 4D-Tensoren mit Achsen für Batch und Kanal (:numref:`sec_channels`). Deshalb bringen wir `X` und `Y` in die entsprechende Form.

<!--en:c29c5d61 kind=paragraph-->
Initially the loss sits near 12 (the near-zero random kernel misses all 12
$\pm 1$ edge pixels by roughly 1), and within just nine steps it drops below
0.01. Inspecting the learned kernel confirms what it found:
<!--tr:c29c5d61-->
Anfangs liegt der Verlust (engl. loss) bei etwa 12. Der zufällig initialisierte Faltungskern hat Gewichte nahe null und verfehlt alle 12 Kantenpixel mit Zielwerten von $\pm 1$ um ungefähr 1. Schon innerhalb von neun Schritten sinkt der Verlust unter 0.01. Ein Blick auf den gelernten Faltungskern bestätigt das Ergebnis.

<!--en:67d25b0b kind=paragraph-->
Both entries land within roughly 0.02 of `[1, -1]`. Nothing in this loop knew
anything about edges: given a loss signal, gradient descent discovered the
filter on its own. Here we supervised the single layer directly with `Y`; in a
deep CNN, backpropagation learns every layer's kernels end to end from the final
task loss alone.
<!--tr:67d25b0b-->
Beide Einträge liegen höchstens etwa 0.02 von den entsprechenden Werten in `[1, -1]` entfernt. Die Schleife enthält keinerlei Wissen über Kanten. Allein anhand des Verlustsignals hat der Gradientenabstieg den Filter selbst gefunden. Hier haben wir der einzigen Schicht `Y` direkt als Ziel vorgegeben. In einem tiefen konvolutionären neuronalen Netz (engl. convolutional neural network, CNN) lernt die Backpropagation die Faltungskerne aller Schichten durchgängig allein aus dem Verlust am Ausgang des Netzes für die eigentliche Aufgabe.

<!--en:91b0010d kind=slide_text-->
Convolutions for Images<br>**2D cross-correlation · edge detection · learned kernels · im2col · receptive fields**
<!--tr:91b0010d-->
Faltungen für Bilder<br>**2D-Kreuzkorrelation · Kantenerkennung · gelernte Faltungskerne · im2col · rezeptive Felder**

<!--en:1d94295a kind=slide_title-->
::: {.slide title="Detecting Local Edges Across an Image"}
<!--tr:1d94295a-->
::: {.slide title="Lokale Kantenerkennung im ganzen Bild"}

<!--en:687ed302 kind=kicker-->
[Motivation & Core Mental Model]{.kicker}
<!--tr:687ed302-->
[Motivation und Grundidee]{.kicker}

<!--en:05fceed0 kind=slide_text-->
- **Synthetic $6 \times 8$ test image:** White outer columns (`1`) surrounding a 4-column black vertical band (`0`).
<!--tr:05fceed0-->
- **Künstliches $6 \times 8$-Testbild:** Weiße Außenspalten (`1`) umgeben einen 4 Spalten breiten senkrechten schwarzen Streifen (`0`).

<!--en:71e700b3 kind=slide_text-->
- **Horizontal finite difference:** Sweeping $\mathbf{K} = [1, -1]$ computes $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$, firing `+1` at white-to-black and `-1` at black-to-white edges.
<!--tr:71e700b3-->
- **Horizontale endliche Differenz:** Beim Verschieben berechnet $\mathbf{K} = [1, -1]$ die Differenz $[\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ mit `+1` für Weiß nach Schwarz und `-1` für Schwarz nach Weiß.

<!--en:957dc033 kind=slide_text-->
- **Learning vs. hand-crafting:** Deep CNNs discover thousands of such kernels automatically via gradient descent.
<!--tr:957dc033-->
- **Lernen statt Entwurf von Hand:** Tiefe CNNs finden per Gradientenabstieg selbstständig Tausende solcher Faltungskerne.

<!--en:cb1045e1 kind=slide_title-->
::: {.slide title="The 2D Cross-Correlation Operation"}
<!--tr:cb1045e1-->
::: {.slide title="Zweidimensionale Kreuzkorrelation"}

<!--en:17bcece5 kind=kicker-->
[Sliding-Window Inner Products]{.kicker}
<!--tr:17bcece5-->
[Innere Produkte im gleitenden Fenster]{.kicker}

<!--en:8d45aae8 kind=slide_text-->
Given an $n_\textrm{h} \times n_\textrm{w}$ input matrix $\mathbf{X}$ and a $k_\textrm{h} \times k_\textrm{w}$ kernel $\mathbf{K}$, the 2D cross-correlation $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ places the window's top-left corner at $(i, j)$ and sums elementwise products:
<!--tr:8d45aae8-->
Gegeben sind die $n_\textrm{h} \times n_\textrm{w}$-Eingabematrix $\mathbf{X}$ und der $k_\textrm{h} \times k_\textrm{w}$-Faltungskern $\mathbf{K}$. Für die 2D-Kreuzkorrelation $\mathbf{Y} = \mathbf{X} \star \mathbf{K}$ die linke obere Fensterecke auf $(i, j)$ setzen und die elementweisen Produkte summieren.

<!--en:e484380c kind=slide_text-->
- **Top-left anchor:** Zero-based indices $a, b \ge 0$ match array indexing in code and handle even-sized kernels (like $2 \times 2$ or $1 \times 2$) cleanly.
<!--tr:e484380c-->
- **Bezugspunkt links oben:** Die bei null beginnenden Indizes $a, b \ge 0$ entsprechen der Indizierung im Code und eignen sich auch für Faltungskerne mit geraden Seitenlängen wie $2 \times 2$ oder $1 \times 2$.

<!--en:137c0980 kind=slide_text-->
- **Shaded top-left output $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.
<!--tr:137c0980-->
- **Schattierter Ausgabeeintrag links oben $(0,0)$:**
  $0\cdot 0 + 1\cdot 1 + 3\cdot 2 + 4\cdot 3 = 19$.

<!--en:575c4aea kind=slide_title-->
::: {.slide title="Valid Window Positions and Output Spatial Shape"}
<!--tr:575c4aea-->
::: {.slide title="Gültige Fensterpositionen und räumliche Ausgabeform"}

<!--en:7eda4531 kind=kicker-->
[Why Unpadded Convolutions Shrink the Grid]{.kicker}
<!--tr:7eda4531-->
[Warum Faltungen ohne Padding das Raster verkleinern]{.kicker}

<!--en:891243ae kind=slide_text-->
- **Valid window placements:** A $k_\textrm{h} \times k_\textrm{w}$ window fits inside an $n_\textrm{h} \times n_\textrm{w}$ grid at $n - k + 1$ positions per axis.
<!--tr:891243ae-->
- **Gültige Fensterpositionen:** Ein $k_\textrm{h} \times k_\textrm{w}$-Fenster passt vollständig in ein $n_\textrm{h} \times n_\textrm{w}$-Raster, und zwar an $n - k + 1$ Positionen pro Achse.

<!--en:55cc8c32 kind=slide_text-->
- **Shape examples:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$); $28 \times 28 \star 5 \times 5 \to 24 \times 24$.
<!--tr:55cc8c32-->
- **Beispiele für Ausgabeformen:** $3 \times 3 \star 2 \times 2 \to 2 \times 2$ ($19, 25, 37, 43$) und $28 \times 28 \star 5 \times 5 \to 24 \times 24$.

<!--en:1e3463da kind=slide_title-->
::: {.slide title="Implementing 2D Cross-Correlation in Code"}
<!--tr:1e3463da-->
::: {.slide title="Implementierung der 2D-Kreuzkorrelation"}

<!--en:07566992 kind=kicker-->
[Direct Nested-Loop Implementation]{.kicker}
<!--tr:07566992-->
[Direkte Implementierung mit verschachtelten Schleifen]{.kicker}

<!--en:202aa5d3 kind=slide_text-->
The function `corr2d` allocates the $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$ output tensor and computes each entry as the sum of `X[i:i+h, j:j+w] * K`:
<!--tr:202aa5d3-->
Die Funktion `corr2d` reserviert Speicher für den $(n_\textrm{h} - k_\textrm{h} + 1) \times (n_\textrm{w} - k_\textrm{w} + 1)$-Ausgabetensor. Jeder Eintrag ist die Summe über `X[i:i+h, j:j+w] * K`.

<!--en:bef130ba kind=slide_text-->
Each iteration slices a $k_\textrm{h} \times k_\textrm{w}$ sub-window of `X`, multiplies elementwise with `K`, and reduces via `sum`.
<!--tr:bef130ba-->
In jeder Iteration ein $k_\textrm{h} \times k_\textrm{w}$-Fenster aus `X` ausschneiden, elementweise mit `K` multiplizieren und die Produkte mit `sum` addieren.

<!--en:4ee9f6b1 kind=slide_title-->
::: {.slide title="Verifying corr2d and the Flipped-Kernel Equivalence"}
<!--tr:4ee9f6b1-->
::: {.slide title="Prüfung von corr2d und Äquivalenz bei Kerndrehung"}

<!--en:2b0be423 kind=kicker-->
[Cross-Correlation vs. Mathematical Convolution]{.kicker}
<!--tr:2b0be423-->
[Kreuzkorrelation und mathematische Faltung]{.kicker}

<!--en:74b9b745 kind=slide_text-->
- **180° flipped kernel $\mathbf{K}'$:** Cross-correlating with $\mathbf{K}$ equals convolving with $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.
<!--tr:74b9b745-->
- **Um 180° gedrehter Faltungskern $\mathbf{K}'$:** Die Kreuzkorrelation mit $\mathbf{K}$ entspricht der Faltung mit $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$.

<!--en:5c8af7cd kind=slide_text-->
- **Identical learned class:** Frameworks compute cross-correlation and call it convolution.
<!--tr:5c8af7cd-->
- **Dieselbe Klasse lernbarer Schichten:** Frameworks berechnen die Kreuzkorrelation und nennen sie Faltung.

<!--en:a69e29b5 kind=slide_title-->
::: {.slide title="Building a 2D Convolutional Layer Module"}
<!--tr:a69e29b5-->
::: {.slide title="Implementierung einer 2D-Faltungsschicht"}

<!--en:55763603 kind=kicker-->
[Learnable Kernel Weights Plus Scalar Bias]{.kicker}
<!--tr:55763603-->
[Lernbare Gewichte des Faltungskerns und skalarer Bias]{.kicker}

<!--en:5b8bad24 kind=slide_text-->
A convolutional layer wraps `corr2d` with a learnable $k_\textrm{h} \times k_\textrm{w}$ weight parameter (initialized from $\mathcal{N}(0, 0.01^2)$) and a scalar bias (initialized to $0$):
<!--tr:5b8bad24-->
Eine Faltungsschicht kapselt `corr2d` mit einem lernbaren $k_\textrm{h} \times k_\textrm{w}$-Gewichtsparameter (Anfangswerte aus $\mathcal{N}(0, 0.01^2)$) und einem skalaren Bias (Anfangswert $0$).

<!--en:c16d97d6 kind=slide_title-->
::: {.slide title="Resolution Independence of Convolutional Layers"}
<!--tr:c16d97d6-->
::: {.slide title="Faltungsschichten bei beliebiger Auflösung"}

<!--en:a7d1e48f kind=kicker-->
[Fixed Parameters Across Arbitrary Input Sizes]{.kicker}
<!--tr:a7d1e48f-->
[Dieselben Parameter für jede Eingabegröße]{.kicker}

<!--en:53a2bce7 kind=slide_text-->
- **Decoupled from $H \times W$:** The exact same 5 parameters ($2 \times 2$ kernel + 1 bias) process $3 \times 3$, $6 \times 8$, or megapixel inputs without reshaping weights.
<!--tr:53a2bce7-->
- **Unabhängig von $H \times W$:** Dieselben 5 Parameter ($2 \times 2$-Faltungskern + 1 Bias) verarbeiten Eingaben mit $3 \times 3$, $6 \times 8$ oder im Megapixelbereich ohne Änderung der Gewichtsform.

<!--en:017fec24 kind=slide_title-->
::: {.slide title="Constructing a Synthetic Vertical-Edge Image"}
<!--tr:017fec24-->
::: {.slide title="Künstliches Testbild mit vertikalen Kanten"}

<!--en:896e641c kind=kicker-->
[A 6×8 Binary Test Pattern]{.kicker}
<!--tr:896e641c-->
[Binäres 6×8-Testmuster]{.kicker}

<!--en:0a4b456b kind=slide_text-->
We construct the $6 \times 8$ test image `X` with ones on the left and right borders and zeros across the middle four columns (`X[:, 2:6] = 0`):
<!--tr:0a4b456b-->
Das $6 \times 8$-Testbild `X` links und rechts mit Einsen und in den mittleren vier Spalten mit Nullen belegen (`X[:, 2:6] = 0`).

<!--en:dfc7e3e3 kind=slide_text-->
- **Piecewise-constant columns:** Intensity is constant (`1` or `0`) within columns $0\text{--}1$, $2\text{--}5$, and $6\text{--}7$, jumping only between columns $1 \to 2$ and $5 \to 6$.
<!--tr:dfc7e3e3-->
- **Abschnittsweise konstante Spalten:** Konstante Helligkeit (`1` oder `0`) in den Spalten $0\text{--}1$, $2\text{--}5$ und $6\text{--}7$. Sprünge nur bei den Übergängen $1 \to 2$ und $5 \to 6$.

<!--en:3ce31cbe kind=slide_title-->
::: {.slide title="Applying the Horizontal Finite-Difference Kernel"}
<!--tr:3ce31cbe-->
::: {.slide title="Faltungskern für horizontale endliche Differenzen"}

<!--en:b37a1029 kind=kicker-->
[Detecting Vertical Edges with K = [1, -1]]{.kicker}
<!--tr:b37a1029-->
[Vertikale Kanten mit K = [1, -1]]{.kicker}

<!--en:9be85019 kind=slide_text-->
Cross-correlating `X` with the $1 \times 2$ kernel $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ computes $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ at every horizontal pair:
<!--tr:9be85019-->
Die Kreuzkorrelation von `X` mit dem $1 \times 2$-Faltungskern $\mathbf{K} = \begin{bmatrix} 1 & -1 \end{bmatrix}$ berechnet $[\mathbf{Y}]_{i,j} = [\mathbf{X}]_{i,j} - [\mathbf{X}]_{i,j+1}$ für jedes waagerechte Pixelpaar.

<!--en:f649cfae kind=slide_text-->
- **Signed edge map:** Column 1 fires `+1` (white-to-black transition), column 5 fires `-1` (black-to-white transition), and all constant interior columns evaluate to `0`.
<!--tr:f649cfae-->
- **Kantenkarte mit Vorzeichen:** Spalte 1 liefert `+1` (Weiß nach Schwarz), Spalte 5 liefert `-1` (Schwarz nach Weiß). Alle übrigen Spalten ohne Helligkeitswechsel liefern `0`.

<!--en:e6d155be kind=slide_title-->
::: {.slide title="Orientation Selectivity of Edge Kernels"}
<!--tr:e6d155be-->
::: {.slide title="Richtungsabhängige Kantenerkennung"}

<!--en:b525cde4 kind=kicker-->
[Why One Kernel Cannot Detect All Orientations]{.kicker}
<!--tr:b525cde4-->
[Warum ein Faltungskern nur bestimmte Richtungen erfasst]{.kicker}

<!--en:38159de6 kind=slide_text-->
- **Blind to horizontal edges:** Transposing `X` (`X.T`) makes horizontal neighbors identical, so $[1, -1]$ outputs all zeros.
<!--tr:38159de6-->
- **Keine Reaktion auf horizontale Kanten:** Nach dem Transponieren von `X` (`X.T`) sind waagerecht benachbarte Pixel gleich. Daher liefert $[1, -1]$ nur Nullen.

<!--en:f0f30f76 kind=slide_text-->
- **Filter banks (§{sec_channels}):** Vision layers learn parallel channels for horizontal, vertical, and diagonal orientations.
<!--tr:f0f30f76-->
- **Filterbänke (§{sec_channels}):** Schichten zur Bildverarbeitung lernen parallele Kanäle für horizontale, vertikale und diagonale Ausrichtungen.

<!--en:b755e31b kind=slide_title-->
::: {.slide title="Learning an Edge Detector via Gradient Descent"}
<!--tr:b755e31b-->
::: {.slide title="Lernen eines Kantendetektors per Gradientenabstieg"}

<!--en:86bbeea7 kind=kicker-->
[Supervising a Built-In Convolutional Layer on (X, Y)]{.kicker}
<!--tr:86bbeea7-->
[Überwachtes Lernen der eingebauten Faltungsschicht mit (X, Y)]{.kicker}

<!--en:a569780c kind=slide_text-->
- **4D framework tensors:** Reshape `X` and `Y` to 4D tensors (one example, one channel) and minimize $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ at $\eta = 0.03$.
<!--tr:a569780c-->
- **4D-Tensoren im Framework:** `X` und `Y` in 4D-Tensoren umformen (ein Beispiel, ein Kanal) und $\mathcal{L} = \|\hat{\mathbf{Y}} - \mathbf{Y}\|_2^2$ mit $\eta = 0.03$ minimieren.

<!--en:2489ce6e kind=slide_title-->
::: {.slide title="Inspecting the Learned Kernel Weights"}
<!--tr:2489ce6e-->
::: {.slide title="Die gelernten Gewichte des Faltungskerns"}

<!--en:7e4ef44b kind=kicker-->
[Why Initial Loss Is 12 and Converges to [1, -1]]{.kicker}
<!--tr:7e4ef44b-->
[Anfangsverlust 12 und Konvergenz der Gewichte gegen [1, -1]]{.kicker}

<!--en:9cd1c645 kind=slide_text-->
- **Why loss starts at $12.0$:** Near-zero weights predict $\hat{Y} \approx 0$, missing the twelve $\pm 1$ edge pixels in `Y` ($12 \times 1^2 = 12$).
<!--tr:9cd1c645-->
- **Warum der Verlust bei $12.0$ beginnt:** Gewichte nahe null liefern $\hat{Y} \approx 0$ und verfehlen die zwölf Kantenpixel mit Werten von $\pm 1$ in `Y` ($12 \times 1^2 = 12$).

<!--en:456670c5 kind=slide_text-->
- **Convergence in 9 steps:** SGD recovers $[0.99, -0.99]$ automatically from data.
<!--tr:456670c5-->
- **Konvergenz in 9 Schritten:** SGD findet $[0.99, -0.99]$ selbstständig aus den Daten.

=== DECISIONS ===
- „edge detection“ → „Kantenerkennung“, „edge detector“ → „Kantendetektor“, „edge map“ → „Kantenkarte“.
- „patch“ → „Ausschnitt“ (das Fenster bezeichnet die Auswahl, der Ausschnitt die ausgewählten Einträge).
- „dense layer“, „fully connected layer“ → „vollständig verbundene Schicht“.
- „framework“ → „das Framework“ (Plural „die Frameworks“).
- „filter bank“ → „Filterbank“.

=== NOTES ===
- 7e4ef44b: Im englischen Zwischentitel scheint der Verlust gegen [1, -1] zu konvergieren. Gemeint sind die Gewichte.
- 456670c5: Auf der Folie steht „SGD“. Die Schleife berechnet bei jedem Schritt den Gradienten über alle 42 Ausgabeeinträge.
